import argparse
import math
import random
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import datetime
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from tensorboardX import SummaryWriter
from pytorch_msssim import ms_ssim

from model import ICIP2020, CheckerboardAR, CheckerboardARCheng2020Attention, \
    CheckerboardARCheng2020, ICIP2020Cheng, MeanScaleSGDN, ICIP2020GELU, ICIP2020ResB, ICIP2020ResBAtten
from data.datasets import Datasets1
from utils import setup_logger
from tqdm import tqdm
from PIL import Image
from compressai.zoo import mbt2018_mean, mbt2018


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer, logger):
    model.train()
    device = next(model.parameters()).device

    train_loss = AverageMeter()
    train_bpp = AverageMeter()
    train_psrn = AverageMeter()
    train_ms_ssim = AverageMeter()
    train_aux_loss = AverageMeter()

    num = len(train_dataloader)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        step = epoch * num + i
        psrn = 10 * np.log10(1.0 / out_criterion["mse_loss"].item())
        msssim = ms_ssim(d, out_net['x_hat'], data_range=1.0)
        writer.add_scalar('train_psnr', psrn, step)
        writer.add_scalar('train_mssim', msssim.detach().item(), step)
        writer.add_scalar('train_bpp', out_criterion["bpp_loss"].item(), step)
        writer.add_scalar('train_aux_loss', aux_loss.item(), step)

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tPSNR: {psrn:.3f} |'
                f'\tBpp: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux: {aux_loss.item():.2f}"
            )

        train_loss.update(out_criterion["loss"].item())
        train_bpp.update(out_criterion["bpp_loss"].item())
        train_psrn.update(psrn)
        train_ms_ssim.update(msssim.detach().item())
        train_aux_loss.update(aux_loss.item())

    logger.info(f"Train Epoch={epoch} LOSS={train_loss.avg:.4f}, AUX={train_aux_loss.avg:.4f}, "
                f"PSNR={train_psrn.avg:.4f}, MSSSIM={train_ms_ssim.avg:.4f}, BPP={train_bpp.avg:.4f}")


def test_epoch(epoch, test_dataloader, model, criterion, writer, logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    val_psrn = AverageMeter()
    val_ms_ssim = AverageMeter()
    aux_loss = AverageMeter()

    num = len(test_dataloader)
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            step = epoch * num + i
            psrn = 10 * np.log10(1.0 / out_criterion["mse_loss"].item())
            msssim = ms_ssim(d, out_net['x_hat'], data_range=1.0)
            val_psrn.update(psrn)
            val_ms_ssim.update(msssim.detach().item())
            writer.add_scalar('val_psnr', psrn, step)
            writer.add_scalar('val_mssim', msssim.detach().item(), step)
            writer.add_scalar('val_bpp', out_criterion["bpp_loss"].item(), step)
            writer.add_scalar('val_aux_loss', model.aux_loss().item(), step)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {psrn:.3f} |"
        f"\tBpp: {bpp_loss.avg:.2f} |"
        f"\tAux: {aux_loss.avg:.2f}\n"
    )

    logger.info(f"Test Epoch={epoch} LOSS={loss.avg:.4f}, AUX={aux_loss.avg:.4f}, PSNR={val_psrn.avg:.4f}, "
                f"MSSSIM={val_ms_ssim.avg:.4f}, BPP={bpp_loss.avg:.4f}")

    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('checkpoint', 'checkpoint_best_loss'))


def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    # parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0067,  # 0.0067， 0.0130， 0.0250， 0.0483   0.0932
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")  # 16
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,  # 64
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--seed", type=float, default=16, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint",
                        default='/home/tdx/LHB/LVC/ImageCoding/version1/logs/ICIP2020ResB_0.0932_20221003_125321/checkpoint_162.pth'
                        )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    date = str(datetime.datetime.now())
    date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    log_dir = os.path.join('./logs', f"ICIP2020Res_load0932_{args.lmbda}_{date}")
    os.makedirs(log_dir, exist_ok=True)

    summary_dir = os.path.join(log_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(logdir=summary_dir, comment='info')

    setup_logger('base', log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(f'[*] Start Log To {log_dir}')

    with open(os.path.join(log_dir, 'setting.json'), 'w') as f:
        flags_dict = {k: vars(args)[k] for k in vars(args)}
        json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset1 = Datasets1('/home/tdx/VideoData/vimeo_interp_train', train_transforms)
    train_dataset2 = Datasets1('/home/tdx/LHB/data/ICLC2020/train', train_transforms)
    train_dataset3 = Datasets1('/home/tdx/LHB/data/zip/flicker_2W_images', train_transforms)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])
    test_dataset = Datasets1('/home/tdx/LHB/LVC/ImageCoding/version1/data/image/kodim', test_transforms)
    logger.info(f'[*] Train File Account For {len(train_dataset)}, val {len(test_dataset)}')

    device = "cuda"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ICIP2020ResB()
    net = net.to(device)
    logger.info(f'[*] Total Parameters = {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    # [80, 100, 120]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 200], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 180], gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        optimizer.param_groups[0]['lr'] = 5e-6

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            logger,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, writer, logger)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            filename=os.path.join(log_dir, f"checkpoint_{epoch}.pth")
        )


if __name__ == "__main__":
    p = '/home/fzus/LHB/Dataset/train_image/ICLC2020/train'
    #   /home/fzus/LHB/Dataset/train_image/flicker_2W_images
    #   /home/fzus/LHB/Dataset/train_image/ICLC2020/train
    print(len(os.listdir(p)))
    for pp in tqdm(os.listdir(p)):
        im = np.array(Image.open(os.path.join(p, pp)))
        if im.shape[0] < 256 or im.shape[1] < 256 or len(im.shape) != 3:
            print(pp)
            # os.remove(os.path.join(p, pp))
    # main()
