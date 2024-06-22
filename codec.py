# -*- coding: utf-8 -*-
import glob
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
from pathlib import Path
import torch
import compressai
from compressai.zoo import image_models, models
from compressai.utils import plot
from model import ICIP2020ResB, ELIC, ICIP2020ResB_1, ICIP2020ResBx5, NIC, ICIP2020ResB1, NIC1, NIC2
from utils import load_pretrained, read_body, torch2img, show_image, load_image, img2torch, pad, write_body, filesize
from utils import compute_psnr, compute_msssim, compute_bpp, ms_ssim, read_image, cal_psnr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def ImageCodec():
    fun = 'encode'  # decode  encode
    compressai.set_entropy_coder("ans")

    input_image_path = 'data/image/stmalo_fracape.png'
    output_string_path = os.path.join('./output', 'bin', 'stmalo_fracape.bin')
    rec_imgae_path = os.path.join('./output', 'rec', 'rec_stmalo_fracape.png')
    # output = Path(Path(input_image_path).resolve().name).with_suffix(".bin")

    codec = Cheng2020Anchor()
    state_dict = torch.load(
        r'D:\Project\Pytorch\DeepVideoCoding\DCVC\checkpoints\cheng2020-anchor-3-e49be189.pth.tar',
        map_location='cuda:0')
    state_dict = load_pretrained(state_dict)
    codec.load_state_dict(state_dict)
    codec.update(force=True)

    if fun == 'encode':
        img = load_image(input_image_path)
        x = img2torch(img)

        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        x = pad(x, p)

        enc_start = time.time()
        with torch.no_grad():
            out = codec.compress(x)
        enc_time = time.time() - enc_start

        shape = out["shape"]

        with Path(output_string_path).open("wb") as f:
            ll = write_body(f, shape, out["strings"])
            print(ll)

        size = filesize(output_string_path)
        bpp = float(size) * 8 / (h * w)

        s1 = 0
        for s in out["strings"]:
            s1 += len(s[0])
        print(s1, size)

        print(f"{bpp:.4f} bpp | Encoded in {enc_time:.2f}s")

    elif fun == 'decode':
        show = True
        dec_start = time.time()
        with Path(output_string_path).open("rb") as f:
            strings, shape = read_body(f)
            with torch.no_grad():
                out = codec.decompress(strings, shape)

            # x_hat = crop(out["x_hat"], codec.original_size)
            x_hat = out["x_hat"]

            img = torch2img(x_hat)
            img.save(rec_imgae_path, quality=95)  # 65-95

            dec_time = time.time() - dec_start
            print(f"Decoded in {dec_time:.2f}s")

            if show:
                # For video, only the last frame is shown
                show_image(img)
    return 0


def TestImageCodec():
    # fun 0, 1 write strings and save images; 2, 3 fast test

    # 202 BPP 1.1435 | PSRN 39.4890 | MSSSIM 0.9933
    # 200 BPP 1.1436 | PSRN 39.4898 | MSSSIM 0.9934 **
    # 198 BPP 1.1437 | PSRN 39.4895 | MSSSIM 0.9934
    # 196 BPP 1.1436 | PSRN 39.4898 | MSSSIM 0.9934 **
    compressai.set_entropy_coder("ans")
    fun = 2
    if fun == 0:
        quality = 0.01
        # ckpt = './ckpt/0611/MeanScaleHyperprior_PSNR0.01/checkpoint.pth'
        # ckpt = './ckpt/0611/Cheng2020Anchor_PSNR0.01/checkpoint.pth'
        ckpt = r'E:\temp\0615\MeanScaleHyperprior_PSNR0.01\checkpoint.pth'
        img_path = './data/image/kodim'
        string_path = f'./output/bin/{quality}'
        os.makedirs(string_path, exist_ok=True)
        rec_path = f'./output/rec/{quality}'
        os.makedirs(rec_path, exist_ok=True)

        codec = MeanScaleHyperprior1()
        state_dict = torch.load(ckpt, map_location='cuda:0')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        images = glob.glob(os.path.join(img_path, '*.png'))
        print(f'* Find {len(images)} Images')
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for path in images:
            name = path.split('\\')[-1].split('.')[0]
            image_org = load_image(path)
            image = img2torch(image_org)

            h, w = image.size(2), image.size(3)
            # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

            enc_start = time.time()
            with torch.no_grad():
                enc_out = codec.compress(image)
            enc_time = time.time() - enc_start
            encT.append(enc_time)
            shape = enc_out["shape"]

            output_string_path = os.path.join(string_path, name + '.bin')
            with Path(output_string_path).open("wb") as f:
                write_body(f, shape, enc_out["strings"])

            size = filesize(output_string_path)
            bpp = float(size) * 8 / (h * w)
            Bpp.append(bpp)

            # decode
            dec_start = time.time()
            with Path(output_string_path).open("rb") as f:
                strings, shape = read_body(f)
                with torch.no_grad():
                    out = codec.decompress(strings, shape)

                x_hat = out["x_hat"]

                rec_imgae_path = os.path.join(rec_path, path.split('\\')[-1])
                dec_img = torch2img(x_hat)
                dec_img.save(rec_imgae_path, quality=95)  # 65-95

                ms_ssim1 = compute_msssim(x_hat.clamp_(0, 1), image)
                MS_SSIM.append(ms_ssim1)
                psnr = compute_psnr(x_hat.clamp_(0, 1), image)
                PSNR.append(psnr)
                dec_time = time.time() - dec_start
                decT.append(dec_time)

            print(f"{name} | Quality {quality} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                  f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
        print(f'Quality {quality} | Average BPP {np.mean(Bpp):.4f} | PSRN {np.mean(PSNR):.4f} | MSSSIM {np.mean(MS_SSIM):.4f}'
              f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')

    elif fun == 1:
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for quality in [0.1, 0.01, 0.001, 0.001]:
            # ckpt = './ckpt/0611/MeanScaleHyperprior_PSNR0.01/checkpoint.pth'
            ckpt = f'./ckpt/0611/Cheng2020Anchor_PSNR{quality}/checkpoint.pth'
            img_path = './data/image/kodim'
            string_path = f'./output/bin/{quality}'
            os.makedirs(string_path, exist_ok=True)
            rec_path = f'./output/rec/{quality}'
            os.makedirs(rec_path, exist_ok=True)

            codec = Cheng2020Anchor()
            state_dict = torch.load(ckpt, map_location='cuda:0')["state_dict"]
            state_dict = load_pretrained(state_dict)
            codec.load_state_dict(state_dict)
            codec.update(force=True)

            images = glob.glob(os.path.join(img_path, '*.png'))
            print(f'* Find {len(images)} Images')
            _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
            for path in images:
                name = path.split('\\')[-1].split('.')[0]
                image_org = load_image(path)
                image = img2torch(image_org)

                h, w = image.size(2), image.size(3)
                # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

                enc_start = time.time()
                with torch.no_grad():
                    enc_out = codec.compress(image)
                enc_time = time.time() - enc_start
                _encT.append(enc_time)
                shape = enc_out["shape"]

                output_string_path = os.path.join(string_path, name + '.bin')
                with Path(output_string_path).open("wb") as f:
                    write_body(f, shape, enc_out["strings"])

                size = filesize(output_string_path)
                bpp = float(size) * 8 / (h * w)
                _Bpp.append(bpp)

                # decode
                dec_start = time.time()
                with Path(output_string_path).open("rb") as f:
                    strings, shape = read_body(f)
                    with torch.no_grad():
                        out = codec.decompress(strings, shape)

                    x_hat = out["x_hat"]

                    rec_imgae_path = os.path.join(rec_path, path.split('\\')[-1])
                    dec_img = torch2img(x_hat)
                    dec_img.save(rec_imgae_path, quality=95)  # 65-95

                    ms_ssim1 = compute_msssim(x_hat.clamp_(0, 1), image)
                    _MS_SSIM.append(ms_ssim1)
                    psnr = compute_psnr(x_hat.clamp_(0, 1), image)
                    _PSNR.append(psnr)
                    dec_time = time.time() - dec_start
                    _decT.append(dec_time)

                print(
                    f"{name} | Quality {quality} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                    f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
            print(
                f'Quality {quality} | Average BPP {np.mean(_Bpp):.4f} | PSRN {np.mean(_PSNR):.4f} | MSSSIM {np.mean(_MS_SSIM):.4f}'
                f' | Encode Time {np.mean(_encT):.4f} | Decode Time {np.mean(_decT):.4f}')
            PSNR.append(np.mean(_PSNR))
            MS_SSIM.append(np.mean(_MS_SSIM))
            Bpp.append(np.mean(_Bpp))
            encT.append(np.mean(_encT))
            decT.append(np.mean(_decT))
        print(f'BPP: {Bpp}')
        print(f'PSNR : {PSNR}')
        print(f'MSSSIM : {MS_SSIM}')

    elif fun == 2:
        quality = 0.01
        # mbt2018CKBD
        # 192 BPP 1.22000 | PSRN 38.90172 | MSSSIM 0.99296
        # 189 BPP 1.21991 | PSRN 38.90205 | MSSSIM 0.99295

        # ms-ssim 220.0
        # 165  BPP 1.14243 | PSRN 33.96035 | MSSSIM 0.99669

        ckpt = './logs/ICIP2020ResB_MSSSIM_from0_220.0_20221230_193336/checkpoint_165.pth'
        # ckpt = './logs/ICIP2020ResBx5_0.0932_20221223_111327/checkpoint_80.pth'
        # ckpt = './logs/ELIC_0.0932_20221228_195642/checkpoint_13.pth'
        img_path = './data/image/kodim'

        # codec = ELIC().cuda()
        codec = ICIP2020ResB().cuda()  # ICIP2020ResBx5
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        images = glob.glob(os.path.join(img_path, '*.png'))
        images = sorted(images)
        print(f'* Find {len(images)} Images')
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for path in images:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()

            num_pixels = image.size(0) * image.size(2) * image.size(3)
            # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image)
                enc_time = time.time() - start
                encT.append(enc_time)

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                dec_time = time.time() - start
                decT.append(dec_time)

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)
            print(f"{name} | Quality {quality} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                  f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
        print(f'Quality {quality} | Average BPP {np.mean(Bpp):.5f} | PSRN {np.mean(PSNR):.5f} | MSSSIM {np.mean(MS_SSIM):.5f}'
              f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')

        results = {"psnr": np.mean(PSNR), "ms-ssim": np.mean(MS_SSIM), "bpp": np.mean(Bpp),
                   "encoding_time": np.mean(encT), "decoding_time": np.mean(decT)}
        output = {
            "name": 'lhb-fixhyperior_mse',
            "description": "Inference (ans)",
            "results": results,
        }
        # print(json.dumps(output, indent=2))
        with open("./output/test.json", 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, indent=2)

    elif fun == 3:
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for quality in [0.1, 0.01, 0.001, 0.001]:
            # ckpt = './ckpt/0611/MeanScaleHyperprior_PSNR0.01/checkpoint.pth'
            ckpt = f'./ckpt/0611/Cheng2020Anchor_PSNR{quality}/checkpoint.pth'
            img_path = './data/image/kodim'
            string_path = f'./output/bin/{quality}'
            os.makedirs(string_path, exist_ok=True)
            rec_path = f'./output/rec/{quality}'
            os.makedirs(rec_path, exist_ok=True)

            codec = Cheng2020Anchor()
            state_dict = torch.load(ckpt, map_location='cuda:0')["state_dict"]
            state_dict = load_pretrained(state_dict)
            codec.load_state_dict(state_dict)
            codec.update(force=True)

            images = glob.glob(os.path.join(img_path, '*.png'))
            print(f'* Find {len(images)} Images')
            _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
            for path in images:
                name = path.split('\\')[-1].split('.')[0]
                image = read_image(path).unsqueeze(0)

                num_pixels = image.size(0) * image.size(2) * image.size(3)
                # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

                with torch.no_grad():
                    start = time.time()
                    out_enc = codec.compress(image)
                    enc_time = time.time() - start
                    encT.append(enc_time)

                    start = time.time()
                    out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                    dec_time = time.time() - start
                    decT.append(dec_time)

                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    psnr = cal_psnr(image, out_dec["x_hat"])
                    ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                    PSNR.append(psnr)
                    Bpp.append(bpp)
                    MS_SSIM.append(ms_ssim1)

                print(
                    f"{name} | Quality {quality} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                    f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
            print(
                f'Quality {quality} | Average BPP {np.mean(_Bpp):.4f} | PSRN {np.mean(_PSNR):.4f} | MSSSIM {np.mean(_MS_SSIM):.4f}'
                f' | Encode Time {np.mean(_encT):.4f} | Decode Time {np.mean(_decT):.4f}')
            PSNR.append(np.mean(_PSNR))
            MS_SSIM.append(np.mean(_MS_SSIM))
            Bpp.append(np.mean(_Bpp))
            encT.append(np.mean(_encT))
            decT.append(np.mean(_decT))
        print(f'BPP: {Bpp}')
        print(f'PSNR : {PSNR}')
        print(f'MSSSIM : {MS_SSIM}')

        results = {"psnr": PSNR, "ms-ssim": MS_SSIM, "bpp": Bpp, "encoding_time": encT, "decoding_time": decT}
        output = {"name": 'lhb-fixhyperior_mse', "description": "Inference (ans)", "results": results}
        with open("./output/test.json", 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, indent=2)
    return 0


def parse_json_file(filepath, metric, db_ssim=False):
    # psnr  ms-ssim  bpp  encoding_time  decoding_time
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if "results" in data:
        results = data["results"]
    else:
        results = data

    if metric not in results:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
        )

    try:
        if metric == "ms-ssim" and db_ssim:
            # Convert to db
            values = np.array(results[metric])
            results[metric] = -10 * np.log10(1 - values)

        return {
            "name": data.get("name", name),
            "xs": results["bpp"],
            "ys": results[metric],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')


def matplotlib_plt(scatters, title, ylabel, output_file, limits=None, show=False, figsize=None):
    linestyle = "-"
    hybrid_matches = ["HM", "VTM", "JPEG", "JPEG2000", "WebP", "BPG", "AV1"]
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        if any(x in sc["name"] for x in hybrid_matches):
            linestyle = "--"
        ax.plot(
            sc["xs"],
            sc["ys"],
            marker=".",
            linestyle=linestyle,
            linewidth=0.7,
            label=sc["name"],
        )

    ax.set_xlabel("Bit-rate [bpp]")
    ax.set_ylabel(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc="lower right")

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300)


def plot_result():
    metric = 'psnr'
    scatters = []
    # UVG
    # results_file = glob.glob(os.path.join('./data/UVG', '*.json'))
    results_file = glob.glob(os.path.join('./data/kodak_result/tgt/mse', '*.json'))
    for f in results_file:
        rv = parse_json_file(f, metric)
        scatters.append(rv)

    ylabel = f"{metric} [dB]"

    matplotlib_plt(scatters, title='Demo', ylabel=ylabel, output_file='./output/test.png', show=True)
    return 0


def save_new_model_weights():
    from collections import OrderedDict
    for e in [25, 26]:
        ckpt11 = f'./logs/ICIP2020ResBx3_bg512_0.0067/checkpoint_{e}.pth'
        ckpt1 = torch.load(ckpt11)
        print('epoch', ckpt1['epoch'])
        print('loss', ckpt1['loss'])
        # exit()
        new_dict = OrderedDict()
        new_dict['epoch'] = ckpt1['epoch']
        new_dict['state_dict'] = ckpt1['state_dict']
        torch.save(new_dict, f'./logs/ICIP2020ResBx3_bg512_0.0067/checkpoint_{e}_1.pth')
    return 0


def test_elic_checkpoints():
    lmbda = f'psnr_0.0932'
    mark = 'ELIC_0.0932'
    # ckpt_path = './logs/ICIP2020ResBx5_0.0932_20230102_120818'
    ckpt_path = './ckpt/ELIC_0.0932'
    checkpoints = sorted(glob.glob(os.path.join(ckpt_path, '*.pth')))
    img_path = './data/image/kodim'
    Bpp1, MS_SSIM1, PSNR1, encT1, decT1, CKPT = [], [], [], [], [], []
    for ii, ckpt in enumerate(checkpoints):
        # print(ckpt)
        ckptt = ckpt.split('/')[-1]
        CKPT.append(ckptt)
        codec = ELIC().cuda()
        # codec = ICIP2020ResBx5().cuda()
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        images = glob.glob(os.path.join(img_path, '*.png'))
        images = sorted(images)
        # print(f'* Find {len(images)} Images')
        bar = tqdm(images)
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for path in bar:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image)
                enc_time = time.time() - start
                encT.append(enc_time)

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                dec_time = time.time() - start
                decT.append(dec_time)

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)
            # print(f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
            #       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}")
            bar.desc = f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s " \
                       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}"
        print(
            f'{ckptt} | {lmbda} | Average BPP {np.mean(Bpp):.6f} | PSRN {np.mean(PSNR):.6f} | MSSSIM {np.mean(MS_SSIM):.7f}'
            f' | Encode Time {np.mean(encT):.3f} | Decode Time {np.mean(decT):.3f}')
        Bpp1.append(np.mean(Bpp))
        MS_SSIM1.append(np.mean(MS_SSIM))
        PSNR1.append(np.mean(PSNR))
        encT1.append(np.mean(encT))
        decT1.append(np.mean(decT))

    results = {"psnr": PSNR1, "ms-ssim": MS_SSIM1, "bpp": Bpp1, "CKPT": CKPT,
               "encoding_time": encT1, "decoding_time": decT1}
    output = {
        "name": 'ICIP2020ResB',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(f"{ckpt_path}/ICIP2020ResB_{lmbda}.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    print(f'best {lmbda} {checkpoints[PSNR1.index(max(PSNR1))]}, {max(PSNR1)}')
    # print(f'best {lmbda} {checkpoints[MS_SSIM1.index(max(MS_SSIM1))]}, {max(MS_SSIM1)}')

    # plt.scatter(Bpp1, PSNR1, c="r", label=f'{mark}')
    for bpp, psnr in zip(Bpp1, PSNR1):
        ckptn = checkpoints[PSNR1.index(psnr)]
        ckptn = ckptn.split('/')[-1].split('.')[0].split('_')[-1]
        plt.scatter(bpp, psnr, label=f'epoch{ckptn}')

    Bpp = [
      0.3097127278645833,
      0.4498155381944444,
      0.6407640245225694,
      0.8683946397569445,
      1.1432020399305556
    ]
    PSNR = [
      32.377160696098976,
      34.10368974430478,
      35.96023583934226,
      37.754727226130875,
      39.49347830473297
    ]
    plt.plot(Bpp, PSNR, label='ICIP2020ResBx3')
    plt.grid()
    plt.xlim(1.091, 1.10)
    plt.ylim(39.15, 39.22)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("bpp (bit/pixel)")
    plt.legend()
    plt.savefig(f"./{mark}.png")
    # plt.savefig(f"{ckpt_path}/{mark}.png")

    return 0


def test_resbx5_checkpoints():
    index = 1
    if index == 1:
        lmbda = f'psnr_0.013'
        mark = '/ICIP2020ResBx5_0.013'
        ckpt_path = './logs/ICIP2020ResBx5_0.013_20230112_104254'
    elif index == 2:
        lmbda = f'psnr_0.025'
        mark = '/ICIP2020ResBx5_0.025'
        ckpt_path = './logs/ICIP2020ResBx5_0.025_20230116_112004'
    elif index == 3:
        lmbda = f'psnr_0.0483'
        mark = '/ICIP2020ResBx5_0.0483'
        ckpt_path = './logs/ICIP2020ResBx5_0.0483_20230109_111257'

    checkpoints = sorted(glob.glob(os.path.join(ckpt_path, '*.pth')))
    img_path = './data/image/kodim'
    Bpp1, MS_SSIM1, PSNR1, encT1, decT1, CKPT = [], [], [], [], [], []
    for ii, ckpt in enumerate(checkpoints):
        # print(ckpt)
        ckptt = ckpt.split('/')[-1]
        CKPT.append(ckptt)
        codec = ICIP2020ResBx5().cuda()
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        images = glob.glob(os.path.join(img_path, '*.png'))
        images = sorted(images)
        # print(f'* Find {len(images)} Images')
        bar = tqdm(images)
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for path in bar:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image)
                enc_time = time.time() - start
                encT.append(enc_time)

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                dec_time = time.time() - start
                decT.append(dec_time)

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)
            # print(f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
            #       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}")
            bar.desc = f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s " \
                       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}"
        print(
            f'{ckptt} | {lmbda} | Average BPP {np.mean(Bpp):.6f} | PSRN {np.mean(PSNR):.6f} | MSSSIM {np.mean(MS_SSIM):.7f}'
            f' | Encode Time {np.mean(encT):.3f} | Decode Time {np.mean(decT):.3f}')
        Bpp1.append(np.mean(Bpp))
        MS_SSIM1.append(np.mean(MS_SSIM))
        PSNR1.append(np.mean(PSNR))
        encT1.append(np.mean(encT))
        decT1.append(np.mean(decT))

    results = {"psnr": PSNR1, "ms-ssim": MS_SSIM1, "bpp": Bpp1, "CKPT": CKPT,
               "encoding_time": encT1, "decoding_time": decT1}
    output = {
        "name": 'ICIP2020ResB',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(f"{ckpt_path}/ICIP2020ResBx5_{lmbda}.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    print(f'best {lmbda} {checkpoints[PSNR1.index(max(PSNR1))]}, {max(PSNR1)}')
    # print(f'best {lmbda} {checkpoints[MS_SSIM1.index(max(MS_SSIM1))]}, {max(MS_SSIM1)}')
    # plt.scatter(Bpp1, PSNR1, c="r", label=f'{mark}')

    for bpp, psnr in zip(Bpp1, PSNR1):
        ckptn = checkpoints[PSNR1.index(psnr)]
        ckptn = ckptn.split('/')[-1].split('.')[0].split('_')[-1]
        plt.scatter(bpp, psnr, label=f'epoch{ckptn}')

    Bpp = [
      0.3097127278645833,
      0.4498155381944444,
      0.6407640245225694,
      0.8683946397569445,
      1.1432020399305556
    ]
    PSNR = [
      32.377160696098976,
      34.10368974430478,
      35.96023583934226,
      37.754727226130875,
      39.49347830473297
    ]
    plt.plot(Bpp, PSNR, label='ICIP2020ResBx3')
    plt.grid()
    # mean_x = np.mean(Bpp1)
    # mean_y = np.mean(PSNR1)
    if index == 1:
        plt.xlim(0.32, 0.55)
        plt.ylim(33, 34.5)
    elif index == 2:
        plt.xlim(0.52, 0.65)
        plt.ylim(35.4, 36.0)
    elif index == 3:
        plt.xlim(0.70, 0.95)
        plt.ylim(37, 40)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("bpp (bit/pixel)")
    plt.legend()
    plt.savefig(f"./{mark}.png")
    # plt.savefig(f"{ckpt_path}/{mark}.png")

    return 0


def test_resbx3_checkpoints():
    fun = 2
    
    if fun == 0:
        lmbda = f'psnr_0.0932'
        mark = 'NIC_0.0932'
        ckpt_path = './logs/NIC_0.0932_20240402_143315'
    elif fun == 1:
        lmbda = f'psnr_0.0932'
        mark = 'NIC1_0.0932'
        ckpt_path = './logs/NIC1_0.0932_20240401_143655'
    elif fun == 2:
        lmbda = f'psnr_0.0932'
        mark = 'NIC2_0.0932'
        ckpt_path = './logs/NIC2_0.0932_20240405_151317'
    elif fun == 3:
        lmbda = f'psnr_0.0932'
        mark = 'ICIP2020ResB1_0.0932'
        ckpt_path = './logs/ICIP2020ResB1_0.0932_20240401_200715'

    checkpoints = sorted(
        glob.glob(os.path.join(ckpt_path, '*.pth')),
        key=lambda checkpoint: int(checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
    )
    img_path = './data/kodim'
    images = glob.glob(os.path.join(img_path, '*.png'))
    images = sorted(images)
    print(f'* Find {len(images)} Images')

    Bpp1, MS_SSIM1, PSNR1, encT1, decT1, CKPT = [], [], [], [], [], []
    for ii, ckpt in enumerate(checkpoints):
        # print(ckpt)
        ckptt = ckpt.split('/')[-1]
        CKPT.append(ckptt)
        if fun == 0:
            codec = NIC().cuda()
        elif fun == 1:
            codec = NIC1().cuda()
        elif fun == 2:
            codec = NIC2().cuda()
        elif fun == 3:
            codec = ICIP2020ResB1().cuda()
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        bar = tqdm(images)
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        for path in bar:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image)
                enc_time = time.time() - start
                encT.append(enc_time)

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                dec_time = time.time() - start
                decT.append(dec_time)

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)
            # print(f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
            #       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}")
            bar.desc = f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s " \
                       f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}"
        print(
            f'{ckptt} | {lmbda} | Average BPP {np.mean(Bpp):.6f} | PSRN {np.mean(PSNR):.6f} | MSSSIM {np.mean(MS_SSIM):.7f}'
            f' | Encode Time {np.mean(encT):.3f} | Decode Time {np.mean(decT):.3f}')
        Bpp1.append(np.mean(Bpp))
        MS_SSIM1.append(np.mean(MS_SSIM))
        PSNR1.append(np.mean(PSNR))
        encT1.append(np.mean(encT))
        decT1.append(np.mean(decT))

    results = {"psnr": PSNR1, "ms-ssim": MS_SSIM1, "bpp": Bpp1, "CKPT": CKPT,
               "encoding_time": encT1, "decoding_time": decT1}
    output = {
        "name": 'ICIP2020ResB',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(f"{ckpt_path}/ICIP2020ResB_{lmbda}.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    print(f'best {lmbda} {checkpoints[PSNR1.index(max(PSNR1))]}, {max(PSNR1)}')
    # print(f'best {lmbda} {checkpoints[MS_SSIM1.index(max(MS_SSIM1))]}, {max(MS_SSIM1)}')
    # plt.scatter(Bpp1, PSNR1, c="r", label=f'{mark}')

    for bpp, psnr in zip(Bpp1, PSNR1):
        ckptn = checkpoints[PSNR1.index(psnr)]
        ckptn = ckptn.split('/')[-1].split('.')[0].split('_')[-1]
        plt.scatter(bpp, psnr, label=f'epoch{ckptn}')

    Bpp = [
      0.3097127278645833,
      0.4498155381944444,
      0.6407640245225694,
      0.8683946397569445,
      1.1432020399305556
    ]
    PSNR = [
      32.377160696098976,
      34.10368974430478,
      35.96023583934226,
      37.754727226130875,
      39.49347830473297
    ]
    plt.plot(Bpp, PSNR, 'k', label='ICIP2020ResBx3')
    plt.grid()
    plt.xlim(1.05, 1.2)
    plt.ylim(38.3, 40)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("bpp (bit/pixel)")
    plt.legend()
    plt.savefig(f"./{mark}.png")
    # plt.savefig(f"{ckpt_path}/{mark}.png")

    return 0


def test_image_codec_checkpoints():
    index = -1
    lmbda = ''
    ckpt_path = ''
    if index == -1:
        lmbda = f'msssim_{4.58}'
        ckpt_path = './logs/ICIP2020ResB_MSSSIM_from0_4.58'
    elif index == 0:
        lmbda = f'msssim_{8.73}'
        ckpt_path = './logs2/ICIP2020ResBx5_MSSSIM_from0_8.73_20230107_182500'
    elif index == 1:
        lmbda = f'msssim_{16.64}'
        ckpt_path = './logs2/ICIP2020ResBx5_MSSSIM_from0_16.64_20230109_212349'
    elif index == 2:
        lmbda = f'msssim_{31.73}'
        ckpt_path = './logs2/ICIP2020ResBx5_MSSSIM_from0_31.73_20230110_233259'
    elif index == 3:
        lmbda = f'msssim_{60.5}'
        ckpt_path = './logs2/ICIP2020ResBx5_MSSSIM_from0_60.5_20230110_233450'
    elif index == 4:
        lmbda = f'msssim_{115.37}'
        ckpt_path = './logs2/ICIP2020ResBx5_MSSSIM_from0_115.37_20230110_233622'
    checkpoints = sorted(glob.glob(os.path.join(ckpt_path, '*.pth')))
    img_path = './data/image/kodim'
    Bpp1, MS_SSIM1, PSNR1, encT1, decT1, CKPT = [], [], [], [], [], []
    for ii, ckpt in enumerate(checkpoints):
        # print(ckpt)
        ckptt = ckpt.split('/')[-1]
        CKPT.append(ckptt)
        codec = ICIP2020ResB().cuda()
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        state_dict = load_pretrained(state_dict)
        codec.load_state_dict(state_dict)
        codec.update(force=True)

        images = glob.glob(os.path.join(img_path, '*.png'))
        images = sorted(images)
        # print(f'* Find {len(images)} Images')
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        bar = tqdm(images)
        for path in bar:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image)
                enc_time = time.time() - start
                encT.append(enc_time)

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                dec_time = time.time() - start
                decT.append(dec_time)

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)
                # print(f"{name} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                #       f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
                bar.desc = f"{name} | {lmbda} | {ckptt} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s " \
                           f"| MS-SSIM {ms_ssim1:.5f} | PSNR {psnr:.4f}"
        print(
            f'{ckptt} | Average BPP {np.mean(Bpp):.5f} | PSRN {np.mean(PSNR):.5f} | MSSSIM {np.mean(MS_SSIM):.5f}'
            f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')
        Bpp1.append(np.mean(Bpp))
        MS_SSIM1.append(np.mean(MS_SSIM))
        PSNR1.append(np.mean(PSNR))
        encT1.append(np.mean(encT))
        decT1.append(np.mean(decT))

    results = {"psnr": PSNR1, "ms-ssim": MS_SSIM1, "bpp": Bpp1, "CKPT": CKPT,
               "encoding_time": encT1, "decoding_time": decT1}
    output = {
        "name": 'ICIP2020ResBx5',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(f"{ckpt_path}/ICIP2020ResBx5_{lmbda}.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    print(f'best {lmbda} {checkpoints[MS_SSIM1.index(max(MS_SSIM1))]}, '
          f'MS_SSIM = {max(MS_SSIM1)}, BPP= {Bpp1[MS_SSIM1.index(max(MS_SSIM1))]}')

    # plt.scatter(Bpp1, MS_SSIM1, c="r", label=f'{mark}')

    for bpp, msssim in zip(Bpp1, MS_SSIM1):
        ckptn = checkpoints[MS_SSIM1.index(msssim)]
        ckptn = ckptn.split('/')[-1].split('.')[0].split('_')[-1]
        plt.scatter(bpp, msssim, label=f'epoch{ckptn}')

    mean_x = np.mean(Bpp1)
    mean_y = np.mean(MS_SSIM1)

    Bpp = [
        0.23639933268229166,
        0.3303697374131944,
        0.45316229926215285,
        0.6289130316840278,
        0.8599378797743057,
        1.1433546278211806
    ]
    MSSSIM = [
        0.9771117394169172,
        0.9844505339860916,
        0.9892362629373869,
        0.992717963953813,
        0.9951162909468015,
        0.9966983944177628
    ]
    plt.plot(Bpp, MSSSIM, label='ICIP2020ResBx3')
    plt.grid()
    plt.xlim(mean_x - 0.03, mean_x + 0.06)
    plt.ylim(mean_y - 0.04, mean_y + 0.03)
    plt.ylabel("MSSSIM")
    plt.xlabel("bpp (bit/pixel)")
    plt.legend()
    plt.savefig(f"./{lmbda}.png")
    return 0


def test():
    img_path = './data/kodim'
    images = glob.glob(os.path.join(img_path, '*.png'))
    images = sorted(images)
    print(f'* Find {len(images)} Images')

    codec = ICIP2020ResB().cuda()
    codec.update(force=True)

    Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
    for path in images:
        name = path.split('/')[-1].split('.')[0]
        image = read_image(path).unsqueeze(0).cuda()
        num_pixels = image.size(0) * image.size(2) * image.size(3)

        with torch.no_grad():
            start = time.time()
            out_enc = codec.compress(image)
            enc_time = time.time() - start
            encT.append(enc_time)

            start = time.time()
            out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - start
            decT.append(dec_time)

            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            psnr = cal_psnr(image, out_dec["x_hat"])
            ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
            PSNR.append(psnr)
            Bpp.append(bpp)
            MS_SSIM.append(ms_ssim1)
            print(f"{name} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                  f"| MS-SSIM {ms_ssim1:.6f} | PSNR {psnr:.6f}")

    return 0


if __name__ == "__main__":
    # test_image_codec_checkpoints()
    # test()
    test_resbx3_checkpoints()
    # save_new_model_weights()
    # test_resbx5_checkpoints()
