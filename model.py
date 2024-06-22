# -*- coding: utf-8 -*-
import os
import math
import warnings
import time

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import subpel_conv3x3, ResidualBlockUpsample, ResidualBlockWithStride, ResidualBlock, conv3x3
from compressai.layers import MaskedConv2d, GDN, AttentionBlock
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from compressai.models.waseda import Cheng2020Anchor

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from torch.autograd import Function
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from utils import load_pretrained, compute_psnr, compute_msssim, compute_bpp

from typing import Any

from thop import profile


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

out_channel_N = 64
out_channel_M = 96  # 96
out_channel_mv = 128  # 128


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


# class GDN(nn.Module):
#     def __init__(self,
#                  ch,
#                  inverse=False,
#                  beta_min=1e-6,
#                  gamma_init=0.1,
#                  reparam_offset=2 ** -18):
#         super(GDN, self).__init__()
#         self.inverse = inverse
#         self.beta_min = beta_min
#         self.gamma_init = gamma_init
#         self.reparam_offset = reparam_offset
#
#         self.build(ch)
#
#     def build(self, ch):
#         self.pedestal = self.reparam_offset ** 2
#         self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
#         self.gamma_bound = self.reparam_offset
#
#         beta = torch.sqrt(torch.ones(ch) + self.pedestal)
#         self.beta = nn.Parameter(beta)
#
#         eye = torch.eye(ch)
#         g = self.gamma_init * eye
#         g = g + self.pedestal
#         gamma = torch.sqrt(g)
#
#         self.gamma = nn.Parameter(gamma)
#         self.pedestal = self.pedestal
#
#     def forward(self, inputs):
#         unfold = False
#         if inputs.dim() == 5:
#             unfold = True
#             bs, ch, d, w, h = inputs.size()
#             inputs = inputs.view(bs, ch, d * w, h)
#
#         _, ch, _, _ = inputs.size()
#
#         # Beta bound and reparam
#         beta = LowerBound.apply(self.beta, self.beta_bound)
#         beta = beta ** 2 - self.pedestal
#
#         # Gamma bound and reparam
#         gamma = LowerBound.apply(self.gamma, self.gamma_bound)
#         gamma = gamma ** 2 - self.pedestal
#         gamma = gamma.view(ch, ch, 1, 1)
#
#         # Norm pool calc
#         norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
#         # print(norm_.shape, gamma.shape, beta.shape)
#         # exit()
#         norm_ = torch.sqrt(norm_)
#
#         # Apply norm
#         if self.inverse:
#             outputs = inputs * norm_
#         else:
#             outputs = inputs / norm_
#
#         if unfold:
#             outputs = outputs.view(bs, ch, d, w, h)
#         return outputs


class GSDN(nn.Module):
    """Generalized Subtractive and Divisive Normalization layer.
    y[i] = (x[i] - )/ sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super().__init__()
        self.inverse = inverse
        self.build(ch, beta_min, gamma_init, reparam_offset)

    def build(self, ch, beta_min, gamma_init, reparam_offset):
        self.pedestal = reparam_offset ** 2
        self.beta_bound = torch.FloatTensor([(beta_min + reparam_offset ** 2) ** .5])
        self.gamma_bound = torch.FloatTensor([reparam_offset])

        ###### param for divisive ######
        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)
        # Create gamma param
        eye = torch.eye(ch)
        g = gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

        ###### param for subtractive ######
        # Create beta2 param
        beta2 = torch.zeros(ch)
        self.beta2 = nn.Parameter(beta2)
        # Create gamma2 param
        eye = torch.eye(ch)
        g = gamma_init * eye
        g = g + self.pedestal
        gamma2 = torch.sqrt(g)
        self.gamma2 = nn.Parameter(gamma2)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        if self.inverse:
            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta ** 2 - self.pedestal
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma ** 2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
            norm_ = torch.sqrt(norm_)

            outputs = inputs * norm_  # modified

            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2 ** 2 - self.pedestal
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2 ** 2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)

            outputs = outputs + mean_
        else:
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2 ** 2 - self.pedestal
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2 ** 2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)

            outputs = inputs - mean_  # modified

            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta ** 2 - self.pedestal
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma ** 2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
            norm_ = torch.sqrt(norm_)

            outputs = outputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs


class SGDN(nn.Module):
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(SGDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)

        # print(gamma.shape)
        # exit()

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 3, 3)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta, padding=1)
        # print(norm_.shape, gamma.shape, beta.shape)
        # exit()
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


def mean1(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def sum1(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


class ActNorm(nn.Module):
    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = mean1(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            var = mean1((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(var) + self.eps))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, x, reverse=False):
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias

    def _scale(self, x, reverse=False):
        logs = self.logs
        if not reverse:
            x = x * torch.exp(logs)
        else:
            x = x * torch.exp(-logs)
        return x

    def forward(self, x, reverse=False):
        if not self.inited:
            self.initialize_parameters(x)
        if not reverse:
            # center and scale
            x = self._center(x, reverse)
            x = self._scale(x, reverse)
        else:
            # scale and center
            x = self._scale(x, reverse)
            x = self._center(x, reverse)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]


class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CheckerboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class Space2Depth(nn.Module):
    """
    ref: https://github.com/huzi96/Coarse2Fine-PyTorch/blob/master/networks.py
    """

    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


class Depth2Space(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r ** 2)
        out_h = h * r
        out_w = w * r
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


def Demultiplexer(x):
    """
    See Supplementary Material: Figure 2.
    This operation can also implemented by slicing.
    """
    x_prime = Space2Depth(r=2)(x)

    _, C, _, _ = x_prime.shape
    anchor_index = tuple(range(C // 4, C * 3 // 4))
    non_anchor_index = tuple(range(0, C // 4)) + tuple(range(C * 3 // 4, C))

    anchor = x_prime[:, anchor_index, :, :]
    non_anchor = x_prime[:, non_anchor_index, :, :]

    return anchor, non_anchor


def Multiplexer(anchor, non_anchor):
    """
    The inverse opperation of Demultiplexer.
    This operation can also implemented by slicing.
    """
    _, C, _, _ = non_anchor.shape
    x_prime = torch.cat((non_anchor[:, : C // 2, :, :], anchor, non_anchor[:, C // 2:, :, :]), dim=1)
    return Depth2Space(r=2)(x_prime)


class MeanScaleSGDN(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(3, N),
            SGDN(N),
            conv(N, N),
            SGDN(N),
            conv(N, N),
            SGDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            SGDN(N, inverse=True),
            deconv(N, N),
            SGDN(N, inverse=True),
            deconv(N, N),
            SGDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# Total Parameters = 17561699, computation cost: 28.75 GMACs, parameters: 17.33 M
class MeanScale2018(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# Total Parameters = 10561996, computation cost: 26.97 GMACs, parameters: 10.46 M
class MeanScaleCheng(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class MeanScaleChengRefine(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 16, 2),
        )

        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# 25504596
class mbt2018CKBD(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardMaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        # Notion: in compressai, the means must be subtracted before quantification.
        # In order to get y_half, we need subtract y_anchor's means and then quantize,
        # to get y_anchor's means, we have to go through 'gep' here
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_anchor, y_non_anchor = Demultiplexer(y)
        scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
        means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

        anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor,
                                                                means=means_hat_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 3
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)

        # PASS 1: anchor
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_anchor, _ = Demultiplexer(scales_hat)
        means_hat_anchor, _ = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor,
                                                        means=means_hat_anchor)  # [1, 384, 8, 8]
        y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))  # [1, 192, 16, 16]

        # PASS 2: non-anchor
        ctx_params = self.context_prediction(y_anchor)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_non_anchor = Demultiplexer(scales_hat)
        _, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor,
                                                            means=means_hat_non_anchor)  # [1, 384, 8, 8]
        y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)  # [1, 192, 16, 16]

        # gather
        y_hat = y_anchor + y_non_anchor
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
        }


# 37464660
class mbt2018CKBD_ResBx5(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )
        self.context_prediction = CheckerboardMaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        # Notion: in compressai, the means must be subtracted before quantification.
        # In order to get y_half, we need subtract y_anchor's means and then quantize,
        # to get y_anchor's means, we have to go through 'gep' here
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_anchor, y_non_anchor = Demultiplexer(y)
        scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
        means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

        anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor,
                                                                means=means_hat_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 3
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)

        # PASS 1: anchor
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_anchor, _ = Demultiplexer(scales_hat)
        means_hat_anchor, _ = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor,
                                                        means=means_hat_anchor)  # [1, 384, 8, 8]
        y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))  # [1, 192, 16, 16]

        # PASS 2: non-anchor
        ctx_params = self.context_prediction(y_anchor)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_non_anchor = Demultiplexer(scales_hat)
        _, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor,
                                                            means=means_hat_non_anchor)  # [1, 384, 8, 8]
        y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)  # [1, 192, 16, 16]

        # gather
        y_hat = y_anchor + y_non_anchor
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
        }


# 32591700
class mbt2018CKBD_ResBx3(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )
        self.context_prediction = CheckerboardMaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        # Notion: in compressai, the means must be subtracted before quantification.
        # In order to get y_half, we need subtract y_anchor's means and then quantize,
        # to get y_anchor's means, we have to go through 'gep' here
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_anchor, y_non_anchor = Demultiplexer(y)
        scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
        means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

        anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor,
                                                                means=means_hat_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 3
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)

        # PASS 1: anchor
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_anchor, _ = Demultiplexer(scales_hat)
        means_hat_anchor, _ = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor,
                                                        means=means_hat_anchor)  # [1, 384, 8, 8]
        y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))  # [1, 192, 16, 16]

        # PASS 2: non-anchor
        ctx_params = self.context_prediction(y_anchor)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_non_anchor = Demultiplexer(scales_hat)
        _, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor,
                                                            means=means_hat_non_anchor)  # [1, 384, 8, 8]
        y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)  # [1, 192, 16, 16]

        # gather
        y_hat = y_anchor + y_non_anchor
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
        }


class mbt2018CKBD1(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardMaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        hyper_params = self.h_s(z_hat)
        ctx_params = self.context_prediction(y_hat)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def validate(self, x):
        """
        Estimate true distortion by ste(y-means) + means instead of adding uniform noise.
        This function can also be used to train a LIC model.
        """
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        # mask non-anchor
        gaussian_params_anchor[:, :, 0::2, 0::2] = 0
        gaussian_params_anchor[:, :, 1::2, 1::2] = 0
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        ctx_params = self.context_prediction(ste_round(y - means_anchor) + means_anchor)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = ste_round(y - means_hat) + means_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        """
        Compress by ste(y-mu) + mu, which leads to two-pass encoding.
        For one-pass encoding, you can use Round(y) and range-coder for AE/AD.
        When adopting range-coder, this repo https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention may be helpful.
        """
        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.compress_anchor(y, scales_anchor, means_anchor, symbols_list, indexes_list)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        # nonanchor_hat = self.compress_nonanchor(y, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        torch.cuda.synchronize()
        start_time = time.process_time()

        y_strings = strings[0][0]
        z_strings = strings[1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([z_hat.size(0), self.M * 2, z_hat.size(2) * 4, z_hat.size(3) * 4],
                                        device=z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.decompress_anchor(scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.decompress_nonanchor(scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)

        y_hat = anchor_hat + nonanchor_hat
        x_hat = self.g_s(y_hat).clamp(0.0, 1.0)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def ckbd_anchor_sequeeze(self, y):
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y):
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor):
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor):
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor

    def compress_anchor(self, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
        # squeeze anchor to avoid non-anchor symbols
        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
        return anchor_hat

    def compress_nonanchor(self, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
        return nonanchor_hat

    def decompress_anchor(self, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(
            scales_anchor.device) + means_anchor_squeeze
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(self, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(
            scales_nonanchor.device) + means_nonanchor_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


# Total Parameters = 11833149, computation cost: 27.09 GMACs, parameters: 10.91 M
class Cheng2020Anchor1(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 16, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020(nn.Module):
    def __init__(self, N=192, M=320, attention=False):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        if not attention:
            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                AttentionBlock(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
                AttentionBlock(M),
            )

            self.g_s = nn.Sequential(
                AttentionBlock(M),
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                AttentionBlock(N),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )

        self.h_a = nn.Sequential(
            conv(M, 256, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(256, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated


class ICIP2020Cheng(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 8
        self.max_support_slices = 4
        # self.num_slices = 2
        # self.max_support_slices = 1

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        # self.h_mean_s = nn.Sequential(
        #     deconv(N, N),
        #     nn.LeakyReLU(inplace=True),
        #     deconv(N, 256),
        #     nn.LeakyReLU(inplace=True),
        #     conv(256, M, stride=1, kernel_size=3),
        # )

        self.h_mean_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, M),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, M),
        )

        # self.h_scale_s = nn.Sequential(
        #     deconv(N, N),
        #     nn.LeakyReLU(inplace=True),
        #     deconv(N, 256),
        #     nn.LeakyReLU(inplace=True),
        #     conv(256, M, stride=1, kernel_size=3),
        # )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated


class ICIP2020GELU(nn.Module):
    def __init__(self, N=192, M=320, attention=False):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        if not attention:
            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                AttentionBlock(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
                AttentionBlock(M),
            )

            self.g_s = nn.Sequential(
                AttentionBlock(M),
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                AttentionBlock(N),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )

        self.h_a = nn.Sequential(
            conv(M, 256, stride=1, kernel_size=3),
            nn.GELU(),
            conv(256, N),
            nn.GELU(),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.GELU(),
            deconv(N, 256),
            nn.GELU(),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.GELU(),
            deconv(N, 256),
            nn.GELU(),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated


class MeanScale(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(128, N),
            GDN(N),
            conv(N, N, stride=1, kernel_size=3),
            GDN(N),
            conv(N, N, stride=1, kernel_size=3),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            conv(N, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            conv(N, 128, stride=1, kernel_size=3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=3),
        )

        self.h_s = nn.Sequential(
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        # print(z.shape)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ResBottleneckBlock(nn.Module):
    def __init__(self, channel, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if slope < 0.0001:
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return x + out


class ResBottleneckBlock1(nn.Module):
    def __init__(self, channel, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // 2, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(channel // 2, channel // 2, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(channel // 2, channel, 1, 1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if slope < 0.0001:
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return x + out


# 57914819
class ICIP2020ResB(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResB1(nn.Module):
    def __init__(self, N=225, M=225):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 5
        self.max_support_slices = 3

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            conv(N, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            conv(N, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            deconv(N, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            deconv(N, N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            ResBottleneckBlock1(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResBx5(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResBAtten(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResB_1(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            nn.LeakyReLU(True),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# 64746723
class ELIC(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.slices_depth = [16, 16, 32, 64, 192]
        self.num_slices = len(self.slices_depth)

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.g_sp = nn.ModuleList(
            nn.Sequential(
                CheckerboardMaskedConv2d(in_channels=self.slices_depth[i],
                                         out_channels=self.slices_depth[i] * 2,
                                         kernel_size=5, stride=1, padding=2),
            ) for i in range(self.num_slices)
        )

        self.g_ch = nn.ModuleList(
            nn.Sequential(
                conv(2 * M + sum(self.slices_depth[:i]), M,
                     stride=1, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(M, M // 2,
                     stride=1, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(M // 2, self.slices_depth[i] * 2,
                     stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.slices_depth[i] * 4 + 2 * M, M * 3 // 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 3 // 2, M, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M, self.slices_depth[i] * 2, 1),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        hyper_params = self.h_s(z_hat)  # [b, 640, w//16, h//16]

        y_slices = y.split(self.slices_depth, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index, 'y_slice.shape', y_slice.shape)
            y_hat_slice = self.gaussian_conditional.quantize(y_slice,
                                                             "noise" if self.training else "dequantize")
            y_hat_slice_half = y_hat_slice.clone()
            y_hat_slice_half[:, :, 0::2, 0::2] = 0
            y_hat_slice_half[:, :, 1::2, 1::2] = 0
            ctx_params = self.g_sp[slice_index](y_hat_slice_half)
            ctx_params[:, :, 0::2, 1::2] = 0
            ctx_params[:, :, 1::2, 0::2] = 0

            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            # print(0, slice_index, ctx_params.shape, cc_params.shape, hyper_params.shape)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))

            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_slice_hat,
                                                              means=means_slice_hat)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        N, _, H, W = z_hat.shape

        hyper_params = self.h_s(z_hat)  # [b, 640, w//16, h//16]

        y_slices = y.split(self.slices_depth, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            y_anchor, y_non_anchor = Demultiplexer(y_slice)

            zero_ctx_params = torch.zeros([N, 2 * self.slices_depth[slice_index],
                                           H * 4, W * 4]).to(z_hat.device)
            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            # print(1, slice_index, zero_ctx_params.shape, cc_params.shape, hyper_params.shape)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([zero_ctx_params, cc_params, hyper_params], dim=1))
            # anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            scales_hat_anchor, _ = Demultiplexer(scales_slice_hat)
            means_hat_anchor, _ = Demultiplexer(means_slice_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            y_q_slice_anchor = self.gaussian_conditional.quantize(y_anchor, "symbols", means_hat_anchor)
            y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

            symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
            indexes_list.extend(index_anchor.reshape(-1).tolist())

            y_hat_slice_anchor = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            ctx_params = self.g_sp[slice_index](y_hat_slice_anchor)

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))
            # non_anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, scales_hat_non_anchor = Demultiplexer(scales_slice_hat)
            _, means_hat_non_anchor = Demultiplexer(means_slice_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_non_anchor, "symbols", means_hat_non_anchor)
            y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

            y_hat_slice_non_anchor = Multiplexer(torch.zeros_like(y_hat_slice_non_anchor), y_hat_slice_non_anchor)
            # gather
            y_hat_slice = y_hat_slice_anchor + y_hat_slice_non_anchor
            y_hat_slices.append(y_hat_slice)

            symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        N, _, H, W = z_hat.shape
        hyper_params = self.h_s(z_hat)
        y_hat_slices = []
        y_string = strings[0][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            zero_ctx_params = torch.zeros([N, 2 * self.slices_depth[slice_index],
                                           H * 4, W * 4]).to(z_hat.device)
            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([zero_ctx_params, cc_params, hyper_params], dim=1))
            # anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            scales_hat_anchor, _ = Demultiplexer(scales_slice_hat)
            means_hat_anchor, _ = Demultiplexer(means_slice_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

            y_hat_slice_anchor = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            ctx_params = self.g_sp[slice_index](y_hat_slice_anchor)

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))
            # non_anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, scales_hat_non_anchor = Demultiplexer(scales_slice_hat)
            _, means_hat_non_anchor = Demultiplexer(means_slice_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

            y_hat_slice_non_anchor = Multiplexer(torch.zeros_like(y_hat_slice_non_anchor), y_hat_slice_non_anchor)
            # gather
            y_hat_slice = y_hat_slice_anchor + y_hat_slice_non_anchor
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# 107044323
class ELICeven(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.slices_depth = [32] * 10
        self.num_slices = len(self.slices_depth)

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.g_sp = nn.ModuleList(
            nn.Sequential(
                CheckerboardMaskedConv2d(in_channels=self.slices_depth[i],
                                         out_channels=self.slices_depth[i] * 2,
                                         kernel_size=5, stride=1, padding=2),
            ) for i in range(self.num_slices)
        )

        self.g_ch = nn.ModuleList(
            nn.Sequential(
                conv(2 * M + sum(self.slices_depth[:i]), M,
                     stride=1, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(M, M // 2,
                     stride=1, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(M // 2, self.slices_depth[i] * 2,
                     stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.slices_depth[i] * 4 + 2 * M, M * 3 // 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 3 // 2, M, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M, self.slices_depth[i] * 2, 1),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        hyper_params = self.h_s(z_hat)  # [b, 640, w//16, h//16]

        y_slices = y.split(self.slices_depth, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index, 'y_slice.shape', y_slice.shape)
            y_hat_slice = self.gaussian_conditional.quantize(y_slice,
                                                             "noise" if self.training else "dequantize")
            y_hat_slice_half = y_hat_slice.clone()
            y_hat_slice_half[:, :, 0::2, 0::2] = 0
            y_hat_slice_half[:, :, 1::2, 1::2] = 0
            ctx_params = self.g_sp[slice_index](y_hat_slice_half)
            ctx_params[:, :, 0::2, 1::2] = 0
            ctx_params[:, :, 1::2, 0::2] = 0

            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            # print(0, slice_index, ctx_params.shape, cc_params.shape, hyper_params.shape)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))

            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_slice_hat,
                                                              means=means_slice_hat)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        N, _, H, W = z_hat.shape

        hyper_params = self.h_s(z_hat)  # [b, 640, w//16, h//16]

        y_slices = y.split(self.slices_depth, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            y_anchor, y_non_anchor = Demultiplexer(y_slice)

            zero_ctx_params = torch.zeros([N, 2 * self.slices_depth[slice_index],
                                           H * 4, W * 4]).to(z_hat.device)
            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            # print(1, slice_index, zero_ctx_params.shape, cc_params.shape, hyper_params.shape)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([zero_ctx_params, cc_params, hyper_params], dim=1))
            # anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            scales_hat_anchor, _ = Demultiplexer(scales_slice_hat)
            means_hat_anchor, _ = Demultiplexer(means_slice_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            y_q_slice_anchor = self.gaussian_conditional.quantize(y_anchor, "symbols", means_hat_anchor)
            y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

            symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
            indexes_list.extend(index_anchor.reshape(-1).tolist())

            y_hat_slice_anchor = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            ctx_params = self.g_sp[slice_index](y_hat_slice_anchor)

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))
            # non_anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, scales_hat_non_anchor = Demultiplexer(scales_slice_hat)
            _, means_hat_non_anchor = Demultiplexer(means_slice_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_non_anchor, "symbols", means_hat_non_anchor)
            y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

            y_hat_slice_non_anchor = Multiplexer(torch.zeros_like(y_hat_slice_non_anchor), y_hat_slice_non_anchor)
            # gather
            y_hat_slice = y_hat_slice_anchor + y_hat_slice_non_anchor
            y_hat_slices.append(y_hat_slice)

            symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        N, _, H, W = z_hat.shape
        hyper_params = self.h_s(z_hat)
        y_hat_slices = []
        y_string = strings[0][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            zero_ctx_params = torch.zeros([N, 2 * self.slices_depth[slice_index],
                                           H * 4, W * 4]).to(z_hat.device)
            support_slices = torch.cat([hyper_params] + y_hat_slices, dim=1)
            cc_params = self.g_ch[slice_index](support_slices)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([zero_ctx_params, cc_params, hyper_params], dim=1))
            # anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            scales_hat_anchor, _ = Demultiplexer(scales_slice_hat)
            means_hat_anchor, _ = Demultiplexer(means_slice_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

            y_hat_slice_anchor = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            ctx_params = self.g_sp[slice_index](y_hat_slice_anchor)

            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat([ctx_params, cc_params, hyper_params], dim=1))
            # non_anchor
            scales_slice_hat, means_slice_hat = gaussian_params.chunk(2, 1)
            _, scales_hat_non_anchor = Demultiplexer(scales_slice_hat)
            _, means_hat_non_anchor = Demultiplexer(means_slice_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

            y_hat_slice_non_anchor = Multiplexer(torch.zeros_like(y_hat_slice_non_anchor), y_hat_slice_non_anchor)
            # gather
            y_hat_slice = y_hat_slice_anchor + y_hat_slice_non_anchor
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# --------------------------------- NIC --------------------------------------------------

class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch, inplace=False):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.LeakyReLU(inplace=inplace),
        )

    def forward(self, x):
        x = self.down(x)
        identity = x
        out = self.conv(x)
        out = out + identity
        return out


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class DepthConv2(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.out_conv(x1 * x2)
        return identity + x


class ConvFFN3(nn.Module):
    def __init__(self, in_ch, inplace=False):
        super().__init__()
        expansion_factor = 2
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = self.relu1(x1) + self.relu2(x2)
        return identity + self.conv_out(out)


class ConvFFN2(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)


class DepthConvBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv2(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlockUpsample1(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out


class DepthConvBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN3(out_ch, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


# 25941763
class NIC(nn.Module):
    def __init__(self, N=256, M=128, z_channel=128, inplace=False):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 8
        self.max_support_slices = 5

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            ResidualBlockWithStride2(3, 128, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockWithStride2(128, 192, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockWithStride2(192, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

        self.g_s = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample1(N, N, 2, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample1(N, 192, 2, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockUpsample1(192, 128, 2, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockUpsample1(128, 128, 2, inplace=inplace),
            conv3x3(128, 3),
        )

        self.h_a = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )

        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        # self.lrp_transforms = nn.ModuleList(
        #     nn.Sequential(
        #         conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
        #         nn.LeakyReLU(inplace=True),
        #         conv(224, 128, stride=1, kernel_size=3),
        #         nn.LeakyReLU(inplace=True),
        #         conv(128, slice_depth, stride=1, kernel_size=3)
        #     ) for i in range(self.num_slices)
        # )

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        # y, slice_shape = self.pad_for_y(y)
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            # lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # # print(slice_index, lrp_support.shape)
            # lrp = self.lrp_transforms[slice_index](lrp_support)
            # lrp = 0.5 * torch.tanh(lrp)
            # y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            # lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # lrp = self.lrp_transforms[slice_index](lrp_support)
            # lrp = 0.5 * torch.tanh(lrp)
            # y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            # lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # lrp = self.lrp_transforms[slice_index](lrp_support)
            # lrp = 0.5 * torch.tanh(lrp)
            # y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# 34561795
class NIC1(nn.Module):
    def __init__(self, N=256, M=256, z_channel=256, inplace=False):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 8
        self.max_support_slices = 5

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            ResidualBlockWithStride2(3, 128, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockWithStride2(128, 192, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockWithStride2(192, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

        self.g_s = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample1(N, N, 2, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample1(N, 192, 2, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockUpsample1(192, 128, 2, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockUpsample1(128, 128, 2, inplace=inplace),
            conv3x3(128, 3),
        )

        self.h_a = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )

        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample1(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        # y, slice_shape = self.pad_for_y(y)
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class NIC2(nn.Module):
    def __init__(self, N=256, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5
        inplace = False

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            ResidualBlockWithStride2(3, 128, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockWithStride2(128, 192, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockWithStride2(192, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, M, 3, stride=2, padding=1),
        )

        self.g_s = nn.Sequential(
            DepthConvBlock3(M, N, inplace=inplace),
            ResidualBlockUpsample1(N, N, 2, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample1(N, 192, 2, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockUpsample1(192, 128, 2, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockUpsample1(128, 128, 2, inplace=inplace),
            conv3x3(128, 3),
        )

        self.h_a = nn.Sequential(
            DepthConvBlock4(M, N, inplace=inplace),
            # nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            # nn.LeakyReLU(inplace=True),
            DepthConvBlock4(256, M),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            # nn.LeakyReLU(inplace=True),
            DepthConvBlock4(256, M),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                DepthConvBlock2(M + slice_depth * min(i, self.max_support_slices), 224, inplace=inplace),
                DepthConvBlock2(224, 128, inplace=inplace),
                DepthConvBlock2(128, slice_depth, inplace=inplace),

                # conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(224, 128, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                DepthConvBlock2(M + slice_depth * min(i, self.max_support_slices), 224, inplace=inplace),
                DepthConvBlock2(224, 128, inplace=inplace),
                DepthConvBlock2(128, slice_depth, inplace=inplace),

                # conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(224, 128, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                DepthConvBlock2(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, inplace=inplace),
                DepthConvBlock2(224, 128, inplace=inplace),
                DepthConvBlock2(128, slice_depth, inplace=inplace),

                # conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(224, 128, stride=1, kernel_size=3),
                # nn.LeakyReLU(inplace=True),
                # conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


if __name__ == "__main__":
    fun = 1

    h, w = 512, 768
    # h, w = 240, 416
    model = NIC2()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x = torch.rand((1, 3, h, w))
    out = model(x)
    print(out['x_hat'].shape)
    exit()

    x = torch.rand((1, 1, 3, h, w)).cuda()
    flops, params = profile(model, inputs=x, verbose=False)
    print('[profile] computation cost: {:.2f} GMACs, parameters: {:.2f} M'.format(
        flops / 10 ** 9, params / 10 ** 6))
    exit()

    if fun == 0:
        model = Cheng2020Anchor().cuda()
        print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        state_dict = torch.load(
            r'D:\Project\Pytorch\DeepVideoCoding\DCVC\checkpoints\cheng2020-anchor-3-e49be189.pth.tar',
            map_location='cuda:0')
        state_dict = load_pretrained(state_dict)
        model.load_state_dict(state_dict)

        img = Image.open('./stmalo_fracape.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).cuda()
        print(x.size())
        with torch.no_grad():
            out_net = model.forward(x)
        out_net['x_hat'].clamp_(0, 1)
        print(out_net.keys())

        print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
        print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
        print(f'Bit-rate: {compute_bpp(out_net):.3f} bpp')

        rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
        rec_net.save('./rec_stmalo_fracape.png', quality=95)

        diff = torch.mean((out_net['x_hat'] - x).abs(), axis=1).squeeze().cpu()
        fix, axes = plt.subplots(1, 3, figsize=(16, 12))
        for ax in axes:
            ax.axis('off')

        axes[0].imshow(img)
        axes[0].title.set_text('Original')
        axes[1].imshow(rec_net)
        axes[1].title.set_text('Reconstructed')
        axes[2].imshow(diff, cmap='viridis')
        axes[2].title.set_text('Difference')
        plt.show()
