#
# (c) 2025-2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by
# Triad National Security, LLC for the U.S. Department of Energy/National
# Nuclear Security Administration. All rights in the program are reserved
# by Triad National Security, LLC, and the U.S. Department of Energy/
# National Nuclear Security Administration.
# The Government is granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly
# and display publicly, and to permit others to do so.
#
# Author:
#   Kai Gao, kaigao@lanl.gov
#
# Building blocks of SCF-Net in 2D. Batch normalization uses fixed running statistics at inference, and the
# modal fusion is local, so the output at a pixel depends only on the image and the seismicity around it, and
# not on the size or the content of the whole image.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_like(src, tar):

    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


def pad_like(source, target):

    pY = target.size()[2] - source.size()[2]
    pX = target.size()[3] - source.size()[3]

    return F.pad(source, [pX // 2, pX - pX // 2, pY // 2, pY - pY // 2])


def maxpool(l, kernel_size=2):

    return F.max_pool2d(l, kernel_size=kernel_size)


class conv(nn.Module):
    """Convolution, batch normalization (if bn) and activation ('relu', 'sigmoid' or None)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, activation='relu', bn=True):

        super(conv, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same')]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.c = nn.Sequential(*layers)

    def forward(self, x):

        # A sigmoid output is computed in float32, so that under mixed precision a probability near 0 or 1 keeps
        # its gradient (in bfloat16, sigmoid(x) rounds to 1 for x >= 7, and its gradient vanishes)
        if isinstance(self.c[-1], nn.Sigmoid):
            return self.c[-1](self.c[:-1](x).float())
        return self.c(x)


class resu1(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):

        super(resu1, self).__init__()

        n = out_channels

        self.conv0 = conv(in_channels, n)
        self.conv1 = conv(n, n)
        self.conv2 = conv(n, 2 * n)
        self.conv3 = conv(2 * n, 4 * n)
        self.convd = conv(4 * n, 16 * n, dilation=2)
        self.up3 = conv(4 * n + 16 * n, 4 * n)
        self.up2 = conv(2 * n + 4 * n, 2 * n)
        self.up1 = conv(2 * n + n, n)
        self.up0 = conv(n, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):

        conv0 = self.conv0(x)

        conv1 = self.conv1(conv0)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        d = self.convd(pool3)

        up3 = pad_like(self.up(d), conv3)
        up3 = self.up3(torch.cat((up3, conv3), dim=1))

        up2 = pad_like(self.up(up3), conv2)
        up2 = self.up2(torch.cat((up2, conv2), dim=1))

        up1 = pad_like(self.up(up2), conv1)
        up1 = self.up1(torch.cat((up1, conv1), dim=1))

        up = pad_like(self.up0(up1), conv0)
        up = up + conv0

        return up


class resu2(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):

        super(resu2, self).__init__()

        n = out_channels

        self.conv0 = conv(in_channels, n)
        self.conv1 = conv(n, n)
        self.conv2 = conv(n, 2 * n)
        self.convd = conv(2 * n, 8 * n, dilation=2)
        self.up2 = conv(2 * n + 8 * n, 2 * n)
        self.up1 = conv(2 * n + n, n)
        self.up0 = conv(n, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):

        conv0 = self.conv0(x)

        conv1 = self.conv1(conv0)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        d = self.convd(pool2)

        up2 = pad_like(self.up(d), conv2)
        up2 = self.up2(torch.cat((up2, conv2), dim=1))

        up1 = pad_like(self.up(up2), conv1)
        up1 = self.up1(torch.cat((up1, conv1), dim=1))

        up = pad_like(self.up0(up1), conv0)
        up = up + conv0

        return up


class resu3(nn.Module):
    """Bottleneck block with dilated convolutions (dilations 2, 4 and 8)."""

    def __init__(self, in_channels=1, out_channels=1):

        super(resu3, self).__init__()

        n = out_channels

        self.conv0 = conv(in_channels, n)
        self.conv1 = conv(n, n)
        self.conv2 = conv(n, 2 * n, dilation=2)
        self.conv3 = conv(2 * n, 2 * n, dilation=4)
        self.convd = conv(2 * n, 4 * n, dilation=8)
        self.up3 = conv(2 * n + 4 * n, 2 * n, dilation=4)
        self.up2 = conv(2 * n + 2 * n, 2 * n, dilation=2)
        self.up1 = conv(2 * n + n, n)
        self.up0 = conv(n, out_channels)

    def forward(self, x):

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        d = self.convd(conv3)

        up3 = self.up3(torch.cat((d, conv3), dim=1))
        up2 = self.up2(torch.cat((up3, conv2), dim=1))
        up1 = self.up1(torch.cat((up2, conv1), dim=1))
        up = self.up0(up1)
        up = up + conv0

        return up


class SpatialAttention(nn.Module):

    def __init__(self, in_channels, kernel_size=7):

        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.convout = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv(x)
        x = torch.cat((avg_out, max_out, x), dim=1)

        return self.convout(x)


class LocalSpatialChannelAttention(nn.Module):
    """
    Modal fusion of the image features e1 and the seismicity features e2 by a spatial-channel attention that is
    local and residual: the channel attention is computed at every pixel with 1 x 1 convolutions, so the fused
    features at a pixel depend only on the encoder features around that pixel, and the attention a in [0, 1]
    scales the features by 1 + a, so it can emphasize features but cannot remove the image features where
    there is no seismicity.
    """

    def __init__(self, in_channels_encoder, out_channels, channel_ratio=16, spatial_kernel=7):

        super(LocalSpatialChannelAttention, self).__init__()

        self.sa_encoder1 = SpatialAttention(in_channels_encoder[0], spatial_kernel)
        self.sa_encoder2 = SpatialAttention(in_channels_encoder[1], spatial_kernel)

        self.n = max(in_channels_encoder) // channel_ratio
        self.ca_encoder1 = nn.Conv2d(in_channels_encoder[0], self.n, kernel_size=1, bias=False)
        self.ca_encoder2 = nn.Conv2d(in_channels_encoder[1], self.n, kernel_size=1, bias=False)

        self.cconv = nn.Conv2d(self.n, out_channels, kernel_size=1, bias=False)

    def forward(self, e1, e2):

        sag = torch.sigmoid(self.sa_encoder1(e1) + self.sa_encoder2(e2))
        cag = torch.sigmoid(self.cconv(F.relu(self.ca_encoder1(e1) + self.ca_encoder2(e2))))

        return torch.cat((e1, e2), dim=1) * (1.0 + sag * cag)


class mtl_decoder_fusion(nn.Module):
    """Decoder that fuses the image and seismicity encoder features at the two finest levels."""

    def __init__(self, l1, l2, l3, out_ch=1, last_kernel_size=1):

        super(mtl_decoder_fusion, self).__init__()

        self.out_ch = out_ch
        self.last_kernel_size = last_kernel_size
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.decoder2 = resu2(2 * self.l2 + self.l3, self.l2)
        self.decoder1 = resu1(2 * self.l1 + self.l2, self.l1)

        self.out = nn.Sequential(nn.Conv2d(self.l1, self.l1, 3, padding='same'),
                                 nn.Conv2d(self.l1, self.l1, 3, padding='same'),
                                 nn.Conv2d(self.l1, self.out_ch, self.last_kernel_size, padding='same'))

        self.fusion2 = LocalSpatialChannelAttention((self.l2, self.l2), 2 * self.l2, channel_ratio=8)
        self.fusion1 = LocalSpatialChannelAttention((self.l1, self.l1), 2 * self.l1, channel_ratio=8)

    def forward(self, x, out_encoder1_a, out_encoder2_a, out_encoder1_b, out_encoder2_b, l):

        up3 = upsample_like(l, out_encoder2_a)
        f2 = self.fusion2(out_encoder2_a, out_encoder2_b)
        up3 = self.decoder2(torch.cat((up3, f2), dim=1))

        up2 = upsample_like(up3, out_encoder1_a)
        f1 = self.fusion1(out_encoder1_a, out_encoder1_b)
        up2 = self.decoder1(torch.cat((up2, f1), dim=1))

        return upsample_like(self.out(up2), x)


class mtl_subdecoder(nn.Module):
    """Output head: four residual convolutions and an output convolution with an activation."""

    def __init__(self, in_ch, out_ch=1, bn=False, mid_activation=None, activation=None):

        super(mtl_subdecoder, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Sequential(conv(self.in_ch, 2 * self.in_ch, activation=mid_activation, bn=bn),
                                  conv(2 * self.in_ch, 2 * self.in_ch, activation=mid_activation, bn=bn),
                                  conv(2 * self.in_ch, 2 * self.in_ch, activation=mid_activation, bn=bn),
                                  conv(2 * self.in_ch, self.in_ch, activation=mid_activation, bn=bn))
        self.out = nn.Sequential(conv(self.in_ch, self.out_ch, activation=activation, bn=False))

    def forward(self, x):

        return self.out(x + self.conv(x))
