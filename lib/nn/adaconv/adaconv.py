from math import ceil, floor

import torch
from torch import nn
from torch.nn import functional as F


class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = self._normalize(x)

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        kernel_size = w_spatial.size(-1)
        padding = (kernel_size - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups, bias=bias)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups)
        return x

    def _normalize(self, x, eps=1e-5):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True)
        x_norm = (x - mean) / (std + eps)
        return x_norm
