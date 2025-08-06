import torch
import torch.nn as nn

import config
from utils import alm_len_from_nside, make_module

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class Degrade(nn.Module):
    def __init__(self, nside, degradation_factor):
        """
        degradation module to downsample nside resolution by degradation_factor
        Args:
            nside (int): Original healpix NSIDE.
            degradation_factor (int): Downsampling factor.
        """
        super().__init__()
        self._new_len = alm_len_from_nside(nside // degradation_factor)

    def forward(self, x):
        return x[..., :self._new_len]


class Upgrade(nn.Module):
    def __init__(self, nside, upgrade_factor):
        """
        upgrade module to upsample (zero-pad) nside resolution by upgrade_factor
        Args:
            nside (int): Original healpix NSIDE.
            upgrade_factor (int): Upsampling factor.
        """
        super().__init__()
        self.N_small = alm_len_from_nside(nside)
        self.N_large = alm_len_from_nside(nside * upgrade_factor)

    def forward(self, x):
        B, C, _ = x.shape
        out = x.new_zeros((B, C, self.N_large))
        out[..., :self.N_small] = x
        return out


class LearnableUpgrade(nn.Module):
    def __init__(self, nsides, upgrade_factor, channels):
        """
        upgrade module to upsample (learned padding) nside resolution by upgrade_factor
        Args:
            nsides (int): Original healpix NSIDE.
            upgrade_factor (int): Upsampling factor.
            channels (int): Number of channels.
        """
        super().__init__()
        self.N_small = alm_len_from_nside(nsides)
        self.N_large = alm_len_from_nside(nsides * upgrade_factor)

        # learnable params to act as extended values
        self.expand = nn.Parameter(torch.zeros(1, channels, self.N_large - self.N_small))

        # conv sequence
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, C, _ = x.shape
        pad = self.expand.expand(B, -1, -1)
        x_full = torch.cat([x, pad], dim=-1)
        return self.proj(x_full)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_block_levels=config.N_CONV_LAYERS_IN_ONE_BLOCK, activation=nn.ReLU,
                 dilation=None, bias=config.BIAS):
        """
        Two-layer 1D convolutional block with activation. Uses dilation in kernel.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            kernel_size (int, optional): Kernel size. Must be odd.
            conv_block_levels (int, optional): Number of conv layers in one block.
            activation (Callable[[], nn.Module], optional): Activation module class. Default nn.ReLU.
            dilation (int, optional): Dilation factor. Defaults to one less than the integer half of kernel size.
            bias (bool, optional): Whether to use bias or not.
        """
        super().__init__()

        # preserve shape by deriving p/d from k
        if dilation is None:
            dilation = (kernel_size // 2) - 1 if kernel_size > 3 else 1
        padding = (dilation * (kernel_size - 1)) // 2

        N_CONV_LAYERS_IN_ONE_BLOCK = conv_block_levels

        # dilation and padding same value to preserve shape
        self.chain = []
        for _ in range(N_CONV_LAYERS_IN_ONE_BLOCK):
            self.chain.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=bias))
            self.chain.append(make_module(activation))
            in_ch = out_ch

        self.block = nn.Sequential(*self.chain)

    def forward(self, x):
        return self.block(x)
