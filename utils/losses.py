from typing import List, Tuple, Optional, Sequence

import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import joblib

import config
from .harmonic_helpers import alm_len_from_lmax, recombine


def ell_index_for_lmax(lmax: int) -> torch.Tensor:
    """Return ell for each coefficient in healpy's packed alm ordering."""
    return torch.tensor(
        [ell for m in range(lmax + 1) for ell in range(m, lmax + 1)],
        dtype=torch.long,
    )


class WeightedCoefficientLoss(nn.Module):
    """Bounded ell-aware coefficient reconstruction loss.

    This keeps the dense coefficient-space MSE objective, but gives selected
    ell bands more influence. Weights are normalized to mean one so changing
    band emphasis does not silently rescale the optimizer step.
    """

    def __init__(
        self,
        bands: Optional[Sequence[Tuple[int, int]]] = None,
        band_weights: Optional[Sequence[float]] = None,
        b_channel_weight: float = 1.0,
        lmax: Optional[int] = None,
    ):
        super().__init__()
        self.lmax = config.LMAX if lmax is None else lmax
        self.b_channel_weight = float(b_channel_weight)
        bands = config.BANDS if bands is None else bands
        if band_weights is None:
            band_weights = [4.0, 2.0, 1.0, 1.0]
        if len(bands) != len(band_weights):
            raise ValueError("bands and band_weights must have the same length.")

        ell_index = ell_index_for_lmax(self.lmax)
        alm_weights = torch.ones(alm_len_from_lmax(self.lmax), dtype=torch.float32)
        for (lmin, lmax_exclusive), weight in zip(bands, band_weights):
            if lmin < 0 or lmax_exclusive <= lmin or lmax_exclusive > self.lmax + 1:
                raise ValueError(f"Invalid ell band ({lmin}, {lmax_exclusive}) for lmax={self.lmax}.")
            alm_weights[(ell_index >= lmin) & (ell_index < lmax_exclusive)] = float(weight)
        alm_weights = alm_weights / alm_weights.mean()

        channel_weights = torch.ones(config.INPUT_DIM, dtype=torch.float32)
        if config.INPUT_DIM >= 4:
            channel_weights[2:] = self.b_channel_weight

        self.register_buffer("alm_weights", alm_weights.view(1, -1, 1))
        self.register_buffer("channel_weights", channel_weights.view(1, 1, -1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = self.alm_weights.to(device=x.device, dtype=x.dtype) * self.channel_weights.to(device=x.device, dtype=x.dtype)
        return torch.mean(weights * torch.square(x - y))


class SpectralBinLoss(nn.Module):
    def __init__(self, bands: List[Tuple[int, int]],
                 band_weights: Optional[List[float]] = None,
                 gamma: Optional[float] = 1.0,
                 lmax: Optional[int] = None,
                 eps: Optional[float] = 1e-8,
                 decay_sharpness: Optional[int] = 0.4):
        """
        weighted combination loss for binnined BB log-MSE and EB leakage penalty

        Args:
            bands : list of multipole ranges depicting bins, e.g. [(0, 10), (11, 20), (21, 30)]
            band_weights : list of weights len of bands for each bin summing to 1, e.g. [0.5, 0.35, 0.15],
                defaults to exp decay from front
            gamma : weight for EB leakage penalty, default 1.0
            lmax : maximum multipole, default None
            eps : small value to prevent log of zero, note in clamp call it uses eps**2, default 1e-8
            decay_sharpness : decay sharpness of band weight decay from front, lower means less front weight, default 0.4
        """
        super().__init__()
        self.bands = bands
        if band_weights is None:
            priorities = np.arange(len(bands), 0, -1)
            self.band_weights = np.exp(decay_sharpness * priorities)
            self.band_weights /= self.band_weights.sum()
        else:
            self.band_weights = band_weights
        self.gamma = gamma
        self.lmax = config.LMAX if lmax is None else lmax
        self.nside = config.NSIDE
        self.eps = eps

        mask = hp.read_map(config.MASK_FILE, field=1)
        self.low_mask = hp.ud_grade(mask, nside_out=self.nside, dtype=np.int32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device  # b, n, 4
        if config.SCALE:
            scalers = joblib.load(config.SCALER_FILE)
            target_scaler = scalers['target_scaler']
            outputs = target_scaler.inverse_transform(x)
            targets = target_scaler.inverse_transform(y)

            # healpy needs cpu numpy
            outputs, targets = outputs.detach().cpu().numpy(), targets.detach().cpu().numpy()
        else:
            outputs, targets = x.detach().cpu().numpy(), y.detach().cpu().numpy()

        b = len(outputs)
        alm_len = alm_len_from_lmax(self.lmax)
        alm_b_out = np.zeros((b, alm_len), dtype=np.complex128)
        alm_b_targ = np.zeros((b, alm_len), dtype=np.complex128)
        for i in range(b):
            alm_b_out[i] = recombine(outputs[i])[1]
            alm_b_targ[i] = recombine(targets[i])[1]

        b_out = np.zeros((b, self.lmax + 1), dtype=np.float64)
        b_targ = np.zeros((b, self.lmax + 1), dtype=np.float64)
        for i in range(b):
            b_out_masked_map = hp.alm2map(alm_b_out[i], nside=self.nside) * self.low_mask
            b_targ_masked_map = hp.alm2map(alm_b_targ[i], nside=self.nside) * self.low_mask
            b_out[i] = hp.anafast(b_out_masked_map, lmax=self.lmax)
            b_targ[i] = hp.anafast(b_targ_masked_map, lmax=self.lmax)

        l2_bb_out = 0
        for i, (lmin, lmax) in enumerate(self.bands):
            b_o_band = b_out[:, lmin:lmax]
            b_t_band = b_targ[:, lmin:lmax]

            l2_bb_out_metric = (np.sum((b_o_band - b_t_band) ** 2, axis=-1) /
                                (np.sum(b_t_band ** 2, axis=-1) + self.eps ** 2))
            l2_bb_out += self.band_weights[i] * l2_bb_out_metric.mean()

        return torch.tensor(l2_bb_out, device=device, dtype=torch.float32)


def zero_one_approximation_loss(guess, target, sigma):
    """
    Zero One Approximation loss, "well" shaped around target
    :param guess: torch.Tensor, guess
    :param target: torch.Tensor, target
    :param sigma: width of well around target, higher value means larger width
    :return: loss
    """
    diff_squared = (guess - target) ** 2
    loss = 1 - torch.exp(-diff_squared / (2 * sigma ** 2))
    return loss.mean()


def rmwe_loss(g, t, reduction='mean', eps=1e-8):
    """
    Relative mean weighted error
    :param g: array-like, guess
    :param t: array-like, target
    :param reduction: str, 'mean' or 'sum'
    :param eps: float, small number to avoid division by zero, default 1e-8
    :return: rmwe of guess from target
    """
    if reduction == 'mean':
        return torch.mean(torch.square(t - g) / torch.square(t).clamp(min=eps ** 2))
    elif reduction == 'sum':
        return torch.sum(torch.square(t - g) / torch.square(t).clamp(min=eps ** 2))
    else:
        raise ValueError('reduction must be either "mean" or "sum"')


def mape_loss(g, t, reduction='mean', eps=1e-8):
    """
    MAPE loss
    :param g: array-like, guess
    :param t: array-like, target
    :param reduction: str, only 'mean'
    :param eps: float, small number to avoid division by zero, default 1e-10
    :return: MAPE of guess from target
    """
    if reduction == 'mean':
        return torch.mean(torch.abs((g - t) / (t + eps ** 2))) * 100
    else:
        raise NotImplementedError('Only mean reduction is supported.')


def calc_mae(g: torch.Tensor, t: torch.Tensor, axis=None) -> torch.Tensor:
    """
    mae
    :param g: torch.Tensor, guess
    :param t: torch.Tensor, target
    :param axis: int, dimension along which to calculate MAE, None for entire mean
    :return: torch.Tensor, mae of guess from target
    """
    return torch.mean(torch.abs(t - g), dim=axis)


def calc_mse(g: torch.Tensor, t: torch.Tensor, axis=None) -> torch.Tensor:
    """
    mse
    :param g: torch.Tensor, guess
    :param t: torch.Tensor, target
    :param axis: int, dimension along which to calculate MSE, None for entire mean
    :return: torch.Tensor, mse of guess from target
    """
    return torch.mean((t - g) ** 2, dim=axis)


def calc_mape(g: torch.Tensor, t: torch.Tensor, eps=1e-8, axis=None) -> torch.Tensor:
    """
    mape
    :param g: torch.Tensor, guess
    :param t: torch.Tensor, target
    :param eps: float, epsilon in denominator to avoid division by zero, default 1e-8
    :param axis: int, dimension along which to calculate MAPE, None for entire mean
    :return: torch.Tensor, mape of guess from target
    """
    return torch.mean(torch.abs((t - g) / (t + eps ** 2)), dim=axis) * 100
