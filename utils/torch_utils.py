from typing import Union, Callable, List, Tuple, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

import config

preferred = [
    SDPBackend.FLASH_ATTENTION,         # fastest when available
    SDPBackend.CUDNN_ATTENTION,         # PyTorch >= 2.5 adds cuDNN SDPA
    SDPBackend.EFFICIENT_ATTENTION,     # xFormers-style
    SDPBackend.MATH,                    # fallback
]


def build_rope_cache(max_len: int, dh: int, base: float = 10000.0, device=None, dtype=torch.float32):
    assert dh % 2 == 0
    idx = torch.arange(0, dh, 2, dtype=dtype, device=device)
    inv_freq = base ** (-idx / dh)
    pos = torch.arange(max_len, dtype=dtype, device=device)[:, None]
    freqs = pos * inv_freq[None, :]
    cos = torch.cos(freqs).repeat_interleave(2, dim=1)  # [N,dh]
    sin = torch.sin(freqs).repeat_interleave(2, dim=1)
    return cos, sin


def rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    # q,k: [B,H,N,dh]; cos/sin: [Nmax,dh]
    B, H, N, dh = q.shape
    cos = cos[offset:offset + N].to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin = sin[offset:offset + N].to(device=q.device, dtype=q.dtype)[None, None, :, :]

    def rot(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = -x_odd
        out[..., 1::2] = x_even
        return out

    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        inner = int(dim * mult)
        self.wg = nn.Linear(dim, inner, bias=bias)
        self.wu = nn.Linear(dim, inner, bias=bias)
        self.wo = nn.Linear(inner, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wo(F.silu(self.wg(x)) * self.wu(x)))


class SDPAEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 mlp_ratio: float = 2.0, mlp_drop: float = 0.0,
                 rope_base: float = 10000.0, causal: bool = False):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        self.h = heads
        self.dh = dim // heads
        self.causal = causal

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_ratio, dropout=mlp_drop, bias=False)

        self.rope_base = rope_base
        self._rope_cache = None

    def rope(self, n: int, device, dtype=torch.float32):
        if self._rope_cache is None or self._rope_cache[0].size(0) < n:
            self._rope_cache = build_rope_cache(n, self.dh, base=self.rope_base, device=device, dtype=dtype)
        return self._rope_cache

    def forward(self, x):
        # x: [B,N,D]
        B, N, D = x.shape

        # Attention
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)  # [B,H,N,dh]
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(N, q.device, torch.float32)
        q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        q, k = rope_apply(q, k, cos, sin)

        with sdpa_kernel(preferred):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                is_causal=self.causal,
            )

        y = y.transpose(1, 2).contiguous().view(B, N, D).to(x.dtype)
        y = self.proj_drop(self.o(y))
        x = self.norm1(x + y)
        x = self.norm2(x + self.mlp(x))
        return x


class LambdaLayer(nn.Module):
    """Wrap any Callable[[Tensor],Tensor] so it behaves like an nn.Module."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def make_module(act):
    """
    Given one of:
      • an nn.Module subclass (e.g. nn.ReLU),
      • an nn.Module instance (e.g. nn.ReLU()),
      • any Callable Tensor→Tensor (e.g. F.relu or a lambda),
    return an nn.Module instance you can safely put into nn.Sequential.
    """
    # already an instance?
    if isinstance(act, nn.Module):
        return act
    # a Module *class*?
    if isinstance(act, type) and issubclass(act, nn.Module):
        return act()
    # any other callable → wrap in LambdaLayer
    if callable(act):
        return LambdaLayer(act)
    raise ValueError(f"Cannot make a module out of {act!r}")


class Scaler:
    """
    Standard scaler for 3D tensor inputs of shape [B, F, N].
    fit() accepts numpy arrays or torch.Tensor.
    transform() and inverse_transform() require torch.Tensor and return torch.Tensor.
    """
    eps: float
    mean_: Optional[torch.Tensor]
    std_: Optional[torch.Tensor]
    is_fitted: bool

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> "Scaler":
        """
        Compute feature-wise mean and std from data.

        Args:
            data: numpy.ndarray or torch.Tensor of shape [B, F, N]

        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().to(dtype=torch.get_default_dtype())
        else:
            arr = torch.from_numpy(np.asarray(data)).to(dtype=torch.get_default_dtype())

        if arr.ndim != 3:
            raise ValueError(f"Input must be 3D [B, F, N], got {tuple(arr.shape)}")

        mean = arr.mean(dim=(0, 2), keepdim=True)
        std = arr.std(dim=(0, 2), unbiased=False, keepdim=True) + self.eps

        self.mean_ = mean
        self.std_ = std
        self.is_fitted = True
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale new data using previously fitted parameters.

        Args:
            data: torch.Tensor of shape [B, F, N]

        Returns:
            torch.Tensor: scaled data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"transform() requires a torch.Tensor, got {type(data)}")

        mean = self.mean_.to(device=data.device, dtype=data.dtype)  # type: ignore
        std = self.std_.to(device=data.device, dtype=data.dtype)  # type: ignore
        return (data - mean) / std

    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Fit to data, then return scaled data.

        Args:
            data: numpy.ndarray or torch.Tensor of shape [B, F, N]

        Returns:
            torch.Tensor: scaled data
        """
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.asarray(data)).to(dtype=torch.get_default_dtype())
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Revert the scaling transformation.

        Args:
            data: torch.Tensor of scaled data, shape [B, F, N]

        Returns:
            torch.Tensor: original-scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"inverse_transform() requires a torch.Tensor, got {type(data)}")

        mean = self.mean_.to(device=data.device, dtype=data.dtype)  # type: ignore
        std = self.std_.to(device=data.device, dtype=data.dtype)  # type: ignore
        return data * std + mean


class PredictorMixin:
    """
    Generic prediction mixin for use with unscaled inputs on a model that was trained with scaled inputs
    """

    def predict(self, X):
        """
        Prediction method that uses the same scaling function as training for use after model training.
        Must have scaler used with the original dataset saved as pkl file in cwd with name specified in config.
        :param X: torch.Tensor, list of value pairs to use as input (excluding target), must be raw numbers unscaled.
        :return: Tensor of predicted target values.
        """
        self.eval()
        if config.SCALE:
            device = next(self.parameters()).device
            scalers = joblib.load(config.SCALER_FILE)
            input_scaler = scalers['input_scaler']
            target_scaler = scalers['target_scaler']

            input_data = input_scaler.transform(X.to(dtype=torch.get_default_dtype(), device=device))
            with torch.no_grad():
                output_scaled = self.forward(input_data)
                output = target_scaler.inverse_transform(output_scaled)
        else:
            with torch.no_grad():
                output = self.forward(X)
        return output


class RollingBufferCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ds = trainer.datamodule.train_dataset
        if hasattr(ds, 'on_epoch_end'):
            ds.on_epoch_end()


class GradientNormCallback(Callback):
    def __init__(self):
        """
        Callback for logging gradient norm
        """
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        if trainer.training:
            total_norm = 0.0
            for param in pl_module.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            pl_module.log('grad_norm', total_norm, prog_bar=True, logger=True, sync_dist=True, on_epoch=True,
                          on_step=config.ON_STEP)
