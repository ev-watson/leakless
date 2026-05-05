"""Utility classes for training, scaling, and callbacks."""
from typing import Union, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

import config


class LambdaLayer(nn.Module):
    """Wrap any Callable[[Tensor], Tensor] as an nn.Module."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def make_module(act):
    """Convert a class, instance, or callable into an nn.Module."""
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
    """Standard scaler for 3D tensor inputs of shape [B, N, F].

    fit() accepts numpy arrays or torch.Tensor.
    transform() and inverse_transform() require torch.Tensor.
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
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().to(dtype=torch.float32)
        else:
            arr = torch.from_numpy(np.asarray(data)).to(dtype=torch.float32)

        if arr.ndim != 3:
            raise ValueError(f"Input must be 3D [B, N, F], got {tuple(arr.shape)}")

        self.mean_ = arr.mean(dim=(0, 1), keepdim=True)
        self.std_ = arr.std(dim=(0, 1), unbiased=False, keepdim=True) + self.eps
        self.is_fitted = True
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        mean = self.mean_.to(device=data.device, dtype=data.dtype)  # type: ignore
        std = self.std_.to(device=data.device, dtype=data.dtype)  # type: ignore
        return (data - mean) / std

    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.asarray(data)).to(dtype=torch.float32)
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        mean = self.mean_.to(device=data.device, dtype=data.dtype)  # type: ignore
        std = self.std_.to(device=data.device, dtype=data.dtype)  # type: ignore
        return data * std + mean


class PredictorMixin:
    """Fallback prediction mixin for models that do not override predict()."""

    def predict(self, X):
        self.eval()
        if config.SCALE:
            device = next(self.parameters()).device
            scalers = joblib.load(config.SCALER_FILE)
            input_scaler = scalers['input_scaler']
            target_scaler = scalers['target_scaler']
            input_data = input_scaler.transform(X.to(dtype=torch.float32, device=device))
            with torch.no_grad():
                output_scaled = self.forward(input_data)
                if isinstance(output_scaled, tuple):
                    output_scaled = output_scaled[0]
                output = target_scaler.inverse_transform(output_scaled)
        else:
            with torch.no_grad():
                output = self.forward(X)
                if isinstance(output, tuple):
                    output = output[0]
        return output


class RollingBufferCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ds = trainer.datamodule.train_dataset
        if hasattr(ds, 'on_epoch_end'):
            ds.on_epoch_end()


class GradientNormCallback(Callback):
    """Log the total gradient L2 norm after each backward pass."""

    def on_after_backward(self, trainer, pl_module):
        if trainer.training:
            total_norm = 0.0
            for param in pl_module.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            pl_module.log('grad_norm', total_norm, prog_bar=True, logger=True,
                          sync_dist=True, on_epoch=True, on_step=config.ON_STEP)
