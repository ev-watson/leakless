from typing import Optional, Union

import joblib
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

import config


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
        device = next(self.parameters()).device
        scalers = joblib.load(config.SCALER_FILE)
        input_scaler = scalers['input_scaler']
        target_scaler = scalers['target_scaler']

        input_data = input_scaler.transform(X.to(dtype=torch.get_default_dtype(), device=device))
        with torch.no_grad():
            output_scaled = self.forward(input_data)
        return target_scaler.inverse_transform(output_scaled)


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
