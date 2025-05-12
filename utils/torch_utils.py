import joblib
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

import config


class Scaler:
    """
    Standard scaler for 3D array-like inputs of shape [B, F, N], where
      - B = batch size,
      - F = number of features,
      - N = feature length.

    Computes per-feature mean and population std (ddof=0) over axes (0, 2),
    then transforms with (x – mean) / std.

    Args:
        eps (float): small constant to avoid division by zero. Default: 1e-12.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_ = None  # shape [1, F, 1]
        self.std_ = None  # shape [1, F, 1]
        self.is_fitted = False

    def fit(self, data):
        """
        Compute feature-wise mean and std from data.

        Args:
            data: array-like, shape [B, F, N]

        Returns:
            self
        """
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise ValueError(f"Input must be 3D [B, F, N], got {arr.shape}")
        # mean over batch and length dims
        self.mean_ = arr.mean(axis=(0, 2), keepdims=True)
        # population std + eps
        self.std_ = arr.std(axis=(0, 2), keepdims=True) + self.eps
        self.is_fitted = True
        return self

    def transform(self, data):
        """
        Scale new data using previously fitted parameters.

        Args:
            data: array-like, shape [B, F, N]

        Returns:
            numpy.ndarray of same shape, scaled.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise ValueError(f"Input must be 3D [B, F, N], got {arr.shape}")
        return (arr - self.mean_) / self.std_

    def fit_transform(self, data):
        """
        Convenience: fit to data, then return scaled data.

        Args:
            data: array-like, shape [B, F, N]

        Returns:
            numpy.ndarray: scaled data
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        """
        Revert the scaling transformation.

        Args:
            data: array-like, scaled, shape [B, F, N]

        Returns:
            numpy.ndarray: original-scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise ValueError(f"Input must be 3D [B, F, N], got {arr.shape}")
        return arr * self.std_ + self.mean_


class PredictorMixin:
    """
    Generic prediction mixin for use with unscaled inputs on a model that was trained with scaled inputs
    If no scaling was involved this function is no different than calling self.forward with eval and no_grad)
    """
    input_slice = None
    output_slice = None

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
