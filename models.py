import torch.nn.functional as F
import torch.optim
from lightning.pytorch import LightningModule
from torch import nn

import config
from modules import HRM
from utils import PredictorMixin, SpectralBinLoss, alm_span_from_m_band

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class HarmonicHRM(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = kwargs.get("input_dim", config.INPUT_DIM)
        hidden_dim = kwargs.get("hidden_dim", config.HIDDEN_DIM)
        nlevels = kwargs.get("num_levels", config.NUM_LEVELS)
        drop_rate = kwargs.get("drop_rate", config.DROP_RATE)
        activation = kwargs.get("activation", nn.ReLU)
        bands = kwargs.get("bands", config.BANDS)
        nsteps_per_cycle = kwargs.get("nsteps_per_cycle", 3)
        ncycles = kwargs.get("ncycles", 2)

        alm_bands = []
        for band in bands:
            alm_bands.append(alm_span_from_m_band(config.LMAX, band))

        self.net = HRM(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       num_layers=nlevels,
                       dropout=drop_rate,
                       activation=activation,
                       bands=alm_bands,
                       nsteps_per_cycle=nsteps_per_cycle,
                       ncycles=ncycles, )

        self.save_hyperparameters(kwargs)

    def forward(self, x):
        x = self.net(x)
        return x


class Leakless(HarmonicHRM, PredictorMixin):
    """
    Lightning training wrapper for models.

    Args:
        lr (float, optional): Learning rate. Default config.LEARNING_RATE.
        activation (Callable[[Tensor], Tensor], optional): Activation function. Default nn.ReLU.
        loss (Callable, optional): Loss function. Default torch.nn.functional.mse_loss.
        optimizer (Type[torch.optim.Optimizer], optional): Optimizer class. Default torch.optim.NAdam.
        scheduler (Type[torch.optim.lr_scheduler._LRScheduler], optional): LR scheduler class.
            Default torch.optim.lr_scheduler.ReduceLROnPlateau.
        loss_kwargs (dict, optional): Additional loss kwargs. Default {}.
        optimizer_kwargs (dict, optional): Additional optimizer kwargs. Default {}.
        scheduler_kwargs (dict, optional): Additional scheduler kwargs. Default {}.
        All other kwargs are passed to parent class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lr = kwargs.get('lr', config.LEARNING_RATE)
        self.activation = kwargs.get('activation', nn.ReLU)
        self.loss = kwargs.get('loss', F.mse_loss)
        self.val_loss = SpectralBinLoss(bands=config.BANDS)
        self.optimizer = kwargs.get('optimizer', torch.optim.NAdam)
        self.scheduler = kwargs.get('scheduler', torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.loss_kwargs = {}
        self.loss_kwargs.update(kwargs.get('loss_kwargs', {}))
        self.optimizer_kwargs = {'params': self.parameters(), 'lr': self.lr, 'weight_decay': config.WEIGHT_DECAY}
        self.optimizer_kwargs.update(kwargs.get('optimizer_kwargs', {}))
        self.scheduler_kwargs = {}
        self.scheduler_kwargs.update(kwargs.get('scheduler_kwargs', {}))

        self.save_hyperparameters(config.hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=config.ON_STEP)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.val_loss(y_hat, y.view_as(y_hat))
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.val_loss(y_hat, y.view_as(y_hat))
        self.log('test_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(**self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        if self.scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.s_config = {"scheduler": scheduler, "monitor": "val_loss"}
        elif self.scheduler == torch.optim.lr_scheduler.OneCycleLR or self.scheduler == torch.optim.lr_scheduler.CyclicLR:
            self.s_config = {'scheduler': scheduler, 'interval': 'step'}
        return {"optimizer": optimizer, "lr_scheduler": self.s_config}
