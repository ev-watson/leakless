import torch.nn.functional as F
import torch.optim
from lightning.pytorch import LightningModule
from torch import nn

import config
from modules import ConvBlock, Degrade, Upgrade, LearnableUpgrade
from utils import PredictorMixin, SpectralBinLoss

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class SpectralUNet(LightningModule):
    def __init__(self, **kwargs):
        """
        U-Net style branch for CMB alm arrays configurable version inspired by Guo-Jian Wang et al 2022 ApJS 260 13

        Args:
            nsides (int, optional): healpix NSIDE. Default config.NSIDE.
            in_channels (int, optional): Number of input channels. Default config.IN_CHANNELS.
            base_channels (int, optional): Channels in first encoding level. Default config.BASE_CHANNELS.
            num_levels (int, optional): Depth of the encoder/decoder (number of levels). Default config.NUM_LEVELS.
            kernel_list (List[int], optional): Kernel sizes for each encoder/decoder level. Default [config.KERNEL_SIZE] * num_levels.
            activation (Callable[[], nn.Module], optional): Activation module class. Default nn.ReLU.
            bias (bool, optional): Whether to include bias in convolutional layers. Default config.BIAS.
            degradation_factor (int, optional): Downsampling upgrade_factor applied to NSIDE at each level. Default config.SAMPLE_FACTOR.
        """
        super().__init__()
        nside_eff = kwargs.get("nside_eff", config.NSIDE_EFF)
        in_ch = kwargs.get("in_channels", config.IN_CHANNELS)
        base_ch = kwargs.get("base_channels", config.BASE_CHANNELS)
        nlevels = kwargs.get("num_levels", config.NUM_LEVELS)
        factor = kwargs.get("sample_factor", config.SAMPLE_FACTOR)
        kernel_list = kwargs.get("kernel_list", config.KERNEL_LIST)
        conv_block_levels = kwargs.get("conv_block_levels", config.N_CONV_LAYERS_IN_ONE_BLOCK)
        drop_rate = kwargs.get("drop_rate", config.DROP_RATE)
        activation = kwargs.get("activation", nn.ReLU)
        bias = kwargs.get("bias", config.BIAS)
        learnable_upgrade = kwargs.get("learnable_upgrade", config.LEARN_UP)

        assert len(kernel_list) == nlevels, f"Wrong kernel list: {kernel_list}, for nlevels={nlevels}"

        # encoder: convblock then degrade
        self.encoders = nn.ModuleList()
        channels = [in_ch] + [base_ch * (2 ** i) for i in range(nlevels)]
        for i in range(nlevels):
            lvl_nsides = nside_eff // (factor ** i)
            self.encoders.append(nn.ModuleDict({
                "conv": ConvBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_list[i],
                    conv_block_levels=conv_block_levels,
                    activation=activation,
                    bias=bias,
                ),
                "degrade": Degrade(lvl_nsides, factor)
            }))

        # bottleneck at coarsest resolution
        bot_ch = base_ch * (2 ** nlevels)
        self.bottleneck = ConvBlock(
            in_ch=channels[-1],
            out_ch=bot_ch,
            kernel_size=kernel_list[-1],
            conv_block_levels=conv_block_levels,
            activation=activation,
            bias=bias,
        )
        self.dropout = nn.Dropout(p=drop_rate)

        # decoder: upsample, concat skip, convblock
        self.decoders = nn.ModuleList()
        for i in range(nlevels - 1, -1, -1):
            lvl_nsides = nside_eff // (factor ** (i + 1))
            up_ch = bot_ch if i == nlevels - 1 else channels[i + 2]
            self.decoders.append(nn.ModuleDict({
                "upgrade": LearnableUpgrade(lvl_nsides, factor, up_ch) if learnable_upgrade else Upgrade(lvl_nsides, factor),
                "conv": ConvBlock(
                    in_ch=up_ch + channels[i + 1],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_list[i],
                    conv_block_levels=conv_block_levels,
                    activation=activation,
                    bias=bias
                ),
                # currently disabled
                # 'drop': nn.Dropout(p=drop_rate) if i == nlevels - 1 else nn.Identity(),
            }))

        # final 1x1 conv to restore input channels
        self.final = nn.Conv1d(base_ch, in_ch, kernel_size=1, bias=bias)

        self.save_hyperparameters(kwargs)

    def forward(self, x):
        skips = []  # [B, C_in, N]
        # encoding
        for i, enc in enumerate(self.encoders):
            x = enc["conv"](x)
            skips.append(x)  # [B, BC*2**i, N/D**i]
            x = enc["degrade"](x)  # [B, BC*2**i, N/D**(i+1)]

        # bottleneck
        x = self.bottleneck(x)  # [B, BC*2**L, N/D**L]
        x = self.dropout(x)

        # decoding
        for idx, dec in enumerate(self.decoders):
            skip = skips[-(idx + 1)]
            x = dec["upgrade"](x)  # [B, BC*2**(L-i), N/D**(L-i-1)]
            x = torch.cat([x, skip], dim=1)  # [B, BC*2**(L-i) + BC*2**(L-i-1), N/D**(L-i-1)]
            x = dec["conv"](x)  # [B, BC*2**(L-i-1), N/D**(L-i-1)]
            # x = dec['drop'](x)

        x = self.final(x)
        return x


class Leakless(SpectralUNet, PredictorMixin):
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
