import torch.nn.functional as F
import torch.optim
from lightning.pytorch import LightningModule
from torch import nn

import config
from utils import alm_len_from_nsides, print_block, PredictorMixin

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class Degrade(nn.Module):
    def __init__(self, nsides, degradation_factor):
        """
        degradation module to downsample nsides resolution by degradation_factor
        Args:
            nsides (int): Original healpix NSIDE.
            degradation_factor (int): Downsampling factor.
        """
        super().__init__()
        new_nsides = nsides // degradation_factor
        self._new_len = alm_len_from_nsides(new_nsides)

    def forward(self, x):
        return x[..., :self._new_len]


class Upgrade(nn.Module):
    def __init__(self, nsides, upgrade_factor):
        """
        upgrade module to upsample (zero-pad) nsides resolution by upgrade_factor
        Args:
            nsides (int): Original healpix NSIDE.
            upgrade_factor (int): Upsampling factor.
        """
        super().__init__()
        new_nsides = nsides * upgrade_factor
        self.N_small = alm_len_from_nsides(nsides)
        self.N_large = alm_len_from_nsides(new_nsides)

    def forward(self, x):
        B, C, _ = x.shape
        out = x.new_zeros((B, C, self.N_large))
        out[..., :self.N_small] = x
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation=nn.ReLU, dilation=None, bias=config.BIAS):
        """
        Two-layer 1D convolutional block with activation. Uses dilation in kernel.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            kernel_size (int, optional): Kernel size. Must be odd.
            activation (Callable[[], nn.Module], optional): Activation module class. Default nn.ReLU.
            dilation (int, optional): Dilation factor. Defaults to one less than the integer half of kernel size.
            bias (bool, optional): Whether to use bias or not.
        """
        super().__init__()

        # preserve shape by deriving p/d from k
        if dilation is None:
            dilation = (kernel_size // 2) - 1 if kernel_size > 3 else 1
        padding = (dilation*(kernel_size - 1)) // 2

        N_CONV_LAYERS_IN_ONE_BLOCK = config.N_CONV_LAYERS_IN_ONE_BLOCK

        # dilation and padding same value to preserve shape
        self.chain = []
        for _ in range(N_CONV_LAYERS_IN_ONE_BLOCK):
            self.chain.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=bias))
            self.chain.append(activation())
            in_ch = out_ch

        self.block = nn.Sequential(*self.chain)

    def forward(self, x):
        return self.block(x)


class DenseHead(nn.Module):
    def __init__(self, channels: int, length: int, dense_dim: int, drop_rate: float, activation: nn.Module = nn.ReLU):
        """
        bottleneck dense head for global mixing
        Args:
            channels (int): feature channels at bottleneck.
            length (int): length of feature sequence.
            dense_dim (int): hidden units.
            drop_rate (float): dropout probability.
            activation (Callable[[], nn.Module], optional): activation module class. default nn.ReLU.
        """
        super().__init__()
        self.channels = channels
        self.length = length
        self.fc = nn.Linear(channels * length, dense_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.output = nn.Linear(dense_dim, channels * length)
        self.activation = activation()

    def forward(self, x):
        B = x.shape[0]
        x = x.flatten(1)
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x.view(B, self.channels, self.length)


class CrossScaleGate(nn.Module):
    def __init__(self, channels: int, thresh_idx: int):
        """
        cross-scale gating between low and high bands
        Args:
            channels (int): number of channels.
            thresh_idx (int): split index along length.
        """
        super().__init__()
        self.thresh = thresh_idx
        self.high_to_low = nn.Linear(channels, channels)
        self.low_to_high = nn.Linear(channels, channels)

    def forward(self, x):
        low, high = x[:, :, :self.thresh], x[:, :, self.thresh:]
        mean_low = low.mean(dim=2)      # [B, C]
        mean_high = high.mean(dim=2)    # [B, C]
        gate_low = torch.sigmoid(self.high_to_low(mean_high)).unsqueeze(-1)     # use high-ell summary in low gate
        gate_high = torch.sigmoid(self.low_to_high(mean_low)).unsqueeze(-1)     # use low-ell summary in high gate
        low = low * gate_low        # [B, C, N_low]
        high = high * gate_high     # [B, C, N_high]
        return torch.cat([low, high], dim=2)


class SpectralUNet(LightningModule):
    def __init__(self, **kwargs):
        """
        U-Net style branch for CMB alm arrays configurable version inspired by Guo-Jian Wang et al 2022 ApJS 260 13

        Args:
            nsides (int, optional): healpix NSIDE. Default config.NSIDES.
            in_channels (int, optional): Number of input channels. Default config.IN_CHANNELS.
            base_channels (int, optional): Channels in first encoding level. Default config.BASE_CHANNELS.
            num_levels (int, optional): Depth of the encoder/decoder (number of levels). Default config.NUM_LEVELS.
            kernel_list (List[int], optional): Kernel sizes for each encoder/decoder level. Default [config.KERNEL_SIZE] * num_levels.
            activation (Callable[[], nn.Module], optional): Activation module class. Default nn.ReLU.
            bias (bool, optional): Whether to include bias in convolutional layers. Default config.BIAS.
            degradation_factor (int, optional): Downsampling upgrade_factor applied to NSIDE at each level. Default config.SAMPLE_FACTOR.
        """
        super().__init__()
        nsides = kwargs.get("nsides", config.NSIDES)
        in_ch = kwargs.get("in_channels", config.IN_CHANNELS)
        base_ch = kwargs.get("base_channels", config.BASE_CHANNELS)
        nlevels = kwargs.get("num_levels", config.NUM_LEVELS)
        factor = kwargs.get("sample_factor", config.SAMPLE_FACTOR)
        kernel_list = kwargs.get("kernel_list", [config.KERNEL_SIZE] * nlevels)
        dense_dim = kwargs.get("dense_dim", config.DENSE_DIM)
        drop_rate = kwargs.get("drop_rate", config.DROP_RATE)
        gate_resolution = kwargs.get("gate_resolution", config.GATE_RESOLUTION)
        activation = kwargs.get("activation", nn.ReLU)
        bias = kwargs.get("bias", config.BIAS)

        # encoder: convblock then degrade
        self.encoders = nn.ModuleList()
        channels = [in_ch] + [base_ch * (2 ** i) for i in range(nlevels)]
        for i in range(nlevels):
            lvl_nsides = nsides // (factor ** i)
            self.encoders.append(nn.ModuleDict({
                "conv": ConvBlock(
                    in_ch=channels[i],
                    out_ch=channels[i+1],
                    kernel_size=kernel_list[i],
                    activation=activation,
                    bias=bias,
                ),
                "degrade": Degrade(lvl_nsides, factor)
            }))

        # bottleneck at coarsest resolution
        bot_ch = base_ch * (2 ** nlevels)
        bot_nsides = nsides // (factor ** nlevels)
        self.bottleneck = ConvBlock(
            in_ch=channels[-1],
            out_ch=bot_ch,
            kernel_size=kernel_list[-1],
            activation=activation,
            bias=bias,
        )
        self.dense_head = DenseHead(
            channels=bot_ch,
            length=alm_len_from_nsides(bot_nsides),
            dense_dim=dense_dim,
            drop_rate=drop_rate,
            activation=activation,
        )

        # decoder: upsample, concat skip, convblock
        self.decoders = nn.ModuleList()
        for i in range(nlevels - 1, -1, -1):
            lvl_nsides = nsides // (factor ** (i+1))
            self.decoders.append(nn.ModuleDict({
                "upgrade": Upgrade(lvl_nsides, factor),
                "conv": ConvBlock(
                    in_ch=(bot_ch if i == nlevels - 1 else channels[i + 2]) + channels[i + 1],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_list[i],
                    activation=activation,
                    bias=bias
                ),
            }))

        # final 1x1 conv to restore input channels
        self.final = nn.Conv1d(base_ch, in_ch, kernel_size=1, bias=bias)

        # cross-scale gate to capture global interactions
        self.gate = CrossScaleGate(channels=in_ch, thresh_idx=alm_len_from_nsides(gate_resolution))

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
        x = self.dense_head(x)  # [B, BC*2**L, N/D**L]

        # decoding
        for idx, dec in enumerate(self.decoders):
            skip = skips[-(idx + 1)]
            x = dec["upgrade"](x)  # [B, BC*2**(L-i), N/D**(L-i-1)]
            x = torch.cat([x, skip], dim=1)  # [B, BC*2**(L-i) + BC*2**(L-i-1), N/D**(L-i-1)]
            x = dec["conv"](x)  # [B, BC*2**(L-i-1), N/D**(L-i-1)]

        x = self.final(x)
        x = self.gate(x)
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
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('test_loss', loss, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(**self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        if self.scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.s_config = {"scheduler": scheduler, "monitor": "val_loss"}
        elif self.scheduler == torch.optim.lr_scheduler.OneCycleLR or self.scheduler == torch.optim.lr_scheduler.CyclicLR:
            self.s_config = {'scheduler': scheduler, 'interval': 'step'}
        return {"optimizer": optimizer, "lr_scheduler": self.s_config}


"""
OUTDATED
"""


class spectralNBB(nn.Module):
    def __init__(self, **kwargs):
        """
        complex spectral Network Building Block (NBB) as described in Krachmalnicoff & Tomasi (A&A, 2019, 628, A129)

        input: tensor shaped [b, c, n] where b is batch size, c is # of channels,
        and n is (lmax+1)*(lmax+2)/2) where lmax must be 3*nsides-1
            channels: [E_alm_re, E_alm,im, B_alm_re, B_alm_im]

        output: tensor shaped [b, c, new_n] where new_n is n but with nside/degrade_factor
        so new_n = (3*nside)*(3*nside + d)/2d)
            channels: [E_alm_re, E_alm,im, B_alm_re, B_alm_im]

        Args:
            nsides (int, optional): healpix NSIDE. Default config.NSIDES.
            in_channels (int, optional): Number of input channels. Default config.IN_CHANNELS.
            out_channels (int, optional): Number of output channels. Default config.OUT_CHANNELS.
            kernel_size (int, optional): 1D convolution kernel size. Default config.KERNEL_SIZE.
            degradation_factor (int, optional): Downsampling upgrade_factor. Default config.SAMPLE_FACTOR.
            activation (Callable[[Tensor], Tensor], optional): Activation function. Default F.relu.
            bias (bool, optional): Whether to include bias in conv. Default config.BIAS.
        """
        super().__init__()
        self.nsides = kwargs.get("nsides", config.NSIDES)
        self.in_channels = kwargs.get("in_channels", config.IN_CHANNELS)
        self.out_channels = kwargs.get("out_channels", config.OUT_CHANNELS)
        self.kernel_size = kwargs.get("kernel_size", config.KERNEL_SIZE)
        self.degradation_factor = kwargs.get("degradation_factor", config.SAMPLE_FACTOR)
        self.activation = kwargs.get("activation", F.relu)
        self.bias = kwargs.get("bias", config.BIAS)

        # pad by k//2 on each side to preserve size
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2, bias=self.bias)

        # degradation layer
        self.degrade = Degrade(self.nsides, self.degradation_factor)

    def forward(self, x):
        x = self.conv(x)  # [b, c, n]
        x = self.activation(x)  # [b, c, n]
        x = self.degrade(x)  # [b, c, new_n]
        return x


class shallowCNN(LightningModule):
    def __init__(self, **kwargs):
        """
        Shallow spectral-space CNN model chaining multiple NBB blocks and a fully-connected head
        as described in Krachmalnicoff & Tomasi (A&A, 2019, 628, A129)
        Args:
            nsides (int, optional): healpix NSIDE. Default config.NSIDES.
            in_channels (int, optional): Number of input channels. Default config.IN_CHANNELS.
            out_channels (int, optional): Number of output channels. Default config.OUT_CHANNELS.
            degradation_factor (int, optional): Downsampling upgrade_factor per NBB. Default config.SAMPLE_FACTOR.
            activation (Callable[[Tensor], Tensor], optional): Activation function. Default F.relu.
            num_nbb (int, optional): Number of NBB blocks. Default config.NUM_NBB.
            fc_neurons (int, optional): Neurons in fully-connected layer. Default config.DENSE_DIM.
            drop_rate (float, optional): Dropout probability. Default config.DROP_RATE.
        """
        super().__init__()
        self.nsides = kwargs.get("nsides", config.NSIDES)
        self.in_channels = kwargs.get("in_channels", config.IN_CHANNELS)
        self.out_channels = kwargs.get("out_channels", config.OUT_CHANNELS)
        self.degradation_factor = kwargs.get("degradation_factor", config.SAMPLE_FACTOR)
        self.activation = kwargs.get("activation", F.relu)
        self.num_nbb = kwargs.get("nlevels", config.NUM_NBB)
        self.fc_neurons = kwargs.get("dense_dim", config.DENSE_DIM)
        self.drop_rate = kwargs.get("drop_rate", config.DROP_RATE)

        # get before and after len for NBB block for defining exit layers
        self.init_len = alm_len_from_nsides(self.nsides)
        self.final_len = alm_len_from_nsides(self.nsides // self.degradation_factor ** self.num_nbb)

        # nsides -> nsides/d -> nsides/d**2 -> ... -> nsides/d**nlevels
        self.nbbBlock = nn.ModuleList([
            spectralNBB(nsides=self.nsides // (self.degradation_factor ** i),
                        in_channels=self.in_channels if i == 0 else self.out_channels,
                        **kwargs) for i in range(self.num_nbb)
        ])
        self.fc = nn.Linear(self.out_channels * self.final_len, self.fc_neurons)
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.output_layer = nn.Linear(self.fc_neurons, self.in_channels * self.init_len)

        self.save_hyperparameters()

    def forward(self, x):
        B = x.shape[0]  # [b, c_in, n]
        for nbb in self.nbbBlock:
            x = nbb(x)  # [b, c_out, new_n]
        x = x.flatten(1)  # [b, c_out*final_n]
        x = self.fc(x)  # [b, dense_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)  # [b, 4*init_len]
        x = x.view(B, self.in_channels, self.init_len)
        return x
