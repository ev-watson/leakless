import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import config
from data_construction import leaklessDataModule
from models import Leakless
from utils import GradientNormCallback, print_block, rmwe_loss, RollingBufferCallback

seed = config.SEED if config.SEED else np.random.randint(1, 10000)
print_block(f"SEED: {seed}")
seed_everything(seed)

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

RUN_NAME = "lowell"

params = {
    'lr': 1e-3,
    'base_channels': 2*config.NSIDE,
    'sample_factor': config.SAMPLE_FACTOR,
    'num_levels': 4,
    'kernel_list': [15, 7, 3, 1],
    'conv_block_layers': 1,
    'activation': nn.ReLU,
    'learnable_upgrade': False,
    'drop_rate': 0.35,
    'loss': F.mse_loss,
    # 'loss_kwargs': {
    #     'beta': 1.575184625409794,
    # },
    'optimizer': torch.optim.NAdam,
    'optimizer_kwargs': {
        'betas': (0.9241000987227369, 0.9996730603074183),
        'eps': 1e-12,
        'weight_decay': 3.0464200451047e-08,
        'momentum_decay': 0.08245131333657932,
        'decoupled_weight_decay': True
    },
    # 'scheduler': optim.lr_scheduler.CyclicLR,
    # 'scheduler_kwargs': {
    #     'base_lr': 7e-4,
    #     'max_lr': .01,
    #     'step_size_up': 2000,
    #     'scale_fn': None,
    #     'mode': 'triangular',   # only used if 'scale_fn' is None
    #     'gamma': 1.0,   # only used if 'mode' = 'exp_range'
    # },
    'scheduler_kwargs': {
        'factor': 0.25,
        'patience': 4,
    },
}

config.update_hparams(params)

data_module = leaklessDataModule()

model = Leakless(**params)

# (training batches)/(4 gpus)/5 to log 5 times per epoch
ngpus = 4
freq = 5
log_steps = int(0.8*config.NUM_SAMPLES/config.BATCH_SIZE/ngpus/freq)
if log_steps == 0:
    log_steps = 1


trainer = Trainer(
    max_epochs=200,
    callbacks=[
        ModelCheckpoint(
            dirpath=f'tlogs/{RUN_NAME}/checkpoints',
            filename='{epoch}-{step}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
        ),
        RollingBufferCallback(),
        GradientNormCallback(),
        LearningRateMonitor(logging_interval='step' if config.ON_STEP else 'epoch'),
        TQDMProgressBar(refresh_rate=log_steps),
    ],
    gradient_clip_val=config.GRADIENT_CLIP_VAL,
    accelerator='gpu',
    devices=-1,
    strategy='ddp' if not config.MAC else 'auto',
    sync_batchnorm=True,
    logger=TensorBoardLogger('tlogs', name=f"{RUN_NAME}"),
    log_every_n_steps=log_steps,
)

print_block("TRAINING...")

trainer.fit(model, datamodule=data_module, ckpt_path='last')

trainer.test(model, datamodule=data_module)
