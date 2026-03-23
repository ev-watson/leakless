import os

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

import config
from data_construction import leaklessDataModule
from models import Leakless
from utils import GradientNormCallback, print_block, RollingBufferCallback

seed = config.SEED if config.SEED else np.random.randint(1, 10000)
print_block(f"SEED: {seed}")
seed_everything(seed)

RUN_NAME = config.RUN_NAME
ckpt_path = f'tlogs/{RUN_NAME}/checkpoints'
last_path = ckpt_path + '/last.ckpt'
if not config.CONTINUOUS and os.path.exists(last_path):
    os.remove(last_path)

params = {
    # HRM architecture
    'hidden_size': config.HIDDEN_SIZE,
    'H_layers': config.H_LAYERS,
    'L_layers': config.L_LAYERS,
    'H_cycles': config.H_CYCLES,
    'L_cycles': config.L_CYCLES,
    'num_heads': config.NUM_HEADS,
    'expansion': config.EXPANSION,
    'n_supervision': config.N_SUPERVISION,

    # Training
    'lr': 1e-3,
    'weight_decay': 0.01,
    'loss': F.mse_loss,
    'optimizer': torch.optim.AdamW,
    'optimizer_kwargs': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    },
    # 'scheduler': torch.optim.lr_scheduler.CyclicLR,
    # 'scheduler_kwargs': {
    #     'base_lr': 7e-4,
    #     'max_lr': .01,
    #     'step_size_up': 2000,
    #     'scale_fn': None,
    #     'mode': 'triangular',   # only used if 'scale_fn' is None
    #     'gamma': 1.0,           # only used if 'mode' = 'exp_range'
    # },
    'scheduler_kwargs': {
        'factor': 0.25,
        'patience': 4,
    },
}

config.update_hparams(params)

data_module = leaklessDataModule()
model = Leakless(**params)

# Log frequency: ~10 times per epoch
ngpus = 4 if not config.MAC else 1
freq = 10
log_steps = max(1, int(0.8 * config.NUM_SAMPLES / config.BATCH_SIZE / ngpus / freq))

trainer = Trainer(
    max_epochs=config.MAX_EPOCHS,
    callbacks=[
        ModelCheckpoint(
            dirpath=ckpt_path,
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
    accelerator='gpu' if not config.MAC else 'auto',
    devices=-1 if not config.MAC else 'auto',
    strategy='ddp' if not config.MAC else 'auto',
    sync_batchnorm=not config.MAC,
    logger=TensorBoardLogger('tlogs', name=RUN_NAME),
    log_every_n_steps=log_steps,
    profiler=PyTorchProfiler(
        dirpath='tlogs/profiles',
        filename="trace",
    ) if not config.MAC else None,
)

print_block("TRAINING...")
trainer.fit(model, datamodule=data_module, ckpt_path='last')
trainer.test(model, datamodule=data_module)
