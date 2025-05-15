import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import config
from data_construction import leaklessDataModule
from models import Leakless
from utils import GradientNormCallback, print_block, rmwe_loss

seed = config.SEED if config.SEED else np.random.randint(1, 10000)
print_block(f"SEED: {seed}")
seed_everything(seed)

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

params = {
    'lr': 1e-3,
    'base_channels': config.BASE_CHANNELS,
    'kernel_size': config.KERNEL_SIZE,
    'sample_factor': config.SAMPLE_FACTOR,
    'num_levels': 4,
    'kernel_list': [13, 11, 7, 5],
    'activation': nn.ReLU,
    'drop_rate': config.DROP_RATE,
    'loss': rmwe_loss,
    # 'loss_kwargs': {
    #
    # },
    'optimizer': torch.optim.NAdam,
    'optimizer_kwargs': {
        'betas': (0.90, 0.99),
        'weight_decay': 0,
        'eps': 1e-8,
        'momentum_decay': 0.003288402439883893,
        'decoupled_weight_decay': False,
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
        'factor': 0.1,
        'patience': 4,
    },
}

config.update_hparams(params)

data_module = leaklessDataModule()

model = Leakless(**params)

# training batches/4 gpus/5 to log 5 times per epoch
ngpus = 4
freq = 5
log_steps = int(0.8*config.NUM_SAMPLES/config.BATCH_SIZE/ngpus/freq)
if log_steps == 0:
    log_steps = 1


trainer = Trainer(
    max_epochs=config.MAX_EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min'),
               GradientNormCallback(),
               LearningRateMonitor(logging_interval='step' if config.ON_STEP else 'epoch'),
               TQDMProgressBar(refresh_rate=0),
               ],
    gradient_clip_val=config.GRADIENT_CLIP_VAL,
    precision=config.PRECISION,
    accelerator='gpu',
    devices=-1,
    strategy='auto',
    sync_batchnorm=True,
    logger=TensorBoardLogger('prelim', name=f"unet"),
    log_every_n_steps=log_steps,
)

print_block("TRAINING...")

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)
