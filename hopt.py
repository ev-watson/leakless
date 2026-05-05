"""Hyperparameter optimization for Leakless HRM via Optuna."""
import argparse
import signal

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

import config
from data_construction import leaklessDataModule
from models import Leakless
from utils import (print_block, print_err, leak_test, GradientNormCallback,
                   rmwe_loss, sample_hyperparams, clear_local_ckpt_files)

seed = config.SEED if config.SEED else np.random.randint(1, 10000)

activation_functions = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'mish': F.mish,
    'hardswish': F.hardswish,
    'sigmoid': F.sigmoid,
    # 'swish': lambda x: x * F.sigmoid(x),
    # 'sinu': lambda x: x + torch.sin(x) ** 2,
}

loss_functions = {
    'l1': F.l1_loss,
    'smooth_l1': F.smooth_l1_loss,
    'huber': F.huber_loss,
    'mse': F.mse_loss,
    # 'rmwe': rmwe_loss,
    # 'zero-one': zero_one_approximation_loss,
}

optimizer_functions = {
    'adamw': torch.optim.AdamW,
    'nadam': torch.optim.NAdam,
    'radam': torch.optim.RAdam,
    'sgd': torch.optim.SGD,
    # 'adabound': optim.AdaBound,
    # 'swats': optim.SWATS,
    # 'lion': Lion,
}

base_opt_kwargs = {
    'betas1': {'type': 'float', 'low': 0.9, 'high': 0.99},  # Log inherently included in sampler function in utils
    'betas2': {'type': 'float', 'low': 0.99, 'high': 0.9999},
    'eps': {'type': 'float', 'default': 1e-8},
    'weight_decay': {'type': 'float', 'low': 1e-10, 'high': 1e-2, 'log': True},
}

optimizer_hyperparams = {
    'adamw': {
        **base_opt_kwargs,
    },
    'nadam': {
        **base_opt_kwargs,
        'momentum_decay': {'type': 'float', 'low': 1e-6, 'high': 5e-1},
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'radam': {
        **base_opt_kwargs,
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'sgd': {
        'momentum': {'type': 'float', 'low': 0.8, 'high': 0.99999},
        'weight_decay': {'type': 'float', 'low': 1e-10, 'high': 1e-2, 'log': True},
        'nesterov': {'type': 'bool', 'default': True},
    },
    # 'adabound': {
    #     **base_opt_kwargs,
    #     'final_lr': {'type': 'float', 'low': 1e-8, 'high': 1e-1},
    #     'gamma': {'type': 'float', 'low': 1e-6, 'high': 1e-1},
    #     'amsbound': {'type': 'bool'},
    # },
    # 'swats': {
    #     **base_opt_kwargs,
    #     'amsgrad': {'type': 'bool'},
    #     'nesterov': {'type': 'bool'},
    # },
    # 'lion': {
    #     **{k: v for k, v in base_opt_kwargs.items() if k != 'eps'},
    #     'decoupled_weight_decay': {'type': 'bool'},
    # },
}

loss_hyperparams = {
    'huber': {
        'delta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    'smooth_l1': {
        'beta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    # 'zero-one': {
    #     'sigma': {'type': 'float', 'low': .1, 'high': 1.},
    # },
}

parser = argparse.ArgumentParser(description="Hyper-optimization")
optimizer_choices = list(optimizer_functions.keys())
parser.add_argument("--opt", "-o", type=str, default="adamw",
                    choices=optimizer_choices,
                    help=f"Optimizer function. Defaults to adamw")
args = parser.parse_args()


def objective(trial):
    print_block(f"TRIAL: {trial.number}, SEED: {seed}", err=True)
    seed_everything(seed)
    clear_local_ckpt_files()

    params = {}
    loss_params = {}

    # ── Training ──────────────────────────────────────────────────
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    # ── HRM architecture ──────────────────────────────────────────
    params['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 128, 256])
    params['H_layers'] = trial.suggest_int('H_layers', 1, 6)
    params['L_layers'] = trial.suggest_int('L_layers', 1, 6)
    params['H_cycles'] = trial.suggest_int('H_cycles', 1, 4)
    params['L_cycles'] = trial.suggest_int('L_cycles', 2, 6)
    params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
    params['expansion'] = trial.suggest_float('expansion', 2.0, 6.0)
    params['n_supervision'] = trial.suggest_int('n_supervision', 1, 8)
    params['rope_theta'] = trial.suggest_float('rope_theta', 1000.0, 100000.0, log=True)

    # Validate: hidden_size must be divisible by num_heads, head_dim must be even
    if params['hidden_size'] % params['num_heads'] != 0:
        raise optuna.TrialPruned("hidden_size not divisible by num_heads")
    head_dim = params['hidden_size'] // params['num_heads']
    if head_dim % 2 != 0:
        raise optuna.TrialPruned("head_dim must be even for RoPE")

    # ── Loss ──────────────────────────────────────────────────────
    params['loss_name'] = trial.suggest_categorical('loss_name', list(loss_functions.keys()))
    params['loss'] = loss_functions[params['loss_name']]
    if params['loss_name'] in loss_hyperparams:
        loss_params = sample_hyperparams(trial, loss_hyperparams[params['loss_name']])
    params['loss_kwargs'] = loss_params

    # ── Optimizer ─────────────────────────────────────────────────
    params['optimizer'] = optimizer_functions[args.opt]
    params['optimizer_kwargs'] = sample_hyperparams(trial, optimizer_hyperparams[args.opt])

    # ── Scheduler ─────────────────────────────────────────────────
    params['scheduler_kwargs'] = {
        'factor': trial.suggest_float('scheduler_factor', 0.05, 0.5),
        'patience': trial.suggest_int('patience', 3, 7),
    }

    config.update_hparams(params)

    data_module = leaklessDataModule()
    model = Leakless(**params)

    # training batches / gpus / freq to log freq times per epoch
    ngpus = 4 if not config.MAC else 1
    freq = 5
    log_steps = max(1, int(0.8 * config.NUM_SAMPLES / config.BATCH_SIZE / ngpus / freq))

    print_err(f"Starting trial with parameters: {params}")

    trainer = Trainer(
        max_epochs=100,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        callbacks=[
            EarlyStopping(monitor='train_loss', patience=config.PATIENCE, mode='min'),
            GradientNormCallback(),
            TQDMProgressBar(refresh_rate=log_steps),
        ],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)] if not config.MAC else None,
        accelerator='gpu' if not config.MAC else 'auto',
        devices=-1 if not config.MAC else 'auto',
        strategy='ddp' if not config.MAC else 'auto',
        sync_batchnorm=not config.MAC,
        precision=config.PRECISION if not config.MAC else '32-true',
        benchmark=True,
        logger=TensorBoardLogger('hopt', name=config.RUN_NAME),
        log_every_n_steps=log_steps,
    )

    trainer.fit(model, datamodule=data_module)

    # trainer.test(model, datamodule=data_module)
    # return trainer.callback_metrics['test_loss'].item()

    rtrials = 2000
    metric = leak_test(model, ntrials=rtrials, hopt=True, err=True)
    return metric.item()


# multi-objective sampler
# sampler = optuna.samplers.NSGAIISampler(
#     population_size=100,    # 50
#     crossover_prob=0.915,   # 0.9
#     swapping_prob=0.51,     # 0.5
#     mutation_prob=0.08,     # None
# )

# single-objective sampler
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,  # 10
    n_ei_candidates=24,   # 24
    seed=seed,
)

study_name = config.RUN_NAME
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(
    direction='minimize',
    storage=storage_name,
    sampler=sampler,
    study_name=study_name,
    load_if_exists=True,
)
study.optimize(objective, n_trials=5000)
