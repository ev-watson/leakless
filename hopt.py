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

loss_functions = {
    'l1': F.l1_loss,
    'smooth_l1': F.smooth_l1_loss,
    'huber': F.huber_loss,
    'mse': F.mse_loss,
}

loss_hyperparams = {
    'huber': {
        'delta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    'smooth_l1': {
        'beta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
}


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
    params['optimizer'] = torch.optim.AdamW
    params['optimizer_kwargs'] = {
        'betas': (
            trial.suggest_float('beta1', 0.85, 0.99),
            trial.suggest_float('beta2', 0.99, 0.9999),
        ),
        'eps': 1e-8,
    }

    # ── Scheduler ─────────────────────────────────────────────────
    params['scheduler_kwargs'] = {
        'factor': trial.suggest_float('scheduler_factor', 0.05, 0.5),
        'patience': trial.suggest_int('patience', 3, 7),
    }

    config.update_hparams(params)

    data_module = leaklessDataModule()
    model = Leakless(**params)

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
        benchmark=True,
        logger=TensorBoardLogger('hopt', name=config.RUN_NAME),
        log_every_n_steps=log_steps,
    )

    trainer.fit(model, datamodule=data_module)

    rtrials = 2000
    metric = leak_test(model, ntrials=rtrials, hopt=True, err=True)
    return metric.item()


sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,
    n_ei_candidates=24,
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
