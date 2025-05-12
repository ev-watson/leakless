import argparse
import signal

import optuna
import torch_optimizer as optim
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lion_pytorch import Lion

from data_construction import *

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

seed = config.SEED if config.SEED else np.random.randint(1, 10000)

activation_functions = {
    'relu': F.relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'mish': F.mish,
    'hardswish': F.hardswish,
    'sigmoid': F.sigmoid,
    # 'swish': lambda x: x * F.sigmoid(x),
    'sinu': lambda x: x + torch.sin(x) ** 2,
}

loss_functions = {
    'l1': F.l1_loss,
    'smooth_l1': F.smooth_l1_loss,
    'huber': F.huber_loss,
    'mse': F.mse_loss,
    'rmwe': rmwe_loss,
    'zero-one': zero_one_approximation_loss,
}

optimizer_functions = {
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,
    'nadam': torch.optim.NAdam,
    'radam': torch.optim.RAdam,
    'adabound': optim.AdaBound,
    'swats': optim.SWATS,
    'lion': Lion,
}

base_opt_kwargs = {
    'betas1': {'type': 'float', 'low': 0.9, 'high': 0.99},  # Log inherently included in sampler function in utils
    'betas2': {'type': 'float', 'low': 0.99, 'high': 0.9999},
    'eps': {'type': 'float', 'default': 1e-8},
    'weight_decay': {'type': 'float', 'low': 1e-10, 'high': 1e-2, 'log': True},
}

optimizer_hyperparams = {
    'sgd': {
        'momentum': {'type': 'float', 'low': 0.8, 'high': 0.99999},
        'weight_decay': {'type': 'float', 'low': 1e-10, 'high': 1e-2, 'log': True},
        'nesterov': {'type': 'bool', 'default': True},
    },
    'adamw': {
        **base_opt_kwargs,
    },
    'nadam': {
        **base_opt_kwargs,
        'momentum_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-1, 'log': True},
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'radam': {
        **base_opt_kwargs,
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'adabound': {
        **base_opt_kwargs,
        'final_lr': {'type': 'float', 'low': 1e-8, 'high': 1e-1},
        'gamma': {'type': 'float', 'low': 1e-6, 'high': 1e-1},
        'amsbound': {'type': 'bool'},
    },
    'swats': {
        **base_opt_kwargs,
        'amsgrad': {'type': 'bool'},
        'nesterov': {'type': 'bool'},
    },
    'lion': {
        **{k: v for k, v in base_opt_kwargs.items() if k != 'eps'},
        'decoupled_weight_decay': {'type': 'bool'},
    },
}

loss_hyperparams = {
    'huber': {
        'delta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    'smooth_l1': {
        'beta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    'zero-one': {
        'sigma': {'type': 'float', 'low': .1, 'high': 1.},
    },
}

parser = argparse.ArgumentParser(description="Hyper-optimization")
optimizer_choices = list(optimizer_functions.keys())
model_choices = ['interp', 'gnn']
parser.add_argument("--opt", "-o", type=str, default="adamw",
                    choices=optimizer_choices,
                    help=f"Optimizer function. Defaults to adamw")
parser.add_argument("--model", '-m', type=str, default='gnn',
                    choices=model_choices,
                    help=f"Model to run. Defaults to gnn")
args = parser.parse_args()
config.TYPE = args.model


def objective(trial):
    print_block(f"TRIAL: {trial.number}, SEED: {seed}", err=True)
    seed_everything(seed)
    clear_local_ckpt_files()

    """
    Build the hyperparameter dictionary and study parameter list.

    This method ensures that the order of trial.suggest calls (study parameter list)
    matches the hparam dictionary with support for hparam dependencies (e.g. 'dropout_frequency' depending on 'num_layers')
    """
    params = {}
    loss_params = {}

    # TRAINING
    params['lr'] = trial.suggest_float('lr', 1e-7, 1e0)
    # params['batch_size'] = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    # params['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.7, 1.5)

    # ARCHITECTURE
    params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512, 1024, 2048])
    params['num_layers'] = trial.suggest_int('num_layers', 2, 8)
    params['drop_rate'] = trial.suggest_float('drop_rate', 5e-3, 0.5)
    params['dropout_frequency'] = trial.suggest_int('dropout_frequency', 1, params['num_layers'])
    # params['se_block'] = trial.suggest_categorical('se_block', [True, False])
    params['se_reduction'] = trial.suggest_categorical('se_reduction', [2, 4, 8, 16, 32, 64, 128])
    # params['rotational_equivariance'] = trial.suggest_categorical('rotational_equivariance', [True, False])
    # params['windowed'] = trial.suggest_categorical('windowed', [True, False])

    # ALGORITHMS
    # ---activation---
    params['activation_name'] = trial.suggest_categorical('activation', list(activation_functions.keys()))

    # ---loss---
    params['loss_name'] = trial.suggest_categorical('loss_name', list(loss_functions.keys()))
    params['loss'] = loss_functions[params['loss_name']]
    if params['loss_name'] in loss_hyperparams:
        loss_params = sample_hyperparams(trial, loss_hyperparams[params['loss_name']])
    params['loss_kwargs'] = loss_params

    # ---optimizer---
    params['optimizer'] = optimizer_functions[args.opt]
    params['optimizer_kwargs'] = sample_hyperparams(trial, optimizer_hyperparams[args.opt])

    # ---scheduler---
    params['scheduler_kwargs'] = {
        'factor': trial.suggest_float('scheduler_factor', 0.01, .5),
        'patience': trial.suggest_int('patience', 3, 6),
    }

    config.ROTATIONAL_EQUIVARIANCE = params.get('rotational_equivariance', config.ROTATIONAL_EQUIVARIANCE)
    config.WINDOWED = params.get('windowed', config.WINDOWED)
    if config.WINDOWED:
        params['seqlen'] = trial.suggest_categorical('seqlen', [50, 100, 150, 200, 250, 300])
        config.SEQUENCE_LENGTH = params['seqlen']

    config.update_hparams(params)

    data_module = NNDataModule()

    model = config.retrieve('model')(
        **params
    )

    print_err(f"Starting trial with parameters: {params}")

    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        gradient_clip_val=params.get('gradient_clip_val', config.GRADIENT_CLIP_VAL),
        callbacks=[EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min'),
                   GradientNormCallback()],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)] if not config.MAC else None,
        accelerator='gpu',
        devices=-1,
        strategy='ddp' if not config.MAC else 'auto',
        sync_batchnorm=True,
        benchmark=True,
        logger=TensorBoardLogger('hopt', name=f'{args.model}_{args.opt}_logs'),
    )

    trainer.fit(model, datamodule=data_module)

    # trainer.test(model, datamodule=data_module)

    # return trainer.callback_metrics['test_loss'].item()

    rtrials = 921600
    if args.model == 'interp':
        mae, mape = interp_test(model, ntrials=rtrials, mape=True, err=True)
    else:
        mae, mape = gnn_test(model, ntrials=rtrials, mape=True, err=True, mean_axis=None)

    return mae.item()


# multi-objective sampler
# sampler = optuna.samplers.NSGAIISampler(
#     population_size=100,    # 50
#     crossover_prob=0.915,   # 0.9
#     swapping_prob=0.51,     # 0.5
#     mutation_prob=0.08,     # None
# )

# single-objective sampler
sampler = optuna.samplers.TPESampler(
    n_startup_trials=15,  # 10
    n_ei_candidates=36,  # 24
    seed=seed,
)

study_name = f"{args.model}_{args.opt}_study"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(direction='minimize',
                            storage=storage_name,
                            sampler=sampler,
                            study_name=study_name,
                            load_if_exists=True)
study.optimize(objective, n_trials=5000)
