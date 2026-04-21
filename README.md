# Leakless

CMB E-to-B polarization leakage cleaning via deep learning on spherical harmonic coefficients.

## Overview

Leakless trains a **Hierarchical Reasoning Model (HRM)** — a dual-timescale recurrent transformer — to remove spurious E-mode leakage from B-mode power spectra in partial-sky CMB observations. The network operates directly on complex spherical harmonic coefficient sequences produced by HEALPix, exploiting Rotary Position Embeddings to encode the (ℓ, m) ordering.

Key architectural features:
- **H-module** (slow): captures global correlations across harmonic modes
- **L-module** (fast): handles local mode-to-mode interactions
- **Deep supervision**: multiple forward segments per training step provide regularization and more frequent gradient feedback to the H-module
- **Scaled Dot-Product Attention (SDPA)** with Flash/memory-efficient backends

## Project Structure

```
config.py            # All hyperparameters and environment flags
modules.py           # HRM core: RoPE, SwiGLU, H/L-modules, attention blocks
models.py            # Lightning training wrapper with deep supervision
train.py             # Training entry point
data_construction.py # HEALPix data pipeline and Lightning DataModule
hopt.py              # Optuna hyperparameter optimization
results.py           # Evaluation and analysis utilities
utils/               # Losses, logging, analysis, harmonic helpers
job_scripts/         # SLURM / shell scripts for cluster submission
```

## Usage

```bash
# Generate / refresh data
bash job_scripts/run_data_init.sh

# Train
bash job_scripts/run_train.sh

# Hyperparameter search
bash job_scripts/run_hopt.sh
```

Training uses PyTorch Lightning with TensorBoard logging, gradient clipping, early stopping, and ReduceLROnPlateau scheduling.

## Requirements

- Python 3.9+
- PyTorch, PyTorch Lightning
- HEALPy, NaMaster (pymaster)
- NumPy, joblib, Optuna

## AI Disclosure

AI-assisted tools (Claude, Anthropic) were used during development of this repository for code architecture, implementation, and documentation.
