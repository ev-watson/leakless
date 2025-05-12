import os

import joblib
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import config
from utils import Scaler

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class leaklessDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features  # [b, 2c, n], b is number of samples
        self.input_slice = slice(None, config.IN_CHANNELS)
        self.target_slice = slice(config.IN_CHANNELS, None)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx, self.input_slice, :]
        y = self.features[idx, self.target_slice, :]
        return x, y


class leaklessDataModule(LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.batch_size = batch_size if batch_size else config.BATCH_SIZE
        self.dataset = leaklessDataset
        self.features = np.load(config.DATA_FILE)  # [B, F, N], B is number of sample

        if config.MAC:  # MAC rejects float64
            self.features = self.features.astype(np.float32)

        total_len = self.features.shape[0]
        train_size = int(0.8 * total_len)
        val_size = int(0.1 * total_len)

        if config.SCALE:
            self.input_scaler = Scaler()
            self.target_scaler = Scaler()
            self.inputs = self.input_scaler.fit_transform(self.features[:, :config.IN_CHANNELS, :])  # [b, 4, n]
            self.targets = self.target_scaler.fit_transform(self.features[:, config.IN_CHANNELS:, :])  # [b, 4, n]
            self.features = np.concatenate((self.inputs, self.targets), axis=1)  # [b, 8, n]

            if config.SCALER_FILE and not os.path.exists(config.SCALER_FILE):
                joblib.dump({
                    'input_scaler': self.input_scaler,
                    'target_scaler': self.target_scaler,
                }, config.SCALER_FILE)

        self.train_dataset = self.dataset(self.features[:train_size])
        self.val_dataset = self.dataset(self.features[train_size:train_size + val_size])
        self.test_dataset = self.dataset(self.features[train_size + val_size:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)
