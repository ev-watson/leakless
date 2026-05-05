import os

import joblib
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import config
from utils import Scaler

torch.set_default_dtype(torch.float32)

DATA_DTYPE = np.float32


class leaklessDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features  # [b, n, 2f], b is number of samples
        self.input_slice = slice(None, config.INPUT_DIM)
        self.target_slice = slice(config.INPUT_DIM, None)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx, :, self.input_slice]
        y = self.features[idx, :, self.target_slice]
        return x, y


class indexedDataset(Dataset):
    def __init__(self, indices, datafile=config.DATA_FILE):
        super().__init__()
        self._data = np.load(datafile, mmap_mode='r')  # [b, n, 2f], b is number of samples
        self.indices = np.array(indices, dtype=np.int64)    # must be np array for indexing
        self.input_slice = slice(None, config.INPUT_DIM)
        self.target_slice = slice(config.INPUT_DIM, None)

        self.dtype = DATA_DTYPE

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw = torch.from_numpy(self._data[self.indices[idx]].astype(self.dtype, copy=False).copy())
        x = raw[:, self.input_slice]
        y = raw[:, self.target_slice]
        return x, y


class rollingDataset(indexedDataset):
    # inherent from indexedDataset to modify indices property
    def __init__(self, indices, datafile=config.DATA_FILE, replace_frac=config.REPLACE_FRAC):
        super().__init__(indices, datafile)
        self._data = np.load(datafile, mmap_mode='r')
        self.input_slice = slice(None, config.INPUT_DIM)
        self.target_slice = slice(config.INPUT_DIM, None)
        self.buffer_size = len(self.indices)
        self.replace_frac = replace_frac

    # for torch callback
    def on_epoch_end(self):
        n_replace = int(self.replace_frac * self.buffer_size)
        old_idx = np.random.choice(self.buffer_size, size=n_replace, replace=False)
        new_idx = np.random.choice(len(self._data), size=n_replace, replace=False)
        self.indices[old_idx] = new_idx


class leaklessDataModule(LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.batch_size = batch_size if batch_size else config.BATCH_SIZE

        # default dataset
        self.dataset = indexedDataset if config.ROLLING else leaklessDataset

        # get unique samples by disabling replace
        n_samples = config.STACK_SIZE
        total_len = config.NUM_SAMPLES
        idxs = np.random.choice(n_samples, size=total_len, replace=False)

        train_idx = int(0.8 * total_len)
        val_idx = int(0.9 * total_len)

        if config.ROLLING:
            self.train_dataset = rollingDataset(idxs[:train_idx])
            self.val_dataset = self.dataset(idxs[train_idx:val_idx])
            self.test_dataset = self.dataset(idxs[val_idx:])

        else:
            # make memmap, dont read anything yet
            data = np.load(config.DATA_FILE, mmap_mode='r')  # [B, N, F], B is number of sample

            # only these rows get read into memory
            self.features = data[idxs]

            self.features = self.features.astype(DATA_DTYPE, copy=False)

            if config.SCALE:
                # scale input and targets separately
                self.input_scaler = Scaler()
                self.target_scaler = Scaler()
                self.inputs = self.input_scaler.fit_transform(self.features[..., :config.INPUT_DIM])  # [b, n, 4]
                self.targets = self.target_scaler.fit_transform(self.features[..., config.INPUT_DIM:])  # [b, n, 4]
                self.features = np.concatenate((self.inputs, self.targets), axis=-1)  # [b, n, 8]

                if config.SCALER_FILE:
                    joblib.dump({
                        'input_scaler': self.input_scaler,
                        'target_scaler': self.target_scaler,
                    }, config.SCALER_FILE)

            self.train_dataset = self.dataset(self.features[:train_idx])
            self.val_dataset = self.dataset(self.features[train_idx:val_idx])
            self.test_dataset = self.dataset(self.features[val_idx:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)
