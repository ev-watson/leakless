import torch
import torch.nn as nn
from typing import Union, Callable

import config
from utils import alm_len_from_nside, make_module

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class BiGRUBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, activation: Union[Callable, nn.Module]):
        super().__init__()
        self.unit = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.activation = make_module(activation)
        self.output_layer = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        # x = (B, N, F=4)
        out, _ = self.unit(x)       # (b, n, 2*f)
        out = self.activation(out)
        out = self.output_layer(out)        # (b, n, f)
        return out


# class HRN(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
#         super().__init__()
#
#         self.lowlevel_gru = BiGRUBlock
#
#     def forward(self, x: torch.Tensor):
