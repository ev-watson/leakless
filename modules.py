from typing import Union, Callable, List, Tuple, Optional

import torch
import torch.nn as nn

import config
from utils import alm_len_from_lmax, make_module, slice_len

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class BiGRUBlock(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: Union[Callable, nn.Module],
                 output_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True,
                 return_hidden: Optional[bool] = False,
                 ):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=batch_first,
                          bidirectional=True)
        self.activation = make_module(activation)
        self.output_layer = nn.Linear(2 * hidden_dim, self.output_dim)

        self.return_z = return_hidden

    def forward(self, x, z=None):
        # x = (B, N, F=4)
        out, z = self.gru(x, z)  # (b, n, 2*f)
        out = self.activation(out)
        out = self.output_layer(out)  # (b, n, f)

        return (out, z) if self.return_z else out


class HRM(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: Union[Callable, nn.Module],
                 bands: List[Tuple[int, int]],
                 nsteps_per_cycle: int,
                 ncycles: int, ):
        """
        Hierarchical Reasoning RNN
        Args:
            input_dim: input dimension
            hidden_dim: hidden dimension
            num_layers: number of layers
            dropout: dropout rate
            activation: activation function/module
            bands: list of tuples of indices to mark bands
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bands = bands
        self.T = nsteps_per_cycle
        self.N = ncycles
        self.nbands = len(bands)
        self.band_lengths = [slice_len(slice(*band).indices(alm_len_from_lmax(config.LMAX))) for band in self.bands]

        self.band_grus = nn.ModuleList(
            [BiGRUBlock(input_dim + hidden_dim, hidden_dim, num_layers, dropout, activation, output_dim=input_dim,
                        return_hidden=True) for _ in range(self.nbands)]
        )
        self.low_level_gru = BiGRUBlock(hidden_dim, hidden_dim, num_layers, dropout, activation, batch_first=False,
                                        return_hidden=True)
        self.high_level_gru = BiGRUBlock(hidden_dim, hidden_dim, num_layers, dropout, activation, batch_first=False,
                                         return_hidden=True)

        self.zL = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))
        self.zH = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))

    def embed_hidden_state_for_band(self, x, band_idx):
        # take last layer and turn into input shape to embed with input
        return x[-1].unsqueeze(1).expand(-1, self.band_lengths[band_idx], -1)

    def forward(self, x: torch.Tensor):
        b, n, f = x.shape
        assert sum(self.band_lengths) == n, f"Sum of band lengths ({sum(self.band_lengths)}) do not match npoints ({n}) in input, "

        banded_input = [x[:, slice(*band), :].clone() for band in self.bands]  # (nbands, b, band_length, f)

        with torch.no_grad():
            zL, zH = self.zL.to(dtype=x.dtype, device=x.device), self.zH.to(dtype=x.dtype, device=x.device)
            zL_band = [zL.clone() for _ in range(self.nbands)]

            for H_step in range(self.N - 1):
                for L_step in range(self.T - 1):
                    for i in range(self.nbands):
                        zH_embed = self.embed_hidden_state_for_band(zH, i)
                        # pass embedded input along with low level hidden state
                        banded_input[i], zL_band[i] = self.band_grus[i](torch.cat((banded_input[i], zH_embed), dim=-1),
                                                                        zL_band[i])

                # pass low level summary as input and current zH as hidden state to get new high level state
                _, zH = self.high_level_gru(torch.stack(zL_band, dim=0).mean(dim=0), zH)

        # one step with grad
        for i in range(self.nbands):
            zH_embed = self.embed_hidden_state_for_band(zH, i)
            banded_input[i], zL_band[i] = self.band_grus[i](torch.cat((banded_input[i], zH_embed), dim=-1), zL_band[i])

        _, zL = self.low_level_gru(torch.stack(zL_band, dim=0).mean(dim=0) + zH, zL)  # combine low and high states
        _, zH = self.high_level_gru(zL, zH)  # get high states like before

        self.zL, self.zH = zL.detach(), zH.detach()  # detach and carry

        x = torch.cat(banded_input, dim=1)  # replace inputs

        return x
