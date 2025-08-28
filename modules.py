from typing import Union, Callable, List, Tuple, Optional

import torch
import torch.nn as nn

import config
from utils import alm_len_from_lmax, make_module, slice_len, SDPAEncoderBlock

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
                 use_attention: bool = True,
                 attn_heads: int = 8,
                 attn_drop: float = 0.0,
                 attn_proj_drop: float = 0.0,
                 mlp_ratio: float = 2.0,
                 mlp_drop: float = 0.0,
                 rope_base: float = 10000.0,
                 causal: bool = False):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=batch_first,
                          bidirectional=True)
        self.activation = make_module(activation)
        self.output_layer = nn.Linear(2 * hidden_dim, self.output_dim)

        self.return_z = return_hidden

        self.use_attention = use_attention
        if use_attention:
            self.attention = SDPAEncoderBlock(
                dim=2*hidden_dim,
                heads=attn_heads,
                attn_drop=attn_drop,
                proj_drop=attn_proj_drop,
                mlp_ratio=mlp_ratio,
                mlp_drop=mlp_drop,
                rope_base=rope_base,
                causal=causal,
            )

    def forward(self, x, z=None):
        # x = (B, N, F=4)
        out, z = self.gru(x, z)  # (b, n, 2*f)
        if self.use_attention:
            out = self.attention(out)
        out = self.activation(out)
        out = self.output_layer(out)  # (b, n, f)

        return (out, z) if self.return_z else out


class HRM(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: Union[Callable, nn.Module],
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
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = nsteps_per_cycle
        self.N = ncycles

        self.low_level_gru = BiGRUBlock(input_dim + hidden_dim, hidden_dim, num_layers, dropout, activation, output_dim=input_dim,
                                        batch_first=True, return_hidden=True)
        self.high_level_gru = BiGRUBlock(hidden_dim, hidden_dim, num_layers, dropout, activation, batch_first=False,
                                         return_hidden=True)
        self.output_head = nn.Linear(hidden_dim, input_dim, bias=False)
        self.mixer = nn.Linear(2 * input_dim, input_dim, bias=False)

        self.zL = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))
        self.zH = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))

    @staticmethod
    def prepare_embedding(x, n):
        # take last layer and turn into input shape to embed with input
        # after slicing, shape is (B, HD), unsqueeze to (B, 1, HD) and expand to (B, n, HD)
        return x[-1].unsqueeze(1).expand(-1, n, -1)

    @staticmethod
    def embed(x, y, dim=-1):
        return torch.cat((x, y), dim=dim)

    def forward(self, x: torch.Tensor):
        b, n, f = x.shape

        with torch.no_grad():
            zL, zH = self.zL.to(dtype=x.dtype, device=x.device), self.zH.to(dtype=x.dtype, device=x.device)

            for H_step in range(self.N - 1):
                for L_step in range(self.T - 1):
                    zH_embed = self.prepare_embedding(zH, n)
                    # pass embedded input along with low level hidden state
                    x, zL = self.low_level_gru(self.embed(x, zH_embed), zL)

                # pass low level summary as input and current zH as hidden state to get new high level state
                _, zH = self.high_level_gru(zL, zH)

        # one step with grad
        zH_embed = self.prepare_embedding(zH, n)
        x, zL = self.low_level_gru(self.embed(x, zH_embed), zL)
        H_summary, zH = self.high_level_gru(zL, zH)

        head_input = self.prepare_embedding(H_summary, n)  # B, N, HD
        x_embed = self.output_head(head_input)  # B, N, F
        x = self.mixer(self.embed(x, x_embed))  # B, N, F

        self.zL, self.zH = zL.detach(), zH.detach()  # detach and carry

        return x


class BandedHRM(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: Union[Callable, nn.Module],
                 bands: List[Tuple[int, int]],
                 nsteps_per_cycle: int,
                 ncycles: int, ):
        """
        Banded Hierarchical Reasoning RNN
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
        self.output_head = nn.Linear(hidden_dim, input_dim, bias=False)
        self.mixer = nn.Linear(2 * input_dim, input_dim, bias=False)

        self.zL = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))
        self.zH = torch.zeros((2 * self.num_layers, config.BATCH_SIZE, self.hidden_dim))

    def embed_hidden_state_for_band(self, x, band_idx):
        # take last layer and turn into input shape to embed with input
        return x[-1].unsqueeze(1).expand(-1, self.band_lengths[band_idx], -1)

    def forward(self, x: torch.Tensor):
        b, n, f = x.shape
        assert sum(
            self.band_lengths) == n, f"Sum of band lengths ({sum(self.band_lengths)}) do not match npoints ({n}) in input, "

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

        L_summary, zL = self.low_level_gru(torch.stack(zL_band, dim=0).mean(dim=0) + zH, zL)  # combine low and high states
        H_summary, zH = self.high_level_gru(L_summary, zH)  # get high states like before

        head_input = H_summary[-1].unsqueeze(1).expand(-1, n, -1)  # B, N, HD
        x_embed = self.output_head(head_input)  # B, N, F

        mix_input = torch.cat((torch.cat(banded_input, dim=1), x_embed), dim=-1)  # mix inputs and embedding
        x = self.mixer(mix_input)

        self.zL, self.zH = zL.detach(), zH.detach()  # detach and carry

        return x
