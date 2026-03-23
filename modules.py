"""
Hierarchical Reasoning Model (HRM) for CMB E->B leakage cleaning.

Adapted from Wang et al. (2025) "Hierarchical Reasoning Model" (arXiv:2506.21734)
for continuous-valued spherical harmonic coefficient sequences.

Core principle: Two coupled recurrent modules at different timescales --
  H-module (slow): captures global correlations across harmonic modes (low-ell)
  L-module (fast): handles local mode-to-mode interactions (high-ell)

Training uses 1-step gradient approximation with deep supervision.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# ──────────────────────────────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────────────────────────────

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal initialization (in-place), following the HRM reference."""
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(a * std, b * std)
    return tensor


# ──────────────────────────────────────────────────────────────────────
# Hidden state carry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HRMCarry:
    """Carries the hidden states z_H and z_L between forward passes.

    Shapes: (batch, seq_len, hidden_size)
    """
    z_H: torch.Tensor
    z_L: torch.Tensor


# ──────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Functional RMS normalization without learnable parameters.

    The HRM paper excludes scale/bias from RMSNorm for Q-learning stability
    (parameters remain bounded under weight decay).
    """
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


# ──────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings
# ──────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cosine/sine cache.

    Positions along the alm sequence encode the (l, m) ordering from
    HEALPix.  RoPE lets attention exploit relative position, mapping to
    relative ell separation within each m-band.
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key.  q, k: (B, seq_len, H, D_h)."""
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2)   # (seq_len, 1, D_h)
    sin = sin.unsqueeze(-2)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


# ──────────────────────────────────────────────────────────────────────
# Linear layer with truncated LeCun normal init
# ──────────────────────────────────────────────────────────────────────

class CastedLinear(nn.Module):
    """Linear layer with truncated LeCun normal init and dtype casting."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features),
                               std=1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


# ──────────────────────────────────────────────────────────────────────
# Feed-forward: SwiGLU
# ──────────────────────────────────────────────────────────────────────

def _find_multiple(a: int, b: int) -> int:
    """Ceiling division then multiply: smallest multiple of b >= a."""
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU activation (SiLU-Gate Linear Unit)."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 64)
        self.gate_up = CastedLinear(hidden_size, inter * 2)
        self.down = CastedLinear(inter, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


# ──────────────────────────────────────────────────────────────────────
# Multi-head Self-Attention
# ──────────────────────────────────────────────────────────────────────

SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,     # fastest when available
    SDPBackend.CUDNN_ATTENTION,     # PyTorch >= 2.5 adds cuDNN SDPA
    SDPBackend.EFFICIENT_ATTENTION, # xFormers-style
    SDPBackend.MATH,                # fallback
]


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, using PyTorch SDPA."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = CastedLinear(hidden_size, 3 * hidden_size)
        self.out_proj = CastedLinear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor,
                cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)            # each (B, N, H, D_h)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, k, cos[:N], sin[:N])

        # (B, H, N, D_h) for SDPA
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        with sdpa_kernel(SDPA_BACKENDS):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────
# Transformer Block (Post-Norm architecture)
# ──────────────────────────────────────────────────────────────────────

class HRMBlock(nn.Module):
    """Single Post-Norm transformer block.

    Post-Norm differs from the Pre-Norm used in most LLMs: the residual
    connection is applied *before* normalization:

        x = RMSNorm(x + Attention(x))
        x = RMSNorm(x + SwiGLU(x))

    This architecture is critical for Q-learning stability in ACT and
    keeps parameters bounded under weight decay (see HRM paper Sec. 2).
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 expansion: float = 4.0, norm_eps: float = 1e-5):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.norm_eps = norm_eps

    def forward(self, x: torch.Tensor,
                cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = rms_norm(x + self.attn(x, cos_sin), self.norm_eps)
        x = rms_norm(x + self.mlp(x), self.norm_eps)
        return x


# ──────────────────────────────────────────────────────────────────────
# Reasoning Module
# ──────────────────────────────────────────────────────────────────────

class ReasoningModule(nn.Module):
    """Stack of HRMBlocks with additive input injection.

    At each reasoning step:
      1. hidden_states += input_injection    (additive, not concat)
      2. hidden_states  = Block_1(hidden_states)
      3. hidden_states  = Block_2(hidden_states)
      ...
    """

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int,
                 expansion: float = 4.0, norm_eps: float = 1e-5):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMBlock(hidden_size, num_heads, expansion, norm_eps)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor,
                cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)
        return hidden_states


# ──────────────────────────────────────────────────────────────────────
# HRM
# ──────────────────────────────────────────────────────────────────────

class HRM(nn.Module):
    """Hierarchical Reasoning Model for continuous sequence-to-sequence tasks.

    Adapted from the discrete-token HRM (Wang et al. 2025) for CMB
    polarimetric cleaning in spherical harmonic space.

    The key adaptation: token embeddings are replaced by linear
    projections (input_proj, output_head) since the data lives on a
    continuous manifold of spherical harmonic coefficients rather than a
    discrete vocabulary.

    The hierarchical structure maps naturally to CMB angular scales:
      - L-module (fast, many steps per cycle): local interactions
        between nearby harmonic modes --> high-ell cleaning
      - H-module (slow, one step per cycle): global integration
        across the full mode spectrum --> low-ell cleaning

    Training uses the 1-step gradient approximation: all N*T reasoning
    iterations run under torch.no_grad() except the final L + H step
    pair, which carries gradients.  This gives O(1) memory cost
    regardless of reasoning depth.

    Args:
        input_dim:    Feature channels (4 = [Re_E, Im_E, Re_B, Im_B]).
        hidden_size:  Transformer hidden dimension.
        H_layers:     Transformer blocks in the H-module.
        L_layers:     Transformer blocks in the L-module.
        H_cycles:     High-level reasoning cycles (N in paper).
        L_cycles:     Low-level steps per cycle (T in paper).
        num_heads:    Attention heads.
        expansion:    SwiGLU expansion ratio.
        max_seq_len:  Maximum sequence length for RoPE cache.
        rope_theta:   RoPE base frequency.
        norm_eps:     RMSNorm epsilon.
    """

    def __init__(self, input_dim: int = 4,
                 hidden_size: int = 128,
                 H_layers: int = 4,
                 L_layers: int = 4,
                 H_cycles: int = 2,
                 L_cycles: int = 3,
                 num_heads: int = 4,
                 expansion: float = 4.0,
                 max_seq_len: int = 8192,
                 rope_theta: float = 10000.0,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # I/O projections for continuous data
        self.input_proj = CastedLinear(input_dim, hidden_size)
        self.output_head = CastedLinear(hidden_size, input_dim)

        # Positional encoding
        head_dim = hidden_size // num_heads
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

        # Reasoning modules
        self.L_level = ReasoningModule(L_layers, hidden_size, num_heads, expansion, norm_eps)
        self.H_level = ReasoningModule(H_layers, hidden_size, num_heads, expansion, norm_eps)

        # Fixed initial states (truncated normal, NOT learned -- kept fixed
        # throughout training, following the paper Sec. 2)
        self.register_buffer(
            'H_init', trunc_normal_init_(torch.empty(hidden_size), std=1.0), persistent=True
        )
        self.register_buffer(
            'L_init', trunc_normal_init_(torch.empty(hidden_size), std=1.0), persistent=True
        )

    # ── carry management ──────────────────────────────────────────────

    def initial_carry(self, batch_size: int, seq_len: int) -> HRMCarry:
        """Create a fresh carry initialized to (H_init, L_init)."""
        return HRMCarry(
            z_H=self.H_init.expand(batch_size, seq_len, -1).clone(),
            z_L=self.L_init.expand(batch_size, seq_len, -1).clone(),
        )

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                carry: Optional[HRMCarry] = None) -> Tuple[torch.Tensor, HRMCarry]:
        """
        Single HRM forward pass with 1-step gradient approximation.

        Args:
            x:     Input alm tensor, shape (B, N, input_dim).
            carry: Hidden state from a previous segment.  If *None* the
                   carry is initialized from the fixed buffers.

        Returns:
            output:    Predicted tensor, shape (B, N, input_dim).
            new_carry: Detached carry for the next deep-supervision segment.
        """
        B, N, _ = x.shape

        if carry is None:
            carry = self.initial_carry(B, N)

        # Project continuous input into hidden space
        input_emb = self.input_proj(x)

        cos_sin = self.rotary_emb()

        # ── no-grad reasoning iterations ──────────────────────────────
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    # Skip the very last (h, l) pair -- done below with grad
                    if h == self.H_cycles - 1 and l == self.L_cycles - 1:
                        continue
                    z_L = self.L_level(z_L, z_H + input_emb, cos_sin)

                if h == self.H_cycles - 1:
                    continue
                z_H = self.H_level(z_H, z_L, cos_sin)

        assert not z_H.requires_grad and not z_L.requires_grad

        # ── 1-step with gradient ──────────────────────────────────────
        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin)

        # Prediction from the H-module's final state
        output = self.output_head(z_H)

        new_carry = HRMCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return output, new_carry
