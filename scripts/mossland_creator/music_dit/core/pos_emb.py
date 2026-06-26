"""1-D positional id / embedding helpers."""

from __future__ import annotations

import math

import torch


def build_pos_ids_1d(seq_len: int, device=None) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(-1)


def build_pos_ids_3d_shim(seq_len: int, device=None) -> torch.Tensor:
    t = torch.arange(seq_len, device=device, dtype=torch.long)
    z = torch.zeros_like(t)
    return torch.stack([t, z, z], dim=-1)


def sincos_1d(positions: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding for scalar positions."""
    if dim % 2 != 0:
        raise ValueError(f"sincos dim must be even, got {dim}")
    positions = positions.float()
    if positions.ndim > 1 and positions.shape[-1] == 1:
        positions = positions[..., 0]
    orig_shape = positions.shape
    positions = positions.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=positions.device).float() / half
    )
    args = positions[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb.reshape(*orig_shape, dim)

