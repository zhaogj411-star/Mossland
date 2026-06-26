"""1-D patchify helpers for audio latents."""

from __future__ import annotations

import torch


def patchify_1d(latent: torch.Tensor, patch_t: int) -> torch.Tensor:
    """Convert ``[C, T]`` latent to ``[S, D]`` tokens."""
    channels, frames = latent.shape
    if frames % patch_t != 0:
        raise ValueError(f"T={frames} is not divisible by patch_t={patch_t}")
    steps = frames // patch_t
    return latent.reshape(channels, steps, patch_t).permute(1, 2, 0).reshape(steps, patch_t * channels)


def unpatchify_1d(tokens: torch.Tensor, patch_t: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`patchify_1d`."""
    steps, dim = tokens.shape
    expected_dim = patch_t * channels
    if dim != expected_dim:
        raise ValueError(f"expected token dim {expected_dim}, got {dim}")
    return tokens.reshape(steps, patch_t, channels).permute(2, 0, 1).reshape(channels, steps * patch_t)


def patchify_1d_batched(latent: torch.Tensor, patch_t: int) -> torch.Tensor:
    """Convert ``[B, C, T]`` latent to ``[B, S, D]`` tokens."""
    batch, channels, frames = latent.shape
    if frames % patch_t != 0:
        raise ValueError(f"T={frames} is not divisible by patch_t={patch_t}")
    steps = frames // patch_t
    return latent.reshape(batch, channels, steps, patch_t).permute(0, 2, 3, 1).reshape(
        batch, steps, patch_t * channels
    )


def unpatchify_1d_batched(tokens: torch.Tensor, patch_t: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`patchify_1d_batched`."""
    batch, steps, dim = tokens.shape
    expected_dim = patch_t * channels
    if dim != expected_dim:
        raise ValueError(f"expected token dim {expected_dim}, got {dim}")
    return tokens.reshape(batch, steps, patch_t, channels).permute(0, 3, 1, 2).reshape(
        batch, channels, steps * patch_t
    )


def pad_to_multiple(latent: torch.Tensor, patch_t: int, value: float = 0.0) -> tuple[torch.Tensor, int]:
    """Pad latent time axis to a multiple of ``patch_t``."""
    channels, frames = latent.shape
    pad = (patch_t - frames % patch_t) % patch_t
    if pad == 0:
        return latent, frames
    padded = latent.new_full((channels, frames + pad), value)
    padded[:, :frames] = latent
    return padded, frames


def mask_to_token_level(mask_1t: torch.Tensor, patch_t: int, mode: str = "any") -> torch.Tensor:
    """Downsample latent-rate ``[1, T]`` mask to token-rate ``[S, 1]``."""
    frames = mask_1t.shape[-1]
    if frames % patch_t != 0:
        raise ValueError(f"T={frames} is not divisible by patch_t={patch_t}")
    mask = mask_1t.reshape(1, frames // patch_t, patch_t).permute(1, 2, 0).float()
    if mode == "any":
        token_mask = (mask.sum(dim=1) > 0).float()
    elif mode == "all":
        token_mask = (mask.sum(dim=1) == patch_t).float()
    elif mode == "mean":
        token_mask = mask.mean(dim=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return token_mask

