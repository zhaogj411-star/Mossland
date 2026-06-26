"""Channel-concat conditioning assembly."""

from __future__ import annotations

import torch

from .patchify import mask_to_token_level, patchify_1d_batched


def assemble_model_input(
    noisy_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    gated_context = context_tokens * mask_token
    return torch.cat([noisy_tokens, gated_context, mask_token], dim=-1)


def split_input_width(latent_token_dim: int) -> int:
    return 2 * latent_token_dim + 1


def build_context_tokens(
    context_latent: torch.Tensor,
    cond_mask: torch.Tensor,
    patch_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_tokens = patchify_1d_batched(context_latent, patch_t)
    batch = cond_mask.shape[0]
    mask_tokens = torch.stack(
        [mask_to_token_level(cond_mask[idx], patch_t, mode="any") for idx in range(batch)],
        dim=0,
    )
    return context_tokens, mask_tokens

