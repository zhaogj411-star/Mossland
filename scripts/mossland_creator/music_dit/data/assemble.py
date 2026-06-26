"""Assemble latent + task output into token-space batch tensors."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..core import contract as C
from ..core.contract import TaskOutput
from ..core.patchify import mask_to_token_level, pad_to_multiple, patchify_1d
from ..core.pos_emb import build_pos_ids_1d, build_pos_ids_3d_shim


def encode_sample_to_tokens(
    z0: torch.Tensor,
    task_out: TaskOutput,
    patch_t: int,
    pos_dim: int = 1,
    seq_length: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Convert one latent/task pair into the token contract."""
    channels, frames = z0.shape
    z0, _ = pad_to_multiple(z0, patch_t)
    context, _ = pad_to_multiple(task_out.context_latent, patch_t)
    cond_mask, _ = pad_to_multiple(task_out.cond_mask, patch_t)
    gen_mask, _ = pad_to_multiple(task_out.gen_mask, patch_t)
    loss_mask = task_out.loss_mask if task_out.loss_mask is not None else gen_mask
    loss_mask, _ = pad_to_multiple(loss_mask, patch_t)

    target_tokens = patchify_1d(z0, patch_t)
    context_tokens = patchify_1d(context, patch_t)
    cond_tokens = mask_to_token_level(cond_mask, patch_t, mode="any")
    loss_tokens = mask_to_token_level(loss_mask, patch_t, mode="any").squeeze(-1)
    seq_len = target_tokens.shape[0]

    if seq_length is not None:
        if seq_len > seq_length:
            return None
        pad = seq_length - seq_len
        target_tokens = F.pad(target_tokens, (0, 0, 0, pad))
        context_tokens = F.pad(context_tokens, (0, 0, 0, pad))
        cond_tokens = F.pad(cond_tokens, (0, 0, 0, pad))
        padded_loss = torch.zeros(seq_length, dtype=loss_tokens.dtype)
        padded_loss[:seq_len] = loss_tokens
        loss_tokens = padded_loss
        out_len = seq_length
    else:
        out_len = seq_len

    if pos_dim == 1:
        pos_ids = build_pos_ids_1d(out_len)
    else:
        pos_ids = build_pos_ids_3d_shim(out_len)

    return {
        C.KEY_LATENT: target_tokens,
        C.KEY_CONTEXT: context_tokens,
        C.KEY_COND_MASK: cond_tokens,
        C.KEY_LOSS_MASK: loss_tokens,
        C.KEY_POS_IDS: pos_ids,
        C.KEY_SEQLEN_Q: torch.tensor(seq_len, dtype=torch.int32),
        C.KEY_LATENT_SHAPE: torch.tensor([channels, frames], dtype=torch.int32),
    }


def collate_tokens(
    samples: list[dict[str, torch.Tensor]],
    crossattn_emb: torch.Tensor,
    crossattn_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([sample[key] for sample in samples], dim=0)

    batch_size = len(samples)
    text_len = crossattn_emb.shape[1]
    return {
        C.KEY_LATENT: stack(C.KEY_LATENT).to(dtype),
        C.KEY_CONTEXT: stack(C.KEY_CONTEXT).to(dtype),
        C.KEY_COND_MASK: stack(C.KEY_COND_MASK).to(dtype),
        C.KEY_LOSS_MASK: stack(C.KEY_LOSS_MASK).to(dtype),
        C.KEY_POS_IDS: stack(C.KEY_POS_IDS),
        C.KEY_CROSSATTN: crossattn_emb.to(dtype),
        C.KEY_CROSSATTN_MASK: crossattn_mask.to(dtype),
        C.KEY_SEQLEN_Q: stack(C.KEY_SEQLEN_Q),
        C.KEY_SEQLEN_KV: torch.full((batch_size,), text_len, dtype=torch.int32),
        C.KEY_LATENT_SHAPE: stack(C.KEY_LATENT_SHAPE),
    }

