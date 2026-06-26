"""Single source of truth for the Music-DiT tensor contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

KEY_LATENT = "latent"
KEY_CONTEXT = "context_latent"
KEY_COND_MASK = "cond_mask"
KEY_LOSS_MASK = "loss_mask"
KEY_CROSSATTN = "crossattn_emb"
KEY_CROSSATTN_MASK = "crossattn_mask"
KEY_POS_IDS = "pos_ids"
KEY_SEQLEN_Q = "seq_len_q"
KEY_SEQLEN_KV = "seq_len_kv"
KEY_LATENT_SHAPE = "latent_shape"
KEY_TASK = "task_name"
KEY_TARGET_KEY = "target_key"


@dataclass
class TaskOutput:
    """Result of applying one task to one clean latent ``[C, T]`` sample."""

    context_latent: object
    cond_mask: object
    gen_mask: object
    loss_mask: Optional[object] = None
    text_prompt: Optional[str] = None
    info: dict = field(default_factory=dict)

