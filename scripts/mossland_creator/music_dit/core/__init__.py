"""Pure-torch Music-DiT tensor contract helpers."""

from . import contract  # noqa: F401
from .masking import assemble_model_input, split_input_width  # noqa: F401
from .patchify import (  # noqa: F401
    mask_to_token_level,
    pad_to_multiple,
    patchify_1d,
    patchify_1d_batched,
    unpatchify_1d,
    unpatchify_1d_batched,
)
from .pos_emb import build_pos_ids_1d, build_pos_ids_3d_shim, sincos_1d  # noqa: F401

