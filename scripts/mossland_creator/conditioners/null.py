# Copyright (c) 2025.
"""Unconditional conditioner and deterministic random text conditioner."""

from __future__ import annotations

from typing import Dict, List

import torch

from .base import Conditioner
from .registry import register_conditioner


class NullConditioner(Conditioner):
    def __init__(self, output_dim: int = 4096, max_len: int = 1, device: str = "cpu"):
        super().__init__(ucg_rate=1.0, device=device)
        self.output_dim = output_dim
        self.max_len = max_len

    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        b = len(samples)
        return {
            "crossattn_emb": torch.zeros(b, self.max_len, self.output_dim, device=self.device),
            "crossattn_mask": torch.zeros(b, self.max_len, device=self.device),
        }


class RandomTextConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int = 4096,
        max_len: int = 8,
        ucg_rate: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(ucg_rate=ucg_rate, device=device)
        self.output_dim = output_dim
        self.max_len = max_len

    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        b = len(samples)
        g = torch.Generator().manual_seed(
            abs(hash(str([s.get("__key__", i) for i, s in enumerate(samples)]))) % (2**31)
        )
        emb = torch.randn(b, self.max_len, self.output_dim, generator=g).to(self.device)
        mask = torch.ones(b, self.max_len, device=self.device)
        return {"crossattn_emb": emb, "crossattn_mask": mask}


@register_conditioner("null")
def _build_null(**kw):
    return NullConditioner(**kw)


@register_conditioner("random")
def _build_random(**kw):
    return RandomTextConditioner(**kw)
