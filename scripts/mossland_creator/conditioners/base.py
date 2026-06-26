# Copyright (c) 2025.
"""Conditioner interface."""

from __future__ import annotations

import abc
from typing import Dict, List

import torch


class Conditioner(abc.ABC):
    output_dim: int
    max_len: int

    def __init__(self, ucg_rate: float = 0.0, device: str = "cpu"):
        self.ucg_rate = ucg_rate
        self.device = device

    @abc.abstractmethod
    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        """List[sample-meta] -> cross-attention embeddings and mask."""

    def null_embedding(self, batch_size: int) -> Dict[str, torch.Tensor]:
        emb = torch.zeros(batch_size, self.max_len, self.output_dim, device=self.device)
        mask = torch.zeros(batch_size, self.max_len, device=self.device)
        return {"crossattn_emb": emb, "crossattn_mask": mask}

    def to(self, device: str):
        self.device = device
        return self


def pad_or_trim(emb: torch.Tensor, mask: torch.Tensor, max_len: int):
    b, s, e = emb.shape
    if s >= max_len:
        return emb[:, :max_len], mask[:, :max_len]
    pad = max_len - s
    emb = torch.nn.functional.pad(emb, (0, 0, 0, pad))
    mask = torch.nn.functional.pad(mask, (0, pad))
    return emb, mask


class MultiConditioner(Conditioner):
    def __init__(self, conditioners: List[Conditioner]):
        assert conditioners, "need >=1 conditioner"
        dims = {c.output_dim for c in conditioners}
        assert len(dims) == 1, f"conditioners must share output_dim, got {dims}"
        super().__init__(ucg_rate=0.0, device=conditioners[0].device)
        self.conditioners = conditioners
        self.output_dim = conditioners[0].output_dim
        self.max_len = sum(c.max_len for c in conditioners)

    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        embs, masks = [], []
        for c in self.conditioners:
            out = c.encode(samples)
            embs.append(out["crossattn_emb"])
            masks.append(out["crossattn_mask"])
        return {
            "crossattn_emb": torch.cat(embs, dim=1),
            "crossattn_mask": torch.cat(masks, dim=1),
        }

    def to(self, device: str):
        for c in self.conditioners:
            c.to(device)
        self.device = device
        return self
