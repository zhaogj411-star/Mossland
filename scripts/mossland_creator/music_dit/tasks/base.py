"""Task abstraction for Music-DiT masking."""

from __future__ import annotations

import abc
import random
from typing import Optional

import torch

from ..core.contract import TaskOutput


class Task(abc.ABC):
    name: str = "task"
    same_domain: bool = True
    loss_on_gen_only: bool = True

    @abc.abstractmethod
    def apply(self, z0: torch.Tensor, meta: dict, rng: random.Random) -> TaskOutput:
        ...

    @staticmethod
    def _empty_context(z0: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z0)

    @staticmethod
    def _zeros_mask(z0: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, z0.shape[-1], dtype=z0.dtype, device=z0.device)

    @staticmethod
    def _ones_mask(z0: torch.Tensor) -> torch.Tensor:
        return torch.ones(1, z0.shape[-1], dtype=z0.dtype, device=z0.device)

    def _finalize(
        self,
        context_latent: torch.Tensor,
        cond_mask: torch.Tensor,
        gen_mask: torch.Tensor,
        text_prompt: Optional[str] = None,
        info: Optional[dict] = None,
    ) -> TaskOutput:
        loss_mask = gen_mask.clone() if self.loss_on_gen_only else self._ones_mask(context_latent)
        if self.loss_on_gen_only:
            loss_mask = loss_mask * (1.0 - cond_mask)
            if loss_mask.sum() == 0:
                loss_mask = gen_mask.clone()
        return TaskOutput(
            context_latent=context_latent,
            cond_mask=cond_mask,
            gen_mask=gen_mask,
            loss_mask=loss_mask,
            text_prompt=text_prompt,
            info={"task": self.name, **(info or {})},
        )

