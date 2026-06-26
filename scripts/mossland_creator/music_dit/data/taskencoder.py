"""Energon task encoder for prepared Mossland Music-DiT shards.

Raw shard schema:
- ``.pth`` / ``pth``: latent bank, usually ``dict[str, Tensor]`` such as
  ``mixture``, ``vocals`` and ``accompaniment``. Each latent is ``[256, T]``.
- ``.pickle`` / ``pickle``: UMT5 text embedding, usually ``[3500, 768]``.
- ``.json`` / ``json``: metadata such as ``frame_rate``, ``codec`` and
  ``latent_frames``.

Training dataloader batch schema after ``MusicDiffusionTaskEncoder``:
- ``latent``: ``Tensor[B, S, 256]`` target latent tokens.
- ``context_latent``: ``Tensor[B, S, 256]`` conditional latent tokens.
- ``cond_mask``: ``Tensor[B, S, 1]`` context-valid token mask.
- ``crossattn_emb``: ``Tensor[B, 3500, 768]`` UMT5 text embedding.
- ``crossattn_mask``: ``Tensor[B, 3500]`` text-valid mask.
- ``loss_mask``: ``Tensor[B, S]`` tokens included in loss.
- ``pos_ids``: ``Tensor[B, S, 1]`` 1-D position ids.
- ``seq_len_q``: ``Tensor[B]`` true latent token lengths before padding.
- ``seq_len_kv``: ``Tensor[B]`` valid text token lengths.
- ``latent_shape``: ``Tensor[B, 2]`` original ``[channels, frames]``.
- ``task_name``: ``list[str]`` sampled task names.
- ``target_key``: ``list[str]`` target latent-bank keys, currently usually
  ``mixture``.

With the current flow-matching YAML, ``patch_t=1`` and
``text_embedding_padding_size=3500``. When ``max_duration_seconds`` is set,
``S`` is the configured duration converted through metadata ``frame_rate``
(for SAME-L, 300 seconds is about 3229 tokens); shorter samples are padded
with zero latent/context tokens and zero ``loss_mask`` after ``seq_len_q``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from megatron.energon import DefaultTaskEncoder, Sample, SkipSample
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from ..core import contract as C
from ..data.assemble import encode_sample_to_tokens
from ..tasks.sampler import TaskMixSampler


@dataclass
class MusicDiffusionSample(Sample):
    latent: torch.Tensor
    context_latent: torch.Tensor
    cond_mask: torch.Tensor
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    loss_mask: torch.Tensor
    pos_ids: torch.Tensor
    seq_len_q: torch.Tensor
    seq_len_kv: torch.Tensor
    latent_shape: torch.Tensor
    task_name: object
    target_key: object

    def to_dict(self) -> dict:
        return {
            C.KEY_LATENT: self.latent,
            C.KEY_CONTEXT: self.context_latent,
            C.KEY_COND_MASK: self.cond_mask,
            C.KEY_CROSSATTN: self.crossattn_emb,
            C.KEY_CROSSATTN_MASK: self.crossattn_mask,
            C.KEY_LOSS_MASK: self.loss_mask,
            C.KEY_POS_IDS: self.pos_ids,
            C.KEY_SEQLEN_Q: self.seq_len_q,
            C.KEY_SEQLEN_KV: self.seq_len_kv,
            C.KEY_LATENT_SHAPE: self.latent_shape,
            C.KEY_TASK: self.task_name,
            C.KEY_TARGET_KEY: self.target_key,
        }


@stateless
def cook_music(sample: dict) -> dict:
    latent_bank = sample[".pth"] if ".pth" in sample else sample["pth"]
    text = sample[".pickle"] if ".pickle" in sample else sample["pickle"]
    meta = sample[".json"] if ".json" in sample else sample["json"]
    return dict(
        **basic_sample_keys(sample),
        latent_bank=latent_bank,
        text=text,
        json=meta,
    )


DEFAULT_TASK_MIX = [
    {"name": "text2music", "weight": 2.0, "target_key": "mixture"},
    {"name": "continuation", "weight": 1.0, "target_key": "mixture", "min_ctx": 0.25, "max_ctx": 0.75},
    {"name": "inpaint", "weight": 1.0, "target_key": "mixture", "min_hole": 0.1, "max_hole": 0.5},
    {"name": "cover", "weight": 0.5, "target_key": "mixture", "noise_std": 1.0},
]


class MusicDiffusionTaskEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook_music)]

    def __init__(
        self,
        *args,
        seq_length: int | None = 256,
        patch_t: int = 1,
        text_embedding_padding_size: int = 256,
        task_specs: Optional[list[dict]] = None,
        context_dropout: float = 0.0,
        text_dropout: float = 0.0,
        max_duration_seconds: float | None = None,
        max_abs_latent: float = 1e3,
        pos_dim: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seq_length = None if seq_length is None else int(seq_length)
        self.patch_t = int(patch_t)
        self.text_embedding_padding_size = int(text_embedding_padding_size)
        self.text_dropout = float(text_dropout)
        self.max_duration_seconds = None if max_duration_seconds is None else float(max_duration_seconds)
        self.max_abs_latent = float(max_abs_latent)
        self.pos_dim = int(pos_dim)
        self.task_sampler = TaskMixSampler.from_config(
            task_specs or DEFAULT_TASK_MIX,
            context_dropout=context_dropout,
        )

    @staticmethod
    def _coerce_latent_bank(latent_bank_raw) -> dict[str, torch.Tensor]:
        if torch.is_tensor(latent_bank_raw):
            return {"mixture": latent_bank_raw.float()}
        if isinstance(latent_bank_raw, dict):
            bank = {}
            for key, value in latent_bank_raw.items():
                tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
                tensor = tensor.float()
                if tensor.ndim != 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)
                bank[str(key)] = tensor
            if bank:
                return bank
        raise SkipSample()

    @staticmethod
    def _trim_to_common_frames(latent_bank: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = min(tensor.shape[-1] for tensor in latent_bank.values())
        return {key: tensor[..., :frames] for key, tensor in latent_bank.items()}

    def _clip_latent_bank_from_start(
        self,
        latent_bank: dict[str, torch.Tensor],
        meta: dict,
    ) -> tuple[dict[str, torch.Tensor], int | None]:
        frames = next(iter(latent_bank.values())).shape[-1]
        target_frames = None
        if self.max_duration_seconds is not None:
            frame_rate = float(meta.get("frame_rate") or 0.0)
            if frame_rate > 0.0:
                target_frames = max(1, int(self.max_duration_seconds * frame_rate))
        if target_frames is None and self.seq_length is not None:
            target_frames = max(1, self.seq_length * self.patch_t)
        if target_frames is None:
            return latent_bank, None
        clipped_frames = min(frames, target_frames)
        return {key: tensor[..., :clipped_frames] for key, tensor in latent_bank.items()}, target_frames

    @staticmethod
    def _coerce_text_embedding(text_raw) -> torch.Tensor:
        text = text_raw if torch.is_tensor(text_raw) else torch.as_tensor(np.asarray(text_raw))
        text = text.float()
        if text.ndim == 1:
            text = text.unsqueeze(0)
        if text.ndim == 3:
            text = text[0]
        if text.ndim != 2:
            raise SkipSample()
        return text

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> MusicDiffusionSample:
        latent_bank = self._trim_to_common_frames(self._coerce_latent_bank(sample["latent_bank"]))
        if any(torch.isnan(latent).any() or torch.isinf(latent).any() for latent in latent_bank.values()):
            raise SkipSample()
        if max(latent.abs().max().item() for latent in latent_bank.values()) > self.max_abs_latent:
            raise SkipSample()

        rng = random
        meta = dict(sample.get("json", {}) or {})
        latent_bank, target_frames = self._clip_latent_bank_from_start(latent_bank, meta)
        sampled_task = self.task_sampler.sample_task(rng)
        if sampled_task.target_key not in latent_bank:
            raise SkipSample()
        target_latent = latent_bank[sampled_task.target_key]

        for key, latent in latent_bank.items():
            meta[f"{key}_latent"] = latent
        if "mixture" in latent_bank and "cond_latent" not in meta:
            meta["cond_latent"] = latent_bank["mixture"]

        task_out = sampled_task.task.apply(target_latent, meta, rng)
        task_out = self.task_sampler.maybe_dropout_context(task_out, rng)
        seq_length = None
        if target_frames is not None:
            seq_length = (target_frames + self.patch_t - 1) // self.patch_t
        elif self.seq_length is not None:
            seq_length = self.seq_length
        token_dict = encode_sample_to_tokens(
            target_latent,
            task_out,
            self.patch_t,
            pos_dim=self.pos_dim,
            seq_length=seq_length,
        )
        if token_dict is None:
            raise SkipSample()

        text = self._coerce_text_embedding(sample["text"])
        pad = self.text_embedding_padding_size
        valid = min(text.shape[0], pad)
        if text.shape[0] > pad:
            text = text[:pad]
        else:
            text = F.pad(text, (0, 0, 0, pad - text.shape[0]))
        text_mask = torch.zeros(pad, dtype=torch.float32)
        text_mask[:valid] = 1.0
        if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
            text = torch.zeros_like(text)
            # Keep at least one zero-valued text token visible. PyTorch MHA
            # returns NaNs when every cross-attention key is masked.
            text_mask.zero_()
            text_mask[0] = 1.0

        return MusicDiffusionSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__"),
            __sources__=sample.get("__sources__"),
            latent=token_dict[C.KEY_LATENT].float(),
            context_latent=token_dict[C.KEY_CONTEXT].float(),
            cond_mask=token_dict[C.KEY_COND_MASK].float(),
            crossattn_emb=text.float(),
            crossattn_mask=text_mask,
            loss_mask=token_dict[C.KEY_LOSS_MASK].float(),
            pos_ids=token_dict[C.KEY_POS_IDS],
            seq_len_q=token_dict[C.KEY_SEQLEN_Q],
            seq_len_kv=torch.tensor(valid, dtype=torch.int32),
            latent_shape=token_dict[C.KEY_LATENT_SHAPE],
            task_name=task_out.info.get("task", sampled_task.task.name),
            target_key=sampled_task.target_key,
        )

    @stateless
    def batch(self, samples: list[MusicDiffusionSample]) -> dict:
        return super().batch(samples).to_dict()
