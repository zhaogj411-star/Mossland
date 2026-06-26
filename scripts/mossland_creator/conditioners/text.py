# Copyright (c) 2025.
"""Text conditioners (T5 / CLAP)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from .base import Conditioner, pad_or_trim
from .prompt_builder import PromptBuilder, PromptBuilderConfig
from .registry import register_conditioner

DEFAULT_UMT5_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "checkpoints" / "umt5-base"
)


def _resolve_prompt(sample: dict, builder: PromptBuilder) -> str:
    if sample.get("text_prompt"):
        return str(sample["text_prompt"])
    if sample.get("prompt"):
        return str(sample["prompt"])
    return builder.build(sample)


class T5TextConditioner(Conditioner):
    def __init__(
        self,
        version: str = "google/t5-v1_1-xxl",
        max_len: int = 128,
        ucg_rate: float = 0.1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        prompt_builder: Optional[PromptBuilder] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__(ucg_rate=ucg_rate, device=device)
        self.version = version
        self.max_len = max_len
        self.dtype = dtype
        self.builder = prompt_builder or PromptBuilder(PromptBuilderConfig())
        self.output_dim = output_dim or 4096
        self._tok = None
        self._enc = None

    def _lazy(self):
        if self._enc is None:
            from transformers import T5EncoderModel, T5Tokenizer

            self._tok = T5Tokenizer.from_pretrained(self.version)
            self._enc = T5EncoderModel.from_pretrained(self.version, torch_dtype=self.dtype)
            self._enc = self._enc.to(self.device).eval()
            for p in self._enc.parameters():
                p.requires_grad = False
            self.output_dim = self._enc.config.d_model
        return self._tok, self._enc

    @torch.no_grad()
    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        tok, enc = self._lazy()
        prompts = [_resolve_prompt(s, self.builder) for s in samples]
        batch = tok(
            prompts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        emb = enc(input_ids=ids, attention_mask=attn).last_hidden_state
        emb, mask = pad_or_trim(emb.float(), attn.float(), self.max_len)
        return {"crossattn_emb": emb, "crossattn_mask": mask}


class UMT5TextConditioner(Conditioner):
    """UMT5 encoder wired the same way as LongCat-AudioDiT.

    LongCat-AudioDiT requests `output_hidden_states=True`, takes
    `last_hidden_state`, optionally layer-normalizes it, and then optionally
    adds `hidden_states[0]` after applying the same normalization. We mirror
    that behavior here so the conditioner can produce the same text features.
    """

    def __init__(
        self,
        version: str = str(DEFAULT_UMT5_CHECKPOINT),
        max_len: int = 3500,
        ucg_rate: float = 0.1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        prompt_builder: Optional[PromptBuilder] = None,
        output_dim: Optional[int] = None,
        text_add_embed: bool = True,
        text_norm_feat: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__(ucg_rate=ucg_rate, device=device)
        self.version = version
        self.max_len = max_len
        self.dtype = dtype
        self.builder = prompt_builder or PromptBuilder(PromptBuilderConfig())
        self.output_dim = output_dim or 768
        self.text_add_embed = text_add_embed
        self.text_norm_feat = text_norm_feat
        self.norm_eps = norm_eps
        self._tok = None
        self._enc = None

    def _lazy(self):
        if self._enc is None:
            from transformers import AutoTokenizer, UMT5EncoderModel

            load_source = Path(self.version)
            from_pretrained_kwargs = {}
            if load_source.exists():
                from_pretrained_kwargs["local_files_only"] = True

            self._tok = AutoTokenizer.from_pretrained(
                self.version,
                **from_pretrained_kwargs,
            )
            self._enc = UMT5EncoderModel.from_pretrained(
                self.version,
                torch_dtype=self.dtype,
                **from_pretrained_kwargs,
            )
            self._enc = self._enc.to(self.device).eval()
            for p in self._enc.parameters():
                p.requires_grad = False
            self.output_dim = self._enc.config.d_model
        return self._tok, self._enc

    @torch.no_grad()
    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        tok, enc = self._lazy()
        prompts = [_resolve_prompt(s, self.builder) for s in samples]
        batch = tok(
            prompts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        output = enc(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
        )
        emb = output.last_hidden_state
        d_model = self._enc.config.d_model

        if self.text_norm_feat:
            emb = torch.nn.functional.layer_norm(emb, (d_model,), eps=self.norm_eps)

        if self.text_add_embed:
            # Match LongCat-AudioDiT exactly: add hidden_states[0], not an
            # intermediate transformer block output.
            first_hidden = output.hidden_states[0]
            if self.text_norm_feat:
                first_hidden = torch.nn.functional.layer_norm(
                    first_hidden,
                    (d_model,),
                    eps=self.norm_eps,
                )
            emb = emb + first_hidden

        emb, mask = pad_or_trim(emb.float(), attn.float(), self.max_len)
        return {"crossattn_emb": emb, "crossattn_mask": mask}


class CLAPTextConditioner(Conditioner):
    def __init__(
        self,
        version: str = "laion/clap-htsat-unfused",
        ucg_rate: float = 0.1,
        device: str = "cpu",
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        super().__init__(ucg_rate=ucg_rate, device=device)
        self.version = version
        self.max_len = 1
        self.output_dim = 512
        self.builder = prompt_builder or PromptBuilder(PromptBuilderConfig())
        self._model = None
        self._proc = None

    def _lazy(self):
        if self._model is None:
            from transformers import ClapModel, ClapProcessor

            self._proc = ClapProcessor.from_pretrained(self.version)
            self._model = ClapModel.from_pretrained(self.version).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad = False
            self.output_dim = self._model.config.projection_dim
        return self._proc, self._model

    @torch.no_grad()
    def encode(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        proc, model = self._lazy()
        prompts = [_resolve_prompt(s, self.builder) for s in samples]
        inp = proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        feat = model.get_text_features(**inp).float()
        emb = feat.unsqueeze(1)
        mask = torch.ones(emb.shape[0], 1, device=self.device)
        return {"crossattn_emb": emb, "crossattn_mask": mask}


@register_conditioner("t5")
def _build_t5(**kw):
    return T5TextConditioner(**kw)


@register_conditioner("umt5")
@register_conditioner("audiodit_umt5")
@register_conditioner("longcat_umt5")
def _build_umt5(**kw):
    return UMT5TextConditioner(**kw)


@register_conditioner("clap")
def _build_clap(**kw):
    return CLAPTextConditioner(**kw)
