# Copyright (c) 2025.
"""SAME (Stable Audio 3 music autoencoder) adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .base import AudioCodec, LatentStats
from .registry import register_codec

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SAME_VARIANTS = {
    "same-l": REPO_ROOT / "checkpoints" / "SAME-L",
    "same-s": REPO_ROOT / "checkpoints" / "SAME-S",
}


def _resolve_variant(variant: str) -> str:
    local_dir = LOCAL_SAME_VARIANTS.get(variant)
    if local_dir is not None and local_dir.is_dir():
        return str(local_dir)
    return variant


class SAMECodec(AudioCodec):
    sample_rate = 44100
    audio_channels = 2
    latent_channels = 256
    hop_length = 4096

    def __init__(
        self,
        variant: str = "same-l",
        stats: Optional[LatentStats] = None,
        device: str = "cuda",
        chunked: bool = True,
    ):
        super().__init__(stats=stats, device=device)
        self.variant = _resolve_variant(variant)
        self.chunked = chunked
        self._ae = None

    def _lazy(self):
        if self._ae is None:
            from scripts.mossland_creator.codecs.stable_audio_3 import AutoencoderModel
            from scripts.mossland_creator.codecs.stable_audio_3.loading_utils import (
                load_autoencoder,
            )

            variant_path = Path(self.variant)
            if variant_path.is_dir():
                config_path = variant_path / "model_config.json"
                ckpt_path = variant_path / "model.safetensors"
                if not config_path.is_file() or not ckpt_path.is_file():
                    raise FileNotFoundError(
                        f"local SAME checkpoint is incomplete under {variant_path}; "
                        "expected model_config.json and model.safetensors"
                    )
                with config_path.open("r", encoding="utf-8") as handle:
                    sample_rate = int(json.load(handle)["sample_rate"])
                autoencoder = load_autoencoder(
                    str(config_path),
                    str(ckpt_path),
                    device=self.device,
                )
                autoencoder.eval().requires_grad_(False)
                self._ae = AutoencoderModel(autoencoder, sample_rate, self.device)
            else:
                self._ae = AutoencoderModel.from_pretrained(
                    self.variant,
                    device=self.device,
                )
            self.sample_rate = self._ae.sample_rate
            self.device = self._ae.device
            ds = getattr(self._ae.autoencoder, "downsampling_ratio", self.hop_length)
            self.hop_length = ds
        return self._ae

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        ae = self._lazy()
        wav = waveform.to(self.device)
        outs = []
        for i in range(wav.shape[0]):
            lat = ae.encode(wav[i], self.sample_rate, chunked=self.chunked)
            outs.append(lat if lat.dim() == 3 else lat.unsqueeze(0))
        return torch.cat(outs, dim=0).float()

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        ae = self._lazy()
        z = latent.to(self.device)
        outs = []
        for i in range(z.shape[0]):
            wav = ae.decode(z[i].unsqueeze(0), chunked=self.chunked)
            outs.append(wav if wav.dim() == 3 else wav.unsqueeze(0))
        return torch.cat(outs, dim=0).float()

    def to(self, device: str):
        self.device = device
        return self


@register_codec("same")
@register_codec("same-l")
def _build_same(**kw) -> SAMECodec:
    kw.setdefault("variant", "same-l")
    return SAMECodec(**kw)
