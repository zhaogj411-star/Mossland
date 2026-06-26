# Copyright (c) 2025.
"""CoDiCodec continuous-feature adapter (plug-point)."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import AudioCodec, LatentStats
from .registry import register_codec


class CoDiCodecCodec(AudioCodec):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        latent_channels: int = 64,
        hop_length: int = 1024,
        sample_rate: int = 44100,
        audio_channels: int = 2,
        backbone_builder: Optional[Callable[[str], torch.nn.Module]] = None,
        stats: Optional[LatentStats] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(stats=stats, device=device)
        self.checkpoint = checkpoint
        self.latent_channels = latent_channels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.dtype = dtype
        self._backbone_builder = backbone_builder
        self._model = None

    def _build_backbone(self) -> torch.nn.Module:
        if self._backbone_builder is not None:
            return self._backbone_builder(self.checkpoint)
        try:
            from codicodec import CoDiCodec  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "CoDiCodec backbone not available. Provide `backbone_builder=...` "
                f"or install your CoDiCodec package. Original error: {e!r}"
            )
        return CoDiCodec.from_pretrained(self.checkpoint)

    def _lazy(self):
        if self._model is None:
            self._model = self._build_backbone().to(self.device).eval()
        return self._model

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self._lazy()
        waveform = waveform.to(self.device, self.dtype)
        z = model.encode(waveform)
        z = z.continuous if hasattr(z, "continuous") else z
        return z.float()

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        model = self._lazy()
        latent = latent.to(self.device, self.dtype)
        wav = model.decode(latent)
        wav = wav.audio if hasattr(wav, "audio") else wav
        return wav.float()

    def to(self, device: str):
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


@register_codec("codicodec")
def _build_codicodec(**kw) -> CoDiCodecCodec:
    return CoDiCodecCodec(**kw)
