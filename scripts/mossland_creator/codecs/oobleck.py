# Copyright (c) 2025.
"""Stable-Audio VAE (Oobleck) adapter."""

from __future__ import annotations

from typing import Optional

import torch

from .base import AudioCodec, LatentStats
from .registry import register_codec


class OobleckCodec(AudioCodec):
    sample_rate = 44100
    audio_channels = 2
    latent_channels = 64
    hop_length = 2048

    def __init__(
        self,
        pretrained: str = "stabilityai/stable-audio-open-1.0",
        subfolder: Optional[str] = "vae",
        sample_posterior: bool = False,
        stats: Optional[LatentStats] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(stats=stats, device=device)
        self.pretrained = pretrained
        self.subfolder = subfolder
        self.sample_posterior = sample_posterior
        self.dtype = dtype
        self._model = None

    def _lazy(self):
        if self._model is None:
            from diffusers import AutoencoderOobleck

            kw = {"torch_dtype": self.dtype}
            if self.subfolder:
                kw["subfolder"] = self.subfolder
            self._model = AutoencoderOobleck.from_pretrained(self.pretrained, **kw).to(
                self.device
            ).eval()
            self.sample_rate = getattr(
                self._model.config, "sampling_rate", self.sample_rate
            )
            self.audio_channels = getattr(
                self._model.config, "audio_channels", self.audio_channels
            )
            self.latent_channels = (
                getattr(self._model.config, "decoder_channels", self.latent_channels)
                if hasattr(self._model.config, "decoder_channels")
                else self.latent_channels
            )
        return self._model

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self._lazy()
        waveform = waveform.to(self.device, self.dtype)
        posterior = model.encode(waveform).latent_dist
        return (
            posterior.sample() if self.sample_posterior else posterior.mode()
        ).float()

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        model = self._lazy()
        latent = latent.to(self.device, self.dtype)
        return model.decode(latent).sample.float()

    def to(self, device: str):
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


@register_codec("oobleck")
@register_codec("stable_audio_vae")
def _build_oobleck(**kw) -> OobleckCodec:
    return OobleckCodec(**kw)
