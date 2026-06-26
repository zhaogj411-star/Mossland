# Copyright (c) 2025.
"""Weightless deterministic codec for unit tests and pipeline smoke-tests."""

from __future__ import annotations

import math

import torch
from einops import rearrange

from .base import AudioCodec, LatentStats
from .registry import register_codec


def _dct2_basis(hop: int, comps: int) -> torch.Tensor:
    n = torch.arange(hop).float()
    k = torch.arange(comps).float()
    basis = torch.cos(math.pi / hop * (n[:, None] + 0.5) * k[None, :])
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return basis


class DummyCodec(AudioCodec):
    def __init__(
        self,
        latent_channels: int = 64,
        hop_length: int = 2048,
        sample_rate: int = 44100,
        audio_channels: int = 2,
        stats: LatentStats | None = None,
        device: str = "cpu",
    ):
        super().__init__(stats=stats, device=device)
        if latent_channels % audio_channels != 0:
            raise ValueError("latent_channels must be divisible by audio_channels")
        self.latent_channels = latent_channels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self._comps = latent_channels // audio_channels
        self._basis = _dct2_basis(hop_length, self._comps)

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        b, a, n = waveform.shape
        pad = (self.hop_length - n % self.hop_length) % self.hop_length
        if pad:
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        frames = rearrange(waveform, "b a (t hop) -> b a t hop", hop=self.hop_length)
        basis = self._basis.to(waveform.device, waveform.dtype)
        coeff = frames @ basis
        return rearrange(coeff, "b a t c -> b (a c) t")

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        coeff = rearrange(latent, "b (a c) t -> b a t c", a=self.audio_channels)
        basis = self._basis.to(latent.device, latent.dtype)
        frames = coeff @ basis.T
        return rearrange(frames, "b a t hop -> b a (t hop)")


@register_codec("dummy")
def _build_dummy(**kw) -> DummyCodec:
    return DummyCodec(**kw)
