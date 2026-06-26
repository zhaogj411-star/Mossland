# Copyright (c) 2025.
"""Abstract audio codec / VAE interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatentStats:
    mean: float | torch.Tensor = 0.0
    std: float | torch.Tensor = 1.0

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._b(self.mean, z)) / self._b(self.std, z)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._b(self.std, z) + self._b(self.mean, z)

    @staticmethod
    def _b(v, z):
        if isinstance(v, torch.Tensor) and v.dim() == 1:
            shape = [1] * z.dim()
            shape[-2] = v.shape[0]
            return v.to(z.device, z.dtype).view(shape)
        return v


class AudioCodec(abc.ABC):
    sample_rate: int = 44100
    audio_channels: int = 2
    latent_channels: int = 64
    hop_length: int = 2048
    is_continuous: bool = True

    def __init__(self, stats: Optional[LatentStats] = None, device: str = "cpu"):
        self.stats = stats or LatentStats()
        self.device = device

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.hop_length

    def num_samples(self, num_frames: int) -> int:
        return num_frames * self.hop_length

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        batched = waveform.dim() == 3
        wav = waveform if batched else waveform.unsqueeze(0)
        z = self._encode(wav.to(self.device))
        if normalize:
            z = self.stats.normalize(z)
        return z if batched else z.squeeze(0)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        batched = latent.dim() == 3
        z = latent if batched else latent.unsqueeze(0)
        if denormalize:
            z = self.stats.denormalize(z)
        wav = self._decode(z.to(self.device))
        return wav if batched else wav.squeeze(0)

    @abc.abstractmethod
    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, A, N] -> [B, C, T] (un-normalized)."""

    @abc.abstractmethod
    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, A, N] (latent already denormalized)."""

    def to(self, device: str) -> "AudioCodec":
        self.device = device
        return self
