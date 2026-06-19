import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .autoencoders import AudioAutoencoder


@dataclass
class SAMEOutput:
    audio: torch.Tensor
    latents: torch.Tensor
    length: int
    softnorm_loss: torch.Tensor


class SAMEAutoencoder(nn.Module):
    """Mossland wrapper around the SAME continuous audio autoencoder.

    The implementation is self-contained under `scripts/same`. Construction is
    fully driven by recursive Hydra targets; checkpoint loading belongs to
    `scripts.factory.load_model()`.
    """

    def __init__(
        self,
        autoencoder: AudioAutoencoder,
        sample_rate: int | None = None,
        audio_channels: int | None = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        freeze_bottleneck: bool = False,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.sample_rate = int(sample_rate or self.autoencoder.sample_rate)
        self.audio_channels = int(audio_channels or self.autoencoder.io_channels)
        self.downsampling_ratio = int(self.autoencoder.downsampling_ratio)
        self.latent_dim = int(self.autoencoder.latent_dim)
        if freeze_encoder:
            self.autoencoder.encoder.requires_grad_(False)
        if freeze_decoder:
            self.autoencoder.decoder.requires_grad_(False)
        if freeze_bottleneck and self.autoencoder.bottleneck is not None:
            self.autoencoder.bottleneck.requires_grad_(False)

    def disable_inference_noise(self):
        if self.autoencoder.bottleneck is not None and hasattr(
            self.autoencoder.bottleneck, "noise_regularize"
        ):
            self.autoencoder.bottleneck.noise_regularize = False
        for module in self.modules():
            if hasattr(module, "mask_noise"):
                module.mask_noise = 0.0
        return self

    @property
    def token_rate(self):
        return self.sample_rate / self.downsampling_ratio

    def preprocess(self, audio: torch.Tensor):
        if audio.ndim == 2:
            audio = audio[:, None, :]
        if audio.ndim != 3:
            raise ValueError(f"expected [B,C,T] audio, got {tuple(audio.shape)}")
        if audio.shape[1] != self.audio_channels:
            raise ValueError(
                f"expected {self.audio_channels} channels, got {audio.shape[1]}"
            )
        length = int(audio.shape[-1])
        pad = math.ceil(length / self.downsampling_ratio) * self.downsampling_ratio - length
        if pad > 0:
            audio = F.pad(audio, (0, pad))
        return audio, length

    def encode(self, audio: torch.Tensor, return_info: bool = False):
        audio, length = self.preprocess(audio)
        result = self.autoencoder.encode(audio, return_info=return_info)
        if return_info:
            latents, info = result
            info["length"] = length
            return latents, info
        return result

    def decode(self, latents: torch.Tensor, length: int | None = None):
        audio = self.autoencoder.decode(latents)
        if length is None:
            return audio
        if audio.shape[-1] < length:
            audio = F.pad(audio, (0, length - audio.shape[-1]))
        return audio[..., :length]

    def forward(self, audio: torch.Tensor):
        audio, length = self.preprocess(audio)
        latents, info = self.autoencoder.encode(audio, return_info=True)
        reconstructed = self.decode(latents, length=length)
        softnorm_loss = info.get("softnorm_loss")
        if softnorm_loss is None:
            softnorm_loss = reconstructed.new_zeros(())
        return SAMEOutput(
            audio=reconstructed,
            latents=latents,
            length=length,
            softnorm_loss=softnorm_loss,
        )
