import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scripts.MOSS_Audio_Tokenizer.modeling_moss_audio_tokenizer import (
    MossAudioTokenizerTransformer,
)

from .layers import Decoder, Encoder, init_weights, norm_conv1d
from .quantize import ResidualVectorQuantize


@dataclass
class FlowCodecOutput:
    audio_discrete: torch.Tensor
    audio_continuous: torch.Tensor
    codes: torch.Tensor
    continuous: torch.Tensor
    quantized: torch.Tensor
    encoded: torch.Tensor
    vq_commitment_loss: torch.Tensor
    vq_codebook_loss: torch.Tensor
    length: int


class MossAudioFlowCodec(nn.Module):
    """Stereo codec with 25 Hz local-transformer bottleneck tokens.

        At 48 kHz, the default total stride is 1920, so a one-second stereo waveform
        becomes 25 temporal tokens. The discrete bottleneck uses 32 RVQ codebooks by
        default with a 1024-entry vocabulary per codebook; the continuous bottleneck
        exposes 32 channels at the same 25 Hz frame rate.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        audio_channels: int = 2,
        encoder_dim: int = 48,
        encoder_rates: list[int] | tuple[int, ...] = (5, 4, 4, 4, 3, 2),
        latent_dim: int | None = None,
        decoder_dim: int | None = None,
        decoder_rates: list[int] | tuple[int, ...] | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int | list[int] = 8,
        quantizer_dropout: float = 0.0,
        continuous_dim: int = 32,
        local_transformer_layers: int = 0,
        local_transformer_heads: int = 8,
        local_transformer_context: int | None = None,
        local_transformer_causal: bool = True,
        local_transformer_ff_mult: int = 4,
        local_transformer_positional_embedding: str = "rope",
        local_transformer_attention: str = "sdpa",
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.audio_channels = int(audio_channels)
        self.encoder_dim = int(encoder_dim)
        self.encoder_rates = tuple(int(rate) for rate in encoder_rates)
        self.downsample_ratio = math.prod(self.encoder_rates)
        self.token_rate = self.sample_rate / self.downsample_ratio
        if abs(self.token_rate - round(self.token_rate, 8)) > 1e-6:
            raise ValueError(
                f"sample_rate/downsample_ratio must be integral enough, got {self.token_rate}"
            )
        if decoder_rates is None:
            decoder_rates = tuple(reversed(self.encoder_rates))
        self.decoder_rates = tuple(int(rate) for rate in decoder_rates)
        if math.prod(self.decoder_rates) != self.downsample_ratio:
            raise ValueError("decoder_rates product must match encoder_rates product")

        if latent_dim is None:
            latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))
        if decoder_dim is None:
            decoder_dim = latent_dim
        self.latent_dim = int(latent_dim)
        self.decoder_dim = int(decoder_dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = codebook_dim
        self.continuous_dim = int(continuous_dim)
        self.local_transformer_layers = int(local_transformer_layers)
        self.local_transformer_heads = int(local_transformer_heads)
        self.local_transformer_context = (
            None if local_transformer_context is None else int(local_transformer_context)
        )
        self.local_transformer_causal = bool(local_transformer_causal)
        self.local_transformer_ff_mult = int(local_transformer_ff_mult)
        self.local_transformer_positional_embedding = local_transformer_positional_embedding
        self.local_transformer_attention = local_transformer_attention

        self.encoder = Encoder(
            in_channels=self.audio_channels,
            d_model=self.encoder_dim,
            strides=self.encoder_rates,
            d_latent=self.latent_dim,
        )
        if self.local_transformer_layers > 0:
            self.local_transformer = MossAudioTokenizerTransformer(
                d_model=self.latent_dim,
                num_heads=self.local_transformer_heads,
                num_layers=self.local_transformer_layers,
                dim_feedforward=self.latent_dim * self.local_transformer_ff_mult,
                causal=self.local_transformer_causal,
                context=self.local_transformer_context,
                positional_embedding=self.local_transformer_positional_embedding,
                attention_implementation=self.local_transformer_attention,
                layer_scale=0.01,
            )
        else:
            self.local_transformer = nn.Identity()
        self.discrete_pre = nn.Sequential(
            norm_conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
            nn.SiLU(),
            norm_conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
        )
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_codebooks=self.num_quantizers,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.continuous_encoder = nn.Sequential(
            norm_conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
            nn.SiLU(),
            norm_conv1d(self.latent_dim, self.continuous_dim, kernel_size=1),
        )
        self.continuous_decoder_in = norm_conv1d(
            self.continuous_dim, self.latent_dim, kernel_size=1
        )
        self.decoder = Decoder(
            input_channels=self.latent_dim,
            channels=self.decoder_dim,
            rates=self.decoder_rates,
            out_channels=self.audio_channels,
        )
        self.apply(init_weights)

    def _apply_local_transformer(self, features: torch.Tensor) -> torch.Tensor:
        if self.local_transformer_layers <= 0:
            return features
        tokens = features.transpose(1, 2).contiguous()
        tokens = self.local_transformer(tokens)
        return tokens.transpose(1, 2).contiguous()

    def preprocess(self, audio: torch.Tensor):
        if audio.ndim == 2:
            audio = audio[:, None, :]
        if audio.ndim != 3:
            raise ValueError(f"expected [B,C,T] audio, got shape {tuple(audio.shape)}")
        if audio.shape[1] != self.audio_channels:
            raise ValueError(
                f"expected {self.audio_channels} channels, got {audio.shape[1]}"
            )
        length = int(audio.shape[-1])
        right_pad = math.ceil(length / self.downsample_ratio) * self.downsample_ratio - length
        if right_pad > 0:
            audio = F.pad(audio, (0, right_pad))
        return audio, length

    def encode_features(self, audio: torch.Tensor):
        audio, length = self.preprocess(audio)
        features = self.encoder(audio)
        features = self._apply_local_transformer(features)
        return features, length

    def encode_discrete(self, audio: torch.Tensor, n_quantizers: int | None = None):
        features, length = self.encode_features(audio)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            self.discrete_pre(features),
            n_quantizers=n_quantizers,
        )
        return {
            "z": z,
            "codes": codes,
            "latents": latents,
            "features": features,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "length": length,
        }

    def encode_continuous(self, audio: torch.Tensor):
        features, length = self.encode_features(audio)
        continuous = self.continuous_encoder(features)
        return {"continuous": continuous, "features": features, "length": length}

    def decode_discrete_z(self, z: torch.Tensor, length: int | None = None):
        audio = self.decoder(z)
        return self._match_length(audio, length)

    def decode_discrete_codes(self, codes: torch.Tensor, length: int | None = None):
        z, _, _ = self.quantizer.from_codes(codes)
        return self.decode_discrete_z(z, length=length)

    def decode_continuous(self, continuous: torch.Tensor, length: int | None = None):
        z = self.continuous_decoder_in(continuous)
        audio = self.decoder(z)
        return self._match_length(audio, length)

    def _match_length(self, audio: torch.Tensor, length: int | None):
        if length is None:
            return audio
        if audio.shape[-1] < length:
            audio = F.pad(audio, (0, length - audio.shape[-1]))
        return audio[..., :length]

    def forward(self, audio: torch.Tensor, n_quantizers: int | None = None):
        features, length = self.encode_features(audio)
        quantized, codes, _, commitment_loss, codebook_loss = self.quantizer(
            self.discrete_pre(features),
            n_quantizers=n_quantizers,
        )
        continuous = self.continuous_encoder(features)
        audio_discrete = self.decode_discrete_z(quantized, length=length)
        audio_continuous = self.decode_continuous(continuous, length=length)
        return FlowCodecOutput(
            audio_discrete=audio_discrete,
            audio_continuous=audio_continuous,
            codes=codes,
            continuous=continuous,
            quantized=quantized,
            encoded=features,
            vq_commitment_loss=commitment_loss,
            vq_codebook_loss=codebook_loss,
            length=length,
        )
