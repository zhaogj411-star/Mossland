import math
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

from scripts.MOSS_Audio_Tokenizer.modeling_moss_audio_tokenizer import (
    MossAudioTokenizerPatchedPretransform,
    MossAudioTokenizerTransformer,
)
from scripts.music2latent.audio import AudioProcessor


def zero_init(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    return x.view(*x.shape, *((1,) * (target_ndim - x.ndim)))


def split_layers(total_layers: int, num_stages: int) -> list[int]:
    total_layers = int(total_layers)
    base = total_layers // num_stages
    extra = total_layers % num_stages
    return [base + (1 if idx < extra else 0) for idx in range(num_stages)]


class SpectralResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: Optional[int] = None, zero_output: bool = False):
        super().__init__()
        hidden_channels = int(hidden_channels or channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        if zero_output:
            zero_init(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size: int, max_positions: int = 10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_size // 2
        emb = math.log(self.max_positions) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb
        )
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.embedding_size:
            emb = F.pad(emb, (0, self.embedding_size - emb.shape[-1]))
        return emb


class MossAudioTokenizerFlow(nn.Module):
    """Music2Latent-compatible consistency model with MOSS local causal transformers.

    The training API intentionally matches ``scripts.music2latent.models.Music2Latent``:
    ``forward(latents, x, sigma)`` receives clean STFT representation as ``latents``
    and noisy STFT representation as ``x`` from the Music2Latent consistency trainer.
    Internally the 2D real/imag spectrogram is flattened into a 1D time-token stream.
    """

    def __init__(
        self,
        audio_processor: Optional[AudioProcessor] = None,
        sample_rate: int = 44100,
        data_channels: int = 2,
        number_channels: Optional[int] = None,
        hop: int = 512,
        fac: int = 4,
        alpha_rescale: float = 0.65,
        beta_rescale: float = 0.34,
        freq_bins: Optional[int] = None,
        spec_frames_per_token: int = 1,
        hidden_dim: int = 512,
        token_dim: Optional[int] = None,
        continuous_dim: int = 64,
        latent_dim: Optional[int] = None,
        encoder_layers: int = 8,
        encoder_heads: int = 8,
        decoder_layers: int = 8,
        decoder_heads: int = 8,
        dim_feedforward: Optional[int] = None,
        context: Optional[int] = None,
        context_duration: Optional[float] = None,
        token_rate: Optional[float] = None,
        causal: bool = True,
        positional_embedding: str = "sin",
        attention_implementation: str = "sdpa",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        init_as_zero: bool = True,
        max_waveform_length_encode: int = 44100 * 60,
        max_batch_size_encode: int = 1,
        max_waveform_length_decode: int = 44100 * 60,
        max_batch_size_decode: int = 1,
        patch_sizes: Optional[list[int]] = None,
        stage_dims: Optional[list[int]] = None,
        encoder_layers_per_stage: Optional[list[int]] = None,
        decoder_layers_per_stage: Optional[list[int]] = None,
        spectral_frontend_channels: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop = hop
        self.fac = fac
        self.data_channels = data_channels
        self.number_channels = number_channels
        self.freq_bins = int(freq_bins or hop * 2)
        self.patch_sizes = [int(x) for x in (patch_sizes or [2, 2, 2])]
        patch_product = int(np.prod(self.patch_sizes)) if self.patch_sizes else 1
        if int(spec_frames_per_token) != patch_product:
            raise ValueError(
                "spec_frames_per_token must equal the product of patch_sizes "
                f"for staged PatchedPretransform flow, got {spec_frames_per_token} vs {patch_product}"
            )
        self.spec_frames_per_token = patch_product
        self.hidden_dim = int(token_dim or hidden_dim)
        self.latent_dim = int(latent_dim or continuous_dim)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.max_waveform_length_encode = max_waveform_length_encode
        self.max_batch_size_encode = max_batch_size_encode
        self.max_waveform_length_decode = max_waveform_length_decode
        self.max_batch_size_decode = max_batch_size_decode

        self.freq_downsample_list = []
        self.audio_processor = audio_processor or AudioProcessor(
            alpha_rescale=alpha_rescale,
            beta_rescale=beta_rescale,
            hop_size=hop,
            fac=fac,
        )

        if context is None and context_duration is not None:
            if token_rate is not None:
                context = max(1, int(round(context_duration * token_rate)))
            else:
                seconds_per_token = hop * self.spec_frames_per_token / sample_rate
                context = max(1, int(round(context_duration / seconds_per_token)))
        self.context = context
        self.causal = causal

        self.frame_dim = self.data_channels * self.freq_bins
        self.spectral_in = SpectralResidualBlock(
            self.data_channels,
            hidden_channels=spectral_frontend_channels,
            zero_output=False,
        )
        self.spectral_out = SpectralResidualBlock(
            self.data_channels,
            hidden_channels=spectral_frontend_channels,
            zero_output=True,
        )
        num_stages = len(self.patch_sizes) + 1
        self.stage_dims = [int(x) for x in (stage_dims or [self.hidden_dim] * num_stages)]
        if len(self.stage_dims) != num_stages:
            raise ValueError(
                f"stage_dims must have {num_stages} entries for patch_sizes={self.patch_sizes}, got {self.stage_dims}"
            )
        encoder_layers_per_stage = encoder_layers_per_stage or split_layers(encoder_layers, num_stages)
        decoder_layers_per_stage = decoder_layers_per_stage or split_layers(decoder_layers, num_stages)
        if len(encoder_layers_per_stage) != num_stages or len(decoder_layers_per_stage) != num_stages:
            raise ValueError("encoder_layers_per_stage and decoder_layers_per_stage must match stage count")

        def make_transformer(d_model: int, heads: int, layers: int) -> MossAudioTokenizerTransformer:
            return MossAudioTokenizerTransformer(
                d_model=d_model,
                num_heads=heads,
                num_layers=int(layers),
                dim_feedforward=int(dim_feedforward or d_model * 4),
                causal=causal,
                context=context,
                positional_embedding=positional_embedding,
                attention_implementation=attention_implementation,
            )

        self.encoder_in = nn.Linear(self.frame_dim, self.stage_dims[0], bias=False)
        self.encoder_stages = nn.ModuleList(
            [
                make_transformer(self.stage_dims[idx], encoder_heads, encoder_layers_per_stage[idx])
                for idx in range(num_stages)
            ]
        )
        self.encoder_patch_downs = nn.ModuleList(
            [
                MossAudioTokenizerPatchedPretransform(
                    patch_size=patch_size,
                    is_downsample=True,
                    module_type="PatchedPretransform",
                )
                for patch_size in self.patch_sizes
            ]
        )
        self.encoder_down_projs = nn.ModuleList(
            [
                nn.Linear(self.stage_dims[idx] * self.patch_sizes[idx], self.stage_dims[idx + 1], bias=False)
                for idx in range(len(self.patch_sizes))
            ]
        )
        self.latent_out = nn.Linear(self.stage_dims[-1], self.latent_dim, bias=False)

        self.latent_in = nn.Linear(self.latent_dim, self.stage_dims[-1], bias=False)
        self.latent_conditioner = make_transformer(
            self.stage_dims[-1],
            decoder_heads,
            decoder_layers_per_stage[-1],
        )
        self.conditioner_patch_ups = nn.ModuleList(
            [
                MossAudioTokenizerPatchedPretransform(
                    patch_size=patch_size,
                    is_downsample=False,
                    module_type="PatchedPretransform",
                )
                for patch_size in reversed(self.patch_sizes)
            ]
        )
        self.conditioner_up_projs = nn.ModuleList(
            [
                nn.Linear(
                    self.stage_dims[idx + 1],
                    self.stage_dims[idx] * self.patch_sizes[idx],
                    bias=False,
                )
                for idx in reversed(range(len(self.patch_sizes)))
            ]
        )
        self.conditioner_up_stages = nn.ModuleList(
            [
                make_transformer(
                    self.stage_dims[idx],
                    decoder_heads,
                    decoder_layers_per_stage[idx],
                )
                for idx in reversed(range(len(self.patch_sizes)))
            ]
        )

        self.sigma_embedding = PositionalEmbedding(self.stage_dims[-1])
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.stage_dims[-1], self.stage_dims[-1]),
            nn.SiLU(),
            nn.Linear(self.stage_dims[-1], self.stage_dims[-1]),
        )
        self.stage_sigma_projs = nn.ModuleList(
            [nn.Linear(self.stage_dims[-1], self.stage_dims[idx], bias=False) for idx in range(num_stages)]
        )
        self.stage_condition_projs = nn.ModuleList(
            [nn.Linear(self.stage_dims[idx] * 2, self.stage_dims[idx], bias=False) for idx in range(num_stages)]
        )
        self.denoiser_in = nn.Linear(self.frame_dim, self.stage_dims[0], bias=False)
        self.denoiser_down_stages = nn.ModuleList(
            [
                make_transformer(self.stage_dims[idx], decoder_heads, decoder_layers_per_stage[idx])
                for idx in range(num_stages)
            ]
        )
        self.denoiser_patch_downs = nn.ModuleList(
            [
                MossAudioTokenizerPatchedPretransform(
                    patch_size=patch_size,
                    is_downsample=True,
                    module_type="PatchedPretransform",
                )
                for patch_size in self.patch_sizes
            ]
        )
        self.denoiser_down_projs = nn.ModuleList(
            [
                nn.Linear(self.stage_dims[idx] * self.patch_sizes[idx], self.stage_dims[idx + 1], bias=False)
                for idx in range(len(self.patch_sizes))
            ]
        )
        self.denoiser_mid = make_transformer(
            self.stage_dims[-1],
            decoder_heads,
            max(1, decoder_layers_per_stage[-1]),
        )
        self.denoiser_patch_ups = nn.ModuleList(
            [
                MossAudioTokenizerPatchedPretransform(
                    patch_size=patch_size,
                    is_downsample=False,
                    module_type="PatchedPretransform",
                )
                for patch_size in reversed(self.patch_sizes)
            ]
        )
        self.denoiser_up_projs = nn.ModuleList(
            [
                nn.Linear(
                    self.stage_dims[idx + 1],
                    self.stage_dims[idx] * self.patch_sizes[idx],
                    bias=False,
                )
                for idx in reversed(range(len(self.patch_sizes)))
            ]
        )
        self.denoiser_up_stages = nn.ModuleList(
            [
                make_transformer(
                    self.stage_dims[idx],
                    decoder_heads,
                    decoder_layers_per_stage[idx],
                )
                for idx in reversed(range(len(self.patch_sizes)))
            ]
        )
        denoiser_out = nn.Linear(self.stage_dims[0], self.frame_dim, bias=False)
        self.denoiser_out = zero_init(denoiser_out) if init_as_zero else denoiser_out

    def to_representation_encoder(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert mono/stereo waveform to flattened real/imag STFT channels."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        added_batch = False
        if waveform.ndim == 2 and self.number_channels and waveform.shape[0] == self.number_channels:
            waveform = waveform.unsqueeze(0)
            added_batch = True
        if waveform.ndim == 2:
            return self.audio_processor.to_representation_encoder(waveform)
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform [B,L] or [B,C,L], got {tuple(waveform.shape)}")

        batch, channels, samples = waveform.shape
        if self.number_channels is not None and channels != self.number_channels:
            raise ValueError(f"Expected {self.number_channels} waveform channels, got {channels}")
        representation = self.audio_processor.to_representation_encoder(
            waveform.reshape(batch * channels, samples)
        )
        representation = representation.reshape(
            batch, channels, 2, representation.shape[-2], representation.shape[-1]
        )
        representation = representation.reshape(
            batch, channels * 2, representation.shape[-2], representation.shape[-1]
        )
        return representation[0] if added_batch else representation

    def to_waveform(self, representation: torch.Tensor) -> torch.Tensor:
        """Invert flattened mono/stereo real/imag STFT channels to waveform."""
        if representation.ndim == 3:
            representation = representation.unsqueeze(0)
        if representation.ndim != 4:
            raise ValueError(
                f"Expected representation [B,C,F,T], got {tuple(representation.shape)}"
            )
        batch, rep_channels, freq_bins, frames = representation.shape
        if rep_channels % 2 != 0:
            raise ValueError(f"Representation channels must be even, got {rep_channels}")
        audio_channels = rep_channels // 2
        representation = representation.reshape(batch * audio_channels, 2, freq_bins, frames)
        waveform = self.audio_processor.to_waveform(representation, self.hop)
        return waveform.reshape(batch, audio_channels, waveform.shape[-1])

    def _input_lengths(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (x.shape[0],),
            x.shape[1],
            device=x.device,
            dtype=torch.long,
        )

    def _run_transformer(
        self,
        transformer: MossAudioTokenizerTransformer,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return transformer(x, input_lengths=input_lengths if input_lengths is not None else self._input_lengths(x))

    def _patch_down(
        self,
        patch: MossAudioTokenizerPatchedPretransform,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, input_lengths = patch(x.transpose(1, 2), input_lengths)
        return x.transpose(1, 2), input_lengths

    def _patch_up(
        self,
        patch: MossAudioTokenizerPatchedPretransform,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, input_lengths = patch(x.transpose(1, 2), input_lengths)
        return x.transpose(1, 2), input_lengths

    def _pad_representation(
        self, representation: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        if representation.ndim != 4:
            raise ValueError(
                f"Expected representation [B, C, F, T], got {tuple(representation.shape)}"
            )
        if representation.shape[1] != self.data_channels:
            raise ValueError(
                f"Expected {self.data_channels} representation channels, got {representation.shape[1]}"
            )
        if representation.shape[2] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {representation.shape[2]}"
            )
        original_frames = representation.shape[-1]
        remainder = original_frames % self.spec_frames_per_token
        if remainder:
            pad_frames = self.spec_frames_per_token - remainder
            representation = F.pad(representation, (0, pad_frames))
        return representation, original_frames, representation.shape[-1]

    def representation_to_tokens(self, representation: torch.Tensor) -> torch.Tensor:
        representation, _, padded_frames = self._pad_representation(representation)
        representation = self.spectral_in(representation)
        batch = representation.shape[0]
        return representation.permute(0, 3, 1, 2).reshape(batch, padded_frames, -1)

    def tokens_to_representation(
        self, tokens: torch.Tensor, original_frames: Optional[int] = None
    ) -> torch.Tensor:
        batch, token_count, _ = tokens.shape
        representation = (
            tokens.reshape(
                batch,
                token_count,
                self.data_channels,
                self.freq_bins,
            )
            .permute(0, 2, 3, 1)
        )
        representation = self.spectral_out(representation)
        if original_frames is not None:
            representation = representation[..., :original_frames]
        return representation

    def encode_representation(self, representation: torch.Tensor) -> torch.Tensor:
        tokens = self.representation_to_tokens(representation)
        lengths = self._input_lengths(tokens)
        hidden = self.encoder_in(tokens)
        for idx, stage in enumerate(self.encoder_stages):
            hidden = self._run_transformer(stage, hidden, lengths)
            if idx < len(self.encoder_patch_downs):
                hidden, lengths = self._patch_down(self.encoder_patch_downs[idx], hidden, lengths)
                hidden = self.encoder_down_projs[idx](hidden)
        return torch.tanh(self.latent_out(hidden))

    def condition_latent_tokens(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.latent_in(latent_tokens)
        return self._run_transformer(self.latent_conditioner, hidden)

    def condition_latent_pyramid(self, latent_tokens: torch.Tensor) -> list[torch.Tensor]:
        lengths = self._input_lengths(latent_tokens)
        hidden = self.condition_latent_tokens(latent_tokens)
        pyramid_reversed = [hidden]
        for proj, patch, stage in zip(
            self.conditioner_up_projs,
            self.conditioner_patch_ups,
            self.conditioner_up_stages,
        ):
            hidden = proj(hidden)
            hidden, lengths = self._patch_up(patch, hidden, lengths)
            hidden = self._run_transformer(stage, hidden, lengths)
            pyramid_reversed.append(hidden)
        return list(reversed(pyramid_reversed))

    def add_stage_condition(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        sigma_base: torch.Tensor,
        stage_idx: int,
    ) -> torch.Tensor:
        if condition.shape[1] != hidden.shape[1]:
            raise ValueError(
                f"Stage {stage_idx} condition length {condition.shape[1]} does not match hidden length {hidden.shape[1]}"
            )
        sigma_stage = self.stage_sigma_projs[stage_idx](sigma_base)
        sigma_stage = sigma_stage[:, None, :].expand(-1, hidden.shape[1], -1)
        return hidden + self.stage_condition_projs[stage_idx](
            torch.cat([condition, sigma_stage], dim=-1)
        )

    def denoise_tokens(
        self,
        noisy_tokens: torch.Tensor,
        latent_tokens: torch.Tensor,
        sigma_base: torch.Tensor,
    ) -> torch.Tensor:
        lengths = self._input_lengths(noisy_tokens)
        condition_pyramid = self.condition_latent_pyramid(latent_tokens)
        hidden = self.denoiser_in(noisy_tokens)
        for idx, stage in enumerate(self.denoiser_down_stages):
            hidden = self._run_transformer(stage, hidden, lengths)
            hidden = self.add_stage_condition(hidden, condition_pyramid[idx], sigma_base, idx)
            if idx < len(self.denoiser_patch_downs):
                hidden, lengths = self._patch_down(self.denoiser_patch_downs[idx], hidden, lengths)
                hidden = self.denoiser_down_projs[idx](hidden)

        hidden = self._run_transformer(self.denoiser_mid, hidden, lengths)

        for stage_idx, proj, patch, stage in zip(
            reversed(range(len(self.patch_sizes))),
            self.denoiser_up_projs,
            self.denoiser_patch_ups,
            self.denoiser_up_stages,
        ):
            hidden = proj(hidden)
            hidden, lengths = self._patch_up(patch, hidden, lengths)
            hidden = self._run_transformer(stage, hidden, lengths)
            hidden = self.add_stage_condition(hidden, condition_pyramid[stage_idx], sigma_base, stage_idx)
        return self.denoiser_out(hidden)

    def _get_c(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = (self.sigma_data * (sigma - self.sigma_min)) / (
            self.sigma_data**2 + sigma**2
        ).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        return c_skip, c_out, c_in

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        pyramid_latents=None,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        latents = latents.to(dtype)
        if sigma is None:
            sigma = torch.full((x.shape[0],), self.sigma_max, device=x.device, dtype=x.dtype)
        elif not torch.is_tensor(sigma):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
        else:
            sigma = sigma.to(device=x.device, dtype=x.dtype).flatten()
            if sigma.numel() == 1:
                sigma = sigma.expand(x.shape[0])

        noisy_tokens = self.representation_to_tokens(x)
        original_frames = x.shape[-1]
        if latents.ndim == 4:
            latent_tokens = self.encode_representation(latents)
        elif latents.ndim == 3:
            latent_tokens = latents
        else:
            raise ValueError(f"Unsupported latents shape: {tuple(latents.shape)}")

        c_skip, c_out, c_in = self._get_c(sigma)
        c_skip_full = append_dims(c_skip, noisy_tokens.ndim)
        c_out_full = append_dims(c_out, noisy_tokens.ndim)
        c_in_full = append_dims(c_in, noisy_tokens.ndim)

        sigma_log = torch.log(sigma.clamp_min(self.sigma_min)) / 4.0
        sigma_base = self.sigma_mlp(self.sigma_embedding(sigma_log.to(dtype)))

        predicted_tokens = self.denoise_tokens(c_in_full * noisy_tokens, latent_tokens, sigma_base)
        output_tokens = c_skip_full * noisy_tokens + c_out_full * predicted_tokens
        return self.tokens_to_representation(output_tokens, original_frames=original_frames)

    @torch.no_grad()
    def encode(self, path_or_audio, *args, rescale: float = 1.0, **kwargs) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        if isinstance(path_or_audio, str):
            audio, _ = sf.read(path_or_audio, dtype="float32", always_2d=True)
            audio = np.transpose(audio, (1, 0))
            audio = torch.from_numpy(audio)
        else:
            audio = path_or_audio
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio)
        audio = audio.to(device=device, dtype=next(self.parameters()).dtype)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if self.number_channels == 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        representation = self.to_representation_encoder(audio)
        if representation.ndim == 3:
            representation = representation.unsqueeze(0)
        latent = self.encode_representation(representation)
        return latent.squeeze(0) / rescale

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        denoising_steps: int = 1,
        *args,
        rescale: float = 1.0,
        target_length: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        latent = latent.to(device=device, dtype=next(self.parameters()).dtype) * rescale
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        representation = self._decode_to_representation(latent, denoising_steps, device)
        waveform = self.to_waveform(representation)
        if target_length is not None:
            waveform = waveform[..., :target_length]
        return waveform

    def _decode_to_representation(
        self,
        latents: torch.Tensor,
        diffusion_steps: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        frame_count = latents.shape[1] * self.spec_frames_per_token
        initial_noise = (
            torch.randn(
                latents.shape[0],
                self.data_channels,
                self.freq_bins,
                frame_count,
                device=device,
                dtype=dtype,
            )
            * self.sigma_max
        )
        return self._reverse_diffusion(initial_noise, int(diffusion_steps), latents)

    def _get_sigma(self, i: int, k: int) -> float:
        return (
            self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / (k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def _reverse_step(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        return x + ((sigma**2 - self.sigma_min**2) ** 0.5) * noise

    @torch.no_grad()
    def _denoise(
        self,
        noisy_samples: torch.Tensor,
        sigma: float,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_tensor = torch.full(
            (noisy_samples.shape[0],),
            float(sigma),
            dtype=noisy_samples.dtype,
            device=noisy_samples.device,
        )
        pred_samples = self(latents=latents, x=noisy_samples, sigma=sigma_tensor)
        pred_noises = torch.randn_like(pred_samples)
        return pred_noises, pred_samples

    @torch.no_grad()
    def _reverse_diffusion(
        self,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        diffusion_steps = max(1, int(diffusion_steps))
        next_noisy_samples = initial_noise
        pred_samples = initial_noise
        for step in range(diffusion_steps):
            sigma = self._get_sigma(diffusion_steps + 1 - step, diffusion_steps + 1)
            next_sigma = self._get_sigma(diffusion_steps - step, diffusion_steps + 1)
            pred_noises, pred_samples = self._denoise(next_noisy_samples, sigma, latents)
            next_noisy_samples = self._reverse_step(pred_samples, pred_noises, next_sigma)
        return pred_samples
