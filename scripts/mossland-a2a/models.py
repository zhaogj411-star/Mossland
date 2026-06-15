from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

_codec_models = import_module("scripts.mossland-codec.models")
_codec_audio = import_module("scripts.mossland-codec.audio")
_codec_transformer = import_module("scripts.mossland-codec.transformer")
_codec_layers = import_module("scripts.mossland-codec.transformer_layers")
_codec_utils = import_module("scripts.mossland-codec.utils")

DownFrontend = _codec_models.DownFrontend
UpFrontend = _codec_models.UpFrontend
PositionalEmbedding = _codec_models.PositionalEmbedding
init = _codec_models.init
zero_init = _codec_models.zero_init
waveform_length_for_stft_frames = _codec_models.waveform_length_for_stft_frames
Transformer_Diffusion = _codec_transformer.Transformer_Diffusion
block_causal_attention_mask = _codec_layers.block_causal_attention_mask
TASK_NAMES = (
    "separate_vocals",
    "separate_drums",
    "separate_bass",
    "separate_other",
    "super_resolution",
    "mono_to_stereo",
)


class MosslandA2ATransformer(nn.Module):
    """Mossland audio-to-audio separation model without codec latent tokens."""

    def __init__(
        self,
        torch_compile_cache_dir: str | None = None,
        mixed_precision: bool = True,
        stereo: bool = True,
        default_denoising_steps_parallel: int = 5,
        hop: int = 1024,
        fac: int = 2,
        sample_rate: int = 44100,
        alpha_rescale: float = 0.65,
        beta_rescale: float = 0.34,
        dim: int = 512,
        head_dim: int = 128,
        mlp_mult: int = 4,
        pos_emb: str = "learned",
        num_layers: int = 12,
        num_layers_encoder: int | None = None,
        cond_channels: int = 512,
        num_latents: int | None = None,
        num_more_latents: int = 0,
        bottleneck_channels: int | None = None,
        use_fsq: bool = False,
        frontend_base_channels: int = 64,
        frontend_multipliers_list: list[int] | None = None,
        frontend_layers_list: list[int] | None = None,
        frontend_encoder_layers_list: list[int] | None = None,
        frontend_freq_downsample_list: list[int] | None = None,
        spec_length: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        max_batch_size_decode: int = 32,
        sigma_rescale: float = 0.8,
        task_names: list[str] | tuple[str, ...] | None = None,
        task_embedding_init_std: float = 0.02,
    ):
        super().__init__()
        if torch_compile_cache_dir is not None:
            import os

            os.environ["TORCHINDUCTOR_CACHE_DIR"] = torch_compile_cache_dir
        if frontend_multipliers_list is None:
            frontend_multipliers_list = [1, 2, 4, dim // frontend_base_channels]
        if frontend_layers_list is None:
            frontend_layers_list = [3, 3, 3, 1]
        if frontend_encoder_layers_list is None:
            frontend_encoder_layers_list = list(frontend_layers_list)
        if frontend_freq_downsample_list is None:
            frontend_freq_downsample_list = [0, 1, 0]
        if num_layers_encoder is None:
            num_layers_encoder = num_layers
        if task_names is None:
            task_names = TASK_NAMES
        task_names = tuple(task_names)
        if not task_names:
            raise ValueError("task_names must not be empty")

        self.torch_compile_cache_dir = torch_compile_cache_dir
        self.task_names = task_names
        self.mixed_precision = bool(mixed_precision)
        self.stereo = bool(stereo)
        self.default_denoising_steps_parallel = int(default_denoising_steps_parallel)
        self.hop = int(hop)
        self.fac = int(fac)
        self.sample_rate = int(sample_rate)
        self.alpha_rescale = float(alpha_rescale)
        self.beta_rescale = float(beta_rescale)
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.heads = self.dim // self.head_dim
        self.mlp_mult = int(mlp_mult)
        self.pos_emb = pos_emb
        self.num_layers = int(num_layers)
        self.num_layers_encoder = int(num_layers_encoder)
        self.cond_channels = int(cond_channels)
        self.num_more_latents = int(num_more_latents)
        self.use_fsq = bool(use_fsq)
        self.frontend_base_channels = int(frontend_base_channels)
        self.frontend_multipliers_list = list(frontend_multipliers_list)
        self.frontend_layers_list = list(frontend_layers_list)
        self.frontend_encoder_layers_list = list(frontend_encoder_layers_list)
        self.frontend_freq_downsample_list = list(frontend_freq_downsample_list)
        self.spec_length = int(spec_length)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.max_batch_size_decode = int(max_batch_size_decode)
        self.sigma_rescale = float(sigma_rescale)
        self.task_embedding_init_std = float(task_embedding_init_std)
        self.stft_channels = 4 if self.stereo else 2
        self.decoder_input_channels = self.stft_channels

        self.downsample_ratio = (
            (4**self.frontend_freq_downsample_list.count(0))
            * (4**self.frontend_freq_downsample_list.count(1))
            * (2**self.frontend_freq_downsample_list.count(2))
            * (2**self.frontend_freq_downsample_list.count(3))
        )
        self.data_length = (
            self.hop * (self.fac // 2) * self.spec_length
        ) // self.downsample_ratio
        self.freq_dim = (self.hop * (self.fac // 2)) // (
            4**self.frontend_freq_downsample_list.count(1)
        )
        self.freq_dim = self.freq_dim // (
            2**self.frontend_freq_downsample_list.count(0)
        )
        self.freq_dim = self.freq_dim // (
            2**self.frontend_freq_downsample_list.count(2)
        )
        self.time_dim = self.spec_length // (
            2**self.frontend_freq_downsample_list.count(0)
        )
        self.time_dim = self.time_dim // (
            2**self.frontend_freq_downsample_list.count(3)
        )
        if num_latents is None:
            num_latents = self.data_length
        self.num_latents = int(num_latents)
        if self.num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if self.num_latents != self.data_length:
            raise ValueError("A2A encoder is non-compressive: num_latents must equal data_length")
        if bottleneck_channels is None:
            bottleneck_channels = self.dim
        self.bottleneck_channels = int(bottleneck_channels)
        if self.bottleneck_channels != self.dim:
            raise ValueError("A2A encoder is non-compressive: bottleneck_channels must equal dim")
        if self.num_more_latents != 0:
            raise ValueError("A2A encoder does not use extra latent tokens")
        if self.use_fsq:
            raise ValueError("A2A encoder does not use FSQ quantization")

        self.emb = PositionalEmbedding(embedding_size=self.cond_channels)
        self.emb_proj = nn.Sequential(
            init(nn.Linear(self.cond_channels, self.cond_channels)),
            nn.SiLU(),
            init(nn.Linear(self.cond_channels, self.cond_channels)),
            nn.SiLU(),
            init(nn.Linear(self.cond_channels, self.cond_channels)),
            nn.SiLU(),
        )
        self.task_to_idx = {name: idx for idx, name in enumerate(self.task_names)}
        self.task_embedding = nn.Embedding(len(self.task_names), self.cond_channels)
        nn.init.normal_(
            self.task_embedding.weight,
            mean=0.0,
            std=self.task_embedding_init_std,
        )
        self.gain_decoder = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(self.cond_channels, self.hop * 2 * (self.fac // 2))),
        )

        self.frontend_encoder_down = DownFrontend(
            self.frontend_encoder_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
        ).to(memory_format=torch.channels_last)
        self.encoder = Transformer_Diffusion(
            self.dim,
            self.dim,
            training_length=self.data_length,
            cond_dim=self.cond_channels,
            dim=self.dim,
            num_layers=self.num_layers_encoder,
            heads=self.heads,
            mlp_mult=self.mlp_mult,
            pos_emb=self.pos_emb,
        )

        self.lat2patch = nn.Identity()
        self.frontend_pre_decoder_up = UpFrontend(
            self.frontend_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
            self.dim,
            self.hop,
            self.fac,
        ).to(memory_format=torch.channels_last)
        self.frontend_decoder_down = DownFrontend(
            self.frontend_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.decoder_input_channels,
            cond_dim=self.cond_channels,
        ).to(memory_format=torch.channels_last)
        self.decoder = Transformer_Diffusion(
            self.dim,
            self.dim,
            training_length=(self.data_length + self.num_latents) * 2,
            cond_dim=self.cond_channels,
            dim=self.dim,
            num_layers=self.num_layers,
            heads=self.heads,
            mlp_mult=self.mlp_mult,
            pos_emb=self.pos_emb,
        )
        self.frontend_decoder_up = UpFrontend(
            self.frontend_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
            self.dim,
            self.hop,
            self.fac,
            cond_dim=self.cond_channels,
        ).to(memory_format=torch.channels_last)

    def get_attn_mask(self, x):
        if x.shape[-1] == (2 * self.spec_length):
            return block_causal_attention_mask(self.data_length + self.num_latents)
        raise ValueError(
            f"Invalid data length. Must be {(2 * self.spec_length)} while it is {x.shape[-1]}."
        )

    def to_representation_encoder(self, x):
        return _codec_audio.to_representation_encoder(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def to_representation(self, x):
        return _codec_audio.to_representation(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def to_waveform(self, x):
        return _codec_audio.to_waveform(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def prepare_audio_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(next(self.parameters()).dtype)
        if not self.stereo:
            batch = batch.mean(dim=-2, keepdim=True)
        target_length = waveform_length_for_stft_frames(
            2 * self.spec_length,
            hop=self.hop,
            fac=self.fac,
        )
        if batch.shape[-1] < target_length:
            return F.pad(batch, (0, target_length - batch.shape[-1]))
        return batch[..., :target_length]

    def encode_source(self, x, log_magnitude=False):
        """Encode source spectrogram without latent-count or channel bottlenecks."""
        if x.shape[-1] % self.spec_length != 0:
            raise ValueError(
                f"Input shape {x.shape[-1]} is not divisible by spec_length={self.spec_length}."
            )
        factor = None
        if x.shape[-1] > self.spec_length:
            x_chunks = torch.split(x, self.spec_length, dim=-1)
            factor = len(x_chunks)
            x = torch.cat(x_chunks, dim=0)

        tokens = self.frontend_encoder_down(
            x,
            gain=None,
            log_magnitude=log_magnitude,
        )[0]
        if tokens.shape[-2] != self.num_latents:
            raise ValueError(
                f"source encoder produced {tokens.shape[-2]} tokens, expected {self.num_latents}"
            )
        encoder_cond = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            self.cond_channels,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens = self.encoder(
            tokens,
            encoder_cond,
            latent=None,
            skip_input_layer=True,
            skip_output_layer=True,
            print_magnitudes=log_magnitude,
        )
        features = self.frontend_pre_decoder_up(
            tokens,
            skip_output_layer=True,
            log_magnitude=log_magnitude,
        )[1]
        if factor is not None:
            tokens = torch.cat(torch.chunk(tokens, factor, dim=0), dim=-2)
            features = [torch.cat(torch.chunk(el, factor, dim=0), dim=-1) for el in features]
        return tokens, features

    @staticmethod
    def _lookup_condition_index(lookup, value, strict: bool):
        key = str(value)
        if key in lookup:
            return lookup[key]
        if strict:
            raise KeyError(key)
        return 0

    def _coerce_condition_indices(self, names, lookup, values, indices, batch_size, device):
        if indices is not None:
            if torch.is_tensor(indices):
                idx = indices.to(device=device, dtype=torch.long).reshape(-1)
            else:
                idx = torch.as_tensor(indices, device=device, dtype=torch.long).reshape(-1)
        else:
            if values is None:
                values = names[0]
            if isinstance(values, str):
                idx = torch.full(
                    (batch_size,),
                    self._lookup_condition_index(lookup, values, strict=True),
                    device=device,
                    dtype=torch.long,
                )
            else:
                value_list = list(values) if isinstance(values, (list, tuple)) else [values]
                idx = torch.tensor(
                    [
                        self._lookup_condition_index(lookup, value, strict=False)
                        for value in value_list
                    ],
                    device=device,
                    dtype=torch.long,
                )
        if idx.numel() == 1:
            return idx.expand(batch_size)
        if idx.numel() == batch_size:
            return idx
        if idx.numel() * 2 == batch_size:
            return torch.cat((idx, idx), dim=0)
        raise ValueError(
            f"condition batch size mismatch: got {idx.numel()}, expected 1, "
            f"{batch_size // 2}, or {batch_size}"
        )

    def _condition_embedding(self, sigma_embedding, task_id="reconstruct", task_idx=None):
        batch_size = sigma_embedding.shape[0]
        task_idx = self._coerce_condition_indices(
            self.task_names,
            self.task_to_idx,
            task_id,
            task_idx,
            batch_size,
            sigma_embedding.device,
        )
        cond = sigma_embedding + self.task_embedding(task_idx).to(sigma_embedding.dtype)
        return self.emb_proj(cond)

    def get_sigma_continuous(self, i):
        return _codec_utils.get_sigma_continuous(
            i,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )

    def decoder_forward(
        self,
        x,
        src_tokens,
        features=None,
        sigma_left=None,
        sigma_right=None,
        output="both",
        task_id="reconstruct",
        task_idx=None,
        log_magnitude=False,
    ):
        if sigma_left is None:
            sigma_left = self.sigma_min if output != "left" else self.sigma_max
        if sigma_right is None:
            sigma_right = self.sigma_max

        if features is None:
            raise ValueError("A2A decoder_forward requires source encoder features")

        sigma_left = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) * sigma_left
        sigma_right = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) * sigma_right
        sigma = torch.cat([sigma_left, sigma_right], dim=0)
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self._condition_embedding(
            emb_sigma_log,
            task_id=task_id,
            task_idx=task_idx,
        )
        gain = self.gain_decoder(time_emb).unsqueeze(-2).unsqueeze(-1) + 1.0
        gain_inp, gain_out = torch.chunk(gain, 2, dim=-2)

        c_skip, c_out, c_in = _codec_utils.get_c(
            sigma,
            sigma_min=self.sigma_min,
            sigma_data=self.sigma_data,
        )
        attn_mask = self.get_attn_mask(x)
        features = [torch.chunk(el, 2, dim=-1) for el in features]
        features = [torch.cat(el, dim=0) for el in features]
        x = torch.chunk(x, 2, dim=-1)
        x = torch.cat(x, dim=0)
        inp = x.clone()
        x = c_in * x
        x, features_dec = self.frontend_decoder_down(
            x,
            cond=time_emb,
            features=features,
            gain=gain_inp,
            log_magnitude=log_magnitude,
        )
        x = torch.cat(torch.chunk(x, 2, dim=0), dim=-2)
        time_emb_left, time_emb_right = torch.chunk(time_emb, 2, dim=0)
        time_emb_transformer = torch.cat(
            (
                torch.ones(
                    (x.shape[0], x.shape[1] // 2 + self.num_latents, time_emb.shape[-1]),
                    dtype=x.dtype,
                    device=x.device,
                )
                * time_emb_left.unsqueeze(-2),
                torch.ones(
                    (x.shape[0], x.shape[1] // 2 + self.num_latents, time_emb.shape[-1]),
                    dtype=x.dtype,
                    device=x.device,
                )
                * time_emb_right.unsqueeze(-2),
            ),
            dim=-2,
        )
        x = self.decoder(
            x,
            time_emb_transformer,
            latent=self.lat2patch(src_tokens),
            skip_input_layer=True,
            skip_output_layer=True,
            attn_mask=attn_mask,
            print_magnitudes=log_magnitude,
        )
        x = torch.cat(torch.chunk(x, 2, dim=-2), dim=0)
        x = self.frontend_decoder_up(
            x,
            cond=time_emb,
            features=features_dec,
            gain=gain_out,
            log_magnitude=log_magnitude,
        )[0]
        x = c_skip * inp + c_out * x
        x_left, x_right = torch.chunk(x, 2, dim=0)
        if output == "left":
            return x_left
        if output == "right":
            return x_right
        if output == "both":
            return torch.cat((x_left, x_right), dim=-1)
        raise ValueError("output must be one of: left, right, both")

    @torch.no_grad()
    def generate_waveform(
        self,
        src: torch.Tensor,
        task_id: str = "reconstruct",
        dont_quantize: bool = True,
    ):
        del dont_quantize
        src = self.prepare_audio_batch(src)
        src_representation = self.to_representation_encoder(src)
        src_tokens, src_features = self.encode_source(src_representation)
        noise = torch.randn_like(src_representation) * self.sigma_max
        generated = self.decoder_forward(
            noise,
            src_tokens,
            features=src_features,
            sigma_left=self.sigma_max,
            sigma_right=self.sigma_max,
            output="both",
            task_id=task_id,
        )
        waveform = self.to_waveform(generated[..., : src_representation.shape[-1]])
        return src.detach().cpu(), waveform.detach().cpu()
