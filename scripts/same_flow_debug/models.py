from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .audio import AudioProcessor
from scripts.codec_common.quantize import ResidualVectorQuantize
from .transformer import TransformerBlock


def zero_init(module):
    for param in module.parameters():
        param.detach().zero_()
    return module


class FreqGain(nn.Module):
    def __init__(self, freq_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, int(freq_dim), 1)))

    def forward(self, x):
        return x * self.scale


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 128, scale: float = 0.02):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size: int = 128, max_positions: int = 10000):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.max_positions = int(max_positions)

    def forward(self, x):
        freqs = torch.arange(
            start=0,
            end=self.embedding_size // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


@dataclass
class QuantizedLatents:
    continuous: torch.Tensor
    discrete: torch.Tensor
    codes: torch.Tensor
    projected_latents: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    distill_loss: torch.Tensor


class SameFlowBlock(nn.Module):
    """Residual block with SAME-style local transformer sequence modeling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int | None = None,
        *,
        use_2d: bool = False,
        normalize: bool = True,
        dropout_rate: float = 0.0,
        dim_heads: int = 64,
        transformer_depth: int = 1,
        sliding_window: tuple[int, int] | list[int] | None = (4, 4),
        differential: bool = True,
        dyt: bool = True,
        ff_mult: int = 3,
        init_as_zero: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_2d = bool(use_2d)
        self.normalize = bool(normalize)
        self.sliding_window = sliding_window
        self.in_proj = nn.Linear(in_channels, out_channels)
        self.res_proj = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        if self.normalize:
            self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout_rate))
        self.cond_proj = (
            zero_init(nn.Linear(cond_channels, out_channels))
            if cond_channels is not None
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    out_channels,
                    dim_heads=min(int(dim_heads), int(out_channels)),
                    zero_init_branch_outputs=init_as_zero,
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": bool(differential),
                    },
                    ff_kwargs={"mult": ff_mult},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(transformer_depth))
            ]
        )
        self.out_proj = nn.Linear(out_channels, out_channels)
        if init_as_zero:
            zero_init(self.out_proj)

    def _to_tokens(self, x: torch.Tensor):
        if self.use_2d:
            batch, channels, freq, frames = x.shape
            tokens = x.permute(0, 2, 3, 1).reshape(batch * freq, frames, channels)
            return tokens, (batch, freq)
        return x.transpose(1, 2), None

    def _from_tokens(self, tokens: torch.Tensor, shape_info):
        if self.use_2d:
            batch, freq = shape_info
            frames, channels = tokens.shape[1], tokens.shape[2]
            return tokens.reshape(batch, freq, frames, channels).permute(0, 3, 1, 2)
        return tokens.transpose(1, 2)

    def _window_for(self, x: torch.Tensor):
        if self.sliding_window is None:
            return None
        left, right = self.sliding_window
        seq_len = x.shape[-1]
        return [
            min(int(left), max(seq_len - 1, 0)),
            min(int(right), max(seq_len - 1, 0)),
        ]

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None):
        tokens, shape_info = self._to_tokens(x)
        residual = self.res_proj(tokens)
        tokens = self.in_proj(tokens)
        if self.normalize:
            tokens = self.norm(tokens)
        tokens = self.activation(tokens)
        if self.cond_proj is not None and time_emb is not None:
            cond = self.cond_proj(time_emb)
            if self.use_2d:
                batch, freq = shape_info
                cond = cond[:, None, None, :].expand(batch, freq, tokens.shape[1], -1)
                cond = cond.reshape(batch * freq, tokens.shape[1], -1)
            else:
                cond = cond[:, None, :]
            tokens = tokens + cond
        window = self._window_for(tokens.transpose(1, 2))
        for layer in self.layers:
            tokens = layer(tokens, self_attention_flash_sliding_window=window)
        tokens = self.dropout(tokens)
        tokens = residual + self.out_proj(tokens)
        return self._from_tokens(tokens, shape_info)


class ZeroTokenProjection(nn.Module):
    """Per-token output projection matching Music2Latent's zero conv_out head."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(int(in_channels), int(out_channels))
        zero_init(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = self.proj(tokens)
        return tokens.transpose(1, 2)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _zero_pad_modulo_sequence(x, size, dim=-2):
    pad_len = (size - x.shape[dim] % size) % size
    if pad_len <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    return torch.cat(
        [x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)],
        dim=dim,
    )


class SameFlowResampling1d(nn.Module):
    """SAME TransformerResamplingBlock, applied to one sequence axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        type: str,
        transformer_depth: int = 1,
        dim_heads: int = 64,
        sliding_window: tuple[int, int] | list[int] | None = None,
        chunk_size: int = 128,
        chunk_midpoint_shift: bool = False,
        differential: bool = True,
        variable_stride: bool = False,
        feat_scale: bool = False,
        sinusoidal_blocks: int = 0,
        mask_noise: float = 0.0,
        mapping_bias: bool = True,
        dyt: bool = True,
        ff_mult: int = 3,
        init_as_zero: bool = True,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        if type not in {"encoder", "decoder"}:
            raise ValueError(f"unknown resampling type {type!r}")
        transformer_dim = out_channels if type == "encoder" else in_channels
        self.type = type
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.variable_stride = bool(variable_stride)
        self.chunk_size = int(chunk_size)
        self.chunk_midpoint_shift = bool(chunk_midpoint_shift)
        self.mask_noise = float(mask_noise)
        self.sliding_window_latents = sliding_window
        self.sliding_window_seq = self._get_sliding_window_size(
            sliding_window,
            self.stride,
        )
        (
            self.input_seg_size,
            self.output_seg_size,
            self.sub_chunk_size,
        ) = self._get_seg_sizes(self.stride)
        kernel_size = 3 if conv_mapping else 1
        self.mapping = (
            WNConv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                bias=mapping_bias,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.new_tokens = nn.Parameter(
            1e-5
            * torch.randn(
                1,
                1 if self.variable_stride else self.output_seg_size,
                transformer_dim,
            )
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    transformer_dim,
                    dim_heads=min(int(dim_heads), int(transformer_dim)),
                    zero_init_branch_outputs=False,
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": bool(differential),
                        "feat_scale": bool(feat_scale),
                    },
                    ff_kwargs={
                        "mult": ff_mult,
                        "no_bias": False,
                        "sinusoidal": (int(transformer_depth) - idx)
                        < int(sinusoidal_blocks),
                    },
                    norm_kwargs={"eps": 1e-3},
                )
                for idx in range(int(transformer_depth))
            ]
        )
        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [win * (stride + 1 + prepend_cond_length) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length=0):
        sub_chunk_size = int(stride) + 1 + int(prepend_cond_length)
        return (
            int(stride) if self.type == "encoder" else 1,
            1 if self.type == "encoder" else int(stride),
            sub_chunk_size,
        )

    def forward(self, x: torch.Tensor, stride=None, return_features=False, override_new_tokens=None):
        # x: [B, C, N]
        batch_size = x.shape[0]
        features = [] if return_features else None
        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride)
            sliding_window = self._get_sliding_window_size(
                self.sliding_window_latents,
                stride,
            )
        if self.type == "encoder":
            if len(self.transformers) > 0:
                x = _zero_pad_modulo_sequence(
                    x,
                    input_seg_size if sliding_window is not None else self.chunk_size,
                    dim=-1,
                )
            x = self.mapping(x)
        if len(self.transformers) > 0:
            x = rearrange(x, "b d n -> b n d")
            if return_features:
                features.append(x)
            if self.type != "encoder":
                active_stride = int(stride or self.stride)
                pad_modulo = (
                    input_seg_size
                    if sliding_window is not None
                    else max(1, self.chunk_size // active_stride)
                )
                x = _zero_pad_modulo_sequence(x, pad_modulo)
            x = rearrange(x, "b (n c) d -> (b n) c d", c=input_seg_size)
            new_tokens = self.new_tokens.expand(x.shape[0], output_seg_size, -1)
            if override_new_tokens is not None:
                new_tokens = rearrange(
                    override_new_tokens,
                    "b (n c) d -> (b n) c d",
                    c=output_seg_size,
                )
                new_tokens = self.new_tokens.expand_as(new_tokens) + new_tokens
            else:
                content_tokens = x.mean(dim=-2, keepdim=True)
                new_tokens = new_tokens + content_tokens.expand(
                    -1,
                    output_seg_size,
                    -1,
                )
                if self.mask_noise > 0:
                    new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x, new_tokens], dim=-2)
            x = rearrange(x, "(b n) c d -> b (n c) d", b=batch_size)
            if sliding_window is None:
                active_stride = int(stride or self.stride)
                effective_chunk_size = self.chunk_size + self.chunk_size // active_stride
                x = _zero_pad_modulo_sequence(x, effective_chunk_size)
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
            for layer in self.transformers:
                x = layer(x, self_attention_flash_sliding_window=sliding_window)
                if return_features:
                    features.append(x)
            if sliding_window is None:
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
            x = rearrange(x, "b (n c) d -> (b n) c d", c=sub_chunk_size)
            x = x[:, -output_seg_size:, :]
            x = rearrange(x, "(b n) c d -> b d (n c)", b=batch_size)
        if self.type == "decoder":
            x = self.mapping(x)
        if return_features:
            return x, features
        return x


class SameFlowResample2d(nn.Module):
    """Apply SAME-style transformer resampling across frequency and/or time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: str,
        type: str,
        same_flow_block: dict | None = None,
    ):
        super().__init__()
        block_kwargs = same_flow_block or {}
        if mode not in {"freq4", "freq2_time2", "time2_freq2"}:
            raise ValueError(f"unknown resampling mode {mode!r}")
        self.mode = mode
        self.type = type
        resampling_kwargs = dict(block_kwargs)
        if mode == "freq4":
            self.freq = SameFlowResampling1d(
                in_channels,
                out_channels,
                stride=4,
                type=type,
                **resampling_kwargs,
            )
            self.time = None
        else:
            if type == "encoder":
                self.freq = SameFlowResampling1d(
                    in_channels,
                    out_channels,
                    stride=2,
                    type="encoder",
                    **resampling_kwargs,
                )
                self.time = SameFlowResampling1d(
                    out_channels,
                    out_channels,
                    stride=2,
                    type="encoder",
                    **resampling_kwargs,
                )
            else:
                self.time = SameFlowResampling1d(
                    in_channels,
                    in_channels,
                    stride=2,
                    type="decoder",
                    **resampling_kwargs,
                )
                self.freq = SameFlowResampling1d(
                    in_channels,
                    out_channels,
                    stride=2,
                    type="decoder",
                    **resampling_kwargs,
                )

    def _apply_freq(self, x):
        batch, channels, freq, frames = x.shape
        seq = x.permute(0, 3, 1, 2).reshape(batch * frames, channels, freq)
        seq = self.freq(seq)
        out_channels, out_freq = seq.shape[1], seq.shape[2]
        return seq.reshape(batch, frames, out_channels, out_freq).permute(0, 2, 3, 1)

    def _apply_time(self, x):
        batch, channels, freq, frames = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(batch * freq, channels, frames)
        seq = self.time(seq)
        out_channels, out_frames = seq.shape[1], seq.shape[2]
        return seq.reshape(batch, freq, out_channels, out_frames).permute(0, 2, 1, 3)

    def forward(self, x):
        if self.mode == "freq4":
            return self._apply_freq(x)
        if self.type == "encoder":
            return self._apply_time(self._apply_freq(x))
        return self._apply_freq(self._apply_time(x))


class SameFlowEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 64,
        layers_list_encoder=(1, 1, 1, 1, 1),
        multipliers_list=(1, 2, 4, 4, 4),
        freq_downsample_list=(1, 0, 0, 0),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        frequency_scaling: bool = True,
        normalization: bool = True,
        bottleneck_channels: int = 64,
        pre_normalize_2d_to_1d: bool = True,
        pre_normalize_downsampling_encoder: bool = True,
        hop: int = 512,
        data_channels: int = 4,
        dropout_rate: float = 0.0,
        same_flow_block: dict | None = None,
    ):
        super().__init__()
        block_kwargs = same_flow_block or {}
        self.layers_list = list(layers_list_encoder)
        self.multipliers_list = list(multipliers_list)
        self.frequency_scaling = bool(frequency_scaling)
        self.pre_normalize_2d_to_1d = bool(pre_normalize_2d_to_1d)
        input_channels = int(base_channels * self.multipliers_list[0])
        self.gain = FreqGain(freq_dim=hop * 2)
        self.input_proj = SameFlowBlock(
            data_channels,
            input_channels,
            use_2d=True,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        self.freq_dim = (hop * 2) // (4 ** list(freq_downsample_list).count(1))
        self.freq_dim = self.freq_dim // (2 ** list(freq_downsample_list).count(0))

        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(self.layers_list, self.multipliers_list)
        ):
            output_channels = int(base_channels * multiplier)
            for _ in range(int(num_layers)):
                down_layers.append(
                    SameFlowBlock(
                        input_channels,
                        output_channels,
                        use_2d=True,
                        normalize=normalization,
                        dropout_rate=dropout_rate,
                        **block_kwargs,
                    )
                )
                input_channels = output_channels
            if i != len(self.layers_list) - 1:
                if freq_downsample_list[i] == 1:
                    down_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            input_channels,
                            mode="freq4",
                            type="encoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                else:
                    down_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            input_channels,
                            mode="freq2_time2",
                            type="encoder",
                            same_flow_block=block_kwargs,
                        )
                    )
        self.down_layers = nn.ModuleList(down_layers)

        if self.pre_normalize_2d_to_1d:
            self.prenorm_1d_to_2d = nn.GroupNorm(
                min(input_channels // 4, 32), input_channels
            )

        bottleneck_layers = [
            SameFlowBlock(
                input_channels * self.freq_dim,
                bottleneck_base_channels,
                use_2d=False,
                normalize=normalization,
                dropout_rate=dropout_rate,
                **block_kwargs,
            )
        ]
        for _ in range(int(num_bottleneck_layers)):
            bottleneck_layers.append(
                SameFlowBlock(
                    bottleneck_base_channels,
                    bottleneck_base_channels,
                    use_2d=False,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    **block_kwargs,
                )
            )
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)
        self.norm_out = nn.GroupNorm(
            min(bottleneck_base_channels // 4, 32), bottleneck_base_channels
        )
        self.activation_out = nn.SiLU()
        self.output_proj = SameFlowBlock(
            bottleneck_base_channels,
            bottleneck_channels,
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        self.activation_bottleneck = nn.Tanh()

    def encode_features(self, x):
        x = self.input_proj(x)
        if self.frequency_scaling:
            x = self.gain(x)
        k = 0
        for i, num_layers in enumerate(self.layers_list):
            for _ in range(int(num_layers)):
                x = self.down_layers[k](x)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.down_layers[k](x)
                k += 1
        if self.pre_normalize_2d_to_1d:
            x = self.prenorm_1d_to_2d(x)
        return x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))

    def project_features(self, x):
        for layer in self.bottleneck_layers:
            x = layer(x)
        hidden = x
        return self.hidden_to_latent(hidden), hidden

    def hidden_to_latent(self, hidden):
        x = self.norm_out(hidden)
        x = self.activation_out(x)
        x = self.output_proj(x)
        return self.activation_bottleneck(x)

    def forward(self, x, extract_features=False, return_hidden=False):
        x = self.encode_features(x)
        if extract_features:
            return x
        continuous, hidden = self.project_features(x)
        if return_hidden:
            return continuous, hidden
        return continuous


class SameFlowDecoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 64,
        layers_list_encoder=(1, 1, 1, 1, 1),
        multipliers_list=(1, 2, 4, 4, 4),
        freq_downsample_list=(1, 0, 0, 0),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        normalization: bool = True,
        bottleneck_channels: int = 64,
        hop: int = 512,
        dropout_rate: float = 0.0,
        same_flow_block: dict | None = None,
    ):
        super().__init__()
        block_kwargs = same_flow_block or {}
        self.layers_list = list(layers_list_encoder)
        input_channels = int(base_channels * list(multipliers_list)[-1])
        self.input_proj = SameFlowBlock(
            bottleneck_channels,
            bottleneck_base_channels,
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        self.freq_dim = (hop * 2) // (4 ** list(freq_downsample_list).count(1))
        self.freq_dim = self.freq_dim // (2 ** list(freq_downsample_list).count(0))
        self.bottleneck_layers = nn.ModuleList(
            [
                SameFlowBlock(
                    bottleneck_base_channels,
                    bottleneck_base_channels,
                    use_2d=False,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    **block_kwargs,
                )
                for _ in range(int(num_bottleneck_layers))
            ]
        )
        self.bottleneck_to_grid = SameFlowBlock(
            bottleneck_base_channels,
            input_channels * self.freq_dim,
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )

        up_layers = []
        multipliers_list_upsampling = (
            list(reversed(multipliers_list))[1:] + list(reversed(multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(freq_downsample_list))
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(self.layers_list), multipliers_list_upsampling)
        ):
            for _ in range(int(num_layers)):
                up_layers.append(
                    SameFlowBlock(
                        input_channels,
                        input_channels,
                        use_2d=True,
                        normalize=normalization,
                        dropout_rate=dropout_rate,
                        **block_kwargs,
                    )
                )
            if i != len(self.layers_list) - 1:
                output_channels = int(base_channels * multiplier)
                if freq_upsample_list[i] == 1:
                    up_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="freq4",
                            type="decoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                else:
                    up_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="time2_freq2",
                            type="decoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                input_channels = output_channels
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.bottleneck_to_grid(x)
        x = torch.cat(torch.chunk(x.unsqueeze(-2), self.freq_dim, -3), -2)

        k = 0
        pyramid_list = []
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for _ in range(int(num_layers)):
                x = self.up_layers[k](x)
                k += 1
            pyramid_list.append(x)
            if i != len(self.layers_list) - 1:
                x = self.up_layers[k](x)
                k += 1
        return pyramid_list[::-1]


class Music2LatentSameFlow2DLegacy(nn.Module):
    """Self-contained Music2Latent consistency model with SAME local transformers."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        sample_rate: int = 48000,
        base_channels: int = 64,
        layers_list=(2, 2, 2, 2, 2),
        multipliers_list=(1, 2, 4, 4, 4),
        attention_list=None,
        freq_downsample_list=(1, 0, 0, 0),
        layers_list_encoder=(1, 1, 1, 1, 1),
        attention_list_encoder=None,
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        frequency_scaling: bool = True,
        heads: int = 4,
        cond_channels: int = 256,
        use_fourier: bool = False,
        fourier_scale: float = 0.2,
        normalization: bool = True,
        dropout_rate: float = 0.0,
        min_res_dropout: int = 16,
        init_as_zero: bool = True,
        bottleneck_channels: int = 64,
        pre_normalize_2d_to_1d: bool = True,
        pre_normalize_downsampling_encoder: bool = True,
        hop: int = 240,
        data_channels: int = 4,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        mixed_precision: bool = True,
        rho: float = 7.0,
        max_waveform_length_encode: int = 48000 * 60,
        max_batch_size_encode: int = 1,
        max_waveform_length_decode: int = 48000 * 60,
        max_batch_size_decode: int = 1,
        quantizer_num_quantizers: int = 0,
        quantizer_codebook_size: int = 1024,
        quantizer_codebook_dim: int | None = 8,
        quantizer_dropout: float = 0.0,
        quantizer_decay: float = 0.8,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 10,
        quantizer_threshold_ema_dead_code: int = 2,
        same_flow_block: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        del attention_list, attention_list_encoder, heads, min_res_dropout
        self.sigma_max = float(sigma_max)
        self.frequency_scaling = bool(frequency_scaling)
        self.layers_list = list(layers_list)
        self.multipliers_list = list(multipliers_list)
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.freq_downsample_list = list(freq_downsample_list)
        self.layers_list_encoder = list(layers_list_encoder)
        self.bottleneck_base_channels = int(bottleneck_base_channels)
        self.num_bottleneck_layers = int(num_bottleneck_layers)
        self.cond_channels = int(cond_channels)
        self.normalization = bool(normalization)
        self.dropout_rate = float(dropout_rate)
        self.init_as_zero = bool(init_as_zero)
        self.bottleneck_channels = int(bottleneck_channels)
        self.pre_normalize_2d_to_1d = bool(pre_normalize_2d_to_1d)
        self.pre_normalize_downsampling_encoder = bool(pre_normalize_downsampling_encoder)
        self.mixed_precision = bool(mixed_precision)
        self.data_channels = int(data_channels)
        self.quantizer_num_quantizers = int(quantizer_num_quantizers)
        self.sigma_min = float(sigma_min)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.audio_processor = audio_processor
        block_kwargs = same_flow_block or {}

        self.encoder = SameFlowEncoder(
            base_channels=base_channels,
            layers_list_encoder=self.layers_list_encoder,
            multipliers_list=self.multipliers_list,
            freq_downsample_list=self.freq_downsample_list,
            bottleneck_base_channels=self.bottleneck_base_channels,
            num_bottleneck_layers=self.num_bottleneck_layers,
            frequency_scaling=self.frequency_scaling,
            normalization=self.normalization,
            bottleneck_channels=self.bottleneck_channels,
            pre_normalize_2d_to_1d=self.pre_normalize_2d_to_1d,
            pre_normalize_downsampling_encoder=self.pre_normalize_downsampling_encoder,
            hop=self.hop,
            data_channels=self.data_channels,
            dropout_rate=self.dropout_rate,
            same_flow_block=block_kwargs,
        )
        self.decoder = SameFlowDecoder(
            base_channels=base_channels,
            layers_list_encoder=self.layers_list_encoder,
            multipliers_list=self.multipliers_list,
            freq_downsample_list=self.freq_downsample_list,
            bottleneck_base_channels=self.bottleneck_base_channels,
            num_bottleneck_layers=self.num_bottleneck_layers,
            normalization=self.normalization,
            bottleneck_channels=self.bottleneck_channels,
            hop=self.hop,
            dropout_rate=self.dropout_rate,
            same_flow_block=block_kwargs,
        )

        self.quantizer = None
        if self.quantizer_num_quantizers > 0:
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.bottleneck_base_channels,
                n_codebooks=self.quantizer_num_quantizers,
                codebook_size=quantizer_codebook_size,
                codebook_dim=quantizer_codebook_dim,
                quantizer_dropout=quantizer_dropout,
                decay=quantizer_decay,
                kmeans_init=quantizer_kmeans_init,
                kmeans_iters=quantizer_kmeans_iters,
                threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            )

        self.emb = (
            GaussianFourierProjection(embedding_size=self.cond_channels, scale=fourier_scale)
            if use_fourier
            else PositionalEmbedding(embedding_size=self.cond_channels)
        )
        self.emb_proj = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
        )
        self.scale_inp = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(self.cond_channels, self.hop * 2))
            if self.init_as_zero
            else nn.Linear(self.cond_channels, self.hop * 2),
        )
        self.scale_out = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(self.cond_channels, self.hop * 2))
            if self.init_as_zero
            else nn.Linear(self.cond_channels, self.hop * 2),
        )

        input_channels = int(base_channels * self.multipliers_list[0])
        self.input_proj = SameFlowBlock(
            self.data_channels,
            input_channels,
            use_2d=True,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **block_kwargs,
        )

        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(self.layers_list, self.multipliers_list)
        ):
            output_channels = int(base_channels * multiplier)
            for _ in range(int(num_layers)):
                down_layers.append(
                    SameFlowBlock(
                        output_channels,
                        output_channels,
                        use_2d=True,
                        normalize=self.normalization,
                        dropout_rate=self.dropout_rate,
                        **block_kwargs,
                    )
                )
                down_layers.append(
                    SameFlowBlock(
                        output_channels,
                        output_channels,
                        cond_channels=self.cond_channels,
                        use_2d=True,
                        normalize=self.normalization,
                        dropout_rate=self.dropout_rate,
                        **block_kwargs,
                    )
                )
                input_channels = output_channels
            if i != len(self.layers_list) - 1:
                output_channels = int(base_channels * self.multipliers_list[i + 1])
                if self.freq_downsample_list[i] == 1:
                    down_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="freq4",
                            type="encoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                else:
                    down_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="freq2_time2",
                            type="encoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                input_channels = output_channels
        self.down_layers = nn.ModuleList(down_layers)

        up_layers = []
        multipliers_list_upsampling = (
            list(reversed(self.multipliers_list))[1:]
            + list(reversed(self.multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(self.freq_downsample_list))
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(self.layers_list), multipliers_list_upsampling)
        ):
            for _ in range(int(num_layers)):
                up_layers.append(
                    SameFlowBlock(
                        input_channels,
                        input_channels,
                        use_2d=True,
                        normalize=self.normalization,
                        dropout_rate=self.dropout_rate,
                        **block_kwargs,
                    )
                )
                up_layers.append(
                    SameFlowBlock(
                        input_channels,
                        input_channels,
                        cond_channels=self.cond_channels,
                        use_2d=True,
                        normalize=self.normalization,
                        dropout_rate=self.dropout_rate,
                        **block_kwargs,
                    )
                )
            if i != len(self.layers_list) - 1:
                output_channels = int(base_channels * multiplier)
                if freq_upsample_list[i] == 1:
                    up_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="freq4",
                            type="decoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                else:
                    up_layers.append(
                        SameFlowResample2d(
                            input_channels,
                            output_channels,
                            mode="time2_freq2",
                            type="decoder",
                            same_flow_block=block_kwargs,
                        )
                    )
                input_channels = output_channels
        self.up_layers = nn.ModuleList(up_layers)

        self.decoded_proj = SameFlowBlock(
            input_channels,
            input_channels,
            use_2d=True,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **block_kwargs,
        )
        self.norm_out = nn.GroupNorm(min(input_channels // 4, 32), input_channels)
        self.activation_out = nn.SiLU()
        output_kwargs = dict(block_kwargs)
        output_kwargs["init_as_zero"] = self.init_as_zero
        self.output_proj = SameFlowBlock(
            input_channels,
            self.data_channels,
            use_2d=True,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **output_kwargs,
        )

        self.max_waveform_length_encode = int(max_waveform_length_encode)
        self.max_batch_size_encode = int(max_batch_size_encode)
        self.max_waveform_length_decode = int(max_waveform_length_decode)
        self.max_batch_size_decode = int(max_batch_size_decode)

    @property
    def has_quantizer(self):
        return self.quantizer is not None

    def quantize_representation(self, representation, detach_encoder=True, n_quantizers=None):
        if self.quantizer is None:
            raise RuntimeError("SameFlow quantizer is disabled")
        continuous, hidden = self.encoder(representation, return_hidden=True)
        quantizer_input = hidden.detach() if detach_encoder else hidden
        quantized_hidden, codes, commitment_loss = self.quantizer(
            quantizer_input,
            n_quantizers=n_quantizers,
        )
        discrete = self.encoder.hidden_to_latent(quantized_hidden)
        distill_loss = F.mse_loss(quantized_hidden.float(), hidden.detach().float())
        return QuantizedLatents(
            continuous=continuous,
            discrete=discrete,
            codes=codes,
            projected_latents=quantized_hidden,
            commitment_loss=commitment_loss,
            codebook_loss=hidden.new_zeros(()),
            distill_loss=distill_loss,
        )

    def latent_from_codes(self, codes):
        if self.quantizer is None:
            raise RuntimeError("SameFlow quantizer is disabled")
        quantized_hidden, _ = self.quantizer.from_codes(codes)
        return self.encoder.hidden_to_latent(quantized_hidden)

    @torch.no_grad()
    def decode_codes(
        self,
        codes,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        sampling_mode="stochastic",
    ):
        codes = (
            torch.from_numpy(codes).to(next(self.parameters()).device)
            if isinstance(codes, np.ndarray)
            else codes.to(next(self.parameters()).device)
        )
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        return self.decode(
            self.latent_from_codes(codes),
            denoising_steps=denoising_steps,
            max_waveform_length=max_waveform_length,
            max_batch_size=max_batch_size,
            rescale=rescale,
            target_length=target_length,
            sampling_mode=sampling_mode,
        )

    def forward(self, latents, x, sigma=None, pyramid_latents=None, latent_override=None):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        latents = latents.to(dtype)
        if sigma is None:
            sigma = self.sigma_max
        inp = x
        sigma = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) * sigma
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = self.emb(sigma_log.to(dtype)).to(dtype)
        time_emb = self.emb_proj(emb_sigma_log)
        scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)
        scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)
        c_skip, c_out, c_in = self._get_c(sigma)
        x = c_in * x

        if latent_override is not None:
            latents = latent_override.to(dtype)
        elif latents.shape == x.shape:
            latents = self.encoder(latents.to(dtype))
        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)

        x = self.input_proj(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_inp) * x

        skip_list = []
        k = 0
        for i, num_layers in enumerate(self.layers_list):
            for _ in range(int(num_layers)):
                d = self.down_layers[k](pyramid_latents[i])
                k += 1
                x = (x + d) / np.sqrt(2.0)
                x = self.down_layers[k](x, time_emb)
                skip_list.append(x)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.down_layers[k](x)
                k += 1

        k = 0
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for _ in range(int(num_layers)):
                d = self.up_layers[k](pyramid_latents[-i - 1])
                k += 1
                x = (x + skip_list.pop() + d) / np.sqrt(3.0)
                x = self.up_layers[k](x, time_emb)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.up_layers[k](x)
                k += 1

        d = self.decoded_proj(pyramid_latents[0])
        x = (x + d) / np.sqrt(2.0)
        x = self.norm_out(x)
        x = self.activation_out(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_out) * x
        x = self.output_proj(x)
        return c_skip * inp + c_out * x

    @torch.no_grad()
    def encode(
        self,
        path_or_audio,
        max_waveform_length=None,
        max_batch_size=None,
        extract_features=False,
        rescale=1,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        self.eval()
        device = next(self.parameters()).device
        max_waveform_length = max_waveform_length or self.max_waveform_length_encode
        max_batch_size = max_batch_size or self.max_batch_size_encode

        if isinstance(path_or_audio, str):
            audio, _ = sf.read(path_or_audio, dtype="float32", always_2d=True)
            audio = np.transpose(audio, [1, 0])
        else:
            audio = path_or_audio
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0) if torch.is_tensor(audio) else np.expand_dims(audio, 0)
        audio = (
            torch.from_numpy(audio).to(device)
            if isinstance(audio, np.ndarray)
            else audio.to(device)
        )
        if audio.ndim == 2 and self.data_channels == audio.shape[0] * 2:
            audio = audio.unsqueeze(0)

        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        original_length = audio.shape[-1]
        min_frames = max(1, int(np.ceil(original_length / self.hop)))
        aligned_frames = int(np.ceil(min_frames / downscaling_factor)) * downscaling_factor
        target_length = aligned_frames * self.hop
        if original_length > target_length:
            audio = audio[..., :target_length]
        elif original_length < target_length:
            audio = F.pad(audio, (0, target_length - original_length))

        repr_encoder = self.audio_processor.to_representation_encoder(audio)
        latent = self._process_long_sequence(
            repr_encoder,
            max_waveform_length,
            max_batch_size,
            downscaling_factor,
            extract_features,
            quantize=quantize,
            return_codes=return_codes,
            n_quantizers=n_quantizers,
        )
        if extract_features or return_codes:
            return latent
        return latent / rescale

    @torch.no_grad()
    def decode(
        self,
        latent,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        sampling_mode="stochastic",
    ):
        self.eval()
        device = next(self.parameters()).device
        max_waveform_length = max_waveform_length or self.max_waveform_length_decode
        max_batch_size = max_batch_size or self.max_batch_size_decode
        latent = latent * rescale
        latent = (
            torch.from_numpy(latent).to(device)
            if isinstance(latent, np.ndarray)
            else latent.to(device)
        )
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(0)
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        max_latent_length = int(max_waveform_length / self.hop) // downscaling_factor

        if latent.shape[-1] > max_latent_length:
            segments = []
            for start in range(0, latent.shape[-1], max_latent_length):
                segments.append(latent[:, :, start : start + max_latent_length])
            repr_segments = [
                self._decode_latent_batch(
                    segment,
                    denoising_steps,
                    device,
                    max_batch_size,
                    sampling_mode,
                )
                for segment in segments
            ]
            repr_out = torch.cat(repr_segments, dim=-1)
        else:
            repr_out = self._decode_latent_batch(
                latent,
                denoising_steps,
                device,
                max_batch_size,
                sampling_mode,
            )

        audio = self.audio_processor.to_waveform(repr_out, self.hop)
        if target_length is not None:
            if audio.shape[-1] > target_length:
                audio = audio[..., :target_length]
            elif audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        return audio

    def _decode_latent_batch(
        self,
        latent,
        denoising_steps,
        device,
        max_batch_size,
        sampling_mode="stochastic",
    ):
        if latent.shape[0] <= max_batch_size:
            return self._decode_to_representation(
                latent,
                denoising_steps,
                device,
                sampling_mode,
            )
        chunks = torch.split(latent, max_batch_size, dim=0)
        return torch.cat(
            [
                self._decode_to_representation(
                    chunk,
                    denoising_steps,
                    device,
                    sampling_mode,
                )
                for chunk in chunks
            ],
            dim=0,
        )

    def _decode_to_representation(
        self,
        latents,
        diffusion_steps=1,
        device=None,
        sampling_mode="stochastic",
    ):
        device = device or next(self.parameters()).device
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        sample_length = int(latents.shape[-1] * downscaling_factor)
        initial_noise = (
            torch.randn(
                (latents.shape[0], self.data_channels, self.hop * 2, sample_length),
                device=device,
            )
            * self.sigma_max
        )
        return self._reverse_diffusion(
            initial_noise,
            diffusion_steps,
            latents,
            sampling_mode=sampling_mode,
        )

    def _get_c(self, sigma):
        sigma_correct = self.sigma_min
        c_skip = (self.sigma_data**2.0) / (
            ((sigma - sigma_correct) ** 2.0) + (self.sigma_data**2.0)
        )
        c_out = (self.sigma_data * (sigma - sigma_correct)) / (
            ((self.sigma_data**2.0) + (sigma**2.0)) ** 0.5
        )
        c_in = 1.0 / (((sigma**2.0) + (self.sigma_data**2.0)) ** 0.5)
        return (
            c_skip.reshape(-1, 1, 1, 1),
            c_out.reshape(-1, 1, 1, 1),
            c_in.reshape(-1, 1, 1, 1),
        )

    def _get_sigma(self, i, k):
        return (
            self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / (k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def _reverse_step(self, x, noise, sigma):
        return x + ((sigma**2 - self.sigma_min**2) ** 0.5) * noise

    def _denoise(self, noisy_samples, sigma, latents=None, sampling_mode="stochastic"):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.mixed_precision,
            ):
                pred_samples = self(latents, noisy_samples, sigma)
        if sampling_mode == "stochastic":
            pred_noises = torch.randn_like(pred_samples)
        elif sampling_mode == "deterministic":
            pred_noises = torch.zeros_like(pred_samples)
        else:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")
        return pred_noises, pred_samples

    def _reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
        latents=None,
        sampling_mode="stochastic",
    ):
        next_noisy_samples = initial_noise
        pred_samples = initial_noise
        for k in range(diffusion_steps):
            sigma = self._get_sigma(diffusion_steps + 1 - k, diffusion_steps + 1)
            next_sigma = self._get_sigma(diffusion_steps - k, diffusion_steps + 1)
            pred_noises, pred_samples = self._denoise(
                next_noisy_samples,
                sigma,
                latents,
                sampling_mode=sampling_mode,
            )
            next_noisy_samples = self._reverse_step(
                pred_samples,
                pred_noises,
                next_sigma,
            )
        return pred_samples.detach().cpu()

    def _process_long_sequence(
        self,
        x,
        max_length,
        max_batch_size,
        downscaling_factor,
        extract_features,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        max_sample_length = (
            int(max_length / self.hop) // downscaling_factor
        ) * downscaling_factor
        if x.shape[-1] > max_sample_length:
            latents = []
            for start in range(0, x.shape[-1], max_sample_length):
                segment = x[:, :, :, start : start + max_sample_length]
                if segment.shape[-1] > 0:
                    latents.append(
                        self._encode_batch(
                            segment,
                            max_batch_size,
                            extract_features,
                            quantize,
                            return_codes,
                            n_quantizers,
                        )
                    )
            latent = torch.cat(latents, dim=-1)
        else:
            latent = self._encode_batch(
                x,
                max_batch_size,
                extract_features,
                quantize,
                return_codes,
                n_quantizers,
            )
        if latent.shape[0] > 1:
            latent = torch.cat(torch.split(latent, 1, 0), -1)
        return latent

    def _encode_batch(
        self,
        x,
        max_batch_size,
        extract_features,
        quantize,
        return_codes,
        n_quantizers,
    ):
        if x.shape[0] <= max_batch_size:
            return self._encode_representation_chunk(
                x,
                extract_features=extract_features,
                quantize=quantize,
                return_codes=return_codes,
                n_quantizers=n_quantizers,
            )
        chunks = torch.split(x, max_batch_size, dim=0)
        return torch.cat(
            [
                self._encode_representation_chunk(
                    chunk,
                    extract_features=extract_features,
                    quantize=quantize,
                    return_codes=return_codes,
                    n_quantizers=n_quantizers,
                )
                for chunk in chunks
            ],
            dim=0,
        )

    def _encode_representation_chunk(
        self,
        x,
        extract_features=False,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        if not quantize:
            return self.encoder(x, extract_features=extract_features)
        if extract_features:
            raise ValueError("extract_features=True is not supported with quantize=True")
        quantized = self.quantize_representation(
            x,
            detach_encoder=False,
            n_quantizers=n_quantizers,
        )
        if return_codes:
            return quantized.codes
        return quantized.discrete


class SpectralFrameAdapter(nn.Module):
    """Compress one normalized STFT frame into a SAME-style 1D token."""

    def __init__(
        self,
        data_channels: int,
        freq_bins: int,
        frame_channels: int = 512,
        freq_patch_size: int = 32,
        patch_channels: int = 32,
    ):
        super().__init__()
        self.data_channels = int(data_channels)
        self.freq_bins = int(freq_bins)
        self.frame_channels = int(frame_channels)
        self.freq_patch_size = int(freq_patch_size)
        self.patch_channels = int(patch_channels)
        if self.freq_bins % self.freq_patch_size != 0:
            raise ValueError(
                f"freq_bins={self.freq_bins} must be divisible by "
                f"freq_patch_size={self.freq_patch_size}"
            )
        self.num_patches = self.freq_bins // self.freq_patch_size
        self.patch_dim = self.data_channels * self.freq_patch_size
        patch_dim = self.patch_dim
        packed_patch_dim = self.num_patches * self.patch_channels
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.patch_channels),
            nn.SiLU(),
            nn.Linear(self.patch_channels, self.patch_channels),
        )
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(packed_patch_dim),
            nn.Linear(packed_patch_dim, self.frame_channels),
        )
        self.frame_unproj = nn.Sequential(
            nn.LayerNorm(self.frame_channels),
            nn.Linear(self.frame_channels, packed_patch_dim),
            nn.SiLU(),
        )
        self.patch_decode = nn.Sequential(
            nn.LayerNorm(self.patch_channels),
            nn.Linear(self.patch_channels, self.patch_channels),
            nn.SiLU(),
            nn.Linear(self.patch_channels, patch_dim),
        )

    def to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        if channels != self.data_channels or freq != self.freq_bins:
            raise ValueError(
                f"expected representation [B,{self.data_channels},{self.freq_bins},T], "
                f"got {tuple(x.shape)}"
            )
        x = x.permute(0, 3, 2, 1).reshape(
            batch,
            frames,
            self.num_patches,
            self.freq_patch_size * self.data_channels,
        )
        x = self.patch_embed(x)
        x = x.reshape(batch, frames, self.num_patches * self.patch_channels)
        x = self.frame_proj(x)
        return x.transpose(1, 2)

    def to_raw_frame_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        if channels != self.data_channels or freq != self.freq_bins:
            raise ValueError(
                f"expected representation [B,{self.data_channels},{self.freq_bins},T], "
                f"got {tuple(x.shape)}"
            )
        x = x.permute(0, 3, 2, 1).reshape(
            batch,
            frames,
            self.freq_bins * self.data_channels,
        )
        return x.transpose(1, 2)

    def to_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        if channels != self.data_channels or freq != self.freq_bins:
            raise ValueError(
                f"expected representation [B,{self.data_channels},{self.freq_bins},T], "
                f"got {tuple(x.shape)}"
            )
        x = x.permute(0, 3, 2, 1).reshape(
            batch,
            frames,
            self.num_patches,
            self.freq_patch_size * self.data_channels,
        )
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.reshape(batch * self.num_patches, self.patch_channels, frames)

    def to_raw_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        if channels != self.data_channels or freq != self.freq_bins:
            raise ValueError(
                f"expected representation [B,{self.data_channels},{self.freq_bins},T], "
                f"got {tuple(x.shape)}"
            )
        x = x.permute(0, 3, 2, 1).reshape(
            batch,
            frames,
            self.num_patches,
            self.freq_patch_size * self.data_channels,
        )
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.reshape(batch * self.num_patches, self.patch_dim, frames)

    def patch_tokens_to_representation(
        self,
        tokens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        batch_patches, channels, frames = tokens.shape
        if channels != self.patch_channels:
            raise ValueError(
                f"expected patch token channels {self.patch_channels}, got {channels}"
            )
        if batch_patches % batch_size != 0:
            raise ValueError(
                f"batch_patches={batch_patches} is not divisible by batch_size={batch_size}"
            )
        num_patches = batch_patches // batch_size
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        x = tokens.reshape(batch_size, num_patches, channels, frames)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.patch_decode(x)
        x = x.reshape(batch_size, frames, self.freq_bins, self.data_channels)
        return x.permute(0, 3, 2, 1)

    def patch_values_to_representation(
        self,
        values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        batch_patches, channels, frames = values.shape
        if channels != self.patch_dim:
            raise ValueError(
                f"expected patch value channels {self.patch_dim}, got {channels}"
            )
        if batch_patches % batch_size != 0:
            raise ValueError(
                f"batch_patches={batch_patches} is not divisible by batch_size={batch_size}"
            )
        num_patches = batch_patches // batch_size
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        x = values.reshape(batch_size, num_patches, channels, frames)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(batch_size, frames, self.freq_bins, self.data_channels)
        return x.permute(0, 3, 2, 1)

    def frame_values_to_representation(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, frames = values.shape
        frame_dim = self.data_channels * self.freq_bins
        if channels != frame_dim:
            raise ValueError(f"expected frame value channels {frame_dim}, got {channels}")
        x = values.transpose(1, 2)
        x = x.reshape(batch, frames, self.freq_bins, self.data_channels)
        return x.permute(0, 3, 2, 1)

    def to_representation(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, frames = tokens.shape
        if channels != self.frame_channels:
            raise ValueError(
                f"expected token channels {self.frame_channels}, got {channels}"
            )
        x = tokens.transpose(1, 2)
        x = self.frame_unproj(x)
        x = x.reshape(batch, frames, self.num_patches, self.patch_channels)
        x = self.patch_decode(x)
        x = x.reshape(
            batch,
            frames,
            self.freq_bins,
            self.data_channels,
        )
        return x.permute(0, 3, 2, 1)


class LatentFiLM1d(nn.Module):
    def __init__(self, channels: int, init_as_zero: bool = True):
        super().__init__()
        self.proj = nn.Linear(int(channels), int(channels) * 2)
        if init_as_zero:
            zero_init(self.proj)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond.transpose(1, 2)).chunk(2, dim=-1)
        scale = torch.tanh(scale).transpose(1, 2)
        shift = shift.transpose(1, 2)
        return x * (1.0 + scale) + shift


class SameFlowFrameEncoder(nn.Module):
    def __init__(
        self,
        *,
        adapter: SpectralFrameAdapter,
        base_channels: int = 64,
        layers_list_encoder=(1, 1, 1, 1, 1),
        multipliers_list=(1, 2, 4, 4, 4),
        transition_strides=(1, 2, 2, 2),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        bottleneck_channels: int = 64,
        normalization: bool = True,
        dropout_rate: float = 0.0,
        same_flow_block: dict | None = None,
    ):
        super().__init__()
        block_kwargs = same_flow_block or {}
        self.adapter = adapter
        self.layers_list = list(layers_list_encoder)
        self.channel_dims = [int(base_channels * m) for m in multipliers_list]
        self.transition_strides = list(transition_strides)
        if len(self.channel_dims) != len(self.layers_list):
            raise ValueError("multipliers_list and layers_list_encoder must have the same length")
        if len(self.transition_strides) != len(self.channel_dims) - 1:
            raise ValueError("transition_strides must have len(stages) - 1 entries")

        self.input_proj = SameFlowBlock(
            adapter.frame_channels,
            self.channel_dims[0],
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        stage_layers = []
        transitions = []
        for idx, (num_layers, channels) in enumerate(zip(self.layers_list, self.channel_dims)):
            stage_layers.append(
                nn.ModuleList(
                    [
                        SameFlowBlock(
                            channels,
                            channels,
                            use_2d=False,
                            normalize=normalization,
                            dropout_rate=dropout_rate,
                            **block_kwargs,
                        )
                        for _ in range(int(num_layers))
                    ]
                )
            )
            if idx != len(self.channel_dims) - 1:
                stride = int(self.transition_strides[idx])
                out_channels = self.channel_dims[idx + 1]
                if stride == 1:
                    transitions.append(
                        SameFlowBlock(
                            channels,
                            out_channels,
                            use_2d=False,
                            normalize=normalization,
                            dropout_rate=dropout_rate,
                            **block_kwargs,
                        )
                    )
                else:
                    transitions.append(
                        SameFlowResampling1d(
                            channels,
                            out_channels,
                            stride=stride,
                            type="encoder",
                            **block_kwargs,
                        )
                    )
        self.stage_layers = nn.ModuleList(stage_layers)
        self.transitions = nn.ModuleList(transitions)

        bottleneck_layers = [
            SameFlowBlock(
                self.channel_dims[-1],
                bottleneck_base_channels,
                use_2d=False,
                normalize=normalization,
                dropout_rate=dropout_rate,
                **block_kwargs,
            )
        ]
        for _ in range(int(num_bottleneck_layers)):
            bottleneck_layers.append(
                SameFlowBlock(
                    bottleneck_base_channels,
                    bottleneck_base_channels,
                    use_2d=False,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    **block_kwargs,
                )
            )
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)
        self.norm_out = nn.GroupNorm(
            min(bottleneck_base_channels // 4, 32),
            bottleneck_base_channels,
        )
        self.activation_out = nn.SiLU()
        self.output_proj = SameFlowBlock(
            bottleneck_base_channels,
            bottleneck_channels,
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        self.activation_bottleneck = nn.Tanh()

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter.to_tokens(x)
        x = self.input_proj(x)
        for idx, layers in enumerate(self.stage_layers):
            for layer in layers:
                x = layer(x)
            if idx != len(self.stage_layers) - 1:
                x = self.transitions[idx](x)
        return x

    def project_features(self, x: torch.Tensor):
        for layer in self.bottleneck_layers:
            x = layer(x)
        hidden = x
        return self.hidden_to_latent(hidden), hidden

    def hidden_to_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm_out(hidden)
        x = self.activation_out(x)
        x = self.output_proj(x)
        return self.activation_bottleneck(x)

    def forward(self, x, extract_features=False, return_hidden=False):
        x = self.encode_features(x)
        if extract_features:
            return x
        continuous, hidden = self.project_features(x)
        if return_hidden:
            return continuous, hidden
        return continuous


class SameFlowFrameDecoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 64,
        layers_list_encoder=(1, 1, 1, 1, 1),
        multipliers_list=(1, 2, 4, 4, 4),
        transition_strides=(1, 2, 2, 2),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        bottleneck_channels: int = 64,
        normalization: bool = True,
        dropout_rate: float = 0.0,
        same_flow_block: dict | None = None,
    ):
        super().__init__()
        block_kwargs = same_flow_block or {}
        self.layers_list = list(layers_list_encoder)
        self.channel_dims = [int(base_channels * m) for m in multipliers_list]
        self.transition_strides = list(transition_strides)
        self.input_proj = SameFlowBlock(
            bottleneck_channels,
            bottleneck_base_channels,
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )
        self.bottleneck_layers = nn.ModuleList(
            [
                SameFlowBlock(
                    bottleneck_base_channels,
                    bottleneck_base_channels,
                    use_2d=False,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    **block_kwargs,
                )
                for _ in range(int(num_bottleneck_layers))
            ]
        )
        self.to_deepest = SameFlowBlock(
            bottleneck_base_channels,
            self.channel_dims[-1],
            use_2d=False,
            normalize=normalization,
            dropout_rate=dropout_rate,
            **block_kwargs,
        )

        stage_layers = []
        transitions = []
        reversed_dims = list(reversed(self.channel_dims))
        reversed_layers = list(reversed(self.layers_list))
        reversed_strides = list(reversed(self.transition_strides))
        for idx, (num_layers, channels) in enumerate(zip(reversed_layers, reversed_dims)):
            stage_layers.append(
                nn.ModuleList(
                    [
                        SameFlowBlock(
                            channels,
                            channels,
                            use_2d=False,
                            normalize=normalization,
                            dropout_rate=dropout_rate,
                            **block_kwargs,
                        )
                        for _ in range(int(num_layers))
                    ]
                )
            )
            if idx != len(reversed_dims) - 1:
                stride = int(reversed_strides[idx])
                out_channels = reversed_dims[idx + 1]
                if stride == 1:
                    transitions.append(
                        SameFlowBlock(
                            channels,
                            out_channels,
                            use_2d=False,
                            normalize=normalization,
                            dropout_rate=dropout_rate,
                            **block_kwargs,
                        )
                    )
                else:
                    transitions.append(
                        SameFlowResampling1d(
                            channels,
                            out_channels,
                            stride=stride,
                            type="decoder",
                            **block_kwargs,
                        )
                    )
        self.stage_layers = nn.ModuleList(stage_layers)
        self.transitions = nn.ModuleList(transitions)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.to_deepest(x)
        pyramid = []
        for idx, layers in enumerate(self.stage_layers):
            for layer in layers:
                x = layer(x)
            pyramid.append(x)
            if idx != len(self.stage_layers) - 1:
                x = self.transitions[idx](x)
        return list(reversed(pyramid))


class Music2LatentSameFlow(nn.Module):
    """Music2Latent consistency model with SAME-L style 1D frame tokens."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        sample_rate: int = 48000,
        base_channels: int = 64,
        layers_list=(1, 1, 1, 1, 1),
        multipliers_list=(1, 2, 4, 4, 4),
        attention_list=None,
        freq_downsample_list=(1, 0, 0, 0),
        layers_list_encoder=(1, 1, 1, 1, 1),
        attention_list_encoder=None,
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        frequency_scaling: bool = True,
        heads: int = 4,
        cond_channels: int = 256,
        use_fourier: bool = False,
        fourier_scale: float = 0.2,
        normalization: bool = True,
        dropout_rate: float = 0.0,
        min_res_dropout: int = 16,
        init_as_zero: bool = True,
        bottleneck_channels: int = 64,
        pre_normalize_2d_to_1d: bool = True,
        pre_normalize_downsampling_encoder: bool = True,
        hop: int = 240,
        data_channels: int = 4,
        frame_channels: int = 512,
        freq_patch_size: int = 32,
        patch_channels: int = 32,
        use_latent_prior: bool = False,
        use_latent_film: bool = False,
        use_patch_denoiser: bool = False,
        direct_patch_output: bool = False,
        raw_patch_tokens: bool = False,
        raw_frame_tokens: bool = False,
        direct_frame_output: bool = False,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        mixed_precision: bool = True,
        rho: float = 7.0,
        max_waveform_length_encode: int = 48000 * 60,
        max_batch_size_encode: int = 1,
        max_waveform_length_decode: int = 48000 * 60,
        max_batch_size_decode: int = 1,
        quantizer_num_quantizers: int = 0,
        quantizer_codebook_size: int = 1024,
        quantizer_codebook_dim: int | None = 8,
        quantizer_dropout: float = 0.0,
        quantizer_decay: float = 0.8,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 10,
        quantizer_threshold_ema_dead_code: int = 2,
        same_flow_block: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        del (
            attention_list,
            attention_list_encoder,
            heads,
            min_res_dropout,
            pre_normalize_2d_to_1d,
            pre_normalize_downsampling_encoder,
            kwargs,
        )
        self.sigma_max = float(sigma_max)
        self.frequency_scaling = bool(frequency_scaling)
        self.layers_list = list(layers_list)
        self.multipliers_list = list(multipliers_list)
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.freq_downsample_list = list(freq_downsample_list)
        self.layers_list_encoder = list(layers_list_encoder)
        self.bottleneck_base_channels = int(bottleneck_base_channels)
        self.num_bottleneck_layers = int(num_bottleneck_layers)
        self.cond_channels = int(cond_channels)
        self.normalization = bool(normalization)
        self.dropout_rate = float(dropout_rate)
        self.init_as_zero = bool(init_as_zero)
        self.bottleneck_channels = int(bottleneck_channels)
        self.mixed_precision = bool(mixed_precision)
        self.data_channels = int(data_channels)
        self.freq_bins = self.hop * 2
        self.frame_channels = int(frame_channels)
        self.use_latent_prior = bool(use_latent_prior)
        self.use_latent_film = bool(use_latent_film)
        self.use_patch_denoiser = bool(use_patch_denoiser)
        self.direct_patch_output = bool(direct_patch_output)
        self.raw_patch_tokens = bool(raw_patch_tokens)
        self.raw_frame_tokens = bool(raw_frame_tokens)
        self.direct_frame_output = bool(direct_frame_output)
        if self.direct_patch_output and not self.use_patch_denoiser:
            raise ValueError("direct_patch_output requires use_patch_denoiser=true")
        if self.raw_patch_tokens and not self.use_patch_denoiser:
            raise ValueError("raw_patch_tokens requires use_patch_denoiser=true")
        if self.raw_patch_tokens and not self.direct_patch_output:
            raise ValueError("raw_patch_tokens requires direct_patch_output=true")
        if self.direct_frame_output and self.use_patch_denoiser:
            raise ValueError("direct_frame_output requires use_patch_denoiser=false")
        if self.raw_frame_tokens and self.use_patch_denoiser:
            raise ValueError("raw_frame_tokens requires use_patch_denoiser=false")
        if self.raw_frame_tokens and not self.direct_frame_output:
            raise ValueError("raw_frame_tokens requires direct_frame_output=true")
        self.quantizer_num_quantizers = int(quantizer_num_quantizers)
        self.sigma_min = float(sigma_min)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.audio_processor = audio_processor
        block_kwargs = same_flow_block or {}
        self.transition_strides = [
            2 if int(flag) == 0 else 1 for flag in self.freq_downsample_list
        ]

        self.adapter = SpectralFrameAdapter(
            data_channels=self.data_channels,
            freq_bins=self.freq_bins,
            frame_channels=self.frame_channels,
            freq_patch_size=freq_patch_size,
            patch_channels=patch_channels,
        )
        self.encoder = SameFlowFrameEncoder(
            adapter=self.adapter,
            base_channels=base_channels,
            layers_list_encoder=self.layers_list_encoder,
            multipliers_list=self.multipliers_list,
            transition_strides=self.transition_strides,
            bottleneck_base_channels=self.bottleneck_base_channels,
            num_bottleneck_layers=self.num_bottleneck_layers,
            bottleneck_channels=self.bottleneck_channels,
            normalization=self.normalization,
            dropout_rate=self.dropout_rate,
            same_flow_block=block_kwargs,
        )
        self.decoder = SameFlowFrameDecoder(
            base_channels=base_channels,
            layers_list_encoder=self.layers_list_encoder,
            multipliers_list=self.multipliers_list,
            transition_strides=self.transition_strides,
            bottleneck_base_channels=self.bottleneck_base_channels,
            num_bottleneck_layers=self.num_bottleneck_layers,
            bottleneck_channels=self.bottleneck_channels,
            normalization=self.normalization,
            dropout_rate=self.dropout_rate,
            same_flow_block=block_kwargs,
        )

        self.quantizer = None
        if self.quantizer_num_quantizers > 0:
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.bottleneck_base_channels,
                n_codebooks=self.quantizer_num_quantizers,
                codebook_size=quantizer_codebook_size,
                codebook_dim=quantizer_codebook_dim,
                quantizer_dropout=quantizer_dropout,
                decay=quantizer_decay,
                kmeans_init=quantizer_kmeans_init,
                kmeans_iters=quantizer_kmeans_iters,
                threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            )

        self.emb = (
            GaussianFourierProjection(
                embedding_size=self.cond_channels,
                scale=fourier_scale,
            )
            if use_fourier
            else PositionalEmbedding(embedding_size=self.cond_channels)
        )
        self.emb_proj = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
        )
        input_channels = int(base_channels * self.multipliers_list[0])
        if self.raw_patch_tokens:
            denoiser_input_channels = self.adapter.patch_dim
        elif self.use_patch_denoiser:
            denoiser_input_channels = self.adapter.patch_channels
        elif self.raw_frame_tokens:
            denoiser_input_channels = self.data_channels * self.freq_bins
        else:
            denoiser_input_channels = self.frame_channels
        if self.direct_patch_output:
            denoiser_output_channels = self.adapter.patch_dim
        elif self.direct_frame_output:
            denoiser_output_channels = self.data_channels * self.freq_bins
        else:
            denoiser_output_channels = denoiser_input_channels
        self.scale_inp = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(self.cond_channels, input_channels))
            if self.init_as_zero
            else nn.Linear(self.cond_channels, input_channels),
        )
        self.scale_out = nn.Sequential(
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            nn.Linear(self.cond_channels, self.cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(self.cond_channels, input_channels))
            if self.init_as_zero
            else nn.Linear(self.cond_channels, input_channels),
        )

        self.input_proj = SameFlowBlock(
            denoiser_input_channels,
            input_channels,
            use_2d=False,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **block_kwargs,
        )
        if self.use_patch_denoiser:
            self.down_patch_embeddings = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, self.adapter.num_patches, channels, 1))
                    for channels in self.encoder.channel_dims
                ]
            )
            self.up_patch_embeddings = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, self.adapter.num_patches, channels, 1))
                    for channels in reversed(self.encoder.channel_dims)
                ]
            )
            self.final_patch_embedding = nn.Parameter(
                torch.zeros(1, self.adapter.num_patches, input_channels, 1)
            )
        self.down_blocks = nn.ModuleList()
        self.down_latent_films = nn.ModuleList()
        self.down_transitions = nn.ModuleList()
        for idx, (num_layers, channels) in enumerate(zip(self.layers_list, self.encoder.channel_dims)):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                SameFlowBlock(
                                    channels,
                                    channels,
                                    use_2d=False,
                                    normalize=self.normalization,
                                    dropout_rate=self.dropout_rate,
                                    **block_kwargs,
                                ),
                                SameFlowBlock(
                                    channels,
                                    channels,
                                    cond_channels=self.cond_channels,
                                    use_2d=False,
                                    normalize=self.normalization,
                                    dropout_rate=self.dropout_rate,
                                    **block_kwargs,
                                ),
                            ]
                        )
                        for _ in range(int(num_layers))
                    ]
                )
            )
            self.down_latent_films.append(
                nn.ModuleList(
                    [
                        LatentFiLM1d(channels, init_as_zero=True)
                        for _ in range(int(num_layers))
                    ]
                )
            )
            if idx != len(self.encoder.channel_dims) - 1:
                stride = int(self.transition_strides[idx])
                out_channels = self.encoder.channel_dims[idx + 1]
                if stride == 1:
                    self.down_transitions.append(
                        SameFlowBlock(
                            channels,
                            out_channels,
                            use_2d=False,
                            normalize=self.normalization,
                            dropout_rate=self.dropout_rate,
                            **block_kwargs,
                        )
                    )
                else:
                    self.down_transitions.append(
                        SameFlowResampling1d(
                            channels,
                            out_channels,
                            stride=stride,
                            type="encoder",
                            **block_kwargs,
                        )
                    )

        reversed_dims = list(reversed(self.encoder.channel_dims))
        reversed_layers = list(reversed(self.layers_list))
        reversed_strides = list(reversed(self.transition_strides))
        self.up_blocks = nn.ModuleList()
        self.up_latent_films = nn.ModuleList()
        self.up_transitions = nn.ModuleList()
        for idx, (num_layers, channels) in enumerate(zip(reversed_layers, reversed_dims)):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                SameFlowBlock(
                                    channels,
                                    channels,
                                    use_2d=False,
                                    normalize=self.normalization,
                                    dropout_rate=self.dropout_rate,
                                    **block_kwargs,
                                ),
                                SameFlowBlock(
                                    channels,
                                    channels,
                                    cond_channels=self.cond_channels,
                                    use_2d=False,
                                    normalize=self.normalization,
                                    dropout_rate=self.dropout_rate,
                                    **block_kwargs,
                                ),
                            ]
                        )
                        for _ in range(int(num_layers))
                    ]
                )
            )
            self.up_latent_films.append(
                nn.ModuleList(
                    [
                        LatentFiLM1d(channels, init_as_zero=True)
                        for _ in range(int(num_layers))
                    ]
                )
            )
            if idx != len(reversed_dims) - 1:
                stride = int(reversed_strides[idx])
                out_channels = reversed_dims[idx + 1]
                if stride == 1:
                    self.up_transitions.append(
                        SameFlowBlock(
                            channels,
                            out_channels,
                            use_2d=False,
                            normalize=self.normalization,
                            dropout_rate=self.dropout_rate,
                            **block_kwargs,
                        )
                    )
                else:
                    self.up_transitions.append(
                        SameFlowResampling1d(
                            channels,
                            out_channels,
                            stride=stride,
                            type="decoder",
                            **block_kwargs,
                        )
                    )

        self.decoded_proj = SameFlowBlock(
            input_channels,
            input_channels,
            use_2d=False,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **block_kwargs,
        )
        self.norm_out = nn.GroupNorm(min(input_channels // 4, 32), input_channels)
        self.activation_out = nn.SiLU()
        output_kwargs = dict(block_kwargs)
        output_kwargs["init_as_zero"] = self.init_as_zero
        self.output_proj = ZeroTokenProjection(input_channels, denoiser_output_channels)
        self.prior_norm = nn.GroupNorm(min(input_channels // 4, 32), input_channels)
        self.prior_activation = nn.SiLU()
        self.prior_proj = SameFlowBlock(
            input_channels,
            self.frame_channels,
            use_2d=False,
            normalize=self.normalization,
            dropout_rate=self.dropout_rate,
            **output_kwargs,
        )

        self.max_waveform_length_encode = int(max_waveform_length_encode)
        self.max_batch_size_encode = int(max_batch_size_encode)
        self.max_waveform_length_decode = int(max_waveform_length_decode)
        self.max_batch_size_decode = int(max_batch_size_decode)

    def _repeat_patch_condition(
        self,
        condition: torch.Tensor,
        patch_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames = condition.shape
        patches = int(self.adapter.num_patches)
        condition = condition[:, None, :, :].expand(batch, patches, channels, frames)
        condition = condition + patch_embedding.to(condition.dtype)
        return condition.reshape(batch * patches, channels, frames)

    @property
    def has_quantizer(self):
        return self.quantizer is not None

    def quantize_representation(self, representation, detach_encoder=True, n_quantizers=None):
        if self.quantizer is None:
            raise RuntimeError("SameFlow quantizer is disabled")
        continuous, hidden = self.encoder(representation, return_hidden=True)
        quantizer_input = hidden.detach() if detach_encoder else hidden
        quantized_hidden, codes, commitment_loss = self.quantizer(
            quantizer_input,
            n_quantizers=n_quantizers,
        )
        discrete = self.encoder.hidden_to_latent(quantized_hidden)
        distill_loss = F.mse_loss(quantized_hidden.float(), hidden.detach().float())
        return QuantizedLatents(
            continuous=continuous,
            discrete=discrete,
            codes=codes,
            projected_latents=quantized_hidden,
            commitment_loss=commitment_loss,
            codebook_loss=hidden.new_zeros(()),
            distill_loss=distill_loss,
        )

    def latent_from_codes(self, codes):
        if self.quantizer is None:
            raise RuntimeError("SameFlow quantizer is disabled")
        quantized_hidden, _ = self.quantizer.from_codes(codes)
        return self.encoder.hidden_to_latent(quantized_hidden)

    @torch.no_grad()
    def decode_codes(
        self,
        codes,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        sampling_mode="stochastic",
    ):
        codes = (
            torch.from_numpy(codes).to(next(self.parameters()).device)
            if isinstance(codes, np.ndarray)
            else codes.to(next(self.parameters()).device)
        )
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        return self.decode(
            self.latent_from_codes(codes),
            denoising_steps=denoising_steps,
            max_waveform_length=max_waveform_length,
            max_batch_size=max_batch_size,
            rescale=rescale,
            target_length=target_length,
            sampling_mode=sampling_mode,
        )

    def latent_prior(self, latents, pyramid_latents=None):
        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)
        prior = self.decoded_proj(pyramid_latents[0])
        prior = self.prior_norm(prior)
        prior = self.prior_activation(prior)
        prior = self.prior_proj(prior)
        return self.adapter.to_representation(prior)

    def forward(self, latents, x, sigma=None, pyramid_latents=None, latent_override=None):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        latents = latents.to(dtype)
        batch_size = x.shape[0]
        if sigma is None:
            sigma = self.sigma_max
        inp = x
        sigma = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) * sigma
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = self.emb(sigma_log.to(dtype)).to(dtype)
        time_emb = self.emb_proj(emb_sigma_log)
        c_skip, c_out, c_in = self._get_c(sigma)
        x = c_in * x

        if latent_override is not None:
            latents = latent_override.to(dtype)
        elif latents.shape == x.shape:
            latents = self.encoder(latents.to(dtype))
        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)
        latent_prior = None
        if self.use_latent_prior:
            latent_prior = self.latent_prior(latents, pyramid_latents=pyramid_latents)
            inp_residual = inp - latent_prior
            x = c_in * inp_residual
        else:
            inp_residual = inp

        if self.use_patch_denoiser:
            if self.raw_patch_tokens:
                x = self.adapter.to_raw_patch_tokens(x)
            else:
                x = self.adapter.to_patch_tokens(x)
            time_emb = time_emb.repeat_interleave(self.adapter.num_patches, dim=0)
            emb_sigma_log = emb_sigma_log.repeat_interleave(
                self.adapter.num_patches,
                dim=0,
            )
            scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0], -1, 1)
            scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0], -1, 1)
        else:
            if self.raw_frame_tokens:
                x = self.adapter.to_raw_frame_tokens(x)
            else:
                x = self.adapter.to_tokens(x)
            scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0], -1, 1)
            scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0], -1, 1)
        x = self.input_proj(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_inp) * x

        skip_list = []
        for idx, blocks in enumerate(self.down_blocks):
            for block_idx, (latent_block, denoise_block) in enumerate(blocks):
                d = latent_block(pyramid_latents[idx])
                if self.use_patch_denoiser:
                    d = self._repeat_patch_condition(
                        d,
                        self.down_patch_embeddings[idx],
                    )
                x = (x + d) / np.sqrt(2.0)
                if self.use_latent_film:
                    x = self.down_latent_films[idx][block_idx](x, d)
                x = denoise_block(x, time_emb)
                skip_list.append(x)
            if idx != len(self.down_blocks) - 1:
                x = self.down_transitions[idx](x)

        for idx, blocks in enumerate(self.up_blocks):
            pyramid = pyramid_latents[-idx - 1]
            for block_idx, (latent_block, denoise_block) in enumerate(blocks):
                d = latent_block(pyramid)
                if self.use_patch_denoiser:
                    d = self._repeat_patch_condition(
                        d,
                        self.up_patch_embeddings[idx],
                    )
                x = (x + skip_list.pop() + d) / np.sqrt(3.0)
                if self.use_latent_film:
                    x = self.up_latent_films[idx][block_idx](x, d)
                x = denoise_block(x, time_emb)
            if idx != len(self.up_blocks) - 1:
                x = self.up_transitions[idx](x)

        d = self.decoded_proj(pyramid_latents[0])
        if self.use_patch_denoiser:
            d = self._repeat_patch_condition(d, self.final_patch_embedding)
        x = (x + d) / np.sqrt(2.0)
        x = self.norm_out(x)
        x = self.activation_out(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_out) * x
        x = self.output_proj(x)
        if self.direct_patch_output:
            x = self.adapter.patch_values_to_representation(x, batch_size)
        elif self.direct_frame_output:
            x = self.adapter.frame_values_to_representation(x)
        elif self.use_patch_denoiser:
            x = self.adapter.patch_tokens_to_representation(x, batch_size)
        else:
            x = self.adapter.to_representation(x)
        if latent_prior is not None:
            return latent_prior + c_skip * inp_residual + c_out * x
        return c_skip * inp + c_out * x

    @torch.no_grad()
    def encode(
        self,
        path_or_audio,
        max_waveform_length=None,
        max_batch_size=None,
        extract_features=False,
        rescale=1,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        self.eval()
        device = next(self.parameters()).device
        max_waveform_length = max_waveform_length or self.max_waveform_length_encode
        max_batch_size = max_batch_size or self.max_batch_size_encode

        if isinstance(path_or_audio, str):
            audio, _ = sf.read(path_or_audio, dtype="float32", always_2d=True)
            audio = np.transpose(audio, [1, 0])
        else:
            audio = path_or_audio
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0) if torch.is_tensor(audio) else np.expand_dims(audio, 0)
        audio = (
            torch.from_numpy(audio).to(device)
            if isinstance(audio, np.ndarray)
            else audio.to(device)
        )
        if audio.ndim == 2 and self.data_channels == audio.shape[0] * 2:
            audio = audio.unsqueeze(0)

        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        original_length = audio.shape[-1]
        min_frames = max(1, int(np.ceil(original_length / self.hop)))
        aligned_frames = int(np.ceil(min_frames / downscaling_factor)) * downscaling_factor
        target_length = aligned_frames * self.hop
        if original_length > target_length:
            audio = audio[..., :target_length]
        elif original_length < target_length:
            audio = F.pad(audio, (0, target_length - original_length))

        repr_encoder = self.audio_processor.to_representation_encoder(audio)
        latent = self._process_long_sequence(
            repr_encoder,
            max_waveform_length,
            max_batch_size,
            downscaling_factor,
            extract_features,
            quantize=quantize,
            return_codes=return_codes,
            n_quantizers=n_quantizers,
        )
        if extract_features or return_codes:
            return latent
        return latent / rescale

    @torch.no_grad()
    def decode(
        self,
        latent,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        sampling_mode="stochastic",
    ):
        self.eval()
        device = next(self.parameters()).device
        max_waveform_length = max_waveform_length or self.max_waveform_length_decode
        max_batch_size = max_batch_size or self.max_batch_size_decode
        latent = latent * rescale
        latent = (
            torch.from_numpy(latent).to(device)
            if isinstance(latent, np.ndarray)
            else latent.to(device)
        )
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(0)
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        max_latent_length = int(max_waveform_length / self.hop) // downscaling_factor

        if latent.shape[-1] > max_latent_length:
            segments = []
            for start in range(0, latent.shape[-1], max_latent_length):
                segments.append(latent[:, :, start : start + max_latent_length])
            repr_segments = [
                self._decode_latent_batch(
                    segment,
                    denoising_steps,
                    device,
                    max_batch_size,
                    sampling_mode,
                )
                for segment in segments
            ]
            repr_out = torch.cat(repr_segments, dim=-1)
        else:
            repr_out = self._decode_latent_batch(
                latent,
                denoising_steps,
                device,
                max_batch_size,
                sampling_mode,
            )

        audio = self.audio_processor.to_waveform(repr_out, self.hop)
        if target_length is not None:
            if audio.shape[-1] > target_length:
                audio = audio[..., :target_length]
            elif audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        return audio

    def _decode_latent_batch(
        self,
        latent,
        denoising_steps,
        device,
        max_batch_size,
        sampling_mode="stochastic",
    ):
        if latent.shape[0] <= max_batch_size:
            return self._decode_to_representation(
                latent,
                denoising_steps,
                device,
                sampling_mode,
            )
        chunks = torch.split(latent, max_batch_size, dim=0)
        return torch.cat(
            [
                self._decode_to_representation(
                    chunk,
                    denoising_steps,
                    device,
                    sampling_mode,
                )
                for chunk in chunks
            ],
            dim=0,
        )

    def _decode_to_representation(
        self,
        latents,
        diffusion_steps=1,
        device=None,
        sampling_mode="stochastic",
    ):
        device = device or next(self.parameters()).device
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        sample_length = int(latents.shape[-1] * downscaling_factor)
        initial_noise = (
            torch.randn(
                (latents.shape[0], self.data_channels, self.freq_bins, sample_length),
                device=device,
            )
            * self.sigma_max
        )
        return self._reverse_diffusion(
            initial_noise,
            diffusion_steps,
            latents,
            sampling_mode=sampling_mode,
        )

    def _get_c(self, sigma):
        sigma_correct = self.sigma_min
        c_skip = (self.sigma_data**2.0) / (
            ((sigma - sigma_correct) ** 2.0) + (self.sigma_data**2.0)
        )
        c_out = (self.sigma_data * (sigma - sigma_correct)) / (
            ((self.sigma_data**2.0) + (sigma**2.0)) ** 0.5
        )
        c_in = 1.0 / (((sigma**2.0) + (self.sigma_data**2.0)) ** 0.5)
        return (
            c_skip.reshape(-1, 1, 1, 1),
            c_out.reshape(-1, 1, 1, 1),
            c_in.reshape(-1, 1, 1, 1),
        )

    def _get_sigma(self, i, k):
        return (
            self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / (k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def _reverse_step(self, x, noise, sigma):
        return x + ((sigma**2 - self.sigma_min**2) ** 0.5) * noise

    def _denoise(self, noisy_samples, sigma, latents=None, sampling_mode="stochastic"):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.mixed_precision,
            ):
                pred_samples = self(latents, noisy_samples, sigma)
        if sampling_mode == "stochastic":
            pred_noises = torch.randn_like(pred_samples)
        elif sampling_mode == "deterministic":
            pred_noises = torch.zeros_like(pred_samples)
        else:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")
        return pred_noises, pred_samples

    def _reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
        latents=None,
        sampling_mode="stochastic",
    ):
        next_noisy_samples = initial_noise
        pred_samples = initial_noise
        for k in range(diffusion_steps):
            sigma = self._get_sigma(diffusion_steps + 1 - k, diffusion_steps + 1)
            next_sigma = self._get_sigma(diffusion_steps - k, diffusion_steps + 1)
            pred_noises, pred_samples = self._denoise(
                next_noisy_samples,
                sigma,
                latents,
                sampling_mode=sampling_mode,
            )
            next_noisy_samples = self._reverse_step(
                pred_samples,
                pred_noises,
                next_sigma,
            )
        return pred_samples.detach().cpu()

    def _process_long_sequence(
        self,
        x,
        max_length,
        max_batch_size,
        downscaling_factor,
        extract_features,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        max_sample_length = (
            int(max_length / self.hop) // downscaling_factor
        ) * downscaling_factor
        if x.shape[-1] > max_sample_length:
            latents = []
            for start in range(0, x.shape[-1], max_sample_length):
                segment = x[:, :, :, start : start + max_sample_length]
                if segment.shape[-1] > 0:
                    latents.append(
                        self._encode_batch(
                            segment,
                            max_batch_size,
                            extract_features,
                            quantize,
                            return_codes,
                            n_quantizers,
                        )
                    )
            latent = torch.cat(latents, dim=-1)
        else:
            latent = self._encode_batch(
                x,
                max_batch_size,
                extract_features,
                quantize,
                return_codes,
                n_quantizers,
            )
        if latent.shape[0] > 1:
            latent = torch.cat(torch.split(latent, 1, 0), -1)
        return latent

    def _encode_batch(
        self,
        x,
        max_batch_size,
        extract_features,
        quantize,
        return_codes,
        n_quantizers,
    ):
        if x.shape[0] <= max_batch_size:
            return self._encode_representation_chunk(
                x,
                extract_features=extract_features,
                quantize=quantize,
                return_codes=return_codes,
                n_quantizers=n_quantizers,
            )
        chunks = torch.split(x, max_batch_size, dim=0)
        return torch.cat(
            [
                self._encode_representation_chunk(
                    chunk,
                    extract_features=extract_features,
                    quantize=quantize,
                    return_codes=return_codes,
                    n_quantizers=n_quantizers,
                )
                for chunk in chunks
            ],
            dim=0,
        )

    def _encode_representation_chunk(
        self,
        x,
        extract_features=False,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        if not quantize:
            return self.encoder(x, extract_features=extract_features)
        if extract_features:
            raise ValueError("extract_features=True is not supported with quantize=True")
        quantized = self.quantize_representation(
            x,
            detach_encoder=False,
            n_quantizers=n_quantizers,
        )
        if return_codes:
            return quantized.codes
        return quantized.discrete
