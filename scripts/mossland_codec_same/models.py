from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from .audio import AudioProcessor
from .quantize import ResidualVectorQuantize
from .same1d import TransformerBlock, TransformerResamplingBlock
from .tasks import TASK_NAMES


def zero_init(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.detach().zero_()
    return module


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, min(channels // 4, 32)), channels)


@dataclass
class QuantizedLatents:
    continuous: torch.Tensor
    discrete: torch.Tensor
    codes: torch.Tensor
    projected_latents: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    distill_loss: torch.Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size: int = 128, max_positions: int = 10000):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.max_positions = int(max_positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            self.embedding_size // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / max(1, self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.float().ger(freqs).to(x.dtype)
        return torch.cat([x.sin(), x.cos()], dim=-1)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 128, scale: float = 0.02):
        super().__init__()
        weight = torch.randn(embedding_size // 2) * float(scale)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.weight[None, :].to(x.dtype) * (2.0 * np.pi)
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


class FreqGain(nn.Module):
    def __init__(self, freq_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, int(freq_dim), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ResidualConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int | None = None,
        dropout_rate: float = 0.0,
        zero_second: bool = True,
    ):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = zero_init(conv2) if zero_second else conv2
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.cond = zero_init(nn.Linear(cond_channels, out_channels)) if cond_channels else None
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        y = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        if self.cond is not None and time_emb is not None:
            x = x + self.cond(time_emb)[:, :, None, None]
        x = self.dropout(F.silu(self.norm2(x)))
        return y + self.conv2(x)


class ResidualConv1d(nn.Module):
    def __init__(self, channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, min(channels // 4, 32)), channels)
        self.norm2 = nn.GroupNorm(max(1, min(channels // 4, 32)), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = zero_init(nn.Conv1d(channels, channels, 3, padding=1))
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.dropout(F.silu(self.norm2(x)))
        return y + self.conv2(x)


def _valid_dim_heads(dim: int, requested_dim_heads: int) -> int:
    dim_heads = max(1, min(int(requested_dim_heads), int(dim)))
    while int(dim) % dim_heads != 0:
        dim_heads -= 1
    return dim_heads


def _valid_num_heads(dim: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(dim)))
    while int(dim) % heads != 0:
        heads -= 1
    return heads


class AxialSameBlock2d(nn.Module):
    """2D block that reuses SAME TransformerBlock along frequency then time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int | None = None,
        heads: int = 4,
        depth: int = 1,
        ff_mult: int = 3,
        freq_sliding_window: Sequence[int] | None = (8, 8),
        time_sliding_window: Sequence[int] | None = (1, 1),
        differential: bool = True,
        dyt: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cond = zero_init(nn.Linear(cond_channels, out_channels)) if cond_channels is not None else None
        dim_heads = max(1, min(int(out_channels), int(out_channels) // max(1, int(heads))))
        self.freq_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    out_channels,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=True,
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                    },
                    ff_kwargs={"mult": ff_mult, "no_bias": False},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(depth))
            ]
        )
        self.time_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    out_channels,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=True,
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                    },
                    ff_kwargs={"mult": ff_mult, "no_bias": False},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(depth))
            ]
        )
        self.freq_window = None if freq_sliding_window is None else tuple(int(v) for v in freq_sliding_window)
        self.time_window = None if time_sliding_window is None else tuple(int(v) for v in time_sliding_window)
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        x = self.in_proj(x)
        if self.cond is not None and time_emb is not None:
            x = x + self.cond(time_emb)[:, :, None, None]
        b, c, f, t = x.shape
        xf = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
        for block in self.freq_blocks:
            xf = block(xf, self_attention_flash_sliding_window=self.freq_window)
        x = xf.reshape(b, t, f, c).permute(0, 3, 2, 1)
        xt = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        for block in self.time_blocks:
            xt = block(xt, self_attention_flash_sliding_window=self.time_window)
        x = xt.reshape(b, f, t, c).permute(0, 3, 1, 2)
        return residual + self.dropout(x)


class SameMultiheadAttention(nn.MultiheadAttention):
    def _reset_parameters(self):
        super()._reset_parameters()
        self.out_proj = zero_init(self.out_proj)


class FullFrequencyAttention2d(nn.Module):
    """Original Mossland full-frequency attention for each time frame."""

    def __init__(self, channels: int, heads: int = 4, normalize: bool = True):
        super().__init__()
        self.normalize = bool(normalize)
        self.norm = _group_norm(channels) if self.normalize else nn.Identity()
        self.mha = SameMultiheadAttention(
            embed_dim=int(channels),
            num_heads=_valid_num_heads(channels, heads),
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
        x = self.mha(x, x, x, need_weights=False)[0]
        x = x.reshape(b, t, f, c).permute(0, 3, 2, 1)
        return residual + x


class AxisResampler2d(nn.Module):
    """Apply SAME TransformerResamplingBlock to one 2D axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        axis: str,
        type: str,
        transformer_depth: int = 3,
        dim_heads: int = 64,
        sliding_window: Sequence[int] | None = (1, 1),
        differential: bool = True,
        variable_stride: bool = False,
        mask_noise: float = 0.0,
        ff_mult: int = 3,
        dyt: bool = True,
    ):
        super().__init__()
        dim_heads = _valid_dim_heads(out_channels if type == "encoder" else in_channels, dim_heads)
        if axis not in {"freq", "time"}:
            raise ValueError(f"axis must be 'freq' or 'time', got {axis!r}")
        self.axis = axis
        self.block = TransformerResamplingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=int(stride),
            sliding_window=sliding_window,
            type=type,
            transformer_depth=transformer_depth,
            dim_heads=dim_heads,
            differential=differential,
            variable_stride=variable_stride,
            mask_noise=mask_noise,
            ff_mult=ff_mult,
            dyt=dyt,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        if self.axis == "freq":
            x = x.permute(0, 3, 1, 2).reshape(b * t, c, f)
            x = self.block(x)
            f_out = x.shape[-1]
            return x.reshape(b, t, x.shape[1], f_out).permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 1, 3).reshape(b * f, c, t)
        x = self.block(x)
        t_out = x.shape[-1]
        return x.reshape(b, f, x.shape[1], t_out).permute(0, 2, 1, 3).contiguous()


class SameDownsampleFreq(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 4, **kwargs):
        super().__init__()
        self.freq = AxisResampler2d(in_channels, out_channels, stride, "freq", "encoder", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.freq(x)


class SameDownsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride_f: int = 2, stride_t: int = 2, **kwargs):
        super().__init__()
        self.freq = AxisResampler2d(in_channels, out_channels, stride_f, "freq", "encoder", **kwargs)
        self.time = AxisResampler2d(out_channels, out_channels, stride_t, "time", "encoder", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.time(self.freq(x))


class SameUpsampleFreq(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 4, **kwargs):
        super().__init__()
        self.freq = AxisResampler2d(in_channels, out_channels, stride, "freq", "decoder", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.freq(x)


class SameUpsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride_f: int = 2, stride_t: int = 2, **kwargs):
        super().__init__()
        self.time = AxisResampler2d(in_channels, out_channels, stride_t, "time", "decoder", **kwargs)
        self.freq = AxisResampler2d(out_channels, out_channels, stride_f, "freq", "decoder", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.freq(self.time(x))


class TransformerStage2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        cond_channels: int | None = None,
        resample: tuple[int, int] | None = None,
        resample_out_channels: int | None = None,
        mode: str = "down",
        heads: int = 4,
        transformer_depth: int = 1,
        ff_mult: int = 3,
        window_size: tuple[int, int] = (4, 4),
        transformer_min_channels: int = 256,
        attention: bool = False,
        attention_normalize: bool = True,
        dropout_rate: float = 0.0,
        norm_type: str = "dyt",
        qk_norm: str = "dyt",
        differential: bool = True,
        add_rope: bool = True,
        zero_init_branch_outputs: bool = True,
        same_sliding_window: Sequence[int] | None = (1, 1),
        same_freq_window: Sequence[int] | None = (8, 8),
        same_time_window: Sequence[int] | None = (1, 1),
        same_resampling_depth: int | None = None,
        same_dim_heads: int = 64,
        same_variable_stride_encoder: bool = False,
        same_variable_stride_decoder: bool = False,
        same_mask_noise_encoder: float = 0.001,
        same_mask_noise_decoder: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        same_resampling_depth = int(same_resampling_depth or transformer_depth)
        current = int(in_channels)
        for _ in range(int(num_layers)):
            if int(out_channels) >= int(transformer_min_channels):
                self.blocks.append(
                    AxialSameBlock2d(
                        current,
                        out_channels,
                        cond_channels=cond_channels,
                        heads=heads,
                        depth=transformer_depth,
                        ff_mult=ff_mult,
                        freq_sliding_window=same_freq_window,
                        time_sliding_window=same_time_window,
                        differential=differential,
                        dyt=norm_type == "dyt" and qk_norm == "dyt",
                        dropout_rate=dropout_rate,
                    )
                )
            else:
                self.blocks.append(
                    ResidualConv2d(current, out_channels, cond_channels, dropout_rate)
                )
            self.attentions.append(
                FullFrequencyAttention2d(out_channels, heads=heads, normalize=attention_normalize)
                if attention
                else nn.Identity()
            )
            current = int(out_channels)
        if resample is None:
            self.resample = None
        elif mode == "down":
            resample_kwargs = dict(
                transformer_depth=same_resampling_depth,
                dim_heads=same_dim_heads,
                sliding_window=same_sliding_window,
                differential=differential,
                variable_stride=same_variable_stride_encoder,
                mask_noise=same_mask_noise_encoder,
                ff_mult=ff_mult,
                dyt=norm_type == "dyt" and qk_norm == "dyt",
            )
            if tuple(resample) == (4, 1):
                self.resample = SameDownsampleFreq(
                    current,
                    int(resample_out_channels or out_channels),
                    stride=4,
                    **resample_kwargs,
                )
            else:
                self.resample = SameDownsample2d(
                    current,
                    int(resample_out_channels or out_channels),
                    stride_f=int(resample[0]),
                    stride_t=int(resample[1]),
                    **resample_kwargs,
                )
        elif mode == "up":
            resample_kwargs = dict(
                transformer_depth=same_resampling_depth,
                dim_heads=same_dim_heads,
                sliding_window=same_sliding_window,
                differential=differential,
                variable_stride=same_variable_stride_decoder,
                mask_noise=same_mask_noise_decoder,
                ff_mult=ff_mult,
                dyt=norm_type == "dyt" and qk_norm == "dyt",
            )
            if tuple(resample) == (4, 1):
                self.resample = SameUpsampleFreq(
                    current,
                    int(resample_out_channels or out_channels),
                    stride=4,
                    **resample_kwargs,
                )
            else:
                self.resample = SameUpsample2d(
                    current,
                    int(resample_out_channels or out_channels),
                    stride_f=int(resample[0]),
                    stride_t=int(resample[1]),
                    **resample_kwargs,
                )
        else:
            raise ValueError(f"Unknown stage mode {mode!r}")

    def forward_block(self, x: torch.Tensor, block_idx: int, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        block = self.blocks[int(block_idx)]
        if isinstance(block, (ResidualConv2d, AxialSameBlock2d)):
            x = block(x, time_emb)
        else:
            x = block(x)
        return self.attentions[int(block_idx)](x)

    def forward_blocks(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        for idx in range(len(self.blocks)):
            x = self.forward_block(x, idx, time_emb)
        return x

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.forward_blocks(x, time_emb)
        if self.resample is not None:
            x = self.resample(x)
        return x


def _down_factor(flag: int) -> tuple[int, int]:
    return (4, 1) if int(flag) == 1 else (2, 2)


class Encoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        layers_list_encoder: Sequence[int] = (1, 1, 1, 1, 1),
        attention_list_encoder: Sequence[int] | None = None,
        multipliers_list: Sequence[int] = (1, 2, 4, 4, 4),
        freq_downsample_list: Sequence[int] = (1, 0, 0, 0),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        frequency_scaling: bool = True,
        heads: int = 4,
        bottleneck_channels: int = 64,
        hop: int = 512,
        data_channels: int = 4,
        dropout_rate: float = 0.0,
        same_transformer_depth: int = 1,
        same_ff_mult: int = 3,
        same_window_size: Sequence[int] = (4, 4),
        same_transformer_min_channels: int | None = None,
        same_norm_type: str = "dyt",
        same_qk_norm: str = "dyt",
        same_differential: bool = True,
        same_add_rope: bool = True,
        same_zero_init_branch_outputs: bool = True,
        same_sliding_window: Sequence[int] | None = (1, 1),
        same_freq_window: Sequence[int] | None = (8, 8),
        same_time_window: Sequence[int] | None = (1, 1),
        same_resampling_depth: int | None = None,
        same_dim_heads: int = 64,
        same_variable_stride_encoder: bool = False,
        same_variable_stride_decoder: bool = False,
        same_mask_noise_encoder: float = 0.001,
        same_mask_noise_decoder: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.layers_list = tuple(int(v) for v in layers_list_encoder)
        self.multipliers_list = tuple(int(v) for v in multipliers_list)
        self.freq_downsample_list = tuple(int(v) for v in freq_downsample_list)
        self.attention_list = tuple(
            int(v) for v in (attention_list_encoder or (0, 0, 1, 1, 1))
        )
        self.frequency_scaling = bool(frequency_scaling)
        input_channels = int(base_channels) * self.multipliers_list[0]
        self.conv_inp = nn.Conv2d(data_channels, input_channels, 3, padding=1)
        self.gain = FreqGain(freq_dim=hop * 2)
        self.freq_dim = (hop * 2) // (4 ** self.freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** self.freq_downsample_list.count(0))
        min_channels = same_transformer_min_channels or int(base_channels) * 4
        stages = []
        current = input_channels
        for idx, (num_layers, multiplier) in enumerate(zip(self.layers_list, self.multipliers_list)):
            out_channels = int(base_channels) * int(multiplier)
            resample = None
            if idx != len(self.layers_list) - 1:
                resample = _down_factor(self.freq_downsample_list[idx])
            stages.append(
                TransformerStage2d(
                    current,
                    out_channels,
                    num_layers,
                    resample=resample,
                    mode="down",
                    heads=heads,
                    transformer_depth=same_transformer_depth,
                    ff_mult=same_ff_mult,
                    window_size=tuple(same_window_size),
                    transformer_min_channels=min_channels,
                    attention=self.attention_list[idx] == 1,
                    dropout_rate=dropout_rate,
                    norm_type=same_norm_type,
                    qk_norm=same_qk_norm,
                    differential=same_differential,
                    add_rope=same_add_rope,
                    zero_init_branch_outputs=same_zero_init_branch_outputs,
                    same_sliding_window=same_sliding_window,
                    same_freq_window=same_freq_window,
                    same_time_window=same_time_window,
                    same_resampling_depth=same_resampling_depth,
                    same_dim_heads=same_dim_heads,
                    same_variable_stride_encoder=same_variable_stride_encoder,
                    same_variable_stride_decoder=same_variable_stride_decoder,
                    same_mask_noise_encoder=same_mask_noise_encoder,
                    same_mask_noise_decoder=same_mask_noise_decoder,
                )
            )
            current = out_channels
        self.stages = nn.ModuleList(stages)
        self.prenorm_2d_to_1d = _group_norm(current)
        self.in_to_bottleneck = nn.Conv1d(current * self.freq_dim, bottleneck_base_channels, 1)
        self.bottleneck_layers = nn.ModuleList(
            [ResidualConv1d(bottleneck_base_channels, dropout_rate) for _ in range(int(num_bottleneck_layers))]
        )
        self.norm_out = nn.GroupNorm(max(1, min(bottleneck_base_channels // 4, 32)), bottleneck_base_channels)
        self.conv_out = nn.Conv1d(bottleneck_base_channels, bottleneck_channels, 1)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = self.gain(x)
        for stage in self.stages:
            x = stage(x)
        x = self.prenorm_2d_to_1d(x)
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

    def project_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_to_bottleneck(x)
        for layer in self.bottleneck_layers:
            x = layer(x)
        hidden = x
        continuous = torch.tanh(self.conv_out(F.silu(self.norm_out(x))))
        return continuous, hidden

    def forward(
        self,
        x: torch.Tensor,
        extract_features: bool = False,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.encode_features(x)
        if extract_features:
            return x
        continuous, hidden = self.project_features(x)
        if return_hidden:
            return continuous, hidden
        return continuous


class PyramidDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        layers_list_encoder: Sequence[int] = (1, 1, 1, 1, 1),
        attention_list_encoder: Sequence[int] | None = None,
        multipliers_list: Sequence[int] = (1, 2, 4, 4, 4),
        freq_downsample_list: Sequence[int] = (1, 0, 0, 0),
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        heads: int = 4,
        cond_channels: int = 256,
        bottleneck_channels: int = 64,
        hop: int = 512,
        dropout_rate: float = 0.0,
        same_transformer_depth: int = 1,
        same_ff_mult: int = 3,
        same_window_size: Sequence[int] = (4, 4),
        same_transformer_min_channels: int | None = None,
        same_norm_type: str = "dyt",
        same_qk_norm: str = "dyt",
        same_differential: bool = True,
        same_add_rope: bool = True,
        same_zero_init_branch_outputs: bool = True,
        same_sliding_window: Sequence[int] | None = (1, 1),
        same_freq_window: Sequence[int] | None = (8, 8),
        same_time_window: Sequence[int] | None = (1, 1),
        same_resampling_depth: int | None = None,
        same_dim_heads: int = 64,
        same_variable_stride_encoder: bool = False,
        same_variable_stride_decoder: bool = False,
        same_mask_noise_encoder: float = 0.001,
        same_mask_noise_decoder: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.layers_list = tuple(int(v) for v in layers_list_encoder)
        self.multipliers_list = tuple(int(v) for v in multipliers_list)
        self.freq_downsample_list = tuple(int(v) for v in freq_downsample_list)
        self.attention_list = tuple(
            int(v) for v in (attention_list_encoder or (0, 0, 1, 1, 1))
        )
        self.freq_dim = (hop * 2) // (4 ** self.freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** self.freq_downsample_list.count(0))
        min_channels = same_transformer_min_channels or int(base_channels) * 4
        low_channels = int(base_channels) * self.multipliers_list[-1]
        self.conv_inp = nn.Conv1d(bottleneck_channels, bottleneck_base_channels, 1)
        self.bottleneck_layers = nn.ModuleList(
            [ResidualConv1d(bottleneck_base_channels, dropout_rate) for _ in range(int(num_bottleneck_layers))]
        )
        self.conv_out_bottleneck = nn.Conv1d(bottleneck_base_channels, low_channels * self.freq_dim, 1)
        stages = []
        current = low_channels
        reversed_layers = list(reversed(self.layers_list))
        reversed_mults = list(reversed(self.multipliers_list))
        reversed_flags = list(reversed(self.freq_downsample_list))
        for idx, num_layers in enumerate(reversed_layers):
            current_multiplier = reversed_mults[idx]
            block_channels = int(base_channels) * current_multiplier
            next_channels = block_channels
            resample = None
            if idx != len(reversed_layers) - 1:
                next_channels = int(base_channels) * reversed_mults[idx + 1]
                resample = _down_factor(reversed_flags[idx])
            stages.append(
                TransformerStage2d(
                    current,
                    block_channels,
                    num_layers,
                    cond_channels=cond_channels,
                    resample=resample,
                    resample_out_channels=next_channels,
                    mode="up",
                    heads=heads,
                    transformer_depth=same_transformer_depth,
                    ff_mult=same_ff_mult,
                    window_size=tuple(same_window_size),
                    transformer_min_channels=min_channels,
                    attention=list(reversed(self.attention_list))[idx] == 1,
                    dropout_rate=dropout_rate,
                    norm_type=same_norm_type,
                    qk_norm=same_qk_norm,
                    differential=same_differential,
                    add_rope=same_add_rope,
                    zero_init_branch_outputs=same_zero_init_branch_outputs,
                    same_sliding_window=same_sliding_window,
                    same_freq_window=same_freq_window,
                    same_time_window=same_time_window,
                    same_resampling_depth=same_resampling_depth,
                    same_dim_heads=same_dim_heads,
                    same_variable_stride_encoder=same_variable_stride_encoder,
                    same_variable_stride_decoder=same_variable_stride_decoder,
                    same_mask_noise_encoder=same_mask_noise_encoder,
                    same_mask_noise_decoder=same_mask_noise_decoder,
                )
            )
            current = next_channels
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> list[torch.Tensor]:
        x = self.conv_inp(x)
        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.conv_out_bottleneck(x)
        b, _, t = x.shape
        x = x.reshape(b, -1, self.freq_dim, t)
        pyramid = []
        for stage in self.stages:
            x = stage.forward_blocks(x, time_emb)
            pyramid.append(x)
            if stage.resample is not None:
                x = stage.resample(x)
        return list(reversed(pyramid))


class DenoiserUNet(nn.Module):
    def __init__(
        self,
        data_channels: int,
        base_channels: int,
        layers_list: Sequence[int],
        multipliers_list: Sequence[int],
        attention_list: Sequence[int],
        freq_downsample_list: Sequence[int],
        cond_channels: int,
        hop: int,
        heads: int,
        frequency_scaling: bool,
        dropout_rate: float,
        init_as_zero: bool,
        same_transformer_depth: int,
        same_ff_mult: int,
        same_window_size: Sequence[int],
        same_transformer_min_channels: int | None,
        same_norm_type: str,
        same_qk_norm: str,
        same_differential: bool,
        same_add_rope: bool,
        same_zero_init_branch_outputs: bool,
        same_sliding_window: Sequence[int] | None,
        same_freq_window: Sequence[int] | None,
        same_time_window: Sequence[int] | None,
        same_resampling_depth: int | None,
        same_dim_heads: int,
        same_variable_stride_encoder: bool,
        same_variable_stride_decoder: bool,
        same_mask_noise_encoder: float,
        same_mask_noise_decoder: float,
    ):
        super().__init__()
        self.layers_list = tuple(int(v) for v in layers_list)
        self.multipliers_list = tuple(int(v) for v in multipliers_list)
        self.freq_downsample_list = tuple(int(v) for v in freq_downsample_list)
        self.frequency_scaling = bool(frequency_scaling)
        min_channels = same_transformer_min_channels or int(base_channels) * 4
        first_channels = int(base_channels) * self.multipliers_list[0]
        self.conv_inp = nn.Conv2d(data_channels, first_channels, 3, padding=1)
        scale_out = zero_init(nn.Linear(cond_channels, hop * 2)) if init_as_zero else nn.Linear(cond_channels, hop * 2)
        self.scale_inp = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(cond_channels, hop * 2)) if init_as_zero else nn.Linear(cond_channels, hop * 2),
        )
        self.scale_out = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            scale_out,
        )
        self.down_stages = nn.ModuleList()
        self.down_cond = nn.ModuleList()
        current = first_channels
        for idx, (num_layers, multiplier) in enumerate(zip(self.layers_list, self.multipliers_list)):
            out_channels = int(base_channels) * int(multiplier)
            self.down_cond.append(
                nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1) for _ in range(int(num_layers))])
            )
            resample = None
            next_channels = out_channels
            if idx != len(self.layers_list) - 1:
                resample = _down_factor(self.freq_downsample_list[idx])
                next_channels = int(base_channels) * int(self.multipliers_list[idx + 1])
            self.down_stages.append(
                TransformerStage2d(
                    current,
                    out_channels,
                    num_layers,
                    cond_channels=cond_channels,
                    resample=resample,
                    resample_out_channels=next_channels,
                    mode="down",
                    heads=heads,
                    transformer_depth=same_transformer_depth,
                    ff_mult=same_ff_mult,
                    window_size=tuple(same_window_size),
                    transformer_min_channels=min_channels,
                    attention=int(attention_list[idx]) == 1,
                    dropout_rate=dropout_rate,
                    norm_type=same_norm_type,
                    qk_norm=same_qk_norm,
                    differential=same_differential,
                    add_rope=same_add_rope,
                    zero_init_branch_outputs=same_zero_init_branch_outputs,
                    same_sliding_window=same_sliding_window,
                    same_freq_window=same_freq_window,
                    same_time_window=same_time_window,
                    same_resampling_depth=same_resampling_depth,
                    same_dim_heads=same_dim_heads,
                    same_variable_stride_encoder=same_variable_stride_encoder,
                    same_variable_stride_decoder=same_variable_stride_decoder,
                    same_mask_noise_encoder=same_mask_noise_encoder,
                    same_mask_noise_decoder=same_mask_noise_decoder,
                )
            )
            current = next_channels
        self.up_stages = nn.ModuleList()
        self.up_cond = nn.ModuleList()
        reversed_layers = list(reversed(self.layers_list))
        reversed_mults = list(reversed(self.multipliers_list))
        reversed_flags = list(reversed(self.freq_downsample_list))
        current = int(base_channels) * self.multipliers_list[-1]
        for idx, num_layers in enumerate(reversed_layers):
            block_channels = int(base_channels) * reversed_mults[idx]
            self.up_cond.append(
                nn.ModuleList([nn.Conv2d(block_channels, block_channels, 1) for _ in range(int(num_layers))])
            )
            resample = None
            next_channels = block_channels
            if idx != len(reversed_layers) - 1:
                next_channels = int(base_channels) * reversed_mults[idx + 1]
                resample = _down_factor(reversed_flags[idx])
            self.up_stages.append(
                TransformerStage2d(
                    current,
                    block_channels,
                    num_layers,
                    cond_channels=cond_channels,
                    resample=resample,
                    resample_out_channels=next_channels,
                    mode="up",
                    heads=heads,
                    transformer_depth=same_transformer_depth,
                    ff_mult=same_ff_mult,
                    window_size=tuple(same_window_size),
                    transformer_min_channels=min_channels,
                    attention=int(list(reversed(attention_list))[idx]) == 1,
                    dropout_rate=dropout_rate,
                    norm_type=same_norm_type,
                    qk_norm=same_qk_norm,
                    differential=same_differential,
                    add_rope=same_add_rope,
                    zero_init_branch_outputs=same_zero_init_branch_outputs,
                    same_sliding_window=same_sliding_window,
                    same_freq_window=same_freq_window,
                    same_time_window=same_time_window,
                    same_resampling_depth=same_resampling_depth,
                    same_dim_heads=same_dim_heads,
                    same_variable_stride_encoder=same_variable_stride_encoder,
                    same_variable_stride_decoder=same_variable_stride_decoder,
                    same_mask_noise_encoder=same_mask_noise_encoder,
                    same_mask_noise_decoder=same_mask_noise_decoder,
                )
            )
            current = next_channels
        self.final_cond = nn.Conv2d(first_channels, first_channels, 1)
        self.norm_out = _group_norm(first_channels)
        self.conv_out = nn.Conv2d(first_channels, data_channels, 3, padding=1)
        if init_as_zero:
            zero_init(self.conv_out)

    def forward(self, x: torch.Tensor, pyramid: Sequence[torch.Tensor], time_emb: torch.Tensor) -> torch.Tensor:
        scale_inp = self.scale_inp(time_emb).reshape(x.shape[0], 1, -1, 1)
        scale_out = self.scale_out(time_emb).reshape(x.shape[0], 1, -1, 1)
        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = x * (1.0 + scale_inp)
        skips = []
        for idx, stage in enumerate(self.down_stages):
            for block_idx in range(len(stage.blocks)):
                x = (x + self.down_cond[idx][block_idx](pyramid[idx])) / np.sqrt(2.0)
                x = stage.forward_block(x, block_idx, time_emb)
                skips.append(x)
            if stage.resample is not None:
                x = stage.resample(x)
        for idx, stage in enumerate(self.up_stages):
            pyramid_idx = len(pyramid) - 1 - idx
            for block_idx in range(len(stage.blocks)):
                skip = skips.pop()
                x = (x + skip + self.up_cond[idx][block_idx](pyramid[pyramid_idx])) / np.sqrt(3.0)
                x = stage.forward_block(x, block_idx, time_emb)
            if stage.resample is not None:
                x = stage.resample(x)
        x = (x + self.final_cond(pyramid[0])) / np.sqrt(2.0)
        x = F.silu(self.norm_out(x))
        if self.frequency_scaling:
            x = x * (1.0 + scale_out)
        return self.conv_out(x)


class MosslandCodecSame(nn.Module):
    def __init__(
        self,
        audio_processor: AudioProcessor,
        sample_rate: int = 44100,
        base_channels: int = 64,
        layers_list: Sequence[int] = (2, 2, 2, 2, 2),
        multipliers_list: Sequence[int] = (1, 2, 4, 4, 4),
        attention_list: Sequence[int] | None = None,
        freq_downsample_list: Sequence[int] = (1, 0, 0, 0),
        layers_list_encoder: Sequence[int] = (1, 1, 1, 1, 1),
        attention_list_encoder: Sequence[int] | None = None,
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
        hop: int = 512,
        data_channels: int = 4,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        mixed_precision: bool = True,
        rho: float = 7.0,
        max_waveform_length_encode: int = 44100 * 60,
        max_batch_size_encode: int = 1,
        max_waveform_length_decode: int = 44100 * 60,
        max_batch_size_decode: int = 1,
        quantizer_num_quantizers: int = 0,
        quantizer_codebook_size: int = 1024,
        quantizer_codebook_dim: int | None = 8,
        quantizer_dropout: float = 0.0,
        quantizer_decay: float = 0.8,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 10,
        quantizer_threshold_ema_dead_code: int = 2,
        task_names: Sequence[str] | None = None,
        task_embedding_init_std: float = 0.02,
        same_transformer_depth: int = 1,
        same_ff_mult: int = 3,
        same_window_size: Sequence[int] = (4, 4),
        same_transformer_min_channels: int | None = None,
        same_norm_type: str = "dyt",
        same_qk_norm: str = "dyt",
        same_differential: bool = True,
        same_add_rope: bool = True,
        same_zero_init_branch_outputs: bool = True,
        same_sliding_window: Sequence[int] | None = (1, 1),
        same_freq_window: Sequence[int] | None = (8, 8),
        same_time_window: Sequence[int] | None = (1, 1),
        same_resampling_depth: int | None = None,
        same_dim_heads: int = 64,
        same_variable_stride_encoder: bool = False,
        same_variable_stride_decoder: bool = False,
        same_mask_noise_encoder: float = 0.001,
        same_mask_noise_decoder: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del normalization, min_res_dropout
        del pre_normalize_2d_to_1d, pre_normalize_downsampling_encoder, kwargs
        task_names = tuple(str(name) for name in (task_names or TASK_NAMES))
        if not task_names:
            raise ValueError("task_names must not be empty")
        attention_list = tuple(int(v) for v in (attention_list or (0, 0, 1, 1, 1)))
        attention_list_encoder = tuple(
            int(v) for v in (attention_list_encoder or (0, 0, 1, 1, 1))
        )
        if len(attention_list) != len(layers_list):
            raise ValueError("attention_list must have the same length as layers_list")
        if len(attention_list_encoder) != len(layers_list_encoder):
            raise ValueError("attention_list_encoder must have the same length as layers_list_encoder")
        self.audio_processor = audio_processor
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.freq_downsample_list = tuple(int(v) for v in freq_downsample_list)
        self.layers_list = tuple(int(v) for v in layers_list)
        self.multipliers_list = tuple(int(v) for v in multipliers_list)
        self.bottleneck_channels = int(bottleneck_channels)
        self.data_channels = int(data_channels)
        self.mixed_precision = bool(mixed_precision)
        self.frequency_scaling = bool(frequency_scaling)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.max_waveform_length_encode = int(max_waveform_length_encode)
        self.max_batch_size_encode = int(max_batch_size_encode)
        self.max_waveform_length_decode = int(max_waveform_length_decode)
        self.max_batch_size_decode = int(max_batch_size_decode)
        self.task_names = task_names
        self.task_to_idx = {name: idx for idx, name in enumerate(self.task_names)}
        common_stage_kwargs = dict(
            same_transformer_depth=same_transformer_depth,
            same_ff_mult=same_ff_mult,
            same_window_size=same_window_size,
            same_transformer_min_channels=same_transformer_min_channels,
            same_norm_type=same_norm_type,
            same_qk_norm=same_qk_norm,
            same_differential=same_differential,
            same_add_rope=same_add_rope,
            same_zero_init_branch_outputs=same_zero_init_branch_outputs,
            same_sliding_window=same_sliding_window,
            same_freq_window=same_freq_window,
            same_time_window=same_time_window,
            same_resampling_depth=same_resampling_depth,
            same_dim_heads=same_dim_heads,
            same_variable_stride_encoder=same_variable_stride_encoder,
            same_variable_stride_decoder=same_variable_stride_decoder,
            same_mask_noise_encoder=same_mask_noise_encoder,
            same_mask_noise_decoder=same_mask_noise_decoder,
        )
        self.encoder = Encoder(
            base_channels=base_channels,
            layers_list_encoder=layers_list_encoder,
            attention_list_encoder=attention_list_encoder,
            multipliers_list=multipliers_list,
            freq_downsample_list=freq_downsample_list,
            bottleneck_base_channels=bottleneck_base_channels,
            num_bottleneck_layers=num_bottleneck_layers,
            frequency_scaling=frequency_scaling,
            heads=heads,
            bottleneck_channels=bottleneck_channels,
            hop=hop,
            data_channels=data_channels,
            dropout_rate=dropout_rate,
            **common_stage_kwargs,
        )
        self.decoder = PyramidDecoder(
            base_channels=base_channels,
            layers_list_encoder=layers_list_encoder,
            attention_list_encoder=attention_list_encoder,
            multipliers_list=multipliers_list,
            freq_downsample_list=freq_downsample_list,
            bottleneck_base_channels=bottleneck_base_channels,
            num_bottleneck_layers=num_bottleneck_layers,
            heads=heads,
            cond_channels=cond_channels,
            bottleneck_channels=bottleneck_channels,
            hop=hop,
            dropout_rate=dropout_rate,
            **common_stage_kwargs,
        )
        self.denoiser = DenoiserUNet(
            data_channels=data_channels,
            base_channels=base_channels,
            layers_list=layers_list,
            multipliers_list=multipliers_list,
            attention_list=attention_list,
            freq_downsample_list=freq_downsample_list,
            cond_channels=cond_channels,
            hop=hop,
            heads=heads,
            frequency_scaling=frequency_scaling,
            dropout_rate=dropout_rate,
            init_as_zero=init_as_zero,
            **common_stage_kwargs,
        )
        if int(quantizer_num_quantizers) > 0:
            self.quantizer = ResidualVectorQuantize(
                input_dim=bottleneck_channels,
                n_codebooks=int(quantizer_num_quantizers),
                codebook_size=int(quantizer_codebook_size),
                codebook_dim=quantizer_codebook_dim,
                quantizer_dropout=float(quantizer_dropout),
                decay=float(quantizer_decay),
                kmeans_init=bool(quantizer_kmeans_init),
                kmeans_iters=int(quantizer_kmeans_iters),
                threshold_ema_dead_code=int(quantizer_threshold_ema_dead_code),
            )
        else:
            self.quantizer = None
        self.emb = (
            GaussianFourierProjection(cond_channels, fourier_scale)
            if use_fourier
            else PositionalEmbedding(cond_channels)
        )
        self.emb_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
        )
        self.task_embedding = nn.Embedding(len(self.task_names), cond_channels)
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=float(task_embedding_init_std))

    @property
    def has_quantizer(self) -> bool:
        return self.quantizer is not None

    def quantize_representation(
        self,
        representation: torch.Tensor,
        detach_encoder: bool = True,
        n_quantizers: int | None = None,
    ) -> QuantizedLatents:
        if self.quantizer is None:
            raise RuntimeError("MosslandCodecSame quantizer is disabled")
        continuous, _hidden = self.encoder(representation, return_hidden=True)
        quantizer_input = continuous.detach() if detach_encoder else continuous
        quantized, codes, commitment_loss = self.quantizer(quantizer_input, n_quantizers=n_quantizers)
        distill_loss = F.mse_loss(quantized.float(), continuous.detach().float())
        return QuantizedLatents(
            continuous=continuous,
            discrete=quantized,
            codes=codes,
            projected_latents=quantized,
            commitment_loss=commitment_loss,
            codebook_loss=continuous.new_zeros(()),
            distill_loss=distill_loss,
        )

    def latent_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.quantizer is None:
            raise RuntimeError("MosslandCodecSame quantizer is disabled")
        quantized, _ = self.quantizer.from_codes(codes)
        return quantized

    @staticmethod
    def _lookup_condition_index(lookup: dict[str, int], value, strict: bool) -> int:
        key = str(value)
        if key in lookup:
            return lookup[key]
        if strict:
            raise KeyError(key)
        return 0

    def _coerce_condition_indices(self, values, indices, batch_size: int, device: torch.device) -> torch.Tensor:
        if indices is not None:
            idx = indices.to(device=device, dtype=torch.long).reshape(-1) if torch.is_tensor(indices) else torch.as_tensor(indices, device=device, dtype=torch.long).reshape(-1)
        elif values is None or isinstance(values, str):
            value = self.task_names[0] if values is None else values
            idx = torch.full((batch_size,), self._lookup_condition_index(self.task_to_idx, value, True), device=device, dtype=torch.long)
        elif torch.is_tensor(values):
            idx = values.to(device=device, dtype=torch.long).reshape(-1)
        else:
            idx = torch.tensor(
                [self._lookup_condition_index(self.task_to_idx, value, False) for value in list(values)],
                device=device,
                dtype=torch.long,
            )
        if idx.numel() == 1:
            return idx.expand(batch_size)
        if idx.numel() == batch_size:
            return idx
        if idx.numel() * 2 == batch_size:
            return torch.cat((idx, idx), dim=0)
        raise ValueError(f"condition batch size mismatch: got {idx.numel()}, expected 1 or {batch_size}")

    def _condition_embedding(self, sigma_embedding: torch.Tensor, task_id="reconstruct", task_idx=None) -> torch.Tensor:
        task_idx = self._coerce_condition_indices(task_id, task_idx, sigma_embedding.shape[0], sigma_embedding.device)
        return self.emb_proj(sigma_embedding + self.task_embedding(task_idx).to(sigma_embedding.dtype))

    def _get_c(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_correct = self.sigma_min
        c_skip = (self.sigma_data**2.0) / (((sigma - sigma_correct) ** 2.0) + (self.sigma_data**2.0))
        c_out = (self.sigma_data * (sigma - sigma_correct)) / (((self.sigma_data**2.0) + (sigma**2.0)) ** 0.5)
        c_in = 1.0 / (((sigma**2.0) + (self.sigma_data**2.0)) ** 0.5)
        return c_skip.reshape(-1, 1, 1, 1), c_out.reshape(-1, 1, 1, 1), c_in.reshape(-1, 1, 1, 1)

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        sigma=None,
        pyramid_latents: Sequence[torch.Tensor] | None = None,
        latent_override: torch.Tensor | None = None,
        task_id="reconstruct",
        task_idx=None,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        latents = latents.to(dtype)
        if sigma is None:
            sigma = self.sigma_max
        sigma = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) * sigma
        time_emb = self._condition_embedding(self.emb((torch.log(sigma) / 4.0).to(dtype)).to(dtype), task_id, task_idx)
        c_skip, c_out, c_in = self._get_c(sigma)
        inp = x
        if latent_override is not None:
            latents = latent_override.to(dtype)
        elif latents.shape == x.shape:
            latents = self.encoder(latents)
        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents, time_emb=time_emb)
        residual = self.denoiser(c_in * x, pyramid_latents, time_emb)
        return c_skip * inp + c_out * residual

    def _downscaling_factor(self) -> int:
        return 2 ** sum(1 for item in self.freq_downsample_list if item == 0)

    def prepare_audio_for_encode(self, audio: torch.Tensor) -> torch.Tensor:
        downscaling_factor = self._downscaling_factor()
        original_length = audio.shape[-1]
        if getattr(self.audio_processor, "center_pad", False):
            frames = max(1, original_length // self.hop)
            target_length = (frames // downscaling_factor) * downscaling_factor * self.hop
        else:
            frame_length = self.audio_processor.fac * self.hop
            frames = max(1, (original_length - frame_length) // self.hop + 1)
            target_length = (frames // downscaling_factor) * downscaling_factor * self.hop
            target_length = target_length + (self.audio_processor.fac - 1) * self.hop
        if original_length > target_length:
            return audio[..., :target_length]
        if original_length < target_length:
            return F.pad(audio, (0, target_length - original_length))
        return audio

    @torch.no_grad()
    def encode(
        self,
        path_or_audio,
        max_waveform_length=None,
        max_batch_size=None,
        extract_features: bool = False,
        rescale: float = 1.0,
        quantize: bool = False,
        return_codes: bool = False,
        n_quantizers: int | None = None,
    ):
        del max_waveform_length, max_batch_size
        self.eval()
        device = next(self.parameters()).device
        if isinstance(path_or_audio, (str, Path)):
            audio, _sr = sf.read(str(path_or_audio), dtype="float32", always_2d=True)
            audio = torch.from_numpy(np.transpose(audio, [1, 0])).to(device)
        else:
            audio = path_or_audio.to(device) if torch.is_tensor(path_or_audio) else torch.from_numpy(path_or_audio).to(device)
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.ndim == 2:
            audio = audio[None, :, :]
        audio = self.prepare_audio_for_encode(audio)
        representation = self.audio_processor.to_representation_encoder(audio)
        if extract_features:
            return self.encoder(representation, extract_features=True)
        if quantize:
            q = self.quantize_representation(representation, detach_encoder=True, n_quantizers=n_quantizers)
            return q.codes if return_codes else q.discrete / rescale
        latent = self.encoder(representation)
        return latent / rescale

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        denoising_steps: int = 1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale: float = 1.0,
        target_length: int | None = None,
        task_id="reconstruct",
        task_idx=None,
    ) -> torch.Tensor:
        del max_waveform_length, max_batch_size
        self.eval()
        device = next(self.parameters()).device
        latent = latent.to(device) if torch.is_tensor(latent) else torch.from_numpy(latent).to(device)
        latent = latent * rescale
        if latent.ndim == 2:
            latent = latent[None, :, :]
        repr_out = self._decode_to_representation(latent, denoising_steps, device, task_id, task_idx)
        audio = self.audio_processor.to_waveform(repr_out, self.hop)
        if target_length is not None:
            if audio.shape[-1] > target_length:
                audio = audio[..., :target_length]
            elif audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        return audio

    def _decode_to_representation(
        self,
        latents: torch.Tensor,
        diffusion_steps: int = 1,
        device=None,
        task_id="reconstruct",
        task_idx=None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        sample_length = int(latents.shape[-1] * self._downscaling_factor())
        x = torch.randn(
            latents.shape[0],
            self.data_channels,
            self.hop * 2,
            sample_length,
            device=device,
            dtype=latents.dtype,
        ) * self.sigma_max
        return self._reverse_diffusion(x, diffusion_steps, latents, task_id=task_id, task_idx=task_idx)

    def _get_sigma(self, i: int, k: int) -> float:
        return float(
            (self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / max(1, k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho)))
            ** self.rho
        )

    def _reverse_diffusion(
        self,
        x: torch.Tensor,
        diffusion_steps: int,
        latents: torch.Tensor,
        task_id="reconstruct",
        task_idx=None,
    ) -> torch.Tensor:
        pyramid = None
        for step in range(int(diffusion_steps), 0, -1):
            sigma = self._get_sigma(step, int(diffusion_steps))
            x = self(latents, x, sigma=sigma, pyramid_latents=pyramid, task_id=task_id, task_idx=task_idx)
            if pyramid is None:
                sigma_tensor = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
                time_emb = self._condition_embedding(self.emb((torch.log(sigma_tensor) / 4.0).to(x.dtype)).to(x.dtype), task_id, task_idx)
                pyramid = self.decoder(latents, time_emb=time_emb)
        return x

    @torch.no_grad()
    def decode_codes(self, codes, denoising_steps: int = 1, target_length: int | None = None, task_id="reconstruct", task_idx=None, **kwargs):
        del kwargs
        codes = codes.to(next(self.parameters()).device) if torch.is_tensor(codes) else torch.from_numpy(codes).to(next(self.parameters()).device)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        return self.decode(self.latent_from_codes(codes), denoising_steps=denoising_steps, target_length=target_length, task_id=task_id, task_idx=task_idx)

    def export_model(self):
        return self


MosslandCodec = MosslandCodecSame
