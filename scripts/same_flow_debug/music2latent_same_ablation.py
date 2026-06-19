from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm

from scripts.music2latent import models as music2latent_models
from scripts.same.autoencoders import TransformerResamplingBlock
from scripts.same.transformer import TransformerBlock


def zero_init(module: nn.Module):
    for param in module.parameters():
        param.detach().zero_()
    return module


def remove_legacy_weight_norm(module: nn.Module):
    try:
        remove_weight_norm(module)
    except ValueError:
        pass
    return module


def configure_resampling_rope(module: TransformerResamplingBlock, *, add_rope: bool):
    if add_rope:
        return module
    for layer in module.transformers:
        layer.rope = None
    return module


def choose_num_heads(channels: int, dim_heads: int) -> int:
    heads = max(1, int(channels) // max(1, int(dim_heads)))
    while heads > 1 and int(channels) % heads != 0:
        heads -= 1
    return heads


class Local2DAttentionConv(nn.Module):
    """2D local attention with a Conv2d-like kernel footprint.

    Each output position attends only to a fixed kxk neighborhood in the input
    feature map. This keeps the locality and variable-length behavior of a
    same-padded Conv2d while making the mixing content-adaptive.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dim_heads: int = 64,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("Local2DAttentionConv requires an odd kernel_size")
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        self.out_channels = int(out_channels)
        self.heads = choose_num_heads(self.out_channels, int(dim_heads))
        self.head_dim = self.out_channels // self.heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, self.kernel_size * self.kernel_size))
        if zero_init_output:
            zero_init(self.to_out)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, freq, frames = x.shape
        return x.reshape(batch, self.heads, self.head_dim, freq, frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.to_q(x))
        k = self._heads(self.to_k(x))
        v = self._heads(self.to_v(x))
        pad = (self.radius, self.radius, self.radius, self.radius)
        k_pad = F.pad(k, pad)
        v_pad = F.pad(v, pad)
        _batch, _heads, _dim, freq, frames = q.shape

        logits = []
        offsets: list[tuple[int, int]] = []
        for df in range(self.kernel_size):
            for dt in range(self.kernel_size):
                k_local = k_pad[..., df:df + freq, dt:dt + frames]
                logits.append((q * k_local).sum(dim=2) * self.scale)
                offsets.append((df, dt))
        attn_logits = torch.stack(logits, dim=2)
        bias = self.relative_bias[None, :, :, None, None].to(dtype=attn_logits.dtype)
        attn = (attn_logits + bias).float().softmax(dim=2).to(dtype=v.dtype)

        out = torch.zeros_like(v)
        for idx, (df, dt) in enumerate(offsets):
            v_local = v_pad[..., df:df + freq, dt:dt + frames]
            out = out + attn[:, :, idx:idx + 1] * v_local
        batch = x.shape[0]
        out = out.reshape(batch, self.out_channels, freq, frames)
        return self.to_out(out)


class LocalFreqAttentionConv(nn.Module):
    """Frequency-only local attention with a Conv2d-like (k, 1) footprint."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        stride: int = 1,
        dim_heads: int = 64,
    ):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("LocalFreqAttentionConv requires an odd kernel_size")
        if int(stride) < 1:
            raise ValueError("LocalFreqAttentionConv requires stride >= 1")
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        self.stride = int(stride)
        self.out_channels = int(out_channels)
        self.heads = choose_num_heads(self.out_channels, int(dim_heads))
        self.head_dim = self.out_channels // self.heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, self.kernel_size))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, freq, frames = x.shape
        return x.reshape(batch, self.heads, self.head_dim, freq, frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = x.shape[-2]
        out_freq = (freq + self.stride - 1) // self.stride
        center_idx = torch.arange(
            out_freq,
            device=x.device,
            dtype=torch.long,
        ) * self.stride

        q_input = x.index_select(dim=2, index=center_idx)
        q = self._heads(self.to_q(q_input))
        k = self._heads(self.to_k(x))
        v = self._heads(self.to_v(x))
        k_pad = F.pad(k, (0, 0, self.radius, self.radius))
        v_pad = F.pad(v, (0, 0, self.radius, self.radius))

        logits = []
        values = []
        for df in range(self.kernel_size):
            k_local = k_pad[..., df:df + freq, :].index_select(dim=3, index=center_idx)
            v_local = v_pad[..., df:df + freq, :].index_select(dim=3, index=center_idx)
            logits.append((q * k_local).sum(dim=2) * self.scale)
            values.append(v_local)
        attn_logits = torch.stack(logits, dim=2)
        bias = self.relative_bias[None, :, :, None, None].to(dtype=attn_logits.dtype)
        attn = (attn_logits + bias).float().softmax(dim=2).to(dtype=v.dtype)

        out = torch.zeros_like(values[0])
        for idx, v_local in enumerate(values):
            out = out + attn[:, :, idx:idx + 1] * v_local
        batch = x.shape[0]
        out = out.reshape(batch, self.out_channels, out_freq, x.shape[-1])
        return self.to_out(out)


class SAMELocal2DResBlockAdapter(nn.Module):
    """Conv2d-shaped local-attention replacement for Music2Latent ResBlock."""

    def __init__(
        self,
        reference: music2latent_models.ResBlock,
        *,
        kernel_size: int = 3,
        dim_heads: int = 64,
        init_as_zero: bool = True,
    ):
        super().__init__()
        if not bool(getattr(reference, "use_2d", False)):
            raise ValueError("SAMELocal2DResBlockAdapter currently expects use_2d=True")
        conv1 = reference.conv1
        self.in_channels = int(conv1.in_channels)
        self.out_channels = int(conv1.out_channels)
        self.normalize = bool(getattr(reference, "normalize", True))
        self.normalize_residual = bool(getattr(reference, "normalize_residual", False))
        self.downsample = bool(getattr(reference, "downsample", False))
        self.upsample = bool(getattr(reference, "upsample", False))
        self.dropout_rate = float(getattr(reference, "dropout_rate", 0.0))
        self.min_res_dropout = int(getattr(reference, "min_res_dropout", 16))
        if self.normalize:
            self.norm1 = nn.GroupNorm(min(self.in_channels // 4, 32), self.in_channels)
            self.norm2 = nn.GroupNorm(min(self.out_channels // 4, 32), self.out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2) if getattr(reference, "leaky", False) else nn.SiLU()
        cond_channels = int(reference.proj_emb.in_features) if hasattr(reference, "proj_emb") else None
        self.proj_emb = zero_init(nn.Linear(cond_channels, self.out_channels)) if cond_channels is not None else None
        self.local1 = Local2DAttentionConv(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            dim_heads=dim_heads,
            zero_init_output=False,
        )
        self.local2 = Local2DAttentionConv(
            self.out_channels,
            self.out_channels,
            kernel_size=kernel_size,
            dim_heads=dim_heads,
            zero_init_output=bool(init_as_zero),
        )
        self.res_conv = (
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.attention = bool(getattr(reference, "attention", False))
        self.att = reference.att if self.attention else None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None):
        if not self.normalize_residual:
            residual = x.clone()
        if self.normalize:
            x = self.norm1(x)
        if self.normalize_residual:
            residual = x.clone()
        x = self.activation(x)
        if self.downsample:
            x = music2latent_models.downsample_2d(x)
            residual = music2latent_models.downsample_2d(residual)
        if self.upsample:
            x = music2latent_models.upsample_2d(x)
            residual = music2latent_models.upsample_2d(residual)
        x = self.local1(x)
        if self.proj_emb is not None and time_emb is not None:
            x = x + self.proj_emb(time_emb)[:, :, None, None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1] <= self.min_res_dropout:
            x = self.dropout(x)
        x = self.local2(x)
        x = x + self.res_conv(residual)
        if self.attention and self.att is not None:
            x = self.att(x)
        return x.to(dtype=residual.dtype).contiguous()


class SAMEResBlockAdapter(nn.Module):
    """Drop-in debug replacement for Music2Latent denoising ResBlock.

    The adapter intentionally preserves Music2Latent's zero-initialized residual
    contract: at initialization it behaves like an identity mapping, but the
    residual branch is implemented with `scripts.same.transformer.TransformerBlock`.
    """

    def __init__(
        self,
        reference: music2latent_models.ResBlock,
        *,
        transformer_depth: int = 1,
        dim_heads: int = 64,
        sliding_window: tuple[int, int] | list[int] | None = (8, 8),
        differential: bool = True,
        dyt: bool = True,
        ff_mult: int = 2,
        init_as_zero: bool = True,
        use_freq_axis: bool = False,
    ):
        super().__init__()
        conv1 = reference.conv1
        self.in_channels = int(conv1.in_channels)
        self.out_channels = int(conv1.out_channels)
        self.use_2d = bool(getattr(reference, "use_2d", False))
        self.normalize = bool(getattr(reference, "normalize", True))
        self.sliding_window = sliding_window
        self.use_freq_axis = bool(use_freq_axis and self.use_2d)
        self.in_proj = nn.Linear(self.in_channels, self.out_channels)
        self.res_proj = (
            nn.Linear(self.in_channels, self.out_channels)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        if self.normalize:
            self.norm = nn.LayerNorm(self.out_channels)
        self.activation = nn.SiLU()
        cond_channels = (
            int(reference.proj_emb.in_features)
            if hasattr(reference, "proj_emb")
            else None
        )
        self.cond_proj = (
            zero_init(nn.Linear(cond_channels, self.out_channels))
            if cond_channels is not None
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.out_channels,
                    dim_heads=min(int(dim_heads), self.out_channels),
                    zero_init_branch_outputs=bool(init_as_zero),
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": bool(differential),
                    },
                    ff_kwargs={"mult": int(ff_mult)},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(transformer_depth))
            ]
        )
        self.freq_layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.out_channels,
                    dim_heads=min(int(dim_heads), self.out_channels),
                    zero_init_branch_outputs=bool(init_as_zero),
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": bool(differential),
                    },
                    ff_kwargs={"mult": int(ff_mult)},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(transformer_depth))
            ]
        ) if self.use_freq_axis else nn.ModuleList()
        self.out_proj = nn.Linear(self.out_channels, self.out_channels)
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
            return tokens.reshape(batch, freq, frames, channels).permute(0, 3, 1, 2).contiguous()
        return tokens.transpose(1, 2).contiguous()

    def _window_for(self, tokens: torch.Tensor):
        if self.sliding_window is None:
            return None
        left, right = self.sliding_window
        seq_len = tokens.shape[1]
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
        window = self._window_for(tokens)
        for layer in self.layers:
            tokens = layer(tokens, self_attention_flash_sliding_window=window)
        if self.use_freq_axis:
            batch, freq = shape_info
            frames, channels = tokens.shape[1], tokens.shape[2]
            freq_tokens = (
                tokens.reshape(batch, freq, frames, channels)
                .permute(0, 2, 1, 3)
                .reshape(batch * frames, freq, channels)
            )
            freq_window = self._window_for(freq_tokens)
            for layer in self.freq_layers:
                freq_tokens = layer(
                    freq_tokens,
                    self_attention_flash_sliding_window=freq_window,
                )
            tokens = (
                freq_tokens.reshape(batch, frames, freq, channels)
                .permute(0, 2, 1, 3)
                .reshape(batch * freq, frames, channels)
            )
        tokens = residual + self.out_proj(tokens)
        return self._from_tokens(tokens, shape_info).to(dtype=x.dtype).contiguous()


class SAMEAttentionAdapter(nn.Module):
    """Drop-in replacement for Music2Latent Attention using SAME TransformerBlock."""

    def __init__(
        self,
        reference: music2latent_models.Attention,
        *,
        transformer_depth: int = 1,
        dim_heads: int = 64,
        sliding_window: tuple[int, int] | list[int] | None = (8, 8),
        differential: bool = True,
        dyt: bool = True,
        ff_mult: int = 2,
        init_as_zero: bool = True,
    ):
        super().__init__()
        self.use_2d = bool(getattr(reference, "use_2d", False))
        self.normalize = bool(getattr(reference, "normalize", True))
        self.sliding_window = sliding_window
        if self.normalize:
            channels = int(reference.norm.num_channels)
            self.norm = nn.LayerNorm(channels)
        else:
            channels = int(reference.mha.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    dim_heads=min(int(dim_heads), channels),
                    zero_init_branch_outputs=bool(init_as_zero),
                    norm_type="dyt" if dyt else "rms_norm",
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": bool(differential),
                    },
                    ff_kwargs={"mult": int(ff_mult)},
                    norm_kwargs={"eps": 1e-3},
                )
                for _ in range(int(transformer_depth))
            ]
        )

    def _to_tokens(self, x: torch.Tensor):
        if self.use_2d:
            batch, channels, freq, frames = x.shape
            tokens = x.permute(0, 3, 2, 1).reshape(batch * frames, freq, channels)
            return tokens, (batch, frames)
        return x.transpose(1, 2), None

    def _from_tokens(self, tokens: torch.Tensor, shape_info):
        if self.use_2d:
            batch, frames = shape_info
            freq, channels = tokens.shape[1], tokens.shape[2]
            return tokens.reshape(batch, frames, freq, channels).permute(0, 3, 2, 1).contiguous()
        return tokens.transpose(1, 2).contiguous()

    def _window_for(self, tokens: torch.Tensor):
        if self.sliding_window is None:
            return None
        left, right = self.sliding_window
        seq_len = tokens.shape[1]
        return [
            min(int(left), max(seq_len - 1, 0)),
            min(int(right), max(seq_len - 1, 0)),
        ]

    def forward(self, x: torch.Tensor):
        residual = x
        tokens, shape_info = self._to_tokens(x)
        if self.normalize:
            tokens = self.norm(tokens)
        tokens_in = tokens
        window = self._window_for(tokens)
        for layer in self.layers:
            tokens = layer(tokens, self_attention_flash_sliding_window=window)
        branch = self._from_tokens(tokens - tokens_in, shape_info)
        return (branch + residual).to(dtype=x.dtype).contiguous()


class SAMEFreqDownsampleAdapter(nn.Module):
    """Replace frequency-only downsampling with SAME token resampling."""

    def __init__(
        self,
        reference: music2latent_models.DownsampleFreqConv,
        *,
        transformer_depth: int = 1,
        dim_heads: int = 64,
        sliding_window: tuple[int, int] | list[int] | None = (8, 8),
        differential: bool = False,
        dyt: bool = True,
        ff_mult: int = 1,
        chunk_size: int = 128,
        add_rope: bool = True,
    ):
        super().__init__()
        self.normalize = bool(getattr(reference, "normalize", False))
        in_channels = int(reference.c.in_channels)
        out_channels = int(reference.c.out_channels)
        if self.normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.resampling = TransformerResamplingBlock(
            in_channels,
            out_channels,
            stride=4,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            type="encoder",
            transformer_depth=int(transformer_depth),
            dim_heads=min(int(dim_heads), out_channels),
            differential=bool(differential),
            dyt=bool(dyt),
            ff_mult=int(ff_mult),
            conv_mapping=True,
        )
        remove_legacy_weight_norm(self.resampling.mapping)
        configure_resampling_rope(self.resampling, add_rope=bool(add_rope))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        if self.normalize:
            x = self.norm(x)
        batch, channels, freq, frames = x.shape
        tokens = x.permute(0, 3, 1, 2).reshape(batch * frames, channels, freq)
        tokens = self.resampling(tokens)
        out_channels, out_freq = tokens.shape[1], tokens.shape[2]
        x = tokens.reshape(batch, frames, out_channels, out_freq).permute(0, 2, 3, 1)
        return x.to(dtype=dtype).contiguous()


class SAMEFreqUpsampleAdapter(nn.Module):
    """Replace frequency-only upsampling with SAME token resampling."""

    def __init__(
        self,
        reference: music2latent_models.UpsampleFreqConv,
        *,
        transformer_depth: int = 1,
        dim_heads: int = 64,
        sliding_window: tuple[int, int] | list[int] | None = (8, 8),
        differential: bool = False,
        dyt: bool = True,
        ff_mult: int = 1,
        chunk_size: int = 128,
        add_rope: bool = True,
    ):
        super().__init__()
        self.normalize = bool(getattr(reference, "normalize", False))
        in_channels = int(reference.c.in_channels)
        out_channels = int(reference.c.out_channels)
        if self.normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.resampling = TransformerResamplingBlock(
            in_channels,
            out_channels,
            stride=4,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            type="decoder",
            transformer_depth=int(transformer_depth),
            dim_heads=min(int(dim_heads), in_channels),
            differential=bool(differential),
            dyt=bool(dyt),
            ff_mult=int(ff_mult),
            conv_mapping=True,
        )
        remove_legacy_weight_norm(self.resampling.mapping)
        configure_resampling_rope(self.resampling, add_rope=bool(add_rope))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        if self.normalize:
            x = self.norm(x)
        batch, channels, freq, frames = x.shape
        tokens = x.permute(0, 3, 1, 2).reshape(batch * frames, channels, freq)
        tokens = self.resampling(tokens)
        out_channels, out_freq = tokens.shape[1], tokens.shape[2]
        x = tokens.reshape(batch, frames, out_channels, out_freq).permute(0, 2, 3, 1)
        return x.to(dtype=dtype).contiguous()


class SAMELocalFreqDownsampleAdapter(nn.Module):
    """Conv-shaped local attention replacement for frequency downsampling."""

    def __init__(
        self,
        reference: music2latent_models.DownsampleFreqConv,
        *,
        dim_heads: int = 64,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.normalize = bool(getattr(reference, "normalize", False))
        in_channels = int(reference.c.in_channels)
        out_channels = int(reference.c.out_channels)
        if self.normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.local = LocalFreqAttentionConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=4,
            dim_heads=dim_heads,
        )

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        if self.normalize:
            x = self.norm(x)
        return self.local(x).to(dtype=dtype).contiguous()


class SAMELocalFreqUpsampleAdapter(nn.Module):
    """Conv-shaped local attention replacement for frequency upsampling."""

    def __init__(
        self,
        reference: music2latent_models.UpsampleFreqConv,
        *,
        dim_heads: int = 64,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.normalize = bool(getattr(reference, "normalize", False))
        in_channels = int(reference.c.in_channels)
        out_channels = int(reference.c.out_channels)
        if self.normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.local = LocalFreqAttentionConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dim_heads=dim_heads,
        )

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        if self.normalize:
            x = self.norm(x)
        x = F.interpolate(x, scale_factor=(4, 1), mode="nearest")
        return self.local(x).to(dtype=dtype).contiguous()


def _replace_resblocks(
    layers: Iterable[nn.Module],
    *,
    transformer_depth: int,
    dim_heads: int,
    sliding_window: tuple[int, int] | list[int] | None,
    differential: bool,
    dyt: bool,
    ff_mult: int,
    init_as_zero: bool,
    use_freq_axis: bool = False,
    min_channels: int = 0,
    adapter_kind: str = "time",
    local2d_kernel_size: int = 3,
) -> int:
    replaced = 0
    for parent in layers:
        for name, child in list(parent.named_children()):
            if isinstance(child, music2latent_models.ResBlock):
                out_channels = int(child.conv1.out_channels)
                if out_channels < int(min_channels):
                    continue
                if adapter_kind == "local2d":
                    adapter = SAMELocal2DResBlockAdapter(
                        child,
                        kernel_size=local2d_kernel_size,
                        dim_heads=dim_heads,
                        init_as_zero=init_as_zero,
                    )
                else:
                    adapter = SAMEResBlockAdapter(
                        child,
                        transformer_depth=transformer_depth,
                        dim_heads=dim_heads,
                        sliding_window=sliding_window,
                        differential=differential,
                        dyt=dyt,
                        ff_mult=ff_mult,
                        init_as_zero=init_as_zero,
                        use_freq_axis=use_freq_axis,
                    )
                setattr(
                    parent,
                    name,
                    adapter,
                )
                replaced += 1
    return replaced


def _replace_freq_resampling(
    layers: Iterable[nn.Module],
    *,
    transformer_depth: int,
    dim_heads: int,
    sliding_window: tuple[int, int] | list[int] | None,
    differential: bool,
    dyt: bool,
    ff_mult: int,
    replace_downsample: bool = True,
    replace_upsample: bool = True,
    add_rope: bool = True,
    adapter_kind: str = "local",
    local_kernel_size: int = 5,
) -> int:
    replaced = 0
    for parent in layers:
        for name, child in list(parent.named_children()):
            if replace_downsample and isinstance(child, music2latent_models.DownsampleFreqConv):
                if adapter_kind == "local":
                    adapter = SAMELocalFreqDownsampleAdapter(
                        child,
                        dim_heads=dim_heads,
                        kernel_size=local_kernel_size,
                    )
                elif adapter_kind == "transformer":
                    adapter = SAMEFreqDownsampleAdapter(
                        child,
                        transformer_depth=transformer_depth,
                        dim_heads=dim_heads,
                        sliding_window=sliding_window,
                        differential=differential,
                        dyt=dyt,
                        ff_mult=ff_mult,
                        add_rope=add_rope,
                    )
                else:
                    raise ValueError(f"Unknown freq resampling adapter_kind={adapter_kind!r}")
                setattr(
                    parent,
                    name,
                    adapter,
                )
                replaced += 1
            elif replace_upsample and isinstance(child, music2latent_models.UpsampleFreqConv):
                if adapter_kind == "local":
                    adapter = SAMELocalFreqUpsampleAdapter(
                        child,
                        dim_heads=dim_heads,
                        kernel_size=local_kernel_size,
                    )
                elif adapter_kind == "transformer":
                    adapter = SAMEFreqUpsampleAdapter(
                        child,
                        transformer_depth=transformer_depth,
                        dim_heads=dim_heads,
                        sliding_window=sliding_window,
                        differential=differential,
                        dyt=dyt,
                        ff_mult=ff_mult,
                        add_rope=add_rope,
                    )
                else:
                    raise ValueError(f"Unknown freq resampling adapter_kind={adapter_kind!r}")
                setattr(
                    parent,
                    name,
                    adapter,
                )
                replaced += 1
    return replaced


def _replace_attention(
    layers: Iterable[nn.Module],
    *,
    transformer_depth: int,
    dim_heads: int,
    sliding_window: tuple[int, int] | list[int] | None,
    differential: bool,
    dyt: bool,
    ff_mult: int,
    init_as_zero: bool,
) -> int:
    replaced = 0
    for parent in layers:
        for child in parent.modules():
            if isinstance(child, music2latent_models.ResBlock) and getattr(
                child,
                "attention",
                False,
            ):
                child.att = SAMEAttentionAdapter(
                    child.att,
                    transformer_depth=transformer_depth,
                    dim_heads=dim_heads,
                    sliding_window=sliding_window,
                    differential=differential,
                    dyt=dyt,
                    ff_mult=ff_mult,
                    init_as_zero=init_as_zero,
                )
                replaced += 1
    return replaced


class Music2LatentSAMEAblation(music2latent_models.Music2Latent):
    """Music2Latent baseline with controlled SAME module replacements.

    `ablation_variant="official"` is a normal Music2Latent model through the
    same Hydra target. `ablation_variant="denoiser_attention_same_freq"` keeps
    Music2Latent ResBlocks but replaces their attention submodules.
    `ablation_variant="denoiser_freq_resampling_same"` only replaces
    frequency-only denoising resampling layers. These variants are diagnostic:
    `same_freq_resampling_kind="transformer"` keeps the older
    TransformerResamplingBlock path, while `"local"` uses a (5, 1) local
    frequency attention footprint. The `_down_only` and `_up_only` variants
    split that replacement for memory/debug isolation.
    `ablation_variant="denoiser_attention_freq_resampling_same"` accumulates
    the attention and frequency-resampling replacements while keeping the
    Music2Latent Conv ResBlocks and output head.
    `ablation_variant="denoiser_resblock_same_time"` replaces the whole
    denoising U-Net ResBlock.
    `ablation_variant="denoiser_resblock_same_local2d"` replaces ResBlocks with
    Conv2d-shaped 2D local attention over kxk neighborhoods.
    `ablation_variant="denoiser_resblock_freq_resampling_same_time"` accumulates
    SAME frequency resampling with the local time-axis ResBlock replacement.
    `ablation_variant="denoiser_resblock_freq_resampling_same_local2d"`
    accumulates SAME frequency resampling with the 2D local-attention ResBlock.
    `ablation_variant="denoiser_resblock_same_time_freq"` replaces the whole
    denoising U-Net ResBlock with axial local time + local frequency SAME
    blocks, preserving local attention on both axes.
    """

    def __init__(
        self,
        *args,
        ablation_variant: str = "official",
        same_transformer_depth: int = 1,
        same_dim_heads: int = 64,
        same_sliding_window: tuple[int, int] | list[int] | None = (8, 8),
        same_differential: bool = True,
        same_dyt: bool = True,
        same_ff_mult: int = 2,
        same_init_as_zero: bool = True,
        same_resampling_add_rope: bool = True,
        same_resblock_min_channels: int = 0,
        same_local2d_kernel_size: int = 3,
        same_freq_resampling_kind: str = "transformer",
        same_freq_kernel_size: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ablation_variant = str(ablation_variant)
        self.same_replaced_modules = 0
        freq_adapter_kind = str(same_freq_resampling_kind)
        if self.ablation_variant.endswith("_transformer"):
            freq_adapter_kind = "transformer"
            self.ablation_variant = self.ablation_variant[: -len("_transformer")]
        if freq_adapter_kind not in {"local", "transformer"}:
            raise ValueError(
                "same_freq_resampling_kind must be 'local' or 'transformer', "
                f"got {same_freq_resampling_kind!r}"
            )
        if (
            self.ablation_variant
            in {
                "denoiser_resblock_same_time",
                "denoiser_resblock_freq_resampling_same_time",
                "denoiser_resblock_same_time_freq",
            }
            and same_sliding_window is None
        ):
            raise ValueError(
                f"{self.ablation_variant} replaces time-axis ResBlocks and "
                "must use local sliding-window attention; same_sliding_window=None "
                "would make time attention full-context."
            )
        if self.ablation_variant == "official":
            return
        if self.ablation_variant == "denoiser_attention_same_freq":
            self.same_replaced_modules += _replace_attention(
                [self.down_layers, self.up_layers],
                transformer_depth=same_transformer_depth,
                dim_heads=same_dim_heads,
                sliding_window=same_sliding_window,
                differential=same_differential,
                dyt=same_dyt,
                ff_mult=same_ff_mult,
                init_as_zero=same_init_as_zero,
            )
            return
        if self.ablation_variant in {
            "denoiser_freq_resampling_same",
            "denoiser_freq_resampling_same_down_only",
            "denoiser_freq_resampling_same_up_only",
            "denoiser_attention_freq_resampling_same",
            "denoiser_attention_freq_resampling_same_down_only",
            "denoiser_attention_freq_resampling_same_up_only",
        }:
            if self.ablation_variant.startswith("denoiser_attention_"):
                self.same_replaced_modules += _replace_attention(
                    [self.down_layers, self.up_layers],
                    transformer_depth=same_transformer_depth,
                    dim_heads=same_dim_heads,
                    sliding_window=same_sliding_window,
                    differential=same_differential,
                    dyt=same_dyt,
                    ff_mult=same_ff_mult,
                    init_as_zero=same_init_as_zero,
                )
            self.same_replaced_modules += _replace_freq_resampling(
                [self.down_layers, self.up_layers],
                transformer_depth=same_transformer_depth,
                dim_heads=same_dim_heads,
                sliding_window=same_sliding_window,
                differential=same_differential,
                dyt=same_dyt,
                ff_mult=same_ff_mult,
                replace_downsample=not self.ablation_variant.endswith("_up_only"),
                replace_upsample=not self.ablation_variant.endswith("_down_only"),
                add_rope=same_resampling_add_rope,
                adapter_kind=freq_adapter_kind,
                local_kernel_size=same_freq_kernel_size,
            )
            return
        if self.ablation_variant not in {
            "denoiser_resblock_same_time",
            "denoiser_resblock_freq_resampling_same_time",
            "denoiser_resblock_same_local2d",
            "denoiser_resblock_freq_resampling_same_local2d",
            "denoiser_resblock_same_time_freq",
        }:
            raise ValueError(f"Unknown ablation_variant={self.ablation_variant!r}")
        if self.ablation_variant in {
            "denoiser_resblock_freq_resampling_same_time",
            "denoiser_resblock_freq_resampling_same_local2d",
        }:
            self.same_replaced_modules += _replace_freq_resampling(
                [self.down_layers, self.up_layers],
                transformer_depth=same_transformer_depth,
                dim_heads=same_dim_heads,
                sliding_window=same_sliding_window,
                differential=same_differential,
                dyt=same_dyt,
                ff_mult=same_ff_mult,
                add_rope=same_resampling_add_rope,
                adapter_kind=freq_adapter_kind,
                local_kernel_size=same_freq_kernel_size,
            )
        self.same_replaced_modules += _replace_resblocks(
            [self.down_layers, self.up_layers],
            transformer_depth=same_transformer_depth,
            dim_heads=same_dim_heads,
            sliding_window=same_sliding_window,
            differential=same_differential,
            dyt=same_dyt,
            ff_mult=same_ff_mult,
            init_as_zero=same_init_as_zero,
            use_freq_axis=self.ablation_variant.endswith("_time_freq"),
            min_channels=same_resblock_min_channels,
            adapter_kind="local2d" if self.ablation_variant.endswith("_local2d") else "time",
            local2d_kernel_size=same_local2d_kernel_size,
        )
