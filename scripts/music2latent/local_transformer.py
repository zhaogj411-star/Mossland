from functools import reduce
from typing import Callable, Literal

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F


def zero_init(module):
    for param in module.parameters():
        param.detach().zero_()
    return module


def _sliding_window_chunked_sdpa(q, k, v, w_left, w_right, chunk_size=1024):
    outputs = []
    seq_len = q.shape[-2]
    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        k_start = max(0, q_start - int(w_left))
        k_end = min(seq_len, q_end + int(w_right))
        q_chunk = q[..., q_start:q_end, :]
        k_chunk = k[..., k_start:k_end, :]
        v_chunk = v[..., k_start:k_end, :]
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        delta = k_idx[None, :] - q_idx[:, None]
        keep = (delta >= -int(w_left)) & (delta <= int(w_right))
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device)
        mask = mask.masked_fill(~keep, float("-inf"))
        outputs.append(
            F.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=mask,
                is_causal=False,
            )
        )
    return torch.cat(outputs, dim=-2)


def _valid_dim_heads(dim, requested):
    requested = max(1, min(int(requested), int(dim)))
    for dim_heads in range(requested, 0, -1):
        if int(dim) % dim_heads == 0:
            return dim_heads
    return 1


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @autocast("cuda", enabled=False)
    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.float())
        return torch.cat((freqs, freqs), dim=-1), 1.0


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t.to(out_dtype), t_unrotated.to(out_dtype)), dim=-1)


class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=self.eps)


class LayerScale(nn.Module):
    def __init__(self, dim, init_val=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))

    def forward(self, x):
        return x * self.scale


class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv=False,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.act = activation
        if use_conv:
            self.proj = nn.Conv1d(
                dim_in,
                dim_out * 2,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
            )
        else:
            self.proj = nn.Linear(dim_in, dim_out * 2)
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, "b n d -> b d n")
            x = self.proj(x)
            x = rearrange(x, "b d n -> b n d")
        else:
            x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(torch.pi * x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        no_bias=False,
        glu=True,
        use_conv=False,
        conv_kernel_size=3,
        zero_init_output=True,
        sinusoidal=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        activation = nn.SiLU() if not sinusoidal else Sin()
        if glu:
            linear_in = GLU(dim, inner_dim, activation, use_conv, conv_kernel_size)
        else:
            linear_in = nn.Sequential(
                Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias=not no_bias)
                if not use_conv
                else nn.Conv1d(
                    dim,
                    inner_dim,
                    conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    bias=not no_bias,
                ),
                Rearrange("b d n -> b n d") if use_conv else nn.Identity(),
                activation,
            )
        linear_out = (
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
            if not use_conv
            else nn.Conv1d(
                inner_dim,
                dim_out,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                bias=not no_bias,
            )
        )
        if zero_init_output:
            zero_init(linear_out)
        self.ff = nn.Sequential(
            linear_in,
            Rearrange("b d n -> b n d") if use_conv else nn.Identity(),
            linear_out,
            Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        dim_context=None,
        causal=False,
        zero_init_output=True,
        qk_norm_eps=1e-6,
        qk_norm: Literal["l2", "ln", "rms", "dyt", "none"] = "none",
        differential=False,
        feat_scale=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = _valid_dim_heads(dim, dim_heads)
        self.differential = differential
        dim_kv = dim if dim_context is None else dim_context
        self.num_heads = max(1, dim // self.dim_heads)
        self.kv_heads = max(1, dim_kv // self.dim_heads)
        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim * (2 if differential else 1), bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * (3 if differential else 2), bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * (5 if differential else 3), bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            zero_init(self.to_out)
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(self.dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(self.dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif qk_norm == "rms":
            self.q_norm = RMSNorm(self.dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.dim_heads, eps=qk_norm_eps)
        elif qk_norm == "dyt":
            self.q_norm = DynamicTanh(self.dim_heads)
            self.k_norm = DynamicTanh(self.dim_heads)
        self.feat_scale = feat_scale
        if feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))
        self.causal = causal

    def _apply_attn(self, q, k, v, causal=None, sliding_window=None):
        if self.num_heads != self.kv_heads:
            repeat = self.num_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if sliding_window is not None:
            return _sliding_window_chunked_sdpa(q, k, v, sliding_window[0], sliding_window[1])
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal if causal is None else causal,
        )

    def forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flash_attn_sliding_window=None,
        **kwargs,
    ):
        h, kv_h = self.num_heads, self.kv_heads
        kv_input = x if context is None else context
        if hasattr(self, "to_q"):
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, q_diff))
                q = torch.stack([q, q_diff], dim=1)
                k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
                k, k_diff, v = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=kv_h),
                    (k, k_diff, v),
                )
                k = torch.stack([k, k_diff], dim=1)
            else:
                q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=kv_h),
                    (k, v),
                )
        else:
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                    (q, k, v, q_diff, k_diff),
                )
                q = torch.stack([q, q_diff], dim=1)
                k = torch.stack([k, k_diff], dim=1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)
        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            k_freqs = freqs if rotary_pos_emb_k is None else rotary_pos_emb_k[0]
            q = apply_rotary_pos_emb(q.float(), freqs.float()).to(v.dtype)
            k = apply_rotary_pos_emb(k.float(), k_freqs.float()).to(v.dtype)
        if self.differential:
            q, q_diff = q.unbind(dim=1)
            k, k_diff = k.unbind(dim=1)
            out = self._apply_attn(q, k, v, causal=causal, sliding_window=flash_attn_sliding_window)
            out = out - self._apply_attn(
                q_diff,
                k_diff,
                v,
                causal=causal,
                sliding_window=flash_attn_sliding_window,
            )
        else:
            out = self._apply_attn(q, k, v, causal=causal, sliding_window=flash_attn_sliding_window)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if self.feat_scale:
            out_dc = out.mean(dim=-2, keepdim=True)
            out = out + self.lambda_dc * out_dc + self.lambda_hf * (out - out_dc)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        cross_attend=False,
        dim_context=None,
        causal=False,
        zero_init_branch_outputs=True,
        conformer=False,
        layer_ix=-1,
        add_rope=False,
        layer_scale=False,
        norm_type="layer_norm",
        attn_kwargs=None,
        ff_kwargs=None,
        norm_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        del conformer, layer_ix, kwargs
        attn_kwargs = attn_kwargs or {}
        ff_kwargs = ff_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        norm_map = {
            "layer_norm": nn.LayerNorm,
            "rms_norm": RMSNorm,
            "dyt": DynamicTanh,
        }
        norm_layer = norm_map[norm_type]
        head_dim = _valid_dim_heads(dim, dim_heads)
        self.pre_norm = norm_layer(dim, **norm_kwargs)
        self.self_attn = Attention(
            dim,
            dim_heads=head_dim,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads=head_dim,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.rope = RotaryEmbedding(max(head_dim // 2, 1)) if add_rope else None

    def forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        cross_attn_rotary_pos_emb=None,
        self_attention_flash_sliding_window=None,
        cross_attention_flash_sliding_window=None,
        **kwargs,
    ):
        if rotary_pos_emb is None and self.rope is not None:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])
        x = x + self.self_attn_scale(
            self.self_attn(
                self.pre_norm(x),
                rotary_pos_emb=rotary_pos_emb,
                flash_attn_sliding_window=self_attention_flash_sliding_window,
            )
        )
        if context is not None and self.cross_attend:
            x = x + self.cross_attn_scale(
                self.cross_attn(
                    self.cross_attend_norm(x),
                    context=context,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_emb_k=cross_attn_rotary_pos_emb,
                    flash_attn_sliding_window=cross_attention_flash_sliding_window,
                )
            )
        x = x + self.ff_scale(self.ff(self.ff_norm(x)))
        return x


def _bounded_window(window, seq_len):
    if window is None:
        return None
    left, right = window
    return [min(int(left), max(seq_len - 1, 0)), min(int(right), max(seq_len - 1, 0))]


class AxialLocalAttention2d(nn.Module):
    """Conv2d-shaped axial local Transformer over `[B, C, F, T]` features."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        *,
        transformer_depth=1,
        dim_heads=64,
        time_window=(1, 1),
        freq_window=(1, 1),
        use_freq_axis=True,
        differential=False,
        dyt=True,
        ff_mult=1,
        zero_output=False,
        add_rope=True,
    ):
        super().__init__()
        out_channels = int(in_channels if out_channels is None else out_channels)
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.time_window = time_window
        self.freq_window = freq_window
        self.use_freq_axis = bool(use_freq_axis)
        self.input_proj = nn.Linear(self.in_channels, self.out_channels)
        block_kwargs = dict(
            dim_heads=_valid_dim_heads(self.out_channels, dim_heads),
            zero_init_branch_outputs=True,
            norm_type="dyt" if dyt else "rms_norm",
            add_rope=bool(add_rope),
            attn_kwargs={
                "qk_norm": "dyt" if dyt else "rms",
                "qk_norm_eps": 1e-3,
                "differential": bool(differential),
            },
            ff_kwargs={"mult": int(ff_mult), "no_bias": False},
            norm_kwargs={"eps": 1e-3},
        )
        self.time_layers = nn.ModuleList(
            [TransformerBlock(self.out_channels, **block_kwargs) for _ in range(int(transformer_depth))]
        )
        self.freq_layers = nn.ModuleList(
            [TransformerBlock(self.out_channels, **block_kwargs) for _ in range(int(transformer_depth))]
        )
        self.output_proj = nn.Linear(self.out_channels, self.out_channels)
        if zero_output:
            zero_init(self.output_proj)

    def _apply_time(self, x):
        batch, channels, freq, frames = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch * freq, frames, channels)
        window = _bounded_window(self.time_window, frames)
        for layer in self.time_layers:
            tokens = layer(tokens, self_attention_flash_sliding_window=window)
        return tokens.reshape(batch, freq, frames, channels).permute(0, 3, 1, 2)

    def _apply_freq(self, x):
        batch, channels, freq, frames = x.shape
        tokens = x.permute(0, 3, 2, 1).reshape(batch * frames, freq, channels)
        window = _bounded_window(self.freq_window, freq)
        for layer in self.freq_layers:
            tokens = layer(tokens, self_attention_flash_sliding_window=window)
        return tokens.reshape(batch, frames, freq, channels).permute(0, 3, 2, 1)

    def forward(self, x):
        dtype = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = self.input_proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self._apply_time(x)
        if self.use_freq_axis:
            x = self._apply_freq(x)
        x = x.permute(0, 2, 3, 1)
        x = self.output_proj(x)
        return x.permute(0, 3, 1, 2).to(dtype=dtype).contiguous()


class AxialResidualAttention2d(nn.Module):
    def __init__(
        self,
        channels,
        *,
        normalize=True,
        transformer_depth=1,
        dim_heads=64,
        time_window=(1, 1),
        freq_window=(1, 1),
        use_freq_axis=True,
        differential=False,
        dyt=True,
        ff_mult=1,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        if self.normalize:
            self.norm = nn.GroupNorm(min(int(channels) // 4, 32), int(channels))
        self.branch = AxialLocalAttention2d(
            channels,
            channels,
            transformer_depth=transformer_depth,
            dim_heads=dim_heads,
            time_window=time_window,
            freq_window=freq_window,
            use_freq_axis=use_freq_axis,
            differential=differential,
            dyt=dyt,
            ff_mult=ff_mult,
            zero_output=True,
        )

    def forward(self, x):
        y = self.norm(x) if self.normalize else x
        return x + self.branch(y)


class AxialLocalAttentionResBlock(nn.Module):
    """Music2Latent ResBlock adapter using axial local SAME-style attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_channels=None,
        *,
        downsample=False,
        upsample=False,
        normalize=True,
        leaky=False,
        attention=False,
        normalize_residual=False,
        dropout_rate=0.0,
        min_res_dropout=16,
        transformer_depth=1,
        dim_heads=64,
        time_window=(1, 1),
        freq_window=(1, 1),
        use_freq_axis=True,
        differential=False,
        dyt=True,
        ff_mult=1,
        init_as_zero=True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.attention = bool(attention)
        self.upsample = bool(upsample)
        self.downsample = bool(downsample)
        self.normalize_residual = bool(normalize_residual)
        self.dropout_rate = float(dropout_rate)
        self.min_res_dropout = int(min_res_dropout)
        self.use_2d = True
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.normalize:
            self.norm1 = nn.GroupNorm(min(self.in_channels // 4, 32), self.in_channels)
            self.norm2 = nn.GroupNorm(min(self.out_channels // 4, 32), self.out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2) if leaky else nn.SiLU()
        self.proj_emb = zero_init(nn.Linear(int(cond_channels), self.out_channels)) if cond_channels else None
        common = dict(
            transformer_depth=transformer_depth,
            dim_heads=dim_heads,
            time_window=time_window,
            freq_window=freq_window,
            use_freq_axis=use_freq_axis,
            differential=differential,
            dyt=dyt,
            ff_mult=ff_mult,
        )
        self.conv1 = AxialLocalAttention2d(
            self.in_channels,
            self.out_channels,
            zero_output=False,
            **common,
        )
        self.conv2 = AxialLocalAttention2d(
            self.out_channels,
            self.out_channels,
            zero_output=bool(init_as_zero),
            **common,
        )
        self.res_conv = (
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.att = (
            AxialResidualAttention2d(
                self.out_channels,
                normalize=True,
                **common,
            )
            if self.attention
            else None
        )

    @classmethod
    def from_resblock(cls, reference, **kwargs):
        conv1 = reference.conv1
        cond_channels = reference.proj_emb.in_features if hasattr(reference, "proj_emb") else None
        return cls(
            conv1.in_channels,
            conv1.out_channels,
            cond_channels=cond_channels,
            downsample=getattr(reference, "downsample", False),
            upsample=getattr(reference, "upsample", False),
            normalize=getattr(reference, "normalize", True),
            leaky=getattr(reference, "leaky", False),
            attention=getattr(reference, "attention", False),
            normalize_residual=getattr(reference, "normalize_residual", False),
            dropout_rate=getattr(reference, "dropout_rate", 0.0),
            min_res_dropout=getattr(reference, "min_res_dropout", 16),
            **kwargs,
        )

    def forward(self, x, time_emb=None):
        if not self.normalize_residual:
            residual = x.clone()
        if self.normalize:
            x = self.norm1(x)
        if self.normalize_residual:
            residual = x.clone()
        x = self.activation(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            residual = F.avg_pool2d(residual, kernel_size=2, stride=2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        if self.proj_emb is not None and time_emb is not None:
            x = x + self.proj_emb(time_emb)[:, :, None, None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1] <= self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        x = x + self.res_conv(residual)
        if self.att is not None:
            x = self.att(x)
        return x


__all__ = [
    "TransformerBlock",
    "AxialLocalAttention2d",
    "AxialResidualAttention2d",
    "AxialLocalAttentionResBlock",
]
