from functools import reduce
from typing import Callable, Literal

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F


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
                q_chunk, k_chunk, v_chunk, attn_mask=mask, is_causal=False
            )
        )
    return torch.cat(outputs, dim=-2)


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
                dim_in, dim_out * 2, conv_kernel_size, padding=conv_kernel_size // 2
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
            nn.init.zeros_(linear_out.weight)
            if getattr(linear_out, "bias", None) is not None:
                nn.init.zeros_(linear_out.bias)
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
        self.dim_heads = dim_heads
        self.differential = differential
        dim_kv = dim if dim_context is None else dim_context
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim * (2 if differential else 1), bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * (3 if differential else 2), bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * (5 if differential else 3), bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif qk_norm == "dyt":
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)
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
            q, k, v, is_causal=self.causal if causal is None else causal
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
                q, q_diff = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                    (q, q_diff),
                )
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
                q, k, v = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                    (q, k, v),
                )
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)
        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
            else:
                k_freqs = freqs
            q = apply_rotary_pos_emb(q.float(), freqs.float()).to(v.dtype)
            k = apply_rotary_pos_emb(k.float(), k_freqs.float()).to(v.dtype)
        if self.differential:
            q, q_diff = q.unbind(dim=1)
            k, k_diff = k.unbind(dim=1)
            out = self._apply_attn(q, k, v, causal=causal, sliding_window=flash_attn_sliding_window)
            out = out - self._apply_attn(
                q_diff, k_diff, v, causal=causal, sliding_window=flash_attn_sliding_window
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
        attn_kwargs = attn_kwargs or {}
        ff_kwargs = ff_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        norm_map = {
            "layer_norm": nn.LayerNorm,
            "rms_norm": RMSNorm,
            "dyt": DynamicTanh,
        }
        norm_layer = norm_map[norm_type]
        self.pre_norm = norm_layer(dim, **norm_kwargs)
        self.self_attn = Attention(
            dim,
            dim_heads=min(dim_heads, dim),
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
                dim_heads=min(dim_heads, dim),
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.rope = RotaryEmbedding(max(min(dim_heads, dim) // 2, 1)) if add_rope else None

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


def WNConv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def _zero_pad_modulo_sequence(x, size, dim=-2):
    pad_len = (int(size) - x.shape[dim] % int(size)) % int(size)
    if pad_len <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    return torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)


class TransformerResamplingBlock(nn.Module):
    """SAME 1D resampling block copied for local use in mossland_codec_same."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        sliding_window=None,
        chunk_size=128,
        chunk_midpoint_shift=False,
        type="encoder",
        transformer_depth=3,
        checkpointing=False,
        conformer=False,
        layer_scale=False,
        dim_heads=128,
        differential=True,
        variable_stride=False,
        feat_scale=False,
        sinusoidal_blocks=0,
        mask_noise=0,
        ff_mult=3,
        mapping_bias=True,
        cross_attn=False,
        dyt=True,
        conv_mapping=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        del checkpointing, chunk_midpoint_shift, kwargs
        if type not in {"encoder", "decoder"}:
            raise ValueError(f"unknown TRB type {type!r}")
        transformer_dim = out_channels if type == "encoder" else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = bool(variable_stride)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.type = type
        self.mask_noise = float(mask_noise)
        self.sliding_window_latents = sliding_window
        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, self.stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(self.stride)
        kernel_size = 3 if conv_mapping else 1
        self.mapping = (
            WNConv1d(in_channels, out_channels, kernel_size, padding="same", bias=mapping_bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    transformer_dim,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=not layer_scale,
                    norm_type="dyt" if dyt else "rms_norm",
                    conformer=conformer,
                    layer_scale=layer_scale,
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                        "feat_scale": feat_scale,
                    },
                    ff_kwargs={
                        "mult": ff_mult,
                        "no_bias": False,
                        "sinusoidal": (transformer_depth - idx) < sinusoidal_blocks,
                    },
                    norm_kwargs={"eps": 1e-3},
                    cross_attend=cross_attn,
                )
                for idx in range(int(transformer_depth))
            ]
        )
        token_dim = out_channels if type == "encoder" else in_channels
        new_token_count = 1 if self.variable_stride else self.output_seg_size
        self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, new_token_count, token_dim))
        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [win * (int(stride) + 1 + int(prepend_cond_length)) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length=0):
        sub_chunk_size = int(stride) + 1 + int(prepend_cond_length)
        return (
            int(stride) if self.type == "encoder" else 1,
            1 if self.type == "encoder" else int(stride),
            sub_chunk_size,
        )

    def forward(self, x, stride=None, return_features=False, override_new_tokens=None, **kwargs):
        batch_size = x.shape[0]
        features = [] if return_features else None
        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride)
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
                pad_modulo = (
                    input_seg_size
                    if sliding_window is not None
                    else max(1, self.chunk_size // int(stride or self.stride))
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
            elif self.mask_noise > 0:
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
