from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


_flash_attn_func = None
_flash_attn_imported = False


def _get_flash_attn_func():
    global _flash_attn_func, _flash_attn_imported
    if _flash_attn_func is not None:
        return _flash_attn_func
    if _flash_attn_imported:
        return None
    _flash_attn_imported = True
    try:
        from flash_attn import flash_attn_func
    except Exception:
        return None
    _flash_attn_func = flash_attn_func
    return _flash_attn_func


def samples_per_token(sample_rate: int, latent_rate_hz: int) -> int:
    sample_rate = int(sample_rate)
    latent_rate_hz = int(latent_rate_hz)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if latent_rate_hz <= 0:
        raise ValueError("latent_rate_hz must be positive")
    if sample_rate % latent_rate_hz != 0:
        raise ValueError(
            f"sample_rate / latent_rate_hz must be an integer, got "
            f"{sample_rate}/{latent_rate_hz}"
        )
    return sample_rate // latent_rate_hz


def token_count_for_samples(num_samples: int, sample_rate: int, latent_rate_hz: int) -> int:
    samples = samples_per_token(sample_rate, latent_rate_hz)
    num_samples = int(num_samples)
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    return num_samples // samples


class SlidingWindowSelfAttention(nn.Module):
    """Bidirectional local attention with FlashAttention window fallback."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_window_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if local_window_tokens < 0:
            raise ValueError("local_window_tokens must be non-negative")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.local_window_tokens = int(local_window_tokens)
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.to_out = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)

        if self._flash_attention_supported(q):
            out = self._flash_attention(q, k, v)
        else:
            out = self._sliding_window_attention(q, k, v)
        out = out.reshape(batch, length, self.dim)
        return self.to_out(out)

    def _flash_attention_supported(self, q: torch.Tensor) -> bool:
        if not q.is_cuda:
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self.head_dim > 256 or self.head_dim % 8 != 0:
            return False
        return _get_flash_attn_func() is not None

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        flash_attn_func = _get_flash_attn_func()
        if flash_attn_func is None:
            raise RuntimeError("flash_attn_func is not available")
        dropout_p = self.dropout.p if self.training else 0.0
        window = self.local_window_tokens
        return flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dropout_p=dropout_p,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            window_size=(window, window),
            deterministic=not self.training,
        )

    def _sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        window = self.local_window_tokens
        k = F.pad(k, (0, 0, window, window))
        v = F.pad(v, (0, 0, window, window))
        k_windows = k.unfold(dimension=2, size=2 * window + 1, step=1).permute(0, 1, 2, 4, 3)
        v_windows = v.unfold(dimension=2, size=2 * window + 1, step=1).permute(0, 1, 2, 4, 3)

        scores = torch.einsum("bhtd,bhtwd->bhtw", q, k_windows)
        scores = scores / (self.head_dim ** 0.5)
        if window > 0:
            length = q.shape[2]
            positions = torch.arange(length, device=q.device)
            offsets = torch.arange(-window, window + 1, device=q.device)
            window_positions = positions[:, None] + offsets[None, :]
            valid = (window_positions >= 0) & (window_positions < length)
            scores = scores.masked_fill(~valid.view(1, 1, length, -1), -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhtw,bhtwd->bhtd", attn, v_windows)
        return out.transpose(1, 2)


class LocalTimeTokenBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_window_tokens: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = dim * int(mlp_mult)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = SlidingWindowSelfAttention(
            dim=dim,
            num_heads=num_heads,
            local_window_tokens=local_window_tokens,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class LocalTimeTokenEncoder(nn.Module):
    """Small bidirectional local Transformer over fixed-rate time tokens."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        local_window_tokens: int = 16,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if local_window_tokens < 0:
            raise ValueError("local_window_tokens must be non-negative")
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.local_window_tokens = int(local_window_tokens)
        self.input_proj = nn.Linear(self.input_dim, self.token_dim)
        self.layers = nn.ModuleList(
            [
                LocalTimeTokenBlock(
                    dim=self.token_dim,
                    num_heads=int(num_heads),
                    local_window_tokens=self.local_window_tokens,
                    mlp_mult=int(mlp_mult),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input dim {self.input_dim}, got {x.shape[-1]}")
        tokens = self.input_proj(x)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.output_norm(tokens)
