"""Pure-torch Music-DiT model for Mossland training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..core.masking import assemble_model_input, split_input_width
from ..core.pos_emb import sincos_1d


@dataclass
class MusicDiTConfig:
    latent_channels: int = 256
    patch_t: int = 1
    crossattn_emb_size: int = 768
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    use_context_conditioning: bool = True
    sigma_data: float = 0.5
    max_frames: int = 4096

    @property
    def latent_token_dim(self) -> int:
        return self.latent_channels * self.patch_t


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _lengths_to_padding_mask(lengths: torch.Tensor | None, seq_len: int, device: torch.device):
    if lengths is None:
        return None
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    return positions >= lengths.to(device=device).unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, text_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            batch_first=True,
            kdim=text_dim,
            vdim=text_dim,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        crossattn_emb: torch.Tensor,
        self_padding_mask: torch.Tensor | None = None,
        crossattn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.ada(timestep_emb).chunk(
            6,
            dim=-1,
        )
        h = _modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            h,
            h,
            h,
            key_padding_mask=self_padding_mask,
            need_weights=False,
        )[0]
        key_padding_mask = None if crossattn_mask is None else crossattn_mask < 0.5
        x = x + self.cross_attn(
            self.norm_ca(x),
            crossattn_emb,
            crossattn_emb,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class MusicDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 256,
        patch_t: int = 1,
        crossattn_emb_size: int = 768,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        use_context_conditioning: bool = True,
        sigma_data: float = 0.5,
        max_frames: int = 4096,
    ):
        super().__init__()
        self.config = MusicDiTConfig(
            latent_channels=latent_channels,
            patch_t=patch_t,
            crossattn_emb_size=crossattn_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            use_context_conditioning=use_context_conditioning,
            sigma_data=sigma_data,
            max_frames=max_frames,
        )
        self.latent_channels = self.config.latent_channels
        self.patch_t = self.config.patch_t
        self.latent_token_dim = self.config.latent_token_dim
        self.use_context_conditioning = self.config.use_context_conditioning
        self.sigma_data = float(self.config.sigma_data)

        input_width = (
            split_input_width(self.latent_token_dim)
            if self.use_context_conditioning
            else self.latent_token_dim
        )
        self.x_embedder = nn.Linear(input_width, self.config.hidden_size)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    text_dim=self.config.crossattn_emb_size,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.final = nn.Linear(self.config.hidden_size, self.latent_token_dim)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
        context_latent: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
        crossattn_mask: torch.Tensor | None = None,
        seq_len_q: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_context_conditioning and context_latent is not None and cond_mask is not None:
            x = assemble_model_input(x, context_latent, cond_mask)

        hidden = self.x_embedder(x)
        if pos_ids is None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            pos = sincos_1d(positions, self.config.hidden_size).unsqueeze(0)
        else:
            if pos_ids.ndim == 3:
                positions = pos_ids[..., 0]
            else:
                positions = pos_ids
            pos = sincos_1d(positions, self.config.hidden_size)
        hidden = hidden + pos.to(hidden.dtype)

        timestep_emb = self.t_mlp(sincos_1d(timesteps.float(), self.config.hidden_size).to(hidden.dtype))
        self_padding_mask = _lengths_to_padding_mask(seq_len_q, hidden.shape[1], hidden.device)
        for block in self.blocks:
            hidden = block(
                hidden,
                timestep_emb,
                crossattn_emb,
                self_padding_mask=self_padding_mask,
                crossattn_mask=crossattn_mask,
            )
        return self.final(self.norm_final(hidden))

