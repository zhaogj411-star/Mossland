from typing import Union

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .layers import norm_conv1d


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = norm_conv1d(input_dim, self.codebook_dim, kernel_size=1)
        self.out_proj = norm_conv1d(self.codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z: torch.Tensor):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def decode_code(self, embed_id: torch.Tensor):
        return F.embedding(embed_id, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        return self.decode_code(indices), indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list[int]] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        if len(codebook_dim) != n_codebooks:
            raise ValueError("codebook_dim list length must match n_codebooks")

        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = list(codebook_dim)
        self.quantizer_dropout = float(quantizer_dropout)
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, self.codebook_size, self.codebook_dim[idx])
                for idx in range(self.n_codebooks)
            ]
        )

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if self.training and self.quantizer_dropout > 0.0:
            active = torch.full(
                (z.shape[0],),
                fill_value=self.n_codebooks + 1,
                device=z.device,
                dtype=torch.long,
            )
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],), device=z.device)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            active[:n_dropout] = dropout[:n_dropout]
        else:
            active = torch.full(
                (z.shape[0],),
                fill_value=int(n_quantizers),
                device=z.device,
                dtype=torch.long,
            )

        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = z.new_zeros(())
        codebook_loss = z.new_zeros(())
        codebook_indices = []
        latents = []

        for idx, quantizer in enumerate(self.quantizers):
            if (not self.training) and idx >= int(n_quantizers):
                break
            z_q_i, commitment_i, codebook_i, indices_i, z_e_i = quantizer(residual)
            mask = (torch.full_like(active, idx) < active).to(z.dtype)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss = commitment_loss + (commitment_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_i * mask).mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        return z_q, torch.stack(codebook_indices, dim=1), torch.cat(latents, dim=1), commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = 0.0
        projected = []
        for idx in range(codes.shape[1]):
            z_p_i = self.quantizers[idx].decode_code(codes[:, idx, :])
            projected.append(z_p_i)
            z_q = z_q + self.quantizers[idx].out_proj(z_p_i)
        return z_q, torch.cat(projected, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        z_q = 0.0
        projected = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])
        n_codebooks = int(np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0])
        for idx in range(n_codebooks):
            start, end = dims[idx], dims[idx + 1]
            z_p_i, codes_i = self.quantizers[idx].decode_latents(latents[:, start:end, :])
            projected.append(z_p_i)
            codes.append(codes_i)
            z_q = z_q + self.quantizers[idx].out_proj(z_p_i)
        return z_q, torch.cat(projected, dim=1), torch.stack(codes, dim=1)
