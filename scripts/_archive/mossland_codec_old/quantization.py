from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RVQOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    per_quantizer_loss: torch.Tensor


class ResidualVectorQuantizer(nn.Module):
    """Token-wise residual vector quantizer with dynamic active codebooks."""

    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        num_quantizers: int = 16,
        codebook_init_scale: float = 0.02,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive")
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.codebooks = nn.Parameter(
            torch.randn(self.num_quantizers, self.codebook_size, self.dim)
            * float(codebook_init_scale)
        )

    def _check_depth(self, n_quantizers: int | None) -> int:
        if n_quantizers is None:
            return self.num_quantizers
        n_quantizers = int(n_quantizers)
        if n_quantizers < 1 or n_quantizers > self.num_quantizers:
            raise ValueError(
                f"n_quantizers must be in [1, {self.num_quantizers}], got {n_quantizers}"
            )
        return n_quantizers

    def forward(self, tokens: torch.Tensor, n_quantizers: int | None = None) -> RVQOutput:
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {tokens.shape[-1]}")
        depth = self._check_depth(n_quantizers)
        flat = tokens.reshape(-1, self.dim)
        residual = flat
        quantized_total = torch.zeros_like(flat)
        codes = []
        losses = []

        for idx in range(depth):
            codebook = self.codebooks[idx]
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(dim=1)
            )
            code = distances.argmin(dim=1)
            quantized = F.embedding(code, codebook)
            quantized_total = quantized_total + quantized
            losses.append(F.mse_loss(quantized, residual.detach()))
            residual = residual - quantized.detach()
            codes.append(code)

        quantized_st = flat + (quantized_total - flat).detach()
        codes_tensor = torch.stack(codes, dim=-1).reshape(*tokens.shape[:-1], depth)
        per_loss = torch.stack(losses)
        return RVQOutput(
            quantized=quantized_st.reshape_as(tokens),
            codes=codes_tensor.to(torch.int64),
            loss=per_loss.sum(),
            per_quantizer_loss=per_loss,
        )

    def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] < 1 or codes.shape[-1] > self.num_quantizers:
            raise ValueError(
                f"codes last dimension must be in [1, {self.num_quantizers}], "
                f"got {codes.shape[-1]}"
            )
        codes = codes.long()
        if codes.numel() and (codes.min().item() < 0 or codes.max().item() >= self.codebook_size):
            raise ValueError("RVQ codes are outside the codebook range")

        flat_codes = codes.reshape(-1, codes.shape[-1])
        quantized = torch.zeros(
            flat_codes.shape[0],
            self.dim,
            device=codes.device,
            dtype=self.codebooks.dtype,
        )
        for idx in range(flat_codes.shape[-1]):
            quantized = quantized + F.embedding(flat_codes[:, idx], self.codebooks[idx])
        return quantized.reshape(*codes.shape[:-1], self.dim)
