from __future__ import annotations

import torch
from torch import nn
from einops import reduce
from vector_quantize_pytorch import ResidualVQ


class ResidualVectorQuantize(nn.Module):
    """Thin channel-first wrapper around lucidrains ResidualVQ.

    The underlying codebook is EMA/kmeans style, not a directly optimized
    nn.Embedding codebook. Inputs and outputs use the local codec convention
    [B, C, T], while lucidrains uses [B, T, C].
    """

    def __init__(
        self,
        input_dim: int,
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int | None = None,
        quantizer_dropout: float = 0.0,
        decay: float = 0.8,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = float(quantizer_dropout)
        self.vq = ResidualVQ(
            dim=input_dim,
            num_quantizers=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=codebook_dim,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
            learnable_codebook=False,
            ema_update=True,
            quantize_dropout=self.quantizer_dropout > 0.0,
        )

    def _output_from_indices(self, indices: torch.Tensor):
        if indices.shape[-1] < self.n_codebooks:
            indices = torch.nn.functional.pad(
                indices,
                (0, self.n_codebooks - indices.shape[-1]),
                value=-1,
            )
        codes = self.vq.get_codes_from_indices(indices)
        codes_summed = reduce(codes, "q ... -> ...", "sum")
        return self.vq.project_out(codes_summed)

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        if n_quantizers is not None and not (1 <= int(n_quantizers) <= self.n_codebooks):
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")

        x = z.transpose(1, 2).contiguous()
        quantized, indices, losses = self.vq(x)
        if n_quantizers is not None and int(n_quantizers) < self.n_codebooks:
            indices = indices[..., : int(n_quantizers)]
            quantized = self._output_from_indices(indices)
            losses = losses[..., : int(n_quantizers)]
        return (
            quantized.transpose(1, 2).contiguous(),
            indices.transpose(1, 2).contiguous(),
            losses.mean(),
        )

    def from_codes(self, codes: torch.Tensor):
        if codes.ndim != 3:
            raise ValueError(f"Expected codes with shape [B, Q, T], got {tuple(codes.shape)}")
        indices = codes.transpose(1, 2).contiguous()
        quantized = self._output_from_indices(indices)
        return quantized.transpose(1, 2).contiguous(), codes
