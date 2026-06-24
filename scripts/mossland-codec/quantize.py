from __future__ import annotations

import torch
from torch import nn
from einops import reduce
from vector_quantize_pytorch import ResidualVQ
from vector_quantize_pytorch import vector_quantize_pytorch as vq_backend


class ResidualVectorQuantize(nn.Module):
    """Thin channel-first wrapper around lucidrains ResidualVQ.

    Inputs and outputs use the local codec convention [B, C, T], while
    lucidrains uses [B, T, C]. This matches scripts.mossland_codec's RVQ
    behavior, with encode_codes kept for this legacy package's inference API.
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

    def set_distributed_sync(self, enabled: bool) -> None:
        """Toggle lucidrains EMA codebook synchronization after DDP init."""
        for layer in self.vq.layers:
            codebook = getattr(layer, "_codebook", None)
            if codebook is None:
                continue

            if hasattr(codebook, "sync_codebook"):
                codebook.sync_codebook = bool(enabled)

            if hasattr(codebook, "use_ddp"):
                codebook.use_ddp = bool(enabled)

            sync_kmeans = bool(enabled)
            codebook.sample_fn = (
                vq_backend.sample_vectors_distributed
                if sync_kmeans
                else vq_backend.batched_sample_vectors
            )
            codebook.replace_sample_fn = (
                vq_backend.sample_vectors_distributed
                if sync_kmeans
                else vq_backend.batched_sample_vectors
            )
            codebook.kmeans_all_reduce_fn = (
                vq_backend.distributed.all_reduce if sync_kmeans else vq_backend.noop
            )
            codebook.all_reduce_fn = (
                vq_backend.distributed.all_reduce if bool(enabled) else vq_backend.noop
            )

    def initialized_codebook_count(self) -> int:
        count = 0
        for layer in self.vq.layers:
            codebook = getattr(layer, "_codebook", None)
            initted = getattr(codebook, "initted", None)
            if initted is None:
                count += 1
            elif bool(initted.detach().cpu().item()):
                count += 1
        return count

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

    def encode_codes(self, z: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        if n_quantizers is not None and not (1 <= int(n_quantizers) <= self.n_codebooks):
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")
        x = z.transpose(1, 2).contiguous()
        _quantized, indices, _losses = self.vq(x)
        if n_quantizers is not None:
            indices = indices[..., : int(n_quantizers)]
        return indices.transpose(1, 2).contiguous()

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
