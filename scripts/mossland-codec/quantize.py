from __future__ import annotations

import torch
import torch.distributed as distributed
from torch import nn
from vector_quantize_pytorch import ResidualVQ
from vector_quantize_pytorch.vector_quantize_pytorch import (
    batched_sample_vectors,
    noop,
    sample_vectors_distributed,
)


class ResidualVectorQuantize(nn.Module):
    """Channel-first EMA ResidualVQ wrapper for Mossland packed embeddings."""

    def __init__(
        self,
        input_dim: int,
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int | None = None,
        quantizer_dropout: float = 0.0,
        quantizer_dropout_cutoff_index: int = 0,
        quantizer_dropout_multiple_of: int = 1,
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
        self.quantizer_dropout_cutoff_index = int(quantizer_dropout_cutoff_index)
        self.quantizer_dropout_multiple_of = int(quantizer_dropout_multiple_of)
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
            quantize_dropout_cutoff_index=self.quantizer_dropout_cutoff_index,
            quantize_dropout_multiple_of=self.quantizer_dropout_multiple_of,
        )

    def set_distributed_sync(self, enabled: bool) -> None:
        """Toggle lucidrains codebook sync after Lightning initializes DDP."""
        for layer in self.vq.layers:
            codebook = layer._codebook
            codebook.use_ddp = bool(enabled)
            codebook.sample_fn = sample_vectors_distributed if enabled else batched_sample_vectors
            codebook.replace_sample_fn = sample_vectors_distributed if enabled else batched_sample_vectors
            codebook.kmeans_all_reduce_fn = distributed.all_reduce if enabled else noop
            codebook.all_reduce_fn = distributed.all_reduce if enabled else noop

    def initialized_codebook_count(self) -> int:
        count = 0
        for layer in self.vq.layers:
            initted = getattr(layer._codebook, "initted", None)
            if initted is None:
                count += 1
                continue
            if not bool(initted.detach().item()):
                break
            count += 1
        return count

    def _has_initialized_codebooks(self, n_quantizers: int | None = None) -> bool:
        required = self.n_codebooks if n_quantizers is None else int(n_quantizers)
        return self.initialized_codebook_count() >= required

    def _assert_eval_codebooks_initialized(self, n_quantizers: int | None = None) -> None:
        # lucidrains initializes kmeans codebooks inside forward(), even in eval.
        # For inference/demo we want a saved tokenizer, not input-dependent init.
        if self.training or self._has_initialized_codebooks(n_quantizers):
            return
        required = self.n_codebooks if n_quantizers is None else int(n_quantizers)
        raise RuntimeError(
            f"RVQ codebook is not initialized for {required} quantizers "
            f"(initialized prefix: {self.initialized_codebook_count()}). "
            "Run enough training forwards or load a checkpoint with initialized "
            "RVQ state before eval/inference."
        )

    def _output_from_indices(self, indices: torch.Tensor):
        codes_summed = None
        active_codebooks = min(indices.shape[-1], self.n_codebooks)
        for quantizer_index in range(active_codebooks):
            layer_indices = indices[..., quantizer_index]
            valid = layer_indices >= 0
            if not bool(valid.any()):
                continue
            layer_indices = layer_indices.masked_fill(~valid, 0)
            codes = self.vq.layers[quantizer_index].get_codes_from_indices(layer_indices)
            codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
            codes_summed = codes if codes_summed is None else codes_summed + codes
        if codes_summed is None:
            shape = (*indices.shape[:-1], self.vq.codebook_dim)
            dtype = self.vq.layers[0].codebook.dtype
            codes_summed = torch.zeros(shape, device=indices.device, dtype=dtype)
        return self.vq.project_out(codes_summed)

    def encode_codes(self, z: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        if n_quantizers is not None and not (1 <= int(n_quantizers) <= self.n_codebooks):
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")
        self._assert_eval_codebooks_initialized(n_quantizers)
        x = z.transpose(1, 2).contiguous()
        _quantized, indices, _losses = self.vq(x)
        if n_quantizers is not None:
            indices = indices[..., : int(n_quantizers)]
        return indices.transpose(1, 2).contiguous()

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        if n_quantizers is not None and not (1 <= int(n_quantizers) <= self.n_codebooks):
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")
        self._assert_eval_codebooks_initialized(n_quantizers)

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
