# Mossland Codec RVQ Bottleneck Implementation Plan

> **Superseded:** Do not execute this plan. It implements chunk-summary RVQ on existing learned-query latents and does not solve the time-alignment issue. Use the revised time-aligned local-Transformer RVQ design instead: `docs/superpowers/specs/2026-06-17-mossland-codec-time-aligned-rvq-design.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `scripts/mossland-codec` FSQ with a chunk-level RVQ bottleneck using `codebook_size=1024` and max `num_quantizers=256`, while supporting DAC-style dynamic active codebook count and preserving the continuous path and existing decoder latent layout.

**Architecture:** Add a focused RVQ module that flattens each chunk from `[128, 4]` to `[512]`, quantizes the 512-D vector with the first `n_quantizers` residual codebooks, then unflattens back to `[128, 4]` for the current pre-decoder and diffusion decoder. The model exposes continuous, RVQ, and code conversion helpers; the wrapper controls warmup, path selection, active codebook count, and RVQ auxiliary loss.

**Tech Stack:** Python, PyTorch, Lightning, Hydra, pytest, existing `scripts/mossland-codec` Transformer codec.

---

## File Structure

- Create `scripts/mossland-codec/quantization.py`: self-contained RVQ implementation and Mossland-specific chunk reshape wrapper.
- Create `tests/mossland_codec/test_quantization.py`: fast CPU tests for RVQ shape, code round-trip, and continuous bounding.
- Modify `scripts/mossland-codec/models.py`: remove local FSQ implementation, instantiate `MosslandRVQBottleneck`, expose latent/code helpers, and keep decoder-facing latent layout unchanged.
- Modify `scripts/mossland-codec/wrapper.py`: replace FSQ dropout naming with RVQ path naming, add RVQ warmup, dynamic active codebook count, and auxiliary losses.
- Modify `scripts/mossland-codec/inference.py`: route integer codes through `model.codes_to_latents()` and discrete encode through `model.latents_to_codes(..., n_quantizers=...)`.
- Modify `scripts/mossland-codec/infer_demo.py`: replace direct `.fsq` access with model helper calls and save one source/RVQ/continuous concat demo wav.
- Modify `scripts/configs/experiment/mossland-codec.yaml`, `scripts/configs/experiment/mossland-codec-small.yaml`, and `scripts/configs/experiment/codicodec-paper-repro.yaml`: replace `fsq_levels` and `fsq_dropout_prob` with RVQ config.
- Modify `tests/mossland_codec/test_wrapper_imports.py`: update model/config/wrapper expectations.
- Modify `docs/mossland-codec.md`, `docs/code-index.md`, and `docs/memory/current.md`: persist the new codec bottleneck shape, config names, and current handoff.

Do not modify `scripts/codicodec/` or A2A model code. A2A configs may still contain `use_fsq=false` because they belong to `scripts/mossland-a2a`, not this codec bottleneck.

---

### Task 1: Add RVQ Quantization Module

**Files:**
- Create: `scripts/mossland-codec/quantization.py`
- Create: `tests/mossland_codec/test_quantization.py`

- [ ] **Step 1: Write failing quantization tests**

Create `tests/mossland_codec/test_quantization.py`:

```python
import importlib

import pytest
import torch


quantization = importlib.import_module("scripts.mossland-codec.quantization")


def test_latent_tanh_bound_limits_values_without_changing_shape():
    bound = quantization.LatentTanhBound(eps=1e-4)
    x = torch.tensor([[-100.0, 0.0, 100.0]])

    out = bound(x)

    assert out.shape == x.shape
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)
    assert out[0, 0] < -0.99
    assert out[0, 2] > 0.99


def test_rvq_bottleneck_flattens_chunks_and_decodes_codes():
    torch.manual_seed(0)
    bottleneck = quantization.MosslandRVQBottleneck(
        num_latents=4,
        bottleneck_channels=2,
        codebook_size=8,
        num_quantizers=3,
    )
    latents = torch.randn(2, 12, 2)

    result = bottleneck(latents)
    decoded = bottleneck.codes_to_latents(result.codes)

    assert result.continuous.shape == latents.shape
    assert result.quantized.shape == latents.shape
    assert result.codes.shape == (2, 3, 3)
    assert result.codes.dtype == torch.long
    torch.testing.assert_close(decoded, result.quantized)
    assert result.loss.ndim == 0
    assert result.recon_loss.ndim == 0
    assert result.codebook_loss.ndim == 0
    assert result.commitment_loss.ndim == 0
    assert result.code_usage.shape == (3,)


def test_rvq_bottleneck_supports_dynamic_quantizer_count():
    torch.manual_seed(0)
    bottleneck = quantization.MosslandRVQBottleneck(
        num_latents=4,
        bottleneck_channels=2,
        codebook_size=8,
        num_quantizers=3,
    )
    latents = torch.randn(2, 12, 2)

    result = bottleneck(latents, n_quantizers=2)
    decoded = bottleneck.codes_to_latents(result.codes)

    assert result.quantized.shape == latents.shape
    assert result.codes.shape == (2, 3, 2)
    assert result.code_usage.shape == (2,)
    torch.testing.assert_close(decoded, result.quantized)


def test_rvq_bottleneck_rejects_non_chunk_multiple():
    bottleneck = quantization.MosslandRVQBottleneck(
        num_latents=4,
        bottleneck_channels=2,
        codebook_size=8,
        num_quantizers=3,
    )

    with pytest.raises(ValueError, match="multiple of num_latents"):
        bottleneck(torch.randn(1, 5, 2))


def test_rvq_bottleneck_rejects_wrong_channel_count():
    bottleneck = quantization.MosslandRVQBottleneck(
        num_latents=4,
        bottleneck_channels=2,
        codebook_size=8,
        num_quantizers=3,
    )

    with pytest.raises(ValueError, match="bottleneck_channels"):
        bottleneck(torch.randn(1, 4, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_quantization.py
```

Expected: FAIL with `ModuleNotFoundError` or `AttributeError` because `scripts.mossland-codec.quantization` does not exist yet.

- [ ] **Step 3: Implement `quantization.py`**

Create `scripts/mossland-codec/quantization.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RVQBottleneckOutput:
    continuous: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    recon_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    code_usage: torch.Tensor


class LatentTanhBound(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * (1.0 - self.eps)


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.codebook = nn.Parameter(torch.empty(self.codebook_size, self.dim))
        nn.init.uniform_(self.codebook, -self.dim ** -0.5, self.dim ** -0.5)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape[-1]}")

        flat = x.reshape(-1, self.dim).float()
        codebook = self.codebook.float()
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        codes = distances.argmin(dim=1)
        quantized = F.embedding(codes, self.codebook).reshape_as(x)
        codebook_loss = F.mse_loss(quantized.float(), x.detach().float())
        commitment_loss = F.mse_loss(x.float(), quantized.detach().float())
        quantized = quantized.to(x.dtype)
        quantized_st = x + (quantized - x).detach()
        return quantized_st, codes.reshape(x.shape[:-1]), codebook_loss, commitment_loss

    def codes_to_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes.to(torch.long), self.codebook)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, num_quantizers: int):
        super().__init__()
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.layers = nn.ModuleList(
            VectorQuantizer(self.dim, self.codebook_size)
            for _ in range(self.num_quantizers)
        )

    def _normalize_n_quantizers(self, n_quantizers: int | None) -> int:
        if n_quantizers is None:
            return self.num_quantizers
        n_quantizers = int(n_quantizers)
        if n_quantizers <= 0 or n_quantizers > self.num_quantizers:
            raise ValueError(
                f"n_quantizers must be in [1, {self.num_quantizers}], got {n_quantizers}"
            )
        return n_quantizers

    def forward(self, x: torch.Tensor, n_quantizers: int | None = None):
        active_quantizers = self._normalize_n_quantizers(n_quantizers)
        quantized_sum = torch.zeros_like(x)
        residual = x
        codes = []
        codebook_loss = x.new_zeros(())
        commitment_loss = x.new_zeros(())

        for layer in self.layers[:active_quantizers]:
            quantized, layer_codes, layer_codebook_loss, layer_commitment_loss = layer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            codes.append(layer_codes)
            codebook_loss = codebook_loss + layer_codebook_loss
            commitment_loss = commitment_loss + layer_commitment_loss

        stacked_codes = torch.stack(codes, dim=-1)
        code_usage = self._code_usage(stacked_codes)
        normalizer = float(active_quantizers)
        return (
            quantized_sum,
            stacked_codes,
            codebook_loss / normalizer,
            commitment_loss / normalizer,
            code_usage,
        )

    def codes_to_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        active_quantizers = self._normalize_n_quantizers(codes.shape[-1])
        if active_quantizers != codes.shape[-1]:
            raise ValueError(
                f"expected at most {self.num_quantizers} RVQ code layers, got {codes.shape[-1]}"
            )

        vectors = None
        for idx, layer in enumerate(self.layers[:active_quantizers]):
            layer_vectors = layer.codes_to_vectors(codes[..., idx])
            vectors = layer_vectors if vectors is None else vectors + layer_vectors
        assert vectors is not None
        return vectors

    def _code_usage(self, codes: torch.Tensor) -> torch.Tensor:
        usage = []
        for idx in range(codes.shape[-1]):
            unique = torch.unique(codes[..., idx]).numel()
            usage.append(
                torch.tensor(
                    float(unique) / float(self.codebook_size),
                    device=codes.device,
                    dtype=torch.float32,
                )
            )
        return torch.stack(usage)


class MosslandRVQBottleneck(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        bottleneck_channels: int = 4,
        codebook_size: int = 1024,
        num_quantizers: int = 256,
        bound_eps: float = 1e-5,
        codebook_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_latents = int(num_latents)
        self.bottleneck_channels = int(bottleneck_channels)
        self.embedding_dim = self.num_latents * self.bottleneck_channels
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.codebook_loss_weight = float(codebook_loss_weight)
        self.commitment_loss_weight = float(commitment_loss_weight)
        self.bound = LatentTanhBound(eps=bound_eps)
        self.rvq = ResidualVectorQuantizer(
            dim=self.embedding_dim,
            codebook_size=self.codebook_size,
            num_quantizers=self.num_quantizers,
        )

    def flatten_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.bottleneck_channels:
            raise ValueError(
                f"expected bottleneck_channels={self.bottleneck_channels}, got {latents.shape[-1]}"
            )
        if latents.shape[-2] % self.num_latents != 0:
            raise ValueError(
                f"latent length must be a multiple of num_latents={self.num_latents}"
            )

        chunks = latents.shape[-2] // self.num_latents
        return latents.contiguous().reshape(
            *latents.shape[:-2],
            chunks,
            self.embedding_dim,
        )

    def unflatten_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.embedding_dim:
            raise ValueError(f"expected embedding_dim={self.embedding_dim}, got {latents.shape[-1]}")

        return latents.contiguous().reshape(
            *latents.shape[:-2],
            latents.shape[-2] * self.num_latents,
            self.bottleneck_channels,
        )

    def forward(
        self,
        latents: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> RVQBottleneckOutput:
        continuous = self.bound(latents)
        flattened = self.flatten_latents(continuous)
        quantized_flat, codes, codebook_loss, commitment_loss, code_usage = self.rvq(
            flattened,
            n_quantizers=n_quantizers,
        )
        quantized = self.unflatten_latents(quantized_flat)
        recon_loss = F.mse_loss(quantized_flat.float(), flattened.detach().float())
        loss = (
            recon_loss
            + self.codebook_loss_weight * codebook_loss
            + self.commitment_loss_weight * commitment_loss
        )
        return RVQBottleneckOutput(
            continuous=continuous,
            quantized=quantized,
            codes=codes,
            loss=loss,
            recon_loss=recon_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            code_usage=code_usage,
        )

    def latents_to_codes(
        self,
        latents: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        return self.forward(latents, n_quantizers=n_quantizers).codes

    def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if codes.ndim == 2:
            squeeze = True
            codes = codes.unsqueeze(0)
        quantized_flat = self.rvq.codes_to_vectors(codes)
        latents = self.unflatten_latents(quantized_flat)
        return latents.squeeze(0) if squeeze else latents
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_quantization.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/mossland-codec/quantization.py tests/mossland_codec/test_quantization.py
git commit -m "Add Mossland RVQ quantization module"
```

---

### Task 2: Integrate RVQ Into `MosslandCodecTransformer`

**Files:**
- Modify: `scripts/mossland-codec/models.py`
- Modify: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing model tests**

In `tests/mossland_codec/test_wrapper_imports.py`, replace `test_mossland_model_can_disable_fsq_for_continuous_bottleneck` with:

```python
def test_mossland_model_can_disable_quantizer_for_continuous_bottleneck():
    module = importlib.import_module("scripts.mossland-codec.models")
    model = module.MosslandCodecTransformer(
        dim=8,
        head_dim=4,
        cond_channels=8,
        num_layers=1,
        num_layers_encoder=1,
        num_latents=2,
        num_more_latents=0,
        bottleneck_channels=8,
        use_quantizer=False,
        frontend_base_channels=4,
        frontend_multipliers_list=[1],
        frontend_layers_list=[1],
        frontend_encoder_layers_list=[1],
        frontend_freq_downsample_list=[],
        hop=8,
        fac=2,
        spec_length=2,
    )

    assert model.quantizer is None
    assert model.bottleneck_channels == model.dim
    assert not hasattr(model, "fsq")
```

Add this test near it:

```python
def test_mossland_model_uses_rvq_bottleneck_by_default():
    module = importlib.import_module("scripts.mossland-codec.models")
    model = module.MosslandCodecTransformer(
        dim=8,
        head_dim=4,
        cond_channels=8,
        num_layers=1,
        num_layers_encoder=1,
        num_latents=2,
        num_more_latents=0,
        bottleneck_channels=2,
        rvq_codebook_size=8,
        rvq_num_quantizers=3,
        frontend_base_channels=4,
        frontend_multipliers_list=[1],
        frontend_layers_list=[1],
        frontend_encoder_layers_list=[1],
        frontend_freq_downsample_list=[],
        hop=8,
        fac=2,
        spec_length=2,
    )

    assert model.quantizer.codebook_size == 8
    assert model.quantizer.num_quantizers == 3
    assert model.rvq_codebook_size == 8
    assert model.rvq_num_quantizers == 3
    assert model.bottleneck_channels == 2
    assert not hasattr(model, "fsq")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_mossland_model_can_disable_quantizer_for_continuous_bottleneck tests/mossland_codec/test_wrapper_imports.py::test_mossland_model_uses_rvq_bottleneck_by_default
```

Expected: FAIL because `use_quantizer`, `rvq_codebook_size`, and `model.quantizer` are not implemented yet.

- [ ] **Step 3: Modify model imports and remove local FSQ**

In `scripts/mossland-codec/models.py`:

```python
from .quantization import MosslandRVQBottleneck, RVQBottleneckOutput
```

Delete the local `round_ste()` function and `FSQ` class. The `scripts/codicodec/models.py` FSQ reference implementation stays untouched.

- [ ] **Step 4: Replace constructor quantizer parameters**

In `MosslandCodecTransformer.__init__`, replace:

```python
fsq_levels: list[int] | None = None,
use_fsq: bool = True,
bottleneck_channels: int | None = None,
```

with:

```python
use_quantizer: bool = True,
bottleneck_channels: int | None = None,
rvq_codebook_size: int = 1024,
rvq_num_quantizers: int = 256,
rvq_bound_eps: float = 1e-5,
rvq_codebook_loss_weight: float = 1.0,
rvq_commitment_loss_weight: float = 0.0,
```

Replace the stored attributes block:

```python
self.num_latents = int(num_latents)
self.num_more_latents = int(num_more_latents)
self.use_quantizer = bool(use_quantizer)
if bottleneck_channels is None:
    bottleneck_channels = 4 if self.use_quantizer else self.dim
self.bottleneck_channels = int(bottleneck_channels)
self.rvq_codebook_size = int(rvq_codebook_size)
self.rvq_num_quantizers = int(rvq_num_quantizers)
self.rvq_bound_eps = float(rvq_bound_eps)
self.rvq_codebook_loss_weight = float(rvq_codebook_loss_weight)
self.rvq_commitment_loss_weight = float(rvq_commitment_loss_weight)
```

Replace final quantizer construction:

```python
self.quantizer = (
    MosslandRVQBottleneck(
        num_latents=self.num_latents,
        bottleneck_channels=self.bottleneck_channels,
        codebook_size=self.rvq_codebook_size,
        num_quantizers=self.rvq_num_quantizers,
        bound_eps=self.rvq_bound_eps,
        codebook_loss_weight=self.rvq_codebook_loss_weight,
        commitment_loss_weight=self.rvq_commitment_loss_weight,
    )
    if self.use_quantizer
    else None
)
```

Update the class docstring from FSQ wording to:

```python
"""带多任务条件的 Mossland consistency Transformer autoencoder。

Provides an encoder that maps spectrogram patches to latents and a decoder
that reconstructs spectrograms conditioned on noise level and Mossland task
metadata. Supports both parallel and autoregressive decoding schedules and
a chunk-level residual vector quantizer (RVQ) bottleneck.
"""
```

- [ ] **Step 5: Add encoder helper methods**

Inside `MosslandCodecTransformer`, replace `encoder_forward()` and `encoder_forward_fast()` bodies with helpers that produce raw latents once and then apply RVQ:

```python
def _encode_bottleneck_raw(self, x, log_magnitude=False):
    assert x.shape[-1] % self.spec_length == 0, (
        f"Input shape {x.shape[-1]} is not divisible by {self.spec_length}."
    )
    factor = None
    if x.shape[-1] > self.spec_length:
        x_ls = torch.split(x, self.spec_length, dim=-1)
        factor = len(x_ls)
        x = torch.cat(x_ls, dim=0)
    x = self.frontend_encoder_down(
        x,
        gain=self.gain_encoder,
        log_magnitude=log_magnitude,
    )[0]
    if self.more_latents_encoder is not None:
        latent_queries = torch.cat(
            (
                self.latents.expand(x.shape[0], -1, -1),
                self.more_latents_encoder.expand(x.shape[0], -1, -1),
            ),
            -2,
        )
    else:
        latent_queries = self.latents.expand(x.shape[0], -1, -1)
    x = self.encoder(
        x,
        latent_queries,
        return_latents=True,
        skip_input_layer=True,
        skip_output_layer=False,
        print_magnitudes=log_magnitude,
    )[:, : self.num_latents]
    if factor is not None:
        x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
    return x

def _apply_quantizer(
    self,
    latents: torch.Tensor,
    dont_quantize: bool,
    n_quantizers: int | None = None,
):
    if self.quantizer is None:
        return latents, None
    output = self.quantizer(latents, n_quantizers=n_quantizers)
    return (output.continuous if dont_quantize else output.quantized), output

def encoder_forward(
    self,
    x,
    dont_quantize=False,
    log_magnitude=False,
    n_quantizers: int | None = None,
):
    raw = self._encode_bottleneck_raw(x, log_magnitude=log_magnitude)
    latents, _ = self._apply_quantizer(
        raw,
        dont_quantize=dont_quantize,
        n_quantizers=n_quantizers,
    )
    return latents

def encoder_forward_with_quantizer(
    self,
    x,
    use_quantizer: bool = False,
    log_magnitude: bool = False,
    n_quantizers: int | None = None,
):
    raw = self._encode_bottleneck_raw(x, log_magnitude=log_magnitude)
    latents, output = self._apply_quantizer(
        raw,
        dont_quantize=not use_quantizer,
        n_quantizers=n_quantizers,
    )
    return latents, output

@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
def encoder_forward_fast(
    self,
    x,
    dont_quantize=False,
    log_magnitude=False,
    n_quantizers: int | None = None,
):
    raw = self._encode_bottleneck_raw(x, log_magnitude=log_magnitude)
    latents, _ = self._apply_quantizer(
        raw,
        dont_quantize=dont_quantize,
        n_quantizers=n_quantizers,
    )
    return latents
```

Add conversion helpers near the encoder methods:

```python
def latents_to_codes(
    self,
    latents: torch.Tensor,
    n_quantizers: int | None = None,
) -> torch.Tensor:
    if self.quantizer is None:
        raise ValueError("discrete encoding requires an enabled RVQ quantizer")
    return self.quantizer.latents_to_codes(latents, n_quantizers=n_quantizers)

def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
    if self.quantizer is None:
        raise ValueError("discrete decoding requires an enabled RVQ quantizer")
    return self.quantizer.codes_to_latents(codes)
```

- [ ] **Step 6: Run model tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_quantization.py tests/mossland_codec/test_wrapper_imports.py::test_mossland_model_can_disable_quantizer_for_continuous_bottleneck tests/mossland_codec/test_wrapper_imports.py::test_mossland_model_uses_rvq_bottleneck_by_default
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/mossland-codec/models.py tests/mossland_codec/test_wrapper_imports.py
git commit -m "Replace Mossland codec FSQ with RVQ bottleneck"
```

---

### Task 3: Update Training Wrapper for RVQ Warmup and Auxiliary Loss

**Files:**
- Modify: `scripts/mossland-codec/wrapper.py`
- Modify: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing wrapper tests**

Add these tests to `tests/mossland_codec/test_wrapper_imports.py`:

```python
def test_training_wrapper_rvq_path_active_respects_warmup_and_dropout(monkeypatch):
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    wrapper = SimpleNamespace(
        rvq_warmup_steps=10,
        rvq_path_dropout_prob=0.0,
        global_step=9,
    )

    assert module.MosslandCodecTrainingWrapper._rvq_path_active(wrapper, torch.device("cpu")) is False

    wrapper.global_step = 10
    assert module.MosslandCodecTrainingWrapper._rvq_path_active(wrapper, torch.device("cpu")) is True

    wrapper.rvq_path_dropout_prob = 1.0
    assert module.MosslandCodecTrainingWrapper._rvq_path_active(wrapper, torch.device("cpu")) is False


def test_training_wrapper_selects_dynamic_rvq_quantizer_count(monkeypatch):
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    wrapper = SimpleNamespace(
        model=SimpleNamespace(
            quantizer=SimpleNamespace(num_quantizers=256),
        ),
        rvq_active_quantizers=None,
        rvq_min_active_quantizers=None,
    )

    assert module.MosslandCodecTrainingWrapper._rvq_n_quantizers(wrapper, torch.device("cpu")) is None

    wrapper.rvq_active_quantizers = 64
    assert module.MosslandCodecTrainingWrapper._rvq_n_quantizers(wrapper, torch.device("cpu")) == 64

    wrapper.rvq_active_quantizers = None
    wrapper.rvq_min_active_quantizers = 32
    monkeypatch.setattr(module.torch, "randint", lambda low, high, size, device=None: torch.tensor([77], device=device))
    assert module.MosslandCodecTrainingWrapper._rvq_n_quantizers(wrapper, torch.device("cpu")) == 77


def test_training_wrapper_rvq_aux_loss_combines_weighted_terms():
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    wrapper = SimpleNamespace(
        rvq_loss_weight=2.0,
        rvq_codebook_loss_weight=3.0,
        rvq_commitment_loss_weight=5.0,
    )
    output = SimpleNamespace(
        recon_loss=torch.tensor(0.25),
        codebook_loss=torch.tensor(0.5),
        commitment_loss=torch.tensor(0.125),
        code_usage=torch.tensor([0.1, 0.3, 0.5]),
    )

    loss, metrics = module.MosslandCodecTrainingWrapper._rvq_aux_loss(wrapper, output)

    expected_inner = torch.tensor(0.25 + 3.0 * 0.5 + 5.0 * 0.125)
    torch.testing.assert_close(loss, expected_inner * 2.0)
    torch.testing.assert_close(metrics["loss/rvq"], expected_inner.detach())
    torch.testing.assert_close(metrics["loss/rvq_weighted"], (expected_inner * 2.0).detach())
    torch.testing.assert_close(metrics["latent/rvq_code_usage_mean"], torch.tensor(0.3))
```

Replace `test_demo_callback_saves_quantized_and_continuous_demos` with:

```python
def test_demo_callback_saves_discrete_and_continuous_concat_demo(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    callback = module.MosslandCodecTrainingCallback(
        demo_dir=str(tmp_path),
        demo_num=1,
        demo_every=1000,
        sample_rate=4,
        use_ema=False,
        silence_seconds=0.0,
    )

    saved = []
    generate_calls = []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.torchaudio,
        "save",
        lambda path, audio, sample_rate: saved.append((Path(path).name, audio.clone(), sample_rate)),
    )

    class FakeModel:
        def generate_waveform(self, src, **kwargs):
            generate_calls.append(kwargs["dont_quantize"])
            value = 3.0 if kwargs["dont_quantize"] is False else 4.0
            return src.detach().cpu(), torch.full_like(src, value).detach().cpu()

        def prepare_audio_batch(self, audio):
            return audio

    payload = {
        "src": torch.ones(1, 2, 8),
        "target": torch.zeros(1, 2, 8),
        "task_id": "reconstruct",
    }
    trainer = SimpleNamespace(global_step=1, global_rank=0)
    lightning_module = SimpleNamespace(model=FakeModel())

    callback.on_train_batch_end(trainer, lightning_module, None, (payload, {}), 0)

    assert generate_calls == [False, True]
    assert len(saved) == 1
    name, audio, sample_rate = saved[0]
    assert name == "1_0_reconstruct_rank0_src_target_rvq_continuous.wav"
    assert sample_rate == 4
    assert audio.shape == (2, 32)
    torch.testing.assert_close(audio[:, 0:8], torch.ones(2, 8))
    torch.testing.assert_close(audio[:, 8:16], torch.zeros(2, 8))
    torch.testing.assert_close(audio[:, 16:24], torch.full((2, 8), 3.0))
    torch.testing.assert_close(audio[:, 24:32], torch.full((2, 8), 4.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_rvq_path_active_respects_warmup_and_dropout tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_selects_dynamic_rvq_quantizer_count tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_rvq_aux_loss_combines_weighted_terms tests/mossland_codec/test_wrapper_imports.py::test_demo_callback_saves_discrete_and_continuous_concat_demo
```

Expected: FAIL because `_rvq_path_active`, `_rvq_n_quantizers`, `_rvq_aux_loss`, and the one-file concat demo behavior do not exist.

- [ ] **Step 3: Update wrapper constructor**

In `MosslandCodecTrainingWrapper.__init__`, replace `fsq_dropout_prob` with:

```python
rvq_path_dropout_prob: float = 0.6,
rvq_warmup_steps: int = 10_000,
rvq_active_quantizers: int | None = None,
rvq_min_active_quantizers: int | None = None,
rvq_loss_weight: float = 1.0,
rvq_codebook_loss_weight: float = 1.0,
rvq_commitment_loss_weight: float = 0.0,
```

Replace stored fields:

```python
self.rvq_path_dropout_prob = float(rvq_path_dropout_prob)
self.rvq_warmup_steps = int(rvq_warmup_steps)
self.rvq_active_quantizers = _normalize_positive_int_or_none(
    rvq_active_quantizers,
    "rvq_active_quantizers",
)
self.rvq_min_active_quantizers = _normalize_positive_int_or_none(
    rvq_min_active_quantizers,
    "rvq_min_active_quantizers",
)
self.rvq_loss_weight = float(rvq_loss_weight)
self.rvq_codebook_loss_weight = float(rvq_codebook_loss_weight)
self.rvq_commitment_loss_weight = float(rvq_commitment_loss_weight)
```

- [ ] **Step 4: Replace dropout helper and add RVQ loss helper**

Replace `_fsq_dropout_active()` with:

```python
def _rvq_path_dropout_active(self, device: torch.device) -> bool:
    if self.rvq_path_dropout_prob <= 0.0:
        return False
    if self.rvq_path_dropout_prob >= 1.0:
        return True
    return bool(torch.rand((), device=device) < self.rvq_path_dropout_prob)

def _rvq_path_active(self, device: torch.device) -> bool:
    if int(getattr(self, "global_step", 0)) < self.rvq_warmup_steps:
        return False
    return not self._rvq_path_dropout_active(device)

def _rvq_n_quantizers(self, device: torch.device) -> int | None:
    quantizer = getattr(self.model, "quantizer", None)
    max_quantizers = getattr(quantizer, "num_quantizers", None)
    if self.rvq_active_quantizers is not None:
        if max_quantizers is not None and self.rvq_active_quantizers > max_quantizers:
            raise ValueError(
                f"rvq_active_quantizers={self.rvq_active_quantizers} exceeds max {max_quantizers}"
            )
        return self.rvq_active_quantizers
    if self.rvq_min_active_quantizers is None:
        return None
    if max_quantizers is None:
        raise ValueError("rvq_min_active_quantizers requires model.quantizer.num_quantizers")
    if self.rvq_min_active_quantizers > max_quantizers:
        raise ValueError(
            f"rvq_min_active_quantizers={self.rvq_min_active_quantizers} exceeds max {max_quantizers}"
        )
    return int(
        torch.randint(
            self.rvq_min_active_quantizers,
            max_quantizers + 1,
            (1,),
            device=device,
        ).item()
    )

def _rvq_aux_loss(self, quantizer_output, device: torch.device | None = None):
    if quantizer_output is None or self.rvq_loss_weight <= 0.0:
        return torch.zeros((), device=device), {}
    rvq_loss = (
        quantizer_output.recon_loss
        + self.rvq_codebook_loss_weight * quantizer_output.codebook_loss
        + self.rvq_commitment_loss_weight * quantizer_output.commitment_loss
    )
    weighted = self.rvq_loss_weight * rvq_loss
    metrics = {
        "loss/rvq": rvq_loss.detach(),
        "loss/rvq_weighted": weighted.detach(),
        "loss/rvq_recon": quantizer_output.recon_loss.detach(),
        "loss/rvq_codebook": quantizer_output.codebook_loss.detach(),
        "loss/rvq_commitment": quantizer_output.commitment_loss.detach(),
        "latent/rvq_code_usage_mean": quantizer_output.code_usage.detach().float().mean(),
    }
    return weighted, metrics
```

- [ ] **Step 5: Update `training_step()`**

Replace the current FSQ path:

```python
dont_quantize = self._fsq_dropout_active(src_representation.device)
latents = self.model.encoder_forward(src_representation, dont_quantize=dont_quantize)
```

with:

```python
rvq_active = self._rvq_path_active(src_representation.device)
rvq_n_quantizers = self._rvq_n_quantizers(src_representation.device) if rvq_active else None
if hasattr(self.model, "encoder_forward_with_quantizer"):
    latents, quantizer_output = self.model.encoder_forward_with_quantizer(
        src_representation,
        use_quantizer=rvq_active,
        n_quantizers=rvq_n_quantizers,
    )
else:
    latents = self.model.encoder_forward(
        src_representation,
        dont_quantize=not rvq_active,
        n_quantizers=rvq_n_quantizers,
    )
    quantizer_output = None
```

After `_consistency_loss(...)` returns, add:

```python
rvq_loss, rvq_metrics = self._rvq_aux_loss(quantizer_output, device=loss.device)
loss = loss + rvq_loss
metrics.update(rvq_metrics)
```

Replace the old log:

```python
self.log(
    "latent/fsq_dropout",
    torch.tensor(float(dont_quantize), device=loss.device),
    prog_bar=False,
    on_step=True,
    on_epoch=False,
    sync_dist=False,
)
```

with:

```python
self.log(
    "latent/rvq_active",
    torch.tensor(float(rvq_active), device=loss.device),
    prog_bar=False,
    on_step=True,
    on_epoch=False,
    sync_dist=False,
)
self.log(
    "latent/rvq_path_dropout",
    torch.tensor(float(not rvq_active), device=loss.device),
    prog_bar=False,
    on_step=True,
    on_epoch=False,
    sync_dist=False,
)
self.log(
    "latent/rvq_n_quantizers",
    torch.tensor(float(rvq_n_quantizers or 0), device=loss.device),
    prog_bar=False,
    on_step=True,
    on_epoch=False,
    sync_dist=False,
)
```

- [ ] **Step 6: Update demo callback to save one concat comparison**

In `MosslandCodecTrainingCallback.on_train_batch_end()`, keep the two generation calls but save a single comparison file per sample. Replace the `generated_versions` loop with:

```python
src_audio, rvq_generated = model.generate_waveform(
    src,
    task_id=demo_task_id,
    dont_quantize=False,
)
_, continuous_generated = model.generate_waveform(
    src,
    task_id=demo_task_id,
    dont_quantize=True,
)
target = model.prepare_audio_batch(target).detach().cpu()
for idx, (src_item, target_item, rvq_item, continuous_item) in enumerate(
    zip(src_audio, target, rvq_generated, continuous_generated)
):
    task_id = _label_at(demo_task_id, idx)
    base = f"{trainer.global_step}_{idx}_{task_id}_rank{trainer.global_rank}"
    comparison = self._concat_demo_audio(
        src_item,
        target_item,
        rvq_item,
        continuous_item,
    )
    torchaudio.save(
        os.path.join(
            self.demo_dir,
            f"{base}_src_target_rvq_continuous.wav",
        ),
        comparison.float(),
        self.sample_rate,
    )
```

Also update local cleanup variable names in the `try/finally` block from `quantized_generated` to `rvq_generated`, and remove `generated_versions`.

- [ ] **Step 7: Run wrapper tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_rvq_path_active_respects_warmup_and_dropout tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_selects_dynamic_rvq_quantizer_count tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_rvq_aux_loss_combines_weighted_terms tests/mossland_codec/test_wrapper_imports.py::test_demo_callback_saves_discrete_and_continuous_concat_demo
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/mossland-codec/wrapper.py tests/mossland_codec/test_wrapper_imports.py
git commit -m "Train Mossland RVQ bottleneck with warmup"
```

---

### Task 4: Update Inference and Demo Discrete Code Paths

**Files:**
- Modify: `scripts/mossland-codec/inference.py`
- Modify: `scripts/mossland-codec/infer_demo.py`
- Modify: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing inference tests**

Add to `tests/mossland_codec/test_wrapper_imports.py`:

```python
def test_encoder_decoder_discrete_encode_uses_rvq_helpers(monkeypatch):
    module = importlib.import_module("scripts.mossland-codec.inference")
    codec = object.__new__(module.EncoderDecoder)
    codec.device = torch.device("cpu")
    codec.max_batch_size_encode = 4

    class FakeGen:
        mixed_precision = False
        num_latents = 4
        bottleneck_channels = 2

        def latents_to_codes(self, latents, n_quantizers=None):
            assert latents.shape == (1, 2, 4, 2)
            assert n_quantizers == 2
            return torch.ones(1, 2, 2, dtype=torch.long)

    codec.gen = FakeGen()
    monkeypatch.setattr(
        module,
        "encode_audio_inference",
        lambda *args, **kwargs: torch.zeros(1, 2, 4, 2),
    )

    codes = module.EncoderDecoder.encode(
        codec,
        torch.zeros(2, 16),
        discrete=True,
        n_quantizers=2,
    )

    assert codes.shape == (1, 2, 2)
    assert codes.dtype == torch.long


def test_encoder_decoder_discrete_decode_uses_rvq_helpers(monkeypatch):
    module = importlib.import_module("scripts.mossland-codec.inference")
    codec = object.__new__(module.EncoderDecoder)
    codec.device = torch.device("cpu")
    codec.max_batch_size_decode = 4

    calls = {}

    class FakeGen:
        num_latents = 4
        bottleneck_channels = 2

        def codes_to_latents(self, codes):
            calls["codes_shape"] = tuple(codes.shape)
            return torch.zeros(1, 2, 4, 2)

    codec.gen = FakeGen()
    monkeypatch.setattr(
        module,
        "decode_latent_inference",
        lambda latent, *args, **kwargs: latent,
    )

    out = module.EncoderDecoder.decode(codec, torch.ones(1, 2, 3, dtype=torch.long))

    assert calls["codes_shape"] == (1, 2, 3)
    assert out.shape == (1, 2, 4, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_encoder_decoder_discrete_encode_uses_rvq_helpers tests/mossland_codec/test_wrapper_imports.py::test_encoder_decoder_discrete_decode_uses_rvq_helpers
```

Expected: FAIL because inference still calls `self.gen.fsq`.

- [ ] **Step 3: Update `EncoderDecoder.encode()`**

In `scripts/mossland-codec/inference.py`, add an `n_quantizers` argument to `EncoderDecoder.encode()`:

```python
def encode(
    self,
    path_or_audio,
    max_batch_size=None,
    discrete=False,
    preprocess_on_gpu=True,
    desired_channels=64,
    fix_batch_size=False,
    n_quantizers: int | None = None,
):
```

In the discrete branch, replace the old quantized-FSQ path with a continuous-latent-to-codes path:

```python
if discrete:
    latents = encode_audio_inference(
        path_or_audio,
        self,
        max_batch_size,
        device=self.device,
        dont_quantize=True,
        preprocess_on_gpu=preprocess_on_gpu,
        fix_batch_size=fix_batch_size,
    )
    return self.gen.latents_to_codes(latents, n_quantizers=n_quantizers)
```

- [ ] **Step 4: Update `EncoderDecoder.decode()` and `decode_next()`**

Replace both direct FSQ calls:

```python
latents = self.gen.fsq.indexes_to_codes(latent)
```

with:

```python
latents = self.gen.codes_to_latents(latent)
```

For `decode_next(..., discrete=True)`, use the same helper. `decode_next()` still expects one timestep of codes and passes decoded latents to `decode_next_latent_inference()`.

- [ ] **Step 5: Update `infer_demo.py`**

Replace the direct FSQ decode:

```python
discrete_codes = codec.gen.fsq.indexes_to_codes(discrete_indexes)
```

with:

```python
discrete_latents = codec.gen.codes_to_latents(discrete_indexes)
```

Then generate both paths and save one concat comparison wav:

```python
continuous_latents = codec.encode(audio, discrete=False, preprocess_on_gpu=True)
discrete_indexes = codec.encode(
    audio,
    discrete=True,
    preprocess_on_gpu=True,
    n_quantizers=None,
)
discrete_latents = codec.gen.codes_to_latents(discrete_indexes)

discrete_generated = codec.decode(
    discrete_indexes,
    mode=MODE,
    denoising_steps=DENOISING_STEPS,
    preprocess_on_gpu=True,
    task_id=TASK_ID,
)
continuous_generated = codec.decode(
    continuous_latents,
    mode=MODE,
    denoising_steps=DENOISING_STEPS,
    preprocess_on_gpu=True,
    task_id=TASK_ID,
)

comparison = torch.cat(
    [
        audio.float().cpu(),
        discrete_generated.float().cpu(),
        continuous_generated.float().cpu(),
    ],
    dim=-1,
)
torchaudio.save(
    os.path.join(OUTPUT_DIR, f"{TASK_ID}_source_rvq_continuous.wav"),
    comparison,
    codec.gen.sample_rate,
)
```

Keep shape prints for `continuous_latents`, `discrete_indexes`, `discrete_latents`, `discrete_generated`, and `continuous_generated`. The standalone `TASK_ID_generated.wav` output is no longer needed for this demo file.

- [ ] **Step 6: Run inference tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_encoder_decoder_discrete_encode_uses_rvq_helpers tests/mossland_codec/test_wrapper_imports.py::test_encoder_decoder_discrete_decode_uses_rvq_helpers
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/mossland-codec/inference.py scripts/mossland-codec/infer_demo.py tests/mossland_codec/test_wrapper_imports.py
git commit -m "Route Mossland discrete inference through RVQ"
```

---

### Task 5: Update Hydra Configs and Documentation

**Files:**
- Modify: `scripts/configs/experiment/mossland-codec.yaml`
- Modify: `scripts/configs/experiment/mossland-codec-small.yaml`
- Modify: `scripts/configs/experiment/codicodec-paper-repro.yaml`
- Modify: `tests/mossland_codec/test_wrapper_imports.py`
- Modify: `docs/mossland-codec.md`
- Modify: `docs/code-index.md`
- Modify: `docs/memory/current.md`

- [ ] **Step 1: Write failing config tests**

In `test_mossland_experiment_config_points_to_self_contained_codec()`, replace the missing assertions around FSQ with:

```python
assert cfg.model.bottleneck_channels == 4
assert cfg.model.rvq_codebook_size == 1024
assert cfg.model.rvq_num_quantizers == 256
assert "fsq_levels" not in cfg.model
assert cfg.wrapper.rvq_path_dropout_prob == 0.6
assert cfg.wrapper.rvq_warmup_steps == 10000
assert cfg.wrapper.rvq_active_quantizers is None
assert cfg.wrapper.rvq_min_active_quantizers is None
assert cfg.wrapper.rvq_loss_weight == 1.0
assert "fsq_dropout_prob" not in cfg.wrapper
```

Add this focused small-config test:

```python
def test_mossland_codec_small_config_uses_rvq_bottleneck():
    config_dir = str((Path.cwd() / "scripts/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["experiment=mossland-codec-small"])

    assert cfg.model.rvq_codebook_size == 1024
    assert cfg.model.rvq_num_quantizers == 256
    assert cfg.model.bottleneck_channels == 4
    assert "fsq_levels" not in cfg.model
    assert cfg.wrapper.rvq_path_dropout_prob == 0.75
    assert cfg.wrapper.rvq_active_quantizers is None
    assert cfg.wrapper.rvq_min_active_quantizers is None
    assert "fsq_dropout_prob" not in cfg.wrapper
```

Add this paper-repro config test:

```python
def test_codicodec_paper_repro_config_uses_rvq_bottleneck():
    config_dir = str((Path.cwd() / "scripts/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["experiment=codicodec-paper-repro"])

    assert cfg.model._target_ == "scripts.mossland-codec.models.MosslandCodecTransformer"
    assert cfg.model.rvq_codebook_size == 1024
    assert cfg.model.rvq_num_quantizers == 256
    assert cfg.model.bottleneck_channels == 4
    assert "fsq_levels" not in cfg.model
    assert cfg.wrapper.rvq_path_dropout_prob == 0.75
    assert cfg.wrapper.rvq_active_quantizers is None
    assert cfg.wrapper.rvq_min_active_quantizers is None
    assert "fsq_dropout_prob" not in cfg.wrapper
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_mossland_experiment_config_points_to_self_contained_codec tests/mossland_codec/test_wrapper_imports.py::test_mossland_codec_small_config_uses_rvq_bottleneck tests/mossland_codec/test_wrapper_imports.py::test_codicodec_paper_repro_config_uses_rvq_bottleneck
```

Expected: FAIL because configs still use `fsq_levels` and `fsq_dropout_prob`.

- [ ] **Step 3: Update codec configs**

In each of these files:

- `scripts/configs/experiment/mossland-codec.yaml`
- `scripts/configs/experiment/mossland-codec-small.yaml`
- `scripts/configs/experiment/codicodec-paper-repro.yaml`

Replace:

```yaml
fsq_levels: [11, 11, 11, 11]
```

with:

```yaml
bottleneck_channels: 4
rvq_codebook_size: 1024
rvq_num_quantizers: 256
rvq_bound_eps: 1e-5
rvq_codebook_loss_weight: 1.0
rvq_commitment_loss_weight: 0.0
```

Replace wrapper `fsq_dropout_prob` with `rvq_path_dropout_prob` and add the RVQ loss and dynamic layer-count controls:

```yaml
rvq_path_dropout_prob: 0.6
rvq_warmup_steps: 10000
rvq_active_quantizers: null
rvq_min_active_quantizers: null
rvq_loss_weight: 1.0
rvq_codebook_loss_weight: 1.0
rvq_commitment_loss_weight: 0.0
```

For configs that currently use `fsq_dropout_prob: 0.75`, preserve that path probability as:

```yaml
rvq_path_dropout_prob: 0.75
rvq_warmup_steps: 10000
rvq_active_quantizers: null
rvq_min_active_quantizers: null
rvq_loss_weight: 1.0
rvq_codebook_loss_weight: 1.0
rvq_commitment_loss_weight: 0.0
```

Do not edit `scripts/configs/experiment/codicodec.yaml`; it targets the separate `scripts/codicodec` reference.

- [ ] **Step 4: Update persistent docs**

In `docs/mossland-codec.md`, replace the compression sentence:

```text
压缩行为保持不变：`hop=1024`、`fac=2`、`spec_length=32`、`num_latents=128`、`fsq_levels=[11,11,11,11]` 和 `frontend_freq_downsample_list=[0,1,0]` 不变；实测实例化后 `data_length=512`、`freq_dim=64`、`time_dim=8`、`downsample_ratio=64`。
```

with:

```text
压缩时间布局保持不变：`hop=1024`、`fac=2`、`spec_length=32`、`num_latents=128`、`bottleneck_channels=4` 和 `frontend_freq_downsample_list=[0,1,0]` 不变；实测实例化后 `data_length=512`、`freq_dim=64`、`time_dim=8`、`downsample_ratio=64`。离散瓶颈从 FSQ 改为 chunk-level RVQ：每个 chunk 把 `[128,4]` 展平成 512 维，默认 `rvq_codebook_size=1024`、最大 `rvq_num_quantizers=256`，使用全部层时约 `2560 bits/chunk`；训练和推理可指定 `n_quantizers` 使用更少层数。
```

In `docs/code-index.md`, update the `scripts/mossland-codec/` and `mossland-codec.yaml` bullets to mention RVQ config names instead of `fsq_levels`.

In `docs/memory/current.md`, add a short top bullet under `## 当前工作`:

```text
- 2026-06-17 RVQ bottleneck 设计已确认并进入实施计划：`scripts/mossland-codec` 将删除本地 FSQ，新增 chunk-level RVQ；每个 chunk 的 `[128,4]` bottleneck latent 展平成 512 维，默认 `codebook_size=1024`、最大 `num_quantizers=256`，使用全部层时约为原 FSQ 码率 1.45 倍；需像 DAC 一样支持动态 `n_quantizers <= 256`。demo 输出每个样本只保存一个 concat wav，顺序为 `src -> target -> rvq/discrete decode -> continuous decode`。计划文档见 `docs/superpowers/plans/2026-06-17-mossland-codec-rvq-implementation.md`。
```

- [ ] **Step 5: Run config tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_mossland_experiment_config_points_to_self_contained_codec tests/mossland_codec/test_wrapper_imports.py::test_mossland_codec_small_config_uses_rvq_bottleneck tests/mossland_codec/test_wrapper_imports.py::test_codicodec_paper_repro_config_uses_rvq_bottleneck
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/configs/experiment/mossland-codec.yaml scripts/configs/experiment/mossland-codec-small.yaml scripts/configs/experiment/codicodec-paper-repro.yaml tests/mossland_codec/test_wrapper_imports.py docs/mossland-codec.md docs/code-index.md docs/memory/current.md
git commit -m "Configure Mossland codec RVQ bottleneck"
```

---

### Task 6: Final Verification and Cleanup

**Files:**
- Modify only if verification reveals a concrete issue in files touched by Tasks 1-5.

- [ ] **Step 1: Run Python compile checks**

Run:

```bash
python -m py_compile scripts/mossland-codec/quantization.py scripts/mossland-codec/models.py scripts/mossland-codec/wrapper.py scripts/mossland-codec/inference.py scripts/mossland-codec/infer_demo.py
```

Expected: command exits with status 0 and no output.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_quantization.py tests/mossland_codec/test_wrapper_imports.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run existing codec smoke tests**

Run:

```bash
python -m pytest -q tests/mossland_codec/test_tasks.py tests/mossland_codec/test_training_base.py tests/mossland_codec/test_attention.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Search for stale FSQ references in Mossland codec code/config**

Run:

```bash
grep -RInE "class FSQ|round_ste|fsq_levels|fsq_dropout_prob|\\.fsq" \
  scripts/mossland-codec \
  scripts/configs/experiment/mossland-codec.yaml \
  scripts/configs/experiment/mossland-codec-small.yaml \
  scripts/configs/experiment/codicodec-paper-repro.yaml
```

Expected: no output. If output appears in generated caches under `__pycache__`, remove the cache directory rather than editing source.

- [ ] **Step 5: Review final diff**

Run:

```bash
git status --short
git diff --stat HEAD
git diff -- scripts/mossland-codec/quantization.py scripts/mossland-codec/models.py scripts/mossland-codec/wrapper.py scripts/mossland-codec/inference.py scripts/mossland-codec/infer_demo.py
```

Expected: only RVQ implementation, tests, config, and documentation files are modified. Existing unrelated changes in `bash/push_oss_files.sh`, `bash/rclone_files.sh`, or `docs/memory/progress.md` should remain unstaged unless the user explicitly asks to include them.

- [ ] **Step 6: Commit verification fixes if needed**

If Task 6 required edits, commit only those edits:

```bash
git add <files-edited-during-task-6>
git commit -m "Verify Mossland RVQ bottleneck integration"
```

If no edits were needed, do not create an empty commit.

---

## Execution Notes

- Use TDD for Tasks 1-5: write or update the named test first, run the focused command and observe failure, implement the smallest change, then rerun the focused command.
- Commit after each task. Do not stage pre-existing unrelated changes.
- `scripts/codicodec/` keeps its FSQ implementation as a reference baseline.
- A2A configs and tests can still say `use_fsq=false`; those refer to the A2A no-latent-bottleneck model and are outside this implementation.
