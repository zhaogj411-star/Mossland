# Mossland Codec Time-Aligned RVQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Mossland codec FSQ chunk bottleneck with 25Hz time-aligned continuous tokens and dynamic RVQ codes.

**Architecture:** The model should encode audio into `[B, T_token, 32]` continuous time tokens where each token maps to a fixed 40 ms span at 48 kHz. RVQ quantizes each token independently with up to 16 codebooks of size 1024 and supports DAC-style dynamic `n_quantizers`. The first decoder integration keeps the existing diffusion decoder/windowing machinery by adapting consecutive time tokens back into the decoder conditioning/features internally, while the public encode/decode interface no longer depends on the old learned-query chunk boundary.

**Tech Stack:** PyTorch, Lightning, Hydra, existing `scripts/mossland-codec` Transformer/frontend modules, `pytest`.

---

## File Structure

- Create `scripts/mossland-codec/quantization.py`: self-contained RVQ module and output container. No imports from `scripts.codicodec`.
- Create `scripts/mossland-codec/time_tokens.py`: local Transformer token encoder and token-to-window adapter helpers. This keeps token-rate logic out of `models.py`.
- Modify `scripts/mossland-codec/models.py`: remove the local `FSQ` class, add RVQ/time-token constructor parameters, expose continuous/RVQ token APIs, and adapt `encoder_forward()` / `pre_decoder_forward()` to time-aligned latents.
- Modify `scripts/mossland-codec/wrapper.py`: rename FSQ dropout semantics to RVQ path dropout, train through continuous or RVQ path, log RVQ reconstruction/codebook metrics, and write a single demo concat wav per sample.
- Modify `scripts/mossland-codec/inference.py`: return continuous tokens as `[channels, token_dim, time]`, return discrete RVQ codes as `[time, n_quantizers]` per batch/audio item, and decode dynamic `n_quantizers`.
- Modify `scripts/mossland-codec/infer_demo.py`: save one concat wav containing `src -> target/input -> rvq decode -> continuous decode`.
- Modify `scripts/configs/experiment/mossland-codec.yaml`: set `sample_rate=48000`, `latent_rate_hz=25`, `token_dim=32`, `rvq_max_quantizers=16`, `rvq_codebook_size=1024`, and RVQ dropout/warmup parameters.
- Modify `tests/mossland_codec/test_wrapper_imports.py`: replace FSQ assertions with RVQ/time-token assertions and demo concat behavior.
- Create `tests/mossland_codec/test_rvq.py`: focused RVQ unit tests.
- Create `tests/mossland_codec/test_time_tokens.py`: focused time-token shape/rate tests.
- Update `docs/README.md`, `docs/code-index.md`, `docs/mossland-codec.md`, and `docs/memory/current.md` after behavior changes.

## Task 1: RVQ Module

**Files:**
- Create: `scripts/mossland-codec/quantization.py`
- Test: `tests/mossland_codec/test_rvq.py`

- [ ] **Step 1: Write failing RVQ shape and dynamic-depth tests**

Add:

```python
import importlib

import pytest
import torch


def test_rvq_quantizes_token_sequence_with_dynamic_depth():
    module = importlib.import_module("scripts.mossland-codec.quantization")
    rvq = module.ResidualVectorQuantizer(
        dim=4,
        codebook_size=8,
        num_quantizers=3,
    )
    tokens = torch.randn(2, 5, 4)

    full = rvq(tokens)
    partial = rvq(tokens, n_quantizers=2)

    assert full.quantized.shape == tokens.shape
    assert full.codes.shape == (2, 5, 3)
    assert full.loss.shape == ()
    assert partial.quantized.shape == tokens.shape
    assert partial.codes.shape == (2, 5, 2)
    assert torch.all(partial.codes >= 0)
    assert torch.all(partial.codes < 8)


def test_rvq_rejects_invalid_dynamic_depth():
    module = importlib.import_module("scripts.mossland-codec.quantization")
    rvq = module.ResidualVectorQuantizer(dim=4, codebook_size=8, num_quantizers=3)
    tokens = torch.randn(1, 2, 4)

    with pytest.raises(ValueError, match="n_quantizers"):
        rvq(tokens, n_quantizers=0)
    with pytest.raises(ValueError, match="n_quantizers"):
        rvq(tokens, n_quantizers=4)
```

- [ ] **Step 2: Run the RVQ tests and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_rvq.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.mossland-codec.quantization'`.

- [ ] **Step 3: Implement minimal RVQ**

Create `scripts/mossland-codec/quantization.py` with:

```python
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
                f"codes last dimension must be in [1, {self.num_quantizers}], got {codes.shape[-1]}"
            )
        if codes.min().item() < 0 or codes.max().item() >= self.codebook_size:
            raise ValueError("RVQ codes are outside the codebook range")
        flat_codes = codes.reshape(-1, codes.shape[-1]).long()
        quantized = torch.zeros(flat_codes.shape[0], self.dim, device=codes.device)
        for idx in range(flat_codes.shape[-1]):
            quantized = quantized + F.embedding(flat_codes[:, idx], self.codebooks[idx])
        return quantized.reshape(*codes.shape[:-1], self.dim)
```

- [ ] **Step 4: Run the RVQ tests and confirm GREEN**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_rvq.py
```

Expected: PASS.

## Task 2: Time-Token Encoder Helpers

**Files:**
- Create: `scripts/mossland-codec/time_tokens.py`
- Test: `tests/mossland_codec/test_time_tokens.py`

- [ ] **Step 1: Write failing token-rate tests**

Add:

```python
import importlib

import pytest
import torch


def test_token_count_uses_sample_rate_and_latent_rate():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")

    assert module.samples_per_token(48000, 25) == 1920
    assert module.token_count_for_samples(48000, 48000, 25) == 25


def test_local_token_encoder_preserves_time_axis():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")
    encoder = module.LocalTimeTokenEncoder(
        input_dim=6,
        token_dim=4,
        num_layers=1,
        num_heads=2,
        local_window_tokens=2,
    )
    x = torch.randn(2, 7, 6)

    out = encoder(x)

    assert out.shape == (2, 7, 4)


def test_samples_per_token_rejects_non_integer_rate():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")

    with pytest.raises(ValueError, match="integer"):
        module.samples_per_token(44100, 25)
```

- [ ] **Step 2: Run the time-token tests and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_time_tokens.py
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement local token helpers**

Create `scripts/mossland-codec/time_tokens.py` with functions `samples_per_token()`, `token_count_for_samples()`, and class `LocalTimeTokenEncoder`. Use `nn.TransformerEncoderLayer(batch_first=True)` with an additive local mask for the first implementation; the mask is acceptable here because the 25Hz token sequence is much shorter than frontend patch tokens.

- [ ] **Step 4: Run the time-token tests and confirm GREEN**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_time_tokens.py
```

Expected: PASS.

## Task 3: Model API and Shape Migration

**Files:**
- Modify: `scripts/mossland-codec/models.py`
- Test: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing model assertions**

In `tests/mossland_codec/test_wrapper_imports.py`, replace FSQ-specific assertions with:

```python
def test_mossland_model_exposes_time_aligned_rvq_configuration():
    module = importlib.import_module("scripts.mossland-codec.models")
    model = module.MosslandCodecTransformer(
        dim=8,
        head_dim=4,
        cond_channels=8,
        num_layers=1,
        num_layers_encoder=1,
        num_more_latents=0,
        token_dim=4,
        latent_rate_hz=25,
        rvq_codebook_size=8,
        rvq_max_quantizers=3,
        use_rvq=True,
        frontend_base_channels=4,
        frontend_multipliers_list=[1],
        frontend_layers_list=[1],
        frontend_encoder_layers_list=[1],
        frontend_freq_downsample_list=[],
        hop=8,
        fac=2,
        sample_rate=200,
        spec_length=2,
    )

    assert not hasattr(model, "fsq")
    assert model.rvq.num_quantizers == 3
    assert model.token_dim == 4
    assert model.latent_rate_hz == 25
```

- [ ] **Step 2: Run the model assertion and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_mossland_model_exposes_time_aligned_rvq_configuration
```

Expected: FAIL because `token_dim` / `use_rvq` are not constructor parameters yet.

- [ ] **Step 3: Add constructor parameters and aliases**

Modify `MosslandCodecTransformer.__init__()`:

- Add `latent_rate_hz: int = 25`
- Add `token_dim: int = 32`
- Add `rvq_codebook_size: int = 1024`
- Add `rvq_max_quantizers: int = 16`
- Add `use_rvq: bool = True`
- Do not keep `use_fsq` / `fsq_levels` compatibility in the main codec path.
- Derive internal `decoder_window_tokens` from `latent_rate_hz`, `sample_rate`, `hop`, `fac`, and `spec_length`. This is only an internal diffusion decoder window; public encode output remains an arbitrary-length `[B, T_token, token_dim]` sequence.
- Instantiate `self.rvq = ResidualVectorQuantizer(token_dim, rvq_codebook_size, rvq_max_quantizers)` when `use_rvq=True`, else `None`.

- [ ] **Step 4: Run focused model assertion and import suite**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py
```

Expected: PASS after updating old FSQ tests and any source-text assertions.

## Task 4: Token Encoder and Decoder Adapter

**Files:**
- Modify: `scripts/mossland-codec/models.py`
- Test: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing encode/decode shape test**

Add a small-model test that builds a short waveform, calls `to_representation_encoder()`, `encoder_forward(..., quantize=False)`, and `pre_decoder_forward()`, and asserts:

```python
assert latents.shape[-1] == model.token_dim
assert latents.shape[-2] > 0
assert features
```

- [ ] **Step 2: Run the shape test and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_time_token_encoder_feeds_decoder_features
```

Expected: FAIL until `encoder_forward()` emits `token_dim` tokens and `pre_decoder_forward()` accepts them.

- [ ] **Step 3: Implement encoder time-token path**

Change `encoder_forward()` and `encoder_forward_fast()` so they:

- Encode the full representation time axis and produce arbitrary-length ordered time tokens.
- Run `frontend_encoder_down()` to get `[B, F_down, T_down, dim]`.
- Pool/project frequency into `[B, T_down, dim]`.
- Resample or group the time axis to the configured `latent_rate_hz`.
- Run `LocalTimeTokenEncoder` and a final projection to `[B, T_token, token_dim]`.
- If RVQ is active and `quantize=True`, return `self.rvq(tokens).quantized`; otherwise return continuous tokens.

- [ ] **Step 4: Implement decoder adapter**

Change `pre_decoder_forward(latents)` so it accepts `[B, T_token, token_dim]`, projects tokens to `dim`, windows consecutive tokens as needed, and feeds the existing `pre_decoder` / `frontend_pre_decoder_up()` path. The assertion must be based on token-window size, not the old learned-query latent assumption.

- [ ] **Step 5: Run focused and import tests**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py tests/mossland_codec/test_time_tokens.py tests/mossland_codec/test_rvq.py
```

Expected: PASS.

## Task 5: Training Wrapper RVQ Path

**Files:**
- Modify: `scripts/mossland-codec/wrapper.py`
- Test: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing wrapper API tests**

Add assertions that `MosslandCodecTrainingWrapper.__init__` accepts `rvq_dropout_prob`, `rvq_warmup_steps`, `rvq_active_quantizers`, and does not expose `_fsq_dropout_active`.

- [ ] **Step 2: Run wrapper API test and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_training_wrapper_uses_rvq_path_controls
```

Expected: FAIL until wrapper parameters are renamed/implemented.

- [ ] **Step 3: Implement RVQ training switch**

Replace `_fsq_dropout_active()` with `_continuous_path_active()` or `_rvq_path_active()`. In `training_step()`, encode continuous tokens first, then choose:

```python
continuous_tokens = self.model.encoder_forward(src_representation, quantize=False)
rvq_output = self.model.quantize_tokens(continuous_tokens, n_quantizers=active_quantizers)
latents = continuous_tokens if use_continuous else rvq_output.quantized
loss = consistency_loss + rvq_loss_weight * rvq_output.loss
```

Use `continuous_tokens.detach()` inside RVQ codebook loss if the chosen mode is codebook-only warmup.

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py
```

Expected: PASS.

## Task 6: Inference Dynamic Codes

**Files:**
- Modify: `scripts/mossland-codec/inference.py`
- Modify: `scripts/mossland-codec/infer_demo.py`
- Test: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Write failing inference source/API assertions**

Assert `EncoderDecoder.encode()` has `n_quantizers` in its signature and source no longer references `.fsq`.

- [ ] **Step 2: Run inference test and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_inference_uses_rvq_dynamic_codebooks
```

Expected: FAIL until `inference.py` is migrated.

- [ ] **Step 3: Implement inference migration**

- `encode(discrete=True, n_quantizers=None)` calls model continuous encode, then `model.quantize_tokens(..., n_quantizers=n_quantizers).codes`.
- `decode()` detects integer tensors as RVQ codes and calls `model.codes_to_tokens(codes)`.
- Continuous decode uses token tensors directly and no longer calls `dim2latents()` unless compatibility is explicitly required for old checkpoints.
- `decode_next()` applies the same conversion.

- [ ] **Step 4: Run inference/import tests**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py
```

Expected: PASS.

## Task 7: Demo Single Concat WAV

**Files:**
- Modify: `scripts/mossland-codec/wrapper.py`
- Modify: `scripts/mossland-codec/infer_demo.py`
- Test: `tests/mossland_codec/test_wrapper_imports.py`

- [ ] **Step 1: Update demo callback test**

Change the fake model test so it expects one `torchaudio.save()` per sample and the filename contains `_src_target_rvq_continuous.wav`.

- [ ] **Step 2: Run demo test and confirm RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py::test_demo_callback_writes_single_discrete_and_continuous_concat
```

Expected: FAIL because the callback currently writes two separate files.

- [ ] **Step 3: Implement concat behavior**

Generate `rvq_generated` with quantized/RVQ path and `continuous_generated` with continuous path, then call:

```python
comparison = self._concat_demo_audio(src_item, target_item, rvq_item, continuous_item)
```

Save to `f"{base}_src_target_rvq_continuous.wav"`.

- [ ] **Step 4: Run demo tests**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py
```

Expected: PASS.

## Task 8: Config and Docs

**Files:**
- Modify: `scripts/configs/experiment/mossland-codec.yaml`
- Modify: `docs/README.md`
- Modify: `docs/code-index.md`
- Modify: `docs/mossland-codec.md`
- Modify: `docs/memory/current.md`

- [ ] **Step 1: Update Hydra config**

Set:

```yaml
data:
  dataset:
    dataset:
      sample_size: 96000
      sample_rate: 48000
    sample_rate: 48000
model:
  sample_rate: 48000
  latent_rate_hz: 25
  token_dim: 32
  rvq_codebook_size: 1024
  rvq_max_quantizers: 16
  use_rvq: true
wrapper:
  rvq_dropout_prob: 0.6
  rvq_warmup_steps: 10000
  rvq_loss_weight: 1.0
callbacks:
  demo_callback:
    sample_rate: 48000
```

Remove `fsq_levels` and `fsq_dropout_prob` from the active config.

- [ ] **Step 2: Update persistent docs**

Document that `scripts/mossland-codec` now uses time-aligned RVQ and old FSQ checkpoints are not compatible with the main codec path.

- [ ] **Step 3: Run config/import checks**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py tests/mossland_codec/test_rvq.py tests/mossland_codec/test_time_tokens.py
```

Expected: PASS.

## Task 9: Final Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q tests/mossland_codec/test_wrapper_imports.py tests/mossland_codec/test_attention.py tests/mossland_codec/test_training_base.py tests/mossland_codec/test_rvq.py tests/mossland_codec/test_time_tokens.py
```

Expected: PASS.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
python -m py_compile scripts/mossland-codec/models.py scripts/mossland-codec/quantization.py scripts/mossland-codec/time_tokens.py scripts/mossland-codec/wrapper.py scripts/mossland-codec/inference.py scripts/mossland-codec/infer_demo.py
```

Expected: PASS.

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff -- scripts/mossland-codec scripts/configs/experiment/mossland-codec.yaml tests/mossland_codec docs/README.md docs/code-index.md docs/mossland-codec.md docs/memory/current.md
```

Expected: diff only contains the time-aligned RVQ implementation and docs updates.
