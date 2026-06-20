# Music2Latent Axial Local Denoiser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Music2Latent variant whose denoising U-Net ResBlocks use SAME-style chunked axial local Transformer blocks without depending on NATTEN.

**Architecture:** Keep the official Music2Latent encoder, latent pyramid decoder, wrapper, loss, and down/up resampling layers. Copy the reusable SAME Transformer/sliding-window components into `scripts/music2latent/local_transformer.py`, then expose `Music2Latent_axiallocal` from `scripts/music2latent/models.py` by replacing denoising U-Net `ResBlock` modules with axial local attention ResBlocks. After H200 OOM, the default is stage-wise replacement with `axial_min_channels=256`; full high-resolution replacement requires `model.axial_min_channels=0` and is diagnostic only.

**Tech Stack:** PyTorch, einops, Hydra configs, pytest.

---

### Task 1: Add Red Tests

**Files:**
- Modify: `tests/music2latent/test_music2latent.py`

- [x] Add tests that import `scripts.music2latent.local_transformer`, verify an axial ResBlock starts as identity when zero-initialized, and verify a small `Music2Latent_axiallocal` model runs a forward pass.
- [x] Run `python -m pytest tests/music2latent/test_music2latent.py::test_axial_local_resblock_starts_as_identity tests/music2latent/test_music2latent.py::test_music2latent_axiallocal_forward_shape -q`.
- [x] Expected before implementation: fail because `scripts.music2latent.local_transformer` and `Music2Latent_axiallocal` do not exist.

### Task 2: Copy SAME-Style Local Transformer Components

**Files:**
- Create: `scripts/music2latent/local_transformer.py`

- [x] Copy/adapt SAME sliding-window SDPA, normalization, feed-forward, attention, and `TransformerBlock` components from `scripts/same/transformer.py`.
- [x] Add `AxialLocalAttention2d`, `AxialResidualAttention2d`, and `AxialLocalAttentionResBlock`.
- [x] Keep the API channel-first for Music2Latent: inputs and outputs are `[B, C, F, T]`.

### Task 3: Expose New Music2Latent Target

**Files:**
- Modify: `scripts/music2latent/models.py`
- Add: `scripts/configs/experiment/music2latent_axiallocal.yaml`

- [x] Add `Music2Latent_axiallocal` as a subclass of official `Music2Latent`.
- [x] After `super().__init__`, replace `ResBlock` instances in `self.down_layers` and `self.up_layers` with `AxialLocalAttentionResBlock`, gated by `axial_min_channels`.
- [x] Add a Hydra config mirroring `music2latent.yaml` with `_target_: scripts.music2latent.models.Music2Latent_axiallocal` and axial-local parameters.

### Task 4: Verify

**Files:**
- Test: `tests/music2latent/test_music2latent.py`
- Compile: `scripts/music2latent/local_transformer.py`, `scripts/music2latent/models.py`

- [x] Run the two new targeted tests.
- [x] Run `python -m py_compile scripts/music2latent/local_transformer.py scripts/music2latent/models.py`.
- [x] If CUDA is available, run a one-sample forward smoke for the new Hydra target.
