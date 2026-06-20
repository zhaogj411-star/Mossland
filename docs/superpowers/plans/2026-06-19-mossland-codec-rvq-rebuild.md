# Mossland Codec RVQ Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old codec implementation with a minimal Mossland RVQ package while preserving the old package as `scripts/mossland_codec_old`.

**Architecture:** The new package contains only the 0.5B stereo RVQ codec code plus a compact multi-task dataset adapter. Training uses source audio to produce conditioning latents and target audio as the denoising target, enabling reconstruct, separation, super-resolution, and mono-to-stereo tasks without carrying over the old codec model/eval stack.

**Tech Stack:** PyTorch, Lightning, Hydra, `vector_quantize_pytorch`, existing `scripts.data.datasets` datasets.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/mossland_codec/test_mossland_codec_rebuild.py`

- [ ] Add tests that assert `mossland-codec.yaml` points to `scripts.mossland_codec.models.MosslandCodec`, `scripts.mossland_codec.wrapper.MosslandCodecTrainingWrapper`, and `scripts.mossland_codec.tasks.MosslandTaskDataset`.
- [ ] Add tests that instantiate `MosslandTaskDataset` around a tiny tensor dataset and verify it returns `src`, `target`, and `task_id`.
- [ ] Add tests that assert the new package does not contain `eval_benchmark`, `transformer.py`, or `transformer_layers.py`.

### Task 2: Package Move

**Files:**
- Move the current old codec package into `scripts/mossland_codec_old`
- Create: `scripts/mossland_codec/`

- [ ] Move the old package out of the import path.
- [ ] Create the new package directory with `__init__.py`.
- [ ] Do not copy old eval or transformer files into the new package.

### Task 3: Minimal Mossland Codec Code

**Files:**
- Create: `scripts/mossland_codec/audio.py`
- Create: `scripts/mossland_codec/models.py`
- Create: `scripts/mossland_codec/quantize.py`
- Create: `scripts/mossland_codec/training_base.py`

- [ ] Copy the relevant STFT, model, RVQ, and training base code.
- [ ] Keep imports relative inside the new package.
- [ ] Exclude SAME/NATTEN/local attention experiments from the new model file.

### Task 4: Multi-Task Training Wrapper

**Files:**
- Create: `scripts/mossland_codec/tasks.py`
- Create: `scripts/mossland_codec/wrapper.py`

- [ ] Copy the compact task adapter from the old package.
- [ ] Implement `MosslandCodecTrainingWrapper` from the RVQ wrapper semantics.
- [ ] Make training consume `MosslandTaskBatch`: encode `src`, denoise `target`, and log task ratios.
- [ ] Make demo save `src -> target -> discrete rates -> continuous`.

### Task 5: Hydra Config and Docs

**Files:**
- Modify: `scripts/configs/experiment/mossland-codec.yaml`
- Modify: `docs/code-index.md`
- Modify: `docs/mossland-codec.md`
- Modify: `docs/memory/current.md`

- [ ] Update `mossland-codec.yaml` to use the new package and 0.5B Mossland RVQ model parameters.
- [ ] Keep multi-task data structure from old `mossland-codec.yaml`.
- [ ] Record the package move and new scope in docs.

### Task 6: Verification

- [ ] Run `python -m pytest tests/mossland_codec/test_mossland_codec_rebuild.py -q`.
- [ ] Run `python -m py_compile scripts/mossland_codec/*.py`.
- [ ] Run Hydra compose checks for `experiment=mossland-codec`.
- [ ] Confirm `scripts/mossland_codec` lacks old eval/transformer files.
