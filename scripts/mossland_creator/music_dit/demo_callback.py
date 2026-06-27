"""Demo callback for Music-DiT flow-matching training."""

from __future__ import annotations

import gc
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from megatron.energon import DefaultTaskEncoder, Sample, SkipSample, WorkerConfig
from megatron.energon import get_savable_loader, get_train_dataset
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker

from scripts.mossland_creator.codecs.same import SAMECodec

from .core import contract as C
from .core.patchify import patchify_1d
from .core.pos_emb import build_pos_ids_1d
from .data.taskencoder import MusicDiffusionTaskEncoder, cook_music
from .diffusion import MusicFlowMatching, MusicFlowMatchingEulerSampler


DEMO_STEMS = ("mixture", "vocals", "accompaniment")


def _safe_name(value: object, fallback: str = "sample") -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-._")
    return text[:96] or fallback


@dataclass
class MusicDiTDemoSample(Sample):
    mixture: torch.Tensor
    vocals: torch.Tensor
    accompaniment: torch.Tensor
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    pos_ids: torch.Tensor
    seq_len_q: torch.Tensor
    prompt_text: object
    frame_rate: torch.Tensor


class MusicDiTDemoTaskEncoder(DefaultTaskEncoder):
    """Return one sample with all stems needed by the demo callback."""

    cookers = [Cooker(cook_music)]

    def __init__(
        self,
        *args,
        max_duration_seconds: float | None = None,
        patch_t: int = 1,
        text_embedding_padding_size: int = 3500,
        max_abs_latent: float = 1e3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_duration_seconds = None if max_duration_seconds is None else float(max_duration_seconds)
        self.patch_t = int(patch_t)
        self.text_embedding_padding_size = int(text_embedding_padding_size)
        self.max_abs_latent = float(max_abs_latent)

    def _tokenize_stem(self, latent: torch.Tensor, seq_length: int) -> torch.Tensor:
        latent = latent[:, : seq_length * self.patch_t]
        pad_frames = seq_length * self.patch_t - latent.shape[-1]
        if pad_frames > 0:
            latent = F.pad(latent, (0, pad_frames))
        return patchify_1d(latent, self.patch_t).float()

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> MusicDiTDemoSample:
        latent_bank = MusicDiffusionTaskEncoder._trim_to_common_frames(
            MusicDiffusionTaskEncoder._coerce_latent_bank(sample["latent_bank"])
        )
        if any(stem not in latent_bank for stem in DEMO_STEMS):
            raise SkipSample()
        if any(torch.isnan(latent).any() or torch.isinf(latent).any() for latent in latent_bank.values()):
            raise SkipSample()
        if max(latent.abs().max().item() for latent in latent_bank.values()) > self.max_abs_latent:
            raise SkipSample()

        meta = dict(sample.get("json", {}) or {})
        frame_rate = float(meta.get("frame_rate") or SAMECodec.sample_rate / SAMECodec.hop_length)
        true_frames = min(latent_bank[stem].shape[-1] for stem in DEMO_STEMS)
        if self.max_duration_seconds is not None:
            max_frames = max(1, int(self.max_duration_seconds * frame_rate))
            true_frames = min(true_frames, max_frames)
        seq_length = max(1, math.ceil(true_frames / self.patch_t))

        text = MusicDiffusionTaskEncoder._coerce_text_embedding(sample["text"])
        valid = min(text.shape[0], self.text_embedding_padding_size)
        if text.shape[0] > self.text_embedding_padding_size:
            text = text[: self.text_embedding_padding_size]
        else:
            text = F.pad(text, (0, 0, 0, self.text_embedding_padding_size - text.shape[0]))
        text_mask = torch.zeros(self.text_embedding_padding_size, dtype=torch.float32)
        text_mask[:valid] = 1.0

        prompt = meta.get("prompt")
        if not prompt:
            prompt = f"caption: {meta.get('caption', '')} lyric: {meta.get('lyric', '')}".strip()

        return MusicDiTDemoSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__"),
            __sources__=sample.get("__sources__"),
            mixture=self._tokenize_stem(latent_bank["mixture"], seq_length),
            vocals=self._tokenize_stem(latent_bank["vocals"], seq_length),
            accompaniment=self._tokenize_stem(latent_bank["accompaniment"], seq_length),
            crossattn_emb=text.float(),
            crossattn_mask=text_mask,
            pos_ids=build_pos_ids_1d(seq_length),
            seq_len_q=torch.tensor(seq_length, dtype=torch.int32),
            prompt_text=prompt,
            frame_rate=torch.tensor(frame_rate, dtype=torch.float32),
        )

    @stateless
    def batch(self, samples: list[MusicDiTDemoSample]) -> dict:
        def stack(key: str) -> torch.Tensor:
            return torch.stack([getattr(sample, key) for sample in samples], dim=0)

        return {
            "__key__": [sample.__key__ for sample in samples],
            "mixture": stack("mixture"),
            "vocals": stack("vocals"),
            "accompaniment": stack("accompaniment"),
            C.KEY_CROSSATTN: stack("crossattn_emb"),
            C.KEY_CROSSATTN_MASK: stack("crossattn_mask"),
            C.KEY_POS_IDS: stack("pos_ids"),
            C.KEY_SEQLEN_Q: stack("seq_len_q"),
            "prompt_text": [str(sample.prompt_text) for sample in samples],
            "frame_rate": stack("frame_rate"),
        }


class MusicDiTFlowDemoCallback(pl.Callback):
    """Generate grouped text and audio demos during Music-DiT training.

    For each demo sample the callback writes four files with the same sortable
    prefix: prompt text, text-only generation, text+vocals generation, and
    text+accompaniment generation.
    """

    def __init__(
        self,
        demo_dir: str,
        dataset_path: str,
        demo_num: int = 1,
        demo_every: int = 1000,
        demo_start_step: int = 1,
        duration_seconds: float | None = None,
        num_steps: int = 32,
        guidance_scale: float = 1.0,
        demo_seed: int = 1234,
        codec_variant: str = "same-l",
        split_part: str = "train",
        text_embedding_padding_size: int = 3500,
        patch_t: int = 1,
        max_abs_latent: float = 1e3,
        save_format: str = "mp3",
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.dataset_path = dataset_path
        self.demo_num = int(demo_num)
        self.demo_every = int(demo_every)
        self.demo_start_step = int(demo_start_step)
        self.duration_seconds = None if duration_seconds is None else float(duration_seconds)
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.demo_seed = int(demo_seed)
        self.codec_variant = codec_variant
        self.split_part = split_part
        self.text_embedding_padding_size = int(text_embedding_padding_size)
        self.patch_t = int(patch_t)
        self.max_abs_latent = float(max_abs_latent)
        self.save_format = save_format
        self.last_demo_step = -1
        self._codec: Optional[SAMECodec] = None
        self._demo_iterator = None

    def _should_run_demo(self, global_step: int) -> bool:
        if self.demo_every <= 0 or global_step < self.demo_start_step:
            return False
        if self.last_demo_step == global_step:
            return False
        return (global_step - self.demo_start_step) % self.demo_every == 0

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _capture_rng_state(self):
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return torch.random.get_rng_state(), cuda_state

    def _restore_rng_state(self, state) -> None:
        cpu_state, cuda_state = state
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _demo_loader(self):
        encoder = MusicDiTDemoTaskEncoder(
            max_duration_seconds=self.duration_seconds,
            patch_t=self.patch_t,
            text_embedding_padding_size=self.text_embedding_padding_size,
            max_abs_latent=self.max_abs_latent,
        )
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        dataset = get_train_dataset(
            self.dataset_path,
            split_part=self.split_part,
            worker_config=worker_config,
            batch_size=self.demo_num,
            shuffle_buffer_size=1,
            max_samples_per_sequence=1,
            task_encoder=encoder,
            repeat=False,
        )
        return get_savable_loader(dataset, prefetch_factor=2)

    def _next_demo_batch(self) -> dict:
        if self._demo_iterator is None:
            self._demo_iterator = iter(self._demo_loader())
        try:
            return next(self._demo_iterator)
        except StopIteration:
            self._demo_iterator = iter(self._demo_loader())
            return next(self._demo_iterator)

    def _codec_for_device(self, device: torch.device) -> SAMECodec:
        device_name = str(device)
        if self._codec is None or self._codec.device != device_name:
            self._codec = SAMECodec(variant=self.codec_variant, device=device_name)
        return self._codec

    def _make_generation_batch(
        self,
        demo_batch: dict,
        index: int,
        context_key: Optional[str],
        device: torch.device,
        seq_len: int,
    ) -> dict[str, torch.Tensor]:
        if context_key is None:
            context = torch.zeros(1, seq_len, demo_batch["mixture"].shape[-1], device=device)
            cond_mask = torch.zeros(1, seq_len, 1, device=device)
        else:
            context = demo_batch[context_key][index : index + 1, :seq_len].to(device)
            cond_mask = torch.ones(1, seq_len, 1, device=device)
        return {
            C.KEY_CROSSATTN: demo_batch[C.KEY_CROSSATTN][index : index + 1].to(device),
            C.KEY_CROSSATTN_MASK: demo_batch[C.KEY_CROSSATTN_MASK][index : index + 1].to(device),
            C.KEY_POS_IDS: demo_batch[C.KEY_POS_IDS][index : index + 1, :seq_len].to(device),
            C.KEY_SEQLEN_Q: torch.tensor([seq_len], dtype=torch.int32, device=device),
            C.KEY_CONTEXT: context,
            C.KEY_COND_MASK: cond_mask,
        }

    def _save_prompt(
        self,
        path: Path,
        prompt: str,
        key: str,
        step: int,
        seq_len: int,
        duration_seconds: float,
    ) -> None:
        path.write_text(
            "\n".join(
                [
                    f"step: {step}",
                    f"key: {key}",
                    f"duration_seconds: {duration_seconds:.3f}",
                    f"latent_tokens: {seq_len}",
                    "",
                    prompt,
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None or trainer.global_rank != 0:
            return
        if not self._should_run_demo(trainer.global_step):
            return
        if not isinstance(getattr(module, "objective", None), MusicFlowMatching):
            return

        self.last_demo_step = trainer.global_step
        output_dir = Path(self.demo_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng_state = self._capture_rng_state()
        was_training = bool(module.model.training)
        module.model.eval()
        self._clear_cuda_cache()
        try:
            demo_batch = self._next_demo_batch()
            codec = self._codec_for_device(module.device)
            sampler = MusicFlowMatchingEulerSampler(
                module.model,
                estimator_target=module.objective.estimator_target,
                flow=module.objective.flow,
                num_steps=self.num_steps,
            )
            data_scale = float(module.objective.data_scale)
            modes = [
                ("01_text2music", None),
                ("02_text_vocals2music", "vocals"),
                ("03_text_accompaniment2music", "accompaniment"),
            ]
            for i in range(min(self.demo_num, len(demo_batch["prompt_text"]))):
                key = _safe_name(demo_batch["__key__"][i], fallback=f"sample{i:02d}")
                seq_len = int(demo_batch[C.KEY_SEQLEN_Q][i].item())
                frame_rate = float(demo_batch["frame_rate"][i].item())
                duration_seconds = seq_len * self.patch_t / frame_rate
                prefix = f"step{trainer.global_step:08d}_sample{i:02d}_{key}"
                self._save_prompt(
                    output_dir / f"{prefix}__00_prompt.txt",
                    demo_batch["prompt_text"][i],
                    key,
                    trainer.global_step,
                    seq_len,
                    duration_seconds,
                )
                for mode_index, (mode_name, context_key) in enumerate(modes):
                    gen_batch = self._make_generation_batch(demo_batch, i, context_key, module.device, seq_len)
                    generator = torch.Generator(device=module.device)
                    generator.manual_seed(self.demo_seed + trainer.global_step * 100_000 + i * 100 + mode_index)
                    latent_tokens = sampler.sample(
                        gen_batch,
                        shape=(1, seq_len, module.model.latent_token_dim),
                        generator=generator,
                        guidance_scale=self.guidance_scale,
                    )
                    latent = (latent_tokens / data_scale).permute(0, 2, 1).contiguous()
                    audio = codec.decode(latent)[0].detach().float().cpu().clamp(-1.0, 1.0)
                    target_samples = codec.num_samples(seq_len * self.patch_t)
                    audio = audio[..., :target_samples]
                    torchaudio.save(
                        str(output_dir / f"{prefix}__{mode_name}.{self.save_format}"),
                        audio,
                        codec.sample_rate,
                        format=self.save_format,
                    )
        finally:
            module.model.train(was_training)
            self._restore_rng_state(rng_state)
            self._clear_cuda_cache()
