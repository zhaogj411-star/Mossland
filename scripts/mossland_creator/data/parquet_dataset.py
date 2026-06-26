# Copyright (c) 2025.
"""Minimal Sonata parquet audio helpers used by prepare_energon_shards.

This is intentionally smaller than the original `music_dit.data.parquet_dataset`
module: the offline shard-prep script only needs path resolution and waveform
loading, not the training dataset/task assembly stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class ParquetDatasetConfig:
    parquet_glob: str
    clip_seconds: float = 10.0
    patch_t: int = 1
    pos_dim: int = 1
    audio_path_field: str = "audio_path"
    audio_root: Optional[str] = None
    oss_prefix: str = "qz_oss2:"
    sample_seed: int = 0
    skip_on_error: bool = True


def default_audio_resolver(cfg: ParquetDatasetConfig) -> Callable[[str], str]:
    """Map a stored `audio_path` to a readable local file path."""

    def resolve(path: str) -> str:
        if os.path.exists(path):
            return path
        key = path
        if cfg.oss_prefix and key.startswith(cfg.oss_prefix):
            key = key[len(cfg.oss_prefix) :]
        key = key.lstrip("/").split(":", 1)[-1]
        if cfg.audio_root:
            cand = os.path.join(cfg.audio_root, key)
            if os.path.exists(cand):
                return cand
            cand2 = os.path.join(cfg.audio_root, os.path.basename(key))
            if os.path.exists(cand2):
                return cand2
        raise FileNotFoundError(
            f"cannot resolve audio '{path}'. Set audio_root or pass a custom "
            f"audio_resolver that downloads from OSS."
        )

    return resolve


def _load_audio(path: str, target_sr: int, target_ch: int) -> torch.Tensor:
    """-> waveform [A, N] at `target_sr` with `target_ch` channels."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).T.contiguous()
    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
    a = wav.shape[0]
    if a == target_ch:
        pass
    elif a == 1 and target_ch == 2:
        wav = wav.repeat(2, 1)
    elif a == 2 and target_ch == 1:
        wav = wav.mean(0, keepdim=True)
    else:
        wav = wav[:target_ch] if a > target_ch else wav.repeat(target_ch, 1)[:target_ch]
    return wav


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    try:
        import torchaudio  # type: ignore

        return torchaudio.functional.resample(wav, sr, target_sr)
    except Exception:
        n = wav.shape[-1]
        new_n = int(round(n * target_sr / sr))
        return torch.nn.functional.interpolate(
            wav.unsqueeze(0), size=new_n, mode="linear", align_corners=False
        ).squeeze(0)
