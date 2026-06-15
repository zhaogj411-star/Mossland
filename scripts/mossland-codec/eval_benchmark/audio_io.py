from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from .manifest import EvalItem


def ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"audio must have shape [C,T], got {tuple(audio.shape)}")
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    return audio[:2]


def align_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(a.shape[-1], b.shape[-1])
    return a[..., :length], b[..., :length]


@lru_cache(maxsize=32)
def _resampler(orig_freq: int, new_freq: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


def load_audio_segment(
    path: str | Path,
    sample_rate: int,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    stereo: bool = True,
) -> torch.Tensor:
    info = torchaudio.info(str(path))
    frame_offset = int(round(start_seconds * info.sample_rate))
    num_frames = -1 if duration_seconds is None else int(round(duration_seconds * info.sample_rate))
    audio, sr = torchaudio.load(str(path), frame_offset=max(frame_offset, 0), num_frames=num_frames)
    if sr != sample_rate:
        audio = _resampler(sr, sample_rate)(audio)
    if stereo:
        audio = ensure_stereo(audio)
    return audio.clamp(-1.0, 1.0)


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu().float().clamp(-1.0, 1.0), sample_rate)


def reference_path_for_item(item: EvalItem) -> Path | None:
    if item.reference_path is not None:
        return item.reference_path
    if item.task_id == "separate_vocals":
        return item.vocals_path
    if item.task_id in {"separate_drums", "separate_bass", "separate_other"}:
        stem = item.task_id.removeprefix("separate_")
        path = item.metadata.get(f"{stem}_path")
        return None if path in (None, "") else Path(str(path))
    if item.task_id == "separate_accompaniment":
        return item.accompaniment_path
    if item.task_id in {"reconstruct", "super_resolution", "mono_to_stereo"}:
        return item.source_path
    return None
