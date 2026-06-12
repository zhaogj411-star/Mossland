from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from tqdm import tqdm


DEFAULT_MODEL_REPO = "KimberleyJensen/Mel-Band-Roformer-Vocal-Model"
DEFAULT_CHECKPOINT_REPO = "KimberleyJSN/melbandroformer"
DEFAULT_MODEL_DIR_NAME = "mel-band-roformer-vocal-model"
DEFAULT_CONFIG_NAME = "config_vocals_mel_band_roformer.yaml"
DEFAULT_CHECKPOINT_NAME = "MelBandRoformer.ckpt"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_NUM_CHANNELS = 2
DEFAULT_NUM_OVERLAP = 2
DEFAULT_MAX_DURATION_SECONDS = 10 * 60
COUNTED_PROGRESS_STATUSES = {"done", "skipped", "skiplong", "error"}
TRACKED_PROGRESS_STATUSES = COUNTED_PROGRESS_STATUSES | {"started"}
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}


class ConfigView(dict):
    """递归 dict 视图，兼容同事脚本里的 config.audio.sample_rate 写法。"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class ModelFiles:
    config_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class SeparationLayout:
    item_dir: Path
    mixture_path: Path
    vocals_path: Path
    accompaniment_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class SeparationResult:
    mixture: torch.Tensor
    vocals: torch.Tensor
    accompaniment: torch.Tensor
    sample_rate: int
    source_sample_rate: int | None = None
    source_num_channels: int | None = None


@dataclass(frozen=True)
class LoadedAudio:
    audio: torch.Tensor
    source_sample_rate: int
    source_num_channels: int


@dataclass(frozen=True)
class WorkerSpec:
    command: list[str]
    env: dict[str, str]
    log_path: Path


@dataclass(frozen=True)
class ProgressEvent:
    status: str
    worker_id: int
    path: str


@dataclass(frozen=True)
class PendingWrite:
    future: Future
    audio_path: Path
    layout: SeparationLayout


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    return repo_root() / "checkpoints" / DEFAULT_MODEL_DIR_NAME


def default_config_path() -> Path:
    return default_model_dir() / DEFAULT_CONFIG_NAME


def default_checkpoint_path() -> Path:
    return default_model_dir() / DEFAULT_CHECKPOINT_NAME


def ensure_model_files(
    model_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
) -> ModelFiles:
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    config_path = Path(config_path) if config_path is not None else model_dir / DEFAULT_CONFIG_NAME
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else model_dir / DEFAULT_CHECKPOINT_NAME
    )
    missing = [str(path) for path in (config_path, checkpoint_path) if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "RoFormer 分离模型文件不存在。请把模型文件放到 "
            f"{model_dir}：\n{joined}"
        )
    return ModelFiles(config_path=config_path, checkpoint_path=checkpoint_path)


def safe_stem_id(audio_path: Path | str, source_root: Path | str | None = None) -> str:
    """把源音频路径转成保留相对目录结构的离线 stem 路径。"""
    audio_path = Path(audio_path)
    if source_root is not None:
        try:
            relpath = audio_path.relative_to(Path(source_root))
        except ValueError:
            relpath = Path(audio_path.name)
    else:
        relpath = Path(audio_path.name)

    rel_without_suffix = relpath.with_suffix("")
    parts = [part for part in rel_without_suffix.parts if part not in ("", ".")]
    return str(Path(*parts)) if parts else audio_path.stem


def separation_layout(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None = None,
) -> SeparationLayout:
    item_dir = Path(output_root) / safe_stem_id(audio_path, source_root)
    return SeparationLayout(
        item_dir=item_dir,
        mixture_path=item_dir / "mixture.mp3",
        vocals_path=item_dir / "vocals.mp3",
        accompaniment_path=item_dir / "accompaniment.mp3",
        metadata_path=item_dir / "metadata.json",
    )


def separation_done(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None = None,
) -> bool:
    layout = separation_layout(audio_path, output_root, source_root)
    required = (
        layout.mixture_path,
        layout.vocals_path,
        layout.accompaniment_path,
        layout.metadata_path,
    )
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("status") == "done"


def separation_terminal(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None = None,
) -> bool:
    return separation_status(audio_path, output_root, source_root) is not None


def separation_status(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None = None,
) -> str | None:
    if separation_done(audio_path, output_root, source_root):
        return "done"
    layout = separation_layout(audio_path, output_root, source_root)
    status = _metadata_status(layout.metadata_path)
    if status in {"error", "skiplong"}:
        return status
    return None


def _progress_status_for_terminal(status: str | None) -> str | None:
    if status == "done":
        return "skipped"
    if status in {"error", "skiplong"}:
        return status
    return None


def filter_pending_files(
    files: list[Path | str],
    output_root: Path | str,
    source_root: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[list[Path], list[Path]]:
    paths = [Path(file_path) for file_path in files]
    if overwrite:
        return paths, []
    pending: list[Path] = []
    skipped: list[Path] = []
    for file_path in paths:
        if separation_terminal(file_path, output_root, source_root):
            skipped.append(file_path)
        else:
            pending.append(file_path)
    return pending, skipped


def _write_progress_event(
    progress_file: Path | str | None,
    status: str,
    worker_id: int,
    audio_path: Path | str,
) -> None:
    if progress_file is None:
        return
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "status": status,
        "worker_id": int(worker_id),
        "path": str(audio_path),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read_progress_events(
    progress_file: Path | str,
    offset: int,
    statuses: set[str] | None = None,
) -> tuple[int, list[ProgressEvent]]:
    path = Path(progress_file)
    if not path.exists():
        return offset, []
    allowed_statuses = statuses or COUNTED_PROGRESS_STATUSES
    events: list[ProgressEvent] = []
    with path.open(encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(raw.get("status", ""))
            if status not in allowed_statuses:
                continue
            events.append(
                ProgressEvent(
                    status=status,
                    worker_id=int(raw.get("worker_id", -1)),
                    path=str(raw.get("path", "")),
                )
            )
        return f.tell(), events


def load_model_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path is not None else default_config_path()
    with path.open(encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _to_config_view(value):
    if isinstance(value, Mapping):
        return ConfigView({key: _to_config_view(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config_view(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_config_view(item) for item in value)
    return value


def _config_with_num_overlap(model_config: Mapping, num_overlap: int | None) -> dict:
    copied = json.loads(json.dumps(model_config))
    if num_overlap is not None:
        copied.setdefault("inference", {})["num_overlap"] = int(num_overlap)
    return copied


def _config_sample_rate(model_config: Mapping) -> int:
    audio_cfg = model_config.get("audio", {})
    model_cfg = model_config.get("model", {})
    return int(audio_cfg.get("sample_rate", model_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)))


def _config_num_channels(model_config: Mapping) -> int:
    audio_cfg = model_config.get("audio", {})
    if "num_channels" in audio_cfg:
        return int(audio_cfg["num_channels"])
    model_cfg = model_config.get("model", {})
    if "stereo" in model_cfg:
        return 2 if bool(model_cfg["stereo"]) else 1
    return DEFAULT_NUM_CHANNELS


def _extract_model_summary(model_config: Mapping) -> dict:
    inference_cfg = model_config.get("inference", {})
    training_cfg = model_config.get("training", {})
    return {
        "sample_rate": _config_sample_rate(model_config),
        "num_channels": _config_num_channels(model_config),
        "num_overlap": int(inference_cfg.get("num_overlap", DEFAULT_NUM_OVERLAP)),
        "chunk_size": inference_cfg.get("chunk_size"),
        "dim_t": inference_cfg.get("dim_t"),
        "instruments": list(training_cfg.get("instruments", ["vocals", "other"])),
        "target_instrument": training_cfg.get("target_instrument"),
    }


def write_metadata(
    layout: SeparationLayout,
    source_path: Path | str,
    model_repo: str,
    model_config: Mapping,
    status: str,
    model_files: ModelFiles | None = None,
    error: str | None = None,
    reason: str | None = None,
    duration_seconds: float | None = None,
    max_duration_seconds: float | None = None,
) -> None:
    model_info = {
        "repo": model_repo,
        **_extract_model_summary(model_config),
    }
    if model_files is not None:
        model_info["config_path"] = str(model_files.config_path)
        model_info["checkpoint_path"] = str(model_files.checkpoint_path)
    metadata = {
        "status": status,
        "source_path": str(source_path),
        "model": model_info,
        "stems": {
            "mixture": layout.mixture_path.name,
            "vocals": layout.vocals_path.name,
            "accompaniment": layout.accompaniment_path.name,
        },
    }
    if error is not None:
        metadata["error"] = error
    if reason is not None:
        metadata["reason"] = reason
    if duration_seconds is not None:
        metadata["duration_seconds"] = float(duration_seconds)
    if max_duration_seconds is not None:
        metadata["max_duration_seconds"] = float(max_duration_seconds)
    layout.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_audio_with_source_info(path: Path, sample_rate: int, num_channels: int) -> LoadedAudio:
    audio, in_sample_rate = torchaudio.load(str(path))
    source_num_channels = int(audio.shape[0])
    audio = audio.float()
    if in_sample_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, in_sample_rate, sample_rate)
    if num_channels == 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif audio.shape[0] == 1 and num_channels == 2:
        audio = audio.repeat_interleave(2, dim=0)
    elif audio.shape[0] > num_channels:
        audio = audio[:num_channels]
    return LoadedAudio(
        audio=audio.contiguous().clamp(-1.0, 1.0),
        source_sample_rate=int(in_sample_rate),
        source_num_channels=source_num_channels,
    )


def _normalize_max_duration_seconds(max_duration_seconds: float | int | None) -> float | None:
    if max_duration_seconds is None:
        return None
    max_duration = float(max_duration_seconds)
    if max_duration <= 0:
        return None
    return max_duration


def _probe_audio_duration_seconds(path: Path | str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            duration_text = completed.stdout.strip().splitlines()
            if duration_text:
                with contextlib.suppress(ValueError):
                    duration = float(duration_text[0])
                    if duration > 0:
                        return duration

    info_fn = getattr(torchaudio, "info", None)
    if info_fn is None:
        return None
    try:
        info = info_fn(str(path))
    except Exception:
        return None
    sample_rate = int(getattr(info, "sample_rate", 0) or 0)
    num_frames = int(getattr(info, "num_frames", 0) or 0)
    if sample_rate <= 0 or num_frames <= 0:
        return None
    return num_frames / sample_rate


def _skip_if_duration_exceeds_limit(
    audio_path: Path | str,
    layout: SeparationLayout,
    model_config: Mapping,
    model_files: ModelFiles,
    max_duration_seconds: float | int | None,
    worker_id: int,
    progress_file: Path | str | None = None,
) -> bool:
    max_duration = _normalize_max_duration_seconds(max_duration_seconds)
    if max_duration is None:
        return False
    duration = _probe_audio_duration_seconds(audio_path)
    if duration is None or duration < max_duration:
        return False

    layout.item_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        layout,
        audio_path,
        DEFAULT_MODEL_REPO,
        model_config,
        status="skiplong",
        model_files=model_files,
        reason="max_duration_exceeded",
        duration_seconds=duration,
        max_duration_seconds=max_duration,
    )
    _write_progress_event(progress_file, "skiplong", worker_id, audio_path)
    print(
        f"[worker-{worker_id}] skip overlong audio: {audio_path} "
        f"duration={duration:.1f}s max={max_duration:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return True


def _load_audio(path: Path, sample_rate: int, num_channels: int) -> torch.Tensor:
    return _load_audio_with_source_info(path, sample_rate, num_channels).audio


def _save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu().float(), sample_rate)


def _cleanup_cuda_memory(device: torch.device | str | None = None) -> None:
    if device is None:
        is_cuda_device = torch.cuda.is_available()
    else:
        try:
            is_cuda_device = torch.device(device).type == "cuda"
        except (RuntimeError, TypeError):
            return
    if not is_cuda_device:
        return

    gc.collect()
    if not torch.cuda.is_available():
        return
    with contextlib.suppress(RuntimeError):
        torch.cuda.empty_cache()
    with contextlib.suppress(RuntimeError):
        torch.cuda.ipc_collect()


def _get_inference_params(config: ConfigView) -> tuple[int, int]:
    if "chunk_size" in config.inference:
        chunk_size = int(config.inference.chunk_size)
    else:
        chunk_size = int(config.audio.chunk_size)
    return chunk_size, int(config.inference.num_overlap)


def _windowing_array(window_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size, device=device)
    fadeout = torch.linspace(1, 0, fade_size, device=device)
    window = torch.ones(window_size, device=device)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


@torch.no_grad()
def _demix(
    model: torch.nn.Module,
    config: ConfigView,
    mixture: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    chunk_batch_size: int = 1,
) -> dict[str, np.ndarray]:
    chunk_size, num_overlap = _get_inference_params(config)
    step = chunk_size // num_overlap
    fade_size = chunk_size // 10
    border = chunk_size - step
    chunk_batch_size = max(1, int(chunk_batch_size))

    if mixture.shape[1] > 2 * border and border > 0:
        mixture = nn.functional.pad(mixture, (border, border), mode="reflect")

    windowing = _windowing_array(chunk_size, fade_size, device)
    instruments = list(config.training.instruments)
    target_instrument = config.training.get("target_instrument")
    if target_instrument is not None:
        req_shape = (1,) + tuple(mixture.shape)
    else:
        req_shape = (len(instruments),) + tuple(mixture.shape)

    mixture = mixture.to(device)
    result = torch.zeros(req_shape, dtype=torch.float32, device=device)
    counter = torch.zeros(req_shape, dtype=torch.float32, device=device)

    total_length = mixture.shape[1]
    index = 0
    if use_amp and device.type == "cuda":
        amp_context = torch.amp.autocast("cuda")
    else:
        amp_context = contextlib.nullcontext()

    with amp_context:
        batch_parts: list[torch.Tensor] = []
        batch_lengths: list[int] = []
        batch_indexes: list[int] = []

        def flush_batch() -> None:
            if not batch_parts:
                return
            estimates = model(torch.stack(batch_parts, dim=0))
            for batch_index, (start, length) in enumerate(zip(batch_indexes, batch_lengths)):
                estimate = estimates[batch_index]
                window = windowing.clone()
                if start == 0:
                    window[:fade_size] = 1
                elif start + chunk_size >= total_length:
                    window[-fade_size:] = 1

                result[..., start : start + length] += estimate[..., :length] * window[..., :length]
                counter[..., start : start + length] += window[..., :length]
            batch_parts.clear()
            batch_lengths.clear()
            batch_indexes.clear()

        while index < total_length:
            part = mixture[:, index : index + chunk_size]
            length = part.shape[-1]
            if length < chunk_size:
                if length > chunk_size // 2 + 1:
                    part = nn.functional.pad(part, (0, chunk_size - length), mode="reflect")
                else:
                    part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), value=0)

            batch_parts.append(part)
            batch_lengths.append(length)
            batch_indexes.append(index)
            if len(batch_parts) >= chunk_batch_size:
                flush_batch()
            index += step
        flush_batch()

    estimated = (result / counter.clamp(min=1e-8)).float().cpu().numpy()
    np.nan_to_num(estimated, copy=False, nan=0.0)

    if mixture.shape[1] > 2 * border and border > 0:
        estimated = estimated[..., border:-border]

    if target_instrument is None:
        return {name: stem for name, stem in zip(instruments, estimated)}
    return {target_instrument: estimated[0]}


def _select_vocals(stems: Mapping[str, np.ndarray]) -> np.ndarray:
    if "vocals" in stems:
        return stems["vocals"]
    raise RuntimeError(f"RoFormer 输出缺少 vocals stem: {list(stems)}")


def _select_accompaniment(
    stems: Mapping[str, np.ndarray],
    mixture: torch.Tensor,
    vocals: np.ndarray,
) -> np.ndarray:
    del stems
    return mixture.cpu().numpy() - vocals


def _as_audio_tensor(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        tensor = audio.detach().cpu().float()
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(audio)).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.clamp(-1.0, 1.0)


def _write_separation_result(
    layout: SeparationLayout,
    result: SeparationResult,
    audio_path: Path | str,
    model_config: Mapping,
    model_files: ModelFiles,
) -> None:
    layout.item_dir.mkdir(parents=True, exist_ok=True)
    if _can_copy_source_mixture(audio_path, result):
        shutil.copyfile(audio_path, layout.mixture_path)
    else:
        _save_audio(layout.mixture_path, result.mixture, result.sample_rate)
    _save_audio(layout.vocals_path, result.vocals, result.sample_rate)
    _save_audio(layout.accompaniment_path, result.accompaniment, result.sample_rate)
    write_metadata(
        layout,
        audio_path,
        DEFAULT_MODEL_REPO,
        model_config,
        status="done",
        model_files=model_files,
    )


def _can_copy_source_mixture(audio_path: Path | str, result: SeparationResult) -> bool:
    audio_path = Path(audio_path)
    return (
        audio_path.suffix.lower() == ".mp3"
        and result.source_sample_rate == result.sample_rate
        and result.source_num_channels == int(result.mixture.shape[0])
    )


class AsyncSeparationWriter:
    """后台保存 stem，让下一首推理和上一首 MP3 编码/落盘重叠。"""

    def __init__(
        self,
        model_config: Mapping,
        model_files: ModelFiles,
        worker_id: int,
        progress_file: Path | str | None = None,
        max_workers: int = 1,
        max_pending: int | None = None,
    ):
        self.model_config = model_config
        self.model_files = model_files
        self.worker_id = int(worker_id)
        self.progress_file = progress_file
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self.max_pending = max(1, int(max_pending or max_workers * 2))
        self.pending: list[PendingWrite] = []

    def submit(
        self,
        layout: SeparationLayout,
        result: SeparationResult,
        audio_path: Path | str,
    ) -> tuple[int, int]:
        done, errors = self._wait_for_capacity()
        future = self.executor.submit(
            _write_separation_result,
            layout,
            result,
            audio_path,
            self.model_config,
            self.model_files,
        )
        self.pending.append(PendingWrite(future=future, audio_path=Path(audio_path), layout=layout))
        delta_done, delta_errors = self.drain_completed()
        return done + delta_done, errors + delta_errors

    def _wait_for_capacity(self) -> tuple[int, int]:
        done = 0
        errors = 0
        while len(self.pending) >= self.max_pending:
            futures = [pending.future for pending in self.pending]
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            delta_done, delta_errors = self._collect_completed(completed)
            done += delta_done
            errors += delta_errors
        return done, errors

    def drain_completed(self) -> tuple[int, int]:
        completed = {pending.future for pending in self.pending if pending.future.done()}
        return self._collect_completed(completed)

    def close(self) -> tuple[int, int]:
        done = 0
        errors = 0
        while self.pending:
            completed, _ = wait([pending.future for pending in self.pending], return_when=FIRST_COMPLETED)
            delta_done, delta_errors = self._collect_completed(completed)
            done += delta_done
            errors += delta_errors
        self.executor.shutdown()
        return done, errors

    def _collect_completed(self, completed: set[Future]) -> tuple[int, int]:
        if not completed:
            return 0, 0
        done = 0
        errors = 0
        still_pending: list[PendingWrite] = []
        for pending in self.pending:
            if pending.future not in completed:
                still_pending.append(pending)
                continue
            try:
                pending.future.result()
                done += 1
                _write_progress_event(self.progress_file, "done", self.worker_id, pending.audio_path)
            except Exception as exc:
                errors += 1
                pending.layout.item_dir.mkdir(parents=True, exist_ok=True)
                write_metadata(
                    pending.layout,
                    pending.audio_path,
                    DEFAULT_MODEL_REPO,
                    self.model_config,
                    status="error",
                    model_files=self.model_files,
                    error=str(exc),
                )
                _write_progress_event(self.progress_file, "error", self.worker_id, pending.audio_path)
                print(
                    f"[worker-{self.worker_id}] error saving: {pending.audio_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        self.pending = still_pending
        return done, errors


class RoformerSeparator:
    """本地 Mel-Band RoFormer 推理器；伴奏由 mixture - vocals 生成。"""

    def __init__(
        self,
        checkpoint_path: Path | str,
        config_path: Path | str,
        device: str | torch.device | None = None,
        num_overlap: int | None = DEFAULT_NUM_OVERLAP,
        use_amp: bool = True,
        chunk_batch_size: int = 1,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.raw_config = load_model_config(self.config_path)
        self.effective_config = _to_config_view(_config_with_num_overlap(self.raw_config, num_overlap))
        self.use_amp = use_amp
        self.chunk_batch_size = max(1, int(chunk_batch_size))
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        try:
            from scripts.third_party.mel_band_roformer import MelBandRoformer
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 MelBandRoformer。请在 roformer 环境运行，或安装 "
                "beartype、rotary_embedding_torch 等依赖。"
            ) from exc

        model = MelBandRoformer(**dict(self.raw_config["model"]))
        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def separate(self, audio_path: Path | str) -> SeparationResult:
        sample_rate = _config_sample_rate(self.effective_config)
        num_channels = _config_num_channels(self.effective_config)
        loaded = _load_audio_with_source_info(
            Path(audio_path),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        mixture = loaded.audio
        stems = _demix(
            self.model,
            self.effective_config,
            mixture,
            self.device,
            use_amp=self.use_amp,
            chunk_batch_size=self.chunk_batch_size,
        )
        vocals = _select_vocals(stems)
        accompaniment = _select_accompaniment(stems, mixture, vocals)
        return SeparationResult(
            mixture=mixture,
            vocals=_as_audio_tensor(vocals),
            accompaniment=_as_audio_tensor(accompaniment),
            sample_rate=sample_rate,
            source_sample_rate=loaded.source_sample_rate,
            source_num_channels=loaded.source_num_channels,
        )


def process_file(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None = None,
    model_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str | None = None,
    num_overlap: int | None = DEFAULT_NUM_OVERLAP,
    use_amp: bool = True,
    chunk_batch_size: int = 1,
    overwrite: bool = False,
) -> SeparationLayout:
    audio_path = Path(audio_path)
    output_root = Path(output_root)
    layout = separation_layout(audio_path, output_root, source_root)
    if (
        not overwrite
        and layout.metadata_path.exists()
        and layout.mixture_path.exists()
        and layout.vocals_path.exists()
        and layout.accompaniment_path.exists()
    ):
        return layout

    model_files = ensure_model_files(
        model_dir=model_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    model_config = _config_with_num_overlap(load_model_config(model_files.config_path), num_overlap)

    separator = None
    result = None
    try:
        separator = RoformerSeparator(
            checkpoint_path=model_files.checkpoint_path,
            config_path=model_files.config_path,
            device=device,
            num_overlap=num_overlap,
            use_amp=use_amp,
            chunk_batch_size=chunk_batch_size,
        )
        result = separator.separate(audio_path)
        _write_separation_result(layout, result, audio_path, model_config, model_files)
    except Exception as exc:
        layout.item_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(
            layout,
            audio_path,
            DEFAULT_MODEL_REPO,
            model_config,
            status="error",
            model_files=model_files,
            error=str(exc),
        )
        raise
    finally:
        cleanup_device = getattr(separator, "device", device)
        result = None
        separator = None
        _cleanup_cuda_memory(cleanup_device)
    return layout


def process_files(
    files: list[Path | str],
    output_root: Path | str,
    source_root: Path | str | None = None,
    model_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str | None = None,
    num_overlap: int | None = DEFAULT_NUM_OVERLAP,
    use_amp: bool = True,
    chunk_batch_size: int = 1,
    overwrite: bool = False,
    progress: bool = True,
    worker_id: int = 0,
    progress_file: Path | str | None = None,
    save_workers: int = 1,
    max_pending_writes: int | None = None,
    max_duration_seconds: float | int | None = DEFAULT_MAX_DURATION_SECONDS,
) -> tuple[int, int, int, int]:
    done = 0
    skipped = 0
    skiplong = 0
    errors = 0
    saw_pending = False
    model_files: ModelFiles | None = None
    model_config: dict | None = None
    separator = None
    writer: AsyncSeparationWriter | None = None
    cleanup_device = device

    def ensure_model_context() -> tuple[ModelFiles, dict]:
        nonlocal model_files, model_config
        if model_files is None:
            model_files = ensure_model_files(
                model_dir=model_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
            model_config = _config_with_num_overlap(
                load_model_config(model_files.config_path),
                num_overlap,
            )
        assert model_config is not None
        return model_files, model_config

    def ensure_separator() -> tuple[RoformerSeparator, AsyncSeparationWriter]:
        nonlocal separator, writer, cleanup_device
        current_model_files, current_model_config = ensure_model_context()
        if separator is None:
            separator = RoformerSeparator(
                checkpoint_path=current_model_files.checkpoint_path,
                config_path=current_model_files.config_path,
                device=device,
                num_overlap=num_overlap,
                use_amp=use_amp,
                chunk_batch_size=chunk_batch_size,
            )
            cleanup_device = getattr(separator, "device", device)
            writer = AsyncSeparationWriter(
                model_config=current_model_config,
                model_files=current_model_files,
                worker_id=worker_id,
                progress_file=progress_file,
                max_workers=save_workers,
                max_pending=max_pending_writes,
            )
        assert writer is not None
        return separator, writer

    try:
        iterator = tqdm(
            files,
            desc=f"worker-{worker_id}",
            unit="file",
            disable=not progress,
            dynamic_ncols=True,
        )
        for audio_path in iterator:
            audio_path = Path(audio_path)
            if not overwrite:
                terminal_status = separation_status(audio_path, output_root, source_root)
                progress_status = _progress_status_for_terminal(terminal_status)
                if progress_status is not None:
                    if terminal_status == "error":
                        current_model_files, current_model_config = ensure_model_context()
                        layout = separation_layout(audio_path, output_root, source_root)
                        if _skip_if_duration_exceeds_limit(
                            audio_path,
                            layout,
                            current_model_config,
                            current_model_files,
                            max_duration_seconds,
                            worker_id,
                            progress_file,
                        ):
                            skiplong += 1
                            continue
                    _write_progress_event(progress_file, progress_status, worker_id, audio_path)
                    if progress_status == "skipped":
                        skipped += 1
                    elif progress_status == "skiplong":
                        skiplong += 1
                    elif progress_status == "error":
                        errors += 1
                    continue
            saw_pending = True
            current_model_files, current_model_config = ensure_model_context()
            layout = separation_layout(audio_path, output_root, source_root)
            _write_progress_event(progress_file, "started", worker_id, audio_path)
            if _skip_if_duration_exceeds_limit(
                audio_path,
                layout,
                current_model_config,
                current_model_files,
                max_duration_seconds,
                worker_id,
                progress_file,
            ):
                skiplong += 1
                continue
            current_separator, current_writer = ensure_separator()
            needs_cuda_cleanup = False
            result = None
            try:
                needs_cuda_cleanup = True
                result = current_separator.separate(audio_path)
                delta_done, delta_errors = current_writer.submit(layout, result, audio_path)
                done += delta_done
                errors += delta_errors
            except Exception as exc:
                errors += 1
                layout.item_dir.mkdir(parents=True, exist_ok=True)
                write_metadata(
                    layout,
                    audio_path,
                    DEFAULT_MODEL_REPO,
                    current_model_config,
                    status="error",
                    model_files=current_model_files,
                    error=str(exc),
                )
                _write_progress_event(progress_file, "error", worker_id, audio_path)
                print(f"[worker-{worker_id}] error: {audio_path}: {exc}", file=sys.stderr, flush=True)
            finally:
                result = None
                if needs_cuda_cleanup:
                    _cleanup_cuda_memory(cleanup_device)
        if writer is not None:
            delta_done, delta_errors = writer.close()
            done += delta_done
            errors += delta_errors
        if saw_pending:
            print(
                f"[worker-{worker_id}] done={done} skipped={skipped} "
                f"skiplong={skiplong} errors={errors}",
                flush=True,
            )
        else:
            print(
                f"[worker-{worker_id}] pending=0 skipped={skipped} "
                f"skiplong={skiplong} errors={errors}",
                flush=True,
            )
        return done, skipped, skiplong, errors
    finally:
        if separator is not None:
            separator = None
            _cleanup_cuda_memory(cleanup_device)


def _fast_audio_files(
    input_dir: Path | str,
    extensions: set[str],
    max_files: int | None = None,
) -> list[str]:
    """轻量版 fast_scandir，避免为 CLI 扫描导入训练 dataset 依赖。"""
    files: list[str] = []
    stack = [str(input_dir)]
    while stack and (max_files is None or len(files) < max_files):
        current_dir = stack.pop()
        try:
            entries = sorted(os.scandir(current_dir), key=lambda entry: entry.name)
        except OSError:
            continue
        subdirs: list[str] = []
        for entry in entries:
            try:
                if entry.is_dir():
                    if not entry.name.startswith("."):
                        subdirs.append(entry.path)
                    continue
                if not entry.is_file():
                    continue
                if entry.name.startswith("."):
                    continue
                if Path(entry.name).suffix.lower() in extensions:
                    files.append(entry.path)
                    if max_files is not None and len(files) >= max_files:
                        break
            except OSError:
                continue
        stack.extend(reversed(subdirs))
    return files


def _iter_audio_files(
    input_dirs: list[str],
    files_list: str | None,
    max_files: int | None = None,
) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()

    def append_unique(file_path: str) -> None:
        if file_path and file_path not in seen:
            seen.add(file_path)
            files.append(file_path)

    if files_list is not None:
        for line in Path(files_list).read_text(encoding="utf-8").splitlines():
            append_unique(line.strip())
            if max_files is not None and len(files) >= max_files:
                return files
    for input_dir in input_dirs:
        remaining = None if max_files is None else max_files - len(files)
        if remaining is not None and remaining <= 0:
            break
        for file_path in _fast_audio_files(input_dir, AUDIO_EXTENSIONS, max_files=remaining):
            append_unique(file_path)
            if max_files is not None and len(files) >= max_files:
                return files
    if max_files is not None:
        return files
    return sorted(files)


def select_shard(files: list[Path | str], shard_id: int, num_shards: int) -> list[Path | str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return [file_path for index, file_path in enumerate(files) if index % num_shards == shard_id]


def parse_devices(devices: str | None) -> list[str]:
    if not devices:
        return []
    parsed = [device.strip() for device in devices.split(",") if device.strip()]
    if not parsed:
        raise ValueError("--devices 不能为空")
    return parsed


def _write_manifest(files: list[str], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(files) + "\n", encoding="utf-8")


def build_worker_spec(
    args: argparse.Namespace,
    manifest_path: Path,
    shard_id: int,
    num_shards: int,
    visible_device: str,
) -> WorkerSpec:
    log_dir = Path(args.worker_log_dir or Path(args.output_root) / "_logs" / "prepare_separation")
    safe_device = visible_device.replace(",", "_").replace(":", "_")
    log_path = log_dir / f"worker_{shard_id:02d}_gpu{safe_device}.log"
    command = [
        sys.executable,
        "-m",
        "scripts.data.prepare_separation",
        "--files-list",
        str(manifest_path),
        "--output-root",
        str(args.output_root),
        "--num-shards",
        str(num_shards),
        "--shard-id",
        str(shard_id),
        "--device",
        "cuda:0",
        "--worker-id",
        str(shard_id),
        "--num-overlap",
        str(args.num_overlap),
        "--chunk-batch-size",
        str(args.chunk_batch_size),
        "--max-duration-seconds",
        str(float(args.max_duration_seconds)),
    ]
    if args.source_root is not None:
        command.extend(["--source-root", str(args.source_root)])
    if args.model_dir is not None:
        command.extend(["--model-dir", str(args.model_dir)])
    if args.config_path is not None:
        command.extend(["--config-path", str(args.config_path)])
    if args.checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(args.checkpoint_path)])
    if args.max_files is not None:
        command.extend(["--max-files", str(args.max_files)])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_amp:
        command.append("--no-amp")
    if args.no_progress:
        command.append("--no-progress")
    progress_file = log_dir / "progress.jsonl"
    command.extend(["--progress-file", str(progress_file)])
    command.extend(["--save-workers", str(args.save_workers)])
    if args.max_pending_writes is not None:
        command.extend(["--max-pending-writes", str(args.max_pending_writes)])
    if args.fail_on_error:
        command.append("--fail-on-error")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible_device
    env["PYTHONUNBUFFERED"] = "1"
    return WorkerSpec(command=command, env=env, log_path=log_path)


def _worker_id_from_spec(spec: WorkerSpec) -> int:
    try:
        return int(spec.command[spec.command.index("--worker-id") + 1])
    except (ValueError, IndexError):
        return -1


def _start_worker_process(spec: WorkerSpec, restart_count: int = 0):
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if restart_count else "w"
    log_file = spec.log_path.open(mode, encoding="utf-8")
    if restart_count:
        print(
            f"\n[launcher] restart attempt {restart_count}: {' '.join(spec.command)}",
            file=log_file,
            flush=True,
        )
        print(
            f"[launcher] restart {spec.log_path.name} attempt={restart_count}: "
            f"{' '.join(spec.command)}",
            flush=True,
        )
    else:
        print(f"[launcher] start {spec.log_path.name}: {' '.join(spec.command)}", flush=True)
    process = subprocess.Popen(
        spec.command,
        env=spec.env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_file


def _metadata_status(path: Path) -> str | None:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = metadata.get("status")
    return str(status) if status is not None else None


def _crash_target_already_terminal(
    audio_path: Path | str,
    output_root: Path | str,
    source_root: Path | str | None,
) -> bool:
    if separation_done(audio_path, output_root, source_root):
        return True
    layout = separation_layout(audio_path, output_root, source_root)
    return _metadata_status(layout.metadata_path) in {"error", "skiplong"}


def _mark_worker_crash_file(
    args: argparse.Namespace,
    audio_path: Path | str,
    worker_id: int,
    return_code: int,
) -> bool:
    if _crash_target_already_terminal(audio_path, args.output_root, args.source_root):
        return False

    model_files = ensure_model_files(
        model_dir=args.model_dir,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    model_config = _config_with_num_overlap(load_model_config(model_files.config_path), args.num_overlap)
    layout = separation_layout(audio_path, args.output_root, args.source_root)
    layout.item_dir.mkdir(parents=True, exist_ok=True)
    max_duration = _normalize_max_duration_seconds(args.max_duration_seconds)
    duration = _probe_audio_duration_seconds(audio_path)
    if max_duration is not None and duration is not None and duration >= max_duration:
        write_metadata(
            layout,
            audio_path,
            DEFAULT_MODEL_REPO,
            model_config,
            status="skiplong",
            model_files=model_files,
            reason="max_duration_exceeded",
            duration_seconds=duration,
            max_duration_seconds=max_duration,
        )
        return True

    error = f"worker-{worker_id} exited with rc={return_code} while processing this file"
    write_metadata(
        layout,
        audio_path,
        DEFAULT_MODEL_REPO,
        model_config,
        status="error",
        model_files=model_files,
        error=error,
        reason="worker_crash",
    )
    return True


def _update_file_progress(progress_bar, counts: dict[str, int], events: list[ProgressEvent]) -> None:
    counted_events = [event for event in events if event.status in COUNTED_PROGRESS_STATUSES]
    if not counted_events:
        return
    for event in counted_events:
        counts[event.status] += 1
    progress_bar.update(len(counted_events))
    progress_bar.set_postfix(
        done=counts["done"],
        skipped=counts["skipped"],
        skiplong=counts["skiplong"],
        errors=counts["error"],
    )


def launch_workers(args: argparse.Namespace, files: list[str], devices: list[str]) -> int:
    run_dir = Path(args.output_root) / "_logs" / "prepare_separation"
    manifest_path = run_dir / "manifest.txt"
    progress_file = run_dir / "progress.jsonl"
    _write_manifest(files, manifest_path)
    progress_file.unlink(missing_ok=True)

    specs = [
        build_worker_spec(args, manifest_path, shard_id=index, num_shards=len(devices), visible_device=device)
        for index, device in enumerate(devices)
    ]
    processes = []
    for spec in specs:
        process, log_file = _start_worker_process(spec)
        processes.append((process, spec, log_file, 0))

    failed = 0
    offset = 0
    counts = {"done": 0, "skipped": 0, "skiplong": 0, "error": 0}
    max_restarts = max(0, int(args.worker_restarts))
    last_started_by_worker: dict[int, str] = {}
    active = list(processes)
    with tqdm(
        total=len(files),
        desc="files",
        unit="file",
        disable=args.no_progress,
        dynamic_ncols=True,
    ) as progress_bar:
        while active:
            offset, events = _read_progress_events(
                progress_file,
                offset,
                statuses=TRACKED_PROGRESS_STATUSES,
            )
            for event in events:
                if event.status == "started":
                    last_started_by_worker[event.worker_id] = event.path
            _update_file_progress(progress_bar, counts, events)

            still_active = []
            for process, spec, log_file, restart_count in active:
                return_code = process.poll()
                if return_code is None:
                    still_active.append((process, spec, log_file, restart_count))
                    continue
                log_file.close()
                if return_code != 0:
                    worker_id = _worker_id_from_spec(spec)
                    current_path = last_started_by_worker.get(worker_id)
                    if current_path:
                        try:
                            marked = _mark_worker_crash_file(
                                args,
                                current_path,
                                worker_id,
                                return_code,
                            )
                        except Exception as exc:
                            marked = False
                            print(
                                f"[launcher] failed to mark crashed file for {spec.log_path}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        if marked:
                            print(
                                f"[launcher] marked crashed file from worker-{worker_id}: {current_path}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if restart_count < max_restarts:
                        next_restart_count = restart_count + 1
                        process, log_file = _start_worker_process(spec, next_restart_count)
                        still_active.append((process, spec, log_file, next_restart_count))
                        continue
                    failed += 1
                    print(
                        f"[launcher] failed {spec.log_path}: rc={return_code} "
                        f"restarts={restart_count}/{max_restarts}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[launcher] done {spec.log_path}", flush=True)
            active = still_active
            if active:
                time.sleep(1.0)
        offset, events = _read_progress_events(
            progress_file,
            offset,
            statuses=TRACKED_PROGRESS_STATUSES,
        )
        _update_file_progress(progress_bar, counts, events)
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 Mossland codec 离线准备人声/伴奏分离结果。")
    parser.add_argument("--input-dir", action="append", default=[], help="待扫描的音频目录，可重复传入。")
    parser.add_argument("--files-list", default=None, help="逐行列出音频路径的文本文件。")
    parser.add_argument("--output-root", required=True, help="分离结果输出根目录。")
    parser.add_argument("--source-root", default=None, help="用于生成稳定相对 stem id 的源音频根目录。")
    parser.add_argument(
        "--model-dir",
        default=None,
        help=f"本地 RoFormer 模型目录；默认 {default_model_dir()}。",
    )
    parser.add_argument("--config-path", default=None, help="本地 config_vocals_mel_band_roformer.yaml。")
    parser.add_argument("--checkpoint-path", default=None, help="本地 MelBandRoformer.ckpt。")
    parser.add_argument("--device", default=None, help="推理设备，例如 cuda:0；默认自动选择 cuda/cpu。")
    parser.add_argument("--devices", default=None, help="单机多卡列表，例如 0,1,2,3,4,5,6,7。")
    parser.add_argument("--num-shards", type=int, default=1, help="worker 分片总数。")
    parser.add_argument("--shard-id", type=int, default=0, help="当前 worker 分片 id。")
    parser.add_argument("--worker-id", type=int, default=0, help="进度条显示用 worker id。")
    parser.add_argument("--worker-log-dir", default=None, help="多卡 worker 日志目录。")
    parser.add_argument(
        "--num-overlap",
        type=int,
        default=DEFAULT_NUM_OVERLAP,
        help="覆盖 RoFormer inference.num_overlap；Kimberley vocal config 默认 2。",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=1,
        help="每次 RoFormer 前向同时处理多少个音频 chunk；调大可提高 GPU 利用率，但会增加显存占用。",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=float(DEFAULT_MAX_DURATION_SECONDS),
        help="只处理短于该秒数的音频；默认 600。传 0 可关闭长音频跳过。",
    )
    parser.add_argument("--no-amp", action="store_true", help="禁用 CUDA mixed precision。")
    parser.add_argument("--no-progress", action="store_true", help="禁用 tqdm 进度条。")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--save-workers",
        type=int,
        default=1,
        help="每个推理 worker 用于后台保存 MP3 的线程数；默认 1，用于让 MP3 写入和下一首推理重叠。",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=None,
        help="每个推理 worker 最多缓存多少个待保存结果；默认 2 * save-workers。",
    )
    parser.add_argument(
        "--worker-restarts",
        type=int,
        default=100,
        help="多卡 launcher 中每个 worker 崩溃后的最多自动重启次数；默认 100。",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="遇到单条样本处理失败时返回非零退出码；默认只记录错误并继续完成 worker。",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    files = _iter_audio_files(args.input_dir, args.files_list, max_files=args.max_files)
    if not files:
        raise SystemExit("没有找到待处理音频。请传入 --input-dir 或 --files-list。")

    devices = parse_devices(args.devices)
    if devices:
        raise SystemExit(launch_workers(args, files, devices))

    shard_files = select_shard(files, args.shard_id, args.num_shards)
    _, _, _, errors = process_files(
        shard_files,
        output_root=args.output_root,
        source_root=args.source_root,
        model_dir=args.model_dir,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        num_overlap=args.num_overlap,
        use_amp=not args.no_amp,
        chunk_batch_size=args.chunk_batch_size,
        overwrite=args.overwrite,
        progress=not args.no_progress,
        worker_id=args.worker_id,
        progress_file=args.progress_file,
        save_workers=args.save_workers,
        max_pending_writes=args.max_pending_writes,
        max_duration_seconds=args.max_duration_seconds,
    )
    if errors:
        print(
            f"[worker-{args.worker_id}] per-file errors={errors}; "
            "bad samples were recorded with metadata status=error.",
            file=sys.stderr,
            flush=True,
        )
        if args.fail_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
