from __future__ import annotations

import contextlib
import gc
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml


DEFAULT_MODEL_REPO = "KimberleyJensen/Mel-Band-Roformer-Vocal-Model"
DEFAULT_MODEL_DIR_NAME = "mel-band-roformer-vocal-model"
DEFAULT_CONFIG_NAME = "config_vocals_mel_band_roformer.yaml"
DEFAULT_CHECKPOINT_NAME = "MelBandRoformer.ckpt"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_OUTPUT_SAMPLE_RATE = 44100
DEFAULT_NUM_CHANNELS = 2
DEFAULT_NUM_OVERLAP = 2
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".webm",
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


def config_with_num_overlap(model_config: Mapping, num_overlap: int | None) -> dict:
    copied = json.loads(json.dumps(model_config))
    if num_overlap is not None:
        copied.setdefault("inference", {})["num_overlap"] = int(num_overlap)
    return copied


def config_sample_rate(model_config: Mapping) -> int:
    audio_cfg = model_config.get("audio", {})
    model_cfg = model_config.get("model", {})
    return int(audio_cfg.get("sample_rate", model_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)))


def config_num_channels(model_config: Mapping) -> int:
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
        "sample_rate": config_sample_rate(model_config),
        "num_channels": config_num_channels(model_config),
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
    source_oss_path: str | None = None,
    output_sample_rate: int | None = None,
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
    if source_oss_path is not None:
        metadata["source_oss_path"] = source_oss_path
    if output_sample_rate is not None:
        metadata["output_sample_rate"] = int(output_sample_rate)
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


def probe_audio_duration_seconds(path: Path | str) -> float | None:
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


def _load_audio_with_source_info(path: Path, sample_rate: int, num_channels: int) -> LoadedAudio:
    try:
        audio, in_sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        loaded = _load_audio_with_ffmpeg(path, sample_rate, num_channels)
        if loaded is None:
            raise exc
        return loaded

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


def _load_audio_with_ffmpeg(path: Path, sample_rate: int, num_channels: int) -> LoadedAudio | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    source_sample_rate, source_num_channels = _probe_audio_stream_info(path)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(num_channels),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None

    decoded = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if decoded.size == 0:
        return None
    usable = decoded.size - (decoded.size % num_channels)
    if usable <= 0:
        return None
    audio = torch.from_numpy(decoded[:usable].reshape(-1, num_channels).T).float()
    return LoadedAudio(
        audio=audio.contiguous().clamp(-1.0, 1.0),
        source_sample_rate=int(source_sample_rate or sample_rate),
        source_num_channels=int(source_num_channels or num_channels),
    )


def _probe_audio_stream_info(path: Path | str) -> tuple[int | None, int | None]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None, None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if completed.returncode != 0:
        return None, None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None, None
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None, None
    stream = streams[0]
    if not isinstance(stream, Mapping):
        return None, None
    sample_rate = stream.get("sample_rate")
    channels = stream.get("channels")
    try:
        parsed_sample_rate = int(sample_rate) if sample_rate is not None else None
    except (TypeError, ValueError):
        parsed_sample_rate = None
    try:
        parsed_channels = int(channels) if channels is not None else None
    except (TypeError, ValueError):
        parsed_channels = None
    return parsed_sample_rate, parsed_channels


def _save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu().float(), sample_rate)


def _resample_for_save(audio: torch.Tensor, sample_rate: int, output_sample_rate: int) -> torch.Tensor:
    if sample_rate == output_sample_rate:
        return audio
    return torchaudio.functional.resample(audio.detach().cpu().float(), sample_rate, output_sample_rate)


def cleanup_cuda_memory(device: torch.device | str | None = None) -> None:
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


def _can_copy_source_mixture(audio_path: Path | str, result: SeparationResult) -> bool:
    audio_path = Path(audio_path)
    return (
        audio_path.suffix.lower() == ".mp3"
        and result.sample_rate == DEFAULT_OUTPUT_SAMPLE_RATE
        and result.source_sample_rate == DEFAULT_OUTPUT_SAMPLE_RATE
        and result.source_num_channels == int(result.mixture.shape[0])
    )


def write_separation_result(
    layout: SeparationLayout,
    result: SeparationResult,
    audio_path: Path | str,
    model_config: Mapping,
    model_files: ModelFiles,
    source_oss_path: str | None = None,
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    save_workers: int = 3,
) -> None:
    layout.item_dir.mkdir(parents=True, exist_ok=True)
    save_workers = max(1, int(save_workers))
    save_tasks = []
    if _can_copy_source_mixture(audio_path, result):
        shutil.copyfile(audio_path, layout.mixture_path)
    else:
        save_tasks.append(
            (
                layout.mixture_path,
                _resample_for_save(result.mixture, result.sample_rate, output_sample_rate),
                output_sample_rate,
            )
        )
    save_tasks.append(
        (
            layout.vocals_path,
            _resample_for_save(result.vocals, result.sample_rate, output_sample_rate),
            output_sample_rate,
        )
    )
    save_tasks.append(
        (
            layout.accompaniment_path,
            _resample_for_save(result.accompaniment, result.sample_rate, output_sample_rate),
            output_sample_rate,
        )
    )

    if save_workers == 1 or len(save_tasks) <= 1:
        for path, audio, sample_rate in save_tasks:
            _save_audio(path, audio, sample_rate)
    else:
        with ThreadPoolExecutor(max_workers=min(save_workers, len(save_tasks))) as executor:
            futures = [executor.submit(_save_audio, path, audio, sample_rate) for path, audio, sample_rate in save_tasks]
            for future in futures:
                future.result()

    write_metadata(
        layout,
        audio_path,
        DEFAULT_MODEL_REPO,
        model_config,
        status="done",
        model_files=model_files,
        source_oss_path=source_oss_path,
        output_sample_rate=output_sample_rate,
    )


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
        self.effective_config = _to_config_view(config_with_num_overlap(self.raw_config, num_overlap))
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
        sample_rate = config_sample_rate(self.effective_config)
        num_channels = config_num_channels(self.effective_config)
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
