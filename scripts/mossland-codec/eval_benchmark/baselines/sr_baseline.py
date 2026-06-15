from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.sr_baseline ..."
    ) from exc


BASELINE_NAME = "audiosr"
AUDIOSR_MODEL_SAMPLE_RATE = 48000
OFFICIAL_AUDIOSR_REPO = "https://github.com/haoheliu/versatile_audio_super_resolution"
OFFICIAL_AUDIOSR_CHECKPOINT = "haoheliu/audiosr_basic"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_audiosr_source_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/sr/versatile_audio_super_resolution"


def _add_audiosr_source_dir(source_dir: str | Path) -> None:
    source_path = Path(source_dir)
    if not source_path.exists():
        return
    source_string = str(source_path)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)


def _load_audiosr(args: argparse.Namespace) -> Any:
    _add_audiosr_source_dir(args.audiosr_source_dir)
    try:
        from audiosr import build_model
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "AudioSR is not importable. Pull the official repository with "
            f"`git clone {OFFICIAL_AUDIOSR_REPO} tmp/eval_baseline_refs/sr/versatile_audio_super_resolution` "
            "or install the official package `python -m pip install audiosr==0.0.7`. "
            "On this machine, importing the local official source may also require clearing "
            "HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and setting HF_ENDPOINT=https://huggingface.co so "
            "`RobertaTokenizer.from_pretrained('roberta-base')` resolves correctly."
        ) from exc

    try:
        return build_model(model_name=args.model_name, device=args.device)
    except Exception as exc:  # pragma: no cover - network/checkpoint dependent
        raise RuntimeError(
            "Failed to load the official AudioSR model. This adapter uses the official "
            f"AudioSR code from {OFFICIAL_AUDIOSR_REPO} and the Hugging Face checkpoint "
            f"{OFFICIAL_AUDIOSR_CHECKPOINT} (`pytorch_model.bin`) for --model-name basic. "
            "If download or tokenizer loading fails, retry with "
            "`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY HF_ENDPOINT=https://huggingface.co ...`."
        ) from exc


def _low_bandwidth_audio(item: EvalItem) -> torch.Tensor:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if audio.ndim != 2:
        raise RuntimeError(f"Expected source audio [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if item.low_sample_rate is not None and item.low_sample_rate > 0:
        if item.low_sample_rate >= item.sample_rate:
            raise RuntimeError(
                f"item_id={item.item_id!r} has low_sample_rate={item.low_sample_rate}, "
                f"which is not below sample_rate={item.sample_rate}."
            )
        audio = AF.resample(audio, item.sample_rate, item.low_sample_rate)
        audio = AF.resample(audio, item.low_sample_rate, item.sample_rate)
    return audio.clamp(-1.0, 1.0)


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def _run_audiosr_item(model: Any, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    try:
        from audiosr import super_resolution, super_resolution_long_audio
    except Exception as exc:  # pragma: no cover - import covered by _load_audiosr
        raise RuntimeError("AudioSR import disappeared after model load.") from exc

    low_band_audio = _low_bandwidth_audio(item)
    target_length = low_band_audio.shape[-1]

    with tempfile.TemporaryDirectory(prefix="mossland_audiosr_") as tmpdir:
        low_band_path = Path(tmpdir) / f"{item.item_id}_lowband.wav"
        save_audio(low_band_path, low_band_audio, item.sample_rate)

        try:
            with torch.inference_mode():
                if args.chunking:
                    waveform = super_resolution_long_audio(
                        model,
                        str(low_band_path),
                        seed=item.seed,
                        guidance_scale=args.guidance_scale,
                        ddim_steps=args.ddim_steps,
                        chunk_duration_s=args.chunk_duration,
                        overlap_duration_s=args.overlap_duration,
                    )
                else:
                    waveform = super_resolution(
                        model,
                        str(low_band_path),
                        seed=item.seed,
                        guidance_scale=args.guidance_scale,
                        ddim_steps=args.ddim_steps,
                    )
        except Exception as exc:  # pragma: no cover - model runtime dependent
            raise RuntimeError(
                f"Official AudioSR inference failed for item_id={item.item_id!r}. "
                "This is a real AudioSR runtime failure, not a fallback upsampler."
            ) from exc

    prediction = torch.as_tensor(waveform).detach().cpu().float()
    if prediction.ndim == 3:
        prediction = prediction[0]
    elif prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 2:
        raise RuntimeError(
            f"AudioSR returned audio with unexpected shape {tuple(prediction.shape)} "
            f"for item_id={item.item_id!r}."
        )

    if prediction.shape[0] > 1:
        prediction = prediction[:1]
    if item.sample_rate != AUDIOSR_MODEL_SAMPLE_RATE:
        prediction = AF.resample(prediction, AUDIOSR_MODEL_SAMPLE_RATE, item.sample_rate)
    prediction = _fit_length(prediction, target_length)
    save_audio(output_path, prediction, item.sample_rate)


def _super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def _run_audiosr_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = _load_audiosr(args)
    prediction_paths: list[Path] = []
    for item in items:
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            continue
        _run_audiosr_item(model, item, output_path, args)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an official audio super-resolution baseline for Mossland eval manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--baseline",
        default="audiosr",
        choices=("audiosr", "aero", "nu-wave2"),
        help="Official SR baseline to run. Only AudioSR is currently wired locally.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for baseline inference.",
    )
    parser.add_argument(
        "--audiosr-source-dir",
        default=str(_default_audiosr_source_dir()),
        help="Path to the cloned official AudioSR repository; used before installed packages.",
    )
    parser.add_argument(
        "--model-name",
        default="basic",
        choices=("basic", "speech"),
        help="AudioSR checkpoint name passed to official build_model().",
    )
    parser.add_argument("--ddim-steps", type=int, default=50, help="AudioSR DDIM sampling steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="AudioSR unconditional guidance scale.",
    )
    parser.add_argument("--chunking", action="store_true", help="Use AudioSR long-audio chunking path.")
    parser.add_argument("--chunk-duration", type=int, default=15, help="AudioSR chunk duration in seconds.")
    parser.add_argument("--overlap-duration", type=int, default=2, help="AudioSR overlap duration in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline != "audiosr":
        raise RuntimeError(
            f"--baseline {args.baseline!r} is not wired in this adapter yet. "
            "AudioSR was attempted first as requested; add the official AERO or NU-Wave2 "
            "repository/checkpoint integration before using this option."
        )

    items = _super_resolution_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return

    prediction_paths = _run_audiosr_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
