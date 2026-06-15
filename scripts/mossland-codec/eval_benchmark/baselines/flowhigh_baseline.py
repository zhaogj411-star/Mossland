from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import (
        add_common_args,
        baseline_prediction_path,
        maybe_print_progress,
        write_predicted_manifest,
    )
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.flowhigh_baseline ..."
    ) from exc


BASELINE_NAME = "flowhigh"
DEFAULT_TARGET_SAMPLE_RATE = 48000
DEFAULT_SR_IN = 16000
DEFAULT_TIMESTEP = 1
OFFICIAL_FLOWHIGH_REPO = "https://github.com/resemble-ai/flowhigh"
OFFICIAL_FLOWHIGH_CHECKPOINT = "ResembleAI/FlowHigh"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/flowhigh/flowhigh/src"


def default_checkpoint_dir() -> Path:
    return _repo_root() / "checkpoints/flowhigh"


def add_official_repo_to_path(repo_dir: str | Path) -> None:
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise FileNotFoundError(
            f"FlowHigh source tree not found at {repo_path}. Clone {OFFICIAL_FLOWHIGH_REPO} "
            "there or pass --repo-dir pointing at the repository's flowhigh/src directory."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def load_flowhigh(args: argparse.Namespace) -> Any:
    if not str(args.device).startswith("cuda"):
        raise RuntimeError(
            "FlowHigh official from_local()/generate() currently hard-code CUDA paths. "
            "Use --device cuda or cuda:N on a CUDA-compatible machine."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("FlowHigh official inference requires torch.cuda.is_available().")
    if str(args.device) != "cuda":
        torch.cuda.set_device(torch.device(args.device))

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"FlowHigh checkpoint directory not found: {checkpoint_dir}. Download "
            f"{OFFICIAL_FLOWHIGH_CHECKPOINT} there or pass --checkpoint-dir/--model-dir."
        )

    add_official_repo_to_path(args.repo_dir)
    try:
        from flowhigh import FlowHighSR
    except Exception as exc:  # pragma: no cover - optional external source path
        raise RuntimeError(
            "FlowHigh is not importable. This adapter expects the maintained official source "
            f"from {OFFICIAL_FLOWHIGH_REPO} at --repo-dir."
        ) from exc

    try:
        return FlowHighSR.from_local(checkpoint_dir, device=args.device)
    except Exception as exc:  # pragma: no cover - checkpoint/CUDA dependent
        raise RuntimeError(
            "Failed to load FlowHigh from the local checkpoint directory. This adapter uses "
            f"the official checkpoint layout from {OFFICIAL_FLOWHIGH_CHECKPOINT}: "
            "FLowHigh_basic_400k.pt, bigvgan_48khz_256band.pt, and the matching json configs."
        ) from exc


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def downmix_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim != 2:
        raise RuntimeError(f"Expected audio [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] == 1:
        return audio
    return audio.mean(dim=0, keepdim=True)


def fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def prepare_flowhigh_input(item: EvalItem, target_sample_rate: int) -> tuple[np.ndarray, int, int]:
    low_sample_rate = item.low_sample_rate or DEFAULT_SR_IN
    if low_sample_rate >= target_sample_rate:
        raise RuntimeError(
            f"item_id={item.item_id!r} has low_sample_rate={low_sample_rate}, "
            f"which must be below target_sample_rate={target_sample_rate} for FlowHigh."
        )

    audio = load_audio_segment(
        item.source_path,
        sample_rate=target_sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    mono = downmix_mono(audio).clamp(-1.0, 1.0)
    low = AF.resample(mono, target_sample_rate, low_sample_rate).clamp(-1.0, 1.0)
    return low.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False), low_sample_rate, mono.shape[-1]


def call_flowhigh_generate(
    model: Any,
    low_audio: np.ndarray,
    sr_in: int,
    target_sample_rate: int,
    timestep: int,
) -> Any:
    signature = inspect.signature(model.generate)
    kwargs: dict[str, Any] = {
        "target_sampling_rate": target_sample_rate,
        "timestep": timestep,
    }
    if "sr_in" in signature.parameters:
        kwargs["sr_in"] = sr_in
        return model.generate(low_audio, **kwargs)
    kwargs["sr"] = sr_in
    return model.generate(low_audio, **kwargs)


def tensor_from_flowhigh_output(output: Any) -> torch.Tensor:
    if isinstance(output, np.ndarray):
        prediction = torch.from_numpy(output)
    else:
        prediction = torch.as_tensor(output)
    prediction = prediction.detach().cpu().float()
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    elif prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise RuntimeError(f"FlowHigh returned audio with unexpected shape {tuple(prediction.shape)}.")
    return downmix_mono(prediction).clamp(-1.0, 1.0)


def run_flowhigh_item(model: Any, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    low_audio, sr_in, target_length = prepare_flowhigh_input(item, args.target_sample_rate)
    with torch.inference_mode():
        output = call_flowhigh_generate(
            model,
            low_audio,
            sr_in=sr_in,
            target_sample_rate=args.target_sample_rate,
            timestep=args.timestep,
        )
    prediction = fit_length(tensor_from_flowhigh_output(output), target_length)
    save_audio(output_path, prediction, args.target_sample_rate)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_flowhigh(args)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_flowhigh_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official FlowHigh super-resolution baseline for Mossland manifests."
    )
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to FlowHigh flowhigh/src.")
    parser.add_argument(
        "--checkpoint-dir",
        "--model-dir",
        dest="checkpoint_dir",
        default=str(default_checkpoint_dir()),
        help="Local FlowHigh checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string passed to FlowHighSR.from_local(). Official code requires CUDA.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="FlowHigh output sample rate.",
    )
    parser.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP, help="FlowHigh generation timestep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = super_resolution_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return
    prediction_paths = run_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
