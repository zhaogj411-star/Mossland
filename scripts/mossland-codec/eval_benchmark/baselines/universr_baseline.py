from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "universr_audio"
TARGET_SAMPLE_RATE = 48000
SUPPORTED_INPUT_RATES = {8000, 12000, 16000, 24000}
OFFICIAL_REPO = "https://github.com/woongzip1/UniverSR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/universr/UniverSR"


def default_checkpoint_dir() -> Path:
    return _repo_root() / "checkpoints/universr-audio"


def add_official_repo_to_path(repo_dir: str | Path) -> None:
    repo_path = Path(repo_dir)
    if not (repo_path / "universr").is_dir():
        raise FileNotFoundError(
            f"UniverSR official repo not found at {repo_path}. Clone {OFFICIAL_REPO} there "
            "or pass --repo-dir."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def load_universr(args: argparse.Namespace):
    add_official_repo_to_path(args.repo_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "config.yaml").exists() or not (checkpoint_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError(
            f"UniverSR checkpoint directory must contain config.yaml and pytorch_model.bin: {checkpoint_dir}. "
            "Download with `HF_ENDPOINT=https://huggingface.co huggingface-cli download "
            "woongzip1/universr-audio --local-dir checkpoints/universr-audio`."
        )
    try:
        from universr import UniverSR
    except Exception as exc:  # pragma: no cover - optional external source path
        raise RuntimeError(
            "Failed to import the official UniverSR source. This adapter expects the official "
            f"repository from {OFFICIAL_REPO} at --repo-dir."
        ) from exc
    return UniverSR.from_pretrained(str(checkpoint_dir), device=args.device)


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def low_bandwidth_audio_48k(item: EvalItem) -> torch.Tensor:
    low_sample_rate = item.low_sample_rate or 16000
    if low_sample_rate not in SUPPORTED_INPUT_RATES:
        raise RuntimeError(
            f"item_id={item.item_id!r} has low_sample_rate={low_sample_rate}; "
            f"official UniverSR supports {sorted(SUPPORTED_INPUT_RATES)} only."
        )
    audio = load_audio_segment(
        item.source_path,
        sample_rate=TARGET_SAMPLE_RATE,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    original_len = audio.shape[-1]
    audio = AF.resample(audio, TARGET_SAMPLE_RATE, low_sample_rate)
    audio = AF.resample(audio, low_sample_rate, TARGET_SAMPLE_RATE)
    return audio[..., :original_len].clamp(-1.0, 1.0)


def run_universr_item(model, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    low_sr = item.low_sample_rate or 16000
    torch.manual_seed(item.seed)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(item.seed)
    with torch.inference_mode():
        output = model.enhance(
            low_bandwidth_audio_48k(item),
            input_sr=low_sr,
            ode_method=args.ode_method,
            ode_steps=args.ode_steps,
            guidance_scale=None if args.guidance_scale <= 0 else args.guidance_scale,
        )
    if output.ndim == 1:
        output = output.unsqueeze(0)
    if output.ndim == 3:
        output = output[0]
    save_audio(output_path, output[:1].detach().cpu(), TARGET_SAMPLE_RATE)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_universr(args)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_universr_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official UniverSR audio super-resolution baseline for Mossland manifests."
    )
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to the official UniverSR repo.")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(default_checkpoint_dir()),
        help="Local Hugging Face snapshot directory for woongzip1/universr-audio.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for UniverSR inference.",
    )
    parser.add_argument("--ode-method", default="midpoint", choices=("euler", "midpoint", "rk4"))
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
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
