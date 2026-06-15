from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.ambisonizer_baseline ..."
    ) from exc


BASELINE_NAME = "ambisonizer"
OFFICIAL_SAMPLE_RATE = 44100
OFFICIAL_LENGTH = 480000
DEFAULT_REPO_DIR = Path("tmp/eval_baseline_refs/ambisonizer/ambisonizer")
DEFAULT_REPO_URL = "https://github.com/yongyizang/ambisonizer.git"
DEFAULT_CHECKPOINT = Path("checkpoints/ambisonizer/ambisonizer_state_dict.pt")


def _clone_repo(repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", DEFAULT_REPO_URL, str(repo_dir)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Ambisonizer official repo is missing at {repo_dir} and automatic clone "
            f"from {DEFAULT_REPO_URL} failed. Clone it manually or pass --repo-dir."
        ) from exc


def _add_official_repo_to_path(repo_dir: Path, clone_if_missing: bool) -> None:
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "model" / "seanet.py").exists():
        if clone_if_missing and not repo_dir.exists():
            _clone_repo(repo_dir)
    if not (repo_dir / "model" / "seanet.py").exists():
        raise RuntimeError(
            f"Ambisonizer official repo not found at {repo_dir}. Clone "
            f"{DEFAULT_REPO_URL} there, or pass --repo-dir."
        )
    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Ambisonizer checkpoint is missing: {path}. Download the official model "
            "weights from the Ambisonizer README and pass --checkpoint, or place them "
            f"at the default path {DEFAULT_CHECKPOINT}."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Ambisonizer checkpoint at {path}.") from exc

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Ambisonizer checkpoint at {path} does not contain a state dict.")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(key, str)
    }


def load_ambisonizer(args: argparse.Namespace, device: torch.device):
    _add_official_repo_to_path(Path(args.repo_dir), clone_if_missing=not args.no_clone)
    try:
        from model.seanet import SEANet
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Failed to import the official Ambisonizer SEANet. Install the official "
            "dependencies from requirements.txt, especially torch, torchaudio, numpy "
            "and einops, then rerun this baseline."
        ) from exc

    model = SEANet(OFFICIAL_LENGTH, 64)
    state_dict = _load_state_dict(Path(args.checkpoint), device)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            "Ambisonizer checkpoint did not match the official SEANet(480000, 64) "
            "architecture used by inference.ipynb."
        ) from exc
    return model.to(device).eval()


def _input_path_for_item(item: EvalItem) -> Path:
    if item.source_path is not None and str(item.source_path):
        return item.source_path
    if item.reference_path is not None:
        return item.reference_path
    raise RuntimeError(f"item_id={item.item_id!r} has no source_path or reference_path.")


def _mono_condition_for_item(item: EvalItem) -> tuple[torch.Tensor, int]:
    input_path = _input_path_for_item(item)
    audio = load_audio_segment(
        input_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    original_length = audio.shape[-1]
    mono = audio.mean(dim=0, keepdim=True)
    if item.sample_rate != OFFICIAL_SAMPLE_RATE:
        mono = AF.resample(mono, item.sample_rate, OFFICIAL_SAMPLE_RATE)
    mono = mono.clamp(-1.0, 1.0)
    if mono.shape[-1] > OFFICIAL_LENGTH:
        mono = mono[..., :OFFICIAL_LENGTH]
    elif mono.shape[-1] < OFFICIAL_LENGTH:
        mono = torch.nn.functional.pad(mono, (0, OFFICIAL_LENGTH - mono.shape[-1]))
    return mono, original_length


def _decode_stereo(foa_wxy: torch.Tensor, left_azimuth: float, right_azimuth: float, w_gain: float) -> torch.Tensor:
    azimuths = torch.deg2rad(
        torch.tensor([left_azimuth, right_azimuth], dtype=foa_wxy.dtype, device=foa_wxy.device)
    )
    w = foa_wxy[0]
    x = foa_wxy[1]
    y = foa_wxy[2]
    left = float(w_gain) * w + x * torch.cos(azimuths[0]) + y * torch.sin(azimuths[0])
    right = float(w_gain) * w + x * torch.cos(azimuths[1]) + y * torch.sin(azimuths[1])
    stereo = torch.stack((left, right), dim=0)
    peak = stereo.abs().amax()
    if peak > 1.0:
        stereo = stereo / peak
    return stereo.clamp(-1.0, 1.0)


def infer_item(
    item: EvalItem,
    output_path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    mono, original_length = _mono_condition_for_item(item)
    model_input = mono.repeat(2, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        predicted_xy, _, _ = model(model_input)
    xy = predicted_xy.squeeze(0).detach().cpu().float()
    if xy.shape[0] != 2:
        raise RuntimeError(
            f"Ambisonizer returned unexpected output shape {tuple(predicted_xy.shape)} "
            f"for item_id={item.item_id!r}; expected [B, 2, {OFFICIAL_LENGTH}]."
        )
    foa_wxy = torch.cat((mono.cpu().float(), xy[..., :OFFICIAL_LENGTH]), dim=0)
    prediction = _decode_stereo(
        foa_wxy,
        left_azimuth=args.left_azimuth,
        right_azimuth=args.right_azimuth,
        w_gain=args.w_gain,
    )

    if args.preserve_input_length:
        target_length = original_length
        if item.sample_rate != OFFICIAL_SAMPLE_RATE:
            prediction = AF.resample(prediction, OFFICIAL_SAMPLE_RATE, item.sample_rate)
        prediction = prediction[..., :target_length]
        if prediction.shape[-1] < target_length:
            prediction = torch.nn.functional.pad(prediction, (0, target_length - prediction.shape[-1]))
        save_audio(output_path, prediction, item.sample_rate)
    else:
        save_audio(output_path, prediction, OFFICIAL_SAMPLE_RATE)


def _mono_to_stereo_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "mono_to_stereo"]
    if max_items > 0:
        return items[:max_items]
    return items


def _run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    device = torch.device(args.device)
    model = load_ambisonizer(args, device)
    prediction_paths: list[Path] = []
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, len(items), item.item_id, args.progress_every)
            continue
        infer_item(item, output_path, model, args, device)
        maybe_print_progress(index, len(items), item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ambisonizer SEANet as a mono-to-stereo baseline via FOA decoding."
    )
    add_common_args(parser)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for Ambisonizer inference.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help=f"Local clone of {DEFAULT_REPO_URL}. Missing default repo is cloned automatically.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Do not automatically clone the official Ambisonizer repo when --repo-dir is missing.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the official Ambisonizer SEANet state_dict checkpoint.",
    )
    parser.add_argument(
        "--left-azimuth",
        type=float,
        default=-60.0,
        help="Left speaker azimuth in degrees for FOA W/X/Y stereo decoding.",
    )
    parser.add_argument(
        "--right-azimuth",
        type=float,
        default=120.0,
        help="Right speaker azimuth in degrees for FOA W/X/Y stereo decoding.",
    )
    parser.add_argument(
        "--w-gain",
        type=float,
        default=0.3,
        help="W-channel gain used by the official synthesize.ipynb stereo decode.",
    )
    parser.add_argument(
        "--preserve-input-length",
        action="store_true",
        help=(
            "Resample/crop the decoded stereo output back to each manifest item's input length. "
            "By default the official fixed 480000-sample, 44.1 kHz output is saved."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = _mono_to_stereo_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return
    prediction_paths = _run_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
