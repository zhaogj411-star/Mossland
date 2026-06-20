from __future__ import annotations

import argparse
import contextlib
import io
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
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.s3a_baseline ..."
    ) from exc


BASELINE_NAME = "s3a"
OFFICIAL_SAMPLE_RATE = 48000
DEFAULT_REPO_DIR = Path("tmp/eval_baseline_refs/s3a/s3a-decorrelation-toolbox")


def _add_official_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "s3a_decorrelation_toolbox" / "s3a_decorrelator.py").exists():
        raise RuntimeError(
            f"s3a-decorrelation-toolbox official repo not found at {repo_dir}. Clone "
            "https://github.com/s3a-spatialaudio/decorrelation-python-toolbox.git there, "
            "or pass --repo-dir."
        )
    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _load_official_s3a(repo_dir: str | Path) -> Any:
    _add_official_repo_to_path(Path(repo_dir))
    try:
        import s3a_decorrelation_toolbox.s3a_decorrelator as s3a
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Failed to import the official s3a-decorrelation-toolbox modules. Install "
            "its dependencies from setup.py, especially librosa, scipy, soundfile, "
            "acoustics, pyloudnorm and matplotlib, then rerun with --mode official."
        ) from exc
    return s3a


def _mono_condition_for_item(item: EvalItem) -> torch.Tensor:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    return audio.mean(dim=0, keepdim=True).clamp(-1.0, 1.0)


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def _to_stereo_tensor(audio: np.ndarray, item: EvalItem, target_length: int) -> torch.Tensor:
    prediction = torch.as_tensor(audio, dtype=torch.float32)
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 2:
        raise RuntimeError(
            f"Official s3a returned unexpected audio shape {tuple(prediction.shape)} "
            f"for item_id={item.item_id!r}."
        )
    prediction = prediction.transpose(0, 1)
    if prediction.shape[0] == 1:
        prediction = prediction.repeat(2, 1)
    elif prediction.shape[0] > 2:
        prediction = prediction[:2]
    if item.sample_rate != OFFICIAL_SAMPLE_RATE:
        prediction = AF.resample(prediction, OFFICIAL_SAMPLE_RATE, item.sample_rate)
    return _fit_length(prediction, target_length).clamp(-1.0, 1.0)


def _run_official_item(
    s3a: Any,
    item: EvalItem,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    mono = _mono_condition_for_item(item)
    target_length = mono.shape[-1]
    official_mono = mono
    if item.sample_rate != OFFICIAL_SAMPLE_RATE:
        official_mono = AF.resample(mono, item.sample_rate, OFFICIAL_SAMPLE_RATE)
    audio_np = official_mono.squeeze(0).detach().cpu().numpy().astype(np.float64)

    # The official toolbox has stochastic routing defaults and prints parsed kwargs.
    # Fix routing/seed and suppress that print so manifest generation stays stable.
    state = np.random.get_state()
    np.random.seed(int(item.seed) & 0xFFFFFFFF)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prediction_np = s3a.s3a_decorrelator(
                audio_np,
                output_filename=None,
                preset=args.preset,
                duration=None,
                make_mono=False,
                fs=OFFICIAL_SAMPLE_RATE,
                num_out_chans=2,
                transient_routing=[0, 1],
                steady_state_routing=[0, 1],
            )
    finally:
        np.random.set_state(state)

    prediction = _to_stereo_tensor(prediction_np, item, target_length)
    save_audio(output_path, prediction, item.sample_rate)


def _allpass_filter(x: torch.Tensor, delay: int, gain: float) -> torch.Tensor:
    y = torch.empty_like(x)
    for n in range(x.numel()):
        delayed_x = x[n - delay] if n >= delay else x.new_tensor(0.0)
        delayed_y = y[n - delay] if n >= delay else x.new_tensor(0.0)
        y[n] = -gain * x[n] + delayed_x + gain * delayed_y
    return y


def _fallback_decorrelate(mono: torch.Tensor, sample_rate: int, seed: int, width: float) -> torch.Tensor:
    mono_1d = mono.squeeze(0).float()
    if mono_1d.numel() == 0:
        return mono_1d.new_zeros(2, 0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    delay_ms = 6.0 + float(torch.randint(0, 8, (1,), generator=generator).item())
    delay = max(1, int(round(sample_rate * delay_ms / 1000.0)))
    gain = 0.62

    # Fallback until this adapter is replaced by a stricter official s3a API path:
    # generate a decorrelated side signal with a short all-pass/phase path and use
    # mid-side rendering, so (L + R) / 2 remains the original mono condition.
    side = _allpass_filter(mono_1d, delay=delay, gain=gain)
    side = side - mono_1d
    side = side - side.mean()
    side_peak = side.abs().amax().clamp_min(1e-6)
    mono_peak = mono_1d.abs().amax().clamp_min(1e-6)
    side = side * (mono_peak / side_peak) * float(width)

    stereo = torch.stack((mono_1d + side, mono_1d - side), dim=0)
    peak = stereo.abs().amax()
    if peak > 1.0:
        stereo = stereo / peak
    return stereo.clamp(-1.0, 1.0)


def _run_fallback_item(item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    mono = _mono_condition_for_item(item)
    prediction = _fallback_decorrelate(
        mono,
        sample_rate=item.sample_rate,
        seed=item.seed,
        width=args.fallback_width,
    )
    save_audio(output_path, prediction, item.sample_rate)


def _mono_to_stereo_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "mono_to_stereo"]
    if max_items > 0:
        return items[:max_items]
    return items


def _run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    s3a = None
    if args.mode in {"official", "auto"}:
        try:
            s3a = _load_official_s3a(args.repo_dir)
        except RuntimeError:
            if args.mode == "official":
                raise
            print("official s3a unavailable; using fallback DSP decorrelator", flush=True)

    prediction_paths: list[Path] = []
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, len(items), item.item_id, args.progress_every)
            continue
        if s3a is None:
            _run_fallback_item(item, output_path, args)
        else:
            try:
                _run_official_item(s3a, item, output_path, args)
            except Exception:
                if args.mode != "auto":
                    raise
                print(
                    f"official s3a failed for item_id={item.item_id!r}; "
                    "using fallback DSP decorrelator",
                    flush=True,
                )
                _run_fallback_item(item, output_path, args)
        maybe_print_progress(index, len(items), item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run s3a-decorrelation-toolbox as a mono-to-stereo DSP baseline."
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Local clone of s3a-spatialaudio/decorrelation-python-toolbox.",
    )
    parser.add_argument(
        "--mode",
        choices=("official", "fallback", "auto"),
        default="auto",
        help=(
            "official requires the s3a toolbox import/runtime to work; fallback uses a "
            "mono-compatible all-pass decorrelator; auto tries official then fallback."
        ),
    )
    parser.add_argument(
        "--preset",
        default="upmix",
        choices=("upmix", "diffuse", "upmix_lauridsen4"),
        help="s3a preset passed to the official toolbox when --mode is official or auto.",
    )
    parser.add_argument(
        "--fallback-width",
        type=float,
        default=0.35,
        help="Mid-side side gain for the fallback all-pass decorrelator.",
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
