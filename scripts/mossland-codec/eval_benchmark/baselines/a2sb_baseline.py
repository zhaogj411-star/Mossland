from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, reference_path_for_item, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.a2sb_baseline ..."
    ) from exc


BASELINE_NAME = "a2sb"
TARGET_SAMPLE_RATE = 44100
OFFICIAL_REPO = "https://github.com/NVIDIA/diffusion-audio-restoration"
OFFICIAL_CHECKPOINT = "nvidia/audio_to_audio_schrodinger_bridge"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return repo_root() / "tmp/eval_baseline_refs/a2sb/diffusion-audio-restoration"


def default_checkpoint_dir() -> Path:
    return repo_root() / "checkpoints/a2sb"


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def ensure_a2sb_repo(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    required = [
        repo_path / "ensembled_inference_api.py",
        repo_path / "configs/ensemble_2split_sampling.yaml",
        repo_path / "configs/inference_files_upsampling.yaml",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"A2SB official repo is incomplete at {repo_path}. Missing: "
            f"{', '.join(str(path) for path in missing)}. Clone {OFFICIAL_REPO} there "
            "or pass --repo-dir pointing at the official repository root."
        )
    return repo_path


def discover_checkpoints(checkpoint_dir: str | Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        checkpoints = [Path(path).expanduser() for path in explicit]
    else:
        root = Path(checkpoint_dir)
        checkpoints = sorted(root.rglob("*.ckpt")) if root.exists() else []
        two_split = [path for path in checkpoints if "twosplit" in path.name]
        if len(two_split) == 2:
            checkpoints = sorted(two_split)

    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"A2SB checkpoint path(s) do not exist: {', '.join(str(path) for path in missing)}")
    if len(checkpoints) < 1:
        raise FileNotFoundError(
            f"No A2SB .ckpt files found under {checkpoint_dir}. Download {OFFICIAL_CHECKPOINT} "
            "to checkpoints/a2sb, or pass one or more --checkpoint paths. The default official "
            "sampling config expects the two split checkpoints used by A2SB bandwidth extension."
        )
    return [path.resolve() for path in checkpoints]


def cutoff_for_item(item: EvalItem, override_hz: float | None) -> float:
    if override_hz is not None:
        cutoff = float(override_hz)
    else:
        if item.low_sample_rate is None or item.low_sample_rate <= 0:
            raise RuntimeError(
                f"item_id={item.item_id!r} has no positive low_sample_rate; pass --cutoff-hz "
                "or use a super_resolution manifest with low_sample_rate."
            )
        cutoff = float(item.low_sample_rate) / 2.0
    nyquist = TARGET_SAMPLE_RATE / 2.0
    if cutoff <= 0 or cutoff >= nyquist:
        raise RuntimeError(
            f"item_id={item.item_id!r} resolved cutoff_hz={cutoff}, expected 0 < cutoff < {nyquist}."
        )
    return cutoff


def low_bandwidth_reference(item: EvalItem, cutoff_hz: float) -> tuple[torch.Tensor, int]:
    reference_path = reference_path_for_item(item) or item.source_path
    audio = load_audio_segment(
        reference_path,
        sample_rate=TARGET_SAMPLE_RATE,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if audio.ndim != 2:
        raise RuntimeError(f"Expected reference audio [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    target_length = audio.shape[-1]
    if item.low_sample_rate is not None and item.low_sample_rate > 0 and abs(cutoff_hz - item.low_sample_rate / 2.0) < 1e-6:
        degraded = AF.resample(audio, TARGET_SAMPLE_RATE, item.low_sample_rate)
        degraded = AF.resample(degraded, item.low_sample_rate, TARGET_SAMPLE_RATE)
    else:
        degraded = AF.lowpass_biquad(audio, TARGET_SAMPLE_RATE, cutoff_hz)
    return fit_length(degraded.clamp(-1.0, 1.0), target_length), target_length


def fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def write_inference_config(
    path: Path,
    degraded_path: Path,
    cutoff_hz: float,
    checkpoints: list[Path],
    device: str,
) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("A2SB adapter needs PyYAML to write the temporary official config.") from exc

    if len(checkpoints) == 1:
        t_cutoffs: list[float] = []
    else:
        t_cutoffs = [round((idx + 1) / len(checkpoints), 6) for idx in range(len(checkpoints) - 1)]

    accelerator = "cpu"
    if device.startswith("cuda"):
        accelerator = "gpu"

    config: dict[str, Any] = {
        "trainer": {
            "accelerator": accelerator,
            "devices": 1,
            "strategy": "auto",
            "logger": None,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
        },
        "model": {
            "pretrained_checkpoints": [str(path) for path in checkpoints],
            "t_cutoffs": t_cutoffs,
        },
        "data": {
            "predict_filelist": [{"filepath": str(degraded_path), "output_subdir": "."}],
            "num_workers": 0,
            "batch_size": 1,
            "transforms_aug": [
                {
                    "class_path": "corruption.corruptions.MultinomialInpaintMaskTransform",
                    "init_args": {
                        "p_upsample_mask": 1.0,
                        "p_extension_mask": 0.0,
                        "p_inpaint_mask": 0.0,
                        "fill_noise_level": 0.5,
                        "sampling_rate": TARGET_SAMPLE_RATE,
                        "upsample_mask_kwargs": {
                            "min_cutoff_freq": int(round(cutoff_hz)),
                            "max_cutoff_freq": int(round(cutoff_hz)),
                        },
                        "inpainting_mask_kwargs": {
                            "min_inpainting_frac": 0.1013,
                            "max_inpainting_frac": 0.1013,
                            "is_random": False,
                        },
                    },
                }
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def subprocess_env(device: str) -> dict[str, str]:
    env = os.environ.copy()
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    return env


def run_official_a2sb(
    repo_dir: Path,
    config_path: Path,
    output_path: Path,
    predict_n_steps: int,
    device: str,
) -> None:
    command = [
        sys.executable,
        "ensembled_inference_api.py",
        "predict",
        "-c",
        "configs/ensemble_2split_sampling.yaml",
        "-c",
        str(config_path),
        f"--model.predict_n_steps={predict_n_steps}",
        f"--model.output_audio_filename={output_path}",
    ]
    # The official simple API wraps this same Lightning entry point but auto-detects
    # cutoff from spectral rolloff. The adapter calls the script directly so the
    # eval manifest can supply the cutoff protocol deterministically.
    result = subprocess.run(
        command,
        cwd=str(repo_dir),
        env=subprocess_env(device),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Official A2SB inference failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Official A2SB command completed but did not create a non-empty output wav: {output_path}"
        )


def normalize_prediction(temp_output_path: Path, final_output_path: Path, target_length: int) -> None:
    prediction, sr = torchaudio.load(str(temp_output_path))
    if sr != TARGET_SAMPLE_RATE:
        prediction = AF.resample(prediction, sr, TARGET_SAMPLE_RATE)
    if prediction.ndim != 2:
        raise RuntimeError(f"A2SB returned audio with unexpected shape {tuple(prediction.shape)}.")
    if prediction.shape[0] > 1:
        prediction = prediction.mean(dim=0, keepdim=True)
    save_audio(final_output_path, fit_length(prediction[:1], target_length), TARGET_SAMPLE_RATE)


def run_a2sb_item(item: EvalItem, output_path: Path, args: argparse.Namespace, repo_dir: Path, checkpoints: list[Path]) -> None:
    cutoff_hz = cutoff_for_item(item, args.cutoff_hz)
    degraded, target_length = low_bandwidth_reference(item, cutoff_hz)

    with tempfile.TemporaryDirectory(prefix="mossland_a2sb_") as tmpdir:
        tmp_path = Path(tmpdir)
        degraded_path = tmp_path / f"{item.item_id}_lowband.wav"
        temp_output_path = tmp_path / f"{item.item_id}_a2sb.wav"
        config_path = tmp_path / "a2sb_inference.yaml"
        save_audio(degraded_path, degraded, TARGET_SAMPLE_RATE)
        write_inference_config(config_path, degraded_path, cutoff_hz, checkpoints, args.device)
        run_official_a2sb(repo_dir, config_path, temp_output_path, args.predict_n_steps, args.device)
        normalize_prediction(temp_output_path, output_path, target_length)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    repo_dir = ensure_a2sb_repo(args.repo_dir)
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoint)
    if not shutil.which(sys.executable):
        raise RuntimeError(f"Python executable not found: {sys.executable}")

    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_a2sb_item(item, output_path, args, repo_dir, checkpoints)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NVIDIA A2SB bandwidth-extension baseline for Mossland super_resolution manifests."
    )
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to official A2SB repo root.")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(default_checkpoint_dir()),
        help="Directory containing downloaded NVIDIA A2SB .ckpt files.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Explicit A2SB .ckpt path. Repeat for split checkpoints; overrides --checkpoint-dir discovery.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for official A2SB subprocess. cuda:N is mapped through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=None,
        help="Override cutoff frequency. By default uses item.low_sample_rate / 2.",
    )
    parser.add_argument("--predict-n-steps", type=int, default=50, help="A2SB diffusion sampling steps.")
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
