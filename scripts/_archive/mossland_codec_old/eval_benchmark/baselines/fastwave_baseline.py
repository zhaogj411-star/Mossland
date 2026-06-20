from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import cheby1, resample_poly, sosfiltfilt

try:
    from omegaconf import OmegaConf

    from ..audio_io import load_audio_segment, reference_path_for_item, save_audio
    from ..manifest import EvalItem, read_manifest, write_jsonl
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.fastwave_baseline ..."
    ) from exc


BASELINE_NAME = "fastwave"
TARGET_SAMPLE_RATE = 48000
DEFAULT_INPUT_SAMPLE_RATE = 16000
DEFAULT_STEPS = 8
DEFAULT_RHO = 8.0
DEFAULT_GUIDANCE = 1.6
DEFAULT_SIGMA_MIN = 0.002
DEFAULT_SIGMA_MAX = 80.0
OFFICIAL_REPO = "https://github.com/Nikait/FastWave"
OFFICIAL_CHECKPOINT_ID = "1oNCxrKjgiWsYGW6P49rsI84vFYR5G3m8"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return repo_root() / "tmp/eval_baseline_refs/fastwave/FastWave"


def default_checkpoint() -> Path:
    return repo_root() / "checkpoints/fastwave/checkpoint-epoch140.pth"


def ensure_repo(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    required = [
        repo_path / "src/model/baseline_model.py",
        repo_path / "src/trainer/inferencer.py",
        repo_path / "src/configs/inference.yaml",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"FastWave official repo is incomplete at {repo_path}. Missing: "
            f"{', '.join(str(path) for path in missing)}. Clone {OFFICIAL_REPO} there or pass --repo-dir."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    return repo_path


def drop_stale_src_modules(repo_dir: Path) -> None:
    repo_resolved = repo_dir.resolve()
    for name in list(sys.modules):
        if name != "src" and not name.startswith("src."):
            continue
        module_file = getattr(sys.modules[name], "__file__", None)
        if module_file is None:
            sys.modules.pop(name, None)
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            sys.modules.pop(name, None)
            continue
        if repo_resolved not in module_path.parents:
            sys.modules.pop(name, None)


def load_fastwave(args: argparse.Namespace) -> torch.nn.Module:
    repo_dir = ensure_repo(args.repo_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"FastWave checkpoint not found: {checkpoint}. Download the official Google Drive file "
            f"{OFFICIAL_CHECKPOINT_ID} there or pass --checkpoint."
        )

    drop_stale_src_modules(repo_dir)
    model_module = importlib.import_module("src.model")

    package = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = package.get("config") if isinstance(package, dict) else None
    if config is None or "model" not in config:
        config = OmegaConf.load(repo_dir / "src/configs/inference.yaml")
    model_config = config.model
    model = model_module.EDMPrecond(model_config.hparams)
    state = package["state_dict"] if isinstance(package, dict) and "state_dict" in package else package
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FastWave checkpoint did not match the official model. "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(args.device)
    model.eval()
    return model


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return F.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def align_to_hop(length: int, hop_length: int) -> int:
    aligned = (length // hop_length) * hop_length
    return max(hop_length, aligned)


def low_quality_condition(audio_48k: torch.Tensor, input_sample_rate: int) -> torch.Tensor:
    wav = audio_48k.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if input_sample_rate >= TARGET_SAMPLE_RATE:
        return torch.from_numpy(wav.copy()).float().unsqueeze(0)

    highcut = input_sample_rate // 2
    nyquist = 0.5 * TARGET_SAMPLE_RATE
    hi = min(max(highcut / nyquist, 0.0), 1.0)
    sos = cheby1(8, 0.05, hi, btype="lowpass", output="sos")
    low = sosfiltfilt(sos, wav).astype(np.float32)
    low = resample_poly(low, input_sample_rate, TARGET_SAMPLE_RATE).astype(np.float32)
    low = resample_poly(low, TARGET_SAMPLE_RATE, input_sample_rate).astype(np.float32)
    if low.shape[0] < wav.shape[0]:
        low = np.pad(low, (0, wav.shape[0] - low.shape[0]))
    elif low.shape[0] > wav.shape[0]:
        low = low[: wav.shape[0]]
    return torch.from_numpy(low.copy()).float().unsqueeze(0).clamp(-1.0, 1.0)


def band_mask(input_sample_rate: int, device: str | torch.device) -> torch.Tensor:
    fft_size = 1024 // 2 + 1
    cutoff_ratio = min(max(input_sample_rate / TARGET_SAMPLE_RATE, 0.0), 1.0)
    cutoff_bin = max(1, min(fft_size, int(round(cutoff_ratio * fft_size))))
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:cutoff_bin] = 1
    return band.unsqueeze(0).to(device)


def low_resolution_input(item: EvalItem, input_sample_rate: int) -> tuple[torch.Tensor, int]:
    reference_path = reference_path_for_item(item) or item.source_path
    target = load_audio_segment(
        reference_path,
        sample_rate=TARGET_SAMPLE_RATE,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if target.shape[0] > 1:
        target = target.mean(dim=0, keepdim=True)
    peak = target.abs().max()
    if peak > 0:
        target = target / peak
    return low_quality_condition(target.clamp(-1.0, 1.0), input_sample_rate), target.shape[-1]


def edm_sampler(
    net: torch.nn.Module,
    x_init: torch.Tensor,
    wav_l: torch.Tensor,
    band: torch.Tensor,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    guidance: float,
    randn_like: Any = torch.randn_like,
) -> torch.Tensor:
    dtype = x_init.dtype
    device = x_init.device

    def denoise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        denoised = net(x, sigma, wav_l, band).to(dtype)
        return denoised if guidance == 1.0 else denoised

    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_next = x_init.to(dtype) * t_steps[0]

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next
        x_hat = x_cur
        sigma_hat = t_cur * torch.ones(x_hat.shape[0], device=device, dtype=dtype)
        d_cur = (x_hat - denoise(x_hat, sigma_hat)) / sigma_hat[:, None]
        x_next = x_hat + (t_next - t_cur) * d_cur

    return x_next


def run_chunk(
    model: torch.nn.Module,
    chunk: torch.Tensor,
    band: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    wav_l = chunk.to(args.device)
    noise = torch.randn_like(wav_l)
    prediction = edm_sampler(
        net=model,
        x_init=noise,
        wav_l=wav_l,
        band=band,
        num_steps=args.steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        guidance=args.guidance,
    )
    return prediction.detach().cpu().float()


def run_fastwave_item(
    model: torch.nn.Module,
    item: EvalItem,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    input_sample_rate = item.low_sample_rate or args.input_sample_rate
    if input_sample_rate >= TARGET_SAMPLE_RATE:
        raise RuntimeError(
            f"item_id={item.item_id!r} has input_sample_rate={input_sample_rate}, "
            f"which must be below {TARGET_SAMPLE_RATE} for FastWave."
        )

    low, target_length = low_resolution_input(item, input_sample_rate)
    chunk_size = align_to_hop(int(round(args.chunk_seconds * TARGET_SAMPLE_RATE)), args.hop_length)
    band = band_mask(input_sample_rate, args.device)

    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    outputs = []
    try:
        torch.manual_seed(item.seed)
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(item.seed)
        with torch.inference_mode():
            for start in range(0, low.shape[-1], chunk_size):
                chunk = low[:, start : start + chunk_size]
                if chunk.shape[-1] < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - chunk.shape[-1]))
                outputs.append(run_chunk(model, chunk, band, args)[..., : chunk.shape[-1]])
    finally:
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    output = fit_length(torch.cat(outputs, dim=-1), target_length).clamp(-1.0, 1.0)
    save_audio(output_path, output[:1], TARGET_SAMPLE_RATE)


def output_item(item: EvalItem, args: argparse.Namespace) -> EvalItem:
    input_sample_rate = item.low_sample_rate or args.input_sample_rate
    metadata = dict(item.metadata)
    metadata["fastwave_input_sample_rate"] = input_sample_rate
    metadata["fastwave_target_sample_rate"] = TARGET_SAMPLE_RATE
    metadata["fastwave_steps"] = args.steps
    metadata["fastwave_guidance"] = args.guidance
    metadata["fastwave_chunk_seconds"] = args.chunk_seconds
    return replace(item, low_sample_rate=input_sample_rate, sample_rate=TARGET_SAMPLE_RATE, metadata=metadata)


def write_predicted_manifest(items: list[EvalItem], prediction_paths: list[Path], output_manifest: str | Path | None) -> None:
    if output_manifest is None:
        return
    rows = []
    for item, prediction_path in zip(items, prediction_paths, strict=True):
        rows.append({**item.to_json(), "prediction_path": str(prediction_path.resolve())})
    write_jsonl(rows, output_manifest)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> tuple[list[EvalItem], list[Path]]:
    model = load_fastwave(args)
    output_items = [output_item(item, args) for item in items]
    prediction_paths: list[Path] = []
    total = len(output_items)
    for index, item in enumerate(output_items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_fastwave_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return output_items, prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official FastWave super-resolution baseline.")
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to the official FastWave repo.")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()), help="Path to the official FastWave checkpoint.")
    parser.add_argument("--input-sample-rate", type=int, default=DEFAULT_INPUT_SAMPLE_RATE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="EDM sampler function evaluations.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_SIGMA_MAX)
    parser.add_argument("--chunk-seconds", type=float, default=32768 / TARGET_SAMPLE_RATE)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for FastWave inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = super_resolution_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return
    output_items, prediction_paths = run_items(items, args)
    write_predicted_manifest(output_items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
