from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio.functional as AF

try:
    from omegaconf import OmegaConf as OC

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
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.nuwave2_baseline ..."
    ) from exc


BASELINE_NAME = "nuwave2"
MODEL_SAMPLE_RATE = 48000
DEFAULT_STEPS = 8
OFFICIAL_NUWAVE2_REPO = "https://github.com/mindslab-ai/nuwave2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/nuwave2"


def default_checkpoint() -> Path:
    return default_repo_dir() / "checkpoints/nuwave2_48k_official.ckpt"


def install_lightning_stub() -> None:
    module = types.ModuleType("pytorch_lightning")
    module.__path__ = []  # type: ignore[attr-defined]

    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, hparams: Any = None, *args: Any, **kwargs: Any) -> None:
            self.hparams = hparams

        def log(self, *args: Any, **kwargs: Any) -> None:
            return None

    class Callback:
        pass

    class LightningDataModule:
        pass

    def dynamic_class(name: str) -> type:
        return type(name, (Callback,), {})

    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.__path__ = []  # type: ignore[attr-defined]
    callbacks.Callback = Callback
    callbacks.__getattr__ = dynamic_class  # type: ignore[attr-defined]
    model_checkpoint = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")
    model_checkpoint.ModelCheckpoint = dynamic_class("ModelCheckpoint")
    early_stopping = types.ModuleType("pytorch_lightning.callbacks.early_stopping")
    early_stopping.EarlyStopping = dynamic_class("EarlyStopping")
    lr_monitor = types.ModuleType("pytorch_lightning.callbacks.lr_monitor")
    lr_monitor.LearningRateMonitor = dynamic_class("LearningRateMonitor")
    progress = types.ModuleType("pytorch_lightning.callbacks.progress")
    progress.ProgressBar = dynamic_class("ProgressBar")
    progress.TQDMProgressBar = dynamic_class("TQDMProgressBar")

    module.LightningModule = LightningModule
    module.LightningDataModule = LightningDataModule
    module.Callback = Callback
    module.callbacks = callbacks
    module.__getattr__ = dynamic_class  # type: ignore[attr-defined]
    sys.modules["pytorch_lightning"] = module
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = model_checkpoint
    sys.modules["pytorch_lightning.callbacks.early_stopping"] = early_stopping
    sys.modules["pytorch_lightning.callbacks.lr_monitor"] = lr_monitor
    sys.modules["pytorch_lightning.callbacks.progress"] = progress


def add_official_repo_to_path(repo_dir: str | Path) -> None:
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise FileNotFoundError(
            f"NU-Wave2 official repo not found at {repo_path}. Clone {OFFICIAL_NUWAVE2_REPO} there "
            "or pass --repo-dir."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _drop_stale_official_modules(repo_dir: Path) -> None:
    repo_resolved = repo_dir.resolve()
    for name in ("lightning_model", "diffusion", "model", "dataloader"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module is not None and (module_file is None or repo_resolved not in Path(module_file).resolve().parents):
            sys.modules.pop(name, None)


def torch_load_weights_false(path: str | Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


@contextlib.contextmanager
def istft_complex_compat() -> Iterator[None]:
    original_istft = torch.istft

    def compat_istft(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not torch.is_complex(input_tensor) and input_tensor.shape[-1] == 2:
            input_tensor = torch.view_as_complex(input_tensor.contiguous())
        return original_istft(input_tensor, *args, **kwargs)

    torch.istft = compat_istft  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.istft = original_istft  # type: ignore[assignment]


def load_nuwave2(args: argparse.Namespace) -> tuple[torch.nn.Module, Any]:
    repo_dir = Path(args.repo_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"NU-Wave2 checkpoint not found: {checkpoint}")

    install_lightning_stub()
    add_official_repo_to_path(repo_dir)
    _drop_stale_official_modules(repo_dir)
    try:
        lightning_model = importlib.import_module("lightning_model")
    except Exception as exc:  # pragma: no cover - optional external source path
        raise RuntimeError(
            "Failed to import the official NU-Wave2 source. This adapter expects the official "
            f"repository from {OFFICIAL_NUWAVE2_REPO} at --repo-dir."
        ) from exc

    hparams = OC.load(repo_dir / "hparameter.yaml")
    model = lightning_model.NuWave2(hparams).to(args.device)
    state = torch_load_weights_false(checkpoint)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def low_bandwidth_audio_48k(item: EvalItem) -> torch.Tensor:
    low_sample_rate = item.low_sample_rate or 16000
    if low_sample_rate >= MODEL_SAMPLE_RATE:
        raise RuntimeError(
            f"item_id={item.item_id!r} has low_sample_rate={low_sample_rate}, "
            f"which must be below {MODEL_SAMPLE_RATE} for NU-Wave2."
        )

    audio = load_audio_segment(
        item.source_path,
        sample_rate=MODEL_SAMPLE_RATE,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak
    audio = AF.resample(audio, MODEL_SAMPLE_RATE, low_sample_rate)
    audio = AF.resample(audio, low_sample_rate, MODEL_SAMPLE_RATE)
    return trim_to_hop(audio.clamp(-1.0, 1.0), hop_length=256)


def trim_to_hop(audio: torch.Tensor, hop_length: int) -> torch.Tensor:
    length = audio.shape[-1] - (audio.shape[-1] % hop_length)
    if length <= 0:
        raise RuntimeError("NU-Wave2 input is shorter than one hop after trimming.")
    return audio[..., :length]


def band_mask(low_sample_rate: int, hparams: Any, device: str | torch.device) -> torch.Tensor:
    fft_size = int(hparams.audio.filter_length) // 2 + 1
    highcut = low_sample_rate // 2
    nyquist = 0.5 * int(hparams.audio.sampling_rate)
    hi = max(0.0, min(1.0, highcut / nyquist))
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[: int(hi * fft_size)] = 1
    return band.unsqueeze(0).to(device)


def run_nuwave2_item(
    model: torch.nn.Module,
    hparams: Any,
    item: EvalItem,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    low_sr = item.low_sample_rate or 16000
    wav_l = low_bandwidth_audio_48k(item).to(args.device)
    band = band_mask(low_sr, hparams, args.device)
    noise_schedule = eval(hparams.dpm.infer_schedule, {"torch": torch}) if args.steps == DEFAULT_STEPS else None
    if noise_schedule is not None:
        noise_schedule = noise_schedule.to(args.device)

    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(item.seed)
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(item.seed)
        with torch.inference_mode(), istft_complex_compat():
            prediction, _ = model.inference(wav_l, band, args.steps, noise_schedule)
    finally:
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    prediction = prediction.detach().cpu().float()
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.shape[0] > 1:
        prediction = prediction[:1]
    save_audio(output_path, prediction, MODEL_SAMPLE_RATE)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model, hparams = load_nuwave2(args)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_nuwave2_item(model, hparams, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official NU-Wave2 48 kHz super-resolution baseline for Mossland manifests."
    )
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to the official NU-Wave2 repo.")
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint()),
        help="Official NU-Wave2 48 kHz checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for NU-Wave2 inference.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=argparse.SUPPRESS)
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
