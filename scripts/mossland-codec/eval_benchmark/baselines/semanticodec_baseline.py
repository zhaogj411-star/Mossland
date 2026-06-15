from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.semanticodec_baseline ..."
    ) from exc


BASELINE_NAME = "semanticodec"
OFFICIAL_REPO_URL = "https://github.com/haoheliu/SemantiCodec-inference.git"
SEMANTICODEC_SAMPLE_RATE = 16000
SEMANTICODEC_CHANNELS = 1
VALID_TOKEN_RATES = (25, 50, 100)
VALID_SEMANTIC_VOCAB_SIZES = (4096, 8192, 16384, 32768)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/semanticodec/SemantiCodec-inference"


def _default_cache_dir() -> Path:
    return _repo_root() / "checkpoints/semanticodec"


def ensure_repo_dir(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    if repo_path.exists():
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(repo_path)],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"SemantiCodec official repository is not available at {repo_path}, and automatic "
            f"`git clone --depth 1 {OFFICIAL_REPO_URL} {repo_path}` failed. Clone it manually "
            "or pass --repo-dir pointing at an existing checkout."
        ) from exc
    return repo_path


def _add_repo_dir(repo_dir: str | Path) -> Path:
    repo_path = ensure_repo_dir(repo_dir).resolve()
    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    return repo_path


def _checkpoint_path(args: argparse.Namespace) -> str | None:
    if not args.checkpoint_root:
        return None

    root = Path(args.checkpoint_root)
    if (root / "encoder.ckpt").exists() or (root / "decoder.ckpt").exists():
        return str(root)
    return str(root / f"semanticodec_tokenrate_{args.token_rate}")


@contextlib.contextmanager
def _official_device_selection(device: str) -> Iterator[None]:
    """Force the official constructor onto CPU when requested.

    The official SemantiCodec class picks cuda/mps automatically and does not
    accept a device argument. For cuda devices we move modules after init; for
    CPU we hide accelerator availability during construction to avoid a large
    temporary allocation on cuda:0.
    """

    if torch.device(device).type != "cpu":
        yield
        return

    cuda_is_available = torch.cuda.is_available
    mps_is_available = getattr(torch.backends.mps, "is_available", None)
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    if mps_is_available is not None:
        torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.cuda.is_available = cuda_is_available  # type: ignore[assignment]
        if mps_is_available is not None:
            torch.backends.mps.is_available = mps_is_available  # type: ignore[method-assign]


def _move_semanticodec(model: Any, device: str) -> Any:
    torch_device = torch.device(device)
    model.device = torch_device
    if hasattr(model, "encoder"):
        model.encoder = model.encoder.to(torch_device)
    if hasattr(model, "decoder"):
        model.decoder = model.decoder.to(torch_device)
        if hasattr(model.decoder, "device"):
            model.decoder.device = torch_device
    return model.eval()


def load_semanticodec(args: argparse.Namespace) -> Any:
    _add_repo_dir(args.repo_dir)
    try:
        from semanticodec import SemantiCodec
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "SemantiCodec is not importable from --repo-dir. This adapter expects the official "
            f"repository from {OFFICIAL_REPO_URL}. Install its dependencies, typically "
            "`pip install -e <repo-dir>` or `pip install torch torchaudio soundfile "
            "vector-quantize-pytorch huggingface_hub timm scipy`."
        ) from exc

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with _official_device_selection(args.device):
            model = SemantiCodec(
                token_rate=args.token_rate,
                semantic_vocab_size=args.semantic_vocab_size,
                ddim_sample_step=args.ddim_sample_step,
                cfg_scale=args.cfg_scale,
                checkpoint_path=_checkpoint_path(args),
                cache_path=str(cache_dir),
            )
    except AssertionError as exc:
        raise RuntimeError(
            "Invalid SemantiCodec setting. Official token_rate must be one of "
            f"{VALID_TOKEN_RATES}; semantic_vocab_size must be one of {VALID_SEMANTIC_VOCAB_SIZES}."
        ) from exc
    except Exception as exc:  # pragma: no cover - checkpoint/dependency dependent
        raise RuntimeError(
            "Failed to initialize official SemantiCodec. Ensure --repo-dir points at "
            "haoheliu/SemantiCodec-inference, dependencies are installed, and Hugging Face "
            "checkpoint files can be downloaded from repo `haoheliu/SemantiCodec` into "
            f"{cache_dir}. To use local weights, pass --checkpoint-root containing "
            "`semanticodec_tokenrate_<rate>/encoder.ckpt`, `decoder.ckpt`, and `codebook_idx/...`."
        ) from exc
    return _move_semanticodec(model, args.device)


def reconstruct_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    if max_items > 0:
        return items[:max_items]
    return items


def prepare_model_audio(audio: torch.Tensor, source_sample_rate: int, model_sample_rate: int) -> torch.Tensor:
    if audio.ndim != 2:
        raise RuntimeError(f"Expected source audio [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] != SEMANTICODEC_CHANNELS:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sample_rate != model_sample_rate:
        audio = AF.resample(audio, source_sample_rate, model_sample_rate)
    return audio.clamp(-1.0, 1.0)


def _to_audio_tensor(decoded: Any) -> torch.Tensor:
    audio = torch.as_tensor(decoded).detach().cpu().float()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise RuntimeError(f"SemantiCodec returned audio with unexpected shape {tuple(audio.shape)}.")
    if audio.shape[0] != SEMANTICODEC_CHANNELS:
        audio = audio[:1]
    return audio.contiguous()


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def reconstruct_item(model: Any, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    model_audio = prepare_model_audio(audio, item.sample_rate, args.sample_rate)

    temp_parent = Path(args.output_dir).resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semanticodec_", dir=str(temp_parent)) as tmp_dir:
        input_path = Path(tmp_dir) / "input.wav"
        save_audio(input_path, model_audio, args.sample_rate)
        with torch.inference_mode():
            tokens = model.encode(str(input_path))
            decoded = model.decode(tokens)

    prediction = _to_audio_tensor(decoded)
    prediction = _fit_length(prediction, model_audio.shape[-1])
    if item.sample_rate != args.sample_rate:
        prediction = AF.resample(prediction, args.sample_rate, item.sample_rate)
    prediction = _fit_length(prediction, audio.shape[-1])
    save_audio(output_path, prediction, item.sample_rate)


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_semanticodec(args)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        reconstruct_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SemantiCodec reconstruction baseline for Mossland eval manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default=str(_default_repo_dir()),
        help="Path to the official SemantiCodec-inference repository; cloned to this path when missing.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for SemantiCodec inference. Official code auto-selects during init; modules are moved after load.",
    )
    parser.add_argument(
        "--token-rate",
        type=int,
        default=100,
        choices=VALID_TOKEN_RATES,
        help="Official SemantiCodec token rate in tokens/second.",
    )
    parser.add_argument(
        "--semantic-vocab-size",
        type=int,
        default=16384,
        choices=VALID_SEMANTIC_VOCAB_SIZES,
        help="Official SemantiCodec semantic vocabulary size.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SEMANTICODEC_SAMPLE_RATE,
        help="Model I/O sample rate. Official SemantiCodec protocol is 16000 Hz mono.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_default_cache_dir()),
        help="Hugging Face cache/checkpoint directory for official SemantiCodec files.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help=(
            "Optional local root for official checkpoint files. Pass either the folder containing "
            "encoder.ckpt/decoder.ckpt or a parent containing semanticodec_tokenrate_<rate>/."
        ),
    )
    parser.add_argument(
        "--ddim-sample-step",
        type=int,
        default=50,
        help="ddim_sample_step passed to official SemantiCodec.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=2.0,
        help="cfg_scale passed to official SemantiCodec.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = reconstruct_manifest_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return

    prediction_paths = reconstruct_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
