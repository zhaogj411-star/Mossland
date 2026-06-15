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
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.wavtokenizer_baseline ..."
    ) from exc


BASELINE_NAME = "wavtokenizer"
OFFICIAL_REPO_URL = "https://github.com/jishengpeng/WavTokenizer.git"
DEFAULT_HF_REPO = "novateur/WavTokenizer-medium-music-audio-75token"
DEFAULT_CONFIG_FILENAME = "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
DEFAULT_CHECKPOINT_FILENAME = "wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
WAVTOKENIZER_SAMPLE_RATE = 24000
WAVTOKENIZER_CHANNELS = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/wavtokenizer/WavTokenizer"


def _default_cache_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/wavtokenizer/checkpoints"


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
            f"WavTokenizer official repository is not available at {repo_path}, and automatic "
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


def _download_hf_file(repo_id: str, filename: str, cache_dir: str | Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "huggingface_hub is required to download the default WavTokenizer checkpoint. "
            "Install it or pass both --model-config and --checkpoint as local files."
        ) from exc

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir)))
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download {filename!r} from Hugging Face repo {repo_id!r}. "
            "Retry with working Hugging Face connectivity, for example "
            "`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY HF_ENDPOINT=https://huggingface.co ...`, "
            "or pass local --model-config and --checkpoint paths."
        ) from exc


def resolve_model_files(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = Path(args.model_config) if args.model_config else None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if config_path is None:
        config_path = _download_hf_file(args.hf_repo, args.hf_config_filename, args.hf_cache_dir)
    if checkpoint_path is None:
        checkpoint_path = _download_hf_file(args.hf_repo, args.hf_checkpoint_filename, args.hf_cache_dir)

    missing = [str(path) for path in (config_path, checkpoint_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            "WavTokenizer model files are missing: "
            + ", ".join(missing)
            + ". Pass local --model-config/--checkpoint or allow HF download."
        )
    return config_path, checkpoint_path


def load_wavtokenizer(args: argparse.Namespace) -> Any:
    _add_repo_dir(args.repo_dir)
    try:
        from decoder.pretrained import WavTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "WavTokenizer is not importable from --repo-dir. This adapter expects the official "
            f"repository from {OFFICIAL_REPO_URL}. Its inference dependencies include encodec, "
            "pyyaml, einops, transformers, pytorch-lightning, and other packages listed in "
            "`requirements.txt`; install them in an isolated environment if import fails."
        ) from exc

    config_path, checkpoint_path = resolve_model_files(args)
    try:
        model = WavTokenizer.from_pretrained0802(str(config_path), str(checkpoint_path))
    except Exception as exc:  # pragma: no cover - checkpoint/dependency dependent
        raise RuntimeError(
            "Failed to initialize official WavTokenizer with "
            f"config={config_path} checkpoint={checkpoint_path}. Ensure the checkpoint matches "
            "the official 0802 YAML format and that the official repo dependencies are installed."
        ) from exc
    return model.to(args.device).eval()


def reconstruct_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    if max_items > 0:
        return items[:max_items]
    return items


def prepare_model_audio(audio: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
    if audio.ndim != 2:
        raise RuntimeError(f"Expected source audio [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] != WAVTOKENIZER_CHANNELS:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sample_rate != WAVTOKENIZER_SAMPLE_RATE:
        audio = AF.resample(audio, source_sample_rate, WAVTOKENIZER_SAMPLE_RATE)
    return audio.clamp(-1.0, 1.0)


def _to_audio_tensor(decoded: Any) -> torch.Tensor:
    audio = torch.as_tensor(decoded).detach().cpu().float()
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise RuntimeError(f"WavTokenizer returned batched audio with shape {tuple(audio.shape)}.")
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise RuntimeError(f"WavTokenizer returned audio with unexpected shape {tuple(audio.shape)}.")
    return audio


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
    model_audio = prepare_model_audio(audio, item.sample_rate)
    device_audio = model_audio.to(args.device)
    bandwidth_id = torch.tensor([args.bandwidth_id], device=args.device)

    with torch.inference_mode():
        features, _discrete_code = model.encode_infer(device_audio, bandwidth_id=bandwidth_id)
        decoded = model.decode(features, bandwidth_id=bandwidth_id)

    prediction = _to_audio_tensor(decoded)
    prediction = _fit_length(prediction, model_audio.shape[-1])
    if item.sample_rate != WAVTOKENIZER_SAMPLE_RATE:
        prediction = AF.resample(prediction, WAVTOKENIZER_SAMPLE_RATE, item.sample_rate)
    prediction = _fit_length(prediction, audio.shape[-1])
    save_audio(output_path, prediction, item.sample_rate)


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_wavtokenizer(args)
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
        description="Run the official WavTokenizer reconstruction baseline for Mossland eval manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default=str(_default_repo_dir()),
        help="Path to the official WavTokenizer repository; cloned to this path when missing.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for WavTokenizer inference.",
    )
    parser.add_argument(
        "--model-config",
        default="",
        help="Optional local WavTokenizer YAML config. Defaults to Hugging Face download.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional local WavTokenizer checkpoint. Defaults to Hugging Face download.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo used when --model-config or --checkpoint is omitted.",
    )
    parser.add_argument(
        "--hf-config-filename",
        default=DEFAULT_CONFIG_FILENAME,
        help="Config filename inside --hf-repo.",
    )
    parser.add_argument(
        "--hf-checkpoint-filename",
        default=DEFAULT_CHECKPOINT_FILENAME,
        help="Checkpoint filename inside --hf-repo.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=str(_default_cache_dir()),
        help="Local Hugging Face cache directory for WavTokenizer model files.",
    )
    parser.add_argument(
        "--bandwidth-id",
        type=int,
        default=0,
        help="bandwidth_id passed to official encode_infer/decode. Official examples use 0.",
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
