from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

from ..manifest import EvalItem, read_manifest
from .common import add_common_args, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "stable_audio3_same"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REF_DIR = REPO_ROOT / "tmp" / "eval_baseline_refs" / "stable-audio-3"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "stable-audio-3"


def configure_hf_cache(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(checkpoint_dir / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(checkpoint_dir / "hub"))
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")


def import_autoencoder(ref_dir: Path):
    if not ref_dir.exists():
        raise FileNotFoundError(f"Stable Audio 3 repo not found: {ref_dir}")
    sys.path.insert(0, str(ref_dir))
    from stable_audio_3 import AutoencoderModel

    return AutoencoderModel


def stable_audio_items(manifest: str | Path, max_items: int, num_shards: int, shard_id: int) -> list[EvalItem]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    items = [item for index, item in enumerate(items) if num_shards == 1 or index % num_shards == shard_id]
    if max_items > 0:
        return items[:max_items]
    return items


def prediction_path(output_dir: str | Path, variant: str, item: EvalItem) -> Path:
    return Path(output_dir) / f"{BASELINE_NAME}_{variant}" / item.task_id / f"{item.item_id}_seed{item.seed}.wav"


def reconstruct_item(model, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    waveform, sr = torchaudio.load(str(item.source_path))
    if item.start_seconds or item.duration_seconds is not None:
        start = int(round(item.start_seconds * sr))
        end = waveform.shape[-1] if item.duration_seconds is None else start + int(round(item.duration_seconds * sr))
        waveform = waveform[..., start:end]
    with torch.inference_mode():
        latents = model.encode(
            waveform,
            sr,
            chunked=args.chunked,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        reconstructed = model.decode(
            latents,
            chunked=args.chunked,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    reconstructed = reconstructed[0].detach().cpu().float().clamp(-1.0, 1.0)
    target_len = int(round(waveform.shape[-1] * model.sample_rate / sr))
    reconstructed = reconstructed[..., :target_len]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), reconstructed, model.sample_rate)


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    configure_hf_cache(Path(args.checkpoint_dir))
    AutoencoderModel = import_autoencoder(Path(args.ref_dir))
    model = AutoencoderModel.from_pretrained(args.variant, device=args.device)
    prediction_paths = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = prediction_path(args.output_dir, args.variant, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        reconstruct_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stable Audio 3 SAME autoencoder reconstruction baseline.")
    add_common_args(parser)
    parser.add_argument("--variant", choices=("same-s", "same-l"), default="same-s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--chunked", action="store_true", default=True)
    parser.add_argument("--no-chunked", action="store_false", dest="chunked")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = stable_audio_items(args.manifest, args.max_items, args.num_shards, args.shard_id)
    prediction_paths = reconstruct_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
