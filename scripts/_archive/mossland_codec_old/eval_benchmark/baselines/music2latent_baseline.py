from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.music2latent_baseline ..."
    ) from exc


BASELINE_NAME = "music2latent"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "music2latent" / "music2latent.pt"


def load_music2latent(checkpoint: str | Path, device: str | torch.device):
    try:
        from music2latent import EncoderDecoder
    except ImportError as exc:
        raise RuntimeError(
            "The official `music2latent` package is not installed. Install SonyCSLParis/music2latent "
            "or use the project environment that already contains it."
        ) from exc

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Music2Latent checkpoint not found: {checkpoint_path}")
    return EncoderDecoder(load_path_inference=str(checkpoint_path), device=torch.device(device))


def reconstruction_items(manifest: str | Path, max_items: int, num_shards: int, shard_id: int) -> list[EvalItem]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    items = [item for index, item in enumerate(items) if num_shards == 1 or index % num_shards == shard_id]
    if max_items > 0:
        return items[:max_items]
    return items


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_music2latent(args.checkpoint, args.device)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue

        audio = load_audio_segment(
            item.source_path,
            sample_rate=args.model_sample_rate,
            start_seconds=item.start_seconds,
            duration_seconds=item.duration_seconds,
            stereo=True,
        )
        latent = model.encode(
            audio.numpy(),
            max_waveform_length=args.max_waveform_length_encode,
            max_batch_size=args.max_batch_size_encode,
        )
        decoded = model.decode(
            latent,
            denoising_steps=args.denoising_steps,
            max_waveform_length=args.max_waveform_length_decode,
            max_batch_size=args.max_batch_size_decode,
        )
        decoded = torch.as_tensor(decoded, dtype=torch.float32)
        if decoded.ndim != 2:
            raise RuntimeError(f"Expected Music2Latent decoded audio to be 2-D, got shape {tuple(decoded.shape)}")
        if decoded.shape[0] > decoded.shape[1]:
            decoded = decoded.transpose(0, 1)
        if decoded.shape[-1] < audio.shape[-1]:
            decoded = torch.nn.functional.pad(decoded, (0, audio.shape[-1] - decoded.shape[-1]))
        decoded = decoded[..., : audio.shape[-1]].clamp(-1.0, 1.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(output_path, decoded.cpu(), args.model_sample_rate)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official SonyCSLParis Music2Latent reconstruction baseline.")
    add_common_args(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-sample-rate", type=int, default=44100)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--max-waveform-length-encode", type=int, default=44100 * 60)
    parser.add_argument("--max-waveform-length-decode", type=int, default=44100 * 60)
    parser.add_argument("--max-batch-size-encode", type=int, default=1)
    parser.add_argument("--max-batch-size-decode", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = reconstruction_items(args.manifest, args.max_items, args.num_shards, args.shard_id)
    prediction_paths = reconstruct_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
