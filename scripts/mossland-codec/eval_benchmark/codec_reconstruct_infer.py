from __future__ import annotations

import argparse
import importlib
from dataclasses import replace
from pathlib import Path

import torch
import torchaudio

from .audio_io import load_audio_segment, save_audio
from .fulltrack_infer import select_items
from .infer import build_task_input, prediction_path_for_item
from .manifest import EvalItem, read_manifest, write_jsonl


inference_module = importlib.import_module("scripts.mossland-codec.inference")

CODEC_PARALLEL_TASKS = {"reconstruct", "super_resolution"}


def _source_path_for_item(item: EvalItem) -> Path:
    if item.source_path is not None:
        return item.source_path
    if item.mixture_path is not None:
        return item.mixture_path
    raise RuntimeError(f"item_id={item.item_id!r} has no source_path or mixture_path.")


def load_encoder_decoder(
    checkpoint_dir: str | Path,
    device: str | torch.device | None = None,
    *,
    max_batch_size_encode: int | None = None,
    max_batch_size_decode: int | None = None,
):
    return inference_module.EncoderDecoder(
        load_path_inference=str(checkpoint_dir),
        device=device,
        max_batch_size_encode=max_batch_size_encode,
        max_batch_size_decode=max_batch_size_decode,
    )


@torch.inference_mode()
def generate_codec_reconstruction(
    codec,
    item: EvalItem,
    output_dir: str | Path,
    *,
    desired_channels: int = 64,
    overwrite: bool = False,
    preprocess_on_gpu: bool = False,
) -> tuple[Path, dict[str, int | float | str]]:
    if item.task_id not in CODEC_PARALLEL_TASKS:
        raise ValueError(
            "codec parallel inference only supports "
            f"{sorted(CODEC_PARALLEL_TASKS)}, got {item.task_id!r}"
        )

    output_path = prediction_path_for_item(item, output_dir)
    if item.task_id == "reconstruct":
        source_path = _source_path_for_item(item)
        source = load_audio_segment(
            source_path,
            sample_rate=item.sample_rate,
            start_seconds=item.start_seconds,
            duration_seconds=item.duration_seconds,
        )
    else:
        source, _ = build_task_input(item)
    source_frames = int(source.shape[-1])
    metadata: dict[str, int | float | str] = {
        "inference_mode": f"codec_parallel_{item.task_id}",
        "source_frames": source_frames,
        "desired_channels": int(desired_channels),
        "max_batch_size_encode": int(codec.max_batch_size_encode),
        "max_batch_size_decode": int(codec.max_batch_size_decode),
    }

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        info = torchaudio.info(str(output_path))
        metadata["prediction_frames"] = int(info.num_frames)
        metadata["prediction_source_length_ratio"] = float(info.num_frames) / max(source_frames, 1)
        return output_path.resolve(), metadata

    torch.manual_seed(int(item.seed))
    latents = codec.encode(
        source,
        desired_channels=desired_channels,
        preprocess_on_gpu=preprocess_on_gpu,
    )
    prediction = codec.decode(
        latents,
        mode="parallel",
        preprocess_on_gpu=preprocess_on_gpu,
        task_id=item.task_id,
    )
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.as_tensor(prediction)
    prediction = prediction.detach().cpu().float()
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    prediction = prediction[..., :source_frames]
    save_audio(output_path, prediction, item.sample_rate)

    info = torchaudio.info(str(output_path))
    metadata["prediction_frames"] = int(info.num_frames)
    metadata["prediction_source_length_ratio"] = float(info.num_frames) / max(source_frames, 1)
    return output_path.resolve(), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mossland reconstruction/SR through EncoderDecoder encode/decode parallel inference."
    )
    parser.add_argument("--manifest", required=True, help="Evaluation manifest JSONL/JSON/CSV.")
    parser.add_argument("--checkpoint-dir", required=True, help="Mossland checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction wav files.")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Prediction manifest path. Defaults to <output-dir>/prediction_manifest.jsonl.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--desired-channels", type=int, default=64)
    parser.add_argument("--max-batch-size-encode", type=int, default=None)
    parser.add_argument("--max-batch-size-decode", type=int, default=None)
    parser.add_argument("--preprocess-on-gpu", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Number of manifest shards for this inference run.")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard id for --num-shards.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_items = read_manifest(args.manifest)
    items = select_items(
        all_items,
        max_items=args.max_items,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    output_manifest = Path(args.output_manifest) if args.output_manifest else Path(args.output_dir) / "prediction_manifest.jsonl"

    codec = load_encoder_decoder(
        args.checkpoint_dir,
        device=args.device,
        max_batch_size_encode=args.max_batch_size_encode,
        max_batch_size_decode=args.max_batch_size_decode,
    )
    rows = []
    if args.num_shards > 1 or args.max_items > 0:
        print(
            (
                f"selected {len(items)}/{len(all_items)} rows "
                f"(shard {args.shard_id}/{args.num_shards}, max_items={args.max_items})"
            ),
            flush=True,
        )
    for index, item in enumerate(items, start=1):
        prediction_path, extra_metadata = generate_codec_reconstruction(
            codec,
            item,
            args.output_dir,
            desired_channels=args.desired_channels,
            overwrite=args.overwrite,
            preprocess_on_gpu=args.preprocess_on_gpu,
        )
        metadata = dict(item.metadata)
        metadata.update(extra_metadata)
        rows.append(replace(item, prediction_path=prediction_path, metadata=metadata).to_json())
        if args.progress_every > 0 and (index == 1 or index == len(items) or index % args.progress_every == 0):
            print(f"processed={index}/{len(items)} last={item.item_id}", flush=True)
    write_jsonl(rows, output_manifest)


if __name__ == "__main__":
    main()
