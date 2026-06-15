from __future__ import annotations

import argparse
import importlib
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from .audio_io import load_audio_segment, save_audio
from .infer import load_mossland_model, prediction_path_for_item
from .manifest import EvalItem, read_manifest, write_jsonl


CHUNKED_TASKS = {
    "reconstruct",
    "separate_vocals",
    "separate_drums",
    "separate_bass",
    "separate_other",
    "separate_accompaniment",
}


models_module = importlib.import_module("scripts.mossland-codec.models")


def _source_path_for_item(item: EvalItem) -> Path:
    if item.mixture_path is not None:
        return item.mixture_path
    if item.source_path is not None:
        return item.source_path
    raise RuntimeError(f"item_id={item.item_id!r} has no mixture_path or source_path.")


def _chunk_frames_for_model(model) -> int:
    return int(
        models_module.waveform_length_for_stft_frames(
            2 * int(model.spec_length),
            hop=int(model.hop),
            fac=int(model.fac),
        )
    )


def _chunk_ranges(total_frames: int, chunk_frames: int, step_frames: int) -> list[tuple[int, int]]:
    if total_frames <= 0:
        return [(0, chunk_frames)]
    starts = list(range(0, total_frames, step_frames))
    if starts[-1] + chunk_frames < total_frames:
        starts.append(max(total_frames - chunk_frames, 0))
    ranges: list[tuple[int, int]] = []
    previous_start = -1
    for start in starts:
        start = min(start, max(total_frames - 1, 0))
        if start == previous_start:
            continue
        ranges.append((start, min(start + chunk_frames, total_frames)))
        previous_start = start
        if start + chunk_frames >= total_frames:
            break
    return ranges


def select_items(
    items: list[EvalItem],
    *,
    max_items: int = 0,
    num_shards: int = 1,
    shard_id: int = 0,
) -> list[EvalItem]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    selected = [
        item
        for index, item in enumerate(items)
        if num_shards == 1 or index % num_shards == shard_id
    ]
    if max_items > 0:
        selected = selected[:max_items]
    return selected


def _crop_generated_chunk(
    generated: torch.Tensor,
    index: int,
    chunk_count: int,
    overlap_frames: int,
) -> torch.Tensor:
    if overlap_frames <= 0 or chunk_count == 1:
        return generated
    left_crop = 0 if index == 0 else overlap_frames // 2
    right_crop = 0 if index == chunk_count - 1 else overlap_frames - left_crop
    end = generated.shape[-1] - right_crop if right_crop > 0 else generated.shape[-1]
    return generated[..., left_crop:end]


def _chunk_crossfade_weights(
    length: int,
    *,
    index: int,
    chunk_count: int,
    overlap_frames: int,
) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32)
    if overlap_frames <= 0 or chunk_count == 1 or length <= 0:
        return weights

    if index > 0:
        fade_frames = min(overlap_frames, length)
        weights[:fade_frames] *= torch.linspace(0.0, 1.0, fade_frames, dtype=torch.float32)
    if index < chunk_count - 1:
        fade_frames = min(overlap_frames, length)
        weights[-fade_frames:] *= torch.linspace(1.0, 0.0, fade_frames, dtype=torch.float32)
    return weights


@torch.inference_mode()
def generate_chunked_prediction(
    model,
    item: EvalItem,
    output_dir: str | Path,
    chunk_overlap_seconds: float,
    chunk_batch_size: int = 1,
    dont_quantize: bool = True,
    overwrite: bool = False,
) -> tuple[Path, dict[str, int | float | str]]:
    if item.task_id not in CHUNKED_TASKS:
        raise ValueError(f"chunked inference does not support task_id={item.task_id!r}")
    if item.task_id != "reconstruct" and item.duration_seconds is not None:
        raise ValueError(
            f"item_id={item.item_id!r} has duration_seconds={item.duration_seconds}; "
            "use regular eval_benchmark.run for windowed items."
        )

    output_path = prediction_path_for_item(item, output_dir)
    source_path = _source_path_for_item(item)
    source = load_audio_segment(
        source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds if item.task_id == "reconstruct" else None,
    )
    source_frames = int(source.shape[-1])
    chunk_frames = _chunk_frames_for_model(model)
    overlap_frames = int(round(float(chunk_overlap_seconds) * item.sample_rate))
    if overlap_frames < 0:
        raise ValueError("--chunk-overlap-seconds must be non-negative.")
    if overlap_frames >= chunk_frames:
        raise ValueError(
            f"chunk overlap ({overlap_frames} frames) must be smaller than chunk length ({chunk_frames})."
        )
    chunk_batch_size = max(1, int(chunk_batch_size))
    step_frames = chunk_frames - overlap_frames
    chunk_ranges = _chunk_ranges(source_frames, chunk_frames, step_frames)

    metadata: dict[str, int | float | str] = {
        "inference_mode": "chunked_reconstruct" if item.task_id == "reconstruct" else "chunked_fulltrack",
        "chunk_frames": chunk_frames,
        "chunk_seconds": chunk_frames / float(item.sample_rate),
        "overlap_frames": overlap_frames,
        "overlap_seconds": overlap_frames / float(item.sample_rate),
        "step_frames": step_frames,
        "source_frames": source_frames,
        "chunk_count": len(chunk_ranges),
        "chunk_batch_size": chunk_batch_size,
    }

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        info = torchaudio.info(str(output_path))
        metadata["prediction_frames"] = int(info.num_frames)
        metadata["prediction_source_length_ratio"] = float(info.num_frames) / max(source_frames, 1)
        return output_path.resolve(), metadata

    torch.manual_seed(int(item.seed))
    device = next(model.parameters()).device
    prediction_sum = torch.zeros(source.shape[0], source_frames, dtype=torch.float32)
    weight_sum = torch.zeros(1, source_frames, dtype=torch.float32)
    for batch_start in range(0, len(chunk_ranges), chunk_batch_size):
        batch_ranges = chunk_ranges[batch_start : batch_start + chunk_batch_size]
        chunks = []
        for start, end in batch_ranges:
            chunk = source[..., start:end]
            valid_frames = int(chunk.shape[-1])
            if valid_frames < chunk_frames:
                chunk = F.pad(chunk, (0, chunk_frames - valid_frames))
            chunks.append(chunk)
        chunk_batch = torch.stack(chunks, dim=0).to(device)
        _, generated_batch = model.generate_waveform(
            chunk_batch,
            task_id=item.task_id,
            dont_quantize=dont_quantize,
        )
        generated_batch = generated_batch.detach().cpu().float()
        for offset, (start, _) in enumerate(batch_ranges):
            index = batch_start + offset
            generated = generated_batch[offset]
            output_end = min(start + generated.shape[-1], source_frames)
            output_frames = max(output_end - start, 0)
            if output_frames <= 0:
                continue
            generated = generated[..., :output_frames]
            weights = _chunk_crossfade_weights(
                output_frames,
                index=index,
                chunk_count=len(chunk_ranges),
                overlap_frames=overlap_frames,
            )
            prediction_sum[..., start:output_end] += generated * weights.unsqueeze(0)
            weight_sum[..., start:output_end] += weights.unsqueeze(0)

    prediction = prediction_sum / weight_sum.clamp_min(1e-8)
    save_audio(output_path, prediction, item.sample_rate)

    info = torchaudio.info(str(output_path))
    metadata["prediction_frames"] = int(info.num_frames)
    metadata["prediction_source_length_ratio"] = float(info.num_frames) / max(source_frames, 1)
    return output_path.resolve(), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mossland long-form inference by chunking around the fixed codec window."
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
    parser.add_argument("--chunk-overlap-seconds", type=float, default=0.5)
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=1,
        help="Number of fixed-length chunks from one long audio item to generate per model call.",
    )
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Number of manifest shards for this inference run.")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard id for --num-shards.")
    parser.add_argument("--dont-quantize", action=argparse.BooleanOptionalAction, default=True)
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

    model = load_mossland_model(args.checkpoint_dir, device=args.device)
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
        prediction_path, extra_metadata = generate_chunked_prediction(
            model,
            item,
            args.output_dir,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            chunk_batch_size=args.chunk_batch_size,
            dont_quantize=args.dont_quantize,
            overwrite=args.overwrite,
        )
        metadata = dict(item.metadata)
        metadata.update(extra_metadata)
        rows.append(replace(item, prediction_path=prediction_path, metadata=metadata).to_json())
        if args.progress_every > 0 and (index == 1 or index == len(items) or index % args.progress_every == 0):
            print(f"processed={index}/{len(items)} last={item.item_id}", flush=True)
    write_jsonl(rows, output_manifest)


if __name__ == "__main__":
    main()
