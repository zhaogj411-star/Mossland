from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .audio_io import load_audio_segment, reference_path_for_item
from .fad_backends import FadBackend, build_fad_backend, summarize_fad
from .infer import build_task_input, generate_prediction, load_mossland_model
from .manifest import EvalItem, read_manifest, write_jsonl
from .metrics import aggregate_metric_rows, pair_metrics


def write_summary_tables(summary: dict[str, dict[str, float]], output_dir: Path) -> None:
    metric_names = sorted({name for metrics in summary.values() for name in metrics})
    rows = []
    for bucket, metrics in sorted(summary.items()):
        row = {"bucket": bucket}
        row.update(metrics)
        rows.append(row)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        headers = ["bucket", *metric_names]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            values = [row.get(header, "") for header in headers]
            formatted = [
                f"{value:.6g}" if isinstance(value, float) else str(value)
                for value in values
            ]
            handle.write("| " + " | ".join(formatted) + " |\n")


def evaluate_item(
    item: EvalItem,
    metrics_device: str | torch.device | None = None,
    fad_backend: FadBackend | None = None,
) -> dict:
    if item.prediction_path is None:
        raise ValueError(f"item {item.item_id} has no prediction_path")
    reference_path = reference_path_for_item(item)
    if reference_path is None:
        raise ValueError(f"item {item.item_id} has no reference path")

    prediction = load_audio_segment(item.prediction_path, item.sample_rate)
    reference = load_audio_segment(
        reference_path,
        item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
    )
    source, _ = build_task_input(item)
    row = {
        **item.to_json(),
        "reference_path": str(reference_path),
        "metrics": pair_metrics(
            prediction,
            reference,
            sample_rate=item.sample_rate,
            task_id=item.task_id,
            low_sample_rate=item.low_sample_rate,
            source=source,
            device=metrics_device,
        ),
    }
    if fad_backend is not None:
        fad = fad_backend.pair(reference, prediction, item.sample_rate)
        row["fad_embedding_backend"] = fad.backend_name
        row["fad_metric_name"] = fad.metric_name
        row["reference_fad_embedding"] = fad.reference_embedding.tolist()
        row["prediction_fad_embedding"] = fad.prediction_embedding.tolist()
    return row


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


def run(args: argparse.Namespace) -> None:
    all_items = read_manifest(args.manifest)
    items = select_items(
        all_items,
        max_items=args.max_items,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    output_dir = Path(args.output_dir)
    metrics_device = args.metrics_device
    if metrics_device is None and torch.cuda.is_available():
        metrics_device = "cuda"
    fad_backend = build_fad_backend(
        args.fad_backend,
        device=metrics_device,
        model_name=args.fad_model,
        cache_dir=args.fad_cache_dir,
    )
    model = None
    if args.checkpoint_dir:
        model = load_mossland_model(args.checkpoint_dir, device=args.device)

    evaluated = []
    total = len(items)
    if args.num_shards > 1 or args.max_items > 0:
        print(
            (
                f"selected {total}/{len(all_items)} rows "
                f"(shard {args.shard_id}/{args.num_shards}, max_items={args.max_items})"
            ),
            file=sys.stderr,
            flush=True,
        )
    for index, item in enumerate(items, start=1):
        if model is not None:
            prediction_path = generate_prediction(
                model,
                item,
                output_dir=output_dir / "predictions",
                dont_quantize=not args.quantized,
                overwrite=args.overwrite_predictions,
            )
            item = EvalItem(**{**item.__dict__, "prediction_path": prediction_path})
        evaluated.append(
            evaluate_item(
                item,
                metrics_device=metrics_device,
                fad_backend=fad_backend,
            )
        )
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == total):
            print(
                f"processed={index}/{total} last={item.item_id}",
                file=sys.stderr,
                flush=True,
            )

    fad_summary = summarize_fad(evaluated, fad_backend.metric_name)
    summary = aggregate_metric_rows(evaluated)
    for key, fad_values in fad_summary.items():
        summary.setdefault(key, {}).update(fad_values)

    write_jsonl(evaluated, output_dir / "results.jsonl")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mossland task evaluation.")
    parser.add_argument("--manifest", required=True, help="JSONL/JSON/CSV evaluation manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions and metrics.")
    parser.add_argument("--max-items", type=int, default=0, help="Maximum rows to evaluate; <=0 means all selected rows.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of manifest shards for this eval run.")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard id for --num-shards.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N rows; <=0 disables progress.")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional Mossland checkpoint directory.")
    parser.add_argument("--device", default=None, help="Torch device for checkpoint inference.")
    parser.add_argument(
        "--metrics-device",
        default=None,
        help="Torch device for metric STFT/FAD embedding; defaults to cuda when available.",
    )
    parser.add_argument(
        "--fad-backend",
        choices=("mel_proxy", "clap", "vggish", "none"),
        default="mel_proxy",
        help="FAD embedding backend. Use clap for CoDiCodec-style FAD_clap and vggish for FAD.",
    )
    parser.add_argument(
        "--fad-model",
        default=None,
        help="Optional model name/path for --fad-backend clap.",
    )
    parser.add_argument(
        "--fad-cache-dir",
        default="checkpoints/clap",
        help="Model cache directory for FAD backends that download checkpoints.",
    )
    parser.add_argument("--quantized", action="store_true", help="Use quantized latent path for generation.")
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Regenerate prediction wav files even if the deterministic output path already exists.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
