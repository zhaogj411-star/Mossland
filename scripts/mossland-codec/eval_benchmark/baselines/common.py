from __future__ import annotations

import argparse
from pathlib import Path

from ..manifest import EvalItem, read_manifest, write_jsonl


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, help="Input eval manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for baseline predictions.")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Optional JSONL manifest with prediction_path fields for eval_benchmark.run.",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Maximum items to process; <=0 means all.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate baseline predictions even when output wav already exists.",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N processed items.")


def maybe_print_progress(index: int, total: int, item_id: str, progress_every: int) -> None:
    if progress_every > 0 and (index % progress_every == 0 or index == total):
        print(f"processed={index}/{total} last={item_id}", flush=True)


def iter_items(args: argparse.Namespace) -> list[EvalItem]:
    items = read_manifest(args.manifest)
    if args.max_items > 0:
        return items[: args.max_items]
    return items


def baseline_prediction_path(output_dir: str | Path, baseline_name: str, item: EvalItem) -> Path:
    return Path(output_dir) / baseline_name / item.task_id / f"{item.item_id}_seed{item.seed}.wav"


def write_predicted_manifest(items: list[EvalItem], prediction_paths: list[Path], output_manifest: str | Path | None) -> None:
    if output_manifest is None:
        return
    rows = []
    for item, prediction_path in zip(items, prediction_paths, strict=True):
        rows.append({**item.to_json(), "prediction_path": str(prediction_path.resolve())})
    write_jsonl(rows, output_manifest)
