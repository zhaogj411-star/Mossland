from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .audio_io import align_length, load_audio_segment, reference_path_for_item
from .manifest import read_manifest, write_jsonl
from .metrics import si_sdr, snr
from .run import write_summary_tables


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name, value in row["metrics"].items():
            buckets[row["task_id"]][name].append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for task_id, metric_values in sorted(buckets.items()):
        first_metric = next(iter(metric_values.values()))
        summary[task_id] = {"count": float(len(first_metric))}
        for name, values in sorted(metric_values.items()):
            tensor = torch.tensor(values, dtype=torch.float64)
            summary[task_id][f"{name}/mean"] = float(tensor.mean().item())
            summary[task_id][f"{name}/median"] = float(tensor.median().item())
    return summary


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    items = read_manifest(args.manifest)
    for index, item in enumerate(items, start=1):
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
        prediction, reference = align_length(prediction.float(), reference.float())
        row = {
            **item.to_json(),
            "reference_path": str(reference_path),
            "metrics": {
                "sdr_db": snr(prediction, reference),
                "si_sdr_db": si_sdr(prediction, reference),
            },
        }
        rows.append(row)
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(items)):
            print(f"processed={index}/{len(items)} last={item.item_id}", flush=True)

    summary = summarize(rows)
    write_jsonl(rows, output_dir / "results.jsonl")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute fast waveform SDR and SI-SDR for separation prediction manifests."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
