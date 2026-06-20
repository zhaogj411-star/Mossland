from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fad_backends import summarize_fad
from .manifest import write_jsonl
from .metrics import aggregate_metric_rows
from .run import write_summary_tables


def read_result_rows(paths: list[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        result_path = Path(path)
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def infer_fad_metric_name(rows: list[dict], explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    names = {
        str(row["fad_metric_name"])
        for row in rows
        if row.get("reference_fad_embedding") is not None
        and row.get("prediction_fad_embedding") is not None
        and row.get("fad_metric_name")
    }
    if not names:
        return None
    if len(names) > 1:
        raise ValueError(f"multiple FAD metric names found: {sorted(names)}")
    return next(iter(names))


def aggregate_results(
    result_paths: list[str | Path],
    output_dir: str | Path,
    *,
    fad_metric_name: str | None = None,
) -> dict[str, dict[str, float]]:
    rows = read_result_rows(result_paths)
    summary = aggregate_metric_rows(rows)
    metric_name = infer_fad_metric_name(rows, fad_metric_name)
    if metric_name is not None:
        for key, fad_values in summarize_fad(rows, metric_name).items():
            summary.setdefault(key, {}).update(fad_values)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_path / "results.jsonl")
    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eval_benchmark results JSONL files.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more eval_benchmark results.jsonl files to merge.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for merged results and summary files.")
    parser.add_argument(
        "--fad-metric-name",
        default=None,
        help="Optional FAD metric name override; defaults to fad_metric_name stored in rows.",
    )
    args = parser.parse_args()
    summary = aggregate_results(args.results, args.output_dir, fad_metric_name=args.fad_metric_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
