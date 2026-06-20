from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import EvalItem, read_manifest, write_jsonl


DEFAULT_TASKS = ("reconstruct", "super_resolution", "mono_to_stereo")
DEFAULT_SR_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 40000)


def parse_csv(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in parse_csv(value))


def base_id(item: EvalItem) -> str:
    return item.item_id.replace("/", "__")


def build_rows(
    items: list[EvalItem],
    tasks: tuple[str, ...],
    sr_rates: tuple[int, ...],
    stereo_seeds: tuple[int, ...],
    max_items: int,
    duration_seconds: float | None,
) -> list[dict]:
    if max_items > 0:
        items = items[:max_items]

    rows = []
    for item in items:
        reference = item.reference_path or item.source_path
        common = {
            "source_path": str(item.source_path.resolve()),
            "reference_path": str(reference.resolve()),
            "start_seconds": item.start_seconds,
            "duration_seconds": item.duration_seconds if duration_seconds is None else duration_seconds,
            "sample_rate": item.sample_rate,
            "metadata": {
                **item.metadata,
                "derived_from_item_id": item.item_id,
                "derived_from_task_id": item.task_id,
            },
        }
        item_prefix = base_id(item)
        if "reconstruct" in tasks:
            rows.append(
                {
                    **common,
                    "item_id": f"{item_prefix}__reconstruct",
                    "task_id": "reconstruct",
                }
            )
        if "super_resolution" in tasks:
            for rate in sr_rates:
                rows.append(
                    {
                        **common,
                        "item_id": f"{item_prefix}__super_resolution_{rate}",
                        "task_id": "super_resolution",
                        "low_sample_rate": rate,
                    }
                )
        if "mono_to_stereo" in tasks:
            for seed in stereo_seeds:
                rows.append(
                    {
                        **common,
                        "item_id": f"{item_prefix}__mono_to_stereo_seed{seed}",
                        "task_id": "mono_to_stereo",
                        "seed": seed,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive reconstruction, super-resolution and mono-to-stereo manifests from audio rows."
    )
    parser.add_argument("--input", required=True, help="Existing JSONL/JSON/CSV manifest with source audio paths.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--max-items", type=int, default=50, help="Maximum source rows; <=0 means all.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--sr-rates", default=",".join(str(rate) for rate in DEFAULT_SR_RATES))
    parser.add_argument("--stereo-seeds", default="0,1,2,3")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override clip duration. Omit to preserve each source row duration.",
    )
    args = parser.parse_args()

    rows = build_rows(
        read_manifest(args.input),
        tasks=parse_csv(args.tasks),
        sr_rates=parse_int_csv(args.sr_rates),
        stereo_seeds=parse_int_csv(args.stereo_seeds),
        max_items=args.max_items,
        duration_seconds=args.duration_seconds,
    )
    write_jsonl(rows, Path(args.output))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
