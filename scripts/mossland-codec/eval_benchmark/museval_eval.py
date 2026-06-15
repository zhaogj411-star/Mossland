from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from .audio_io import align_length, load_audio_segment, reference_path_for_item
from .manifest import EvalItem, read_manifest, write_jsonl
from .run import write_summary_tables


TARGET_BY_TASK = {
    "separate_vocals": "vocals",
    "separate_drums": "drums",
    "separate_bass": "bass",
    "separate_other": "other",
    "separate_accompaniment": "accompaniment",
}


def install_imp_compat() -> None:
    """Provide the tiny part of removed stdlib imp needed by future/past.

    museval imports musdb -> stempeg -> ffmpeg-python -> past.builtins, and
    older future/past releases still import `imp.reload` on Python 3.12.
    Keeping the shim local avoids patching site-packages.
    """
    if "imp" in sys.modules:
        return
    module = types.ModuleType("imp")
    module.reload = importlib.reload
    sys.modules["imp"] = module


def import_museval():
    install_imp_compat()
    import museval

    return museval


def track_key(item: EvalItem) -> str:
    value = item.metadata.get("track_name") if isinstance(item.metadata, dict) else None
    if value:
        return str(value)
    for suffix in ("__vocals", "__drums", "__bass", "__other", "__accompaniment"):
        if item.item_id.endswith(suffix):
            return item.item_id[: -len(suffix)]
    return item.item_id


def target_name(item: EvalItem) -> str | None:
    return TARGET_BY_TASK.get(item.task_id)


def group_items(items: Iterable[EvalItem]) -> dict[str, dict[str, EvalItem]]:
    grouped: dict[str, dict[str, EvalItem]] = defaultdict(dict)
    for item in items:
        target = target_name(item)
        if target is None or item.prediction_path is None:
            continue
        grouped[track_key(item)][target] = item
    return dict(grouped)


def load_pair(item: EvalItem, align: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if item.prediction_path is None:
        raise ValueError(f"item {item.item_id} has no prediction_path")
    reference_path = reference_path_for_item(item)
    if reference_path is None:
        raise ValueError(f"item {item.item_id} has no reference path")
    reference = load_audio_segment(
        reference_path,
        item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    estimate = load_audio_segment(item.prediction_path, item.sample_rate, stereo=True)
    reference = reference.float()
    estimate = estimate.float()
    if align:
        estimate, reference = align_length(estimate, reference)
    return reference, estimate


def should_check_lengths(item: EvalItem, mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return True
    if mode != "auto":
        raise ValueError(f"unsupported length_check mode: {mode!r}")
    return item.duration_seconds is None


def validate_pair_lengths(
    track: str,
    target: str,
    item: EvalItem,
    reference_frames: int,
    prediction_frames: int,
    min_ratio: float,
    max_mismatch_seconds: float,
    mode: str,
) -> None:
    if not should_check_lengths(item, mode):
        return
    if reference_frames <= 0 or prediction_frames <= 0:
        raise ValueError(
            f"track {track!r} target {target!r} has empty audio: "
            f"reference_frames={reference_frames}, prediction_frames={prediction_frames}"
        )

    mismatch_seconds = abs(reference_frames - prediction_frames) / float(item.sample_rate)
    ratio = min(reference_frames, prediction_frames) / float(max(reference_frames, prediction_frames))
    if mismatch_seconds <= max_mismatch_seconds or ratio >= min_ratio:
        return

    reference_seconds = reference_frames / float(item.sample_rate)
    prediction_seconds = prediction_frames / float(item.sample_rate)
    raise ValueError(
        f"track {track!r} target {target!r} length mismatch: "
        f"reference={reference_seconds:.3f}s ({reference_frames} frames), "
        f"prediction={prediction_seconds:.3f}s ({prediction_frames} frames), "
        f"ratio={ratio:.4f}. Refusing to min-length truncate this "
        f"{'full-track' if item.duration_seconds is None else 'configured'} evaluation; "
        "use --length-check off to restore legacy truncation."
    )


def _to_museval_array(audio: torch.Tensor) -> torch.Tensor:
    return audio[:2].T.contiguous()


def _finite_stats(values) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
    }


def evaluate_track(
    museval_module,
    track: str,
    items_by_target: dict[str, EvalItem],
    win_seconds: float,
    hop_seconds: float,
    length_check: str = "auto",
    min_prediction_reference_ratio: float = 0.95,
    max_length_mismatch_seconds: float = 2.0,
) -> list[dict]:
    target_order = ("vocals", "drums", "bass", "other", "accompaniment")
    targets = [target for target in target_order if target in items_by_target]
    if len(targets) < 2:
        raise ValueError(f"track {track!r} has {len(targets)} targets; museval v4 needs at least two targets")

    references = []
    estimates = []
    rows = []
    sample_rate = None
    min_length = None
    loaded = {}
    for target in targets:
        item = items_by_target[target]
        reference, estimate = load_pair(item, align=False)
        sample_rate = item.sample_rate if sample_rate is None else sample_rate
        if item.sample_rate != sample_rate:
            raise ValueError(f"track {track!r} mixes sample rates")
        validate_pair_lengths(
            track=track,
            target=target,
            item=item,
            reference_frames=reference.shape[-1],
            prediction_frames=estimate.shape[-1],
            min_ratio=min_prediction_reference_ratio,
            max_mismatch_seconds=max_length_mismatch_seconds,
            mode=length_check,
        )
        min_length = reference.shape[-1] if min_length is None else min(min_length, reference.shape[-1])
        min_length = min(min_length, estimate.shape[-1])
        loaded[target] = (item, reference, estimate)

    assert sample_rate is not None and min_length is not None
    for target in targets:
        item, reference, estimate = loaded[target]
        references.append(_to_museval_array(reference[..., :min_length]).numpy())
        estimates.append(_to_museval_array(estimate[..., :min_length]).numpy())

    sdr, isr, sir, sar = museval_module.evaluate(
        references,
        estimates,
        win=int(round(win_seconds * sample_rate)),
        hop=int(round(hop_seconds * sample_rate)),
        mode="v4",
        padding=True,
    )
    metric_arrays = {
        "museval_v4_sdr_db": sdr,
        "museval_v4_isr_db": isr,
        "museval_v4_sir_db": sir,
        "museval_v4_sar_db": sar,
    }

    for index, target in enumerate(targets):
        item = items_by_target[target]
        metrics = {}
        frames = {}
        for name, values in metric_arrays.items():
            target_values = values[index]
            stats = _finite_stats(target_values)
            metrics[name] = stats["mean"]
            metrics[f"{name}_frame_median"] = stats["median"]
            frames[name] = torch.as_tensor(target_values, dtype=torch.float64).flatten().tolist()
        rows.append(
            {
                **item.to_json(),
                "track": track,
                "target": target,
                "museval_mode": "v4",
                "museval_win_seconds": win_seconds,
                "museval_hop_seconds": hop_seconds,
                "museval_evaluated_seconds": min_length / float(sample_rate),
                "museval_length_check": length_check,
                "museval_min_prediction_reference_ratio": min_prediction_reference_ratio,
                "museval_max_length_mismatch_seconds": max_length_mismatch_seconds,
                "metrics": metrics,
                "museval_frame_metrics": frames,
            }
        )
    return rows


def aggregate(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        buckets[row["target"]].append(row["metrics"])

    summary: dict[str, dict[str, float]] = {}
    for bucket, metric_rows in sorted(buckets.items()):
        metric_names = sorted({name for metrics in metric_rows for name in metrics})
        summary[bucket] = {"count": float(len(metric_rows))}
        for name in metric_names:
            values = [
                float(metrics[name])
                for metrics in metric_rows
                if name in metrics and math.isfinite(float(metrics[name]))
            ]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            summary[bucket][f"{name}/mean"] = float(tensor.mean().item())
            summary[bucket][f"{name}/median"] = float(tensor.median().item())
    return summary


def write_track_summary(rows: Iterable[dict], output_dir: Path) -> None:
    path = output_dir / "track_summary.csv"
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track", "target", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow({"track": row["track"], "target": row["target"], **row["metrics"]})


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_items(read_manifest(args.manifest))
    if args.max_tracks > 0:
        grouped = dict(list(sorted(grouped.items()))[: args.max_tracks])
    if not grouped:
        raise ValueError("no separation rows with prediction_path found")

    museval_module = import_museval()
    rows = []
    errors = []
    for track, items_by_target in sorted(grouped.items()):
        try:
            rows.extend(
                evaluate_track(
                    museval_module,
                    track,
                    items_by_target,
                    win_seconds=args.win_seconds,
                    hop_seconds=args.hop_seconds,
                    length_check=args.length_check,
                    min_prediction_reference_ratio=args.min_prediction_reference_ratio,
                    max_length_mismatch_seconds=args.max_length_mismatch_seconds,
                )
            )
        except Exception as exc:
            errors.append({"track": track, "error": str(exc)})
            if not args.keep_going:
                raise

    write_jsonl(rows, output_dir / "results.jsonl")
    if errors:
        write_jsonl(errors, output_dir / "errors.jsonl")
    summary = aggregate(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_dir)
    write_track_summary(rows, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run museval BSS Eval v4 on paired separation predictions.")
    parser.add_argument("--manifest", required=True, help="Manifest/results JSONL with vocals and accompaniment prediction_path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for museval summaries.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Maximum grouped tracks to evaluate; 0 means all.")
    parser.add_argument("--win-seconds", type=float, default=1.0, help="museval window size in seconds.")
    parser.add_argument("--hop-seconds", type=float, default=1.0, help="museval hop size in seconds.")
    parser.add_argument(
        "--length-check",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Protect against misleading min-length truncation. "
            "'auto' checks only full-track rows without duration_seconds; "
            "'always' checks every row; 'off' restores legacy truncation."
        ),
    )
    parser.add_argument(
        "--min-prediction-reference-ratio",
        type=float,
        default=0.95,
        help="Minimum shorter/longer audio length ratio accepted when length checking is enabled.",
    )
    parser.add_argument(
        "--max-length-mismatch-seconds",
        type=float,
        default=2.0,
        help="Absolute length mismatch tolerated before applying the ratio guard.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Record per-track errors instead of stopping.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
