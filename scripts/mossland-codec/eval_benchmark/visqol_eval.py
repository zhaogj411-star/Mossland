from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

from .audio_io import align_length, load_audio_segment, reference_path_for_item
from .manifest import EvalItem, read_manifest, write_jsonl


DEFAULT_VISQOL_BIN = Path("tmp/eval_metric_refs/visqol/bazel-bin/visqol")
DEFAULT_VISQOL_AUDIO_MODEL = Path("tmp/eval_metric_refs/visqol/model/libsvm_nu_svr_model.txt")


def bucket_key(item: EvalItem) -> str:
    if item.low_sample_rate:
        return f"{item.task_id}@{item.low_sample_rate}"
    return item.task_id


def parse_csv_list(value: str | None) -> set[str] | None:
    if value is None or value.strip() == "":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def select_items(
    items: list[EvalItem],
    tasks: set[str] | None = None,
    max_items: int = 0,
    num_shards: int = 1,
    shard_id: int = 0,
) -> list[EvalItem]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < --num-shards")
    selected = []
    for index, item in enumerate(items):
        if item.prediction_path is None:
            continue
        if tasks is not None and item.task_id not in tasks and bucket_key(item) not in tasks:
            continue
        if index % num_shards != shard_id:
            continue
        selected.append(item)
        if max_items and len(selected) >= max_items:
            break
    return selected


def _mono_48k(audio: torch.Tensor, source_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sample_rate != target_sample_rate:
        audio = AF.resample(audio, source_sample_rate, target_sample_rate)
    return audio.clamp(-1.0, 1.0)


def save_visqol_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path),
        audio.detach().cpu().float().clamp(-1.0, 1.0),
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def converted_pair_paths(output_dir: Path, item: EvalItem) -> tuple[Path, Path]:
    bucket = bucket_key(item)
    stem = f"{item.item_id}_seed{item.seed}"
    wav_dir = output_dir / "wav48_mono" / bucket
    return wav_dir / f"{stem}_ref.wav", wav_dir / f"{stem}_deg.wav"


def prepare_visqol_pair(
    item: EvalItem,
    output_dir: Path,
    sample_rate: int = 48000,
    reuse_wavs: bool = True,
) -> tuple[Path, Path]:
    if item.prediction_path is None:
        raise ValueError(f"item {item.item_id} has no prediction_path")
    reference_path = reference_path_for_item(item)
    if reference_path is None:
        raise ValueError(f"item {item.item_id} has no reference path")

    ref_out, deg_out = converted_pair_paths(output_dir, item)
    if reuse_wavs and ref_out.exists() and ref_out.stat().st_size > 0 and deg_out.exists() and deg_out.stat().st_size > 0:
        return ref_out, deg_out

    reference = load_audio_segment(
        reference_path,
        item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    degraded = load_audio_segment(item.prediction_path, item.sample_rate, stereo=True)
    reference, degraded = align_length(reference.float(), degraded.float())
    reference = _mono_48k(reference, item.sample_rate, sample_rate)
    degraded = _mono_48k(degraded, item.sample_rate, sample_rate)
    reference, degraded = align_length(reference, degraded)
    save_visqol_wav(ref_out, reference, sample_rate)
    save_visqol_wav(deg_out, degraded, sample_rate)
    return ref_out, deg_out


def write_batch_csv(pairs: list[tuple[Path, Path]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "degraded"])
        for reference, degraded in pairs:
            writer.writerow([str(reference), str(degraded)])


def read_visqol_results(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values = []
        for row in reader:
            raw = row.get("moslqo") or row.get("mos_lqo") or row.get("MOS-LQO")
            if raw is None:
                raise ValueError(f"ViSQOL result row has no MOS-LQO column: {row}")
            values.append(float(raw))
    return values


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = row["metrics"]["visqol_moslqo"]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            buckets.setdefault(row["bucket"], []).append(float(value))

    summary = {}
    for key, values in sorted(buckets.items()):
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[key] = {
            "count": float(len(values)),
            "visqol_moslqo/mean": float(tensor.mean().item()),
            "visqol_moslqo/median": float(tensor.median().item()),
        }
    return summary


def write_summary_tables(summary: dict[str, dict[str, float]], output_dir: Path) -> None:
    metric_names = sorted({name for metrics in summary.values() for name in metrics})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", *metric_names])
        writer.writeheader()
        for bucket, metrics in sorted(summary.items()):
            writer.writerow({"bucket": bucket, **metrics})

    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        headers = ["bucket", *metric_names]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for bucket, metrics in sorted(summary.items()):
            row = {"bucket": bucket, **metrics}
            values = []
            for header in headers:
                value = row.get(header, "")
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def run_visqol(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visqol_bin = Path(args.visqol_bin)
    if not visqol_bin.exists():
        raise FileNotFoundError(f"ViSQOL binary not found: {visqol_bin}")

    items = select_items(
        read_manifest(args.manifest),
        tasks=parse_csv_list(args.tasks),
        max_items=args.max_items,
        num_shards=getattr(args, "num_shards", 1),
        shard_id=getattr(args, "shard_id", 0),
    )
    if not items:
        raise ValueError("no manifest rows with prediction_path matched the requested filters")

    pairs = [
        prepare_visqol_pair(
            item,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            reuse_wavs=not args.overwrite_wavs,
        )
        for item in items
    ]
    batch_csv = output_dir / "batch.csv"
    results_csv = output_dir / "visqol_results.csv"
    write_batch_csv(pairs, batch_csv)
    results_csv.unlink(missing_ok=True)

    command = [
        str(visqol_bin),
        f"--batch_input_csv={batch_csv}",
        f"--results_csv={results_csv}",
    ]
    if args.similarity_model:
        command.append(f"--similarity_to_quality_model={args.similarity_model}")
    if args.use_speech_mode:
        command.append("--use_speech_mode")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (output_dir / "visqol_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "visqol_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ViSQOL failed with exit code {completed.returncode}; "
            f"see {output_dir / 'visqol_stderr.txt'}"
        )

    scores = read_visqol_results(results_csv)
    if len(scores) != len(items):
        raise RuntimeError(f"ViSQOL returned {len(scores)} rows for {len(items)} input pairs")

    rows = []
    for item, (reference_wav, degraded_wav), score in zip(items, pairs, scores):
        rows.append(
            {
                **item.to_json(),
                "bucket": bucket_key(item),
                "visqol_reference_wav": str(reference_wav),
                "visqol_degraded_wav": str(degraded_wav),
                "visqol_sample_rate": args.sample_rate,
                "visqol_mode": "speech" if args.use_speech_mode else "audio",
                "metrics": {"visqol_moslqo": float(score)},
            }
        )

    summary = summarize(rows)
    write_jsonl(rows, output_dir / "results.jsonl")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction manifests with Google ViSQOL.")
    parser.add_argument("--manifest", required=True, help="Manifest/results JSONL with prediction_path.")
    parser.add_argument("--output-dir", required=True, help="Directory for ViSQOL inputs and outputs.")
    parser.add_argument("--visqol-bin", default=str(DEFAULT_VISQOL_BIN), help="Path to built ViSQOL binary.")
    parser.add_argument(
        "--similarity-model",
        default=str(DEFAULT_VISQOL_AUDIO_MODEL),
        help="Path to ViSQOL audio-mode libsvm model; ignored by ViSQOL speech mode.",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs or bucket IDs to include.")
    parser.add_argument("--max-items", type=int, default=0, help="Maximum matched rows to evaluate; 0 means all.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of disjoint manifest shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--sample-rate", type=int, default=48000, help="WAV sample rate for ViSQOL audio mode.")
    parser.add_argument("--use-speech-mode", action="store_true", help="Use ViSQOL speech mode instead of audio mode.")
    parser.add_argument("--overwrite-wavs", action="store_true", help="Regenerate converted 48 kHz mono WAVs.")
    run_visqol(parser.parse_args())


if __name__ == "__main__":
    main()
