from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from .visqol_eval import write_summary_tables


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}


def merge_shards(shard_root: Path, output_dir: Path, num_shards: int) -> dict[str, dict[str, float]]:
    rows = []
    for shard_id in range(num_shards):
        result_path = shard_root / f"shard{shard_id:02d}of{num_shards}" / "results.jsonl"
        with result_path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["metrics"]["visqol_moslqo"])
        if math.isfinite(value):
            buckets.setdefault(row["bucket"], []).append(value)

    summary: dict[str, dict[str, float]] = {}
    for bucket, values in sorted(buckets.items()):
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[bucket] = {
            "count": float(len(values)),
            "visqol_moslqo/mean": float(tensor.mean().item()),
            "visqol_moslqo/median": float(tensor.median().item()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_summary_tables(summary, output_dir)
    return summary


def run_sharded(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    shard_root = Path(args.shard_output_dir) if args.shard_output_dir else output_dir.with_name(output_dir.name + "_shards")
    shard_root.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    if args.limit_threads:
        env_base.update(THREAD_ENV)
    if args.pythonpath:
        env_base["PYTHONPATH"] = args.pythonpath

    processes: list[tuple[int, Path, subprocess.Popen]] = []
    for shard_id in range(args.num_shards):
        shard_name = f"shard{shard_id:02d}of{args.num_shards}"
        shard_dir = shard_root / shard_name
        log_path = shard_root / f"{shard_name}.log"
        command = [
            sys.executable,
            "-m",
            "scripts.mossland-codec.eval_benchmark.visqol_eval",
            "--manifest",
            args.manifest,
            "--output-dir",
            str(shard_dir),
            "--num-shards",
            str(args.num_shards),
            "--shard-id",
            str(shard_id),
            "--visqol-bin",
            args.visqol_bin,
            "--similarity-model",
            args.similarity_model,
        ]
        if args.tasks:
            command.extend(["--tasks", args.tasks])
        if args.sample_rate:
            command.extend(["--sample-rate", str(args.sample_rate)])
        if args.overwrite_wavs:
            command.append("--overwrite-wavs")
        if args.use_speech_mode:
            command.append("--use-speech-mode")
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env_base)
        processes.append((shard_id, log_path, process))
        print(f"started shard={shard_id}/{args.num_shards} pid={process.pid} log={log_path}", flush=True)

    failed = []
    for shard_id, log_path, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_id, return_code, log_path))
    if failed:
        details = ", ".join(f"shard {sid} rc={rc} log={log}" for sid, rc, log in failed)
        raise RuntimeError(f"one or more shards failed: {details}")

    summary = merge_shards(shard_root, output_dir, args.num_shards)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViSQOL in CPU shards and aggregate results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-output-dir", default=None)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--visqol-bin", default="tmp/eval_metric_refs/visqol/bazel-bin/visqol")
    parser.add_argument("--similarity-model", default="tmp/eval_metric_refs/visqol/model/libsvm_nu_svr_model.txt")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--use-speech-mode", action="store_true")
    parser.add_argument("--overwrite-wavs", action="store_true")
    parser.add_argument("--pythonpath", default=str(Path.cwd()), help="PYTHONPATH for child processes.")
    parser.add_argument("--limit-threads", action="store_true", default=True)
    parser.add_argument("--no-limit-threads", action="store_false", dest="limit_threads")
    run_sharded(parser.parse_args())


if __name__ == "__main__":
    main()
