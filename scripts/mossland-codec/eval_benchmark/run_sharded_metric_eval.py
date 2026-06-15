from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}


def parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device id")
    return gpus


def run_sharded(args: argparse.Namespace) -> None:
    gpus = parse_gpus(args.gpus)
    num_shards = args.num_shards or len(gpus)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    output_dir = Path(args.output_dir)
    shard_root = Path(args.shard_output_dir) if args.shard_output_dir else output_dir.with_name(output_dir.name + "_shards")
    shard_root.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    if args.limit_threads:
        env_base.update(THREAD_ENV)
    if args.pythonpath:
        env_base["PYTHONPATH"] = args.pythonpath

    processes: list[tuple[int, Path, subprocess.Popen]] = []
    for shard_id in range(num_shards):
        shard_name = f"shard{shard_id:02d}of{num_shards}"
        shard_dir = shard_root / shard_name
        log_path = shard_root / f"{shard_name}.log"
        gpu = gpus[shard_id % len(gpus)]
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable,
            "-m",
            "scripts.mossland-codec.eval_benchmark.run",
            "--manifest",
            args.manifest,
            "--output-dir",
            str(shard_dir),
            "--num-shards",
            str(num_shards),
            "--shard-id",
            str(shard_id),
            "--fad-backend",
            args.fad_backend,
            "--metrics-device",
            args.metrics_device,
            "--progress-every",
            str(args.progress_every),
        ]
        if args.checkpoint_dir:
            command.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.device:
            command.extend(["--device", args.device])
        if args.quantized:
            command.append("--quantized")
        if args.overwrite_predictions:
            command.append("--overwrite-predictions")
        if args.fad_model:
            command.extend(["--fad-model", args.fad_model])
        if args.fad_cache_dir:
            command.extend(["--fad-cache-dir", args.fad_cache_dir])
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        processes.append((shard_id, log_path, process))
        print(f"started shard={shard_id}/{num_shards} gpu={gpu} pid={process.pid} log={log_path}", flush=True)

    failed = []
    for shard_id, log_path, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_id, return_code, log_path))
    if failed:
        details = ", ".join(f"shard {sid} rc={rc} log={log}" for sid, rc, log in failed)
        raise RuntimeError(f"one or more shards failed: {details}")

    results = [str(shard_root / f"shard{shard_id:02d}of{num_shards}" / "results.jsonl") for shard_id in range(num_shards)]
    command = [
        sys.executable,
        "-m",
        "scripts.mossland-codec.eval_benchmark.aggregate_results",
        "--results",
        *results,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, env=env_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval_benchmark.run in GPU shards and aggregate the results.")
    parser.add_argument("--manifest", required=True, help="Input prediction manifest JSONL/JSON/CSV.")
    parser.add_argument("--output-dir", required=True, help="Final aggregated output directory.")
    parser.add_argument("--shard-output-dir", default=None, help="Optional directory for per-shard outputs.")
    parser.add_argument("--fad-backend", choices=("mel_proxy", "clap", "vggish", "none"), required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids. Shards are assigned round-robin.")
    parser.add_argument("--num-shards", type=int, default=0, help="Defaults to the number of GPUs.")
    parser.add_argument("--metrics-device", default="cuda", help="Device visible inside each shard process.")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional Mossland checkpoint dir for sharded inference.")
    parser.add_argument("--device", default=None, help="Inference device visible inside each shard process.")
    parser.add_argument("--quantized", action="store_true", help="Use quantized latent path for generation.")
    parser.add_argument("--overwrite-predictions", action="store_true", help="Regenerate prediction wavs.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--fad-model", default=None)
    parser.add_argument("--fad-cache-dir", default="checkpoints/clap")
    parser.add_argument("--pythonpath", default=str(Path.cwd()), help="PYTHONPATH for child processes.")
    parser.add_argument("--limit-threads", action="store_true", default=True, help="Limit CPU library threads per shard.")
    parser.add_argument("--no-limit-threads", action="store_false", dest="limit_threads")
    run_sharded(parser.parse_args())


if __name__ == "__main__":
    main()
