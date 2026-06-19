import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def _probe_duration_seconds(path: Path) -> float | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        value = float(completed.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None
    return value if value > 0 else None


def _decode_segment(
    path: Path,
    *,
    offset_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    channels: int,
    sample_size: int,
) -> torch.Tensor:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{offset_seconds:.6f}",
            "-t",
            f"{duration_seconds:.6f}",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr}")
    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced empty audio for {path}")
    usable = audio.size - (audio.size % channels)
    audio = audio[:usable].reshape(-1, channels).T
    tensor = torch.from_numpy(audio)
    if tensor.shape[-1] > sample_size:
        tensor = tensor[..., :sample_size]
    elif tensor.shape[-1] < sample_size:
        tensor = torch.nn.functional.pad(tensor, (0, sample_size - tensor.shape[-1]))
    return tensor.clamp(-1.0, 1.0).contiguous()


def _segment_offsets(duration: float | None, segment_seconds: float, count: int):
    if count <= 1:
        return [0.0]
    if duration is None or duration <= segment_seconds:
        return [0.0 for _ in range(count)]
    max_start = max(duration - segment_seconds, 0.0)
    return [max_start * i / (count - 1) for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--sample-size", type=int, default=96000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--segments-per-file", type=int, default=1)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    args = parser.parse_args()

    source_index = Path(args.source_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_seconds = args.sample_size / args.sample_rate
    rows = []

    with open(source_index, "r", encoding="utf-8") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]

    for file_index, path in enumerate(paths):
        duration = _probe_duration_seconds(path)
        if (
            args.max_duration_seconds is not None
            and duration is not None
            and duration >= args.max_duration_seconds
        ):
            print(f"skip long file {path}: duration={duration:.3f}", flush=True)
            continue
        for segment_index, offset in enumerate(
            _segment_offsets(duration, segment_seconds, args.segments_per_file)
        ):
            cache_name = f"{file_index:03d}_{segment_index:03d}_{path.stem}.pt"
            cache_path = output_dir / cache_name
            if not cache_path.exists():
                audio = _decode_segment(
                    path,
                    offset_seconds=offset,
                    duration_seconds=segment_seconds,
                    sample_rate=args.sample_rate,
                    channels=args.channels,
                    sample_size=args.sample_size,
                )
                torch.save(
                    {
                        "audio": audio,
                        "info": {
                            "source_path": str(path),
                            "source_duration_seconds": duration,
                            "offset_seconds": offset,
                            "segment_seconds": segment_seconds,
                            "sample_rate": args.sample_rate,
                        },
                    },
                    cache_path,
                )
            rows.append(
                {
                    "cache_path": str(cache_path),
                    "source_path": str(path),
                    "source_duration_seconds": duration,
                    "offset_seconds": offset,
                    "segment_seconds": segment_seconds,
                }
            )
            print(f"cached {cache_path}", flush=True)

    if not rows:
        raise RuntimeError("no cached segments were created")
    index_path = output_dir / "index.jsonl"
    tmp_path = output_dir / f".{index_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(index_path)
    print(f"wrote {len(rows)} rows to {index_path}")


if __name__ == "__main__":
    main()
