from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .manifest import write_jsonl


DEFAULT_DATASET = "mahendra0203/musiccaps_processed_full"


def row_youtube_id(row: dict[str, Any]) -> str:
    for key in ("youtube_id", "ytid", "yt_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError("row has no youtube_id/ytid field")


def row_audio_bytes(row: dict[str, Any]) -> bytes:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise TypeError("row audio field is not a dict")
    data = audio.get("bytes")
    if isinstance(data, bytes):
        return data
    path = audio.get("path")
    if path:
        path = Path(path)
        if path.exists():
            return path.read_bytes()
    raise ValueError("row audio field has neither bytes nor readable path")


def download_rows(
    dataset_name: str,
    split: str,
    audio_dir: Path,
    max_items: int = 0,
    start_index: int = 0,
    streaming: bool = True,
    progress_every: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    ds = ds.cast_column("audio", Audio(decode=False))
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    audio_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(ds):
        if index < start_index:
            continue
        if max_items > 0 and len(successes) + len(failures) >= max_items:
            break
        try:
            ytid = row_youtube_id(row)
            output_path = audio_dir / f"{ytid}.wav"
            if output_path.exists() and output_path.stat().st_size > 0:
                status = "exists"
            else:
                output_path.write_bytes(row_audio_bytes(row))
                status = "downloaded"
            successes.append(
                {
                    "ytid": ytid,
                    "audio_path": str(output_path.resolve()),
                    "status": status,
                    "dataset": dataset_name,
                    "start_s": row.get("start_time", row.get("start_s")),
                    "end_s": row.get("end_time", row.get("end_s")),
                    "caption": row.get("caption", ""),
                }
            )
            processed = len(successes) + len(failures)
            if progress_every > 0 and processed % progress_every == 0:
                print(f"processed={processed} success={len(successes)} failed={len(failures)} last={ytid}", flush=True)
        except Exception as exc:
            failures.append({"status": "failed", "dataset": dataset_name, "index": index, "error": str(exc)})
            processed = len(successes) + len(failures)
            if progress_every > 0 and (processed % progress_every == 0 or progress_every == 1):
                print(f"failed index={index}: {exc}", flush=True)
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MusicCaps audio from a Hugging Face audio mirror.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset containing audio and youtube_id fields.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-dir", required=True, help="Output directory for wav files named {youtube_id}.wav.")
    parser.add_argument("--success-output", default=None, help="Optional JSONL receipt for successful clips.")
    parser.add_argument("--failure-output", default=None, help="Optional JSONL receipt for failed clips.")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1, help="Print one progress line every N processed rows.")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download the HF dataset locally before iterating; default streams rows.",
    )
    args = parser.parse_args()

    successes, failures = download_rows(
        dataset_name=args.dataset,
        split=args.split,
        audio_dir=Path(args.audio_dir),
        max_items=args.max_items,
        start_index=args.start_index,
        streaming=not args.no_streaming,
        progress_every=args.progress_every,
    )
    if args.success_output:
        write_jsonl(successes, args.success_output)
    if args.failure_output:
        write_jsonl(failures, args.failure_output)
    print(f"success={len(successes)} failed={len(failures)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
