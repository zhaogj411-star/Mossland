from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import write_jsonl


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def find_audio(audio_root: Path, ytid: str) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        path = audio_root / f"{ytid}{ext}"
        if path.exists():
            return path.resolve()
    for ext in AUDIO_EXTENSIONS:
        matches = list(audio_root.rglob(f"{ytid}{ext}"))
        if matches:
            return matches[0].resolve()
    return None


def build_rows(
    metadata_path: Path,
    audio_root: Path,
    sample_rate: int,
    max_items: int,
    audio_is_clipped: bool = False,
) -> tuple[list[dict], list[dict]]:
    rows = []
    missing = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            audio_path = find_audio(audio_root, item["ytid"])
            if audio_path is None:
                missing.append(item)
                continue
            start = float(item["start_s"])
            end = float(item["end_s"])
            manifest_start = 0.0 if audio_is_clipped else start
            rows.append(
                {
                    "item_id": f"musiccaps__{item['ytid']}__{int(start)}_{int(end)}",
                    "task_id": "reconstruct",
                    "source_path": str(audio_path),
                    "reference_path": str(audio_path),
                    "start_seconds": manifest_start,
                    "duration_seconds": end - start,
                    "sample_rate": sample_rate,
                    "metadata": {
                        "dataset": "google/MusicCaps",
                        "ytid": item["ytid"],
                        "caption": item.get("caption", ""),
                        "is_audioset_eval": item.get("is_audioset_eval"),
                        "original_start_s": start,
                        "original_end_s": end,
                        "audio_is_clipped": audio_is_clipped,
                    },
                }
            )
            if max_items > 0 and len(rows) >= max_items:
                break
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CoDiCodec-style MusicCaps codec eval manifest.")
    parser.add_argument("--metadata", required=True, help="MusicCaps metadata JSONL from google/MusicCaps.")
    parser.add_argument("--audio-root", required=True, help="Directory containing downloaded MusicCaps audio by ytid.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--missing-output", default=None, help="Optional JSONL path for metadata rows without local audio.")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-items", type=int, default=0, help="Maximum rows to emit; <=0 means all available audio.")
    parser.add_argument(
        "--audio-is-clipped",
        action="store_true",
        help="Use start_seconds=0 because local audio files already contain the MusicCaps clip.",
    )
    args = parser.parse_args()

    rows, missing = build_rows(
        metadata_path=Path(args.metadata),
        audio_root=Path(args.audio_root),
        sample_rate=args.sample_rate,
        max_items=args.max_items,
        audio_is_clipped=args.audio_is_clipped,
    )
    write_jsonl(rows, args.output)
    if args.missing_output:
        write_jsonl(missing, args.missing_output)
    print(f"wrote {len(rows)} rows; missing_audio={len(missing)}")


if __name__ == "__main__":
    main()
