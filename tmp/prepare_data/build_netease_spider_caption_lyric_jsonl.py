#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency for one-off script
    tqdm = None


# Hard-coded input/output parameters.
RESULTS_ROOT = Path(
    "/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/work/results/NETEASE_SPIDER"
)
SEPARATION_ROOT = Path(
    "/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/source_seperation/NETEASE_SPIDER"
)
LYRIC_DIR = Path(
    "/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/raw/NETEASE_SPIDER/wy-music-song-id-lyric"
)
OUTPUT_JSONL = Path(
    "/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/tmp/prepare_data/netease_spider_caption_lyric_paths.jsonl"
)
MAX_NUMS: int | None = 300000
SHOW_PROGRESS = True

ASR_GLOB = "**/lyrics-Qwen3-Omni/shard_*/qwen_direct_parts/part_*.asr.jsonl"
LYRIC_GLOB = "wy-music-song-id_*.jsons"
CAPTION_FIELD = "caption_Qwen3-Omni"
RAW_NETEASE_PREFIXES = (
    "qz_oss2:embodied-multimodality/public/Sonata/data/raw/NETEASE_SPIDER/",
    "/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/raw/NETEASE_SPIDER/",
)

T = TypeVar("T")


@dataclass
class LyricMatch:
    lyric: str | None
    lyric_file_path: str
    score: int


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def find_asr_files() -> list[Path]:
    return sorted(RESULTS_ROOT.glob(ASR_GLOB))


def find_lyric_files() -> list[Path]:
    return sorted(LYRIC_DIR.glob(LYRIC_GLOB))


def iter_with_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str,
) -> Iterable[T]:
    if not SHOW_PROGRESS or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def extract_song_id(record: dict) -> str | None:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        song_id = metadata.get("song_id")
        if song_id is not None:
            return str(song_id)

    source_audio_path = record.get("source_audio_path")
    if isinstance(source_audio_path, str) and source_audio_path:
        return Path(source_audio_path).stem
    return None


def extract_lrc_lyric(payload: dict) -> str | None:
    lyric_payload = payload.get("lyric")
    if isinstance(lyric_payload, str):
        return lyric_payload or None
    if not isinstance(lyric_payload, dict):
        return None

    lrc_payload = lyric_payload.get("lrc")
    if isinstance(lrc_payload, dict):
        lyric_text = lrc_payload.get("lyric")
        if isinstance(lyric_text, str):
            return lyric_text or None
    return None


def lyric_score(lyric: str | None) -> int:
    if lyric is None:
        return -1
    return len(lyric.strip())


def collect_song_ids(asr_files: Iterable[Path]) -> tuple[set[str], int]:
    asr_files = list(asr_files)
    song_ids: set[str] = set()
    row_count = 0
    valid_candidate_count = 0
    for asr_file in iter_with_progress(
        asr_files,
        total=len(asr_files),
        desc="Collect ASR rows",
    ):
        for record in iter_jsonl(asr_file):
            row_count += 1
            song_id = extract_song_id(record)
            if song_id is not None:
                if MAX_NUMS is None:
                    song_ids.add(song_id)
                    continue

                caption = record.get(CAPTION_FIELD)
                if not isinstance(caption, str) or not caption:
                    continue
                if not has_complete_separation(record):
                    continue
                song_ids.add(song_id)
                valid_candidate_count += 1
                if valid_candidate_count >= MAX_NUMS:
                    return song_ids, row_count
    return song_ids, row_count


def build_lyric_index(song_ids: set[str], lyric_files: Iterable[Path]) -> dict[str, LyricMatch]:
    if MAX_NUMS is not None and shutil.which("rg") is not None:
        return build_lyric_index_with_rg(song_ids)

    lyric_files = list(lyric_files)
    lyric_index: dict[str, LyricMatch] = {}
    for lyric_file in iter_with_progress(
        lyric_files,
        total=len(lyric_files),
        desc="Scan lyric files",
    ):
        for payload in iter_jsonl(lyric_file):
            song_id = payload.get("song_id")
            if song_id is None:
                continue
            song_id = str(song_id)
            if song_id not in song_ids:
                continue

            lyric_text = extract_lrc_lyric(payload)
            score = lyric_score(lyric_text)
            existing = lyric_index.get(song_id)
            if existing is None or score > existing.score:
                lyric_index[song_id] = LyricMatch(
                    lyric=lyric_text,
                    lyric_file_path=str(lyric_file),
                    score=score,
                )
    return lyric_index


def build_lyric_index_with_rg(song_ids: set[str]) -> dict[str, LyricMatch]:
    lyric_index: dict[str, LyricMatch] = {}
    sorted_song_ids = sorted(song_ids)
    for song_id in iter_with_progress(
        sorted_song_ids,
        total=len(sorted_song_ids),
        desc="Lookup lyrics",
    ):
        pattern = rf'"song_id"\s*:\s*"?{re.escape(song_id)}"?'
        result = subprocess.run(
            ["rg", "-n", "-g", LYRIC_GLOB, pattern, str(LYRIC_DIR)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            continue
        if result.returncode != 0:
            raise RuntimeError(
                f"rg failed while searching lyric for song_id={song_id}: {result.stderr.strip()}"
            )

        best_match: LyricMatch | None = None
        for match_line in result.stdout.splitlines():
            try:
                file_path, _line_number, payload_json = match_line.split(":", 2)
            except ValueError:
                continue
            payload = json.loads(payload_json)
            lyric_text = extract_lrc_lyric(payload)
            candidate = LyricMatch(
                lyric=lyric_text,
                lyric_file_path=file_path,
                score=lyric_score(lyric_text),
            )
            if best_match is None or candidate.score > best_match.score:
                best_match = candidate

        if best_match is not None:
            lyric_index[song_id] = best_match

    return lyric_index


def derive_separation_dir(source_audio_path: str) -> Path | None:
    for prefix in RAW_NETEASE_PREFIXES:
        if source_audio_path.startswith(prefix):
            relative_audio_path = Path(source_audio_path[len(prefix) :])
            return SEPARATION_ROOT / relative_audio_path.with_suffix("")
    return None


def path_or_none(path: Path) -> str | None:
    if path.exists():
        return str(path)
    return None


def has_complete_separation(record: dict) -> bool:
    source_audio_path = record.get("source_audio_path")
    if not isinstance(source_audio_path, str) or not source_audio_path:
        return False

    separation_dir = derive_separation_dir(source_audio_path)
    if separation_dir is None:
        return False

    return (
        (separation_dir / "mixture.mp3").exists()
        and (separation_dir / "vocals.mp3").exists()
        and (separation_dir / "accompaniment.mp3").exists()
    )


def write_output(asr_files: Iterable[Path], lyric_index: dict[str, LyricMatch]) -> dict[str, int]:
    asr_files = list(asr_files)
    stats = {
        "written_rows": 0,
        "missing_caption": 0,
        "missing_song_id": 0,
        "missing_lyric": 0,
        "missing_source_audio_path": 0,
        "skipped_missing_separation_dir": 0,
        "skipped_missing_separation_file": 0,
    }

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for asr_file in iter_with_progress(
            asr_files,
            total=len(asr_files),
            desc="Write output",
        ):
            if MAX_NUMS is not None and stats["written_rows"] >= MAX_NUMS:
                break
            for record in iter_jsonl(asr_file):
                if MAX_NUMS is not None and stats["written_rows"] >= MAX_NUMS:
                    break

                caption = record.get(CAPTION_FIELD)
                if not isinstance(caption, str) or not caption:
                    stats["missing_caption"] += 1
                    continue

                song_id = extract_song_id(record)
                if song_id is None:
                    stats["missing_song_id"] += 1
                    continue

                lyric_match = lyric_index.get(song_id)
                lyric = lyric_match.lyric if lyric_match is not None else None
                if lyric is None:
                    stats["missing_lyric"] += 1
                    continue

                source_audio_path = record.get("source_audio_path")
                if not isinstance(source_audio_path, str) or not source_audio_path:
                    stats["missing_source_audio_path"] += 1
                    separation_dir = None
                else:
                    separation_dir = derive_separation_dir(source_audio_path)

                if separation_dir is None:
                    stats["skipped_missing_separation_dir"] += 1
                    continue
                else:
                    mixture_file = separation_dir / "mixture.mp3"
                    vocals_file = separation_dir / "vocals.mp3"
                    accompaniment_file = separation_dir / "accompaniment.mp3"
                    mixture_path = path_or_none(mixture_file)
                    vocals_path = path_or_none(vocals_file)
                    accompaniment_path = path_or_none(accompaniment_file)
                    if (
                        mixture_path is None
                        or vocals_path is None
                        or accompaniment_path is None
                    ):
                        stats["skipped_missing_separation_file"] += 1
                        continue

                row = {
                    "caption": caption,
                    "lyric": lyric,
                    "mixture_path": mixture_path,
                    "vocals_path": vocals_path,
                    "accompaniment_path": accompaniment_path,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_rows"] += 1

    return stats


def main() -> None:
    asr_files = find_asr_files()
    if not asr_files:
        raise FileNotFoundError(f"No ASR files matched {RESULTS_ROOT / ASR_GLOB}")

    lyric_files = find_lyric_files()
    if not lyric_files:
        raise FileNotFoundError(f"No lyric files matched {LYRIC_DIR / LYRIC_GLOB}")

    song_ids, input_rows = collect_song_ids(asr_files)
    lyric_index = build_lyric_index(song_ids, lyric_files)
    stats = write_output(asr_files, lyric_index)

    print(f"asr_files={len(asr_files)}")
    print(f"lyric_files={len(lyric_files)}")
    print(f"max_nums={MAX_NUMS}")
    print(f"input_rows={input_rows}")
    print(f"unique_song_ids={len(song_ids)}")
    print(f"matched_lyric_song_ids={len(lyric_index)}")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output_jsonl={OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
