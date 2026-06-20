from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torchaudio

from .manifest import write_jsonl


DEFAULT_TASKS = (
    "reconstruct",
    "separate_vocals",
    "separate_accompaniment",
    "super_resolution",
    "mono_to_stereo",
)
DEFAULT_SR_RATES = (8000, 12000, 16000, 24000, 32000, 40000)


def find_prepared_items(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.rglob("mixture.mp3")
        if (path.parent / "vocals.mp3").exists()
        and (path.parent / "accompaniment.mp3").exists()
    )


def audio_duration_seconds(path: Path) -> float:
    info = torchaudio.info(str(path))
    if info.sample_rate <= 0:
        return 0.0
    return float(info.num_frames) / float(info.sample_rate)


def choose_start(duration: float, clip_duration: float, rng: random.Random) -> float:
    if duration <= clip_duration:
        return 0.0
    return rng.uniform(0.0, duration - clip_duration)


def row_id(item_dir: Path, root: Path) -> str:
    rel = item_dir.relative_to(root).as_posix()
    if rel == ".":
        rel = item_dir.name
    return rel.replace("/", "__")


def build_rows(
    root: Path,
    max_items: int,
    clip_duration: float,
    sample_rate: int,
    tasks: tuple[str, ...],
    sr_rates: tuple[int, ...],
    stereo_seeds: tuple[int, ...],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    item_dirs = find_prepared_items(root)
    rng.shuffle(item_dirs)
    if max_items > 0:
        item_dirs = item_dirs[:max_items]

    rows = []
    for item_dir in item_dirs:
        mixture = item_dir / "mixture.mp3"
        vocals = item_dir / "vocals.mp3"
        accompaniment = item_dir / "accompaniment.mp3"
        try:
            duration = audio_duration_seconds(mixture)
        except Exception as exc:
            print(f"skip unreadable prepared item {item_dir}: {exc}", file=sys.stderr)
            continue
        start = choose_start(duration, clip_duration, rng)
        base = {
            "source_path": str(mixture.resolve()),
            "reference_path": str(mixture.resolve()),
            "mixture_path": str(mixture.resolve()),
            "vocals_path": str(vocals.resolve()),
            "accompaniment_path": str(accompaniment.resolve()),
            "start_seconds": round(start, 6),
            "duration_seconds": min(clip_duration, duration) if duration > 0 else clip_duration,
            "sample_rate": sample_rate,
        }
        base_id = row_id(item_dir, root)
        if "reconstruct" in tasks:
            rows.append({"item_id": f"{base_id}__reconstruct", "task_id": "reconstruct", **base})
        if "separate_vocals" in tasks:
            rows.append(
                {
                    **base,
                    "item_id": f"{base_id}__separate_vocals",
                    "task_id": "separate_vocals",
                    "reference_path": str(vocals.resolve()),
                }
            )
        if "separate_accompaniment" in tasks:
            rows.append(
                {
                    **base,
                    "item_id": f"{base_id}__separate_accompaniment",
                    "task_id": "separate_accompaniment",
                    "reference_path": str(accompaniment.resolve()),
                }
            )
        if "super_resolution" in tasks:
            for rate in sr_rates:
                rows.append(
                    {
                        "item_id": f"{base_id}__super_resolution_{rate}",
                        "task_id": "super_resolution",
                        "low_sample_rate": rate,
                        **base,
                    }
                )
        if "mono_to_stereo" in tasks:
            for stereo_seed in stereo_seeds:
                rows.append(
                    {
                        "item_id": f"{base_id}__mono_to_stereo_seed{stereo_seed}",
                        "task_id": "mono_to_stereo",
                        "seed": stereo_seed,
                        **base,
                    }
                )
    return rows


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_task_tuple(value: str) -> tuple[str, ...]:
    if not value:
        return DEFAULT_TASKS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed Mossland eval manifest from prepared stems.")
    parser.add_argument("--prepared-root", required=True, help="Root containing mixture/vocals/accompaniment mp3 files.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--max-items", type=int, default=50, help="Maximum prepared items to sample; <=0 means all.")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Clip duration per item.")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--sr-rates", default=",".join(str(rate) for rate in DEFAULT_SR_RATES))
    parser.add_argument("--stereo-seeds", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(
        root=Path(args.prepared_root),
        max_items=args.max_items,
        clip_duration=args.duration_seconds,
        sample_rate=args.sample_rate,
        tasks=parse_task_tuple(args.tasks),
        sr_rates=parse_int_tuple(args.sr_rates),
        stereo_seeds=parse_int_tuple(args.stereo_seeds),
        seed=args.seed,
    )
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
