from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torchaudio

from .manifest import write_jsonl


STEM_NAMES = ("mixture", "vocals", "drums", "bass", "other")


def find_tracks(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"missing MUSDB split directory: {split_dir}")
    tracks = []
    for track_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        if all((track_dir / f"{stem}.wav").exists() for stem in STEM_NAMES):
            tracks.append(track_dir)
    return tracks


def _match_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def write_accompaniment(track_dir: Path, output_root: Path, musdb_root: Path) -> Path:
    rel = track_dir.relative_to(musdb_root)
    output_path = output_root / rel / "accompaniment.wav"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path.resolve()
    stems = []
    sample_rate = None
    for stem in ("drums", "bass", "other"):
        audio, sr = torchaudio.load(str(track_dir / f"{stem}.wav"))
        sample_rate = sr if sample_rate is None else sample_rate
        if sr != sample_rate:
            raise ValueError(f"sample rate mismatch in {track_dir}")
        stems.append(audio)
    length = max(stem.shape[-1] for stem in stems)
    accompaniment = sum(_match_length(stem, length) for stem in stems).clamp(-1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), accompaniment, sample_rate)
    return output_path.resolve()


def _load_window_rms(path: Path, start_seconds: float, duration_seconds: float) -> float:
    info = torchaudio.info(str(path))
    frame_offset = max(0, int(round(start_seconds * info.sample_rate)))
    num_frames = max(1, int(round(duration_seconds * info.sample_rate)))
    audio, _ = torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames)
    if audio.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(audio.float().square())).item())


def _load_window_min_frame_rms(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
    frame_seconds: float,
) -> float:
    info = torchaudio.info(str(path))
    frame_offset = max(0, int(round(start_seconds * info.sample_rate)))
    num_frames = max(1, int(round(duration_seconds * info.sample_rate)))
    frame_size = max(1, int(round(frame_seconds * info.sample_rate)))
    audio, _ = torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames)
    if audio.numel() == 0:
        return 0.0
    rms_values = []
    for frame_start in range(0, audio.shape[-1], frame_size):
        frame = audio[..., frame_start : frame_start + frame_size]
        if frame.shape[-1] < frame_size:
            break
        rms_values.append(torch.sqrt(torch.mean(frame.float().square())))
    if not rms_values:
        return 0.0
    return float(torch.stack(rms_values).min().item())


def _track_duration_seconds(path: Path) -> float:
    info = torchaudio.info(str(path))
    if info.sample_rate <= 0:
        raise ValueError(f"invalid sample rate for {path}: {info.sample_rate}")
    return info.num_frames / info.sample_rate


def select_non_silent_start(
    vocals_path: Path,
    accompaniment_path: Path,
    duration_seconds: float,
    hop_seconds: float,
    min_source_rms: float,
    frame_seconds: float,
) -> float | None:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive when selecting non-silent windows")
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive")

    max_duration = min(_track_duration_seconds(vocals_path), _track_duration_seconds(accompaniment_path))
    if max_duration < duration_seconds:
        return None
    max_start = max_duration - duration_seconds
    steps = int(math.floor(max_start / hop_seconds)) + 1
    candidate_starts = [step * hop_seconds for step in range(steps)]
    if not candidate_starts or candidate_starts[-1] < max_start:
        candidate_starts.append(max_start)

    for start in candidate_starts:
        vocals_rms = _load_window_min_frame_rms(
            vocals_path,
            start,
            duration_seconds,
            frame_seconds=frame_seconds,
        )
        if vocals_rms < min_source_rms:
            continue
        accompaniment_rms = _load_window_min_frame_rms(
            accompaniment_path,
            start,
            duration_seconds,
            frame_seconds=frame_seconds,
        )
        if accompaniment_rms >= min_source_rms:
            return float(start)
    return None


def build_rows(
    root: Path,
    split: str,
    sample_rate: int,
    max_tracks: int,
    duration_seconds: float | None,
    accompaniment_root: Path,
    include_vocals: bool = True,
    include_drums: bool = False,
    include_bass: bool = False,
    include_other: bool = False,
    include_accompaniment: bool = True,
    select_non_silent: bool = False,
    window_hop_seconds: float = 5.0,
    min_source_rms: float = 1e-5,
    non_silent_frame_seconds: float = 1.0,
) -> list[dict]:
    tracks = find_tracks(root, split)
    if max_tracks > 0:
        tracks = tracks[:max_tracks]
    rows = []
    for track_dir in tracks:
        mixture = (track_dir / "mixture.wav").resolve()
        vocals = (track_dir / "vocals.wav").resolve()
        drums = (track_dir / "drums.wav").resolve()
        bass = (track_dir / "bass.wav").resolve()
        other = (track_dir / "other.wav").resolve()
        accompaniment = write_accompaniment(track_dir, accompaniment_root, root)
        start_seconds = 0.0
        selection_metadata = {
            "window_selection": "start_0",
        }
        if select_non_silent:
            if duration_seconds is None:
                raise ValueError("--select-non-silent requires --duration-seconds")
            selected_start = select_non_silent_start(
                vocals,
                accompaniment,
                duration_seconds=duration_seconds,
                hop_seconds=window_hop_seconds,
                min_source_rms=min_source_rms,
                frame_seconds=non_silent_frame_seconds,
            )
            if selected_start is None:
                continue
            start_seconds = selected_start
            selection_metadata = {
                "window_selection": "first_non_silent_vocals_and_accompaniment",
                "window_hop_seconds": window_hop_seconds,
                "min_source_rms": min_source_rms,
                "non_silent_frame_seconds": non_silent_frame_seconds,
            }
        common = {
            "source_path": str(mixture),
            "mixture_path": str(mixture),
            "vocals_path": str(vocals),
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "sample_rate": sample_rate,
            "metadata": {
                "dataset": "MUSDB18-HQ" if root.name.lower().endswith("hq") else "MUSDB18",
                "split": split,
                "track": track_dir.name,
                "drums_path": str(drums),
                "bass_path": str(bass),
                "other_path": str(other),
                **selection_metadata,
            },
        }
        track_id = track_dir.name.replace(" ", "_")
        if include_vocals:
            rows.append(
                {
                    **common,
                    "item_id": f"musdb__{split}__{track_id}__vocals",
                    "task_id": "separate_vocals",
                    "reference_path": str(vocals),
                }
            )
        for stem_name, include, reference in (
            ("drums", include_drums, drums),
            ("bass", include_bass, bass),
            ("other", include_other, other),
        ):
            if include:
                rows.append(
                    {
                        **common,
                        "item_id": f"musdb__{split}__{track_id}__{stem_name}",
                        "task_id": f"separate_{stem_name}",
                        "reference_path": str(reference),
                    }
                )
        if include_accompaniment:
            rows.append(
                {
                    **common,
                    "item_id": f"musdb__{split}__{track_id}__accompaniment",
                    "task_id": "separate_accompaniment",
                    "reference_path": str(accompaniment),
                    "accompaniment_path": str(accompaniment),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MUSDB18/MUSDB18-HQ separation eval manifest.")
    parser.add_argument("--musdb-root", required=True, help="Unzipped MUSDB root containing train/test directories.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest.")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument(
        "--select-non-silent",
        action="store_true",
        help="Choose the first window where both vocals and accompaniment exceed --min-source-rms.",
    )
    parser.add_argument(
        "--window-hop-seconds",
        type=float,
        default=5.0,
        help="Hop size for --select-non-silent window scanning.",
    )
    parser.add_argument(
        "--min-source-rms",
        type=float,
        default=1e-5,
        help="Minimum per-frame RMS for each reference source when selecting non-silent windows.",
    )
    parser.add_argument(
        "--non-silent-frame-seconds",
        type=float,
        default=1.0,
        help="Frame size used by --select-non-silent; default matches museval --win-seconds.",
    )
    parser.add_argument(
        "--accompaniment-root",
        default="scripts/mossland-codec/eval_benchmark/data/musdb_derived_accompaniment",
    )
    parser.add_argument("--no-vocals", action="store_true")
    parser.add_argument("--include-drums", action="store_true")
    parser.add_argument("--include-bass", action="store_true")
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--no-accompaniment", action="store_true")
    args = parser.parse_args()

    rows = build_rows(
        root=Path(args.musdb_root),
        split=args.split,
        sample_rate=args.sample_rate,
        max_tracks=args.max_tracks,
        duration_seconds=args.duration_seconds,
        accompaniment_root=Path(args.accompaniment_root),
        include_vocals=not args.no_vocals,
        include_drums=args.include_drums,
        include_bass=args.include_bass,
        include_other=args.include_other,
        include_accompaniment=not args.no_accompaniment,
        select_non_silent=args.select_non_silent,
        window_hop_seconds=args.window_hop_seconds,
        min_source_rms=args.min_source_rms,
        non_silent_frame_seconds=args.non_silent_frame_seconds,
    )
    write_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
