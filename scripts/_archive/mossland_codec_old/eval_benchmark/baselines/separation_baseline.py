from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "demucs_htdemucs"
TARGET_STEM_BY_TASK = {
    "separate_vocals": "vocals",
    "separate_accompaniment": "no_vocals",
}


class SeparationGroupKey(NamedTuple):
    source_path: str
    start_seconds: float
    duration_seconds: float | None
    sample_rate: int
    seed: int


def _rounded_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def separation_group_key(item: EvalItem) -> SeparationGroupKey:
    return SeparationGroupKey(
        source_path=str(item.source_path.resolve()),
        start_seconds=float(_rounded_seconds(item.start_seconds) or 0.0),
        duration_seconds=_rounded_seconds(item.duration_seconds),
        sample_rate=item.sample_rate,
        seed=item.seed,
    )


def separation_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id in TARGET_STEM_BY_TASK]
    if max_items > 0:
        return items[:max_items]
    return items


def grouped_items(items: list[EvalItem]) -> list[list[EvalItem]]:
    groups: dict[SeparationGroupKey, list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[separation_group_key(item)].append(item)
    return [groups[key] for key in sorted(groups)]


def demucs_output_path(output_root: Path, model: str, input_stem: str, stem: str) -> Path:
    expected = output_root / model / input_stem / f"{stem}.wav"
    if expected.exists():
        return expected
    matches = sorted(output_root.glob(f"**/{stem}.wav"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Demucs did not write {stem!r} under {output_root}. "
        f"Expected {expected}; found {len(matches)} candidate files."
    )


def ensure_demucs_available() -> None:
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError(
            "Demucs is not installed in this Python environment. Install the official package with "
            "`python -m pip install demucs`, then rerun this baseline. The adapter calls "
            "`python -m demucs.separate --two-stems vocals` and expects the official "
            "HTDemucs-compatible CLI output layout."
        )


def run_demucs_cli(input_wav: Path, output_root: Path, args: argparse.Namespace) -> None:
    ensure_demucs_available()
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--name",
        args.model,
        "--out",
        str(output_root),
        "--device",
        args.device,
        "--two-stems",
        "vocals",
        "--shifts",
        str(args.shifts),
        "--overlap",
        str(args.overlap),
    ]
    if args.segment_seconds > 0:
        command.extend(["--segment", str(args.segment_seconds)])
    command.append(str(input_wav))
    subprocess.run(command, check=True)


def _safe_temp_stem(index: int, item: EvalItem) -> str:
    keep = []
    for char in item.item_id:
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    token = "".join(keep).strip("_") or "item"
    return f"{index:06d}_{token}"


def _group_prediction_paths(group: list[EvalItem], output_dir: str | Path) -> list[Path]:
    return [baseline_prediction_path(output_dir, BASELINE_NAME, item) for item in group]


def separate_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    groups = grouped_items(items)
    path_by_item_id: dict[tuple[str, str, int], Path] = {}
    temp_root = Path(args.temp_dir) if args.temp_dir else None
    total = len(groups)

    with tempfile.TemporaryDirectory(prefix="mossland_demucs_", dir=temp_root) as tmpdir:
        work_root = Path(tmpdir)
        for group_index, group in enumerate(groups, start=1):
            output_paths = _group_prediction_paths(group, args.output_dir)
            for item, output_path in zip(group, output_paths, strict=True):
                path_by_item_id[(item.item_id, item.task_id, item.seed)] = output_path
            if (
                all(path.exists() and path.stat().st_size > 0 for path in output_paths)
                and not args.overwrite
            ):
                maybe_print_progress(group_index, total, group[0].item_id, args.progress_every)
                continue

            first = group[0]
            input_stem = _safe_temp_stem(group_index, first)
            input_wav = work_root / "inputs" / f"{input_stem}.wav"
            audio = load_audio_segment(
                first.source_path,
                sample_rate=first.sample_rate,
                start_seconds=first.start_seconds,
                duration_seconds=first.duration_seconds,
                stereo=True,
            )
            save_audio(input_wav, audio, first.sample_rate)

            demucs_root = work_root / "demucs"
            if args.clean_demucs_output and demucs_root.exists():
                shutil.rmtree(demucs_root)
            run_demucs_cli(input_wav, demucs_root, args)

            for item, output_path in zip(group, output_paths, strict=True):
                stem = TARGET_STEM_BY_TASK[item.task_id]
                stem_path = demucs_output_path(demucs_root, args.model, input_stem, stem)
                prediction = load_audio_segment(stem_path, item.sample_rate, stereo=True)
                save_audio(output_path, prediction, item.sample_rate)
            maybe_print_progress(group_index, total, first.item_id, args.progress_every)

        if args.keep_temp:
            keep_path = Path(args.keep_temp)
            if keep_path.exists():
                shutil.rmtree(keep_path)
            shutil.copytree(work_root, keep_path)

    return [path_by_item_id[(item.item_id, item.task_id, item.seed)] for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official Demucs/HTDemucs source-separation baseline on MUSDB-style "
            "separation manifest rows."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Official Demucs model name passed to `python -m demucs.separate --name`.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device passed to the official Demucs CLI.",
    )
    parser.add_argument("--shifts", type=int, default=1, help="Demucs random shifts for test-time augmentation.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Demucs segment overlap.")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=0.0,
        help="Optional Demucs segment length. <=0 leaves the model default unchanged.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional parent directory for temporary clipped mixtures and raw Demucs outputs.",
    )
    parser.add_argument(
        "--keep-temp",
        default="",
        help="Optional directory to copy temporary Demucs inputs/outputs into for debugging.",
    )
    parser.add_argument(
        "--clean-demucs-output",
        action="store_true",
        help="Remove raw temporary Demucs output before each group; useful when debugging output discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = separation_manifest_items(args.manifest, args.max_items)
    prediction_paths = separate_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
