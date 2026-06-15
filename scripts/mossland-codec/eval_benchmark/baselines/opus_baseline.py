from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "opus"


def opus_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    if max_items > 0:
        return items[:max_items]
    return items


def ffmpeg_encode_decode_command(
    input_path: Path,
    encoded_path: Path,
    output_path: Path,
    bitrate_kbps: float,
    sample_rate: int,
) -> list[list[str]]:
    bitrate = f"{bitrate_kbps:g}k"
    encode = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-application",
        "audio",
        str(encoded_path),
    ]
    decode = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(encoded_path),
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    return [encode, decode]


def run_ffmpeg_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def reconstruct_item(item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    with tempfile.TemporaryDirectory(prefix="mossland_opus_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / f"{item.item_id}_input.wav"
        encoded_path = tmpdir_path / f"{item.item_id}_{args.bitrate_kbps:g}kbps.opus"
        decoded_path = tmpdir_path / f"{item.item_id}_decoded.wav"
        save_audio(input_path, audio, item.sample_rate)
        for command in ffmpeg_encode_decode_command(
            input_path,
            encoded_path,
            decoded_path,
            args.bitrate_kbps,
            item.sample_rate,
        ):
            run_ffmpeg_command(command)
        reconstructed = load_audio_segment(decoded_path, sample_rate=item.sample_rate, stereo=True)
        reconstructed = reconstructed[..., : audio.shape[-1]]
        save_audio(output_path, reconstructed, item.sample_rate)


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    prediction_paths = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        reconstruct_item(item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an FFmpeg/libopus reconstruction baseline.")
    add_common_args(parser)
    parser.add_argument(
        "--bitrate-kbps",
        type=float,
        required=True,
        help="Target Opus audio bitrate in kbps, for example 8, 14, or 24.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = opus_manifest_items(args.manifest, args.max_items)
    prediction_paths = reconstruct_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
