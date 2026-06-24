from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "flashsr"
FLASHSR_INPUT_SAMPLE_RATE = 16000
FLASHSR_OUTPUT_SAMPLE_RATE = 48000
OFFICIAL_REPO = "https://github.com/ysharma3501/FlashSR"
HF_REPO_ID = "YatharthS/FlashSR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_source_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/flashsr/FlashSR"


def _add_source_dir(source_dir: str | Path) -> None:
    source = Path(source_dir)
    if not source.exists():
        raise RuntimeError(
            f"FlashSR source tree is missing: {source}. Clone {OFFICIAL_REPO} there, "
            "or pass --source-dir."
        )
    source_string = str(source.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="upsampler.pth",
            local_dir=str(Path(args.cache_dir)),
        )
    )


def load_flashsr(args: argparse.Namespace):
    _add_source_dir(args.source_dir)
    try:
        from FastAudioSR import FASR
    except Exception as exc:
        raise RuntimeError("Failed to import FlashSR FastAudioSR from the cloned source tree.") from exc
    model = FASR(str(resolve_checkpoint(args)))
    if args.half:
        model.model.half()
    return model


def _low_band_audio(item: EvalItem) -> torch.Tensor:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    low_rate = item.low_sample_rate or FLASHSR_INPUT_SAMPLE_RATE
    if low_rate != item.sample_rate:
        audio = AF.resample(audio, item.sample_rate, low_rate)
        audio = AF.resample(audio, low_rate, FLASHSR_INPUT_SAMPLE_RATE)
    elif item.sample_rate != FLASHSR_INPUT_SAMPLE_RATE:
        audio = AF.resample(audio, item.sample_rate, FLASHSR_INPUT_SAMPLE_RATE)
    return audio.clamp(-1.0, 1.0)


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def _run_item(model, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    low_band = _low_band_audio(item)
    dtype = torch.float16 if args.half else torch.float32
    with torch.inference_mode():
        prediction = model.run(low_band.to(dtype=dtype))
    prediction = torch.as_tensor(prediction).detach().cpu().float()
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim == 0:
        raise RuntimeError(f"FlashSR returned scalar output for item_id={item.item_id!r}.")
    if FLASHSR_OUTPUT_SAMPLE_RATE != item.sample_rate:
        prediction = AF.resample(prediction, FLASHSR_OUTPUT_SAMPLE_RATE, item.sample_rate)
    expected = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    ).shape[-1]
    prediction = _fit_length(prediction[:1], expected)
    save_audio(output_path, prediction, item.sample_rate)


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = load_flashsr(args)
    prediction_paths = []
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            _run_item(model, item, output_path, args)
        maybe_print_progress(index, len(items), item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlashSR on super-resolution eval manifest rows.")
    add_common_args(parser)
    parser.add_argument("--source-dir", default=str(default_source_dir()))
    parser.add_argument("--checkpoint", default="", help="Path to upsampler.pth; empty downloads from HF.")
    parser.add_argument("--cache-dir", default="tmp/eval_baseline_refs/flashsr/checkpoints")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = super_resolution_items(args.manifest, args.max_items)
    prediction_paths = run_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
