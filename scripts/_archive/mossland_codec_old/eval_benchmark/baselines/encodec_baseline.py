from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.encodec_baseline ..."
    ) from exc


BASELINE_NAME = "encodec"


def load_encodec_model(device: str | torch.device, bandwidth: float):
    try:
        from encodec import EncodecModel
    except ImportError as exc:
        raise RuntimeError(
            "Facebook Research EnCodec is not installed. Install the official package with "
            "`python -m pip install encodec`, then rerun this baseline."
        ) from exc

    try:
        model = EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(float(bandwidth))
        return model.to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the official Facebook Research EnCodec 48 kHz model. "
            "This may require network access for the first checkpoint download into "
            "the default torch cache, and the requested --bandwidth must be supported "
            "by the model."
        ) from exc


def convert_audio_for_model(audio: torch.Tensor, source_sample_rate: int, model) -> torch.Tensor:
    try:
        from encodec.utils import convert_audio
    except ImportError as exc:
        raise RuntimeError(
            "Installed `encodec` package does not expose `encodec.utils.convert_audio`; "
            "please use the official Facebook Research EnCodec package."
        ) from exc

    return convert_audio(audio, source_sample_rate, model.sample_rate, model.channels)


def pad_for_encodec_segments(audio: torch.Tensor, model) -> tuple[torch.Tensor, int]:
    """Pad audio so EnCodec's overlap-add decoder sees only full segments."""
    original_length = audio.shape[-1]
    segment_length = getattr(model, "segment_length", None)
    segment_stride = getattr(model, "segment_stride", None)
    if segment_length is None or segment_stride is None:
        return audio, original_length
    if original_length <= segment_length:
        target_length = segment_length
    else:
        remainder = (original_length - segment_length) % segment_stride
        target_length = original_length if remainder == 0 else original_length + segment_stride - remainder
    if target_length > original_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - original_length))
    return audio, original_length


def reconstruct_items(items: list[EvalItem], output_dir: str | Path, model, overwrite: bool) -> list[Path]:
    prediction_paths: list[Path] = []
    device = next(model.parameters()).device

    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
            maybe_print_progress(index, total, item.item_id, getattr(model, "progress_every", 100))
            continue

        audio = load_audio_segment(
            item.source_path,
            sample_rate=item.sample_rate,
            start_seconds=item.start_seconds,
            duration_seconds=item.duration_seconds,
            stereo=True,
        )
        model_audio = convert_audio_for_model(audio, item.sample_rate, model)
        model_audio, model_audio_length = pad_for_encodec_segments(model_audio, model)
        model_audio = model_audio.unsqueeze(0).to(device)
        with torch.inference_mode():
            encoded_frames = model.encode(model_audio)
            decoded = model.decode(encoded_frames).squeeze(0).detach().cpu()
        decoded = decoded[..., :model_audio_length]

        if model.sample_rate != item.sample_rate:
            decoded = AF.resample(decoded, model.sample_rate, item.sample_rate)
        decoded = decoded[..., : audio.shape[-1]]
        save_audio(output_path, decoded, item.sample_rate)
        maybe_print_progress(index, total, item.item_id, getattr(model, "progress_every", 100))

    return prediction_paths


def reconstruct_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    if max_items > 0:
        return items[:max_items]
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official Facebook Research EnCodec reconstruct baseline."
    )
    add_common_args(parser)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for EnCodec inference.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=24.0,
        help="Target EnCodec bandwidth in kbps. The official 48 kHz model supports 3, 6, 12, and 24.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = reconstruct_manifest_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return

    device = torch.device(args.device)
    model = load_encodec_model(device=device, bandwidth=args.bandwidth)
    model.progress_every = args.progress_every
    prediction_paths = reconstruct_items(items, args.output_dir, model=model, overwrite=args.overwrite)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
