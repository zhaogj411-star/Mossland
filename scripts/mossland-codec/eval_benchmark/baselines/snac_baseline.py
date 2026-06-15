from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.snac_baseline ..."
    ) from exc


BASELINE_NAME = "snac"
DEFAULT_MODEL_NAME = "hubertsiuzdak/snac_44khz"


def _load_snac_model(model_name: str, device: str | torch.device) -> Any:
    try:
        from snac import SNAC
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SNAC is not installed. Install the official package with "
            "`python -m pip install snac`, then rerun this baseline."
        ) from exc

    try:
        model = SNAC.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - network/checkpoint dependent
        raise RuntimeError(
            f"Failed to load SNAC model {model_name!r}. The official package downloads "
            "weights from Hugging Face on first use; check network access or pass "
            "--model-name pointing at a local SNAC checkpoint directory."
        ) from exc

    return model.to(device).eval()


def _model_sample_rate(model: Any) -> int:
    sample_rate = getattr(model, "sampling_rate", None) or getattr(model, "sample_rate", None)
    if sample_rate is None:
        raise RuntimeError("SNAC model does not expose sampling_rate or sample_rate.")
    return int(sample_rate)


def _downmix_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"audio must have shape [C,T], got {tuple(audio.shape)}")
    return audio.mean(dim=0, keepdim=True)


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def _to_audio_tensor(decoded: Any) -> torch.Tensor:
    audio = torch.as_tensor(decoded).detach().cpu().float()
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise RuntimeError(f"SNAC returned batched audio with shape {tuple(audio.shape)}.")
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise RuntimeError(f"SNAC returned audio with unexpected shape {tuple(audio.shape)}.")
    if audio.shape[0] > audio.shape[1] and audio.shape[1] == 1:
        audio = audio.T
    return audio[:1].contiguous()


def _reconstruct_item(model: Any, item: EvalItem, output_path: Path) -> None:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    model_sample_rate = _model_sample_rate(model)
    model_audio = _downmix_mono(audio)
    if item.sample_rate != model_sample_rate:
        model_audio = AF.resample(model_audio, item.sample_rate, model_sample_rate)

    device = next(model.parameters()).device
    model_input = model_audio.unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(model_input)
        decoded = model.decode(codes)

    prediction = _to_audio_tensor(decoded)
    prediction = _fit_length(prediction, model_audio.shape[-1])
    if model_sample_rate != item.sample_rate:
        prediction = AF.resample(prediction, model_sample_rate, item.sample_rate)
    prediction = _fit_length(prediction, audio.shape[-1])
    save_audio(output_path, prediction, item.sample_rate)


def snac_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "reconstruct"]
    if max_items > 0:
        return items[:max_items]
    return items


def reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = _load_snac_model(args.model_name, args.device)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        _reconstruct_item(model, item, output_path)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SNAC reconstruction baseline for Mossland eval manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SNAC Hugging Face repo id or local checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for SNAC inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = snac_manifest_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return

    prediction_paths = reconstruct_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
