from __future__ import annotations

import argparse
import sys
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
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.codicodec_baseline ..."
    ) from exc


BASELINE_NAME = "codicodec"
CODICODEC_SAMPLE_RATE = 48000


def _add_repo_dir(repo_dir: str | Path) -> None:
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise RuntimeError(
            f"CoDiCodec repository not found at {repo_path}. Clone the official repository "
            "or pass --repo-dir pointing at an installed source checkout."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _load_codicodec(args: argparse.Namespace) -> Any:
    _add_repo_dir(args.repo_dir)
    try:
        from codicodec import EncoderDecoder
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "CoDiCodec is not importable from --repo-dir. This adapter expects the official "
            "SonyCSLParis/codicodec repository, whose package exports EncoderDecoder."
        ) from exc

    try:
        return EncoderDecoder(device=torch.device(args.device))
    except Exception as exc:  # pragma: no cover - checkpoint/network dependent
        raise RuntimeError(
            "Failed to initialize official CoDiCodec EncoderDecoder. The official code expects "
            "`codicodec/models/codicodec.pt`; if it is missing, it downloads `codicodec.pt` "
            "from Hugging Face repo SonyCSLParis/codicodec via huggingface_hub."
        ) from exc


def _codec_input_path(item: EvalItem) -> Path | None:
    if item.task_id == "reconstruct":
        return item.source_path
    if item.reference_path is not None:
        return item.reference_path
    if item.task_id == "separate_vocals":
        return item.vocals_path
    if item.task_id == "separate_accompaniment":
        return item.accompaniment_path
    return None


def codicodec_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if _codec_input_path(item) is not None]
    if max_items > 0:
        return items[:max_items]
    return items


def _to_audio_tensor(decoded: Any) -> torch.Tensor:
    audio = torch.as_tensor(decoded).detach().cpu().float()
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise RuntimeError(f"CoDiCodec returned batched audio with shape {tuple(audio.shape)}.")
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise RuntimeError(f"CoDiCodec returned audio with unexpected shape {tuple(audio.shape)}.")
    if audio.shape[0] > audio.shape[1] and audio.shape[1] <= 2:
        audio = audio.T
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    return audio[:2].contiguous()


def _fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def _reconstruct_item(encdec: Any, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    input_path = _codec_input_path(item)
    if input_path is None:
        raise RuntimeError(f"item_id={item.item_id!r} is not a reconstruct/reference-audio item.")

    audio = load_audio_segment(
        input_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    codec_audio = audio
    if item.sample_rate != CODICODEC_SAMPLE_RATE:
        codec_audio = AF.resample(codec_audio, item.sample_rate, CODICODEC_SAMPLE_RATE)

    with torch.inference_mode():
        latent = encdec.encode(
            codec_audio,
            max_batch_size=args.max_batch_size_encode,
            discrete=args.discrete,
            preprocess_on_gpu=args.preprocess_on_gpu,
            desired_channels=args.desired_channels,
            fix_batch_size=args.fix_batch_size,
        )
        decoded = encdec.decode(
            latent,
            mode=args.decode_mode,
            max_batch_size=args.max_batch_size_decode,
            denoising_steps=args.denoising_steps,
            time_prompt=args.time_prompt,
            preprocess_on_gpu=args.preprocess_on_gpu,
        )

    prediction = _to_audio_tensor(decoded)
    prediction = _fit_length(prediction, codec_audio.shape[-1])
    if item.sample_rate != CODICODEC_SAMPLE_RATE:
        prediction = AF.resample(prediction, CODICODEC_SAMPLE_RATE, item.sample_rate)
    prediction = _fit_length(prediction, audio.shape[-1])
    save_audio(output_path, prediction, item.sample_rate)


def _run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    encdec = _load_codicodec(args)
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        _reconstruct_item(encdec, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official CoDiCodec reconstruction baseline for Mossland eval manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default="tmp/codicodec",
        help="Path to the cloned official SonyCSLParis/codicodec repository.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for CoDiCodec inference.",
    )
    parser.add_argument(
        "--max-batch-size-encode",
        type=int,
        default=None,
        help="Optional max_batch_size passed to EncoderDecoder.encode().",
    )
    parser.add_argument(
        "--max-batch-size-decode",
        type=int,
        default=None,
        help="Optional max_batch_size passed to EncoderDecoder.decode().",
    )
    parser.add_argument(
        "--decode-mode",
        default="parallel",
        choices=("parallel", "autoregressive"),
        help="CoDiCodec decode mode.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        help="Optional denoising_steps passed to EncoderDecoder.decode().",
    )
    parser.add_argument(
        "--time-prompt",
        type=float,
        default=None,
        help="Optional time_prompt passed to EncoderDecoder.decode().",
    )
    parser.add_argument(
        "--desired-channels",
        type=int,
        default=64,
        help="Continuous latent channel dimension passed to EncoderDecoder.encode().",
    )
    parser.add_argument("--discrete", action="store_true", help="Use CoDiCodec discrete tokens instead of continuous latents.")
    parser.add_argument("--fix-batch-size", action="store_true", help="Pass fix_batch_size=True to EncoderDecoder.encode().")
    parser.add_argument(
        "--preprocess-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CoDiCodec STFT/iSTFT preprocessing on the inference device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = codicodec_manifest_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return

    prediction_paths = _run_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
