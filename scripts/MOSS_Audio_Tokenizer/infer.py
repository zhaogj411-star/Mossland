#!/usr/bin/env python3
"""Run MOSS-Audio-Tokenizer reconstruction for a single audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from scripts.MOSS_Audio_Tokenizer.modeling_moss_audio_tokenizer import (
    MossAudioTokenizerModel,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "OpenMOSS-Team" / "MOSS-Audio-Tokenizer-v2"
DEFAULT_INPUT = REPO_ROOT / "tmp" / "34078087" / "mixture.mp3"
DEFAULT_OUTPUT = REPO_ROOT / "tmp" / "34078087" / "mixture_reconstructed.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-quantizers", type=int, default=None)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.96,
        help="Streaming chunk length in seconds. Use 0 to disable chunked inference.",
    )
    parser.add_argument(
        "--codec-weight-dtype",
        choices=("fp32", "bf16"),
        default=None,
        help="Override checkpoint codec weight dtype. bf16 reduces GPU memory use.",
    )
    return parser.parse_args()


def load_audio(path: Path, target_sr: int, channels: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        raise ValueError(f"Input audio is empty: {path}")
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] == 1 and channels > 1:
        wav = wav.repeat(channels, 1)
    elif wav.shape[0] < channels:
        wav = torch.cat([wav, wav[-1:].repeat(channels - wav.shape[0], 1)], dim=0)
    else:
        wav = wav[:channels]
    return wav.clamp(-1.0, 1.0)


def save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float().clamp(-1.0, 1.0)
    torchaudio.save(str(path), wav, sample_rate=sample_rate)


def main() -> None:
    args = parse_args()
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input audio not found: {args.input}")

    load_kwargs = {"low_cpu_mem_usage": True}
    if args.codec_weight_dtype is not None:
        load_kwargs["codec_weight_dtype"] = args.codec_weight_dtype

    device = torch.device(args.device)
    model = MossAudioTokenizerModel.from_pretrained(args.checkpoint_dir, **load_kwargs)
    model.eval().to(device)

    wav = load_audio(args.input, model.sampling_rate, model.config.number_channels).to(device)
    chunk_duration = args.chunk_duration if args.chunk_duration and args.chunk_duration > 0 else None

    with torch.inference_mode():
        encoded = model.encode(
            wav.unsqueeze(0),
            num_quantizers=args.num_quantizers,
            return_dict=True,
            chunk_duration=chunk_duration,
        )
        decoded = model.decode(
            encoded.audio_codes,
            return_dict=True,
            chunk_duration=chunk_duration,
            num_quantizers=args.num_quantizers,
        )

    rec = decoded.audio.squeeze(0)[..., : wav.shape[-1]]
    save_audio(args.output, rec, model.sampling_rate)
    print(f"input: {args.input}")
    print(f"checkpoint: {args.checkpoint_dir}")
    print(f"codes: {tuple(encoded.audio_codes.shape)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
