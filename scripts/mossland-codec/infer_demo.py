from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))

EncoderDecoder = importlib.import_module("scripts.mossland-codec.inference").EncoderDecoder


DEFAULT_CKPT = REPO_ROOT / "ckpt/mosslandcodec0617"
DEFAULT_INPUT_AUDIO = Path(
    "/inspire/sj-ssd3/project/embodied-multimodality/public/"
    "Sonata/data/source_seperation/NETEASE_SPIDER/audio/20260528/"
    "2030433496/mixture.mp3"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp/mossland-codec-infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mossland codec continuous/RVQ demo inference."
    )
    parser.add_argument("--ckpt", default=os.fspath(DEFAULT_CKPT))
    parser.add_argument("--input-audio", default=os.fspath(DEFAULT_INPUT_AUDIO))
    parser.add_argument("--output-dir", default=os.fspath(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--task-id",
        default="reconstruct",
        choices=[
            "reconstruct",
            "separate_vocals",
            "separate_drums",
            "separate_bass",
            "separate_other",
            "separate_accompaniment",
            "super_resolution",
            "mono_to_stereo",
        ],
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--mode", choices=["parallel", "autoregressive"], default="parallel")
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav"])
    return parser.parse_args()


def print_shape(name: str, x) -> None:
    print(f"{name}: {tuple(x.shape)}")


def load_audio(
    path: str,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    start = max(0, int(start_seconds * sr))
    end = start + int(duration_seconds * sr)
    audio = audio[:, start:end]
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int, audio_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    torchaudio.save(
        os.fspath(path),
        audio.float().cpu(),
        sample_rate,
        format=audio_format,
    )
    print(f"saved {path}")


def decode_and_save(
    codec,
    name: str,
    latents: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    generated = codec.decode(
        latents,
        mode=args.mode,
        denoising_steps=args.denoising_steps,
        preprocess_on_gpu=True,
        task_id=args.task_id,
    )
    print_shape(f"{name}_generated_audio", generated)
    suffix = args.format
    save_audio(
        Path(args.output_dir) / f"{args.task_id}_{name}.{suffix}",
        generated,
        codec.gen.sample_rate,
        args.format,
    )
    return generated


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    codec = EncoderDecoder(load_path_inference=args.ckpt, device=args.device)
    audio = load_audio(
        args.input_audio,
        codec.gen.sample_rate,
        args.start_seconds,
        args.duration_seconds,
    )

    print(f"checkpoint: {args.ckpt}")
    print(f"input: {args.input_audio}")
    print(f"output_dir: {args.output_dir}")
    print_shape("source_audio", audio)
    save_audio(
        Path(args.output_dir) / f"{args.task_id}_source.{args.format}",
        audio,
        codec.gen.sample_rate,
        args.format,
    )

    continuous_latents = codec.encode(audio, discrete=False, preprocess_on_gpu=True)
    print_shape("continuous_latents", continuous_latents)
    decode_and_save(codec, "continuous", continuous_latents, args)

    if codec.gen.bottleneck_type not in {"packed_rvq", "causal_rvq"}:
        raise RuntimeError(
            f"rvq demo requires bottleneck_type='packed_rvq' or 'causal_rvq', "
            f"got {codec.gen.bottleneck_type!r}"
        )

    rates = (8, 16, 32) if codec.gen.rvq_num_quantizers <= 32 else (16, 128)
    for n_quantizers in rates:
        codes = codec.encode(
            audio,
            discrete=True,
            preprocess_on_gpu=True,
            n_quantizers=n_quantizers,
        )
        print_shape(f"rvq{n_quantizers}_codes", codes)
        decode_and_save(codec, f"rvq{n_quantizers}", codes, args)


if __name__ == "__main__":
    main()
