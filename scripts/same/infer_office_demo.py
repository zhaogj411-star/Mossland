from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.factory import load_model


DEFAULT_CKPT_DIR = REPO_ROOT / "ckpt/SAME-L"
DEFAULT_OUTPUT_AUDIO = REPO_ROOT / "tmp/same-infer/same_l_reconstruct.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAME-L official checkpoint inference with scripts.same model code."
    )
    parser.add_argument("--ckpt-dir", default=os.fspath(DEFAULT_CKPT_DIR))
    parser.add_argument(
        "--input-audio",
        default="/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/source_seperation/NETEASE_SPIDER/audio/20260520/68484/mixture.mp3",
        help="Optional input audio. If omitted, a short stereo sine test signal is generated.",
    )
    parser.add_argument("--output-audio", default=os.fspath(DEFAULT_OUTPUT_AUDIO))
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32" if torch.cuda.is_available() else "fp32",
    )
    parser.add_argument("--save-source", action="store_true")
    parser.add_argument("--save-concat", action="store_true")
    parser.add_argument(
        "--keep-inference-noise",
        action="store_true",
        help="Keep SAME-L stochastic bottleneck/new-token noise enabled.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_audio(
    path: str | Path,
    sample_rate: int,
    channels: int,
    start_seconds: float,
    duration_seconds: float,
) -> torch.Tensor:
    audio, sr = torchaudio.load(os.fspath(path))
    start = max(0, int(start_seconds * sr))
    if duration_seconds > 0:
        end = start + int(duration_seconds * sr)
        audio = audio[:, start:end]
    else:
        audio = audio[:, start:]
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    if audio.shape[0] == 1 and channels == 2:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > channels:
        audio = audio[:channels]
    elif audio.shape[0] < channels:
        repeats = (channels + audio.shape[0] - 1) // audio.shape[0]
        audio = audio.repeat(repeats, 1)[:channels]
    return audio.unsqueeze(0).contiguous()


def make_synthetic_audio(
    sample_rate: int,
    channels: int,
    duration_seconds: float,
) -> torch.Tensor:
    duration_seconds = duration_seconds if duration_seconds > 0 else 4.0
    samples = max(1, int(sample_rate * duration_seconds))
    t = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    left = 0.25 * torch.sin(2 * torch.pi * 220.0 * t)
    right = 0.25 * torch.sin(2 * torch.pi * 330.0 * t)
    audio = torch.stack([left, right], dim=0)
    if channels == 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif channels > 2:
        repeats = (channels + 1) // 2
        audio = audio.repeat(repeats, 1)[:channels]
    return audio.unsqueeze(0).contiguous()


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    torchaudio.save(os.fspath(path), audio.float().cpu().clamp(-1, 1), sample_rate)
    print(f"saved {path}")


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    config_path = ckpt_dir / "config.yaml"
    checkpoint_path = ckpt_dir / "checkpoint.ckpt"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    model = load_model(ckpt_dir=os.fspath(ckpt_dir))
    model.eval().to(device)
    if not args.keep_inference_noise:
        model.disable_inference_noise()
    if dtype != torch.float32:
        model.to(dtype=dtype)

    if args.input_audio is None:
        audio = make_synthetic_audio(
            sample_rate=model.sample_rate,
            channels=model.audio_channels,
            duration_seconds=args.duration_seconds,
        )
        input_label = "synthetic stereo sine"
    else:
        audio = load_audio(
            args.input_audio,
            sample_rate=model.sample_rate,
            channels=model.audio_channels,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
        )
        input_label = args.input_audio
    audio = audio.to(device=device, dtype=dtype)

    print(f"checkpoint_dir: {ckpt_dir}")
    print(f"input_audio: {input_label}")
    print(f"output_audio: {args.output_audio}")
    print(f"device: {device}, dtype: {dtype}")
    print(f"source_audio: {tuple(audio.shape)} @ {model.sample_rate} Hz")

    with torch.inference_mode():
        output = model(audio)
    reconstructed = output.audio.float().cpu()
    source = audio.float().cpu()[..., : reconstructed.shape[-1]]

    print(f"latents: {tuple(output.latents.shape)}")
    print(f"reconstructed_audio: {tuple(reconstructed.shape)}")
    save_audio(args.output_audio, reconstructed, model.sample_rate)

    output_path = Path(args.output_audio)
    if args.save_source:
        save_audio(output_path.with_name(output_path.stem + "_source.wav"), source, model.sample_rate)
    if args.save_concat:
        silence = torch.zeros(
            source.shape[:-1] + (int(0.25 * model.sample_rate),), dtype=source.dtype
        )
        concat = torch.cat([source, silence, reconstructed], dim=-1)
        save_audio(output_path.with_name(output_path.stem + "_source_reconstruct.wav"), concat, model.sample_rate)


if __name__ == "__main__":
    main()
