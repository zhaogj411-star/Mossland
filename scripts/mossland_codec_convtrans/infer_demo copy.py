from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.mossland_codec_convtrans.inference import EncoderDecoder  # noqa: E402


DEFAULT_CKPT = '/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/ckpt/mossland-codec-convtrans-rvq64-step2000'
DEFAULT_INPUT_AUDIO = os.path.join(REPO_ROOT, "tmp/data/许嵩 - 如约而至.mp3")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "tmp/mossland-codec-convtrans-infer-demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mossland ConvTrans codec continuous/RVQ demo inference."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hydra config for Lightning .ckpt. If omitted, tries ../../.hydra/config.yaml.",
    )
    parser.add_argument("--input-audio", default=DEFAULT_INPUT_AUDIO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task-id",
        default="reconstruct",
        choices=[
            "reconstruct",
            "separate_vocals",
            "separate_accompaniment",
            "super_resolution",
            "mono_to_stereo",
        ],
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--mode", choices=["parallel", "autoregressive"], default="parallel")
    parser.add_argument("--denoising-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav"])
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When loading Lightning checkpoints, prefer ema.ema_model weights if present.",
    )
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


def save_audio(path: str, audio: torch.Tensor, sample_rate: int, audio_format: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    torchaudio.save(path, audio.float().cpu(), sample_rate, format=audio_format)
    print(f"saved {path}")


def _default_lightning_config_path(ckpt_path: str) -> str | None:
    path = Path(ckpt_path)
    run_dir = path.parent.parent
    candidate = run_dir / ".hydra" / "config.yaml"
    return str(candidate) if candidate.exists() else None


def _strip_prefix_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    return {key[plen:]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_lightning_codec(args: argparse.Namespace) -> EncoderDecoder:
    config_path = args.config or _default_lightning_config_path(args.ckpt)
    if config_path is None:
        raise FileNotFoundError(
            "Lightning checkpoint needs --config, or a sibling run .hydra/config.yaml."
        )

    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model_state = _strip_prefix_state_dict(state_dict, "model.")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"loaded lightning model weights: missing={len(missing)} unexpected={len(unexpected)}"
    )

    if args.use_ema:
        ema_state = _strip_prefix_state_dict(state_dict, "ema.ema_model.")
        if ema_state:
            missing, unexpected = model.load_state_dict(ema_state, strict=False)
            print(
                f"loaded lightning EMA weights: missing={len(missing)} unexpected={len(unexpected)}"
            )

    codec = EncoderDecoder.__new__(EncoderDecoder)
    codec.device = torch.device(args.device)
    codec.load_path_inference = args.ckpt
    codec.model_kwargs = {}
    codec.gen = model.to(codec.device).eval()
    codec.latents_per_timestep = codec.gen.num_latents
    codec.bottleneck_channels = codec.gen.bottleneck_channels
    codec.max_batch_size_encode = codec.gen.max_batch_size_encode
    codec.max_batch_size_decode = codec.gen.max_batch_size_decode
    codec.sigma_rescale = codec.gen.sigma_rescale
    codec.past_spec = None
    codec.past_latents = None
    return codec


def load_codec(args: argparse.Namespace) -> EncoderDecoder:
    try:
        return EncoderDecoder(load_path_inference=args.ckpt, device=args.device)
    except Exception as exc:
        print(f"standard ckpt load failed, trying Lightning load: {exc}")
        return _load_lightning_codec(args)


def decode_and_save(
    codec: EncoderDecoder,
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
    save_audio(
        os.path.join(args.output_dir, f"{args.task_id}_{name}.{args.format}"),
        generated,
        codec.gen.sample_rate,
        args.format,
    )
    return generated


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    codec = load_codec(args)
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
        os.path.join(args.output_dir, f"{args.task_id}_source.{args.format}"),
        audio,
        codec.gen.sample_rate,
        args.format,
    )

    continuous_latents = codec.encode(audio, discrete=False, preprocess_on_gpu=True)
    continuous_latents = continuous_latents[:,3:-5]
    print_shape("continuous_latents", continuous_latents)
    decode_and_save(codec, "continuous", continuous_latents, args)

    if codec.gen.rvq is None:
        raise RuntimeError("rvq demo requires RVQ to be enabled")

    max_quantizers = int(codec.gen.rvq_num_quantizers)
    rates = [8, 16, 32] if max_quantizers <= 32 else [16, 32, 64, 128]
    rates = [rate for rate in rates if rate <= max_quantizers]
    initialized = codec.gen.rvq.initialized_codebook_count()
    if initialized > 0:
        rates = [rate for rate in rates if rate <= initialized]
    if not rates:
        print("RVQ codebooks are not initialized yet; skipping discrete demos.")
        return

    for n_quantizers in rates:
        codes = codec.encode(
            audio,
            discrete=True,
            preprocess_on_gpu=True,
            n_quantizers=n_quantizers,
        )
        codes = codes[:,3:-5]
        print_shape(f"rvq{n_quantizers}_codes", codes)
        decode_and_save(codec, f"rvq{n_quantizers}", codes, args)


if __name__ == "__main__":
    main()
