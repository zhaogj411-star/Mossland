from __future__ import annotations

import importlib
import os
import sys

import torch
import torchaudio


REPO_ROOT = "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland"
sys.path.insert(0, REPO_ROOT)

EncoderDecoder = importlib.import_module("scripts.mossland-codec.inference").EncoderDecoder


CKPT_DIR = "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland/ckpt/mossland-codec0610"
INPUT_AUDIO = "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW/audio/20260526/1352955032/mixture.mp3"
OUTPUT_DIR = "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland/tmp/mossland-codec-infer"

# options: reconstruct, separate_vocals, separate_accompaniment, super_resolution, mono_to_stereo
TASK_ID = "reconstruct"
START_SECONDS = 0.0
DURATION_SECONDS = 6.0
USE_CONTINUOUS_LATENTS = True
MODE = "parallel"
DENOISING_STEPS = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_shape(name: str, x) -> None:
    print(f"{name}: {tuple(x.shape)}")


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    start = int(START_SECONDS * sr)
    end = start + int(DURATION_SECONDS * sr)
    audio = audio[:, start:end]
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    codec = EncoderDecoder(load_path_inference=CKPT_DIR, device=DEVICE)
    audio = load_audio(INPUT_AUDIO, codec.gen.sample_rate)

    continuous_latents = codec.encode(audio, discrete=False, preprocess_on_gpu=True)
    discrete_indexes = codec.encode(audio, discrete=True, preprocess_on_gpu=True)
    discrete_codes = codec.gen.fsq.indexes_to_codes(discrete_indexes)

    latents = continuous_latents if USE_CONTINUOUS_LATENTS else discrete_indexes
    generated = codec.decode(
        latents,
        mode=MODE,
        denoising_steps=DENOISING_STEPS,
        preprocess_on_gpu=True,
        task_id=TASK_ID,
    )

    print_shape("source_audio", audio)
    print_shape("continuous_latents", continuous_latents)
    print_shape("discrete_indexes", discrete_indexes)
    print_shape("discrete_codes", discrete_codes)
    print_shape("decode_latents", latents)
    print_shape("generated_audio", generated)

    torchaudio.save(os.path.join(OUTPUT_DIR, "source.wav"), audio.float().cpu(), codec.gen.sample_rate)
    torchaudio.save(
        os.path.join(OUTPUT_DIR, f"{TASK_ID}_generated.wav"),
        generated.float().cpu(),
        codec.gen.sample_rate,
    )
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
