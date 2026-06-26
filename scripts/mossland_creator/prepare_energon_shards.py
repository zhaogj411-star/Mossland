# Copyright (c) 2025.
"""Offline data-prep: prepared jsonl -> energon/WebDataset shards.

Recommended launch modes:

1. Single process:
   `python -m scripts.mossland_creator.prepare_energon_shards --out <out_dir>`

2. Direct 8-GPU launch on one node (copy-paste from repo root):
   ```bash
   OUT_DIR=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/tmp/data/training_data/mossland-create-exp
   NUM_WORKERS=8
   mkdir -p "$OUT_DIR"
   for i in $(seq 0 $((NUM_WORKERS - 1))); do
     python -m scripts.mossland_creator.prepare_energon_shards \
       --out "$OUT_DIR" \
       --num-workers "$NUM_WORKERS" \
       --worker-id "$i" \
       --device "cuda:$i" \
       > "$OUT_DIR/worker_${i}.log" 2>&1 &
   done
   wait
   ```

3. Manual per-worker launch on one node:
   worker 0: `python -m scripts.mossland_creator.prepare_energon_shards --out <out_dir> --num-workers 8 --worker-id 0 --device cuda:0`
   worker 1: `python -m scripts.mossland_creator.prepare_energon_shards --out <out_dir> --num-workers 8 --worker-id 1 --device cuda:1`
   ...
   worker 7: `python -m scripts.mossland_creator.prepare_energon_shards --out <out_dir> --num-workers 8 --worker-id 7 --device cuda:7`

4. Torch launcher:
   ```bash
   OUT_DIR=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/tmp/data/training_data/mossland-create-exp
   NPROC_PER_NODE=8
   mkdir -p "$OUT_DIR/logs"
   python -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
     --log-dir "$OUT_DIR/logs" \
     -r 3 \
     -m scripts.mossland_creator.prepare_energon_shards \
     --out "$OUT_DIR" \
     --device cuda --num-workers 1 --worker-id 0
   ```
   If the launcher injects `LOCAL_RANK/RANK/WORLD_SIZE`, leaving
   `--num-workers 1 --worker-id 0` at their defaults will auto-detect them.

Input rows come from `tmp/prepare_data/netease_spider_caption_lyric_paths.jsonl`
and contain:

- `caption`
- `lyric`
- `mixture_path`
- `vocals_path`
- `accompaniment_path`

For each row, the script:

1. builds a text prompt as `caption: ... lyric: ...`
2. loads `mixture/vocals/accompaniment`
3. resamples/channel-fixes them for the selected codec
4. trims all three waveforms to the same shortest sample length
5. encodes the three aligned waveforms into latents
6. trims latent lengths again to the shortest latent frame count
7. writes one shard sample containing the 3-way latent dict, text embedding,
   and metadata json
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch

from .codecs import build_codec
from .conditioners import build_conditioner
from .data.parquet_dataset import _load_audio

DEFAULT_INPUT_JSONL = (
    Path(__file__).resolve().parents[2]
    / "tmp"
    / "prepare_data"
    / "netease_spider_caption_lyric_paths.jsonl"
)
STEM_KEYS = ("mixture", "vocals", "accompaniment")


def _tar_add(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json in {path}:{line_number}") from exc


def _format_prompt(row: dict) -> str:
    caption = str(row.get("caption") or "").strip()
    lyric = str(row.get("lyric") or "").strip()
    return f"caption: {caption} lyric: {lyric}"


def _load_aligned_waveforms(row: dict, sample_rate: int, audio_channels: int) -> tuple[dict[str, torch.Tensor], int]:
    waveforms: dict[str, torch.Tensor] = {}
    for stem in STEM_KEYS:
        stem_path = row.get(f"{stem}_path")
        if not isinstance(stem_path, str) or not stem_path:
            raise ValueError(f"missing {stem}_path")
        waveforms[stem] = _load_audio(stem_path, sample_rate, audio_channels)

    min_samples = min(waveform.shape[-1] for waveform in waveforms.values())
    if min_samples <= 0:
        raise ValueError("aligned waveform length is zero")
    aligned = {
        stem: waveform[:, :min_samples].contiguous()
        for stem, waveform in waveforms.items()
    }
    return aligned, min_samples


def _encode_aligned_latents(codec, waveforms: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    latents = {stem: codec.encode(waveform).cpu() for stem, waveform in waveforms.items()}
    min_frames = min(latent.shape[-1] for latent in latents.values())
    if min_frames < 4:
        raise ValueError(f"latent too short after alignment: {min_frames}")
    aligned = {
        stem: latent[:, :min_frames].contiguous()
        for stem, latent in latents.items()
    }
    return aligned, min_frames


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _resolve_worker_context(args) -> tuple[int, int, int, str]:
    env_world_size = _get_env_int("WORLD_SIZE", 1)
    env_rank = _get_env_int("RANK", 0)
    env_local_rank = _get_env_int("LOCAL_RANK", env_rank)

    use_torchrun_env = (
        "LOCAL_RANK" in os.environ
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and env_world_size > 1
        and args.num_workers == 1
        and args.worker_id == 0
    )
    if use_torchrun_env:
        world_size = env_world_size
        rank = env_rank
        local_rank = env_local_rank
    else:
        world_size = args.num_workers
        rank = args.worker_id
        local_rank = rank

    if world_size < 1:
        raise ValueError(f"invalid num_workers={world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"worker_id/rank out of range: rank={rank}, world_size={world_size}")

    device = args.device
    if device == "cuda":
        device = f"cuda:{local_rank}"
    return world_size, rank, local_rank, device


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _cleanup_worker_memory(device: str, aggressive: bool = False) -> None:
    gc.collect()
    if aggressive and _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="prepared jsonl with caption/lyric and aligned stem paths",
    )
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--codec", default="same")
    ap.add_argument("--codec-kwargs", default="{}")
    ap.add_argument("--text", default="umt5")
    ap.add_argument("--text-kwargs", default="{}")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="print one progress log every N worker-local rows; <=0 disables periodic progress logs",
    )
    ap.add_argument(
        "--cuda-empty-cache-every",
        type=int,
        default=100,
        help="call torch.cuda.empty_cache every N processed rows on CUDA; <=0 disables periodic empty_cache",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="split rows across N workers; torchrun env auto-detected when left at default",
    )
    ap.add_argument("--worker-id", type=int, default=0, help="worker index in [0, num_workers)")
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl)
    os.makedirs(args.out, exist_ok=True)

    world_size, rank, local_rank, device = _resolve_worker_context(args)
    worker_prefix = f"[worker {rank}/{world_size} local_rank={local_rank} device={device}]"
    print(f"{worker_prefix} start input={input_jsonl} out={args.out}")

    codec = build_codec(args.codec, **json.loads(args.codec_kwargs)).to(device)
    conditioner = build_conditioner(args.text, **json.loads(args.text_kwargs)).to(device)

    processed = written = skipped = shard_idx = 0
    tar = None

    def open_shard(i):
        if world_size == 1:
            filename = f"shard_{i:06d}.tar"
        else:
            filename = f"shard_rank{rank:02d}_{i:06d}.tar"
        path = os.path.join(args.out, filename)
        return tarfile.open(path, "w")

    for row_index, (line_number, row) in enumerate(_iter_jsonl(input_jsonl)):
        if args.limit and row_index >= args.limit:
            break
        if row_index % world_size != rank:
            continue
        processed += 1
        if written % args.shard_size == 0:
            if tar is not None:
                tar.close()
            tar = open_shard(shard_idx)
            shard_idx += 1
        prompt = None
        waveforms = None
        latents = None
        emb = None
        buf = None
        try:
            with torch.inference_mode():
                prompt = _format_prompt(row)
                waveforms, aligned_samples = _load_aligned_waveforms(
                    row,
                    codec.sample_rate,
                    codec.audio_channels,
                )
                latents, latent_frames = _encode_aligned_latents(codec, waveforms)
                emb = conditioner.encode([{"text_prompt": prompt}])["crossattn_emb"][
                    0
                ].cpu().numpy()

                if world_size == 1:
                    key = f"{shard_idx:06d}_{written:08d}"
                else:
                    key = f"r{rank:02d}_{shard_idx:06d}_{written:08d}"
                buf = io.BytesIO()
                torch.save(latents, buf)
                _tar_add(tar, f"{key}.pth", buf.getvalue())
                _tar_add(tar, f"{key}.pickle", pickle.dumps(emb.astype(np.float16)))
                meta = {
                    "frame_rate": codec.frame_rate,
                    "latent_channels": int(next(iter(latents.values())).shape[0]),
                    "latent_frames": int(latent_frames),
                    "aligned_samples": int(aligned_samples),
                    "prompt": prompt,
                    "caption": row.get("caption"),
                    "lyric": row.get("lyric"),
                    "mixture_path": row.get("mixture_path"),
                    "vocals_path": row.get("vocals_path"),
                    "accompaniment_path": row.get("accompaniment_path"),
                    "stem_keys": list(STEM_KEYS),
                    "text_conditioner": args.text,
                    "codec": args.codec,
                    "source_line_number": line_number,
                    "worker_rank": rank,
                    "worker_world_size": world_size,
                    "worker_local_rank": local_rank,
                }
                _tar_add(
                    tar,
                    f"{key}.json",
                    json.dumps(meta, ensure_ascii=False).encode(),
                )
                written += 1
        except Exception as e:
            skipped += 1
            error_tag = "oom" if _is_cuda_oom_error(e) else "error"
            print(
                f"{worker_prefix} {error_tag} row {line_number}: "
                f"{type(e).__name__}: {e}"
            )
            _cleanup_worker_memory(device, aggressive=True)
        finally:
            del prompt, waveforms, latents, emb, buf
            if (
                args.cuda_empty_cache_every > 0
                and _is_cuda_device(device)
                and processed % args.cuda_empty_cache_every == 0
            ):
                _cleanup_worker_memory(device, aggressive=True)
        if args.log_every > 0 and processed % args.log_every == 0:
            print(
                f"{worker_prefix} progress: processed={processed} "
                f"written={written} skipped={skipped} shards={shard_idx}"
            )

    if tar is not None:
        tar.close()
    print(
        f"{worker_prefix} done: processed={processed} written={written} "
        f"skipped={skipped} shards={shard_idx} out={args.out}"
    )
    if rank == 0:
        print("next: run `energon prepare", args.out, "` to create the .nv-meta splits")


if __name__ == "__main__":
    main()
