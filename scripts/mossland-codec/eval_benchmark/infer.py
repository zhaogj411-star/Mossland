from __future__ import annotations

import importlib
from pathlib import Path

import torch

from .audio_io import load_audio_segment, save_audio
from .manifest import EvalItem


tasks = importlib.import_module("scripts.mossland-codec.tasks")


def load_mossland_model(checkpoint_dir: str | Path, device: str | torch.device | None = None):
    from scripts.factory import load_model

    model = load_model(ckpt_dir=str(checkpoint_dir))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()


def prediction_path_for_item(item: EvalItem, output_dir: str | Path) -> Path:
    return Path(output_dir) / item.task_id / f"{item.item_id}_seed{item.seed}.wav"


def build_task_input(item: EvalItem) -> tuple[torch.Tensor, torch.Tensor | None]:
    if item.task_id in {"reconstruct", "super_resolution", "mono_to_stereo"}:
        audio = load_audio_segment(
            item.source_path,
            sample_rate=item.sample_rate,
            start_seconds=item.start_seconds,
            duration_seconds=item.duration_seconds,
        )
        payload = audio
    else:
        mixture = item.mixture_path or item.source_path
        payload = {
            "mixture": load_audio_segment(
                mixture,
                sample_rate=item.sample_rate,
                start_seconds=item.start_seconds,
                duration_seconds=item.duration_seconds,
            )
        }
        if item.vocals_path is not None:
            payload["vocals"] = load_audio_segment(
                item.vocals_path,
                sample_rate=item.sample_rate,
                start_seconds=item.start_seconds,
                duration_seconds=item.duration_seconds,
            )
        if item.accompaniment_path is not None:
            payload["accompaniment"] = load_audio_segment(
                item.accompaniment_path,
                sample_rate=item.sample_rate,
                start_seconds=item.start_seconds,
                duration_seconds=item.duration_seconds,
            )
        for stem in ("drums", "bass", "other"):
            path = item.metadata.get(f"{stem}_path")
            if path not in (None, ""):
                payload[stem] = load_audio_segment(
                    path,
                    sample_rate=item.sample_rate,
                    start_seconds=item.start_seconds,
                    duration_seconds=item.duration_seconds,
                )

    task = tasks.build_task_batch(
        payload,
        item.task_id,
        sample_rate=item.sample_rate,
        low_sample_rate=item.low_sample_rate or 16000,
    )
    return task.src, task.target


@torch.inference_mode()
def generate_prediction(
    model,
    item: EvalItem,
    output_dir: str | Path,
    quantize: bool = False,
    overwrite: bool = False,
) -> Path:
    output_path = prediction_path_for_item(item, output_dir)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return output_path.resolve()

    torch.manual_seed(int(item.seed))
    source, _ = build_task_input(item)
    device = next(model.parameters()).device
    source = source.unsqueeze(0).to(device)
    _, generated = model.generate_waveform(
        source,
        task_id=item.task_id,
        quantize=quantize,
    )
    prediction = generated.squeeze(0)
    save_audio(output_path, prediction, item.sample_rate)
    return output_path.resolve()
