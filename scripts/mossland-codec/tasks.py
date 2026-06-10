from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torchaudio.transforms as AT
from torch.utils.data import Dataset


TASK_NAMES = (
    "reconstruct",
    "separate_vocals",
    "separate_accompaniment",
    "super_resolution",
    "mono_to_stereo",
)
SUPER_RESOLUTION_RATE_BUCKETS = (
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    40000,
)
_RESAMPLER_CACHE = {}


@dataclass(frozen=True)
class MosslandTaskBatch:
    src: torch.Tensor
    target: torch.Tensor
    task_id: str | tuple[str, ...]

    def to_payload(self) -> dict:
        return {
            "src": self.src,
            "target": self.target,
            "task_id": self.task_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "MosslandTaskBatch":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "MosslandCodecTrainingWrapper expects task payloads produced by "
                "MosslandTaskDataset"
            )
        missing = {"src", "target", "task_id"} - set(payload)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise KeyError(
                "MosslandTaskDataset task payload missing required keys: "
                f"{missing_keys}"
            )

        src = _flatten_collated_crops(payload["src"])
        target = _flatten_collated_crops(payload["target"])
        label_count = src.shape[0] if torch.is_tensor(src) and src.ndim >= 3 else None
        return cls(
            src=src,
            target=target,
            task_id=coerce_batch_label(payload["task_id"], label_count),
        )


def _flatten_collated_crops(audio: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(audio) and audio.ndim == 4:
        return audio.flatten(0, 1)
    return audio


def _as_batched_audio(audio: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if audio.ndim == 2:
        return audio.unsqueeze(0), True
    if audio.ndim == 3:
        return audio, False
    raise ValueError(f"audio must have shape [C,T] or [B,C,T], got {tuple(audio.shape)}")


def _restore_batch_dim(audio: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return audio.squeeze(0) if squeezed else audio


def _payload_get(payload, *keys: str):
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def coerce_label(value) -> str:
    """把 DataLoader 默认 collate 后的字符串标签还原为单个字符串。"""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return str(value.item())
        if value.numel() == 1:
            return str(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return coerce_label(value[0])
    return str(value)


def _is_label_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def coerce_batch_label(
    value,
    expected_length: int | None = None,
) -> str | tuple[str, ...]:
    """还原 batch/crop 维上的字符串标签，必要时保留为 tuple。"""
    if isinstance(value, (str, bytes)):
        return coerce_label(value)
    if torch.is_tensor(value):
        if value.ndim == 0 or value.numel() == 1:
            return coerce_label(value)
        labels = tuple(str(item.item()) for item in value.reshape(-1))
        return labels if len(labels) > 1 else labels[0]
    if not _is_label_sequence(value):
        return coerce_label(value)
    if not value:
        return ""

    if all(_is_label_sequence(item) for item in value):
        columns = [[coerce_label(cell) for cell in item] for item in value]
        if all(len(column) == len(columns[0]) for column in columns):
            labels = tuple(
                columns[crop_idx][batch_idx]
                for batch_idx in range(len(columns[0]))
                for crop_idx in range(len(columns))
            )
        else:
            labels = tuple(coerce_label(item) for item in value)
    else:
        labels = tuple(coerce_label(item) for item in value)

    if len(labels) == 1 and (expected_length is None or expected_length == 1):
        return labels[0]
    return labels


def _default_audio(payload) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    audio = _payload_get(payload, "audio", "waveform", "music", "target", "mixture")
    if audio is None:
        raise KeyError("payload must be a Tensor or contain one of: audio, waveform, music, target, mixture")
    return audio


def sample_low_sample_rate(low_sample_rate: int | Sequence[int]) -> int:
    if isinstance(low_sample_rate, Sequence) and not isinstance(low_sample_rate, (str, bytes)):
        rates = tuple(int(rate) for rate in low_sample_rate)
        if not rates:
            raise ValueError("low_sample_rate must not be empty")
        if len(rates) == 1:
            return rates[0]
        if len(rates) > 2:
            return int(random.choice(rates))

        min_rate, max_rate = sorted(rates)
        bucket_rates = tuple(
            rate for rate in SUPER_RESOLUTION_RATE_BUCKETS
            if min_rate <= rate <= max_rate
        )
        if bucket_rates:
            return int(random.choice(bucket_rates))

        first_grid_rate = ((min_rate + 99) // 100) * 100
        grid_rates = tuple(range(first_grid_rate, max_rate + 1, 100))
        if grid_rates:
            return int(random.choice(grid_rates))
        return min_rate
    return int(low_sample_rate)


def _match_time_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = audio.shape[-1]
    if current_length == target_length:
        return audio
    if current_length > target_length:
        return audio[..., :target_length]
    if current_length == 0:
        return audio.new_zeros(*audio.shape[:-1], target_length)

    pad_length = target_length - current_length
    tail = audio[..., -1:].expand(*audio.shape[:-1], pad_length)
    return torch.cat((audio, tail), dim=-1)


def _resampler_cache_key(
    orig_freq: int,
    new_freq: int,
    audio: torch.Tensor,
) -> tuple[int, int, str, int, torch.dtype]:
    device_index = -1 if audio.device.index is None else int(audio.device.index)
    return (
        int(orig_freq),
        int(new_freq),
        audio.device.type,
        device_index,
        audio.dtype,
    )


def _get_resampler(orig_freq: int, new_freq: int, audio: torch.Tensor):
    key = _resampler_cache_key(orig_freq, new_freq, audio)
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = AT.Resample(int(orig_freq), int(new_freq)).to(
            device=audio.device,
            dtype=audio.dtype,
        )
        _RESAMPLER_CACHE[key] = resampler
    return resampler


def _downsample_upsample(audio: torch.Tensor, sample_rate: int, low_sample_rate: int) -> torch.Tensor:
    if low_sample_rate <= 0 or low_sample_rate >= sample_rate:
        return audio.clone()

    target_length = audio.shape[-1]
    low = _get_resampler(sample_rate, low_sample_rate, audio)(audio)
    restored = _get_resampler(low_sample_rate, sample_rate, audio)(low)
    return _match_time_length(restored, target_length)


def _mono_to_stereo(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batched, squeezed = _as_batched_audio(audio)
    if batched.shape[-2] == 1:
        target = batched.repeat_interleave(2, dim=-2)
    else:
        target = batched[..., :2, :]
    mono = target.mean(dim=-2, keepdim=True)
    src = mono.repeat_interleave(2, dim=-2)
    return _restore_batch_dim(src, squeezed), _restore_batch_dim(target, squeezed)


def _mix_from_stems(payload: Mapping, target_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    target = _payload_get(payload, target_key)
    if target is None:
        raise KeyError(f"payload missing required stem {target_key!r}")

    mixture = _payload_get(payload, "mixture")
    if mixture is None:
        raise KeyError("payload missing required stem 'mixture'")
    return mixture, target


def build_task_batch(
    payload,
    task_id: str,
    sample_rate: int,
    low_sample_rate: int | Sequence[int] = 16000,
) -> MosslandTaskBatch:
    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported task_id={task_id!r}")

    if task_id == "reconstruct":
        audio = _default_audio(payload)
        return MosslandTaskBatch(src=audio, target=audio, task_id=task_id)

    if task_id == "super_resolution":
        target = _default_audio(payload)
        sampled_low_sample_rate = sample_low_sample_rate(low_sample_rate)
        src = _downsample_upsample(target, sample_rate, sampled_low_sample_rate)
        return MosslandTaskBatch(
            src=src,
            target=target,
            task_id=task_id,
        )

    if task_id == "mono_to_stereo":
        target = _default_audio(payload)
        src, target = _mono_to_stereo(target)
        return MosslandTaskBatch(
            src=src,
            target=target,
            task_id=task_id,
        )

    if not isinstance(payload, Mapping):
        raise TypeError(f"{task_id} requires a mapping payload with mixture/stem tensors")

    if task_id == "separate_vocals":
        src, target = _mix_from_stems(payload, "vocals")
        return MosslandTaskBatch(src=src, target=target, task_id=task_id)

    src, target = _mix_from_stems(payload, "accompaniment")
    return MosslandTaskBatch(src=src, target=target, task_id=task_id)


def _select_crop_payload(payload, crop_index: int, crop_count: int):
    if torch.is_tensor(payload):
        if payload.ndim >= 3 and payload.shape[0] == crop_count:
            return payload[crop_index]
        return payload
    if not isinstance(payload, Mapping):
        return payload
    return {
        key: (
            value[crop_index]
            if torch.is_tensor(value)
            and value.ndim >= 3
            and value.shape[0] == crop_count
            else value
        )
        for key, value in payload.items()
    }


def build_task_batch_for_tasks(
    payload,
    task_ids: Sequence[str],
    sample_rate: int,
    low_sample_rate: int | Sequence[int] = 16000,
) -> MosslandTaskBatch:
    task_ids = tuple(task_ids)
    if not task_ids:
        raise ValueError("task_ids must not be empty")

    crop_tasks = [
        build_task_batch(
            _select_crop_payload(payload, crop_index, len(task_ids)),
            task_id,
            sample_rate=sample_rate,
            low_sample_rate=low_sample_rate,
        )
        for crop_index, task_id in enumerate(task_ids)
    ]
    return MosslandTaskBatch(
        src=torch.stack([task.src for task in crop_tasks], dim=0),
        target=torch.stack([task.target for task in crop_tasks], dim=0),
        task_id=task_ids,
    )


def sample_task_id(active_tasks: Sequence[str], task_weights: Mapping[str, float] | None = None) -> str:
    if not active_tasks:
        raise ValueError("active_tasks must not be empty")
    for task in active_tasks:
        if task not in TASK_NAMES:
            raise ValueError(f"Unsupported task_id={task!r}")
    if task_weights is None:
        return random.choice(tuple(active_tasks))
    weights = [max(float(task_weights.get(task, 0.0)), 0.0) for task in active_tasks]
    if sum(weights) <= 0:
        return random.choice(tuple(active_tasks))
    return random.choices(tuple(active_tasks), weights=weights, k=1)[0]


class MosslandTaskDataset(Dataset):
    """把普通音频或 stem dataset 适配成 src/target/task_id payload。"""

    def __init__(
        self,
        dataset: Dataset,
        active_tasks: Sequence[str] = ("reconstruct",),
        task_weights: Mapping[str, float] | None = None,
        sample_rate: int = 48000,
        low_sample_rate: int | Sequence[int] = 16000,
    ):
        self.dataset = dataset
        self.active_tasks = tuple(active_tasks)
        self.task_weights = dict(task_weights or {})
        self.sample_rate = sample_rate
        self.low_sample_rate = low_sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        get_item_for_tasks = getattr(self.dataset, "get_item_for_tasks", None)
        crops_per_file = int(getattr(self.dataset, "crops_per_file", 1))
        if get_item_for_tasks is not None and crops_per_file > 1:
            task_ids = tuple(
                sample_task_id(self.active_tasks, self.task_weights)
                for _ in range(crops_per_file)
            )
            item = get_item_for_tasks(index, task_ids)
            if isinstance(item, tuple) and len(item) == 2:
                payload, info = item
            else:
                payload, info = item, {}
            task = build_task_batch_for_tasks(
                payload,
                task_ids,
                sample_rate=self.sample_rate,
                low_sample_rate=self.low_sample_rate,
            )
            return task.to_payload(), info

        task_id = sample_task_id(self.active_tasks, self.task_weights)
        get_item_for_task = getattr(self.dataset, "get_item_for_task", None)
        if get_item_for_task is not None:
            item = get_item_for_task(index, task_id)
        else:
            item = self.dataset[index]
        if isinstance(item, tuple) and len(item) == 2:
            payload, info = item
        else:
            payload, info = item, {}
        task = build_task_batch(
            payload,
            task_id,
            sample_rate=self.sample_rate,
            low_sample_rate=self.low_sample_rate,
        )
        return task.to_payload(), info
