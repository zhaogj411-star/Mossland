import importlib
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

tasks = importlib.import_module("scripts.mossland-codec.tasks")
TASK_NAMES = tasks.TASK_NAMES
MosslandTaskDataset = tasks.MosslandTaskDataset
build_task_batch = tasks.build_task_batch
coerce_label = tasks.coerce_label


def test_task_registry_matches_mossland_document_tasks():
    assert TASK_NAMES == (
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "super_resolution",
        "mono_to_stereo",
    )


def test_build_reconstruct_task_uses_same_source_and_target():
    audio = torch.randn(2, 2, 128)

    task = build_task_batch(audio, "reconstruct", sample_rate=48000)

    assert task.task_id == "reconstruct"
    torch.testing.assert_close(task.src, audio)
    torch.testing.assert_close(task.target, audio)


def test_build_mono_to_stereo_uses_duplicate_mono_source_and_stereo_target():
    left = torch.linspace(-1.0, 1.0, steps=32)
    right = torch.linspace(1.0, -1.0, steps=32)
    audio = torch.stack([left, right], dim=0).unsqueeze(0)

    task = build_task_batch(audio, "mono_to_stereo", sample_rate=48000)

    mono = audio.mean(dim=-2, keepdim=True)
    expected_src = mono.repeat_interleave(2, dim=-2)
    torch.testing.assert_close(task.src, expected_src)
    torch.testing.assert_close(task.target, audio)
    assert not hasattr(task, "channel_mode")


def test_build_super_resolution_keeps_shape_without_recording_degradation():
    audio = torch.randn(1, 2, 480)

    task = build_task_batch(
        audio,
        "super_resolution",
        sample_rate=48000,
        low_sample_rate=12000,
    )

    assert task.src.shape == audio.shape
    assert task.target.shape == audio.shape
    assert not hasattr(task, "degradation_id")


def test_build_super_resolution_filters_above_low_sample_rate_nyquist():
    sample_rate = 48000
    low_sample_rate = 16000
    duration = sample_rate // 2
    time = torch.arange(duration) / sample_rate
    high_tone = torch.sin(2 * torch.pi * 12000 * time).reshape(1, 1, -1)

    task = build_task_batch(
        high_tone,
        "super_resolution",
        sample_rate=sample_rate,
        low_sample_rate=low_sample_rate,
    )

    input_rms = high_tone.pow(2).mean().sqrt()
    source_rms = task.src.pow(2).mean().sqrt()
    assert source_rms < input_rms * 0.05


def test_time_length_matching_trims_or_repeats_last_sample():
    audio = torch.tensor([[[1.0, 2.0, 3.0]]])

    matched = tasks._match_time_length(audio, 5)

    assert matched.shape[-1] == 5
    torch.testing.assert_close(matched, torch.tensor([[[1.0, 2.0, 3.0, 3.0, 3.0]]]))
    torch.testing.assert_close(tasks._match_time_length(matched, 2), torch.tensor([[[1.0, 2.0]]]))


def test_sample_low_sample_rate_samples_within_range():
    rates = {tasks.sample_low_sample_rate((8000, 24000)) for _ in range(40)}

    assert min(rates) >= 8000
    assert max(rates) <= 24000


def test_sample_low_sample_rate_range_uses_fast_audio_sr_buckets(monkeypatch):
    chosen = {}

    def fake_choice(values):
        chosen["values"] = tuple(values)
        return values[-1]

    monkeypatch.setattr(tasks.random, "choice", fake_choice)

    assert tasks.sample_low_sample_rate((8000, 40000)) == 40000
    assert chosen["values"] == (
        8000,
        11025,
        12000,
        16000,
        22050,
        24000,
        32000,
        40000,
    )


def test_sample_low_sample_rate_accepts_explicit_choices(monkeypatch):
    monkeypatch.setattr(tasks.random, "choice", lambda values: values[1])

    assert tasks.sample_low_sample_rate((12000, 16000, 24000)) == 16000


def test_downsample_upsample_reuses_cached_resamplers(monkeypatch):
    created = []

    class FakeResampler:
        def __init__(self, orig_freq, new_freq):
            created.append((orig_freq, new_freq))

        def to(self, *, device=None, dtype=None):
            return self

        def __call__(self, waveform):
            return waveform

    monkeypatch.setattr(tasks, "AT", SimpleNamespace(Resample=FakeResampler), raising=False)
    if hasattr(tasks, "_RESAMPLER_CACHE"):
        tasks._RESAMPLER_CACHE.clear()

    audio = torch.randn(1, 2, 8)

    tasks._downsample_upsample(audio, sample_rate=48000, low_sample_rate=16000)
    tasks._downsample_upsample(audio, sample_rate=48000, low_sample_rate=16000)

    assert created == [(48000, 16000), (16000, 48000)]


def test_build_separation_tasks_use_offline_mixture_and_stems():
    payload = {
        "mixture": torch.full((2, 64), 0.1),
        "vocals": torch.full((2, 64), 0.2),
        "accompaniment": torch.full((2, 64), 0.3),
    }

    vocal_task = build_task_batch(payload, "separate_vocals", sample_rate=48000)
    accompaniment_task = build_task_batch(
        payload,
        "separate_accompaniment",
        sample_rate=48000,
    )

    torch.testing.assert_close(vocal_task.src, payload["mixture"])
    torch.testing.assert_close(vocal_task.target, payload["vocals"])
    torch.testing.assert_close(accompaniment_task.src, payload["mixture"])
    torch.testing.assert_close(accompaniment_task.target, payload["accompaniment"])


def test_build_separation_task_requires_explicit_prepared_mixture():
    payload = {
        "vocals": torch.full((2, 64), 0.2),
        "accompaniment": torch.full((2, 64), 0.3),
    }

    try:
        build_task_batch(payload, "separate_vocals", sample_rate=48000)
    except KeyError as exc:
        assert "mixture" in str(exc)
    else:
        raise AssertionError("separation task must require explicit prepared mixture")


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.ones(2, 64), {"path": "tiny.wav"}


def test_dataset_adapter_returns_src_target_task_payload():
    dataset = MosslandTaskDataset(
        TinyDataset(),
        active_tasks=("reconstruct",),
        sample_rate=48000,
    )

    payload, info = dataset[0]

    assert payload["task_id"] == "reconstruct"
    assert set(payload) == {"src", "target", "task_id"}
    assert payload["src"].shape == (2, 64)
    assert payload["target"].shape == (2, 64)
    assert info["path"] == "tiny.wav"


def test_coerce_label_handles_default_dataloader_collated_strings():
    dataset = MosslandTaskDataset(
        TinyDataset(),
        active_tasks=("reconstruct",),
        sample_rate=48000,
    )
    payload, _ = next(iter(DataLoader(dataset, batch_size=1)))

    assert coerce_label(payload["task_id"]) == "reconstruct"


def test_task_batch_flattens_dataloader_collated_crop_dimension():
    task = tasks.MosslandTaskBatch.from_payload(
        {
            "src": torch.ones(1, 4, 2, 8),
            "target": torch.zeros(1, 4, 2, 8),
            "task_id": [
                ("reconstruct",),
                ("separate_vocals",),
                ("separate_accompaniment",),
                ("mono_to_stereo",),
            ],
        }
    )

    assert task.src.shape == (4, 2, 8)
    assert task.target.shape == (4, 2, 8)
    assert task.task_id == (
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "mono_to_stereo",
    )
    assert not hasattr(task, "channel_mode")


def test_task_batch_flattens_collated_crop_labels_in_batch_major_order():
    task = tasks.MosslandTaskBatch.from_payload(
        {
            "src": torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1),
            "target": torch.zeros(2, 2, 1, 1),
            "task_id": [("item0_crop0", "item1_crop0"), ("item0_crop1", "item1_crop1")],
        }
    )

    assert task.src.reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert task.task_id == (
        "item0_crop0",
        "item0_crop1",
        "item1_crop0",
        "item1_crop1",
    )


def test_task_dataset_samples_task_per_crop_for_prepared_multi_crop_dataset(monkeypatch):
    sampled = iter(
        (
            "reconstruct",
            "separate_vocals",
            "separate_accompaniment",
            "mono_to_stereo",
        )
    )

    monkeypatch.setattr(tasks, "sample_task_id", lambda active_tasks, task_weights: next(sampled))

    class MultiCropPreparedDataset(torch.utils.data.Dataset):
        crops_per_file = 4

        def __len__(self):
            return 1

        def get_item_for_tasks(self, index, task_ids):
            assert task_ids == (
                "reconstruct",
                "separate_vocals",
                "separate_accompaniment",
                "mono_to_stereo",
            )
            return (
                {
                    "mixture": torch.full((4, 2, 8), 1.0),
                    "vocals": torch.full((4, 2, 8), 2.0),
                    "accompaniment": torch.full((4, 2, 8), 3.0),
                },
                {"task_ids": task_ids},
            )

    dataset = MosslandTaskDataset(
        MultiCropPreparedDataset(),
        active_tasks=TASK_NAMES,
        sample_rate=48000,
    )

    payload, info = dataset[0]

    assert payload["src"].shape == (4, 2, 8)
    assert payload["target"].shape == (4, 2, 8)
    assert set(payload) == {"src", "target", "task_id"}
    assert payload["task_id"] == (
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "mono_to_stereo",
    )
    torch.testing.assert_close(payload["target"][0], torch.full((2, 8), 1.0))
    torch.testing.assert_close(payload["target"][1], torch.full((2, 8), 2.0))
    torch.testing.assert_close(payload["target"][2], torch.full((2, 8), 3.0))
    torch.testing.assert_close(payload["src"][3], torch.full((2, 8), 1.0))
    assert info["task_ids"] == payload["task_id"]
