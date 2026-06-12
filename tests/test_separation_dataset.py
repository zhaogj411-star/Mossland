from pathlib import Path
import importlib
from types import SimpleNamespace

import pytest
import torch

datasets = pytest.importorskip("scripts.data.datasets")
tasks = importlib.import_module("scripts.mossland-codec.tasks")


class TinyCountDataset(torch.utils.data.Dataset):
    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return torch.zeros(2, 8), {"index": index}


def _write_placeholder(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())


def _write_prepared_item(item_dir: Path):
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")


def test_prepared_separation_dataset_reads_done_items_directly(tmp_path, monkeypatch):
    done_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(done_dir / "mixture.mp3")
    _write_placeholder(done_dir / "vocals.mp3")
    _write_placeholder(done_dir / "accompaniment.mp3")
    (done_dir / "metadata.json").write_text(
        '{"status": "done", "source_path": "/data/source/audio/20260520/1001230.mp3"}',
        encoding="utf-8",
    )

    long_dir = tmp_path / "separated" / "audio" / "20260520" / "1009999"
    long_dir.mkdir(parents=True)
    (long_dir / "metadata.json").write_text('{"status": "skiplong"}', encoding="utf-8")

    def fake_load_stem(self, path):
        values = {
            "mixture.mp3": torch.arange(40, dtype=torch.float32),
            "vocals.mp3": torch.arange(100, 140, dtype=torch.float32),
            "accompaniment.mp3": torch.arange(200, 240, dtype=torch.float32),
        }
        return values[path.name].repeat(2, 1)

    monkeypatch.setattr(
        datasets.PreparedSeparationDataset,
        "_load_stem",
        fake_load_stem,
    )
    dataset = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )

    payload, info = dataset[0]

    assert len(dataset) == 1
    assert set(payload) == {"audio", "mixture", "vocals", "accompaniment"}
    torch.testing.assert_close(payload["audio"], torch.arange(8, dtype=torch.float32).repeat(2, 1))
    torch.testing.assert_close(payload["mixture"], payload["audio"])
    torch.testing.assert_close(payload["vocals"], torch.arange(100, 108, dtype=torch.float32).repeat(2, 1))
    torch.testing.assert_close(
        payload["accompaniment"],
        torch.arange(200, 208, dtype=torch.float32).repeat(2, 1),
    )
    assert info["path"] == "/data/source/audio/20260520/1001230.mp3"
    assert info["separation_dir"] == str(done_dir)
    assert info["relpath"] == "audio/20260520/1001230"
    assert (tmp_path / "separated" / "index.list").read_text(encoding="utf-8") == (
        "audio/20260520/1001230\n"
    )


def test_prepared_separation_dataset_stacks_multiple_crops_per_loaded_item(
    tmp_path,
    monkeypatch,
):
    item_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")
    (item_dir / "metadata.json").write_text('{"status": "done"}', encoding="utf-8")
    loaded = []

    def fake_load_stem(self, path):
        loaded.append(path.name)
        values = {
            "mixture.mp3": torch.arange(40, dtype=torch.float32),
            "vocals.mp3": torch.arange(100, 140, dtype=torch.float32),
            "accompaniment.mp3": torch.arange(200, 240, dtype=torch.float32),
        }
        return values[path.name].repeat(2, 1)

    monkeypatch.setattr(datasets.PreparedSeparationDataset, "_load_stem", fake_load_stem)

    dataset = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
        crops_per_file=3,
    )

    payload, info = dataset.get_item_for_task(0, "separate_vocals")

    assert loaded == ["mixture.mp3", "vocals.mp3"]
    assert payload["mixture"].shape == (3, 2, 8)
    assert payload["vocals"].shape == (3, 2, 8)
    assert "accompaniment" not in payload
    assert info["crops_per_file"] == 3


def test_prepared_separation_dataset_loads_stem_union_for_per_crop_tasks(
    tmp_path,
    monkeypatch,
):
    item_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")
    (item_dir / "metadata.json").write_text('{"status": "done"}', encoding="utf-8")
    loaded = []

    def fake_load_stem(self, path):
        loaded.append(path.name)
        values = {
            "mixture.mp3": torch.arange(40, dtype=torch.float32),
            "vocals.mp3": torch.arange(100, 140, dtype=torch.float32),
            "accompaniment.mp3": torch.arange(200, 240, dtype=torch.float32),
        }
        return values[path.name].repeat(2, 1)

    monkeypatch.setattr(datasets.PreparedSeparationDataset, "_load_stem", fake_load_stem)

    dataset = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
        crops_per_file=4,
    )

    payload, info = dataset.get_item_for_tasks(
        0,
        (
            "reconstruct",
            "separate_vocals",
            "separate_accompaniment",
            "mono_to_stereo",
        ),
    )

    assert loaded == ["mixture.mp3", "vocals.mp3", "accompaniment.mp3"]
    assert payload["mixture"].shape == (4, 2, 8)
    assert payload["vocals"].shape == (4, 2, 8)
    assert payload["accompaniment"].shape == (4, 2, 8)
    assert info["task_ids"] == (
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "mono_to_stereo",
    )


def test_prepared_separation_dataset_reuses_existing_index_list(tmp_path):
    item_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")
    first = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )

    new_item_dir = tmp_path / "separated" / "audio" / "20260520" / "1009999"
    _write_placeholder(new_item_dir / "mixture.mp3")
    _write_placeholder(new_item_dir / "vocals.mp3")
    _write_placeholder(new_item_dir / "accompaniment.mp3")
    second = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )

    assert first.item_dirs == [item_dir]
    assert second.item_dirs == [item_dir]


def test_prepared_separation_dataset_fixed_length_uses_item_count_modulo(
    tmp_path,
    monkeypatch,
):
    first_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    second_dir = tmp_path / "separated" / "audio" / "20260520" / "1009999"
    _write_prepared_item(first_dir)
    _write_prepared_item(second_dir)

    loaded_dirs = []

    def fake_load_item(self, item_dir, task_id=None, task_ids=None):
        loaded_dirs.append(item_dir)
        return {"audio": torch.ones(2, 8), "mixture": torch.ones(2, 8)}, {
            "separation_dir": str(item_dir)
        }

    monkeypatch.setattr(datasets.PreparedSeparationDataset, "_load_item", fake_load_item)

    dataset = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
        length=5,
    )

    _, info = dataset.get_item_for_task(4, "reconstruct")

    assert len(dataset) == 5
    assert loaded_dirs == [first_dir]
    assert info["separation_dir"] == str(first_dir)


def test_prepared_separation_dataset_rebuild_index_updates_item_dirs(tmp_path):
    root = tmp_path / "separated"
    first_dir = root / "audio" / "20260520" / "1001230"
    _write_prepared_item(first_dir)
    dataset = datasets.PreparedSeparationDataset(
        dirs=[root],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )

    second_dir = root / "audio" / "20260520" / "1009999"
    _write_prepared_item(second_dir)

    dataset.rebuild_index()

    assert dataset.item_dirs == [first_dir, second_dir]
    assert (root / "index.list").read_text(encoding="utf-8") == (
        "audio/20260520/1001230\naudio/20260520/1009999\n"
    )


def test_prepared_separation_dataset_lazily_filters_overlong_items_with_ffprobe(
    tmp_path,
    monkeypatch,
):
    short_dir = tmp_path / "separated" / "audio" / "20260520" / "short"
    long_dir = tmp_path / "separated" / "audio" / "20260520" / "long"
    _write_prepared_item(short_dir)
    _write_prepared_item(long_dir)
    probed_paths = []

    def fake_run(cmd, capture_output, text, timeout, check):
        assert cmd[0] == "ffprobe"
        path = Path(cmd[-1])
        probed_paths.append(path)
        stdout = "301.0\n" if path.parent == long_dir else "299.0\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        datasets.PreparedSeparationDataset,
        "_load_stem",
        lambda self, path: torch.ones(2, 16),
    )

    dataset = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
        max_duration_seconds=300,
    )

    assert dataset.item_dirs == [long_dir, short_dir]
    assert probed_paths == []

    payload, info = dataset.get_item_for_task(0, "reconstruct")

    assert dataset.item_dirs == [short_dir]
    assert probed_paths == [long_dir / "mixture.mp3", short_dir / "mixture.mp3"]
    assert info["separation_dir"] == str(short_dir)
    assert payload["mixture"].shape == (2, 8)
    assert (tmp_path / "separated" / "index.list").read_text(encoding="utf-8") == (
        "audio/20260520/long\naudio/20260520/short\n"
    )


def test_prepared_separation_dataset_lazily_filters_existing_index_with_ffprobe(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "separated"
    short_dir = root / "audio" / "20260520" / "short"
    long_dir = root / "audio" / "20260520" / "long"
    _write_prepared_item(short_dir)
    _write_prepared_item(long_dir)
    (root / "index.list").write_text(
        "audio/20260520/short\naudio/20260520/long\n",
        encoding="utf-8",
    )

    def fake_run(cmd, capture_output, text, timeout, check):
        path = Path(cmd[-1])
        stdout = "300.0\n" if path.parent == long_dir else "120.0\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        datasets.PreparedSeparationDataset,
        "_load_stem",
        lambda self, path: torch.ones(2, 16),
    )

    dataset = datasets.PreparedSeparationDataset(
        dirs=[root],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
        max_duration_seconds=300,
    )

    assert dataset.item_dirs == [long_dir, short_dir]

    payload, info = dataset.get_item_for_task(0, "reconstruct")

    assert dataset.item_dirs == [short_dir]
    assert info["separation_dir"] == str(short_dir)
    assert payload["mixture"].shape == (2, 8)
    assert (root / "index.list").read_text(encoding="utf-8") == (
        "audio/20260520/short\naudio/20260520/long\n"
    )


def test_experiment_dataset_keeps_one_validation_item_for_small_positive_split():
    data = datasets.Experiment_Dataset(
        TinyCountDataset(3),
        train_batch_size=1,
        val_batch_size=1,
        val_split=0.0001,
    )

    data.setup()

    assert len(data.train_dataset) == 2
    assert len(data.val_dataset) == 1


def test_task_dataset_reconstruct_loads_only_prepared_mixture(tmp_path, monkeypatch):
    item_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")
    (item_dir / "metadata.json").write_text('{"status": "done"}', encoding="utf-8")
    loaded = []

    def fake_load_stem(self, path):
        loaded.append(path.name)
        return torch.ones(2, 16)

    monkeypatch.setattr(datasets.PreparedSeparationDataset, "_load_stem", fake_load_stem)

    base = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )
    dataset = tasks.MosslandTaskDataset(
        base,
        active_tasks=("reconstruct",),
        sample_rate=48000,
    )

    payload, _ = dataset[0]

    assert loaded == ["mixture.mp3"]
    assert payload["task_id"] == "reconstruct"
    assert payload["src"].shape == (2, 8)
    assert payload["target"].shape == (2, 8)


def test_task_dataset_vocal_separation_skips_unused_accompaniment(tmp_path, monkeypatch):
    item_dir = tmp_path / "separated" / "audio" / "20260520" / "1001230"
    _write_placeholder(item_dir / "mixture.mp3")
    _write_placeholder(item_dir / "vocals.mp3")
    _write_placeholder(item_dir / "accompaniment.mp3")
    (item_dir / "metadata.json").write_text('{"status": "done"}', encoding="utf-8")
    loaded = []

    def fake_load_stem(self, path):
        loaded.append(path.name)
        values = {
            "mixture.mp3": 1.0,
            "vocals.mp3": 2.0,
            "accompaniment.mp3": 3.0,
        }
        return torch.full((2, 16), values[path.name])

    monkeypatch.setattr(datasets.PreparedSeparationDataset, "_load_stem", fake_load_stem)

    base = datasets.PreparedSeparationDataset(
        dirs=[tmp_path / "separated"],
        sample_size=8,
        sample_rate=48000,
        random_crop=False,
    )
    dataset = tasks.MosslandTaskDataset(
        base,
        active_tasks=("separate_vocals",),
        sample_rate=48000,
    )

    payload, _ = dataset[0]

    assert loaded == ["mixture.mp3", "vocals.mp3"]
    assert payload["task_id"] == "separate_vocals"
    torch.testing.assert_close(payload["src"], torch.ones(2, 8))
    torch.testing.assert_close(payload["target"], torch.full((2, 8), 2.0))
