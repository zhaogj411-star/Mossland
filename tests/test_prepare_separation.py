import gc
import importlib
import json
import sys
import threading
from subprocess import CompletedProcess
from pathlib import Path

import pytest
import torch

prepare = importlib.import_module("scripts.data.prepare_separation")


def test_safe_stem_id_preserves_relative_directory_structure():
    source_root = Path("/data/music")
    audio_path = Path("/data/music/a/b/song name.wav")

    stem_id = prepare.safe_stem_id(audio_path, source_root)

    assert stem_id == str(Path("a") / "b" / "song name")


def test_separation_layout_preserves_relative_directory_structure(tmp_path):
    layout = prepare.separation_layout(
        tmp_path / "source" / "artist" / "album" / "track.wav",
        tmp_path / "separated",
        source_root=tmp_path / "source",
    )

    assert layout.item_dir == tmp_path / "separated" / "artist" / "album" / "track"
    assert layout.mixture_path == layout.item_dir / "mixture.mp3"
    assert layout.vocals_path == layout.item_dir / "vocals.mp3"
    assert layout.accompaniment_path == layout.item_dir / "accompaniment.mp3"
    assert layout.metadata_path == layout.item_dir / "metadata.json"


def test_write_metadata_records_local_model_config(tmp_path):
    layout = prepare.separation_layout(
        tmp_path / "track.wav",
        tmp_path / "separated",
    )
    layout.item_dir.mkdir(parents=True)

    prepare.write_metadata(
        layout,
        source_path=tmp_path / "track.wav",
        model_repo="KimberleyJensen/Mel-Band-Roformer-Vocal-Model",
        model_config={
            "model": {"sample_rate": 44100, "stereo": True},
            "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
            "inference": {"num_overlap": 2, "chunk_size": 352800},
        },
        status="done",
    )

    metadata = json.loads(layout.metadata_path.read_text())
    assert metadata["status"] == "done"
    assert metadata["stems"]["mixture"] == "mixture.mp3"
    assert metadata["stems"]["vocals"] == "vocals.mp3"
    assert metadata["stems"]["accompaniment"] == "accompaniment.mp3"
    assert metadata["model"]["repo"] == "KimberleyJensen/Mel-Band-Roformer-Vocal-Model"
    assert metadata["model"]["sample_rate"] == 44100
    assert metadata["model"]["num_channels"] == 2
    assert metadata["model"]["num_overlap"] == 2
    assert metadata["model"]["target_instrument"] == "vocals"
    assert "config_url" not in metadata["model"]
    assert "checkpoint_url" not in metadata["model"]


def test_default_model_dir_points_to_repo_checkpoints():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "checkpoints" / "mel-band-roformer-vocal-model"

    assert prepare.default_model_dir() == model_dir
    assert prepare.default_config_path() == model_dir / prepare.DEFAULT_CONFIG_NAME
    assert prepare.default_checkpoint_path() == model_dir / prepare.DEFAULT_CHECKPOINT_NAME


def test_default_model_points_to_kimberley_vocal_model_files():
    assert prepare.DEFAULT_MODEL_REPO == "KimberleyJensen/Mel-Band-Roformer-Vocal-Model"
    assert prepare.DEFAULT_CONFIG_NAME == "config_vocals_mel_band_roformer.yaml"
    assert prepare.DEFAULT_CHECKPOINT_NAME == "MelBandRoformer.ckpt"


def test_load_model_config_reads_default_local_config(tmp_path, monkeypatch):
    config_path = tmp_path / prepare.DEFAULT_CONFIG_NAME
    config_path.write_text(
        """
model:
  sample_rate: 44100
  stereo: true
  num_stems: 1
training:
  instruments:
  - vocals
  - other
  target_instrument: vocals
inference:
  num_overlap: 2
  chunk_size: 352800
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "default_config_path", lambda: config_path)

    config = prepare.load_model_config(None)

    assert config["model"]["num_stems"] == 1
    assert config["training"]["instruments"] == ["vocals", "other"]
    assert config["training"]["target_instrument"] == "vocals"
    assert config["inference"]["num_overlap"] == 2
    assert config["inference"]["chunk_size"] == 352800


def test_load_model_config_requires_default_local_config(tmp_path, monkeypatch):
    missing_config = tmp_path / prepare.DEFAULT_CONFIG_NAME
    monkeypatch.setattr(prepare, "default_config_path", lambda: missing_config)

    with pytest.raises(FileNotFoundError):
        prepare.load_model_config(None)


def test_prepare_separation_does_not_import_training_dataset_module():
    source = Path(prepare.__file__).read_text(encoding="utf-8")

    assert "scripts.data.datasets" not in source


def test_prepare_separation_uses_modern_torch_amp_autocast_api():
    source = Path(prepare.__file__).read_text(encoding="utf-8")

    assert "torch.cuda.amp.autocast" not in source
    assert 'torch.amp.autocast("cuda")' in source


def test_cleanup_cuda_memory_releases_cuda_allocator(monkeypatch):
    calls = []

    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(prepare.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prepare.torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(prepare.torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect"))

    prepare._cleanup_cuda_memory(torch.device("cuda:0"))

    assert calls == ["gc", "empty_cache", "ipc_collect"]


def test_cleanup_cuda_memory_skips_cpu_devices(monkeypatch):
    calls = []

    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(prepare.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prepare.torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(prepare.torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect"))

    prepare._cleanup_cuda_memory(torch.device("cpu"))

    assert calls == []


def test_select_shard_uses_round_robin_distribution():
    files = [f"track-{idx}.wav" for idx in range(10)]

    assert prepare.select_shard(files, shard_id=0, num_shards=3) == [
        "track-0.wav",
        "track-3.wav",
        "track-6.wav",
        "track-9.wav",
    ]
    assert prepare.select_shard(files, shard_id=1, num_shards=3) == [
        "track-1.wav",
        "track-4.wav",
        "track-7.wav",
    ]
    assert prepare.select_shard(files, shard_id=2, num_shards=3) == [
        "track-2.wav",
        "track-5.wav",
        "track-8.wav",
    ]


def test_build_worker_spec_binds_visible_device_and_shard(tmp_path):
    args = prepare.parse_args(
        [
            "--files-list",
            str(tmp_path / "input.txt"),
            "--output-root",
            str(tmp_path / "out"),
            "--source-root",
            "/data/music",
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--num-overlap",
            "2",
            "--chunk-batch-size",
            "3",
            "--max-duration-seconds",
            "540",
            "--save-workers",
            "2",
            "--max-pending-writes",
            "5",
            "--overwrite",
        ]
    )
    manifest = tmp_path / "manifest.txt"

    spec = prepare.build_worker_spec(args, manifest, shard_id=3, num_shards=8, visible_device="3")

    assert spec.command[0] == sys.executable
    assert spec.command[1:3] == ["-m", "scripts.data.prepare_separation"]
    assert "--devices" not in spec.command
    assert spec.command[spec.command.index("--files-list") + 1] == str(manifest)
    assert spec.command[spec.command.index("--num-shards") + 1] == "8"
    assert spec.command[spec.command.index("--shard-id") + 1] == "3"
    assert spec.command[spec.command.index("--device") + 1] == "cuda:0"
    assert spec.command[spec.command.index("--progress-file") + 1] == str(
        tmp_path / "out" / "_logs" / "prepare_separation" / "progress.jsonl"
    )
    assert spec.command[spec.command.index("--chunk-batch-size") + 1] == "3"
    assert spec.command[spec.command.index("--max-duration-seconds") + 1] == "540.0"
    assert spec.command[spec.command.index("--save-workers") + 1] == "2"
    assert spec.command[spec.command.index("--max-pending-writes") + 1] == "5"
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "3"
    assert spec.env["PYTHONUNBUFFERED"] == "1"
    assert spec.log_path.name == "worker_03_gpu3.log"


def test_filter_pending_files_skips_completed_outputs(tmp_path):
    done_audio = tmp_path / "done.wav"
    pending_audio = tmp_path / "pending.wav"
    done_audio.write_bytes(b"audio")
    pending_audio.write_bytes(b"audio")
    output_root = tmp_path / "out"

    done_layout = prepare.separation_layout(done_audio, output_root)
    done_layout.item_dir.mkdir(parents=True)
    done_layout.mixture_path.write_bytes(b"mixture")
    done_layout.vocals_path.write_bytes(b"vocals")
    done_layout.accompaniment_path.write_bytes(b"accompaniment")
    done_layout.metadata_path.write_text(json.dumps({"status": "done"}), encoding="utf-8")

    pending, skipped = prepare.filter_pending_files(
        [done_audio, pending_audio],
        output_root=output_root,
    )

    assert pending == [pending_audio]
    assert skipped == [done_audio]
    assert prepare.filter_pending_files([done_audio], output_root=output_root, overwrite=True) == (
        [done_audio],
        [],
    )


def test_filter_pending_files_skips_terminal_error_and_skiplong_metadata(tmp_path):
    error_audio = tmp_path / "error.wav"
    skiplong_audio = tmp_path / "long.wav"
    pending_audio = tmp_path / "pending.wav"
    for audio_path in (error_audio, skiplong_audio, pending_audio):
        audio_path.write_bytes(b"audio")
    output_root = tmp_path / "out"

    error_layout = prepare.separation_layout(error_audio, output_root)
    error_layout.item_dir.mkdir(parents=True)
    error_layout.metadata_path.write_text(json.dumps({"status": "error"}), encoding="utf-8")

    skiplong_layout = prepare.separation_layout(skiplong_audio, output_root)
    skiplong_layout.item_dir.mkdir(parents=True)
    skiplong_layout.metadata_path.write_text(json.dumps({"status": "skiplong"}), encoding="utf-8")

    pending, skipped = prepare.filter_pending_files(
        [error_audio, skiplong_audio, pending_audio],
        output_root=output_root,
    )

    assert pending == [pending_audio]
    assert skipped == [error_audio, skiplong_audio]


def test_probe_audio_duration_uses_ffprobe_when_available(tmp_path, monkeypatch):
    audio_path = tmp_path / "track.mp3"
    audio_path.write_bytes(b"audio")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return CompletedProcess(command, 0, stdout="601.25\n", stderr="")

    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)
    monkeypatch.setattr(prepare.subprocess, "run", fake_run)

    duration = prepare._probe_audio_duration_seconds(audio_path)

    assert duration == 601.25
    assert captured["command"][0] == "/usr/bin/ffprobe"
    assert str(audio_path) in captured["command"]
    assert captured["kwargs"]["timeout"] == 10


def test_progress_events_round_trip_incrementally(tmp_path):
    progress_file = tmp_path / "progress.jsonl"

    prepare._write_progress_event(progress_file, "done", worker_id=2, audio_path=tmp_path / "a.mp3")
    offset, events = prepare._read_progress_events(progress_file, 0)

    assert [event.status for event in events] == ["done"]
    assert events[0].worker_id == 2
    assert events[0].path == str(tmp_path / "a.mp3")

    prepare._write_progress_event(progress_file, "skipped", worker_id=3, audio_path=tmp_path / "b.mp3")
    next_offset, next_events = prepare._read_progress_events(progress_file, offset)

    assert next_offset > offset
    assert [event.status for event in next_events] == ["skipped"]

    prepare._write_progress_event(progress_file, "skiplong", worker_id=4, audio_path=tmp_path / "c.mp3")
    final_offset, final_events = prepare._read_progress_events(progress_file, next_offset)

    assert final_offset > next_offset
    assert [event.status for event in final_events] == ["skiplong"]


def test_iter_audio_files_uses_fast_scandir_for_input_dirs(tmp_path, monkeypatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    files_list = tmp_path / "files.txt"
    listed_audio = tmp_path / "listed.mp3"
    listed_audio.write_bytes(b"audio")
    files_list.write_text(str(listed_audio) + "\n", encoding="utf-8")
    captured = {}

    def fake_fast_scandir(input_dir, extensions, max_files=None):
        captured["input_dir"] = Path(input_dir)
        captured["extensions"] = set(extensions)
        return [str(audio_root / "song.mp3"), str(audio_root / "song.mp3")]

    monkeypatch.setattr(prepare, "_fast_audio_files", fake_fast_scandir)

    files = prepare._iter_audio_files([str(audio_root)], str(files_list))

    assert captured["input_dir"] == audio_root
    assert ".mp3" in captured["extensions"]
    assert files == [str(audio_root / "song.mp3"), str(listed_audio)]


def test_iter_audio_files_honors_max_files_before_scanning_dirs(tmp_path, monkeypatch):
    files_list = tmp_path / "files.txt"
    listed_audio = tmp_path / "listed.mp3"
    listed_audio.write_bytes(b"audio")
    files_list.write_text(str(listed_audio) + "\n", encoding="utf-8")

    def fail_fast_scandir(input_dir, extensions, max_files=None):
        raise AssertionError("input dirs should not be scanned after max_files is satisfied")

    monkeypatch.setattr(prepare, "_fast_audio_files", fail_fast_scandir)

    files = prepare._iter_audio_files([str(tmp_path / "audio")], str(files_list), max_files=1)

    assert files == [str(listed_audio)]


def test_process_files_reuses_separator_for_worker_batch(tmp_path, monkeypatch):
    audio_paths = [tmp_path / f"track-{idx}.wav" for idx in range(3)]
    for path in audio_paths:
        path.write_bytes(b"fake")
    default_config = prepare.load_model_config(None)
    counters = {"inits": 0, "separates": 0}

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            counters["inits"] += 1

        def separate(self, path):
            counters["separates"] += 1
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    progress_calls = []

    def fake_tqdm(items, **kwargs):
        progress_calls.append(kwargs)
        return list(items)

    def fake_save_audio(path, audio, sample_rate):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", fake_tqdm)

    prepare.process_files(audio_paths, output_root=tmp_path / "out", worker_id=2)

    assert counters == {"inits": 1, "separates": 3}
    assert progress_calls
    assert progress_calls[0]["desc"] == "worker-2"
    assert progress_calls[0]["unit"] == "file"


def test_process_files_does_not_prefilter_all_files_before_first_separation(tmp_path, monkeypatch):
    audio_paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in audio_paths:
        path.write_bytes(b"fake")
    default_config = {
        "model": {"sample_rate": 44100, "stereo": True},
        "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
        "inference": {"num_overlap": 2, "chunk_size": 352800},
    }
    state = {"first_separated": False}

    def fake_separation_status(audio_path, output_root, source_root=None):
        if Path(audio_path).name == "second.wav" and not state["first_separated"]:
            raise AssertionError("second file was checked before first file was separated")
        return None

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            if Path(path).name == "first.wav":
                state["first_separated"] = True
            return prepare.SeparationResult(
                mixture=torch.full((2, 16), 0.5),
                vocals=torch.full((2, 16), 0.2),
                accompaniment=torch.full((2, 16), 0.3),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "separation_status", fake_separation_status)
    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    done, skipped, skiplong, errors = prepare.process_files(
        audio_paths,
        output_root=tmp_path / "out",
        progress=False,
    )

    assert (done, skipped, skiplong, errors) == (2, 0, 0, 0)
    assert state["first_separated"] is True


def test_process_files_cleans_cuda_cache_after_each_file_and_batch(tmp_path, monkeypatch):
    audio_paths = [tmp_path / f"track-{idx}.wav" for idx in range(2)]
    for path in audio_paths:
        path.write_bytes(b"fake")
    default_config = prepare.load_model_config(None)
    cleanup_calls = []

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cuda:0")

        def separate(self, path):
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))
    monkeypatch.setattr(prepare, "_cleanup_cuda_memory", lambda device=None: cleanup_calls.append(str(device)))

    prepare.process_files(audio_paths, output_root=tmp_path / "out", progress=False)

    assert cleanup_calls == ["cuda:0", "cuda:0", "cuda:0"]


def test_process_files_writes_file_progress_events(tmp_path, monkeypatch):
    done_audio = tmp_path / "done.wav"
    pending_audio = tmp_path / "pending.wav"
    done_audio.write_bytes(b"done")
    pending_audio.write_bytes(b"pending")
    output_root = tmp_path / "out"
    progress_file = tmp_path / "progress.jsonl"
    default_config = prepare.load_model_config(None)

    done_layout = prepare.separation_layout(done_audio, output_root)
    done_layout.item_dir.mkdir(parents=True)
    done_layout.mixture_path.write_bytes(b"mixture")
    done_layout.vocals_path.write_bytes(b"vocals")
    done_layout.accompaniment_path.write_bytes(b"accompaniment")
    done_layout.metadata_path.write_text(json.dumps({"status": "done"}), encoding="utf-8")

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    prepare.process_files(
        [done_audio, pending_audio],
        output_root=output_root,
        worker_id=5,
        progress_file=progress_file,
    )
    _, events = prepare._read_progress_events(progress_file, 0)

    assert [(event.status, Path(event.path).name, event.worker_id) for event in events] == [
        ("skipped", "done.wav", 5),
        ("done", "pending.wav", 5),
    ]


def test_process_files_skips_audio_at_or_above_max_duration_before_separation(
    tmp_path,
    monkeypatch,
):
    long_audio = tmp_path / "long.wav"
    short_audio = tmp_path / "short.wav"
    long_audio.write_bytes(b"long")
    short_audio.write_bytes(b"short")
    output_root = tmp_path / "out"
    progress_file = tmp_path / "progress.jsonl"
    default_config = prepare.load_model_config(None)
    separated = []

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            separated.append(Path(path).name)
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(
        prepare,
        "_probe_audio_duration_seconds",
        lambda path: 600.0 if Path(path) == long_audio else 599.0,
        raising=False,
    )
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    done, skipped, skiplong, errors = prepare.process_files(
        [long_audio, short_audio],
        output_root=output_root,
        progress_file=progress_file,
        max_duration_seconds=600,
    )

    assert (done, skipped, skiplong, errors) == (1, 0, 1, 0)
    assert separated == ["short.wav"]
    long_metadata = json.loads(
        prepare.separation_layout(long_audio, output_root).metadata_path.read_text(
            encoding="utf-8"
        )
    )
    assert long_metadata["status"] == "skiplong"
    assert long_metadata["reason"] == "max_duration_exceeded"
    assert long_metadata["duration_seconds"] == 600.0
    _, events = prepare._read_progress_events(progress_file, 0)
    assert [(event.status, Path(event.path).name) for event in events] == [
        ("skiplong", "long.wav"),
        ("done", "short.wav"),
    ]


def test_process_files_reclassifies_existing_error_as_skiplong_when_over_max_duration(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"long")
    output_root = tmp_path / "out"
    progress_file = tmp_path / "progress.jsonl"
    layout = prepare.separation_layout(audio_path, output_root)
    layout.item_dir.mkdir(parents=True)
    layout.metadata_path.write_text(
        json.dumps({"status": "error", "reason": "worker_crash"}),
        encoding="utf-8",
    )
    default_config = prepare.load_model_config(None)
    separated = []

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            separated.append(Path(path).name)

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_probe_audio_duration_seconds", lambda path: 601.0)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    done, skipped, skiplong, errors = prepare.process_files(
        [audio_path],
        output_root=output_root,
        progress_file=progress_file,
        max_duration_seconds=600,
    )

    assert (done, skipped, skiplong, errors) == (0, 0, 1, 0)
    assert separated == []
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "skiplong"
    assert metadata["reason"] == "max_duration_exceeded"
    _, events = prepare._read_progress_events(progress_file, 0)
    assert [(event.status, Path(event.path).name) for event in events] == [
        ("skiplong", "long.wav"),
    ]


def test_launch_workers_marks_started_file_and_restarts_after_worker_crash(
    tmp_path,
    monkeypatch,
):
    bad_audio = tmp_path / "bad.wav"
    good_audio = tmp_path / "good.wav"
    bad_audio.write_bytes(b"bad")
    good_audio.write_bytes(b"good")
    output_root = tmp_path / "out"
    default_config = prepare.load_model_config(None)
    popen_calls = []

    args = prepare.parse_args(
        [
            "--files-list",
            str(tmp_path / "files.txt"),
            "--output-root",
            str(output_root),
            "--source-root",
            str(tmp_path),
            "--devices",
            "0",
            "--worker-restarts",
            "2",
        ]
    )

    class FakeProcess:
        def __init__(self, return_code):
            self.return_code = return_code

        def poll(self):
            return self.return_code

    def fake_popen(command, env, stdout, stderr, text):
        popen_calls.append(command)
        progress_file = Path(command[command.index("--progress-file") + 1])
        if len(popen_calls) == 1:
            prepare._write_progress_event(progress_file, "started", 0, bad_audio)
            return FakeProcess(-11)
        prepare._write_progress_event(progress_file, "done", 0, good_audio)
        return FakeProcess(0)

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.updates = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            self.updates.append(value)

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(prepare.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(prepare, "_probe_audio_duration_seconds", lambda path: None)
    monkeypatch.setattr(prepare, "tqdm", FakeTqdm)

    return_code = prepare.launch_workers(args, [str(bad_audio), str(good_audio)], ["0"])

    assert return_code == 0
    assert len(popen_calls) == 2
    metadata = json.loads(
        prepare.separation_layout(bad_audio, output_root, tmp_path).metadata_path.read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "error"
    assert metadata["reason"] == "worker_crash"
    assert "rc=-11" in metadata["error"]


def test_mark_worker_crash_file_records_skiplong_when_duration_exceeds_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"long")
    output_root = tmp_path / "out"
    default_config = prepare.load_model_config(None)
    args = prepare.parse_args(
        [
            "--files-list",
            str(tmp_path / "files.txt"),
            "--output-root",
            str(output_root),
            "--source-root",
            str(tmp_path),
            "--max-duration-seconds",
            "600",
        ]
    )

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "_probe_audio_duration_seconds", lambda path: 12812.891833)

    marked = prepare._mark_worker_crash_file(args, audio_path, worker_id=0, return_code=-11)

    assert marked is True
    metadata = json.loads(
        prepare.separation_layout(audio_path, output_root, tmp_path).metadata_path.read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "skiplong"
    assert metadata["reason"] == "max_duration_exceeded"
    assert metadata["duration_seconds"] == 12812.891833
    assert metadata["max_duration_seconds"] == 600.0


def test_main_keeps_worker_successful_when_process_files_reports_data_errors_by_default(
    tmp_path,
    monkeypatch,
):
    bad_audio = tmp_path / "bad.wav"
    captured = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_separation",
            "--files-list",
            str(tmp_path / "files.txt"),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(
        prepare,
        "_iter_audio_files",
        lambda input_dirs, files_list, max_files=None: [str(bad_audio)],
    )

    def fake_process_files(files, **kwargs):
        captured["files"] = files
        captured["kwargs"] = kwargs
        return 0, 0, 0, 1

    monkeypatch.setattr(prepare, "process_files", fake_process_files)

    prepare.main()

    assert captured["files"] == [str(bad_audio)]


def test_main_can_fail_on_per_file_errors_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_separation",
            "--files-list",
            str(tmp_path / "files.txt"),
            "--output-root",
            str(tmp_path / "out"),
            "--fail-on-error",
        ],
    )
    monkeypatch.setattr(
        prepare,
        "_iter_audio_files",
        lambda input_dirs, files_list, max_files=None: [str(tmp_path / "bad.wav")],
    )
    monkeypatch.setattr(prepare, "process_files", lambda files, **kwargs: (0, 0, 0, 1))

    with pytest.raises(SystemExit) as exc_info:
        prepare.main()

    assert exc_info.value.code == 1


def test_process_files_overlaps_save_with_next_separation(tmp_path, monkeypatch):
    audio_paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in audio_paths:
        path.write_bytes(b"fake")
    default_config = prepare.load_model_config(None)
    second_separation_started = threading.Event()
    observed = {"overlapped": False}
    counters = {"separates": 0, "saves": 0}
    save_lock = threading.Lock()

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            counters["separates"] += 1
            if counters["separates"] == 2:
                second_separation_started.set()
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        with save_lock:
            counters["saves"] += 1
            should_wait = counters["saves"] == 1
        if should_wait:
            observed["overlapped"] = second_separation_started.wait(timeout=0.3)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    prepare.process_files(audio_paths, output_root=tmp_path / "out", progress=False)

    assert observed["overlapped"] is True


def test_process_files_counts_writes_finished_while_waiting_for_capacity(tmp_path, monkeypatch):
    audio_paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in audio_paths:
        path.write_bytes(b"fake")
    default_config = prepare.load_model_config(None)
    second_separation_done = threading.Event()
    counters = {"separates": 0, "saves": 0}
    save_lock = threading.Lock()

    class FakeSeparator:
        def __init__(self, *args, **kwargs):
            pass

        def separate(self, path):
            counters["separates"] += 1
            if counters["separates"] == 2:
                second_separation_done.set()
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        with save_lock:
            counters["saves"] += 1
            should_wait = counters["saves"] == 1
        if should_wait:
            assert second_separation_done.wait(timeout=1.0)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)
    monkeypatch.setattr(prepare, "tqdm", lambda items, **kwargs: list(items))

    done, skipped, skiplong, errors = prepare.process_files(
        audio_paths,
        output_root=tmp_path / "out",
        progress=False,
        max_pending_writes=1,
    )

    assert (done, skipped, skiplong, errors) == (2, 0, 0, 0)


def test_demix_batches_chunks_to_reduce_gpu_dispatches():
    config = prepare._to_config_view(
            {
                "audio": {"chunk_size": 20},
                "inference": {"num_overlap": 2},
                "training": {"instruments": ["Vocals", "Instrumental"]},
            }
    )
    mixture = torch.linspace(-1.0, 1.0, steps=100).repeat(2, 1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch):
            self.batch_sizes.append(batch.shape[0])
            return torch.stack([batch + 0.25, batch - 0.25], dim=1)

    model = FakeModel()

    stems = prepare._demix(
        model,
        config,
        mixture,
        torch.device("cpu"),
        use_amp=False,
        chunk_batch_size=5,
    )

    assert model.batch_sizes == [5, 5, 2]
    assert set(stems) == {"Vocals", "Instrumental"}
    assert stems["Vocals"].shape == tuple(mixture.shape)


def test_select_vocals_uses_exact_official_stem_name():
    vocals = torch.full((2, 4), 3.0).numpy()

    assert prepare._select_vocals({"vocals": vocals}) is vocals


def test_select_vocals_errors_without_exact_official_stem_name():
    with pytest.raises(RuntimeError, match="vocals"):
        prepare._select_vocals({"Vocals": torch.full((2, 4), 3.0).numpy()})

    with pytest.raises(RuntimeError, match="vocals"):
        prepare._select_vocals({"other": torch.full((2, 4), 3.0).numpy()})


def test_select_accompaniment_subtracts_vocals_from_mixture_even_when_model_outputs_instrumental():
    mixture = torch.full((2, 4), 10.0)
    vocals = torch.full((2, 4), 3.0).numpy()
    instrumental = torch.full((2, 4), 99.0).numpy()

    accompaniment = prepare._select_accompaniment(
        {"Vocals": vocals, "Instrumental": instrumental},
        mixture,
        vocals,
    )

    torch.testing.assert_close(torch.from_numpy(accompaniment), torch.full((2, 4), 7.0))


def test_select_accompaniment_subtracts_vocals_without_accompaniment_stem():
    mixture = torch.tensor([[0.5, -0.5], [0.25, -0.25]])
    vocals = torch.tensor([[0.2, -0.1], [0.1, -0.05]]).numpy()

    accompaniment = prepare._select_accompaniment({"Vocals": vocals}, mixture, vocals)

    torch.testing.assert_close(
        torch.from_numpy(accompaniment),
        torch.tensor([[0.3, -0.4], [0.15, -0.2]]),
    )


def test_config_audio_summary_uses_kimberley_model_section():
    model_config = {
        "model": {"sample_rate": 44100, "stereo": True},
        "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
        "inference": {"num_overlap": 2, "chunk_size": 352800},
    }

    summary = prepare._extract_model_summary(model_config)

    assert summary["sample_rate"] == 44100
    assert summary["num_channels"] == 2
    assert summary["num_overlap"] == 2
    assert summary["chunk_size"] == 352800
    assert summary["dim_t"] is None
    assert summary["instruments"] == ["vocals", "other"]
    assert summary["target_instrument"] == "vocals"


def test_write_separation_result_copies_matching_source_mp3_mixture(tmp_path, monkeypatch):
    audio_path = tmp_path / "source.mp3"
    audio_path.write_bytes(b"original-mp3")
    layout = prepare.separation_layout(audio_path, tmp_path / "out")
    result = prepare.SeparationResult(
        mixture=torch.zeros(2, 16),
        vocals=torch.ones(2, 16),
        accompaniment=torch.full((2, 16), 2.0),
        sample_rate=44100,
        source_sample_rate=44100,
        source_num_channels=2,
    )
    saved = []

    def fake_save_audio(path, audio, sample_rate):
        saved.append(Path(path).name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(Path(path).name.encode())

    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)

    prepare._write_separation_result(
        layout,
        result,
        audio_path,
        prepare.load_model_config(None),
        prepare.ModelFiles(
            config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
            checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
        ),
    )

    assert layout.mixture_path.read_bytes() == b"original-mp3"
    assert saved == ["vocals.mp3", "accompaniment.mp3"]
    assert json.loads(layout.metadata_path.read_text(encoding="utf-8"))["status"] == "done"


def test_process_file_uses_local_roformer_separator_and_writes_stems(tmp_path, monkeypatch):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"fake")
    captured = {}
    default_config = prepare.load_model_config(None)

    class FakeSeparator:
        def __init__(
            self,
            checkpoint_path,
            config_path,
            device=None,
            num_overlap=None,
            use_amp=True,
            chunk_batch_size=1,
        ):
            captured["checkpoint_path"] = Path(checkpoint_path)
            captured["config_path"] = Path(config_path)
            captured["device"] = device
            captured["num_overlap"] = num_overlap
            captured["use_amp"] = use_amp
            captured["chunk_batch_size"] = chunk_batch_size

        def separate(self, path):
            captured["audio_path"] = Path(path)
            return prepare.SeparationResult(
                mixture=torch.zeros(2, 16),
                vocals=torch.ones(2, 16),
                accompaniment=torch.full((2, 16), 2.0),
                sample_rate=44100,
            )

    saved = {}

    def fake_save_audio(path, audio, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        saved[Path(path).name] = (audio.clone(), sample_rate)
        path.write_bytes(Path(path).name.encode())

    monkeypatch.setattr(prepare, "load_model_config", lambda config_path: default_config)
    monkeypatch.setattr(prepare, "ensure_model_files", lambda model_dir=None, config_path=None, checkpoint_path=None: prepare.ModelFiles(
        config_path=tmp_path / prepare.DEFAULT_CONFIG_NAME,
        checkpoint_path=tmp_path / prepare.DEFAULT_CHECKPOINT_NAME,
    ))
    monkeypatch.setattr(prepare, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(prepare, "_save_audio", fake_save_audio)

    layout = prepare.process_file(
        audio_path,
        output_root=tmp_path / "out",
        device="cuda:0",
        num_overlap=2,
        use_amp=False,
    )

    assert captured["audio_path"] == audio_path
    assert captured["checkpoint_path"] == tmp_path / prepare.DEFAULT_CHECKPOINT_NAME
    assert captured["config_path"] == tmp_path / prepare.DEFAULT_CONFIG_NAME
    assert captured["device"] == "cuda:0"
    assert captured["num_overlap"] == 2
    assert captured["use_amp"] is False
    assert captured["chunk_batch_size"] == 1
    assert set(saved) == {"mixture.mp3", "vocals.mp3", "accompaniment.mp3"}
    assert layout.metadata_path.exists()
