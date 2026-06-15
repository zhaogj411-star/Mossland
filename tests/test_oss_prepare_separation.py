from __future__ import annotations

import json

from scripts.data_processing import oss_prepare_separation as oss_prepare
from scripts.data_processing import separation_core
from scripts.tools.local_queue import TaskRecord


class FakeOssClient:
    uploaded: dict[str, bytes] = {}

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def exists(self, oss_path):
        return False

    def download_file(self, oss_path, local_path):
        del oss_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake audio")

    def upload_file(self, local_path, oss_path):
        self.uploaded[str(oss_path)] = local_path.read_bytes()


def test_oss_prepare_marks_overlong_audio_as_skiplong(monkeypatch, tmp_path):
    FakeOssClient.uploaded = {}
    monkeypatch.setattr(oss_prepare, "OssClient", FakeOssClient)
    monkeypatch.setattr(oss_prepare.separation_core, "probe_audio_duration_seconds", lambda path: 601.0)

    handler = oss_prepare.OssPrepareSeparationHandler()
    task = TaskRecord(
        task_id="task-long",
        task_type=handler.task_type,
        payload={
            "input_audio": "qz_oss2:bucket/audio/long.mp3",
            "output_dir": "qz_oss2:bucket/prepared/audio/long",
            "relative_stem": "audio/long",
            "config": {"max_duration_seconds": 600},
        },
    )

    result = handler.process(
        task,
        root=tmp_path / "queue",
        job="job",
        worker_id="worker-0",
        device="cuda:0",
        tmp_dir=tmp_path / "tmp",
    )

    assert result.metadata["status"] == "skiplong"
    assert result.metadata["skipped"] is False
    assert result.metadata["duration_seconds"] == 601.0

    uploaded_metadata = json.loads(
        FakeOssClient.uploaded["qz_oss2:bucket/prepared/audio/long/metadata.json"].decode("utf-8")
    )
    assert uploaded_metadata["status"] == "skiplong"
    assert uploaded_metadata["reason"] == "max_duration_exceeded"
    assert uploaded_metadata["duration_seconds"] == 601.0
    assert uploaded_metadata["max_duration_seconds"] == 600.0
    assert list(FakeOssClient.uploaded) == ["qz_oss2:bucket/prepared/audio/long/metadata.json"]


def test_oss_prepare_reuses_separator_for_multiple_tasks(monkeypatch, tmp_path):
    FakeOssClient.uploaded = {}
    constructed = []
    separated = []
    cleaned = []

    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "model.ckpt"
    config_path.write_text("config", encoding="utf-8")
    checkpoint_path.write_bytes(b"ckpt")
    model_files = separation_core.ModelFiles(config_path=config_path, checkpoint_path=checkpoint_path)
    model_config = {
        "model": {"sample_rate": 44100, "stereo": True},
        "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
        "inference": {"num_overlap": 2, "chunk_size": 352800},
    }

    class FakeSeparator:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.device = kwargs["device"]

        def separate(self, audio_path):
            separated.append(str(audio_path))
            return object()

    def fake_write_separation_result(
        layout,
        result,
        audio_path,
        received_model_config,
        received_model_files,
        source_oss_path=None,
        save_workers=3,
    ):
        del result, audio_path, save_workers
        assert received_model_config == model_config
        assert received_model_files == model_files
        layout.item_dir.mkdir(parents=True, exist_ok=True)
        layout.mixture_path.write_bytes(b"mixture")
        layout.vocals_path.write_bytes(b"vocals")
        layout.accompaniment_path.write_bytes(b"accompaniment")
        layout.metadata_path.write_text(
            json.dumps({"status": "done", "source_oss_path": source_oss_path}),
            encoding="utf-8",
        )

    monkeypatch.setattr(oss_prepare, "OssClient", FakeOssClient)
    monkeypatch.setattr(oss_prepare.separation_core, "probe_audio_duration_seconds", lambda path: 1.0)
    monkeypatch.setattr(oss_prepare.separation_core, "ensure_model_files", lambda **kwargs: model_files)
    monkeypatch.setattr(oss_prepare.separation_core, "load_model_config", lambda path: model_config)
    monkeypatch.setattr(oss_prepare.separation_core, "config_with_num_overlap", lambda config, num_overlap: config)
    monkeypatch.setattr(oss_prepare.separation_core, "RoformerSeparator", FakeSeparator)
    monkeypatch.setattr(oss_prepare.separation_core, "write_separation_result", fake_write_separation_result)
    monkeypatch.setattr(oss_prepare.separation_core, "cleanup_cuda_memory", lambda device: cleaned.append(device))

    handler = oss_prepare.OssPrepareSeparationHandler()
    common_config = {
        "max_duration_seconds": 600,
        "model_dir": str(tmp_path),
        "device": "cuda:0",
        "chunk_batch_size": 1,
    }
    tasks = [
        TaskRecord(
            task_id="task-a",
            task_type=handler.task_type,
            payload={
                "input_audio": "qz_oss2:bucket/audio/a.webm",
                "output_dir": "qz_oss2:bucket/prepared/audio/a",
                "relative_stem": "audio/a",
                "config": common_config,
            },
        ),
        TaskRecord(
            task_id="task-b",
            task_type=handler.task_type,
            payload={
                "input_audio": "qz_oss2:bucket/audio/b.mp3",
                "output_dir": "qz_oss2:bucket/prepared/audio/b",
                "relative_stem": "audio/b",
                "config": common_config,
            },
        ),
    ]

    results = [
        handler.process(
            task,
            root=tmp_path / "queue",
            job="job",
            worker_id="worker-0",
            device="cuda:0",
            tmp_dir=tmp_path / "tmp",
        )
        for task in tasks
    ]

    assert [result.metadata["status"] for result in results] == ["done", "done"]
    assert len(constructed) == 1
    assert len(separated) == 2
    assert cleaned == ["cuda:0", "cuda:0"]

    first_metadata = json.loads(
        FakeOssClient.uploaded["qz_oss2:bucket/prepared/audio/a/metadata.json"].decode("utf-8")
    )
    second_metadata = json.loads(
        FakeOssClient.uploaded["qz_oss2:bucket/prepared/audio/b/metadata.json"].decode("utf-8")
    )
    assert first_metadata["source_oss_path"] == "qz_oss2:bucket/audio/a.webm"
    assert second_metadata["source_oss_path"] == "qz_oss2:bucket/audio/b.mp3"


def test_data_processing_audio_extensions_include_video_containers():
    assert ".webm" in separation_core.AUDIO_EXTENSIONS
    assert ".mp4" in separation_core.AUDIO_EXTENSIONS
