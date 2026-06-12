from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.tools.local_queue import TaskHandler, TaskRecord, TaskResult
from scripts.tools.oss_tools import OssClient, OssPath


STEM_FILES = ("mixture.mp3", "vocals.mp3", "accompaniment.mp3", "metadata.json")


@dataclass(frozen=True)
class RemoteSeparationPaths:
    input_audio: str
    output_dir: str
    relative_stem: str


def _join_oss_dir(root: str, *parts: str) -> str:
    parsed = OssPath.parse(root)
    clean_parts = [parsed.path.strip("/")]
    clean_parts.extend(part.strip("/") for part in parts if part.strip("/"))
    return f"{parsed.remote}:{'/'.join(part for part in clean_parts if part)}"


def _relative_stem(input_audio: str, source_prefix: str | None) -> str:
    input_path = OssPath.parse(input_audio).path
    if source_prefix:
        source_path = OssPath.parse(source_prefix).path.rstrip("/") + "/"
        if input_path.startswith(source_path):
            input_path = input_path[len(source_path) :]
    return str(Path(input_path).with_suffix(""))


def remote_paths(input_audio: str, output_prefix: str, source_prefix: str | None = None) -> RemoteSeparationPaths:
    relative_stem = _relative_stem(input_audio, source_prefix)
    return RemoteSeparationPaths(
        input_audio=input_audio,
        output_dir=_join_oss_dir(output_prefix, relative_stem),
        relative_stem=relative_stem,
    )


def remote_separation_done(client: OssClient, output_dir: str) -> bool:
    return all(client.exists(_join_oss_dir(output_dir, name)) for name in STEM_FILES)


def _iter_inputs(config: Mapping[str, Any]) -> Iterable[str]:
    for item in config.get("inputs", []):
        yield str(item)
    inputs_file = config.get("inputs_file")
    if inputs_file:
        for line in Path(str(inputs_file)).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


class OssPrepareSeparationHandler(TaskHandler):
    """Local-queue handler: OSS download -> local separation -> OSS upload -> cleanup.

    Config shape:

    {
      "inputs": ["qz_oss2:bucket/audio/a.mp3"],
      "inputs_file": "oss_inputs.txt",
      "source_prefix": "qz_oss2:bucket/audio",
      "output_prefix": "qz_oss2:bucket/prepared",
      "config_hash": "separation_v1",
      "model_dir": "checkpoints/mel-band-roformer-vocal-model",
      "device": "cuda:0"
    }
    """

    task_type = "oss_prepare_separation"

    def build_tasks(self, root: Path, job: str, config: Mapping[str, Any]) -> Iterable[TaskRecord]:
        del root
        output_prefix = str(config["output_prefix"])
        source_prefix = config.get("source_prefix")
        config_hash = str(config.get("config_hash", job))
        for input_audio in _iter_inputs(config):
            paths = remote_paths(str(input_audio), output_prefix, str(source_prefix) if source_prefix else None)
            raw_id = f"{paths.input_audio}|{paths.output_dir}|{config_hash}"
            yield TaskRecord(
                task_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
                task_type=self.task_type,
                payload={
                    "input_audio": paths.input_audio,
                    "output_dir": paths.output_dir,
                    "relative_stem": paths.relative_stem,
                    "config": dict(config),
                },
            )

    def process(
        self,
        task: TaskRecord,
        *,
        root: Path,
        job: str,
        worker_id: str,
        device: str,
        tmp_dir: Path,
    ) -> TaskResult:
        del root, job
        payload = dict(task.payload)
        config = dict(payload.get("config", {}))
        input_audio = str(payload["input_audio"])
        output_dir = str(payload["output_dir"])

        client = OssClient(timeout_seconds=float(config.get("oss_timeout_seconds", 3600)))
        work_dir = tmp_dir / task.task_id
        input_dir = work_dir / "input"
        output_root = work_dir / "output"
        receipt = tmp_dir / f"{task.task_id}.receipt.json"
        local_audio = input_dir / Path(OssPath.parse(input_audio).path).name

        try:
            if not config.get("overwrite", False) and remote_separation_done(client, output_dir):
                return _write_receipt(receipt, task, input_audio, output_dir, skipped=True)

            input_dir.mkdir(parents=True, exist_ok=True)
            client.download_file(input_audio, local_audio)

            from scripts.data.prepare_separation import process_file

            layout = process_file(
                local_audio,
                output_root=output_root,
                source_root=input_dir,
                model_dir=config.get("model_dir"),
                config_path=config.get("config_path"),
                checkpoint_path=config.get("checkpoint_path"),
                device=str(config.get("device", device)),
                num_overlap=config.get("num_overlap", 2),
                use_amp=not bool(config.get("no_amp", False)),
                chunk_batch_size=int(config.get("chunk_batch_size", 1)),
                overwrite=True,
            )

            for name in STEM_FILES:
                client.upload_file(layout.item_dir / name, _join_oss_dir(output_dir, name))

            return _write_receipt(receipt, task, input_audio, output_dir, skipped=False)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _write_receipt(
    receipt: Path,
    task: TaskRecord,
    input_audio: str,
    output_dir: str,
    *,
    skipped: bool,
) -> TaskResult:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_id": task.task_id,
        "input_audio": input_audio,
        "output_dir": output_dir,
        "skipped": skipped,
    }
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TaskResult(output_path=receipt, metadata=metadata)
