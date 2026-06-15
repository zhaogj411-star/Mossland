from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.data_processing import separation_core
from scripts.tools.local_queue import TaskHandler, TaskRecord, TaskResult
from scripts.tools.oss_tools import OssClient, OssPath


STEM_FILES = ("mixture.mp3", "vocals.mp3", "accompaniment.mp3", "metadata.json")
DEFAULT_MAX_DURATION_SECONDS = 10 * 60


@dataclass(frozen=True)
class RemoteSeparationPaths:
    input_audio: str
    output_dir: str
    relative_stem: str


@dataclass
class CachedSeparator:
    separator: separation_core.RoformerSeparator
    model_files: separation_core.ModelFiles
    model_config: dict


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


def _remote_metadata_status(client: OssClient, output_dir: str, tmp_dir: Path, task_id: str) -> str | None:
    metadata_oss = _join_oss_dir(output_dir, "metadata.json")
    if not client.exists(metadata_oss):
        return None
    metadata_path = tmp_dir / f"{task_id}.remote_metadata.json"
    client.download_file(metadata_oss, metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    finally:
        metadata_path.unlink(missing_ok=True)
    status = metadata.get("status")
    return str(status) if status is not None else None


def remote_separation_status(client: OssClient, output_dir: str, tmp_dir: Path, task_id: str) -> str | None:
    if remote_separation_done(client, output_dir):
        return "done"
    status = _remote_metadata_status(client, output_dir, tmp_dir, task_id)
    if status == "skiplong":
        return "skiplong"
    return None


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

    def __init__(self) -> None:
        self._separator_cache: dict[tuple[str, str, str, int | None, bool, int], CachedSeparator] = {}

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
            if not config.get("overwrite", False):
                remote_status = remote_separation_status(client, output_dir, tmp_dir, task.task_id)
                if remote_status == "done":
                    return _write_receipt(receipt, task, input_audio, output_dir, status="skipped")
                if remote_status == "skiplong":
                    return _write_receipt(receipt, task, input_audio, output_dir, status="skiplong")

            input_dir.mkdir(parents=True, exist_ok=True)
            started_at = time.perf_counter()
            client.download_file(input_audio, local_audio)
            downloaded_at = time.perf_counter()

            max_duration_seconds = _normalize_max_duration_seconds(
                config.get("max_duration_seconds", DEFAULT_MAX_DURATION_SECONDS)
            )
            probed_at = downloaded_at
            if max_duration_seconds is not None:
                duration_seconds = separation_core.probe_audio_duration_seconds(local_audio)
                probed_at = time.perf_counter()
                if duration_seconds is not None and duration_seconds >= max_duration_seconds:
                    metadata_path = _write_local_skiplong_metadata(
                        work_dir,
                        input_audio=input_audio,
                        duration_seconds=duration_seconds,
                        max_duration_seconds=max_duration_seconds,
                    )
                    client.upload_file(metadata_path, _join_oss_dir(output_dir, "metadata.json"))
                    print(
                        f"[{worker_id}] skiplong OSS audio: {input_audio} "
                        f"duration={duration_seconds:.1f}s max={max_duration_seconds:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    return _write_receipt(
                        receipt,
                        task,
                        input_audio,
                        output_dir,
                        status="skiplong",
                        duration_seconds=duration_seconds,
                        max_duration_seconds=max_duration_seconds,
                    )

            cached = self._get_separator(config, default_device=str(device))
            layout = separation_core.separation_layout(
                local_audio,
                output_root=output_root,
                source_root=input_dir,
            )
            result = None
            try:
                result = cached.separator.separate(local_audio)
                separated_at = time.perf_counter()
                separation_core.write_separation_result(
                    layout,
                    result,
                    local_audio,
                    cached.model_config,
                    cached.model_files,
                    source_oss_path=input_audio,
                    save_workers=int(config.get("save_workers", 3)),
                )
                saved_at = time.perf_counter()
            except Exception as exc:
                layout.item_dir.mkdir(parents=True, exist_ok=True)
                separation_core.write_metadata(
                    layout,
                    local_audio,
                    separation_core.DEFAULT_MODEL_REPO,
                    cached.model_config,
                    status="error",
                    model_files=cached.model_files,
                    error=str(exc),
                    source_oss_path=input_audio,
                )
                raise
            finally:
                result = None
                separation_core.cleanup_cuda_memory(cached.separator.device)

            upload_started_at = time.perf_counter()
            _upload_stem_files(
                client,
                layout.item_dir,
                output_dir,
                max_workers=int(config.get("upload_workers", 4)),
            )
            uploaded_at = time.perf_counter()
            _log_timing_if_needed(
                worker_id,
                input_audio,
                config,
                total_seconds=uploaded_at - started_at,
                download_seconds=downloaded_at - started_at,
                probe_seconds=probed_at - downloaded_at,
                separate_seconds=separated_at - probed_at,
                save_seconds=saved_at - separated_at,
                upload_seconds=uploaded_at - upload_started_at,
            )

            return _write_receipt(receipt, task, input_audio, output_dir, status="done")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _get_separator(self, config: Mapping[str, Any], default_device: str) -> CachedSeparator:
        num_overlap_value = config.get("num_overlap", separation_core.DEFAULT_NUM_OVERLAP)
        num_overlap = None if num_overlap_value is None else int(num_overlap_value)
        use_amp = not bool(config.get("no_amp", False))
        chunk_batch_size = int(config.get("chunk_batch_size", 1))
        device = str(config.get("device", default_device))
        model_files = separation_core.ensure_model_files(
            model_dir=config.get("model_dir"),
            config_path=config.get("config_path"),
            checkpoint_path=config.get("checkpoint_path"),
        )
        key = (
            str(model_files.config_path.resolve()),
            str(model_files.checkpoint_path.resolve()),
            device,
            num_overlap,
            use_amp,
            chunk_batch_size,
        )
        cached = self._separator_cache.get(key)
        if cached is not None:
            return cached

        model_config = separation_core.config_with_num_overlap(
            separation_core.load_model_config(model_files.config_path),
            num_overlap,
        )
        separator = separation_core.RoformerSeparator(
            checkpoint_path=model_files.checkpoint_path,
            config_path=model_files.config_path,
            device=device,
            num_overlap=num_overlap,
            use_amp=use_amp,
            chunk_batch_size=chunk_batch_size,
        )
        cached = CachedSeparator(separator=separator, model_files=model_files, model_config=model_config)
        self._separator_cache[key] = cached
        return cached


def _normalize_max_duration_seconds(max_duration_seconds: Any) -> float | None:
    if max_duration_seconds is None:
        return None
    max_duration = float(max_duration_seconds)
    if max_duration <= 0:
        return None
    return max_duration


def _write_local_skiplong_metadata(
    work_dir: Path,
    *,
    input_audio: str,
    duration_seconds: float,
    max_duration_seconds: float,
) -> Path:
    metadata_path = work_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "skiplong",
        "source_path": input_audio,
        "reason": "max_duration_exceeded",
        "duration_seconds": float(duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def _upload_stem_files(
    client: OssClient,
    item_dir: Path,
    output_dir: str,
    *,
    max_workers: int = 4,
) -> None:
    max_workers = max(1, int(max_workers))
    uploads = [(item_dir / name, _join_oss_dir(output_dir, name)) for name in STEM_FILES]
    if max_workers == 1:
        for local_path, remote_path in uploads:
            client.upload_file(local_path, remote_path)
        return
    with ThreadPoolExecutor(max_workers=min(max_workers, len(uploads))) as executor:
        futures = [executor.submit(client.upload_file, local_path, remote_path) for local_path, remote_path in uploads]
        for future in futures:
            future.result()


def _log_timing_if_needed(
    worker_id: str,
    input_audio: str,
    config: Mapping[str, Any],
    *,
    total_seconds: float,
    download_seconds: float,
    probe_seconds: float,
    separate_seconds: float,
    save_seconds: float,
    upload_seconds: float,
) -> None:
    threshold = float(config.get("timing_log_threshold_seconds", 60))
    if not bool(config.get("log_timing", False)) and total_seconds < threshold:
        return
    print(
        f"[{worker_id}] timing total={total_seconds:.2f}s "
        f"download={download_seconds:.2f}s probe={probe_seconds:.2f}s "
        f"separate={separate_seconds:.2f}s save={save_seconds:.2f}s "
        f"upload={upload_seconds:.2f}s input={input_audio}",
        file=sys.stderr,
        flush=True,
    )


def _write_receipt(
    receipt: Path,
    task: TaskRecord,
    input_audio: str,
    output_dir: str,
    *,
    status: str,
    duration_seconds: float | None = None,
    max_duration_seconds: float | None = None,
) -> TaskResult:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_id": task.task_id,
        "input_audio": input_audio,
        "output_dir": output_dir,
        "status": status,
        "skipped": status == "skipped",
    }
    if duration_seconds is not None:
        metadata["duration_seconds"] = float(duration_seconds)
    if max_duration_seconds is not None:
        metadata["max_duration_seconds"] = float(max_duration_seconds)
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TaskResult(output_path=receipt, metadata=metadata)
