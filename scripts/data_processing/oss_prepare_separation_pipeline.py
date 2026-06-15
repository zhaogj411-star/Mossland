from __future__ import annotations

import argparse
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.data_processing import oss_prepare_separation as oss_prepare
from scripts.data_processing import separation_core
from scripts.tools.local_queue import TaskRecord, TaskResult
from scripts.tools.local_queue.lease import release_lease
from scripts.tools.local_queue.runner import (
    LostLeaseError,
    commit_result,
    count_error_logs,
    log_error,
    make_worker_id,
    mark_failed,
    refresh_loop,
    select_task,
    write_heartbeat,
)
from scripts.tools.local_queue.paths import tmp_worker_dir
from scripts.tools.oss_tools import OssClient, OssPath


@dataclass
class LeaseState:
    root: Path
    job: str
    task: TaskRecord
    worker_id: str
    device: str
    lease_ttl: int
    heartbeat_interval: int
    stop: threading.Event
    thread: threading.Thread

    @classmethod
    def start(
        cls,
        *,
        root: Path,
        job: str,
        task: TaskRecord,
        worker_id: str,
        device: str,
        lease_ttl: int,
        heartbeat_interval: int,
    ) -> "LeaseState":
        stop = threading.Event()
        thread = threading.Thread(
            target=refresh_loop,
            args=(root, job, task.task_id, worker_id, lease_ttl, heartbeat_interval, stop),
            daemon=True,
        )
        thread.start()
        return cls(
            root=root,
            job=job,
            task=task,
            worker_id=worker_id,
            device=device,
            lease_ttl=lease_ttl,
            heartbeat_interval=heartbeat_interval,
            stop=stop,
            thread=thread,
        )

    def close(self) -> None:
        self.stop.set()
        release_lease(self.root, self.job, self.task.task_id, self.worker_id)


@dataclass(frozen=True)
class PreparedTask:
    task: TaskRecord
    config: dict[str, Any]
    input_audio: str
    output_dir: str
    work_dir: Path
    input_dir: Path
    output_root: Path
    receipt: Path
    local_audio: Path


@dataclass(frozen=True)
class CompletedTask:
    result: TaskResult
    work_dir: Path


@dataclass(frozen=True)
class PipelineTimings:
    started_at: float
    downloaded_at: float
    probed_at: float
    separated_at: float | None = None
    saved_uploaded_at: float | None = None


@dataclass(frozen=True)
class GpuResult:
    prepared: PreparedTask
    result: separation_core.SeparationResult
    cached: oss_prepare.CachedSeparator
    timings: PipelineTimings


class PipelineWorker:
    def __init__(
        self,
        *,
        root: Path,
        job: str,
        device: str,
        lease_ttl: int,
        heartbeat_interval: int,
        max_attempts: int,
        download_workers: int,
        prefetch: int,
        upload_workers: int,
        max_pending_uploads: int,
        once: bool,
    ) -> None:
        self.root = root
        self.job = job
        self.device = device
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = heartbeat_interval
        self.max_attempts = max_attempts
        self.download_workers = max(1, int(download_workers))
        self.prefetch = max(1, int(prefetch))
        self.upload_workers = max(1, int(upload_workers))
        self.max_pending_uploads = max(1, int(max_pending_uploads))
        self.once = once
        self.worker_id = make_worker_id(f"pipeline-{device}")
        self.handler = oss_prepare.OssPrepareSeparationHandler()
        self.tmp_dir = tmp_worker_dir(root, self.worker_id)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.processed = 0

    def run(self) -> None:
        pending_downloads: dict[Future, LeaseState] = {}
        pending_uploads: dict[Future, LeaseState] = {}

        with ThreadPoolExecutor(max_workers=self.download_workers) as download_pool, ThreadPoolExecutor(
            max_workers=self.upload_workers
        ) as upload_pool:
            while True:
                self._drain_completed_uploads(pending_uploads, wait_for_one=False)
                if self.once and self.processed >= 1:
                    return
                self._fill_downloads(download_pool, pending_downloads)

                if not pending_downloads and not pending_uploads:
                    return

                if pending_downloads:
                    done_downloads, _ = wait(list(pending_downloads), timeout=1, return_when=FIRST_COMPLETED)
                    for future in done_downloads:
                        lease = pending_downloads.pop(future)
                        self._handle_download_result(future, lease, upload_pool, pending_uploads)
                else:
                    self._drain_completed_uploads(pending_uploads, wait_for_one=True)

    def _fill_downloads(
        self,
        download_pool: ThreadPoolExecutor,
        pending_downloads: dict[Future, LeaseState],
    ) -> None:
        while len(pending_downloads) < self.prefetch:
            write_heartbeat(self.root, self.job, self.worker_id, self.device, None)
            task = select_task(self.root, self.job, self.worker_id, self.lease_ttl)
            if task is None:
                return
            lease = LeaseState.start(
                root=self.root,
                job=self.job,
                task=task,
                worker_id=self.worker_id,
                device=self.device,
                lease_ttl=self.lease_ttl,
                heartbeat_interval=self.heartbeat_interval,
            )
            pending_downloads[download_pool.submit(self._prepare_task, task)] = lease
            if self.once:
                return

    def _handle_download_result(
        self,
        future: Future,
        lease: LeaseState,
        upload_pool: ThreadPoolExecutor,
        pending_uploads: dict[Future, LeaseState],
    ) -> None:
        try:
            prepared_or_done = future.result()
            if isinstance(prepared_or_done, CompletedTask):
                self._commit_and_close(lease, prepared_or_done.result, prepared_or_done.work_dir)
                return

            while len(pending_uploads) >= self.max_pending_uploads:
                self._drain_completed_uploads(pending_uploads, wait_for_one=True)

            write_heartbeat(self.root, self.job, self.worker_id, self.device, lease.task.task_id)
            gpu_result = self._separate(prepared_or_done)
            pending_uploads[upload_pool.submit(self._save_upload_task, gpu_result)] = lease
        except BaseException as exc:
            self._record_failure_and_close(lease, exc)

    def _drain_completed_uploads(
        self,
        pending_uploads: dict[Future, LeaseState],
        *,
        wait_for_one: bool,
    ) -> None:
        if not pending_uploads:
            return
        if wait_for_one:
            done_uploads, _ = wait(list(pending_uploads), return_when=FIRST_COMPLETED)
        else:
            done_uploads = {future for future in pending_uploads if future.done()}
        for future in done_uploads:
            lease = pending_uploads.pop(future)
            try:
                completed = future.result()
                self._commit_and_close(lease, completed.result, completed.work_dir)
            except BaseException as exc:
                self._record_failure_and_close(lease, exc)

    def _prepare_task(self, task: TaskRecord) -> PreparedTask | CompletedTask:
        payload = dict(task.payload)
        config = dict(payload.get("config", {}))
        input_audio = str(payload["input_audio"])
        output_dir = str(payload["output_dir"])
        client = OssClient(timeout_seconds=float(config.get("oss_timeout_seconds", 3600)))
        work_dir = self.tmp_dir / task.task_id
        input_dir = work_dir / "input"
        output_root = work_dir / "output"
        receipt = self.tmp_dir / f"{task.task_id}.receipt.json"
        local_audio = input_dir / Path(OssPath.parse(input_audio).path).name

        if not config.get("overwrite", False):
            remote_status = oss_prepare.remote_separation_status(client, output_dir, self.tmp_dir, task.task_id)
            if remote_status == "done":
                result = oss_prepare._write_receipt(receipt, task, input_audio, output_dir, status="skipped")
                return CompletedTask(result=result, work_dir=work_dir)
            if remote_status == "skiplong":
                result = oss_prepare._write_receipt(receipt, task, input_audio, output_dir, status="skiplong")
                return CompletedTask(result=result, work_dir=work_dir)

        input_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        client.download_file(input_audio, local_audio)
        downloaded_at = time.perf_counter()

        max_duration_seconds = oss_prepare._normalize_max_duration_seconds(
            config.get("max_duration_seconds", oss_prepare.DEFAULT_MAX_DURATION_SECONDS)
        )
        probed_at = downloaded_at
        if max_duration_seconds is not None:
            duration_seconds = separation_core.probe_audio_duration_seconds(local_audio)
            probed_at = time.perf_counter()
            if duration_seconds is not None and duration_seconds >= max_duration_seconds:
                metadata_path = oss_prepare._write_local_skiplong_metadata(
                    work_dir,
                    input_audio=input_audio,
                    duration_seconds=duration_seconds,
                    max_duration_seconds=max_duration_seconds,
                )
                client.upload_file(metadata_path, oss_prepare._join_oss_dir(output_dir, "metadata.json"))
                print(
                    f"[{self.worker_id}] skiplong OSS audio: {input_audio} "
                    f"duration={duration_seconds:.1f}s max={max_duration_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                result = oss_prepare._write_receipt(
                    receipt,
                    task,
                    input_audio,
                    output_dir,
                    status="skiplong",
                    duration_seconds=duration_seconds,
                    max_duration_seconds=max_duration_seconds,
                )
                return CompletedTask(result=result, work_dir=work_dir)

        prepared = PreparedTask(
            task=task,
            config={**config, "_timing_started_at": started_at, "_timing_downloaded_at": downloaded_at, "_timing_probed_at": probed_at},
            input_audio=input_audio,
            output_dir=output_dir,
            work_dir=work_dir,
            input_dir=input_dir,
            output_root=output_root,
            receipt=receipt,
            local_audio=local_audio,
        )
        return prepared

    def _separate(self, prepared: PreparedTask) -> GpuResult:
        cached = self.handler._get_separator(prepared.config, default_device=self.device)
        result = None
        try:
            result = cached.separator.separate(prepared.local_audio)
            separated_at = time.perf_counter()
            timings = PipelineTimings(
                started_at=float(prepared.config["_timing_started_at"]),
                downloaded_at=float(prepared.config["_timing_downloaded_at"]),
                probed_at=float(prepared.config["_timing_probed_at"]),
                separated_at=separated_at,
            )
            return GpuResult(prepared=prepared, result=result, cached=cached, timings=timings)
        except Exception as exc:
            layout = separation_core.separation_layout(
                prepared.local_audio,
                output_root=prepared.output_root,
                source_root=prepared.input_dir,
            )
            layout.item_dir.mkdir(parents=True, exist_ok=True)
            separation_core.write_metadata(
                layout,
                prepared.local_audio,
                separation_core.DEFAULT_MODEL_REPO,
                cached.model_config,
                status="error",
                model_files=cached.model_files,
                error=str(exc),
                source_oss_path=prepared.input_audio,
            )
            raise
        finally:
            if result is None:
                separation_core.cleanup_cuda_memory(getattr(cached.separator, "device", self.device))

    def _save_upload_task(self, gpu_result: GpuResult) -> CompletedTask:
        prepared = gpu_result.prepared
        cached = gpu_result.cached
        layout = separation_core.separation_layout(
            prepared.local_audio,
            output_root=prepared.output_root,
            source_root=prepared.input_dir,
        )
        separation_core.write_separation_result(
            layout,
            gpu_result.result,
            prepared.local_audio,
            cached.model_config,
            cached.model_files,
            source_oss_path=prepared.input_audio,
            save_workers=int(prepared.config.get("save_workers", 3)),
        )
        gpu_result = GpuResult(
            prepared=prepared,
            result=gpu_result.result,
            cached=cached,
            timings=PipelineTimings(
                started_at=gpu_result.timings.started_at,
                downloaded_at=gpu_result.timings.downloaded_at,
                probed_at=gpu_result.timings.probed_at,
                separated_at=gpu_result.timings.separated_at,
                saved_uploaded_at=None,
            ),
        )
        client = OssClient(timeout_seconds=float(prepared.config.get("oss_timeout_seconds", 3600)))
        upload_started_at = time.perf_counter()
        oss_prepare._upload_stem_files(
            client,
            layout.item_dir,
            prepared.output_dir,
            max_workers=int(prepared.config.get("upload_workers", 4)),
        )
        uploaded_at = time.perf_counter()
        timings = gpu_result.timings
        assert timings.separated_at is not None
        oss_prepare._log_timing_if_needed(
            self.worker_id,
            prepared.input_audio,
            prepared.config,
            total_seconds=uploaded_at - timings.started_at,
            download_seconds=timings.downloaded_at - timings.started_at,
            probe_seconds=timings.probed_at - timings.downloaded_at,
            separate_seconds=timings.separated_at - timings.probed_at,
            save_seconds=upload_started_at - timings.separated_at,
            upload_seconds=uploaded_at - upload_started_at,
        )
        result = oss_prepare._write_receipt(
            prepared.receipt,
            prepared.task,
            prepared.input_audio,
            prepared.output_dir,
            status="done",
        )
        return CompletedTask(result=result, work_dir=prepared.work_dir)

    def _commit_and_close(self, lease: LeaseState, result: TaskResult, work_dir: Path) -> None:
        try:
            commit_result(self.root, self.job, lease.task, self.worker_id, result, self.lease_ttl)
            self.processed += 1
        except LostLeaseError:
            pass
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            lease.close()
            write_heartbeat(self.root, self.job, self.worker_id, self.device, None)

    def _record_failure_and_close(self, lease: LeaseState, exc: BaseException) -> None:
        try:
            attempts = log_error(self.root, self.job, lease.task, self.worker_id, exc)
            if attempts >= self.max_attempts:
                mark_failed(self.root, self.job, lease.task, self.worker_id, exc, attempts)
        finally:
            shutil.rmtree(self.tmp_dir / lease.task.task_id, ignore_errors=True)
            lease.close()
            write_heartbeat(self.root, self.job, self.worker_id, self.device, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lease-ttl", type=int, default=1800)
    parser.add_argument("--heartbeat-interval", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--max-pending-uploads", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = PipelineWorker(
        root=Path(args.root),
        job=args.job,
        device=args.device,
        lease_ttl=args.lease_ttl,
        heartbeat_interval=args.heartbeat_interval,
        max_attempts=args.max_attempts,
        download_workers=args.download_workers,
        prefetch=args.prefetch,
        upload_workers=args.upload_workers,
        max_pending_uploads=args.max_pending_uploads,
        once=args.once,
    )
    print(
        f"pipeline_worker host={socket.gethostname()} worker_id={worker.worker_id} "
        f"device={args.device} prefetch={args.prefetch} "
        f"download_workers={args.download_workers} upload_workers={args.upload_workers}",
        flush=True,
    )
    worker.run()


if __name__ == "__main__":
    main()
