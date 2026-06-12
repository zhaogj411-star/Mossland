from __future__ import annotations

import os
import random
import socket
import threading
import time
import uuid
from pathlib import Path

from .api import TaskHandler, TaskRecord, TaskResult
from .io import atomic_write_json, read_json
from .lease import acquire_task, now, refresh_lease, release_lease
from .manifest import iter_tasks_shuffled
from .paths import done_path, error_log_path, heartbeat_path, output_dir, tmp_worker_dir


class RetryableTaskError(Exception):
    """Raise this from handlers when a task should be retried."""


class FatalTaskError(Exception):
    """Raise this from handlers when a task should be marked failed."""


class LostLeaseError(Exception):
    """Raised when a worker finishes after another worker has reclaimed its lease."""


def make_worker_id(device: str) -> str:
    host = socket.gethostname()
    return f"{host}-{os.getpid()}-{device.replace(':', '')}-{uuid.uuid4().hex[:8]}"


def write_heartbeat(root: Path, job: str, worker_id: str, device: str, task_id: str | None) -> None:
    atomic_write_json(
        heartbeat_path(root, job, worker_id),
        {
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "device": device,
            "task_id": task_id,
            "updated_at": now(),
        },
    )


def refresh_loop(root: Path, job: str, task_id: str, worker_id: str, lease_ttl: int, interval: int, stop: threading.Event) -> None:
    while not stop.wait(interval):
        if not refresh_lease(root, job, task_id, worker_id, lease_ttl):
            return


def count_error_logs(root: Path, job: str, task_id: str) -> int:
    err_dir = root / "logs" / job / "errors" / task_id[:2]
    if not err_dir.exists():
        return 0
    return len(list(err_dir.glob(f"{task_id}.*.json")))


def log_error(root: Path, job: str, task: TaskRecord, worker_id: str, exc: BaseException) -> int:
    suffix = f"{int(now())}.{worker_id}.{uuid.uuid4().hex[:8]}"
    path = error_log_path(root, job, task.task_id, suffix)
    atomic_write_json(
        path,
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "worker_id": worker_id,
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "time": now(),
        },
    )
    return count_error_logs(root, job, task.task_id)


def mark_failed(root: Path, job: str, task: TaskRecord, worker_id: str, exc: BaseException, attempts: int) -> None:
    atomic_write_json(
        root / "state" / job / "failed" / task.task_id[:2] / f"{task.task_id}.json",
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "worker_id": worker_id,
            "attempts": attempts,
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "time": now(),
        },
    )


def commit_result(
    root: Path,
    job: str,
    task: TaskRecord,
    worker_id: str,
    result: TaskResult,
    lease_ttl: int = 300,
) -> None:
    done = done_path(root, job, task.task_id)
    if done.exists():
        if result.output_path.exists() and result.output_path.is_file():
            result.output_path.unlink(missing_ok=True)
        return

    if not refresh_lease(root, job, task.task_id, worker_id, lease_ttl):
        if result.output_path.exists() and result.output_path.is_file():
            result.output_path.unlink(missing_ok=True)
        raise LostLeaseError(f"worker {worker_id} no longer owns lease for task {task.task_id}")

    final_dir = output_dir(root, job, task.task_id)
    final_dir.mkdir(parents=True, exist_ok=True)
    final = final_dir / result.output_path.name

    if not final.exists():
        os.replace(result.output_path, final)
    elif result.output_path.exists() and result.output_path.is_file():
        result.output_path.unlink(missing_ok=True)

    meta = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "worker_id": worker_id,
        "output": str(final),
        "completed_at": now(),
        "metadata": dict(result.metadata),
    }
    atomic_write_json(final.with_suffix(final.suffix + ".meta.json"), meta)
    atomic_write_json(done, meta)


def run_one_task(
    *,
    root: Path,
    job: str,
    handler: TaskHandler,
    task: TaskRecord,
    worker_id: str,
    device: str,
    lease_ttl: int,
    heartbeat_interval: int,
    max_attempts: int,
) -> None:
    stop = threading.Event()
    lease_thread = threading.Thread(
        target=refresh_loop,
        args=(root, job, task.task_id, worker_id, lease_ttl, heartbeat_interval, stop),
        daemon=True,
    )
    lease_thread.start()

    try:
        write_heartbeat(root, job, worker_id, device, task.task_id)
        tmp_dir = tmp_worker_dir(root, worker_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        result = handler.process(task, root=root, job=job, worker_id=worker_id, device=device, tmp_dir=tmp_dir)
        commit_result(root, job, task, worker_id, result, lease_ttl)
    except LostLeaseError:
        pass
    except FatalTaskError as exc:
        if not refresh_lease(root, job, task.task_id, worker_id, lease_ttl):
            return
        attempts = log_error(root, job, task, worker_id, exc)
        mark_failed(root, job, task, worker_id, exc, attempts)
    except BaseException as exc:
        if not refresh_lease(root, job, task.task_id, worker_id, lease_ttl):
            return
        attempts = log_error(root, job, task, worker_id, exc)
        if attempts >= max_attempts:
            mark_failed(root, job, task, worker_id, exc, attempts)
    finally:
        stop.set()
        release_lease(root, job, task.task_id, worker_id)
        write_heartbeat(root, job, worker_id, device, None)


def select_task(root: Path, job: str, worker_id: str, lease_ttl: int) -> TaskRecord | None:
    for task in iter_tasks_shuffled(root, job):
        if acquire_task(root, job, task, worker_id, lease_ttl):
            return task
    return None


def run_worker(
    *,
    root: Path,
    job: str,
    handler: TaskHandler,
    device: str,
    lease_ttl: int = 300,
    heartbeat_interval: int = 30,
    max_attempts: int = 5,
    idle_min_sleep: int = 10,
    idle_max_sleep: int = 60,
    once: bool = False,
) -> None:
    worker_id = make_worker_id(device)
    while True:
        write_heartbeat(root, job, worker_id, device, None)
        task = select_task(root, job, worker_id, lease_ttl)
        if task is None:
            if once:
                return
            time.sleep(random.randint(idle_min_sleep, idle_max_sleep))
            continue

        if task.task_type != handler.task_type:
            release_lease(root, job, task.task_id, worker_id)
            continue

        run_one_task(
            root=root,
            job=job,
            handler=handler,
            task=task,
            worker_id=worker_id,
            device=device,
            lease_ttl=lease_ttl,
            heartbeat_interval=heartbeat_interval,
            max_attempts=max_attempts,
        )

        if once:
            return
