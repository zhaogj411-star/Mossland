from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from .api import TaskRecord
from .io import atomic_write_json, read_json
from .paths import done_path, failed_path, lease_dir, lease_path


def now() -> float:
    return time.time()


def lease_expired(path: Path, default_ttl: int) -> bool:
    try:
        lease = read_json(path)
    except Exception:
        return True

    updated_at = float(lease.get("updated_at", 0))
    ttl = int(lease.get("ttl_seconds", default_ttl))
    return now() - updated_at > ttl


def acquire_task(root: Path, job: str, task: TaskRecord, worker_id: str, lease_ttl: int) -> bool:
    if done_path(root, job, task.task_id).exists() or failed_path(root, job, task.task_id).exists():
        return False

    ldir = lease_dir(root, job, task.task_id)
    lpath = lease_path(root, job, task.task_id)
    ldir.parent.mkdir(parents=True, exist_ok=True)

    try:
        ldir.mkdir()
    except FileExistsError:
        if not lease_expired(lpath, lease_ttl):
            return False
        expired = ldir.with_name(f"{ldir.name}.expired.{int(now())}.{uuid.uuid4().hex[:8]}")
        try:
            os.replace(ldir, expired)
        except OSError:
            return False
        try:
            ldir.mkdir()
        except FileExistsError:
            return False

    atomic_write_json(
        lpath,
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "worker_id": worker_id,
            "updated_at": now(),
            "ttl_seconds": lease_ttl,
        },
    )
    return True


def owns_lease(root: Path, job: str, task_id: str, worker_id: str) -> bool:
    try:
        lease = read_json(lease_path(root, job, task_id))
    except Exception:
        return False
    return lease.get("worker_id") == worker_id


def refresh_lease(root: Path, job: str, task_id: str, worker_id: str, lease_ttl: int) -> bool:
    lpath = lease_path(root, job, task_id)
    try:
        lease = read_json(lpath)
        if lease.get("worker_id") != worker_id:
            return False
        lease["updated_at"] = now()
        lease["ttl_seconds"] = lease_ttl
        atomic_write_json(lpath, lease)
        return True
    except Exception:
        return False


def release_lease(root: Path, job: str, task_id: str, worker_id: str) -> None:
    ldir = lease_dir(root, job, task_id)
    lpath = lease_path(root, job, task_id)
    try:
        lease = read_json(lpath)
        if lease.get("worker_id") != worker_id:
            return
    except Exception:
        return

    try:
        lpath.unlink(missing_ok=True)
        ldir.rmdir()
    except OSError:
        pass
