from __future__ import annotations

from pathlib import Path


def shard(task_id: str) -> str:
    return task_id[:2]


def job_dir(root: Path, name: str, job: str) -> Path:
    return root / name / job


def manifest_dir(root: Path, job: str) -> Path:
    return job_dir(root, "manifests", job)


def bucket_dir(root: Path, job: str) -> Path:
    return manifest_dir(root, job) / "buckets"


def done_path(root: Path, job: str, task_id: str) -> Path:
    return job_dir(root, "state", job) / "done" / shard(task_id) / f"{task_id}.json"


def failed_path(root: Path, job: str, task_id: str) -> Path:
    return job_dir(root, "state", job) / "failed" / shard(task_id) / f"{task_id}.json"


def lease_dir(root: Path, job: str, task_id: str) -> Path:
    return job_dir(root, "leases", job) / shard(task_id) / f"{task_id}.lock"


def lease_path(root: Path, job: str, task_id: str) -> Path:
    return lease_dir(root, job, task_id) / "lease.json"


def output_dir(root: Path, job: str, task_id: str) -> Path:
    return job_dir(root, "outputs", job) / shard(task_id)


def tmp_worker_dir(root: Path, worker_id: str) -> Path:
    return root / "tmp" / worker_id


def heartbeat_path(root: Path, job: str, worker_id: str) -> Path:
    return root / "heartbeats" / job / f"{worker_id}.json"


def error_log_path(root: Path, job: str, task_id: str, suffix: str) -> Path:
    return root / "logs" / job / "errors" / shard(task_id) / f"{task_id}.{suffix}.json"
