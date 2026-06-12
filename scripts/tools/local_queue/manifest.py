from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .api import TaskRecord
from .io import atomic_write_json, append_jsonl
from .paths import bucket_dir, manifest_dir


def task_to_json(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "priority": task.priority,
        "payload": dict(task.payload),
    }


def task_from_json(obj: dict[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=obj["task_id"],
        task_type=obj["task_type"],
        priority=int(obj.get("priority", 0)),
        payload=obj.get("payload", {}),
    )


def write_manifest(root: Path, job: str, tasks: Iterator[TaskRecord], meta: dict[str, Any]) -> int:
    mdir = manifest_dir(root, job)
    if mdir.exists():
        raise FileExistsError(f"manifest already exists for job={job}: {mdir}")

    buckets = bucket_dir(root, job)
    buckets.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    count = 0
    for task in tasks:
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        append_jsonl(buckets / f"{task.task_id[:2]}.jsonl", task_to_json(task))
        count += 1

    meta = dict(meta)
    meta["task_count"] = count
    atomic_write_json(mdir / "manifest.json", meta)
    return count


def iter_tasks(root: Path, job: str) -> Iterator[TaskRecord]:
    buckets = bucket_dir(root, job)
    for path in sorted(buckets.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield task_from_json(json.loads(line))


def iter_tasks_shuffled(root: Path, job: str) -> Iterator[TaskRecord]:
    """Yield tasks in a randomized bucket order without loading all tasks."""

    buckets = list(bucket_dir(root, job).glob("*.jsonl"))
    random.shuffle(buckets)
    for path in buckets:
        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        random.shuffle(lines)
        for line in lines:
            yield task_from_json(json.loads(line))
