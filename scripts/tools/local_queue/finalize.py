from __future__ import annotations

from pathlib import Path

from .io import atomic_write_json, read_json
from .manifest import iter_tasks
from .paths import done_path, job_dir


def finalize(root: Path, job: str, allow_failed: bool = False) -> dict:
    total = 0
    done = 0
    missing: list[str] = []
    failed: list[str] = []
    outputs: list[str] = []

    for task in iter_tasks(root, job):
        total += 1
        marker = done_path(root, job, task.task_id)
        if marker.exists():
            done += 1
            meta = read_json(marker)
            outputs.append(str(meta.get("output", "")))
            continue

        fail_marker = root / "state" / job / "failed" / task.task_id[:2] / f"{task.task_id}.json"
        if fail_marker.exists():
            failed.append(task.task_id)
        else:
            missing.append(task.task_id)

    success = not missing and (allow_failed or not failed)
    summary = {
        "job": job,
        "total": total,
        "done": done,
        "failed": len(failed),
        "missing": len(missing),
        "success": success,
        "outputs": outputs,
        "failed_task_ids": failed[:1000],
        "missing_task_ids": missing[:1000],
    }

    final_dir = job_dir(root, "final", job)
    atomic_write_json(final_dir / "summary.json", summary)
    if success:
        atomic_write_json(final_dir / "_SUCCESS", {"job": job, "total": total, "done": done})

    return summary
