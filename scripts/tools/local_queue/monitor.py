from __future__ import annotations

from pathlib import Path

from .manifest import iter_tasks


def count_files(path: Path, pattern: str = "*.json") -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def snapshot(root: Path, job: str) -> dict[str, int]:
    total = sum(1 for _ in iter_tasks(root, job))
    done = count_files(root / "state" / job / "done")
    failed = count_files(root / "state" / job / "failed")
    leases = count_files(root / "leases" / job, "lease.json")
    workers = count_files(root / "heartbeats" / job)
    return {
        "total": total,
        "done": done,
        "failed": failed,
        "running_or_leased": leases,
        "workers_seen": workers,
        "remaining": max(total - done - failed, 0),
    }
