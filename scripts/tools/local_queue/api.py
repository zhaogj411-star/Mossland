from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TaskRecord:
    """A stable unit of work stored in the manifest."""

    task_id: str
    task_type: str
    payload: Mapping[str, Any]
    priority: int = 0


@dataclass(frozen=True)
class TaskResult:
    """Result returned by a task handler."""

    output_path: Path
    metadata: Mapping[str, Any]


class TaskHandler:
    """Base class for pluggable data processing tasks.

    Subclasses define a task type. The queue owns scheduling, leases, retries,
    idempotent commits, and monitoring. Handlers only define how tasks are
    created and processed.
    """

    task_type: str = "base"

    def build_tasks(self, root: Path, job: str, config: Mapping[str, Any]) -> Iterable[TaskRecord]:
        """Yield tasks for a job manifest."""

        raise NotImplementedError

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
        """Run one task and write its output under tmp_dir."""

        raise NotImplementedError
