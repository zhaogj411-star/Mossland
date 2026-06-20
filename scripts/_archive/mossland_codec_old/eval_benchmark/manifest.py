from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TASK_NAMES = (
    "reconstruct",
    "separate_vocals",
    "separate_drums",
    "separate_bass",
    "separate_other",
    "separate_accompaniment",
    "super_resolution",
    "mono_to_stereo",
)


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    task_id: str
    source_path: Path
    reference_path: Path | None = None
    mixture_path: Path | None = None
    vocals_path: Path | None = None
    accompaniment_path: Path | None = None
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    sample_rate: int = 44100
    low_sample_rate: int | None = None
    seed: int = 0
    prediction_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any], base_dir: Path | None = None) -> "EvalItem":
        task_id = str(row["task_id"])
        if task_id not in TASK_NAMES:
            raise ValueError(f"unsupported task_id={task_id!r}")

        def path_value(name: str) -> Path | None:
            value = row.get(name)
            if value in (None, ""):
                return None
            path = Path(str(value))
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            return path

        metadata = dict(row.get("metadata") or {})
        known = {
            "item_id",
            "task_id",
            "source_path",
            "reference_path",
            "mixture_path",
            "vocals_path",
            "accompaniment_path",
            "start_seconds",
            "duration_seconds",
            "sample_rate",
            "low_sample_rate",
            "seed",
            "prediction_path",
            "metadata",
        }
        for key, value in row.items():
            if key not in known and value not in (None, ""):
                metadata[key] = value

        return cls(
            item_id=str(row.get("item_id") or row.get("id") or Path(str(row["source_path"])).stem),
            task_id=task_id,
            source_path=path_value("source_path") or path_value("mixture_path") or Path(),
            reference_path=path_value("reference_path"),
            mixture_path=path_value("mixture_path"),
            vocals_path=path_value("vocals_path"),
            accompaniment_path=path_value("accompaniment_path"),
            start_seconds=float(row.get("start_seconds") or 0.0),
            duration_seconds=(
                None
                if row.get("duration_seconds") in (None, "")
                else float(row["duration_seconds"])
            ),
            sample_rate=int(row.get("sample_rate") or 44100),
            low_sample_rate=(
                None if row.get("low_sample_rate") in (None, "") else int(row["low_sample_rate"])
            ),
            seed=int(row.get("seed") or 0),
            prediction_path=path_value("prediction_path"),
            metadata=metadata,
        )

    def to_json(self) -> dict[str, Any]:
        def string(path: Path | None) -> str | None:
            return None if path is None else str(path)

        return {
            "item_id": self.item_id,
            "task_id": self.task_id,
            "source_path": string(self.source_path),
            "reference_path": string(self.reference_path),
            "mixture_path": string(self.mixture_path),
            "vocals_path": string(self.vocals_path),
            "accompaniment_path": string(self.accompaniment_path),
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "low_sample_rate": self.low_sample_rate,
            "seed": self.seed,
            "prediction_path": string(self.prediction_path),
            "metadata": self.metadata,
        }


def read_manifest(path: str | Path, base_dir: str | Path | None = None) -> list[EvalItem]:
    path = Path(path)
    resolved_base = Path(base_dir) if base_dir is not None else path.parent
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data["items"] if isinstance(data, dict) and "items" in data else data
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported manifest format: {path}")
    return [EvalItem.from_mapping(row, base_dir=resolved_base) for row in rows]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
