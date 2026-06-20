from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.data.datasets import PreparedSeparationDataset
from scripts.mossland_codec.tasks import TASK_NAMES, build_task_batch


DEFAULT_SOURCE_ROOT = (
    "/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/"
    "source_seperation/NETEASE_SPIDER"
)
DEFAULT_OUTPUT_ROOT = "tmp/mossland_codec_debug_pt"


def _info_value(value):
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        if flat.numel() == 1:
            return flat[0].item()
        return flat.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_info_value(item) for item in value]
    return str(value)


def _save_task_pt(out_file: Path, task_payload: dict, info: dict, sample_rate: int):
    payload = {
        "src": task_payload["src"].float().cpu().contiguous(),
        "target": task_payload["target"].float().cpu().contiguous(),
        "task_id": task_payload["task_id"],
        "sample_rate": int(sample_rate),
        "info": {key: _info_value(value) for key, value in info.items()},
    }
    source_path = payload["info"].get("path")
    if source_path is not None:
        payload["source_path"] = source_path
    if "relpath" in payload["info"]:
        payload["relpath"] = payload["info"]["relpath"]
    if "sample_start" in payload["info"]:
        payload["sample_start"] = payload["info"]["sample_start"]
    if "sample_end" in payload["info"]:
        payload["sample_end"] = payload["info"]["sample_end"]
    torch.save(payload, out_file)


def materialize_debug_pt_dataset(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    index_file: str | Path | None = None,
    num_samples: int = 10,
    sample_size: int = 88200,
    sample_rate: int = 44100,
    num_channels: int = 2,
    tasks: tuple[str, ...] = TASK_NAMES,
    low_sample_rate: tuple[int, ...] = (8000, 40000),
    seed: int = 42,
    max_duration_seconds: float | None = 300.0,
    overwrite: bool = False,
) -> Path:
    random.seed(seed)
    torch.manual_seed(seed)

    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = Path(index_file) if index_file is not None else source_root / "index.list"
    files_list_path = output_root / "files.list"
    metadata_path = output_root / "index.jsonl"

    base_dataset = PreparedSeparationDataset(
        dirs=[source_root],
        index_file=index_path,
        sample_size=sample_size,
        sample_rate=sample_rate,
        random_crop=True,
        num_channels=num_channels,
        strict=False,
        max_duration_seconds=max_duration_seconds,
        crops_per_file=1,
        length=None,
    )
    if len(base_dataset) == 0:
        raise RuntimeError(f"No prepared separation items found under {source_root}")

    written_files: list[str] = []
    metadata_lines: list[str] = []
    task_names = tuple(tasks)
    if not task_names:
        raise ValueError("tasks must not be empty")

    for sample_idx in tqdm(range(int(num_samples)), desc="materialize Mossland task pt"):
        task_id = task_names[sample_idx % len(task_names)]
        out_file = output_root / f"{sample_idx:06d}_{task_id}.pt"
        if out_file.exists() and not overwrite:
            written_files.append(out_file.name)
            continue

        payload, info = base_dataset.get_item_for_task(sample_idx, task_id)
        task = build_task_batch(
            payload,
            task_id,
            sample_rate=sample_rate,
            low_sample_rate=low_sample_rate,
        )
        _save_task_pt(out_file, task.to_payload(), info, sample_rate)
        written_files.append(out_file.name)
        metadata_lines.append(
            json.dumps(
                {
                    "pt_path": out_file.name,
                    "task_id": task_id,
                    "source_path": _info_value(info.get("path")),
                    "relpath": _info_value(info.get("relpath")),
                    "sample_start": _info_value(info.get("sample_start")),
                    "sample_end": _info_value(info.get("sample_end")),
                },
                ensure_ascii=True,
            )
        )

    files_list_path.write_text("\n".join(written_files) + "\n", encoding="utf-8")
    if metadata_lines:
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return files_list_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize fixed multi-task Mossland debug samples as .pt files."
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-file", default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=88200)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--num-channels", type=int, default=2)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES))
    parser.add_argument("--low-sample-rate", nargs="+", type=int, default=[8000, 40000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    files_list_path = materialize_debug_pt_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        index_file=args.index_file,
        num_samples=args.num_samples,
        sample_size=args.sample_size,
        sample_rate=args.sample_rate,
        num_channels=args.num_channels,
        tasks=tuple(args.tasks),
        low_sample_rate=tuple(args.low_sample_rate),
        seed=args.seed,
        max_duration_seconds=args.max_duration_seconds,
        overwrite=args.overwrite,
    )
    print(f"Wrote Mossland task pt index: {files_list_path}")


if __name__ == "__main__":
    main()
