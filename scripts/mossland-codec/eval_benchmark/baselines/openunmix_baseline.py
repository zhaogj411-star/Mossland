from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio.functional as AF

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "openunmix_umxhq"
TARGET_STEM_BY_TASK = {
    "separate_vocals": "vocals",
    "separate_accompaniment": "accompaniment",
}


class SeparationGroupKey(NamedTuple):
    source_path: str
    start_seconds: float
    duration_seconds: float | None
    sample_rate: int
    seed: int


def _rounded_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def separation_group_key(item: EvalItem) -> SeparationGroupKey:
    return SeparationGroupKey(
        source_path=str(item.source_path.resolve()),
        start_seconds=float(_rounded_seconds(item.start_seconds) or 0.0),
        duration_seconds=_rounded_seconds(item.duration_seconds),
        sample_rate=item.sample_rate,
        seed=item.seed,
    )


def separation_manifest_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id in TARGET_STEM_BY_TASK]
    if max_items > 0:
        return items[:max_items]
    return items


def grouped_items(items: list[EvalItem]) -> list[list[EvalItem]]:
    groups: dict[SeparationGroupKey, list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[separation_group_key(item)].append(item)
    return [groups[key] for key in sorted(groups)]


def ensure_openunmix_available() -> None:
    if importlib.util.find_spec("openunmix") is None:
        raise RuntimeError(
            "Open-Unmix is not installed in this Python environment. Install the official "
            "package with `python -m pip install openunmix musdb`, then rerun this baseline. "
            "The adapter uses `openunmix.utils.load_separator()` and "
            "`openunmix.predict.separate()` with the official pretrained model name, "
            "defaulting to `umxhq`."
        )


def load_openunmix_separator(args: argparse.Namespace):
    ensure_openunmix_available()
    import openunmix.utils

    targets = None if not args.targets else [target.strip() for target in args.targets.split(",") if target.strip()]
    try:
        separator = openunmix.utils.load_separator(
            model_str_or_path=args.model,
            targets=targets,
            niter=args.niter,
            residual=args.residual,
            wiener_win_len=args.wiener_win_len,
            device=args.device,
            pretrained=True,
            filterbank=args.filterbank,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the official Open-Unmix separator. Check that the model name is "
            f"valid for this openunmix version (requested {args.model!r}) and that pretrained "
            "weights can be downloaded or found in the local torch/hub cache. Typical install: "
            "`python -m pip install openunmix musdb`."
        ) from exc
    separator.freeze()
    if args.device:
        separator.to(args.device)
    return separator


def run_openunmix_separate(
    audio: torch.Tensor,
    sample_rate: int,
    separator,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    ensure_openunmix_available()
    import openunmix.predict

    aggregate_dict = None
    if args.aggregate_accompaniment:
        aggregate_dict = {"vocals": ["vocals"], "accompaniment": ["drums", "bass", "other"]}

    try:
        return openunmix.predict.separate(
            audio,
            rate=sample_rate,
            model_str_or_path=args.model,
            targets=None,
            niter=args.niter,
            residual=args.residual,
            wiener_win_len=args.wiener_win_len,
            aggregate_dict=aggregate_dict,
            separator=separator,
            device=args.device,
            filterbank=args.filterbank,
        )
    except Exception as exc:
        raise RuntimeError(
            "Open-Unmix separation failed. The adapter expects the official API "
            "`openunmix.predict.separate(audio, rate=..., separator=...)` to return a dict "
            "of estimates such as vocals/drums/bass/other or vocals/accompaniment."
        ) from exc


def _estimate_to_audio(estimate: torch.Tensor) -> torch.Tensor:
    if estimate.ndim == 3:
        if estimate.shape[0] != 1:
            raise ValueError(f"expected a single Open-Unmix batch item, got shape {tuple(estimate.shape)}")
        estimate = estimate[0]
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)
    if estimate.ndim != 2:
        raise ValueError(f"Open-Unmix estimate must have shape [C,T] or [1,C,T], got {tuple(estimate.shape)}")
    if estimate.shape[0] == 1:
        estimate = estimate.repeat(2, 1)
    return estimate[:2]


def estimate_for_task(estimates: dict[str, torch.Tensor], task_id: str) -> torch.Tensor:
    if task_id == "separate_vocals":
        if "vocals" not in estimates:
            raise KeyError(f"Open-Unmix estimates do not contain 'vocals': {sorted(estimates)}")
        return _estimate_to_audio(estimates["vocals"])

    if task_id != "separate_accompaniment":
        raise ValueError(f"unsupported separation task_id={task_id!r}")

    for key in ("accompaniment", "no_vocals"):
        if key in estimates:
            return _estimate_to_audio(estimates[key])

    source_names = [key for key in ("drums", "bass", "other", "residual") if key in estimates]
    if not source_names:
        source_names = [key for key in estimates if key != "vocals"]
    if not source_names:
        raise KeyError(
            "Open-Unmix estimates do not contain accompaniment/no_vocals or any non-vocals stems. "
            f"Available estimates: {sorted(estimates)}"
        )

    accompaniment = _estimate_to_audio(estimates[source_names[0]]).clone()
    for name in source_names[1:]:
        accompaniment = accompaniment + _estimate_to_audio(estimates[name])
    return accompaniment


def _fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def _model_sample_rate(separator, fallback: int) -> int:
    return int(getattr(separator, "sample_rate", fallback) or fallback)


def _group_prediction_paths(group: list[EvalItem], output_dir: str | Path) -> list[Path]:
    return [baseline_prediction_path(output_dir, BASELINE_NAME, item) for item in group]


def separate_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    groups = grouped_items(items)
    path_by_item_id: dict[tuple[str, str, int], Path] = {}
    total = len(groups)
    separator = load_openunmix_separator(args) if groups else None

    for group_index, group in enumerate(groups, start=1):
        output_paths = _group_prediction_paths(group, args.output_dir)
        for item, output_path in zip(group, output_paths, strict=True):
            path_by_item_id[(item.item_id, item.task_id, item.seed)] = output_path
        if all(path.exists() and path.stat().st_size > 0 for path in output_paths) and not args.overwrite:
            maybe_print_progress(group_index, total, group[0].item_id, args.progress_every)
            continue

        first = group[0]
        mixture = load_audio_segment(
            first.source_path,
            sample_rate=first.sample_rate,
            start_seconds=first.start_seconds,
            duration_seconds=first.duration_seconds,
            stereo=True,
        )
        estimates = run_openunmix_separate(mixture, first.sample_rate, separator, args)
        model_sample_rate = _model_sample_rate(separator, first.sample_rate)

        for item, output_path in zip(group, output_paths, strict=True):
            prediction = estimate_for_task(estimates, item.task_id)
            if model_sample_rate != item.sample_rate:
                prediction = AF.resample(prediction.detach().cpu(), model_sample_rate, item.sample_rate)
            prediction = _fit_length(prediction, mixture.shape[-1])
            save_audio(output_path, prediction, item.sample_rate)
        maybe_print_progress(group_index, total, first.item_id, args.progress_every)

    return [path_by_item_id[(item.item_id, item.task_id, item.seed)] for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official Open-Unmix/UMXHQ source-separation baseline on MUSDB-style "
            "separation manifest rows."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        default="umxhq",
        help="Official Open-Unmix pretrained model name or local model parent directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device passed to the official Open-Unmix separator.",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Optional comma-separated Open-Unmix targets. Empty loads all targets for the model.",
    )
    parser.add_argument("--niter", type=int, default=1, help="Open-Unmix Wiener filtering EM iterations.")
    parser.add_argument("--wiener-win-len", type=int, default=300, help="Open-Unmix Wiener window length.")
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Ask Open-Unmix to emit a residual target when not all stems are loaded.",
    )
    parser.add_argument(
        "--filterbank",
        default="torch",
        choices=("torch", "asteroid"),
        help="Open-Unmix filterbank implementation.",
    )
    parser.add_argument(
        "--aggregate-accompaniment",
        action="store_true",
        help="Use Open-Unmix aggregate_dict to request vocals/accompaniment directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = separation_manifest_items(args.manifest, args.max_items)
    prediction_paths = separate_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
