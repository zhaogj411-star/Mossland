from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torchaudio.functional as AF

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


OFFICIAL_REPO = "https://github.com/ByteDance/music_source_separation"
ZENODO_BASE = "https://zenodo.org/record/5804160/files"
BASELINE_PREFIX = "bytesep"
TARGET_SOURCE_BY_TASK = {
    "separate_vocals": "vocals",
    "separate_accompaniment": "accompaniment",
}
CHECKPOINTS = {
    ("MobileNet_Subbandtime", "vocals"): (
        "mobilenet_subbtandtime_vocals_7.2dB_500k_steps_v2.pth",
        4_621_773,
    ),
    ("MobileNet_Subbandtime", "accompaniment"): (
        "mobilenet_subbtandtime_accompaniment_14.6dB_500k_steps_v2.pth",
        4_621_773,
    ),
    ("ResUNet143_Subbandtime", "vocals"): (
        "resunet143_subbtandtime_vocals_8.7dB_500k_steps_v2.pth",
        414_046_363,
    ),
    ("ResUNet143_Subbandtime", "accompaniment"): (
        "resunet143_subbtandtime_accompaniment_16.4dB_500k_steps_v2.pth",
        414_036_369,
    ),
}
CONFIG_NAMES = {
    ("MobileNet_Subbandtime", "vocals"): "vocals-accompaniment,mobilenet_subbandtime.yaml",
    ("MobileNet_Subbandtime", "accompaniment"): "accompaniment-vocals,mobilenet_subbandtime.yaml",
    ("ResUNet143_Subbandtime", "vocals"): "vocals-accompaniment,resunet_subbandtime.yaml",
    ("ResUNet143_Subbandtime", "accompaniment"): "accompaniment-vocals,resunet_subbandtime.yaml",
}


class SeparationGroupKey(NamedTuple):
    source_path: str
    start_seconds: float
    duration_seconds: float | None
    sample_rate: int
    seed: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/bytesep/music_source_separation"


def default_checkpoint_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/bytesep/checkpoints"


def default_bytesep_home() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/bytesep/home"


def baseline_name(model_type: str) -> str:
    return f"{BASELINE_PREFIX}_{model_type.lower()}"


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
    items = [item for item in read_manifest(manifest) if item.task_id in TARGET_SOURCE_BY_TASK]
    if max_items > 0:
        return items[:max_items]
    return items


def grouped_items(items: list[EvalItem]) -> list[list[EvalItem]]:
    groups: dict[SeparationGroupKey, list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[separation_group_key(item)].append(item)
    return [groups[key] for key in sorted(groups)]


def _checkpoint_filename(model_type: str, source_type: str) -> str:
    return CHECKPOINTS[(model_type, source_type)][0]


def _expected_checkpoint_size(model_type: str, source_type: str) -> int:
    return CHECKPOINTS[(model_type, source_type)][1]


def _config_name(model_type: str, source_type: str) -> str:
    return CONFIG_NAMES[(model_type, source_type)]


def config_path(repo_dir: Path, model_type: str, source_type: str) -> Path:
    return repo_dir / "scripts/4_train/musdb18/configs" / _config_name(model_type, source_type)


def checkpoint_path(checkpoint_dir: Path, model_type: str, source_type: str) -> Path:
    return checkpoint_dir / _checkpoint_filename(model_type, source_type)


def _download_url(filename: str) -> str:
    return f"{ZENODO_BASE}/{filename}?download=1"


def _urllib_opener(no_proxy: bool):
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def download_file(url: str, output_path: Path, no_proxy: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    opener = _urllib_opener(no_proxy)
    with opener.open(url, timeout=60) as response, temp_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temp_path.replace(output_path)


def ensure_repo_available(repo_dir: Path) -> None:
    if (repo_dir / "bytesep").is_dir():
        return
    raise RuntimeError(
        "ByteSep official source tree is missing. Clone it without installing old dependencies, for example: "
        f"`git clone --depth 1 {OFFICIAL_REPO} {repo_dir}`."
    )


def prepare_bytesep_home(repo_dir: Path, bytesep_home: Path) -> None:
    filters_src = repo_dir / "bytesep/models/subband_tools/filters"
    filters_dst = bytesep_home / "bytesep_data/filters"
    filters_dst.mkdir(parents=True, exist_ok=True)
    for name in ("f_4_64.mat", "h_4_64.mat"):
        src = filters_src / name
        dst = filters_dst / name
        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
    os.environ["HOME"] = str(bytesep_home.resolve())


def ensure_checkpoint(
    checkpoint_dir: Path,
    model_type: str,
    source_type: str,
    download_missing: bool,
    no_proxy: bool,
) -> Path:
    path = checkpoint_path(checkpoint_dir, model_type, source_type)
    expected_size = _expected_checkpoint_size(model_type, source_type)
    if path.exists() and path.stat().st_size == expected_size:
        return path
    if path.exists() and path.stat().st_size != expected_size:
        if not download_missing:
            raise RuntimeError(
                f"ByteSep checkpoint has unexpected size: {path} is {path.stat().st_size} bytes, "
                f"expected {expected_size}. Rerun with --download-missing to replace it."
            )
        path.unlink()
    if not download_missing:
        raise RuntimeError(
            f"Missing official ByteSep checkpoint {path}. Rerun with --download-missing to fetch "
            f"{_download_url(path.name)}."
        )
    download_file(_download_url(path.name), path, no_proxy=no_proxy)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Downloaded {path} has {actual_size} bytes, expected {expected_size}.")
    return path


def _install_lightning_shim_if_needed() -> None:
    try:
        import pytorch_lightning  # noqa: F401

        return
    except Exception:
        pass

    module = types.ModuleType("pytorch_lightning")
    module.LightningModule = torch.nn.Module
    sys.modules["pytorch_lightning"] = module


def _add_bytesep_to_path(repo_dir: Path) -> None:
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def build_separator(
    repo_dir: Path,
    checkpoint_dir: Path,
    bytesep_home: Path,
    model_type: str,
    source_type: str,
    device: str,
    segment_seconds: float,
    batch_size: int,
    download_missing: bool,
    no_proxy: bool,
):
    ensure_repo_available(repo_dir)
    prepare_bytesep_home(repo_dir, bytesep_home)
    config = config_path(repo_dir, model_type, source_type)
    if not config.exists():
        raise RuntimeError(f"ByteSep config not found: {config}")
    checkpoint = ensure_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        source_type=source_type,
        download_missing=download_missing,
        no_proxy=no_proxy,
    )

    _install_lightning_shim_if_needed()
    _add_bytesep_to_path(repo_dir)

    from bytesep.models.lightning_modules import get_model_class
    from bytesep.separator import Separator
    from bytesep.utils import read_yaml

    configs = read_yaml(str(config))
    train_config = configs["train"]
    sample_rate = int(train_config["sample_rate"])
    input_channels = int(train_config["input_channels"])
    output_channels = int(train_config["output_channels"])
    target_sources_num = len(train_config["target_source_types"])

    model_cls = get_model_class(model_type)
    model = model_cls(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    return Separator(
        model=model,
        segment_samples=int(segment_seconds * sample_rate),
        batch_size=batch_size,
        device=device,
    ), sample_rate, input_channels


def _match_audio_channels(audio: torch.Tensor, input_channels: int) -> torch.Tensor:
    if audio.shape[0] == input_channels:
        return audio
    if audio.shape[0] == 1 and input_channels == 2:
        return audio.repeat(2, 1)
    if audio.shape[0] == 2 and input_channels == 1:
        return audio.mean(dim=0, keepdim=True)
    raise ValueError(f"Cannot match audio shape {tuple(audio.shape)} to {input_channels} input channels.")


def _fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def _prediction_paths(group: list[EvalItem], output_dir: str | Path, model_type: str) -> list[Path]:
    name = baseline_name(model_type)
    return [baseline_prediction_path(output_dir, name, item) for item in group]


def separate_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    groups = grouped_items(items)
    path_by_item_id: dict[tuple[str, str, int], Path] = {}
    total = len(groups)
    repo_dir = Path(args.repo_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    bytesep_home = Path(args.bytesep_home)
    separators = {}

    for group_index, group in enumerate(groups, start=1):
        output_paths = _prediction_paths(group, args.output_dir, args.model_type)
        for item, output_path in zip(group, output_paths, strict=True):
            path_by_item_id[(item.item_id, item.task_id, item.seed)] = output_path
        if all(path.exists() and path.stat().st_size > 0 for path in output_paths) and not args.overwrite:
            maybe_print_progress(group_index, total, group[0].item_id, args.progress_every)
            continue

        first = group[0]
        needed_sources = sorted({TARGET_SOURCE_BY_TASK[item.task_id] for item in group})
        mixture_sample_rate = 44100
        separator_sample_rate = mixture_sample_rate
        mixture = load_audio_segment(
            first.source_path,
            sample_rate=mixture_sample_rate,
            start_seconds=first.start_seconds,
            duration_seconds=first.duration_seconds,
            stereo=True,
        )

        estimates: dict[str, torch.Tensor] = {}
        for source_type in needed_sources:
            key = (args.model_type, source_type)
            if key not in separators:
                separators[key] = build_separator(
                    repo_dir=repo_dir,
                    checkpoint_dir=checkpoint_dir,
                    bytesep_home=bytesep_home,
                    model_type=args.model_type,
                    source_type=source_type,
                    device=args.device,
                    segment_seconds=args.segment_seconds,
                    batch_size=args.batch_size,
                    download_missing=args.download_missing,
                    no_proxy=args.no_proxy,
                )
            separator, model_sample_rate, input_channels = separators[key]
            separator_sample_rate = model_sample_rate
            model_input = mixture
            if mixture_sample_rate != model_sample_rate:
                model_input = AF.resample(model_input, mixture_sample_rate, model_sample_rate)
            model_input = _match_audio_channels(model_input, input_channels)
            estimate = separator.separate({"waveform": model_input.detach().cpu().numpy().astype(np.float32)})
            estimates[source_type] = torch.from_numpy(estimate).float()

        for item, output_path in zip(group, output_paths, strict=True):
            source_type = TARGET_SOURCE_BY_TASK[item.task_id]
            prediction = estimates[source_type]
            if separator_sample_rate != item.sample_rate:
                prediction = AF.resample(prediction, separator_sample_rate, item.sample_rate)
            prediction = _fit_length(prediction, int(round((item.duration_seconds or 0.0) * item.sample_rate)) or mixture.shape[-1])
            save_audio(output_path, prediction, item.sample_rate)
        maybe_print_progress(group_index, total, first.item_id, args.progress_every)

    return [path_by_item_id[(item.item_id, item.task_id, item.seed)] for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official ByteDance ByteSep music_source_separation baseline on MUSDB-style manifests."
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default=str(default_repo_dir()),
        help="Local clone of the official ByteDance/music_source_separation repository.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(default_checkpoint_dir()),
        help="Directory for official Zenodo ByteSep checkpoints.",
    )
    parser.add_argument(
        "--bytesep-home",
        default=str(default_bytesep_home()),
        help="Temporary HOME for ByteSep runtime cache files such as PQMF filters.",
    )
    parser.add_argument(
        "--model-type",
        default="ResUNet143_Subbandtime",
        choices=("ResUNet143_Subbandtime", "MobileNet_Subbandtime"),
        help="Official ByteSep model type. ResUNet matches the official CLI default; MobileNet is faster for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for ByteSep inference.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="ByteSep evaluation segment size; official scripts use 30 seconds.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="ByteSep segment mini-batch size.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing official checkpoints from Zenodo into --checkpoint-dir.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Download checkpoints without HTTP_PROXY/HTTPS_PROXY/ALL_PROXY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)
    items = separation_manifest_items(args.manifest, args.max_items)
    prediction_paths = separate_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
