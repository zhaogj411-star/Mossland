from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


OFFICIAL_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
BASELINE_PREFIX = "msst"
SUPPORTED_MODEL_TYPES = ("bs_roformer", "mel_band_roformer", "scnet")
TARGET_STEM_BY_TASK = {
    "separate_vocals": "vocals",
    "separate_drums": "drums",
    "separate_bass": "bass",
    "separate_other": "other",
    "separate_accompaniment": "accompaniment",
}
KNOWN_PRETRAINED = {
    "bs_roformer": {
        "config_name": "config_bs_roformer_384_8_2_485100.yaml",
        "checkpoint_name": "model_bs_roformer_ep_17_sdr_9.6568.ckpt",
        "config_url": (
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
        ),
        "checkpoint_url": (
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
        ),
    },
    "scnet": {
        "config_name": "config_musdb18_scnet_xl_more_wide_v5.yaml",
        "checkpoint_name": "model_scnet_ep_36_sdr_10.0891.ckpt",
        "config_url": (
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml"
        ),
        "checkpoint_url": (
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt"
        ),
    },
}
MUSDB_STEMS = ("vocals", "bass", "drums", "other")


class SeparationGroupKey(NamedTuple):
    source_path: str
    start_seconds: float
    duration_seconds: float | None
    sample_rate: int
    seed: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return _repo_root() / "tmp/eval_baseline_refs/music-source-separation-training"


def default_checkpoint_dir() -> Path:
    return default_repo_dir() / "checkpoints"


def default_config_path(repo_dir: Path, model_type: str) -> Path:
    return repo_dir / "configs" / f"config_musdb18_{model_type}.yaml"


def default_checkpoint_path(model_type: str) -> Path:
    info = KNOWN_PRETRAINED.get(model_type)
    name = info["checkpoint_name"] if info else f"{model_type}.ckpt"
    return default_checkpoint_dir() / name


def baseline_name(model_type: str) -> str:
    return f"{BASELINE_PREFIX}_{model_type}"


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


def _safe_temp_stem(index: int, item: EvalItem) -> str:
    keep = [char if char.isalnum() or char in ("-", "_") else "_" for char in item.item_id]
    token = "".join(keep).strip("_") or "item"
    return f"{index:06d}_{token}"


def _prediction_paths(group: list[EvalItem], output_dir: str | Path, model_type: str) -> list[Path]:
    return [baseline_prediction_path(output_dir, baseline_name(model_type), item) for item in group]


def ensure_repo_available(repo_dir: Path) -> None:
    if (repo_dir / "inference.py").is_file() and (repo_dir / "utils" / "settings.py").is_file():
        return
    raise RuntimeError(
        "MSST official source tree is missing. Clone it without running inference, for example: "
        f"`git clone --depth 1 {OFFICIAL_REPO} {repo_dir}`."
    )


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


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config_path:
        return Path(args.config_path)
    return default_config_path(Path(args.repo_dir), args.model_type)


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return default_checkpoint_path(args.model_type)


def ensure_config_and_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    ensure_repo_available(Path(args.repo_dir))
    config_path = resolve_config_path(args)
    checkpoint_path = resolve_checkpoint_path(args)
    info = KNOWN_PRETRAINED.get(args.model_type)

    if args.download_missing and info:
        if not config_path.exists() and config_path.name == info["config_name"]:
            download_file(info["config_url"], config_path, args.no_proxy)
        if not checkpoint_path.exists() and checkpoint_path.name == info["checkpoint_name"]:
            download_file(info["checkpoint_url"], checkpoint_path, args.no_proxy)

    if not config_path.exists():
        raise RuntimeError(
            f"MSST config not found: {config_path}. Pass --config-path explicitly. "
            "For the MUSDB18-HQ BS RoFormer baseline, use "
            "`configs/config_musdb18_bs_roformer.yaml` from the official repo or rerun "
            "with --download-missing to fetch the v1.0.12 inference config."
        )
    if not checkpoint_path.exists():
        hint = ""
        if info:
            hint = f" Rerun with --download-missing to fetch {info['checkpoint_url']}."
        raise RuntimeError(
            f"MSST checkpoint not found: {checkpoint_path}. Pass --checkpoint to a compatible "
            f"{args.model_type} .ckpt file.{hint}"
        )
    return config_path, checkpoint_path


def _device_cli_args(device: str) -> list[str]:
    if device == "cpu":
        return ["--force_cpu"]
    if device.startswith("cuda:"):
        return ["--device_ids", device.split(":", 1)[1]]
    if device == "cuda":
        return ["--device_ids", "0"]
    if device.isdigit():
        return ["--device_ids", device]
    raise ValueError("MSST --device must be cpu, cuda, cuda:N, or an integer GPU id.")


def run_msst_cli(input_dir: Path, store_dir: Path, config_path: Path, checkpoint_path: Path, args: argparse.Namespace) -> None:
    repo_path = Path(args.repo_dir).resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    ensure_repo_available(repo_path)
    command = [
        sys.executable,
        str(repo_path / "inference.py"),
        "--model_type",
        args.model_type,
        "--config_path",
        str(config_path),
        "--start_check_point",
        str(checkpoint_path),
        "--input_folder",
        str(input_dir),
        "--store_dir",
        str(store_dir),
        "--filename_template",
        "{file_name}/{instr}",
        "--pcm_type",
        "FLOAT",
        "--bigshifts",
        str(args.bigshifts),
        "--disable_detailed_pbar",
    ]
    command.extend(_device_cli_args(args.device))
    if args.use_tta:
        command.append("--use_tta")
    env = os.environ.copy()
    repo = str(repo_path)
    env["PYTHONPATH"] = repo if not env.get("PYTHONPATH") else f"{repo}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(command, cwd=repo, env=env, check=True)


def _load_estimate(output_dir: Path, input_stem: str, stem: str, sample_rate: int) -> torch.Tensor | None:
    for suffix in ("wav", "flac"):
        path = output_dir / input_stem / f"{stem}.{suffix}"
        if path.exists():
            return load_audio_segment(path, sample_rate=sample_rate, stereo=True)
    return None


def _fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def estimate_for_task(estimates: dict[str, torch.Tensor], task_id: str) -> torch.Tensor:
    if task_id in TARGET_STEM_BY_TASK and task_id != "separate_accompaniment":
        stem = TARGET_STEM_BY_TASK[task_id]
        if stem not in estimates:
            raise KeyError(f"MSST estimates do not contain {stem!r}: {sorted(estimates)}")
        return estimates[stem]
    if task_id != "separate_accompaniment":
        raise ValueError(f"unsupported separation task_id={task_id!r}")
    if "accompaniment" in estimates:
        return estimates["accompaniment"]
    source_names = [name for name in ("bass", "drums", "other") if name in estimates]
    if not source_names:
        source_names = [name for name in estimates if name != "vocals"]
    if source_names:
        accompaniment = estimates[source_names[0]].clone()
        for name in source_names[1:]:
            accompaniment = accompaniment + estimates[name]
        return accompaniment
    if "vocals" in estimates and "mixture" in estimates:
        return estimates["mixture"] - estimates["vocals"]
    raise KeyError(
        "MSST estimates do not contain accompaniment or non-vocal stems. "
        f"Available estimates: {sorted(estimates)}"
    )


def separate_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    groups = grouped_items(items)
    path_by_item_id: dict[tuple[str, str, int], Path] = {}
    total = len(groups)
    config_path, checkpoint_path = ensure_config_and_checkpoint(args) if groups else (Path(), Path())
    temp_root = Path(args.temp_dir) if args.temp_dir else None

    with tempfile.TemporaryDirectory(prefix="mossland_msst_", dir=temp_root) as tmpdir:
        work_root = Path(tmpdir)
        for group_index, group in enumerate(groups, start=1):
            output_paths = _prediction_paths(group, args.output_dir, args.model_type)
            for item, output_path in zip(group, output_paths, strict=True):
                path_by_item_id[(item.item_id, item.task_id, item.seed)] = output_path
            if all(path.exists() and path.stat().st_size > 0 for path in output_paths) and not args.overwrite:
                maybe_print_progress(group_index, total, group[0].item_id, args.progress_every)
                continue

            first = group[0]
            input_stem = _safe_temp_stem(group_index, first)
            input_dir = work_root / "inputs" / input_stem
            input_dir.mkdir(parents=True, exist_ok=True)
            input_wav = input_dir / f"{input_stem}.wav"
            mixture = load_audio_segment(
                first.source_path,
                sample_rate=first.sample_rate,
                start_seconds=first.start_seconds,
                duration_seconds=first.duration_seconds,
                stereo=True,
            )
            save_audio(input_wav, mixture, first.sample_rate)

            raw_output_dir = work_root / "msst_outputs" / input_stem
            if raw_output_dir.exists():
                shutil.rmtree(raw_output_dir)
            run_msst_cli(input_dir, raw_output_dir, config_path, checkpoint_path, args)

            estimates = {"mixture": mixture}
            for stem in MUSDB_STEMS:
                estimate = _load_estimate(raw_output_dir, input_stem, stem, first.sample_rate)
                if estimate is not None:
                    estimates[stem] = estimate

            for item, output_path in zip(group, output_paths, strict=True):
                prediction = estimate_for_task(estimates, item.task_id)
                if prediction.shape[-1] != mixture.shape[-1]:
                    prediction = _fit_length(prediction, mixture.shape[-1])
                if prediction.shape[0] == 1:
                    prediction = prediction.repeat(2, 1)
                save_audio(output_path, prediction[:2], item.sample_rate)
            maybe_print_progress(group_index, total, first.item_id, args.progress_every)

        if args.keep_temp:
            keep_path = Path(args.keep_temp)
            if keep_path.exists():
                shutil.rmtree(keep_path)
            shutil.copytree(work_root, keep_path)

    return [path_by_item_id[(item.item_id, item.task_id, item.seed)] for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ZFTurbo/MVSep Music-Source-Separation-Training baseline on "
            "MUSDB-style full-track separation manifest rows."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--repo-dir",
        default=str(default_repo_dir()),
        help="Local clone of ZFTurbo/Music-Source-Separation-Training.",
    )
    parser.add_argument(
        "--model-type",
        default="bs_roformer",
        choices=SUPPORTED_MODEL_TYPES,
        help="Official MSST model_type passed to inference.py.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="MSST config YAML. Empty uses configs/config_musdb18_<model-type>.yaml in --repo-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="MSST checkpoint .ckpt. Empty uses a model-type default under the local checkpoint cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cpu, cuda, cuda:N, or integer GPU id.",
    )
    parser.add_argument("--bigshifts", type=int, default=1, help="MSST bigshifts inference option.")
    parser.add_argument("--use-tta", action="store_true", help="Enable MSST test-time augmentation.")
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional parent directory for temporary clipped mixtures and raw MSST outputs.",
    )
    parser.add_argument(
        "--keep-temp",
        default="",
        help="Optional directory to copy temporary MSST inputs/outputs into for debugging.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download known missing config/checkpoint assets when the requested model has a registered URL.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Download known assets without HTTP_PROXY/HTTPS_PROXY/ALL_PROXY.",
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
