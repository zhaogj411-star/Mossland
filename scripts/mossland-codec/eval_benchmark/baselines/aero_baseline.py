from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..audio_io import load_audio_segment, reference_path_for_item, save_audio
from ..manifest import EvalItem, read_manifest, write_jsonl
from .common import add_common_args, baseline_prediction_path, maybe_print_progress


BASELINE_NAME = "aero"
TARGET_SAMPLE_RATE = 48000
DEFAULT_INPUT_SAMPLE_RATE = 12000
OFFICIAL_REPO = "https://github.com/slp-rl/aero"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_repo_dir() -> Path:
    return repo_root() / "tmp/eval_baseline_refs/aero"


def default_checkpoint() -> Path:
    return default_repo_dir() / "checkpoints/12-48/aero-nfft=512-hl=128/checkpoint.th"


def ensure_repo(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    required = [repo_path / "src/models/aero.py", repo_path / "conf/experiment/aero_4-16_512_128.yaml"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"AERO official repo is incomplete at {repo_path}. Missing: "
            f"{', '.join(str(path) for path in missing)}. Clone {OFFICIAL_REPO} there or pass --repo-dir."
        )
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    return repo_path


def load_experiment_config(repo_dir: Path, experiment_config: str, input_sample_rate: int) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("AERO adapter needs PyYAML to read the official experiment config.") from exc

    config_path = repo_dir / "conf/experiment" / experiment_config
    with config_path.open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment.pop("# @package experiment", None)
    experiment["lr_sr"] = int(input_sample_rate)
    experiment["hr_sr"] = TARGET_SAMPLE_RATE
    experiment.setdefault("model", "aero")

    aero = dict(experiment["aero"])
    aero["nfft"] = int(experiment["nfft"])
    aero["hop_length"] = int(experiment["hop_length"])
    aero["lr_sr"] = int(experiment["lr_sr"])
    aero["hr_sr"] = int(experiment["hr_sr"])
    aero["dconv_init"] = float(aero["dconv_init"])
    aero["freq_emb"] = float(aero["freq_emb"])
    aero["rescale"] = float(aero["rescale"])
    experiment["aero"] = aero
    return experiment


def load_aero(args: argparse.Namespace) -> torch.nn.Module:
    repo_dir = ensure_repo(args.repo_dir)
    from src.model_serializer import SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_MODELS, SERIALIZE_KEY_STATE
    from src.models import modelFactory
    from omegaconf import OmegaConf

    experiment = load_experiment_config(repo_dir, args.experiment_config, args.input_sample_rate)
    config = OmegaConf.create({"experiment": experiment})
    model = modelFactory.get_model(config)["generator"]
    package = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if args.continue_best:
        state = package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]["generator"][SERIALIZE_KEY_STATE]
    else:
        state = package[SERIALIZE_KEY_MODELS]["generator"][SERIALIZE_KEY_STATE]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    return model


def super_resolution_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "super_resolution"]
    if max_items > 0:
        return items[:max_items]
    return items


def fit_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    if audio.shape[-1] < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    return audio


def low_resolution_input(item: EvalItem, input_sample_rate: int) -> tuple[torch.Tensor, int]:
    reference_path = reference_path_for_item(item) or item.source_path
    target = load_audio_segment(
        reference_path,
        sample_rate=TARGET_SAMPLE_RATE,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    low = load_audio_segment(
        reference_path,
        sample_rate=input_sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=False,
    )
    if low.shape[0] > 1:
        low = low.mean(dim=0, keepdim=True)
    return low.clamp(-1.0, 1.0), target.shape[-1]


def run_aero_item(model: torch.nn.Module, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    low, target_length = low_resolution_input(item, args.input_sample_rate)
    segment_samples = args.segment_seconds * args.input_sample_rate
    chunks = [low[:, start : start + segment_samples] for start in range(0, low.shape[-1], segment_samples)]
    outputs = []
    with torch.inference_mode():
        for chunk in chunks:
            prediction = model(chunk.unsqueeze(0).to(args.device)).squeeze(0).detach().cpu()
            outputs.append(prediction)
    output = fit_length(torch.cat(outputs, dim=-1), target_length).clamp(-1.0, 1.0)
    save_audio(output_path, output[:1], TARGET_SAMPLE_RATE)


def output_item(item: EvalItem, input_sample_rate: int) -> EvalItem:
    metadata = dict(item.metadata)
    metadata["aero_input_sample_rate"] = input_sample_rate
    metadata["aero_target_sample_rate"] = TARGET_SAMPLE_RATE
    return replace(item, low_sample_rate=input_sample_rate, metadata=metadata)


def write_predicted_manifest(items: list[EvalItem], prediction_paths: list[Path], output_manifest: str | Path | None) -> None:
    if output_manifest is None:
        return
    rows = []
    for item, prediction_path in zip(items, prediction_paths, strict=True):
        rows.append({**item.to_json(), "prediction_path": str(prediction_path.resolve())})
    write_jsonl(rows, output_manifest)


def run_items(items: list[EvalItem], args: argparse.Namespace) -> tuple[list[EvalItem], list[Path]]:
    model = load_aero(args)
    output_items = [output_item(item, args.input_sample_rate) for item in items]
    prediction_paths: list[Path] = []
    total = len(items)
    for index, item in enumerate(output_items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite):
            run_aero_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return output_items, prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official AERO super-resolution baseline.")
    add_common_args(parser)
    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help="Path to the official AERO repo.")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()), help="Path to an official AERO checkpoint.th.")
    parser.add_argument(
        "--experiment-config",
        default="aero_4-16_512_128.yaml",
        help="Official AERO experiment config filename. The adapter overrides lr_sr/hr_sr for 12->48.",
    )
    parser.add_argument("--input-sample-rate", type=int, default=DEFAULT_INPUT_SAMPLE_RATE)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--continue-best", action="store_true", help="Load best state instead of last state.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for AERO inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = super_resolution_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return
    output_items, prediction_paths = run_items(items, args)
    write_predicted_manifest(output_items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
