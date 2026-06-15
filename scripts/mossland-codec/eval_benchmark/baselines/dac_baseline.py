from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from ..audio_io import load_audio_segment, save_audio
from ..manifest import EvalItem, read_manifest
from .common import add_common_args, baseline_prediction_path, maybe_print_progress, write_predicted_manifest


BASELINE_NAME = "dac"
DEFAULT_WEIGHTS_CACHE = Path("tmp/eval_baseline_refs/dac/checkpoints")


def _official_weights_url(dac_module, model_type: str, model_bitrate: str, model_tag: str) -> str:
    model_type = model_type.lower()
    model_bitrate = model_bitrate.lower()
    model_tag = model_tag.lower()
    if model_tag == "latest":
        try:
            model_tag = dac_module.utils.__MODEL_LATEST_TAGS__[(model_type, model_bitrate)]
        except Exception as exc:
            raise RuntimeError(
                f"Official DAC package does not expose a latest tag for "
                f"model_type={model_type!r}, model_bitrate={model_bitrate!r}."
            ) from exc

    try:
        url = dac_module.utils.__MODEL_URLS__[(model_type, model_tag, model_bitrate)]
    except Exception as exc:
        raise RuntimeError(
            f"Official DAC package does not expose weights for model_type={model_type!r}, "
            f"model_bitrate={model_bitrate!r}, model_tag={model_tag!r}."
        ) from exc
    return str(url)


def _download_official_weights(dac_module, args: argparse.Namespace) -> Path:
    if args.weights_path:
        return Path(args.weights_path)

    url = _official_weights_url(dac_module, args.model_type, args.model_bitrate, args.model_tag)
    cache_dir = Path(args.weights_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / Path(url).name
    if weights_path.exists() and weights_path.stat().st_size > 0:
        return weights_path

    try:
        import requests

        part_path = weights_path.with_suffix(weights_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=(10, args.download_timeout)) as response:
            response.raise_for_status()
            with part_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        part_path.replace(weights_path)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download official DAC checkpoint from {url}. "
            f"Pass --weights-path with a local checkpoint if this machine cannot reach GitHub releases."
        ) from exc
    return weights_path


def _load_dac_model(args: argparse.Namespace):
    try:
        import dac
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Descript Audio Codec is not importable. Install the official package with "
            "`python -m pip install descript-audio-codec`."
        ) from exc

    try:
        weights_path = _download_official_weights(dac, args)
        model = dac.DAC.load(weights_path)
    except Exception as exc:  # pragma: no cover - network/checkpoint dependent
        raise RuntimeError(
            "Failed to load the official Descript Audio Codec checkpoint. "
            "Check network access to the public DAC release assets, or pass "
            "`--weights-path` pointing at a downloaded DAC weights file."
        ) from exc

    model.to(args.device)
    model.eval()
    return model


def _reconstruct_item(model, item: EvalItem, output_path: Path, args: argparse.Namespace) -> None:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )

    with tempfile.TemporaryDirectory(prefix="mossland_dac_") as tmpdir:
        input_path = Path(tmpdir) / f"{item.item_id}.wav"
        save_audio(input_path, audio, item.sample_rate)

        with torch.inference_mode():
            artifact = model.compress(
                input_path,
                win_duration=args.win_duration,
                verbose=args.verbose,
                normalize_db=args.normalize_db,
                n_quantizers=args.n_quantizers,
            )
            reconstructed = model.decompress(artifact, verbose=args.verbose)

        output_audio = reconstructed.audio_data.detach().cpu().float()
        if output_audio.ndim == 3:
            output_audio = output_audio[0]
        if output_audio.ndim != 2:
            raise RuntimeError(
                f"DAC returned audio with unexpected shape {tuple(output_audio.shape)} "
                f"for item_id={item.item_id!r}."
            )
        save_audio(output_path, output_audio, reconstructed.sample_rate)


def _reconstruct_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    model = _load_dac_model(args)
    prediction_paths = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            maybe_print_progress(index, total, item.item_id, args.progress_every)
            continue
        _reconstruct_item(model, item, output_path, args)
        maybe_print_progress(index, total, item.item_id, args.progress_every)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official Descript Audio Codec reconstruction baseline."
    )
    add_common_args(parser)
    parser.add_argument("--device", default="cuda", help="Torch device for DAC inference.")
    parser.add_argument(
        "--model-type",
        default="44khz",
        choices=("44khz", "24khz", "16khz"),
        help="Official DAC release model type.",
    )
    parser.add_argument(
        "--model-bitrate",
        default="8kbps",
        choices=("8kbps", "16kbps"),
        help="Official DAC release model bitrate.",
    )
    parser.add_argument(
        "--model-tag",
        default="latest",
        help="Official DAC release tag.",
    )
    parser.add_argument(
        "--weights-path",
        default="",
        help="Optional local DAC weights file. Skips release checkpoint download when set.",
    )
    parser.add_argument(
        "--weights-cache-dir",
        default=str(DEFAULT_WEIGHTS_CACHE),
        help="Local cache for official DAC release weights.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=120.0,
        help="Read timeout in seconds for official checkpoint download.",
    )
    parser.add_argument(
        "--win-duration",
        type=float,
        default=5.0,
        help="Window duration passed to DAC compress().",
    )
    parser.add_argument(
        "--normalize-db",
        type=float,
        default=-16.0,
        help="Loudness normalization target passed to DAC compress(); use NaN to disable.",
    )
    parser.add_argument(
        "--n-quantizers",
        type=int,
        default=None,
        help="Optional number of DAC quantizers passed to compress().",
    )
    parser.add_argument("--verbose", action="store_true", help="Show official DAC progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.normalize_db != args.normalize_db:
        args.normalize_db = None

    items = [item for item in read_manifest(args.manifest) if item.task_id == "reconstruct"]
    if args.max_items > 0:
        items = items[: args.max_items]

    prediction_paths = _reconstruct_items(items, args) if items else []
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
