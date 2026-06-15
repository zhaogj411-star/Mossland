from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF

try:
    from ..audio_io import load_audio_segment, save_audio
    from ..manifest import EvalItem, read_manifest
    from .common import add_common_args, baseline_prediction_path, write_predicted_manifest
except ImportError as exc:  # pragma: no cover - direct file execution convenience.
    raise RuntimeError(
        "Run this baseline as a module from the repository root, for example: "
        "PYTHONPATH=. python -m scripts.mossland-codec.eval_benchmark.baselines.diffstereo_baseline ..."
    ) from exc


BASELINE_NAME = "diffstereo"
OFFICIAL_SAMPLE_RATE = 24000
OFFICIAL_CHUNK_SECONDS = 9.1
OFFICIAL_CHUNK_SAMPLES = int(OFFICIAL_SAMPLE_RATE * OFFICIAL_CHUNK_SECONDS)
DEFAULT_REPO_DIR = Path("tmp/eval_baseline_refs/diffstereo/DiffStereo")
DEFAULT_CHECKPOINT = DEFAULT_REPO_DIR / "checkpoints/model_epoch_80000.pt"


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False


def _add_official_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "models.py").exists():
        raise RuntimeError(
            f"DiffStereo official repo not found at {repo_dir}. Clone "
            "https://github.com/SAKi-77/DiffStereo.git there, or pass --repo-dir."
        )
    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"DiffStereo checkpoint is missing: {path}. The official repo stores "
            "checkpoints in Git LFS. Install git-lfs and run `git lfs pull` in the "
            "official repo, or pass --checkpoint pointing to model_epoch_80000.pt."
        )
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"DiffStereo checkpoint is still a Git LFS pointer, not model weights: {path}. "
            "Install git-lfs and run `git lfs pull`, or manually download the public "
            "LFS object for model_epoch_80000.pt."
        )
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load DiffStereo checkpoint at {path}.") from exc


def load_diffstereo(args: argparse.Namespace, device: torch.device):
    repo_dir = Path(args.repo_dir)
    _add_official_repo_to_path(repo_dir)
    try:
        from diffusion import create_diffusion
        from models import DiT
        from tools.transforms_audio.audio2spec import AudioToSpec
        from tools.transforms_audio.spec2audio import SpecToAudio
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the official DiffStereo modules. Install the official "
            "dependencies from requirement.txt, especially timm, einops, soundfile, "
            "and scipy, then rerun this baseline."
        ) from exc

    checkpoint = _load_checkpoint(Path(args.checkpoint), device)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"DiffStereo checkpoint at {args.checkpoint} does not contain a state dict."
        )

    model = DiT(
        input_size=(729, 513),
        patch_size=9,
        in_channels=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
    )
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            "DiffStereo checkpoint did not match the official DiT architecture used by sample.py."
        ) from exc

    model.to(device).eval()
    diffusion = create_diffusion(timestep_respacing=args.timestep_respacing)
    return model, diffusion, AudioToSpec(), SpecToAudio()


def _mono_condition_for_item(item: EvalItem) -> torch.Tensor:
    audio = load_audio_segment(
        item.source_path,
        sample_rate=item.sample_rate,
        start_seconds=item.start_seconds,
        duration_seconds=item.duration_seconds,
        stereo=True,
    )
    mono = audio.mean(dim=0, keepdim=True)
    if item.sample_rate != OFFICIAL_SAMPLE_RATE:
        mono = AF.resample(mono, item.sample_rate, OFFICIAL_SAMPLE_RATE)
    return mono.clamp(-1.0, 1.0)


def _chunk_mono(mono: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
    chunks = []
    total = mono.shape[-1]
    for start in range(0, total, OFFICIAL_CHUNK_SAMPLES):
        chunk = mono[:, start : start + OFFICIAL_CHUNK_SAMPLES]
        valid = chunk.shape[-1]
        if valid < OFFICIAL_CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, OFFICIAL_CHUNK_SAMPLES - valid))
        chunks.append((chunk, valid))
    if not chunks:
        chunks.append((mono.new_zeros(1, OFFICIAL_CHUNK_SAMPLES), 0))
    return chunks


def infer_item(
    item: EvalItem,
    output_path: Path,
    model,
    diffusion,
    audio2spec,
    spec2audio,
    device: torch.device,
) -> None:
    mono = _mono_condition_for_item(item)
    outputs = []

    for chunk, valid_samples in _chunk_mono(mono):
        chunk = chunk.unsqueeze(0).to(device)
        with torch.inference_mode():
            cond_spec = audio2spec(chunk, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(item.seed) + len(outputs) * 1000003)
            noise = torch.randn(1, 4, 729, 513, device=device, generator=generator)
            generated_spec = diffusion.p_sample_loop(
                model.forward,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs={"y": cond_spec},
                progress=False,
                device=device,
            )
            output_audio = spec2audio(generated_spec, device).squeeze(0).detach().cpu()
        outputs.append(output_audio[..., :valid_samples])

    prediction = torch.cat(outputs, dim=-1) if outputs else torch.zeros(2, 0)
    if OFFICIAL_SAMPLE_RATE != item.sample_rate and prediction.numel() > 0:
        prediction = AF.resample(prediction, OFFICIAL_SAMPLE_RATE, item.sample_rate)

    expected_samples = mono.shape[-1]
    if OFFICIAL_SAMPLE_RATE != item.sample_rate:
        expected_samples = int(round(expected_samples * item.sample_rate / OFFICIAL_SAMPLE_RATE))
    prediction = prediction[..., :expected_samples]
    save_audio(output_path, prediction, item.sample_rate)


def mono_to_stereo_items(manifest: str | Path, max_items: int) -> list[EvalItem]:
    items = [item for item in read_manifest(manifest) if item.task_id == "mono_to_stereo"]
    if max_items > 0:
        return items[:max_items]
    return items


def run_items(items: list[EvalItem], args: argparse.Namespace) -> list[Path]:
    device = torch.device(args.device)
    model, diffusion, audio2spec, spec2audio = load_diffstereo(args, device)
    prediction_paths = []
    for item in items:
        output_path = baseline_prediction_path(args.output_dir, BASELINE_NAME, item)
        prediction_paths.append(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            continue
        infer_item(item, output_path, model, diffusion, audio2spec, spec2audio, device)
    return prediction_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official DiffStereo mono-to-stereo generation baseline."
    )
    add_common_args(parser)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DiffStereo inference.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the official DiffStereo model_epoch_80000.pt checkpoint.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Local clone of https://github.com/SAKi-77/DiffStereo.git.",
    )
    parser.add_argument(
        "--timestep-respacing",
        default="",
        help=(
            "Diffusion timestep respacing passed to official create_diffusion(). "
            "Use the empty default for the paper-style 1000-step sampler; shorter "
            "values such as `ddim25` are useful only for smoke tests."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = mono_to_stereo_items(args.manifest, args.max_items)
    if not items:
        write_predicted_manifest([], [], args.output_manifest)
        return
    prediction_paths = run_items(items, args)
    write_predicted_manifest(items, prediction_paths, args.output_manifest)


if __name__ == "__main__":
    main()
