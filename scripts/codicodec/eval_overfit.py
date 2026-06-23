import argparse
import json
from pathlib import Path

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _load_checkpoint(module, checkpoint_path: str, use_ema: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if use_ema:
        ema_prefix = "ema.ema_model."
        model_state = {
            key.removeprefix(ema_prefix): value
            for key, value in state_dict.items()
            if key.startswith(ema_prefix)
        }
        if not model_state:
            raise RuntimeError(f"checkpoint has no EMA weights: {checkpoint_path}")
        module.model.load_state_dict(model_state, strict=False)
    else:
        module.load_state_dict(state_dict, strict=False)


def _rep_metrics(model, source: torch.Tensor, reconstructed: torch.Tensor):
    device = next(model.parameters()).device
    length = min(source.shape[-1], reconstructed.shape[-1])
    source = source[..., :length]
    reconstructed = reconstructed[..., :length]
    source_rep = model.audio_processor.to_representation_encoder(source.float().to(device))
    reconstructed_rep = model.audio_processor.to_representation_encoder(
        reconstructed.float().to(device)
    )
    frames = min(source_rep.shape[-1], reconstructed_rep.shape[-1])
    source_rep = source_rep[..., :frames].float()
    reconstructed_rep = reconstructed_rep[..., :frames].float()
    diff = reconstructed_rep - source_rep
    return {
        "rep_l1": float(diff.abs().mean().detach().cpu()),
        "rep_mse": float(diff.pow(2).mean().detach().cpu()),
        "wave_l1": float((reconstructed - source).float().abs().mean().detach().cpu()),
    }


def _concat_segments(segments: list[torch.Tensor], sample_rate: int):
    silence = segments[0].new_zeros(segments[0].shape[:-1] + (sample_rate // 4,))
    pieces = []
    for segment in segments:
        if pieces:
            pieces.append(silence)
        pieces.append(segment)
    return torch.cat(pieces, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="codicodec")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--n-quantizers", type=int, default=None)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--slide", action="store_true")
    parser.add_argument("--initial-state", choices=["noise", "zero"], default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--save-audio", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixed-crop", action="store_true")
    parser.add_argument("--sub-length", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    args = parser.parse_args()

    config_dir = str((Path.cwd() / "scripts/configs").resolve())
    overrides = [
        f"experiment={args.experiment}",
        "data.num_workers=0",
        "data.train_batch_size=1",
    ]
    if args.fixed_crop:
        overrides.append("data.dataset.random_crop=false")
    if args.sub_length is not None:
        overrides.append(f"data.dataset.sub_length={args.sub_length}")
    if args.target_length is not None:
        overrides.append(f"data.dataset.target_length={args.target_length}")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=overrides,
        )

    torch.manual_seed(args.seed)
    datamodule = instantiate(cfg.data)
    datamodule.setup()
    model = instantiate(cfg.model).to(args.device)
    module = instantiate(cfg.wrapper, model=model).to(args.device)
    module.eval()
    if args.checkpoint:
        _load_checkpoint(module, args.checkpoint, use_ema=args.use_ema)
        module.eval()

    metrics = []
    audio_pairs = []
    dataset = datamodule.dataset
    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            if batch_idx >= len(dataset):
                break
            batch = dataset[batch_idx]
            audio, info = batch
            audio = audio.unsqueeze(0)
            batch = (audio, info)
            torch.manual_seed(args.seed + batch_idx)
            audio, _info = batch
            audio = audio.to(args.device)
            source, reconstructed = module.reconstruct_waveform(
                audio,
                model=module.model,
                n_quantizers=args.n_quantizers,
                slide=args.slide,
                denoising_steps=args.denoising_steps,
                initial_state=args.initial_state,
            )
            item = _rep_metrics(module.model, source, reconstructed)
            item["batch_idx"] = batch_idx
            metrics.append(item)
            if args.save_audio and batch_idx == 0:
                audio_pairs.append(source[0].cpu())
                audio_pairs.append(reconstructed[0].cpu())

    aggregate = {
        key: sum(item[key] for item in metrics) / max(1, len(metrics))
        for key in ["rep_l1", "rep_mse", "wave_l1"]
    }
    result = {
        "checkpoint": args.checkpoint,
        "use_ema": args.use_ema,
        "n_quantizers": args.n_quantizers,
        "denoising_steps": args.denoising_steps,
        "slide": args.slide,
        "initial_state": args.initial_state,
        "num_batches": len(metrics),
        "mean": aggregate,
        "items": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_audio and audio_pairs:
        out = _concat_segments(audio_pairs, sample_rate=module.model.sample_rate)
        out_path = Path(args.save_audio)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), out.float(), module.model.sample_rate)


if __name__ == "__main__":
    main()
