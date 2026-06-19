"""Compare Music2Latent and SameFlow training dynamics on one fixed batch.

This script is intentionally diagnostic: it uses the repository Hydra configs,
then runs a controlled consistency-loss forward/backward with the same batch,
noise and sigma pair for both models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra import compose, initialize_config_dir

from scripts.music2latent.training_base import add_noise, pseudo_huber_loss


ROOT = Path(__file__).resolve().parents[2]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    out: dict[str, Any] = {
        "shape": list(value.shape),
        "finite": bool(finite.all().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "abs_mean": float(value.abs().mean().item()),
        "abs_max": float(value.abs().max().item()),
    }
    if not out["finite"]:
        out["nonfinite"] = int((~finite).sum().item())
    return out


def grad_norm_for(parameters) -> float:
    norms = []
    for param in parameters:
        if param.grad is None:
            continue
        norms.append(torch.linalg.vector_norm(param.grad.detach().float(), ord=2))
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        top = name.split(".", 1)[0]
        groups.setdefault(top, []).append(param)
        if "output_proj" in name or "conv_out" in name:
            groups.setdefault("_output_head", []).append(param)
        if "adapter" in name:
            groups.setdefault("_adapter", []).append(param)
        if "encoder" in name:
            groups.setdefault("_encoder", []).append(param)
        if "decoder" in name:
            groups.setdefault("_decoder", []).append(param)
    return {key: grad_norm_for(params) for key, params in sorted(groups.items())}


def parameter_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        top = name.split(".", 1)[0]
        groups.setdefault(top, []).append(param)
        if "output_proj" in name or "conv_out" in name:
            groups.setdefault("_output_head", []).append(param)
    out = {}
    for key, params in sorted(groups.items()):
        norms = [torch.linalg.vector_norm(p.detach().float(), ord=2) for p in params]
        out[key] = float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())
    return out


def load_experiment(
    experiment: str,
    device: torch.device,
    batch_size: int,
    extra_overrides: list[str] | None = None,
):
    overrides = [
        f"experiment={experiment}",
        "trainer.devices=1",
        "trainer.strategy=auto",
        "data.num_workers=0",
        f"data.train_batch_size={batch_size}",
        "logger.wandb=null",
        "extras.print_config=false",
    ]
    overrides.extend(extra_overrides or [])
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "scripts" / "configs"),
    ):
        cfg = compose(config_name="train", overrides=overrides)
    data = hydra.utils.instantiate(cfg.data)
    data.setup()
    batch = next(iter(data.train_dataloader()))
    model = hydra.utils.instantiate(cfg.model).to(device)
    wrapper = hydra.utils.instantiate(cfg.wrapper, model=model).to(device)
    wrapper.train()
    return cfg, wrapper, batch


def crop_and_represent(wrapper, batch: torch.Tensor):
    batch = batch.to(next(wrapper.model.parameters()).device)
    batch = batch.to(next(wrapper.model.parameters()).dtype)
    if hasattr(wrapper, "_crop_to_valid_stft_length"):
        batch = wrapper._crop_to_valid_stft_length(batch)
    else:
        hop = wrapper.model.hop
        fac = getattr(wrapper.model.audio_processor, "fac", 4)
        downscaling_factor = 2 ** sum(
            1 for x in wrapper.model.freq_downsample_list if x == 0
        )
        if getattr(wrapper.model.audio_processor, "center_pad", False):
            stft_frames = batch.shape[-1] // hop
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = cropped_frames * hop * downscaling_factor
        else:
            frame_length = fac * hop
            stft_frames = max(0, (batch.shape[-1] - frame_length) // hop + 1)
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = cropped_frames * hop * downscaling_factor + (fac - 1) * hop
        batch = batch[..., :cropped_length]
    return batch, wrapper.model.audio_processor.to_representation_encoder(batch)


def get_output_head(model):
    if hasattr(model, "conv_out"):
        return "conv_out", model.conv_out
    if hasattr(model, "output_proj"):
        return "output_proj", model.output_proj
    return None, None


def run_one(
    label: str,
    wrapper,
    batch: torch.Tensor,
    noise: torch.Tensor,
    sigma_low: torch.Tensor,
    sigma_high: torch.Tensor,
) -> dict[str, Any]:
    model = wrapper.model
    model.zero_grad(set_to_none=True)
    cropped, representation = crop_and_represent(wrapper, batch)
    latent = model.encoder(representation)
    pyramid = model.decoder(latent)
    noisy_high = add_noise(representation, noise.to(representation.device), sigma_high)
    noisy_low = add_noise(representation, noise.to(representation.device), sigma_low)

    activations: dict[str, Any] = {}
    head_name, head = get_output_head(model)
    handle = None
    if head is not None:
        def hook(_module, inputs, output):
            activations[f"{head_name}/input"] = tensor_stats(inputs[0])
            activations[f"{head_name}/output"] = tensor_stats(output)

        handle = head.register_forward_hook(hook)

    predicted = model(
        representation,
        noisy_high,
        sigma=sigma_high,
        latent_override=latent,
    )
    with torch.no_grad():
        target = model(
            representation,
            noisy_low,
            sigma=sigma_low,
            latent_override=latent.detach(),
        )
    if handle is not None:
        handle.remove()

    sigma_delta = (sigma_high - sigma_low).clamp(min=wrapper.consistency_min_sigma_delta)
    weights = (1.0 / sigma_delta).reshape(representation.shape[0], 1, 1, 1)
    loss_values = (
        pseudo_huber_loss(
            predicted.float(),
            target.float(),
            delta=wrapper.consistency_loss_delta,
        )
        * weights.float()
    )
    loss = loss_values.mean()
    loss.backward()

    return {
        "label": label,
        "loss": float(loss.detach().float().item()),
        "cropped": tensor_stats(cropped),
        "representation": tensor_stats(representation),
        "latent": tensor_stats(latent),
        "pyramid": [tensor_stats(x) for x in pyramid],
        "noisy_high": tensor_stats(noisy_high),
        "predicted": tensor_stats(predicted),
        "target": tensor_stats(target),
        "pred_minus_target": tensor_stats(predicted - target),
        "pred_minus_clean": tensor_stats(predicted - representation),
        "target_minus_clean": tensor_stats(target - representation),
        "head_activations": activations,
        "grad_norm_total": grad_norm_for(model.parameters()),
        "grad_norms": grad_norms(model),
        "param_norms": parameter_norms(model),
    }


def fixed_batch_descent(
    wrapper,
    batch: torch.Tensor,
    noise: torch.Tensor,
    sigma_low: torch.Tensor,
    sigma_high: torch.Tensor,
    steps: int,
    lr: float,
) -> list[float]:
    if steps <= 0:
        return []
    opt = torch.optim.RAdam(wrapper.model.parameters(), lr=lr, betas=(0.9, 0.999))
    losses = []
    for _ in range(steps):
        wrapper.model.zero_grad(set_to_none=True)
        _, representation = crop_and_represent(wrapper, batch)
        latent = wrapper.model.encoder(representation)
        noisy_high = add_noise(representation, noise.to(representation.device), sigma_high)
        noisy_low = add_noise(representation, noise.to(representation.device), sigma_low)
        predicted = wrapper.model(
            representation,
            noisy_high,
            sigma=sigma_high,
            latent_override=latent,
        )
        with torch.no_grad():
            target = wrapper.model(
                representation,
                noisy_low,
                sigma=sigma_low,
                latent_override=latent.detach(),
            )
        sigma_delta = (sigma_high - sigma_low).clamp(
            min=wrapper.consistency_min_sigma_delta
        )
        weights = (1.0 / sigma_delta).reshape(representation.shape[0], 1, 1, 1)
        loss = (
            pseudo_huber_loss(
                predicted.float(),
                target.float(),
                delta=wrapper.consistency_loss_delta,
            )
            * weights.float()
        ).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().float().item()))
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-high", type=float, default=0.522835373878479)
    parser.add_argument("--sigma-low", type=float, default=0.17247170209884644)
    parser.add_argument("--descent-steps", type=int, default=0)
    parser.add_argument("--descent-lr", type=float, default=1e-4)
    parser.add_argument(
        "--same-experiment",
        default="same_flow_baseline",
        help="Hydra experiment used for the candidate model side.",
    )
    parser.add_argument(
        "--same-override",
        action="append",
        default=[],
        help="Additional Hydra override for the same_flow experiment. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "tmp" / "same_flow_debug" / "training_dynamics.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    music_cfg, music_wrapper, batch = load_experiment("music2latent", device, args.batch_size)
    same_cfg, same_wrapper, _ = load_experiment(
        args.same_experiment,
        device,
        args.batch_size,
        args.same_override,
    )

    audio, info = batch
    audio = audio.to(device)
    _, representation = crop_and_represent(music_wrapper, audio)
    torch.manual_seed(args.seed + 1000)
    noise = torch.randn_like(representation)
    sigma_high = torch.full((audio.shape[0],), args.sigma_high, device=device)
    sigma_low = torch.full((audio.shape[0],), args.sigma_low, device=device)

    out = {
        "settings": {
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sigma_high": args.sigma_high,
            "sigma_low": args.sigma_low,
            "music_precision": str(music_cfg.trainer.precision),
            "same_precision": str(same_cfg.trainer.precision),
            "same_experiment": args.same_experiment,
            "same_overrides": list(args.same_override),
            "paths": info.get("path", None) if isinstance(info, dict) else None,
        },
        "music2latent": run_one(
            "music2latent",
            music_wrapper,
            audio,
            noise,
            sigma_low,
            sigma_high,
        ),
        "same_flow": run_one(
            "same_flow",
            same_wrapper,
            audio,
            noise,
            sigma_low,
            sigma_high,
        ),
    }
    if args.descent_steps > 0:
        out["fixed_batch_descent"] = {
            "steps": args.descent_steps,
            "lr": args.descent_lr,
            "music2latent": fixed_batch_descent(
                music_wrapper,
                audio,
                noise,
                sigma_low,
                sigma_high,
                args.descent_steps,
                args.descent_lr,
            ),
            "same_flow": fixed_batch_descent(
                same_wrapper,
                audio,
                noise,
                sigma_low,
                sigma_high,
                args.descent_steps,
                args.descent_lr,
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "music_loss": out["music2latent"]["loss"],
        "same_loss": out["same_flow"]["loss"],
        "music_grad": out["music2latent"]["grad_norm_total"],
        "same_grad": out["same_flow"]["grad_norm_total"],
        "music_head_input_rms": out["music2latent"]["head_activations"],
        "same_head_input_rms": out["same_flow"]["head_activations"],
    }, indent=2))


if __name__ == "__main__":
    main()
