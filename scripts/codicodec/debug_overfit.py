import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _load_first_audio(files_list: str, key: str = "audio") -> torch.Tensor:
    with open(files_list, "r") as f:
        path = next(line.strip() for line in f if line.strip())
    item = torch.load(path, map_location="cpu", weights_only=False)
    audio = item[key]
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return audio


def _module_grad_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not name.startswith(prefix) or param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def _rep_metrics(model, source_rep: torch.Tensor, pred_rep: torch.Tensor) -> dict[str, float]:
    frames = min(source_rep.shape[-1], pred_rep.shape[-1])
    diff = pred_rep[..., :frames].float() - source_rep[..., :frames].float()
    return {
        "rep_l1": float(diff.abs().mean().detach().cpu()),
        "rep_mse": float(diff.pow(2).mean().detach().cpu()),
    }


@torch.no_grad()
def _decode_metrics(model, representation, latents, *, seed: int, denoising_steps: int):
    torch.manual_seed(seed)
    prior_pred = model.decode(latents, denoising_steps=denoising_steps, slide=False, initial_state="noise")
    zero_init_pred = model.decode(latents, denoising_steps=denoising_steps, slide=False, initial_state="zero")
    zero_latents = torch.zeros_like(latents)
    zero_latent_pred = model.decode(
        zero_latents,
        denoising_steps=denoising_steps,
        slide=False,
        initial_state="zero",
    )
    zero_x = torch.zeros_like(representation)
    t = torch.full((representation.shape[0],), model.t_max, device=representation.device)
    features = model.pre_decoder_forward(latents)
    zero_pred = model.denoise(zero_x, t, latents, features=features)
    out = {}
    out.update({f"prior_{k}": v for k, v in _rep_metrics(model, representation, prior_pred).items()})
    out.update({f"zero_init_{k}": v for k, v in _rep_metrics(model, representation, zero_init_pred).items()})
    out.update({f"zero_latent_{k}": v for k, v in _rep_metrics(model, representation, zero_latent_pred).items()})
    out["true_vs_zero_latent_mse"] = float(
        (zero_init_pred.float() - zero_latent_pred.float()).pow(2).mean().detach().cpu()
    )
    out.update({f"zero_{k}": v for k, v in _rep_metrics(model, representation, zero_pred).items()})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="codicodec")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["radam", "adamw"], default=None)
    parser.add_argument("--direct-weight", type=float, default=1.0e-5)
    parser.add_argument("--direct-mode", default="trigflow_noise")
    parser.add_argument("--detach-encoder", action="store_true")
    parser.add_argument("--fixed-train-noise", action="store_true")
    args = parser.parse_args()

    config_dir = str((Path.cwd() / "scripts/configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={args.experiment}",
                "data.num_workers=0",
                "data.train_batch_size=1",
                "wrapper.use_ema=false",
                *([] if args.optimizer is None else [f"wrapper.optimizer_name={args.optimizer}"]),
                "wrapper.consistency_weight=0.0",
                "wrapper.use_jvp_tangent=false",
                f"wrapper.direct_denoise_weight={args.direct_weight}",
                f"wrapper.direct_denoise_mode={args.direct_mode}",
                f"wrapper.direct_detach_encoder={str(args.detach_encoder).lower()}",
                "wrapper.p_mean=5.0",
                "wrapper.p_std=0.0",
                "wrapper.random_mix_prob=0.0",
                "wrapper.rvq_latent_train_prob=0.0",
                "wrapper.rvq_commitment_weight=0.0",
                "wrapper.rvq_codebook_weight=0.0",
                "wrapper.rvq_hidden_recon_weight=0.0",
                "wrapper.latent_time_shift_prob=0.0",
            ],
        )

    torch.manual_seed(args.seed)
    model = instantiate(cfg.model).to(args.device)
    module = instantiate(cfg.wrapper, model=model).to(args.device)
    module.train()
    opt_conf = module.configure_optimizers()
    optimizer = opt_conf["optimizer"] if isinstance(opt_conf, dict) else opt_conf
    if args.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    audio = _load_first_audio(cfg.data.dataset.files_list).to(args.device)
    audio = module._prepare_audio_batch(audio)

    reports = []
    for step in range(args.steps + 1):
        if args.fixed_train_noise:
            torch.manual_seed(args.seed)
        representation = model.audio_processor.to_representation_encoder(audio)
        latents = model.encoder_forward(representation)
        if args.detach_encoder:
            latents = latents.detach()
        features = model.pre_decoder_forward(latents)
        loss, metrics, _predicted, _target = module._consistency_loss(
            representation,
            latents,
            features,
        )

        if step % args.log_every == 0 or step == args.steps:
            decode = _decode_metrics(
                model,
                representation,
                latents,
                seed=args.seed,
                denoising_steps=args.denoising_steps,
            )
            report = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "direct_mse": float(metrics["loss/direct_denoise_mse"].detach().cpu()),
                "direct_sse": float(metrics["loss/direct_denoise_sse"].detach().cpu()),
                "latent_std": float(latents.detach().float().std().cpu()),
                **decode,
            }
            print(json.dumps(report, ensure_ascii=False), flush=True)
            reports.append(report)

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            grad_report = {
                "step": step,
                "grad/frontend_encoder_down": _module_grad_norm(model, "frontend_encoder_down"),
                "grad/encoder": _module_grad_norm(model, "encoder"),
                "grad/lat2patch_pre_decoder": _module_grad_norm(model, "lat2patch_pre_decoder"),
                "grad/pre_decoder": _module_grad_norm(model, "pre_decoder"),
                "grad/frontend_pre_decoder_up": _module_grad_norm(model, "frontend_pre_decoder_up"),
                "grad/lat2patch": _module_grad_norm(model, "lat2patch"),
                "grad/frontend_decoder_down": _module_grad_norm(model, "frontend_decoder_down"),
                "grad/decoder": _module_grad_norm(model, "decoder"),
                "grad/frontend_decoder_up": _module_grad_norm(model, "frontend_decoder_up"),
                "grad/gain_decoder": _module_grad_norm(model, "gain_decoder"),
            }
            print(json.dumps(grad_report, ensure_ascii=False), flush=True)
        optimizer.step()


if __name__ == "__main__":
    main()
