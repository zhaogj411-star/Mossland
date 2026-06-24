from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from scripts.codec_common.training_base import add_noise, get_sigma_continuous, pseudo_huber_loss


def _load_cfg(path: str | Path):
    return OmegaConf.load(Path(path))


def _make_optimizer(model: torch.nn.Module, cfg):
    name = str(cfg.wrapper.optimizer_name).lower()
    if name == "radam":
        opt = torch.optim.RAdam(
            model.parameters(),
            lr=float(cfg.wrapper.learning_rate),
            betas=(0.9, 0.999),
            weight_decay=float(cfg.wrapper.weight_decay),
        )
    elif name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.wrapper.learning_rate),
            betas=(0.9, 0.999),
            weight_decay=float(cfg.wrapper.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer_name={cfg.wrapper.optimizer_name!r}")

    if str(cfg.wrapper.lr_schedule) == "constant":
        return opt, None
    if str(cfg.wrapper.lr_schedule) != "cosine_decay":
        raise ValueError(f"Unsupported lr_schedule={cfg.wrapper.lr_schedule!r}")

    warmup_steps = int(cfg.wrapper.lr_warmup_steps)
    total_steps = int(cfg.wrapper.lr_schedule_total_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(
        opt,
        T_max=decay_steps,
        eta_min=float(cfg.wrapper.final_learning_rate),
    )
    if warmup_steps <= 0:
        return opt, cosine
    warmup = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    return opt, SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _consistency_step(step: int, cfg) -> float:
    base = float(cfg.wrapper.consistency_step)
    schedule = str(cfg.wrapper.consistency_step_schedule)
    if schedule == "constant":
        return base
    if schedule != "exponential":
        raise ValueError(f"Unsupported consistency_step_schedule={schedule!r}")
    progress = min(max(float(step) / max(1, int(cfg.wrapper.consistency_step_total_steps)), 0.0), 1.0)
    return base * (10.0 ** (-(float(cfg.wrapper.consistency_step_end_exp) - 1.0) * progress))


def _lognormal_positions(batch_size: int, model, cfg, generator: torch.Generator, num_bins: int = 10_000):
    bins = torch.linspace(0.0, 1.0, num_bins - 1)
    sigmas = get_sigma_continuous(
        bins,
        sigma_min=float(model.sigma_min),
        sigma_max=float(model.sigma_max),
        rho=float(model.rho),
    )
    weights = torch.exp(
        -0.5 * ((torch.log(sigmas) - float(cfg.wrapper.lognormal_mean)) / float(cfg.wrapper.lognormal_std)) ** 2
    ) / (float(cfg.wrapper.lognormal_std) * (2.0 * torch.pi) ** 0.5)
    inds = torch.multinomial(weights, batch_size, replacement=True, generator=generator).float()
    jitter = torch.rand(batch_size, generator=generator)
    return (inds + jitter) / float(num_bins - 1)


def _sample_sigma_pair(batch_size: int, model, cfg, step: int, generator: torch.Generator, device: torch.device):
    step_size = min(max(_consistency_step(step, cfg), 0.0), 1.0)
    if str(cfg.wrapper.sigma_sampling) == "lognormal":
        high_pos = _lognormal_positions(batch_size, model, cfg, generator)
    elif str(cfg.wrapper.sigma_sampling) == "uniform":
        high_pos = torch.rand(batch_size, generator=generator)
    else:
        raise ValueError(f"Unsupported sigma_sampling={cfg.wrapper.sigma_sampling!r}")

    high_pos = high_pos.clamp(min=step_size)
    low_pos = (high_pos - step_size).clamp(min=0.0)
    sigma_low = get_sigma_continuous(
        low_pos,
        sigma_min=float(model.sigma_min),
        sigma_max=float(model.sigma_max),
        rho=float(model.rho),
    ).clamp(float(model.sigma_min), float(model.sigma_max))
    sigma_high = get_sigma_continuous(
        high_pos,
        sigma_min=float(model.sigma_min),
        sigma_max=float(model.sigma_max),
        rho=float(model.rho),
    ).clamp(float(model.sigma_min), float(model.sigma_max))
    return sigma_low.to(device), sigma_high.to(device), step_size


def _sample_n_quantizers(cfg, generator: torch.Generator):
    choices = cfg.wrapper.get("train_n_quantizers_choices")
    if choices:
        idx = int(torch.randint(len(choices), (), generator=generator).item())
        return int(choices[idx])
    value = cfg.wrapper.get("train_n_quantizers")
    return None if value is None else int(value)


def _batch_from_dataset(cfg, batch_size: int, device: torch.device):
    ds = instantiate(cfg.data.dataset)
    payloads = [ds[i][0] for i in range(batch_size)]
    src = torch.stack([item["src"] for item in payloads], dim=0).to(device)
    target = torch.stack([item["target"] for item in payloads], dim=0).to(device)
    task_id = [str(item["task_id"]) for item in payloads]
    return src, target, task_id


def _representations(model, src_audio, target_audio):
    with torch.no_grad():
        src_rep = model.audio_processor.to_representation_encoder(src_audio)
        target_rep = model.audio_processor.to_representation_encoder(target_audio)
    return src_rep.detach(), target_rep.detach()


def _grad_norm(model: torch.nn.Module) -> float:
    norms = [
        torch.linalg.vector_norm(param.grad.detach().float(), ord=2)
        for param in model.parameters()
        if param.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())


def _run_step(
    name: str,
    model,
    cfg,
    optimizer,
    src_rep,
    target_rep,
    task_id,
    sigma_low,
    sigma_high,
    noise,
    n_quantizers,
    discrete_mask,
    train: bool,
):
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=src_rep.is_cuda):
            quantized = model.quantize_representation(
                src_rep,
                detach_encoder=bool(cfg.wrapper.rvq_detach_encoder),
                n_quantizers=n_quantizers,
            )
            latent_override = quantized.continuous
            if float(cfg.wrapper.rvq_latent_train_prob) > 0:
                mask = discrete_mask.to(device=latent_override.device)
                mask = mask.view(mask.shape[0], *([1] * (latent_override.ndim - 1)))
                latent_override = torch.where(mask, quantized.discrete, quantized.continuous)

            noisy_high = add_noise(target_rep, noise, sigma_high)
            noisy_low = add_noise(target_rep, noise, sigma_low)
            predicted = model(
                target_rep,
                noisy_high,
                sigma=sigma_high,
                latent_override=latent_override,
                task_id=task_id,
            )
            with torch.no_grad():
                target = model(
                    target_rep,
                    noisy_low,
                    sigma=sigma_low,
                    latent_override=latent_override.detach(),
                    task_id=task_id,
                )
            sigma_delta = (sigma_high - sigma_low).clamp(min=float(cfg.wrapper.consistency_min_sigma_delta))
            weights = (1.0 / sigma_delta).reshape(target_rep.shape[0], 1, 1, 1)
            loss = (
                pseudo_huber_loss(
                    predicted.float(),
                    target.float(),
                    delta=float(cfg.wrapper.consistency_loss_delta),
                )
                * weights.float()
            ).mean()
            rvq_loss = (
                float(cfg.wrapper.rvq_commitment_weight) * quantized.commitment_loss
                + float(cfg.wrapper.rvq_codebook_weight) * quantized.codebook_loss
                + float(cfg.wrapper.rvq_hidden_recon_weight) * quantized.distill_loss
            )
            total = loss + rvq_loss
    if train:
        total.backward()
        grad_before = _grad_norm(model)
        clip = cfg.trainer.get("gradient_clip_val")
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        grad_after = _grad_norm(model)
        optimizer.step()
    else:
        grad_before = 0.0
        grad_after = 0.0

    return {
        f"{name}_loss": float(loss.detach().item()),
        f"{name}_total": float(total.detach().item()),
        f"{name}_rvq_commit": float(quantized.commitment_loss.detach().item()),
        f"{name}_rvq_hidden": float(quantized.distill_loss.detach().item()),
        f"{name}_latent_std": float(quantized.continuous.detach().float().std().item()),
        f"{name}_discrete_std": float(quantized.discrete.detach().float().std().item()),
        f"{name}_pred_clean_l1": float((predicted.detach().float() - target_rep.float()).abs().mean().item()),
        f"{name}_grad_before": grad_before,
        f"{name}_grad_after": grad_after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-config", default="scripts/configs/experiment/mossland-codec-same-overfit.yaml")
    parser.add_argument("--original-config", default="scripts/configs/experiment/mossland-codec-overfit-small.yaml")
    parser.add_argument("--output-dir", default="tmp/mossland_codec_pair_debug")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    same_cfg = _load_cfg(args.same_config)
    original_cfg = _load_cfg(args.original_config)

    torch.manual_seed(args.seed)
    same_model = instantiate(same_cfg.model).to(device)
    torch.manual_seed(args.seed)
    original_model = instantiate(original_cfg.model).to(device)
    same_opt, same_sched = _make_optimizer(same_model, same_cfg)
    original_opt, original_sched = _make_optimizer(original_model, original_cfg)

    src_audio, target_audio, task_id = _batch_from_dataset(same_cfg, args.batch_size, device)
    same_src_rep, same_target_rep = _representations(same_model, src_audio, target_audio)
    original_src_rep, original_target_rep = _representations(original_model, src_audio, target_audio)
    if not torch.equal(same_src_rep, original_src_rep) or not torch.equal(same_target_rep, original_target_rep):
        max_src = float((same_src_rep - original_src_rep).abs().max().item())
        max_target = float((same_target_rep - original_target_rep).abs().max().item())
        raise RuntimeError(f"representation mismatch: src={max_src}, target={max_target}")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed + 1)
    eval_rng = torch.Generator(device="cpu")
    eval_rng.manual_seed(args.seed + 999)
    eval_sigma_low, eval_sigma_high, _ = _sample_sigma_pair(
        args.batch_size, same_model, same_cfg, 0, eval_rng, device
    )
    eval_noise = torch.randn(same_target_rep.shape, generator=eval_rng).to(device)
    eval_mask = torch.rand(args.batch_size, generator=eval_rng) < float(same_cfg.wrapper.rvq_latent_train_prob)
    eval_nq = _sample_n_quantizers(same_cfg, eval_rng)

    fields = [
        "step",
        "seconds",
        "lr",
        "n_quantizers",
        "source_discrete",
        "sigma_low_mean",
        "sigma_high_mean",
        "sigma_weight_mean",
        "same_loss",
        "same_total",
        "same_rvq_commit",
        "same_rvq_hidden",
        "same_latent_std",
        "same_discrete_std",
        "same_pred_clean_l1",
        "same_grad_before",
        "same_grad_after",
        "original_loss",
        "original_total",
        "original_rvq_commit",
        "original_rvq_hidden",
        "original_latent_std",
        "original_discrete_std",
        "original_pred_clean_l1",
        "original_grad_before",
        "original_grad_after",
        "same_eval_loss",
        "same_eval_pred_clean_l1",
        "original_eval_loss",
        "original_eval_pred_clean_l1",
    ]
    csv_path = output_dir / "metrics.csv"
    start = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps):
            sigma_low, sigma_high, _ = _sample_sigma_pair(args.batch_size, same_model, same_cfg, step, rng, device)
            noise = torch.randn(same_target_rep.shape, generator=rng).to(device)
            n_quantizers = _sample_n_quantizers(same_cfg, rng)
            discrete_mask = torch.rand(args.batch_size, generator=rng) < float(same_cfg.wrapper.rvq_latent_train_prob)
            same_metrics = _run_step(
                "same",
                same_model,
                same_cfg,
                same_opt,
                same_src_rep,
                same_target_rep,
                task_id,
                sigma_low,
                sigma_high,
                noise,
                n_quantizers,
                discrete_mask,
                train=True,
            )
            original_metrics = _run_step(
                "original",
                original_model,
                original_cfg,
                original_opt,
                original_src_rep,
                original_target_rep,
                task_id,
                sigma_low,
                sigma_high,
                noise,
                n_quantizers,
                discrete_mask,
                train=True,
            )
            if same_sched is not None:
                same_sched.step()
            if original_sched is not None:
                original_sched.step()

            row = {
                "step": step,
                "seconds": time.time() - start,
                "lr": same_opt.param_groups[0]["lr"],
                "n_quantizers": n_quantizers if n_quantizers is not None else "",
                "source_discrete": float(discrete_mask.float().mean().item()),
                "sigma_low_mean": float(sigma_low.mean().item()),
                "sigma_high_mean": float(sigma_high.mean().item()),
                "sigma_weight_mean": float((1.0 / (sigma_high - sigma_low).clamp(min=float(same_cfg.wrapper.consistency_min_sigma_delta))).mean().item()),
            }
            row.update(same_metrics)
            row.update(original_metrics)

            if step % args.eval_every == 0 or step == args.steps - 1:
                same_eval = _run_step(
                    "same_eval",
                    same_model,
                    same_cfg,
                    same_opt,
                    same_src_rep,
                    same_target_rep,
                    task_id,
                    eval_sigma_low,
                    eval_sigma_high,
                    eval_noise,
                    eval_nq,
                    eval_mask,
                    train=False,
                )
                original_eval = _run_step(
                    "original_eval",
                    original_model,
                    original_cfg,
                    original_opt,
                    original_src_rep,
                    original_target_rep,
                    task_id,
                    eval_sigma_low,
                    eval_sigma_high,
                    eval_noise,
                    eval_nq,
                    eval_mask,
                    train=False,
                )
                row["same_eval_loss"] = same_eval["same_eval_loss"]
                row["same_eval_pred_clean_l1"] = same_eval["same_eval_pred_clean_l1"]
                row["original_eval_loss"] = original_eval["original_eval_loss"]
                row["original_eval_pred_clean_l1"] = original_eval["original_eval_pred_clean_l1"]
            writer.writerow(row)
            if step % args.log_every == 0 or step == args.steps - 1:
                print(
                    f"step={step} "
                    f"same={row['same_loss']:.6f} orig={row['original_loss']:.6f} "
                    f"same_eval={row.get('same_eval_loss', '')} "
                    f"orig_eval={row.get('original_eval_loss', '')} "
                    f"t={row['seconds']:.1f}s",
                    flush=True,
                )

    torch.save(
        {
            "same_model": same_model.state_dict(),
            "original_model": original_model.state_dict(),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "task_id": task_id,
        },
        output_dir / "last.pt",
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
