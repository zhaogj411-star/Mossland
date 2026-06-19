import argparse
import json
from pathlib import Path
from types import MethodType

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def _load_config(args):
    root = Path(__file__).resolve().parents[2]
    overrides = [f"experiment={args.experiment}"]
    overrides.extend(args.override or [])
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "scripts" / "configs"),
    ):
        return compose(config_name="train", overrides=overrides)


def _optimizer_scheduler_from_config(wrapper):
    configured = wrapper.configure_optimizers()
    if isinstance(configured, dict):
        scheduler_config = configured.get("lr_scheduler")
        scheduler = None
        if isinstance(scheduler_config, dict):
            scheduler = scheduler_config.get("scheduler")
        else:
            scheduler = scheduler_config
        return configured["optimizer"], scheduler
    if isinstance(configured, (list, tuple)):
        return configured[0], None
    return configured, None


def _save_checkpoint(wrapper, path, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": int(step),
        "state_dict": {
            f"model.{name}": value.detach().cpu()
            for name, value in wrapper.model.state_dict().items()
        },
    }
    if hasattr(wrapper, "ema"):
        state["state_dict"].update(
            {
                f"ema.ema_model.{name}": value.detach().cpu()
                for name, value in wrapper.ema.ema_model.state_dict().items()
            }
        )
    torch.save(state, path)


def _strip_prefix(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return None
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _load_model_checkpoint(model, checkpoint_path, variant):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if variant == "ema":
        selected = _strip_prefix(state_dict, "ema.ema_model.")
        if selected is None:
            raise ValueError(f"{checkpoint_path} does not contain ema.ema_model.* keys")
    elif variant == "raw":
        selected = _strip_prefix(state_dict, "model.")
        if selected is None:
            selected = state_dict
    else:
        raise ValueError(f"unsupported checkpoint variant {variant!r}")

    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, value in selected.items():
        if name in current and current[name].shape == value.shape:
            compatible[name] = value
        else:
            skipped.append(name)
    current.update(compatible)
    model.load_state_dict(current, strict=True)
    return {
        "loaded": len(compatible),
        "missing": len(current) - len(compatible),
        "skipped": len(skipped),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="same-flow-overfit10")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-variant", choices=["raw", "ema"], default="raw")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args)
    model = instantiate(cfg.model)
    load_info = None
    if args.checkpoint is not None:
        load_info = _load_model_checkpoint(
            model,
            args.checkpoint,
            args.checkpoint_variant,
        )
        print(f"loaded checkpoint {args.checkpoint}: {load_info}", flush=True)
    wrapper = instantiate(cfg.wrapper, model=model).to(device)
    wrapper.train()

    logged = {}

    def capture_log(self, name, value, *log_args, **log_kwargs):
        del log_args, log_kwargs
        logged[name] = value.detach().float().cpu() if torch.is_tensor(value) else value

    wrapper.log = MethodType(capture_log, wrapper)
    optimizer, scheduler = _optimizer_scheduler_from_config(wrapper)

    dataset = instantiate(cfg.data.dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    iterator = iter(loader)
    rows = []
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = (batch[0].to(device), batch[1])

        optimizer.zero_grad(set_to_none=True)
        wrapper._debug_global_step_override = step - 1
        loss = wrapper.training_step(batch, step - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        wrapper.on_before_zero_grad()

        row = {
            "step": step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss_total": float(loss.detach().cpu()),
            "loss_consistency": float(logged.get("loss/consistency", float("nan"))),
            "loss_rvq": float(logged.get("loss/rvq", float("nan"))),
            "rvq_commitment_loss": float(
                logged.get("rvq/commitment_loss", float("nan"))
            ),
            "rvq_hidden_recon_loss": float(
                logged.get("rvq/hidden_recon_loss", float("nan"))
            ),
            "rvq_n_quantizers": float(logged.get("rvq/n_quantizers", float("nan"))),
            "latent_source_std": float(
                logged.get("latent/source_discrete_std", float("nan"))
            ),
            "sigma_high_mean": float(logged.get("sigma/high_mean", float("nan"))),
            "sigma_low_mean": float(logged.get("sigma/low_mean", float("nan"))),
            "consistency_step_size": float(
                logged.get("consistency/step_size", float("nan"))
            ),
        }
        rows.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step} loss={row['loss_total']:.6f} "
                f"lr={row['lr']:.3e} "
                f"latent_std={row['latent_source_std']:.5f} "
                f"sigma={row['sigma_low_mean']:.4f}->{row['sigma_high_mean']:.4f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            _save_checkpoint(wrapper, output_dir / f"step_{step:06d}.ckpt", step)

    with open(output_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if load_info is not None:
        with open(output_dir / "load_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "variant": args.checkpoint_variant,
                    **load_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    _save_checkpoint(wrapper, output_dir / "last.ckpt", args.max_steps)
    print(f"saved {output_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
