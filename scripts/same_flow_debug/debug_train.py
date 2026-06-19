import argparse
import json
from pathlib import Path
from types import MethodType

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def _rms(x):
    return float(x.detach().float().pow(2).mean().sqrt().cpu())


def _snr_db(source, pred):
    source = source.detach().float()
    pred = pred.detach().float()
    noise = source - pred
    signal = source.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    return float((10.0 * torch.log10(signal / noise_power)).cpu())


def _as_mono_batch(audio):
    audio = audio.detach().float()
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 2:
        return audio
    if audio.ndim == 3:
        return audio.mean(dim=-2)
    raise ValueError(f"expected audio with 1-3 dims, got {tuple(audio.shape)}")


def _lsd_db(source, pred, n_fft=2048, hop_length=512):
    source = _as_mono_batch(source)
    pred = _as_mono_batch(pred)
    window = torch.hann_window(n_fft, device=source.device)
    source_spec = torch.stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    pred_spec = torch.stft(
        pred,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    diff = 20.0 * (
        torch.log10(pred_spec.clamp_min(1e-5))
        - torch.log10(source_spec.clamp_min(1e-5))
    )
    return float(diff.pow(2).mean().sqrt().cpu())


def _hf_ratio(source, pred, sample_rate, cutoff_hz=8000.0, n_fft=2048, hop_length=512):
    source = _as_mono_batch(source)
    pred = _as_mono_batch(pred)
    window = torch.hann_window(n_fft, device=source.device)
    source_spec = torch.stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    pred_spec = torch.stft(
        pred,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(source.device)
    high = freqs >= float(cutoff_hz)
    source_hf = source_spec[:, high, :].pow(2).mean().sqrt().clamp_min(1e-12)
    pred_hf = pred_spec[:, high, :].pow(2).mean().sqrt()
    return float((pred_hf / source_hf).clamp_max(1e6).cpu())


def _mrstft_metrics(source, pred):
    source = _as_mono_batch(source)
    pred = _as_mono_batch(pred)
    rows = []
    for n_fft in (2048, 1024, 512, 256, 128, 64, 32):
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft, device=source.device)
        source_spec = torch.stft(
            source,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        pred_spec = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        sc = (source_spec - pred_spec).norm() / source_spec.norm().clamp_min(1e-12)
        log_mag = (
            torch.log(source_spec.clamp_min(1e-5))
            - torch.log(pred_spec.clamp_min(1e-5))
        ).abs().mean()
        rows.append((sc, log_mag))
    return {
        "probe_mrstft_sc": float(torch.stack([x[0] for x in rows]).mean().cpu()),
        "probe_mrstft_log_mag_l1": float(
            torch.stack([x[1] for x in rows]).mean().cpu()
        ),
    }


def _logmel_metrics(source, pred, sample_rate):
    source = _as_mono_batch(source)
    pred = _as_mono_batch(pred)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sample_rate),
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20.0,
        f_max=min(float(sample_rate) / 2.0, 24000.0),
        power=2.0,
    ).to(source.device)
    source_mel = torch.log10(mel(source).clamp_min(1e-10))
    pred_mel = torch.log10(mel(pred).clamp_min(1e-10))
    diff = source_mel - pred_mel
    return {
        "probe_logmel_l1": float(diff.abs().mean().cpu()),
        "probe_logmel_l2": float(diff.pow(2).mean().sqrt().cpu()),
    }


def _load_config(args):
    root = Path(__file__).resolve().parents[2]
    overrides = [f"experiment={args.experiment}"]
    overrides.extend(args.override or [])
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "scripts" / "configs"),
    ):
        return compose(config_name="train", overrides=overrides)


def _crop_to_valid_stft_length(model, audio):
    hop = model.hop
    downscaling_factor = 2 ** sum(1 for x in model.freq_downsample_list if x == 0)
    center_pad = bool(getattr(model.audio_processor, "center_pad", True))
    fac = int(getattr(model.audio_processor, "fac", 4))
    if center_pad:
        cropped_frames = (audio.shape[-1] // hop) // downscaling_factor
        cropped_length = cropped_frames * hop * downscaling_factor
    else:
        min_extra = max(fac - 1, 0) * hop
        available_frames = max((audio.shape[-1] - min_extra) // hop, 0)
        cropped_frames = available_frames // downscaling_factor
        cropped_length = cropped_frames * hop * downscaling_factor + min_extra
    return audio[..., :cropped_length]


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


@torch.no_grad()
def _make_probe(wrapper, dataset, device, seed, probe_sigma=None):
    source, info = dataset[0]
    if source.ndim == 3:
        source = source[0]
    source = source.unsqueeze(0).to(device)
    source = _crop_to_valid_stft_length(wrapper.model, source)
    representation = wrapper.model.audio_processor.to_representation_encoder(source)
    latent = wrapper.model.encoder(representation)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 999)
    noise = torch.randn(
        representation.shape,
        generator=generator,
        device=device,
        dtype=representation.dtype,
    )
    sigma_value = wrapper.model.sigma_max if probe_sigma is None else float(probe_sigma)
    sigma = torch.full(
        (representation.shape[0],),
        float(sigma_value),
        device=device,
        dtype=representation.dtype,
    )
    noisy = representation + noise * sigma.reshape(-1, 1, 1, 1)
    return {
        "source": source,
        "representation": representation,
        "latent": latent,
        "noisy": noisy,
        "sigma": sigma,
        "sigma_value": float(sigma_value),
        "path": info.get("path") if isinstance(info, dict) else None,
    }


@torch.no_grad()
def _run_probe(wrapper, probe, sample_rate):
    was_training = wrapper.model.training
    wrapper.model.eval()
    pred_repr = wrapper.model(
        probe["representation"],
        probe["noisy"],
        sigma=probe["sigma"],
        latent_override=probe["latent"],
    )
    pred_audio = wrapper.model.audio_processor.to_waveform(
        pred_repr,
        wrapper.model.hop,
    )[..., : probe["source"].shape[-1]]
    if was_training:
        wrapper.model.train()
    repr_mse = float(F.mse_loss(pred_repr.float(), probe["representation"].float()).cpu())
    row = {
        "probe_repr_mse": repr_mse,
        "probe_source_rms": _rms(probe["source"]),
        "probe_pred_rms": _rms(pred_audio),
        "probe_snr_db": _snr_db(probe["source"], pred_audio),
        "probe_lsd_db": _lsd_db(probe["source"], pred_audio),
        "probe_hf_ratio": _hf_ratio(probe["source"], pred_audio, sample_rate),
        "probe_sigma": float(probe["sigma_value"]),
    }
    row.update(_mrstft_metrics(probe["source"], pred_audio))
    row.update(_logmel_metrics(probe["source"], pred_audio, sample_rate))
    return row


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
    parser.add_argument("--probe-every", type=int, default=0)
    parser.add_argument("--probe-sigma", type=float, default=None)
    parser.add_argument(
        "--fixed-probe-batch",
        action="store_true",
        help="Train every step on dataset[0] with the same latent, noise, and sigma.",
    )
    parser.add_argument(
        "--fixed-source-random-noise",
        action="store_true",
        help="Train every step on dataset[0] and fixed latent, but resample noise each step.",
    )
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
    probe = None
    if args.probe_every > 0:
        probe = _make_probe(wrapper, dataset, device, args.seed, args.probe_sigma)
        print(
            f"probe_path={probe['path']} probe_sigma={probe['sigma_value']}",
            flush=True,
        )
    rows = []
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        wrapper._debug_global_step_override = step - 1
        if args.fixed_probe_batch or args.fixed_source_random_noise:
            if probe is None:
                raise RuntimeError(
                    "--fixed-probe-batch/--fixed-source-random-noise requires "
                    "--probe-every > 0"
                )
            noisy = probe["noisy"]
            if args.fixed_source_random_noise:
                noise = torch.randn_like(probe["representation"])
                noisy = probe["representation"] + noise * probe["sigma"].reshape(
                    -1,
                    1,
                    1,
                    1,
                )
            loss, metrics, _ = wrapper.fixed_max_denoise_step(
                probe["representation"],
                probe["latent"],
                noisy,
                probe["sigma"],
            )
            for name, value in metrics.items():
                logged[name] = value.detach().float().cpu()
            logged["latent/source_discrete_std"] = (
                probe["latent"].detach().float().std().cpu()
            )
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = (batch[0].to(device), batch[1])
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
            "latent_source_std": float(
                logged.get("latent/source_discrete_std", float("nan"))
            ),
            "sigma_high_mean": float(logged.get("sigma/high_mean", float("nan"))),
            "sigma_low_mean": float(logged.get("sigma/low_mean", float("nan"))),
            "consistency_step_size": float(
                logged.get("consistency/step_size", float("nan"))
            ),
        }
        if probe is not None and (step == 1 or step % args.probe_every == 0):
            row.update(_run_probe(wrapper, probe, int(cfg.model.sample_rate)))
        rows.append(row)
        if step % args.log_every == 0 or step == 1:
            probe_text = ""
            if "probe_snr_db" in row:
                probe_text = (
                    f" probe_mse={row['probe_repr_mse']:.5f}"
                    f" probe_snr={row['probe_snr_db']:.2f}"
                    f" probe_lsd={row['probe_lsd_db']:.2f}"
                    f" probe_mrstft={row['probe_mrstft_sc']:.3f}"
                    f" probe_mel={row['probe_logmel_l1']:.3f}"
                    f" probe_hf={row['probe_hf_ratio']:.3f}"
                    f" probe_rms={row['probe_pred_rms']:.4f}"
                )
            print(
                f"step={step} loss={row['loss_total']:.6f} "
                f"lr={row['lr']:.3e} "
                f"latent_std={row['latent_source_std']:.5f} "
                f"sigma={row['sigma_low_mean']:.4f}->{row['sigma_high_mean']:.4f}"
                f"{probe_text}",
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
