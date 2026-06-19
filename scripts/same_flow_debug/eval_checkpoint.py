import argparse
import json
import math
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _load_config(args):
    root = Path(__file__).resolve().parents[2]
    if args.run_dir is not None:
        config_path = Path(args.run_dir) / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"missing Hydra config: {config_path}")
        return OmegaConf.load(config_path)

    overrides = [f"experiment={args.experiment}"]
    overrides.extend(args.override or [])
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "scripts" / "configs"),
    ):
        return compose(config_name="train", overrides=overrides)


def _strip_prefix(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return None
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _load_checkpoint(model, checkpoint_path, variant):
    if checkpoint_path is None:
        return {"loaded": 0, "missing": len(model.state_dict()), "skipped": 0}

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if variant == "ema":
        selected = _strip_prefix(state_dict, "ema.ema_model.")
        if selected is None:
            raise ValueError("checkpoint does not contain ema.ema_model.* keys")
    elif variant == "raw":
        selected = _strip_prefix(state_dict, "model.")
        if selected is None:
            selected = state_dict
    else:
        raise ValueError(f"unsupported variant={variant!r}")

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


def _safe_name(path_value, index):
    if isinstance(path_value, (list, tuple)):
        path_value = path_value[0]
    if path_value is None:
        return f"item_{index:03d}"
    stem = Path(str(path_value)).stem
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:80]


def _align_audio(source, pred):
    length = min(source.shape[-1], pred.shape[-1])
    return source[..., :length], pred[..., :length]


def _as_audio_channels(audio):
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 3 and audio.shape[0] == 1:
        return audio[0]
    return audio


def _save_audio(path, audio, sample_rate):
    audio = audio.detach().cpu().clamp(-1, 1)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def _corrcoef(source, pred):
    source = source.reshape(-1).float()
    pred = pred.reshape(-1).float()
    source = source - source.mean()
    pred = pred - pred.mean()
    denom = source.norm() * pred.norm()
    if denom <= 1e-12:
        return float("nan")
    return float((source @ pred / denom).clamp(-1.0, 1.0))


def _snr_db(source, pred):
    source = source.float()
    pred = pred.float()
    noise = source - pred
    signal_power = source.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(signal_power.clamp_min(1e-12) / noise_power))


def _optimal_gain(source, pred):
    source = source.float()
    pred = pred.float()
    denom = pred.pow(2).sum().clamp_min(1e-12)
    return float((source * pred).sum() / denom)


def _rms(audio):
    return float(audio.float().pow(2).mean().sqrt())


def _lsd_db(source, pred, sample_rate, n_fft=2048, hop_length=512):
    source = source.mean(dim=0).float()
    pred = pred.mean(dim=0).float()
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
    return float(diff.pow(2).mean().sqrt())


def _band_metrics(source, pred, sample_rate, cutoff_hz=8000.0, n_fft=2048, hop_length=512):
    source = source.mean(dim=0).float()
    pred = pred.mean(dim=0).float()
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
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(source_spec.device)
    high = freqs >= float(cutoff_hz)
    low = ~high

    def energy(spec, mask):
        return spec[mask].pow(2).mean().clamp_min(1e-12)

    source_hf = energy(source_spec, high)
    pred_hf = energy(pred_spec, high)
    source_lf = energy(source_spec, low)
    pred_lf = energy(pred_spec, low)
    hf_diff = 20.0 * (
        torch.log10(pred_spec[high].clamp_min(1e-5))
        - torch.log10(source_spec[high].clamp_min(1e-5))
    )
    lf_diff = 20.0 * (
        torch.log10(pred_spec[low].clamp_min(1e-5))
        - torch.log10(source_spec[low].clamp_min(1e-5))
    )
    return {
        "hf_cutoff_hz": float(cutoff_hz),
        "source_hf_rms": float(source_hf.sqrt()),
        "prediction_hf_rms": float(pred_hf.sqrt()),
        "hf_rms_ratio": float((pred_hf.sqrt() / source_hf.sqrt()).clamp_max(1e6)),
        "source_lf_rms": float(source_lf.sqrt()),
        "prediction_lf_rms": float(pred_lf.sqrt()),
        "lf_rms_ratio": float((pred_lf.sqrt() / source_lf.sqrt()).clamp_max(1e6)),
        "hf_lsd_db": float(hf_diff.pow(2).mean().sqrt()),
        "lf_lsd_db": float(lf_diff.pow(2).mean().sqrt()),
    }


def _mrstft_metrics(source, pred):
    source = source.mean(dim=0).float()
    pred = pred.mean(dim=0).float()
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
        "mrstft_sc": float(torch.stack([x[0] for x in rows]).mean()),
        "mrstft_log_mag_l1": float(torch.stack([x[1] for x in rows]).mean()),
    }


def _logmel_metrics(source, pred, sample_rate):
    source = source.mean(dim=0).float()
    pred = pred.mean(dim=0).float()
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
        "logmel_l1": float(diff.abs().mean()),
        "logmel_l2": float(diff.pow(2).mean().sqrt()),
    }


def _mean(values):
    values = [v for v in values if not math.isnan(v)]
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="same-flow-overfit10")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=["raw", "ema"], default="raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument(
        "--sampling-mode",
        choices=["stochastic", "deterministic"],
        default="stochastic",
    )
    parser.add_argument("--direct-sigma", type=float, default=None)
    parser.add_argument(
        "--mismatch-latent-offset",
        type=int,
        default=0,
        help="With --direct-sigma, also denoise each item with latent from index+offset.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quiet-rms-threshold", type=float, default=0.01)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = _load_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model).to(device)
    load_info = _load_checkpoint(model, args.checkpoint, args.variant)
    model.eval()

    dataset = instantiate(cfg.data.dataset)
    sample_rate = int(cfg.model.sample_rate)
    rows = []
    for index in range(min(args.max_items, len(dataset))):
        source, info = dataset[index]
        source = _as_audio_channels(source).to(device)
        torch.manual_seed(args.seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + index)
        representation = model.audio_processor.to_representation_encoder(
            source.unsqueeze(0)
        )
        latent = model.encoder(representation)
        pred = _as_audio_channels(
            model.decode(
                latent,
                denoising_steps=args.denoising_steps,
                target_length=source.shape[-1],
                sampling_mode=args.sampling_mode,
            ).to(device)
        )
        source_aligned, pred_aligned = _align_audio(source, pred)

        name = _safe_name(info.get("path") if isinstance(info, dict) else None, index)
        item_dir = output_dir / f"{index:03d}_{name}"
        item_dir.mkdir(parents=True, exist_ok=True)
        _save_audio(item_dir / "source.wav", source_aligned, sample_rate)
        _save_audio(
            item_dir / f"reconstruct_{args.variant}.wav",
            pred_aligned,
            sample_rate,
        )

        row = {
            "index": index,
            "path": info.get("path") if isinstance(info, dict) else None,
            "source_wav": str(item_dir / "source.wav"),
            "prediction_wav": str(item_dir / f"reconstruct_{args.variant}.wav"),
            "latent_shape": list(latent.shape),
            "latent_mean": float(latent.detach().float().mean()),
            "latent_std": float(latent.detach().float().std()),
            "source_rms": _rms(source_aligned),
            "prediction_rms": _rms(pred_aligned),
            "corr": _corrcoef(source_aligned, pred_aligned),
            "snr_db": _snr_db(source_aligned, pred_aligned),
            "lsd_db": _lsd_db(source_aligned, pred_aligned, sample_rate),
        }
        row.update(_band_metrics(source_aligned, pred_aligned, sample_rate))
        row.update(_mrstft_metrics(source_aligned, pred_aligned))
        row.update(_logmel_metrics(source_aligned, pred_aligned, sample_rate))
        gain = _optimal_gain(source_aligned, pred_aligned)
        row.update(
            {
                "optimal_gain": gain,
                "optimal_gain_snr_db": _snr_db(source_aligned, pred_aligned * gain),
                "optimal_gain_prediction_rms": _rms(pred_aligned * gain),
            }
        )
        if args.direct_sigma is not None:
            noise = torch.randn_like(representation)
            sigma = torch.full(
                (representation.shape[0],),
                float(args.direct_sigma),
                device=representation.device,
                dtype=representation.dtype,
            )
            noisy = representation + noise * sigma.reshape(-1, 1, 1, 1)
            denoised_repr = model(
                representation,
                noisy,
                sigma=sigma,
                latent_override=latent,
            )
            denoised = _as_audio_channels(
                model.audio_processor.to_waveform(denoised_repr, model.hop)
            )
            denoised = denoised[..., : source.shape[-1]]
            direct_source, direct_pred = _align_audio(source, denoised)
            _save_audio(
                item_dir / f"direct_sigma{args.direct_sigma:g}_{args.variant}.wav",
                direct_pred,
                sample_rate,
            )
            row.update(
                {
                    "direct_sigma": float(args.direct_sigma),
                    "direct_rms": _rms(direct_pred),
                    "direct_corr": _corrcoef(direct_source, direct_pred),
                    "direct_snr_db": _snr_db(direct_source, direct_pred),
                    "direct_lsd_db": _lsd_db(direct_source, direct_pred, sample_rate),
                }
            )
            row.update(
                {
                    f"direct_{key}": value
                    for key, value in _band_metrics(
                        direct_source,
                        direct_pred,
                        sample_rate,
                    ).items()
                }
            )
            row.update(
                {
                    f"direct_{key}": value
                    for key, value in _mrstft_metrics(
                        direct_source,
                        direct_pred,
                    ).items()
                }
            )
            row.update(
                {
                    f"direct_{key}": value
                    for key, value in _logmel_metrics(
                        direct_source,
                        direct_pred,
                        sample_rate,
                    ).items()
                }
            )
            direct_gain = _optimal_gain(direct_source, direct_pred)
            row.update(
                {
                    "direct_optimal_gain": direct_gain,
                    "direct_optimal_gain_snr_db": _snr_db(
                        direct_source,
                        direct_pred * direct_gain,
                    ),
                }
            )
            if args.mismatch_latent_offset:
                mismatch_index = (index + args.mismatch_latent_offset) % len(dataset)
                mismatch_source, _ = dataset[mismatch_index]
                mismatch_source = _as_audio_channels(mismatch_source).to(device)
                mismatch_representation = model.audio_processor.to_representation_encoder(
                    mismatch_source.unsqueeze(0)
                )
                mismatch_latent = model.encoder(mismatch_representation)
                mismatch_repr = model(
                    representation,
                    noisy,
                    sigma=sigma,
                    latent_override=mismatch_latent,
                )
                mismatch_audio = _as_audio_channels(
                    model.audio_processor.to_waveform(
                        mismatch_repr,
                        model.hop,
                    )
                )
                mismatch_audio = mismatch_audio[..., : source.shape[-1]]
                mismatch_source_aligned, mismatch_pred = _align_audio(
                    source,
                    mismatch_audio,
                )
                _save_audio(
                    item_dir
                    / f"direct_sigma{args.direct_sigma:g}_mismatch{args.mismatch_latent_offset}_{args.variant}.wav",
                    mismatch_pred,
                    sample_rate,
                )
                row.update(
                    {
                        "mismatch_latent_index": int(mismatch_index),
                        "mismatch_latent_std": float(
                            mismatch_latent.detach().float().std()
                        ),
                        "mismatch_direct_rms": _rms(mismatch_pred),
                        "mismatch_direct_delta_rms": _rms(
                            direct_pred - mismatch_pred
                        ),
                        "mismatch_direct_corr": _corrcoef(
                            mismatch_source_aligned,
                            mismatch_pred,
                        ),
                        "mismatch_direct_snr_db": _snr_db(
                            mismatch_source_aligned,
                            mismatch_pred,
                        ),
                        "mismatch_direct_lsd_db": _lsd_db(
                            mismatch_source_aligned,
                            mismatch_pred,
                            sample_rate,
                        ),
                    }
                )
        rows.append(row)
        print(
            f"{index:03d} corr={row['corr']:.4f} "
            f"snr={row['snr_db']:.2f} lsd={row['lsd_db']:.2f} "
            f"hf_ratio={row['hf_rms_ratio']:.3f} "
            f"latent_std={row['latent_std']:.4f}"
            + (
                f" direct_corr={row['direct_corr']:.4f} "
                f"direct_snr={row['direct_snr_db']:.2f}"
                if args.direct_sigma is not None
                else ""
            ),
            flush=True,
        )

    summary = {
        "experiment": args.experiment,
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "load_info": load_info,
        "max_items": args.max_items,
        "denoising_steps": args.denoising_steps,
        "sampling_mode": args.sampling_mode,
        "direct_sigma": args.direct_sigma,
        "seed": args.seed,
        "corr_mean": _mean([row["corr"] for row in rows]),
        "snr_db_mean": _mean([row["snr_db"] for row in rows]),
        "lsd_db_mean": _mean([row["lsd_db"] for row in rows]),
        "latent_std_mean": _mean([row["latent_std"] for row in rows]),
        "source_rms_mean": _mean([row["source_rms"] for row in rows]),
        "prediction_rms_mean": _mean([row["prediction_rms"] for row in rows]),
        "hf_rms_ratio_mean": _mean([row["hf_rms_ratio"] for row in rows]),
        "lf_rms_ratio_mean": _mean([row["lf_rms_ratio"] for row in rows]),
        "hf_lsd_db_mean": _mean([row["hf_lsd_db"] for row in rows]),
        "lf_lsd_db_mean": _mean([row["lf_lsd_db"] for row in rows]),
        "mrstft_sc_mean": _mean([row["mrstft_sc"] for row in rows]),
        "mrstft_log_mag_l1_mean": _mean(
            [row["mrstft_log_mag_l1"] for row in rows]
        ),
        "logmel_l1_mean": _mean([row["logmel_l1"] for row in rows]),
        "logmel_l2_mean": _mean([row["logmel_l2"] for row in rows]),
        "optimal_gain_mean": _mean([row["optimal_gain"] for row in rows]),
        "optimal_gain_snr_db_mean": _mean(
            [row["optimal_gain_snr_db"] for row in rows]
        ),
        "optimal_gain_prediction_rms_mean": _mean(
            [row["optimal_gain_prediction_rms"] for row in rows]
        ),
        "items": rows,
    }
    quiet_rows = [
        row for row in rows if row["source_rms"] < float(args.quiet_rms_threshold)
    ]
    nonquiet_rows = [
        row for row in rows if row["source_rms"] >= float(args.quiet_rms_threshold)
    ]
    summary.update(
        {
            "quiet_rms_threshold": float(args.quiet_rms_threshold),
            "quiet_count": len(quiet_rows),
            "nonquiet_count": len(nonquiet_rows),
            "nonquiet_corr_mean": _mean([row["corr"] for row in nonquiet_rows]),
            "nonquiet_snr_db_mean": _mean(
                [row["snr_db"] for row in nonquiet_rows]
            ),
            "nonquiet_mrstft_sc_mean": _mean(
                [row["mrstft_sc"] for row in nonquiet_rows]
            ),
            "nonquiet_logmel_l1_mean": _mean(
                [row["logmel_l1"] for row in nonquiet_rows]
            ),
            "quiet_source_rms_mean": _mean(
                [row["source_rms"] for row in quiet_rows]
            ),
            "quiet_prediction_rms_mean": _mean(
                [row["prediction_rms"] for row in quiet_rows]
            ),
            "quiet_snr_db_mean": _mean([row["snr_db"] for row in quiet_rows]),
        }
    )
    if args.direct_sigma is not None:
        summary.update(
            {
                "direct_corr_mean": _mean([row["direct_corr"] for row in rows]),
                "direct_snr_db_mean": _mean([row["direct_snr_db"] for row in rows]),
                "direct_lsd_db_mean": _mean([row["direct_lsd_db"] for row in rows]),
                "direct_rms_mean": _mean([row["direct_rms"] for row in rows]),
                "direct_hf_rms_ratio_mean": _mean(
                    [row["direct_hf_rms_ratio"] for row in rows]
                ),
                "direct_lf_rms_ratio_mean": _mean(
                    [row["direct_lf_rms_ratio"] for row in rows]
                ),
                "direct_hf_lsd_db_mean": _mean(
                    [row["direct_hf_lsd_db"] for row in rows]
                ),
                "direct_lf_lsd_db_mean": _mean(
                    [row["direct_lf_lsd_db"] for row in rows]
                ),
                "direct_mrstft_sc_mean": _mean(
                    [row["direct_mrstft_sc"] for row in rows]
                ),
                "direct_mrstft_log_mag_l1_mean": _mean(
                    [row["direct_mrstft_log_mag_l1"] for row in rows]
                ),
                "direct_logmel_l1_mean": _mean(
                    [row["direct_logmel_l1"] for row in rows]
                ),
                "direct_logmel_l2_mean": _mean(
                    [row["direct_logmel_l2"] for row in rows]
                ),
                "direct_optimal_gain_mean": _mean(
                    [row["direct_optimal_gain"] for row in rows]
                ),
                "direct_optimal_gain_snr_db_mean": _mean(
                    [row["direct_optimal_gain_snr_db"] for row in rows]
                ),
                "nonquiet_direct_corr_mean": _mean(
                    [row["direct_corr"] for row in nonquiet_rows]
                ),
                "nonquiet_direct_snr_db_mean": _mean(
                    [row["direct_snr_db"] for row in nonquiet_rows]
                ),
                "nonquiet_direct_mrstft_sc_mean": _mean(
                    [row["direct_mrstft_sc"] for row in nonquiet_rows]
                ),
                "quiet_direct_rms_mean": _mean(
                    [row["direct_rms"] for row in quiet_rows]
                ),
            }
        )
        if args.mismatch_latent_offset:
            summary.update(
                {
                    "mismatch_direct_corr_mean": _mean(
                        [row["mismatch_direct_corr"] for row in rows]
                    ),
                    "mismatch_direct_snr_db_mean": _mean(
                        [row["mismatch_direct_snr_db"] for row in rows]
                    ),
                    "mismatch_direct_lsd_db_mean": _mean(
                        [row["mismatch_direct_lsd_db"] for row in rows]
                    ),
                    "mismatch_direct_rms_mean": _mean(
                        [row["mismatch_direct_rms"] for row in rows]
                    ),
                    "mismatch_direct_delta_rms_mean": _mean(
                        [row["mismatch_direct_delta_rms"] for row in rows]
                    ),
                }
            )
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
