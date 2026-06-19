import argparse
import json
from pathlib import Path

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _load_config(args):
    root = Path(__file__).resolve().parents[2]
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
            raise ValueError(f"{checkpoint_path} has no ema.ema_model.* state")
    else:
        selected = _strip_prefix(state_dict, "model.")
        if selected is None:
            selected = state_dict
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
    return {"loaded": len(compatible), "missing": len(current) - len(compatible), "skipped": len(skipped)}


def _tensor_stats(x):
    x = x.detach().float()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().cpu()),
        "std": float(x.std().cpu()),
        "rms": float(x.pow(2).mean().sqrt().cpu()),
        "abs_mean": float(x.abs().mean().cpu()),
    }


def _audio_align(a, b):
    length = min(a.shape[-1], b.shape[-1])
    return a[..., :length], b[..., :length]


def _corr(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom <= 1e-12:
        return float("nan")
    return float((a @ b / denom).clamp(-1, 1).cpu())


def _snr_db(source, pred):
    source = source.detach().float()
    pred = pred.detach().float()
    signal = source.pow(2).mean().clamp_min(1e-12)
    noise = (source - pred).pow(2).mean().clamp_min(1e-12)
    return float((10 * torch.log10(signal / noise)).cpu())


def _stft_mag(x, n_fft, hop):
    x = x.detach().float().mean(dim=-2)
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()


def _mrstft(source, pred):
    rows = []
    for n_fft in (2048, 1024, 512, 256, 128, 64, 32):
        hop = n_fft // 4
        src = _stft_mag(source, n_fft, hop)
        prd = _stft_mag(pred, n_fft, hop)
        sc = (src - prd).norm() / src.norm().clamp_min(1e-12)
        log_mag = (torch.log(src.clamp_min(1e-5)) - torch.log(prd.clamp_min(1e-5))).abs().mean()
        rows.append((sc, log_mag))
    return {
        "mrstft_sc": float(torch.stack([r[0] for r in rows]).mean().cpu()),
        "mrstft_log_mag_l1": float(torch.stack([r[1] for r in rows]).mean().cpu()),
    }


def _module_interesting(name):
    exact = {
        "adapter",
        "encoder",
        "decoder",
        "input_proj",
        "decoded_proj",
        "output_proj",
        "prior_proj",
    }
    if name in exact:
        return True
    return (
        name.startswith("down_blocks.")
        or name.startswith("up_blocks.")
        or name.startswith("down_transitions.")
        or name.startswith("up_transitions.")
        or name.startswith("down_latent_films.")
        or name.startswith("up_latent_films.")
        or name.startswith("down_patch_embeddings.")
        or name.startswith("up_patch_embeddings.")
    )


class TraceCollector:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.rows = {}

    def _hook(self, name):
        def fn(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.rows.setdefault(name, []).append(_tensor_stats(output))
            elif isinstance(output, (list, tuple)):
                values = []
                for item in output:
                    if isinstance(item, torch.Tensor):
                        values.append(_tensor_stats(item))
                if values:
                    self.rows.setdefault(name, []).append(values)
        return fn

    def __enter__(self):
        for name, module in self.model.named_modules():
            if _module_interesting(name):
                self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        return False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="same-flow-debug-overfit10")
    parser.add_argument("--checkpoint")
    parser.add_argument("--variant", choices=["raw", "ema"], default="raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--item", type=int, default=0)
    parser.add_argument("--mismatch-item", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = _load_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model).to(device)
    load_info = _load_checkpoint(model, args.checkpoint, args.variant)
    model.eval()
    dataset = instantiate(cfg.data.dataset)
    source, info = dataset[args.item]
    mismatch_source, mismatch_info = dataset[args.mismatch_item]
    if source.ndim == 3:
        source = source[0]
    if mismatch_source.ndim == 3:
        mismatch_source = mismatch_source[0]
    source = source.unsqueeze(0).to(device)
    mismatch_source = mismatch_source.unsqueeze(0).to(device)

    representation = model.audio_processor.to_representation_encoder(source)
    mismatch_representation = model.audio_processor.to_representation_encoder(mismatch_source)
    latent = model.encoder(representation)
    mismatch_latent = model.encoder(mismatch_representation)
    pyramid = model.decoder(latent)
    mismatch_pyramid = model.decoder(mismatch_latent)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(representation.shape, device=device, dtype=representation.dtype, generator=generator)
    sigma = torch.full((representation.shape[0],), float(args.sigma), device=device, dtype=representation.dtype)
    noisy = representation + noise * sigma.reshape(-1, 1, 1, 1)

    with TraceCollector(model) as tracer:
        pred_repr = model(representation, noisy, sigma=sigma, latent_override=latent)
    with TraceCollector(model) as mismatch_tracer:
        mismatch_pred_repr = model(
            representation,
            noisy,
            sigma=sigma,
            latent_override=mismatch_latent,
        )
    pred_audio = model.audio_processor.to_waveform(pred_repr, model.hop)[..., : source.shape[-1]]
    mismatch_audio = model.audio_processor.to_waveform(mismatch_pred_repr, model.hop)[..., : source.shape[-1]]
    source_aligned, pred_aligned = _audio_align(source[0], pred_audio[0])
    _, mismatch_aligned = _audio_align(source[0], mismatch_audio[0])

    out = {
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "load_info": load_info,
        "item": int(args.item),
        "item_path": info.get("path") if isinstance(info, dict) else None,
        "mismatch_item": int(args.mismatch_item),
        "mismatch_path": mismatch_info.get("path") if isinstance(mismatch_info, dict) else None,
        "sigma": float(args.sigma),
        "representation": _tensor_stats(representation),
        "latent": _tensor_stats(latent),
        "mismatch_latent": _tensor_stats(mismatch_latent),
        "latent_delta_rms": float((latent - mismatch_latent).detach().float().pow(2).mean().sqrt().cpu()),
        "pyramid": [_tensor_stats(x) for x in pyramid],
        "mismatch_pyramid": [_tensor_stats(x) for x in mismatch_pyramid],
        "pyramid_delta_rms": [
            float((a - b).detach().float().pow(2).mean().sqrt().cpu())
            for a, b in zip(pyramid, mismatch_pyramid)
        ],
        "pred_repr": _tensor_stats(pred_repr),
        "mismatch_pred_repr": _tensor_stats(mismatch_pred_repr),
        "pred_delta_rms": float((pred_repr - mismatch_pred_repr).detach().float().pow(2).mean().sqrt().cpu()),
        "audio": {
            "source_rms": float(source_aligned.pow(2).mean().sqrt().cpu()),
            "pred_rms": float(pred_aligned.pow(2).mean().sqrt().cpu()),
            "mismatch_pred_rms": float(mismatch_aligned.pow(2).mean().sqrt().cpu()),
            "pred_vs_source_corr": _corr(source_aligned, pred_aligned),
            "mismatch_vs_source_corr": _corr(source_aligned, mismatch_aligned),
            "snr_db": _snr_db(source_aligned, pred_aligned),
            "mismatch_snr_db": _snr_db(source_aligned, mismatch_aligned),
            "pred_mismatch_audio_delta_rms": float((pred_aligned - mismatch_aligned).pow(2).mean().sqrt().cpu()),
            **_mrstft(source_aligned, pred_aligned),
        },
        "trace": tracer.rows,
        "mismatch_trace": mismatch_tracer.rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in {"trace", "mismatch_trace"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
