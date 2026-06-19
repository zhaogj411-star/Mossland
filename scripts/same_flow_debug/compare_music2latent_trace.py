import argparse
import json
import sys
import types
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT / "tmp" / "music2latent-training (1)" / "music2latent-training"


class _NoFadClapModule:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("CLAP/FAD is disabled in trace")


def _install_music2latent_imports():
    sys.path.insert(0, str(ORIGINAL_ROOT))
    stub = types.ModuleType("laion_clap")
    stub.CLAP_Module = _NoFadClapModule
    sys.modules.setdefault("laion_clap", stub)


def _stats(tensor):
    x = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "mean": float(x.mean().cpu()),
        "std": float(x.std().cpu()),
        "rms": float(x.pow(2).mean().sqrt().cpu()),
        "abs_mean": float(x.abs().mean().cpu()),
        "min": float(x.min().cpu()),
        "max": float(x.max().cpu()),
    }


def _add(rows, model_name, stage, tensor, ref=None):
    row = {"model": model_name, "stage": stage}
    row.update(_stats(tensor))
    if ref is not None and tuple(ref.shape) == tuple(tensor.shape):
        ref_x = ref.detach().float()
        x = tensor.detach().float()
        row["mse_to_ref"] = float(torch.nn.functional.mse_loss(x, ref_x).cpu())
        row["rms_ratio_to_ref"] = float(
            (x.pow(2).mean().sqrt() / ref_x.pow(2).mean().sqrt().clamp_min(1e-12)).cpu()
        )
    rows.append(row)
    return tensor


def _load_same_model(experiment, checkpoint, device, overrides=None):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "scripts" / "configs"),
    ):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"] + list(overrides or []),
        )
    model = instantiate(cfg.model).to(device).eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        selected = {
            key[len("model.") :]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
        if not selected:
            selected = state
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in selected.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(compatible)
        model.load_state_dict(current, strict=True)
    dataset = instantiate(cfg.data.dataset)
    return cfg, model, dataset


def _load_official_model(config_path, checkpoint, device):
    _install_music2latent_imports()
    from music2latent.config_loader import load_config
    from music2latent.models import UNet

    load_config(config_path)
    model = UNet().to(device).eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["gen_state_dict"], strict=True)
    return model


def _same_audio_from_dataset(dataset, index, device):
    audio, info = dataset[index]
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    elif audio.ndim == 2:
        audio = audio.mean(dim=0)
    return audio.unsqueeze(0).to(device), info


def _official_trace(model, audio, sigma_value, seed, device):
    from music2latent.audio import to_representation, to_representation_encoder
    from music2latent.utils import add_noise, get_c, huber

    rows = []
    with torch.no_grad():
        data = to_representation(audio)
        data_encoder = to_representation_encoder(audio)
        _add(rows, "music2latent", "representation", data)
        _add(rows, "music2latent", "representation_encoder", data_encoder)
        latents = _add(rows, "music2latent", "encoder.latent", model.encoder(data_encoder))
        pyramid = model.decoder(latents)
        for idx, value in enumerate(pyramid):
            _add(rows, "music2latent", f"decoder.pyramid.{idx}", value)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(data.shape, generator=generator, device=device, dtype=data.dtype)
        sigma = torch.full((data.shape[0],), float(sigma_value), device=device)
        noisy = add_noise(data, noise, sigma)
        _add(rows, "music2latent", "noise", noise)
        _add(rows, "music2latent", "noisy", noisy, ref=data)

        inp = noisy
        c_skip, c_out, c_in = get_c(sigma)
        x = _add(rows, "music2latent", "edm.c_in_x", c_in * noisy)
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = model.emb(sigma_log)
        time_emb = model.emb_proj(emb_sigma_log)
        scale_w_inp = model.scale_inp(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)
        scale_w_out = model.scale_out(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)

        x = model.conv_inp(x)
        _add(rows, "music2latent", "denoiser.input_proj", x)
        if getattr(sys.modules["music2latent.hparams"], "hparams").frequency_scaling:
            x = (1.0 + scale_w_inp) * x
            _add(rows, "music2latent", "denoiser.input_scaled", x)

        skip_list = []
        k = 0
        for idx, num_layers in enumerate(model.layers_list):
            for block_idx in range(int(num_layers)):
                d = model.down_layers[k](pyramid[idx])
                _add(rows, "music2latent", f"down.{idx}.{block_idx}.latent_proj", d)
                k += 1
                x = (x + d) / (2.0**0.5)
                _add(rows, "music2latent", f"down.{idx}.{block_idx}.mixed", x)
                x = model.down_layers[k](x, time_emb)
                _add(rows, "music2latent", f"down.{idx}.{block_idx}.block", x)
                skip_list.append(x)
                k += 1
            if idx != len(model.layers_list) - 1:
                x = model.down_layers[k](x)
                _add(rows, "music2latent", f"down.{idx}.transition", x)
                k += 1

        k = 0
        reversed_layers = list(reversed(model.layers_list))
        for idx, num_layers in enumerate(reversed_layers):
            pyramid_idx = len(pyramid) - idx - 1
            for block_idx in range(int(num_layers)):
                d = model.up_layers[k](pyramid[pyramid_idx])
                _add(rows, "music2latent", f"up.{idx}.{block_idx}.latent_proj", d)
                k += 1
                skip = skip_list.pop()
                x = (x + skip + d) / (3.0**0.5)
                _add(rows, "music2latent", f"up.{idx}.{block_idx}.mixed", x)
                x = model.up_layers[k](x, time_emb)
                _add(rows, "music2latent", f"up.{idx}.{block_idx}.block", x)
                k += 1
            if idx != len(reversed_layers) - 1:
                x = model.up_layers[k](x)
                _add(rows, "music2latent", f"up.{idx}.transition", x)
                k += 1

        d = model.conv_decoded(pyramid[0])
        _add(rows, "music2latent", "decoded_proj", d)
        x = (x + d) / (2.0**0.5)
        _add(rows, "music2latent", "final.mixed", x)
        x = model.norm_out(x)
        x = model.activation_out(x)
        _add(rows, "music2latent", "final.norm_act", x)
        x = (1.0 + scale_w_out) * x
        _add(rows, "music2latent", "final.scaled", x)
        residual = model.conv_out(x)
        _add(rows, "music2latent", "final.residual_head", residual)
        out = c_skip * inp + c_out * residual
        _add(rows, "music2latent", "edm.output", out, ref=data)
        loss = huber(out, data)
    return rows, {
        "representation": _stats(data),
        "output": _stats(out),
        "direct_huber_to_clean": float(loss.cpu()),
    }


def _same_trace(model, audio, sigma_value, seed, device):
    rows = []
    with torch.no_grad():
        data = model.audio_processor.to_representation_encoder(audio)
        _add(rows, "same_flow", "representation", data)
        latents = _add(rows, "same_flow", "encoder.latent", model.encoder(data))
        pyramid = model.decoder(latents)
        for idx, value in enumerate(pyramid):
            _add(rows, "same_flow", f"decoder.pyramid.{idx}", value)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(data.shape, generator=generator, device=device, dtype=data.dtype)
        sigma = torch.full((data.shape[0],), float(sigma_value), device=device)
        noisy = data + noise * sigma.reshape(-1, 1, 1, 1)
        _add(rows, "same_flow", "noise", noise)
        _add(rows, "same_flow", "noisy", noisy, ref=data)

        inp = noisy
        c_skip, c_out, c_in = model._get_c(sigma)
        x = _add(rows, "same_flow", "edm.c_in_x", c_in * noisy)
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = model.emb(sigma_log.to(x.dtype)).to(x.dtype)
        time_emb = model.emb_proj(emb_sigma_log)

        if model.use_patch_denoiser:
            x = model.adapter.to_patch_tokens(x)
            _add(rows, "same_flow", "adapter.to_patch_tokens", x)
            time_emb = time_emb.repeat_interleave(model.adapter.num_patches, dim=0)
            emb_sigma_log = emb_sigma_log.repeat_interleave(model.adapter.num_patches, dim=0)
        else:
            x = model.adapter.to_tokens(x)
            _add(rows, "same_flow", "adapter.to_frame_tokens", x)

        scale_w_inp = model.scale_inp(emb_sigma_log).reshape(x.shape[0], -1, 1)
        scale_w_out = model.scale_out(emb_sigma_log).reshape(x.shape[0], -1, 1)
        x = model.input_proj(x)
        _add(rows, "same_flow", "denoiser.input_proj", x)
        if model.frequency_scaling:
            x = (1.0 + scale_w_inp) * x
            _add(rows, "same_flow", "denoiser.input_scaled", x)

        skip_list = []
        for idx, blocks in enumerate(model.down_blocks):
            for block_idx, (latent_block, denoise_block) in enumerate(blocks):
                d = latent_block(pyramid[idx])
                if model.use_patch_denoiser:
                    d = model._repeat_patch_condition(d, model.down_patch_embeddings[idx])
                _add(rows, "same_flow", f"down.{idx}.{block_idx}.latent_proj", d)
                x = (x + d) / (2.0**0.5)
                _add(rows, "same_flow", f"down.{idx}.{block_idx}.mixed", x)
                if model.use_latent_film:
                    x = model.down_latent_films[idx][block_idx](x, d)
                    _add(rows, "same_flow", f"down.{idx}.{block_idx}.film", x)
                x = denoise_block(x, time_emb)
                _add(rows, "same_flow", f"down.{idx}.{block_idx}.block", x)
                skip_list.append(x)
            if idx != len(model.down_blocks) - 1:
                x = model.down_transitions[idx](x)
                _add(rows, "same_flow", f"down.{idx}.transition", x)

        for idx, blocks in enumerate(model.up_blocks):
            pyramid_value = pyramid[-idx - 1]
            for block_idx, (latent_block, denoise_block) in enumerate(blocks):
                d = latent_block(pyramid_value)
                if model.use_patch_denoiser:
                    d = model._repeat_patch_condition(d, model.up_patch_embeddings[idx])
                _add(rows, "same_flow", f"up.{idx}.{block_idx}.latent_proj", d)
                skip = skip_list.pop()
                x = (x + skip + d) / (3.0**0.5)
                _add(rows, "same_flow", f"up.{idx}.{block_idx}.mixed", x)
                if model.use_latent_film:
                    x = model.up_latent_films[idx][block_idx](x, d)
                    _add(rows, "same_flow", f"up.{idx}.{block_idx}.film", x)
                x = denoise_block(x, time_emb)
                _add(rows, "same_flow", f"up.{idx}.{block_idx}.block", x)
            if idx != len(model.up_blocks) - 1:
                x = model.up_transitions[idx](x)
                _add(rows, "same_flow", f"up.{idx}.transition", x)

        d = model.decoded_proj(pyramid[0])
        if model.use_patch_denoiser:
            d = model._repeat_patch_condition(d, model.final_patch_embedding)
        _add(rows, "same_flow", "decoded_proj", d)
        x = (x + d) / (2.0**0.5)
        _add(rows, "same_flow", "final.mixed", x)
        x = model.norm_out(x)
        x = model.activation_out(x)
        _add(rows, "same_flow", "final.norm_act", x)
        x = (1.0 + scale_w_out) * x
        _add(rows, "same_flow", "final.scaled", x)
        residual_tokens = model.output_proj(x)
        _add(rows, "same_flow", "final.residual_head_tokens", residual_tokens)
        if model.direct_patch_output:
            residual = model.adapter.patch_values_to_representation(residual_tokens, data.shape[0])
        elif model.direct_frame_output:
            residual = model.adapter.frame_values_to_representation(residual_tokens)
        elif model.use_patch_denoiser:
            residual = model.adapter.patch_tokens_to_representation(residual_tokens, data.shape[0])
        else:
            residual = model.adapter.to_representation(residual_tokens)
        _add(rows, "same_flow", "final.residual_head_representation", residual)
        out = c_skip * inp + c_out * residual
        _add(rows, "same_flow", "edm.output", out, ref=data)
        loss = torch.nn.functional.mse_loss(out.float(), data.float())
    return rows, {
        "representation": _stats(data),
        "output": _stats(out),
        "direct_mse_to_clean": float(loss.cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-experiment", default="same-flow-debug-m2l-parity")
    parser.add_argument("--same-override", action="append", default=[])
    parser.add_argument("--same-checkpoint")
    parser.add_argument(
        "--official-config",
        default=str(ROOT / "tmp" / "original_music2latent_official_overfit10_fast_config.py"),
    )
    parser.add_argument("--official-checkpoint")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    same_cfg, same_model, same_dataset = _load_same_model(
        args.same_experiment,
        args.same_checkpoint,
        device,
        args.same_override,
    )
    audio, info = _same_audio_from_dataset(same_dataset, args.index, device)
    official_model = _load_official_model(args.official_config, args.official_checkpoint, device)

    official_rows, official_summary = _official_trace(
        official_model,
        audio,
        args.sigma,
        args.seed,
        device,
    )
    same_rows, same_summary = _same_trace(
        same_model,
        audio,
        args.sigma,
        args.seed,
        device,
    )
    output = {
        "input": {
            "index": args.index,
            "path": info.get("path"),
            "audio": _stats(audio),
            "sigma": args.sigma,
            "seed": args.seed,
        },
        "official": official_summary,
        "same": same_summary,
        "rows": official_rows + same_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output": str(out_path),
        "official": official_summary,
        "same": same_summary,
        "num_rows": len(output["rows"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
