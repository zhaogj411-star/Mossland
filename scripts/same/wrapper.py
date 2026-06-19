import os
from contextlib import nullcontext
from typing import Any

import lightning as pl
import torch
import torchaudio
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .discriminators import EncodecDiscriminator
from .losses import MultiResolutionSTFTLoss, SumAndDifferenceSTFTLoss
from .models import SAMEAutoencoder


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def trim_to_shortest(left: torch.Tensor, right: torch.Tensor):
    length = min(left.shape[-1], right.shape[-1])
    return left[..., :length], right[..., :length]


class StateDictEMA:
    def __init__(self, model: nn.Module, beta: float = 0.9999, update_after_step: int = 2000):
        self.beta = float(beta)
        self.update_after_step = int(update_after_step)
        self.step = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self._copy_from_model(model)

    @torch.no_grad()
    def _copy_from_model(self, model: nn.Module):
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        state = model.state_dict()
        if not self.shadow or self.step <= self.update_after_step:
            self.shadow = {name: value.detach().clone() for name, value in state.items()}
            return
        for name, value in state.items():
            detached = value.detach()
            if name not in self.shadow or self.shadow[name].shape != detached.shape:
                self.shadow[name] = detached.clone()
                continue
            shadow = self.shadow[name]
            if shadow.device != detached.device:
                shadow = shadow.to(detached.device)
                self.shadow[name] = shadow
            if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                shadow.lerp_(detached, 1.0 - self.beta)
            else:
                shadow.copy_(detached)

    @torch.no_grad()
    def state_dict(self):
        return {
            "step": torch.tensor(self.step, dtype=torch.long),
            "shadow": self.shadow,
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        self.step = int(state.get("step", 0))
        shadow = state.get("shadow", {})
        self.shadow = {name: value.detach().clone() for name, value in shadow.items()}


class SAMETrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model: SAMEAutoencoder,
        loss_config: dict[str, Any],
        learning_rate: float = 1e-4,
        discriminator_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 1000,
        lr_schedule_total_steps: int = 500000,
        warmup_steps: int = 0,
        warmup_mode: str = "full",
        decoder_finetune: bool = False,
        clip_grad_norm: float = 0.0,
        use_ema: bool = False,
        ema_beta: float = 0.9999,
        ema_update_after_step: int = 2000,
        fail_on_nonfinite: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.learning_rate = float(learning_rate)
        self.discriminator_learning_rate = float(discriminator_learning_rate or learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.warmup_steps = int(warmup_steps)
        self.warmup_mode = warmup_mode
        self.decoder_finetune = bool(decoder_finetune)
        self.clip_grad_norm = float(clip_grad_norm)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        self.use_ema = bool(use_ema)
        if self.use_ema:
            self.ema = StateDictEMA(
                self.model,
                beta=float(ema_beta),
                update_after_step=int(ema_update_after_step),
            )

        if loss_config is None:
            raise ValueError("SAMETrainingWrapper requires wrapper.loss_config in YAML")
        self.loss_config = loss_config

        spectral_cfg = self.loss_config.get("spectral", {}).get("config", {})
        spectral_cfg.setdefault("sample_rate", self.model.sample_rate)
        if self.model.audio_channels == 2:
            self.sdstft = SumAndDifferenceSTFTLoss(**spectral_cfg)
            self.lrstft = MultiResolutionSTFTLoss(**spectral_cfg)
        else:
            self.sdstft = MultiResolutionSTFTLoss(**spectral_cfg)
            self.lrstft = None

        self.l1 = nn.L1Loss()
        disc_cfg = self.loss_config.get("discriminator")
        self.use_disc = disc_cfg is not None and disc_cfg.get("type", "encodec") != "none"
        if self.use_disc:
            if disc_cfg.get("type", "encodec") != "encodec":
                raise ValueError("scripts.same currently implements the EnCodec MS-STFT discriminator")
            disc_kwargs = dict(disc_cfg.get("config", {}))
            disc_kwargs.setdefault("in_channels", self.model.audio_channels)
            self.discriminator = EncodecDiscriminator(**disc_kwargs)
        else:
            self.discriminator = None

    def _make_optimizer(self, params, lr: float):
        name = self.optimizer_name.lower()
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=(0.8, 0.99), weight_decay=self.weight_decay)
        if name == "radam":
            return torch.optim.RAdam(params, lr=lr, betas=(0.8, 0.99), weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}")

    def _make_scheduler(self, optimizer):
        if self.lr_schedule == "constant":
            return None
        if self.lr_schedule != "cosine_decay":
            raise ValueError(f"Unsupported lr_schedule={self.lr_schedule!r}")
        decay_steps = max(1, self.lr_schedule_total_steps - self.lr_warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0.0)
        if self.lr_warmup_steps <= 0:
            return cosine
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.lr_warmup_steps)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[self.lr_warmup_steps])

    def configure_optimizers(self):
        if self.decoder_finetune:
            gen_params = list(self.model.autoencoder.decoder.parameters())
            if self.model.autoencoder.pretransform is not None:
                gen_params += list(self.model.autoencoder.pretransform.parameters())
        else:
            gen_params = list(self.model.parameters())
        opt_gen = self._make_optimizer(gen_params, self.learning_rate)
        sched_gen = self._make_scheduler(opt_gen)
        if not self.use_disc:
            if sched_gen is None:
                return opt_gen
            return {"optimizer": opt_gen, "lr_scheduler": {"scheduler": sched_gen, "interval": "step"}}

        opt_disc = self._make_optimizer(self.discriminator.parameters(), self.discriminator_learning_rate)
        sched_disc = self._make_scheduler(opt_disc)
        optimizers = [opt_gen, opt_disc]
        schedulers = []
        if sched_gen is not None:
            schedulers.append({"scheduler": sched_gen, "interval": "step"})
        if sched_disc is not None:
            schedulers.append({"scheduler": sched_disc, "interval": "step"})
        if schedulers:
            return optimizers, schedulers
        return optimizers

    def _prepare_audio(self, batch: torch.Tensor):
        if batch.ndim == 4:
            batch = batch.flatten(0, 1)
        if batch.ndim == 2:
            batch = batch[:, None, :]
        if batch.ndim != 3:
            raise ValueError(f"expected [B,C,T] audio, got {tuple(batch.shape)}")
        if batch.shape[1] != self.model.audio_channels:
            raise ValueError(f"expected {self.model.audio_channels} audio channels, got {batch.shape[1]}")
        return batch.clamp(-1.0, 1.0)

    def _assert_finite(self, name: str, tensor: torch.Tensor):
        if torch.isfinite(tensor).all():
            return
        if self.fail_on_nonfinite:
            raise FloatingPointError(f"{name} contains non-finite values")
        print(f"{name} contains non-finite values")

    def _format_tensor_stats(self, name: str, tensor: torch.Tensor | None):
        if tensor is None:
            return f"{name}: None"
        with torch.no_grad():
            detached = tensor.detach()
            finite = torch.isfinite(detached)
            finite_count = int(finite.sum().item())
            total = detached.numel()
            if finite_count == 0:
                return f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} finite=0/{total}"
            values = detached[finite].float()
            return (
                f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} finite={finite_count}/{total} "
                f"min={values.min().item():.6g} max={values.max().item():.6g} "
                f"mean={values.mean().item():.6g} std={values.std(unbiased=False).item():.6g}"
            )

    def _raise_nonfinite_loss(
        self,
        loss: torch.Tensor,
        decoded: torch.Tensor,
        target: torch.Tensor,
        latents: torch.Tensor,
        softnorm_loss: torch.Tensor,
        losses: dict[str, torch.Tensor],
        recon_loss: torch.Tensor,
        adv_loss: torch.Tensor,
        feature_matching: torch.Tensor,
    ):
        parts = [
            self._format_tensor_stats("loss", loss),
            self._format_tensor_stats("recon_loss", recon_loss),
            self._format_tensor_stats("adv_loss", adv_loss),
            self._format_tensor_stats("feature_matching", feature_matching),
            self._format_tensor_stats("decoded", decoded),
            self._format_tensor_stats("target", target),
            self._format_tensor_stats("latents", latents),
            self._format_tensor_stats("softnorm_loss", softnorm_loss),
        ]
        for name, value in losses.items():
            parts.append(self._format_tensor_stats(name, value))
        raise FloatingPointError("loss contains non-finite values\n" + "\n".join(parts))

    def _forward_autoencoder(self, audio: torch.Tensor, encoder_grad: bool, decoder_grad: bool):
        audio, length = self.model.preprocess(audio)
        encode_context = nullcontext() if encoder_grad else torch.no_grad()
        autocast_device = "cuda" if audio.is_cuda else "cpu"
        with encode_context, torch.autocast(device_type=autocast_device, enabled=False):
            audio = audio.float()
            latents, info = self.model.autoencoder.encode(audio, return_info=True)
        decode_context = nullcontext() if decoder_grad else torch.no_grad()
        with decode_context, torch.autocast(device_type=autocast_device, enabled=False):
            latents = latents.float()
            decoded = self.model.autoencoder.decode(latents)
        decoded = decoded[..., :length]
        softnorm_loss = info.get("softnorm_loss")
        if softnorm_loss is None:
            softnorm_loss = decoded.new_zeros(())
        return decoded, latents, softnorm_loss

    def _reconstruction_losses(self, decoded: torch.Tensor, target: torch.Tensor, softnorm_loss: torch.Tensor):
        losses = {}
        weights = self.loss_config
        time_weight = float(weights.get("time", {}).get("weights", {}).get("l1", 0.0))
        if time_weight:
            losses["l1_time_loss"] = self.l1(decoded, target)
        else:
            losses["l1_time_loss"] = decoded.new_zeros(())
        spectral_weights = weights.get("spectral", {}).get("weights", {})
        mrstft_weight = float(spectral_weights.get("mrstft", 1.0))
        lr_mrstft_weight = mrstft_weight if self.lrstft is not None else 0.0
        losses["mrstft_loss"] = self.sdstft(decoded, target) if mrstft_weight else decoded.new_zeros(())
        if self.lrstft is not None and lr_mrstft_weight:
            losses["lr_stft_loss"] = self.lrstft(decoded, target)
        else:
            losses["lr_stft_loss"] = decoded.new_zeros(())
        losses["softnorm_loss"] = softnorm_loss
        total = (
            time_weight * losses["l1_time_loss"]
            + mrstft_weight * losses["mrstft_loss"]
            + lr_mrstft_weight * losses["lr_stft_loss"]
            + float(weights.get("bottleneck", {}).get("weights", {}).get("softnorm", 0.0)) * softnorm_loss
        )
        return total, losses

    def _step_schedulers(self, schedulers, use_disc_step: bool):
        if schedulers is None:
            return
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        if self.use_disc and len(schedulers) >= 2:
            schedulers[1 if use_disc_step else 0].step()
        elif schedulers:
            schedulers[0].step()

    def training_step(self, batch, batch_idx):
        audio, _ = batch
        audio = self._prepare_audio(audio)
        self._assert_finite("audio", audio)
        warmed_up = self.global_step >= self.warmup_steps
        use_disc_step = bool(
            self.use_disc
            and self.global_step % 2 == 1
            and ((self.warmup_mode == "full" and warmed_up) or self.warmup_mode == "adv")
        )
        opts = self.optimizers()
        opt_gen, opt_disc = (opts if self.use_disc else (opts, None))
        schedulers = self.lr_schedulers()
        if self.use_disc:
            set_requires_grad(self.discriminator, use_disc_step)

        if use_disc_step:
            with torch.no_grad():
                decoded, latents, softnorm_loss = self._forward_autoencoder(audio, encoder_grad=False, decoder_grad=False)
                decoded, target = trim_to_shortest(decoded, audio)
            loss_dis, _, _ = self.discriminator.loss(reals=target, fakes=decoded.detach())
            self._assert_finite("loss_dis", loss_dis)
            opt_disc.zero_grad()
            self.manual_backward(loss_dis)
            if self.clip_grad_norm > 0:
                self.clip_gradients(opt_disc, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
            opt_disc.step()
            self._step_schedulers(schedulers, use_disc_step=True)
            self.log("train/discriminator_loss", loss_dis.detach(), prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/disc_lr", opt_disc.param_groups[0]["lr"], on_step=True, on_epoch=False)
            self.log("latent/std", latents.detach().float().std(), on_step=True, on_epoch=False)
            return loss_dis

        encoder_grad = not self.decoder_finetune
        decoded, latents, softnorm_loss = self._forward_autoencoder(audio, encoder_grad=encoder_grad, decoder_grad=True)
        decoded, target = trim_to_shortest(decoded, audio)
        recon_loss, losses = self._reconstruction_losses(decoded, target, softnorm_loss)

        adv_loss = decoded.new_zeros(())
        feature_matching = decoded.new_zeros(())
        if self.use_disc and warmed_up:
            _, adv_loss, feature_matching = self.discriminator.loss(reals=target, fakes=decoded)

        disc_weights = self.loss_config.get("discriminator", {}).get("weights", {})
        loss = (
            recon_loss
            + float(disc_weights.get("adversarial", 0.0)) * adv_loss
            + float(disc_weights.get("feature_matching", 0.0)) * feature_matching
        )
        if not torch.isfinite(loss).all():
            self._raise_nonfinite_loss(
                loss=loss,
                decoded=decoded,
                target=target,
                latents=latents,
                softnorm_loss=softnorm_loss,
                losses=losses,
                recon_loss=recon_loss,
                adv_loss=adv_loss,
                feature_matching=feature_matching,
            )
        opt_gen.zero_grad()
        self.manual_backward(loss)
        if self.clip_grad_norm > 0:
            self.clip_gradients(opt_gen, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_gen.step()
        if hasattr(self, "ema"):
            self.ema.update(self.model)
        self._step_schedulers(schedulers, use_disc_step=False)

        self.log("train/loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=False)
        for name, value in losses.items():
            self.log(f"train/{name}", value.detach(), on_step=True, on_epoch=False)
        self.log("train/loss_adv", adv_loss.detach(), on_step=True, on_epoch=False)
        self.log("train/feature_matching_loss", feature_matching.detach(), on_step=True, on_epoch=False)
        self.log("train/gen_lr", opt_gen.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("latent/token_rate", torch.tensor(self.model.token_rate, device=loss.device), on_step=True, on_epoch=False)
        self.log("latent/std", latents.detach().float().std(), on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def reconstruct_waveform(self, audio: torch.Tensor, use_ema: bool = False):
        audio = self._prepare_audio(audio)
        was_training = self.model.training
        raw_state = None
        if use_ema:
            if not hasattr(self, "ema"):
                raise RuntimeError("EMA reconstruction requested but wrapper has no EMA state")
            raw_state = {
                name: value.detach().clone()
                for name, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.ema.shadow, strict=True)
        self.model.eval()
        output = self.model(audio)
        if raw_state is not None:
            self.model.load_state_dict(raw_state, strict=True)
        if was_training:
            self.model.train()
        return audio.detach().cpu(), output.audio.detach().cpu(), output.latents.detach().cpu()

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["same_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self, "ema") and "same_ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["same_ema"])


class SAMECallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 2,
        demo_every: int = 1000,
        sample_rate: int = 44100,
        silence_seconds: float = 0.25,
        save_separate_files: bool = False,
        save_latents: bool = False,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = int(demo_num)
        self.demo_every = int(demo_every)
        self.sample_rate = int(sample_rate)
        self.silence_seconds = float(silence_seconds)
        self.save_separate_files = bool(save_separate_files)
        self.save_latents = bool(save_latents)
        self.last_demo_step = -1

    def _concat_audio(self, original: torch.Tensor, reconstructed: torch.Tensor):
        silence_samples = int(self.sample_rate * self.silence_seconds)
        if silence_samples <= 0:
            return torch.cat([original, reconstructed], dim=-1)
        silence = original.new_zeros(original.shape[:-1] + (silence_samples,))
        return torch.cat([original, silence, reconstructed], dim=-1)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None or self.demo_every <= 0:
            return
        if trainer.global_step % self.demo_every != 1:
            return
        if self.last_demo_step == trainer.global_step:
            return
        self.last_demo_step = trainer.global_step
        os.makedirs(self.demo_dir, exist_ok=True)
        audio, _ = batch
        demo_audio = audio[: self.demo_num]
        demo_models = [("raw", False)]
        if hasattr(module, "ema"):
            demo_models.append(("ema", True))
        latents_by_model = {}
        for model_label, use_ema in demo_models:
            originals, reconstructed, latents = module.reconstruct_waveform(demo_audio, use_ema=use_ema)
            latents_by_model[model_label] = latents
            for idx, (original, recon) in enumerate(zip(originals, reconstructed)):
                base = f"{trainer.global_step}_{idx}_rank{trainer.global_rank}_{model_label}"
                if self.save_separate_files:
                    torchaudio.save(os.path.join(self.demo_dir, f"{base}_input.wav"), original.float(), self.sample_rate)
                    torchaudio.save(os.path.join(self.demo_dir, f"{base}_reconstruct.wav"), recon.float(), self.sample_rate)
                torchaudio.save(
                    os.path.join(self.demo_dir, f"{base}_input_reconstruct.wav"),
                    self._concat_audio(original, recon).float(),
                    self.sample_rate,
                )
        if self.save_latents:
            torch.save(
                {"latents": latents_by_model},
                os.path.join(self.demo_dir, f"{trainer.global_step}_rank{trainer.global_rank}_latents.pt"),
            )
