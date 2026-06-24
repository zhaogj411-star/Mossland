from __future__ import annotations

import gc
import os
from typing import Sequence
from safetensors.torch import save_file, save_model

import lightning as pl
import torch
import torchaudio
from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import MosslandCodecSame as MosslandCodec
from .tasks import MosslandTaskBatch
from .training_base import (
    CodecTrainingBase,
    add_noise,
    get_sigma_continuous,
    pseudo_huber_loss,
)


def _label_tuple(task_id: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(task_id, str):
        return (task_id,)
    return tuple(str(item) for item in task_id)


def _label_at(task_id: str | Sequence[str], index: int) -> str:
    labels = _label_tuple(task_id)
    return labels[index] if index < len(labels) else labels[0]


class MosslandCodecTrainingWrapper(CodecTrainingBase):
    """Mossland RVQ trainer with src/target task batches."""

    def __init__(
        self,
        model: MosslandCodec,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "radam",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 100,
        lr_schedule_total_steps: int = 800_000,
        final_learning_rate: float = 1e-6,
        rvq_commitment_weight: float = 0.0,
        rvq_codebook_weight: float = 0.0,
        rvq_hidden_recon_weight: float = 0.0,
        rvq_latent_train_prob: float = 0.25,
        rvq_detach_encoder: bool = False,
        train_n_quantizers: int | None = None,
        train_n_quantizers_choices: list[int] | None = None,
        consistency_step: float = 0.1,
        consistency_step_schedule: str = "exponential",
        consistency_step_total_steps: int = 800_000,
        consistency_step_end_exp: float = 3.0,
        sigma_sampling: str = "lognormal",
        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
        consistency_loss_delta: float = 0.00054,
        consistency_min_sigma_delta: float = 0.001,
        index_data_every_step: int | None = None,
        fail_on_nonfinite: bool = True,
        fail_on_nonfinite_grad: bool | None = None,
    ):
        super().__init__(
            model=model,
            use_ema=False,
            learning_rate=learning_rate,
            fail_on_nonfinite=fail_on_nonfinite,
            fail_on_nonfinite_grad=fail_on_nonfinite_grad,
        )
        self.ema_beta = float(ema_beta)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.final_learning_rate = float(final_learning_rate)
        self.rvq_commitment_weight = float(rvq_commitment_weight)
        self.rvq_codebook_weight = float(rvq_codebook_weight)
        self.rvq_hidden_recon_weight = float(rvq_hidden_recon_weight)
        self.rvq_latent_train_prob = float(rvq_latent_train_prob)
        self.rvq_detach_encoder = bool(rvq_detach_encoder)
        self.train_n_quantizers = train_n_quantizers
        self.train_n_quantizers_choices = train_n_quantizers_choices
        self.consistency_step = float(consistency_step)
        self.consistency_step_schedule = consistency_step_schedule
        self.consistency_step_total_steps = int(consistency_step_total_steps)
        self.consistency_step_end_exp = float(consistency_step_end_exp)
        self.sigma_sampling = sigma_sampling
        self.lognormal_mean = float(lognormal_mean)
        self.lognormal_std = float(lognormal_std)
        self.consistency_loss_delta = float(consistency_loss_delta)
        self.consistency_min_sigma_delta = float(consistency_min_sigma_delta)
        self.index_data_every_step = index_data_every_step
        if use_ema:
            self.ema = EMA(
                self.model,
                beta=self.ema_beta,
                power=3 / 4,
                update_every=1,
                update_after_step=2000,
            )

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "radam":
            opt = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}")

        if self.lr_schedule == "constant":
            return opt
        if self.lr_schedule != "cosine_decay":
            raise ValueError(f"Unsupported lr_schedule={self.lr_schedule!r}")

        decay_steps = max(1, self.lr_schedule_total_steps - self.lr_warmup_steps)
        cosine = CosineAnnealingLR(
            opt,
            T_max=decay_steps,
            eta_min=self.final_learning_rate,
        )
        if self.lr_warmup_steps <= 0:
            scheduler = cosine
        else:
            warmup = LinearLR(
                opt,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.lr_warmup_steps,
            )
            scheduler = SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[self.lr_warmup_steps],
            )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _log_current_lr(self):
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None or not getattr(trainer, "optimizers", None):
            return
        self.log(
            "optim/lr",
            trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def _log_tensor_stats(self, prefix: str, tensor: torch.Tensor):
        stats = tensor.detach().float()
        self.log(f"{prefix}_mean", stats.mean(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{prefix}_std", stats.std(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

    def _sample_n_quantizers(self):
        if self.train_n_quantizers_choices:
            idx = torch.randint(len(self.train_n_quantizers_choices), (), device=self.device).item()
            return int(self.train_n_quantizers_choices[idx])
        if self.train_n_quantizers is not None:
            return int(self.train_n_quantizers)
        return None

    def _consistency_step_for_update(self) -> float:
        if self.consistency_step_schedule == "constant":
            return self.consistency_step
        if self.consistency_step_schedule != "exponential":
            raise ValueError(f"Unsupported consistency_step_schedule={self.consistency_step_schedule!r}")
        progress = min(max(float(self.global_step) / max(1, self.consistency_step_total_steps), 0.0), 1.0)
        return self.consistency_step * (10.0 ** (-(self.consistency_step_end_exp - 1.0) * progress))

    def _official_lognormal_positions(self, batch_size: int, device: torch.device, num_bins: int = 10_000):
        bins = torch.linspace(0.0, 1.0, num_bins - 1, device=device)
        sigmas = get_sigma_continuous(
            bins,
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
        )
        weights = torch.exp(
            -0.5 * ((torch.log(sigmas) - self.lognormal_mean) / self.lognormal_std) ** 2
        ) / (self.lognormal_std * (2.0 * torch.pi) ** 0.5)
        inds = torch.multinomial(weights, batch_size, replacement=True).float()
        return (inds + torch.rand_like(inds)) / float(num_bins - 1)

    def _sample_sigma_pair(self, batch_size: int, device: torch.device):
        step_size = min(max(self._consistency_step_for_update(), 0.0), 1.0)
        if self.sigma_sampling == "lognormal":
            high_pos = self._official_lognormal_positions(batch_size, device)
        elif self.sigma_sampling == "uniform":
            high_pos = torch.rand(batch_size, device=device)
        else:
            raise ValueError(f"Unsupported sigma_sampling={self.sigma_sampling!r}")

        high_pos = high_pos.clamp(min=step_size)
        low_pos = (high_pos - step_size).clamp(min=0.0)
        sigma_low = get_sigma_continuous(
            low_pos,
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
        ).clamp(self.model.sigma_min, self.model.sigma_max)
        sigma_high = get_sigma_continuous(
            high_pos,
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
        ).clamp(self.model.sigma_min, self.model.sigma_max)
        return sigma_low, sigma_high, step_size

    def _align_audio_for_model(self, audio: torch.Tensor) -> torch.Tensor:
        hop = self.model.hop
        fac = getattr(self.model.audio_processor, "fac", 4)
        downscaling_factor = 2 ** sum(1 for x in self.model.freq_downsample_list if x == 0)
        if getattr(self.model.audio_processor, "center_pad", False):
            stft_frames = audio.shape[-1] // hop
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = cropped_frames * hop * downscaling_factor
        else:
            frame_length = fac * hop
            stft_frames = max(0, (audio.shape[-1] - frame_length) // hop + 1)
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = cropped_frames * hop * downscaling_factor + (fac - 1) * hop
        return audio[..., :cropped_length]

    def _consistency_loss(self, target_representation, latent_override, task_id):
        batch_size = target_representation.shape[0]
        sigma_low, sigma_high, step_size = self._sample_sigma_pair(batch_size, target_representation.device)
        noise = torch.randn_like(target_representation)
        noisy_high = add_noise(target_representation, noise, sigma_high)
        noisy_low = add_noise(target_representation, noise, sigma_low)

        predicted = self.model(
            target_representation,
            noisy_high,
            sigma=sigma_high,
            latent_override=latent_override,
            task_id=task_id,
        )
        with torch.no_grad():
            target = self.model(
                target_representation,
                noisy_low,
                sigma=sigma_low,
                latent_override=latent_override,
                task_id=task_id,
            )

        sigma_delta = (sigma_high - sigma_low).clamp(min=self.consistency_min_sigma_delta)
        weights = (1.0 / sigma_delta).reshape(batch_size, 1, 1, 1)
        loss_values = (
            pseudo_huber_loss(
                predicted.float(),
                target.float(),
                delta=self.consistency_loss_delta,
            )
            * weights.float()
        )
        loss = loss_values.mean()
        metrics = {
            "loss/consistency": loss.detach(),
            "sigma/weight_mean": weights.mean().detach(),
            "sigma/step": torch.tensor(step_size, device=target_representation.device),
            "sigma/low_mean": sigma_low.mean().detach(),
            "sigma/high_mean": sigma_high.mean().detach(),
        }
        return loss, metrics, predicted, target

    def training_step(self, batch, batch_idx):
        self._log_current_lr()
        payload, info = batch
        task = MosslandTaskBatch.from_payload(payload)
        dtype = next(self.model.parameters()).dtype
        src = self._align_audio_for_model(task.src.to(device=self.device, dtype=dtype))
        target_audio = self._align_audio_for_model(task.target.to(device=self.device, dtype=dtype))
        min_len = min(src.shape[-1], target_audio.shape[-1])
        src = src[..., :min_len]
        target_audio = target_audio[..., :min_len]
        self._assert_finite("src", src, info)
        self._assert_finite("target_audio", target_audio, info)

        src_representation = self.model.audio_processor.to_representation_encoder(src)
        target_representation = self.model.audio_processor.to_representation_encoder(target_audio)
        self._assert_finite("src_representation", src_representation, info)
        self._assert_finite("target_representation", target_representation, info)

        quantized = None
        latent_override = None
        source_discrete = src_representation.new_zeros(())
        if self.model.has_quantizer:
            n_quantizers = self._sample_n_quantizers()
            quantized = self.model.quantize_representation(
                src_representation,
                detach_encoder=self.rvq_detach_encoder,
                n_quantizers=n_quantizers,
            )
            latent_override = quantized.continuous
            rvq_latent_train_prob = max(0.0, min(1.0, self.rvq_latent_train_prob))
            if rvq_latent_train_prob > 0.0:
                batch_size = quantized.continuous.shape[0]
                discrete_mask = (
                    torch.ones(batch_size, dtype=torch.bool, device=quantized.continuous.device)
                    if rvq_latent_train_prob >= 1.0
                    else torch.rand(batch_size, device=quantized.continuous.device) < rvq_latent_train_prob
                )
                source_discrete = discrete_mask.float().mean()
                discrete_mask = discrete_mask.view(batch_size, *([1] * (quantized.continuous.ndim - 1)))
                latent_override = torch.where(discrete_mask, quantized.discrete, quantized.continuous)
            if n_quantizers is not None:
                self.log("rvq/n_quantizers", float(n_quantizers), prog_bar=False)
            self._log_tensor_stats("latent/continuous", quantized.continuous)
            self._log_tensor_stats("latent/discrete", quantized.discrete)
        else:
            latent_override = self.model.encoder(src_representation)
            self._log_tensor_stats("latent/continuous", latent_override)

        consistency_loss, metrics, predicted, target = self._consistency_loss(
            target_representation,
            latent_override=latent_override,
            task_id=task.task_id,
        )
        self._assert_finite("predicted", predicted, info)
        self._assert_finite("target", target, info)
        loss = consistency_loss
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False)
        self.log("latent/source_discrete", source_discrete, prog_bar=False)

        if quantized is not None:
            rvq_loss = (
                self.rvq_commitment_weight * quantized.commitment_loss
                + self.rvq_codebook_weight * quantized.codebook_loss
                + self.rvq_hidden_recon_weight * quantized.distill_loss
            )
            loss = loss + rvq_loss
            self.log("rvq/commitment_loss", quantized.commitment_loss, prog_bar=True)
            self.log("rvq/codebook_loss", quantized.codebook_loss, prog_bar=False)
            self.log("rvq/hidden_recon_loss", quantized.distill_loss, prog_bar=False)
            self.log("loss/rvq", rvq_loss, prog_bar=False)

        task_ids = _label_tuple(task.task_id)
        task_names = tuple(dict.fromkeys((*getattr(self.model, "task_names", ()), *task_ids)))
        for task_id in task_names:
            self.log(
                f"task/{task_id}",
                task_ids.count(task_id) / len(task_ids),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self._assert_finite("loss", loss, info)
        self.log("loss/total", loss, prog_bar=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if hasattr(self, "ema"):
            self.ema.update()
    def export_model(self, path, use_safetensors=False, export_ema=False):
        model = self.model
        if export_ema:
            model = self.ema.ema_model

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)



class MosslandCodecTrainingCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num=2,
        demo_every=2000,
        demo_start_step: int = 1,
        sample_rate=44100,
        demo_n_quantizers_choices: list[int] | None = None,
        denoising_steps: int = 1,
        demo_seed: int = 1234,
        silence_seconds: float = 0.25,
        use_ema: bool = True,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = int(demo_num)
        self.demo_every = int(demo_every)
        self.demo_start_step = int(demo_start_step)
        self.sample_rate = int(sample_rate)
        self.demo_n_quantizers_choices = demo_n_quantizers_choices
        self.denoising_steps = int(denoising_steps)
        self.demo_seed = int(demo_seed)
        self.silence_seconds = float(silence_seconds)
        self.use_ema = bool(use_ema)
        self.last_demo_step = -1

    def _clear_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _set_demo_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _capture_rng_state(self):
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
        return torch.random.get_rng_state(), cuda_state

    def _restore_rng_state(self, rng_state):
        cpu_state, cuda_state = rng_state
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _set_eval(self, model):
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        return was_training

    def _restore_training(self, model, was_training: bool):
        if hasattr(model, "train"):
            model.train(was_training)

    def _demo_rates_for_model(self, model):
        if not model.has_quantizer:
            return []
        rates = self.demo_n_quantizers_choices or [8, 16, 32]
        return [int(n) for n in rates if 1 <= int(n) <= model.quantizer.n_codebooks]

    def _should_run_demo(self, global_step: int) -> bool:
        if self.demo_every <= 0:
            return False
        if global_step < self.demo_start_step:
            return False
        if self.last_demo_step == global_step:
            return False
        return (global_step - self.demo_start_step) % self.demo_every == 0

    def _save_demo_for_model(self, model, model_label, src, target, task_id, trainer):
        os.makedirs(self.demo_dir, exist_ok=True)
        rates = self._demo_rates_for_model(model)
        src = model.prepare_audio_for_encode(src) if hasattr(model, "prepare_audio_for_encode") else src
        for i in range(src.shape[0]):
            task_label = _label_at(task_id, i)
            src_item = src[i].detach().cpu()
            target_item = target[i].detach().cpu()
            continuous_latent = model.encode(src[i])
            self._set_demo_seed(self.demo_seed + trainer.global_step * 100_000 + i * 100)
            continuous = model.decode(
                continuous_latent,
                denoising_steps=self.denoising_steps,
                target_length=target_item.shape[-1],
                task_id=task_label,
            )[0].detach().cpu()
            segments = [src_item, target_item]
            labels = []
            for rate_idx, n_quantizers in enumerate(rates):
                discrete_latent = model.encode(src[i], quantize=True, n_quantizers=n_quantizers)
                self._set_demo_seed(self.demo_seed + trainer.global_step * 100_000 + i * 100 + rate_idx + 1)
                discrete = model.decode(
                    discrete_latent,
                    denoising_steps=self.denoising_steps,
                    target_length=target_item.shape[-1],
                    task_id=task_label,
                )[0].detach().cpu()
                segments.append(discrete)
                labels.append(f"discrete{n_quantizers}")
            segments.append(continuous)
            labels.append("continuous")
            silence = torch.zeros(
                src_item.shape[0],
                int(self.sample_rate * self.silence_seconds),
                dtype=src_item.dtype,
            )
            comparison = torch.cat(
                [part for segment in segments for part in (segment, silence)][:-1],
                dim=-1,
            )
            torchaudio.save(
                os.path.join(
                    self.demo_dir,
                    f"{trainer.global_step}_{i}_{task_label}_{model_label}_{'_'.join(labels)}_src_target_generated.mp3",
                ),
                comparison.clamp(-1.0, 1.0),
                self.sample_rate,
                format="mp3",
            )

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None or trainer.global_rank != 0:
            return
        if not self._should_run_demo(trainer.global_step):
            return
        self.last_demo_step = trainer.global_step
        payload, info = batch
        task = MosslandTaskBatch.from_payload(payload)
        src = task.src[: self.demo_num].to(module.device)
        target = task.target[: self.demo_num].to(module.device)
        rng_state = self._capture_rng_state()
        model_states = []
        self._clear_cuda_cache()
        try:
            model_states.append((module.model, self._set_eval(module.model)))
            self._save_demo_for_model(module.model, "raw", src, target, task.task_id, trainer)
            if self.use_ema and hasattr(module, "ema"):
                model_states.append((module.ema.ema_model, self._set_eval(module.ema.ema_model)))
                self._save_demo_for_model(module.ema.ema_model, "ema", src, target, task.task_id, trainer)
        finally:
            for demo_model, was_training in model_states:
                self._restore_training(demo_model, was_training)
            self._restore_rng_state(rng_state)
            self._clear_cuda_cache()
