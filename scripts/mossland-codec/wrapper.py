from __future__ import annotations

import gc
import os

import lightning as pl
import torch
import torchaudio
from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import MosslandCodecTransformer
from .tasks import MosslandTaskBatch
from .training_base import CodecTrainingBase, pseudo_huber_loss
from .utils import add_noise, get_sigma_continuous


def _label_tuple(label: str | tuple[str, ...]) -> tuple[str, ...]:
    return label if isinstance(label, tuple) else (label,)


def _slice_label(label: str | tuple[str, ...], count: int) -> str | tuple[str, ...]:
    if isinstance(label, tuple):
        return label[:count]
    return label


def _label_at(label: str | tuple[str, ...], index: int) -> str:
    if isinstance(label, tuple):
        if not label:
            return "unknown"
        return label[min(index, len(label) - 1)]
    return label


class MosslandCodecTrainingWrapper(CodecTrainingBase):
    """Mossland 多任务 codec 训练 wrapper。"""

    def __init__(
        self,
        model: MosslandCodecTransformer,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "radam",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 10_000,
        lr_schedule_total_steps: int = 2_000_000,
        fsq_dropout_prob: float = 0.75,
        consistency_step: float = 0.1,
        consistency_step_schedule: str = "exponential",
        consistency_step_total_steps: int = 2_000_000,
        consistency_step_end_exp: float = 2.0,
        sigma_sampling: str = "lognormal",
        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
        consistency_loss_delta: float = 0.00054,
        consistency_min_sigma_delta: float = 0.001,
        fail_on_nonfinite: bool = True,
    ):
        super().__init__(
            model=model,
            use_ema=False,
            learning_rate=learning_rate,
            lr_warmup_steps=lr_warmup_steps,
            consistency_total_step=consistency_step_total_steps,
            fail_on_nonfinite=fail_on_nonfinite,
        )
        if use_ema:
            self.ema = EMA(
                self.model,
                beta=ema_beta,
                power=3 / 4,
                update_every=1,
                update_after_step=2000,
            )
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.fsq_dropout_prob = fsq_dropout_prob
        self.consistency_step = consistency_step
        self.consistency_step_schedule = consistency_step_schedule
        self.consistency_step_total_steps = int(consistency_step_total_steps)
        self.consistency_step_end_exp = consistency_step_end_exp
        self.sigma_sampling = sigma_sampling
        self.lognormal_mean = lognormal_mean
        self.lognormal_std = lognormal_std
        self.consistency_loss_delta = consistency_loss_delta
        self.consistency_min_sigma_delta = consistency_min_sigma_delta

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
        cosine = CosineAnnealingLR(opt, T_max=decay_steps, eta_min=0.0)
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
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _consistency_step_for_update(self) -> float:
        if self.consistency_step_schedule == "constant":
            return self.consistency_step
        if self.consistency_step_schedule != "exponential":
            raise ValueError(
                f"Unsupported consistency_step_schedule={self.consistency_step_schedule!r}"
            )

        progress = min(
            max(float(self.global_step) / max(1, self.consistency_step_total_steps), 0.0),
            1.0,
        )
        return self.consistency_step * (
            10.0 ** (-(self.consistency_step_end_exp - 1.0) * progress)
        )

    def _schedule_position_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        rho_inv = 1.0 / self.model.rho
        return (
            (sigma**rho_inv - self.model.sigma_min**rho_inv)
            / (self.model.sigma_max**rho_inv - self.model.sigma_min**rho_inv)
        ).clamp(0.0, 1.0)

    def _sample_sigma_pair(self, batch_size: int, device: torch.device):
        step_size = min(max(self._consistency_step_for_update(), 0.0), 1.0)
        if self.sigma_sampling == "lognormal":
            sigma_high = torch.exp(
                torch.randn(batch_size, device=device) * self.lognormal_std
                + self.lognormal_mean
            ).clamp(self.model.sigma_min, self.model.sigma_max)
            high_pos = self._schedule_position_from_sigma(sigma_high)
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

    def _expand_half_sigmas(self, sigma_left: torch.Tensor, sigma_right: torch.Tensor):
        left = sigma_left[:, None].expand(-1, self.model.spec_length)
        right = sigma_right[:, None].expand(-1, self.model.spec_length)
        return torch.cat([left, right], dim=1)

    def _fsq_dropout_active(self, device: torch.device) -> bool:
        if self.fsq_dropout_prob <= 0.0:
            return False
        if self.fsq_dropout_prob >= 1.0:
            return True
        return bool(torch.rand((), device=device) < self.fsq_dropout_prob)

    def _pseudo_huber_loss(self, predicted: torch.Tensor, target: torch.Tensor):
        if self.consistency_loss_delta == 0.00054:
            return pseudo_huber_loss(predicted, target)
        c = self.consistency_loss_delta * (predicted[0].numel() ** 0.5)
        return torch.sqrt((predicted - target) ** 2 + c**2) - c

    def _consistency_loss(
        self,
        representation: torch.Tensor,
        latents: torch.Tensor,
        features: list[torch.Tensor],
        task_id: str,
    ):
        batch_size = representation.shape[0]
        sigma_low_left, sigma_high_left, step_size = self._sample_sigma_pair(
            batch_size,
            representation.device,
        )
        sigma_low_right, sigma_high_right, _ = self._sample_sigma_pair(
            batch_size,
            representation.device,
        )
        noise = torch.randn_like(representation)
        sigma_high = self._expand_half_sigmas(sigma_high_left, sigma_high_right)
        sigma_low = self._expand_half_sigmas(sigma_low_left, sigma_low_right)
        noisy_high = add_noise(representation, noise, sigma_high)
        noisy_low = add_noise(representation, noise, sigma_low)

        predicted = self.model.decoder_forward(
            noisy_high,
            latents,
            features=features,
            sigma_left=sigma_high_left,
            sigma_right=sigma_high_right,
            output="both",
            task_id=task_id,
        )
        with torch.no_grad():
            target = self.model.decoder_forward(
                noisy_low,
                latents,
                features=features,
                sigma_left=sigma_low_left,
                sigma_right=sigma_low_right,
                output="both",
                task_id=task_id,
            )

        sigma_delta = (sigma_high - sigma_low).clamp(
            min=self.consistency_min_sigma_delta
        )
        weights = (1.0 / sigma_delta).reshape(batch_size, 1, 1, -1)
        loss_values = (
            self._pseudo_huber_loss(predicted.float(), target.float()) * weights.float()
        )
        loss = loss_values.mean()
        metrics = {
            "loss/consistency": loss.detach(),
            "loss/consistency_weight_mean": weights.mean().detach(),
            "sigma/step": torch.tensor(step_size, device=representation.device),
            "sigma/low_mean": sigma_low.mean().detach(),
            "sigma/high_mean": sigma_high.mean().detach(),
        }
        return loss, metrics, predicted, target

    def training_step(self, batch, batch_idx):
        payload, info = batch
        task = MosslandTaskBatch.from_payload(payload)

        src = self.model.prepare_audio_batch(task.src)
        target_audio = self.model.prepare_audio_batch(task.target)
        self._assert_finite("src", src, info)
        self._assert_finite("target_audio", target_audio, info)

        src_representation = self.model.to_representation_encoder(src)
        target_representation = self.model.to_representation_encoder(target_audio)
        self._assert_finite("src_representation", src_representation, info)
        self._assert_finite("target_representation", target_representation, info)

        dont_quantize = self._fsq_dropout_active(src_representation.device)
        latents = self.model.encoder_forward(src_representation, dont_quantize=dont_quantize)
        self._assert_finite("latents", latents, info)
        features = self.model.pre_decoder_forward(latents)
        for idx, feature in enumerate(features):
            self._assert_finite(f"features[{idx}]", feature, info)

        loss, metrics, predicted, target = self._consistency_loss(
            target_representation,
            latents,
            features,
            task_id=task.task_id,
        )
        self._assert_finite("predicted", predicted, info)
        self._assert_finite("target", target, info)
        self._assert_finite("loss", loss, info)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        task_ids = _label_tuple(task.task_id)
        for task_id in dict.fromkeys(task_ids):
            self.log(
                f"task/{task_id}",
                torch.tensor(
                    task_ids.count(task_id) / len(task_ids),
                    device=loss.device,
                ),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        self.log(
            "latent/fsq_dropout",
            torch.tensor(float(dont_quantize), device=loss.device),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "latent/std",
            latents.detach().float().std(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        return loss


class MosslandCodecTrainingCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 2,
        demo_every: int = 1000,
        sample_rate: int = 48000,
        use_ema: bool = True,
        silence_seconds: float = 0.25,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = demo_num
        self.demo_every = demo_every
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.silence_seconds = silence_seconds
        self.last_demo_step = -1

    def _concat_demo_audio(self, *segments: torch.Tensor) -> torch.Tensor:
        if not segments:
            raise ValueError("at least one segment is required")
        silence_samples = int(self.sample_rate * self.silence_seconds)
        silence = segments[0].new_zeros(segments[0].shape[:-1] + (silence_samples,))
        pieces = []
        for segment in segments:
            if pieces and silence_samples > 0:
                pieces.append(silence)
            pieces.append(segment)
        return torch.cat(pieces, dim=-1)

    def _clear_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None:
            return
        if trainer.global_step % self.demo_every != 1 or self.last_demo_step == trainer.global_step:
            return
        self.last_demo_step = trainer.global_step

        payload = task = src = target = src_audio = generated = comparison = None
        generated_versions = quantized_generated = continuous_generated = None
        src_item = target_item = generated_item = None
        try:
            self._clear_cuda_cache()
            model = module.model
            if self.use_ema and hasattr(module, "ema"):
                model = module.ema.ema_model

            os.makedirs(self.demo_dir, exist_ok=True)
            payload, _ = batch
            task = MosslandTaskBatch.from_payload(payload)
            src = task.src[: self.demo_num]
            target = task.target[: self.demo_num]
            demo_count = src.shape[0]
            demo_task_id = _slice_label(task.task_id, demo_count)
            src_audio, quantized_generated = model.generate_waveform(
                src,
                task_id=demo_task_id,
                dont_quantize=False,
            )
            _, continuous_generated = model.generate_waveform(
                src,
                task_id=demo_task_id,
                dont_quantize=True,
            )
            generated_versions = (
                ("quantized", quantized_generated),
                ("continuous", continuous_generated),
            )
            target = model.prepare_audio_batch(target).detach().cpu()
            for mode, generated in generated_versions:
                for idx, (src_item, target_item, generated_item) in enumerate(
                    zip(src_audio, target, generated)
                ):
                    task_id = _label_at(demo_task_id, idx)
                    base = f"{trainer.global_step}_{idx}_{task_id}_rank{trainer.global_rank}"
                    comparison = self._concat_demo_audio(src_item, target_item, generated_item)
                    torchaudio.save(
                        os.path.join(
                            self.demo_dir,
                            f"{base}_{mode}_src_target_generated.wav",
                        ),
                        comparison.float(),
                        self.sample_rate,
                    )
        finally:
            payload = task = src = target = src_audio = generated = comparison = None
            generated_versions = quantized_generated = continuous_generated = None
            src_item = target_item = generated_item = None
            self._clear_cuda_cache()
