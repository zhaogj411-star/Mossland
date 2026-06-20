import lightning as pl

from scripts.codec_common.training_base import (
    CodecTrainingBase,
    add_noise,
    get_sigma_continuous,
    pseudo_huber_loss,
)
from scripts.music2latent.models import Music2Latent

from ema_pytorch import EMA
import torch
from safetensors.torch import save_file, save_model
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torchaudio
import gc
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class Music2LatnetTrainingWrapper(CodecTrainingBase):
    def __init__(
        self,
        model: Music2Latent,
        use_ema: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 10_000,
        lr_schedule_total_steps: int = 2_000_000,
        final_learning_rate: float = 1e-6,
        rvq_commitment_weight: float = 1.0,
        rvq_codebook_weight: float = 1.0,
        rvq_distill_weight: float = 1.0,
        rvq_hidden_recon_weight: float | None = None,
        rvq_latent_train_prob: float = 0.1,
        rvq_detach_encoder: bool = True,
        train_n_quantizers: int | None = None,
        train_n_quantizers_choices: list[int] | None = None,
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
        fail_on_nonfinite_grad: bool | None = None,
    ):
        super().__init__(
            model=model,
            use_ema=False,
            learning_rate=learning_rate,
            fail_on_nonfinite=fail_on_nonfinite,
            fail_on_nonfinite_grad=fail_on_nonfinite_grad,
        )
        self.rvq_commitment_weight = rvq_commitment_weight
        self.rvq_codebook_weight = rvq_codebook_weight
        self.rvq_hidden_recon_weight = (
            rvq_distill_weight
            if rvq_hidden_recon_weight is None
            else rvq_hidden_recon_weight
        )
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.final_learning_rate = float(final_learning_rate)
        self.rvq_latent_train_prob = float(rvq_latent_train_prob)
        self.rvq_detach_encoder = rvq_detach_encoder
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
        if use_ema:
            self.ema = EMA(
                self.model,
                beta=0.9999,
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
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _log_current_lr(self):
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None:
            return
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return
        lr = optimizers[0].param_groups[0]["lr"]
        self.log("optim/lr", lr, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

    def _log_tensor_stats(self, prefix: str, tensor: torch.Tensor):
        stats = tensor.detach().float()
        self.log(f"{prefix}_mean", stats.mean(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{prefix}_std", stats.std(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

    def _sample_n_quantizers(self):
        if self.train_n_quantizers_choices:
            idx = torch.randint(
                len(self.train_n_quantizers_choices),
                (),
                device=self.device,
            ).item()
            return int(self.train_n_quantizers_choices[idx])
        if self.train_n_quantizers is not None:
            return int(self.train_n_quantizers)
        return None

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

    def _official_lognormal_positions(
        self,
        batch_size: int,
        device: torch.device,
        num_bins: int = 10_000,
    ) -> torch.Tensor:
        # Official Music2Latent samples 10000 discretized EDM sigmas with
        # lognormal weights, then jitters inside the selected bin.
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

    def _consistency_loss(self, representation, latent_override):
        batch_size = representation.shape[0]
        sigma_low, sigma_high, step_size = self._sample_sigma_pair(
            batch_size,
            representation.device,
        )
        noise = torch.randn_like(representation)
        noisy_high = add_noise(representation, noise, sigma_high)
        noisy_low = add_noise(representation, noise, sigma_low)

        predicted = self.model(
            representation,
            noisy_high,
            sigma=sigma_high,
            latent_override=latent_override,
        )
        with torch.no_grad():
            target = self.model(
                representation,
                noisy_low,
                sigma=sigma_low,
                latent_override=latent_override,
            )

        sigma_delta = (sigma_high - sigma_low).clamp(
            min=self.consistency_min_sigma_delta
        )
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
            "sigma/step": torch.tensor(step_size, device=representation.device),
            "sigma/low_mean": sigma_low.mean().detach(),
            "sigma/high_mean": sigma_high.mean().detach(),
        }
        return loss, metrics, predicted, target

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.global_step % 3000 == 1:
    #         self.trainer.datamodule.dataset.refresh_filenames()
    #         self.log("data_nums", len(self.trainer.datamodule.dataset.filenames))

    def training_step(self, batch, batch_idx):
        self._log_current_lr()
        batch, info = batch
        batch = batch.to(next(self.model.parameters()).dtype)
        self.model.train()
        hop = self.model.hop
        fac = getattr(self.model.audio_processor, "fac", 4)
        downscaling_factor = 2 ** sum(
            1 for x in self.model.freq_downsample_list if x == 0
        )
        if getattr(self.model.audio_processor, "center_pad", False):
            stft_frames = batch.shape[-1] // hop
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = cropped_frames * hop * downscaling_factor
        else:
            frame_length = fac * hop
            stft_frames = max(0, (batch.shape[-1] - frame_length) // hop + 1)
            cropped_frames = stft_frames // downscaling_factor
            cropped_length = (
                cropped_frames * hop * downscaling_factor + (fac - 1) * hop
            )
        batch = batch[..., :cropped_length]
        representation = self.model.audio_processor.to_representation_encoder(batch)
        self._assert_finite("representation", representation, info)
        quantized = None
        latent_override = None
        source_discrete = representation.new_zeros(())
        if self.model.has_quantizer:
            n_quantizers = self._sample_n_quantizers()
            quantized = self.model.quantize_representation(
                representation,
                detach_encoder=self.rvq_detach_encoder,
                n_quantizers=n_quantizers,
            )
            latent_override = quantized.continuous
            rvq_latent_train_prob = max(0.0, min(1.0, self.rvq_latent_train_prob))
            if rvq_latent_train_prob > 0.0:
                batch_size = quantized.continuous.shape[0]
                if rvq_latent_train_prob >= 1.0:
                    discrete_mask = torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=quantized.continuous.device,
                    )
                else:
                    discrete_mask = (
                        torch.rand(batch_size, device=quantized.continuous.device)
                        < rvq_latent_train_prob
                    )
                source_discrete = discrete_mask.float().mean()
                discrete_mask = discrete_mask.view(
                    batch_size,
                    *([1] * (quantized.continuous.ndim - 1)),
                )
                latent_override = torch.where(
                    discrete_mask,
                    quantized.discrete,
                    quantized.continuous,
                )
            if n_quantizers is not None:
                self.log("rvq/n_quantizers", float(n_quantizers), prog_bar=False)
            self._log_tensor_stats("latent/continuous", quantized.continuous)
            self._log_tensor_stats("latent/discrete", quantized.discrete)
            self._log_tensor_stats("rvq/quantized_hidden", quantized.projected_latents)

        consistency_loss, metrics, predicted, target = self._consistency_loss(
            representation,
            latent_override=latent_override,
        )
        self._assert_finite("predicted", predicted, info)
        self._assert_finite("target", target, info)
        loss = consistency_loss
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False)
        self.log("latent/source_discrete", source_discrete, prog_bar=False)
        if latent_override is not None:
            self.log(
                "latent/selected_mean",
                latent_override.detach().float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            self.log(
                "latent/selected_std",
                latent_override.detach().float().std(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        if quantized is not None:
            rvq_loss = (
                self.rvq_commitment_weight * quantized.commitment_loss
                + self.rvq_codebook_weight * quantized.codebook_loss
                + self.rvq_hidden_recon_weight * quantized.distill_loss
            )
            loss = loss + rvq_loss
            self.log(
                "rvq/commitment_loss",
                quantized.commitment_loss,
                prog_bar=True,
            )
            self.log("rvq/codebook_loss", quantized.codebook_loss, prog_bar=False)
            self.log("rvq/hidden_recon_loss", quantized.distill_loss, prog_bar=False)
            self.log("loss/rvq", rvq_loss, prog_bar=False)
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


class Music2LatnetTrainingCallback(pl.Callback):

    def __init__(
        self,
        demo_dir,
        demo_num=1,
        demo_every=2000,
        sample_rate=44100,
        demo_n_quantizers: int | None = None,
        demo_n_quantizers_choices: list[int] | None = None,
        denoising_steps: int = 4,
        demo_seed: int = 1234,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = demo_num
        self.demo_every = demo_every
        self.sample_rate = sample_rate
        self.demo_n_quantizers = demo_n_quantizers
        self.demo_n_quantizers_choices = demo_n_quantizers_choices
        self.denoising_steps = int(denoising_steps)
        self.demo_seed = int(demo_seed)
        self.last_demo_step = -1

    def _clear_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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

    def _set_demo_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _demo_rates_for_model(self, model):
        demo_rates = self.demo_n_quantizers_choices
        if self.demo_n_quantizers is not None:
            demo_rates = [self.demo_n_quantizers]
        if demo_rates is None and model.has_quantizer:
            demo_rates = [4, 8, 16, 32]
        if demo_rates is not None and model.has_quantizer:
            return [
                int(n)
                for n in demo_rates
                if 1 <= int(n) <= model.quantizer.n_codebooks
            ]
        return []

    def _save_model_demo(
        self,
        model,
        model_label,
        batch,
        trainer,
        current_rank,
    ):
        demo_rates = self._demo_rates_for_model(model)
        for i in range(batch.shape[0]):
            source = batch[i].detach().cpu()
            if source.ndim == 1:
                source = source.unsqueeze(0)
            continuous_latent = model.encode(batch[i])
            self._set_demo_seed(
                self.demo_seed + trainer.global_step * 100_000 + i * 100
            )
            continuous = model.decode(
                continuous_latent,
                denoising_steps=self.denoising_steps,
                target_length=source.shape[-1],
            )[0].detach().cpu()
            segments = [source]
            rate_labels = []
            if model.has_quantizer:
                for rate_idx, n_quantizers in enumerate(demo_rates):
                    discrete_latent = model.encode(
                        batch[i],
                        quantize=True,
                        n_quantizers=n_quantizers,
                    )
                    self._set_demo_seed(
                        self.demo_seed
                        + trainer.global_step * 100_000
                        + i * 100
                        + rate_idx
                        + 1
                    )
                    discrete = model.decode(
                        discrete_latent,
                        denoising_steps=self.denoising_steps,
                        target_length=source.shape[-1],
                    )[0].detach().cpu()
                    if discrete.ndim == 1:
                        discrete = discrete.unsqueeze(0)
                    segments.append(discrete)
                    rate_labels.append(str(n_quantizers))
            if continuous.ndim == 1:
                continuous = continuous.unsqueeze(0)
            silence = torch.zeros(
                source.shape[0],
                int(self.sample_rate * 0.25),
                dtype=source.dtype,
            )
            segments.append(continuous)
            results = torch.cat(
                [part for segment in segments for part in (segment, silence)][:-1],
                dim=-1,
            )
            discrete_label = "_".join(rate_labels) if rate_labels else "none"
            torchaudio.save(
                os.path.join(
                    self.demo_dir,
                    f"{trainer.global_step}_{i}_rank{current_rank}_{model_label}_gt_discrete{discrete_label}_continuous.mp3",
                ),
                results.clamp(-1.0, 1.0),
                self.sample_rate,
                format="mp3",
            )

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None:
            return
        if trainer.global_rank != 0:
            return
        if (
            trainer.global_step
        ) % self.demo_every != 1 or self.last_demo_step == trainer.global_step:
            return
        current_rank = trainer.global_rank

        self.last_demo_step = trainer.global_step
        self._clear_cuda_cache()
        rng_state = self._capture_rng_state()
        try:
            # trainer.datamodule.dataset.refresh_filenames()
            os.makedirs(self.demo_dir, exist_ok=True)
            batch, info = batch
            batch = batch[: self.demo_num]
            self._save_model_demo(module.model, "raw", batch, trainer, current_rank)
            if hasattr(module, "ema"):
                self._save_model_demo(
                    module.ema.ema_model,
                    "ema",
                    batch,
                    trainer,
                    current_rank,
                )
        finally:
            self._restore_rng_state(rng_state)
            self._clear_cuda_cache()
