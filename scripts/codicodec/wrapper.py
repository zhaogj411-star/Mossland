import os

import lightning as pl
import torch
import torchaudio
from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from scripts.codec_common.training_base import CodecTrainingBase
from scripts.codicodec import trigflow
from scripts.codicodec.models import UNet


class CoDiCodecTrainingWrapper(CodecTrainingBase):
    def __init__(
        self,
        model: UNet,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "radam",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 10_000,
        lr_schedule_total_steps: int = 2_000_000,
        final_learning_rate: float = 0.0,
        train_num_chunks: int | None = 2,
        random_mix_prob: float = 0.5,
        rvq_commitment_weight: float = 0.0,
        rvq_codebook_weight: float = 0.0,
        rvq_hidden_recon_weight: float = 0.0,
        rvq_latent_train_prob: float = 0.25,
        rvq_detach_encoder: bool = False,
        train_n_quantizers: int | None = None,
        train_n_quantizers_choices: list[int] | None = None,
        latent_time_shift_prob: float = 0.0,
        latent_time_shift_max_tokens: int | None = None,
        latent_time_shift_loss_weight: float = 1.0,
        p_mean: float = -1.0,
        p_std: float = 1.6,
        tangent_warmup_steps: int = 10_000,
        tangent_norm_const: float = 0.1,
        adaptive_weight_dim: int = 128,
        use_jvp_tangent: bool = True,
        consistency_weight: float = 1.0,
        direct_denoise_weight: float = 0.0,
        direct_denoise_mode: str = "trigflow_noise",
        direct_detach_encoder: bool = False,
        decode_initial_state: str = "noise",
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
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.final_learning_rate = float(final_learning_rate)
        self.train_num_chunks = None if train_num_chunks is None else int(train_num_chunks)
        self.random_mix_prob = random_mix_prob
        self.rvq_commitment_weight = float(rvq_commitment_weight)
        self.rvq_codebook_weight = float(rvq_codebook_weight)
        self.rvq_hidden_recon_weight = float(rvq_hidden_recon_weight)
        self.rvq_latent_train_prob = float(rvq_latent_train_prob)
        self.rvq_detach_encoder = bool(rvq_detach_encoder)
        self.train_n_quantizers = train_n_quantizers
        self.train_n_quantizers_choices = train_n_quantizers_choices
        self.latent_time_shift_prob = float(latent_time_shift_prob)
        self.latent_time_shift_max_tokens = latent_time_shift_max_tokens
        self.latent_time_shift_loss_weight = float(latent_time_shift_loss_weight)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.tangent_warmup_steps = int(tangent_warmup_steps)
        self.tangent_norm_const = float(tangent_norm_const)
        self.use_jvp_tangent = bool(use_jvp_tangent)
        self.consistency_weight = float(consistency_weight)
        self.direct_denoise_weight = float(direct_denoise_weight)
        self.direct_denoise_mode = str(direct_denoise_mode)
        self.direct_detach_encoder = bool(direct_detach_encoder)
        self.decode_initial_state = str(decode_initial_state)
        # Learned EDM2-style uncertainty weighting w_phi(t), trained jointly.
        self.adaptive_weight = trigflow.AdaptiveWeight(dim=int(adaptive_weight_dim))

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower()
        # The adaptive weighting w_phi(t) is trained jointly with the model.
        params = list(self.model.parameters()) + list(self.adaptive_weight.parameters())
        if optimizer_name == "radam":
            opt = torch.optim.RAdam(
                params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            opt = torch.optim.AdamW(
                params,
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
        cosine = CosineAnnealingLR(opt, T_max=decay_steps, eta_min=self.final_learning_rate)
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

    def _prepare_audio_batch(
        self,
        batch: torch.Tensor,
        num_chunks: int | None = None,
    ) -> torch.Tensor:
        num_chunks = self.train_num_chunks if num_chunks is None else int(num_chunks)
        prepared, _ = self.model.prepare_waveform(batch, num_chunks=num_chunks)
        return prepared

    def _maybe_random_mix(self, batch: torch.Tensor):
        if self.random_mix_prob <= 0.0 or batch.shape[0] < 2:
            return batch, torch.zeros((), device=batch.device)
        apply_mix = torch.rand((), device=batch.device) < self.random_mix_prob
        if not bool(apply_mix):
            return batch, torch.zeros((), device=batch.device)
        shuffled = batch[torch.randperm(batch.shape[0], device=batch.device)]
        return (batch + shuffled).clamp(-1.0, 1.0), torch.ones((), device=batch.device)

    def _tangent_warmup(self) -> float:
        """Linear tangent-warmup factor ``r = min(1, step / H)`` (sCM stabilization)."""
        if self.tangent_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.global_step) / float(self.tangent_warmup_steps))

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

    def _should_use_rvq_path(self) -> bool:
        if not self.model.has_quantizer:
            return False
        return (
            self.rvq_latent_train_prob > 0.0
            or self.rvq_commitment_weight > 0.0
            or self.rvq_codebook_weight > 0.0
            or self.rvq_hidden_recon_weight > 0.0
        )

    def _log_tensor_stats(self, prefix: str, tensor: torch.Tensor):
        stats = tensor.detach().float()
        self.log(
            f"{prefix}_mean",
            stats.mean(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_std",
            stats.std(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def _apply_latent_time_shift(
        self,
        representation: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Make encode/decode chunk boundaries disagree by a sub-chunk token shift.

        The encoder always produces latents on the original chunk grid. Right
        before decoding, we crop the target representation from a later frame
        offset and crop the latent stream from the matching 64-d token offset.
        This makes each decoder chunk span a different boundary than the chunks
        used by the encoder, so the chunk-internal latent token order has to carry
        real temporal information.
        """
        spec_length = self.model.spec_length
        num_latents = self.model.num_latents
        bottleneck_channels = self.model.bottleneck_channels
        rvq_dim = self.model.rvq_dim
        rvq_tokens_per_chunk = self.model.rvq_tokens_per_chunk
        if representation.shape[-1] % spec_length != 0:
            raise ValueError(
                f"representation frames {representation.shape[-1]} must be divisible by {spec_length}"
            )
        if latents.shape[-2] % num_latents != 0:
            raise ValueError(
                f"latent tokens {latents.shape[-2]} must be divisible by {num_latents}"
            )

        num_chunks = representation.shape[-1] // spec_length
        latent_chunks = latents.shape[-2] // num_latents
        if num_chunks != latent_chunks:
            raise ValueError(f"representation has {num_chunks} chunks but latents have {latent_chunks}")
        if spec_length % rvq_tokens_per_chunk != 0:
            raise ValueError(
                f"spec_length={spec_length} must be divisible by "
                f"rvq_tokens_per_chunk={rvq_tokens_per_chunk} for aligned latent time shift"
            )

        shift_tokens = 0
        configured_max = (
            rvq_tokens_per_chunk - 1
            if self.latent_time_shift_max_tokens is None
            else int(self.latent_time_shift_max_tokens)
        )
        max_shift_tokens = max(0, min(configured_max, rvq_tokens_per_chunk - 1))
        frames_per_rvq_token = spec_length // rvq_tokens_per_chunk
        if (
            self.latent_time_shift_prob > 0.0
            and num_chunks > 1
            and max_shift_tokens > 0
            and bool(torch.rand((), device=latents.device) < self.latent_time_shift_prob)
        ):
            # If the shift path is active, force a non-zero offset. No-shift
            # examples are controlled only by latent_time_shift_prob.
            shift_tokens = int(torch.randint(1, max_shift_tokens + 1, (), device=latents.device).item())

        # Keep the 64-d RVQ/LLM token intact: reshape [B, chunks*num_latents, C]
        # -> [B, chunks*rvq_tokens_per_chunk, rvq_dim], shift on that sequence,
        # then reshape back for the decoder's chunk-local latent interface.
        usable_chunks = num_chunks if shift_tokens == 0 else num_chunks - 1
        if shift_tokens > 0:
            frame_offset = shift_tokens * frames_per_rvq_token
            target_frames = usable_chunks * spec_length
            target_rvq_tokens = usable_chunks * rvq_tokens_per_chunk
            representation = representation[..., frame_offset : frame_offset + target_frames].contiguous()
            rvq_tokens = latents.reshape(latents.shape[0], num_chunks * rvq_tokens_per_chunk, rvq_dim)
            rvq_tokens = rvq_tokens[:, shift_tokens : shift_tokens + target_rvq_tokens].contiguous()
            latents = rvq_tokens.reshape(
                latents.shape[0],
                usable_chunks * num_latents,
                bottleneck_channels,
            )
        else:
            frame_offset = 0

        metrics = {
            "latent/time_shift_tokens": torch.tensor(float(shift_tokens), device=latents.device),
            "latent/time_shift_token_fraction": torch.tensor(
                float(shift_tokens) / float(rvq_tokens_per_chunk),
                device=latents.device,
            ),
            "latent/time_shift_frames": torch.tensor(float(frame_offset), device=latents.device),
            "latent/usable_chunks": torch.tensor(float(usable_chunks), device=latents.device),
        }
        return representation, latents, metrics

    def _consistency_loss(
        self,
        representation: torch.Tensor,
        latents: torch.Tensor,
        features: list[torch.Tensor],
    ):
        """Continuous-time (sCT) consistency loss in the TrigFlow parameterization.

        Samples a time ``t``, builds ``x_t = cos(t) x0 + sin(t) z``, computes the
        exact tangent via JVP at stop-gradient params, then regresses the trainable
        network output against the (detached) tangent target under learned weighting.
        """
        x0 = representation
        batch_size = x0.shape[0]
        if self.direct_denoise_mode == "trigflow_noise":
            t = trigflow.sample_t(
                batch_size,
                sigma_data=self.model.sigma_data,
                p_mean=self.p_mean,
                p_std=self.p_std,
                t_min=self.model.t_min,
                t_max=self.model.t_max,
                device=x0.device,
            )
            z = torch.randn_like(x0) * self.model.sigma_data
            x_t = trigflow.trigflow_noise(x0, z, t)
        elif self.direct_denoise_mode == "zero_tmax":
            t = torch.full((batch_size,), self.model.t_max, device=x0.device, dtype=x0.dtype)
            z = torch.zeros_like(x0)
            x_t = torch.zeros_like(x0)
        elif self.direct_denoise_mode == "prior_tmax":
            t = torch.full((batch_size,), self.model.t_max, device=x0.device, dtype=x0.dtype)
            z = torch.randn_like(x0) * self.model.sigma_data
            x_t = z
        else:
            raise ValueError(f"Unsupported direct_denoise_mode={self.direct_denoise_mode!r}")

        t_b = t.reshape(-1, 1, 1, 1)
        dxt_dt = torch.cos(t_b) * z - torch.sin(t_b) * x0

        # Pure function of (x_t, t) for the stop-gradient (theta^-) tangent: the
        # conditioning is detached so gradients reach the encoder only via f_pred.
        # Cast to fp32 because consistency_tangent runs the JVP with autocast
        # disabled (fp32 weights), so bf16 conditioning would dtype-mismatch.
        latents_detached = latents.detach().float()
        features_detached = [f.detach().float() for f in features]

        def network_fn(x_in, t_in):
            return self.model._decoder_network(
                x_in, t_in, latents_detached, features=features_detached
            )

        f_pred = self.model._decoder_network(x_t, t, latents, features=features)
        t_clean = t.reshape(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
        clean_pred = torch.cos(t_clean) * x_t - torch.sin(t_clean) * self.model.sigma_data * f_pred

        zero = x0.new_zeros(())
        consistency_loss = zero
        metrics = {}
        f_detached = x0.detach()
        if self.consistency_weight > 0.0 and self.use_jvp_tangent:
            f_detached, g = trigflow.consistency_tangent(
                network_fn,
                x_t,
                t,
                dxt_dt=dxt_dt,
                sigma_data=self.model.sigma_data,
                warmup_r=self._tangent_warmup(),
                norm_const=self.tangent_norm_const,
            )
            t_detached = t.reshape(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
            clean_detached = (
                torch.cos(t_detached) * x_t.float()
                - torch.sin(t_detached) * self.model.sigma_data * f_detached.float()
            )
            raw_consistency_loss, metrics = trigflow.scm_loss(
                clean_pred,
                clean_detached,
                g,
                w_phi=self.adaptive_weight(t),
            )
            consistency_loss = raw_consistency_loss * self.consistency_weight
        elif self.consistency_weight > 0.0:
            # Debug path: isolate DDP/autocast forward+backward from torch.func.jvp.
            with torch.no_grad():
                f_detached = network_fn(x_t.float(), t.float()).detach()
            g = torch.zeros_like(f_detached)
            t_detached = t.reshape(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
            clean_detached = (
                torch.cos(t_detached) * x_t.float()
                - torch.sin(t_detached) * self.model.sigma_data * f_detached.float()
            )
            raw_consistency_loss, metrics = trigflow.scm_loss(
                clean_pred,
                clean_detached,
                g,
                w_phi=self.adaptive_weight(t),
            )
            consistency_loss = raw_consistency_loss * self.consistency_weight

        direct_residual_sq = (clean_pred.float() - x0.float()).pow(2).flatten(1)
        direct_mse = direct_residual_sq.mean(dim=1).mean()
        direct_sse = direct_residual_sq.sum(dim=1).mean()
        direct_loss = direct_sse
        loss = consistency_loss + self.direct_denoise_weight * direct_loss
        metrics["loss/consistency"] = consistency_loss.detach()
        metrics["loss/direct_denoise"] = direct_loss.detach()
        metrics["loss/direct_denoise_mse"] = direct_mse.detach()
        metrics["loss/direct_denoise_sse"] = direct_sse.detach()
        metrics["loss/direct_denoise_weighted"] = (
            self.direct_denoise_weight * direct_loss
        ).detach()
        metrics["sigma/t_mean"] = t.mean().detach()
        metrics["tangent/warmup_r"] = torch.tensor(self._tangent_warmup(), device=x0.device)
        # Clean-estimate prediction kept for the finite-value guard in training_step.
        predicted = clean_pred
        target = x0.detach()
        return loss, metrics, predicted, target

    def _loss_on_aligned_pair(
        self,
        representation: torch.Tensor,
        latents: torch.Tensor,
        info,
        *,
        finite_prefix: str,
    ):
        self._assert_finite(f"{finite_prefix}_representation", representation, info)
        self._assert_finite(f"{finite_prefix}_latents", latents, info)
        features = self.model.pre_decoder_forward(latents)
        for idx, feature in enumerate(features):
            self._assert_finite(f"{finite_prefix}_features[{idx}]", feature, info)
        loss, metrics, predicted, target = self._consistency_loss(
            representation, latents, features
        )
        self._assert_finite(f"{finite_prefix}_predicted", predicted, info)
        self._assert_finite(f"{finite_prefix}_target", target, info)
        return loss, metrics, predicted, target

    def training_step(self, batch, batch_idx):
        batch, info = batch
        self._assert_finite("batch", batch, info)
        batch = self._prepare_audio_batch(batch)
        batch, mix_applied = self._maybe_random_mix(batch)
        self._assert_finite("prepared_batch", batch, info)

        representation = self.model.audio_processor.to_representation_encoder(batch)
        self._assert_finite("representation", representation, info)

        quantized = None
        source_discrete = representation.new_zeros(())
        if self._should_use_rvq_path():
            n_quantizers = self._sample_n_quantizers()
            quantized = self.model.quantize_representation(
                representation,
                detach_encoder=self.rvq_detach_encoder,
                n_quantizers=n_quantizers,
            )
            latents = quantized.continuous
            rvq_latent_train_prob = max(0.0, min(1.0, self.rvq_latent_train_prob))
            if rvq_latent_train_prob > 0.0:
                batch_size = quantized.continuous.shape[0]
                discrete_mask = (
                    torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=quantized.continuous.device,
                    )
                    if rvq_latent_train_prob >= 1.0
                    else torch.rand(
                        batch_size,
                        device=quantized.continuous.device,
                    )
                    < rvq_latent_train_prob
                )
                source_discrete = discrete_mask.float().mean()
                discrete_mask = discrete_mask.view(
                    batch_size,
                    *([1] * (quantized.continuous.ndim - 1)),
                )
                latents = torch.where(
                    discrete_mask,
                    quantized.discrete,
                    quantized.continuous,
                )
            if n_quantizers is not None:
                self.log("rvq/n_quantizers", float(n_quantizers), prog_bar=False)
            self._log_tensor_stats("latent/continuous", quantized.continuous)
            self._log_tensor_stats("latent/discrete", quantized.discrete)
        else:
            latents = self.model.encoder_forward(representation)
            self._log_tensor_stats("latent/continuous", latents)
        self._assert_finite("latents", latents, info)

        if self.direct_detach_encoder:
            latents = latents.detach()

        aligned_loss, metrics, predicted, target = self._loss_on_aligned_pair(
            representation, latents, info, finite_prefix="aligned"
        )
        loss = aligned_loss
        metrics["loss/aligned"] = aligned_loss.detach()

        shifted_loss = representation.new_zeros(())
        shifted_metrics = {}
        shifted_representation, shifted_latents, shift_metrics = self._apply_latent_time_shift(
            representation, latents
        )
        if (
            self.latent_time_shift_loss_weight > 0.0
            and float(shift_metrics["latent/time_shift_tokens"].detach().cpu()) > 0.0
        ):
            shifted_loss, shifted_metrics, _shifted_predicted, _shifted_target = (
                self._loss_on_aligned_pair(
                    shifted_representation,
                    shifted_latents,
                    info,
                    finite_prefix="shifted",
                )
            )
            loss = loss + self.latent_time_shift_loss_weight * shifted_loss
        metrics["loss/shifted"] = shifted_loss.detach()
        metrics["loss/shifted_weighted"] = (
            self.latent_time_shift_loss_weight * shifted_loss
        ).detach()
        for name, value in shifted_metrics.items():
            metrics[f"shifted/{name}"] = value.detach()

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
        self._assert_finite("loss", loss, info)

        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "augment/random_mix",
            mix_applied,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "latent/source_discrete",
            source_discrete,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        for name, value in metrics.items():
            self.log(
                name,
                value,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        for name, value in shift_metrics.items():
            self.log(
                name,
                value,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return loss

    @torch.no_grad()
    def reconstruct_waveform(
        self,
        batch: torch.Tensor,
        model: UNet | None = None,
        n_quantizers: int | None = None,
        num_chunks: int | None = None,
        slide: bool = True,
        denoising_steps: int | None = None,
        initial_state: str | None = None,
    ):
        model = model or self.model
        batch = self._prepare_audio_batch(batch, num_chunks=num_chunks)
        representation = model.audio_processor.to_representation_encoder(batch)
        if n_quantizers is not None:
            latents = model.quantize_representation(
                representation,
                detach_encoder=True,
                n_quantizers=n_quantizers,
            ).discrete
        else:
            latents = model.encoder_forward(representation)
        reconstructed = model.decode(
            latents,
            denoising_steps=denoising_steps,
            slide=slide,
            initial_state=initial_state or self.decode_initial_state,
        )
        waveform = model.audio_processor.to_waveform(
            reconstructed[..., : representation.shape[-1]],
            model.hop,
        )
        return batch.detach().float().cpu(), waveform.detach().float().cpu()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        dataset = getattr(getattr(self.trainer, "datamodule", None), "dataset", None)
        if dataset is not None and hasattr(dataset, "filenames"):
            self.log("data_nums", len(dataset.filenames), on_step=True, on_epoch=False)


class CoDiCodecTrainingCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 2,
        demo_every: int = 1000,
        sample_rate: int = 44100,
        demo_n_quantizers_choices: list[int] | None = None,
        use_ema: bool = True,
        silence_seconds: float = 0.25,
        demo_seconds: float | None = None,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = demo_num
        self.demo_every = demo_every
        self.sample_rate = sample_rate
        self.demo_n_quantizers_choices = demo_n_quantizers_choices
        self.use_ema = use_ema
        self.silence_seconds = silence_seconds
        self.demo_seconds = demo_seconds
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

    def _demo_rates_for_model(self, model):
        if not model.has_quantizer:
            return []
        rates = (
            [8, 16, 32]
            if self.demo_n_quantizers_choices is None
            else self.demo_n_quantizers_choices
        )
        return [int(n) for n in rates if 1 <= int(n) <= model.quantizer.n_codebooks]

    def _demo_audio_from_batch(self, batch, model):
        audio, _info = batch
        return audio[: self.demo_num], None

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None:
            return
        if (
            trainer.global_step % self.demo_every != 1
            or self.last_demo_step == trainer.global_step
        ):
            return
        self.last_demo_step = trainer.global_step

        model = module.model
        if self.use_ema and hasattr(module, "ema"):
            model = module.ema.ema_model

        os.makedirs(self.demo_dir, exist_ok=True)
        audio, demo_chunks = self._demo_audio_from_batch(batch, model)
        originals = None
        discrete_reconstructions = []
        for n_quantizers in self._demo_rates_for_model(model):
            originals, reconstructed = module.reconstruct_waveform(
                audio,
                model=model,
                n_quantizers=n_quantizers,
                num_chunks=demo_chunks,
            )
            discrete_reconstructions.append((n_quantizers, reconstructed))
        if originals is None:
            originals = module.reconstruct_waveform(
                audio,
                model=model,
                num_chunks=demo_chunks,
            )[0]
        # Continuous reconstruction with both decoders for A/B comparison:
        # sliding-window (shifted-grid seam mitigation) vs plain aligned grid.
        _, continuous_slide = module.reconstruct_waveform(
            audio,
            model=model,
            num_chunks=demo_chunks,
            slide=True,
        )
        _, continuous_noslide = module.reconstruct_waveform(
            audio,
            model=model,
            num_chunks=demo_chunks,
            slide=False,
        )
        for idx, (original, cont_slide, cont_noslide) in enumerate(
            zip(originals, continuous_slide, continuous_noslide)
        ):
            base = f"{trainer.global_step}_{idx}_rank{trainer.global_rank}"
            discrete_segments = [reconstructed[idx] for _, reconstructed in discrete_reconstructions]
            labels = "_".join(
                f"discrete{label}" if isinstance(label, int) else str(label)
                for label, _ in discrete_reconstructions
            )
            if not labels:
                labels = "no_discrete"
            # Layout: input | discrete rates... | continuous(slide) | continuous(no-slide)
            comparison = self._concat_demo_audio(
                original, *discrete_segments, cont_slide, cont_noslide
            )
            torchaudio.save(
                os.path.join(self.demo_dir, f"{base}_input_{labels}_contSlide_contAligned.mp3"),
                comparison.float(),
                self.sample_rate,
            )
