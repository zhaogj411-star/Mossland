from __future__ import annotations

import gc
import json
import os
from collections.abc import Iterable

import lightning as pl
import torch
import torchaudio
from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import MosslandCodecConvTrans
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


def _normalize_positive_int_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return normalized


def _iter_reindexable_datasets(root, seen: set[int] | None = None):
    if root is None:
        return
    if seen is None:
        seen = set()
    obj_id = id(root)
    if obj_id in seen:
        return
    seen.add(obj_id)

    rebuild_index = getattr(root, "rebuild_index", None)
    if callable(rebuild_index):
        yield root

    for attr_name in ("dataset", "datasets", "train_dataset", "val_dataset", "test_dataset"):
        child = getattr(root, attr_name, None)
        if child is None:
            continue
        if isinstance(child, dict):
            children = child.values()
        elif isinstance(child, Iterable) and not isinstance(child, (str, bytes)):
            children = child
        else:
            children = (child,)
        for item in children:
            yield from _iter_reindexable_datasets(item, seen)


def _rebuild_dataset_indexes(datamodule) -> int:
    rebuilt = 0
    seen = set()
    for dataset in _iter_reindexable_datasets(datamodule):
        dataset_id = id(dataset)
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        dataset.rebuild_index()
        rebuilt += 1
    return rebuilt


class MosslandCodecTrainingWrapper(CodecTrainingBase):
    """Mossland 多任务 codec 训练 wrapper。"""

    def __init__(
        self,
        model: MosslandCodecConvTrans,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "radam",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 10_000,
        lr_schedule_total_steps: int = 2_000_000,
        random_mix_prob: float = 0.0,
        rvq_commitment_weight: float = 0.0,
        rvq_codebook_weight: float = 0.0,
        rvq_distill_weight: float = 0.0,
        rvq_latent_train_prob: float = 0.25,
        rvq_detach_encoder: bool = False,
        train_n_quantizers: int | None = None,
        train_n_quantizers_choices: list[int] | None = None,
        rvq_log_n_quantizers_choices: list[int] | None = None,
        consistency_step: float = 0.1,
        consistency_step_schedule: str = "exponential",
        consistency_step_total_steps: int = 2_000_000,
        consistency_step_end_exp: float = 2.0,
        sigma_sampling: str = "lognormal",
        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
        consistency_loss_delta: float = 0.00054,
        consistency_min_sigma_delta: float = 0.001,
        train_input_chunks: int = 2,
        index_data_every_step: int | None = None,
        rvq_sync_debug_every_step: int | None = None,
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
        self.random_mix_prob = float(random_mix_prob)
        self.rvq_commitment_weight = float(rvq_commitment_weight)
        self.rvq_codebook_weight = float(rvq_codebook_weight)
        self.rvq_distill_weight = float(rvq_distill_weight)
        self.rvq_latent_train_prob = float(rvq_latent_train_prob)
        self.rvq_detach_encoder = bool(rvq_detach_encoder)
        self.train_n_quantizers = train_n_quantizers
        self.train_n_quantizers_choices = train_n_quantizers_choices
        self.rvq_log_n_quantizers_choices = (
            [int(value) for value in rvq_log_n_quantizers_choices]
            if rvq_log_n_quantizers_choices is not None
            else [16, 32, 64, 128]
        )
        self.consistency_step = float(consistency_step)
        self.consistency_step_schedule = consistency_step_schedule
        self.consistency_step_total_steps = int(consistency_step_total_steps)
        self.consistency_step_end_exp = float(consistency_step_end_exp)
        self.sigma_sampling = sigma_sampling
        self.lognormal_mean = float(lognormal_mean)
        self.lognormal_std = float(lognormal_std)
        self.consistency_loss_delta = float(consistency_loss_delta)
        self.consistency_min_sigma_delta = float(consistency_min_sigma_delta)
        self.train_input_chunks = int(train_input_chunks)
        if self.train_input_chunks < 2:
            raise ValueError("train_input_chunks must be >= 2")
        self.index_data_every_step = _normalize_positive_int_or_none(
            index_data_every_step,
            "index_data_every_step",
        )
        self.rvq_sync_debug_every_step = _normalize_positive_int_or_none(
            rvq_sync_debug_every_step,
            "rvq_sync_debug_every_step",
        )
        self._last_index_data_step: int | None = None
        self._last_rvq_sync_debug_step: int | None = None

    def _set_rvq_distributed_sync(self, model: MosslandCodecConvTrans) -> None:
        if getattr(model, "rvq", None) is None:
            return
        enabled = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        model.rvq.set_distributed_sync(enabled)

    def on_train_start(self):
        self._set_rvq_distributed_sync(self.model)
        if hasattr(self, "ema") and getattr(self.ema, "ema_model", None) is not None:
            self._set_rvq_distributed_sync(self.ema.ema_model)

    def _sync_ema_rvq_from_raw(self) -> None:
        if not hasattr(self, "ema"):
            return
        ema_model = getattr(self.ema, "ema_model", None)
        if ema_model is None:
            return
        if getattr(self.model, "rvq", None) is None or getattr(ema_model, "rvq", None) is None:
            return
        ema_model.rvq.load_state_dict(self.model.rvq.state_dict(), strict=True)

    def on_before_zero_grad(self, *args, **kwargs):
        if hasattr(self, "ema"):
            self.ema.update()
            self._sync_ema_rvq_from_raw()

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

    def _sample_n_quantizers(self) -> int | None:
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

    def _log_current_lr(self):
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None or not getattr(trainer, "optimizers", None):
            return
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        self.log("optim/lr", lr, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

    def _log_tensor_stats(self, prefix: str, tensor: torch.Tensor):
        stats = tensor.detach().float()
        self.log(f"{prefix}_mean", stats.mean(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{prefix}_std", stats.std(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

    def _log_conv_bottleneck_stats(self):
        pre_tanh = getattr(self.model, "_last_conv_pre_tanh", None)
        tokens = getattr(self.model, "_last_conv_tokens", None)
        if pre_tanh is not None:
            self._log_tensor_stats("latent/pre_tanh", pre_tanh)
            pre = pre_tanh.detach().float()
            self.log(
                "latent/pre_tanh_absmax",
                pre.abs().max(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        if tokens is not None:
            bounded = tokens.detach().float()
            self.log(
                "latent/saturation_0p95",
                (bounded.abs() > 0.95).float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            self.log(
                "latent/saturation_0p99",
                (bounded.abs() > 0.99).float().mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    @torch.no_grad()
    def _rvq_rank_checksum(self) -> torch.Tensor:
        rvq = getattr(self.model, "rvq", None)
        if rvq is None:
            return torch.zeros(1, device=self.device)
        device = self.device
        values = []
        layers = getattr(getattr(rvq, "vq", None), "layers", [])
        layer_indices = [0, 15, 31, 63, 127]
        for index in layer_indices:
            if index >= len(layers):
                continue
            codebook = layers[index]._codebook
            for name in ("embed", "cluster_size", "embed_avg"):
                tensor = getattr(codebook, name, None)
                if tensor is None:
                    continue
                stats = tensor.detach().float()
                values.extend(
                    [
                        stats.mean(),
                        stats.std(),
                        stats.norm() / max(1, stats.numel()),
                    ]
                )
        if not values:
            return torch.zeros(1, device=device)
        return torch.stack([value.to(device=device) for value in values])

    @torch.no_grad()
    def _maybe_log_rvq_rank_sync(self, trainer) -> None:
        if self.rvq_sync_debug_every_step is None or getattr(self.model, "rvq", None) is None:
            return
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step == self._last_rvq_sync_debug_step:
            return
        if step % self.rvq_sync_debug_every_step != 0:
            return
        self._last_rvq_sync_debug_step = step

        local = self._rvq_rank_checksum()
        world_size = 1
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        gathered = [torch.empty_like(local) for _ in range(world_size)]
        if world_size > 1:
            torch.distributed.all_gather(gathered, local)
        else:
            gathered[0].copy_(local)
        stacked = torch.stack(gathered)
        max_abs_diff = (stacked - stacked[:1]).abs().max()
        checksum_norm = stacked.norm(dim=1)

        self.log(
            "rvq_sync/max_abs_diff_from_rank0",
            max_abs_diff,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "rvq_sync/rank_checksum_norm",
            checksum_norm[rank],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        if rank != 0:
            return
        try:
            log_dir = getattr(trainer, "log_dir", None) or getattr(trainer, "default_root_dir", None)
        except RuntimeError:
            log_dir = None
        if not log_dir:
            return
        path = os.path.join(log_dir, "rvq_rank_sync_debug.jsonl")
        record = {
            "step": step,
            "world_size": world_size,
            "max_abs_diff_from_rank0": float(max_abs_diff.detach().cpu()),
            "checksum_norms": [float(value) for value in checksum_norm.detach().cpu()],
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def _log_rvq_rate_errors(self, packed_continuous: torch.Tensor):
        if self.model.rvq is None or not self.rvq_log_n_quantizers_choices:
            return
        was_training = self.model.rvq.training
        self.model.rvq.eval()
        try:
            quantizer_input = packed_continuous.detach()
            initialized_count = self.model.rvq.initialized_codebook_count()
            for n_quantizers in self.rvq_log_n_quantizers_choices:
                if n_quantizers < 1 or n_quantizers > self.model.rvq_num_quantizers:
                    continue
                if int(n_quantizers) > initialized_count:
                    continue
                packed_quantized, _, _ = self.model.rvq(
                    quantizer_input,
                    n_quantizers=int(n_quantizers),
                )
                error = torch.nn.functional.mse_loss(
                    packed_quantized.float(),
                    packed_continuous.detach().float(),
                )
                self.log(
                    f"rvq/quantization_error_{int(n_quantizers)}",
                    error.detach(),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
        finally:
            self.model.rvq.train(was_training)

    def _maybe_random_mix(
        self,
        src: torch.Tensor,
        target_audio: torch.Tensor,
        task_id: str | tuple[str, ...],
    ):
        if self.random_mix_prob <= 0.0 or src.shape[0] < 2:
            return src, target_audio, torch.zeros((), device=src.device)

        task_ids = _label_tuple(task_id)
        if any(label != "reconstruct" for label in task_ids):
            raise ValueError(
                "random_mix_prob is only supported for reconstruction-only "
                "Mossland batches because mixing breaks task src/target pairs"
            )

        apply_mix = torch.rand((), device=src.device) < self.random_mix_prob
        if not bool(apply_mix):
            return src, target_audio, torch.zeros((), device=src.device)

        permutation = torch.randperm(src.shape[0], device=src.device)
        src = (src + src[permutation]).clamp(-1.0, 1.0)
        target_audio = (target_audio + target_audio[permutation]).clamp(-1.0, 1.0)
        return src, target_audio, torch.ones((), device=src.device)

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

    def _crop_decoder_training_window(
        self,
        representation: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the original chunk-aligned 2-chunk window for decoder training."""
        target_frames = 2 * self.model.spec_length
        target_tokens = 2 * self.model.rvq_tokens_per_chunk
        if representation.shape[-1] < target_frames:
            raise ValueError(
                f"not enough representation frames for decoder window: "
                f"frames={representation.shape[-1]}, need={target_frames}"
            )
        packed = self.model.pack_rvq_tokens(latents)
        if packed.shape[-1] < target_tokens:
            raise ValueError(
                f"not enough latent tokens for decoder window: "
                f"tokens={packed.shape[-1]}, need={target_tokens}"
            )
        representation = representation[..., :target_frames]
        latents = self.model.unpack_rvq_tokens(packed[..., :target_tokens])
        return representation, latents

    def training_step(self, batch, batch_idx):
        self._log_current_lr()
        payload, info = batch
        task = MosslandTaskBatch.from_payload(payload)

        src = self.model.prepare_audio_batch(task.src, num_chunks=self.train_input_chunks)
        target_audio = self.model.prepare_audio_batch(
            task.target,
            num_chunks=self.train_input_chunks,
        )
        src, target_audio, mix_applied = self._maybe_random_mix(
            src,
            target_audio,
            task.task_id,
        )
        self._assert_finite("src", src, info)
        self._assert_finite("target_audio", target_audio, info)

        src_representation = self.model.to_representation_encoder(src)
        target_representation = self.model.to_representation_encoder(target_audio)
        self._assert_finite("src_representation", src_representation, info)
        self._assert_finite("target_representation", target_representation, info)

        rvq_encoded = None
        rvq_loss = src_representation.new_zeros(())
        n_quantizers = None
        source_discrete = src_representation.new_zeros(())
        if self.model.rvq is not None:
            n_quantizers = self._sample_n_quantizers()
            rvq_encoded = self.model.encode_bottleneck(
                src_representation,
                quantize=True,
                detach_encoder=self.rvq_detach_encoder,
                n_quantizers=n_quantizers,
            )
            latents = rvq_encoded.continuous
            rvq_latent_train_prob = max(0.0, min(1.0, self.rvq_latent_train_prob))
            if rvq_latent_train_prob > 0.0:
                batch_size = rvq_encoded.continuous.shape[0]
                discrete_mask = (
                    torch.ones(batch_size, dtype=torch.bool, device=rvq_encoded.continuous.device)
                    if rvq_latent_train_prob >= 1.0
                    else torch.rand(batch_size, device=rvq_encoded.continuous.device) < rvq_latent_train_prob
                )
                source_discrete = discrete_mask.float().mean()
                discrete_mask = discrete_mask.view(batch_size, *([1] * (rvq_encoded.continuous.ndim - 1)))
                latents = torch.where(discrete_mask, rvq_encoded.latents, rvq_encoded.continuous)
        else:
            encoded = self.model.encode_bottleneck(
                src_representation,
                quantize=False,
            )
            latents = encoded.latents
        target_representation, latents = self._crop_decoder_training_window(
            target_representation,
            latents,
        )
        self._assert_finite("latents", latents, info)
        features = self.model.pre_decoder_forward(latents)
        for idx, feature in enumerate(features):
            self._assert_finite(f"features[{idx}]", feature, info)

        consistency_loss, metrics, predicted, target = self._consistency_loss(
            target_representation,
            latents,
            features,
            task_id=task.task_id,
        )
        self._assert_finite("predicted", predicted, info)
        self._assert_finite("target", target, info)
        loss = consistency_loss
        if rvq_encoded is not None:
            rvq_loss = (
                self.rvq_commitment_weight * rvq_encoded.commitment_loss
                + self.rvq_codebook_weight * rvq_encoded.codebook_loss
                + self.rvq_distill_weight * rvq_encoded.distill_loss
            )
            if self.rvq_commitment_weight or self.rvq_codebook_weight or self.rvq_distill_weight:
                loss = loss + rvq_loss
        self._assert_finite("loss", loss, info)

        self.log("loss/total", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        self.log("loss/rvq", rvq_loss.detach(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(
            "augment/random_mix",
            mix_applied,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
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
            "latent/source_discrete",
            source_discrete.to(device=loss.device),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self._log_tensor_stats("latent/selected", latents)
        self._log_conv_bottleneck_stats()
        if self.model.rvq is not None and rvq_encoded is not None:
            if n_quantizers is not None:
                self.log("rvq/n_quantizers", float(n_quantizers), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
            self._log_tensor_stats("latent/continuous", rvq_encoded.continuous)
            self._log_tensor_stats("latent/discrete", rvq_encoded.latents)
            if rvq_encoded.packed_continuous is not None:
                self._log_tensor_stats("rvq/packed_continuous", rvq_encoded.packed_continuous)
            if rvq_encoded.packed_quantized is not None:
                self._log_tensor_stats("rvq/packed_quantized", rvq_encoded.packed_quantized)
            self._log_tensor_stats("rvq/projected_latents", rvq_encoded.projected_latents)
            self.log("rvq/quantization_error", rvq_encoded.commitment_loss.detach(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
            self._log_rvq_rate_errors(rvq_encoded.packed_continuous)
            self.log("rvq/codebook_loss", rvq_encoded.codebook_loss.detach(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
            self.log("rvq/distill_loss", rvq_encoded.distill_loss.detach(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def _maybe_rebuild_data_index(self, trainer) -> int:
        if self.index_data_every_step is None:
            return 0
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step == self._last_index_data_step:
            return 0
        if step % self.index_data_every_step != 0:
            return 0

        self._last_index_data_step = step
        return _rebuild_dataset_indexes(getattr(trainer, "datamodule", None))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._maybe_log_rvq_rank_sync(self.trainer)
        rebuilt = self._maybe_rebuild_data_index(self.trainer)
        if rebuilt:
            self.log(
                "data/reindexed_datasets",
                float(rebuilt),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )


class MosslandCodecTrainingCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 2,
        demo_every: int = 1000,
        sample_rate: int = 48000,
        use_ema: bool = True,
        silence_seconds: float = 0.25,
        demo_n_quantizers: int | None = None,
        demo_n_quantizers_choices: list[int] | tuple[int, ...] | None = None,
        demo_all_ranks: bool = True,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = demo_num
        self.demo_every = demo_every
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.silence_seconds = silence_seconds
        self.demo_n_quantizers = demo_n_quantizers
        self.demo_n_quantizers_choices = demo_n_quantizers_choices
        self.demo_all_ranks = bool(demo_all_ranks)
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

    def _set_eval(self, model):
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        return was_training

    def _restore_training(self, model, was_training: bool):
        if hasattr(model, "train"):
            model.train(was_training)

    def _demo_quantizer_rates(self, model) -> list[int | None]:
        if getattr(model, "rvq", None) is None:
            return [None]
        if self.demo_n_quantizers is not None:
            return [int(self.demo_n_quantizers)]
        if self.demo_n_quantizers_choices is not None:
            rates = [int(value) for value in self.demo_n_quantizers_choices]
        else:
            max_quantizers = getattr(model, "rvq_num_quantizers", None)
            if max_quantizers is not None and int(max_quantizers) <= 32:
                rates = [8, 16, 32]
            else:
                rates = [16, 32, 64, 128]
        max_quantizers = getattr(model, "rvq_num_quantizers", None)
        if max_quantizers is not None:
            rates = [value for value in rates if 1 <= value <= int(max_quantizers)]
        rvq = getattr(model, "rvq", None)
        if rvq is not None and not model.training:
            initialized_count = rvq.initialized_codebook_count()
            rates = [value for value in rates if value <= initialized_count]
        return rates

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None:
            return
        step = int(trainer.global_step)
        should_run_demo = step == 1 or (
            self.demo_every > 0 and step > 0 and step % self.demo_every == 0
        )
        if not should_run_demo or self.last_demo_step == step:
            return
        self.last_demo_step = step
        global_rank = int(getattr(trainer, "global_rank", 0))
        if global_rank == 0:
            print(f"[demo_callback] saving demo at step={step}", flush=True)
        if not self.demo_all_ranks and global_rank != 0:
            return

        payload = task = src = target = src_audio = generated = comparison = None
        generated_versions = quantized_versions = continuous_generated = None
        src_item = target_item = generated_item = None
        model_states = []
        try:
            self._clear_cuda_cache()
            demo_models = [("raw", module.model)]
            if self.use_ema and hasattr(module, "ema"):
                demo_models.append(("ema", module.ema.ema_model))
            for _, demo_model in demo_models:
                model_states.append((demo_model, self._set_eval(demo_model)))

            os.makedirs(self.demo_dir, exist_ok=True)
            payload, _ = batch
            task = MosslandTaskBatch.from_payload(payload)
            src = task.src[: self.demo_num]
            target = task.target[: self.demo_num]
            demo_count = src.shape[0]
            demo_task_id = _slice_label(task.task_id, demo_count)
            target = module.model.prepare_audio_batch(target).detach().cpu()

            for model_label, demo_model in demo_models:
                quantized_versions = []
                src_audio = None
                for n_quantizers in self._demo_quantizer_rates(demo_model):
                    src_audio, quantized_generated = demo_model.generate_waveform(
                        src,
                        task_id=demo_task_id,
                        dont_quantize=False,
                        n_quantizers=n_quantizers,
                    )
                    label = (
                        "quantized"
                        if n_quantizers is None
                        else f"quantized{int(n_quantizers)}"
                    )
                    quantized_versions.append((label, quantized_generated))
                continuous_src_audio, continuous_generated = demo_model.generate_waveform(
                    src,
                    task_id=demo_task_id,
                    dont_quantize=True,
                )
                if src_audio is None:
                    src_audio = continuous_src_audio
                generated_versions = tuple(quantized_versions) + (
                    ("continuous", continuous_generated),
                )
                for idx, (src_item, target_item) in enumerate(zip(src_audio, target)):
                    task_id = _label_at(demo_task_id, idx)
                    base = f"{trainer.global_step}_{idx}_{task_id}_rank{trainer.global_rank}"
                    ordered_segments = [src_item, target_item]
                    ordered_labels = []
                    for mode, generated in generated_versions:
                        ordered_segments.append(generated[idx])
                        ordered_labels.append(mode)
                    comparison = self._concat_demo_audio(*ordered_segments)
                    torchaudio.save(
                        os.path.join(
                            self.demo_dir,
                            f"{base}_{model_label}_{'_'.join(ordered_labels)}_src_target_generated.mp3",
                        ),
                        comparison.float(),
                        self.sample_rate,
                        format="mp3",
                    )
        finally:
            for demo_model, was_training in model_states:
                self._restore_training(demo_model, was_training)
            payload = task = src = target = src_audio = generated = comparison = None
            generated_versions = quantized_versions = continuous_generated = None
            src_item = target_item = generated_item = None
            self._clear_cuda_cache()
