from __future__ import annotations

from importlib import import_module

import torch

_codec_wrapper = import_module("scripts.mossland-codec.wrapper")
_codec_tasks = import_module("scripts.mossland-codec.tasks")
_codec_utils = import_module("scripts.mossland-codec.utils")

MosslandTaskBatch = _codec_tasks.MosslandTaskBatch
MosslandA2ATrainingCallback = _codec_wrapper.MosslandCodecTrainingCallback
_label_tuple = _codec_wrapper._label_tuple


class MosslandA2ATrainingWrapper(_codec_wrapper.MosslandCodecTrainingWrapper):
    """A2A separation wrapper that conditions on full source spectrograms."""

    def _consistency_loss(
        self,
        target_representation: torch.Tensor,
        src_tokens: torch.Tensor,
        src_features: list[torch.Tensor],
        task_id: str,
    ):
        batch_size = target_representation.shape[0]
        sigma_low_left, sigma_high_left, step_size = self._sample_sigma_pair(
            batch_size,
            target_representation.device,
        )
        sigma_low_right, sigma_high_right, _ = self._sample_sigma_pair(
            batch_size,
            target_representation.device,
        )
        noise = torch.randn_like(target_representation)
        sigma_high = self._expand_half_sigmas(sigma_high_left, sigma_high_right)
        sigma_low = self._expand_half_sigmas(sigma_low_left, sigma_low_right)
        noisy_high = _codec_utils.add_noise(target_representation, noise, sigma_high)
        noisy_low = _codec_utils.add_noise(target_representation, noise, sigma_low)

        predicted = self.model.decoder_forward(
            noisy_high,
            src_tokens,
            features=src_features,
            sigma_left=sigma_high_left,
            sigma_right=sigma_high_right,
            output="both",
            task_id=task_id,
        )
        with torch.no_grad():
            target = self.model.decoder_forward(
                noisy_low,
                src_tokens,
                features=src_features,
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
            "sigma/step": torch.tensor(step_size, device=target_representation.device),
            "sigma/low_mean": sigma_low.mean().detach(),
            "sigma/high_mean": sigma_high.mean().detach(),
        }
        return loss, metrics, predicted, target

    def training_step(self, batch, batch_idx):
        payload, info = batch
        task = MosslandTaskBatch.from_payload(payload)

        src = self.model.prepare_audio_batch(task.src)
        target_audio = self.model.prepare_audio_batch(task.target)
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
        src_tokens, src_features = self.model.encode_source(src_representation)
        self._assert_finite("src_tokens", src_tokens, info)
        for feature_idx, feature in enumerate(src_features):
            self._assert_finite(f"src_features/{feature_idx}", feature, info)

        loss, metrics, predicted, target = self._consistency_loss(
            target_representation,
            src_tokens,
            src_features,
            task_id=task.task_id,
        )
        self._assert_finite("predicted", predicted, info)
        self._assert_finite("target", target, info)
        self._assert_finite("loss", loss, info)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
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
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=False)
        return loss
