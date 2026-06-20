import json
import os
from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from ema_pytorch import EMA

from scripts.data.datasets import SampleDataset
from scripts.codec_common.training_base import (
    add_noise,
    get_sigma_continuous,
    pseudo_huber_loss,
)
from scripts.MOSS_Audio_Tokenizer_Flow.models import MossAudioTokenizerFlow


class MossAudioTokenizerFlowTrainingWrapper(pl.LightningModule):
    """Music2Latent training loop for the MOSS-transformer Flow model."""

    def __init__(
        self,
        model: MossAudioTokenizerFlow,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        consistency_total_step: int = 100000,
        consistency_step: float | None = None,
        consistency_loss_delta: float = 0.00054,
        optimizer_name: str = "adamw",
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.use_ema = use_ema
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.consistency_total_step = int(consistency_total_step)
        self.consistency_step = consistency_step
        self.consistency_loss_delta = float(consistency_loss_delta)
        if use_ema:
            self.ema = EMA(
                self.model,
                beta=ema_beta,
                power=3 / 4,
                update_every=1,
                update_after_step=2000,
            )

    def configure_optimizers(self):
        if self.optimizer_name == "radam":
            optimizer_cls = torch.optim.RAdam
        else:
            optimizer_cls = torch.optim.AdamW
        return [
            optimizer_cls(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        ]

    def _crop_batch(self, batch: torch.Tensor) -> torch.Tensor:
        hop = self.model.hop
        frames_per_token = max(1, int(self.model.spec_frames_per_token))
        cropped_frames = (batch.shape[-1] // hop) // frames_per_token
        cropped_length = cropped_frames * hop * frames_per_token
        return batch[..., :cropped_length]

    def _consistency_step_size(self):
        if self.consistency_step is not None:
            return float(self.consistency_step)
        progress = min(float(self.global_step) / max(1, self.consistency_total_step), 1.0)
        return 1.0 / (10.0 ** (1.0 + progress))

    def _get_sigma(self, position):
        return get_sigma_continuous(
            position,
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
        )

    def _consistency_loss(self, representation):
        batch_size = representation.shape[0]
        device = representation.device
        step_size = min(max(self._consistency_step_size(), 0.0), 1.0)
        high_pos = torch.rand(batch_size, device=device).clamp(min=step_size)
        low_pos = (high_pos - step_size).clamp(min=0.0)
        sigma_high = self._get_sigma(high_pos)
        sigma_low = self._get_sigma(low_pos)
        noise = torch.randn_like(representation)
        noisy_high = add_noise(representation, noise, sigma_high)
        noisy_low = add_noise(representation, noise, sigma_low)
        predicted = self.model(representation, noisy_high, sigma=sigma_high)
        with torch.no_grad():
            target = self.model(representation, noisy_low, sigma=sigma_low)
        weights = (1.0 / (sigma_high - sigma_low).clamp(min=1e-3)).reshape(
            batch_size, 1, 1, 1
        )
        loss = (
            pseudo_huber_loss(
                predicted.float(),
                target.float(),
                delta=self.consistency_loss_delta,
            )
            * weights.float()
        ).mean()
        return loss

    def training_step(self, batch, batch_idx):
        batch, info = batch
        batch = batch.to(next(self.model.parameters()).dtype)
        self.model.train()
        batch = self._crop_batch(batch)
        representation = self.model.to_representation_encoder(batch)
        loss = self._consistency_loss(representation)
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.use_ema:
            self.ema.update()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        dataset = getattr(getattr(self.trainer, "datamodule", None), "dataset", None)
        if (
            dataset is not None
            and hasattr(dataset, "refresh_filenames")
            and self.global_step % 3000 == 1
        ):
            dataset.refresh_filenames()
            self.log("data_nums", len(dataset.filenames))


class MossAudioTokenizerFlowCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 1,
        demo_every: int = 2000,
        sample_rate: int = 44100,
        use_ema: bool = True,
        silence_seconds: float = 0.25,
        denoising_steps: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = demo_num
        self.demo_every = demo_every
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.silence_seconds = silence_seconds
        self.denoising_steps = int(denoising_steps)
        self.last_demo_step = -1

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir is None:
            return
        if trainer.global_step % self.demo_every != 1:
            return
        if self.last_demo_step == trainer.global_step:
            return
        self.last_demo_step = trainer.global_step

        os.makedirs(self.demo_dir, exist_ok=True)
        model = module.ema.ema_model if self.use_ema and getattr(module, "use_ema", False) else module.model
        model.eval()
        batch, info = batch
        batch = module._crop_batch(batch[: self.demo_num]).to(next(model.parameters()).device)
        silence = int(round(self.silence_seconds * self.sample_rate))

        for i in range(batch.shape[0]):
            latent = model.encode(batch[i])
            recon = model.decode(
                latent,
                denoising_steps=self.denoising_steps,
                target_length=batch.shape[-1],
            )[0]
            source = self._as_audio_channels(batch[i].detach().cpu())
            recon = self._as_audio_channels(recon.detach().cpu())
            if recon.shape[0] != source.shape[0]:
                if recon.shape[0] == 1:
                    recon = recon.expand(source.shape[0], -1)
                else:
                    recon = recon[: source.shape[0]]
            spacer = torch.zeros(source.shape[0], max(0, silence))
            comparison = torch.cat([source, spacer, recon], dim=-1)
            base = os.path.join(
                self.demo_dir,
                f"{trainer.global_step}_{i}_rank{trainer.global_rank}",
            )
            torchaudio.save(
                f"{base}_source.wav",
                source.float().clamp(-1.0, 1.0),
                self.sample_rate,
            )
            torchaudio.save(
                f"{base}_recon.wav",
                recon.float().clamp(-1.0, 1.0),
                self.sample_rate,
            )
            torchaudio.save(
                f"{base}.wav",
                comparison.float().clamp(-1.0, 1.0),
                self.sample_rate,
            )

    @staticmethod
    def _as_audio_channels(audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            return audio.unsqueeze(0)
        if audio.ndim == 2:
            return audio
        raise ValueError(f"Expected audio [C,L] or [L], got {tuple(audio.shape)}")


class MossAudioTokenizerFlowFixedEvalCallback(pl.Callback):
    def __init__(
        self,
        eval_dir,
        dirs,
        index_file,
        eval_every: int = 0,
        max_items: int = 10,
        sample_size: int = 32768,
        sample_rate: int = 44100,
        num_channels: int = 1,
        use_ema: bool = False,
        audio_cache_dir=None,
        fail_on_error: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eval_dir = eval_dir
        self.dirs = dirs
        self.index_file = index_file
        self.eval_every = int(eval_every)
        self.max_items = int(max_items)
        self.sample_size = int(sample_size)
        self.sample_rate = int(sample_rate)
        self.num_channels = int(num_channels)
        self.use_ema = use_ema
        self.audio_cache_dir = audio_cache_dir
        self.fail_on_error = fail_on_error
        self.last_eval_step = -1

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.eval_every <= 0:
            return
        if trainer.global_step % self.eval_every != 1:
            return
        if self.last_eval_step == trainer.global_step:
            return
        self.last_eval_step = trainer.global_step
        if trainer.global_rank != 0:
            return
        try:
            self._run_eval(trainer, module)
        except Exception as exc:
            if self.fail_on_error:
                raise
            step_dir = Path(self.eval_dir) / f"step_{trainer.global_step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "eval_error.json").write_text(
                json.dumps({"error": repr(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _run_eval(self, trainer, module):
        step_dir = Path(self.eval_dir) / f"step_{trainer.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        dataset = SampleDataset(
            dirs=self.dirs,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            random_crop=False,
            num_channels=self.num_channels,
            index_file=self.index_file,
            audio_cache_dir=self.audio_cache_dir,
            no_channel_dim=self.num_channels == 1,
        )
        model = module.ema.ema_model if self.use_ema and getattr(module, "use_ema", False) else module.model
        model.eval()
        device = next(model.parameters()).device
        rows = []
        for idx in range(min(self.max_items, len(dataset.filenames))):
            source, info = dataset[idx]
            source = source.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
            source = module._crop_batch(source)
            latent = model.encode(source[0])
            recon = model.decode(latent, target_length=source.shape[-1])[0]
            source_cpu = source[0].detach().cpu().float()
            recon_cpu = recon.detach().cpu().float()
            torchaudio.save(step_dir / f"{idx}_source.wav", source_cpu, self.sample_rate)
            torchaudio.save(step_dir / f"{idx}_recon.wav", recon_cpu.clamp(-1.0, 1.0), self.sample_rate)
            rows.append(
                {
                    "idx": idx,
                    "mse": float(F.mse_loss(recon_cpu, source_cpu).item()),
                    "source_rms": float(source_cpu.square().mean().sqrt().item()),
                    "recon_rms": float(recon_cpu.square().mean().sqrt().item()),
                }
            )
        (step_dir / "results.json").write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
