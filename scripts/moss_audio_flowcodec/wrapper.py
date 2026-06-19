import os

import lightning as pl
import torch
import torchaudio
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import MossAudioFlowCodec


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: list[int] | tuple[int, ...] = (2048, 1024, 512, 256),
        hop_ratio: float = 0.25,
        log_weight: float = 1.0,
        mag_weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.hop_ratio = float(hop_ratio)
        self.log_weight = float(log_weight)
        self.mag_weight = float(mag_weight)
        self.eps = float(eps)

    def _stft_mag(self, audio: torch.Tensor, fft_size: int):
        audio = audio.float()
        batch, channels, samples = audio.shape
        flat = audio.reshape(batch * channels, samples)
        window = torch.hann_window(fft_size, device=audio.device, dtype=torch.float32)
        spec = torch.stft(
            flat,
            n_fft=fft_size,
            hop_length=max(1, int(fft_size * self.hop_ratio)),
            win_length=fft_size,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        loss = estimate.new_zeros(())
        for fft_size in self.fft_sizes:
            estimate_mag = self._stft_mag(estimate, fft_size)
            target_mag = self._stft_mag(target, fft_size)
            loss = loss + self.mag_weight * torch.nn.functional.l1_loss(
                estimate_mag, target_mag
            )
            loss = loss + self.log_weight * torch.nn.functional.l1_loss(
                torch.log(estimate_mag.clamp_min(self.eps)),
                torch.log(target_mag.clamp_min(self.eps)),
            )
        return loss / len(self.fft_sizes)


class MossAudioFlowCodecTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model: MossAudioFlowCodec,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 1000,
        lr_schedule_total_steps: int = 500000,
        waveform_loss_weight: float = 1.0,
        stft_loss_weight: float = 1.0,
        discrete_loss_weight: float = 1.0,
        continuous_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
        codebook_loss_weight: float = 1.0,
        train_n_quantizers: int | None = None,
        train_n_quantizers_choices: list[int] | tuple[int, ...] | None = None,
        fail_on_nonfinite: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.waveform_loss_weight = float(waveform_loss_weight)
        self.stft_loss_weight = float(stft_loss_weight)
        self.discrete_loss_weight = float(discrete_loss_weight)
        self.continuous_loss_weight = float(continuous_loss_weight)
        self.commitment_loss_weight = float(commitment_loss_weight)
        self.codebook_loss_weight = float(codebook_loss_weight)
        self.train_n_quantizers = train_n_quantizers
        self.train_n_quantizers_choices = (
            tuple(int(value) for value in train_n_quantizers_choices)
            if train_n_quantizers_choices is not None
            else None
        )
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        self.l1 = nn.L1Loss()
        self.stft_loss = MultiScaleSTFTLoss()

    def configure_optimizers(self):
        name = self.optimizer_name.lower()
        if name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif name == "radam":
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}")

        if self.lr_schedule == "constant":
            return optimizer
        if self.lr_schedule != "cosine_decay":
            raise ValueError(f"Unsupported lr_schedule={self.lr_schedule!r}")
        decay_steps = max(1, self.lr_schedule_total_steps - self.lr_warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0.0)
        if self.lr_warmup_steps <= 0:
            scheduler = cosine
        else:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.lr_warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.lr_warmup_steps],
            )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _prepare_audio(self, batch: torch.Tensor):
        if batch.ndim == 4:
            batch = batch.flatten(0, 1)
        if batch.ndim == 2:
            batch = batch[:, None, :]
        if batch.shape[1] != self.model.audio_channels:
            raise ValueError(
                f"expected {self.model.audio_channels} audio channels, got {batch.shape[1]}"
            )
        return batch.clamp(-1.0, 1.0)

    def _assert_finite(self, name: str, tensor: torch.Tensor):
        if torch.isfinite(tensor).all():
            return
        message = f"{name} contains non-finite values"
        if self.fail_on_nonfinite:
            raise FloatingPointError(message)
        print(message)

    def _branch_recon_loss(self, name: str, estimate: torch.Tensor, target: torch.Tensor):
        waveform = self.l1(estimate, target)
        stft = self.stft_loss(estimate, target)
        total = self.waveform_loss_weight * waveform + self.stft_loss_weight * stft
        self.log(f"loss/{name}_waveform", waveform.detach(), on_step=True, on_epoch=False)
        self.log(f"loss/{name}_stft", stft.detach(), on_step=True, on_epoch=False)
        return total

    def _sample_train_n_quantizers(self):
        if self.train_n_quantizers_choices:
            idx = torch.randint(
                0,
                len(self.train_n_quantizers_choices),
                (),
                device=self.device,
            ).item()
            return self.train_n_quantizers_choices[int(idx)]
        if self.train_n_quantizers is not None:
            return int(self.train_n_quantizers)
        return None

    def training_step(self, batch, batch_idx):
        audio, info = batch
        audio = self._prepare_audio(audio)
        self._assert_finite("audio", audio)
        n_quantizers = self._sample_train_n_quantizers()
        output = self.model(audio, n_quantizers=n_quantizers)
        target = audio[..., : output.audio_discrete.shape[-1]]

        discrete_recon = self._branch_recon_loss(
            "discrete", output.audio_discrete, target
        )
        continuous_recon = self._branch_recon_loss(
            "continuous", output.audio_continuous, target
        )
        vq_loss = (
            self.commitment_loss_weight * output.vq_commitment_loss
            + self.codebook_loss_weight * output.vq_codebook_loss
        )
        loss = (
            self.discrete_loss_weight * discrete_recon
            + self.continuous_loss_weight * continuous_recon
            + vq_loss
        )
        self._assert_finite("loss", loss)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("loss/discrete_recon", discrete_recon.detach(), on_step=True, on_epoch=False)
        self.log("loss/continuous_recon", continuous_recon.detach(), on_step=True, on_epoch=False)
        self.log("loss/vq", vq_loss.detach(), on_step=True, on_epoch=False)
        self.log("latent/token_rate", torch.tensor(self.model.token_rate, device=loss.device), on_step=True, on_epoch=False)
        self.log(
            "latent/n_quantizers",
            torch.tensor(
                n_quantizers or self.model.num_quantizers,
                device=loss.device,
                dtype=torch.float32,
            ),
            on_step=True,
            on_epoch=False,
        )
        self.log("latent/codes_std", output.codes.detach().float().std(), on_step=True, on_epoch=False)
        self.log("latent/continuous_std", output.continuous.detach().float().std(), on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def reconstruct_waveform(self, audio: torch.Tensor, model: MossAudioFlowCodec | None = None):
        model = model or self.model
        audio = self._prepare_audio(audio)
        output = model(audio)
        return (
            audio.detach().cpu(),
            output.audio_discrete.detach().cpu(),
            output.audio_continuous.detach().cpu(),
            output.codes.detach().cpu(),
            output.continuous.detach().cpu(),
        )


class MossAudioFlowCodecCallback(pl.Callback):
    def __init__(
        self,
        demo_dir,
        demo_num: int = 2,
        demo_every: int = 1000,
        sample_rate: int = 48000,
        silence_seconds: float = 0.25,
        save_separate_files: bool = False,
        save_tokens: bool = False,
    ):
        super().__init__()
        self.demo_dir = demo_dir
        self.demo_num = int(demo_num)
        self.demo_every = int(demo_every)
        self.sample_rate = int(sample_rate)
        self.silence_seconds = float(silence_seconds)
        self.save_separate_files = bool(save_separate_files)
        self.save_tokens = bool(save_tokens)
        self.last_demo_step = -1

    def _concat_audio(self, *segments: torch.Tensor):
        silence_samples = int(self.sample_rate * self.silence_seconds)
        silence = segments[0].new_zeros(segments[0].shape[:-1] + (silence_samples,))
        pieces = []
        for segment in segments:
            if pieces and silence_samples > 0:
                pieces.append(silence)
            pieces.append(segment)
        return torch.cat(pieces, dim=-1)

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
        originals, discrete, continuous, codes, continuous_tokens = module.reconstruct_waveform(
            audio[: self.demo_num]
        )
        for idx, (original, disc, cont) in enumerate(zip(originals, discrete, continuous)):
            base = f"{trainer.global_step}_{idx}_rank{trainer.global_rank}"
            if self.save_separate_files:
                torchaudio.save(
                    os.path.join(self.demo_dir, f"{base}_input.wav"),
                    original.float(),
                    self.sample_rate,
                )
                torchaudio.save(
                    os.path.join(self.demo_dir, f"{base}_discrete.wav"),
                    disc.float(),
                    self.sample_rate,
                )
                torchaudio.save(
                    os.path.join(self.demo_dir, f"{base}_continuous.wav"),
                    cont.float(),
                    self.sample_rate,
                )
            torchaudio.save(
                os.path.join(self.demo_dir, f"{base}_input_discrete_continuous.wav"),
                self._concat_audio(original, disc, cont).float(),
                self.sample_rate,
            )
        if self.save_tokens:
            torch.save(
                {"codes": codes, "continuous": continuous_tokens},
                os.path.join(
                    self.demo_dir,
                    f"{trainer.global_step}_rank{trainer.global_rank}_tokens.pt",
                ),
            )
