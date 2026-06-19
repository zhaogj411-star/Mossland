import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class PatchedPretransform(nn.Module):
    def __init__(self, channels, patch_size, **kwargs):
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.downsampling_ratio = self.patch_size
        self.io_channels = self.channels
        self.encoded_channels = self.channels * self.patch_size
        self.enable_grad = False

    def _pad(self, x):
        pad_len = (self.patch_size - (x.shape[-1] % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x

    def encode(self, x):
        x = self._pad(x)
        return rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)

    def decode(self, x):
        return rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size, c=self.channels)


class Music2LatentSTFTPretransform(nn.Module):
    def __init__(
        self,
        channels,
        patch_size,
        alpha_rescale=0.65,
        beta_rescale=0.34,
        hop_size=None,
        fac=4,
        clamp_output=True,
        max_normalized_magnitude=100.0,
        phase_eps=1e-8,
        **kwargs,
    ):
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.alpha_rescale = float(alpha_rescale)
        self.beta_rescale = float(beta_rescale)
        self.hop_size = int(hop_size) if hop_size is not None else self.patch_size
        self.fac = int(fac)
        self.clamp_output = bool(clamp_output)
        self.max_normalized_magnitude = float(max_normalized_magnitude)
        self.phase_eps = float(phase_eps)
        self.downsampling_ratio = self.hop_size
        self.io_channels = self.channels
        self.encoded_channels = self.channels * 4 * self.hop_size
        self.enable_grad = False

        if self.fac <= 0:
            raise ValueError("fac must be positive")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be positive")
        if self.max_normalized_magnitude <= 0:
            raise ValueError("max_normalized_magnitude must be positive")
        if self.phase_eps <= 0:
            raise ValueError("phase_eps must be positive")
        n_fft = self.fac * self.hop_size
        bins = 2 * self.hop_size
        k = torch.arange(bins, dtype=torch.float32)[:, None]
        n = torch.arange(n_fft, dtype=torch.float32)[None, :]
        phase = 2.0 * torch.pi * k * n / float(n_fft)
        self.register_buffer("dft_cos", torch.cos(phase), persistent=False)
        self.register_buffer("dft_sin", torch.sin(phase), persistent=False)
        inv_weights = torch.full((bins,), 2.0, dtype=torch.float32)
        inv_weights[0] = 1.0
        if bins > n_fft // 2:
            inv_weights[n_fft // 2] = 1.0
        self.register_buffer("idft_weights", inv_weights, persistent=False)

    def _window(self, device, dtype):
        return torch.hann_window(self.fac * self.hop_size, device=device, dtype=dtype)

    def _normalize_realimag(self, real, imag):
        magnitude = torch.sqrt(real.square() + imag.square() + self.phase_eps * self.phase_eps)
        output_magnitude = self.beta_rescale * (magnitude ** self.alpha_rescale)
        denom = magnitude.clamp_min(self.phase_eps)
        return output_magnitude * real / denom, output_magnitude * imag / denom

    def _denormalize_realimag(self, real, imag):
        real = torch.nan_to_num(
            real.float(),
            nan=0.0,
            posinf=self.max_normalized_magnitude,
            neginf=-self.max_normalized_magnitude,
        ).clamp(-self.max_normalized_magnitude, self.max_normalized_magnitude)
        imag = torch.nan_to_num(
            imag.float(),
            nan=0.0,
            posinf=self.max_normalized_magnitude,
            neginf=-self.max_normalized_magnitude,
        ).clamp(-self.max_normalized_magnitude, self.max_normalized_magnitude)
        normalized_magnitude = torch.sqrt(real.square() + imag.square() + self.phase_eps * self.phase_eps)
        raw_magnitude = normalized_magnitude / self.beta_rescale
        clipped_magnitude = raw_magnitude.clamp(max=self.max_normalized_magnitude)
        output_magnitude = clipped_magnitude ** (1.0 / self.alpha_rescale)
        denom = normalized_magnitude.clamp_min(self.phase_eps)
        return output_magnitude * real / denom, output_magnitude * imag / denom

    def _frame(self, x):
        frame_length = self.fac * self.hop_size
        pad = (frame_length - self.hop_size) // 2
        if x.shape[-1] <= 1:
            x = F.pad(x, (pad, pad), "constant", 0.0)
        else:
            x = F.pad(x, (pad, pad), "reflect")
        return x.unfold(-1, frame_length, self.hop_size)

    def _stft(self, x):
        batch, channels, samples = x.shape
        x = x.reshape(batch * channels, samples)
        x = x.float()
        window = self._window(x.device, x.dtype)
        frames = self._frame(x) * window
        cos = self.dft_cos.to(device=frames.device, dtype=frames.dtype)
        sin = self.dft_sin.to(device=frames.device, dtype=frames.dtype)
        real = torch.matmul(frames, cos.t())
        imag = -torch.matmul(frames, sin.t())
        bins = 2 * self.hop_size
        real = real.transpose(-1, -2).reshape(batch, channels, bins, -1)
        imag = imag.transpose(-1, -2).reshape(batch, channels, bins, -1)
        return real, imag

    def _inverse_stft_window(self, forward_window):
        frame_length = self.fac * self.hop_size
        denom = forward_window**2
        overlaps = -(-frame_length // self.hop_size)
        denom = F.pad(denom, (0, overlaps * self.hop_size - frame_length))
        denom = denom.reshape(overlaps, self.hop_size)
        denom = denom.sum(0, keepdim=True)
        denom = denom.tile(overlaps, 1).reshape(overlaps * self.hop_size)
        return forward_window / denom[:frame_length]

    def _overlap_and_add(self, signal):
        frame_step = self.hop_size
        frame_length = signal.shape[-1]
        frames = signal.shape[-2]
        output_length = frame_length + frame_step * (frames - 1)
        segments = -(-frame_length // frame_step)
        signal = F.pad(signal, (0, segments * frame_step - frame_length, 0, segments))
        signal = signal.reshape(*signal.shape[:-2], frames + segments, segments, frame_step)
        signal = signal.transpose(-3, -2)
        signal = signal.reshape(*signal.shape[:-3], (frames + segments) * segments, frame_step)
        signal = signal[..., : (frames + segments - 1) * segments, :]
        signal = signal.reshape(*signal.shape[:-2], segments, frames + segments - 1, frame_step)
        signal = signal.sum(-3)
        signal = signal.reshape(*signal.shape[:-2], (frames + segments - 1) * frame_step)
        return signal[..., :output_length]

    def _istft(self, real, imag):
        batch, channels, bins, frames = real.shape
        expected_bins = 2 * self.hop_size
        if bins < expected_bins:
            real = F.pad(real, (0, 0, 0, expected_bins - bins))
            imag = F.pad(imag, (0, 0, 0, expected_bins - bins))
        elif bins > expected_bins:
            real = real[:, :, :expected_bins, :]
            imag = imag[:, :, :expected_bins, :]

        real = real.float()
        imag = imag.float()
        cos = self.dft_cos.to(device=real.device, dtype=real.dtype)
        sin = self.dft_sin.to(device=real.device, dtype=real.dtype)
        weights = self.idft_weights.to(device=real.device, dtype=real.dtype)
        x = (
            torch.einsum("bckt,kn,k->bctn", real, cos, weights)
            - torch.einsum("bckt,kn,k->bctn", imag, sin, weights)
        ) / float(self.fac * self.hop_size)
        x = x.reshape(batch * channels, frames, self.fac * self.hop_size).transpose(-1, -2)
        window = self._inverse_stft_window(self._window(x.device, x.dtype))
        x = x * window[:, None]
        reconstructed = self._overlap_and_add(x.transpose(-1, -2))
        pad = (self.fac * self.hop_size - self.hop_size) // 2
        if reconstructed.shape[-1] > 2 * pad:
            reconstructed = reconstructed[..., pad:-pad]
        reconstructed = reconstructed.reshape(batch, channels, -1)
        if self.clamp_output:
            reconstructed = reconstructed.clamp(-1.0, 1.0)
        return reconstructed

    def encode(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected audio shape [batch, channels, samples], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        real, imag = self._stft(x)
        real, imag = self._normalize_realimag(real, imag)
        realimag = torch.stack((real, imag), dim=2)
        return rearrange(realimag, "b c ri f l -> b (c ri f) l")

    def decode(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected representation shape [batch, channels, frames], got {tuple(x.shape)}")
        realimag = rearrange(
            x,
            "b (c ri f) l -> b c ri f l",
            c=self.channels,
            ri=2,
            f=2 * self.hop_size,
        )
        real, imag = self._denormalize_realimag(realimag[:, :, 0], realimag[:, :, 1])
        return self._istft(real, imag)
