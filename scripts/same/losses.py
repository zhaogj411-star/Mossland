import math
from math import comb
from typing import Sequence

import numpy as np
import scipy.signal
import torch
from torch import nn
from torch.nn import functional as F


def _power_sine_tight(n_fft: int, hop_length: int, device=None, dtype=None):
    ratio = n_fft // hop_length
    power = ratio - 1
    if ratio < 2 or power < 1:
        raise ValueError("n_fft must be at least 2x hop_length for a tight window")
    samples = torch.arange(n_fft, device=device, dtype=dtype)
    sine = torch.sin(math.pi * samples / n_fft)
    mean_square = comb(2 * power, power) / (2.0 ** (2 * power))
    amplitude = 1.0 / math.sqrt(ratio * mean_square)
    return (sine**power) * amplitude


def _hermitian_sqrt(n_fft: int, device=None, dtype=None):
    bins = n_fft // 2 + 1
    scale = torch.ones(bins, device=device, dtype=dtype)
    if n_fft % 2 == 0:
        if bins > 2:
            scale[1:-1] = math.sqrt(2.0)
    elif bins > 1:
        scale[1:] = math.sqrt(2.0)
    return scale


def tight_one_sided_complex_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop: int | None = None,
    center: bool = False,
    demod_mode: str | None = None,
):
    if n_fft % 2 != 0:
        raise ValueError("n_fft must be even")
    if hop is None:
        hop = n_fft // 2
    window = _power_sine_tight(n_fft, hop, device=audio.device, dtype=audio.dtype)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=True,
        onesided=True,
        return_complex=True,
    )
    spec = spec * _hermitian_sqrt(n_fft, device=spec.device, dtype=spec.real.dtype)[:, None]
    freq_bins, frames = spec.shape[-2:]
    if demod_mode == "sign":
        k_odd = (torch.arange(freq_bins, device=spec.device, dtype=torch.int8) & 1).view(freq_bins, 1)
        m_odd = (torch.arange(frames, device=spec.device, dtype=torch.int8) & 1).view(1, frames)
        parity = (k_odd & m_odd).to(spec.real.dtype)
        spec = spec * (1.0 - 2.0 * parity)
    elif demod_mode == "rot":
        k = torch.arange(freq_bins, device=spec.device, dtype=torch.float32).view(freq_bins, 1)
        m = torch.arange(frames, device=spec.device, dtype=torch.float32).view(1, frames)
        phase = torch.remainder((2.0 * math.pi) * (k * (hop / float(n_fft))) * m, 2.0 * math.pi)
        spec = spec * torch.complex(torch.cos(phase), -torch.sin(phase)).to(spec.dtype)
    return spec


def normalized_complex_distance_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5):
    numerator = (pred - target).abs() ** 2
    scale = numerator.std(dim=[-1, -2], keepdim=True).detach().clamp(min=eps)
    return torch.log(numerator / scale + 1.0).mean()


def if_gd_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3, w_floor: float = 1e-3):
    pred_t = pred[..., :, 1:] * torch.conj(pred[..., :, :-1])
    target_t = target[..., :, 1:] * torch.conj(target[..., :, :-1])
    denom_pred_t = (pred[..., :, 1:].abs() * pred[..., :, :-1].abs()).clamp_min(eps)
    denom_target_t = (target[..., :, 1:].abs() * target[..., :, :-1].abs()).clamp_min(eps)
    unit_pred_t = pred_t / denom_pred_t
    unit_target_t = target_t / denom_target_t
    weight_t = torch.sqrt(denom_pred_t * denom_target_t).clamp_min(w_floor).detach()
    weight_t = weight_t / weight_t.mean().clamp_min(1e-7)
    loss_t = (1.0 - (unit_pred_t * torch.conj(unit_target_t)).real) * weight_t

    pred_f = pred[..., 1:, :] * torch.conj(pred[..., :-1, :])
    target_f = target[..., 1:, :] * torch.conj(target[..., :-1, :])
    denom_pred_f = (pred[..., 1:, :].abs() * pred[..., :-1, :].abs()).clamp_min(eps)
    denom_target_f = (target[..., 1:, :].abs() * target[..., :-1, :].abs()).clamp_min(eps)
    unit_pred_f = pred_f / denom_pred_f
    unit_target_f = target_f / denom_target_f
    weight_f = torch.sqrt(denom_pred_f * denom_target_f).clamp_min(w_floor).detach()
    weight_f = weight_f / weight_f.mean().clamp_min(1e-7)
    loss_f = (1.0 - (unit_pred_f * torch.conj(unit_target_f)).real) * weight_f
    return loss_t.mean() + loss_f.mean()


class FIRFilter(nn.Module):
    def __init__(self, filter_type: str = "kw", coef: float = 0.85, fs: int = 44100, ntaps: int = 257):
        super().__init__()
        if ntaps % 2 == 0:
            raise ValueError("ntaps must be odd")
        if filter_type == "hp":
            taps = np.array([1.0, -float(coef)], dtype=np.float64)
        elif filter_type == "fd":
            taps = np.array([1.0, 0.0, -float(coef)], dtype=np.float64)
        elif filter_type == "kw":
            taps = self._design_k_weighting_fir(fs=fs, ntaps=ntaps)
        else:
            raise ValueError(f"unsupported filter_type={filter_type!r}")
        self.register_buffer("kernel", torch.from_numpy(taps.astype(np.float32))[None, None, :], persistent=False)

    @staticmethod
    def _design_k_weighting_fir(fs: int, ntaps: int):
        f_hp, q_hp = 38.135, 0.5
        w_hp = 2 * np.pi * f_hp
        num_hp = [1, 0, 0]
        den_hp = [1, w_hp / q_hp, w_hp**2]
        f_shelf, q_shelf, gain_db = 1681.974, 1.69, 4.0
        k = 10 ** (gain_db / 20.0)
        w_s = 2 * np.pi * f_shelf
        num_shelf = [k, (k * w_s) / q_shelf, w_s**2]
        den_shelf = [1, w_s / q_shelf, w_s**2]
        b, a = scipy.signal.bilinear(np.polymul(num_hp, num_shelf), np.polymul(den_hp, den_shelf), fs=fs)
        freq = np.linspace(0.0, fs / 2.0, num=8193, endpoint=True)
        _, response = scipy.signal.freqz(b, a, worN=freq, fs=fs)
        taps = scipy.signal.firwin2(ntaps, freq, np.abs(response), fs=fs)
        ref = 1000.0
        n = np.arange(len(taps))
        h_ref = np.abs(np.sum(taps * np.exp(-1j * (2 * np.pi * ref / fs) * n)))
        return taps / h_ref if h_ref > 0 else taps

    def forward(self, audio: torch.Tensor):
        batch, channels, samples = audio.shape
        flat = audio.reshape(batch * channels, 1, samples)
        kernel = self.kernel.to(device=audio.device, dtype=audio.dtype)
        pad = (kernel.shape[-1] - 1) // 2
        flat = F.pad(flat, (pad, pad), mode="reflect")
        return F.conv1d(flat, kernel).reshape(batch, channels, -1)


class SpectralContrastLoss(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor):
        numerator = torch.norm(target_mag.float() - pred_mag.float(), p="fro", dim=[-1, -2])
        denominator = torch.norm(target_mag.float() + pred_mag.float(), p="fro", dim=[-1, -2]).clamp_min(self.eps)
        return (numerator / denominator).unsqueeze(-1).unsqueeze(-1)


class AdaptiveLogMagnitudeLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.distance = nn.L1Loss(reduction=reduction)

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor):
        pred_std = pred_mag.std(dim=[-1, -2], keepdim=True).detach().clamp(min=1e-4)
        target_std = target_mag.std(dim=[-1, -2], keepdim=True).detach().clamp(min=1e-4)
        log_eps = torch.sqrt(pred_std**2 + target_std**2)
        return self.distance(torch.log(pred_mag / log_eps + 1.0), torch.log(target_mag / log_eps + 1.0))


class STFTLoss(nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int | None = None,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: int | None = None,
        perceptual_weighting: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_length = int(win_length or fft_size)
        self.w_sc = float(w_sc)
        self.w_log_mag = float(w_log_mag)
        self.w_lin_mag = float(w_lin_mag)
        self.w_phs = float(w_phs)
        self.eps = float(eps)
        self.spectral_contrast = SpectralContrastLoss()
        self.log_mag = AdaptiveLogMagnitudeLoss()
        self.lin_mag = nn.L1Loss()
        self.prefilter = FIRFilter("kw", fs=int(sample_rate)) if perceptual_weighting else None

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        estimate = estimate.float()
        target = target.float()
        if self.prefilter is not None:
            self.prefilter.to(estimate.device)
            estimate = self.prefilter(estimate)
            target = self.prefilter(target)
        estimate_spec = tight_one_sided_complex_stft(estimate.flatten(0, 1), self.fft_size, hop=self.hop_size, center=False)
        target_spec = tight_one_sided_complex_stft(target.flatten(0, 1), self.fft_size, hop=self.hop_size, center=False)
        estimate_mag = estimate_spec.abs().clamp_min(self.eps)
        target_mag = target_spec.abs().clamp_min(self.eps)
        loss = estimate.new_zeros(())
        if self.w_sc:
            loss = loss + self.w_sc * self.spectral_contrast(estimate_mag, target_mag).mean()
        if self.w_log_mag:
            loss = loss + self.w_log_mag * self.log_mag(estimate_mag, target_mag)
        if self.w_lin_mag:
            loss = loss + self.w_lin_mag * self.lin_mag(estimate_mag, target_mag)
        if self.w_phs:
            phase_loss = normalized_complex_distance_loss(estimate_spec, target_spec) + if_gd_loss(estimate_spec, target_spec)
            loss = loss + self.w_phs * phase_loss
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64, 32),
        hop_sizes: Sequence[int] | None = None,
        win_lengths: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        if hop_sizes is None:
            hop_sizes = [int(size * 0.25) for size in fft_sizes]
        if win_lengths is None:
            win_lengths = list(fft_sizes)
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have same length")
        self.losses = nn.ModuleList(
            [
                STFTLoss(fft_size=fft, hop_size=hop, win_length=win, **kwargs)
                for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        total = estimate.new_zeros(())
        used = 0
        for loss in self.losses:
            if estimate.shape[-1] <= loss.fft_size:
                continue
            total = total + loss(estimate, target)
            used += 1
        return total / max(used, 1)


class SumAndDifferenceSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64, 32),
        hop_sizes: Sequence[int] | None = None,
        win_lengths: Sequence[int] | None = None,
        w_sum: float = 1.0,
        w_diff: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.w_sum = float(w_sum)
        self.w_diff = float(w_diff)
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, **kwargs)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        if estimate.shape[1] != 2 or target.shape[1] != 2:
            raise ValueError("SumAndDifferenceSTFTLoss expects stereo tensors")
        estimate_sum = (estimate[:, 0, :] + estimate[:, 1, :]).unsqueeze(1)
        estimate_diff = (estimate[:, 0, :] - estimate[:, 1, :]).unsqueeze(1)
        target_sum = (target[:, 0, :] + target[:, 1, :]).unsqueeze(1)
        target_diff = (target[:, 0, :] - target[:, 1, :]).unsqueeze(1)
        sum_loss = self.mrstft(estimate_sum, target_sum)
        diff_loss = self.mrstft(estimate_diff, target_diff)
        return (self.w_sum * sum_loss + self.w_diff * diff_loss) / 2.0
