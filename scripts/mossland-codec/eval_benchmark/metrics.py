from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Iterable

import torch
import torchaudio.functional as AF

from .audio_io import align_length


EPS = 1e-8


def _safe_db(value: torch.Tensor | float) -> float:
    if not torch.is_tensor(value):
        value = torch.tensor(float(value))
    return float(10.0 * torch.log10(value.clamp_min(EPS)))


def snr(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    prediction, reference = align_length(prediction, reference)
    noise = reference - prediction
    return _safe_db(reference.pow(2).mean() / noise.pow(2).mean().clamp_min(EPS))


def si_sdr(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    prediction, reference = align_length(prediction, reference)
    pred = prediction.reshape(-1) - prediction.mean()
    ref = reference.reshape(-1) - reference.mean()
    scale = torch.dot(pred, ref) / torch.dot(ref, ref).clamp_min(EPS)
    target = scale * ref
    error = pred - target
    return _safe_db(target.pow(2).mean() / error.pow(2).mean().clamp_min(EPS))


def stft_magnitude(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return spec.abs().clamp_min(EPS)


def log_spectral_distance(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    prediction, reference = align_length(prediction, reference)
    pred_mag = stft_magnitude(prediction, n_fft=n_fft, hop_length=hop_length)
    ref_mag = stft_magnitude(reference, n_fft=n_fft, hop_length=hop_length)
    freqs = torch.linspace(0.0, sample_rate / 2.0, pred_mag.shape[-2], device=pred_mag.device)
    mask = freqs >= float(freq_min)
    if freq_max is not None:
        mask = mask & (freqs <= float(freq_max))
    if not bool(mask.any()):
        return float("nan")
    diff = 20.0 * (pred_mag[:, mask, :].log10() - ref_mag[:, mask, :].log10())
    return float(torch.sqrt(diff.pow(2).mean()).item())


def multi_resolution_stft_distance(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    fft_sizes: Iterable[int] = (512, 1024, 2048),
) -> float:
    prediction, reference = align_length(prediction, reference)
    values = []
    for n_fft in fft_sizes:
        pred_mag = stft_magnitude(prediction, n_fft=n_fft, hop_length=n_fft // 4)
        ref_mag = stft_magnitude(reference, n_fft=n_fft, hop_length=n_fft // 4)
        values.append((pred_mag - ref_mag).abs().mean() / ref_mag.mean().clamp_min(EPS))
    return float(torch.stack(values).mean().item())


def high_frequency_energy_ratio(audio: torch.Tensor, sample_rate: int, cutoff_hz: float) -> float:
    mag = stft_magnitude(audio)
    freqs = torch.linspace(0.0, sample_rate / 2.0, mag.shape[-2], device=mag.device)
    high = mag[:, freqs >= cutoff_hz, :].pow(2).sum()
    total = mag.pow(2).sum().clamp_min(EPS)
    return float((high / total).item())


def stereo_width(audio: torch.Tensor) -> float:
    if audio.shape[0] < 2:
        return 0.0
    left, right = audio[0], audio[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return float((side.pow(2).mean() / mid.pow(2).mean().clamp_min(EPS)).sqrt().item())


def channel_correlation(audio: torch.Tensor) -> float:
    if audio.shape[0] < 2:
        return 1.0
    left = audio[0] - audio[0].mean()
    right = audio[1] - audio[1].mean()
    denom = left.norm() * right.norm()
    if float(denom) <= EPS:
        return 0.0
    return float(torch.dot(left, right).div(denom).clamp(-1.0, 1.0).item())


def mono_fold_down(audio: torch.Tensor) -> torch.Tensor:
    if audio.shape[0] == 1:
        return audio
    return audio.mean(dim=0, keepdim=True)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def mir_eval_bss_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compute mir_eval BSS metrics for a single estimated source.

    This is useful as a lightweight source-separation diagnostic. It is not a
    replacement for SiSEC 2018's museval/BSS Eval v4 protocol on full MUSDB
    tracks, but it exposes the same SDR/SIR/SAR/ISR vocabulary when the
    dependency is available.
    """
    from mir_eval import separation

    prediction, reference = align_length(prediction.detach().cpu().float(), reference.detach().cpu().float())
    if prediction.shape[0] >= 2 and reference.shape[0] >= 2:
        ref = reference[:2].T.unsqueeze(0).numpy()
        est = prediction[:2].T.unsqueeze(0).numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sdr, isr, sir, sar, _ = separation.bss_eval_images(ref, est, compute_permutation=False)
        return {
            "mir_eval_bss_images_sdr_db": _finite_or_nan(sdr[0]),
            "mir_eval_bss_images_isr_db": _finite_or_nan(isr[0]),
            "mir_eval_bss_images_sir_db": _finite_or_nan(sir[0]),
            "mir_eval_bss_images_sar_db": _finite_or_nan(sar[0]),
        }

    ref = mono_fold_down(reference).numpy()
    est = mono_fold_down(prediction).numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sdr, sir, sar, _ = separation.bss_eval_sources(ref, est, compute_permutation=False)
    return {
        "mir_eval_bss_sources_sdr_db": _finite_or_nan(sdr[0]),
        "mir_eval_bss_sources_sir_db": _finite_or_nan(sir[0]),
        "mir_eval_bss_sources_sar_db": _finite_or_nan(sar[0]),
    }


def mel_frechet_embedding(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if audio.shape[0] > 1:
        audio = mono_fold_down(audio)
    mel = AF.melscale_fbanks(
        n_freqs=1025,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=64,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="htk",
    ).to(audio.device, audio.dtype)
    mag = stft_magnitude(audio, n_fft=2048, hop_length=512).squeeze(0)
    mel_spec = torch.matmul(mel.T, mag).clamp_min(EPS).log()
    return torch.cat([mel_spec.mean(dim=-1), mel_spec.std(dim=-1)], dim=0)


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    if features.shape[0] <= 1:
        return torch.zeros(features.shape[1], features.shape[1], dtype=features.dtype)
    return centered.T @ centered / (features.shape[0] - 1)


def _sqrtm_psd(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    values = values.clamp_min(0.0).sqrt()
    return (vectors * values.unsqueeze(0)) @ vectors.T


def frechet_distance(reference_features: torch.Tensor, prediction_features: torch.Tensor) -> float:
    ref = reference_features.double()
    pred = prediction_features.double()
    mu_ref = ref.mean(dim=0)
    mu_pred = pred.mean(dim=0)
    cov_ref = _covariance(ref)
    cov_pred = _covariance(pred)
    sqrt_ref = _sqrtm_psd(cov_ref)
    covmean = _sqrtm_psd(sqrt_ref @ cov_pred @ sqrt_ref)
    distance = (mu_ref - mu_pred).pow(2).sum() + torch.trace(cov_ref + cov_pred - 2.0 * covmean)
    return float(distance.clamp_min(0.0).item())


def pair_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    task_id: str,
    low_sample_rate: int | None = None,
    source: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    prediction, reference = align_length(prediction.float(), reference.float())
    if device is not None:
        prediction = prediction.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)
        if source is not None:
            source = source.to(device, non_blocking=True)
    values = {
        "snr_db": snr(prediction, reference),
        "si_sdr_db": si_sdr(prediction, reference),
        "lsd": log_spectral_distance(prediction, reference, sample_rate),
        "mrstft": multi_resolution_stft_distance(prediction, reference),
    }
    if task_id == "super_resolution" and low_sample_rate:
        cutoff = low_sample_rate / 2.0
        values["lsd_lf"] = log_spectral_distance(prediction, reference, sample_rate, 0.0, cutoff)
        values["lsd_hf"] = log_spectral_distance(prediction, reference, sample_rate, cutoff, None)
        values["hf_energy_ratio"] = high_frequency_energy_ratio(prediction, sample_rate, cutoff)
    if task_id == "mono_to_stereo":
        source_mono = mono_fold_down(source if source is not None else reference)
        pred_mono = mono_fold_down(prediction)
        values["fold_down_si_sdr_db"] = si_sdr(pred_mono, source_mono)
        values["stereo_width"] = stereo_width(prediction)
        values["channel_correlation"] = channel_correlation(prediction)
    if task_id.startswith("separate_"):
        try:
            values.update(mir_eval_bss_metrics(prediction, reference))
        except Exception as exc:
            values["mir_eval_bss_error"] = str(exc)
    return values


def aggregate_metric_rows(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = row["task_id"]
        if row.get("low_sample_rate"):
            key = f"{key}@{row['low_sample_rate']}"
        buckets[key].append(row["metrics"])

    summary: dict[str, dict[str, float]] = {}
    for key, metrics_rows in buckets.items():
        metric_names = sorted({name for metrics in metrics_rows for name in metrics})
        summary[key] = {"count": float(len(metrics_rows))}
        for name in metric_names:
            numeric_values = [
                metrics[name]
                for metrics in metrics_rows
                if name in metrics
                and isinstance(metrics[name], (int, float))
                and math.isfinite(float(metrics[name]))
            ]
            values = torch.tensor(
                numeric_values,
                dtype=torch.float64,
            )
            if values.numel() == 0:
                continue
            summary[key][f"{name}/mean"] = float(values.mean().item())
            summary[key][f"{name}/median"] = float(values.median().item())
    return summary
