from typing import Literal, Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_2d_padding(kernel_size: tuple[int, int], dilation: tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class NormConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class DiscriminatorSTFT(nn.Module):
    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple[int, int] = (3, 9),
        dilations: Sequence[int] = (1, 2, 4),
        stride: tuple[int, int] = (1, 1),
        normalized: bool = True,
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        spec_scale_pow: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        activation_params = activation_params or {"negative_slope": 0.2}
        self.activation = getattr(nn, activation)(**activation_params)
        self.spec_transform = torch.nn.Sequential()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.normalized = bool(normalized)
        self.spec_scale_pow = float(spec_scale_pow)
        spec_channels = 2 * int(in_channels)
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * filters, max_filters)
        for idx, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (idx + 1)) * filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(int(dilation), 1),
                    padding=get_2d_padding(kernel_size, (int(dilation), 1)),
                )
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
        )

    def _spectrogram(self, audio: torch.Tensor):
        batch, channels, samples = audio.shape
        flat = audio.reshape(batch * channels, samples).float()
        window = torch.hann_window(self.win_length, device=flat.device, dtype=flat.dtype)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])
        if self.spec_scale_pow:
            spec = spec * torch.pow(spec.abs() + 1e-6, self.spec_scale_pow)
        return spec

    def forward(self, audio: torch.Tensor):
        features = []
        spec = self._spectrogram(audio)
        z = torch.cat([spec.real, spec.imag], dim=1)
        z = rearrange(z, "b c f t -> b c t f")
        for layer in self.convs:
            z = checkpoint(layer, z)
            z = self.activation(z)
            features.append(z)
        z = checkpoint(self.conv_post, z)
        return z, features


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: Sequence[int] = (1024, 2048, 512),
        hop_lengths: Sequence[int] = (256, 512, 128),
        win_lengths: Sequence[int] = (1024, 2048, 512),
        **kwargs,
    ):
        super().__init__()
        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError("n_ffts, hop_lengths, and win_lengths must have same length")
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters=filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=int(n_fft),
                    hop_length=int(hop),
                    win_length=int(win),
                    **kwargs,
                )
                for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
            ]
        )

    def forward(self, audio: torch.Tensor):
        logits = []
        features = []
        for discriminator in self.discriminators:
            logit, fmap = discriminator(audio)
            logits.append(logit)
            features.append(fmap)
        return logits, features


def get_hinge_losses(score_real: torch.Tensor, score_fake: torch.Tensor):
    gen_loss = -score_fake.mean()
    dis_loss = F.relu(1.0 - score_real).mean() + F.relu(1.0 + score_fake).mean()
    return dis_loss, gen_loss


def get_relativistic_losses(score_real: torch.Tensor, score_fake: torch.Tensor):
    diff = score_real - score_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss


class EncodecDiscriminator(nn.Module):
    def __init__(
        self,
        normalize_losses: bool = False,
        loss_type: Literal["hinge", "rpgan"] = "rpgan",
        **kwargs,
    ):
        super().__init__()
        self.discriminators = MultiScaleSTFTDiscriminator(**kwargs)
        self.normalize_losses = bool(normalize_losses)
        self.loss_type = loss_type

    def forward(self, audio: torch.Tensor):
        return self.discriminators(audio)

    def _feature_matching_reduction(self, real: torch.Tensor, fake: torch.Tensor):
        distance = (real - fake).abs().mean()
        if self.normalize_losses:
            distance = distance / (real.abs().mean() + 1e-3)
        return distance

    def loss(self, reals: torch.Tensor, fakes: torch.Tensor):
        feature_matching_distance = reals.new_zeros(())
        dis_loss = reals.new_zeros(())
        adv_loss = reals.new_zeros(())
        logits_true, features_true = self.forward(reals)
        logits_fake, features_fake = self.forward(fakes)
        for idx, (scale_true, scale_fake) in enumerate(zip(features_true, features_fake)):
            scale_fm = sum(
                self._feature_matching_reduction(real, fake)
                for real, fake in zip(scale_true, scale_fake)
            ) / len(scale_true)
            feature_matching_distance = feature_matching_distance + scale_fm
            if self.loss_type == "hinge":
                scale_dis, scale_adv = get_hinge_losses(logits_true[idx], logits_fake[idx])
            elif self.loss_type == "rpgan":
                scale_dis, scale_adv = get_relativistic_losses(logits_true[idx], logits_fake[idx])
            else:
                raise ValueError(f"unsupported loss_type={self.loss_type!r}")
            dis_loss = dis_loss + scale_dis
            adv_loss = adv_loss + scale_adv
        num_scales = max(1, len(logits_true))
        return dis_loss / num_scales, adv_loss / num_scales, feature_matching_distance / num_scales
