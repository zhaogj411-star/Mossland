import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


def norm_conv1d(*args, norm: str = "weight_norm", **kwargs):
    conv = nn.Conv1d(*args, **kwargs)
    if norm == "weight_norm":
        return weight_norm(conv)
    if norm == "none":
        return conv
    raise ValueError(f"Unsupported norm={norm!r}")


def norm_conv_transpose1d(*args, norm: str = "weight_norm", **kwargs):
    conv = nn.ConvTranspose1d(*args, **kwargs)
    if norm == "weight_norm":
        return weight_norm(conv)
    if norm == "none":
        return conv
    raise ValueError(f"Unsupported norm={norm!r}")


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            norm_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            norm_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            norm_conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        strides: list[int] | tuple[int, ...] = (5, 4, 4, 4, 3, 2),
        d_latent: int | None = None,
    ):
        super().__init__()
        if d_latent is None:
            d_latent = d_model * (2 ** len(strides))

        layers: list[nn.Module] = [norm_conv1d(in_channels, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride=stride))
        layers.extend(
            [
                Snake1d(d_model),
                norm_conv1d(d_model, d_latent, kernel_size=3, padding=1),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            norm_conv_transpose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        rates: list[int] | tuple[int, ...] = (2, 3, 4, 4, 4, 5),
        out_channels: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            norm_conv1d(input_channels, channels, kernel_size=7, padding=3)
        ]
        for idx, stride in enumerate(rates):
            input_dim = channels // (2**idx)
            output_dim = channels // (2 ** (idx + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride=stride))
        layers.extend(
            [
                Snake1d(output_dim),
                norm_conv1d(output_dim, out_channels, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
