from .audio import AudioProcessor
from .quantize import ResidualVectorQuantize
from .tasks import TASK_NAMES

# from .audio import *
import soundfile as sf
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


def zero_init(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class QuantizedLatents:
    continuous: torch.Tensor
    discrete: torch.Tensor
    codes: torch.Tensor
    projected_latents: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    distill_loss: torch.Tensor


def upsample_1d(x):
    # 1D上采样,使用最近邻插值将尺寸放大2倍
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_1d(x):
    # 1D下采样,使用平均池化将尺寸减小2倍
    return F.avg_pool1d(x, kernel_size=2, stride=2)


def upsample_2d(x):
    # 2D上采样,使用最近邻插值将尺寸放大2倍
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_2d(x):
    # 2D下采样,使用平均池化将尺寸减小2倍
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class LayerNorm(nn.Module):
    # 层归一化模块
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, input):
        x = input.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FreqGain(nn.Module):
    # 频率增益模块,对频率维度进行缩放
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, freq_dim, 1)))

    def forward(self, input):
        return input * self.scale


class UpsampleConv(nn.Module):
    # 上采样卷积模块
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(UpsampleConv, self).__init__()
        self.normalize = normalize

        self.use_2d = use_2d

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding="same"
            )
        else:
            self.c = nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=1, padding="same"
            )

    def forward(self, x):

        if self.normalize:
            x = self.norm(x)

        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)

        return x


class DownsampleConv(nn.Module):
    # 下采样卷积模块
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(DownsampleConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.c = nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):

        if self.normalize:
            x = self.norm(x)
        x = self.c(x)

        return x


class UpsampleFreqConv(nn.Module):
    # 频率维度上采样卷积模块
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(UpsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), stride=1, padding="same"
        )

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = F.interpolate(x, scale_factor=(4, 1), mode="nearest")
        x = self.c(x)
        return x


class DownsampleFreqConv(nn.Module):
    # 频率维度下采样卷积模块
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(DownsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0)
        )

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = self.c(x)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    # 多头注意力模块,继承自PyTorch的MultiheadAttention
    def _reset_parameters(self):
        super()._reset_parameters()
        self.out_proj = zero_init(self.out_proj)


class Attention(nn.Module):
    # 注意力模块
    def __init__(self, dim, heads=4, normalize=True, use_2d=False):
        super(Attention, self).__init__()

        self.normalize = normalize
        self.use_2d = use_2d

        self.mha = MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
        )
        if normalize:
            self.norm = nn.GroupNorm(min(dim // 4, 32), dim)

    def forward(self, x):

        inp = x

        if self.normalize:
            x = self.norm(x)

        if self.use_2d:
            x = x.permute(0, 3, 2, 1)  # shape: [bs,len,freq,channels]
            bs, len, freq, channels = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            # shape: [bs*len,freq,channels]
            x = x.reshape(bs * len, freq, channels)
        else:
            x = x.permute(0, 2, 1)  # shape: [bs,len,channels]

        x = self.mha(x, x, x, need_weights=False)[0]

        if self.use_2d:
            x = x.reshape(bs, len, freq, channels).permute(0, 3, 2, 1)
        else:
            x = x.permute(0, 2, 1)
        x = x + inp

        return x


class ResBlock(nn.Module):
    # 残差块模块
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_channels=None,
        kernel_size=3,
        downsample=False,
        upsample=False,
        normalize=True,
        leaky=False,
        attention=False,
        heads=4,
        use_2d=False,
        normalize_residual=False,
        dropout_rate=0,
        # dropout在小于等于min_res_dropout分辨率的特征图上应用
        min_res_dropout=16,
    ):
        super(ResBlock, self).__init__()
        self.normalize = normalize
        self.attention = attention
        self.upsample = upsample
        self.downsample = downsample
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.normalize_residual = normalize_residual
        self.use_2d = use_2d
        self.dropout_rate = dropout_rate
        self.min_res_dropout = 16

        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d
        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.conv2 = zero_init(
            Conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )
        if in_channels != out_channels:
            self.res_conv = Conv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.res_conv = nn.Identity()
        if normalize:
            self.norm1 = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
            self.norm2 = nn.GroupNorm(min(out_channels // 4, 32), out_channels)
        if leaky:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.SiLU()
        if cond_channels is not None:
            self.proj_emb = zero_init(nn.Linear(cond_channels, out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        if attention:
            self.att = Attention(out_channels, heads, use_2d=use_2d)

    def forward(self, x, time_emb=None):
        if not self.normalize_residual:
            y = x.clone()
        if self.normalize:
            x = self.norm1(x)
        if self.normalize_residual:
            y = x.clone()
        x = self.activation(x)
        if self.downsample:
            if self.use_2d:
                x = downsample_2d(x)
                y = downsample_2d(y)
            else:
                x = downsample_1d(x)
                y = downsample_1d(y)
        if self.upsample:
            if self.use_2d:
                x = upsample_2d(x)
                y = upsample_2d(y)
            else:
                x = upsample_1d(x)
                y = upsample_1d(y)
        x = self.conv1(x)
        if time_emb is not None:
            if self.use_2d:
                x = x + self.proj_emb(time_emb)[:, :, None, None]
            else:
                x = x + self.proj_emb(time_emb)[:, :, None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1] <= self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x + y
        if self.attention:
            x = self.att(x)
        return x


# 从https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py改编
class GaussianFourierProjection(torch.nn.Module):
    """噪声水平的高斯傅里叶嵌入"""

    def __init__(self, embedding_size=128, scale=0.02):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(embedding_size // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    # 位置嵌入模块
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.embedding_size // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class Encoder(nn.Module):
    # 编码器模块
    def __init__(
        self,
        base_channels=64,
        layers_list_encoder=[1, 1, 1, 1, 1],
        multipliers_list=[1, 2, 4, 4, 4],
        attention_list_encoder=[0, 0, 1, 1, 1],
        freq_downsample_list=[1, 0, 0, 0],
        bottleneck_base_channels=512,
        num_bottleneck_layers=4,
        frequency_scaling=True,
        heads=4,
        normalization=True,
        bottleneck_channels=32 * 2,
        pre_normalize_2d_to_1d=True,
        pre_normalize_downsampling_encoder=True,
        hop=128 * 4,
        data_channels=2,
        min_res_dropout=16,
        dropout_rate=0,
    ):
        super(Encoder, self).__init__()

        layers_list = layers_list_encoder
        attention_list = attention_list_encoder
        self.layers_list = layers_list_encoder
        self.multipliers_list = multipliers_list
        self.min_res_dropout = 16
        input_channels = base_channels * multipliers_list[0]
        Conv = nn.Conv2d
        self.gain = FreqGain(freq_dim=hop * 2)
        self.frequency_scaling = frequency_scaling
        self.pre_normalize_2d_to_1d = pre_normalize_2d_to_1d

        channels = data_channels
        self.conv_inp = Conv(
            channels, input_channels, kernel_size=3, stride=1, padding=1
        )

        self.freq_dim = (hop * 2) // (4 ** freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** freq_downsample_list.count(0))

        # 下采样
        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(layers_list, multipliers_list)
        ):
            output_channels = base_channels * multiplier
            for num in range(num_layers):
                down_layers.append(
                    ResBlock(
                        input_channels,
                        output_channels,
                        normalize=normalization,
                        attention=attention_list[i] == 1,
                        heads=heads,
                        use_2d=True,
                        min_res_dropout=min_res_dropout,
                        dropout_rate=dropout_rate,
                    )
                )
                input_channels = output_channels
            if i != (len(layers_list) - 1):
                if freq_downsample_list[i] == 1:
                    down_layers.append(
                        DownsampleFreqConv(
                            input_channels, normalize=pre_normalize_downsampling_encoder
                        )
                    )
                else:
                    down_layers.append(
                        DownsampleConv(
                            input_channels,
                            use_2d=True,
                            normalize=pre_normalize_downsampling_encoder,
                        )
                    )

        if pre_normalize_2d_to_1d:
            self.prenorm_1d_to_2d = nn.GroupNorm(
                min(input_channels // 4, 32), input_channels
            )

        bottleneck_layers = []
        output_channels = bottleneck_base_channels
        bottleneck_layers.append(
            nn.Conv1d(
                input_channels * self.freq_dim,
                output_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )
        )
        for i in range(num_bottleneck_layers):
            bottleneck_layers.append(
                ResBlock(
                    output_channels,
                    output_channels,
                    normalize=normalization,
                    use_2d=False,
                    min_res_dropout=min_res_dropout,
                    dropout_rate=dropout_rate,
                )
            )
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        self.norm_out = nn.GroupNorm(min(output_channels // 4, 32), output_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = nn.Conv1d(
            output_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.activation_bottleneck = nn.Tanh()

        self.down_layers = nn.ModuleList(down_layers)

    def encode_features(self, x):
        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = self.gain(x)

        # 下采样
        k = 0
        for i, num_layers in enumerate(self.layers_list):
            for num in range(num_layers):
                x = self.down_layers[k](x)
                k = k + 1
            if i != (len(self.layers_list) - 1):
                x = self.down_layers[k](x)
                k = k + 1

        if self.pre_normalize_2d_to_1d:
            x = self.prenorm_1d_to_2d(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        return x

    def project_features(self, x):
        for layer in self.bottleneck_layers:
            x = layer(x)

        hidden = x
        continuous = self.hidden_to_latent(hidden)
        return continuous, hidden

    def hidden_to_latent(self, hidden):
        x = hidden
        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)
        return self.activation_bottleneck(x)

    def forward(self, x, extract_features=False, return_hidden=False):
        x = self.encode_features(x)
        if extract_features:
            return x

        continuous, hidden = self.project_features(x)
        if return_hidden:
            return continuous, hidden

        return continuous


class Decoder(nn.Module):
    # 解码器模块
    def __init__(
        self,
        base_channels=64,
        layers_list=[2, 2, 2, 2, 2],
        multipliers_list=[1, 2, 4, 4, 4],
        attention_list=[0, 0, 1, 1, 1],
        freq_downsample_list=[1, 0, 0, 0],
        layers_list_encoder=[1, 1, 1, 1, 1],
        attention_list_encoder=[0, 0, 1, 1, 1],
        bottleneck_base_channels=512,
        num_bottleneck_layers=4,
        heads=4,
        cond_channels=256,
        normalization=True,
        bottleneck_channels=64,
        hop=512,
        dropout_rate=0,
        min_res_dropout=16,
    ):
        super(Decoder, self).__init__()

        layers_list = layers_list_encoder
        attention_list = attention_list_encoder
        self.layers_list = layers_list_encoder
        self.multipliers_list = multipliers_list
        input_channels = base_channels * multipliers_list[-1]

        output_channels = bottleneck_base_channels
        self.conv_inp = nn.Conv1d(
            bottleneck_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.freq_dim = (hop * 2) // (4 ** freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** freq_downsample_list.count(0))

        bottleneck_layers = []
        for i in range(num_bottleneck_layers):
            bottleneck_layers.append(
                ResBlock(
                    output_channels,
                    output_channels,
                    normalize=normalization,
                    use_2d=False,
                    dropout_rate=dropout_rate,
                    min_res_dropout=min_res_dropout,
                )
            )

        self.conv_out_bottleneck = nn.Conv1d(
            output_channels,
            input_channels * self.freq_dim,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        # 上采样
        multipliers_list_upsampling = (
            list(reversed(multipliers_list))[1:] + list(reversed(multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(freq_downsample_list))
        up_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(layers_list), multipliers_list_upsampling)
        ):
            for num in range(num_layers):
                up_layers.append(
                    ResBlock(
                        input_channels,
                        input_channels,
                        normalize=normalization,
                        attention=list(reversed(attention_list))[i] == 1,
                        heads=heads,
                        use_2d=True,
                        min_res_dropout=min_res_dropout,
                        dropout_rate=dropout_rate,
                    )
                )
            if i != (len(layers_list) - 1):
                output_channels = base_channels * multiplier
                if freq_upsample_list[i] == 1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(
                        UpsampleConv(input_channels, output_channels, use_2d=True)
                    )
                input_channels = output_channels

        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, x):

        x = self.conv_inp(x)

        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.conv_out_bottleneck(x)

        x_ls = torch.chunk(x.unsqueeze(-2), self.freq_dim, -3)
        x = torch.cat(x_ls, -2)

        # 上采样
        k = 0
        pyramid_list = []
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for num in range(num_layers):
                x = self.up_layers[k](x)
                k = k + 1
            pyramid_list.append(x)
            if i != (len(self.layers_list) - 1):
                x = self.up_layers[k](x)
                k = k + 1

        pyramid_list = pyramid_list[::-1]

        return pyramid_list


class EncoderSameTemporal(Encoder):
    def __init__(
        self,
        same_transformer_depth=1,
        same_dim_heads=64,
        same_sliding_window=(1, 1),
        same_ff_mult=1,
        same_differential=False,
        same_dyt=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        base_channels = kwargs.get("base_channels", 64)
        multipliers_list = kwargs.get("multipliers_list", [1, 2, 4, 4, 4])
        bottleneck_base_channels = kwargs.get("bottleneck_base_channels", 512)
        bottleneck_channels = kwargs.get("bottleneck_channels", 32 * 2)
        num_bottleneck_layers = kwargs.get("num_bottleneck_layers", 4)
        normalization = kwargs.get("normalization", True)
        dropout_rate = kwargs.get("dropout_rate", 0)
        min_res_dropout = kwargs.get("min_res_dropout", 16)

        input_channels = int(base_channels) * int(multipliers_list[-1])
        output_channels = int(bottleneck_base_channels)
        bottleneck_layers = [
            ChannelLinear1d(input_channels * self.freq_dim, output_channels)
        ]
        for _ in range(int(num_bottleneck_layers)):
            bottleneck_layers.append(
                SameTemporalResBlock1d(
                    output_channels,
                    output_channels,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    min_res_dropout=min_res_dropout,
                    transformer_depth=same_transformer_depth,
                    dim_heads=same_dim_heads,
                    sliding_window=same_sliding_window,
                    ff_mult=same_ff_mult,
                    differential=same_differential,
                    dyt=same_dyt,
                )
            )
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)
        self.conv_out = ChannelLinear1d(output_channels, int(bottleneck_channels))


class DecoderSameTemporal(Decoder):
    def __init__(
        self,
        same_transformer_depth=1,
        same_dim_heads=64,
        same_sliding_window=(1, 1),
        same_ff_mult=1,
        same_differential=False,
        same_dyt=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        base_channels = kwargs.get("base_channels", 64)
        multipliers_list = kwargs.get("multipliers_list", [1, 2, 4, 4, 4])
        bottleneck_base_channels = kwargs.get("bottleneck_base_channels", 512)
        bottleneck_channels = kwargs.get("bottleneck_channels", 64)
        num_bottleneck_layers = kwargs.get("num_bottleneck_layers", 4)
        normalization = kwargs.get("normalization", True)
        dropout_rate = kwargs.get("dropout_rate", 0)
        min_res_dropout = kwargs.get("min_res_dropout", 16)

        input_channels = int(base_channels) * int(multipliers_list[-1])
        output_channels = int(bottleneck_base_channels)
        self.conv_inp = ChannelLinear1d(int(bottleneck_channels), output_channels)
        self.bottleneck_layers = nn.ModuleList(
            [
                SameTemporalResBlock1d(
                    output_channels,
                    output_channels,
                    normalize=normalization,
                    dropout_rate=dropout_rate,
                    min_res_dropout=min_res_dropout,
                    transformer_depth=same_transformer_depth,
                    dim_heads=same_dim_heads,
                    sliding_window=same_sliding_window,
                    ff_mult=same_ff_mult,
                    differential=same_differential,
                    dyt=same_dyt,
                )
                for _ in range(int(num_bottleneck_layers))
            ]
        )
        self.conv_out_bottleneck = ChannelLinear1d(
            output_channels,
            input_channels * self.freq_dim,
        )


class MosslandCodec(nn.Module):
    # U-Net模型
    def __init__(
        self,
        audio_processor: AudioProcessor,
        sample_rate: 44100,
        base_channels=64,  # 基础通道数
        layers_list=[2, 2, 2, 2, 2],  # 每个分辨率层的块数
        multipliers_list=[1, 2, 4, 4, 4],  # 每个分辨率层的通道数倍数
        attention_list=[0, 0, 1, 1, 1],  # 每个分辨率层是否使用注意力
        freq_downsample_list=[1, 0, 0, 0],  # 每个分辨率层的频率下采样方式
        layers_list_encoder=[1, 1, 1, 1, 1],  # 编码器每个分辨率层的块数
        attention_list_encoder=[0, 0, 1, 1, 1],  # 编码器每个分辨率层是否使用注意力
        bottleneck_base_channels=512,  # 瓶颈层基础通道数
        num_bottleneck_layers=4,  # 瓶颈层块数
        frequency_scaling=True,
        heads=4,  # 注意力头数
        cond_channels=256,  # 条件嵌入维度
        use_fourier=False,  # 是否使用傅里叶嵌入
        fourier_scale=0.2,  # 高斯傅里叶层的缩放参数
        normalization=True,  # 是否使用组归一化
        dropout_rate=0.0,  # dropout率
        min_res_dropout=16,  # dropout应用的最小分辨率
        init_as_zero=True,  # 是否将跳跃连接前的卷积核初始化为0
        bottleneck_channels=32 * 2,  # 编码器瓶颈层通道数
        pre_normalize_2d_to_1d=True,  # 是否对编码器2D到1D的连接进行预归一化
        pre_normalize_downsampling_encoder=True,
        hop=128 * 4,  # 变换的hop大小
        data_channels=2,  # 输入数据通道数
        sigma_max=80.0,  # 最大噪声水平
        sigma_min=0.002,  # 最小噪声水平
        sigma_data=0.5,  # 数据噪声水平
        mixed_precision=True,  # 是否使用混合精度训练
        rho=7.0,  # 噪声调度参数
        max_waveform_length_encode=44100 * 60,  # 编码时最大波形长度
        max_batch_size_encode=1,  # 编码时最大批次大小
        max_waveform_length_decode=44100 * 60,  # 解码时最大波形长度
        max_batch_size_decode=1,
        quantizer_num_quantizers=0,
        quantizer_codebook_size=1024,
        quantizer_codebook_dim=8,
        quantizer_dropout=0.0,
        quantizer_decay=0.8,
        quantizer_kmeans_init=True,
        quantizer_kmeans_iters=10,
        quantizer_threshold_ema_dead_code=2,
        task_names: list[str] | tuple[str, ...] | None = None,
        task_embedding_init_std: float = 0.02,
        **kwargs
    ):
        super().__init__()
        if task_names is None:
            task_names = TASK_NAMES
        task_names = tuple(str(name) for name in task_names)
        if not task_names:
            raise ValueError("task_names must not be empty")
        self.sigma_max = sigma_max
        self.frequency_scaling = frequency_scaling
        self.layers_list = layers_list
        self.multipliers_list = multipliers_list
        self.sample_rate = sample_rate
        input_channels = base_channels * multipliers_list[0]
        self.hop = hop
        self.freq_downsample_list = freq_downsample_list
        self.layers_list_encoder = layers_list_encoder
        self.attention_list_encoder = attention_list_encoder
        self.bottleneck_base_channels = bottleneck_base_channels
        self.num_bottleneck_layers = num_bottleneck_layers
        self.heads = heads
        self.cond_channels = cond_channels
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.min_res_dropout = min_res_dropout
        self.init_as_zero = init_as_zero
        self.bottleneck_channels = bottleneck_channels
        self.pre_normalize_2d_to_1d = pre_normalize_2d_to_1d
        self.pre_normalize_downsampling_encoder = pre_normalize_downsampling_encoder
        self.mixed_precision = mixed_precision
        self.data_channels = data_channels
        self.quantizer_num_quantizers = int(quantizer_num_quantizers)
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.rho = rho
        self.task_names = task_names
        self.task_to_idx = {name: idx for idx, name in enumerate(self.task_names)}
        self.task_embedding_init_std = float(task_embedding_init_std)
        Conv = nn.Conv2d
        ## audio processor
        self.audio_processor = audio_processor
        self.encoder = Encoder(
            base_channels=base_channels,
            layers_list_encoder=layers_list_encoder,
            multipliers_list=multipliers_list,
            attention_list_encoder=attention_list_encoder,
            freq_downsample_list=freq_downsample_list,
            bottleneck_base_channels=bottleneck_base_channels,
            num_bottleneck_layers=num_bottleneck_layers,
            frequency_scaling=frequency_scaling,
            heads=heads,
            normalization=normalization,
            bottleneck_channels=bottleneck_channels,
            pre_normalize_2d_to_1d=pre_normalize_2d_to_1d,
            pre_normalize_downsampling_encoder=pre_normalize_downsampling_encoder,
            hop=hop,
            data_channels=data_channels,
            min_res_dropout=min_res_dropout,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            base_channels=base_channels,
            layers_list=layers_list,
            multipliers_list=multipliers_list,
            attention_list=attention_list,
            freq_downsample_list=freq_downsample_list,
            layers_list_encoder=layers_list_encoder,
            attention_list_encoder=attention_list_encoder,
            bottleneck_base_channels=bottleneck_base_channels,
            num_bottleneck_layers=num_bottleneck_layers,
            heads=heads,
            cond_channels=cond_channels,
            normalization=normalization,
            bottleneck_channels=bottleneck_channels,
            hop=hop,
            dropout_rate=dropout_rate,
            min_res_dropout=min_res_dropout,
        )

        if self.quantizer_num_quantizers > 0:
            self.quantizer = ResidualVectorQuantize(
                input_dim=bottleneck_base_channels,
                n_codebooks=self.quantizer_num_quantizers,
                codebook_size=quantizer_codebook_size,
                codebook_dim=quantizer_codebook_dim,
                quantizer_dropout=quantizer_dropout,
                decay=quantizer_decay,
                kmeans_init=quantizer_kmeans_init,
                kmeans_iters=quantizer_kmeans_iters,
                threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            )
        else:
            self.quantizer = None

        if use_fourier:
            self.emb = GaussianFourierProjection(
                embedding_size=cond_channels, scale=fourier_scale
            )
        else:
            self.emb = PositionalEmbedding(embedding_size=cond_channels)

        self.emb_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
        )
        self.task_embedding = nn.Embedding(len(self.task_names), cond_channels)
        nn.init.normal_(
            self.task_embedding.weight,
            mean=0.0,
            std=self.task_embedding_init_std,
        )

        self.scale_inp = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            (
                zero_init(nn.Linear(cond_channels, hop * 2))
                if init_as_zero
                else nn.Linear(cond_channels, hop * 2)
            ),
        )
        self.scale_out = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            (
                zero_init(nn.Linear(cond_channels, hop * 2))
                if init_as_zero
                else nn.Linear(cond_channels, hop * 2)
            ),
        )

        self.conv_inp = Conv(
            data_channels, input_channels, kernel_size=3, stride=1, padding=1
        )

        # 下采样
        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(layers_list, multipliers_list)
        ):
            output_channels = base_channels * multiplier
            for num in range(num_layers):
                down_layers.append(
                    Conv(
                        output_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                down_layers.append(
                    ResBlock(
                        output_channels,
                        output_channels,
                        cond_channels,
                        normalize=normalization,
                        attention=attention_list[i] == 1,
                        heads=heads,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                    )
                )
                input_channels = output_channels
            if i != (len(layers_list) - 1):
                output_channels = base_channels * multipliers_list[i + 1]
                if freq_downsample_list[i] == 1:
                    down_layers.append(
                        DownsampleFreqConv(input_channels, output_channels)
                    )
                else:
                    down_layers.append(
                        DownsampleConv(input_channels, output_channels, use_2d=True)
                    )

        # UPSAMPLING
        multipliers_list_upsampling = (
            list(reversed(multipliers_list))[1:] + list(reversed(multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(freq_downsample_list))
        up_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(layers_list), multipliers_list_upsampling)
        ):
            for num in range(num_layers):
                up_layers.append(
                    Conv(
                        input_channels,
                        input_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                up_layers.append(
                    ResBlock(
                        input_channels,
                        input_channels,
                        cond_channels,
                        normalize=normalization,
                        attention=list(reversed(attention_list))[i] == 1,
                        heads=heads,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                    )
                )
            if i != (len(layers_list) - 1):
                output_channels = base_channels * multiplier
                if freq_upsample_list[i] == 1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(
                        UpsampleConv(input_channels, output_channels, use_2d=True)
                    )
                input_channels = output_channels

        self.conv_decoded = Conv(
            input_channels, input_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm_out = nn.GroupNorm(min(input_channels // 4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = (
            zero_init(
                Conv(input_channels, data_channels, kernel_size=3, stride=1, padding=1)
            )
            if init_as_zero
            else Conv(input_channels, data_channels, kernel_size=3, stride=1, padding=1)
        )

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        # 推理相关参数
        self.max_waveform_length_encode = max_waveform_length_encode
        self.max_batch_size_encode = max_batch_size_encode
        self.max_waveform_length_decode = max_waveform_length_decode
        self.max_batch_size_decode = max_batch_size_decode

    @property
    def has_quantizer(self):
        return self.quantizer is not None

    def quantize_representation(
        self,
        representation,
        detach_encoder=True,
        n_quantizers=None,
    ):
        if self.quantizer is None:
            raise RuntimeError("MosslandCodec quantizer is disabled")

        continuous, hidden = self.encoder(representation, return_hidden=True)
        quantizer_input = hidden.detach() if detach_encoder else hidden
        (
            quantized_hidden,
            codes,
            commitment_loss,
        ) = self.quantizer(quantizer_input, n_quantizers=n_quantizers)
        discrete = self.encoder.hidden_to_latent(quantized_hidden)
        distill_loss = F.mse_loss(quantized_hidden.float(), hidden.detach().float())
        codebook_loss = hidden.new_zeros(())
        return QuantizedLatents(
            continuous=continuous,
            discrete=discrete,
            codes=codes,
            projected_latents=quantized_hidden,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            distill_loss=distill_loss,
        )

    def latent_from_codes(self, codes):
        if self.quantizer is None:
            raise RuntimeError("MosslandCodec quantizer is disabled")
        quantized_hidden, _ = self.quantizer.from_codes(codes)
        return self.encoder.hidden_to_latent(quantized_hidden)

    @torch.no_grad()
    def decode_codes(
        self,
        codes,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        task_id="reconstruct",
        task_idx=None,
    ):
        codes = (
            torch.from_numpy(codes).to(next(self.parameters()).device)
            if isinstance(codes, np.ndarray)
            else codes.to(next(self.parameters()).device)
        )
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        latent = self.latent_from_codes(codes)
        return self.decode(
            latent,
            denoising_steps=denoising_steps,
            max_waveform_length=max_waveform_length,
            max_batch_size=max_batch_size,
            rescale=rescale,
            target_length=target_length,
            task_id=task_id,
            task_idx=task_idx,
        )

    @staticmethod
    def _lookup_condition_index(lookup, value, strict: bool):
        key = str(value)
        if key in lookup:
            return lookup[key]
        if strict:
            raise KeyError(key)
        return 0

    def _coerce_condition_indices(self, values, indices, batch_size, device):
        if indices is not None:
            if torch.is_tensor(indices):
                idx = indices.to(device=device, dtype=torch.long).reshape(-1)
            else:
                idx = torch.as_tensor(indices, device=device, dtype=torch.long).reshape(-1)
        else:
            if values is None:
                values = self.task_names[0]
            if isinstance(values, str):
                idx = torch.full(
                    (batch_size,),
                    self._lookup_condition_index(self.task_to_idx, values, strict=True),
                    device=device,
                    dtype=torch.long,
                )
            else:
                if torch.is_tensor(values):
                    idx = values.to(device=device, dtype=torch.long).reshape(-1)
                else:
                    value_list = list(values) if isinstance(values, (list, tuple)) else [values]
                    idx = torch.tensor(
                        [
                            self._lookup_condition_index(self.task_to_idx, value, strict=False)
                            for value in value_list
                        ],
                        device=device,
                        dtype=torch.long,
                    )

        if idx.numel() == 1:
            return idx.expand(batch_size)
        if idx.numel() == batch_size:
            return idx
        if idx.numel() * 2 == batch_size:
            return torch.cat((idx, idx), dim=0)
        raise ValueError(
            f"condition batch size mismatch: got {idx.numel()}, expected 1, "
            f"{batch_size // 2}, or {batch_size}"
        )

    def _condition_embedding(self, sigma_embedding, task_id="reconstruct", task_idx=None):
        task_idx = self._coerce_condition_indices(
            task_id,
            task_idx,
            sigma_embedding.shape[0],
            sigma_embedding.device,
        )
        cond = sigma_embedding + self.task_embedding(task_idx).to(sigma_embedding.dtype)
        return self.emb_proj(cond)

    @staticmethod
    def _slice_condition_values(values, start: int, end: int):
        if values is None or isinstance(values, str):
            return values
        if torch.is_tensor(values):
            flat = values.reshape(-1)
            if flat.numel() >= end:
                return flat[start:end]
            return values
        if isinstance(values, tuple):
            if len(values) >= end:
                return values[start:end]
            return values
        if isinstance(values, list):
            if len(values) >= end:
                return values[start:end]
            return values
        return values

    def forward(
        self,
        latents,
        x,
        sigma=None,
        pyramid_latents=None,
        latent_override=None,
        task_id="reconstruct",
        task_idx=None,
    ):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        latents = latents.to(dtype)
        if sigma is None:
            sigma = self.sigma_max
        inp = x

        # CONDITIONING
        sigma = torch.ones((x.shape[0],), dtype=x.dtype).to(x.device) * sigma
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = self.emb(sigma_log.to(dtype)).to(dtype)
        # breakpoint()
        time_emb = self._condition_embedding(
            emb_sigma_log,
            task_id=task_id,
            task_idx=task_idx,
        )

        scale_w_inp = self.scale_inp(time_emb).reshape(x.shape[0], 1, -1, 1)
        scale_w_out = self.scale_out(time_emb).reshape(x.shape[0], 1, -1, 1)

        c_skip, c_out, c_in = self._get_c(sigma)

        x = c_in * x

        if latent_override is not None:
            latents = latent_override.to(dtype)
        elif latents.shape == x.shape:
            latents = self.encoder(latents.to(dtype))

        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)

        x = self.conv_inp(x.to(dtype))
        if self.frequency_scaling:
            x = (1.0 + scale_w_inp) * x

        skip_list = []

        # DOWNSAMPLING
        k = 0
        r = 0
        for i, num_layers in enumerate(self.layers_list):
            for num in range(num_layers):
                d = self.down_layers[k](pyramid_latents[i])
                k = k + 1
                x = (x + d) / np.sqrt(2.0)
                x = self.down_layers[k](x, time_emb)
                skip_list.append(x)
                k = k + 1
            if i != (len(self.layers_list) - 1):
                x = self.down_layers[k](x)
                k = k + 1

        # UPSAMPLING
        k = 0
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for num in range(num_layers):
                d = self.up_layers[k](pyramid_latents[-i - 1])
                k = k + 1
                x = (x + skip_list.pop() + d) / np.sqrt(3.0)
                x = self.up_layers[k](x, time_emb)
                k = k + 1
            if i != (len(self.layers_list) - 1):
                x = self.up_layers[k](x)
                k = k + 1

        d = self.conv_decoded(pyramid_latents[0])
        x = (x + d) / np.sqrt(2.0)

        x = self.norm_out(x)
        x = self.activation_out(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_out) * x
        x = self.conv_out(x)

        out = c_skip * inp + c_out * x

        return out

    @torch.no_grad()
    def encode(
        self,
        path_or_audio,
        max_waveform_length=None,
        max_batch_size=None,
        extract_features=False,
        rescale=1,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        """编码音频到潜空间

        Args:
            path_or_audio: 音频文件路径或音频数据
            max_waveform_length: 最大波形长度
            max_batch_size: 最大批次大小
            extract_features: 是否提取特征

        Returns:
            latent: 编码后的潜变量 [audio_channels, dim, length]
        """
        self.eval()
        device = next(self.parameters()).device

        # 设置默认值
        max_waveform_length = max_waveform_length or self.max_waveform_length_encode
        max_batch_size = max_batch_size or self.max_batch_size_encode

        # 加载和预处理音频
        if isinstance(path_or_audio, str):
            audio, sr = sf.read(path_or_audio, dtype="float32", always_2d=True)
            audio = np.transpose(audio, [1, 0])
        else:
            audio = path_or_audio
            if len(audio.shape) == 1:
                audio = (
                    audio.unsqueeze(0)
                    if torch.is_tensor(audio)
                    else np.expand_dims(audio, 0)
                )

        audio = (
            torch.from_numpy(audio).to(device)
            if isinstance(audio, np.ndarray)
            else audio.to(device)
        )
        if audio.ndim == 2 and self.data_channels == audio.shape[0] * 2:
            audio = audio.unsqueeze(0)

        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        original_length = audio.shape[-1]

        if getattr(self.audio_processor, "center_pad", False):
            min_frames = max(1, original_length // self.hop)
            aligned_frames = (min_frames // downscaling_factor) * downscaling_factor
            target_length = aligned_frames * self.hop
        else:
            frame_length = self.audio_processor.fac * self.hop
            min_frames = max(1, (original_length - frame_length) // self.hop + 1)
            aligned_frames = (min_frames // downscaling_factor) * downscaling_factor
            target_length = (
                aligned_frames * self.hop
                + (self.audio_processor.fac - 1) * self.hop
            )

        # 裁剪或填充到目标长度
        if original_length > target_length:
            audio = audio[..., :target_length]
        elif original_length < target_length:
            pad_size = target_length - original_length
            audio = F.pad(audio, (0, pad_size), mode='constant', value=0)

        # 转换为频谱表示
        repr_encoder = self.audio_processor.to_representation_encoder(audio)

        # 处理长序列
        latent = self._process_long_sequence(
            repr_encoder,
            max_waveform_length,
            max_batch_size,
            downscaling_factor,
            extract_features,
            original_length,
            quantize=quantize,
            return_codes=return_codes,
            n_quantizers=n_quantizers,
        )

        if extract_features or return_codes:
            return latent
        return latent / rescale

    @torch.no_grad()
    def decode(
        self,
        latent,
        denoising_steps=1,
        max_waveform_length=None,
        max_batch_size=None,
        rescale=1,
        target_length=None,
        task_id="reconstruct",
        task_idx=None,
    ):
        """解码潜变量到音频

        Args:
            latent: 潜变量 [audio_channels, dim, length]
            denoising_steps: 去噪步数
            max_waveform_length: 最大波形长度
            max_batch_size: 最大批次大小
            rescale: 缩放因子
            target_length: 目标音频长度（用于精确控制输出长度）

        Returns:
            audio: 解码后的音频波形 [waveform_samples, audio_channels]
        """
        self.eval()
        device = next(self.parameters()).device

        # 设置默认值
        max_waveform_length = max_waveform_length or self.max_waveform_length_decode
        max_batch_size = max_batch_size or self.max_batch_size_decode

        # 预处理潜变量
        latent = latent * rescale
        latent = (
            torch.from_numpy(latent).to(device)
            if isinstance(latent, np.ndarray)
            else latent.to(device)
        )
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(0)

        # 计算下采样因子
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)

        latent_length = latent.shape[-1]
        max_latent_length = int(max_waveform_length / self.hop) // downscaling_factor

        # 处理长序列分段
        if latent_length > max_latent_length:
            # 简单分段处理，不使用重叠
            latent_segments = []
            start_idx = 0

            while start_idx < latent_length:
                end_idx = min(start_idx + max_latent_length, latent_length)
                segment = latent[:, :, start_idx:end_idx]

                if segment.shape[-1] > 0:
                    latent_segments.append(segment)

                start_idx = end_idx

            # 批处理解码
            repr_segments = []
            for segment in latent_segments:
                if segment.shape[0] > max_batch_size:
                    segment_chunks = torch.split(segment, max_batch_size, dim=0)
                    segment_reprs = []
                    for chunk in segment_chunks:
                        start = len(segment_reprs) * max_batch_size
                        end = start + chunk.shape[0]
                        repr_chunk = self._decode_to_representation(
                            chunk,
                            denoising_steps,
                            device,
                            task_id=self._slice_condition_values(task_id, start, end),
                            task_idx=self._slice_condition_values(task_idx, start, end),
                        )
                        segment_reprs.append(repr_chunk)
                    segment_repr = torch.cat(segment_reprs, dim=0)
                else:
                    segment_repr = self._decode_to_representation(
                        segment,
                        denoising_steps,
                        device,
                        task_id=task_id,
                        task_idx=task_idx,
                    )
                repr_segments.append(segment_repr)

            # 简单拼接segments
            repr = torch.cat(repr_segments, dim=-1)
        else:
            # 短序列直接处理
            if latent.shape[0] > max_batch_size:
                latent_chunks = torch.split(latent, max_batch_size, dim=0)
                repr_chunks = []
                for chunk in latent_chunks:
                    start = len(repr_chunks) * max_batch_size
                    end = start + chunk.shape[0]
                    repr_chunk = self._decode_to_representation(
                        chunk,
                        denoising_steps,
                        device,
                        task_id=self._slice_condition_values(task_id, start, end),
                        task_idx=self._slice_condition_values(task_idx, start, end),
                    )
                    repr_chunks.append(repr_chunk)
                repr = torch.cat(repr_chunks, dim=0)
            else:
                repr = self._decode_to_representation(
                    latent,
                    denoising_steps,
                    device,
                    task_id=task_id,
                    task_idx=task_idx,
                )

        # 转换为波形
        audio = self.audio_processor.to_waveform(repr, self.hop)

        # 如果指定了目标长度，精确裁剪到原始长度
        if target_length is not None:
            if audio.shape[-1] > target_length:
                audio = audio[..., :target_length]
            elif audio.shape[-1] < target_length:
                # 如果音频太短，用零填充
                pad_size = target_length - audio.shape[-1]
                audio = F.pad(audio, (0, pad_size), mode='constant', value=0)

        return audio

    def _decode_to_representation(
        self,
        latents,
        diffusion_steps=1,
        device=None,
        task_id="reconstruct",
        task_idx=None,
    ):
        """解码潜变量到频谱表示"""
        device = device or next(self.parameters()).device
        num_samples = latents.shape[0]
        downscaling_factor = 2 ** sum(1 for x in self.freq_downsample_list if x == 0)
        sample_length = int(latents.shape[-1] * downscaling_factor)

        initial_noise = (
            torch.randn(
                (num_samples, self.data_channels, self.hop * 2, sample_length)
            ).to(device)
            * self.sigma_max
        )

        return self._reverse_diffusion(
            initial_noise,
            diffusion_steps,
            latents,
            task_id=task_id,
            task_idx=task_idx,
        )

    def _get_c(self, sigma):
        """获取缩放系数 c_skip, c_out, c_in"""
        sigma_correct = self.sigma_min
        c_skip = (self.sigma_data**2.0) / (
            ((sigma - sigma_correct) ** 2.0) + (self.sigma_data**2.0)
        )
        c_out = (self.sigma_data * (sigma - sigma_correct)) / (
            ((self.sigma_data**2.0) + (sigma**2.0)) ** 0.5
        )
        c_in = 1.0 / (((sigma**2.0) + (self.sigma_data**2.0)) ** 0.5)
        return (
            c_skip.reshape(-1, 1, 1, 1),
            c_out.reshape(-1, 1, 1, 1),
            c_in.reshape(-1, 1, 1, 1),
        )

    def _get_sigma(self, i, k):
        """获取离散索引i对应的噪声水平"""
        return (
            self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / (k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def _reverse_step(self, x, noise, sigma):
        """反向一步ODE"""
        return x + ((sigma**2 - self.sigma_min**2) ** 0.5) * noise

    def _denoise(self, noisy_samples, sigma, latents=None, task_id="reconstruct", task_idx=None):
        """对给定噪声水平的样本去噪"""
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
            ):
                if latents is not None:
                    pred_samples = self(
                        latents,
                        noisy_samples,
                        sigma,
                        task_id=task_id,
                        task_idx=task_idx,
                    )
                else:
                    pred_samples = self(noisy_samples, sigma, task_id=task_id, task_idx=task_idx)
        pred_noises = torch.randn_like(pred_samples)
        return pred_noises, pred_samples

    def _reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
        latents=None,
        task_id="reconstruct",
        task_idx=None,
    ):
        """反向扩散过程生成样本"""
        next_noisy_samples = initial_noise
        for k in range(diffusion_steps):
            sigma = self._get_sigma(diffusion_steps + 1 - k, diffusion_steps + 1)
            next_sigma = self._get_sigma(diffusion_steps - k, diffusion_steps + 1)

            noisy_samples = next_noisy_samples
            pred_noises, pred_samples = self._denoise(
                noisy_samples,
                sigma,
                latents,
                task_id=task_id,
                task_idx=task_idx,
            )
            next_noisy_samples = self._reverse_step(
                pred_samples, pred_noises, next_sigma
            )

        return pred_samples.detach().cpu()

    def _process_long_sequence(
        self,
        x,
        max_length,
        max_batch_size,
        downscaling_factor,
        extract_features,
        original_length,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        """处理长序列的辅助函数，简化版本不使用重叠"""
        sample_length = x.shape[-1]
        max_sample_length = (
            int(max_length / self.hop) // downscaling_factor
        ) * downscaling_factor

        # 处理超长序列
        if sample_length > max_sample_length:
            # 简单分段处理，不使用重叠
            x_segments = []
            start_idx = 0

            while start_idx < sample_length:
                end_idx = min(start_idx + max_sample_length, sample_length)
                segment = x[:, :, :, start_idx:end_idx]

                if segment.shape[-1] > 0:
                    x_segments.append(segment)

                start_idx = end_idx

            # 批处理编码
            latent_segments = []
            for segment in x_segments:
                if segment.shape[0] > max_batch_size:
                    segment_chunks = torch.split(segment, max_batch_size, dim=0)
                    segment_latents = []
                    for chunk in segment_chunks:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=self.mixed_precision,
                        ):
                            latent_chunk = self._encode_representation_chunk(
                                chunk,
                                extract_features=extract_features,
                                quantize=quantize,
                                return_codes=return_codes,
                                n_quantizers=n_quantizers,
                            )
                        segment_latents.append(latent_chunk)
                    segment_latent = torch.cat(segment_latents, dim=0)
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
                    ):
                        segment_latent = self._encode_representation_chunk(
                            segment,
                            extract_features=extract_features,
                            quantize=quantize,
                            return_codes=return_codes,
                            n_quantizers=n_quantizers,
                        )
                latent_segments.append(segment_latent)

            # 简单拼接segments
            latent = torch.cat(latent_segments, dim=-1)
        else:
            # 短序列直接处理
            if x.shape[0] > max_batch_size:
                x_chunks = torch.split(x, max_batch_size, dim=0)
                latents = []
                for chunk in x_chunks:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=self.mixed_precision,
                    ):
                        latent_chunk = self._encode_representation_chunk(
                            chunk,
                            extract_features=extract_features,
                            quantize=quantize,
                            return_codes=return_codes,
                            n_quantizers=n_quantizers,
                        )
                    latents.append(latent_chunk)
                latent = torch.cat(latents, dim=0)
            else:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
                ):
                    latent = self._encode_representation_chunk(
                        x,
                        extract_features=extract_features,
                        quantize=quantize,
                        return_codes=return_codes,
                        n_quantizers=n_quantizers,
                    )

        # 如果有多个batch，需要在时间维度上拼接
        if latent.shape[0] > 1:
            latent = torch.cat(torch.split(latent, 1, 0), -1)

        return latent

    def _encode_representation_chunk(
        self,
        x,
        extract_features=False,
        quantize=False,
        return_codes=False,
        n_quantizers=None,
    ):
        if not quantize:
            return self.encoder(x, extract_features=extract_features)
        if extract_features:
            raise ValueError("extract_features=True is not supported with quantize=True")
        quantized = self.quantize_representation(
            x,
            detach_encoder=False,
            n_quantizers=n_quantizers,
        )
        if return_codes:
            return quantized.codes
        return quantized.discrete
