import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .audio import AudioProcessor
from scripts.codec_common.quantize import ResidualVectorQuantize
from .transformer import Transformer, Transformer_Diffusion

@dataclass
class QuantizedLatents:
    continuous: torch.Tensor
    discrete: torch.Tensor
    codes: torch.Tensor
    projected_latents: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    distill_loss: torch.Tensor


os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "torch_compile_cache",
)

torch.backends.cudnn.benchmark = True


def exists(val):
    return val is not None

def init(module):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.)
    return module

def zero_init(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1,1,freq_dim,1)))

    def forward(self, input):
        return input*self.scale


class GroupNorm(nn.Module):
    def __init__(self, dim, cond_dim=None, affine=True):
        super(GroupNorm, self).__init__()
        self.affine = affine
        self.norm = nn.GroupNorm(min(dim//4, 32), dim, affine=False)
        if exists(cond_dim):
            self.cond_proj = zero_init(nn.Linear(cond_dim, dim))
        else:
            if affine:
                self.weight = nn.Parameter(torch.ones((dim,)))

    def forward(self, x, cond=None):
        x = self.norm(x)
        if exists(cond):
            cond = self.cond_proj(cond)
            if x.dim()==4:
                cond = cond.view(cond.shape[0], -1, 1, 1)
            else:
                cond = cond.view(cond.shape[0], -1, 1)
            x = x * (1.+cond)
        else:
            if self.affine:
                if x.dim()==4:
                    x = x * self.weight.view(1,-1,1,1)
                else:
                    x = x * self.weight.view(1,-1,1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, cond_dim=None, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if exists(cond_dim):
            self.cond_proj = zero_init(nn.Linear(cond_dim, dim))
        else:
            if affine:
                self.weight = nn.Parameter(torch.ones((dim,)))

    def forward(self, x, cond=None):
        x = (x.float() * torch.rsqrt(x.float().pow(2).mean(-2, keepdim=True) + self.eps)).type_as(x)
        if exists(cond):
            cond = self.cond_proj(cond)
            cond = cond.view(cond.shape[0], -1, 1)
            x = x * (1.+cond)
        else:
            if self.affine:
                x = x * self.weight.view(1,-1,1)
        return x


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor):
        super(Downsample, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.groups = input_channels

        # Initialize the weights to average filter
        weight = torch.ones(output_channels, input_channels // self.groups, *factor)
        weight = weight / (factor[0] * factor[1])
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=self.factor, padding=0, groups=self.groups)
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor):
        super(Upsample, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.groups = output_channels

        # Initialize the weights for evenly spreading the values
        weight = torch.ones(input_channels, output_channels // self.groups, *factor)
        weight = weight / (input_channels/output_channels)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = F.conv_transpose2d(x, self.weight, stride=self.factor, padding=0, groups=self.groups)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(UpsampleConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(2,2))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DownsampleConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(2,2))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(UpsampleFreqConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(factor,1))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(factor,1), stride=(factor,1), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(DownsampleFreqConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(factor,1))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(factor,1), stride=(factor,1), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class UpsampleTimeConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(UpsampleTimeConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(1,factor))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,factor), stride=(1,factor), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class DownsampleTimeConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(DownsampleTimeConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(1,factor))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(1,factor), stride=(1,factor), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class Feedforward(nn.Module):
    def __init__(self, dim, mlp_mult = 1, use_2d=True):
        super().__init__()
        inner_dim = int(dim * mlp_mult)
        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d

        self.ff1 = init(Conv(dim, inner_dim, 3, padding=1, bias=False))
        self.activation = nn.SiLU()
        self.ff2 = zero_init(Conv(inner_dim, dim, 3, padding=1, bias=False))

    def forward(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_mult=1, cond_dim=None, use_2d=True):
        super(ConvBlock, self).__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, use_2d=use_2d)
        self.norm = GroupNorm(dim, cond_dim)

    def forward(self, x, cond=None):
        inp = x.clone()
        x = self.norm(x, cond)
        x = self.ff(x)
        return x+inp


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        freqs = freqs.to(x.dtype)
        # take outer product
        x = x.unsqueeze(-1) * freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class DownFrontend(nn.Module):
    def __init__(
        self,
        frontend_layers_list,
        cond_dim=None,
        *,
        stft_channels=4,
        frontend_base_channels=64,
        frontend_multipliers_list=None,
        frontend_freq_downsample_list=None,
    ):
        super(DownFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        self.frontend_base_channels = frontend_base_channels
        self.frontend_multipliers_list = list(
            frontend_multipliers_list
            if frontend_multipliers_list is not None
            else [1, 2, 4, 8]
        )
        self.frontend_freq_downsample_list = list(
            frontend_freq_downsample_list
            if frontend_freq_downsample_list is not None
            else [0, 1, 0]
        )

        input_channels = self.frontend_base_channels*self.frontend_multipliers_list[0]

        self.conv_inp = init(nn.Conv2d(stft_channels, input_channels, kernel_size=3, stride=1, padding=1))

        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(self.frontend_layers_list,self.frontend_multipliers_list)):
            output_channels = self.frontend_base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ConvBlock(output_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                next_channels = self.frontend_base_channels*self.frontend_multipliers_list[i+1]
                if self.frontend_freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(output_channels, next_channels, factor=4))
                elif self.frontend_freq_downsample_list[i]==2:
                    down_layers.append(DownsampleFreqConv(output_channels, next_channels, factor=2))
                elif self.frontend_freq_downsample_list[i]==3:
                    down_layers.append(DownsampleTimeConv(output_channels, next_channels, factor=2))
                else:
                    down_layers.append(DownsampleConv(output_channels, next_channels))

        self.down_layers = nn.ModuleList(down_layers)

    def add_feature(self, x, features, index):
        if features is not None:
            x = (x + features[index])/math.sqrt(2.)
        return x

    def forward(self, x, cond=None, features=None, gain=None, log_magnitude=False):

        x = x.to(memory_format=torch.channels_last)
        if features is not None:
            features = [el.to(memory_format=torch.channels_last) for el in features]
        if gain is not None:
            gain = gain.to(memory_format=torch.channels_last)

        x = self.conv_inp(x)

        if gain is not None:
            x = x*gain

        # DOWNSAMPLING
        new_features = []
        k = 0
        k_feat = 0
        for i,num_layers in enumerate(self.frontend_layers_list):
            for num in range(num_layers):
                x = self.add_feature(x, features, k_feat)
                k_feat = k_feat+1
                x = self.down_layers[k](x, cond)
                if log_magnitude:
                    print(f'Enc 2D Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1
                new_features.append(x)
            if i!=(len(self.frontend_layers_list)-1):
                x = self.down_layers[k](x)
                if log_magnitude:
                    print(f'Enc 2D Down Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # shape [batch, freq*time, dim]

        x = x.to(memory_format=torch.contiguous_format)
        new_features = [el.to(memory_format=torch.contiguous_format) for el in new_features]

        return x, new_features[::-1]


class UpFrontend(nn.Module):
    def __init__(
        self,
        frontend_layers_list,
        cond_dim=None,
        *,
        dim=512,
        hop=1024,
        fac=2,
        stft_channels=4,
        frontend_base_channels=64,
        frontend_multipliers_list=None,
        frontend_freq_downsample_list=None,
    ):
        super(UpFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        self.dim = dim
        self.frontend_base_channels = frontend_base_channels
        self.frontend_multipliers_list = list(
            frontend_multipliers_list
            if frontend_multipliers_list is not None
            else [1, 2, 4, 8]
        )
        self.frontend_freq_downsample_list = list(
            frontend_freq_downsample_list
            if frontend_freq_downsample_list is not None
            else [0, 1, 0]
        )

        input_channels = self.frontend_base_channels*self.frontend_multipliers_list[-1]

        self.freq_dim = (hop*(fac//2))//(4**self.frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**self.frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**self.frontend_freq_downsample_list.count(2))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(self.frontend_multipliers_list))[1:]+list(reversed(self.frontend_multipliers_list))[:1]
        freq_upsample_list = list(reversed(self.frontend_freq_downsample_list))
        up_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(reversed(self.frontend_layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ConvBlock(input_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                output_channels = self.frontend_base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels, factor=4))
                elif freq_upsample_list[i]==2:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels, factor=2))
                elif freq_upsample_list[i]==3:
                    up_layers.append(UpsampleTimeConv(input_channels, output_channels, factor=2))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels))
                input_channels = output_channels

        self.up_layers = nn.ModuleList(up_layers)

        self.norm_out = GroupNorm(input_channels, cond_dim=cond_dim)
        self.conv_out = zero_init(nn.Conv2d(input_channels, stft_channels, kernel_size=3, stride=1, padding=1))

    def add_feature(self, x, features, index):
        if features is not None:
            x = (x + features[index])/math.sqrt(2.)
        return x

    def forward(self, x, cond=None, features=None, gain=None, skip_output_layer=False, log_magnitude=False):

        x = x.permute(0,2,1).reshape(x.shape[0], self.dim, self.freq_dim, -1) # shape [batch, dim, freq, time]

        x = x.to(memory_format=torch.channels_last)
        if features is not None:
            features = [el.to(memory_format=torch.channels_last) for el in features]
        if gain is not None:
            gain = gain.to(memory_format=torch.channels_last)

        # UPSAMPLING
        new_features = []
        k = 0
        k_feat = 0
        for i,num_layers in enumerate(reversed(self.frontend_layers_list)):
            for num in range(num_layers):
                x = self.add_feature(x, features, k_feat)
                k_feat = k_feat+1
                x = self.up_layers[k](x, cond)
                if log_magnitude:
                    print(f'Dec 2D Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1
                new_features.append(x)
            if i!=(len(self.frontend_layers_list)-1):
                x = self.up_layers[k](x)
                if log_magnitude:
                    print(f'Dec 2D Up Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1

        if not skip_output_layer:
            x = self.norm_out(x, cond)
            if gain is not None:
                x = x*gain
            x = self.conv_out(x) # shape [batch, stft_channels, freq, time]

        x = x.to(memory_format=torch.contiguous_format)
        new_features = [el.to(memory_format=torch.contiguous_format) for el in new_features]

        return x, new_features[::-1]


class UNet(nn.Module):
    """Consistency Transformer autoencoder used by CodiCodec.

    Provides an encoder that maps spectrogram patches to latents and a decoder
    that reconstructs spectrograms conditioned on noise level. The model accepts
    arbitrary-length waveforms by padding internally to whole STFT chunks and
    cropping decoded audio back to the original length. RVQ is applied to
    stacked per-chunk summary latents when enabled.
    """
    def __init__(
        self,
        mixed_precision: bool = True,
        stereo: bool = True,
        sample_rate: int = 48000,
        audio_processor: AudioProcessor | None = None,
        dim: int = 512,
        head_dim: int = 128,
        heads: int | None = None,
        mlp_mult: int = 4,
        pos_emb: str = "learned",
        num_layers: int = 12,
        num_layers_encoder: int | None = None,
        cond_channels: int = 512,
        num_latents: int = 128,
        num_more_latents: int = 8,
        bottleneck_channels: int | None = None,
        frontend_base_channels: int = 64,
        frontend_multipliers_list: list[int] | None = None,
        frontend_layers_list: list[int] | None = None,
        frontend_encoder_layers_list: list[int] | None = None,
        frontend_freq_downsample_list: list[int] | None = None,
        spec_length: int = 32,
        sigma_data: float = 0.333,
        t_min: float = 0.006,
        t_max: float = 1.5666,
        default_denoising_steps: int = 5,
        quantizer_num_quantizers: int = 0,
        quantizer_codebook_size: int = 1024,
        quantizer_codebook_dim: int | None = None,
        rvq_dim: int = 64,
        quantizer_dropout: float = 0.0,
        quantizer_decay: float = 0.99,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 10,
        quantizer_threshold_ema_dead_code: int = 2,
        latent_tanh_scale: float = 1.0,
    ):
        super(UNet, self).__init__()
        if audio_processor is None:
            audio_processor = AudioProcessor(
                alpha_rescale=0.65,
                beta_rescale=0.34,
                hop_size=960,
                fac=2,
                center_pad=False,
            )
        hop = int(audio_processor.hop_size)
        fac = int(audio_processor.fac)
        frontend_multipliers_list = list(
            frontend_multipliers_list
            if frontend_multipliers_list is not None
            else [1, 2, 4, 8]
        )
        frontend_layers_list = list(
            frontend_layers_list
            if frontend_layers_list is not None
            else [3, 3, 3, 1]
        )
        frontend_encoder_layers_list = list(
            frontend_encoder_layers_list
            if frontend_encoder_layers_list is not None
            else frontend_layers_list
        )
        frontend_freq_downsample_list = list(
            frontend_freq_downsample_list
            if frontend_freq_downsample_list is not None
            else [0, 1, 0]
        )
        heads = int(heads if heads is not None else dim // head_dim)
        num_layers_encoder = int(
            num_layers_encoder if num_layers_encoder is not None else num_layers
        )
        bottleneck_channels = int(
            bottleneck_channels
            if bottleneck_channels is not None
            else 4
        )
        stft_channels = 4 if stereo else 2
        downsample_ratio = (
            (4 ** frontend_freq_downsample_list.count(0))
            * (4 ** frontend_freq_downsample_list.count(1))
            * (2 ** frontend_freq_downsample_list.count(2))
            * (2 ** frontend_freq_downsample_list.count(3))
        )
        data_length = (hop * (fac // 2) * spec_length) // downsample_ratio
        self.mixed_precision = bool(mixed_precision)
        self.stereo = bool(stereo)
        self.sample_rate = int(sample_rate)
        self.audio_processor = audio_processor
        self.hop = int(hop)
        self.fac = int(fac)
        self.stft_channels = int(stft_channels)
        self.alpha_rescale = float(audio_processor.alpha_rescale)
        self.beta_rescale = float(audio_processor.beta_rescale)
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.heads = int(heads)
        self.mlp_mult = int(mlp_mult)
        self.pos_emb = pos_emb
        self.num_layers = int(num_layers)
        self.num_layers_encoder = int(num_layers_encoder)
        self.cond_channels = int(cond_channels)
        self.num_latents = int(num_latents)
        self.num_more_latents = int(num_more_latents)
        self.bottleneck_channels = int(bottleneck_channels)
        self.frontend_base_channels = int(frontend_base_channels)
        self.frontend_multipliers_list = frontend_multipliers_list
        self.frontend_layers_list = frontend_layers_list
        self.frontend_encoder_layers_list = frontend_encoder_layers_list
        self.frontend_freq_downsample_list = frontend_freq_downsample_list
        self.spec_length = int(spec_length)
        self.downsample_ratio = int(downsample_ratio)
        self.data_length = int(data_length)
        self.sigma_data = float(sigma_data)
        # TrigFlow 时间区间 t ∈ (t_min, t_max] ⊂ (0, π/2]：t→0 近干净数据，t→π/2 近纯噪声。
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.default_denoising_steps = int(default_denoising_steps)
        self.quantizer_num_quantizers = int(quantizer_num_quantizers)
        self.latent_tanh_scale = float(latent_tanh_scale)
        # Each chunk's stacked latent (num_latents*bottleneck_channels dims) is
        # reshaped into rvq_tokens_per_chunk tokens of rvq_dim before quantization,
        # so RVQ codes operate on rvq_dim (e.g. 64) rather than the whole chunk.
        self.rvq_dim = int(rvq_dim)
        chunk_latent_dim = self.num_latents * self.bottleneck_channels
        if chunk_latent_dim % self.rvq_dim != 0:
            raise ValueError(
                f"num_latents*bottleneck_channels={chunk_latent_dim} must be divisible by rvq_dim={self.rvq_dim}"
            )
        self.rvq_tokens_per_chunk = chunk_latent_dim // self.rvq_dim

        self.freq_dim = (hop*(fac//2))//(4**frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(2))

        self.time_dim = spec_length//(2**frontend_freq_downsample_list.count(0))
        self.time_dim = self.time_dim//(2**frontend_freq_downsample_list.count(3))

        scale = float(dim) ** -0.5
        self.emb = PositionalEmbedding(embedding_size=cond_channels)
        self.emb_proj = nn.Sequential(init(nn.Linear(cond_channels, cond_channels)), nn.SiLU(), init(nn.Linear(cond_channels, cond_channels)), nn.SiLU(), init(nn.Linear(cond_channels, cond_channels)), nn.SiLU())

        self.latents = nn.Parameter(scale*torch.randn(1, num_latents, dim), requires_grad=True)
        self.mask_embedding = nn.Parameter(scale*torch.randn(1, self.freq_dim, 1, dim), requires_grad=True)

        self.gain_encoder = nn.Parameter(torch.ones(1, 1, hop*(fac//2), 1), requires_grad=True)
        self.gain_decoder = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(), nn.Linear(cond_channels, cond_channels), nn.SiLU(), zero_init(nn.Linear(cond_channels, hop*2*(fac//2))))

        self.frontend_encoder_down = DownFrontend(
            frontend_encoder_layers_list,
            stft_channels=stft_channels,
            frontend_base_channels=frontend_base_channels,
            frontend_multipliers_list=frontend_multipliers_list,
            frontend_freq_downsample_list=frontend_freq_downsample_list,
        ).to(memory_format=torch.channels_last)
        if num_more_latents>0:
            self.more_latents_encoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_encoder = None
        self.encoder = Transformer(dim, bottleneck_channels, training_length=data_length+num_latents+num_more_latents, dim=dim, num_layers=num_layers_encoder, heads=heads, mlp_mult=mlp_mult, pos_emb=pos_emb, zero_output_init=False)

        self.lat2patch_pre_decoder = init(nn.Linear(bottleneck_channels, dim))
        if num_more_latents>0:
            self.more_latents_pre_decoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_pre_decoder = None
        self.pre_decoder = Transformer(
            dim,
            dim,
            training_length=data_length + num_latents + num_more_latents,
            dim=dim,
            num_layers=num_layers_encoder,
            heads=heads,
            mlp_mult=mlp_mult,
            pos_emb=pos_emb,
            zero_output_init=False,
        )
        self.frontend_pre_decoder_up = UpFrontend(
            frontend_encoder_layers_list,
            dim=dim,
            hop=hop,
            fac=fac,
            stft_channels=stft_channels,
            frontend_base_channels=frontend_base_channels,
            frontend_multipliers_list=frontend_multipliers_list,
            frontend_freq_downsample_list=frontend_freq_downsample_list,
        ).to(memory_format=torch.channels_last)

        self.lat2patch = init(nn.Linear(bottleneck_channels, dim))
        self.frontend_decoder_down = DownFrontend(
            frontend_layers_list,
            cond_dim=cond_channels,
            stft_channels=stft_channels,
            frontend_base_channels=frontend_base_channels,
            frontend_multipliers_list=frontend_multipliers_list,
            frontend_freq_downsample_list=frontend_freq_downsample_list,
        ).to(memory_format=torch.channels_last)
        if num_more_latents>0:
            self.more_latents_decoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_decoder = None
        self.decoder = Transformer_Diffusion(
            dim,
            dim,
            training_length=data_length + num_latents + num_more_latents,
            cond_dim=cond_channels,
            dim=dim,
            num_layers=num_layers,
            heads=heads,
            mlp_mult=mlp_mult,
            pos_emb=pos_emb,
            zero_output_init=False,
        )
        self.frontend_decoder_up = UpFrontend(
            frontend_layers_list,
            cond_dim=cond_channels,
            dim=dim,
            hop=hop,
            fac=fac,
            stft_channels=stft_channels,
            frontend_base_channels=frontend_base_channels,
            frontend_multipliers_list=frontend_multipliers_list,
            frontend_freq_downsample_list=frontend_freq_downsample_list,
        ).to(memory_format=torch.channels_last)
        if self.quantizer_num_quantizers > 0:
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.rvq_dim,
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
            raise RuntimeError("CoDiCodec quantizer is disabled")

        continuous = self.encoder_forward(representation)
        if continuous.shape[-2] % self.num_latents != 0:
            raise ValueError(
                f"continuous latent length {continuous.shape[-2]} is not divisible by num_latents={self.num_latents}"
            )
        batch_size = continuous.shape[0]
        num_chunks = continuous.shape[-2] // self.num_latents
        # Stack each chunk's latents, then split into rvq_tokens_per_chunk tokens of
        # rvq_dim so RVQ codes operate on rvq_dim (paper-style chunk-internal reshape).
        rvq_seq = num_chunks * self.rvq_tokens_per_chunk
        continuous_tokens = continuous.reshape(batch_size, rvq_seq, self.rvq_dim)
        quantizer_input = continuous_tokens.detach() if detach_encoder else continuous_tokens
        quantized, codes, commitment_loss = self.quantizer(
            quantizer_input.transpose(1, 2).contiguous(),
            n_quantizers=n_quantizers,
        )
        discrete_tokens = quantized.transpose(1, 2).contiguous()  # [B, rvq_seq, rvq_dim]
        discrete = discrete_tokens.reshape(
            batch_size,
            num_chunks * self.num_latents,
            self.bottleneck_channels,
        )
        distill_loss = F.mse_loss(
            discrete_tokens.float(),
            continuous_tokens.detach().float(),
        )
        codebook_loss = continuous.new_zeros(())
        return QuantizedLatents(
            continuous=continuous,
            discrete=discrete,
            codes=codes,
            projected_latents=discrete_tokens,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            distill_loss=distill_loss,
        )

    def latent_from_codes(self, codes):
        if self.quantizer is None:
            raise RuntimeError("CoDiCodec quantizer is disabled")
        quantized, _ = self.quantizer.from_codes(codes)
        tokens = quantized.transpose(1, 2).contiguous()  # [B, rvq_seq, rvq_dim]
        if tokens.shape[1] % self.rvq_tokens_per_chunk != 0:
            raise ValueError(
                f"rvq token length {tokens.shape[1]} is not divisible by "
                f"rvq_tokens_per_chunk={self.rvq_tokens_per_chunk}"
            )
        num_chunks = tokens.shape[1] // self.rvq_tokens_per_chunk
        return tokens.reshape(
            tokens.shape[0],
            num_chunks * self.num_latents,
            self.bottleneck_channels,
        )

    def waveform_length_for_stft_frames(self, num_frames: int) -> int:
        frame_length = self.fac * self.hop
        return frame_length + self.hop * (int(num_frames) - 1)

    def _chunk_count_for_waveform_length(self, length: int) -> int:
        frame_length = self.fac * self.hop
        if length <= frame_length:
            num_frames = 1
        else:
            num_frames = math.ceil((int(length) - frame_length) / self.hop) + 1
        return max(1, math.ceil(num_frames / self.spec_length))

    def prepare_waveform(
        self,
        audio: torch.Tensor,
        num_chunks: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        parameter = next(self.parameters())
        audio = audio.to(device=parameter.device, dtype=parameter.dtype)
        if audio.ndim == 4:
            audio = audio.flatten(0, 1)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.ndim != 3:
            raise ValueError(f"expected waveform [B,T] or [B,C,T], got {tuple(audio.shape)}")

        original_length = int(audio.shape[-1])
        if self.stereo:
            if audio.shape[-2] == 1:
                audio = audio.repeat_interleave(2, dim=-2)
            elif audio.shape[-2] > 2:
                audio = audio[..., :2, :]
        else:
            audio = audio.mean(dim=-2, keepdim=True)

        chunks = self._chunk_count_for_waveform_length(original_length) if num_chunks is None else int(num_chunks)
        num_frames = chunks * self.spec_length
        target_length = self.waveform_length_for_stft_frames(num_frames)
        if audio.shape[-1] < target_length:
            audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        else:
            audio = audio[..., :target_length]
            original_length = min(original_length, target_length)
        return audio.contiguous(), original_length

    def encode_waveform(
        self,
        audio: torch.Tensor,
        *,
        quantize: bool = False,
        n_quantizers: int | None = None,
        num_chunks: int | None = None,
        detach_encoder: bool = True,
    ):
        audio, original_length = self.prepare_waveform(audio, num_chunks=num_chunks)
        representation = self.audio_processor.to_representation_encoder(audio)
        if quantize:
            encoded = self.quantize_representation(
                representation,
                detach_encoder=detach_encoder,
                n_quantizers=n_quantizers,
            )
        else:
            encoded = self.encoder_forward(representation)
        return encoded, original_length

    def decode_waveform(
        self,
        latents: torch.Tensor,
        *,
        original_length: int | None = None,
        denoising_steps: int | None = None,
        initial_state: str = "noise",
    ) -> torch.Tensor:
        representation = self.decode(
            latents,
            denoising_steps=denoising_steps,
            initial_state=initial_state,
        )
        waveform = self.audio_processor.to_waveform(representation, self.hop)
        if original_length is not None:
            waveform = waveform[..., : int(original_length)]
        return waveform

    def reconstruct_waveform(
        self,
        audio: torch.Tensor,
        *,
        n_quantizers: int | None = None,
        num_chunks: int | None = None,
        denoising_steps: int | None = None,
        initial_state: str = "noise",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared, original_length = self.prepare_waveform(audio, num_chunks=num_chunks)
        representation = self.audio_processor.to_representation_encoder(prepared)
        if n_quantizers is None:
            latents = self.encoder_forward(representation)
        else:
            latents = self.quantize_representation(
                representation,
                detach_encoder=True,
                n_quantizers=n_quantizers,
            ).discrete
        waveform = self.decode_waveform(
            latents,
            original_length=original_length,
            denoising_steps=denoising_steps,
            initial_state=initial_state,
        )
        return prepared[..., :original_length], waveform

    def encoder_forward(self, x, log_magnitude=False):
        """Encode STFT chunks to latents.

        Args:
            x: Input spectrogram chunks [B, C, F, T], T multiple of spec_length.
            log_magnitude: If True, prints intermediate magnitudes (debug).
        Returns:
            Tensor of bounded latents in (-1, 1) with shape
            [B, num_latents*(T/spec_length), bottleneck_channels].
        """
        spec_length = self.spec_length
        num_latents = self.num_latents
        assert x.shape[-1]%spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {spec_length}.'
        factor = None
        if x.shape[-1]>spec_length:
            x_ls = torch.split(x, spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        # Bound the latent to (-1, 1). Continuous-only sCM experiments can reduce
        # latent_tanh_scale to avoid early saturation while keeping RVQ-compatible
        # bounded latents.
        x = torch.tanh(x * self.latent_tanh_scale)
        if factor is not None:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
        return x

    @torch.compile(fullgraph=True, dynamic=False, mode='max-autotune-no-cudagraphs')
    def encoder_forward_fast(self, x, log_magnitude=False):
        """torch.compile-optimized variant of encoder_forward with same outputs."""
        spec_length = self.spec_length
        num_latents = self.num_latents
        assert x.shape[-1]%spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {spec_length}.'
        factor = None
        if x.shape[-1]>spec_length:
            x_ls = torch.split(x, spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        x = torch.tanh(x * self.latent_tanh_scale)
        if factor is not None:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
        return x

    def pre_decoder_forward(self, latents, log_magnitude=False):
        """Project latents and generate multi-scale features for the decoder.

        Args:
            latents: [B, L, dim] where L is multiple of num_latents.
        Returns:
            List of feature maps aligned with decoder frontend stages.
        """
        num_latents = self.num_latents
        dim = self.dim
        assert latents.shape[-2]%num_latents==0, f'Input shape {latents.shape[-2]} is not divisible by {num_latents}.'
        factor = None
        if latents.shape[-2]>num_latents:
            latents_ls = torch.split(latents, num_latents, dim=-2)
            factor = len(latents_ls)
            latents = torch.cat(latents_ls, dim=0)
        mask_embedding = self.mask_embedding.expand(latents.shape[0], -1, self.time_dim, -1).reshape(latents.shape[0], -1, dim)
        if self.more_latents_pre_decoder is not None:
            x = self.pre_decoder(mask_embedding, torch.cat((self.lat2patch_pre_decoder(latents), self.more_latents_pre_decoder.expand(latents.shape[0], -1, -1)), -2), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        else:
            x = self.pre_decoder(mask_embedding, self.lat2patch_pre_decoder(latents), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        features = self.frontend_pre_decoder_up(x, skip_output_layer=True, log_magnitude=log_magnitude)[1]
        if factor is not None:
            features = [torch.cat(torch.chunk(el, factor, dim=0), dim=-1) for el in features]
        return features

    def _normalize_t(self, t, batch_size, dtype, device):
        """Coerce a scalar/0-dim/[B] time into a ``[B]`` float tensor."""
        if t is None:
            t = self.t_max
        if isinstance(t, (int, float)):
            return torch.full((batch_size,), float(t), dtype=dtype, device=device)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.expand(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(f"t must be scalar or shape [{batch_size}], got {tuple(t.shape)}")
        return t

    def _split_features_for_chunks(self, features, factor: int):
        return [
            torch.cat(torch.chunk(feature, factor, dim=-1), dim=0)
            for feature in features
        ]

    def _decoder_network(self, x_t, t, latents, features=None, log_magnitude=False):
        """Raw consistency network ``F_theta(x_t / sigma_data, c_noise=t | latents)``.

        Returns the unparameterized network output (no TrigFlow ``c_skip``/``c_out``
        applied). Transformer work stays chunk-local: inputs are split into
        ``spec_length`` chunks, processed, then merged back. JVP-friendly -- this is
        the function differentiated by :func:`trigflow.consistency_tangent`.
        """
        spec_length = self.spec_length
        num_latents = self.num_latents
        assert x_t.shape[-1] % spec_length == 0, f"Input shape {x_t.shape[-1]} is not divisible by {spec_length}."
        assert latents.shape[-2] % num_latents == 0, f"Latent shape {latents.shape[-2]} is not divisible by {num_latents}."
        x_chunks = x_t.shape[-1] // spec_length
        latent_chunks = latents.shape[-2] // num_latents
        if x_chunks != latent_chunks:
            raise ValueError(f"x has {x_chunks} chunks but latents have {latent_chunks} chunks")

        t = self._normalize_t(t, batch_size=x_t.shape[0], dtype=x_t.dtype, device=x_t.device)
        factor = x_chunks
        if factor > 1:
            x_t = torch.cat(torch.split(x_t, spec_length, dim=-1), dim=0)
            latents = torch.cat(torch.split(latents, num_latents, dim=-2), dim=0)
            t = t.repeat(factor)
            if features is not None:
                features = self._split_features_for_chunks(features, factor)

        # c_noise(t) = t (identity); positional embedding keeps the Fourier scale low.
        emb_t = self.emb(t)
        time_emb = self.emb_proj(emb_t)

        gain = self.gain_decoder(emb_t).unsqueeze(-2).unsqueeze(-1) + 1.
        gain_inp, gain_out = torch.chunk(gain, 2, dim=-2)

        if features is None:
            features = self.pre_decoder_forward(latents, log_magnitude=log_magnitude)

        # c_in = 1 / sigma_data (constant in TrigFlow).
        x = x_t / self.sigma_data
        x, features_dec = self.frontend_decoder_down(x, cond=time_emb, features=features, gain=gain_inp, log_magnitude=log_magnitude)
        latent_tokens = self.lat2patch(latents)
        cond_length = x.shape[1] + latent_tokens.shape[1]
        if self.more_latents_decoder is not None:
            more_latents = self.more_latents_decoder.expand(x.shape[0], -1, -1)
            cond_length += more_latents.shape[1]
        else:
            more_latents = None
        time_emb_transformer = time_emb.unsqueeze(1).expand(-1, cond_length, -1)
        x = self.decoder(
            x,
            time_emb_transformer,
            latent_tokens,
            more_latents,
            skip_input_layer=True,
            skip_output_layer=True,
            print_magnitudes=log_magnitude,
        )
        x = self.frontend_decoder_up(x, cond=time_emb, features=features_dec, gain=gain_out, log_magnitude=log_magnitude)[0]
        if factor > 1:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-1)
        return x

    def denoise(self, x_t, t, latents, features=None, log_magnitude=False):
        """TrigFlow consistency function ``f_theta(x_t, t) = cos(t) x_t - sin(t) sigma_data F_theta``.

        Maps a noised input at time ``t`` directly to the clean estimate ``x0``.
        Satisfies the boundary condition ``f_theta(x, 0) = x``.
        """
        f = self._decoder_network(x_t, t, latents, features=features, log_magnitude=log_magnitude)
        t = self._normalize_t(t, batch_size=x_t.shape[0], dtype=x_t.dtype, device=x_t.device)
        t = t.reshape(-1, 1, 1, 1)
        return torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * f

    def _decode_t_schedule(self, n_steps: int) -> list[float]:
        """Denoising times for the multistep consistency sampler, high noise first.

        The two-step recipe from arXiv:2410.11081 uses an intermediate time of
        ``1.1`` radians; other step counts fall back to a linear-in-``t`` schedule.
        """
        t_max, t_min = self.t_max, self.t_min
        if n_steps <= 1:
            return [t_max]
        if n_steps == 2:
            return [t_max, min(max(1.1, t_min), t_max)]
        return torch.linspace(t_max, t_min, n_steps).tolist()

    def _decode_offset_schedule(self, n_steps: int, n_chunks: int) -> list[int]:
        """Per-step frame offsets for the sliding-window decoder.

        Each intermediate denoising step decodes on a chunk grid shifted by a
        sub-chunk frame offset, so that the (chunk-local) seams land at different
        positions each step and get re-decoded inside chunk interiors on the next
        step; the final step returns to the aligned grid (offset 0) so the output
        sits on the standard boundaries.

        Offsets are restricted to multiples of ``spec_length //
        gcd(spec_length, rvq_tokens_per_chunk)`` frames so the matching 64-d
        RVQ/LLM token offset is an integer. This keeps the audio chunk and its
        conditioning token window temporally aligned without slicing inside a
        64-d token. With spec_length=32 and rvq_tokens_per_chunk=8, valid phases
        are {0, 4, 8, ..., 28} frames ({0, 1, 2, ..., 7} 64-d tokens).

        Returns a list of length ``n_steps`` with the last entry equal to 0.
        """
        if n_steps <= 1 or n_chunks <= 1:
            return [0] * max(1, n_steps)
        g = math.gcd(self.spec_length, self.rvq_tokens_per_chunk)
        step_frames = self.spec_length // g
        phases = [step_frames * k for k in range(self.spec_length // step_frames)]
        nonzero = [p for p in phases if p != 0] or [0]
        offsets = [nonzero[i % len(nonzero)] for i in range(n_steps - 1)]
        offsets.append(0)
        return offsets

    def decode(self, latents, denoising_steps=None, slide=True, initial_state: str = "noise"):
        """Decode latents to an STFT representation via multistep consistency sampling.

        Starts from the TrigFlow prior ``N(0, sigma_data^2)`` at ``t_max`` and
        alternates ``denoise`` (jump to the ``x0`` estimate) with renoising to the
        next scheduled time, returning the final ``x0`` estimate.

        When ``slide`` is True (and there are multiple chunks and steps), each
        intermediate step decodes on a sub-chunk-shifted grid via :func:`
        _decode_offset_schedule`, cycling the (chunk-local) seam positions so they
        are absorbed into chunk interiors on subsequent steps. The representation
        and its conditioning latents are rolled together so each audio chunk keeps
        its temporally-aligned latent window; the final step uses the aligned grid.
        Note: the decoder is chunk-local and was trained on frame-0-anchored chunks,
        so this is a best-effort seam-mitigation heuristic -- verify its benefit on
        a trained checkpoint. Pass ``slide=False`` for the plain aligned sampler.
        """
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        denoising_steps = int(denoising_steps)
        num_chunks = latents.shape[-2] // self.num_latents
        rep_shape = (
            latents.shape[0],
            self.stft_channels,
            self.hop * (self.fac // 2),
            self.spec_length * num_chunks,
        )
        if initial_state == "noise":
            x = torch.randn(rep_shape, dtype=latents.dtype, device=latents.device) * self.sigma_data
        elif initial_state == "zero":
            x = torch.zeros(rep_shape, dtype=latents.dtype, device=latents.device)
        else:
            raise ValueError(f"Unsupported decode initial_state={initial_state!r}")
        times = self._decode_t_schedule(denoising_steps)
        offsets = (
            self._decode_offset_schedule(denoising_steps, num_chunks)
            if slide
            else [0] * len(times)
        )
        g = math.gcd(self.spec_length, self.rvq_tokens_per_chunk)
        token_ratio = self.rvq_tokens_per_chunk // g
        frame_unit = self.spec_length // g

        def shift_left(tensor, amount, dim, fill):
            """Shift toward index 0 by ``amount`` along ``dim``, padding the tail.

            Non-circular (unlike torch.roll): the wrapped region is replaced by
            ``fill`` ('noise' -> fresh prior-scale Gaussian, 'edge' -> last slice),
            avoiding end-to-start audio contamination on long sequences.
            """
            if amount == 0:
                return tensor
            kept = tensor.narrow(dim, amount, tensor.shape[dim] - amount)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = amount
            if fill == "noise":
                tail = torch.randn(pad_shape, dtype=tensor.dtype, device=tensor.device) * self.sigma_data
            else:  # 'edge': repeat the last valid slice
                tail = tensor.narrow(dim, tensor.shape[dim] - 1, 1).expand(*pad_shape)
            return torch.cat([kept, tail], dim=dim)

        def shift_right(tensor, amount, dim):
            """Inverse of shift_left for realigning the decoded x0 (pads the head)."""
            if amount == 0:
                return tensor
            kept = tensor.narrow(dim, 0, tensor.shape[dim] - amount)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = amount
            head = tensor.narrow(dim, 0, 1).expand(*pad_shape)
            return torch.cat([head, kept], dim=dim)

        def shift_latent_tokens_left(latent_tensor, amount):
            if amount == 0:
                return latent_tensor
            latent_chunks = latent_tensor.shape[-2] // self.num_latents
            rvq_tokens = latent_tensor.reshape(
                latent_tensor.shape[0],
                latent_chunks * self.rvq_tokens_per_chunk,
                self.rvq_dim,
            )
            shifted = shift_left(rvq_tokens, amount, -2, fill="edge")
            return shifted.reshape(
                latent_tensor.shape[0],
                latent_chunks * self.num_latents,
                self.bottleneck_channels,
            )

        # Precompute aligned features once; recompute only when a shift is active.
        aligned_features = self.pre_decoder_forward(latents)
        x0 = x
        for i, t_value in enumerate(times):
            offset = offsets[i]
            if offset:
                token_offset = (offset // frame_unit) * token_ratio
                x_in = shift_left(x, offset, -1, fill="noise")
                latents_in = shift_latent_tokens_left(latents, token_offset)
                features_in = None  # conditioning changed; rebuild inside denoise
            else:
                x_in, latents_in, features_in = x, latents, aligned_features
            x0_shifted = self.denoise(x_in, t_value, latents_in, features=features_in)
            x0 = shift_right(x0_shifted, offset, -1) if offset else x0_shifted
            if i < len(times) - 1:
                t_next = times[i + 1]
                noise = torch.randn_like(x0) * self.sigma_data
                x = math.cos(t_next) * x0 + math.sin(t_next) * noise
        return x0
