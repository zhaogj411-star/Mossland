from . import audio as audio_utils
from .utils import get_sigma_continuous, get_c, reverse_step, distribute, preprocess_parallel_input, postprocess_parallel_input, preprocess_parallel_features
from .tasks import TASK_NAMES
from .quantize import ResidualVectorQuantize

from dataclasses import dataclass
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, Transformer_Diffusion
from .transformer_layers import block_causal_attention_mask


torch.backends.cudnn.benchmark = True


@dataclass
class BottleneckOutput:
    latents: torch.Tensor
    continuous: torch.Tensor
    codes: torch.Tensor | None = None
    projected_latents: torch.Tensor | None = None
    commitment_loss: torch.Tensor | None = None
    codebook_loss: torch.Tensor | None = None
    distill_loss: torch.Tensor | None = None
    packed_continuous: torch.Tensor | None = None
    packed_quantized: torch.Tensor | None = None
    quantized: bool = False
    n_quantizers: int | None = None
    rvq_tokens_per_chunk: int | None = None


def waveform_length_for_stft_frames(num_frames: int, hop: int, fac: int) -> int:
    frame_length = fac * hop
    return frame_length + hop * (num_frames - 1)


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


class LocalCausalSelfAttention(nn.Module):
    """Small local-causal SDPA block for codec token streams."""

    def __init__(
        self,
        dim: int,
        heads: int,
        context: int | None = None,
        query_chunk_size: int = 1500,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got dim={dim}, heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.context = None if context is None else int(context)
        if self.context is not None and self.context <= 0:
            raise ValueError(f"context must be positive or None, got {context}")
        self.query_chunk_size = int(query_chunk_size)
        self.to_qkv = init(nn.Linear(dim, 3 * dim, bias=False))
        self.to_out = zero_init(nn.Linear(dim, dim, bias=False))

    def _attention_bias(
        self,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_positions = torch.arange(q_start, q_end, device=device)
        k_positions = torch.arange(k_start, k_end, device=device)
        delta = q_positions.view(-1, 1) - k_positions.view(1, -1)
        mask = delta >= 0
        if self.context is not None:
            mask = mask & (delta < self.context)
        return mask.view(1, 1, q_end - q_start, k_end - k_start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        head_dim = self.dim // self.heads
        qkv = self.to_qkv(x).reshape(batch, length, 3, self.heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if length <= self.query_chunk_size:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                self._attention_bias(0, length, 0, length, x.device),
                dropout_p=0.0,
            )
        else:
            chunks = []
            for q_start in range(0, length, self.query_chunk_size):
                q_end = min(length, q_start + self.query_chunk_size)
                k_start = 0 if self.context is None else max(0, q_start - self.context + 1)
                k_end = q_end
                chunks.append(
                    F.scaled_dot_product_attention(
                        q[:, :, q_start:q_end],
                        k[:, :, k_start:k_end],
                        v[:, :, k_start:k_end],
                        self._attention_bias(q_start, q_end, k_start, k_end, x.device),
                        dropout_p=0.0,
                    )
                )
            out = torch.cat(chunks, dim=2)
        out = out.transpose(1, 2).reshape(batch, length, self.dim)
        return self.to_out(out)


class LocalCausalTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int = 4, context: int | None = None):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = LocalCausalSelfAttention(dim=dim, heads=heads, context=context)
        self.norm_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_mult)
        self.ff = nn.Sequential(
            init(nn.Linear(dim, hidden, bias=False)),
            nn.GELU(),
            zero_init(nn.Linear(hidden, dim, bias=False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class DownFrontend(nn.Module):
    def __init__(
        self,
        frontend_layers_list,
        frontend_base_channels,
        frontend_multipliers_list,
        frontend_freq_downsample_list,
        stft_channels,
        cond_dim=None,
    ):
        super(DownFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        self.frontend_multipliers_list = frontend_multipliers_list
        self.frontend_freq_downsample_list = frontend_freq_downsample_list

        input_channels = frontend_base_channels*self.frontend_multipliers_list[0]

        self.conv_inp = init(nn.Conv2d(stft_channels, input_channels, kernel_size=3, stride=1, padding=1))

        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(self.frontend_layers_list,self.frontend_multipliers_list)):
            output_channels = frontend_base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ConvBlock(output_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                if self.frontend_freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(output_channels, frontend_base_channels*self.frontend_multipliers_list[i+1], factor=4))
                elif self.frontend_freq_downsample_list[i]==2:
                    down_layers.append(DownsampleFreqConv(output_channels, frontend_base_channels*self.frontend_multipliers_list[i+1], factor=2))
                elif self.frontend_freq_downsample_list[i]==3:
                    down_layers.append(DownsampleTimeConv(output_channels, frontend_base_channels*self.frontend_multipliers_list[i+1], factor=2))
                else:
                    down_layers.append(DownsampleConv(output_channels, frontend_base_channels*self.frontend_multipliers_list[i+1]))

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
        frontend_base_channels,
        frontend_multipliers_list,
        frontend_freq_downsample_list,
        stft_channels,
        dim,
        hop,
        fac,
        cond_dim=None,
    ):
        super(UpFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        self.dim = dim

        input_channels = frontend_base_channels*frontend_multipliers_list[-1]

        self.freq_dim = (hop*(fac//2))//(4**frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(2))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(frontend_multipliers_list))[1:]+list(reversed(frontend_multipliers_list))[:1]
        freq_upsample_list = list(reversed(frontend_freq_downsample_list))
        up_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(reversed(self.frontend_layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ConvBlock(input_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                output_channels = frontend_base_channels*multiplier
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


class MosslandCodecTransformer(nn.Module):
    """带多任务条件的 Mossland consistency Transformer autoencoder。

    Provides an encoder that maps spectrogram patches to latents and a decoder
    that reconstructs spectrograms conditioned on noise level and Mossland task
    metadata. Supports both parallel and autoregressive decoding schedules and
    RVQ bottlenecks.
    """
    def __init__(
        self,
        torch_compile_cache_dir: str | None = None,
        mixed_precision: bool = True,
        stereo: bool = True,
        default_time_prompt: float = 0.4,
        default_denoising_steps_parallel: int = 5,
        default_denoising_steps_ar: int = 2,
        hop: int = 1024,
        fac: int = 2,
        sample_rate: int = 48000,
        alpha_rescale: float = 0.65,
        beta_rescale: float = 0.34,
        dim: int = 512,
        head_dim: int = 128,
        mlp_mult: int = 4,
        pos_emb: str = "learned",
        num_layers: int = 12,
        num_layers_encoder: int | None = None,
        cond_channels: int = 512,
        num_latents: int = 128,
        num_more_latents: int = 8,
        bottleneck_channels: int | None = None,
        rvq_num_quantizers: int = 0,
        rvq_codebook_size: int = 1024,
        rvq_codebook_dim: int | None = 8,
        rvq_quantizer_dropout: float = 0.0,
        rvq_quantizer_dropout_cutoff_index: int = 0,
        rvq_quantizer_dropout_multiple_of: int = 1,
        rvq_decay: float = 0.8,
        rvq_kmeans_init: bool = True,
        rvq_kmeans_iters: int = 10,
        rvq_threshold_ema_dead_code: int = 2,
        rvq_token_dim: int = 64,
        rvq_tokens_per_chunk: int | None = None,
        frontend_base_channels: int = 64,
        frontend_multipliers_list: list[int] | None = None,
        frontend_layers_list: list[int] | None = None,
        frontend_encoder_layers_list: list[int] | None = None,
        frontend_freq_downsample_list: list[int] | None = None,
        spec_length: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        max_batch_size_encode: int = 64,
        max_batch_size_decode: int = 32,
        sigma_rescale: float = 0.8,
        load_path_inference_default: str | None = None,
        task_names: list[str] | tuple[str, ...] | None = None,
        task_embedding_init_std: float = 0.02,
    ):
        super().__init__()
        if torch_compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = torch_compile_cache_dir
        if frontend_multipliers_list is None:
            frontend_multipliers_list = [1, 2, 4, dim // frontend_base_channels]
        if frontend_layers_list is None:
            frontend_layers_list = [3, 3, 3, 1]
        if frontend_encoder_layers_list is None:
            frontend_encoder_layers_list = list(frontend_layers_list)
        if frontend_freq_downsample_list is None:
            frontend_freq_downsample_list = [0, 1, 0]
        if num_layers_encoder is None:
            num_layers_encoder = num_layers
        if task_names is None:
            task_names = TASK_NAMES
        task_names = tuple(task_names)
        if not task_names:
            raise ValueError("task_names must not be empty")

        self.torch_compile_cache_dir = torch_compile_cache_dir
        self.task_names = task_names
        self.mixed_precision = bool(mixed_precision)
        self.stereo = bool(stereo)
        self.default_time_prompt = float(default_time_prompt)
        self.default_denoising_steps_parallel = int(default_denoising_steps_parallel)
        self.default_denoising_steps_ar = int(default_denoising_steps_ar)
        self.hop = int(hop)
        self.fac = int(fac)
        self.sample_rate = int(sample_rate)
        self.alpha_rescale = float(alpha_rescale)
        self.beta_rescale = float(beta_rescale)
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.heads = self.dim // self.head_dim
        self.mlp_mult = int(mlp_mult)
        self.pos_emb = pos_emb
        self.num_layers = int(num_layers)
        self.num_layers_encoder = int(num_layers_encoder)
        self.cond_channels = int(cond_channels)
        self.num_latents = int(num_latents)
        self.num_more_latents = int(num_more_latents)
        self.rvq_num_quantizers = int(rvq_num_quantizers)
        self.rvq_codebook_size = int(rvq_codebook_size)
        self.rvq_codebook_dim = rvq_codebook_dim
        self.rvq_quantizer_dropout = float(rvq_quantizer_dropout)
        self.rvq_quantizer_dropout_cutoff_index = int(rvq_quantizer_dropout_cutoff_index)
        self.rvq_quantizer_dropout_multiple_of = int(rvq_quantizer_dropout_multiple_of)
        self.rvq_decay = float(rvq_decay)
        self.rvq_kmeans_init = bool(rvq_kmeans_init)
        self.rvq_kmeans_iters = int(rvq_kmeans_iters)
        self.rvq_threshold_ema_dead_code = int(rvq_threshold_ema_dead_code)
        if bottleneck_channels is None:
            bottleneck_channels = 4
        self.bottleneck_channels = int(bottleneck_channels)
        self.rvq_token_dim = int(rvq_token_dim)
        if self.rvq_token_dim <= 0:
            raise ValueError(f"rvq_token_dim must be positive, got {self.rvq_token_dim}")
        total_chunk_dim = self.num_latents * self.bottleneck_channels
        if rvq_tokens_per_chunk is None:
            if total_chunk_dim % self.rvq_token_dim != 0:
                raise ValueError(
                    f"num_latents*bottleneck_channels={total_chunk_dim} must be divisible "
                    f"by rvq_token_dim={self.rvq_token_dim}"
                )
            rvq_tokens_per_chunk = total_chunk_dim // self.rvq_token_dim
        self.rvq_tokens_per_chunk = int(rvq_tokens_per_chunk)
        if self.rvq_tokens_per_chunk <= 0:
            raise ValueError(
                f"rvq_tokens_per_chunk must be positive, got {self.rvq_tokens_per_chunk}"
            )
        if total_chunk_dim != self.rvq_tokens_per_chunk * self.rvq_token_dim:
            raise ValueError(
                "RVQ tokenization must preserve each chunk exactly: "
                f"num_latents*bottleneck_channels={total_chunk_dim}, "
                f"rvq_tokens_per_chunk*rvq_token_dim="
                f"{self.rvq_tokens_per_chunk * self.rvq_token_dim}"
            )
        if self.rvq_token_dim % self.bottleneck_channels != 0:
            raise ValueError(
                f"rvq_token_dim={self.rvq_token_dim} must be divisible by "
                f"bottleneck_channels={self.bottleneck_channels}"
            )
        self.rvq_latents_per_token = self.rvq_token_dim // self.bottleneck_channels
        if self.rvq_tokens_per_chunk * self.rvq_latents_per_token != self.num_latents:
            raise ValueError(
                f"rvq_tokens_per_chunk={self.rvq_tokens_per_chunk} and "
                f"rvq_latents_per_token={self.rvq_latents_per_token} do not cover "
                f"num_latents={self.num_latents}"
            )
        self.frontend_base_channels = int(frontend_base_channels)
        self.frontend_multipliers_list = list(frontend_multipliers_list)
        self.frontend_layers_list = list(frontend_layers_list)
        self.frontend_encoder_layers_list = list(frontend_encoder_layers_list)
        self.frontend_freq_downsample_list = list(frontend_freq_downsample_list)
        self.spec_length = int(spec_length)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.max_batch_size_encode = int(max_batch_size_encode)
        self.max_batch_size_decode = int(max_batch_size_decode)
        self.sigma_rescale = float(sigma_rescale)
        self.load_path_inference_default = load_path_inference_default
        self.task_embedding_init_std = float(task_embedding_init_std)
        self.stft_channels = 4 if self.stereo else 2
        self.downsample_ratio = (
            (4**self.frontend_freq_downsample_list.count(0))
            * (4**self.frontend_freq_downsample_list.count(1))
            * (2**self.frontend_freq_downsample_list.count(2))
            * (2**self.frontend_freq_downsample_list.count(3))
        )
        self.data_length = (
            self.hop * (self.fac // 2) * self.spec_length
        ) // self.downsample_ratio

        self.freq_dim = (self.hop*(self.fac//2))//(4**self.frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**self.frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**self.frontend_freq_downsample_list.count(2))

        self.time_dim = self.spec_length//(2**self.frontend_freq_downsample_list.count(0))
        self.time_dim = self.time_dim//(2**self.frontend_freq_downsample_list.count(3))

        scale = float(self.dim) ** -0.5
        self.emb = PositionalEmbedding(embedding_size=self.cond_channels)
        self.emb_proj = nn.Sequential(init(nn.Linear(self.cond_channels, self.cond_channels)), nn.SiLU(), init(nn.Linear(self.cond_channels, self.cond_channels)), nn.SiLU(), init(nn.Linear(self.cond_channels, self.cond_channels)), nn.SiLU())
        self.task_to_idx = {name: idx for idx, name in enumerate(self.task_names)}
        self.task_embedding = nn.Embedding(len(self.task_names), self.cond_channels)
        nn.init.normal_(
            self.task_embedding.weight,
            mean=0.0,
            std=self.task_embedding_init_std,
        )

        self.latents = nn.Parameter(scale*torch.randn(1, self.num_latents, self.dim), requires_grad=True)
        self.mask_embedding = nn.Parameter(scale*torch.randn(1, self.freq_dim, 1, self.dim), requires_grad=True)

        self.gain_encoder = nn.Parameter(torch.ones(1, 1, self.hop*(self.fac//2), 1), requires_grad=True)
        self.gain_decoder = nn.Sequential(nn.Linear(self.cond_channels, self.cond_channels), nn.SiLU(), nn.Linear(self.cond_channels, self.cond_channels), nn.SiLU(), zero_init(nn.Linear(self.cond_channels, self.hop*2*(self.fac//2))))

        self.frontend_encoder_down = DownFrontend(
            self.frontend_encoder_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
        ).to(memory_format=torch.channels_last)
        if self.num_more_latents>0:
            self.more_latents_encoder = nn.Parameter(scale*torch.randn(1, self.num_more_latents, self.dim), requires_grad=True)
        else:
            self.more_latents_encoder = None
        self.encoder = Transformer(self.dim, self.bottleneck_channels, training_length=self.data_length+self.num_latents+self.num_more_latents, dim=self.dim, num_layers=self.num_layers_encoder, heads=self.heads, mlp_mult=self.mlp_mult, pos_emb=self.pos_emb)
        self.lat2patch_pre_decoder = init(nn.Linear(self.bottleneck_channels, self.dim))
        if self.num_more_latents>0:
            self.more_latents_pre_decoder = nn.Parameter(scale*torch.randn(1, self.num_more_latents, self.dim), requires_grad=True)
        else:
            self.more_latents_pre_decoder = None
        self.pre_decoder = Transformer(self.dim, self.dim, training_length=self.data_length+self.num_latents+self.num_more_latents, dim=self.dim, num_layers=self.num_layers_encoder, heads=self.heads, mlp_mult=self.mlp_mult, pos_emb=self.pos_emb)
        self.frontend_pre_decoder_up = UpFrontend(
            self.frontend_encoder_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
            self.dim,
            self.hop,
            self.fac,
        ).to(memory_format=torch.channels_last)

        self.lat2patch = init(nn.Linear(self.bottleneck_channels, self.dim))
        self.frontend_decoder_down = DownFrontend(
            self.frontend_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
            cond_dim=self.cond_channels,
        ).to(memory_format=torch.channels_last)
        if self.num_more_latents>0:
            self.more_latents_decoder = nn.Parameter(scale*torch.randn(1, self.num_more_latents, self.dim), requires_grad=True)
        else:
            self.more_latents_decoder = None
        self.decoder = Transformer_Diffusion(self.dim, self.dim, training_length=(self.data_length+self.num_latents+self.num_more_latents)*2, cond_dim=self.cond_channels, dim=self.dim, num_layers=self.num_layers, heads=self.heads, mlp_mult=self.mlp_mult, pos_emb=self.pos_emb)
        self.frontend_decoder_up = UpFrontend(
            self.frontend_layers_list,
            self.frontend_base_channels,
            self.frontend_multipliers_list,
            self.frontend_freq_downsample_list,
            self.stft_channels,
            self.dim,
            self.hop,
            self.fac,
            cond_dim=self.cond_channels,
        ).to(memory_format=torch.channels_last)

        self.rvq = None
        if self.rvq_num_quantizers > 0:
            self.rvq = ResidualVectorQuantize(
                input_dim=self.rvq_token_dim,
                n_codebooks=self.rvq_num_quantizers,
                codebook_size=self.rvq_codebook_size,
                codebook_dim=self.rvq_codebook_dim,
                quantizer_dropout=self.rvq_quantizer_dropout,
                decay=self.rvq_decay,
                kmeans_init=self.rvq_kmeans_init,
                kmeans_iters=self.rvq_kmeans_iters,
                threshold_ema_dead_code=self.rvq_threshold_ema_dead_code,
            )

    def get_attn_mask(self, x):
        """Generate attention mask for autoregressive decoding."""
        if x.shape[-1]==(2*self.spec_length):
            return block_causal_attention_mask(
                self.data_length + self.num_latents + self.num_more_latents
            )
        else:
            raise ValueError(f'Invalid data length. Must be {(2*self.spec_length)} while it is {x.shape[-1]}.')

    def to_representation_encoder(self, x):
        return audio_utils.to_representation_encoder(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def to_representation(self, x):
        return audio_utils.to_representation(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def to_waveform(self, x):
        return audio_utils.to_waveform(
            x,
            self.hop,
            self.fac,
            alpha_rescale=self.alpha_rescale,
            beta_rescale=self.beta_rescale,
        )

    def prepare_audio_batch(self, batch: torch.Tensor, num_chunks: int = 2) -> torch.Tensor:
        batch = batch.to(next(self.parameters()).dtype)
        if not self.stereo:
            batch = batch.mean(dim=-2, keepdim=True)

        num_chunks = int(num_chunks)
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        target_length = waveform_length_for_stft_frames(
            num_chunks * self.spec_length,
            hop=self.hop,
            fac=self.fac,
        )
        if batch.shape[-1] < target_length:
            batch = F.pad(batch, (0, target_length - batch.shape[-1]))
        else:
            batch = batch[..., :target_length]
        return batch

    @torch.no_grad()
    def generate_waveform(
        self,
        src: torch.Tensor,
        task_id: str = "reconstruct",
        dont_quantize: bool = True,
        n_quantizers: int | None = None,
    ):
        src = self.prepare_audio_batch(src)
        representation = self.to_representation_encoder(src)
        encoded = self.encode_bottleneck(
            representation,
            quantize=not dont_quantize,
            n_quantizers=n_quantizers,
        )
        latents = encoded.latents
        features = self.pre_decoder_forward(latents)
        noise = torch.randn_like(representation) * self.sigma_max
        generated = self.decoder_forward(
            noise,
            latents,
            features=features,
            sigma_left=self.sigma_max,
            sigma_right=self.sigma_max,
            output="both",
            task_id=task_id,
        )
        waveform = self.to_waveform(generated[..., : representation.shape[-1]])
        return src.detach().cpu(), waveform.detach().cpu()

    def pack_rvq_tokens(self, latents: torch.Tensor) -> torch.Tensor:
        """Stack per-chunk summary slots into 64-d RVQ tokens.

        The Transformer keeps the original CoDiCodec-style latent layout
        [B, chunks*num_latents, bottleneck_channels]. For the requested RVQ
        layout, each chunk is reshaped into rvq_tokens_per_chunk tokens of
        rvq_token_dim channels, e.g. 128*4 -> 8*64.
        """
        if latents.shape[-2] % self.num_latents != 0:
            raise ValueError(
                f"latents length {latents.shape[-2]} must be divisible by "
                f"num_latents={self.num_latents}"
            )
        batch, total_latents, channels = latents.shape
        if channels != self.bottleneck_channels:
            raise ValueError(
                f"expected bottleneck_channels={self.bottleneck_channels}, got {channels}"
            )
        num_chunks = total_latents // self.num_latents
        latents = latents.reshape(
            batch,
            num_chunks,
            self.rvq_tokens_per_chunk,
            self.rvq_latents_per_token,
            channels,
        )
        return latents.reshape(
            batch,
            num_chunks * self.rvq_tokens_per_chunk,
            self.rvq_token_dim,
        ).transpose(1, 2).contiguous()

    def unpack_rvq_tokens(self, packed: torch.Tensor) -> torch.Tensor:
        """Inverse of pack_rvq_tokens()."""
        batch, packed_channels, total_tokens = packed.shape
        if packed_channels != self.rvq_token_dim:
            raise ValueError(
                f"expected rvq_token_dim={self.rvq_token_dim}, got {packed_channels}"
            )
        if total_tokens % self.rvq_tokens_per_chunk != 0:
            raise ValueError(
                f"packed token length {total_tokens} must be divisible by "
                f"rvq_tokens_per_chunk={self.rvq_tokens_per_chunk}"
            )
        num_chunks = total_tokens // self.rvq_tokens_per_chunk
        return packed.transpose(1, 2).contiguous().reshape(
            batch,
            num_chunks,
            self.rvq_tokens_per_chunk,
            self.rvq_latents_per_token,
            self.bottleneck_channels,
        ).reshape(
            batch,
            num_chunks * self.num_latents,
            self.bottleneck_channels,
        )

    def encode_bottleneck(
        self,
        x: torch.Tensor,
        quantize: bool = False,
        detach_encoder: bool = False,
        n_quantizers: int | None = None,
        log_magnitude: bool = False,
    ) -> BottleneckOutput:
        if self.rvq is not None:
            continuous = self.encoder_forward(
                x,
                dont_quantize=True,
                log_magnitude=log_magnitude,
            )
            packed_continuous = self.pack_rvq_tokens(continuous)
            if not quantize:
                return BottleneckOutput(
                    latents=continuous,
                    continuous=continuous,
                    packed_continuous=packed_continuous,
                    quantized=False,
                    rvq_tokens_per_chunk=self.rvq_tokens_per_chunk,
                )
            quantizer_input = packed_continuous.detach() if detach_encoder else packed_continuous
            (
                packed_quantized,
                codes,
                _rvq_aux_loss,
            ) = self.rvq(quantizer_input, n_quantizers=n_quantizers)
            quantized = self.unpack_rvq_tokens(packed_quantized)
            quantization_error = F.mse_loss(
                packed_quantized.float(),
                packed_continuous.detach().float(),
            )
            zero = continuous.new_zeros(())
            return BottleneckOutput(
                latents=quantized,
                continuous=continuous,
                codes=codes,
                projected_latents=packed_quantized,
                commitment_loss=quantization_error,
                codebook_loss=zero,
                distill_loss=zero,
                packed_continuous=packed_continuous,
                packed_quantized=packed_quantized,
                quantized=True,
                n_quantizers=n_quantizers,
                rvq_tokens_per_chunk=self.rvq_tokens_per_chunk,
            )

        latents = self.encoder_forward(
            x,
            dont_quantize=not quantize,
            log_magnitude=log_magnitude,
        )
        return BottleneckOutput(
            latents=latents,
            continuous=latents,
            quantized=False,
            n_quantizers=None,
        )

    def latent_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.rvq is None:
            raise RuntimeError("No RVQ bottleneck is enabled")
        packed_quantized, _ = self.rvq.from_codes(codes)
        return self.unpack_rvq_tokens(packed_quantized)

    def encode_latents_from_representation(
        self,
        x: torch.Tensor,
        quantize: bool = False,
        detach_encoder: bool = False,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        return self.encode_bottleneck(
            x,
            quantize=quantize,
            detach_encoder=detach_encoder,
            n_quantizers=n_quantizers,
        ).latents

    def encode_packed_from_representation(
        self,
        x: torch.Tensor,
        quantize: bool = False,
        detach_encoder: bool = False,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_bottleneck(
            x,
            quantize=quantize,
            detach_encoder=detach_encoder,
            n_quantizers=n_quantizers,
        )
        packed = encoded.packed_quantized if quantize else encoded.packed_continuous
        if packed is None:
            packed = self.pack_rvq_tokens(encoded.latents)
        return packed

    def encode_codes_from_representation(
        self,
        x: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        if self.rvq is not None:
            continuous = self.encoder_forward(x, dont_quantize=True)
            packed = self.pack_rvq_tokens(continuous)
            return self.rvq.encode_codes(packed, n_quantizers=n_quantizers)
        encoded = self.encode_bottleneck(
            x,
            quantize=True,
            n_quantizers=n_quantizers,
        )
        if encoded.codes is None:
            raise RuntimeError("No RVQ bottleneck is enabled")
        return encoded.codes

    def encoder_forward(self, x, dont_quantize=False, log_magnitude=False):
        """Encode STFT chunks to latents.

        Args:
            x: Input spectrogram chunks [B, C, F, T], T multiple of spec_length.
            dont_quantize: If True, returns continuous latents scaled to [-1,1].
            log_magnitude: If True, prints intermediate magnitudes (debug).
        Returns:
            Tensor of latents with shape [B, num_latents*(T/spec_length), dim].
        """
        assert x.shape[-1]%self.spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {self.spec_length}.'
        factor = None
        if x.shape[-1] > self.spec_length:
            x_ls = torch.split(x, self.spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :self.num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :self.num_latents]
        x = torch.tanh(x)
        if factor is not None:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
        return x

    @torch.compile(fullgraph=True, dynamic=False, mode='max-autotune-no-cudagraphs')
    def encoder_forward_fast(self, x, dont_quantize=False, log_magnitude=False):
        """torch.compile-optimized variant of encoder_forward with same outputs."""
        assert x.shape[-1]%self.spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {self.spec_length}.'
        factor = None
        if x.shape[-1] > self.spec_length:
            x_ls = torch.split(x, self.spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :self.num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :self.num_latents]
        x = torch.tanh(x)
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
        assert latents.shape[-2]%self.num_latents==0, f'Input shape {latents.shape[-2]} is not divisible by {self.num_latents}.'
        factor = None
        if latents.shape[-2]>self.num_latents:
            latents_ls = torch.split(latents, self.num_latents, dim=-2)
            factor = len(latents_ls)
            latents = torch.cat(latents_ls, dim=0)
        mask_embedding = self.mask_embedding.expand(latents.shape[0], -1, self.time_dim, -1).reshape(latents.shape[0], -1, self.dim)
        if self.more_latents_pre_decoder is not None:
            x = self.pre_decoder(mask_embedding, torch.cat((self.lat2patch_pre_decoder(latents), self.more_latents_pre_decoder.expand(latents.shape[0], -1, -1)), -2), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        else:
            x = self.pre_decoder(mask_embedding, self.lat2patch_pre_decoder(latents), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        features = self.frontend_pre_decoder_up(x, skip_output_layer=True, log_magnitude=log_magnitude)[1]
        if factor is not None:
            features = [torch.cat(torch.chunk(el, factor, dim=0), dim=-1) for el in features]
        return features

    @staticmethod
    def _lookup_condition_index(lookup, value, strict: bool):
        key = str(value)
        if key in lookup:
            return lookup[key]
        if strict:
            raise KeyError(key)
        return 0

    def _coerce_condition_indices(self, names, lookup, values, indices, batch_size, device):
        if indices is not None:
            if torch.is_tensor(indices):
                idx = indices.to(device=device, dtype=torch.long).reshape(-1)
            else:
                idx = torch.as_tensor(indices, device=device, dtype=torch.long).reshape(-1)
        else:
            if values is None:
                values = names[0]
            if isinstance(values, str):
                idx = torch.full(
                    (batch_size,),
                    self._lookup_condition_index(lookup, values, strict=True),
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
                            self._lookup_condition_index(lookup, value, strict=False)
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

    def _condition_embedding(
        self,
        sigma_embedding,
        task_id="reconstruct",
        task_idx=None,
    ):
        batch_size = sigma_embedding.shape[0]
        device = sigma_embedding.device
        task_idx = self._coerce_condition_indices(
            self.task_names,
            self.task_to_idx,
            task_id,
            task_idx,
            batch_size,
            device,
        )
        cond = sigma_embedding + self.task_embedding(task_idx).to(sigma_embedding.dtype)
        return self.emb_proj(cond)

    def get_sigma_continuous(self, i):
        return get_sigma_continuous(
            i,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )

    def decoder_forward(
        self,
        x,
        latents,
        features=None,
        sigma_left=None,
        sigma_right=None,
        output='both',
        task_id="reconstruct",
        task_idx=None,
        log_magnitude=False,
    ):
        """Single denoising step conditioned on left/right sigmas.

        Inputs are split along time into left/right halves. Returns left, right,
        or both halves depending on `output`.
        """

        if sigma_left is None:
            if output=='left':
                sigma_left = self.sigma_max
            else:
                sigma_left = self.sigma_min
        if sigma_right is None:
            sigma_right = self.sigma_max

        # CONDITIONING
        sigma_left = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)*sigma_left
        sigma_right = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)*sigma_right
        sigma = torch.cat([sigma_left, sigma_right], dim=0)
        sigma_log = torch.log(sigma)/4.
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self._condition_embedding(
            emb_sigma_log,
            task_id=task_id,
            task_idx=task_idx,
        )

        gain = self.gain_decoder(time_emb).unsqueeze(-2).unsqueeze(-1)+1.
        gain_inp, gain_out = torch.chunk(gain, 2, dim=-2)

        if features is None:
            features = self.pre_decoder_forward(latents, log_magnitude=log_magnitude)
        features = [torch.chunk(el, 2, dim=-1) for el in features]
        features = [torch.cat(el, dim=0) for el in features]

        c_skip, c_out, c_in = get_c(
            sigma,
            sigma_min=self.sigma_min,
            sigma_data=self.sigma_data,
        )
        attn_mask = self.get_attn_mask(x)
        x = torch.chunk(x, 2, dim=-1)
        x = torch.cat(x, dim=0)
        inp = x.clone()
        x = c_in*x
        x, features_dec = self.frontend_decoder_down(x, cond=time_emb, features=features, gain=gain_inp, log_magnitude=log_magnitude)
        x = torch.cat(torch.chunk(x, 2, dim=0), dim=-2)
        time_emb_left, time_emb_right = torch.chunk(time_emb, 2, dim=0)
        time_emb_transformer = torch.cat((torch.ones((x.shape[0], x.shape[1]//2 +self.num_latents+self.num_more_latents, time_emb.shape[-1]), dtype=x.dtype, device=x.device)*time_emb_left.unsqueeze(-2), torch.ones((x.shape[0], x.shape[1]//2 +self.num_latents +self.num_more_latents, time_emb.shape[-1]), dtype=x.dtype, device=x.device)*time_emb_right.unsqueeze(-2)), dim=-2)
        if self.more_latents_decoder is not None:
            x = self.decoder(x, time_emb_transformer, self.lat2patch(latents), self.more_latents_decoder.expand(x.shape[0], -1, -1), skip_input_layer=True, skip_output_layer=True, attn_mask=attn_mask, print_magnitudes=log_magnitude)
        else:
            x = self.decoder(x, time_emb_transformer, self.lat2patch(latents), skip_input_layer=True, skip_output_layer=True, attn_mask=attn_mask, print_magnitudes=log_magnitude)
        x = torch.cat(torch.chunk(x, 2, dim=-2), dim=0)
        x = self.frontend_decoder_up(x, cond=time_emb, features=features_dec, gain=gain_out, log_magnitude=log_magnitude)[0]
        x = c_skip*inp + c_out*x
        x_left, x_right = torch.chunk(x, 2, dim=0)
        if output=='left':
            return x_left
        elif output=='right':
            return x_right
        elif output=='both':
            return torch.cat((x_left, x_right), dim=-1)
        elif output=='features':
            return torch.cat((x_left, x_right), dim=-1), features
        else:
            raise ValueError(f'Invalid output type: {output}. Must be one of: (left, right, both, features).')

    def decode_single_parallel(self, latents, denoising_steps=None, task_id='reconstruct'):
        """Run the full parallel denoising schedule for one sample/batch."""
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps_parallel
        step_size = 1./denoising_steps
        noise = torch.randn((latents.shape[0], self.stft_channels, self.hop*(self.fac//2), self.spec_length*(latents.shape[-2]//self.num_latents)), dtype=latents.dtype, device=latents.device)*self.sigma_max
        latents_ls = torch.split(latents, self.num_latents, dim=-2)
        num_chunks = len(latents_ls)
        features = self.pre_decoder_forward(torch.cat(latents_ls, dim=0))
        sigma = self.sigma_max
        inp = noise
        for step in range(denoising_steps):
            inp, num_samples = preprocess_parallel_input(inp, iteration=step, length=self.spec_length, dim=-1)
            inp_latents, num_samples = preprocess_parallel_input(latents, iteration=step, length=self.num_latents, dim=-2)
            inp_features = preprocess_parallel_features(features, iteration=step, num_samples=num_chunks, dim=-1)
            out = self.decoder_forward(inp, inp_latents, inp_features, output='both', sigma_left=sigma, sigma_right=sigma, task_id=task_id)
            out = postprocess_parallel_input(out, iteration=step, num_samples=num_samples, length=self.spec_length, dim=-1)
            sigma = self.get_sigma_continuous(1.-(step+1)*step_size)
            inp = reverse_step(out, torch.randn_like(out), sigma, sigma_min=self.sigma_min)
        return out

    def decode_parallel(self, latents, denoising_steps=None, max_batch_size=None, task_id='reconstruct'):
        """Decode all timesteps in parallel using `distribute` for batching."""
        device = next(self.parameters()).device
        out = distribute(self.decode_single_parallel, latents, max_batch_size, device, denoising_steps=denoising_steps, task_id=task_id, mixed_precision=self.mixed_precision)
        return out

    def decode_autoregressive(self, latents, time_prompt=None, denoising_steps=None, max_batch_size=None, task_id='reconstruct'):
        """Autoregressive decoding over timesteps."""
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps_ar
        if time_prompt is None:
            time_prompt = self.default_time_prompt
        step_size = 1./denoising_steps
        # step_size_prompt = time_prompt/denoising_steps
        device = next(self.parameters()).device
        sigma_prompt = self.get_sigma_continuous(time_prompt)
        out_ls = []
        for i in range(latents.shape[-2]//self.num_latents):
            if i==0:
                noise = torch.randn((latents.shape[0], self.stft_channels, self.hop*(self.fac//2), self.spec_length), dtype=latents.dtype, device=latents.device)*self.sigma_max
                sigma = self.sigma_max
                features = distribute(self.pre_decoder_forward, latents[:, :self.num_latents*2], max_batch_size, device, mixed_precision=self.mixed_precision)
                for step in range(denoising_steps):
                    out = distribute(self.decoder_forward, torch.cat((noise, torch.zeros_like(noise)), dim=-1), max_batch_size, device, latents[:, :self.num_latents*2], features=features, output='left', sigma_left=sigma, task_id=task_id, mixed_precision=self.mixed_precision)
                    sigma = self.get_sigma_continuous(1.-(step+1)*step_size)
                    noise = reverse_step(out, torch.randn_like(out), sigma, sigma_min=self.sigma_min)
                out_ls.append(out)
            else:
                last_out_clean = out_ls[-1]
                noise = torch.randn_like(last_out_clean)*self.sigma_max
                last_out = reverse_step(last_out_clean, torch.randn_like(last_out_clean), torch.ones_like(last_out_clean)*sigma_prompt, sigma_min=self.sigma_min)
                sigma = self.sigma_max
                sigma_last_out = sigma_prompt
                features = distribute(self.pre_decoder_forward, latents[:, (i-1)*self.num_latents:(i+1)*self.num_latents], max_batch_size, device, mixed_precision=self.mixed_precision)
                for step in range(denoising_steps):
                    out = distribute(self.decoder_forward, torch.cat([last_out, noise], dim=-1), max_batch_size, device, latents[:, (i-1)*self.num_latents:(i+1)*self.num_latents], features=features, output='right', sigma_left=sigma_last_out, sigma_right=sigma, task_id=task_id, mixed_precision=self.mixed_precision)
                    sigma = self.get_sigma_continuous(1.-(step+1)*step_size)
                    noise = reverse_step(out, torch.randn_like(out), sigma, sigma_min=self.sigma_min)
                out_ls.append(out)
        out = torch.cat(out_ls, dim=-1)
        return out

    def decode_autoregressive_step(self, current_latents, past_repr=None, past_latents=None, time_prompt=None, denoising_steps=None, max_batch_size=None, task_id='reconstruct'):
        """Single-step live decoding with optional past spectrogram and latents."""
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps_ar
        if time_prompt is None:
            time_prompt = self.default_time_prompt
        step_size = 1./denoising_steps
        device = next(self.parameters()).device
        sigma_prompt = self.get_sigma_continuous(time_prompt)
        if past_repr is None:
            noise = torch.randn((current_latents.shape[0], self.stft_channels, self.hop*(self.fac//2), self.spec_length), dtype=current_latents.dtype, device=current_latents.device)*self.sigma_max
            sigma = self.sigma_max
            features = distribute(self.pre_decoder_forward, current_latents[:, -self.num_latents:], max_batch_size, device, mixed_precision=self.mixed_precision)
            features = [torch.cat((el, torch.zeros_like(el)), dim=-1) for el in features]
            for step in range(denoising_steps):
                out = distribute(self.decoder_forward, torch.cat((noise, torch.zeros_like(noise)), dim=-1), max_batch_size, device, torch.cat((current_latents[:, -self.num_latents:], torch.zeros_like(current_latents[:, -self.num_latents:])), dim=-2), features=features, output='left', sigma_left=sigma, task_id=task_id, mixed_precision=self.mixed_precision)
                sigma = self.get_sigma_continuous(1.-(step+1)*step_size)
                noise = reverse_step(out, torch.randn_like(out), sigma, sigma_min=self.sigma_min)
            out = torch.cat((torch.zeros_like(out), out), dim=-1)
        else:
            last_out_clean = past_repr[..., -self.spec_length:]
            if past_latents is None:
                past_latents = distribute(self.encoder_forward, last_out_clean, max_batch_size, device, mixed_precision=self.mixed_precision)
            else:
                past_latents = past_latents[:, -self.num_latents:]
            noise = torch.randn_like(last_out_clean)*self.sigma_max
            last_out = reverse_step(last_out_clean, torch.randn_like(last_out_clean), torch.ones_like(last_out_clean)*sigma_prompt, sigma_min=self.sigma_min)
            sigma = self.sigma_max
            sigma_last_out = sigma_prompt
            features = distribute(self.pre_decoder_forward, torch.cat((past_latents, current_latents[:, -self.num_latents:]), dim=-2), max_batch_size, device, mixed_precision=self.mixed_precision)
            for step in range(denoising_steps):
                out = distribute(self.decoder_forward, torch.cat([last_out, noise], dim=-1), max_batch_size, device, torch.cat((past_latents, current_latents[:, -self.num_latents:]), dim=-2), features=features, output='right', sigma_left=sigma_last_out, sigma_right=sigma, task_id=task_id, mixed_precision=self.mixed_precision)
                sigma = self.get_sigma_continuous(1.-(step+1)*step_size)
                noise = reverse_step(out, torch.randn_like(out), sigma, sigma_min=self.sigma_min)
            out = torch.cat((last_out_clean, out), dim=-1)
        return out
