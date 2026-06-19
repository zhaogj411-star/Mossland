import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from .transformer import TransformerBlock


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class Transpose(nn.Module):
    def forward(self, x, **kwargs):
        return rearrange(x, "... a b -> ... b a")


def _zero_pad_modulo_sequence(x, size, dim=-2):
    pad_len = (size - x.shape[dim] % size) % size
    if pad_len <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    return torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)


class TransformerResamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        sliding_window=None,
        chunk_size=128,
        chunk_midpoint_shift=False,
        type="encoder",
        transformer_depth=3,
        checkpointing=False,
        conformer=False,
        layer_scale=False,
        dim_heads=128,
        differential=True,
        variable_stride=False,
        feat_scale=False,
        sinusoidal_blocks=0,
        mask_noise=0,
        ff_mult=3,
        mapping_bias=True,
        cross_attn=False,
        dyt=True,
        conv_mapping=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        if type not in {"encoder", "decoder"}:
            raise ValueError(f"unknown TRB type {type!r}")
        transformer_dim = out_channels if type == "encoder" else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = bool(variable_stride)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.chunk_midpoint_shift = bool(chunk_midpoint_shift)
        self.type = type
        self.mask_noise = float(mask_noise)
        self.sliding_window_latents = sliding_window
        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, self.stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(self.stride)
        kernel_size = 3 if conv_mapping else 1
        self.mapping = (
            WNConv1d(in_channels, out_channels, kernel_size, padding="same", bias=mapping_bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    transformer_dim,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=not layer_scale,
                    norm_type="dyt" if dyt else "rms_norm",
                    conformer=conformer,
                    layer_scale=layer_scale,
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                        "feat_scale": feat_scale,
                    },
                    ff_kwargs={
                        "mult": ff_mult,
                        "no_bias": False,
                        "sinusoidal": (transformer_depth - idx) < sinusoidal_blocks,
                    },
                    norm_kwargs={"eps": 1e-3},
                    cross_attend=cross_attn,
                )
                for idx in range(int(transformer_depth))
            ]
        )
        token_dim = out_channels if type == "encoder" else in_channels
        new_token_count = 1 if self.variable_stride else self.output_seg_size
        self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, new_token_count, token_dim))
        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [win * (stride + 1 + prepend_cond_length) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length=0):
        sub_chunk_size = int(stride) + 1 + int(prepend_cond_length)
        return (
            int(stride) if self.type == "encoder" else 1,
            1 if self.type == "encoder" else int(stride),
            sub_chunk_size,
        )

    def forward(self, x, stride=None, return_features=False, override_new_tokens=None, **kwargs):
        batch_size = x.shape[0]
        features = [] if return_features else None
        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride)
        if self.type == "encoder":
            if len(self.transformers) > 0:
                x = _zero_pad_modulo_sequence(
                    x, input_seg_size if sliding_window is not None else self.chunk_size, dim=-1
                )
            x = self.mapping(x)
        if len(self.transformers) > 0:
            x = rearrange(x, "b d n -> b n d")
            if return_features:
                features.append(x)
            if self.type != "encoder":
                pad_modulo = input_seg_size if sliding_window is not None else max(1, self.chunk_size // int(stride or self.stride))
                x = _zero_pad_modulo_sequence(x, pad_modulo)
            x = rearrange(x, "b (n c) d -> (b n) c d", c=input_seg_size)
            new_tokens = self.new_tokens.expand(x.shape[0], output_seg_size, -1)
            if override_new_tokens is not None:
                new_tokens = rearrange(override_new_tokens, "b (n c) d -> (b n) c d", c=output_seg_size)
                new_tokens = self.new_tokens.expand_as(new_tokens) + new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x, new_tokens], dim=-2)
            x = rearrange(x, "(b n) c d -> b (n c) d", b=batch_size)
            if sliding_window is None:
                active_stride = int(stride or self.stride)
                effective_chunk_size = self.chunk_size + self.chunk_size // active_stride
                x = _zero_pad_modulo_sequence(x, effective_chunk_size)
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
            for layer in self.transformers:
                x = layer(x, self_attention_flash_sliding_window=sliding_window)
                if return_features:
                    features.append(x)
            if sliding_window is None:
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
            x = rearrange(x, "b (n c) d -> (b n) c d", c=sub_chunk_size)
            x = x[:, -output_seg_size:, :]
            x = rearrange(x, "(b n) c d -> b d (n c)", b=batch_size)
        if self.type == "decoder":
            x = self.mapping(x)
        if return_features:
            return x, features
        return x


class SAMEEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        transformer_depths=(3, 3, 3, 3),
        **kwargs,
    ):
        super().__init__()
        channel_dims = [in_channels] + [mult * channels for mult in c_mults]
        layers = []
        for idx in range(len(c_mults)):
            layers.append(
                TransformerResamplingBlock(
                    channel_dims[idx],
                    channel_dims[idx + 1],
                    strides[idx],
                    transformer_depth=transformer_depths[idx],
                    type="encoder",
                    **kwargs,
                )
            )
        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)
        self.depth = len(c_mults)
        self.strides = tuple(strides)

    def forward(self, x, override_stride=None, return_features=False, **kwargs):
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                stride = None if override_stride is None else override_stride[idx]
                if return_features:
                    x, features = layer(x, stride=stride, return_features=True)
                else:
                    x = layer(x, stride=stride)
            else:
                x = layer(x)
        if return_features:
            return x, features
        return x


class SAMEDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        transformer_depths=(3, 3, 3, 3),
        sinusoidal_blocks=None,
        **kwargs,
    ):
        super().__init__()
        sinusoidal_blocks = sinusoidal_blocks or [0 for _ in c_mults]
        channel_dims = [out_channels] + [mult * channels for mult in c_mults]
        layers = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]
        for idx in range(len(c_mults), 0, -1):
            layers.append(
                TransformerResamplingBlock(
                    channel_dims[idx],
                    channel_dims[idx - 1],
                    strides[idx - 1],
                    type="decoder",
                    transformer_depth=transformer_depths[idx - 1],
                    sinusoidal_blocks=sinusoidal_blocks[idx - 1],
                    **kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.depth = len(c_mults)

    def forward(self, x, override_stride=None, **kwargs):
        trb_idx = 0
        for layer in self.layers:
            if isinstance(layer, TransformerResamplingBlock):
                stride = None if override_stride is None else override_stride[trb_idx]
                x = layer(x, stride=stride)
                trb_idx += 1
            else:
                x = layer(x)
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck=None,
        pretransform=None,
        soft_clip=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.pretransform = pretransform
        self.latent_dim = int(latent_dim)
        self.downsampling_ratio = int(downsampling_ratio)
        self.sample_rate = int(sample_rate)
        self.io_channels = int(io_channels)
        self.in_channels = self.io_channels
        self.out_channels = self.io_channels
        self.min_length = self.downsampling_ratio
        self.soft_clip = bool(soft_clip)

    def encode(self, audio, return_info=False, skip_pretransform=False, **kwargs):
        info = {}
        if self.pretransform is not None and not skip_pretransform:
            audio = self.pretransform.encode(audio)
        latents = self.encoder(audio, **kwargs)
        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)
        if return_info:
            return latents, info
        return latents

    def decode(self, latents, **kwargs):
        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)
        audio = self.decoder(latents, **kwargs)
        if self.pretransform is not None:
            audio = self.pretransform.decode(audio)
        if self.soft_clip:
            audio = torch.tanh(audio)
        return audio

    def encode_audio(self, audio, **kwargs):
        return self.encode(audio, **kwargs)

    def decode_audio(self, latents, **kwargs):
        return self.decode(latents, **kwargs)
