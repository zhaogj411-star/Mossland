import sys
import types

import torch

sys.modules.setdefault("librosa", types.SimpleNamespace())

einops_stub = types.ModuleType("einops")


def _reduce_stub(x, _pattern, reduction):
    if reduction != "sum":
        raise NotImplementedError(reduction)
    return x.sum(dim=0)


einops_stub.reduce = _reduce_stub
sys.modules.setdefault("einops", einops_stub)

vq_stub = types.ModuleType("vector_quantize_pytorch")


class _DummyResidualVQ:
    def __init__(self, *args, **kwargs):
        self.layers = []


vq_stub.ResidualVQ = _DummyResidualVQ
vq_stub.vector_quantize_pytorch = types.SimpleNamespace(
    sample_vectors_distributed=None,
    batched_sample_vectors=None,
    distributed=types.SimpleNamespace(all_reduce=None),
    noop=None,
)
sys.modules.setdefault("vector_quantize_pytorch", vq_stub)

from scripts.mossland_codec_convtrans.models import ConvEncoder


def _small_conv_encoder(**overrides):
    kwargs = dict(
        base_channels=8,
        layers_list_encoder=[1, 1, 1, 1, 1],
        multipliers_list=[1, 2, 2, 2, 2],
        attention_list_encoder=[0, 0, 0, 0, 0],
        freq_downsample_list=[1, 0, 0, 0],
        bottleneck_base_channels=32,
        num_bottleneck_layers=1,
        frequency_scaling=True,
        heads=2,
        normalization=True,
        pre_normalize_2d_to_1d=True,
        pre_normalize_downsampling_encoder=True,
        hop=16,
        data_channels=4,
        bottleneck_channels=16,
        min_res_dropout=16,
        dropout_rate=0.0,
    )
    kwargs.update(overrides)
    return ConvEncoder(**kwargs)


def test_convtrans_bottleneck_zero_init_starts_at_zero():
    encoder = _small_conv_encoder()
    x = torch.randn(2, 4, 32, 64)

    continuous, hidden, pre_tanh = encoder.project_features(encoder.encode_features(x))

    assert hidden.shape == (2, 32, 8)
    assert pre_tanh.shape == (2, 16, 8)
    assert continuous.shape == (2, 16, 8)
    assert torch.allclose(pre_tanh, torch.zeros_like(pre_tanh))
    assert torch.allclose(continuous, torch.zeros_like(continuous))


def test_convtrans_bottleneck_small_init_stays_bounded():
    torch.manual_seed(0)
    encoder = _small_conv_encoder(
        latent_head_init_std=1e-3,
        latent_tanh_scale=2.0,
    )
    x = torch.randn(2, 4, 32, 64)

    continuous, _hidden, pre_tanh = encoder.project_features(encoder.encode_features(x))

    assert torch.isfinite(pre_tanh).all()
    assert torch.isfinite(continuous).all()
    assert pre_tanh.abs().sum() > 0
    assert continuous.abs().max() <= 1.0 + 1e-6
