import importlib

import pytest
import torch


def test_token_count_uses_sample_rate_and_latent_rate():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")

    assert module.samples_per_token(48000, 25) == 1920
    assert module.token_count_for_samples(48000, 48000, 25) == 25


def test_local_token_encoder_preserves_time_axis():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")
    encoder = module.LocalTimeTokenEncoder(
        input_dim=6,
        token_dim=4,
        num_layers=1,
        num_heads=2,
        local_window_tokens=2,
    )
    x = torch.randn(2, 7, 6)

    out = encoder(x)

    assert out.shape == (2, 7, 4)


def test_local_token_encoder_uses_sliding_window_attention_not_full_mask():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")
    encoder = module.LocalTimeTokenEncoder(
        input_dim=6,
        token_dim=4,
        num_layers=1,
        num_heads=2,
        local_window_tokens=2,
    )

    assert not any(
        isinstance(layer, torch.nn.TransformerEncoder)
        for layer in encoder.modules()
    )
    assert encoder.layers[0].attention.local_window_tokens == 2


def test_sliding_window_attention_uses_flashattention_window_backend(monkeypatch):
    module = importlib.import_module("scripts.mossland-codec.time_tokens")
    calls = []

    def fake_flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        deterministic,
    ):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "window_size": window_size,
                "deterministic": deterministic,
            }
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(module, "_flash_attn_func", fake_flash_attn_func, raising=False)
    attention = module.SlidingWindowSelfAttention(
        dim=4,
        num_heads=2,
        local_window_tokens=3,
    )
    monkeypatch.setattr(attention, "_flash_attention_supported", lambda q: True)
    x = torch.randn(2, 5, 4)

    out = attention(x)

    assert out.shape == (2, 5, 4)
    assert calls == [
        {
            "q_shape": (2, 5, 2, 2),
            "dropout_p": 0.0,
            "softmax_scale": attention.head_dim ** -0.5,
            "causal": False,
            "window_size": (3, 3),
            "deterministic": not attention.training,
        }
    ]


def test_samples_per_token_rejects_non_integer_rate():
    module = importlib.import_module("scripts.mossland-codec.time_tokens")

    with pytest.raises(ValueError, match="integer"):
        module.samples_per_token(44100, 40)
