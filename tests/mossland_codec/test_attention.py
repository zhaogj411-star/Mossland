import importlib

import pytest
import torch


layers = importlib.import_module("scripts.mossland-codec.transformer_layers")


def test_block_causal_attention_mask_matches_additive_mask():
    torch.manual_seed(0)
    attn = layers.MultiHeadAttention(embed_dim=8, num_heads=2)
    with torch.no_grad():
        attn.to_out.weight.copy_(torch.eye(8))

    x = torch.randn(2, 6, 8)
    additive_mask = torch.zeros(6, 6)
    additive_mask[:3, 3:] = -torch.finfo(x.dtype).max

    expected = attn(x, attn_mask=additive_mask)
    actual = attn(x, attn_mask=layers.block_causal_attention_mask(3))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.backends.cuda.is_flash_attention_available(),
    reason="CUDA flash attention is not available",
)
def test_multi_head_attention_flash_path_runs_on_cuda():
    attn = layers.MultiHeadAttention(embed_dim=64, num_heads=4).cuda().bfloat16()
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y = attn(x)
    torch.cuda.synchronize()

    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.backends.cuda.is_flash_attention_available(),
    reason="CUDA flash attention is not available",
)
def test_block_causal_attention_mask_flash_path_runs_on_cuda():
    attn = layers.MultiHeadAttention(embed_dim=64, num_heads=4).cuda().bfloat16()
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y = attn(x, attn_mask=layers.block_causal_attention_mask(16))
    torch.cuda.synchronize()

    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16
