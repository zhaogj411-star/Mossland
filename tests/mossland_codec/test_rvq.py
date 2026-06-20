import importlib

import pytest
import torch


def test_rvq_quantizes_token_sequence_with_dynamic_depth():
    module = importlib.import_module("scripts.mossland-codec.quantization")
    rvq = module.ResidualVectorQuantizer(
        dim=4,
        codebook_size=8,
        num_quantizers=3,
    )
    tokens = torch.randn(2, 5, 4)

    full = rvq(tokens)
    partial = rvq(tokens, n_quantizers=2)

    assert full.quantized.shape == tokens.shape
    assert full.codes.shape == (2, 5, 3)
    assert full.loss.shape == ()
    assert partial.quantized.shape == tokens.shape
    assert partial.codes.shape == (2, 5, 2)
    assert torch.all(partial.codes >= 0)
    assert torch.all(partial.codes < 8)


def test_rvq_codes_to_latents_supports_partial_depth():
    module = importlib.import_module("scripts.mossland-codec.quantization")
    rvq = module.ResidualVectorQuantizer(dim=4, codebook_size=8, num_quantizers=3)
    tokens = torch.randn(2, 5, 4)
    encoded = rvq(tokens, n_quantizers=2)

    decoded = rvq.codes_to_latents(encoded.codes)

    assert decoded.shape == tokens.shape


def test_rvq_rejects_invalid_dynamic_depth():
    module = importlib.import_module("scripts.mossland-codec.quantization")
    rvq = module.ResidualVectorQuantizer(dim=4, codebook_size=8, num_quantizers=3)
    tokens = torch.randn(1, 2, 4)

    with pytest.raises(ValueError, match="n_quantizers"):
        rvq(tokens, n_quantizers=0)
    with pytest.raises(ValueError, match="n_quantizers"):
        rvq(tokens, n_quantizers=4)
