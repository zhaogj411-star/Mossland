import importlib.util
from pathlib import Path

import pytest
import torch

from scripts.mossland_codec_convtrans import audio as convtrans_audio


ROOT = Path(__file__).resolve().parents[2]


def _load_audio_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_codicodec_audio():
    return _load_audio_module("codicodec_audio_file", ROOT / "scripts" / "codicodec" / "audio.py")


def _load_hyphen_mossland_codec_audio():
    return _load_audio_module(
        "hyphen_mossland_codec_audio",
        ROOT / "scripts" / "mossland-codec" / "audio.py",
    )


AUDIO_MODULES = [
    pytest.param(convtrans_audio, id="mossland_codec_convtrans"),
    pytest.param(_load_codicodec_audio(), id="codicodec"),
    pytest.param(_load_hyphen_mossland_codec_audio(), id="mossland-codec"),
]


def _expected_music2latent_realimag(realimag, alpha, beta):
    if realimag.shape[-3] % 2 != 0:
        raise ValueError("real/imag channel count must be even")
    pairs = realimag.shape[-3] // 2
    paired = realimag.reshape(*realimag.shape[:-3], pairs, 2, *realimag.shape[-2:])
    complex_spec = torch.complex(paired[..., 0, :, :], paired[..., 1, :, :])
    expected_complex = (
        beta
        * complex_spec.abs().pow(alpha).to(torch.complex64)
        * torch.exp(1j * torch.angle(complex_spec).to(torch.complex64))
    )
    return torch.stack((expected_complex.real, expected_complex.imag), dim=-3).reshape_as(realimag)


@pytest.mark.parametrize("audio_module", AUDIO_MODULES)
def test_complex_realimag_rescale_matches_music2latent_formula(audio_module):
    torch.manual_seed(0)
    realimag = torch.randn(3, 4, 17, 5)
    alpha = 0.65
    beta = 0.34

    actual = audio_module.normalize_complex_realimag(
        realimag,
        alpha_rescale=alpha,
        beta_rescale=beta,
    )
    expected = _expected_music2latent_realimag(realimag, alpha, beta)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("audio_module", AUDIO_MODULES)
def test_complex_realimag_rescale_roundtrip(audio_module):
    torch.manual_seed(1)
    realimag = torch.randn(2, 4, 11, 7)
    realimag[:, :, 0, 0] = 0

    normalized = audio_module.normalize_complex_realimag(realimag)
    restored = audio_module.denormalize_complex_realimag(normalized)

    torch.testing.assert_close(restored, realimag, rtol=2e-5, atol=2e-6)
