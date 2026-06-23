from types import SimpleNamespace

import torch

from scripts.codicodec.wrapper import CoDiCodecTrainingWrapper


def test_latent_time_shift_forces_decode_boundary_mismatch_when_active():
    wrapper = CoDiCodecTrainingWrapper.__new__(CoDiCodecTrainingWrapper)
    wrapper.model = SimpleNamespace(
        spec_length=32,
        num_latents=128,
        bottleneck_channels=4,
        rvq_dim=64,
        rvq_tokens_per_chunk=8,
    )
    wrapper.latent_time_shift_prob = 1.0
    wrapper.latent_time_shift_max_tokens = None

    representation = torch.arange(1 * 4 * 960 * 96, dtype=torch.float32).reshape(1, 4, 960, 96)
    latents = torch.arange(1 * 384 * 4, dtype=torch.float32).reshape(1, 384, 4)

    shifted_representation, shifted_latents, metrics = wrapper._apply_latent_time_shift(
        representation,
        latents,
    )

    shift_tokens = int(metrics["latent/time_shift_tokens"].item())
    shift_frames = int(metrics["latent/time_shift_frames"].item())

    assert 1 <= shift_tokens <= 7
    assert shift_frames == shift_tokens * 4
    assert shifted_representation.shape == (1, 4, 960, 64)
    assert shifted_latents.shape == (1, 256, 4)
    assert torch.equal(
        shifted_representation,
        representation[..., shift_frames : shift_frames + 64],
    )
    expected_latent_tokens = latents.reshape(1, 24, 64)[:, shift_tokens : shift_tokens + 16]
    assert torch.equal(shifted_latents.reshape(1, 16, 64), expected_latent_tokens)
