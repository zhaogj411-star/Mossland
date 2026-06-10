import math
import importlib

import pytest
import torch

training_base = importlib.import_module("scripts.mossland-codec.training_base")
CodecTrainingBase = training_base.CodecTrainingBase
pseudo_huber_loss = training_base.pseudo_huber_loss


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, value):
        return value * self.weight


def test_pseudo_huber_loss_matches_music2latent_formula():
    predicted = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
    target = torch.tensor([[0.0, 1.0], [2.0, 1.0]])

    loss = pseudo_huber_loss(predicted, target)

    c = 0.00054 * math.sqrt(math.prod(predicted.shape[1:]))
    expected = torch.sqrt((predicted - target) ** 2 + c**2) - c
    torch.testing.assert_close(loss, expected)


def test_assert_finite_reports_tensor_name_shape_and_path():
    wrapper = CodecTrainingBase(
        model=TinyModule(),
        use_ema=False,
        fail_on_nonfinite=True,
    )

    with pytest.raises(FloatingPointError) as error:
        wrapper._assert_finite(
            "representation",
            torch.tensor([1.0, float("nan")]),
            {"path": "/tmp/example.wav"},
        )

    message = str(error.value)
    assert "Non-finite representation" in message
    assert "shape=(2,)" in message
    assert "/tmp/example.wav" in message
