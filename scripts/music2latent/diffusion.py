from __future__ import annotations

import math

import torch


def get_c(
    sigma: torch.Tensor,
    *,
    sigma_min: float,
    sigma_data: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_correct = float(sigma_min)
    sigma_data_sq = float(sigma_data) ** 2.0
    c_skip = sigma_data_sq / ((sigma - sigma_correct).square() + sigma_data_sq)
    c_out = (float(sigma_data) * (sigma - sigma_correct)) / (
        sigma_data_sq + sigma.square()
    ).sqrt()
    c_in = 1.0 / (sigma.square() + sigma_data_sq).sqrt()
    return (
        c_skip.reshape(-1, 1, 1, 1),
        c_out.reshape(-1, 1, 1, 1),
        c_in.reshape(-1, 1, 1, 1),
    )


def get_sigma_continuous(
    index: torch.Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    rho_inv = 1.0 / float(rho)
    return (
        float(sigma_min) ** rho_inv
        + index * (float(sigma_max) ** rho_inv - float(sigma_min) ** rho_inv)
    ) ** float(rho)


def add_noise(
    representation: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return representation + sigma.reshape(-1, 1, 1, 1) * noise


def get_step_schedule(
    step: int,
    *,
    schedule: str,
    total_steps: int,
    base_step: float,
    start_exp: float,
    end_exp: float,
) -> float:
    if schedule == "constant":
        return float(base_step) ** float(end_exp)
    if schedule != "exponential":
        raise ValueError("schedule must be one of: exponential, constant")
    progress = min(max(float(step) / max(1, int(total_steps)), 0.0), 1.0)
    exponent = progress * (float(end_exp) - float(start_exp)) + float(start_exp)
    return max(float(base_step) ** exponent, float(base_step) ** float(end_exp))


def pseudo_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 0.00054,
) -> torch.Tensor:
    c = float(delta) * math.sqrt(math.prod(predicted.shape[1:]))
    return torch.hypot(predicted - target, predicted.new_tensor(c)) - c


def consistency_weight(
    sigma_high: torch.Tensor,
    sigma_low: torch.Tensor,
    *,
    min_delta: float = 0.0,
) -> torch.Tensor:
    return 1.0 / (sigma_high - sigma_low).clamp_min(float(min_delta))
