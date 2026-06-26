"""Diffusion and flow-matching training utilities for the pure-torch Music-DiT."""

from __future__ import annotations

from typing import Optional

import torch

from .core import contract as C


def _unconditional_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    uncond = dict(batch)
    if C.KEY_CROSSATTN in batch:
        uncond[C.KEY_CROSSATTN] = torch.zeros_like(batch[C.KEY_CROSSATTN])
    if C.KEY_CROSSATTN_MASK in batch:
        # Do not mask every cross-attention key. PyTorch MHA produces NaNs
        # for an all-masked key sequence; zero embeddings plus a visible mask
        # represent the unconditional text path safely.
        uncond[C.KEY_CROSSATTN_MASK] = torch.ones_like(batch[C.KEY_CROSSATTN_MASK])
    if C.KEY_CONTEXT in batch:
        uncond[C.KEY_CONTEXT] = torch.zeros_like(batch[C.KEY_CONTEXT])
    if C.KEY_COND_MASK in batch:
        uncond[C.KEY_COND_MASK] = torch.zeros_like(batch[C.KEY_COND_MASK])
    return uncond


class MusicEDM:
    def __init__(
        self,
        net,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        self.net = net
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def _scalings(self, sigma: torch.Tensor):
        sigma_data_sq = self.sigma_data ** 2
        c_skip = sigma_data_sq / (sigma.square() + sigma_data_sq)
        c_out = sigma * self.sigma_data / (sigma.square() + sigma_data_sq).sqrt()
        c_in = 1.0 / (sigma.square() + sigma_data_sq).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    def denoise(self, x_t: torch.Tensor, sigma: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = x_t.shape[0]
        sigma = sigma.view(batch_size, 1, 1)
        c_skip, c_out, c_in, c_noise = self._scalings(sigma)
        pred = self.net(
            x=c_in * x_t,
            timesteps=c_noise.view(batch_size),
            crossattn_emb=batch[C.KEY_CROSSATTN],
            crossattn_mask=batch.get(C.KEY_CROSSATTN_MASK),
            pos_ids=batch.get(C.KEY_POS_IDS),
            context_latent=batch.get(C.KEY_CONTEXT),
            cond_mask=batch.get(C.KEY_COND_MASK),
            seq_len_q=batch.get(C.KEY_SEQLEN_Q),
        )
        return c_skip * x_t + c_out * pred

    def training_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x0 = batch[C.KEY_LATENT].float() * self.sigma_data
        batch_size = x0.shape[0]
        sigma = (torch.randn(batch_size, device=x0.device) * self.p_std + self.p_mean).exp()
        sigma_broadcast = sigma.view(batch_size, 1, 1)
        noise = torch.randn_like(x0)
        x_t = x0 + sigma_broadcast * noise
        x0_pred = self.denoise(x_t, sigma, batch)
        weights = (sigma.square() + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)
        per_token = (x0_pred - x0).square().mean(dim=-1)

        loss_mask = batch.get(C.KEY_LOSS_MASK)
        if loss_mask is not None:
            loss_mask = loss_mask.float()
            per_token = per_token * loss_mask
            denom = loss_mask.sum().clamp_min(1.0)
            loss = (weights.view(batch_size, 1) * per_token).sum() / denom
        else:
            loss = (weights.view(batch_size, 1) * per_token).mean()

        metrics = {
            "loss/mse": (x0_pred - x0).square().mean().detach(),
            "sigma/mean": sigma.mean().detach(),
            "sigma/min": sigma.min().detach(),
            "sigma/max": sigma.max().detach(),
        }
        return loss, metrics


class OptimalTransportFlow:
    """Minimal OT-CFM path ported from NeMo audio flow utilities.

    This keeps only the pieces needed by the local Music-DiT stack:
    sampling a point on the path and computing the conditional vector field.
    """

    def __init__(
        self,
        time_min: float = 1e-5,
        time_max: float = 1.0,
        sigma_start: float = 1.0,
        sigma_end: float = 1e-4,
    ):
        self.time_min = float(time_min)
        self.time_max = float(time_max)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)

    @staticmethod
    def _broadcast_time(time: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while time.ndim < target_ndim:
            time = time.unsqueeze(-1)
        return time

    def generate_time(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype, generator=generator) * (
            self.time_max - self.time_min
        ) + self.time_min

    def mean(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        time = self._broadcast_time(time, x_start.ndim)
        return (self.time_max - time) * x_start + time * x_end

    def std(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        time = self._broadcast_time(time, x_start.ndim)
        return (self.time_max - time) * self.sigma_start + time * self.sigma_end

    def sample(
        self,
        *,
        time: torch.Tensor,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        mean = self.mean(time=time, x_start=x_start, x_end=x_end)
        std = self.std(time=time, x_start=x_start, x_end=x_end)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + std * noise

    def vector_field(
        self,
        *,
        time: torch.Tensor,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        point: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        time = self._broadcast_time(time, x_start.ndim)
        if self.sigma_start == self.sigma_end:
            return x_end - x_start

        num = self.sigma_end * (point - x_start) - self.sigma_start * (point - x_end)
        denom = (self.time_max - time) * self.sigma_start + time * self.sigma_end
        return num / (denom + eps)


class MusicFlowMatching:
    """Pure-torch flow matching objective for the local Music-DiT contract."""

    def __init__(
        self,
        net,
        data_scale: float = 0.5,
        estimator_target: str = "conditional_vector_field",
        time_min: float = 1e-5,
        time_max: float = 1.0,
        sigma_start: float = 1.0,
        sigma_end: float = 1e-4,
    ):
        estimator_target = str(estimator_target)
        if estimator_target not in {"conditional_vector_field", "data"}:
            raise ValueError(f"Unsupported estimator_target={estimator_target!r}")
        self.net = net
        self.data_scale = float(data_scale)
        self.estimator_target = estimator_target
        self.flow = OptimalTransportFlow(
            time_min=time_min,
            time_max=time_max,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
        )

    def _forward_estimator(
        self,
        sample: torch.Tensor,
        time: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.net(
            x=sample,
            timesteps=time,
            crossattn_emb=batch[C.KEY_CROSSATTN],
            crossattn_mask=batch.get(C.KEY_CROSSATTN_MASK),
            pos_ids=batch.get(C.KEY_POS_IDS),
            context_latent=batch.get(C.KEY_CONTEXT),
            cond_mask=batch.get(C.KEY_COND_MASK),
            seq_len_q=batch.get(C.KEY_SEQLEN_Q),
        )

    def training_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_end = batch[C.KEY_LATENT].float() * self.data_scale
        batch_size = x_end.shape[0]
        x_start = torch.zeros_like(x_end)
        time = self.flow.generate_time(batch_size, device=x_end.device)
        sample = self.flow.sample(time=time, x_start=x_start, x_end=x_end)
        estimate = self._forward_estimator(sample, time, batch)

        if self.estimator_target == "conditional_vector_field":
            target = self.flow.vector_field(time=time, x_start=x_start, x_end=x_end, point=sample)
        else:
            target = x_end

        per_token = (estimate - target).square().mean(dim=-1)
        loss_mask = batch.get(C.KEY_LOSS_MASK)
        if loss_mask is not None:
            loss_mask = loss_mask.float()
            per_token = per_token * loss_mask
            denom = loss_mask.sum().clamp_min(1.0)
            loss = per_token.sum() / denom
        else:
            loss = per_token.mean()

        metrics = {
            "loss/mse": (estimate - target).square().mean().detach(),
            "flow/time_mean": time.mean().detach(),
            "flow/time_min": time.min().detach(),
            "flow/time_max": time.max().detach(),
            "flow/sample_std": sample.std().detach(),
            "flow/target_std": target.std().detach(),
        }
        return loss, metrics


class MusicFlowMatchingEulerSampler:
    """Minimal Euler sampler for the local flow-matching objective."""

    def __init__(
        self,
        net,
        *,
        estimator_target: str = "conditional_vector_field",
        flow: Optional[OptimalTransportFlow] = None,
        num_steps: int = 32,
    ):
        estimator_target = str(estimator_target)
        if estimator_target not in {"conditional_vector_field", "data"}:
            raise ValueError(f"Unsupported estimator_target={estimator_target!r}")
        self.net = net
        self.estimator_target = estimator_target
        self.flow = flow or OptimalTransportFlow()
        self.num_steps = int(num_steps)

    def _estimate(self, state: torch.Tensor, time: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(
            x=state,
            timesteps=time,
            crossattn_emb=batch[C.KEY_CROSSATTN],
            crossattn_mask=batch.get(C.KEY_CROSSATTN_MASK),
            pos_ids=batch.get(C.KEY_POS_IDS),
            context_latent=batch.get(C.KEY_CONTEXT),
            cond_mask=batch.get(C.KEY_COND_MASK),
            seq_len_q=batch.get(C.KEY_SEQLEN_Q),
        )

    @torch.inference_mode()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        shape: Optional[tuple[int, ...]] = None,
        init_state: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if init_state is None:
            if shape is None:
                if C.KEY_LATENT not in batch:
                    raise ValueError("shape is required when batch does not contain latent")
                shape = tuple(batch[C.KEY_LATENT].shape)
            ref = batch[C.KEY_CROSSATTN]
            init_state = torch.randn(shape, device=ref.device, dtype=ref.dtype, generator=generator)
            init_state = init_state * self.flow.sigma_start
        else:
            init_state = init_state.clone()

        state = init_state
        if x_start is None:
            x_start = torch.zeros_like(state)

        times = torch.linspace(
            self.flow.time_min,
            self.flow.time_max,
            self.num_steps + 1,
            device=state.device,
            dtype=state.dtype,
        )
        dt = times[1] - times[0]
        guidance_scale = float(guidance_scale)
        uncond_batch = None if guidance_scale == 1.0 else _unconditional_batch(batch)
        for t in times[:-1]:
            time = torch.full((state.shape[0],), float(t), device=state.device, dtype=state.dtype)
            estimate = self._estimate(state, time, batch)
            if uncond_batch is not None:
                uncond_estimate = self._estimate(state, time, uncond_batch)
                estimate = uncond_estimate + guidance_scale * (estimate - uncond_estimate)
            if self.estimator_target == "conditional_vector_field":
                vector_field = estimate
            else:
                vector_field = self.flow.vector_field(
                    time=time,
                    x_start=x_start,
                    x_end=estimate,
                    point=state,
                )
            state = state + vector_field * dt
        return state
