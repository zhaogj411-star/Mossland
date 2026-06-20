import math
from typing import Any

import lightning as pl
import torch
from ema_pytorch import EMA


def pseudo_huber_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 0.00054):
    c = delta * math.sqrt(math.prod(input.shape[1:]))
    return torch.sqrt((input - target) ** 2 + c**2) - c


def get_sigma_continuous(
    i,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
):
    return (
        sigma_min ** (1.0 / rho)
        + i * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho))
    ) ** rho


def add_noise(x: torch.Tensor, noise: torch.Tensor, sigma):
    if isinstance(sigma, int):
        sigma = float(sigma)
    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x.device)
    if len(sigma.shape) == 1:
        sigma = sigma.reshape(-1, 1, 1, 1)
    elif len(sigma.shape) == 2:
        sigma = sigma.reshape(sigma.shape[0], 1, 1, -1)
    return x + sigma * noise


class CodecTrainingBase(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        use_ema: bool = True,
        learning_rate: float = 1e-4,
        fail_on_nonfinite: bool = True,
        fail_on_nonfinite_grad: bool | None = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.fail_on_nonfinite = fail_on_nonfinite
        self.fail_on_nonfinite_grad = (
            fail_on_nonfinite if fail_on_nonfinite_grad is None else fail_on_nonfinite_grad
        )
        if use_ema:
            self.ema = EMA(
                self.model,
                beta=0.9999,
                power=3 / 4,
                update_every=1,
                update_after_step=2000,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def _optimizer_grad_total_norm(self, optimizer: torch.optim.Optimizer):
        norms = []
        device = self.device
        seen_param_ids = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen_param_ids or param.grad is None:
                    continue
                seen_param_ids.add(id(param))
                grad = param.grad.detach()
                device = grad.device
                norms.append(torch.linalg.vector_norm(grad.float(), ord=2))

        if not norms:
            return torch.zeros((), device=device)
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _first_nonfinite_gradient_summary(self) -> str:
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            num_nonfinite = int((~finite_mask).sum().item())
            return (
                f", first_nonfinite_grad={name}, "
                f"grad_shape={tuple(grad.shape)}, "
                f"grad_nonfinite={num_nonfinite}/{grad.numel()}"
            )
        return ""

    def _training_position(self):
        try:
            rank: Any = self.trainer.global_rank
        except RuntimeError:
            rank = "unknown"
        try:
            step: Any = self.global_step
        except RuntimeError:
            step = "unknown"
        return step, rank

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        grad_norm_before = self._optimizer_grad_total_norm(optimizer)
        if self.fail_on_nonfinite_grad and not bool(torch.isfinite(grad_norm_before)):
            step, rank = self._training_position()
            raise FloatingPointError(
                "Non-finite grad_norm/before_clip at "
                f"step={step}, rank={rank}: "
                f"value={float(grad_norm_before.detach().cpu())}"
                f"{self._first_nonfinite_gradient_summary()}"
            )

        self.log(
            "grad_norm/before_clip",
            grad_norm_before,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grad_norm_after = self._optimizer_grad_total_norm(optimizer)
        if self.fail_on_nonfinite_grad and not bool(torch.isfinite(grad_norm_after)):
            step, rank = self._training_position()
            raise FloatingPointError(
                "Non-finite grad_norm/after_clip at "
                f"step={step}, rank={rank}: "
                f"value={float(grad_norm_after.detach().cpu())}"
                f"{self._first_nonfinite_gradient_summary()}"
            )
        self.log(
            "grad_norm/after_clip",
            grad_norm_after,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def _assert_finite(self, name: str, tensor: torch.Tensor, info=None):
        if not self.fail_on_nonfinite or not torch.is_tensor(tensor):
            return

        check = tensor.detach()
        finite_mask = torch.isfinite(check)
        if bool(finite_mask.all()):
            return

        num_nonfinite = int((~finite_mask).sum().item())
        finite_values = check[finite_mask]
        if finite_values.numel() > 0:
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_summary = f", finite_min={finite_min:.6g}, finite_max={finite_max:.6g}"
        else:
            finite_summary = ", no_finite_values=True"

        step, rank = self._training_position()
        paths = ""
        if isinstance(info, dict) and "path" in info:
            path_info = info["path"]
            if isinstance(path_info, (list, tuple)):
                path_info = list(path_info)[:4]
            paths = f", paths={path_info}"

        raise FloatingPointError(
            f"Non-finite {name} at step={step}, rank={rank}: "
            f"shape={tuple(check.shape)}, dtype={check.dtype}, "
            f"nonfinite={num_nonfinite}/{check.numel()}{finite_summary}{paths}"
        )

    def on_before_zero_grad(self, *args, **kwargs):
        if hasattr(self, "ema"):
            self.ema.update()
