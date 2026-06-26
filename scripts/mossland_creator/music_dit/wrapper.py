"""Lightning wrapper for the pure-torch Mossland Music-DiT."""

from __future__ import annotations

import lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .core import contract as C
from .diffusion import MusicEDM, MusicFlowMatching


class MusicDiTTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        lr_schedule: str = "cosine_decay",
        lr_warmup_steps: int = 1000,
        lr_schedule_total_steps: int = 200000,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        objective_name: str = "edm",
        sigma_data: float | None = None,
        flow_matching_estimator_target: str = "conditional_vector_field",
        flow_matching_time_min: float = 1e-5,
        flow_matching_time_max: float = 1.0,
        flow_matching_sigma_start: float = 1.0,
        flow_matching_sigma_end: float = 1e-4,
        fail_on_nonfinite: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_schedule_total_steps = int(lr_schedule_total_steps)
        self.objective_name = str(objective_name)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        objective_sigma_data = self.model.sigma_data if sigma_data is None else float(sigma_data)
        if self.objective_name == "edm":
            self.objective = MusicEDM(
                net=self.model,
                sigma_data=objective_sigma_data,
                p_mean=p_mean,
                p_std=p_std,
            )
        elif self.objective_name == "flow_matching":
            self.objective = MusicFlowMatching(
                net=self.model,
                data_scale=objective_sigma_data,
                estimator_target=flow_matching_estimator_target,
                time_min=flow_matching_time_min,
                time_max=flow_matching_time_max,
                sigma_start=flow_matching_sigma_start,
                sigma_end=flow_matching_sigma_end,
            )
        else:
            raise ValueError(f"Unsupported objective_name={self.objective_name!r}")

    def _assert_finite(self, name: str, value: torch.Tensor) -> None:
        if not self.fail_on_nonfinite:
            return
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise RuntimeError(f"{name} contains non-finite values")

    def _log_task_stats(self, task_names) -> None:
        if task_names is None:
            return
        if isinstance(task_names, str):
            task_names = [task_names]
        if not isinstance(task_names, (list, tuple)):
            return
        total = len(task_names)
        if total == 0:
            return
        for task_name in dict.fromkeys(task_names):
            self.log(
                f"task/{task_name}",
                torch.tensor(task_names.count(task_name) / total, device=self.device),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self.objective.training_loss(batch)
        self._assert_finite("loss", loss)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.compute_loss(batch)
        self.log("loss/total", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self._log_task_stats(batch.get(C.KEY_TASK))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for name, value in metrics.items():
            self.log(
                f"val/{name.split('/', 1)[-1]}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        return loss

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}")

        if self.lr_schedule == "constant":
            return optimizer
        if self.lr_schedule != "cosine_decay":
            raise ValueError(f"Unsupported lr_schedule={self.lr_schedule!r}")

        decay_steps = max(1, self.lr_schedule_total_steps - self.lr_warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0.0)
        if self.lr_warmup_steps <= 0:
            scheduler = cosine
        else:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.lr_warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.lr_warmup_steps],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
