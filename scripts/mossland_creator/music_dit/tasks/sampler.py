"""Per-sample task mixing with target latent routing."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .base import Task
from .registry import build_task


@dataclass
class TaskSpec:
    name: str
    weight: float = 1.0
    target_key: str = "mixture"
    kwargs: dict = field(default_factory=dict)


@dataclass
class SampledTask:
    task: Task
    target_key: str


class TaskMixSampler:
    def __init__(self, specs: list[TaskSpec], context_dropout: float = 0.0):
        if not specs:
            raise ValueError("need at least one task spec")
        self.tasks: list[tuple[Task, float, str]] = [
            (build_task(spec.name, **spec.kwargs), float(spec.weight), str(spec.target_key))
            for spec in specs
        ]
        self.weights = [weight for _task, weight, _target_key in self.tasks]
        self.context_dropout = float(context_dropout)

    @classmethod
    def from_config(cls, cfg: list[dict] | dict, context_dropout: float = 0.0):
        specs: list[TaskSpec] = []
        if isinstance(cfg, dict):
            for name, weight in cfg.items():
                specs.append(TaskSpec(name=name, weight=weight))
        else:
            for item in cfg:
                payload = dict(item)
                name = payload.pop("name")
                weight = payload.pop("weight", 1.0)
                target_key = payload.pop("target_key", "mixture")
                specs.append(TaskSpec(name=name, weight=weight, target_key=target_key, kwargs=payload))
        return cls(specs, context_dropout=context_dropout)

    def sample_task(self, rng: random.Random) -> SampledTask:
        task, _weight, target_key = rng.choices(self.tasks, weights=self.weights, k=1)[0]
        return SampledTask(task=task, target_key=target_key)

    def maybe_dropout_context(self, out, rng: random.Random):
        if self.context_dropout > 0.0 and rng.random() < self.context_dropout:
            out.context_latent = out.context_latent.new_zeros(out.context_latent.shape)
            out.cond_mask = out.cond_mask.new_zeros(out.cond_mask.shape)
        return out

    def names(self) -> list[str]:
        return [task.name for task, _weight, _target_key in self.tasks]

