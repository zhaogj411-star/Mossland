from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from .audio_io import save_audio
from .infer import build_task_input, load_mossland_model, prediction_path_for_item
from .manifest import EvalItem


@dataclass(frozen=True)
class AdapterContext:
    device: str | torch.device | None = None
    output_dir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)


class PredictionAdapter(Protocol):
    def predict(
        self,
        item: EvalItem,
        source: torch.Tensor,
        target: torch.Tensor | None,
        context: AdapterContext,
    ) -> torch.Tensor:
        """Return predicted audio with shape [channels, frames]."""


class MosslandCheckpointAdapter:
    def __init__(self, checkpoint_dir: str | Path, device: str | torch.device | None = None):
        self.model = load_mossland_model(checkpoint_dir, device=device)

    @torch.inference_mode()
    def predict(
        self,
        item: EvalItem,
        source: torch.Tensor,
        target: torch.Tensor | None,
        context: AdapterContext,
    ) -> torch.Tensor:
        del target
        quantize = bool(context.options.get("quantized", False))
        device = next(self.model.parameters()).device
        _, generated = self.model.generate_waveform(
            source.unsqueeze(0).to(device),
            task_id=item.task_id,
            quantize=quantize,
        )
        return generated.squeeze(0).detach().cpu()


def parse_adapter_config(raw: str | None) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    path = Path(raw)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw)


def load_python_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("--adapter-target must use 'module:object' syntax")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value: Any = module
    for part in object_name.split("."):
        value = getattr(value, part)
    return value


def _call_factory(factory: Any, context: AdapterContext) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()

    parameters = signature.parameters
    if len(parameters) == 0:
        return factory()
    if "context" in parameters:
        return factory(context=context)
    if "device" in parameters or "options" in parameters:
        kwargs: dict[str, Any] = {}
        if "device" in parameters:
            kwargs["device"] = context.device
        if "options" in parameters:
            kwargs["options"] = context.options
        return factory(**kwargs)
    return factory(context)


def load_prediction_adapter(
    *,
    checkpoint_dir: str | Path | None = None,
    adapter_target: str | None = None,
    context: AdapterContext,
) -> PredictionAdapter | None:
    if checkpoint_dir and adapter_target:
        raise ValueError("Use either --checkpoint-dir or --adapter-target, not both.")
    if checkpoint_dir:
        return MosslandCheckpointAdapter(checkpoint_dir, device=context.device)
    if not adapter_target:
        return None

    candidate = load_python_object(adapter_target)
    adapter = candidate if hasattr(candidate, "predict") else _call_factory(candidate, context)
    if not hasattr(adapter, "predict"):
        raise TypeError(
            f"adapter target {adapter_target!r} did not produce an object with a predict() method"
        )
    return adapter


@torch.inference_mode()
def generate_adapter_prediction(
    adapter: PredictionAdapter,
    item: EvalItem,
    output_dir: str | Path,
    *,
    context: AdapterContext,
    overwrite: bool = False,
) -> Path:
    output_path = prediction_path_for_item(item, output_dir)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return output_path.resolve()

    torch.manual_seed(int(item.seed))
    source, target = build_task_input(item)
    prediction = adapter.predict(item, source, target, context)
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.as_tensor(prediction)
    prediction = prediction.detach().cpu().float()
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 2:
        raise ValueError(
            "adapter predict() must return audio with shape [channels, frames], "
            f"got {tuple(prediction.shape)} for item_id={item.item_id!r}"
        )
    save_audio(output_path, prediction, item.sample_rate)
    return output_path.resolve()
