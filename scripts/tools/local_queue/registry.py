from __future__ import annotations

import importlib

from .api import TaskHandler


def load_handler(spec: str) -> TaskHandler:
    """Load a handler from 'module:ClassName'."""

    if ":" not in spec:
        raise ValueError("handler must be in 'module:ClassName' format")

    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    handler = cls()

    if not isinstance(handler, TaskHandler):
        raise TypeError(f"{spec} is not a TaskHandler")

    return handler
