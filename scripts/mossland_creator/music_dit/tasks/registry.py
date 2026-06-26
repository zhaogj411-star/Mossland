"""Task registry."""

from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_task(name: str):
    def deco(fn: Callable):
        _REGISTRY[name] = fn
        return fn

    return deco


def build_task(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; available={sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_tasks() -> list[str]:
    return sorted(_REGISTRY)

