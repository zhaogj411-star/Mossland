# Copyright (c) 2025.
"""String -> conditioner factory."""

from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register_conditioner(name: str):
    def deco(fn: Callable):
        if name in _REGISTRY:
            raise KeyError(f"conditioner '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_conditioner(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown conditioner '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_conditioners():
    return sorted(_REGISTRY)
