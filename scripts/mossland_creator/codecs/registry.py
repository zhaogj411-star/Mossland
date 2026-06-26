# Copyright (c) 2025.
"""String -> codec factory."""

from __future__ import annotations

from typing import Callable, Dict

from .base import AudioCodec

_REGISTRY: Dict[str, Callable[..., AudioCodec]] = {}


def register_codec(name: str):
    def deco(fn: Callable[..., AudioCodec]):
        if name in _REGISTRY:
            raise KeyError(f"codec '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_codec(name: str, **kwargs) -> AudioCodec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_codecs():
    return sorted(_REGISTRY)
