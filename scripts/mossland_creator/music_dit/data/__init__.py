"""Data utilities for Mossland Music-DiT."""

from __future__ import annotations

from .assemble import collate_tokens, encode_sample_to_tokens

__all__ = [
    "collate_tokens",
    "encode_sample_to_tokens",
    "MosslandCreatorEnergonDataModule",
    "MusicDiffusionTaskEncoder",
]


def __getattr__(name: str):
    if name == "MusicDiffusionTaskEncoder":
        from .taskencoder import MusicDiffusionTaskEncoder

        return MusicDiffusionTaskEncoder
    if name == "MosslandCreatorEnergonDataModule":
        from .energon_datamodule import MosslandCreatorEnergonDataModule

        return MosslandCreatorEnergonDataModule
    raise AttributeError(name)
