"""Lightweight Music-DiT training stack for Mossland.

This package ports the useful parts of ``tmp/dit/Sonata_NeMo/music_dit`` into
the local Hydra/Lightning project shape:

- pure torch core/task/model/EDM logic;
- energon-backed datamodule for prepared WebDataset shards;
- a Lightning wrapper that plugs into ``scripts/train.py``.

It intentionally excludes the NeMo/Megatron container-only layer.
"""

