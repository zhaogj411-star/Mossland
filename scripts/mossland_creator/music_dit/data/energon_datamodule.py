"""Lightning datamodule for energon-prepared Music-DiT shards."""

from __future__ import annotations

import os

import lightning as pl
import torch
from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset, get_val_dataset

from .taskencoder import MusicDiffusionTaskEncoder


def _resolve_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


class MosslandCreatorEnergonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        seq_length: int | None = 256,
        patch_t: int = 1,
        text_embedding_padding_size: int = 256,
        task_specs: list[dict] | None = None,
        context_dropout: float = 0.0,
        text_dropout: float = 0.0,
        max_duration_seconds: float | None = None,
        shuffle_buffer_size: int = 256,
        max_samples_per_sequence: int = 1,
        train_split: str = "train",
        val_split: str | None = "val",
        use_train_split_for_val: bool = True,
        val_limit: int | None = 32,
        max_abs_latent: float = 1e3,
        pos_dim: int = 1,
        loader_prefetch_factor: int = 2,
    ):
        super().__init__()
        self.path = path
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.seq_length = None if seq_length is None else int(seq_length)
        self.patch_t = int(patch_t)
        self.text_embedding_padding_size = int(text_embedding_padding_size)
        self.task_specs = task_specs
        self.context_dropout = float(context_dropout)
        self.text_dropout = float(text_dropout)
        self.max_duration_seconds = None if max_duration_seconds is None else float(max_duration_seconds)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.max_samples_per_sequence = int(max_samples_per_sequence)
        self.train_split = train_split
        self.val_split = val_split
        self.use_train_split_for_val = bool(use_train_split_for_val)
        self.val_limit = None if val_limit is None else int(val_limit)
        self.max_abs_latent = float(max_abs_latent)
        self.pos_dim = int(pos_dim)
        self.loader_prefetch_factor = int(loader_prefetch_factor)

    def setup(self, stage: str | None = None) -> None:
        return None

    def _worker_config(self) -> WorkerConfig:
        rank, world_size = _resolve_rank_world()
        return WorkerConfig(rank=rank, world_size=world_size, num_workers=self.num_workers)

    def _task_encoder(self) -> MusicDiffusionTaskEncoder:
        return MusicDiffusionTaskEncoder(
            seq_length=self.seq_length,
            patch_t=self.patch_t,
            text_embedding_padding_size=self.text_embedding_padding_size,
            task_specs=self.task_specs,
            context_dropout=self.context_dropout,
            text_dropout=self.text_dropout,
            max_duration_seconds=self.max_duration_seconds,
            max_abs_latent=self.max_abs_latent,
            pos_dim=self.pos_dim,
        )

    def train_dataloader(self):
        dataset = get_train_dataset(
            self.path,
            split_part=self.train_split,
            worker_config=self._worker_config(),
            batch_size=self.train_batch_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            task_encoder=self._task_encoder(),
            repeat=True,
        )
        return get_savable_loader(
            dataset,
            prefetch_factor=self.loader_prefetch_factor,
        )

    def val_dataloader(self):
        if self.val_split is None and not self.use_train_split_for_val:
            return None
        split_part = self.val_split or self.train_split
        try:
            dataset = get_val_dataset(
                self.path,
                split_part=split_part,
                worker_config=self._worker_config(),
                batch_size=self.val_batch_size,
                limit=self.val_limit,
                task_encoder=self._task_encoder(),
                max_samples_per_sequence=self.max_samples_per_sequence,
            )
        except Exception:
            if not self.use_train_split_for_val:
                raise
            dataset = get_val_dataset(
                self.path,
                split_part=self.train_split,
                worker_config=self._worker_config(),
                batch_size=self.val_batch_size,
                limit=self.val_limit,
                task_encoder=self._task_encoder(),
                max_samples_per_sequence=self.max_samples_per_sequence,
            )
        return get_savable_loader(
            dataset,
            prefetch_factor=self.loader_prefetch_factor,
        )
