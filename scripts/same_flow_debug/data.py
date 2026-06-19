import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedAudioSegmentDataset(Dataset):
    """Read short fixed audio tensors prepared for same_flow overfit/debug runs."""

    def __init__(
        self,
        index_file: str,
        length: int | None = None,
        sample_size: int | None = None,
        num_channels: int = 2,
        no_channel_dim: bool = False,
    ):
        super().__init__()
        self.index_file = Path(index_file)
        self.length = int(length) if length is not None else None
        self.sample_size = int(sample_size) if sample_size is not None else None
        self.num_channels = int(num_channels)
        self.no_channel_dim = bool(no_channel_dim)
        if not self.index_file.exists():
            raise FileNotFoundError(f"missing cached audio index: {self.index_file}")
        self.rows = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"cached audio index is empty: {self.index_file}")

    def __len__(self):
        return self.length if self.length is not None else len(self.rows)

    def _pad_or_crop(self, audio: torch.Tensor) -> torch.Tensor:
        if self.sample_size is None:
            return audio
        if audio.shape[-1] > self.sample_size:
            return audio[..., : self.sample_size]
        if audio.shape[-1] < self.sample_size:
            audio = torch.nn.functional.pad(audio, (0, self.sample_size - audio.shape[-1]))
        return audio

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        payload = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
        audio = payload["audio"].float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > self.num_channels:
            audio = audio[: self.num_channels]
        elif audio.shape[0] < self.num_channels:
            audio = audio.expand(self.num_channels, -1).contiguous()
        audio = self._pad_or_crop(audio).clamp(-1.0, 1.0)
        if self.no_channel_dim:
            audio = audio.mean(dim=0)
        info = dict(row)
        info.update(payload.get("info", {}))
        info["path"] = row.get("source_path", row["cache_path"])
        return audio, info
