import importlib
import json
import numpy as np

import os

import random
import subprocess

import time
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import lightning as pl
from aeiou.core import is_silence
from os import path
from pathlib import Path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List, Sequence
import librosa

from scripts.data.utils import (
    Stereo,
    Mono,
    PhaseFlipper,
    PadCrop_Normalized_T,
)
import logging

# 配置日志记录 - 只输出到文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("dataloading_info.log", mode="a"),
    ],
)

# 创建一个专用的日志记录器
logger = logging.getLogger("dataloading")


def fast_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions,
    # max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = [
        "." + x if x[0] != "." else x for x in ext
    ]  # add starting period to extensions if needed
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ["." + x if x[0] != "." else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == "."
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any([keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words]
                    )
                    if (
                        has_ext
                        and has_keyword
                        and not has_banned
                        and not is_hidden
                        and not os.path.basename(f.path).startswith("._")
                    ):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=[".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus"],
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:  # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames


import os
import torch
from torch.utils.data import Dataset

import random


class PTDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mainkey,
        infokeys,
        target_length,
        nums=None,
        sub_length=None,
        files_list=None,
        use_multiple_start=True,
        if_random_crop=True,
    ):
        super().__init__()
        self.nums = nums
        self.folder_path = folder_path
        self.target_length = target_length
        self.pt_files = []
        self.sub_length = sub_length
        self.files_list = files_list
        self.use_multiple_start = use_multiple_start
        self.if_random_crop = if_random_crop
        self.refresh_filenames()
        self.mainkey = mainkey
        self.infokeys = infokeys

    def refresh_filenames(self):
        self.pt_files = []

        if self.files_list and os.path.exists(self.files_list):
            # 如果提供了文件列表路径，从文件中读取路径
            with open(self.files_list, "r") as f:
                for line in f:
                    file_path = line.strip()
                    if file_path.endswith(".pt"):
                        self.pt_files.append(file_path)
        else:
            # 否则从文件夹中扫描
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if file.endswith(".pt"):
                        self.pt_files.append(os.path.join(root, file))

        # self.pt_files = self.pt_files[100:110]
        if self.sub_length is not None:
            self.pt_files = self.pt_files[: self.sub_length]
        print(f"refresh:Found {len(self.pt_files)} files")

    def ramdom_crop(self, data):
        if len(data.shape) > 2:
            data = data.squeeze(0)
        data_length = data.shape[1]
        if data_length > self.target_length:
            # 确保起始位置是 target_length 的倍数
            max_start = data_length - self.target_length
            if self.use_multiple_start:
                valid_starts = list(range(0, max_start + 1, self.target_length))
                if not valid_starts:  # 如果没有有效的起始位置，则从0开始
                    start = 0
                else:
                    start = random.choice(valid_starts)
            else:
                start = random.randint(0, max_start)
            data = data[:, start : start + self.target_length]
        elif data_length < self.target_length:
            # data = torch.cat(
            #     [data, torch.zeros(data.shape[0], self.target_length - data_length)],
            #     dim=1,
            # )
            raise ValueError(
                f"data length {data_length} is less than target length {self.target_length}"
            )
        return data, start, start + self.target_length

    def __len__(self):
        return len(self.pt_files) if self.nums is None else self.nums

    def __getitem__(self, idx):
        idx = idx % len(self.pt_files)
        try:
            file_path = self.pt_files[idx]
            data = torch.load(file_path, map_location="cpu", weights_only=False)

            main_key_info = data.get(self.mainkey, None)

            # 检查main_key_info是否为None或不是张量
            if main_key_info is None:
                raise ValueError(f"主键 {self.mainkey} 在文件 {file_path} 中不存在")

            # 检查main_key_info是否为整数（这会导致'int' object is not subscriptable错误）
            if isinstance(main_key_info, int):
                raise ValueError(f"主键 {self.mainkey} 的值是整数而不是张量")

            # 确保main_key_info是张量并且有正确的维度
            if not isinstance(main_key_info, torch.Tensor):
                raise ValueError(f"主键 {self.mainkey} 的值不是张量")

            # main_key_info = main_key_info[:,:self.target_length]
            if self.if_random_crop:
                main_key_info, start, end = self.ramdom_crop(main_key_info)
            else:
                if main_key_info.shape[1] > self.target_length:
                    main_key_info = main_key_info[:, : self.target_length]
                    start = 0
                    end = self.target_length
                else:
                    raise ValueError(
                        f"data length {main_key_info.shape[1]} is less than target length {self.target_length}"
                    )

            info_keys_info = {
                key: data.get(key, None) for key in self.infokeys if key in data
            }

            return main_key_info, info_keys_info
        except Exception as e:
            self.pt_files.remove(file_path)
            logging.info(f"total_length:{len(self.pt_files)},{file_path} :{e}")
            return self.__getitem__(idx + 1)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dirs,
        sample_size=65536,
        sample_rate=48000,
        keywords=None,
        random_crop=True,
        no_channel_dim=False,
        num_channels=2,
        audio_cache_dir="/home/gjzhao/audio_cache/",
        crops_per_file=1,
    ):
        super().__init__()
        self.dirs = dirs
        self.filenames = []
        self.no_channel_dim = no_channel_dim
        self.crops_per_file = max(1, int(crops_per_file))
        if audio_cache_dir is not None:
            os.makedirs(audio_cache_dir, exist_ok=True)
            self.audio_cache_dir = audio_cache_dir
        else:
            self.audio_cache_dir = None
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(
            sample_size, sample_rate, randomize=random_crop
        )
        self.num_channels = num_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.num_channels == 2 else torch.nn.Identity(),
            Mono() if self.num_channels == 1 else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}
        self.refresh_filenames()
        # for dir in dirs:
        #     self.filenames.extend(get_audio_filenames(dir, None))
        # # self.filenames = self.filenames[:1]
        # print(f"Found {len(self.filenames)} files")

    def refresh_filenames(self):
        self.filenames = []
        for dir in self.dirs:
            self.filenames.extend(get_audio_filenames(dir, None))
        # self.filenames = self.filenames[:2]
        print(f"refresh:Found {len(self.filenames)} files")

    def load_file(self, filename):
        ext = filename.split(".")[-1]
        basename = path.basename(filename)
        if self.audio_cache_dir is not None:
            target_path = path.join(
                self.audio_cache_dir, f"{basename}_{self.num_channels}_{self.sr}.pt"
            )
            if os.path.exists(target_path):
                audio = torch.load(target_path, map_location="cpu")
                return audio

        if ext == "mp3":
            # with AudioFile(filename) as f:
            #     audio = f.read(f.frames)
            #     audio = torch.from_numpy(audio)
            #     in_sr = f.samplerate
            audio, in_sr = librosa.load(filename, sr=None, mono=False)
            audio = torch.from_numpy(audio)
        else:
            audio, in_sr = torchaudio.load(filename)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        if self.audio_cache_dir is not None:
            torch.save(audio.cpu().detach(), target_path)
        return audio

    def _crop_audio(self, audio):
        audio, t_start, t_end, seconds_start, seconds_total, padding_mask = (
            self.pad_crop(audio)
        )

        if self.augs is not None:
            audio = self.augs(audio)

        audio = audio.clamp(-1, 1)

        if self.encoding is not None:
            audio = self.encoding(audio)

        if self.no_channel_dim:
            audio = audio.mean(dim=0)

        return audio, {
            "timestamps": (t_start, t_end),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
            "padding_mask": padding_mask,
        }

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):

        try:
            idx = idx % len(self.filenames)
            audio_filename = self.filenames[idx]
            start_time = time.time()
            audio = self.load_file(audio_filename)

            crops = []
            crop_infos = []
            for _ in range(self.crops_per_file):
                crop, crop_info = self._crop_audio(audio)
                crops.append(crop)
                crop_infos.append(crop_info)

            if self.crops_per_file == 1:
                audio = crops[0]
                crop_info = crop_infos[0]
            else:
                audio = torch.stack(crops, dim=0)
                crop_info = {
                    "timestamps": torch.tensor(
                        [info["timestamps"] for info in crop_infos],
                        dtype=torch.float32,
                    ),
                    "seconds_start": torch.tensor(
                        [info["seconds_start"] for info in crop_infos],
                        dtype=torch.int64,
                    ),
                    "seconds_total": torch.tensor(
                        [info["seconds_total"] for info in crop_infos],
                        dtype=torch.int64,
                    ),
                    "padding_mask": torch.stack(
                        [info["padding_mask"] for info in crop_infos],
                        dim=0,
                    ),
                }

            info = {"path": audio_filename, "crops_per_file": self.crops_per_file}

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)
            info.update(crop_info)
            end_time = time.time()
            info["load_time"] = end_time - start_time
            return (audio, info)
        except Exception as e:
            print(f"Couldn't load file {audio_filename}: {e}")
            self.filenames.remove(audio_filename)
            return self[random.randrange(len(self))]


def _pad_or_crop_to_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] == target_length:
        return audio
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    return torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))


def _normalize_max_duration_seconds(max_duration_seconds: float | int | None) -> float | None:
    if max_duration_seconds is None:
        return None
    value = float(max_duration_seconds)
    if value <= 0:
        return None
    return value


def _normalize_positive_int_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return normalized


def _probe_audio_duration_seconds_ffprobe(path_: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path_),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    duration_lines = completed.stdout.strip().splitlines()
    if not duration_lines:
        return None
    try:
        duration = float(duration_lines[0])
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


class PreparedSeparationDataset(Dataset):
    """直接读取 prepare_separation.py 已生成的离线分离目录。"""

    def __init__(
        self,
        dirs,
        index_file=None,
        sample_size: int = 65536,
        sample_rate: int = 48000,
        random_crop: bool = True,
        num_channels: int = 2,
        strict: bool = False,
        scan_fallback: bool = True,
        crops_per_file: int = 1,
        max_duration_seconds: float | int | None = None,
        length: int | None = None,
    ):
        super().__init__()
        self.dirs = (
            [Path(dirs)]
            if isinstance(dirs, (str, os.PathLike))
            else [Path(item) for item in dirs]
        )
        if index_file is None:
            self.index_sources = [(root, root / "index.list") for root in self.dirs]
        else:
            index_files = (
                [Path(index_file)]
                if isinstance(index_file, (str, os.PathLike))
                else [Path(item) for item in index_file]
            )
            if len(index_files) == 1:
                self.index_sources = [(self.dirs[0], index_files[0])]
            elif len(index_files) == len(self.dirs):
                self.index_sources = list(zip(self.dirs, index_files))
            else:
                raise ValueError("index_file must be a path or match dirs length")
        self.sample_size = int(sample_size)
        self.sample_rate = int(sample_rate)
        self.random_crop = bool(random_crop)
        self.num_channels = int(num_channels)
        self.strict = bool(strict)
        self.scan_fallback = bool(scan_fallback)
        self.crops_per_file = max(1, int(crops_per_file))
        self.max_duration_seconds = _normalize_max_duration_seconds(max_duration_seconds)
        self.length = _normalize_positive_int_or_none(length, "length")
        self._duration_limit_cache: dict[Path, bool] = {}
        self.item_dirs: list[Path] = []
        self.refresh_items()

    def refresh_items(self, rebuild_index: bool = False):
        item_dirs = []
        for root, index_path in self.index_sources:
            if rebuild_index:
                scanned = self._scan_item_dirs(root)
                self._write_index(root, index_path, scanned)
                item_dirs.extend(scanned)
                continue
            if index_path.exists():
                item_dirs.extend(self._read_index(root, index_path))
                continue
            if self.scan_fallback:
                scanned = self._scan_item_dirs(root)
                self._write_index(root, index_path, scanned)
                item_dirs.extend(scanned)
        self.item_dirs = sorted(set(item_dirs))
        if rebuild_index:
            self._duration_limit_cache.clear()
        print(f"refresh:Found {len(self.item_dirs)} prepared separation items")

    def rebuild_index(self):
        self.refresh_items(rebuild_index=True)

    def _read_index(self, root: Path, index_path: Path) -> list[Path]:
        item_dirs = []
        try:
            lines = index_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return item_dirs
        for line in lines:
            value = line.strip()
            if not value:
                continue
            path_ = Path(value)
            item_dirs.append(path_ if path_.is_absolute() else root / path_)
        return item_dirs

    def _write_index(self, root: Path, index_path: Path, item_dirs: list[Path]) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for item_dir in sorted(set(item_dirs)):
            try:
                lines.append(str(item_dir.relative_to(root)))
            except ValueError:
                lines.append(str(item_dir))
        tmp_path = index_path.with_name(
            f"{index_path.name}.{os.getpid()}.{id(self)}.tmp"
        )
        tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp_path, index_path)

    def _item_exceeds_duration_limit(self, item_dir: Path) -> bool:
        if self.max_duration_seconds is None:
            return False
        if item_dir in self._duration_limit_cache:
            return self._duration_limit_cache[item_dir]
        duration = _probe_audio_duration_seconds_ffprobe(item_dir / "mixture.mp3")
        exceeds = duration is not None and duration >= self.max_duration_seconds
        self._duration_limit_cache[item_dir] = exceeds
        return exceeds

    def _skip_overlong_item(self, index: int, item_dir: Path) -> None:
        print(
            "Skip overlong prepared separation item "
            f"{item_dir}: max_duration_seconds={self.max_duration_seconds}"
        )
        self.item_dirs.pop(index)

    def _scan_item_dirs(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        scan_root = root / "audio" if (root / "audio").exists() else root
        _, files = fast_scandir(str(scan_root), [".mp3"])
        file_set = set(files)
        item_dirs = []
        for file_path in files:
            path_ = Path(file_path)
            if path_.name != "mixture.mp3":
                continue
            item_dir = path_.parent
            if (
                str(item_dir / "vocals.mp3") in file_set
                and str(item_dir / "accompaniment.mp3") in file_set
            ):
                item_dirs.append(item_dir)
        return sorted(set(item_dirs))

    def _read_metadata(self, item_dir: Path) -> dict:
        metadata_path = item_dir / "metadata.json"
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def __len__(self):
        return self.length if self.length is not None else len(self.item_dirs)

    def _load_stem(self, path_: Path) -> torch.Tensor:
        audio, in_sample_rate = torchaudio.load(str(path_))
        audio = audio.float()
        if in_sample_rate != self.sample_rate:
            audio = T.Resample(in_sample_rate, self.sample_rate)(audio)
        if self.num_channels == 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.shape[0] == 1 and self.num_channels == 2:
            audio = audio.repeat_interleave(2, dim=0)
        elif audio.shape[0] > self.num_channels:
            audio = audio[: self.num_channels]
        return audio.clamp(-1.0, 1.0)

    def _crop_start(self, audio: torch.Tensor) -> int:
        max_start = max(0, int(audio.shape[-1]) - self.sample_size)
        if not self.random_crop or max_start <= 0:
            return 0
        return random.randint(0, max_start)

    def _crop_stem(self, audio: torch.Tensor, start: int) -> torch.Tensor:
        crop = audio[..., start : start + self.sample_size]
        return _pad_or_crop_to_length(crop, self.sample_size)

    def _relative_item_path(self, item_dir: Path) -> str:
        for root in self.dirs:
            try:
                return str(item_dir.relative_to(root))
            except ValueError:
                continue
        return item_dir.name

    def _stem_names_for_task(self, task_id: str | None) -> tuple[str, ...]:
        if task_id == "separate_vocals":
            return ("mixture", "vocals")
        if task_id == "separate_accompaniment":
            return ("mixture", "accompaniment")
        if task_id is None:
            return ("mixture", "vocals", "accompaniment")
        return ("mixture",)

    def _stem_names_for_tasks(self, task_ids: Sequence[str]) -> tuple[str, ...]:
        stem_names = ["mixture"]
        if any(task_id == "separate_vocals" for task_id in task_ids):
            stem_names.append("vocals")
        if any(task_id == "separate_accompaniment" for task_id in task_ids):
            stem_names.append("accompaniment")
        return tuple(stem_names)

    def _load_item(
        self,
        item_dir: Path,
        task_id: str | None = None,
        task_ids: Sequence[str] | None = None,
    ):
        metadata = self._read_metadata(item_dir)
        if task_ids is None:
            stem_names = self._stem_names_for_task(task_id)
            crop_count = self.crops_per_file
        else:
            task_ids = tuple(task_ids)
            if not task_ids:
                raise ValueError("task_ids must not be empty")
            stem_names = self._stem_names_for_tasks(task_ids)
            crop_count = len(task_ids)
        stems = {
            stem_name: self._load_stem(item_dir / f"{stem_name}.mp3")
            for stem_name in stem_names
        }
        mixture = stems["mixture"]
        starts = [self._crop_start(mixture) for _ in range(crop_count)]

        cropped_stems = {
            stem_name: torch.stack(
                [self._crop_stem(stem_audio, start) for start in starts],
                dim=0,
            )
            for stem_name, stem_audio in stems.items()
        }
        if crop_count == 1:
            cropped_stems = {
                stem_name: stem_audio[0]
                for stem_name, stem_audio in cropped_stems.items()
            }

        mixture = cropped_stems["mixture"]
        payload = {"audio": mixture, "mixture": mixture}
        for stem_name in stem_names:
            if stem_name == "mixture":
                continue
            payload[stem_name] = cropped_stems[stem_name]
        if crop_count == 1:
            sample_start = starts[0]
            sample_end = sample_start + self.sample_size
        else:
            sample_start = torch.tensor(starts, dtype=torch.int64)
            sample_end = sample_start + self.sample_size
        info = {
            "path": str(metadata.get("source_path") or item_dir),
            "relpath": self._relative_item_path(item_dir),
            "separation_dir": str(item_dir),
            "sample_start": sample_start,
            "sample_end": sample_end,
            "crops_per_file": crop_count,
        }
        if task_ids is not None:
            info["task_ids"] = task_ids
        return payload, info

    def __getitem__(self, index):
        return self.get_item_for_task(index, None)

    def get_item_for_task(self, index, task_id: str | None):
        while self.item_dirs:
            index = index % len(self.item_dirs)
            item_dir = self.item_dirs[index]
            if self._item_exceeds_duration_limit(item_dir):
                self._skip_overlong_item(index, item_dir)
                if not self.item_dirs:
                    break
                index = random.randrange(len(self.item_dirs))
                continue
            try:
                return self._load_item(item_dir, task_id)
            except Exception as exc:
                if self.strict or len(self.item_dirs) <= 1:
                    raise
                print(f"Couldn't load prepared separation item {item_dir}: {exc}")
                self.item_dirs.pop(index)
                index = random.randrange(len(self.item_dirs))
        raise RuntimeError("PreparedSeparationDataset 没有可用的 done 分离结果")

    def get_item_for_tasks(self, index, task_ids: Sequence[str]):
        task_ids = tuple(task_ids)
        while self.item_dirs:
            index = index % len(self.item_dirs)
            item_dir = self.item_dirs[index]
            if self._item_exceeds_duration_limit(item_dir):
                self._skip_overlong_item(index, item_dir)
                if not self.item_dirs:
                    break
                index = random.randrange(len(self.item_dirs))
                continue
            try:
                return self._load_item(item_dir, task_ids=task_ids)
            except Exception as exc:
                if self.strict or len(self.item_dirs) <= 1:
                    raise
                print(f"Couldn't load prepared separation item {item_dir}: {exc}")
                self.item_dirs.pop(index)
                index = random.randrange(len(self.item_dirs))
        raise RuntimeError("PreparedSeparationDataset 没有可用的 done 分离结果")


class Experiment_Dataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.05,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.prefetch_factor = prefetch_factor
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        # 实例化数据集
        dataset = self.dataset
        dataset_size = len(dataset)

        # 计算训练集和验证集的大小
        if self.val_split is not None and dataset_size > 1 and float(self.val_split) > 0:
            if float(self.val_split) < 1:
                val_size = max(1, int(dataset_size * float(self.val_split)))
            else:
                val_size = int(self.val_split)
            val_size = min(val_size, dataset_size - 1)
            train_size = dataset_size - val_size

            # 随机拆分数据集
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = None

    def prepare_data(self):
        # 用于数据下载等操作
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.val_split is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.val_split is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dataset = SampleDataset(
        [
            "/home/gjzhao/data/music/",
        ],
        sample_size=65536,
        sample_rate=48000,
        keywords=["mp3", "wav", "flac", "ogg", "aif", "opus"],
        random_crop=True,
    )
    a = dataset[0]
    b = 0
