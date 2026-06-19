import argparse
import json
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T


def find_audio_files(roots, extensions):
    normalized_extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    )
    files = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix.lower() in normalized_extensions:
            files.append(root_path)
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in normalized_extensions:
                files.append(path)
    return sorted(files)


def read_audio_index(index_files, extensions, max_files=None, seed=42):
    normalized_extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    )
    rng = random.Random(seed)
    files = []
    seen = 0
    for index_file in index_files:
        with Path(index_file).open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                path = Path(value)
                if path.suffix.lower() not in normalized_extensions:
                    continue
                seen += 1
                if max_files is None or len(files) < max_files:
                    files.append(path)
                    continue
                replace_idx = rng.randrange(seen)
                if replace_idx < max_files:
                    files[replace_idx] = path
    return files


def _load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    try:
        data, in_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(data).transpose(0, 1)
    except Exception:
        audio, in_sample_rate = torchaudio.load(str(path))
    if in_sample_rate != sample_rate:
        audio = T.Resample(in_sample_rate, sample_rate)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    return audio[:2]


def _official_random_mono_crop(
    audio: torch.Tensor,
    sample_length: int,
    generator: torch.Generator | None = None,
):
    if audio.shape[-1] < sample_length:
        raise ValueError(
            f"audio length {audio.shape[-1]} is shorter than sample_length {sample_length}"
        )
    max_start = audio.shape[-1] - sample_length
    start = (
        0
        if max_start == 0
        else torch.randint(max_start, size=(1,), generator=generator).item()
    )
    crop = audio[:, start : start + sample_length]
    channel = torch.randint(crop.shape[0], size=(1,), generator=generator).item()
    return crop[channel].clone().contiguous(), start, channel


def _prepare_one_official_pt(task):
    out_index, source_path, out_file, sample_rate, sample_length, seed = task
    out_file = Path(out_file)
    source_path = Path(source_path)
    if out_file.exists():
        return {
            "ok": True,
            "pt_path": str(out_file),
            "source_path": str(source_path),
            "existing": True,
        }
    try:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(out_index))
        audio = _load_audio(source_path, sample_rate)
        crop, start, channel = _official_random_mono_crop(
            audio,
            sample_length,
            generator=generator,
        )
        payload = {
            "audio": crop.cpu(),
            "source_path": str(source_path),
            "source_start": int(start),
            "source_channel": int(channel),
            "sample_rate": int(sample_rate),
            "sample_length": int(sample_length),
        }
        tmp_file = out_file.with_name(f"{out_file.name}.{os.getpid()}.tmp")
        torch.save(payload, tmp_file)
        os.replace(tmp_file, out_file)
        return {
            "ok": True,
            "pt_path": str(out_file),
            "source_path": str(source_path),
            "source_start": int(start),
            "source_channel": int(channel),
            "existing": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "pt_path": str(out_file),
            "source_path": str(source_path),
            "error": str(exc),
        }


class Music2LatentOfficialPTDataset(Dataset):
    """Read pre-cropped official Music2Latent mono waveform `.pt` samples."""

    def __init__(
        self,
        root,
        index_file=None,
        key="audio",
        length=None,
        strict=False,
    ):
        self.root = Path(root)
        self.index_file = Path(index_file) if index_file is not None else self.root / "files.list"
        self.key = key
        self.length = int(length) if length is not None else None
        self.strict = bool(strict)
        self.refresh_files()

    def refresh_files(self):
        if self.index_file.exists():
            lines = self.index_file.read_text(encoding="utf-8").splitlines()
            self.files = [
                path if path.is_absolute() else self.root / path
                for path in (Path(line.strip()) for line in lines if line.strip())
            ]
        else:
            self.files = sorted(self.root.rglob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found under {self.root}")
        print(f"refresh:Loaded {len(self.files)} Music2Latent official pt samples")

    def __len__(self):
        return self.length if self.length is not None else len(self.files)

    def __getitem__(self, index):
        while self.files:
            path = self.files[index % len(self.files)]
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                audio = payload[self.key] if isinstance(payload, dict) else payload
                if audio.ndim != 1:
                    audio = audio.squeeze()
                if audio.ndim != 1:
                    raise ValueError(f"expected mono waveform tensor, got {tuple(audio.shape)}")
                info = {"path": str(path)}
                if isinstance(payload, dict):
                    for key in (
                        "source_path",
                        "source_start",
                        "source_channel",
                        "sample_rate",
                        "sample_length",
                    ):
                        if key in payload:
                            info[key] = payload[key]
                return audio.float().contiguous(), info
            except Exception:
                if self.strict or len(self.files) <= 1:
                    raise
                self.files.pop(index % len(self.files))
                index = random.randrange(len(self.files))
        raise RuntimeError("No valid Music2Latent official pt samples remain")


def prepare_official_pt_dataset(
    source_dirs,
    output_dir,
    source_indexes=None,
    num_samples=10_000,
    sample_rate=44_100,
    hop=512,
    fac=4,
    data_length=64,
    extensions=(".wav", ".flac"),
    seed=42,
    num_workers=1,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files_list_path = output_path / "files.list"
    index_jsonl_path = output_path / "index.jsonl"
    sample_length = int(hop) * int(data_length) + (int(fac) - 1) * int(hop)
    if source_indexes:
        files = read_audio_index(
            source_indexes,
            extensions,
            max_files=max(int(num_samples) * 50, int(num_samples)),
            seed=seed,
        )
    else:
        files = find_audio_files(source_dirs, extensions)
    if not files:
        raise RuntimeError(f"No audio files found in {source_dirs} with {extensions}")

    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    written = []
    records = []

    tasks = [
        (
            idx,
            shuffled[idx % len(shuffled)],
            output_path / f"{idx:06d}.pt",
            int(sample_rate),
            int(sample_length),
            int(seed),
        )
        for idx in range(int(num_samples))
    ]

    completed = 0
    if int(num_workers) <= 1:
        results_iter = (_prepare_one_official_pt(task) for task in tasks)
        for result in results_iter:
            completed += 1
            records.append(result)
            if result["ok"]:
                written.append(Path(result["pt_path"]))
            if completed % 100 == 0 or completed == num_samples:
                print(
                    f"prepared={len(written)}/{num_samples} attempts={completed}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            futures = [executor.submit(_prepare_one_official_pt, task) for task in tasks]
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                records.append(result)
                if result["ok"]:
                    written.append(Path(result["pt_path"]))
                if completed % 100 == 0 or completed == num_samples:
                    print(
                        f"prepared={len(written)}/{num_samples} attempts={completed}",
                        flush=True,
                    )

    written = sorted(set(written))
    if len(written) < num_samples:
        missing = [
            idx
            for idx in range(int(num_samples))
            if not (output_path / f"{idx:06d}.pt").exists()
        ]
        retry_attempts = 0
        source_offset = int(num_samples)
        while missing and retry_attempts < int(num_samples) * 10:
            idx = missing.pop(0)
            task = (
                idx,
                shuffled[(source_offset + retry_attempts) % len(shuffled)],
                output_path / f"{idx:06d}.pt",
                int(sample_rate),
                int(sample_length),
                int(seed) + 1_000_000,
            )
            retry_attempts += 1
            result = _prepare_one_official_pt(task)
            records.append(result)
            if result["ok"]:
                written.append(Path(result["pt_path"]))
            else:
                missing.append(idx)
            if retry_attempts % 100 == 0 or not missing:
                print(
                    f"retry_prepared={len(set(written))}/{num_samples} retry_attempts={retry_attempts}",
                    flush=True,
                )
    written = sorted(set(written))

    files_list_path.write_text(
        "".join(f"{path.relative_to(output_path)}\n" for path in written),
        encoding="utf-8",
    )
    with index_jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(written) < num_samples:
        raise RuntimeError(
            f"Only prepared {len(written)} / {num_samples} samples; see {index_jsonl_path}"
        )
    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare official Music2Latent .pt crops.")
    parser.add_argument("--source-dir", action="append", default=[])
    parser.add_argument("--source-index", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--fac", type=int, default=4)
    parser.add_argument("--data-length", type=int, default=64)
    parser.add_argument("--extension", action="append", default=[".wav", ".flac"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    prepare_official_pt_dataset(
        source_dirs=args.source_dir,
        output_dir=args.output_dir,
        source_indexes=args.source_index,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        hop=args.hop,
        fac=args.fac,
        data_length=args.data_length,
        extensions=args.extension,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
