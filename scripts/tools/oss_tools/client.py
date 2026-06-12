from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_RCLONE = TOOL_ROOT / "bin" / "rclone"
DEFAULT_RCLONE_CONFIG = TOOL_ROOT / "rclone.conf"


@dataclass(frozen=True)
class OssPath:
    """A parsed rclone remote path such as 'qz_oss2:public/path/file.txt'."""

    remote: str
    path: str = ""

    @classmethod
    def parse(cls, value: str) -> "OssPath":
        if ":" not in value:
            raise ValueError("OSS path must be in 'remote:bucket/key' format")
        remote, path = value.split(":", 1)
        if not remote:
            raise ValueError("OSS remote name cannot be empty")
        return cls(remote=remote, path=path.lstrip("/"))

    def __str__(self) -> str:
        return f"{self.remote}:{self.path}"

    def parent(self) -> "OssPath":
        if "/" not in self.path:
            return OssPath(self.remote, "")
        return OssPath(self.remote, self.path.rsplit("/", 1)[0])

    @property
    def name(self) -> str:
        return self.path.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class RcloneResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


class OssClient:
    """Small OSS utility wrapper around rclone.

    Benchmarked on 2026-06-12 from this workspace with
    scripts/tools/oss_tools/bin/rclone, scripts/tools/oss_tools/rclone.conf,
    and a 15-byte object at
    qz_oss2:public/zhaoguojie/Mossland/oss_tools_benchmark/:

    - upload_file(copyto local -> OSS): about 0.156 s
    - exists(lsjson OSS object): about 0.121 s when present
    - download_file(copyto OSS -> local): about 0.043 s
    - delete_file: about 0.111 s
    - exists(lsjson OSS object): about 0.108 s after delete/missing

    These numbers are only startup/network latency samples for tiny files.
    Large objects are dominated by bandwidth and remote-side throttling.
    """

    def __init__(
        self,
        *,
        rclone_bin: Path | str | None = None,
        rclone_config: Path | str | None = None,
        extra_flags: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.rclone_bin = self._resolve_path(
            explicit=rclone_bin,
            env_name="OSS_TOOLS_RCLONE_BIN",
            default=DEFAULT_RCLONE,
        )
        self.rclone_config = self._resolve_path(
            explicit=rclone_config,
            env_name="OSS_TOOLS_RCLONE_CONFIG",
            default=DEFAULT_RCLONE_CONFIG,
        )
        self.extra_flags = tuple(extra_flags or ())
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _resolve_path(
        *,
        explicit: Path | str | None,
        env_name: str,
        default: Path,
    ) -> Path:
        if explicit is not None:
            return Path(explicit)
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
        return default

    def run_rclone(self, args: Sequence[str], *, check: bool = True) -> RcloneResult:
        command = (
            str(self.rclone_bin),
            "--config",
            str(self.rclone_config),
            *self.extra_flags,
            *args,
        )
        started = time.perf_counter()
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        result = RcloneResult(
            command=tuple(command),
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"rclone failed rc={proc.returncode} elapsed={elapsed:.3f}s\n"
                f"command: {' '.join(command)}\n"
                f"stderr: {proc.stderr.strip()}"
            )
        return result

    def upload_file(self, local_path: Path | str, oss_path: str | OssPath) -> RcloneResult:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = str(oss_path)
        return self.run_rclone(("copyto", str(src), dst))

    def download_file(self, oss_path: str | OssPath, local_path: Path | str) -> RcloneResult:
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return self.run_rclone(("copyto", str(oss_path), str(dst)))

    def exists(self, oss_path: str | OssPath) -> bool:
        path = OssPath.parse(str(oss_path))
        result = self.run_rclone(("lsjson", str(path)), check=False)
        lowered = result.stderr.lower()
        if "directory not found" in lowered or "object not found" in lowered:
            return False
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"rclone lsjson failed for {path}")
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return False
        if isinstance(payload, dict):
            return bool(payload)
        if isinstance(payload, list):
            return any(item.get("Name") == path.name or item.get("Path") == path.name for item in payload)
        return False

    def delete_file(self, oss_path: str | OssPath, *, missing_ok: bool = True) -> RcloneResult:
        result = self.run_rclone(("deletefile", str(oss_path)), check=False)
        if result.returncode == 0:
            return result
        lowered = result.stderr.lower()
        if missing_ok and ("not found" in lowered or "directory not found" in lowered):
            return result
        if "not found" in lowered or "directory not found" in lowered:
            raise FileNotFoundError(str(oss_path))
        raise RuntimeError(result.stderr.strip() or f"rclone deletefile failed for {oss_path}")
