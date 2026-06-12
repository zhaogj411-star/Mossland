from __future__ import annotations

import subprocess


def detect_devices() -> list[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        gpus = [line for line in out.splitlines() if line.startswith("GPU ")]
        if gpus:
            return [f"cuda:{i}" for i in range(len(gpus))]
    except Exception:
        pass
    return ["cpu"]
