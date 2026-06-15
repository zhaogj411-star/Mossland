#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "${REPO_ROOT}"

"${PY}" - <<'PY'
import json
from collections import Counter
from pathlib import Path

queue_root = Path(__import__("os").environ["QUEUE_ROOT"])
job = __import__("os").environ["JOB"]
output_root = queue_root / "outputs" / job

status = Counter()
extensions = Counter()
workers = Counter()
for meta_path in output_root.rglob("*.meta.json"):
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    status[str(metadata.get("status", "unknown"))] += 1
    input_audio = str(metadata.get("input_audio", ""))
    if "." in input_audio:
        extensions["." + input_audio.rsplit(".", 1)[-1].lower()] += 1
    workers[str(data.get("worker_id", "unknown"))] += 1

print("status")
for key, value in sorted(status.items()):
    print(f"  {key}: {value}")
print("extensions")
for key, value in sorted(extensions.items()):
    print(f"  {key}: {value}")
print("workers")
for key, value in sorted(workers.items()):
    print(f"  {key}: {value}")

log_dir = Path(__import__("os").environ["LOG_DIR"])
if log_dir.exists():
    print("machine_log_dirs")
    for path in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        print(f"  {path}")

failed_dir = queue_root / "state" / job / "failed"
failed = list(failed_dir.rglob("*.json")) if failed_dir.exists() else []
print(f"failed_records: {len(failed)}")
for path in failed[:10]:
    print(path)
PY
