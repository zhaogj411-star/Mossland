#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "${REPO_ROOT}"

"${PY}" -m scripts.tools.local_queue.cli monitor --root "${QUEUE_ROOT}" --job "${JOB}"

