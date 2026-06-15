#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "${REPO_ROOT}"

if [[ ! -s "${INPUTS_FILE}" ]]; then
  echo "missing inputs file: ${INPUTS_FILE}" >&2
  echo "run 01_make_inputs.sh first" >&2
  exit 1
fi

if [[ "${RESET_QUEUE:-0}" == "1" ]]; then
  if [[ "${FORCE_RESET_QUEUE:-0}" != "1" ]] && find "${QUEUE_ROOT}/heartbeats/${JOB}" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to reset queue because worker heartbeats exist: ${QUEUE_ROOT}/heartbeats/${JOB}" >&2
    echo "stop all workers first, or set FORCE_RESET_QUEUE=1 if you are certain the queue is idle" >&2
    exit 1
  fi
  rm -rf "${QUEUE_ROOT}"
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cat > "${CONFIG_FILE}" <<JSON
{
  "inputs_file": "${INPUTS_FILE}",
  "source_prefix": "${SOURCE_PREFIX}",
  "output_prefix": "${OUTPUT_PREFIX}",
  "config_hash": "${CONFIG_HASH}",
  "model_dir": "${MODEL_DIR}",
  "num_overlap": ${NUM_OVERLAP},
  "chunk_batch_size": ${CHUNK_BATCH_SIZE},
  "save_workers": 3,
  "upload_workers": 4,
  "timing_log_threshold_seconds": 60,
  "max_duration_seconds": ${MAX_DURATION_SECONDS},
  "oss_timeout_seconds": ${OSS_TIMEOUT_SECONDS}
}
JSON

"${PY}" -m scripts.tools.local_queue.cli init \
  --root "${QUEUE_ROOT}" \
  --job "${JOB}" \
  --handler "${HANDLER}" \
  --config "${CONFIG_FILE}"

echo "config_file=${CONFIG_FILE}"
echo "queue_root=${QUEUE_ROOT}"
echo "job=${JOB}"
