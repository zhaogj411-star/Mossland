#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "${REPO_ROOT}"

mkdir -p "${MACHINE_LOG_DIR}"
IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
declare -A PIDS
declare -A RESTARTS

start_worker() {
  local device=$1
  local safe_device=${device//:/}
  local log_file="${MACHINE_LOG_DIR}/worker_${safe_device}.log"
  echo "starting worker ${device}, log=${log_file}" >&2
  if [[ "${USE_PIPELINE_WORKER}" == "1" ]]; then
    "${PY}" -m scripts.data_processing.oss_prepare_separation_pipeline \
      --root "${QUEUE_ROOT}" \
      --job "${JOB}" \
      --device "${device}" \
      --lease-ttl "${LEASE_TTL}" \
      --heartbeat-interval "${HEARTBEAT_INTERVAL}" \
      --max-attempts "${MAX_ATTEMPTS}" \
      --download-workers "${DOWNLOAD_WORKERS}" \
      --prefetch "${PREFETCH}" \
      --upload-workers "${UPLOAD_STAGE_WORKERS}" \
      --max-pending-uploads "${MAX_PENDING_UPLOADS}" \
      >> "${log_file}" 2>&1 &
  else
    "${PY}" -m scripts.tools.local_queue.cli worker \
      --root "${QUEUE_ROOT}" \
      --job "${JOB}" \
      --handler "${HANDLER}" \
      --device "${device}" \
      --lease-ttl "${LEASE_TTL}" \
      --heartbeat-interval "${HEARTBEAT_INTERVAL}" \
      --max-attempts "${MAX_ATTEMPTS}" \
      >> "${log_file}" 2>&1 &
  fi
  PIDS["${device}"]=$!
}

cleanup() {
  for device in "${DEVICE_LIST[@]}"; do
    local pid=${PIDS["${device}"]:-}
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

for device in "${DEVICE_LIST[@]}"; do
  RESTARTS["${device}"]=0
  start_worker "${device}"
done

start_ts=$(date +%s)
echo "machine_tag=${MACHINE_TAG}"
echo "train_job_id=${TRAIN_JOB_ID:-}"
echo "pet_node_rank=${PET_NODE_RANK:-}"
echo "hostname=$(hostname)"
echo "log_dir=${MACHINE_LOG_DIR}"
echo "use_pipeline_worker=${USE_PIPELINE_WORKER}"
echo "prefetch=${PREFETCH}"
echo "download_workers=${DOWNLOAD_WORKERS}"
echo "upload_stage_workers=${UPLOAD_STAGE_WORKERS}"
echo "max_pending_uploads=${MAX_PENDING_UPLOADS}"

while true; do
  snapshot=$("${PY}" -m scripts.tools.local_queue.cli monitor --root "${QUEUE_ROOT}" --job "${JOB}")
  echo "$(date '+%F %T')"
  echo "${snapshot}"

  total=$(echo "${snapshot}" | awk '/^total:/ {print $2}')
  done_count=$(echo "${snapshot}" | awk '/^done:/ {print $2}')
  failed_count=$(echo "${snapshot}" | awk '/^failed:/ {print $2}')
  total=${total:-0}
  done_count=${done_count:-0}
  failed_count=${failed_count:-0}

  if (( total > 0 && done_count + failed_count >= total )); then
    break
  fi

  for device in "${DEVICE_LIST[@]}"; do
    pid=${PIDS["${device}"]}
    if ! kill -0 "${pid}" 2>/dev/null; then
      RESTARTS["${device}"]=$((RESTARTS["${device}"] + 1))
      echo "worker ${device} exited; restart=${RESTARTS["${device}"]}" >&2
      start_worker "${device}"
    fi
  done

  sleep "${MONITOR_INTERVAL}"
done

end_ts=$(date +%s)
echo "elapsed_seconds=$((end_ts - start_ts))"
"${PY}" -m scripts.tools.local_queue.cli finalize --root "${QUEUE_ROOT}" --job "${JOB}" --allow-failed || true
