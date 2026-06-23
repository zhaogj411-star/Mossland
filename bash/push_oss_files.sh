#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RCLONE="${RCLONE:-${REPO_ROOT}/scripts/tools/oss_tools/bin/rclone}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${REPO_ROOT}/scripts/tools/oss_tools/rclone.conf}"

DEFAULT_LOCAL_SRC="/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/logs/mossland-codec/runs/2026-06-20_18-49-34/checkpoints/last.ckpt"
DEFAULT_OSS_DST="qz_oss2:embodied-multimodality/public/zhaoguojie/Mossland/logs/mossland-codec/runs/2026-06-20_18-49-34/checkpoints/last.ckpt"

LOCAL_SRC="${1:-${LOCAL_SRC:-${DEFAULT_LOCAL_SRC}}}"
OSS_DST="${2:-${OSS_DST:-${DEFAULT_OSS_DST}}}"
TRANSFERS="${TRANSFERS:-16}"
CHECKERS="${CHECKERS:-32}"

if [[ ! -x "${RCLONE}" ]]; then
  echo "rclone not executable: ${RCLONE}" >&2
  exit 1
fi

if [[ ! -f "${RCLONE_CONFIG}" ]]; then
  echo "rclone config not found: ${RCLONE_CONFIG}" >&2
  exit 1
fi

if [[ ! -e "${LOCAL_SRC}" ]]; then
  echo "local source not found: ${LOCAL_SRC}" >&2
  exit 1
fi

remote_is_dir() {
  local remote_path="$1"
  local stat_json
  stat_json="$("${RCLONE}" lsjson --stat "${remote_path}" --config "${RCLONE_CONFIG}" 2>/dev/null || true)"
  [[ "${stat_json}" == *'"IsDir": true'* ]] || [[ "${stat_json}" == *'"IsDir":true'* ]]
}

remote_is_file() {
  local remote_path="$1"
  local stat_json
  stat_json="$("${RCLONE}" lsjson --stat "${remote_path}" --config "${RCLONE_CONFIG}" 2>/dev/null || true)"
  [[ "${stat_json}" == *'"IsDir": false'* ]] || [[ "${stat_json}" == *'"IsDir":false'* ]]
}

remote_file_destination() {
  local source_file="$1"
  local remote_dst="$2"
  local source_name dst_name
  source_name="$(basename "${source_file}")"
  dst_name="${remote_dst##*/}"

  if [[ "${remote_dst}" == */ ]] || remote_is_dir "${remote_dst}"; then
    printf '%s/%s\n' "${remote_dst%/}" "${source_name}"
    return
  fi

  if remote_is_file "${remote_dst}"; then
    printf '%s\n' "${remote_dst}"
    return
  fi

  if [[ "${dst_name}" == "${source_name}" ]] || [[ "${dst_name}" == *.* ]]; then
    printf '%s\n' "${remote_dst}"
    return
  fi

  printf '%s/%s\n' "${remote_dst%/}" "${source_name}"
}

echo "Push local data to OSS"
echo "  local: ${LOCAL_SRC}"
echo "  oss: ${OSS_DST}"
echo "  transfers: ${TRANSFERS}"
echo "  checkers: ${CHECKERS}"

if [[ -f "${LOCAL_SRC}" ]]; then
  OSS_FILE_DST="$(remote_file_destination "${LOCAL_SRC}" "${OSS_DST}")"
  echo "  mode: file -> file"
  echo "  oss file: ${OSS_FILE_DST}"
  "${RCLONE}" copyto "${LOCAL_SRC}" "${OSS_FILE_DST}" \
    --config "${RCLONE_CONFIG}" \
    -P \
    --transfers "${TRANSFERS}" \
    --checkers "${CHECKERS}"
else
  echo "  mode: directory -> directory"
  "${RCLONE}" copy "${LOCAL_SRC}" "${OSS_DST}" \
    --config "${RCLONE_CONFIG}" \
    -P \
    --transfers "${TRANSFERS}" \
    --checkers "${CHECKERS}" \
    --fast-list
fi
