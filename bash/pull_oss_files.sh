#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RCLONE="${RCLONE:-${REPO_ROOT}/scripts/tools/oss_tools/bin/rclone}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${REPO_ROOT}/scripts/tools/oss_tools/rclone.conf}"

DEFAULT_OSS_SRC="qz_oss2:embodied-multimodality/public/zhaoguojie/Mossland/logs/mossland-codec/runs/2026-06-12_12-46-36/checkpoints"
DEFAULT_LOCAL_DST="/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland/logs/mossland-codec/runs/2026-06-12_12-46-36/checkpoints"

OSS_SRC="${1:-${OSS_SRC:-${DEFAULT_OSS_SRC}}}"
LOCAL_DST="${2:-${LOCAL_DST:-${DEFAULT_LOCAL_DST}}}"
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

remote_is_file() {
  local remote_path="$1"
  local stat_json
  stat_json="$("${RCLONE}" lsjson --stat "${remote_path}" --config "${RCLONE_CONFIG}" 2>/dev/null || true)"
  [[ "${stat_json}" == *'"IsDir": false'* ]] || [[ "${stat_json}" == *'"IsDir":false'* ]]
}

local_file_destination() {
  local remote_src="$1"
  local local_dst="$2"
  local source_name dst_name
  source_name="${remote_src##*/}"
  dst_name="$(basename "${local_dst}")"

  if [[ "${local_dst}" == */ ]] || [[ -d "${local_dst}" ]]; then
    printf '%s/%s\n' "${local_dst%/}" "${source_name}"
    return
  fi

  if [[ -f "${local_dst}" ]]; then
    printf '%s\n' "${local_dst}"
    return
  fi

  if [[ "${dst_name}" == "${source_name}" ]] || [[ "${dst_name}" == *.* ]]; then
    printf '%s\n' "${local_dst}"
    return
  fi

  printf '%s/%s\n' "${local_dst%/}" "${source_name}"
}

echo "Pull OSS data to local"
echo "  oss: ${OSS_SRC}"
echo "  local: ${LOCAL_DST}"
echo "  transfers: ${TRANSFERS}"
echo "  checkers: ${CHECKERS}"

if remote_is_file "${OSS_SRC}"; then
  LOCAL_FILE_DST="$(local_file_destination "${OSS_SRC}" "${LOCAL_DST}")"
  mkdir -p "$(dirname "${LOCAL_FILE_DST}")"
  echo "  mode: file -> file"
  echo "  local file: ${LOCAL_FILE_DST}"
  "${RCLONE}" copyto "${OSS_SRC}" "${LOCAL_FILE_DST}" \
    --config "${RCLONE_CONFIG}" \
    -P \
    --transfers "${TRANSFERS}" \
    --checkers "${CHECKERS}"
else
  mkdir -p "${LOCAL_DST}"
  echo "  mode: directory -> directory"
  "${RCLONE}" copy "${OSS_SRC}" "${LOCAL_DST}" \
    --config "${RCLONE_CONFIG}" \
    -P \
    --transfers "${TRANSFERS}" \
    --checkers "${CHECKERS}" \
    --fast-list
fi
