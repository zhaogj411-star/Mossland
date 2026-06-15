#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RCLONE="${RCLONE:-${REPO_ROOT}/scripts/tools/oss_tools/bin/rclone}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${REPO_ROOT}/scripts/tools/oss_tools/rclone.conf}"

DEFAULT_LOCAL_SRC="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER"
DEFAULT_OSS_DST="qz_oss2:embodied-multimodality/public/Sonata/data/raw/NETEASE_SPIDER"

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

echo "Push local data to OSS"
echo "  local: ${LOCAL_SRC}"
echo "  oss: ${OSS_DST}"
echo "  transfers: ${TRANSFERS}"
echo "  checkers: ${CHECKERS}"

"${RCLONE}" copy "${LOCAL_SRC}" "${OSS_DST}" \
  --config "${RCLONE_CONFIG}" \
  -P \
  --transfers "${TRANSFERS}" \
  --checkers "${CHECKERS}" \
  --fast-list
