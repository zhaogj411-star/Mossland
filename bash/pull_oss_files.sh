#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RCLONE="${RCLONE:-${REPO_ROOT}/scripts/tools/oss_tools/bin/rclone}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${REPO_ROOT}/scripts/tools/oss_tools/rclone.conf}"

DEFAULT_OSS_SRC="qz_oss2:embodied-multimodality/public/Sonata/data/source_seperation/NETEASE_SPIDER"
DEFAULT_LOCAL_DST="/inspire/sj-ssd3/project/embodied-multimodality/public/Sonata/data/source_seperation/NETEASE_SPIDER"

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

mkdir -p "${LOCAL_DST}"

echo "Pull OSS data to local"
echo "  oss: ${OSS_SRC}"
echo "  local: ${LOCAL_DST}"
echo "  transfers: ${TRANSFERS}"
echo "  checkers: ${CHECKERS}"

"${RCLONE}" copy "${OSS_SRC}" "${LOCAL_DST}" \
  --config "${RCLONE_CONFIG}" \
  -P \
  --transfers "${TRANSFERS}" \
  --checkers "${CHECKERS}" \
  # --fast-list
