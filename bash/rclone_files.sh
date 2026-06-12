#!/usr/bin/env bash
set -euo pipefail

RCLONE="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/low-resource-dialects-quality-sources/tools/rclone-v1.71.0-linux-amd64/rclone"
RCLONE_CONFIG="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/low-resource-dialects-quality-sources/tools/rclone-v1.71.0-linux-amd64/rclone.conf"
SRC="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER"
DST="qz_oss2:embodied-multimodality/public/Sonata/data/raw/NETEASE_SPIDER"

"${RCLONE}" copy "${SRC}" "${DST}" \
  --config "${RCLONE_CONFIG}" \
  -P \
  --transfers 16 \
  --checkers 32 \
  --fast-list
