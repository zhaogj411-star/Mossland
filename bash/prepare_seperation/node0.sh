#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland
PY=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python
MANIFEST=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW/_logs/prepare_separation/manifest.txt
SOURCE_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER
OUTPUT_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW
LOG_DIR=${OUTPUT_ROOT}/_logs/prepare_separation_multinode

NODE_RANK=0
GPUS_PER_NODE=8
TOTAL_SHARDS=32

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for LOCAL_GPU in 0 1 2 3 4 5 6 7; do
  SHARD_ID=$((NODE_RANK * GPUS_PER_NODE + LOCAL_GPU))
  CUDA_VISIBLE_DEVICES=${LOCAL_GPU} PYTHONUNBUFFERED=1 \
  "${PY}" -m scripts.data.prepare_separation \
    --files-list "${MANIFEST}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --num-shards "${TOTAL_SHARDS}" \
    --shard-id "${SHARD_ID}" \
    --worker-id "${SHARD_ID}" \
    --device cuda:0 \
    --num-overlap 2 \
    --chunk-batch-size 1 \
    --max-duration-seconds 600 \
    --save-workers 8 \
    --max-pending-writes 32 \
    --progress-file "${LOG_DIR}/progress_node${NODE_RANK}_gpu${LOCAL_GPU}.jsonl" \
    > "${LOG_DIR}/node${NODE_RANK}_gpu${LOCAL_GPU}.log" 2>&1 &
done

wait
