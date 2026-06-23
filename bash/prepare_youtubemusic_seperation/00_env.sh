#!/usr/bin/env bash

# Source this file from the other step scripts. Override variables before running
# a step when needed, for example: DEVICES=cuda:0,cuda:1 bash 03_start_workers.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland}
export MAIN_ENV=${MAIN_ENV:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/envs/main_env}
export PY=${PY:-${MAIN_ENV}/bin/python}

export RCLONE=${RCLONE:-${REPO_ROOT}/scripts/tools/oss_tools/bin/rclone}
export RCLONE_CONFIG=${RCLONE_CONFIG:-${REPO_ROOT}/scripts/tools/oss_tools/rclone.conf}

export DATASET_NAME=${DATASET_NAME:-NETEASE_SPIDER}
export SOURCE_PREFIX=${SOURCE_PREFIX:-qz_oss2:embodied-multimodality/public/Sonata/data/raw/${DATASET_NAME}}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-qz_oss2:embodied-multimodality/public/Sonata/data/source_seperation/${DATASET_NAME}}
export DATES=${DATES:-}

export RUN_DIR=${RUN_DIR:-${REPO_ROOT}/data_processing_tmp/prepare_${DATASET_NAME,,}_seperation}
export INPUTS_FILE=${INPUTS_FILE:-${RUN_DIR}/inputs.txt}
export CONFIG_FILE=${CONFIG_FILE:-${RUN_DIR}/config.json}
export QUEUE_ROOT=${QUEUE_ROOT:-${RUN_DIR}/local_queue}
export JOB=${JOB:-${DATASET_NAME,,}_source_seperation}
export HANDLER=${HANDLER:-scripts.data_processing.oss_prepare_separation:OssPrepareSeparationHandler}
export LOG_DIR=${LOG_DIR:-${RUN_DIR}/logs}

if [[ -n "${TRAIN_JOB_ID:-}" ]]; then
  DEFAULT_MACHINE_TAG="trainjob_${TRAIN_JOB_ID}"
elif [[ -n "${PET_NODE_RANK:-}" ]]; then
  DEFAULT_MACHINE_TAG="node_${PET_NODE_RANK}"
else
  DEFAULT_MACHINE_TAG="$(hostname)"
fi
export MACHINE_TAG=${MACHINE_TAG:-${DEFAULT_MACHINE_TAG}}
export MACHINE_LOG_DIR=${MACHINE_LOG_DIR:-${LOG_DIR}/${MACHINE_TAG}}

export MODEL_DIR=${MODEL_DIR:-${REPO_ROOT}/checkpoints/mel-band-roformer-vocal-model}
export CONFIG_HASH=${CONFIG_HASH:-${DATASET_NAME,,}_roformer_cached_44100_ppu_complex_fix_v1}
export MAX_DURATION_SECONDS=${MAX_DURATION_SECONDS:-600}
export OSS_TIMEOUT_SECONDS=${OSS_TIMEOUT_SECONDS:-3600}
export NUM_OVERLAP=${NUM_OVERLAP:-2}
export CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-1}
export DEVICES=${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}
export LEASE_TTL=${LEASE_TTL:-1800}
export HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-30}
export MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
export MONITOR_INTERVAL=${MONITOR_INTERVAL:-30}
export USE_PIPELINE_WORKER=${USE_PIPELINE_WORKER:-1}
export PREFETCH=${PREFETCH:-4}
export DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-2}
export UPLOAD_STAGE_WORKERS=${UPLOAD_STAGE_WORKERS:-2}
export MAX_PENDING_UPLOADS=${MAX_PENDING_UPLOADS:-2}

export PATH="${MAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
