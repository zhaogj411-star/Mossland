#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/envs/main_env/bin/python}"
EXPERIMENT="${EXPERIMENT:-mossland-codec}"
PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-8}"
PET_NNODES="${PET_NNODES:-1}"
PET_NODE_RANK="${PET_NODE_RANK:-0}"

export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
export NODE_RANK="${PET_NODE_RANK}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

command=(
  "${PYTHON_BIN}"
  -m scripts.train
  "experiment=${EXPERIMENT}"
  "trainer.devices=${PET_NPROC_PER_NODE}"
  "trainer.num_nodes=${PET_NNODES}"
  "$@"
)

echo "Starting Mossland multinode training"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  PET_NNODES=${PET_NNODES}"
echo "  PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE}"
echo "  EXPERIMENT=${EXPERIMENT}"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  WORLD_SIZE=${WORLD_SIZE:-unset}"
echo "  RANK=${RANK:-unset}"
echo "  TRAIN_JOB_ID=${TRAIN_JOB_ID:-unset}"
echo "  RUNNING_ROUND=${RUNNING_ROUND:-unset}"
printf "  command:"
printf " %q" "${command[@]}"
printf "\n"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_DIR}"
exec "${command[@]}"
