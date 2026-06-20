#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/envs/main_env/bin/python}"
EXPERIMENT="${EXPERIMENT:-music2latent}"
TRAINER_DEVICES="${TRAINER_DEVICES:-8}"

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
  "trainer.devices=${TRAINER_DEVICES}"
  "$@"
)

echo "Starting Mossland training"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  EXPERIMENT=${EXPERIMENT}"
echo "  TRAINER_DEVICES=${TRAINER_DEVICES}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}"
printf "  command:"
printf " %q" "${command[@]}"
printf "\n"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_DIR}"
exec "${command[@]}"
