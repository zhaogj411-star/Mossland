#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland}"
DEFAULT_CKPT="${REPO_ROOT}/logs/mossland-codec/runs/2026-06-12_12-46-36/checkpoints/last.ckpt/last.ckpt"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash bash/eval_mossland_codec_checkpoint.sh [LIGHTNING_CKPT_OR_DIR]

Runs MusicCaps-HF full-clip reconstruction evaluation for a Mossland codec
Lightning checkpoint:
  1. unwrap checkpoint to ckpt/<run-name>
  2. run EncoderDecoder parallel reconstruction from scripts/mossland-codec/inference.py
  3. run CLAP, VGGish, and ViSQOL metrics

Default checkpoint:
  logs/mossland-codec/runs/2026-06-12_12-46-36/checkpoints/last.ckpt/last.ckpt

Useful environment overrides:
  RUN_NAME=...              Output run name under eval_benchmark/runs.
  EXPORT_EMA=true|false     Export EMA model during unwrap. Default: true.
  GPUS=0,1,2,3,4,5,6,7      GPU ids for inference and GPU metrics.
  INFER_SHARDS=16           Full-clip inference shard count.
  METRIC_SHARDS=8           CLAP/VGGish shard count.
  VISQOL_SHARDS=128         CPU ViSQOL shard count.
  MAX_BATCH_SIZE_ENCODE=... Override EncoderDecoder encode batch size.
  MAX_BATCH_SIZE_DECODE=... Override EncoderDecoder parallel decode batch size.
  OVERWRITE_PREDICTIONS=0|1 Regenerate existing prediction wavs. Default: 0.
  PYTHON_BIN=python         Python executable for unwrap/infer/metric commands.
  FORCE_UNWRAP=0|1          Re-export checkpoint even if checkpoint dir exists.
  RUN_UNWRAP=0|1            Enable unwrap stage. Default: 1.
  RUN_INFER=0|1             Enable full-clip inference stage. Default: 1.
  RUN_CLAP=0|1              Enable CLAP metric stage. Default: 1.
  RUN_VGGISH=0|1            Enable VGGish metric stage. Default: 1.
  RUN_VISQOL=0|1            Enable ViSQOL metric stage. Default: 1.

Outputs:
  scripts/mossland-codec/eval_benchmark/runs/<RUN_NAME>/
  scripts/mossland-codec/eval_benchmark/logs/<RUN_NAME>.log
EOF
  exit 0
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CKPT_INPUT="${1:-${DEFAULT_CKPT}}"
if [[ -d "${CKPT_INPUT}" ]]; then
  if [[ -f "${CKPT_INPUT}/last.ckpt" ]]; then
    CKPT_PATH="${CKPT_INPUT}/last.ckpt"
  elif [[ -f "${CKPT_INPUT}/checkpoint.ckpt" && -f "${CKPT_INPUT}/config.yaml" ]]; then
    CKPT_DIR="${CKPT_INPUT}"
    CKPT_PATH=""
  else
    echo "ERROR: directory does not contain last.ckpt or checkpoint.ckpt+config.yaml: ${CKPT_INPUT}" >&2
    exit 2
  fi
else
  CKPT_PATH="${CKPT_INPUT}"
fi

if [[ -n "${CKPT_PATH:-}" && ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CKPT_PATH}" >&2
  exit 2
fi

EXPORT_EMA="${EXPORT_EMA:-true}"
if [[ "${EXPORT_EMA}" == "true" ]]; then
  VARIANT="ema"
else
  VARIANT="raw"
fi

GLOBAL_STEP="unknown"
if [[ -n "${CKPT_PATH:-}" ]]; then
  GLOBAL_STEP="$(
    "${PYTHON_BIN}" - "${CKPT_PATH}" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
step = ckpt.get("global_step", "unknown") if isinstance(ckpt, dict) else "unknown"
print(step if step is not None else "unknown")
PY
  )"
fi

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${GLOBAL_STEP}" != "unknown" ]]; then
    RUN_NAME="mossland_codec_${GLOBAL_STEP}_${VARIANT}_parallel_decode"
  elif [[ -n "${CKPT_DIR:-}" ]]; then
    RUN_NAME="$(basename "${CKPT_DIR}")_${VARIANT}_parallel_decode"
  else
    RUN_NAME="$(basename "$(dirname "${CKPT_PATH}")")_${VARIANT}_parallel_decode"
  fi
fi

EVAL_ROOT="${REPO_ROOT}/scripts/mossland-codec/eval_benchmark"
MANIFEST="${MANIFEST:-${EVAL_ROOT}/data/musiccaps/hf_musiccaps_reconstruct_manifest.jsonl}"
RUN_DIR="${RUN_DIR:-${EVAL_ROOT}/runs/${RUN_NAME}}"
PREDICTION_DIR="${RUN_DIR}/predictions"
PREDICTION_MANIFEST="${RUN_DIR}/prediction_manifest.jsonl"
LOG_DIR="${EVAL_ROOT}/logs"
mkdir -p "${RUN_DIR}" "${PREDICTION_DIR}" "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/${RUN_NAME}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[$(date '+%F %T')] repo=${REPO_ROOT}"
echo "[$(date '+%F %T')] checkpoint_input=${CKPT_INPUT}"
echo "[$(date '+%F %T')] run_name=${RUN_NAME}"
echo "[$(date '+%F %T')] run_dir=${RUN_DIR}"
echo "[$(date '+%F %T')] manifest=${MANIFEST}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 2
fi

if [[ -z "${CKPT_DIR:-}" ]]; then
  CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpt/mossland-codec-${GLOBAL_STEP}-${VARIANT}}"
fi

RUN_UNWRAP="${RUN_UNWRAP:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_CLAP="${RUN_CLAP:-1}"
RUN_VGGISH="${RUN_VGGISH:-1}"
RUN_VISQOL="${RUN_VISQOL:-1}"
FORCE_UNWRAP="${FORCE_UNWRAP:-0}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-0}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
INFER_SHARDS="${INFER_SHARDS:-16}"
METRIC_SHARDS="${METRIC_SHARDS:-8}"
VISQOL_SHARDS="${VISQOL_SHARDS:-128}"
MAX_BATCH_SIZE_ENCODE="${MAX_BATCH_SIZE_ENCODE:-}"
MAX_BATCH_SIZE_DECODE="${MAX_BATCH_SIZE_DECODE:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
METRICS_DEVICE="${METRICS_DEVICE:-cuda}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"

run_unwrap() {
  if [[ "${RUN_UNWRAP}" != "1" ]]; then
    echo "[$(date '+%F %T')] skip unwrap: RUN_UNWRAP=${RUN_UNWRAP}"
    return
  fi
  if [[ -f "${CKPT_DIR}/checkpoint.ckpt" && -f "${CKPT_DIR}/config.yaml" && "${FORCE_UNWRAP}" != "1" ]]; then
    echo "[$(date '+%F %T')] reuse unwrapped checkpoint: ${CKPT_DIR}"
    return
  fi
  if [[ -z "${CKPT_PATH:-}" ]]; then
    echo "[$(date '+%F %T')] skip unwrap: input is already a checkpoint dir: ${CKPT_DIR}"
    return
  fi
  echo "[$(date '+%F %T')] unwrap checkpoint -> ${CKPT_DIR} (export_ema=${EXPORT_EMA})"
  "${PYTHON_BIN}" -m scripts.unwrap experiment=mossland-codec \
    "experiment_ckpt_path=${CKPT_PATH}" \
    "output_path=${CKPT_DIR}" \
    "deepspeed=false" \
    "export_ema=${EXPORT_EMA}"
}

merge_prediction_manifests() {
  local shard_manifest_dir="${RUN_DIR}/prediction_manifest_shards"
  "${PYTHON_BIN}" - "${shard_manifest_dir}" "${PREDICTION_MANIFEST}" "${INFER_SHARDS}" <<'PY'
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
num_shards = int(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as out:
    for shard_id in range(num_shards):
        path = shard_dir / f"shard{shard_id:02d}of{num_shards}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing shard manifest: {path}")
        with path.open("r", encoding="utf-8") as inp:
            for line in inp:
                if line.strip():
                    out.write(line)
print(output)
PY
  echo "[$(date '+%F %T')] merged prediction manifest: ${PREDICTION_MANIFEST}"
  wc -l "${PREDICTION_MANIFEST}"
}

run_infer() {
  if [[ "${RUN_INFER}" != "1" ]]; then
    echo "[$(date '+%F %T')] skip inference: RUN_INFER=${RUN_INFER}"
    return
  fi
  echo "[$(date '+%F %T')] start EncoderDecoder parallel reconstruction shards=${INFER_SHARDS} gpus=${GPUS}"
  local shard_manifest_dir="${RUN_DIR}/prediction_manifest_shards"
  local shard_log_dir="${RUN_DIR}/prediction_logs"
  mkdir -p "${shard_manifest_dir}" "${shard_log_dir}"

  IFS=',' read -r -a gpu_list <<< "${GPUS}"
  local pids=()
  for ((shard_id = 0; shard_id < INFER_SHARDS; shard_id++)); do
    local gpu="${gpu_list[$((shard_id % ${#gpu_list[@]}))]}"
    local shard_name
    shard_name="$(printf 'shard%02dof%d' "${shard_id}" "${INFER_SHARDS}")"
    local cmd=(
      "${PYTHON_BIN}" -m scripts.mossland-codec.eval_benchmark.codec_reconstruct_infer
      --manifest "${MANIFEST}"
      --checkpoint-dir "${CKPT_DIR}"
      --output-dir "${PREDICTION_DIR}"
      --output-manifest "${shard_manifest_dir}/${shard_name}.jsonl"
      --device "${INFER_DEVICE}"
      --num-shards "${INFER_SHARDS}"
      --shard-id "${shard_id}"
      --progress-every "${PROGRESS_EVERY}"
    )
    if [[ -n "${MAX_BATCH_SIZE_ENCODE}" ]]; then
      cmd+=(--max-batch-size-encode "${MAX_BATCH_SIZE_ENCODE}")
    fi
    if [[ -n "${MAX_BATCH_SIZE_DECODE}" ]]; then
      cmd+=(--max-batch-size-decode "${MAX_BATCH_SIZE_DECODE}")
    fi
    if [[ "${OVERWRITE_PREDICTIONS}" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
      export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
      export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
      export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
      echo "[$(date '+%F %T')] start ${shard_name} gpu=${gpu}"
      "${cmd[@]}"
    ) > "${shard_log_dir}/${shard_name}.log" 2>&1 &
    pids+=("$!")
    echo "[$(date '+%F %T')] launched ${shard_name} gpu=${gpu} pid=${pids[$((${#pids[@]} - 1))]} log=${shard_log_dir}/${shard_name}.log"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "ERROR: at least one inference shard failed; see ${shard_log_dir}" >&2
    exit 1
  fi
  merge_prediction_manifests
}

run_metric() {
  local backend="$1"
  local output_dir="$2"
  local shard_dir="$3"
  local enabled="$4"
  if [[ "${enabled}" != "1" ]]; then
    echo "[$(date '+%F %T')] skip ${backend}: disabled"
    return
  fi
  echo "[$(date '+%F %T')] start ${backend} metric -> ${output_dir}"
  "${PYTHON_BIN}" -m scripts.mossland-codec.eval_benchmark.run_sharded_metric_eval \
    --manifest "${PREDICTION_MANIFEST}" \
    --output-dir "${output_dir}" \
    --shard-output-dir "${shard_dir}" \
    --fad-backend "${backend}" \
    --gpus "${GPUS}" \
    --num-shards "${METRIC_SHARDS}" \
    --metrics-device "${METRICS_DEVICE}" \
    --progress-every "${PROGRESS_EVERY}"
}

run_visqol() {
  if [[ "${RUN_VISQOL}" != "1" ]]; then
    echo "[$(date '+%F %T')] skip visqol: RUN_VISQOL=${RUN_VISQOL}"
    return
  fi
  echo "[$(date '+%F %T')] start ViSQOL shards=${VISQOL_SHARDS} -> ${RUN_DIR}/visqol"
  "${PYTHON_BIN}" -m scripts.mossland-codec.eval_benchmark.run_sharded_visqol_eval \
    --manifest "${PREDICTION_MANIFEST}" \
    --output-dir "${RUN_DIR}/visqol" \
    --shard-output-dir "${RUN_DIR}/visqol_shards${VISQOL_SHARDS}" \
    --num-shards "${VISQOL_SHARDS}"
}

run_unwrap
if [[ ! -f "${CKPT_DIR}/checkpoint.ckpt" || ! -f "${CKPT_DIR}/config.yaml" ]]; then
  echo "ERROR: unwrapped checkpoint dir is incomplete: ${CKPT_DIR}" >&2
  exit 2
fi
run_infer
if [[ ! -f "${PREDICTION_MANIFEST}" ]]; then
  echo "ERROR: prediction manifest not found after inference: ${PREDICTION_MANIFEST}" >&2
  exit 2
fi
run_metric clap "${RUN_DIR}/eval_clap" "${RUN_DIR}/eval_clap_shards${METRIC_SHARDS}" "${RUN_CLAP}"
run_metric vggish "${RUN_DIR}/eval_vggish" "${RUN_DIR}/eval_vggish_shards${METRIC_SHARDS}" "${RUN_VGGISH}"
run_visqol

echo "[$(date '+%F %T')] complete"
echo "run_dir=${RUN_DIR}"
echo "prediction_manifest=${PREDICTION_MANIFEST}"
echo "clap_summary=${RUN_DIR}/eval_clap/summary.json"
echo "vggish_summary=${RUN_DIR}/eval_vggish/summary.json"
echo "visqol_summary=${RUN_DIR}/visqol/summary.json"
