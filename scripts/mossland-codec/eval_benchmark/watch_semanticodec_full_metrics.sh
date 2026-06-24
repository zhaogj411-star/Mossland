#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

LOG_DIR="scripts/mossland-codec/eval_benchmark/logs"
mkdir -p "$LOG_DIR"

expected_rows="${SEMANTICODEC_EXPECTED_ROWS:-5355}"
poll_seconds="${SEMANTICODEC_WATCH_POLL_SECONDS:-300}"

wait_for_manifest() {
  local manifest="$1"
  while true; do
    if [[ -f "$manifest" ]]; then
      local rows
      rows="$(wc -l < "$manifest")"
      if [[ "$rows" -ge "$expected_rows" ]]; then
        return 0
      fi
      printf '[%(%F %T)T] waiting for %s rows=%s/%s\n' -1 "$manifest" "$rows" "$expected_rows"
    else
      printf '[%(%F %T)T] waiting for %s missing\n' -1 "$manifest"
    fi
    sleep "$poll_seconds"
  done
}

run_metric() {
  local token_rate="$1"
  local backend="$2"
  local device="$3"
  local base="scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_semanticodec_tr${token_rate}"
  local manifest="${base}/semanticodec_manifest.jsonl"
  local output_dir="${base}/eval_${backend}"
  local summary="${output_dir}/summary.json"

  if [[ -s "$summary" ]]; then
    printf '[%(%F %T)T] skip token_rate=%s backend=%s existing %s\n' -1 "$token_rate" "$backend" "$summary"
    return 0
  fi

  printf '[%(%F %T)T] start token_rate=%s backend=%s device=%s\n' -1 "$token_rate" "$backend" "$device"
  CUDA_VISIBLE_DEVICES="$device" python -m scripts.mossland-codec.eval_benchmark.run \
    --manifest "$manifest" \
    --output-dir "$output_dir" \
    --fad-backend "$backend" \
    --metrics-device cuda \
    --progress-every 100
  printf '[%(%F %T)T] done token_rate=%s backend=%s\n' -1 "$token_rate" "$backend"
}

run_for_rate() {
  local token_rate="$1"
  local device="$2"
  local manifest="scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_semanticodec_tr${token_rate}/semanticodec_manifest.jsonl"
  wait_for_manifest "$manifest"
  run_metric "$token_rate" clap "$device"
  run_metric "$token_rate" vggish "$device"
}

run_for_rate 25 "${SEMANTICODEC_DEVICE_TR25:-3}" &
pid25="$!"
run_for_rate 50 "${SEMANTICODEC_DEVICE_TR50:-4}" &
pid50="$!"
run_for_rate 100 "${SEMANTICODEC_DEVICE_TR100:-5}" &
pid100="$!"

wait "$pid25"
wait "$pid50"
wait "$pid100"

python -m scripts.mossland-codec.eval_benchmark.generate_tables
printf '[%(%F %T)T] semanticodec full metrics watcher complete\n' -1
