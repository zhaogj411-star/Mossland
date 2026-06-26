#!/bin/bash
# Launch Echo TTS training from the Mossland repo.
#
# Usage:
#   bash bash/train_echo_tts.sh scripts/configs/echo_tts/tt_800M_synthetic.yaml
#   sbatch --nodes=1 bash/train_echo_tts.sh scripts/configs/echo_tts/tt_3B.yaml

#SBATCH --job-name=echo-tts
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=logs/echo_tts_%j.out
#SBATCH --error=logs/echo_tts_%j.err

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-scripts/configs/echo_tts/tt_800M_synthetic.yaml}"

if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$REPO_DIR/$CONFIG_PATH"
fi

mkdir -p "$REPO_DIR/logs"

FRAMEWORK=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['framework'])")
if [ "$FRAMEWORK" = "megatron" ]; then
    MODULE="scripts.echo_tts.train.megatron"
else
    MODULE="scripts.echo_tts.train.torchtitan"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${HF_HOME:-$REPO_DIR/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Jupiter / GH200 defaults. Override these in the environment if needed.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-120}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$REPO_DIR"

echo "=== Echo TTS ==="
echo "Config: $CONFIG_PATH"
echo "Framework: $FRAMEWORK"
echo "Module: $MODULE"
echo "Date: $(date)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$REPO_DIR/.cache/torchinductor_${SLURM_PROCID:-0}_${LOCAL_RANK:-0}}

    srun --cpu-bind=none bash -c '
    if [ -n "${CONDA_SH:-}" ]; then
        source "$CONDA_SH"
        conda activate "${CONDA_ENV:-tts}"
    fi
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m '"$MODULE"' --config '"$CONFIG_PATH"'
    '
else
    NPROC_PER_NODE=${NPROC_PER_NODE:-1}
    torchrun --nproc_per_node="$NPROC_PER_NODE" \
        -m "$MODULE" --config "$CONFIG_PATH"
fi

echo "=== Done: $(date) ==="
