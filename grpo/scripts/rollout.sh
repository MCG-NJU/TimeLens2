#!/usr/bin/env bash
# Off-policy rollout, config-driven. All data sources come from a single JSON config;
# add a new source by editing that file (no new bash args).
#
# Multi-node example (one srun task per node, 8 GPUs/node):
#   DATA_CONFIG=configs/data/timelens2.json \
#   MODEL_PATH=/path/to/sft_ckpt \
#   PRED_ROOT=output/rollout/qwen3vl_4b_64k_2fps_1024f \
#   srun -N16 -n16 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8 \
#        bash scripts/rollout.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_CONFIG="${DATA_CONFIG:?set DATA_CONFIG to a data config JSON}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the rollout checkpoint}"
PRED_ROOT="${PRED_ROOT:?set PRED_ROOT to the output directory}"
EMIT_TRAIN_CONFIG="${EMIT_TRAIN_CONFIG:-${PRED_ROOT}/train_data_config.json}"

# Generation / video budget (defaults align with GRPO training).
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MIN_TOKENS="${MIN_TOKENS:-1}"
TOTAL_TOKENS="${TOTAL_TOKENS:-65536}"
FPS="${FPS:-2}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-1024}"
SEED="${SEED:-42}"
MERGE_WAIT_SEC="${MERGE_WAIT_SEC:-86400}"
MERGE_POLL_SEC="${MERGE_POLL_SEC:-10}"

cleanup() { pkill -P $$ 2>/dev/null || true; exit 130; }
trap cleanup SIGINT SIGTERM

nnodes=${SLURM_JOB_NUM_NODES:-1}
if [[ "${nnodes}" -gt 1 ]] && [[ "${SLURM_NTASKS_PER_NODE:-1}" -eq 1 ]]; then
  node_rank=${SLURM_PROCID:-0}
else
  node_rank=0
fi

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1)))}"
local_gpus=${#GPULIST[@]}
global_chunks=$((nnodes * local_gpus))
echo "nnodes=${nnodes} node_rank=${node_rank} local_gpus=${local_gpus} global_chunks=${global_chunks}"

common_args=(
  --data_config "${DATA_CONFIG}"
  --pred_root "${PRED_ROOT}"
  --min_tokens "${MIN_TOKENS}" --total_tokens "${TOTAL_TOKENS}"
  --fps "${FPS}" --fps_max_frames "${FPS_MAX_FRAMES}"
  --num_rollouts "${NUM_ROLLOUTS}" --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --seed "${SEED}"
)

for IDX in $(seq 0 $((local_gpus - 1))); do
  global_idx=$((node_rank * local_gpus + IDX))
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "${PYTHON}" -m timelens.rollout \
    "${common_args[@]}" \
    --model_path "${MODEL_PATH}" \
    --chunk "${global_chunks}" --index "${global_idx}" &
done
wait

if [[ "${node_rank}" -eq 0 ]]; then
  deadline=$((SECONDS + MERGE_WAIT_SEC))
  while true; do
    if "${PYTHON}" -m timelens.rollout "${common_args[@]}" \
        --merge --rm_shards --emit_train_config "${EMIT_TRAIN_CONFIG}"; then
      echo "Merged rollouts; train-ready config: ${EMIT_TRAIN_CONFIG}"
      break
    fi
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      echo "Timeout waiting for all shards to complete." >&2
      exit 1
    fi
    sleep "${MERGE_POLL_SEC}"
  done
else
  echo "node_rank=${node_rank}: local workers done; merge runs on node_rank 0 only."
fi
