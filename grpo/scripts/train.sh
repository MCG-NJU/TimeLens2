#!/usr/bin/env bash
# Config-driven GRPO training (Qwen3-VL). One JSON data config per run; all data
# sources (paths, sampling ratios, per-source reward / prompt) live there.
#
# Single node:
#   DATA_CONFIG=output/rollout/.../train_data_config.json \
#   MODEL_PATH=/path/to/sft_ckpt \
#   bash scripts/train.sh
#
# Multi node (one srun task per node, 8 GPUs/node):
#   DATA_CONFIG=... MODEL_PATH=... \
#   srun -N8 -n8 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8 \
#        bash scripts/train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

DATA_CONFIG="${DATA_CONFIG:?set DATA_CONFIG to a data config JSON}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the (SFT) checkpoint}"
MODEL_ID="${MODEL_ID:-qwen3-vl-4b}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/${MODEL_ID}/grpo}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/zero1.json}"
REPORT_TO="${REPORT_TO:-tensorboard}"

REWARD_FUNCS="${REWARD_FUNCS:-tiou}"
SAMPLE_WEIGHT_STD_POWER="${SAMPLE_WEIGHT_STD_POWER:-2}"
SAMPLE_WEIGHT_MEAN_POWER="${SAMPLE_WEIGHT_MEAN_POWER:-0}"
SAMPLE_WEIGHT_MEAN_SPEC="${SAMPLE_WEIGHT_MEAN_SPEC:-}"
SAVE_TRAIN_ROLLOUTS="${SAVE_TRAIN_ROLLOUTS:-True}"

MIN_TOKENS="${MIN_TOKENS:-1}"
TOTAL_TOKENS="${TOTAL_TOKENS:-65536}"
FPS="${FPS:-2}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-1024}"
MIN_VIDEO_LEN="${MIN_VIDEO_LEN:-3}"
MAX_VIDEO_LEN="${MAX_VIDEO_LEN:-3600}"
MAX_NUM_WORDS="${MAX_NUM_WORDS:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-False}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SEED="${SEED:-42}"
BETA="${BETA:-0.0}"

clusterx_nnodes="${NNODES:-${NODE_COUNT:-}}"
nnodes="${clusterx_nnodes:-${SLURM_JOB_NUM_NODES:-1}}"
if [[ -n "${NODE_RANK:-}" ]]; then
  node_rank="${NODE_RANK}"
elif [[ -n "${clusterx_nnodes}" ]] && [[ -n "${RANK:-}" ]]; then
  node_rank="${RANK}"
elif [[ "${nnodes}" -gt 1 ]] && [[ "${SLURM_NTASKS_PER_NODE:-1}" -eq 1 ]]; then
  node_rank="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
else
  node_rank=0
fi
[[ "${nnodes}" =~ ^[0-9]+$ ]] && [[ "${nnodes}" -ge 1 ]] || {
  echo "Invalid NNODES/NODE_COUNT: ${nnodes}" >&2
  exit 2
}
[[ "${node_rank}" =~ ^[0-9]+$ ]] && [[ "${node_rank}" -lt "${nnodes}" ]] || {
  echo "Invalid NODE_RANK/RANK ${node_rank} for nnodes=${nnodes}" >&2
  exit 2
}

clusterx_master_addr="${MASTER_ADDR:-}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  MASTER_ADDR="${MASTER_ADDR:-${master_node}}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
MASTER_PORT="${MASTER_PORT:-29500}"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L 2>/dev/null | wc -l) - 1)))}"
local_gpus=${#GPULIST[@]}
[[ "${local_gpus}" -ge 1 ]] || { echo "No GPUs found." >&2; exit 1; }
nproc_per_node="${NPROC_PER_NODE:-${local_gpus}}"
[[ "${nproc_per_node}" =~ ^[0-9]+$ ]] && [[ "${nproc_per_node}" -ge 1 ]] \
  && [[ "${nproc_per_node}" -le "${local_gpus}" ]] || {
  echo "Invalid NPROC_PER_NODE=${nproc_per_node}; visible GPUs=${local_gpus}" >&2
  exit 2
}
world_size=$((nnodes * nproc_per_node))

prod=$((BATCH_PER_DEVICE * world_size))
(( GLOBAL_BATCH_SIZE % prod == 0 )) || {
  echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by BATCH_PER_DEVICE*world_size (${prod})." >&2
  exit 1
}
grad_accum_steps=$((GLOBAL_BATCH_SIZE / prod))

# Coordinate a single run_name / hostfile across nodes via shared files.
coord_dir="${ROOT}/.deepspeed_hostfiles"
mkdir -p "${coord_dir}"
deepspeed_hostfile=""
if [[ "${nnodes}" -gt 1 ]] && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  deepspeed_hostfile="${coord_dir}/job-${SLURM_JOB_ID:-local}.hostfile"
  if [[ "${node_rank}" -eq 0 ]]; then
    tmp="${deepspeed_hostfile}.$$.tmp"; : > "${tmp}"
    while IFS= read -r h; do [[ -n "${h}" ]] && echo "${h} slots=${nproc_per_node}" >> "${tmp}"; done \
      < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    mv -f "${tmp}" "${deepspeed_hostfile}"
  else
    until [[ -s "${deepspeed_hostfile}" ]]; do sleep 1; done
  fi
fi

dist_run_id="${DIST_RUN_ID:-${SLURM_JOB_ID:-local}}"
if [[ "${nnodes}" -gt 1 ]] && [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  [[ -n "${DIST_RUN_ID:-}" ]] || {
    echo "Clusterx multi-node launch requires DIST_RUN_ID." >&2
    exit 2
  }
  addr_dir="${ROOT}/.dist_addr"
  master_file="${addr_dir}/${DIST_RUN_ID}.master_addr"
  mkdir -p "${addr_dir}"
  if [[ "${node_rank}" -eq 0 ]]; then
    detected_master_addr="$(hostname -I | awk '{print $1}')"
    [[ -n "${detected_master_addr}" ]] || { echo "Could not resolve rank-0 IP." >&2; exit 2; }
    tmp="${master_file}.$$.tmp"
    echo "${detected_master_addr}" > "${tmp}"
    mv -f "${tmp}" "${master_file}"
    MASTER_ADDR="${detected_master_addr}"
  else
    for ((waited = 0; waited < 300; waited++)); do
      [[ -s "${master_file}" ]] && break
      sleep 1
    done
    [[ -s "${master_file}" ]] || { echo "Timed out waiting for ${master_file}." >&2; exit 2; }
    MASTER_ADDR="$(head -n 1 "${master_file}")"
  fi
fi

run_name_file="${coord_dir}/job-${dist_run_id}.run_name"
if [[ "${node_rank}" -eq 0 ]]; then
  run_tag="$(date +%Y%m%d-%H%M)"
  run_name="grpo-${run_tag}_F${FPS}_MAXF${FPS_MAX_FRAMES}_T${TOTAL_TOKENS}_n${nnodes}x${nproc_per_node}"
  echo "${run_name}" > "${run_name_file}"
else
  for ((waited = 0; waited < 300; waited++)); do
    [[ -s "${run_name_file}" ]] && break
    sleep 1
  done
  [[ -s "${run_name_file}" ]] || { echo "Timed out waiting for ${run_name_file}." >&2; exit 2; }
  run_name="$(cat "${run_name_file}")"
fi
output_dir="${OUTPUT_ROOT}/${run_name}"
mkdir -p "${output_dir}"
echo "launch_context nnodes=${nnodes} node_count=${NODE_COUNT:-unset} node_rank=${node_rank} rank=${RANK:-unset} clusterx_master_addr=${clusterx_master_addr:-unset} master_addr=${MASTER_ADDR} master_port=${MASTER_PORT} nproc_per_node=${nproc_per_node} dist_run_id=${dist_run_id}"
echo "world_size=${world_size} grad_accum=${grad_accum_steps} output_dir=${output_dir}"

if [[ "${nnodes}" -le 1 ]]; then
  launcher=(deepspeed)
elif [[ -n "${deepspeed_hostfile}" ]]; then
  launcher=(deepspeed --hostfile="${deepspeed_hostfile}" --no_ssh
    --num_gpus="${nproc_per_node}" --num_nodes="${nnodes}" --node_rank="${node_rank}"
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
else
  command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found in PATH." >&2; exit 2; }
  launcher=(torchrun --nnodes="${nnodes}" --nproc-per-node="${nproc_per_node}"
    --node_rank="${node_rank}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
fi

"${launcher[@]}" timelens/train.py \
  --bf16 True --fp16 False --tf32 True --disable_flash_attn2 "${DISABLE_FLASH_ATTN2}" \
  --gradient_checkpointing True \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${MODEL_PATH}" \
  --model_id "${MODEL_ID}" \
  --data_config "${DATA_CONFIG}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${MIN_TOKENS}" --total_tokens "${TOTAL_TOKENS}" \
  --fps "${FPS}" --fps_max_frames "${FPS_MAX_FRAMES}" \
  --min_video_len "${MIN_VIDEO_LEN}" --max_video_len "${MAX_VIDEO_LEN}" \
  --max_num_words "${MAX_NUM_WORDS}" \
  --freeze_vision_tower True --freeze_llm False --freeze_merger False \
  --lr_scheduler_type constant --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${EPOCHS}" --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --num_generations "${NUM_GENERATIONS}" --steps_per_generation 1 \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --scale_rewards False \
  --beta "${BETA}" \
  --reward_funcs "${REWARD_FUNCS}" \
  --sample_weight_std_power "${SAMPLE_WEIGHT_STD_POWER}" \
  --sample_weight_mean_power "${SAMPLE_WEIGHT_MEAN_POWER}" \
  --sample_weight_mean_spec "${SAMPLE_WEIGHT_MEAN_SPEC}" \
  --save_train_rollouts "${SAVE_TRAIN_ROLLOUTS}" --save_train_rollouts_every_n_steps 1 \
  --logging_steps 1 --save_strategy steps --save_steps "${SAVE_STEPS}" --save_only_model True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" --log_completions True \
  --use_liger False --use_liger_loss False \
  --max_grad_norm 1.0 --seed "${SEED}" \
  --report_to "${REPORT_TO}" --run_name "${MODEL_ID}-grpo/${run_name}" \
  --logging_dir "${output_dir}/tensorboard"
