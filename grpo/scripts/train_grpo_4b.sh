#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHON="${PYTHON:-python}"
export DATA_CONFIG="${DATA_CONFIG:-${ROOT}/outputs/rollout/timelens2_93k/train_data_config.json}"
export MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the 4B SFT checkpoint}"
export MODEL_ID="qwen3-vl-4b"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/grpo/timelens2-4b}"

export REWARD_FUNCS="twass1,tiou,parse_penalty"
export BETA="0.04"
export SAMPLE_WEIGHT_STD_POWER="0"
export SAMPLE_WEIGHT_MEAN_POWER="2"
export SAMPLE_WEIGHT_MEAN_SPEC="tiou"
export SAVE_TRAIN_ROLLOUTS="True"

export MIN_TOKENS="1"
export TOTAL_TOKENS="16384"
export FPS="2"
export FPS_MAX_FRAMES="512"
export MIN_VIDEO_LEN="3"
export MAX_VIDEO_LEN="3600"
export GLOBAL_BATCH_SIZE="64"
export BATCH_PER_DEVICE="1"
export NUM_GENERATIONS="8"
export LEARNING_RATE="1e-6"
export EPOCHS="1"
export MAX_STEPS="300"
export SAVE_STEPS="50"
export SEED="42"

mkdir -p "${OUTPUT_ROOT}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/train-${LOG_TIMESTAMP}.log}"
echo "Logging train output to ${LOG_FILE}"
bash "${SCRIPT_DIR}/train.sh" 2>&1 | tee -a "${LOG_FILE}"
