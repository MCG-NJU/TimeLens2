#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHON="${PYTHON:-python}"
export DATA_CONFIG="${DATA_CONFIG:-${ROOT}/configs/data/timelens2.json}"
export MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the TimeLens2 SFT checkpoint}"
export PRED_ROOT="${PRED_ROOT:-${ROOT}/outputs/rollout/timelens2}"
export NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export MIN_TOKENS="${MIN_TOKENS:-1}"
export TOTAL_TOKENS="${TOTAL_TOKENS:-16384}"
export FPS="${FPS:-2}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-512}"
export SEED="${SEED:-42}"

mkdir -p "${PRED_ROOT}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
LOG_FILE="${LOG_FILE:-${PRED_ROOT}/rollout-${LOG_TIMESTAMP}.log}"
echo "Logging rollout output to ${LOG_FILE}"
bash "${SCRIPT_DIR}/rollout.sh" 2>&1 | tee -a "${LOG_FILE}"
