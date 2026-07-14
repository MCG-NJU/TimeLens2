#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export DATA_CONFIG="${DATA_CONFIG:-${ROOT}/configs/data/timelens2_sft.json}"
export WORK_DIR="${WORK_DIR:-${ROOT}/outputs/sft-8b}"

mkdir -p "${WORK_DIR}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
LOG_FILE="${LOG_FILE:-${WORK_DIR}/train-${LOG_TIMESTAMP}.log}"
echo "Logging train output to ${LOG_FILE}"
bash "${SCRIPT_DIR}/train.sh" "${ROOT}/configs/qwen3_vl_8b_sft.py" \
  2>&1 | tee -a "${LOG_FILE}"
