#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORK_DIR="${ROOT}/outputs/sft-8b"

mkdir -p "${WORK_DIR}"
LOG_TIMESTAMP="$(date +%Y%m%d%H%M%S)"
LOG_NODE_RANK="${NODE_RANK:-${RANK:-0}}"
LOG_RUN_ID="${DIST_RUN_ID:-local}"
LOG_FILE="${WORK_DIR}/train-${LOG_RUN_ID}-node${LOG_NODE_RANK}-${LOG_TIMESTAMP}.log"
echo "Logging train output to ${LOG_FILE}"
bash "${SCRIPT_DIR}/train.sh" "${ROOT}/configs/qwen3_vl_8b_sft.py" \
  2>&1 | tee -a "${LOG_FILE}"
