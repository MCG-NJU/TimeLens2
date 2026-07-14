#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

read -r -a MODEL_LIST <<< "${MODELS:-TimeLens2-4B TimeLens2-8B}"
read -r -a DATASET_LIST <<< "${DATASETS:-TimeLens_Charades_4fps TimeLens_ActivityNet_4fps TimeLens_QVHighlights_4fps VUE_TR_1fps_limit_2048_px480_ctx128k VUE_TR_V2_1fps_limit_2048_px480_ctx128k MomentSeeker_2fps_limit_2048_px480_ctx128k Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k}"

PYTHON_BIN="${PYTHON:-python}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

export LMUData="${LMUData:-${ROOT}/data/lmudata}"
export VLMEVAL_TRUST_LOCAL_IMAGE_PATHS="${VLMEVAL_TRUST_LOCAL_IMAGE_PATHS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT:-29501}" \
  run.py \
  --data "${DATASET_LIST[@]}" \
  --model "${MODEL_LIST[@]}" \
  --work-dir "${WORK_DIR:-${ROOT}/outputs}" \
  --verbose \
  --reuse \
  --check-extracted-frames "${CHECK_EXTRACTED_FRAMES:-True}"
