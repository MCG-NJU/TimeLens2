#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODELS="${MODELS:-TimeLens2-4B TimeLens2-8B}"
DATASETS="${DATASETS:-TimeLens_Charades_4fps TimeLens_ActivityNet_4fps TimeLens_QVHighlights_4fps VUE_TR_1fps_limit_2048_px480_ctx128k VUE_TR_V2_1fps_limit_2048_px480_ctx128k MomentSeeker_2fps_limit_2048_px480_ctx128k Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k}"
USE_LLM_JUDGE="${USE_LLM_JUDGE:-auto}"
LLM_JUDGE="${LLM_JUDGE:-qwen3-235b-a22b-thinking-2507}"

export LMUData="${LMUData:-$PWD/data/lmudata}"
export VLMEVAL_TRUST_LOCAL_IMAGE_PATHS="${VLMEVAL_TRUST_LOCAL_IMAGE_PATHS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$LLM_JUDGE" == qwen3-235b-a22b-* ]]; then
  export LOCAL_LLM="${LOCAL_LLM:-qwen3-235b}"
fi

[[ "$USE_LLM_JUDGE" =~ ^(auto|true|false)$ ]] || {
  echo "USE_LLM_JUDGE must be auto, true, or false" >&2
  exit 2
}

read -r -a model_list <<< "$MODELS"
read -r -a dataset_list <<< "$DATASETS"

for dataset in "${dataset_list[@]}"; do
  judge="$LLM_JUDGE"
  if [[ "$USE_LLM_JUDGE" == false ]]; then
    judge=exact_matching
  elif [[ "$USE_LLM_JUDGE" == auto ]]; then
    case "$dataset" in
      TimeLens_Charades_*|TimeLens_ActivityNet_*|TimeLens_QVHighlights_*)
        judge=exact_matching
        ;;
    esac
  fi

  echo "[evaluation] dataset=$dataset judge=$judge"
  judge_args=(--judge "$judge")
  [[ "$judge" == exact_matching ]] || judge_args+=(--retry "${JUDGE_RETRY:-3}")

  "${PYTHON:-python}" -m torch.distributed.run \
    --nproc-per-node="${N_GPU:-8}" \
    --master-port="${MASTER_PORT:-29501}" \
    run.py \
    --data "$dataset" \
    --model "${model_list[@]}" \
    --work-dir "${OUTPUT_DIR:-$PWD/outputs}" \
    --verbose --reuse \
    --check-extracted-frames "${CHECK_EXTRACTED_FRAMES:-True}" \
    "${judge_args[@]}"
done
