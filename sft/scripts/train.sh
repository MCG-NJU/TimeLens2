#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.py>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CONFIG=$1
PYTHON_BIN="${PYTHON:-python}"

clusterx_nnodes="${NNODES:-${NODE_COUNT:-}}"
NNODES="${clusterx_nnodes:-${SLURM_JOB_NUM_NODES:-1}}"
if [[ -n "${NODE_RANK:-}" ]]; then
  NODE_RANK="${NODE_RANK}"
elif [[ -n "${clusterx_nnodes}" && -n "${RANK:-}" ]]; then
  NODE_RANK="${RANK}"
elif [[ "${NNODES}" -gt 1 && "${SLURM_NTASKS_PER_NODE:-1}" -eq 1 ]]; then
  NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
else
  NODE_RANK=0
fi
MASTER_PORT="${MASTER_PORT:-29500}"

[[ "${NNODES}" =~ ^[0-9]+$ && "${NNODES}" -ge 1 ]] || {
  echo "Invalid NNODES/NODE_COUNT: ${NNODES}" >&2
  exit 2
}
[[ "${NODE_RANK}" =~ ^[0-9]+$ && "${NODE_RANK}" -lt "${NNODES}" ]] || {
  echo "Invalid NODE_RANK/RANK ${NODE_RANK} for NNODES=${NNODES}" >&2
  exit 2
}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

clusterx_master_addr="${MASTER_ADDR:-}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  master_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  MASTER_ADDR="${MASTER_ADDR:-${master_node}}"
elif [[ "${NNODES}" -gt 1 ]]; then
  [[ -n "${DIST_RUN_ID:-}" ]] || {
    echo "Clusterx multi-node launch requires DIST_RUN_ID." >&2
    exit 2
  }
  coord_dir="${ROOT}/.dist_addr"
  master_file="${coord_dir}/${DIST_RUN_ID}.master_addr"
  mkdir -p "${coord_dir}"

  if [[ "${NODE_RANK}" -eq 0 ]]; then
    detected_master_addr="$(hostname -I | awk '{print $1}')"
    [[ -n "${detected_master_addr}" ]] || {
      echo "Could not resolve rank-0 IP." >&2
      exit 2
    }
    tmp="${master_file}.$$.tmp"
    echo "${detected_master_addr}" > "${tmp}"
    mv -f "${tmp}" "${master_file}"
    MASTER_ADDR="${detected_master_addr}"
  else
    for ((waited = 0; waited < 600; waited++)); do
      [[ -s "${master_file}" ]] && break
      sleep 1
    done
    [[ -s "${master_file}" ]] || {
      echo "Timed out waiting for ${master_file}." >&2
      exit 2
    }
    MASTER_ADDR="$(head -n 1 "${master_file}")"
  fi
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-0}"
export XTUNER_GC_ENABLE="${XTUNER_GC_ENABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "launch_context nnodes=${NNODES} node_count=${NODE_COUNT:-unset} node_rank=${NODE_RANK} rank=${RANK:-unset} clusterx_master_addr=${clusterx_master_addr:-unset} master_addr=${MASTER_ADDR} master_port=${MASTER_PORT} nproc_per_node=${NPROC_PER_NODE} dist_run_id=${DIST_RUN_ID:-local}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)' 2>/dev/null || command -v "${PYTHON_BIN}" || true)"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  xtuner/v1/train/cli/sft.py --config "${CONFIG}"
