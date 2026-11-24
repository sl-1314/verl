#!/usr/bin/env bash
# Exact C10 4-GPU acceptance reproduction wrapper (1 node, using 4 GPUs: vLLM EP=4 x mlite FSDP2 EP=4 CP=1)
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES=1
export NGPUS_PER_NODE=4
export ACTOR_EP="${ACTOR_EP:-4}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ROLLOUT_DP="${ROLLOUT_DP:-4}"
export ROLLOUT_PCP="${ROLLOUT_PCP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_EP="${ROLLOUT_EP:-4}"
exec bash "${SCRIPT_DIR}/run_qwen35_35b_bi_1n8g.sh" "$@"
