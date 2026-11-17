#!/usr/bin/env bash
# C9-E closure run: PCR (prefill-recompute rollout) on the TARGET topology.
# Same actor as C9-D (automodel FSDP2+EP full-param, zero_diff + zero_diff_ep);
# the rollout is replaced by the in-process SPMD PCR engines (rollout.mode=pcr,
# one TP1 BI vLLM engine per worker, EP4 across the 4 workers).
#
# Closure gate: >=2 complete GRPO steps with the in-loop debug metrics
#   training/rollout_logprobs_{mismatch_count,diff_max,diff_mean}
# at 0 / 0.0 bitwise (PCR per-step logprob == actor recompute, E3c-proven),
# plus loop health (grad_norm, update_weights round-trip) vs C9-D.
#
# env: /tmp/workspace/qwen35-vllm-automodel/envs/qwen35-am; requires
# VLLM_BATCH_INVARIANT=1 (BI norm pin, vllm qwen35-0diff-bi-gdn) +
# TRITON_PTXAS_PATH + LD_LIBRARY_PATH=<env>/lib, set below.

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5-ZD-C9E-PCR}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B-A3B-zero-diff-c9e-pcr}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
NNODES=${NNODES:-1}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/tmp/workspace/qwen35-vllm-automodel"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/cache/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/cache/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/cache/data/gsm8k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${RAY_DATA_HOME}/repos/verl"}
########################### end user-adjustable ###########################

# 0-diff launch env (same set C8/C9-D validated; VLLM_BATCH_INVARIANT pins the
# GDN norm tile so every forward length is bitwise row-invariant).
export VLLM_BATCH_INVARIANT=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export LD_LIBRARY_PATH="${RAY_DATA_HOME}/envs/qwen35-am/lib:${LD_LIBRARY_PATH:-}"
export VERL_VLLM_FORCE_VERSION=0.18.0

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=8
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    model_engine=automodel
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.automodel.distributed_strategy=fsdp2
    actor_rollout_ref.actor.automodel.tp_size=1
    actor_rollout_ref.actor.automodel.pp_size=1
    actor_rollout_ref.actor.automodel.cp_size=1
    actor_rollout_ref.actor.automodel.ep_size=4
    actor_rollout_ref.actor.automodel.dp_replicate_size=1
    actor_rollout_ref.actor.automodel.param_offload=True
    actor_rollout_ref.actor.automodel.optimizer_offload=True
    actor_rollout_ref.actor.automodel.activation_checkpointing=False
    actor_rollout_ref.actor.automodel.model_dtype=bf16
    actor_rollout_ref.actor.automodel.attn_implementation=sdpa
    actor_rollout_ref.actor.automodel.use_torch_compile=False
    actor_rollout_ref.actor.automodel.zero_diff=True
    actor_rollout_ref.actor.automodel.zero_diff_freeze_experts=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    # PCR: in-process SPMD rollout object on each actor worker (registry
    # ("vllm","pcr") -> PCRRollout). No server actor is launched.
    actor_rollout_ref.rollout.mode=pcr
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    # The PCR rollout ignores these (mesh is TP1-per-worker by construction)
    # but they satisfy the RolloutConfig validation.
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.prompt_length=512
    actor_rollout_ref.rollout.response_length=512
    actor_rollout_ref.rollout.max_model_len=1536
    actor_rollout_ref.rollout.max_num_batched_tokens=2048
    actor_rollout_ref.rollout.max_num_seqs=64
    # E3c evidence parity: eager mode, no prefix caching (full re-prefill).
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.free_cache_engine=True
    # always record per-token logprobs in the rollout output (drives the
    # training/rollout_logprobs_* debug metrics = the closure gate)
    actor_rollout_ref.rollout.calculate_log_probs=True
    # in-process naive weight sync through the worker's update_weights path
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    # closure gate contract: >=2 complete GRPO steps
    trainer.total_training_steps=2
    trainer.default_local_dir="${RAY_DATA_HOME}/outputs/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    2>&1 | tee "logs/${EXPERIMENT_NAME}_${start_time}.log"
