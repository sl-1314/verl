#!/usr/bin/env bash
# C8 minimal GRPO for the 0-diff (bitwise-vllm) recipe: single GPU colocate,
# vLLM rollout TP1 (the C5-validated BI configuration) + automodel actor with
# zero_diff=True (ddp world=1, full-model replica). Training step freezes the
# routed+shared experts (full-param grads do not fit one H20; FSDP/EP sharding
# breaks the recipe's full-weight contract — see C8 acceptance notes).
# Set VERL_ZD_DUMP=<dir> before launching to capture in-loop actor logits for
# the bitwise actor-vs-vllm gate.
# env: /tmp/workspace/qwen35-vllm-automodel/envs/qwen35-am; requires
# VLLM_BATCH_INVARIANT=1 + TRITON_PTXAS_PATH + LD_LIBRARY_PATH=<env>/lib.

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5-ZD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B-A3B-zero-diff}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/tmp/workspace/qwen35-vllm-automodel"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/cache/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/cache/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/cache/data/gsm8k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${RAY_DATA_HOME}/repos/verl"}
########################### end user-adjustable ###########################

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
    actor_rollout_ref.actor.automodel.distributed_strategy=ddp
    actor_rollout_ref.actor.automodel.tp_size=1
    actor_rollout_ref.actor.automodel.pp_size=1
    actor_rollout_ref.actor.automodel.cp_size=1
    actor_rollout_ref.actor.automodel.ep_size=1
    actor_rollout_ref.actor.automodel.dp_replicate_size=1
    actor_rollout_ref.actor.automodel.param_offload=True
    actor_rollout_ref.actor.automodel.optimizer_offload=True
    actor_rollout_ref.actor.automodel.activation_checkpointing=False
    actor_rollout_ref.actor.automodel.model_dtype=bf16
    actor_rollout_ref.actor.automodel.attn_implementation=sdpa
    actor_rollout_ref.actor.automodel.use_torch_compile=False
    actor_rollout_ref.actor.automodel.zero_diff=True
    actor_rollout_ref.actor.automodel.zero_diff_freeze_experts=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
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
