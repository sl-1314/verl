#!/usr/bin/env bash
# GRPO on the NeMo Automodel engine for Qwen3.5-35B-A3B (hybrid GDN + MoE) with vLLM rollout.
# Validated topology: single node 4x H20 (141GB), colocate, vLLM rollout TP4,
# automodel actor+ref, ep_size=4 (ep>1 required: build_automodel_model force_hf for
# qwen archs when ep_size<=1), dispatcher=torch + experts=torch_mm (no DeepEP /
# grouped_gemm dependency), attn=sdpa (packed Qwen3.5-MoE rejects TE attention).
# Text-only: MTP off (no speculative_config), vision tower stays idle.
# env: /tmp/workspace/qwen35-vllm-automodel/envs/qwen35-am (torch 2.14.0+cu126, vllm source,
# transformers 5.18.0, flash-linear-attention). LD_LIBRARY_PATH must include the env lib
# (conda gcc15 libstdc++) at runtime.

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5-Automodel}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B-A3B}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-4}
EP_SIZE=${EP_SIZE:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/tmp/workspace/qwen35-vllm-automodel"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/cache/Qwen3.5-35B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/outputs/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
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
    data.train_batch_size=32
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    model_engine=automodel
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.automodel.distributed_strategy=fsdp2
    actor_rollout_ref.actor.automodel.tp_size=1
    actor_rollout_ref.actor.automodel.pp_size=1
    actor_rollout_ref.actor.automodel.cp_size=1
    actor_rollout_ref.actor.automodel.ep_size=${EP_SIZE}
    actor_rollout_ref.actor.automodel.dp_replicate_size=1
    actor_rollout_ref.actor.automodel.param_offload=True
    actor_rollout_ref.actor.automodel.optimizer_offload=True
    actor_rollout_ref.actor.automodel.activation_checkpointing=True
    actor_rollout_ref.actor.automodel.model_dtype=bf16
    actor_rollout_ref.actor.automodel.attn_implementation=flash_attention_2
    actor_rollout_ref.actor.automodel.enable_compile=False
    actor_rollout_ref.actor.automodel.backend_config.attn=sdpa
    actor_rollout_ref.actor.automodel.backend_config.linear=te
    actor_rollout_ref.actor.automodel.backend_config.rms_norm=torch_fp32
    actor_rollout_ref.actor.automodel.backend_config.dispatcher=torch
    actor_rollout_ref.actor.automodel.backend_config.experts=torch_mm
    actor_rollout_ref.actor.automodel.backend_config.enable_fsdp_optimizations=True
    actor_rollout_ref.actor.automodel.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.automodel.entropy_checkpointing=True
    actor_rollout_ref.actor.automodel.moe_config.reshard_after_forward=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console','wandb']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    2>&1 | tee "logs/${EXPERIMENT_NAME}_${start_time}.log"
