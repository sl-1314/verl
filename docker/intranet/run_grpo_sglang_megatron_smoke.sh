#!/usr/bin/env bash
# ==============================================================================
# verl 端到端 GRPO 冒烟/训练启动脚本 (sglang Rollout + Megatron Train + Python 3.12)
# 特点: 自动由 main_ppo 内部启停 Ray 集群，任务结束后自动回收，无常驻进程残留
# ==============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PATH="/opt/verl/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/opt/verl/lib:/opt/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/verl/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/opt/sglang-cache/hf"
export TRITON_CACHE_DIR="/opt/sglang-cache/triton"
export FLASHINFER_WORKSPACE_BASE="/opt/sglang-cache/flashinfer"
export RAY_TMPDIR="/tmp/verl-isolated/ray"
export SGLANG_RUST_BUILD_MODE=never
export VERL_USE_UV=0

MODEL_PATH="${MODEL_PATH:-/opt/sglang-cache/hf/Qwen/Qwen3-0.6B}"
TRAIN_FILES="${TRAIN_FILES:-['/tmp/verl-isolated/data/train.parquet']}"
VAL_FILES="${VAL_FILES:-['/tmp/verl-isolated/data/test.parquet']}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"

cd /opt/verl-src
/opt/verl/bin/python -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=64 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=verl_e2e_smoke \
    trainer.experiment_name=qwen3_0_6b_sglang_megatron \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@"
