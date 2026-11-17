#!/usr/bin/env bash
# C9-D GRPO for the 0-diff recipe: TARGET TOPOLOGY — 4xH20 colocate.
# vLLM rollout TP1 x DP4 x EP4 (BI mode, the C9-A-validated reference
# topology) + automodel actor with zero_diff FSDP2+EP full-param
# (zero_diff_ep: recipe-side EP semantics + C9-B/C-validated call-graph
# FSDP2 grouping). No frozen experts.
# Set VERL_ZD_DUMP=<dir> before launching to capture in-loop actor logits
# (per-rank filenames) for the bitwise actor-vs-vllm-EP4 gate (c9_d_gate).
# env: /tmp/workspace/qwen35-vllm-automodel/envs/qwen35-am; requires
# VLLM_BATCH_INVARIANT=1 + TRITON_PTXAS_PATH + LD_LIBRARY_PATH=<env>/lib.

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5-ZD-C9D}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B-A3B-zero-diff-c9d}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-1}
GEN_DP=${GEN_DP:-4}
# EP4 shards the routed experts across the 4 DP ranks: ~16 GiB routed +
# ~14 GiB dense (TP1) per GPU for weights; 0.30 leaves room for KV + the
# colocated actor's offloaded footprint. Rollout-side knobs never touch
# the 0-diff acceptance surface (actor logits vs vllm EP re-scoring).
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}

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
    actor_rollout_ref.actor.automodel.distributed_strategy=fsdp2
    actor_rollout_ref.actor.automodel.tp_size=1
    actor_rollout_ref.actor.automodel.pp_size=1
    actor_rollout_ref.actor.automodel.cp_size=1
    actor_rollout_ref.actor.automodel.ep_size=${GEN_DP}
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
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_DP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=4
    # prompts+responses capped at 512+512; the model default
    # (max_model_len=262144) would demand >5 GiB KV for a single max-len
    # request, which the colocate KV budget cannot cover.
    actor_rollout_ref.rollout.max_model_len=1536
    # 8 prompts x n=4 = 32 decode seqs / DP4 = 8 per DP rank; every decode
    # seq needs a GDN (Mamba-style) state cache block — keep the per-rank
    # budget consistent with C8 (64 per engine).
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=2048
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
    # C9-D gate contract: >=2 complete GRPO steps, then stop so the GPUs are
    # free for the offline vllm re-scoring gate.
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
