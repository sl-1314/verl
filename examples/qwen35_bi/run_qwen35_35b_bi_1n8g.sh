#!/usr/bin/env bash
# Qwen3.5-35B-A3B (40L BF16, Hybrid GDN/GQA + MoE 256 experts) GRPO RL under BI=1 (0-diff)
# Validated for MLP image:
#   registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/xt-ai-search/training_nvidia-cuda_12.9.1-devel-ubuntu22.04_qwen35_bi_6efe9bd4:1.0.0
#
# Usage (via my_starship, default --workdir ./jobs/verl):
#   F1 (submit to running Ray cluster):
#     python submit_task.py --id qwen35_bi_run1 --command bash examples/qwen35_bi/run_qwen35_35b_bi_1n8g.sh
#   F2 (one-shot cluster + task):
#     python launch_cluster.py --id qwen35_bi_f2 --num_gpu 8 --command bash examples/qwen35_bi/run_qwen35_35b_bi_1n8g.sh
set -xeuo pipefail

PY="${PY:-/env/mlite-bi/bin/python}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/env/repos/Megatron-LM}"
EXAMPLE_ROOT="${MEGATRON_ROOT}/experimental/lite/examples/verl"
LITE_ROOT="${MEGATRON_ROOT}/experimental/lite"
VERL_ROOT="${VERL_ROOT:-$(pwd)}"

export PATH="/env/mlite-bi/bin:/env/base-py312/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/env/base-py312/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# Put . and VERL_ROOT first so uploaded jobs/verl always shadows /env/mlite-bi/.../site-packages/verl
export PYTHONPATH=".:${VERL_ROOT}:${EXAMPLE_ROOT}:${LITE_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"

# Batch-Invariant (BI=1) deterministic environment + thread caps for 160-core node
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VERL_MLITE_SKIP_RUNTIME_PATCHES=0
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=4
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export CUBLASLT_WORKSPACE_SIZE=1
export VLLM_USE_AOT_COMPILE=0

DOLPHINFS_USER_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/wangshulin02/qwen35_bi"
MODEL_PATH="${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/LLMbasemodels/huggingface.co/Qwen/Qwen3.5-35B-A3B}"
TRAIN_FILE="${TRAIN_FILE:-${DOLPHINFS_USER_BASE}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${DOLPHINFS_USER_BASE}/data/gsm8k/test.parquet}"

for f in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${TEST_FILE}"; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: required input path not accessible in container: ${f}" >&2
    exit 64
  fi
done

# Topology knobs (default 1 node x 8 H20 GPUs; override NGPUS_PER_NODE=4 for 4-GPU runs)
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
WORLD_SIZE="$((NNODES * NGPUS_PER_NODE))"

ACTOR_TP="${ACTOR_TP:-1}"
ACTOR_PP="${ACTOR_PP:-1}"
ACTOR_VPP="${ACTOR_VPP:-1}"
ACTOR_CP="${ACTOR_CP:-1}"
ACTOR_EP="${ACTOR_EP:-${NGPUS_PER_NODE}}"
ACTOR_ETP="${ACTOR_ETP:-1}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_PCP="${ROLLOUT_PCP:-1}"
ROLLOUT_DP="${ROLLOUT_DP:-$((WORLD_SIZE / (ROLLOUT_TP * ROLLOUT_PCP)))}"
ROLLOUT_EP="${ROLLOUT_EP:-$((ROLLOUT_TP * ROLLOUT_PCP * ROLLOUT_DP))}"

# Hyperparameters & FSDP2 memory settings (validated on H20-3e 141GB HBM)
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
ROLLOUT_N="${ROLLOUT_N:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"

PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-0.0}"
FSDP2_USE_FP32_SHARDS="${FSDP2_USE_FP32_SHARDS:-False}"
FSDP2_USE_FP32_MASTER="${FSDP2_USE_FP32_MASTER:-False}"
FSDP2_ADAMW_FOREACH="${FSDP2_ADAMW_FOREACH:-False}"

RUN_ID="${RUN_ID:-${RAY_JOB_SUBMISSION_ID:-qwen35_35b_bi_$(date +%m%d%H%M%S)}}"
OUT_DIR="${OUT_DIR:-${DOLPHINFS_USER_BASE}/outputs/${RUN_ID}}"
mkdir -p "${OUT_DIR}/ckpt" "${OUT_DIR}/rollout_dump"

export VERL_FILE_LOGGER_PATH="${OUT_DIR}/metrics.jsonl"
export TORCHINDUCTOR_CACHE_DIR="/workdir/cache/torchinductor_${RUN_ID}"
export TRITON_CACHE_DIR="/workdir/cache/triton_${RUN_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

if "${PY}" -c "from torch.utils.tensorboard import SummaryWriter" >/dev/null 2>&1; then
  TRAINER_LOGGERS="${TRAINER_LOGGERS:-[console,file,tensorboard]}"
else
  TRAINER_LOGGERS="${TRAINER_LOGGERS:-[console,file]}"
fi

"${PY}" -m verl_mlite.launch verl.trainer.main_ppo \
  hydra.searchpath=[pkg://verl_mlite.config] \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor@actor_rollout_ref.actor=mlite_actor \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
  actor_rollout_ref.actor.engine.dtype=bfloat16 \
  actor_rollout_ref.actor.engine.model_name=auto \
  actor_rollout_ref.actor.engine.impl=lite \
  actor_rollout_ref.actor.engine.tp="${ACTOR_TP}" \
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}" \
  actor_rollout_ref.actor.engine.vpp="${ACTOR_VPP}" \
  actor_rollout_ref.actor.engine.cp="${ACTOR_CP}" \
  actor_rollout_ref.actor.engine.ep="${ACTOR_EP}" \
  actor_rollout_ref.actor.engine.etp="${ACTOR_ETP}" \
  actor_rollout_ref.actor.engine.param_offload="${PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.engine.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction="${OFFLOAD_FRACTION}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.fsdp2_use_fp32_shards="${FSDP2_USE_FP32_SHARDS}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.fsdp2_use_fp32_master="${FSDP2_USE_FP32_MASTER}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.fsdp2_adamw_foreach="${FSDP2_ADAMW_FOREACH}" \
  actor_rollout_ref.actor.engine.attention_backend_override=flash \
  actor_rollout_ref.actor.engine.impl_cfg.use_thd=True \
  +actor_rollout_ref.actor.engine.impl_cfg.optimizer=fsdp2 \
  +actor_rollout_ref.actor.engine.impl_cfg.bi_mode=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.prefill_context_parallel_size="${ROLLOUT_PCP}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  critic.enable=False \
  trainer.balance_batch=True \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.project_name=qwen35-bi-0diff \
  trainer.experiment_name="${RUN_ID}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${OUT_DIR}/ckpt" \
  trainer.val_before_train=False \
  +trainer.rollout_data_dir="${OUT_DIR}/rollout_dump" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH}" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.OPENBLAS_NUM_THREADS=1 \
  ++ray_kwargs.ray_init.runtime_env.env_vars.MKL_NUM_THREADS=1 \
  ++ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS=4 \
  "$@" 2>&1 | tee "${OUT_DIR}/run.log"
