# -*- coding: utf-8 -*-
# Fast cluster smoke test for Qwen3.5 BI=1 0-diff image (754999989):
# verifies Python env (/env/mlite-bi), BI=1 env vars, C++/CUDA extensions,
# /env/bi-image-manifest.json, and DolphinFS model/data paths.
import json
import os
import platform
import socket
from pathlib import Path

import torch
import triton
import vllm
import fla
import verl
import megatron.lite
import verl_mlite
import transformer_engine.pytorch
import math_verify
import tensorboard
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
import vllm.vllm_flash_attn._vllm_fa3_C

print("=== qwen35-bi-0diff cluster smoke test ===")
print("hostname:", socket.gethostname())
print("python:", platform.python_version())
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
print("verl.__file__:", verl.__file__)
assert "/site-packages/verl/" not in os.path.abspath(verl.__file__), (
    f"Expected uploaded working_dir verl to shadow site-packages, got {verl.__file__}"
)

required_env = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_V1": "1",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    "CUBLASLT_WORKSPACE_SIZE": "1",
    "VLLM_USE_AOT_COMPILE": "0",
}
for k, expected in required_env.items():
    actual = os.environ.get(k)
    print(f"env[{k}] = {actual!r} (expected {expected!r})")
    assert actual == expected, f"Mismatch on {k}: {actual!r} != {expected!r}"

manifest_path = Path("/env/bi-image-manifest.json")
assert manifest_path.is_file(), "Missing /env/bi-image-manifest.json"
manifest = json.loads(manifest_path.read_text())
print("bi-image-manifest:", json.dumps(manifest))

model_cfg = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/LLMbasemodels/huggingface.co/Qwen/Qwen3.5-35B-A3B/config.json")
train_pq = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/wangshulin02/qwen35_bi/data/gsm8k/train.parquet")
assert model_cfg.is_file(), f"Model config not visible: {model_cfg}"
assert train_pq.is_file(), f"GSM8K train.parquet not visible: {train_pq}"
print("SMOKE_BI_ENV_OK")
