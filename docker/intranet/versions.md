# verl (RL) + sglang (Rollout) + Megatron (Train) · Python 3.12 版本锁与冲突解决清单 (`versions.md`)

> 适用环境：内网 `nvidia-cuda:12.9.1-devel-ubuntu22.04`（CUDA 12.9 + NVIDIA H20 `sm90`）+ **Python 3.12.3**
> 对应仓库基线：`verl @ 0938fcb3` + `sglang @ 4349538c02`

---

## 一、核心框架与底层算子矩阵（均在 `/opt/verl` Python 3.12.3 中实测验证）

| 层级 | 包名 | 实测安装版本 | 安装来源 | 关键适配与冲突解决说明 |
|---|---|---|---|---|
| Python & 工具 | `python` / `libprotobuf` | `3.12.3` / `5.26.1` (`protoc 26.1`) | 美团内网 Conda `cloud/conda-forge` | 加 `--override-channels` 免 Anaconda ToS |
| PyTorch | `torch` / `torchvision` / `torchaudio` | `2.13.0+cu129` / `0.28.0+cu129` / `2.11.0+cu129` (`cp312`) | `mirrors.nju.edu.cn/pytorch/whl/cu129` | 底层 3.2 GB `nvidia-*-cu12` 从美团内网 `pip.sankuai.com` 千兆直连预装；Python 3.12 下必须显式预装 `typing-extensions>=4.10.0` |
| Rollout (sglang) | `sglang` | `0.5.23` (`4349538c02`) | `/opt/verl-sglang-src` (`--no-build-isolation --no-deps -e`) | `pyproject.toml` 执行 cu13→cu12 改写 |
| Rollout 算子 | `sglang-kernel` | `0.4.6.post1+cu129` (`cp310-abi3`) | GitHub Release (`gh-proxy.com`) + SHA256 校验 | `cp310-abi3` 稳定 ABI 直接兼容 Python 3.12 |
| Rollout 算子 | `flashinfer_python[cu12]` | `0.6.18` | 美团内网 `pip.sankuai.com` | 显式锁定 `cuda-python==12.9.10`、`cuda-bindings==12.9.10`、`nvidia-cudnn-frontend==1.28.0`、`protobuf==6.33.6` |
| Train (Megatron) | `megatron-core` / `megatron-bridge` | `0.19.2` (`cp312`) / `0.6.2` | 美团内网 `pip.sankuai.com` | `megatron-bridge` 按 `verl/pyproject.toml` 要求 `--no-deps` 安装；配套 `megatron-energon==7.4.1`、`nvidia-modelopt==0.44.0` |
| Train 算子 | `flash-attn` (FA2) + `flash-attn-4` (FA4 CuTe) | `2.8.3` + `4.0.0b37` | 美团内网 `pip.sankuai.com` | **必须先装 `flash-attn==2.8.3`（`FLASH_ATTN_CUDA_ARCHS="90"` 编译），后装 `flash-attn-4==4.0.0b37`（`--no-deps`）**，用适配 `nvidia-cutlass-dsl==4.6.2` 的新版 `flash_attn/cute/` 覆盖 `2.8.3` 自带的旧 `cute/`，否则 `import megatron.core` 报 `AttributeError: ThrMma` |
| Train 算子 | `apex` | `0.1` (`575968bc`) | GitHub `NVIDIA/apex` (`--cpp_ext --cuda_ext`) | 针对 `TORCH_CUDA_ARCH_LIST="9.0"` 编译 `amp_C`、`fused_layer_norm_cuda`、`scaled_masked_softmax_cuda` 等 |
| Train 算子 | `transformer-engine` | `2.16.1` (`cu12` + `torch`) | 美团内网 `pip.sankuai.com` | `transformer-engine-cu12==2.16.1` 为预编译 wheel；编译 `transformer-engine-torch==2.16.1` 时将 `CPATH` / `LIBRARY_PATH` 指向 `/opt/verl/lib/python3.12/site-packages/nvidia/cudnn/{include,lib}`，无需 root 安装系统级 `libcudnn9-dev` |
| Train 通信 | `cupy-cuda12x` | `14.0.1` (`cp312`) | 美团内网 `pip.sankuai.com` | 替代 cu13 下的 `cupy-cuda13x==14.0.1`，供 `ray.util.collective` 与 Megatron MoE 使用 |
| HF 生态公共锁 | `transformers` / `tokenizers` | `5.12.1` / `0.22.2` | 美团内网 `pip.sankuai.com` | 同时满足 `sglang @ 4349538c02`（避免 `transformers 5.16.1` 的 `qwen3_asr` 重复注册报错）与 `verl/requirements.txt`（`>=5.5.3,!=5.6.0,<5.17`） |
| RL 框架 (verl) | `verl` | `0.10.0.dev0` (`0938fcb3`) | `/opt/verl-src` (`--no-build-isolation --no-deps -e`) | 配套 `ray[default]==2.59.0`、`tensordict==0.10.0`、`torchdata==0.11.0`、`trl==0.27.0`、`liger-kernel==0.8.4`、`flash-linear-attention==0.5.2`、`TransferQueue==0.1.11`（`--no-deps`） |
