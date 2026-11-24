# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-wide compatibility and BI=1 long-context memory hooks for VERL + MLite."""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from collections.abc import Callable
from functools import cache

# Cap default OpenBLAS / MKL / OMP thread pools before numpy/scipy/torch import so
# 30+ concurrent Ray actors on a 160-core node do not spawn 128 OpenBLAS threads each
# and exhaust pthread_create ("Resource temporarily unavailable").
for _thr_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thr_var, "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

warnings.filterwarnings("ignore", message=r"The cuda\.(cudart|nvrtc) module is deprecated", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quiet_vllm_optional_module_probe() -> None:
    """Fix vllm.utils.import_utils._has_module so missing parent packages (e.g. 'torchcodec'
    when probing 'torchcodec.decoders') return False quietly instead of logging a full
    ModuleNotFoundError traceback via logger.warning(..., exc_info=True)."""
    try:
        import vllm.utils.import_utils as vllm_iu
    except Exception:
        return
    if getattr(vllm_iu, "_quiet_has_module_patched", False):
        return

    @cache
    def _quiet_has_module(module_name: str) -> bool:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            return False
        except Exception:
            vllm_iu.logger.warning("Module %s was found but failed to import", module_name, exc_info=True)
            return False
        return True

    vllm_iu._has_module = _quiet_has_module
    vllm_iu._quiet_has_module_patched = True


def _install_qwen35_bi_longctx_patches() -> None:
    """Install BI=1 22k long-context memory patches into megatron.lite (0-diff preserved):
    1) Pass inplace_final_state=True to fused_sigmoid_gating_delta_rule_update in _BIFusedGdnFn
       so it does not allocate a [T, HV, V, K] (32 GiB @ 16k tokens) unused tensor per GDN layer.
    2) Use non-reentrant torch.utils.checkpoint.checkpoint in megatron.lite.primitive.recompute.wrap_checkpoint
       so recompute=full works cleanly with FSDP2 hooks and custom BI autograd.Functions.
    3) Chunk vocab projection + cross-entropy in 1024-token tiles when bi_mode=True so the
       [T, 248320] logits tensor (11 GiB @ 22k tokens) is never materialized all at once.
    """
    try:
        import megatron.lite.model.qwen3_5.lite.bi_ops as bi_ops
        import megatron.lite.primitive.recompute as recompute_mod
    except Exception:
        return

    if getattr(bi_ops, "_qwen35_bi_longctx_patched", False):
        return
    bi_ops._qwen35_bi_longctx_patched = True

    # 1. Patch fused_sigmoid_gating_delta_rule_update to avoid [T, HV, V, K] allocation
    orig_gdn_update = bi_ops.fused_sigmoid_gating_delta_rule_update

    def _patched_gdn_update(*args, **kwargs):
        kwargs["inplace_final_state"] = True
        return orig_gdn_update(*args, **kwargs)

    bi_ops.fused_sigmoid_gating_delta_rule_update = _patched_gdn_update

    # 2. Patch wrap_checkpoint to use non-reentrant checkpointing compatible with FSDP2 + BI ops
    def _non_reentrant_wrap_checkpoint(module: nn.Module, *, preserve_rng_state: bool = True) -> None:
        original_forward = module.forward
        _routers = [m for m in module.modules() if hasattr(m, "expert_bias")]

        def _checkpointed_forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original_forward(*args, **kwargs)
            if _routers:
                saved = [r.expert_bias.clone() for r in _routers]
                call_count = [0]

                def _fwd(*a, **kw):
                    call_count[0] += 1
                    if call_count[0] > 1:
                        for r, s in zip(_routers, saved, strict=False):
                            r.expert_bias.copy_(s)
                    return original_forward(*a, **kw)
            else:
                _fwd = original_forward

            return torch.utils.checkpoint.checkpoint(
                _fwd,
                *args,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
                **kwargs,
            )

        module.forward = _checkpointed_forward

    recompute_mod.wrap_checkpoint = _non_reentrant_wrap_checkpoint

    # 3. Chunked BI linear + vocab cross-entropy (bitwise identical to bi_linear + bi_vocab_cross_entropy)
    class _BIChunkedLinearVocabCEFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, temperature: float):
            orig_target_shape = target.shape
            h2 = hidden.reshape(-1, hidden.shape[-1])
            t1 = target.reshape(-1)
            w_local = weight.to_local() if hasattr(weight, "to_local") else weight
            T = h2.shape[0]
            logprobs = torch.empty((T,), dtype=torch.float32, device=h2.device)
            chunk = 1024
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                logits_c = F.linear(h2[start:end], w_local)
                if temperature != 1.0:
                    logits_c = logits_c / temperature
                lp_c = bi_ops.compute_token_logprobs(logits_c, t1[start:end].unsqueeze(-1)).squeeze(-1)
                logprobs[start:end].copy_(lp_c)
            ctx.save_for_backward(hidden, weight, target)
            ctx.temperature = temperature
            return (-logprobs).view(orig_target_shape)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            hidden, weight, target = ctx.saved_tensors
            temperature = ctx.temperature
            h2 = hidden.reshape(-1, hidden.shape[-1])
            t1 = target.reshape(-1)
            w_local = weight.to_local() if hasattr(weight, "to_local") else weight
            g1 = grad_out.reshape(-1, 1).float()
            if temperature != 1.0:
                g1 = g1 / temperature
            T = h2.shape[0]
            grad_h2 = torch.empty_like(h2)
            grad_w = torch.zeros_like(w_local, dtype=torch.float32)
            chunk = 1024
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                hc = h2[start:end]
                logits_c = F.linear(hc, w_local)
                if temperature != 1.0:
                    logits_c = logits_c / temperature
                sm = F.softmax(logits_c.float(), dim=-1)
                rows = torch.arange(end - start, device=hc.device)
                sm[rows, t1[start:end].long()] -= 1.0
                sm.mul_(g1[start:end])
                grad_logits_c = sm.to(hc.dtype)
                grad_h2[start:end].copy_(grad_logits_c @ w_local)
                grad_w.add_(grad_logits_c.t() @ hc)
            gw_out = grad_w.to(w_local.dtype)
            if hasattr(weight, "to_local"):
                from torch.distributed.tensor import DTensor

                gw_out = DTensor.from_local(gw_out, weight.device_mesh, weight.placements, run_check=False)
            return grad_h2.view_as(hidden), gw_out, None, None

    class _LazyVocabLogitsProxy:
        def __init__(self, hidden: torch.Tensor, weight: torch.Tensor, temp: float = 1.0):
            self.hidden = hidden
            self.weight = weight
            self.temp = temp

        def __truediv__(self, other: float | torch.Tensor):
            val = float(other.detach().item()) if isinstance(other, torch.Tensor) else float(other)
            return _LazyVocabLogitsProxy(self.hidden, self.weight, self.temp * val)

        def materialize(self) -> torch.Tensor:
            out = orig_bi_linear(self.hidden, self.weight)
            if self.temp != 1.0:
                out = out / self.temp
            return out

        def transpose(self, dim0: int, dim1: int) -> torch.Tensor:
            return self.materialize().transpose(dim0, dim1)

        @property
        def shape(self):
            return (*self.hidden.shape[:-1], self.weight.shape[0])

        @property
        def dtype(self):
            return self.weight.dtype

        @property
        def device(self):
            return self.hidden.device

        def __getattr__(self, name: str):
            # Unknown attribute (e.g. .float()/.detach()/.to()): fall back to the
            # materialized logits so unexpected call sites keep the pre-patch
            # behavior (memory-heavy but correct) instead of raising AttributeError.
            if name.startswith("__"):
                raise AttributeError(name)
            return getattr(self.materialize(), name)

        def entropy(self, tp_group=None) -> torch.Tensor:
            """Token-chunked mirror of vocab_parallel_entropy (bi_mode has tp_size == 1)."""
            h2 = self.hidden.reshape(-1, self.hidden.shape[-1])
            w_local = self.weight.to_local() if hasattr(self.weight, "to_local") else self.weight
            T = h2.shape[0]
            out = torch.empty((T,), dtype=torch.float32, device=h2.device)
            chunk = 1024
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                logits_c = F.linear(h2[start:end], w_local)
                if self.temp != 1.0:
                    logits_c = logits_c / self.temp
                logits_c = logits_c.float()
                logits_max = logits_c.max(dim=-1).values
                exp_logits = torch.exp(logits_c - logits_max.unsqueeze(-1))
                sum_exp = exp_logits.sum(dim=-1)
                weighted = (exp_logits * logits_c).sum(dim=-1)
                out[start:end] = torch.log(sum_exp) + logits_max - weighted / sum_exp
            return out.view(self.hidden.shape[:-1])

    orig_bi_linear: Callable = bi_ops.bi_linear
    orig_bi_ce: Callable = bi_ops.bi_vocab_cross_entropy

    def _hooked_bi_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
        if bias is None and weight.shape[0] >= 100000:
            return _LazyVocabLogitsProxy(x, weight, 1.0)
        return orig_bi_linear(x, weight, bias)

    def _hooked_bi_vocab_ce(logits, target: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, _LazyVocabLogitsProxy):
            return _BIChunkedLinearVocabCEFn.apply(logits.hidden, logits.weight, target, logits.temp)
        return orig_bi_ce(logits, target)

    bi_ops.bi_linear = _hooked_bi_linear
    bi_ops.bi_vocab_cross_entropy = _hooked_bi_vocab_ce

    # 4. Route vocab_parallel_entropy through the proxy's chunked entropy: the old-log-prob
    #    pass hardcodes calculate_entropy=True (actor/entropy metric), and the vanilla path
    #    would materialize logits.float() ([T, vocab] fp32, 22 GiB @ 22k tokens).
    try:
        import megatron.lite.primitive.ops.logprob as logprob_mod
    except Exception:
        logprob_mod = None
    if logprob_mod is not None and not getattr(logprob_mod, "_qwen35_bi_entropy_patched", False):
        logprob_mod._qwen35_bi_entropy_patched = True
        orig_vocab_parallel_entropy = logprob_mod.vocab_parallel_entropy

        def _patched_vocab_parallel_entropy(logits, tp_group=None):
            if isinstance(logits, _LazyVocabLogitsProxy):
                world_size = 1
                if tp_group is not None:
                    try:
                        import torch.distributed as dist

                        if dist.is_initialized():
                            world_size = dist.get_world_size(tp_group)
                    except Exception:
                        world_size = 1
                if world_size == 1:
                    return logits.entropy(tp_group)
                logits = logits.materialize()
            return orig_vocab_parallel_entropy(logits, tp_group)

        logprob_mod.vocab_parallel_entropy = _patched_vocab_parallel_entropy
        # model.py binds the name at import time ("from ...logprob import ...");
        # rebind it there too if the module is already loaded.
        import sys

        model_mod = sys.modules.get("megatron.lite.model.qwen3_5.lite.model")
        if model_mod is not None and getattr(model_mod, "vocab_parallel_entropy", None) is orig_vocab_parallel_entropy:
            model_mod.vocab_parallel_entropy = _patched_vocab_parallel_entropy


if os.environ.get("VERL_MLITE_SKIP_RUNTIME_PATCHES") != "1":
    _quiet_vllm_optional_module_probe()
    try:
        from verl_mlite.compat import apply_runtime_patches
    except ImportError:
        pass
    else:
        apply_runtime_patches()
        _install_qwen35_bi_longctx_patches()
