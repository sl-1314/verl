# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefill-recompute rollout (PCR) for zero-diff colocated training.

Each trainer DP rank constructs an IN-PROCESS vLLM engine (TP1) inside its
own worker process; the engines across ranks form one SPMD EP group via the
VLLM_DP_* env (offline SPMD mode, no server actor). Generation runs the PCR
loop proven by C9-E E3a/E3c: every step, each rank forwards the FULL current
prefix of ONE sequence as a prefill, samples from the last row, and appends.
All ranks enter each step together via a torch.distributed barrier on the
trainer group, so the EP all-to-all always sees one token-bearing sequence
per rank (the idle-rank hazard C9-D G2 worked around with file barriers).

With the vllm BI norm pin (qwen35-0diff-bi-gdn), every forward length
produces bitwise-identical per-row outputs (E3c-1: n_distinct=1 across all
L), so the per-step logprob from the last row equals the full-sequence
re-score row AND the actor's recompute (G2 anchor) -- the C9-E
"rollout logprob == training recompute" requirement holds by construction.

The weight-update / sleep / wake surface reuses the colocate machinery:
the verl vLLMColocateWorkerExtension receives buckets from a
BucketedWeightSender over ZMQ exactly like the server route, but both ends
live in this process tree (sender here, receiver in the EngineCore child).
"""

import asyncio
import logging
import os
import time
from typing import Generator, Optional

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

WORKER_EXTENSION_CLS = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"


def _dp_master_port() -> int:
    """Pick the vLLM DP-store port on rank 0 and broadcast it over the
    trainer group (already initialized when the rollout is constructed)."""
    import socket

    if not dist.is_initialized():
        # Fallback for rollout-only workers without a trainer group: a fixed
        # high port (all ranks use the same value).
        return 37777
    # NCCL groups require CUDA tensors; the worker has already set the
    # current device to its local rank.
    device = "cuda" if (torch.cuda.is_available() and dist.get_backend() == "nccl") else "cpu"
    port = torch.zeros(1, dtype=torch.int64, device=device)
    if dist.get_rank() == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port[0] = s.getsockname()[1]
    dist.broadcast(port, src=0)
    return int(port.item())


class PCRRollout(BaseRollout):
    """SPMD in-process prefill-recompute rollout (rollout.mode = "pcr")."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        *args,
        **kwargs,
    ):
        super().__init__(config, model_config, device_mesh)
        # Overrides get lost through OmegaConf serialization; re-read them.
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        self.rank = rank
        self.world = world
        self.dp_size = world  # one TP1 engine per trainer rank

        # vLLM SPMD env must be set before LLM construction; the engines pick
        # their EP group up from these.
        port = _dp_master_port()
        os.environ["VLLM_DP_RANK"] = str(rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
        os.environ["VLLM_DP_SIZE"] = str(self.dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = str(port)
        logger.info("PCRRollout rank=%d world=%d dp_master_port=%d", rank, world, port)

        self._llm = None  # lazy: built on first use, after actor memory freed
        self._tokenizer = None

        # Stop tokens: prefer the model dir's generation_config.json (the same
        # source vLLM's sampler uses for stop_token_ids), fall back to
        # hf_config.eos_token_id.
        self.eos_ids = set()
        try:
            import json

            model_path = self.model_config.local_path or self.model_config.path
            with open(os.path.join(model_path, "generation_config.json")) as f:
                gen_cfg = json.load(f)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, (list, tuple)):
                self.eos_ids = {int(e) for e in eos}
            elif eos is not None:
                self.eos_ids = {int(eos)}
        except (OSError, ValueError):
            pass
        if not self.eos_ids:
            eos = getattr(self.model_config.hf_config, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                self.eos_ids = {int(e) for e in eos}
            elif eos is not None:
                self.eos_ids = {int(eos)}

        # ZMQ handle must match the worker-extension receiver's
        # _get_zmq_handle (Ray job id + replica 0 + trainer local rank).
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-0-rank-{local_rank}.sock"
        self._sender_inited = False
        self._released = False  # True after release() (sleep 2); reset on resume/wake

    # ---------------------------------------------------------------- engine
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_path = self.model_config.local_path or self.model_config.path
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self._tokenizer

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        model_path = self.model_config.local_path or self.model_config.path
        # engine_kwargs may be keyed per engine ({"vllm": {...}}) like the
        # server path; tolerate both shapes and drop None values.
        raw_kwargs = dict(self.config.engine_kwargs or {})
        engine_kwargs = raw_kwargs.get("vllm", raw_kwargs) if "vllm" in raw_kwargs else raw_kwargs
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=1,            # one engine per trainer rank
            enable_expert_parallel=True,       # engines form one EP group
            max_model_len=self.config.max_model_len or 4096,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=self.config.enforce_eager,
            enable_prefix_caching=False,       # full re-prefill semantics
            max_num_batched_tokens=self.config.max_num_batched_tokens or 8192,
            worker_extension_cls=WORKER_EXTENSION_CLS,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
            **{k: v for k, v in engine_kwargs.items() if v is not None},
        )
        logger.info("PCRRollout rank=%d LLM loaded", self.rank)
        return self._llm

    # ------------------------------------------------------------- interface
    async def resume(self, tags: list[str]):
        if self.config.free_cache_engine and self._llm is not None:
            await asyncio.to_thread(self._llm.wake_up, tags=tags or ["weights", "kv_cache"])
            self._released = False

    async def release(self):
        if self.config.free_cache_engine and self._llm is not None:
            await asyncio.to_thread(self._llm.sleep, level=2)
            self._released = True

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        assert wire_format == "named_tensors", "PCR rollout supports named_tensors only"
        llm = self._ensure_llm()
        start = time.time()

        # update_weights_from_ipc(peft_config, base_sync_done, use_shm) -- forward
        # only the kwargs it accepts; the worker passes global_steps and friends.
        _rpc_kwargs = {
            "peft_config": kwargs.get("peft_config"),
            "base_sync_done": kwargs.get("base_sync_done", True),
            "use_shm": False,
        }
        llm.collective_rpc("update_weights_from_ipc", kwargs=_rpc_kwargs)
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
            use_shm=False,
        )
        await sender.async_send_weights(weights)
        if self.rank == 0:
            logger.info("PCRRollout update_weights done in %.2fs", time.time() - start)

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        llm = self._ensure_llm()
        # The worker's naive weight-sync path leaves the engine AWAKE after
        # update_weights (resume weights + kv_cache); release() (sleep 2) is
        # called when generation ends so the actor update phase has the GPU.
        if self.config.free_cache_engine and self._released:
            llm.wake_up(tags=["weights", "kv_cache"])
            self._released = False
        cfg = self.config
        plen_cfg, rlen_cfg = int(cfg.prompt_length), int(cfg.response_length)
        validate = bool(prompts.meta_info.get("validate", False))
        temperature = prompts.meta_info.get("temperature", cfg.temperature)
        if validate:
            temperature = cfg.val_kwargs.temperature
        do_sample = getattr(cfg, "do_sample", True)
        if not do_sample or temperature == 0.0:
            temperature, top_p, top_k = 0.0, 1.0, -1
        else:
            top_p, top_k = cfg.top_p, cfg.top_k

        sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                            max_tokens=1, logprobs=1)
        ignore_eos = bool(cfg.ignore_eos)

        input_ids = prompts.batch.get("input_ids") if prompts.batch is not None else None
        attention_mask = prompts.batch.get("attention_mask") if input_ids is not None else None
        n_rows = len(prompts)

        # Extract unpadded prompt ids per row. Two data shapes reach the
        # rollout: a tensor batch (input_ids/attention_mask, older flow) or
        # the agent-loop flow where the prompt lives in non_tensor_batch
        # "raw_prompt" as chat messages and must be tokenized here (same
        # recipe as E2: apply_chat_template(tokenize=False) + explicit
        # tokenizer call -- the new HF tokenize=True returns a BatchEncoding).
        seqs = []
        for i in range(n_rows):
            if input_ids is not None:
                ids = input_ids[i]
                if attention_mask is not None:
                    ids = ids[attention_mask[i].bool()]
                seqs.append([int(t) for t in ids.tolist()])
            else:
                messages = prompts.non_tensor_batch["raw_prompt"][i]
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                ids = list(self.tokenizer(text)["input_ids"])
                # keep the tail (left-padded convention) so PCR logprobs are
                # computed on exactly the tokens the output batch carries
                if len(ids) > plen_cfg:
                    ids = ids[-plen_cfg:]
                seqs.append(ids)

        # ---- PCR loop: every step, ONE sequence per rank per barriered round
        # (the batch shape E3a/E3c anchored: per-rank single-sequence prefill
        # with all ranks stepping in lockstep).
        outs = [[None] * rlen_cfg for _ in range(n_rows)]     # token ids
        lps = [[None] * rlen_cfg for _ in range(n_rows)]      # logprobs
        alive = [True] * n_rows
        for t in range(rlen_cfg):
            for j in range(n_rows):
                if dist.is_initialized():
                    dist.barrier()
                if not alive[j]:
                    continue
                o = llm.generate([TokensPrompt(prompt_token_ids=seqs[j])], sp,
                                 use_tqdm=False)
                resp = o[0].outputs[0]
                tid = int(resp.token_ids[0])
                lp = resp.logprobs[0].get(tid)
                outs[j][t] = tid
                lps[j][t] = None if lp is None else float(lp.logprob)
                seqs[j].append(tid)
                if not ignore_eos and tid in self.eos_ids:
                    alive[j] = False

        # ---- assemble the per-rank DataProto (fixed widths for dispatch
        # gather; padding semantics mirror the agent-loop manager).
        pad_id = 0
        out = DataProto()
        prompts_t = torch.full((n_rows, plen_cfg), pad_id, dtype=torch.int64)
        responses_t = torch.full((n_rows, rlen_cfg), pad_id, dtype=torch.int64)
        response_mask = torch.zeros((n_rows, rlen_cfg), dtype=torch.int64)
        rollout_lp = torch.zeros((n_rows, rlen_cfg), dtype=torch.float32)
        for j in range(n_rows):
            p, r = seqs[j][:plen_cfg], outs[j]
            prompts_t[j, plen_cfg - len(p):] = torch.tensor(p, dtype=torch.int64)
            # tokens stop at EOS (alive[j] flips); trailing entries stay None
            r_valid = [x for x in r if x is not None]
            k = min(len(r_valid), rlen_cfg)
            responses_t[j, :k] = torch.tensor(r_valid[:k], dtype=torch.int64)
            response_mask[j, :k] = 1
            for t in range(k):
                if lps[j][t] is not None:
                    rollout_lp[j, t] = lps[j][t]
        input_ids_t = torch.cat([prompts_t, responses_t], dim=1)
        attention_t = torch.cat(
            [torch.zeros_like(prompts_t), response_mask], dim=1)
        # prompt part of attention: 1 where a real prompt token sits
        for j in range(n_rows):
            p = seqs[j][:plen_cfg]
            attention_t[j, plen_cfg - len(p):plen_cfg] = 1
        position_ids = attention_t.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_min(0)

        out.batch = TensorDict(
            {
                "prompts": prompts_t,
                "responses": responses_t,
                "response_mask": response_mask,
                "input_ids": input_ids_t,
                "attention_mask": attention_t,
                "position_ids": position_ids,
                "rollout_log_probs": rollout_lp,
            },
            batch_size=[n_rows],
        )
        out.non_tensor_batch = {
            "__num_turns__": np.full(n_rows, 2, dtype=np.int32),
        }
        # pass the shard's non-tensor fields through (uid, data_source, ...)
        for k, v in prompts.non_tensor_batch.items():
            out.non_tensor_batch[k] = v
        out.meta_info = {
            "temperature": temperature,
            "validate": validate,
            "timing": {"gen": float(time.time())},
        }

        # Hand the GPU back to the actor before its compute phases; the naive
        # weight-sync path wakes the engine again after update_actor.
        if self.config.free_cache_engine:
            llm.sleep(level=2)
            self._released = True
        return out
