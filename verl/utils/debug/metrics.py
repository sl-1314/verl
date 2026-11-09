# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


_E5_DUMP_SEQ = 0


def _maybe_dump_e5(data: DataProto) -> None:
    """Bisect aid for the online 0-diff investigation (env-gated, default off).

    Persists the full per-token logprob tensors from both sides of the online
    kl (engine ``rollout_log_probs`` vs actor-recomputed ``old_log_probs``)
    plus the token ids and masks, so the *structure* of the divergence (which
    positions / sequences / tokens diverge) can be analyzed offline.
    """
    import os

    root = os.environ.get("VERL_E5_DUMP")
    if not root:
        return
    global _E5_DUMP_SEQ
    _E5_DUMP_SEQ += 1
    try:
        payload = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, v in data.batch.items()
            if torch.is_tensor(v) and k
            in (
                "rollout_log_probs",
                "old_log_probs",
                "response_mask",
                "attention_mask",
                "loss_mask",
                "responses",
                "input_ids",
                "prompts",
                "response_ids",
            )
        }
        os.makedirs(root, exist_ok=True)
        path = f"{root}/e5_step{_E5_DUMP_SEQ}_pid{os.getpid()}.pt"
        torch.save(payload, path)
        logger.info("E5 dump written: %s (%d tensors)", path, len(payload))
    except Exception as e:  # noqa: BLE001 - diagnostics must never break training
        logger.warning("E5 dump failed: %s", e)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    _maybe_dump_e5(data)
    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)

    # ---- live rollout-vs-actor verdict (debug): print immediately, don't
    # wait for the step to end. Per-sequence bitwise + first-divergence info.
    try:
        lp_diff = (rollout_old_log_probs - actor_old_log_probs).abs()
        _m = response_mask_bool
        _lines = [
            "[rollout_corr live] overall: "
            f"valid={int(_m.sum())} "
            f"|lp diff| max/mean={lp_diff[_m].max().item():.4e}/{lp_diff[_m].mean().item():.4e} "
            f"bitwise={int((lp_diff[_m] == 0).sum())}/{int(_m.sum())} "
            f"corr={pearson_corrcoef:.6f}"
        ]
        for _i in range(lp_diff.shape[0]):
            _mi = _m[_i]
            if not bool(_mi.any()):
                continue
            _d = lp_diff[_i][_mi]
            _nz = (_d > 0).nonzero()
            _first = int(_nz[0].item()) if len(_nz) else -1
            _bw = int((_d == 0).sum())
            _lines.append(
                f"[rollout_corr live]   seq{_i}: valid={int(_mi.sum())} "
                f"bitwise={_bw}/{int(_mi.sum())} max={_d.max().item():.4e} "
                f"first_nonzero_resp_pos={_first}"
            )
        print("\n".join(_lines), flush=True)
    except Exception as _e:  # noqa: BLE001 - debug print must never break training
        logger.warning(f"live rollout_corr print failed: {_e}")

    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
