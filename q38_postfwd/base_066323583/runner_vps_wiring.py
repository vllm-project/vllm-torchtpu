# SPDX-License-Identifier: Apache-2.0
"""VPS wiring (q38mm DSTOCK): the vocabulary-parallel verify sampler and draft argmax. Default off.

Flags (PD stack: registered in envs.py; VERIFY / FALLBACK_ROWS / TIEGAP_SLOTS / DRAWS key the compile cache, CHECK /
CHECK_FAULT / STATS_EVERY are host-only and in tpu_compiler._TPU_COMPILE_ENV_IGNORED):
  VLLM_TPU_VPS_VERIFY=1        spec-verify chunks of eligible steps keep the rank's vocab shard of the target logits
                               (no full-logits all-gather) and run layers/adapter/sample/vp_sampler.vp_verify_from_local
  VLLM_TPU_DRAFT_TOP_TOKENS=1  the MTP draft's greedy token comes from LogitsProcessor.get_top_tokens (local argmax and
                               a (value, id) exchange) instead of argmax over all-gathered logits
  VLLM_TPU_VPS_CHECK=N         the first N VPS chunks also run the stock fused verify on all-gathered logits and count
                               mismatching token positions on device (logged as VPS_CHECK)
  VLLM_TPU_VPS_CHECK_FAULT=M   the first M checks perturb the compared copy (a positive control that must be caught)
  PD DEC-FB-2 (PR 2821 adoption item 1; vp_sampler.vp_verify_fb):
  VLLM_TPU_VPS_FALLBACK_ROWS=F the per-row full-vocabulary fallback, at most F rows per chunk (0/unset: off). Chunks
                               whose non-greedy requests keep top_k outside [1, 32] (unset or larger) then also run
                               vocabulary-parallel, with those rows (and exact-tie rows) through the fallback; a chunk
                               that would need more than F rows keeps today's path
  VLLM_TPU_VPS_TIEGAP_SLOTS=S  reserve S fallback rows on every VPS chunk for exact-tie rows (default 0: tie rows use
                               spare rows only; uncovered ones are counted and logged as VPS_FALLBACK)
  VLLM_TPU_VPS_DRAWS=candidate per-candidate uniforms (VPB): no full-vocabulary draws; same distribution, different
                               tokens than stock (VPS_CHECK is then skipped). Default "full" (VPBX, token-identical)

Eligibility (else the step keeps today's path, bit for bit):
  verify: a TP LM head with scale 1, no soft cap, no vocab padding, a shard width that is a whole number of
          vp_sampler.VP_BLOCK blocks (>= VP_KC of them); a non-greedy step without logprobs or structured outputs; a
          chunk with spec-verify metadata whose requests all take the top-k candidate path (1 <= top_k <= 32).
  draft:  a draft model that opts in with `plain_compute_logits` (compute_logits is exactly lm head + logits
          processor) and has no draft->target id map.
Exactness: the draft argmax is exact (first-occurrence rule, float32 exchange of bf16 values and ids < 2**24). The
verify is token-identical to the stock verify given the same uniforms, except the vp_sampler module's documented
float-sum caveats (q38mm VPS-mb1 j-4f37738e: 0 mismatches in 1,440 slots on device).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

VPS_VERIFY_ENV = "VLLM_TPU_VPS_VERIFY"
DRAFT_TOP_TOKENS_ENV = "VLLM_TPU_DRAFT_TOP_TOKENS"
VPS_CHECK_ENV = "VLLM_TPU_VPS_CHECK"
VPS_CHECK_FAULT_ENV = "VLLM_TPU_VPS_CHECK_FAULT"
FALLBACK_ROWS_ENV = "VLLM_TPU_VPS_FALLBACK_ROWS"
TIEGAP_SLOTS_ENV = "VLLM_TPU_VPS_TIEGAP_SLOTS"
DRAWS_ENV = "VLLM_TPU_VPS_DRAWS"
HASHED_ENVS = (VPS_VERIFY_ENV, FALLBACK_ROWS_ENV, TIEGAP_SLOTS_ENV, DRAWS_ENV)   # registered in envs.py, hashed
IGNORED_ENVS = (VPS_CHECK_ENV, VPS_CHECK_FAULT_ENV, "VLLM_TPU_VPS_STATS_EVERY")    # registered, host-only
FALLBACK_BUCKETS = (4, 8, 16, 32, 64)



def _envs():
    # PD stack: the flags are registered in envs.py (INT-3 compile-key rule).
    from vllm_torchtpu import envs

    return envs


def verify_enabled() -> bool:
    return bool(_envs().VLLM_TPU_VPS_VERIFY)


def draft_top_tokens_enabled() -> bool:
    # PD port: the draft top-tokens sites are not ported (DRAFT_TOP_TOKENS unused).
    return False


def check_counts() -> tuple[int, int]:
    e = _envs()
    return int(e.VLLM_TPU_VPS_CHECK), int(e.VLLM_TPU_VPS_CHECK_FAULT)


STATS_EVERY_ENV = "VLLM_TPU_VPS_STATS_EVERY"


def stats_every() -> int:
    """VPS_FALLBACK log period in VPS chunks (default 1000; 0 = never). Host-only (ignored in the compile key)."""
    return int(_envs().VLLM_TPU_VPS_STATS_EVERY)


def fallback_max() -> int:
    return int(_envs().VLLM_TPU_VPS_FALLBACK_ROWS)


def tiegap_slots() -> int:
    return int(_envs().VLLM_TPU_VPS_TIEGAP_SLOTS)


def cand_draws() -> bool:
    v = str(_envs().VLLM_TPU_VPS_DRAWS).strip().lower()
    if v in ("", "full", "vpbx"):
        return False
    if v in ("candidate", "vpb"):
        return True
    raise ValueError(f"Invalid value '{v}' for {DRAWS_ENV}: use full or candidate")


def fallback_path() -> bool:
    """Whether VPS chunks run vp_verify_fb (the fallback, its counters, or per-candidate draws)."""
    return fallback_max() > 0 or tiegap_slots() > 0 or cand_draws()


def range_rows(top_k: torch.Tensor, temperatures: torch.Tensor, vocab_size: int, kc: int, rows_per_req: int) -> int:
    """Upper bound on the chunk's fallback rows for unset / large top_k (host side, the active requests' CPU params):
    non-greedy requests (temperature != 0) whose top_k, padded as `_padded_sampling_params` does (>= vocab -> 0), is
    outside [1, kc], times the verify rows per request (bonus + drafts)."""
    k = torch.where(top_k >= vocab_size, torch.zeros_like(top_k), top_k)
    live = temperatures.to(torch.float32) != 0
    return int((live & ~((k >= 1) & (k <= kc))).sum()) * rows_per_req


def fallback_slots(need_rows: int, tiegap: int, max_rows: int) -> int | None:
    """The chunk's static fallback row count F: 0 when nothing needs it, the smallest bucket that fits (capped at
    max_rows), or None when the chunk needs more than max_rows (it keeps today's path)."""
    need = need_rows + tiegap
    if need == 0:
        return 0
    if need > max_rows:
        return None
    for b in FALLBACK_BUCKETS:
        if need <= b <= max_rows:
            return b
    return max_rows


def fallback_variants(max_rows: int, tiegap: int) -> list[int]:
    """Every F `fallback_slots` can return (for warm-up)."""
    out = {fallback_slots(0, tiegap, max(max_rows, tiegap))}
    for need in range(1, max_rows + 1):
        out.add(fallback_slots(need, tiegap, max_rows))
    return sorted(v for v in out if v is not None)


@dataclass(frozen=True)
class VpsHead:
    """The target LM head as the verify needs it, resolved once at model load."""

    processor: Any
    head: Any
    shard_offset: int
    shard_width: int
    full_vocab: int
    tp_size: int


def _find_head(model: torch.nn.Module):
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        if hasattr(processor, "get_top_tokens") and hasattr(head, "shard_indices"):
            return processor, head
    return None, None


def resolve_vps_head(
    model: torch.nn.Module, block: int, kc: int, dp_lockstep: bool = False
) -> tuple[VpsHead | None, str]:
    """(VpsHead, "") when the model's verify can run vocabulary-parallel, else (None, reason).

    PD: refused under expert-parallel DP lockstep with a TP head, as TPU_GREEDY_TOP_TOKENS is
    (greedy_top_tokens.resolve_head_owner): an idle engine issues the full-logits collective in lockstep with its
    busy peers. The fallback needs kc + 1 whole blocks per shard (the sentinel)."""
    if dp_lockstep:
        return None, "expert-parallel DP lockstep (an idle engine issues the full-logits collective)"
    if not getattr(type(model), "plain_compute_logits", False) and getattr(
        getattr(type(model), "compute_logits", None), "__module__", ""
    ).startswith("vllm_torchtpu."):
        return None, "the model's compute_logits is TPU-specific and does not opt in with plain_compute_logits"
    processor, head = _find_head(model)
    if processor is None:
        return None, "no module with an lm_head and a logits_processor"
    if getattr(processor, "logits_as_input", False):
        return None, "logits_as_input"
    if getattr(processor, "soft_cap", None) is not None or getattr(processor, "scale", 1.0) != 1.0:
        return None, "the logits processor applies a soft cap or a scale"
    si = head.shard_indices
    tp = int(getattr(head, "tp_size", 1))
    width = int(si.org_vocab_end_index - si.org_vocab_start_index)
    if tp < 2:
        return None, "the LM head is not tensor-parallel"
    if int(getattr(si, "num_org_vocab_padding", 0)) != 0:
        return None, "the vocab shard carries padding"
    if width % block or width // block < kc + 1:
        return None, f"shard width {width} is not >= {kc + 1} whole blocks of {block}"
    return VpsHead(processor, head, int(si.org_vocab_start_index), width, int(processor.org_vocab_size), tp), ""


def step_eligible(input_batch: Any, requests: Mapping[str, Any]) -> bool:
    """A step whose verify chunks may run vocabulary-parallel: non-greedy (the fused verify's branch), no logprobs,
    no structured outputs (those need the full logits)."""
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            return False
    return True


def local_logits(vh: VpsHead, hidden: torch.Tensor) -> torch.Tensor:
    """The rank's vocab shard of the target logits, [rows, shard_width] (no all-gather)."""
    return vh.processor._get_logits(hidden, vh.head, None, skip_gather=True)


def draft_top_tokens_ok(draft_model: Any) -> bool:
    """The draft's greedy token may come from get_top_tokens (exact; see the module docstring)."""
    if draft_model is None or getattr(draft_model, "draft_id_to_target_id", None) is not None:
        return False
    if not getattr(type(draft_model), "plain_compute_logits", False):
        return False
    processor, head = getattr(draft_model, "logits_processor", None), getattr(draft_model, "lm_head", None)
    return hasattr(processor, "get_top_tokens") and hasattr(head, "shard_indices")


def draft_top_tokens(draft_model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """The draft's greedy token per row, int32, through LogitsProcessor.get_top_tokens."""
    return draft_model.logits_processor.get_top_tokens(draft_model.lm_head, hidden).to(torch.int32)
