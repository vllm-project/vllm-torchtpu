"""Greedy draft tokens without gathering the full logits
(VLLM_TPU_DRAFT_SHARD_ARGMAX).

The draft's `compute_logits(h).argmax(-1)` all-gathers the [rows, vocab]
logits of the vocab-parallel lm head on every rank, once per draft step, only
to take one index per row. Here each rank takes the argmax of its own vocab
shard (padded vocab rows masked to -inf), and the ranks all-gather one
(value, global id) pair per row instead. The pick is bit-identical to the full
argmax: shards are contiguous, ascending vocab ranges in rank order, and both
argmaxes return the first maximum, so the lowest rank holding the row maximum
gives the lowest global index, exactly as the full argmax does. The logits
values are the same, because the per-shard projection is the one the full path
computes before its gather, and soft cap / scale are elementwise.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

from vllm_torchtpu import envs


def local_candidates(
    local_logits: torch.Tensor, tp_rank: int, org_vocab_size: int
) -> torch.Tensor:
    """[rows, 2] float32 (max value, global id) of one rank's vocab shard.

    Global ids at or past `org_vocab_size` (vocab padding) never win. Ids stay
    exact in float32: vocab sizes are far below 2**24.
    """
    shard = local_logits.shape[-1]
    ids = torch.arange(shard, device=local_logits.device) + tp_rank * shard
    logits = local_logits.float().masked_fill(ids >= org_vocab_size, float("-inf"))
    arg = logits.argmax(dim=-1, keepdim=True)
    val = torch.gather(logits, -1, arg)
    return torch.cat([val, (arg + tp_rank * shard).float()], dim=-1)


def combine_candidates(gathered: torch.Tensor) -> torch.Tensor:
    """int32 [rows] winning global id from the all-gathered [rows, 2 * tp]
    candidates (rank r at columns 2r, 2r + 1); first maximum = lowest rank."""
    vals, ids = gathered[..., 0::2], gathered[..., 1::2]
    best = vals.argmax(dim=-1, keepdim=True)
    return torch.gather(ids, -1, best).squeeze(-1).to(torch.int32)


def _usable(model: Any) -> bool:
    lp = getattr(model, "logits_processor", None)
    head = getattr(model, "lm_head", None)
    return (
        lp is not None
        and head is not None
        and getattr(head, "tp_size", 1) > 1
        and not getattr(lp, "logits_as_input", False)
        and getattr(lp, "use_all_gather", False)
        # Contiguous rank-ordered shards: no added-vocab interleave, and the
        # head is split over the TP group the all-gather below runs on.
        and getattr(head, "num_embeddings_padded", None)
        == getattr(head, "org_vocab_size_padded", None)
        and head.tp_size == get_tensor_model_parallel_world_size()
    )


def greedy_draft_token(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """int32 [rows] = `model.compute_logits(hidden).argmax(-1)`, bit-identical,
    gathering 2 floats per row and rank instead of the vocab-wide logits when
    VLLM_TPU_DRAFT_SHARD_ARGMAX is on."""
    if not (envs.VLLM_TPU_DRAFT_SHARD_ARGMAX and _usable(model)):
        return model.compute_logits(hidden).argmax(dim=-1).to(torch.int32)
    lp, head = model.logits_processor, model.lm_head
    local = lp._get_logits(hidden, head, None, skip_gather=True)
    if lp.soft_cap is not None:
        local = torch.tanh(local / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        local = local * lp.scale
    cand = local_candidates(local, head.tp_rank, lp.org_vocab_size)
    return combine_candidates(tensor_model_parallel_all_gather(cand, dim=-1))
