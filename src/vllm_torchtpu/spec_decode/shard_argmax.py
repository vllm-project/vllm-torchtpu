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

VLLM_TPU_DRAFT_LM_HEAD_FP8 (``attach_fp8_head``) trades that exactness for bytes:
the local logits come from an FP8 copy of the shard, so near-ties may pick another
token.
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


def attach_fp8_head(model: Any) -> bool:
    """VLLM_TPU_DRAFT_LM_HEAD_FP8: register an FP8 copy of this rank's lm_head
    shard on ``model`` (``draft_head_fp8`` [shard, hidden] float8_e4m3fn and
    ``draft_head_scale`` [shard] float32, one scale per vocab row), read by
    ``greedy_draft_token`` instead of the BF16 weight. Returns whether it did:
    the shard-argmax path must be usable, since the copy feeds the local
    candidates only, and the head must hold plain float weights (a quantized
    head's codes are not its values) and no bias."""
    if not (envs.VLLM_TPU_DRAFT_SHARD_ARGMAX and _usable(model)):
        return False
    weight = model.lm_head.weight
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if getattr(model.lm_head, "bias", None) is not None:
        return False  # the FP8 logits carry no bias term
    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    fp8, scale = quantize_tensor(weight.float(), torch.float8_e4m3fn, axis=-1)
    scale = scale.reshape(-1)
    model.register_buffer("draft_head_fp8", fp8, persistent=False)
    model.register_buffer("draft_head_scale", scale, persistent=False)
    return True


def _local_logits(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """This rank's vocab-shard logits, from the FP8 copy when one is attached."""
    lp, head = model.logits_processor, model.lm_head
    fp8 = getattr(model, "draft_head_fp8", None)
    if fp8 is None:
        return lp._get_logits(hidden, head, None, skip_gather=True)
    logits = torch.matmul(hidden, fp8.to(hidden.dtype).t()).float()
    return logits * model.draft_head_scale


def greedy_draft_token(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """int32 [rows] = `model.compute_logits(hidden).argmax(-1)`, gathering 2
    floats per row and rank instead of the vocab-wide logits when
    VLLM_TPU_DRAFT_SHARD_ARGMAX is on. Bit-identical unless ``attach_fp8_head``
    attached an FP8 head, whose logits may pick another token at near-ties."""
    if not (envs.VLLM_TPU_DRAFT_SHARD_ARGMAX and _usable(model)):
        return model.compute_logits(hidden).argmax(dim=-1).to(torch.int32)
    lp, head = model.logits_processor, model.lm_head
    local = _local_logits(model, hidden)
    if lp.soft_cap is not None:
        local = torch.tanh(local / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        local = local * lp.scale
    cand = local_candidates(local, head.tp_rank, lp.org_vocab_size)
    return combine_candidates(tensor_model_parallel_all_gather(cand, dim=-1))
