# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy selection with rank-invariant TPU graphs."""

import torch
from vllm.distributed import tensor_model_parallel_all_gather


def local_argmax_pair(
    logits: torch.Tensor, vocab_start: torch.Tensor, vocab_end: torch.Tensor
) -> torch.Tensor:
    """Return (value, global ID); IDs must be exactly representable in fp32.

    Index arithmetic stays in the bounds' dtype (int32 on TPU), which holds
    every such ID exactly.
    """
    ids = (
        torch.arange(logits.shape[-1], device=logits.device, dtype=vocab_start.dtype)
        + vocab_start
    )
    logits = torch.where(ids < vocab_end, logits, -float("inf"))
    values, indices = logits.max(dim=-1)
    global_ids = indices.to(vocab_start.dtype) + vocab_start
    return torch.stack((values.float(), global_ids.float()), dim=-1)


def select_argmax_pair(pairs: torch.Tensor) -> torch.Tensor:
    """Select the first rank on ties, matching concatenated-vocabulary argmax."""
    rank = pairs[..., 0].argmax(dim=-1, keepdim=True)
    return pairs[..., 1].gather(-1, rank).squeeze(-1).to(torch.int64)


def vocab_parallel_argmax(
    logits: torch.Tensor,
    vocab_start: torch.Tensor,
    vocab_end: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Adapt vLLM LogitsProcessor.get_top_tokens to tensor shard boundaries.

    Rank-dependent Python bounds specialize TPU graphs differently. Existing
    embedding boundary buffers keep the projection/collective graph identical.
    The caller excludes added vocabulary and vocabulary IDs above 2**24.
    """
    pairs = local_argmax_pair(logits, vocab_start, vocab_end)
    if tp_size > 1:
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    return select_argmax_pair(pairs.reshape(*logits.shape[:-1], tp_size, 2))
