# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill for the qwen4_exp backbone.

At prefill sizes every TP rank otherwise repeats the row-wise work between
two attention blocks on all tokens: both hyper-connection mixes and combines
over the [tokens, hc * hidden] streams, the PLE gate and the padding masks.
With QWEN4_EXP_PREFILL_SP=1 a step of at least QWEN4_EXP_PREFILL_SP_MIN_TOKENS
tokens keeps the streams sharded by token across the TP ranks instead:

- the attention block input is all-gathered right before the TP projections,
- the attention output is reduce-scattered instead of all-reduced, and
- the MoE runs on the gathered block (its EP kernel takes replicated rows)
  and each rank keeps its own rows of the reduced output.

Rank r owns rows [r * T / tp, (r + 1) * T / tp), the order the reduce-scatter
produces and the all-gather consumes. Slicing goes through vLLM's
``sequence_parallel_chunk`` custom op so every rank traces the same graph.

The sequence-parallel backbone is a compiled graph of its own
(``Qwen4ExpBackboneSP``): ``Qwen4ExpForCausalLM.forward`` routes each step on
its padded bucket size, outside both graphs, and each graph compiles only its
own buckets (compilation/bucket_filter). Neither branches on the token count.
Only the backbone takes this path; the MTP draft keeps the replicated one.
"""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_torchtpu import envs

# Each rank's shard stays a multiple of a sublane tile.
ROW_ALIGNMENT = 8
# Decode buckets (the megakernel's, <= 64 rows) must never take this path.
MIN_TOKENS_FLOOR = 1024


def enabled(tp_size: int) -> bool:
    """Whether sequence-parallel prefill is on for this TP size."""
    return bool(envs.QWEN4_EXP_PREFILL_SP) and tp_size > 1


def sequence_parallel_eligible(num_tokens: int, tp_size: int) -> bool:
    """Whether a bucket of ``num_tokens`` tokens takes the sequence-parallel
    graph. The router and both graphs' bucket filters use this one rule, so
    every bucket is compiled by exactly one graph."""
    if not enabled(tp_size):
        return False
    minimum = max(
        int(envs.QWEN4_EXP_PREFILL_SP_MIN_TOKENS),
        MIN_TOKENS_FLOOR,
        tp_size * ROW_ALIGNMENT,
    )
    return num_tokens >= minimum and num_tokens % (tp_size * ROW_ALIGNMENT) == 0


def tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tensor_model_parallel_world_size()


def check_divisible(num_tokens) -> None:
    """State that the token count divides by TP so shard shapes stay exact."""
    tp_size = tp_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def shard(x: torch.Tensor) -> torch.Tensor:
    """This rank's contiguous rows of a replicated [tokens, ...] tensor."""
    return sequence_parallel_chunk(x)


def gather(x: torch.Tensor, num_tokens) -> torch.Tensor:
    """All ranks' rows of a sharded tensor, in token order."""
    return tensor_model_parallel_all_gather(x, 0).narrow(0, 0, num_tokens)


def reduce(x: torch.Tensor, sequence_parallel: bool) -> torch.Tensor:
    """Sum TP partial outputs: this rank's rows, or all rows on the TP path."""
    if tp_world_size() <= 1:
        return x
    if sequence_parallel:
        return tensor_model_parallel_reduce_scatter(x, 0)
    return tensor_model_parallel_all_reduce(x)
