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

import math
from collections.abc import Iterable

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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Each rank's shard stays a multiple of a sublane tile.
ROW_ALIGNMENT = 8
# Decode buckets (the megakernel's, <= 64 rows) must never take this path.
MIN_TOKENS_FLOOR = 1024


def enabled(tp_size: int) -> bool:
    """Whether sequence-parallel prefill is on for this TP size."""
    return bool(envs.QWEN4_EXP_PREFILL_SP) and tp_size > 1


def shard_rows(hc_tiles: Iterable[int]) -> int:
    """Row multiple of a shard at which every HC prefill kernel takes its
    Pallas path.

    ``GatedResidual`` picks a kernel or its XLA fallback by whether the row
    count divides the kernel's tile (twice the tile for the full mix). Only
    shards that divide all of them keep the sequence-parallel graph one
    structure across its buckets, so it traces once.
    """
    return math.lcm(ROW_ALIGNMENT, *(tile for tile in hc_tiles if tile))


def pad_hc_enabled() -> bool:
    """QWEN4_EXP_PREFILL_SP_PAD_HC, honoured only with QWEN4_EXP_PREFILL_SP_MOE.

    Padded shards feeding the replicated MoE lowered real-AL to 3.09-3.11
    against 3.24 with GSM8K held, so the pair stays apart until understood.
    Localized so far: it takes a padded bucket (3072 at TP8) together with
    the replicated MoE on the gathered rows; tracing the graph at 2048
    instead of 1024 keeps it (s04, AL 3.109). Suspect a token-count choice
    the replicated MoE makes at trace time without a guard.
    """
    if not envs.QWEN4_EXP_PREFILL_SP_PAD_HC:
        return False
    if not envs.QWEN4_EXP_PREFILL_SP_MOE:
        logger.warning(
            "QWEN4_EXP_PREFILL_SP_PAD_HC needs QWEN4_EXP_PREFILL_SP_MOE=1; ignoring it."
        )
        return False
    return True


def padded_rows(rows, multiple: int):
    """``rows`` rounded up to ``multiple``. Shape arithmetic rather than a
    branch, so one graph serves every bucket."""
    return (rows + multiple - 1) // multiple * multiple


def pad_shard(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """A shard with zero rows appended up to a multiple of ``multiple``.

    With QWEN4_EXP_PREFILL_SP_PAD_HC the sequence-parallel graph keeps the HC
    streams at this padded length, so every HC kernel takes its Pallas path
    for any shard. The padding rows stay zero (or False) and are cut off with
    :func:`unpad` before any collective or token-indexed op.
    """
    if multiple <= 1:
        return x
    rows = x.shape[0]
    pad = padded_rows(rows, multiple) - rows
    return torch.cat([x, x.new_zeros((pad, *x.shape[1:]))])


def unpad(x: torch.Tensor, rows) -> torch.Tensor:
    """The shard's real rows of a padded tensor."""
    return x.narrow(0, 0, rows)


def sequence_parallel_eligible(
    num_tokens: int, tp_size: int, rows: int = ROW_ALIGNMENT
) -> bool:
    """Whether a bucket of ``num_tokens`` tokens takes the sequence-parallel
    graph, each rank's shard a multiple of ``rows``. The router, both graphs'
    bucket filters and the warmup retraces use this one rule, so every bucket
    is compiled by exactly one graph."""
    if not enabled(tp_size):
        return False
    return num_tokens >= min_tokens(tp_size, rows) and num_tokens % (tp_size * rows) == 0


def min_tokens(tp_size: int, rows: int = ROW_ALIGNMENT) -> int:
    """Smallest bucket of the sequence-parallel graph."""
    return max(int(envs.QWEN4_EXP_PREFILL_SP_MIN_TOKENS), MIN_TOKENS_FLOOR, tp_size * rows)


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
