# Copyright 2026 Google LLC
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
"""Torch bridge for the streamed LM-head kernel in ``kernels/streamed_logits.py``."""

from typing import Any

import jax
import torch
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_torchtpu.kernels.streamed_logits import (
    LogitsTiling,
    block_rows,
    streamed_logits,
    streamed_logits_shard,
    vocab_shard_columns,
    vocab_shard_layout,
)
from vllm_torchtpu.runner import utils as runner_utils


def _tiling() -> LogitsTiling:
    return LogitsTiling(
        block_rows=runner_utils.LOGITS_BLOCK_ROWS,
        block_dmas=runner_utils.LOGITS_BLOCK_DMAS,
        buffers=runner_utils.LOGITS_BUFFERS,
    )


def _logits(x: jax.Array, w: jax.Array) -> jax.Array:
    return streamed_logits(x, w, _tiling())


def _fake_streamed_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


# Registered at import: compute_logits runs inside a compiled graph.
_op = pallas.jax_op("pallas::streamed_logits", _logits)
_op.register_fake(_fake_streamed_logits)


def _logits_shard(
    x: jax.Array, w: jax.Array, first_block: jax.Array, num_shards: int
) -> jax.Array:
    return streamed_logits_shard(x, w, first_block, num_shards, _tiling())


def _fake_streamed_logits_shard(
    x: torch.Tensor, w: torch.Tensor, first_block: torch.Tensor, num_shards: int
) -> torch.Tensor:
    _, per_shard, _ = vocab_shard_layout(w.shape[0], num_shards, _tiling())
    cols = per_shard * block_rows(w.shape[0], _tiling())
    return torch.empty((x.shape[0], cols), dtype=x.dtype, device=x.device)


_shard_op = pallas.jax_op("pallas::streamed_logits_shard", _logits_shard)
_shard_op.register_fake(_fake_streamed_logits_shard)

# Set by `prepare_vocab_shard`: the group the head's vocab is split over, this
# rank's index in it as [1] device data in the hidden dtype, and the first
# vocab block of the shard at each collective position as int32 [world].
_SHARD_GROUP: Any = None
_SHARD_RANK: torch.Tensor | None = None
_SHARD_FIRST_BLOCKS: torch.Tensor | None = None
# Lane-aligned width of the rank tag appended to each rank's gathered rows.
_RANK_TAG_COLS = 128


def _is_plain_head(processor: LogitsProcessor, head: VocabParallelEmbedding) -> bool:
    """Whether ``processor(head, h)`` is exactly ``h @ head.weight.T`` for a
    bf16 ``h``."""
    w = head.weight
    return (
        w.device.type == "tpu"
        and w.dtype == torch.bfloat16
        and isinstance(head.quant_method, UnquantizedEmbeddingMethod)
        and head.tp_size == 1
        and getattr(head, "bias", None) is None
        and not processor.logits_as_input
        and processor.soft_cap is None
        and processor.scale == 1.0
        and getattr(processor, "head_dtype", None) in (None, w.dtype)
        and processor.org_vocab_size == w.shape[0]
        and block_rows(w.shape[0], _tiling()) > 0
    )


def _is_plain_matmul(
    processor: LogitsProcessor, head: VocabParallelEmbedding, hidden: torch.Tensor
) -> bool:
    """Whether ``processor(head, hidden)`` is exactly ``hidden @ head.weight.T``."""
    return (
        hidden.device.type == "tpu"
        and hidden.dim() == 2
        and hidden.dtype == torch.bfloat16
        and _is_plain_head(processor, head)
    )


def compute_logits(
    processor: LogitsProcessor, head: VocabParallelEmbedding, hidden: torch.Tensor
) -> torch.Tensor | None:
    """``processor(head, hidden)``, through the streamed kernel when it is a
    plain bf16 head matmul."""
    if _is_plain_matmul(processor, head, hidden):
        return _op(hidden, head.weight)
    return processor(head, hidden)


def prepare_vocab_shard(
    processor: LogitsProcessor, head: VocabParallelEmbedding, group: Any
) -> bool:
    """Arm `compute_logits_vocab_sharded` over ``group`` (a vLLM group
    coordinator whose ranks all hold the same head); returns whether it is
    armed. Call once after the weights load, outside any trace."""
    global _SHARD_GROUP, _SHARD_RANK, _SHARD_FIRST_BLOCKS
    if not _is_plain_head(processor, head):
        return False
    _, per_shard, first = vocab_shard_layout(
        head.weight.shape[0], group.world_size, _tiling()
    )
    if per_shard == 0:
        return False
    device = head.weight.device
    _SHARD_GROUP = group
    _SHARD_RANK = torch.tensor(
        [group.rank_in_group], dtype=head.weight.dtype, device=device
    )
    _SHARD_FIRST_BLOCKS = torch.tensor(first, dtype=torch.int32, device=device)
    return True


def compute_logits_vocab_sharded(
    head: VocabParallelEmbedding, hidden: torch.Tensor
) -> torch.Tensor:
    """`compute_logits` with the head's vocab split over the prepared group.

    Collective: every rank of the group must call it with the same row count.
    Each rank scores every rank's rows against its own vocab shard, so it
    streams 1/world of the head, and gets its own rows' logits back from each
    shard. Every logit is the same full-hidden f32-accumulated dot as
    `compute_logits`, so the result is identical.
    """
    assert _SHARD_GROUP is not None and _SHARD_RANK is not None
    assert _SHARD_FIRST_BLOCKS is not None
    world = _SHARD_GROUP.world_size
    rows, hidden_size = hidden.shape
    # The collectives place ranks in device order, which need not be rank
    # order, so each rank finds its position from the gathered rank tags and
    # streams that position's shard.
    tag = _SHARD_RANK.to(hidden.dtype).expand(rows, _RANK_TAG_COLS)
    gathered = _SHARD_GROUP.all_gather(torch.cat([hidden, tag], dim=1), dim=0)
    ranks = gathered[:, hidden_size].view(world, rows)[:, 0]
    position = (ranks == _SHARD_RANK).float().argmax().view(1)
    first_block = torch.index_select(_SHARD_FIRST_BLOCKS, 0, position)
    x = gathered[:, :hidden_size].contiguous()
    part = _shard_op(x, head.weight, first_block, world)
    # Block p of the leading axis (position p's rows) goes to position p;
    # block p of the result is position p's shard for this rank's rows.
    received = funcol.all_to_all_single(part, None, None, _SHARD_GROUP.device_group)
    if isinstance(received, funcol.AsyncCollectiveTensor):
        received = funcol.wait_tensor(received)
    received = received.view(world, rows, part.shape[1])
    spans = vocab_shard_columns(head.weight.shape[0], world, _tiling())
    return torch.cat(
        [received[r, :, lo:hi] for r, (lo, hi) in enumerate(spans) if hi > lo], dim=1
    )
