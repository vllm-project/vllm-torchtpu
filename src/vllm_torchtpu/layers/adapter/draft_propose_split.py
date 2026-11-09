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
"""Torch bridge for greedy draft tokens with the LM head split over DP
(``TPU_MTP_DRAFT_PROPOSE_DP_SPLIT``).

The draft's vocab-parallel argmax reads the whole TP shard of a hidden-major
LM head on every rank of every DP engine. With the knob on, each rank scores
its DP rank's slice of the shard for the draft rows of every engine and the
ranks exchange (value, id) pairs (``layers.core.draft_propose_split``). The
weights stay as they are; the rank's first column is a buffer on the head.

The exchange spans engines, so every engine must run each propose with the
same row count, which the draft's DP lockstep provides: busy engines propose
on the coordinated row bucket and idle ones replay the same proposes.
"""

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu.layers.core import draft_propose_split as split
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OP = None
# int32 [1] buffer on the LM head: the first column of this DP rank's slice of
# its TP shard.
SLICE_START_ATTR = "_tpu_draft_split_start"


def enable_draft_propose_split(head: torch.nn.Module, processor) -> None:
    """Check that ``head`` projects as a plain hidden-major matmul and give it
    this DP rank's slice start; build the op outside any trace."""
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.layers.adapter.ragged_a2a_moe import get_dp_tp_mesh
    from vllm_torchtpu.runner.utils import HiddenMajorLMHeadMethod

    get_dp_tp_mesh()  # checks the rank-aligned DP x TP layout the split assumes
    dp = get_dp_group()
    dp_rank, dp_size = int(dp.rank_in_group), int(dp.world_size)
    cols = head.weight.shape[-1]
    head_dtype = getattr(processor, "head_dtype", None)
    if (
        type(head.quant_method) is not HiddenMajorLMHeadMethod
        or getattr(head, "bias", None) is not None
        or processor.scale != 1.0
        or processor.soft_cap is not None
        or head_dtype not in (None, head.weight.dtype)
        or head.weight.dtype != torch.bfloat16
    ):
        raise NotImplementedError(
            "TPU_MTP_DRAFT_PROPOSE_DP_SPLIT needs a bf16 [hidden, vocab shard] "
            "LM head (TPU_LM_HEAD_HIDDEN_MAJOR) without bias, logit scale or "
            "soft cap"
        )
    if dp_size == 1 or cols % dp_size:
        raise NotImplementedError(
            "TPU_MTP_DRAFT_PROPOSE_DP_SPLIT needs DP > 1 and a vocabulary shard "
            f"that divides over DP (dp={dp_size}, shard columns={cols})"
        )
    head.register_buffer(
        SLICE_START_ATTR,
        torch.tensor(
            [dp_rank * (cols // dp_size)], dtype=torch.int32, device=head.weight.device
        ),
        persistent=False,
    )
    _get_op()
    rows = [
        r
        for r in range(1, 1025)
        if split.supported(r, head.weight.shape[0], cols, dp_size)
    ]
    logger.info(
        "Draft propose split over DP: rank %d of %d scores shard columns "
        "[%d, %d) for every engine's rows; draft row counts %s",
        dp_rank,
        dp_size,
        dp_rank * (cols // dp_size),
        (dp_rank + 1) * (cols // dp_size),
        f"{rows[0]}-{rows[-1]}" if rows else "none",
    )


def _get_op():
    global _OP
    if _OP is not None:
        return _OP
    from vllm_torchtpu.layers.adapter.ragged_a2a_moe import get_dp_tp_mesh

    mesh = get_dp_tp_mesh()

    # torch_tpu infers static argnums from annotations: every operand is jax.Array.
    def fn(
        x: jax.Array,
        w: jax.Array,
        slice_start: jax.Array,
        vocab_start: jax.Array,
        vocab_end: jax.Array,
    ) -> jax.Array:
        return split.draft_argmax(x, w, slice_start, vocab_start, vocab_end, mesh=mesh)

    spec = PartitionSpec(split.DP_AXIS, split.TP_AXIS)
    op = pallas.jax_op(
        "pallas::draft_propose_split",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 5,
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda x, w, s, a, b: x.new_empty(x.shape[:-1], dtype=torch.int32))
    _OP = op
    return op


def draft_split_tokens(
    head: torch.nn.Module, hidden: torch.Tensor
) -> torch.Tensor | None:
    """Greedy tokens [p] int32 of this engine's draft rows ``hidden`` [p,
    hidden] through the split, or None for rows it does not serve (another
    dtype than the head's, or a row count outside its table)."""
    from vllm_torchtpu.layers.adapter.ragged_a2a_moe import get_dp_tp_mesh

    num_rows, hidden_size = hidden.shape
    w = head.weight
    dp_size = get_dp_tp_mesh().shape[split.DP_AXIS]
    if hidden.dtype != w.dtype or not split.supported(
        num_rows, hidden_size, w.shape[-1], dp_size
    ):
        return None
    out = _get_op()(
        hidden.reshape(1, 1, num_rows, hidden_size),
        w.reshape(1, 1, *w.shape),
        getattr(head, SLICE_START_ATTR).reshape(1, 1, 1),
        head.org_vocab_start_index.reshape(1, 1, 1),
        head.org_vocab_end_index.reshape(1, 1, 1),
    )
    return out.reshape(num_rows)
