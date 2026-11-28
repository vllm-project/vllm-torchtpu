# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Column-parallel sharding of the latent-MoE projections.

Kimi K3's routed experts run in a 3584-wide latent space, reached by a
``[7168, 3584]`` down-projection and left by a ``[3584, 7168]``
up-projection. Both default to ``ReplicatedLinear`` and are unquantized, so
every rank holds all 51 MB of each and reads all of it every decode step.
Profiling K3 at TP=32 puts the pair at ~22% of decode device time, both
running at ~2.2 TB/s -- i.e. purely HBM-bandwidth bound, which is the case
sharding addresses.

Sharding across a rank group reduces the weight bytes each rank stores and
reads. Both projections are made *column*-parallel (sharding the output)
rather than row-parallel (sharding the contraction), so the shards are
concatenated by an all-gather rather than summed by an all-reduce: for a
2-rank group the all-gather moves half the bytes, and for any group size it
keeps the contraction arithmetic bit-identical to the replicated layer.

``make_latent_projection`` selects between two opt-in shard groups (falling
back to ``ReplicatedLinear`` when neither is enabled):

* ``TPU_LATENT_PROJ_TP``: the full TP group. Divides per-rank weight HBM by
  the TP world size on memory-constrained or single-core TPUs, at the cost of
  two cross-chip ``all_gather(dim=-1)`` collectives per MoE layer. Checked
  first when both flags are set.
* ``TPU_LATENT_PROJ_INTRA_CHIP_TP``: the cores of a single multi-core chip.
  Keeps both per-layer all-gathers on-package while halving the HBM bytes
  read per core on 2-core chips.
"""

from __future__ import annotations

import torch
from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class IntraChipColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel over a shard group, gathered back over the same group.

    ``group`` is either a single chip's core subgroup
    (``TPU_LATENT_PROJ_INTRA_CHIP_TP``) or the full TP group
    (``TPU_LATENT_PROJ_TP``); see the module docstring.

    Drop-in for ``ReplicatedLinear``: same output, same ``(tensor, bias)``
    return shape. ``ColumnParallelLinear`` supplies the sharded weight loader;
    this subclass gathers the sharded output over ``shard_group``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
        group,
    ) -> None:
        # `rank_in_group`, not the topology's core index: the all-gather below
        # concatenates shards in group-rank order, and the weight split has to
        # use the same order or the halves come back transposed.
        self.shard_group = group
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=True,
            tp_rank=group.rank_in_group,
            tp_size=group.world_size,
        )

    def forward(  # type: ignore[override]
        self,
        input_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = self.quant_method.apply(self, input_, bias)
        output = self.shard_group.all_gather(output_parallel, dim=-1)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"out_per_shard={self.output_size_per_partition}, "
            f"shard_group_size={self.tp_size}"
        )


def shard_group_or_none(output_size: int):
    """The chip group to shard ``output_size`` across, or ``None`` to replicate.

    Separated from layer construction so the decision -- the part with all the
    ways to be wrong -- can be tested without an initialized TP group.

    Answers ``None`` for every reason the split might not hold: the flag is
    off, the chip topology is unreadable, there is one core per chip, or the
    output does not divide evenly across the cores.
    """
    if not envs.TPU_LATENT_PROJ_INTRA_CHIP_TP:
        return None

    group = get_intra_chip_group()
    if group is None or group.world_size < 2:
        logger.warning_once(
            "TPU_LATENT_PROJ_INTRA_CHIP_TP is set but there is no usable "
            "intra-chip group; leaving the latent projections replicated."
        )
        return None

    if output_size % group.world_size != 0:
        logger.warning_once(
            "Latent projection output %d does not divide across %d cores per "
            "chip; leaving it replicated.",
            output_size,
            group.world_size,
        )
        return None

    return group


def _multi_rank_tp_group() -> GroupCoordinator | None:
    """The TP group if it is initialized and has at least two ranks."""
    if not parallel_state.model_parallel_is_initialized():
        return None
    group = parallel_state.get_tp_group()
    if group is None or group.world_size < 2:
        return None
    return group


def tp_shard_group_or_none(output_size: int) -> GroupCoordinator | None:
    """The TP group to shard ``output_size`` across under ``TPU_LATENT_PROJ_TP``.

    Sharding column-parallel across the TP group with ``all_gather(dim=-1)``
    divides per-rank weight HBM by the TP world size while keeping contraction
    arithmetic bit-identical to ``ReplicatedLinear``. Returns ``None`` when the
    flag is off, no multi-rank TP group exists, or ``output_size`` is not
    evenly divisible by ``tp_group.world_size``.
    """
    if not envs.TPU_LATENT_PROJ_TP:
        return None

    group = _multi_rank_tp_group()
    if group is None:
        logger.warning_once(
            "TPU_LATENT_PROJ_TP is set but there is no multi-rank TP group; "
            "leaving the latent projections replicated."
        )
        return None

    if output_size % group.world_size != 0:
        logger.warning_once(
            "Latent projection output %d does not divide across %d TP ranks; "
            "leaving it replicated.",
            output_size,
            group.world_size,
        )
        return None

    return group


def make_latent_projection(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> ReplicatedLinear | IntraChipColumnParallelLinear:
    """The sharded projection when a shard group applies, else the replicated one."""
    # Check TP sharding first: if both flags are set on a single-core chip,
    # sharding over TP avoids a spurious intra-chip fallback warning.
    group = tp_shard_group_or_none(output_size)
    if group is None:
        group = shard_group_or_none(output_size)
    if group is None:
        return ReplicatedLinear(
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    # Keyed on the shape, not the prefix: `info_once` dedupes on the formatted
    # message, so naming the layer would emit 184 near-identical lines per rank
    # instead of one per distinct projection.
    logger.info_once(
        "Latent MoE projection [%d, %d] sharded across %d ranks (%d columns each).",
        output_size,
        input_size,
        group.world_size,
        output_size // group.world_size,
    )
    return IntraChipColumnParallelLinear(
        input_size,
        output_size,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
        group=group,
    )
