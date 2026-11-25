# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intra-chip tensor parallelism for the latent-MoE projections.

Kimi K3's routed experts run in a 3584-wide latent space, reached by a
``[7168, 3584]`` down-projection and left by a ``[3584, 7168]``
up-projection. Both are ``ReplicatedLinear`` and unquantized, so every rank
holds all 51 MB of each and reads all of it every decode step. Profiling K3 at
TP=32 puts the pair at ~22% of decode device time, both running at ~2.2 TB/s --
i.e. purely HBM-bandwidth bound, which is exactly the case sharding fixes.

Splitting them across the cores of one chip halves the bytes each core reads.
Both are made *column*-parallel (sharding the output) rather than row-parallel
(sharding the contraction), so the two halves are concatenated by an all-gather
rather than summed by an all-reduce: for a 2-core group the all-gather moves
half the bytes, and it keeps the arithmetic bit-identical to the replicated
layer instead of resplitting a sum.

Why the chip and not the full TP group: the exchange happens twice per layer
across 92 layers, so it must stay on-package. Over all 32 ranks the collective
latency would swamp the bandwidth saved.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_warned = False


def _warn_once(message: str, *args) -> None:
    global _warned
    if not _warned:
        logger.warning(message, *args)
        _warned = True


class IntraChipColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel over one chip's cores, gathered back on the chip.

    Drop-in for ``ReplicatedLinear``: same output, same ``(tensor, bias)``
    return shape. ``ColumnParallelLinear`` supplies the sharded weight loader;
    all this adds is doing the gather over the chip group instead of over TP.
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
        self._chip_group = group
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
        output = self._chip_group.all_gather(output_parallel, dim=-1)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"out_per_chip_core={self.output_size_per_partition}, "
            f"chip_tp_size={self.tp_size}"
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
        _warn_once(
            "TPU_LATENT_PROJ_INTRA_CHIP_TP is set but there is no usable "
            "intra-chip group; leaving the latent projections replicated."
        )
        return None

    if output_size % group.world_size != 0:
        _warn_once(
            "Latent projection output %d does not divide across %d cores per "
            "chip; leaving it replicated.",
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
    """The intra-chip sharded projection when it applies, else the replicated one."""
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
        "Latent MoE projection [%d, %d] sharded across %d cores per chip "
        "(%d columns each).",
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
