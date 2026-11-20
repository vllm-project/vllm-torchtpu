# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch ops for Host Engram gathers over the DP ICI mesh.

Every rank owns a head shard of each Host Engram table and one tpu_sync host
signal channel per table (``common/host_channel.py``), which moves the rows
between the device and the Host. In-graph, around the channel:

* ``gather_hashes``: all-gather the DP ranks' padded hashes over ICI.
* ``local_rows``: this owner's row ids of the gathered hashes, token-major;
  the channel posts them to the Host before layer 0.
* ``exchange``: after the channel's wait, mark rows outside this shard
  invalid and all-to-all the rows back to their tokens.
"""

from functools import cache

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import get_pallas_mesh, jax_op

AXIS = "engram_dp"
# The gathered row: 256 FP8 bytes then 8 E8M0 scale bytes.
ROW_WORDS = 66
# Scale words of an invalid row: four E8M0 255 bytes each, which dequantize to 0.
_INVALID_ROW = (0,) * 64 + (-1, -1)


def local_heads(num_heads: int, owners: int) -> int:
    """Heads per owner: the head axis is padded to a multiple of ``owners``."""
    return -(-num_heads // owners)


def _local_rows(hashes: jax.Array, interval: jax.Array, owners: int):
    """This owner's row ids (invalid -> dummy row 0) and validity, token-major."""
    heads = local_heads(hashes.shape[1], owners)
    head_start, start, count = interval.reshape(-1)
    padded = jnp.pad(
        hashes, ((0, 0), (0, heads * owners - hashes.shape[1])), constant_values=-1
    )
    selected = lax.dynamic_slice(padded, (0, head_start), (hashes.shape[0], heads))
    valid = (selected >= start) & (selected < start + count)
    return jnp.where(valid, selected - start, 0).reshape(-1), valid


@cache
def initialize() -> None:
    """Register ``torch.ops.engram_host.*`` once torch.distributed is up."""
    mesh = get_pallas_mesh(axis_names=(AXIS,))
    shard, replicated = P(AXIS), P()

    def mapped(body, in_specs, out_specs):
        return jax.shard_map(
            body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )

    def gather(hashes: jax.Array) -> jax.Array:
        return mapped(
            lambda x: lax.all_gather(x, AXIS, axis=0, tiled=True), shard, replicated
        )(hashes)

    def rows(hashes: jax.Array, interval: jax.Array) -> jax.Array:
        def body(hashes, interval):
            # The channel carries int32 words whatever the hash dtype.
            rows, _ = _local_rows(hashes, interval, lax.axis_size(AXIS))
            return rows.astype(jnp.int32)

        return mapped(body, (replicated, shard), shard)(hashes, interval)

    def exchange(
        hashes: jax.Array, interval: jax.Array, packed: jax.Array, barrier: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        def body(hashes, interval, packed, barrier):
            _, valid = _local_rows(hashes, interval, lax.axis_size(AXIS))
            packed = jnp.where(
                valid.reshape(-1, 1),
                packed.reshape(-1, ROW_WORDS),
                jnp.array(_INVALID_ROW, jnp.int32),
            ).reshape(*valid.shape, ROW_WORDS)
            rows = lax.all_to_all(packed, AXIS, split_axis=0, concat_axis=1, tiled=True)
            # Whatever consumes `barrier` follows the exchange.
            return lax.optimization_barrier((rows[:, : hashes.shape[1]], barrier))

        return mapped(body, (replicated, shard, shard, shard), (shard, shard))(
            hashes, interval, packed, barrier
        )

    # TorchTPU accepts PartitionSpec; its annotation only permits plain tuples.
    # pyrefly: ignore[no-matching-overload]
    gather_op = jax_op(
        "engram_host::gather_hashes",
        gather,
        mesh=mesh,
        input_partition_specs=(shard,),
        output_partition_specs=(replicated,),
    )
    # pyrefly: ignore[no-matching-overload]
    rows_op = jax_op(
        "engram_host::local_rows",
        rows,
        mesh=mesh,
        input_partition_specs=(replicated, shard),
        output_partition_specs=(shard,),
    )
    # pyrefly: ignore[no-matching-overload]
    exchange_op = jax_op(
        "engram_host::exchange",
        exchange,
        mesh=mesh,
        input_partition_specs=(replicated, shard, shard, shard),
        output_partition_specs=(shard, shard),
    )

    # The default fake kernels reject vLLM's symbolic token axis.
    @gather_op.register_fake
    def _(hashes):
        return hashes.new_empty((hashes.shape[0] * mesh.size, *hashes.shape[1:]))

    @rows_op.register_fake
    def _(hashes, interval):
        heads = local_heads(hashes.shape[1], mesh.size)
        return hashes.new_empty((hashes.shape[0] * heads,), dtype=torch.int32)

    @exchange_op.register_fake
    def _(hashes, interval, packed, barrier):
        tokens = hashes.shape[0] // mesh.size
        return (
            packed.new_empty((tokens, hashes.shape[1], ROW_WORDS)),
            torch.empty_like(barrier),
        )


def gather_hashes(hashes: torch.Tensor) -> torch.Tensor:
    return torch.ops.engram_host.gather_hashes(hashes)


def local_rows(hashes: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
    """``[gathered tokens * local heads]`` int32 row ids of this owner's shard."""
    return torch.ops.engram_host.local_rows(hashes, interval)


def exchange(
    hashes: torch.Tensor,
    interval: torch.Tensor,
    packed: torch.Tensor,
    barrier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``packed`` (``local_rows`` order) -> ``[tokens, heads, ROW_WORDS]`` rows."""
    return torch.ops.engram_host.exchange(hashes, interval, packed, barrier)
