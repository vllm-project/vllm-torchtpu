# SPDX-License-Identifier: Apache-2.0
"""Decode-step all-reduce of TP partial outputs on the in-kernel Pallas RS+AG.

VLLM_TPU_DECODE_PALLAS_ALL_REDUCE: the replicated (non sequence-parallel)
``sequence_parallel.reduce`` sums each layer's row-parallel partial output
(GDN ``out_proj``, QSA ``o_proj``; bf16 [tokens, hidden]) with XLA's
all-reduce, which is latency-bound at decode sizes. Armed, steps of at most
``MAX_ROWS`` rows use ``moe_tp_decode``'s reduce-scatter + all-gather kernel
instead (one kernel, each rank sums its column chunk in f32 in rank order);
larger steps keep the XLA all-reduce. The choice is made on the static row
count inside the op, so a graph traced over a dynamic token dimension calls
one op for every bucket. Not bit-identical to the XLA all-reduce: the f32
chunk sum is rounded once to bf16.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.layers.adapter import moe_tp_decode

# Steps above this many rows (prefill buckets on the TP path) keep XLA's
# all-reduce; the kernel holds the whole [rows, hidden] partial in VMEM.
MAX_ROWS = 128

_STATE: dict = {}


def _lane_hidden(hidden: int, n: int, lanes: int) -> int:
    """Hidden padded so each of n column chunks is lane-aligned."""
    unit = n * lanes
    return -(-hidden // unit) * unit


def reduce_partial(x, rank, *, axis: str, n: int):
    """Sum of x over the mesh axis (call under shard_map); x bf16 [m, h]."""
    m, h = x.shape
    if m > MAX_ROWS:
        return lax.psum(x, axis)
    info = pltpu.get_tpu_info()
    tile = info.get_sublane_tiling(jnp.bfloat16)
    rows = max(tile, -(-m // tile) * tile)
    cols = _lane_hidden(h, n, info.num_lanes)
    xp = jnp.pad(x, ((0, rows - m), (0, cols - h)))
    spec = all_reduce.make_spec(n, rows, cols)
    meta = jnp.stack([jnp.int32(m), rank.reshape(()).astype(jnp.int32)])
    return all_reduce.standalone(spec)(meta, xp)[:m, :h]


def arm(device: torch.device, hidden: int) -> bool:
    """Build the op and self-test the kernel at every decode row bucket.
    Collective over the TP group; returns whether the op is armed."""
    topo = moe_tp_decode._topology()
    if isinstance(topo, str):
        _STATE.clear()
        return False
    mesh = topo.mesh
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    info = pltpu.get_tpu_info()
    cols = _lane_hidden(hidden, n, info.num_lanes)
    tile = info.get_sublane_tiling(jnp.bfloat16)
    buckets = [(m, cols) for m in range(tile, MAX_ROWS + 1, tile)]
    failures = moe_tp_decode._selftest(topo, buckets, device)
    seen = moe_tp_decode._gather(dict(mesh_index=topo.mesh_index, failures=failures))
    if any(s["failures"] for s in seen):
        raise RuntimeError(f"decode_all_reduce: kernel self-test failed: {seen}")
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        lambda x, rank: reduce_partial(x, rank, axis=axis, n=n),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        args = (x, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::decode_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _STATE.update(
        op=op,
        hidden=hidden,
        rank=moe_tp_decode._rank_tensor(device, topo.mesh_index),
    )
    return True


def armed_for(x: torch.Tensor) -> bool:
    """Whether the armed op takes this partial (bf16, the armed hidden)."""
    return (
        bool(_STATE)
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[-1] == _STATE["hidden"]
    )


def all_reduce_partial(x: torch.Tensor) -> torch.Tensor:
    return _STATE["op"](x, _STATE["rank"])
