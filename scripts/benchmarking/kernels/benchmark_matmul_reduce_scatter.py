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
"""Benchmark the matmul + reduce-scatter collective at Llama-70B TP shapes.

For each token count M in the sweep this measures

    reduce_scatter(a[M, N // tp] @ w[N // tp, K], axis=0)

for the fused ring kernel and for two XLA formulations of the same collective;
XLA's baseline is the faster of the two, which changes with the shape.

Run on a TPU host, or on every host of a multi-host slice:
    python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
        --k 4096 --n 20736 --m 512,1024,2048,4096,8192
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from collective_bench_lib import AXIS, run
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import matmul_reduce_scatter


def _snake(axes):
    """Boustrophedon coord order: each axis reverses the sub-order it wraps, so
    consecutive coords differ by one step on exactly one axis."""
    if not axes:
        return [()]
    sub = _snake(axes[1:])
    return [(v, ) + s for i, v in enumerate(axes[0])
            for s in (sub if i % 2 == 0 else sub[::-1])]


def build_ring_mesh():
    """Devices ordered as a single-hop Hamiltonian cycle over the chip grid."""
    # A 1-D collective's physical embedding IS the mesh's device order, and the
    # ready-made orders are not single-hop on a 2x2x2 slice: jax.devices() has
    # 4 of 8 chip edges > 1 hop, mesh_utils.create_device_mesh 2 of 8.
    by_chip = {}
    for d in jax.devices():
        by_chip.setdefault(tuple(d.coords), []).append(d)
    ndim = len(next(iter(by_chip)))
    axes = [sorted({c[i] for c in by_chip}) for i in range(ndim)]
    order = []
    for coord in _snake(axes):
        if coord not in by_chip:
            raise ValueError(f"chip grid is not full (missing {coord}); "
                             "order the devices explicitly")
        order.extend(by_chip[coord])
    cycle = [tuple(d.coords) for d in order]
    cycle = [c for i, c in enumerate(cycle) if i == 0 or c != cycle[i - 1]]
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        hops = sum(abs(x - y) for x, y in zip(a, b))
        if hops != 1:
            raise ValueError(f"chip ring edge {a}->{b} is {hops} hops")
    return Mesh(np.asarray(order), (AXIS, ))


def make_inputs(mesh, m, k, n, dtype):
    """a[M, N] and w[N, K] both contraction-sharded over the TP axis."""
    key_a, key_w = jax.random.split(jax.random.key(0), 2)
    a = jax.device_put(
        jax.random.normal(key_a, (m, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    w = jax.device_put(
        jax.random.normal(key_w, (n, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    return jax.block_until_ready((a, w))


def build_xla_baseline(mesh, inputs):
    """Auto-partitioned einsum: XLA inserts the reduce-scatter from the shards."""
    fn = jax.jit(lambda a, w: jnp.einsum("mn,nk->mk", a, w),
                 out_shardings=NamedSharding(mesh, P(AXIS, None)))
    return fn.lower(*inputs).compile()


def build_xla_shard_map(mesh, inputs):
    """The same collective written by hand as a local matmul + psum_scatter."""

    def rs(a_shard, w_shard):
        prod = jnp.dot(a_shard, w_shard, preferred_element_type=jnp.float32)
        return lax.psum_scatter(prod.astype(a_shard.dtype),
                                AXIS,
                                scatter_dimension=0,
                                tiled=True)

    fn = jax.jit(
        jax.shard_map(rs,
                      mesh=mesh,
                      in_specs=(P(None, AXIS), P(AXIS, None)),
                      out_specs=P(AXIS, None),
                      check_vma=False))
    return fn.lower(*inputs).compile()


def build_ring_kernel(mesh, inputs):
    """The fused ring kernel, on the single-hop order it is embedded in."""
    fn = jax.jit(
        functools.partial(matmul_reduce_scatter.matmul_reduce_scatter,
                          mesh=mesh,
                          axis_name=AXIS))
    return fn.lower(*inputs).compile()


IMPLEMENTATIONS = {
    "xla": build_xla_baseline,
    "xla_sm": build_xla_shard_map,
    "ring": build_ring_kernel,
}

if __name__ == "__main__":
    # A no-op on one host; on a slice, run this same command on every host.
    jax.distributed.initialize()
    run(
        make_inputs=make_inputs,
        implementations=IMPLEMENTATIONS,
        default_n=28672,  # N = FFN intermediate dim (the contraction axis)
        mesh_fn=build_ring_mesh)
