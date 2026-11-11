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
"""Latency-bound all-reduce of a small bf16 tensor over the n ranks of a 1-D mesh.

One Pallas call: a barrier (every peer's scratch is live), a reduce-scatter of
n column chunks [m, h / n] (rank r owns chunk r), an f32 sum of the n slots in
rank order rounded to bf16, and an all-gather of the owned chunk into every
peer's output. Every rank gets the same bits. Rank r's i-th copy goes to rank
(r + i) % n, so no two ranks' i-th copies share a target.

`all_reduce` takes any bf16 [rows, hidden] and runs the kernel on a [16 k, w]
view of it (rows padded with zeros when the view needs it): at decode sizes the
data is latency bound, and the 16-row bf16 tile of the narrowest view moves
the fewest tiles per copy. On a tpu7x 2x2x1 (8 TensorCores) it all-reduces
bf16[8, 6144] in 8.3 us against 14.6 us for XLA's TensorCore all-reduce, and
bf16[24, 6144] in 12.1 us against 18.3 us; at 64 rows the two meet.

kernels/experimental/moe_tp_decode/all_reduce.py runs the same protocol as a
fused kernel's epilogue (row paths by n_valid, the own chunk by local DMA);
its `standalone` on the same views takes 9.0 us at 8 rows and 12.7 at 24, so
this unfused call keeps its own, leaner body.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = 0
_MESH = pl.DeviceIdType.MESH
_TILE_ROWS = 16  # bf16 sublane tiling
_LANES = 128
# Input, output and the n receive slots live in VMEM: ~3x the view (the limit
# asks 4x + 1 MiB, 33 MiB at 8 MiB, inside a tpu7x TensorCore's 64 MiB). 8 MiB
# admits the 512-row decode bucket of hidden 6144 (6 MiB).
MAX_VIEW_BYTES = 8 << 20


def view_shape(rows: int, hidden: int, n: int) -> tuple[int, int, int]:
    """(padded rows, view rows, view width) for an all-reduce of [rows, hidden]
    over n ranks: the view is [16, w] with w a multiple of n lanes, the rows
    padded with zeros to the fewest that allow it."""
    quantum = _TILE_ROWS * n * _LANES
    step = quantum // math.gcd(hidden, quantum)
    padded = -(-rows // step) * step
    return padded, _TILE_ROWS, padded * hidden // _TILE_ROWS


def view_bytes(rows: int, hidden: int, n: int) -> int:
    _, m, w = view_shape(rows, hidden, n)
    return m * w * jnp.dtype(BF16).itemsize


@functools.cache
def _call(n: int, m: int, h: int):
    w = h // n

    def kernel(rank_ref, x_ref, out_ref, recv, red, rs_send, rs_recv, ag_send, ag_recv):
        me = rank_ref[0]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(n):
            pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, n)

        def chunk(ref, t):
            return ref.at[:, pl.ds(pl.multiple_of(t * w, w), w)]

        def send_all(src_of, dst, send_sem, recv_sem):
            sends = []
            for step in range(1, n):
                t = lax.rem(me + step, n)
                copy = pltpu.make_async_remote_copy(
                    src_of(t),
                    dst,
                    send_sem.at[step],
                    recv_sem.at[me],
                    device_id=(t,),
                    device_id_type=_MESH,
                )
                copy.start()
                sends.append(copy)
            return sends

        def recv_all(dst_of, recv_sem):
            for step in range(1, n):
                t = lax.rem(me + n - step, n)
                pltpu.make_async_copy(dst_of(t), dst_of(t), recv_sem.at[t]).wait_recv()

        sends = send_all(lambda t: chunk(x_ref, t), recv.at[me], rs_send, rs_recv)
        recv[me] = x_ref[:, pl.ds(pl.multiple_of(me * w, w), w)]
        recv_all(lambda t: recv.at[t], rs_recv)
        acc = recv[0].astype(F32)
        for s in range(1, n):
            acc = acc + recv[s].astype(F32)
        red[...] = acc.astype(BF16)
        out_ref[:, pl.ds(pl.multiple_of(me * w, w), w)] = red[...]
        sends += send_all(lambda t: red, chunk(out_ref, me), ag_send, ag_recv)
        recv_all(lambda t: chunk(out_ref, t), ag_recv)
        for copy in sends:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    view = m * h * jnp.dtype(BF16).itemsize
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((n, m, w), BF16),
                pltpu.VMEM((m, w), BF16),
                *[pltpu.SemaphoreType.DMA((n,)) for _ in range(4)],
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=4 * view + (1 << 20)
        ),
        name=f"tp_all_reduce-m_{m}-h_{h}-n_{n}",
    )


def all_reduce(x: jax.Array, rank: jax.Array, n: int) -> jax.Array:
    """Sum of bf16 x [rows, hidden] over the n ranks; call under shard_map over
    the 1-D mesh. rank (int32, one element) must be this device's mesh index."""
    rows, hidden = x.shape
    padded, m, w = view_shape(rows, hidden, n)
    xp = jnp.pad(x, ((0, padded - rows), (0, 0))) if padded != rows else x
    out = _call(n, m, w)(rank.reshape(1).astype(jnp.int32), xp.reshape(m, w))
    # Relayout to [rows, hidden] here: left to XLA, the reshape sinks into the
    # consumer (the residual add + RMSNorm), which then runs on the [16, w] view
    # at 0.78 us instead of 0.15 us.
    return lax.optimization_barrier(out.reshape(padded, hidden)[:rows])
