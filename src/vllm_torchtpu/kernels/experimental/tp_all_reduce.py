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
"""One-kernel all-reduce of a small bf16 [rows, h] activation over a 1-D mesh.

Protocol, for n ranks (a rank is a device's index in the mesh axis):

  1. Entry barrier: signal every other rank's barrier semaphore once, wait for
     n - 1 signals, so every peer is inside the kernel and its buffers are live.
  2. Reduce-scatter: h splits into n lane-aligned column chunks [rows, h / n];
     rank r owns chunk r. Chunk t of x goes by DMA straight from x (HBM) into
     slot `me` of rank t's VMEM receive buffer; the own chunk by a local copy.
     Each receive slot has its own semaphore, indexed by sender.
  3. f32 sum of the n slots in rank order 0, 1, ..., n - 1, rounded to bf16 once
     (every rank owns one chunk, so every rank gets the same bits).
  4. All-gather: the reduced chunk goes by DMA into column chunk `me` of every
     rank's out (HBM).

Exactly rows x h travel; rows needs no padding. There is no second barrier: a
rank leaves only after it has every peer's all-gather chunk, so no peer can
signal the next call's barrier before every rank has passed this call's. Every
send is waited before exit, so every semaphore ends at 0. rank r's k-th copy goes
to rank (r + k) % n, so no two ranks' k-th copies share a target.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
LANES = 128
COLLECTIVE_ID = 0
_MESH = pl.DeviceIdType.MESH
_ANY = getattr(pl, "ANY", None) or pltpu.ANY
_VMEM_LIMIT = 32 * 1024 * 1024


def supported(n: int, h: int) -> bool:
    """Whether h splits into n lane-aligned column chunks."""
    return n >= 2 and h % n == 0 and (h // n) % LANES == 0


def kernel_name(n: int, rows: int, h: int) -> str:
    return f"tp_all_reduce-m_{rows}-h_{h}-n_{n}"


def _kernel(
    x_ref, o_ref, slots, red, rs_send, rs_recv, ag_send, ag_recv, loc, *, n, cw, axis
):
    me = lax.axis_index(axis)

    def col(ref, t):
        return ref.at[:, pl.ds(pl.multiple_of(t * cw, LANES), cw)]

    barrier = pltpu.get_barrier_semaphore()
    for k in range(1, n):
        pl.semaphore_signal(
            barrier, 1, device_id=(lax.rem(me + k, n),), device_id_type=_MESH
        )
    own = pltpu.make_async_copy(col(x_ref, me), slots.at[me], loc.at[0])
    own.start()
    pl.semaphore_wait(barrier, n - 1)

    sends = []
    for k in range(1, n):
        t = lax.rem(me + k, n)
        copy = pltpu.make_async_remote_copy(
            col(x_ref, t),
            slots.at[me],
            rs_send.at[k],
            rs_recv.at[me],
            device_id=(t,),
            device_id_type=_MESH,
        )
        copy.start()
        sends.append(copy)
    own.wait()
    for k in range(1, n):
        t = lax.rem(me + n - k, n)
        pltpu.make_async_copy(slots.at[t], slots.at[t], rs_recv.at[t]).wait_recv()

    acc = slots[0].astype(F32)
    for s in range(1, n):
        acc = acc + slots[s].astype(F32)
    red[...] = acc.astype(red.dtype)

    own = pltpu.make_async_copy(red, col(o_ref, me), loc.at[1])
    own.start()
    for k in range(1, n):
        t = lax.rem(me + k, n)
        copy = pltpu.make_async_remote_copy(
            red,
            col(o_ref, me),
            ag_send.at[k],
            ag_recv.at[me],
            device_id=(t,),
            device_id_type=_MESH,
        )
        copy.start()
        sends.append(copy)
    own.wait()
    for k in range(1, n):
        t = lax.rem(me + n - k, n)
        pltpu.make_async_copy(col(o_ref, t), col(o_ref, t), ag_recv.at[t]).wait_recv()
    for copy in sends:
        copy.wait_send()


def all_reduce(x: jax.Array, n: int, axis: str, *, interpret=False) -> jax.Array:
    """Sum of x over the n ranks of mesh axis `axis`; call under shard_map.
    x: bf16 [rows, h] with supported(n, h)."""
    rows, h = x.shape
    if x.dtype != BF16 or not supported(n, h):
        raise ValueError(
            f"tp_all_reduce needs bf16 [rows, h] with h / {n} a multiple of {LANES}"
        )
    cw = h // n
    return pl.pallas_call(
        functools.partial(_kernel, n=n, cw=cw, axis=axis),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=_ANY)],
        out_specs=pl.BlockSpec(memory_space=_ANY),
        scratch_shapes=[
            pltpu.VMEM((n, rows, cw), BF16),
            pltpu.VMEM((rows, cw), BF16),
            *[pltpu.SemaphoreType.DMA((n,)) for _ in range(4)],
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=_VMEM_LIMIT
        ),
        interpret=interpret,
        name=kernel_name(n, rows, h),
    )(x)


def selftest_inputs(n: int, rows: int, h: int) -> np.ndarray:
    """[n, rows, h] f32 partials whose partial and full sums are integers exact
    in bf16 and differ between column blocks and rows, so a chunk landing in
    the wrong column block or slot changes the sum: rank d holds
    ((d + 1) 3 + 5 row + 7 lane block) % 11 - 5, rank 0 its lane block % 64."""
    d = np.arange(n)[:, None, None]
    r = np.arange(rows)[None, :, None]
    c = np.arange(h)[None, None, :] // LANES
    x = (((d + 1) * 3 + r * 5 + c * 7) % 11 - 5).astype(np.float32)
    x[0] = np.broadcast_to((c[0] % 64).astype(np.float32), (rows, h))
    return x
