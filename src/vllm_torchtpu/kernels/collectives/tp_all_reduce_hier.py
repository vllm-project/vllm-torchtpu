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
"""Two-level all-reduce of a small bf16 tensor over the 8 TensorCores of a
tpu7x 2x2x1 (4 chips x 2 cores), in one Pallas call
(TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS).

tp_all_reduce.py sends every core's column chunks straight to the 7 other
cores: each chip puts 2 x 6 chunks on its ICI links in the reduce-scatter and
again in the all-gather, because the two cores of a chip send their own copies
to both cores of every other chip. Here the two cores of a chip first combine
over the on-package link, so each chip puts half of those bytes on ICI:

1. d2d reduce-scatter: core c keeps column half c and sends the other half to
   its sibling; the pair sum (f32, rounded to bf16) of half c stays on core c.
2. ICI reduce-scatter: the 4 cores with core index c (one per chip) split half
   c in 4 quarters; chip p owns quarter p. Each sends 3 quarters, one per
   peer, and sums the 4 pair sums of its quarter in chip order (f32, one
   rounding to bf16).
3. ICI all-gather: the owned chunk goes to the 3 peers and to the sibling.
4. d2d all-gather: each core forwards the 3 chunks it received in step 3 to
   its sibling, which then holds all 8.

Each output chunk is reduced once, by its owner, and copied unchanged to every
core, so all 8 cores get the same bits. Numerics differ from tp_all_reduce
(f32 sum of the 8 partials in rank order, one rounding): the pair sums are
rounded to bf16 before the ICI leg, so a chunk is rounded twice. That is the
rounding of a bf16 tree reduction (as XLA's all-reduce does it), not of a
quantized payload.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import tp_all_reduce as flat

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = flat.COLLECTIVE_ID
_MESH = pl.DeviceIdType.MESH
_N = 8
_CHIPS = 4
_LANES = 128
# Rows of the layout table: sibling core, chip position, core index, then the
# core with this core's index on chip position 0..3.
SIB, POS, CORE, AT = 0, 1, 2, 3
_TAB_ROWS = AT + _CHIPS


def layout_table(devices) -> np.ndarray:
    """int32 [7, 8] over the mesh order from devices with `.coords` and
    `.core_on_chip` (e.g. pallas_all_reduce._ChipCore): SIB, POS, CORE, and
    AT + q = the mesh index of the core with the same core index on chip q.
    Chips are numbered in sorted (x, y) order."""
    if len(devices) != _N:
        raise ValueError(f"tp_all_reduce_hier needs 8 TensorCores, got {len(devices)}")
    chips = sorted({tuple(d.coords[:2]) for d in devices})
    if len(chips) != _CHIPS or {d.core_on_chip for d in devices} != {0, 1}:
        raise ValueError(f"tp_all_reduce_hier needs 4 chips x 2 cores: {chips}")
    index = {(tuple(d.coords[:2]), int(d.core_on_chip)): i for i, d in enumerate(devices)}
    if len(index) != _N:
        raise ValueError("tp_all_reduce_hier: two mesh entries on one core")
    tab = np.zeros((_TAB_ROWS, _N), np.int32)
    for i, d in enumerate(devices):
        chip, core = tuple(d.coords[:2]), int(d.core_on_chip)
        tab[SIB, i] = index[(chip, 1 - core)]
        tab[POS, i] = chips.index(chip)
        tab[CORE, i] = core
        for q in range(_CHIPS):
            tab[AT + q, i] = index[(chips[q], core)]
    return tab


def supported(m: int, h: int) -> bool:
    """Whether a [m, h] bf16 kernel layout splits into 8 lane-aligned
    column chunks with whole bf16 tiles."""
    return h % (_N * _LANES) == 0 and m % 8 == 0


MODES = ("ag", "full")


def _owner_chunk(tab_ref, i, full: bool):
    """Column chunk (of 8) that mesh index i reduces and broadcasts."""
    if full:
        return tab_ref[CORE, i] * _CHIPS + tab_ref[POS, i]
    return i


@functools.cache
def _call(m: int, h: int, full: bool):
    h2 = h // 2
    w = h // _N  # one owned chunk

    def kernel(rank_ref, tab_ref, x_ref, out_ref, sib_in, rs_in, red,
               d2d_send, d2d_recv, rs_send, rs_recv, ag_send, ag_recv,
               fwd_send, fwd_recv):
        me = rank_ref[0]
        sib = tab_ref[SIB, me]
        pos = tab_ref[POS, me]
        core = tab_ref[CORE, me]

        def peer(step):  # same core index, chip position pos + step
            return tab_ref[AT + lax.rem(pos + step, _CHIPS), me]

        def col(ref, k):
            return ref.at[:, pl.ds(pl.multiple_of(k * w, w), w)]

        barrier = pltpu.get_barrier_semaphore()
        if full:
            # Barrier with the 4 cores this one writes to (the relation is
            # symmetric, so 4 signals arrive).
            pl.semaphore_signal(barrier, inc=1, device_id=(sib,), device_id_type=_MESH)
            for step in range(1, _CHIPS):
                pl.semaphore_signal(barrier, inc=1, device_id=(peer(step),),
                                    device_id_type=_MESH)
            pl.semaphore_wait(barrier, _CHIPS)
        else:
            for d in range(_N):
                pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
            pl.semaphore_wait(barrier, _N)

        sends = []
        if full:
            own_off = pl.multiple_of(core * h2, h2)
            # 1. d2d reduce-scatter of the column halves.
            c = pltpu.make_async_remote_copy(
                x_ref.at[:, pl.ds(pl.multiple_of((1 - core) * h2, h2), h2)],
                sib_in,
                d2d_send.at[0],
                d2d_recv.at[0],
                device_id=(sib,),
                device_id_type=_MESH,
            )
            c.start()
            sends.append(c)
            c.wait_recv()
            # Pair sum of half `core`, kept in out_ref's own half. A peer's
            # step-3 chunk lands on the quarter this core sent to that peer,
            # which the peer has received before it sends.
            out_ref[:, pl.ds(own_off, h2)] = (
                x_ref[:, pl.ds(own_off, h2)].astype(F32) + sib_in[...].astype(F32)
            ).astype(BF16)
            # 2. ICI reduce-scatter of the quarters among the same-index cores.
            for step in range(1, _CHIPS):
                t = lax.rem(pos + step, _CHIPS)
                c = pltpu.make_async_remote_copy(
                    col(out_ref, core * _CHIPS + t),
                    rs_in.at[pos],
                    rs_send.at[step],
                    rs_recv.at[pos],
                    device_id=(peer(step),),
                    device_id_type=_MESH,
                )
                c.start()
                sends.append(c)
            rs_in[pos] = out_ref[:, pl.ds(pl.multiple_of((core * _CHIPS + pos) * w, w), w)]
            for step in range(1, _CHIPS):
                s = lax.rem(pos + _CHIPS - step, _CHIPS)
                pltpu.make_async_copy(rs_in.at[s], rs_in.at[s], rs_recv.at[s]).wait_recv()
            slots = _CHIPS
        else:
            # 1-2. Flat reduce-scatter, as tp_all_reduce: chunk t to mesh index t.
            for step in range(1, _N):
                t = lax.rem(me + step, _N)
                c = pltpu.make_async_remote_copy(
                    col(x_ref, t),
                    rs_in.at[me],
                    rs_send.at[step],
                    rs_recv.at[me],
                    device_id=(t,),
                    device_id_type=_MESH,
                )
                c.start()
                sends.append(c)
            rs_in[me] = x_ref[:, pl.ds(pl.multiple_of(me * w, w), w)]
            for step in range(1, _N):
                t = lax.rem(me + _N - step, _N)
                pltpu.make_async_copy(rs_in.at[t], rs_in.at[t], rs_recv.at[t]).wait_recv()
            slots = _N
        acc = rs_in[0].astype(F32)
        for q in range(1, slots):
            acc = acc + rs_in[q].astype(F32)
        red[...] = acc.astype(BF16)
        mine = col(out_ref, _owner_chunk(tab_ref, me, full))
        mine[...] = red[...]

        # 3. ICI all-gather of the owned chunk to the same-index cores, plus
        # its copy to the sibling.
        for step in range(1, _CHIPS):
            c = pltpu.make_async_remote_copy(
                red, mine, ag_send.at[step], ag_recv.at[pos],
                device_id=(peer(step),), device_id_type=_MESH,
            )
            c.start()
            sends.append(c)
        c = pltpu.make_async_remote_copy(
            red, mine, ag_send.at[0], fwd_recv.at[pos], device_id=(sib,),
            device_id_type=_MESH,
        )
        c.start()
        sends.append(c)

        # 4. Forward each chunk received in step 3 to the sibling.
        for step in range(1, _CHIPS):
            s = lax.rem(pos + _CHIPS - step, _CHIPS)
            got = col(out_ref, _owner_chunk(tab_ref, tab_ref[AT + s, me], full))
            pltpu.make_async_copy(got, got, ag_recv.at[s]).wait_recv()
            c = pltpu.make_async_remote_copy(
                got, got, fwd_send.at[step], fwd_recv.at[s], device_id=(sib,),
                device_id_type=_MESH,
            )
            c.start()
            sends.append(c)
        # The chunks of the sibling's same-index cores: its own and the 3 it
        # forwards.
        for q in range(_CHIPS):
            got = col(out_ref, _owner_chunk(tab_ref, tab_ref[AT + q, sib], full))
            pltpu.make_async_copy(got, got, fwd_recv.at[q]).wait_recv()
        for c in sends:
            c.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    view = m * h * jnp.dtype(BF16).itemsize
    slots = _CHIPS if full else _N
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((m, h2) if full else (8, _LANES), BF16),
                pltpu.VMEM((slots, m, w), BF16),
                pltpu.VMEM((m, w), BF16),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((_N,)),
                pltpu.SemaphoreType.DMA((_N,)),
                *[pltpu.SemaphoreType.DMA((_CHIPS,)) for _ in range(4)],
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=4 * view + (1 << 20)
        ),
        name=f"tp_all_reduce_hier_{'full' if full else 'ag'}-m_{m}-h_{h}",
    )


def all_reduce(
    x: jax.Array, rank: jax.Array, tab: np.ndarray, direct_rows: int = 0,
    mode: str = "ag",
) -> jax.Array:
    """Sum of bf16 x [rows, hidden] over the 8 cores; call under shard_map
    over the 1-D mesh whose order `tab` (layout_table) describes. rank (int32,
    one element) is this device's mesh index. Layout as tp_all_reduce: x's
    own [rows, hidden] when direct_rows divides rows, else the [16, w] view.
    mode "ag": flat reduce-scatter (bits of tp_all_reduce), two-level
    all-gather; "full": both legs two-level (pair sums rounded to bf16)."""
    if mode not in MODES:
        raise ValueError(f"tp_all_reduce_hier mode {mode!r} not in {MODES}")
    full = mode == "full"
    rows, hidden = x.shape
    r = rank.reshape(1).astype(jnp.int32)
    t = jnp.asarray(tab)
    if flat.direct_ok(rows, hidden, _N, direct_rows):
        return _call(rows, hidden, full)(r, t, x)
    padded, m, w = flat.view_shape(rows, hidden, _N)
    xp = jnp.pad(x, ((0, padded - rows), (0, 0))) if padded != rows else x
    out = _call(m, w, full)(r, t, xp.reshape(m, w))
    return lax.optimization_barrier(out.reshape(padded, hidden)[:rows])
