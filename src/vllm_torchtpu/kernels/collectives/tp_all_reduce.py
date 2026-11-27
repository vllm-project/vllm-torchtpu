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

Above TWO_LEVEL_MIN_VIEW_BYTES the view's bytes, not the hops, set the time,
and `all_reduce_two_level` runs a chip-aware variant on meshes of two-core
chips: the two cores of a chip first exchange halves and sum them (rounded to
bf16), the cores of one parity then reduce-scatter and all-gather their half
across chips, and each gathered piece goes on to the chip's other core as it
lands. The flat kernel sends six of eight chunks off chip in each phase, and
both cores of a chip receive the same chunks; the two-level one sends half
those bytes off chip and adds two on-chip hops. Its sum rounds twice (the pair
sum, then the total), so its bits differ from the flat kernel's.

`all_reduce_carrying` (the flat kernel) also copies HBM operands into VMEM
outputs while the protocol runs, for consumers that want them in VMEM next
(the next layer's weights, such as `fp8_linear`'s e4m3 operand): the copies
start after the barrier signals and are waited at the end, so they overlap the
barrier wait and the exchange instead of following the call.

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
# Input, output and the n receive slots live in VMEM: ~3x the view.
MAX_VIEW_BYTES = 4 << 20
# Views above this take the two-level kernel where the mesh allows it. On a
# tpu7x 2x2x1 a cross-chip DMA hop measured ~3.3 us plus ~75 GB/s per core and
# an on-chip hop ~1.1 us, so halving the off-chip bytes (3/4 of the view per
# core) pays for the two on-chip hops from ~200 KiB; chained per-call times put
# the crossover between 24 rows of 6144 (flat ahead) and 32 (two-level ahead).
TWO_LEVEL_MIN_VIEW_BYTES = 24 * 6144 * 2


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


def _open(n: int, copies: list) -> None:
    """The barrier round (every peer's scratch is live), with the carried
    copies started between the signals and the wait."""
    barrier = pltpu.get_barrier_semaphore()
    for d in range(n):
        pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
    for copy in copies:
        copy.start()
    pl.semaphore_wait(barrier, n)


def _carrying(body, n_in: int, k: int):
    """The Pallas kernel for `body(*refs, copies)`: the k carried HBM inputs
    (after the n_in others), their VMEM outputs (after the sum) and the
    copies' semaphores (the last scratch) are taken out of refs into copies,
    one async copy per carried operand."""
    if not k:
        return lambda *refs: body(*refs, [])

    def kernel(*refs):
        srcs, (out, *rest) = refs[n_in : n_in + k], refs[n_in + k :]
        dsts, rest, sems = rest[:k], rest[k:-1], rest[-1]
        copies = [
            pltpu.make_async_copy(src, dst, sems.at[i])
            for i, (src, dst) in enumerate(zip(srcs, dsts))
        ]
        body(*refs[:n_in], out, *rest, copies)

    return kernel


def _specs(carry: tuple, out_shape, scratch: list):
    """(extra in_specs, out_shape, out_specs, scratch) with the carry added."""
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    if not carry:
        return [], out_shape, vmem, scratch
    return (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * len(carry),
        (out_shape, *(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in carry)),
        (vmem,) * (1 + len(carry)),
        [*scratch, pltpu.SemaphoreType.DMA((len(carry),))],
    )


@functools.cache
def _call(n: int, m: int, h: int, carry: tuple = ()):
    """carry: (shape, dtype) per operand copied into a VMEM output."""
    w = h // n

    def body(
        rank_ref, x_ref, out_ref, recv, red, rs_send, rs_recv, ag_send, ag_recv, cps
    ):
        me = rank_ref[0]
        _open(n, cps)

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
        for copy in cps:
            copy.wait()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    view = m * h * jnp.dtype(BF16).itemsize
    carry_in, out_shape, out_specs, scratch = _specs(
        carry,
        jax.ShapeDtypeStruct((m, h), BF16),
        [
            pltpu.VMEM((n, m, w), BF16),
            pltpu.VMEM((m, w), BF16),
            *[pltpu.SemaphoreType.DMA((n,)) for _ in range(4)],
        ],
    )
    return pl.pallas_call(
        _carrying(body, 2, len(carry)),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, *carry_in],
            out_specs=out_specs,
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=4 * view + (1 << 20)
        ),
        name=f"tp_all_reduce-m_{m}-h_{h}-n_{n}" + ("-carry" if carry else ""),
    )


def _run(call, x: jax.Array, n: int, *scalars: jax.Array, carry=()):
    """Run a kernel built by `_call`-like `call(m, w)` on x's [16 k, w] view;
    returns the sum as [rows, hidden], with the carried VMEM copies if any."""
    rows, hidden = x.shape
    padded, m, w = view_shape(rows, hidden, n)
    xp = jnp.pad(x, ((0, padded - rows), (0, 0))) if padded != rows else x
    out = call(m, w)(*scalars, xp.reshape(m, w), *carry)
    out, *copies = out if carry else (out,)
    # Relayout to [rows, hidden] here: left to XLA, the reshape sinks into the
    # consumer (the residual add + RMSNorm), which then runs on the [16, w] view
    # at 0.78 us instead of 0.15 us.
    total = lax.optimization_barrier(out.reshape(padded, hidden)[:rows])
    return (total, tuple(copies)) if carry else total


def all_reduce(x: jax.Array, rank: jax.Array, n: int) -> jax.Array:
    """Sum of bf16 x [rows, hidden] over the n ranks; call under shard_map over
    the 1-D mesh. rank (int32, one element) must be this device's mesh index."""
    rank = rank.reshape(1).astype(jnp.int32)
    return _run(lambda m, w: _call(n, m, w), x, n, rank)


def all_reduce_carrying(
    x: jax.Array, rank: jax.Array, n: int, carry: tuple[jax.Array, ...]
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """(`all_reduce` of x, the carry copied into VMEM by the same call)."""
    rank = rank.reshape(1).astype(jnp.int32)
    key = tuple((tuple(a.shape), a.dtype) for a in carry)
    # The kernel's HBM in_spec does not bind XLA, which would otherwise
    # prefetch the operand into VMEM ahead of the call: the very copy this
    # call is to hide.
    carry = tuple(pltpu.with_memory_space_constraint(a, pltpu.HBM) for a in carry)
    return _run(lambda m, w: _call(n, m, w, key), x, n, rank, carry=carry)


def two_level_table(chips: list[list[int]]) -> tuple[int, ...] | None:
    """The two-level kernel's static table for a mesh whose indices sit on
    `chips` (each the mesh indices of one chip, chips in one fixed order), or
    None when a chip does not hold exactly two of them or there is one chip.
    Row i: (partner, pos, half, group...), flattened: the other index on i's
    chip, i's chip position, which half of the view i reduces (0 for the lower
    index of its chip), and the indices of that half, one per chip in order."""
    if len(chips) < 2 or any(len(c) != 2 for c in chips):
        return None
    pairs = [sorted(c) for c in chips]
    rows = {}
    for pos, pair in enumerate(pairs):
        for half, me in enumerate(pair):
            group = [p[half] for p in pairs]
            rows[me] = (pair[1 - half], pos, half, *group)
    return tuple(v for i in sorted(rows) for v in rows[i])


@functools.cache
def _call_two_level(n: int, m: int, h: int):
    chips = n // 2
    width = 3 + chips  # table row
    half_w = h // 2
    w = half_w // chips  # one chip's piece of a half: the view's chunk, h / n

    def kernel(rank_ref, tab_ref, x_ref, out_ref, pair_in, pair_sum, recv, red, *sems):
        send_sem, recv_sem = sems
        me = rank_ref[0]
        row = me * width
        partner, pos, half = tab_ref[row], tab_ref[row + 1], tab_ref[row + 2]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(n):
            pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, n)

        # Receive semaphores: [0, chips) the partner's pieces of my half,
        # [chips, 2 chips) the cross-chip reduce-scatter by sender position,
        # [2 chips, 3 chips) the cross-chip all-gather by sender position,
        # [3 chips, 4 chips) the partner's pieces of its half.
        sends = []

        def send(src, dst, k, sem, device):
            copy = pltpu.make_async_remote_copy(
                src,
                dst,
                send_sem.at[k],
                recv_sem.at[sem],
                device_id=(device,),
                device_id_type=_MESH,
            )
            copy.start()
            sends.append(copy)

        def wait(ref, sem):
            pltpu.make_async_copy(ref, ref, recv_sem.at[sem]).wait_recv()

        def cols(ref, start):
            if isinstance(start, int):
                return ref.at[:, pl.ds(start, w)]
            return ref.at[:, pl.ds(pl.multiple_of(start, w), w)]

        def piece(q):  # chip q's piece of my half, in view columns
            return half * half_w + q * w

        def peer(q):  # the device reducing my half's piece q
            return tab_ref[row + 3 + q]

        # The partner's half, piece by piece, so each pair sum can leave early.
        for q in range(chips):
            send(
                cols(x_ref, (1 - half) * half_w + q * w),
                cols(pair_in, q * w),
                q,
                q,
                partner,
            )
        for i in range(1, chips + 1):
            q = lax.rem(pos + i, chips)  # my own piece last
            wait(cols(pair_in, q * w), q)
            mine = x_ref[:, pl.ds(pl.multiple_of(piece(q), w), w)].astype(F32)
            other = pair_in[:, pl.ds(pl.multiple_of(q * w, w), w)].astype(F32)
            total = mine + other
            pair_sum[:, pl.ds(pl.multiple_of(q * w, w), w)] = total.astype(BF16)
            if i < chips:
                send(
                    cols(pair_sum, q * w),
                    recv.at[pos],
                    chips - 1 + i,
                    chips + pos,
                    peer(q),
                )
        recv[pos] = pair_sum[:, pl.ds(pl.multiple_of(pos * w, w), w)]
        for i in range(1, chips):
            q = lax.rem(pos + chips - i, chips)
            wait(recv.at[q], chips + q)
        acc = recv[0].astype(F32)
        for q in range(1, chips):
            acc = acc + recv[q].astype(F32)
        red[...] = acc.astype(BF16)

        mine_at = piece(pos)
        out_ref[:, pl.ds(pl.multiple_of(mine_at, w), w)] = red[...]
        k = 2 * chips - 1
        for i in range(1, chips):
            send(
                red,
                cols(out_ref, mine_at),
                k,
                2 * chips + pos,
                peer(lax.rem(pos + i, chips)),
            )
            k += 1
        send(red, cols(out_ref, mine_at), k, 3 * chips + pos, partner)
        k += 1
        # Forward each gathered piece of my half to the partner as it lands.
        for i in range(1, chips):
            q = lax.rem(pos + chips - i, chips)
            wait(cols(out_ref, piece(q)), 2 * chips + q)
            send(
                cols(out_ref, piece(q)),
                cols(out_ref, piece(q)),
                k,
                3 * chips + q,
                partner,
            )
            k += 1
        for q in range(chips):
            wait(cols(out_ref, (1 - half) * half_w + q * w), 3 * chips + q)
        for copy in sends:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    view = m * h * jnp.dtype(BF16).itemsize
    sends = 4 * chips - 2  # pair pieces, reduce-scatter, gathers, forwards
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((m, half_w), BF16),
                pltpu.VMEM((m, half_w), BF16),
                pltpu.VMEM((chips, m, w), BF16),
                pltpu.VMEM((m, w), BF16),
                pltpu.SemaphoreType.DMA((sends,)),
                pltpu.SemaphoreType.DMA((4 * chips,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=4 * view + (1 << 20)
        ),
        name=f"tp_all_reduce-two_level-m_{m}-h_{h}-n_{n}",
    )


def all_reduce_two_level(
    x: jax.Array, rank: jax.Array, table: tuple[int, ...], n: int
) -> jax.Array:
    """`all_reduce` by the two-level kernel; table is `two_level_table` of
    the mesh."""
    rows, hidden = x.shape
    padded, m, w = view_shape(rows, hidden, n)
    xp = jnp.pad(x, ((0, padded - rows), (0, 0))) if padded != rows else x
    out = _call_two_level(n, m, w)(
        rank.reshape(1).astype(jnp.int32),
        jnp.asarray(table, dtype=jnp.int32),
        xp.reshape(m, w),
    )
    # As in all_reduce.
    return lax.optimization_barrier(out.reshape(padded, hidden)[:rows])
