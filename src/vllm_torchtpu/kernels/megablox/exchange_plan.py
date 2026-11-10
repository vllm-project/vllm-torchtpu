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
"""The capacity exchange's dispatch plan in one kernel.

Between the routing all-gather and the row scatter, the capacity exchange of
the ragged all-to-all derives from the DP group's gathered routing whether the
step fits the capacity buffers on every rank, the token counts between the
members of this rank's TP column, the all-to-all's sizes and offsets, the
token of each send row and where each received token lands. As XLA ops these
are many small fusions, all before the all-to-all can start. This kernel
computes the same integers from the gathered words in VMEM, in one launch.

Each token's EP ranks are kept as a bitmask (one bit per rank) and its routes
per rank as 4-bit counters, both built over the top-k route slabs; the counts
the plan needs are lane sums and popcounts of those.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

I32 = jnp.int32
_LANES = 128
_SUBLANES = 8
# The word an empty route slot holds: id field -1.
_EMPTY_WORD = -1 << 16


class ExchangePlan(NamedTuple):
    """The capacity exchange's plan for this rank (``D`` DP members, ``S``
    tokens per member, ``rows`` capacity rows)."""

    overflow: jax.Array  # bool []: some rank's rows or routes exceed the buffers
    send_start: jax.Array  # [D]: first send row of each destination
    send_sizes: jax.Array  # [D]: rows sent to each destination (0 on overflow)
    send_landing: jax.Array  # [D]: where they land on each destination
    recv_sizes: jax.Array  # [D]: rows received from each source (0 on overflow)
    count: jax.Array  # [1]: send rows the row scatter writes
    src_rows: jax.Array  # [1]: rows received in all
    send_index: jax.Array  # [rows]: token of each send row below ``count``
    in_row: jax.Array  # [D, S]: receive row of each token that reaches this rank


def _divider(divisor: int, limit: int):
    """``x // divisor`` for ``0 <= x < limit`` as a shift or one multiply and
    shift (exact over the range, checked here), or None."""
    if divisor & (divisor - 1) == 0:
        shift = divisor.bit_length() - 1
        return lambda x: x >> shift
    x = np.arange(limit, dtype=np.int64)
    for shift in range(8, 31):
        mul = (1 << shift) // divisor + 1
        if (limit - 1) * mul < 1 << 31 and np.array_equal(
            (x * mul) >> shift, x // divisor
        ):
            return lambda v, mul=mul, shift=shift: (v * mul) >> shift
    return None


def _shapes(num_members, tp_size, num_tokens, topk):
    members = -(-num_members // _SUBLANES) * _SUBLANES
    ranks = num_members * tp_size
    return members, ranks, -(-ranks // 32), -(-ranks // 8)


def exchange_plan_supported(
    *, num_members: int, tp_size: int, num_tokens: int, topk: int, slots_per_rank: int
) -> bool:
    """Whether the kernel takes this exchange: at most 128 tokens per member
    and 64 EP ranks, a power-of-two TP size, top-k at most 15 (4-bit route
    counters) with the counters of all members summing in one byte."""
    members, ranks, _, _ = _shapes(num_members, tp_size, num_tokens, topk)
    return (
        num_tokens <= _LANES
        and ranks <= 64
        and tp_size & (tp_size - 1) == 0
        and topk <= 15
        and topk * members <= 255
        and _divider(slots_per_rank, ranks * slots_per_rank) is not None
    )


def _lane_scan(x):
    """Inclusive prefix sum along the 128 lanes."""
    lane = jax.lax.broadcasted_iota(I32, x.shape, x.ndim - 1)
    d = 1
    while d < x.shape[-1]:
        x = x + jnp.where(lane >= d, pltpu.roll(x, d, x.ndim - 1), 0)
        d *= 2
    return x


def _sublane_scan_excl(x):
    """Exclusive prefix sum along the sublanes (axis 0)."""
    row = jax.lax.broadcasted_iota(I32, x.shape, 0)
    s = x
    d = 1
    while d < x.shape[0]:
        s = s + jnp.where(row >= d, pltpu.roll(s, d, 0), 0)
        d *= 2
    return s - x


def _compact_lanes(bits):
    """Per row of the 0/1 array ``bits`` [N, 128]: ``out[r, i]`` = lane of the
    (i+1)-th one of row r (0 past them), and the inclusive prefix count.

    Each one moves left by the number of zeros before it, in steps of
    increasing powers of two; the moves keep the ones in order, so no two land
    on one lane."""
    width = bits.shape[1]
    lane = jax.lax.broadcasted_iota(I32, bits.shape, 1)
    prefix = _lane_scan(bits)
    val = jnp.where(bits != 0, lane, -1)
    shift = jnp.where(bits != 0, lane + 1 - prefix, 0)
    d = 1
    while d < width:
        moves = (shift & d) != 0
        arrives = (pltpu.roll(moves.astype(I32), width - d, 1) != 0) & (
            lane < width - d
        )
        val = jnp.where(
            arrives, pltpu.roll(val, width - d, 1), jnp.where(moves, -1, val)
        )
        shift = jnp.where(
            arrives, pltpu.roll(shift, width - d, 1), jnp.where(moves, 0, shift)
        )
        d *= 2
    return jnp.maximum(val, 0), prefix


def _column_mask(column: int, tp_size: int) -> int:
    """The bits of one TP column's ranks in a 32-rank reach word, as int32."""
    mask = 0
    for b in range(column, 32, tp_size):
        mask |= 1 << b
    return mask - (1 << 32) if mask >= 1 << 31 else mask


def _plan_kernel(
    rank_ref,
    words_ref,
    start_ref,
    size_ref,
    land_ref,
    rsize_ref,
    over_ref,
    count_ref,
    src_ref,
    index_ref,
    inrow_ref,
    slabs_ref,
    reach_ref,
    *,
    num_members,
    tp_size,
    num_tokens,
    topk,
    slots_per_rank,
    rows,
    routes,
    order_bits,
):
    members, ranks, num_words, num_counters = _shapes(
        num_members, tp_size, num_tokens, topk
    )
    shape = (members, _LANES)
    ep = rank_ref[0]
    mycol = ep % tp_size
    dp = ep // tp_size
    div = _divider(slots_per_rank, ranks * slots_per_rank)
    assert div is not None  # exchange_plan_supported found one
    # The route slabs [K][members, 128]: member rows and token lanes past the
    # group's hold empty words.
    for k in range(topk):
        slabs_ref[k] = jnp.full(shape, _EMPTY_WORD, I32)
        slabs_ref[k, :num_members, :num_tokens] = words_ref[:, k, :]

    # Per token: the EP ranks it reaches (bit r of word r // 32) and its routes
    # per rank (4-bit counter r % 8 of word r // 8).
    reach = [jnp.zeros(shape, I32) for _ in range(num_words)]
    counters = [jnp.zeros(shape, I32) for _ in range(num_counters)]
    for k in range(topk):
        slot = slabs_ref[k] >> 16
        if order_bits:
            slot = slot >> order_bits
        routed = (slot >= 0) & (slot < ranks * slots_per_rank)
        # Unrouted slots take a rank past every counter word.
        rank = jnp.where(routed, div(jnp.maximum(slot, 0)), 8 * num_counters)
        bit = jnp.left_shift(1, rank & 31)
        for w in range(num_words):
            reach[w] = reach[w] | jnp.where(routed & ((rank >> 5) == w), bit, 0)
        one = jnp.left_shift(1, (rank & 7) << 2)
        word = rank >> 3
        for j in range(num_counters):
            counters[j] = counters[j] + jnp.where(word == j, one, 0)
    for w in range(num_words):
        reach_ref[w] = reach[w]

    # Routes per rank: nibbles -> bytes summed over the members (at most
    # topk * members), bytes -> 16-bit fields summed over the lanes.
    parts = []
    for c in counters:
        for nibbles in (c & 0x0F0F0F0F, (c >> 4) & 0x0F0F0F0F):
            per_lane = nibbles.sum(axis=0, keepdims=True)
            parts += [per_lane & 0x00FF00FF, (per_lane >> 8) & 0x00FF00FF]
    totals = jnp.concatenate(parts, axis=0).sum(axis=1, keepdims=True)
    max_routes = jnp.max(jnp.maximum(totals & 0xFFFF, totals >> 16))
    worst = max_routes - routes
    # Rows each member sends to each TP column: popcounts of the column's bits.
    for c in range(tp_size):
        mask = _column_mask(c, tp_size)
        sent = jax.lax.population_count(reach[0] & mask)
        for w in range(1, num_words):
            sent = sent + jax.lax.population_count(reach[w] & mask)
        worst = jnp.maximum(worst, jnp.max(sent.sum(axis=1, keepdims=True)) - rows)

    # Rows each rank receives are at most its routes: counted only when some
    # rank's routes exceed the rows.
    def received_rows(_):
        most = jnp.int32(0)
        for w in range(num_words):
            for i in range(16):
                f = ((reach[w] >> i) & 0x00010001).sum(axis=0, keepdims=True)
                f = f.sum(axis=1, keepdims=True)
                most = jnp.maximum(most, jnp.max(jnp.maximum(f & 0xFFFF, f >> 16)))
        return most - rows

    worst = jnp.maximum(
        worst,
        jax.lax.cond(max_routes > rows, received_rows, lambda _: jnp.int32(-1), 0),
    )
    over = worst > 0

    # This column's counts[s, m]: tokens of member s reaching member m.
    lane = jax.lax.broadcasted_iota(I32, shape, 1)
    row = jax.lax.broadcasted_iota(I32, shape, 0)
    counts = jnp.zeros(shape, I32)
    for m in range(num_members):
        r = m * tp_size + mycol
        word = reach[0]
        if num_words > 1:
            word = jnp.where(r >= 32, reach[1], reach[0])
        reached = ((word >> (r & 31)) & 1).sum(axis=1, keepdims=True)
        counts = jnp.where(lane == m, reached, counts)
    counts = jnp.where(over, 0, counts)
    send_sizes = jnp.where(row == dp, counts, 0).sum(axis=0, keepdims=True)
    landing = jnp.where(row < dp, counts, 0).sum(axis=0, keepdims=True)
    recv_sizes = jnp.where(lane == dp, counts, 0).sum(axis=1, keepdims=True)
    send_start = _lane_scan(send_sizes) - send_sizes
    recv_start = _sublane_scan_excl(jnp.broadcast_to(recv_sizes, shape))
    start_ref[...] = send_start[0, :num_members]
    size_ref[...] = send_sizes[0, :num_members]
    land_ref[...] = landing[0, :num_members]
    rsize_ref[...] = jnp.where(row == lane, jnp.broadcast_to(recv_sizes, shape), 0).sum(
        axis=0
    )[:num_members]

    # Receive rows: the tokens of source s that reach this rank fill rows
    # recv_start[s] on, in token order.
    word = reach[0]
    if num_words > 1:
        word = jnp.where(ep >= 32, reach[1], reach[0])
    mine = (word >> (ep & 31)) & 1
    inrow_ref[...] = recv_start + _lane_scan(mine) - 1

    # This member's tokens per destination m of the column (not masked by the
    # overflow: rows past the buffer are dropped and nothing is sent).
    own = [reach_ref[w, pl.ds(dp, 1), :] for w in range(num_words)]
    r = row * tp_size + mycol
    word = own[0]
    if num_words > 1:
        word = jnp.where(r >= 32, own[1], own[0])
    to_member = jnp.where(row < num_members, (word >> (r & 31)) & 1, 0)
    tokens, prefix = _compact_lanes(to_member)
    sizes = prefix[:, _LANES - 1 : _LANES]
    base = jnp.where(
        row == lane, _sublane_scan_excl(jnp.broadcast_to(sizes, shape)), 0
    ).sum(axis=0, keepdims=True)
    # Send row j: member m(j) = the last member whose rows start at or before j,
    # token tokens[m(j), j - base[m(j)]].
    base8 = jnp.broadcast_to(base, (_SUBLANES, _LANES))
    tile = _SUBLANES * _LANES
    for t in range(index_ref.shape[0] // _SUBLANES):
        j = (
            t * tile
            + jax.lax.broadcasted_iota(I32, (_SUBLANES, _LANES), 0) * _LANES
            + jax.lax.broadcasted_iota(I32, (_SUBLANES, _LANES), 1)
        )
        member = jnp.zeros((_SUBLANES, _LANES), I32) - 1
        for m in range(num_members):
            member = member + (base8[:, m : m + 1] <= j).astype(I32)
        member = jnp.maximum(member, 0)
        at = jnp.clip(j - jnp.take_along_axis(base8, member, axis=1), 0, _LANES - 1)
        out = jnp.zeros((_SUBLANES, _LANES), I32)
        for m in range(num_members):
            picked = jnp.take_along_axis(
                jnp.broadcast_to(tokens[m : m + 1, :], (_SUBLANES, _LANES)), at, axis=1
            )
            out = jnp.where(member == m, picked, out)
        index_ref[t * _SUBLANES : (t + 1) * _SUBLANES, :] = out
    over_ref[0] = over.astype(I32)
    count_ref[0] = jnp.minimum(sizes.sum(), rows)
    src_ref[0] = recv_sizes.sum()


@functools.partial(
    jax.jit,
    static_argnames=(
        "tp_size",
        "slots_per_rank",
        "rows",
        "routes",
        "order_bits",
        "interpret",
    ),
)
def exchange_plan(
    words,
    ep_rank,
    *,
    tp_size: int,
    slots_per_rank: int,
    rows: int,
    routes: int,
    order_bits: int = 0,
    interpret: bool = False,
) -> ExchangePlan:
    """The capacity exchange's plan from the gathered packed routing.

    ``words`` [D, K, S] int32: member s's route k of token t, id field in the
    high 16 bits (the slot that serves the route, shifted left by
    ``order_bits`` when the field also carries the owner order); an empty slot
    reads id -1. ``ep_rank`` [1]: this rank's EP rank. ``slots_per_rank``: the
    expert slots of a rank (rank = slot // slots_per_rank). ``rows`` /
    ``routes``: the capacity buffers. Same integers as the capacity
    exchange's XLA ops (`_capacity_overflow` and the planned branch of
    `_capacity_exchange`), for every input."""
    num_members, topk, num_tokens = words.shape
    if not exchange_plan_supported(
        num_members=num_members,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        slots_per_rank=slots_per_rank,
    ):
        raise ValueError(
            f"exchange_plan does not take {num_members} members x TP {tp_size}, "
            f"{num_tokens} tokens, top-{topk}, {slots_per_rank} slots per rank"
        )
    members, _, num_words, _ = _shapes(num_members, tp_size, num_tokens, topk)
    index_rows = -(-rows // (_SUBLANES * _LANES)) * _SUBLANES
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    plan = pl.pallas_call(
        functools.partial(
            _plan_kernel,
            num_members=num_members,
            tp_size=tp_size,
            num_tokens=num_tokens,
            topk=topk,
            slots_per_rank=slots_per_rank,
            rows=rows,
            routes=routes,
            order_bits=order_bits,
        ),
        out_shape=[jax.ShapeDtypeStruct((num_members,), I32)] * 4
        + [jax.ShapeDtypeStruct((1,), I32)] * 3
        + [
            jax.ShapeDtypeStruct((index_rows, _LANES), I32),
            jax.ShapeDtypeStruct((members, _LANES), I32),
        ],
        in_specs=[smem, vmem],
        out_specs=[vmem] * 4 + [smem] * 3 + [vmem] * 2,
        scratch_shapes=[
            pltpu.VMEM((topk, members, _LANES), I32),
            pltpu.VMEM((num_words, members, _LANES), I32),
        ],
        interpret=interpret,
        name=f"exchange_plan-d_{num_members}-t_{tp_size}-s_{num_tokens}-r_{rows}",
    )(ep_rank, words)
    start, sizes, landing, recv, over, count, src, index, in_row = plan
    return ExchangePlan(
        overflow=over[0] != 0,
        send_start=start,
        send_sizes=sizes,
        send_landing=landing,
        recv_sizes=recv,
        count=count,
        src_rows=src,
        send_index=index.reshape(-1)[:rows],
        in_row=in_row[:num_members, :num_tokens],
    )
