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
"""SparseCore dispatch of the capacity exchange's fp8 token rows.

The capacity exchange sends each token row that reaches a member once, the
receiver copies its routes' rows into route order, and only then can the
expert kernel start. Here a SparseCore kernel sends one row per route straight
to its row of the receiver's route order (the order of `local_route_plan`):
local expert by local expert ("waves"), each wave signalling its own DMA
semaphore in the receiver's TensorCore memory. The SC kernel exits as soon as
its DMAs are issued, so a TensorCore kernel can compute on the waves that
have landed while later ones are in flight.

Protocol, per call:
- The receive buffer and the wave semaphores are outputs of the dispatch
  kernel, so they are a member's only once its kernel runs: a barrier over
  the members (barrier semaphore with a collective id) keeps peers from
  writing earlier.
- A count-based barrier is exact only if every member hears from every other
  member in every call: one that expects nothing from a peer could otherwise
  finish, enter the next call and signal its barrier while the peer still
  counts this one. A member that has no route for a peer therefore sends it
  one sentinel row on the extra semaphore ``num_waves``, into a row past the
  routes' rows.
- Mosaic puts a DMA's send semaphore on its source's core type, so the send
  semaphore is an SC semaphore, output by the dispatch kernel and waited for
  every row sent by `sc_drain`, which also holds the source rows alive.
- The consumer waits ``group_sizes[e]`` rows on wave semaphore ``e`` and
  ``sentinels`` rows on semaphore ``num_waves``, one row's bytes per row.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.megablox.exchange_plan import _divider, _sublane_scan_excl

I32 = jnp.int32
# Header words of a send table ([#sends, 0, ...]); the sends follow, so the
# table stays whole 16-word SMEM rows.
TABLE_HEADER = 16
# A send packs (wave, destination - this member mod members, source token,
# landing row) high to low, so that sorting the words orders the sends.
_LAND_BITS = 14
_SRC_BITS = 6
_DST_BITS = 4
_WAVE_BITS = 4
_SRC_SHIFT = _LAND_BITS
_DST_SHIFT = _SRC_SHIFT + _SRC_BITS
_WAVE_SHIFT = _DST_SHIFT + _DST_BITS
_NO_SEND = (1 << 31) - 1
# Mosaic's tensor core axis of the wave semaphores' memory.
_TC_AXIS = "tc"
_SUBLANES = 8
_LANES = 128
_HALF = _LANES // 2
_EMPTY_WORD = -1 << 16
# 4-bit counters per int32; the even (odd) counters masked out fit bytes.
_COUNTERS = 8
_NIBBLES = 0x0F0F0F0F
# A table slot (< 1024: sends and sentinels) below the landing row.
_SLOT_BITS = 10


class SendPlan(NamedTuple):
    """This member's sends of one SC dispatch, and what it receives."""

    table: jax.Array  # [TABLE_HEADER + M] int32: #sends, then packed sends
    group_sizes: jax.Array  # [L] int32: rows each wave lands here
    sentinels: jax.Array  # [1] int32: sentinel rows landing here


def sc_dispatch_supported(
    *, num_members: int, num_tokens: int, local_num_experts: int, recv_rows: int
) -> bool:
    """Whether a send packs into one word: sentinel rows included, the
    landing rows, tokens, members and waves fit their fields."""
    return (
        recv_rows + num_members <= 1 << _LAND_BITS
        and num_tokens <= 1 << _SRC_BITS
        and num_members <= 1 << _DST_BITS
        and local_num_experts + 1 <= 1 << _WAVE_BITS
    )


def recv_buffer_rows(recv_rows: int, num_members: int) -> int:
    """Rows of the receive buffer: ``recv_rows`` route rows, then one sentinel
    row per member, rounded up to whole 16-row groups."""
    return pl.cdiv(recv_rows + num_members, 16) * 16


def _check_supported(num_members, num_tokens, num_local, recv_rows):
    if not sc_dispatch_supported(
        num_members=num_members,
        num_tokens=num_tokens,
        local_num_experts=num_local,
        recv_rows=recv_rows,
    ):
        raise ValueError(
            f"sc_send_plan does not take {num_members} members, {num_tokens} "
            f"tokens, {num_local} local experts, {recv_rows} receive rows"
        )


@functools.partial(
    jax.jit,
    static_argnames=("local_num_experts", "tp_size", "skip_padded", "recv_rows"),
)
def sc_send_plan_xla(
    words,
    ep_rank,
    fits,
    *,
    local_num_experts: int,
    tp_size: int,
    skip_padded: bool,
    recv_rows: int,
) -> SendPlan:
    """`sc_send_plan` as XLA ops, with pairwise compares over the routes."""
    num_members, topk, num_tokens = words.shape
    num_local = local_num_experts
    _check_supported(num_members, num_tokens, num_local, recv_rows)
    columns = num_members * num_local
    ep = ep_rank[0]
    me = ep // tp_size
    ids = words >> 16
    rank = ids // num_local
    computed = (
        (ids >= 0)
        & (rank < num_members * tp_size)
        & (rank % tp_size == ep % tp_size)
        & (fits[0] != 0)
    )
    if skip_padded:
        # A zero (subnormals flush to zero) or non-finite bf16 weight.
        exponent = words & 0x7F80
        computed = computed & (exponent != 0) & (exponent != 0x7F80)
    # (destination member, local expert) of each route; `columns`: none.
    col = jnp.where(computed, (rank // tp_size) * num_local + ids % num_local, columns)

    column = jnp.arange(columns, dtype=I32)
    per_member = jnp.sum(col[..., None] == column, axis=(1, 2), dtype=I32)  # [D, C]
    total = per_member.sum(axis=0).reshape(num_members, num_local)
    earlier = jnp.arange(num_members, dtype=I32)[:, None] < me
    before = jnp.sum(jnp.where(earlier, per_member, 0), axis=0)
    start = (jnp.cumsum(total, axis=1, dtype=I32) - total).reshape(-1)
    base = start + before
    mine_here = per_member.reshape(num_members, num_members, num_local)
    from_member = lax.dynamic_index_in_dim(mine_here, me, axis=1, keepdims=False)
    group_sizes = lax.dynamic_index_in_dim(total, me, keepdims=False)
    sentinels = jnp.sum(from_member.sum(axis=1) == 0, dtype=I32)[None]

    # This member's routes in (token, k) order.
    mine = lax.dynamic_index_in_dim(col, me, keepdims=False).T.reshape(-1)
    n = mine.shape[0]
    order = jnp.arange(n, dtype=I32)
    within = jnp.sum(
        (mine[None, :] == mine[:, None]) & (order[None, :] < order[:, None]),
        axis=1,
        dtype=I32,
    )
    land = (
        jnp.sum(jnp.where(mine[:, None] == column, base, 0), axis=1, dtype=I32) + within
    )
    dest = mine // num_local
    rot = (dest - me) % num_members
    send = jnp.where(
        mine < columns,
        jnp.left_shift(mine % num_local, _WAVE_SHIFT)
        | jnp.left_shift(rot, _DST_SHIFT)
        | jnp.left_shift(order // topk, _SRC_SHIFT)
        | land,
        _NO_SEND,
    )
    sends_to = jnp.sum(
        dest[:, None] == jnp.arange(num_members, dtype=I32), axis=0, dtype=I32
    )
    peer = jnp.arange(num_members, dtype=I32)
    sentinel = jnp.where(
        sends_to == 0,
        jnp.left_shift(num_local, _WAVE_SHIFT)
        | jnp.left_shift((peer - me) % num_members, _DST_SHIFT)
        | (recv_rows + me),
        _NO_SEND,
    )
    send = jnp.concatenate([send, sentinel])
    # The sends' words are distinct: each one's rank is its table slot.
    slot = jnp.sum(send[None, :] < send[:, None], axis=1, dtype=I32)
    width = pl.cdiv(send.shape[0], TABLE_HEADER) * TABLE_HEADER
    live = send != _NO_SEND
    body = jnp.sum(
        jnp.where(
            live[:, None] & (slot[:, None] == jnp.arange(width, dtype=I32)),
            send[:, None],
            0,
        ),
        axis=0,
        dtype=I32,
    )
    header = jnp.zeros((TABLE_HEADER,), I32).at[0].set(jnp.sum(live, dtype=I32))
    return SendPlan(jnp.concatenate([header, body]), group_sizes, sentinels)


def _send_plan_kernel(
    ep_ref,
    fits_ref,
    words_ref,
    table_ref,
    gs_ref,
    sent_ref,
    slabs_ref,
    mine_ref,
    *,
    local_num_experts,
    tp_size,
    skip_padded,
    recv_rows,
):
    num_members, topk, num_tokens = words_ref.shape
    num_local = local_num_experts
    ranks = num_members * tp_size
    rank_of = _divider(num_local, ranks * num_local)
    tp_of = _divider(tp_size, ranks)
    assert None not in (rank_of, tp_of)  # sc_send_plan checked
    # A (local expert, destination - this member mod members) bin is counter
    # rot % 8 of word local * groups + rot // 8, row 8 * word + rot % 8 of the
    # bins: the bins in send order.
    groups = -(-num_members // _COUNTERS)
    dst_pad = groups * _COUNTERS
    num_words = num_local * groups
    bins = -(-num_words * _COUNTERS // 16) * 16
    ep = ep_ref[0]
    me = ep // tp_size
    # No TP rank matches on a step that overflows: nothing is computed.
    my_tp = jnp.where(fits_ref[0] != 0, ep % tp_size, -1)

    def decode(word):
        ids = word >> 16
        routed = (ids >= 0) & (ids < ranks * num_local)
        rank = rank_of(jnp.where(routed, ids, 0))
        dest = tp_of(rank)
        computed = routed & (rank - dest * tp_size == my_tp)
        if skip_padded:
            # A zero (subnormals flush to zero) or non-finite bf16 weight.
            exponent = word & 0x7F80
            computed = computed & (exponent != 0) & (exponent != 0x7F80)
        rot = dest - me
        return (
            computed,
            ids - rank * num_local,
            jnp.where(rot < 0, rot + num_members, rot),
        )

    def iota(shape, axis):
        return lax.broadcasted_iota(I32, shape, axis)

    # Every member's routes, slab k [8, 128]: member r's tokens in the first
    # lanes of row r, member r + 8's from lane 64.
    shape = (_SUBLANES, _LANES)
    row = iota(shape, 0)
    lane = iota(shape, 1)
    upper = lane >= _HALF
    low = min(num_members, _SUBLANES)
    for h, start, n in ((0, 0, low), (1, _SUBLANES, num_members - low)):
        for k in range(topk if n else 0):
            slabs_ref[h, k] = jnp.full(shape, _EMPTY_WORD, I32)
            slabs_ref[h, k, :n, :num_tokens] = words_ref[start : start + n, k, :]
    # Per token: 4-bit counters of its computed routes per bin (at most top-k
    # in all, so no sum over its bins overflows a counter).
    acc = [jnp.zeros(shape, I32) for _ in range(num_words)]
    for k in range(topk):
        word = slabs_ref[0, k]
        if num_members > _SUBLANES:
            word = jnp.where(upper, pltpu.roll(slabs_ref[1, k], _HALF, 1), word)
        computed, local, rot = decode(word)
        index = jnp.where(computed, local * groups + (rot >> 3), -1)
        one = jnp.left_shift(1, (rot & (_COUNTERS - 1)) << 2)
        for w in range(num_words):
            acc[w] = acc[w] + jnp.where(index == w, one, 0)

    # A bin's base row at its destination: the routes there of lower local
    # experts, and of this local expert from earlier members.
    earlier = row + jnp.where(upper, _SUBLANES, 0) < me
    base = [None] * num_words
    for g in range(groups):
        below = jnp.zeros(shape, I32)
        for e in range(num_local):
            w = e * groups + g
            base[w] = below + jnp.where(earlier, acc[w], 0)
            below = below + acc[w]
    # Summed over the rows, a word's counters fit bytes; row j of its plane
    # is counter j. Then the MXU sums the lanes (0/1 weights, exact).
    odd = (row & 1) != 0
    byte_shift = (row >> 1) << 3

    def plane(x):
        lo = jnp.sum(x & _NIBBLES, axis=0, keepdims=True)
        hi = jnp.sum((x >> 4) & _NIBBLES, axis=0, keepdims=True)
        return (jnp.where(odd, hi, lo) >> byte_shift) & 0xFF

    planes = [plane(x) for x in base]
    planes += [jnp.zeros(shape, I32)] * (bins // _SUBLANES - num_words)
    # This member's group sizes: the routes of rot 0, local expert per row.
    gshape = (16, _LANES)
    here = jnp.zeros(gshape, I32)
    for e in range(num_local):
        n = jnp.sum(acc[e * groups] & 0xF, axis=0, keepdims=True)
        here = jnp.where(iota(gshape, 0) == e, n, here)
    ones = jnp.ones((_LANES, _LANES), jnp.bfloat16)
    sums = _count_mxu(jnp.concatenate(planes + [here]).astype(jnp.bfloat16), ones)
    base_row = sums[:bins]
    diag = iota(gshape, 0) == iota(gshape, 1)
    group = jnp.sum(jnp.where(diag, sums[bins:], 0), axis=0, keepdims=True)
    gs_ref[...] = group.astype(I32)[0, :num_local]
    # The members with no route here: each sends one sentinel row.
    routes_here = sum(acc[e * groups] & 0xF for e in range(num_local))
    silent = 0
    for half, first in ((False, 0), (True, _SUBLANES)):
        n = jnp.sum(jnp.where(upper == half, routes_here, 0), axis=1, keepdims=True)
        member = iota((_SUBLANES, 1), 0) + first
        silent = silent + ((n == 0) & (member < num_members)).astype(I32)
    sent_ref[0] = jnp.sum(silent)

    # This member's routes: route k of token t at row k // 2, lane t (even k)
    # or 64 + t (odd k). Per route: its bin, and the token's routes of lower
    # k in the bin.
    num_pairs = -(-topk // 2)
    mine_ref[...] = jnp.full((2, *shape), _EMPTY_WORD, I32)
    for k in range(topk):
        p = k // 2
        mine_ref[k % 2, p : p + 1, :num_tokens] = words_ref[me, k : k + 1, :]
    computed, local, rot = decode(
        jnp.where(upper, pltpu.roll(mine_ref[1], _HALF, 1), mine_ref[0])
    )
    col = jnp.where(computed, local * dst_pad + rot, -1)
    other = pltpu.roll(col, _HALF, 1)
    same = (upper & (other == col)).astype(I32)
    for d in range(1, num_pairs):
        lower_k = row >= d
        for x in (col, other):
            same = same + (lower_k & (pltpu.roll(x, d, 0) == col)).astype(I32)

    # Per bin and token: the routes of earlier tokens in the bin (MXU over a
    # strictly lower token mask), and the bin's routes (all tokens).
    bin_row = iota((bins, _LANES), 0)
    hist = jnp.zeros((bins, _LANES), I32)
    for p in range(num_pairs):
        hist = hist + (bin_row == col[p : p + 1]).astype(I32)
    hist = hist + pltpu.roll(hist, _HALF, 1)
    src = iota((_LANES, 2 * _LANES), 0)
    dst = iota((_LANES, 2 * _LANES), 1)
    token_mask = (src < _HALF) & ((dst >= _LANES) | (src < (dst & (_HALF - 1))))
    counts = _count_mxu(hist.astype(jnp.bfloat16), token_mask.astype(jnp.bfloat16))
    earlier_tokens = counts[:, :_LANES]
    per_bin = counts[:, _LANES:].astype(I32)
    # The routes of lower bins (the send order), bytes through the MXU.
    bin_lane = iota((bins, _LANES), 1)
    split = jnp.where(
        bin_lane == 0, per_bin & 0xFF, jnp.where(bin_lane == 1, per_bin >> 8, 0)
    )
    depth = -(-bins // _LANES) * _LANES
    if depth > bins:
        split = jnp.concatenate([split, jnp.zeros((depth - bins, _LANES), I32)])
    lower = iota((bins, depth), 1) < iota((bins, depth), 0)
    before = _count_mxu(lower.astype(jnp.bfloat16), split.astype(jnp.bfloat16))
    before = before[:, 0:1] + 256 * before[:, 1:2]
    # Per route: landing row (high bits) and table slot (low bits), read
    # from its bin's row.
    key = (earlier_tokens + base_row) * (1 << _SLOT_BITS) + earlier_tokens + before
    key = key.astype(I32)
    found = jnp.zeros(shape, I32)
    for p in range(num_pairs):
        hit = jnp.where(bin_row == col[p : p + 1], key, 0)
        found = jnp.where(row == p, jnp.sum(hit, axis=0, keepdims=True), found)
    land = (found >> _SLOT_BITS) + same
    slot = (found & ((1 << _SLOT_BITS) - 1)) + same
    dst_bits = dst_pad.bit_length() - 1
    send = (
        jnp.left_shift(col >> dst_bits, _WAVE_SHIFT)
        | jnp.left_shift(col & (dst_pad - 1), _DST_SHIFT)
        | jnp.left_shift(lane & (_HALF - 1), _SRC_SHIFT)
        | land
    )
    pos = jnp.where(col >= 0, TABLE_HEADER + slot, -1)

    # A sentinel per destination this member sends nothing, after the routes.
    sends = per_bin[:dst_pad]
    for e in range(1, num_local):
        sends = sends + per_bin[e * dst_pad : (e + 1) * dst_pad]
    dshape = (dst_pad, _LANES)
    none = ((sends == 0) & (iota(dshape, 0) < num_members)).astype(I32)
    dst_diag = iota(dshape, 0) == iota(dshape, 1)
    num_routes = jnp.sum(sends, axis=0, keepdims=True)
    num_none = jnp.sum(none, axis=0, keepdims=True)
    none_at = jnp.sum(jnp.where(dst_diag, none, 0), axis=0, keepdims=True)
    none_before = jnp.sum(
        jnp.where(dst_diag, _sublane_scan_excl(none), 0), axis=0, keepdims=True
    )
    s_pos = jnp.where(none_at != 0, TABLE_HEADER + num_routes + none_before, -1)
    s_send = (
        jnp.left_shift(num_local, _WAVE_SHIFT)
        | jnp.left_shift(lane[:1], _DST_SHIFT)
        | (recv_rows + me)
    )

    # The table through the MXU: byte b of each send to row (b, slot // 128),
    # lane slot % 128.
    pos = jnp.concatenate([pos[p : p + 1] for p in range(num_pairs)] + [s_pos], 1)
    send = jnp.concatenate([send[p : p + 1] for p in range(num_pairs)] + [s_send], 1)
    width = pos.shape[1]
    hit = iota((_SUBLANES, width), 0) == pos >> 7
    by_byte = jnp.concatenate(
        [jnp.where(hit, (send >> (8 * b)) & 0xFF, 0) for b in range(4)]
    )
    at_lane = (iota((_LANES, width), 0) == (pos & (_LANES - 1))) & (pos >= 0)
    table = lax.dot_general(
        by_byte.astype(jnp.bfloat16),
        at_lane.astype(jnp.bfloat16),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(I32)
    table = sum(
        jnp.left_shift(table[b * _SUBLANES : (b + 1) * _SUBLANES], 8 * b)
        for b in range(4)
    )
    table = jnp.where((row == 0) & (lane == 0), num_routes + num_none, table)
    table_ref[...] = table[: table_ref.shape[0] // _LANES].reshape(-1)


def _count_mxu(lhs, rhs):
    return jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)


def _staged_kernel(*refs, num_outs, **kwargs):
    """`_send_plan_kernel` writing its outputs to VMEM / SMEM stages, then
    copying them to the HBM outputs."""
    ins, outs = refs[:3], refs[3 : 3 + num_outs]
    stages = refs[3 + num_outs : 3 + 2 * num_outs]
    sem = refs[-1]
    _send_plan_kernel(*ins, *stages, *refs[3 + 2 * num_outs : -1], **kwargs)
    copies = [
        pltpu.make_async_copy(stage, out, sem.at[0]) for stage, out in zip(stages, outs)
    ]
    for c in copies:
        c.start()
    for c in copies:
        c.wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "local_num_experts",
        "tp_size",
        "skip_padded",
        "recv_rows",
        "interpret",
    ),
)
def sc_send_plan(
    words,
    ep_rank,
    fits,
    *,
    local_num_experts: int,
    tp_size: int,
    skip_padded: bool,
    recv_rows: int,
    interpret: bool = False,
) -> SendPlan:
    """The send table of this member's SC dispatch, from the gathered routing.

    ``words`` [D, K, S] int32: member s's route k of token t, expert id in the
    high 16 bits, bf16 weight in the low 16 (as `local_route_plan` takes it).
    ``ep_rank`` [1]: this rank's EP rank, DP position ``ep_rank // tp_size``.
    ``fits`` [1] (0: the step overflows, nothing is sent but sentinels).
    ``skip_padded``: a route with a zero or non-finite weight is not computed.

    A route of this member's token t goes to the member owning its expert on
    this TP rank when that member computes it (as `local_route_plan` decides),
    landing at the route's position in that member's route order: (local
    expert, source member, token), a token naming an expert twice taking
    consecutive rows in k order. Sends are wave-major (local expert), then by
    destination starting after this member, then by token.

    One kernel (top-k at most 15), the outputs of `sc_send_plan_xla` bitwise:
    4-bit per-token counters per (local expert, destination) bin give each
    bin's base row at its destination; this member's routes find their bin's
    base, earlier routes and table slot through one-hot matmuls.
    """
    num_members, topk, num_tokens = words.shape
    num_local = local_num_experts
    _check_supported(num_members, num_tokens, num_local, recv_rows)
    ranks = num_members * tp_size
    if (
        topk > 15
        or _divider(num_local, ranks * num_local) is None
        or _divider(tp_size, ranks) is None
    ):
        raise ValueError(
            f"sc_send_plan does not take top-{topk} or {num_local} local "
            f"experts on TP {tp_size}: use at most top-15"
        )
    table_len = TABLE_HEADER + pl.cdiv(topk * num_tokens + num_members, 16) * 16
    out_shape = [
        jax.ShapeDtypeStruct((pl.cdiv(table_len, _LANES) * _LANES,), I32),
        jax.ShapeDtypeStruct((num_local,), I32),
        jax.ShapeDtypeStruct((1,), I32),
    ]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    # Staged in VMEM / SMEM and copied to HBM by the kernel: left to XLA,
    # each small output is copied to HBM after the kernel, one by one.
    stages = [pltpu.VMEM(o.shape, I32) for o in out_shape[:2]] + [pltpu.SMEM((1,), I32)]
    if interpret:
        out_specs = [pl.BlockSpec(memory_space=pl.ANY)] * 3
    else:
        out_shape = [pltpu.HBM(o.shape, o.dtype) for o in out_shape]
        out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * 3
    table, group_sizes, sentinels = pl.pallas_call(
        functools.partial(
            _staged_kernel,
            num_outs=3,
            local_num_experts=num_local,
            tp_size=tp_size,
            skip_padded=skip_padded,
            recv_rows=recv_rows,
        ),
        out_shape=out_shape,
        in_specs=[smem, smem, vmem],
        out_specs=out_specs,
        scratch_shapes=stages
        + [
            pltpu.VMEM((2, topk, _SUBLANES, _LANES), I32),
            pltpu.VMEM((2, _SUBLANES, _LANES), I32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        interpret=interpret,
        name=f"sc_send_plan-d_{num_members}-s_{num_tokens}-k_{topk}-l_{num_local}",
    )(ep_rank.astype(I32), fits.astype(I32), words)
    if table.shape[0] != table_len:
        table = table[:table_len]
    return SendPlan(table, group_sizes, sentinels)


def _sc_mesh():
    return plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)


def sc_dispatch(
    src_words,
    table,
    *,
    axis: str,
    num_members: int,
    num_waves: int,
    recv_rows: int,
    collective_id: int,
):
    """Sends this member's rows by its `sc_send_plan` table; call inside
    shard_map on every member of mesh axis ``axis`` (``num_members``), same
    sizes on all.

    ``src_words`` [S, *row] uint32 (in HBM): the source rows, token-major
    (`fp8_source_rows`). Returns ``(recv, wave_sems, send_sem)``: the receive
    buffer [`recv_buffer_rows`, *row] (route rows from 0, sentinel rows from
    ``recv_rows``), ``num_waves + 1`` TC DMA semaphores (the last one counts
    sentinel rows) and the SC DMA semaphore of this member's sends, for
    `sc_drain`. The rows may still be in flight when this returns: a consumer
    waits for them on the wave semaphores.
    """
    sc_mesh = _sc_mesh()
    tc_mesh = pltpu.create_tensorcore_mesh(_TC_AXIS, num_cores=1)

    def body(src_hbm, tbl_hbm, recv, wave_sems, send_sems, tbl_s, tbl_sem):
        tbl_copy = pltpu.make_async_copy(tbl_hbm, tbl_s, tbl_sem)
        tbl_copy.start()
        me = lax.axis_index(axis)
        if num_members > 1:
            barrier = pltpu.get_barrier_semaphore()
            for k in range(1, num_members):
                peer = lax.rem(me + k, num_members)
                pl.semaphore_signal(barrier, 1, device_id={axis: peer})
            pl.semaphore_wait(barrier, num_members - 1)
        tbl_copy.wait()

        @pl.loop(0, tbl_s[0])
        def _(i):
            word = tbl_s[TABLE_HEADER + i]
            srl = lax.shift_right_logical
            land = word & ((1 << _LAND_BITS) - 1)
            src = srl(word, _SRC_SHIFT) & ((1 << _SRC_BITS) - 1)
            rot = srl(word, _DST_SHIFT) & ((1 << _DST_BITS) - 1)
            wave = srl(word, _WAVE_SHIFT)
            pltpu.async_remote_copy(
                src_hbm.at[src],
                recv.at[land],
                send_sems.at[0],
                wave_sems.at[wave],
                device_id={axis: lax.rem(me + rot, num_members), _TC_AXIS: 0},
            )

    return pl.kernel(
        body,
        out_type=(
            pltpu.HBM(
                (recv_buffer_rows(recv_rows, num_members), *src_words.shape[1:]),
                src_words.dtype,
            ),
            pltpu.SemaphoreType.DMA((num_waves + 1,)) @ tc_mesh,
            pltpu.SemaphoreType.DMA((1,)) @ sc_mesh,
        ),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM(table.shape, I32), pltpu.SemaphoreType.DMA(())],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name="sc_dispatch",
    )(src_words, table)


def _wait_rows(n, wait, limit):
    """Waits ``n`` (dynamic) rows by static power-of-two waits ``wait(b)`` of
    at most min(64, limit) rows."""
    big = 1
    while big * 2 <= min(64, limit):
        big *= 2

    @pl.loop(0, n // big)
    def _(i):
        wait(big)

    b = big // 2
    while b:
        pl.when((n & b) != 0)(functools.partial(wait, b))
        b //= 2


def sc_drain(send_sem, src_words, table, *, axis: str):
    """Waits for every row this member's `sc_dispatch` over mesh axis
    ``axis`` sent (``table``'s count) from ``src_words``, which it keeps alive
    until then. Returns a copy of ``table``: a value for a later op to take,
    so that XLA keeps this kernel and the source rows."""
    sc_mesh = _sc_mesh()
    sem_ref = jax.new_ref(send_sem, memory_space=pltpu.SEMAPHORE @ sc_mesh)

    def body(sem_r, src_hbm, tbl_hbm, out_hbm, tbl_s, tbl_sem):
        pltpu.async_copy(tbl_hbm, tbl_s, tbl_sem).wait()

        def wait(b):
            # A descriptor of b rows: its wait counts b rows' bytes.
            rows = src_hbm.at[pl.ds(0, b)]
            pltpu.make_async_remote_copy(
                rows, rows, sem_r.at[0], sem_r.at[0], device_id={axis: 0}
            ).wait_send()

        _wait_rows(tbl_s[0], wait, src_words.shape[0])
        pltpu.async_copy(tbl_s, out_hbm, tbl_sem).wait()

    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(table.shape, I32),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM(table.shape, I32), pltpu.SemaphoreType.DMA(())],
        name="sc_dispatch_drain",
    )(sem_ref, src_words, table)
