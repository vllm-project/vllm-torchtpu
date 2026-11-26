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
"""The capacity exchange's route plan in one kernel.

The expert side of the capacity exchange orders this rank's local routes by
(local expert, receive row) with one sort over every route of the DP group
(``D * S * K`` keys, almost all of them for other ranks), the fp8 exchange
scatters each received token's scale to its receive row, and the owner
combine orders its owned tokens' routes. As XLA ops that is a sort and a few
dozen small fusions, all while the token rows are in flight. This kernel
computes the same arrays from the gathered routing words in VMEM, with a
counting sort.

Per token, 4-bit counters per local expert (built over the top-k route slabs)
give its routes per expert and, through their prefix sums, the rank of each
local route among the token's routes, which names its destination row. The
routes of one local expert from consecutive tokens are consecutive in the
sorted order: with the tokens in member-major order over whole lane rows,
each (expert, row) of tokens is compacted to the left and rotated to where
its run starts, and the rows are summed into the sorted positions. Equal sort
keys (an expert named twice by one token) take one position per route: the
runs are expanded after.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.exchange_plan import (
    _divider,
    _lane_scan,
    _sublane_scan_excl,
)

I32 = jnp.int32
_LANES = 128
_SUBLANES = 8
_EMPTY_WORD = -1 << 16
# 4-bit route counters per int32: a token's routes per local expert, at most
# top-k (<= 15) in all, so their prefix sums stay within a counter too.
_COUNTERS = 8
_NIBBLE_PREFIX = 0x11111111
# A left move within the lanes takes 7 bits, carried above the payload.
_MOVE_BITS = 7
_PAYLOAD_BITS = 32 - _MOVE_BITS
# Expert ids span the signed 16-bit field of a routing word.
_ID_LIMIT = 1 << 15
# float32 1.0: the scale past the sorted routes.
_ONE_F32_BITS = 0x3F800000
# The owner order's key of a route that names no expert of the mesh, plus k.
_UNROUTED_KEY = 1 << 24


class OwnerPlan(NamedTuple):
    """The owner combine's arrays of this rank (`owner_plan` in
    ragged_a2a_moe.py): ``N`` = ceil(S / tp) owned tokens."""

    tp_rank: jax.Array  # [K, N]: per owner-order position (`_owner_routes`)
    dp_pos: jax.Array  # [K, N]
    sent: jax.Array  # [K, N] int32
    weight: jax.Array  # [K, N] f32
    expected: jax.Array  # [1]: rows other ranks send here (0 on overflow)
    shared_member: jax.Array  # [N * tp]: shared-expert partials' destinations
    shared_row: jax.Array  # [N * tp]
    shared_counts: jax.Array  # [2]: shared rows sent and received (0 on overflow)


class LocalRoutePlan(NamedTuple):
    """`route_plan`'s outputs (``routes`` sorted routes, ``L`` local experts),
    the fp8 exchange's receive-row scales and the owner combine's arrays."""

    token_sorted: jax.Array  # [routes]: receive row (``rows`` past the routes)
    group_sizes: jax.Array  # [L]
    dest_member: jax.Array  # [routes] (0 past the routes)
    dest_row: jax.Array  # [routes] (0 past the routes)
    row_scale: jax.Array | None  # [rows] f32 (1 where no token lands)
    owner: OwnerPlan | None
    route_scale: jax.Array | None = None  # [routes] f32 (1 past the routes)


class _Dests(NamedTuple):
    members: int  # destination members
    recv_rows: int  # receive rows per member
    row_step: int  # receive rows per route position


def _dests(*, num_members, num_tokens, topk, tp_size, owner) -> _Dests:
    """`owner_dests` / `streamed_dests` sizes (ragged_a2a_moe.py)."""
    if owner:
        blk = -(-num_tokens // tp_size)
        return _Dests(num_members * tp_size, blk * topk, blk)
    return _Dests(num_members, num_tokens * topk, num_tokens)


def _field_bits(rows: int, dests: _Dests) -> tuple[int, int, int]:
    """Bits of a route's receive row, destination member and destination row,
    as `route_plan` packs them."""
    return (
        max(1, (rows - 1).bit_length()),
        max(1, (dests.members - 1).bit_length()),
        max(1, (dests.recv_rows - 1).bit_length()),
    )


def _floor_divider(divisor: int):
    """``x // divisor`` (floor) for expert-id-sized ``x`` of either sign, or
    None."""
    div = _divider(divisor, _ID_LIMIT)
    if div is None:
        return None
    return lambda x: jnp.where(
        x >= 0, div(jnp.maximum(x, 0)), -div(jnp.maximum(-x - 1, 0)) - 1
    )


def local_route_plan_supported(
    *,
    num_members: int,
    num_tokens: int,
    topk: int,
    local_num_experts: int,
    tp_size: int,
    rows: int,
    owner: bool,
    owner_plan: bool = False,
) -> bool:
    """Whether the kernel takes this exchange: at most 128 tokens per member,
    top-k at most 15 (4-bit counters), a route's receive row and destination
    packing below the lane-move bits, and exact multiply-shift divisions;
    with ``owner_plan`` (owner destinations), the engine's padded tokens
    within the lanes."""
    if owner_plan and not (
        owner
        and -(-num_tokens // tp_size) * tp_size <= _LANES
        and _floor_divider(local_num_experts) is not None
        and _floor_divider(tp_size) is not None
    ):
        return False
    dests = _dests(
        num_members=num_members,
        num_tokens=num_tokens,
        topk=topk,
        tp_size=tp_size,
        owner=owner,
    )
    ranks = num_members * tp_size
    return (
        num_tokens <= _LANES
        and topk <= 15
        and sum(_field_bits(rows, dests)) <= _PAYLOAD_BITS
        and _divider(local_num_experts, ranks * local_num_experts) is not None
        and _divider(tp_size, ranks) is not None
        and _divider(dests.row_step, _LANES) is not None
    )


def _compact(word, vals=(), *, at):
    """Moves each lane left by its move (bits ``[at, at + 7)`` of ``word``),
    carrying ``vals`` along. The moves are the counts of empty lanes (zero
    word and values) before each live lane, so the live lanes of every row end
    up at its first lanes, in order. Low bits first: no two lanes ever land on
    one, and no lane wraps around."""
    d = 1
    while d < _LANES:
        bit = d << at
        if bit >= 1 << 31:
            bit -= 1 << 32
        moves = (word & bit) != 0
        incoming = pltpu.roll(word, _LANES - d, 1)
        arrives = (incoming & bit) != 0
        vals = [
            jnp.where(arrives, pltpu.roll(v, _LANES - d, 1), jnp.where(moves, 0, v))
            for v in vals
        ]
        word = jnp.where(arrives, incoming, jnp.where(moves, 0, word))
        d *= 2
    return word, vals


def _rotate(x, amount):
    """Each row of ``x`` rolled right by its ``amount`` (0..127, the same over
    the row)."""
    for b in range(_MOVE_BITS):
        x = jnp.where(((amount >> b) & 1) != 0, pltpu.roll(x, 1 << b, 1), x)
    return x


def _flat_shift(x, d, flat):
    """``x`` [R, 128] read in row-major order, moved ``d`` positions later
    (zeros in front, positions past the end dropped)."""
    rows, lanes = divmod(d, _LANES)
    if lanes:
        lane = jax.lax.broadcasted_iota(I32, x.shape, 1)
        moved = pltpu.roll(x, lanes, 1)
        x = jnp.where(lane >= lanes, moved, pltpu.roll(moved, 1, 0))
    if rows:
        x = pltpu.roll(x, rows, 0)
    return jnp.where(flat >= d, x, 0)


def _flat_scan_excl(x):
    """Exclusive prefix sum of ``x`` [R, 128] in row-major order."""
    incl = _lane_scan(x)
    totals = jnp.broadcast_to(incl[:, _LANES - 1 : _LANES], x.shape)
    return _sublane_scan_excl(totals) + incl - x


def _expand(payloads, extra, distinct, total, *, topk):
    """The sorted routes from one entry per distinct key: entry ``q`` (of
    ``distinct``) stands for ``extra[q] + 1`` equal routes. Each entry (of
    every array in ``payloads``) moves ``g`` positions later (``g`` the extra
    routes before it; high bits first, so the entries never meet), then the
    empty positions below ``total`` take the entry before them."""
    shape = payloads[0].shape
    flat = jax.lax.broadcasted_iota(I32, shape, 0) * _LANES + jax.lax.broadcasted_iota(
        I32, shape, 1
    )
    live = (flat < distinct).astype(I32)
    payloads = [jnp.where(live != 0, p, 0) for p in payloads]
    g = _flat_scan_excl(jnp.where(live != 0, extra, 0))
    for b in reversed(range((payloads[0].size - 1).bit_length())):
        d = 1 << b
        moves = (live != 0) & ((g & d) != 0)
        arrives = _flat_shift(moves.astype(I32), d, flat) != 0
        payloads = [
            jnp.where(arrives, _flat_shift(p, d, flat), jnp.where(moves, 0, p))
            for p in payloads
        ]
        g = jnp.where(arrives, _flat_shift(g, d, flat), jnp.where(moves, 0, g))
        live = jnp.where(arrives, 1, jnp.where(moves, 0, live))
    d = 1
    while d < topk:
        payloads = [jnp.where(live != 0, p, _flat_shift(p, d, flat)) for p in payloads]
        live = live | _flat_shift(live, d, flat)
        d *= 2
    return payloads


def _kernel(
    ep_ref,
    fits_ref,
    words_ref,
    *refs,
    num_members,
    topk,
    num_tokens,
    local_num_experts,
    tp_size,
    owner,
    skip_padded,
    rows,
    with_scale,
    owner_plan,
    route_scale,
):
    refs = list(refs)
    scale_ref = refs.pop(0) if with_scale else None
    tok_ref, mem_ref, row_ref, gs_ref = refs[:4]
    rs_ref = refs[4] if with_scale else None
    owner_refs = refs[4 + with_scale : 12 + with_scale] if owner_plan else None
    if route_scale:
        scale_dense_ref = refs.pop()
        scale_rot_ref = refs.pop()
    if owner_plan:
        owned_ref = refs.pop()
    slabs_ref, pad_ref, rot_ref, key_ref, dense_ref, extra_ref = refs[-6:]

    num_local = local_num_experts
    shape = (slabs_ref.shape[1], _LANES)
    ranks = num_members * tp_size
    dests = _dests(
        num_members=num_members,
        num_tokens=num_tokens,
        topk=topk,
        tp_size=tp_size,
        owner=owner,
    )
    token_bits, member_bits, row_bits = _field_bits(rows, dests)
    dest_bits = member_bits + row_bits
    token_mask = (1 << token_bits) - 1
    member_mask = (1 << member_bits) - 1
    row_mask = (1 << row_bits) - 1
    rank_of = _divider(num_local, ranks * num_local)
    tp_of = _divider(tp_size, ranks)
    step_of = _divider(dests.row_step, _LANES)
    assert None not in (rank_of, tp_of, step_of)  # local_route_plan_supported

    ep = ep_ref[0]
    first = ep * num_local
    my_tp = ep % tp_size
    my_key = my_tp * num_members + ep // tp_size
    fits = jnp.zeros(shape, I32) + fits_ref[0]
    row = jax.lax.broadcasted_iota(I32, shape, 0)
    lane = jax.lax.broadcasted_iota(I32, shape, 1)

    # The route slabs [K][members, 128] and the token scales: member rows and
    # token lanes past the group's hold empty routes.
    for k in range(topk):
        slabs_ref[k] = jnp.full(shape, _EMPTY_WORD, I32)
        slabs_ref[k, :num_members, :num_tokens] = words_ref[:, k, :]
    if scale_ref is not None:
        pad_ref[0] = jnp.zeros(shape, I32)
        pad_ref[0, :num_members, :num_tokens] = scale_ref[...]

    # Per token: its routes per local expert (all of them, and the computed
    # ones), and its routes that sort before every local route in the route
    # order of `owner_dests` / `streamed_dests`.
    num_words = -(-num_local // _COUNTERS)
    every = [jnp.zeros(shape, I32) for _ in range(num_words)]
    computed = [jnp.zeros(shape, I32) for _ in range(num_words)]
    before = jnp.zeros(shape, I32)
    for k in range(topk):
        word = slabs_ref[k]
        expert = word >> 16
        local = expert - first
        is_local = (local >= 0) & (local < num_local)
        valid = is_local & (fits != 0)
        if skip_padded:
            # A zero (subnormals flush to zero) or non-finite bf16 weight is
            # not computed.
            exponent = word & 0x7F80
            valid = valid & (exponent != 0) & (exponent != 0x7F80)
        routed = (expert >= 0) & (expert < ranks * num_local)
        rank = rank_of(jnp.where(routed, expert, 0))
        dp = tp_of(rank)
        tp = rank - dp * tp_size
        if owner:
            earlier = tp * num_members + dp < my_key
        else:
            earlier = (tp == my_tp) & (expert < first)
        before = before + (routed & earlier).astype(I32)
        one = jnp.left_shift(1, (local & (_COUNTERS - 1)) << 2)
        slot = local >> 3
        for w in range(num_words):
            hit = slot == w
            every[w] = every[w] + jnp.where(is_local & hit, one, 0)
            computed[w] = computed[w] + jnp.where(valid & hit, one, 0)

    if owner:
        chunk = step_of(lane)
        member = row * tp_size + chunk
        step_row = lane - chunk * dests.row_step
    else:
        member = row
        step_row = lane
    dest = jnp.left_shift(member, row_bits) | (step_row + before * dests.row_step)
    prefix = [w * _NIBBLE_PREFIX for w in every]
    scale = [] if scale_ref is None else [pad_ref[0]]
    # The sort takes the tokens in member-major order over whole lanes, where
    # they fill fewer rows than one row per member.
    sort_shape = (rot_ref.shape[0] // num_local, _LANES)
    if sort_shape != shape:
        dest, *words = _member_major(
            [dest, *computed, *prefix, *scale], num_members, num_tokens, sort_shape
        )
        computed = words[:num_words]
        prefix = words[num_words : 2 * num_words]
        scale = words[2 * num_words :]
    sort_lane = jax.lax.broadcasted_iota(I32, sort_shape, 1)
    # The tokens that reach this rank (some route to its experts) fill the
    # receive rows in member-major order.
    reach = sum((w >> 28) & 0xF for w in prefix) != 0
    in_row = _flat_scan_excl(reach.astype(I32))
    base = jnp.left_shift(in_row, dest_bits) | dest

    def counter(words, e):
        return (words[e // _COUNTERS] >> (4 * (e % _COUNTERS))) & 0xF

    def earlier_local(e):
        """The token's routes to local experts below ``e``."""
        w, i = divmod(e, _COUNTERS)
        n = (prefix[w] >> (4 * i - 4)) & 0xF if i else jnp.zeros(sort_shape, I32)
        for v in range(w):
            n = n + ((prefix[v] >> 28) & 0xF)
        return n

    # The rows of all experts, expert-major: the sorted order. Their prefix
    # counts (along the lanes, then over the rows) are 0/1 sums: exact on the
    # MXU in bf16 with f32 accumulation.
    all_shape = rot_ref.shape
    lane = jax.lax.broadcasted_iota(I32, all_shape, 1)
    square = (_LANES, _LANES)
    upper = (
        jax.lax.broadcasted_iota(I32, square, 0)
        <= jax.lax.broadcasted_iota(I32, square, 1)
    ).astype(jnp.bfloat16)
    depth = -(-all_shape[0] // _LANES) * _LANES
    lower = (
        jax.lax.broadcasted_iota(I32, (all_shape[0], depth), 1)
        < jax.lax.broadcasted_iota(I32, (all_shape[0], depth), 0)
    ).astype(jnp.bfloat16)

    def count_mxu(lhs, rhs):
        return jnp.dot(lhs, rhs, preferred_element_type=jnp.float32).astype(I32)

    def sort_rows(value, carried=None):
        """Per (local expert, row of tokens): ``value(e, count)`` of the
        tokens with a computed route to expert ``e``, compacted and rotated to
        where the row's run starts in the sorted order (`rot_ref`), and each
        lane's sorted row (`key_ref`, -1: none). ``carried``: one more int32
        per token, moved the same way (into `scale_rot_ref`). Returns the
        distinct sort keys."""
        counts = [counter(computed, e) for e in range(num_local)]
        live = jnp.concatenate([(c > 0).astype(I32) for c in counts])
        rank_in_row = count_mxu(live.astype(jnp.bfloat16), upper)
        n = jnp.broadcast_to(rank_in_row[:, _LANES - 1 :], all_shape)
        rows_n = n.astype(jnp.bfloat16)
        if depth > all_shape[0]:
            pad = jnp.zeros((depth - all_shape[0], _LANES), jnp.bfloat16)
            rows_n = jnp.concatenate([rows_n, pad])
        run = count_mxu(lower, rows_n)
        payload = jnp.concatenate([value(e, c) for e, c in enumerate(counts)])
        move = jnp.where(live != 0, lane + 1 - rank_in_row, 0)
        vals = []
        if carried is not None:
            vals = [jnp.where(live != 0, jnp.concatenate([carried] * num_local), 0)]
        word, vals = _compact(
            jnp.where(live != 0, payload | (move << _PAYLOAD_BITS), 0),
            vals,
            at=_PAYLOAD_BITS,
        )
        amount = run & (_LANES - 1)
        rot_ref[...] = _rotate(word & ((1 << _PAYLOAD_BITS) - 1), amount)
        if carried is not None:
            scale_rot_ref[...] = _rotate(vals[0], amount)
        i = (lane - amount) & (_LANES - 1)
        key_ref[...] = jnp.where(i < n, (run + i) >> 7, -1)
        return jnp.max(run + n)

    def place(out_ref, distinct, src_ref=rot_ref):
        """The rotated rows summed into the sorted rows they cover."""

        def body(d, carry):
            hit = jnp.where(key_ref[...] == d, src_ref[...], 0)
            out_ref[pl.ds(d, 1), :] = hit.sum(axis=0, keepdims=True)
            return carry

        num_rows = (distinct + _LANES - 1) // _LANES
        jax.lax.fori_loop(0, jnp.minimum(num_rows, out_ref.shape[0]), body, 0)

    group = jnp.zeros((1, _LANES), I32)
    for e in range(num_local):
        size = counter(computed, e).sum(axis=0, keepdims=True).sum(1, keepdims=True)
        group = jnp.where(sort_lane[:1] == e, size, group)
    gs_ref[...] = group[0, :num_local]
    total = jnp.sum(group)
    distinct = sort_rows(
        lambda e, count: base + earlier_local(e) * dests.row_step,
        scale[0] if route_scale else None,
    )
    place(dense_ref, distinct)
    if route_scale:
        place(scale_dense_ref, distinct, scale_rot_ref)

    @pl.when(total != distinct)
    def _():
        # A token names a local expert more than once: its routes share one
        # sort key and take consecutive positions.
        sort_rows(lambda e, count: count - 1)
        place(extra_ref, distinct)
        expanded = [dense_ref]
        if route_scale:
            expanded.append(scale_dense_ref)
        values = _expand(
            [r[...] for r in expanded], extra_ref[...], distinct, total, topk=topk
        )
        for r, v in zip(expanded, values, strict=True):
            r[...] = v

    flat = jax.lax.broadcasted_iota(
        I32, dense_ref.shape, 0
    ) * _LANES + jax.lax.broadcasted_iota(I32, dense_ref.shape, 1)
    dense = dense_ref[...]
    sorted_route = flat < total
    # The outputs are 1-D, as their readers take them: no relayout in XLA.
    num_out = tok_ref.shape[0] // _LANES
    for ref, value in (
        (tok_ref, jnp.where(sorted_route, (dense >> dest_bits) & token_mask, rows)),
        (mem_ref, jnp.where(sorted_route, (dense >> row_bits) & member_mask, 0)),
        (row_ref, jnp.where(sorted_route, dense & row_mask, 0)),
    ):
        ref[...] = value[:num_out].reshape(-1)
    if route_scale:
        bits = jnp.where(sorted_route, scale_dense_ref[...], _ONE_F32_BITS)
        rs_ref[...] = jax.lax.bitcast_convert_type(bits[:num_out], jnp.float32).reshape(
            -1
        )

    if owner_refs is not None:
        _owner_plan_arrays(
            ep,
            fits_ref[0],
            words_ref,
            owned_ref,
            owner_refs,
            num_members=num_members,
            topk=topk,
            num_tokens=num_tokens,
            local_num_experts=num_local,
            tp_size=tp_size,
            skip_padded=skip_padded,
        )
    if rs_ref is None or route_scale:
        return
    # Each receive row's token scale, as the sorted routes: the reaching
    # tokens' scales compacted per row and rotated to their receive rows.
    rank_in_row = _lane_scan(reach.astype(I32))
    n = jnp.broadcast_to(rank_in_row[:, _LANES - 1 :], sort_shape)
    run = _sublane_scan_excl(n)
    move = jnp.where(reach, sort_lane + 1 - rank_in_row, 0)
    _, (scale,) = _compact(move, [jnp.where(reach, scale[0], 0)], at=0)
    amount = run & (_LANES - 1)
    scale = _rotate(scale, amount)
    i = (sort_lane - amount) & (_LANES - 1)
    key = jnp.where(i < n, (run + i) >> 7, -1)

    for d in range(rs_ref.shape[0] // _LANES):
        hit = key == d
        bits = jnp.where(hit, scale, 0).sum(axis=0, keepdims=True)
        landed = hit.astype(I32).sum(axis=0, keepdims=True)
        rs_ref[pl.ds(d * _LANES, _LANES)] = jnp.where(
            landed > 0, jax.lax.bitcast_convert_type(bits, jnp.float32), 1.0
        ).reshape(-1)


def _member_major(arrays, num_members, num_tokens, shape):
    """Per-token arrays [members, 128] (member rows, token lanes) with the
    tokens in member-major order over ``shape`` (zeros past them)."""
    row = jax.lax.broadcasted_iota(I32, shape, 0)
    lane = jax.lax.broadcasted_iota(I32, shape, 1)
    outs = [jnp.zeros(shape, I32) for _ in arrays]
    for s in range(num_members):
        r, offset = divmod(s * num_tokens, _LANES)
        end = offset + num_tokens
        hit = (row == r) & (lane >= offset) & (lane < end)
        if end > _LANES:
            hit = hit | ((row == r + 1) & (lane < end - _LANES))
        for i, x in enumerate(arrays):
            moved = jnp.broadcast_to(x[s : s + 1], shape)
            if offset:
                moved = pltpu.roll(moved, offset, 1)
            outs[i] = jnp.where(hit, moved, outs[i])
    return outs


def _column_sum(x):
    return x.sum(axis=0, keepdims=True)


def _owner_plan_arrays(
    ep,
    fits,
    words_ref,
    scratch_ref,
    out_refs,
    *,
    num_members,
    topk,
    num_tokens,
    local_num_experts,
    tp_size,
    skip_padded,
):
    """`OwnerPlan` with `owner_plan`'s arithmetic (ragged_a2a_moe.py), in
    ``scratch_ref`` [K rounded up to whole sublanes, 128]."""
    tp_ref, dp_ref, sent_ref, weight_ref, exp_ref, smember_ref, srow_ref = out_refs[:7]
    scount_ref = out_refs[7]
    blk = -(-num_tokens // tp_size)
    ranks = num_members * tp_size
    rank_of = _floor_divider(local_num_experts)
    tp_of = _floor_divider(tp_size)
    chunk_of = _divider(blk, _LANES)
    shape = scratch_ref.shape
    k = jax.lax.broadcasted_iota(I32, shape, 0)
    lane = jax.lax.broadcasted_iota(I32, shape, 1)
    my_tp = ep % tp_size
    dp_pos = ep // tp_size

    # The engine's routes [K, tokens], padded with empty routes, its owned
    # tokens rolled to the first lanes.
    scratch_ref[...] = jnp.full(shape, _EMPTY_WORD, I32)
    scratch_ref[:topk, :num_tokens] = words_ref[dp_pos]
    word = pltpu.roll(scratch_ref[...], (_LANES - my_tp * blk) % _LANES, 1)
    expert = word >> 16
    rank = rank_of(expert)
    valid = (expert >= 0) & (rank < ranks)
    dp = tp_of(rank)
    tp = rank - dp * tp_size
    key = jnp.where(
        valid,
        (tp * num_members + dp) * local_num_experts + expert - rank * local_num_experts,
        _UNROUTED_KEY + k,
    )
    exponent = word & 0x7F80
    finite = exponent != 0x7F80
    weight = jnp.where(
        finite, jax.lax.bitcast_convert_type(jnp.left_shift(word, 16), jnp.float32), 0.0
    )
    sent = valid & (exponent != 0) & finite if skip_padded else valid
    order = jnp.zeros(shape, I32)
    for j in range(topk):
        order = order + (jnp.broadcast_to(key[j : j + 1], shape) < key).astype(I32)
    order = jnp.where(k < topk, order, topk)

    tp_s = jnp.zeros(shape, I32)
    dp_s = jnp.zeros(shape, I32)
    sent_s = jnp.zeros(shape, I32)
    weight_s = jnp.zeros(shape, jnp.float32)
    for j in range(topk):
        pick = order == j
        at = k == j
        tp_s = jnp.where(
            at, _column_sum(jnp.where(pick, jnp.where(valid, tp, tp_size), 0)), tp_s
        )
        dp_s = jnp.where(at, _column_sum(jnp.where(pick, dp, 0)), dp_s)
        sent_s = jnp.where(
            at, (pick & sent).astype(I32).max(axis=0, keepdims=True), sent_s
        )
        weight_s = jnp.where(at, _column_sum(jnp.where(pick, weight, 0.0)), weight_s)
    tp_ref[...] = tp_s[:topk, :blk]
    dp_ref[...] = dp_s[:topk, :blk]
    sent_ref[...] = sent_s[:topk, :blk]
    weight_ref[...] = weight_s[:topk, :blk]
    remote = sent & (rank != ep) & (k < topk) & (lane < blk)
    exp_ref[0] = jnp.where(fits != 0, jnp.sum(remote.astype(I32)), 0)

    # This rank's shared-expert partial of engine token t goes to the token's
    # owner, row my_tp * blk + t % blk there.
    t = lane[:1]
    chunk = chunk_of(t)
    smember_ref[...] = (dp_pos * tp_size + chunk)[0, : blk * tp_size]
    srow_ref[...] = (my_tp * blk + t - chunk * blk)[0, : blk * tp_size]
    scount_ref[0] = jnp.where(fits != 0, (tp_size - 1) * blk, 0)
    scount_ref[1] = jnp.where(fits != 0, blk * tp_size, 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "local_num_experts",
        "tp_size",
        "owner",
        "skip_padded",
        "rows",
        "routes",
        "owner_plan",
        "route_scale",
        "interpret",
    ),
)
def local_route_plan(
    words,
    ep_rank,
    fits,
    scale=None,
    *,
    local_num_experts: int,
    tp_size: int,
    owner: bool,
    skip_padded: bool,
    rows: int,
    routes: int,
    owner_plan: bool = False,
    route_scale: bool = False,
    interpret: bool = False,
) -> LocalRoutePlan:
    """`route_plan` with `owner_dests` (``owner``) or `streamed_dests`, from
    the gathered packed routing, and the fp8 exchange's receive-row scales.

    ``words`` [D, K, S] int32: member s's route k of token t, expert id in the
    high 16 bits, bf16 weight in the low 16 (an empty slot reads id -1).
    A token that reaches this rank (some route to its experts) lands at
    receive row = its rank among those tokens in (member, token) order, as
    the capacity exchange's plan lands them.
    ``ep_rank`` [1]: this rank's EP rank (first local expert ``ep_rank *
    local_num_experts``). ``fits`` [1] (0: the step overflows, no local
    routes). ``scale`` [D, S] float32 (optional): each token's fp8 scale.
    ``skip_padded``: a route with a zero weight is not computed.
    ``owner_plan`` (with ``owner``): also the owner combine's arrays
    (`OwnerPlan`), for bf16 activations.
    ``route_scale`` (with ``scale``): the scales come per sorted route
    (``route_scale[j]`` = the scale of route ``j``'s token, the receive-row
    scale at ``token_sorted[j]``), for a consumer whose rows land in route
    order, instead of per receive row (``row_scale`` is None).

    Same arrays as `route_plan` (and the scale scatter of
    `_capacity_exchange`, `owner_plan`) on a step that fits; past the sorted
    routes the destinations read 0 instead of the unsorted routes' (never
    read)."""
    num_members, topk, num_tokens = words.shape
    if not local_route_plan_supported(
        num_members=num_members,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=local_num_experts,
        tp_size=tp_size,
        rows=rows,
        owner=owner,
        owner_plan=owner_plan,
    ):
        raise ValueError(
            f"local_route_plan does not take {num_members} members x TP "
            f"{tp_size}, {num_tokens} tokens, top-{topk}, {local_num_experts} "
            f"local experts, {rows} rows"
        )
    members = -(-num_members // _SUBLANES) * _SUBLANES
    sort_rows = min(
        members, -(-num_members * num_tokens // (_SUBLANES * _LANES)) * _SUBLANES
    )
    out_rows = -(-routes // _LANES)
    dense_rows = -(-out_rows // _SUBLANES) * _SUBLANES
    with_scale = scale is not None
    assert with_scale or not route_scale
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    operands = [ep_rank.astype(I32), fits.astype(I32), words]
    out_shape = [jax.ShapeDtypeStruct((out_rows * _LANES,), I32)] * 3 + [
        jax.ShapeDtypeStruct((local_num_experts,), I32)
    ]
    out_specs = [vmem] * len(out_shape)
    if with_scale:
        operands.append(jax.lax.bitcast_convert_type(scale, I32))
        scale_len = out_rows if route_scale else -(-rows // _LANES)
        out_shape.append(jax.ShapeDtypeStruct((scale_len * _LANES,), jnp.float32))
        out_specs.append(vmem)
    if owner_plan:
        blk = -(-num_tokens // tp_size)
        out_shape += [jax.ShapeDtypeStruct((topk, blk), I32)] * 3 + [
            jax.ShapeDtypeStruct((topk, blk), jnp.float32),
            jax.ShapeDtypeStruct((1,), I32),
            jax.ShapeDtypeStruct((blk * tp_size,), I32),
            jax.ShapeDtypeStruct((blk * tp_size,), I32),
            jax.ShapeDtypeStruct((2,), I32),
        ]
        out_specs += [vmem] * 4 + [smem] + [vmem] * 2 + [smem]
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            num_members=num_members,
            topk=topk,
            num_tokens=num_tokens,
            local_num_experts=local_num_experts,
            tp_size=tp_size,
            owner=owner,
            skip_padded=skip_padded,
            rows=rows,
            with_scale=with_scale,
            owner_plan=owner_plan,
            route_scale=route_scale,
        ),
        out_shape=out_shape,
        in_specs=[smem, smem] + [vmem] * (len(operands) - 2),
        out_specs=out_specs,
        scratch_shapes=[
            pltpu.VMEM((topk, members, _LANES), I32),
            pltpu.VMEM((1, members, _LANES), I32),
            pltpu.VMEM((local_num_experts * sort_rows, _LANES), I32),
            pltpu.VMEM((local_num_experts * sort_rows, _LANES), I32),
            pltpu.VMEM((dense_rows, _LANES), I32),
            pltpu.VMEM((dense_rows, _LANES), I32),
        ]
        + (
            [pltpu.VMEM((-(-topk // _SUBLANES) * _SUBLANES, _LANES), I32)]
            if owner_plan
            else []
        )
        + (
            [
                pltpu.VMEM((local_num_experts * sort_rows, _LANES), I32),
                pltpu.VMEM((dense_rows, _LANES), I32),
            ]
            if route_scale
            else []
        ),
        interpret=interpret,
        name=(
            f"local_route_plan-d_{num_members}-s_{num_tokens}-k_{topk}"
            f"-l_{local_num_experts}-r_{routes}"
        ),
    )(*operands)
    # Whole lane rows; a slice only where the sizes are not.
    token_sorted, dest_member, dest_row = (o[:routes] for o in outs[:3])
    row_scale = outs[4][:rows] if with_scale and not route_scale else None
    owner_arrays = OwnerPlan(*outs[4 + with_scale :]) if owner_plan else None
    return LocalRoutePlan(
        token_sorted,
        outs[3],
        dest_member,
        dest_row,
        row_scale,
        owner_arrays,
        outs[4][:routes] if route_scale else None,
    )
