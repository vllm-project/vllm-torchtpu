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
"""The routing plan of one shard in one TensorCore kernel.

At the decode buckets the sharded plan (host.build_routing_tables_sharded and
the shard_ helpers after it) is a few hundred integers turned into tables by
about 66 separate XLA operations per layer, most shorter than 0.2 us, plus two
slab scatters the SparseCore runs ahead of the activation all-gather. Their
cost is launching, not work. This kernel reads the gathered routing payload
and writes the same integers in one launch.

Every quantity is an integer. The prefix sums, the transposes and the slab
scatter are matmuls over 0/1 matrices with the integer operand split into
bytes, so each product is exact in any matmul precision and every sum stays
below 2**24, where float32 holds integers exactly.

Orientation: the per-expert tables are built with the expert on sublanes and
the destination shard on lanes ([e_total, 128], lanes past ep zero), which is
the [G, ep] layout the expert kernel reads, and the per-pair tables with the
pair on lanes, the order the payload carries them in.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    COUNT_RECV_ALIGNED_ROWS,
    COUNT_RECV_MIRROR,
    COUNT_RECV_ROWS,
    COUNT_SELF_MIRROR,
    COUNT_SEND_ALIGNED_ROWS,
    COUNT_SEND_MIRROR,
    COUNT_SEND_ROWS,
    COUNT_VISITS,
    HIDDEN_LANE_BLOCK,
    N_COUNTS,
    ROWBLK,
    act_scale_slab_rows,
    align_up,
    pow2_shift,
    token_gather_window_rows,
)

LANES = HIDDEN_LANE_BLOCK
SUBLANES = 8
ROWBLK_SHIFT = ROWBLK.bit_length() - 1
# Lane blocks of the packed per-expert output, in this order: run start,
# aligned rows and contribution offset (blocks), push destination (blocks),
# true rows; then one lane each for the slab rows, slab base and visit list.
PACKED_TABLES = 5
PACKED_ROWS_LANE = 40
PACKED_BASE_LANE = 41
PACKED_VISIT_LANE = 42
# The widest pair row and the longest slab the kernel is built for. Past them
# the per-pair matrices outgrow the decode buckets this kernel exists for.
MAX_PAIR_LANES = 256
MAX_SLAB_ROWS = 4096


class PlanShape(NamedTuple):
    ep: int
    e_total: int
    topk: int
    t_local: int
    block: int
    tile_m: int
    shard_stride: int
    scale_at: int
    count_at: int
    width: int

    @property
    def g_local(self):
        return self.e_total // self.ep

    @property
    def pairs(self):
        return self.t_local * self.topk

    @property
    def pair_lanes(self):
        return align_up(self.pairs, LANES)

    @property
    def gather_rows(self):
        return self.shard_stride + token_gather_window_rows(self.tile_m)

    @property
    def scale_rows(self):
        return act_scale_slab_rows(self.shard_stride)

    @property
    def scatter_rows(self):
        return align_up(max(self.gather_rows, self.scale_rows * LANES), LANES)


def plan_kernel_fits(shape: PlanShape) -> bool:
    """Whether this kernel is built for the shape; otherwise the XLA plan runs."""
    return (
        shape.ep == SUBLANES
        and shape.e_total % shape.ep == 0
        and shape.g_local % SUBLANES == 0
        and PACKED_TABLES * shape.ep <= PACKED_ROWS_LANE
        and shape.pair_lanes <= MAX_PAIR_LANES
        and shape.scale_at == shape.pair_lanes
        and shape.count_at - shape.scale_at == align_up(shape.t_local, LANES)
        and shape.t_local <= LANES
        and shape.scatter_rows <= MAX_SLAB_ROWS
        and shape.pairs + (ROWBLK - 1) < 2**16 // shape.ep
    )


def _bytes_for(bound):
    return max(1, (int(bound).bit_length() + 7) // 8)


def _exact_matmul(a, b01, bound, *, nt=False):
    """``a @ b01`` (or ``a @ b01.T``) for integer a in [0, bound], b01 in {0, 1}.

    Each byte of `a` is a float32 integer below 256 and so exact in any
    matmul precision; the byte products are summed in float32 and shifted
    back as integers.
    """
    dims = (((1,), (1,)), ((), ())) if nt else (((1,), (0,)), ((), ()))

    def byte_product(k):
        part = ((a >> (8 * k)) & 255).astype(jnp.float32)
        return lax.dot_general(
            part, b01, dims, preferred_element_type=jnp.float32
        ).astype(jnp.int32)

    out = byte_product(0)
    for k in range(1, _bytes_for(bound)):
        out = out + (byte_product(k) << (8 * k))
    return out


def _iota(shape, axis):
    return lax.broadcasted_iota(jnp.int32, shape, axis)


def _strict_upper(n):
    """[n, n] float32, 1 where row < column: x @ it is an exclusive prefix."""
    return (_iota((n, n), 0) < _iota((n, n), 1)).astype(jnp.float32)


def _strict_lower(n):
    """[n, n] float32, 1 where column < row: it @ x is an exclusive prefix."""
    return (_iota((n, n), 1) < _iota((n, n), 0)).astype(jnp.float32)


def _pick_lane(x, lane):
    """Column `lane` of x as [rows, 1]."""
    return jnp.sum(jnp.where(_iota(x.shape, 1) == lane, x, 0), axis=1, keepdims=True)


def _pick_sublane(x, row):
    """Row `row` of x as [1, lanes]."""
    return jnp.sum(jnp.where(_iota(x.shape, 0) == row, x, 0), axis=0, keepdims=True)


def _column_to_row(col, lanes):
    """A [n, 1] column as a [1, lanes] row (n <= lanes, the rest zero)."""
    n = col.shape[0]
    diag = _iota((n, lanes), 0) == _iota((n, lanes), 1)
    return jnp.sum(jnp.where(diag, col, 0), axis=0, keepdims=True)


def _mirror_run_tiles(run_start, run_rows, shift):
    last = run_start + jnp.maximum(run_rows, 1) - 1
    return jnp.where(run_rows > 0, (last >> shift) - (run_start >> shift) + 1, 0)


def _plan_kernel(
    s: PlanShape,
    me_ref,
    ex_ref,
    packed_ref,
    counts_ref,
    pos_ref,
    gather_ref,
    scale_ref,
    rows_scratch,
):
    ep, E, G, K, P, PL = s.ep, s.e_total, s.g_local, s.topk, s.pairs, s.pair_lanes
    shift = pow2_shift(s.tile_m, "tile_m")
    me = me_ref[0, 0]
    first = me * G

    pairs = ex_ref[:, 0:PL]  # [ep, PL]: shard s's pairs on row s
    valid = _iota((1, PL), 1) < P

    # Rows per (expert, token shard): the gathered count table, transposed.
    # A pair whose expert is e_total (a row that routes nowhere) is in no
    # expert's count, as in the payload's own histogram.
    expert_sub = _iota((E, PL), 0)
    lane_ep = _iota((E, LANES), 1)
    true_t = jnp.zeros((E, LANES), jnp.int32)
    for src in range(ep):
        hit = (pairs[src : src + 1, :] == expert_sub) & valid
        count = jnp.sum(hit.astype(jnp.int32), axis=1, keepdims=True)
        true_t = jnp.where(lane_ep == src, count, true_t)
    aligned_t = (true_t + (ROWBLK - 1)) & jnp.int32(-ROWBLK)  # 0 past ep
    expert_rows_all = jnp.sum(aligned_t, axis=1, keepdims=True)  # [E, 1]

    run_bound = P + ROWBLK
    # Exclusive prefix over the token shards (lanes): where each run starts in
    # its expert's slab.
    run_start_t = _exact_matmul(aligned_t, _strict_upper(LANES), run_bound)
    # Exclusive prefix over the experts (sublanes), per token shard, with the
    # expert totals in lane ep: the receive base, and the slab base.
    with_totals = jnp.where(lane_ep == ep, expert_rows_all, aligned_t)
    recv_base_t = _exact_matmul_lhs01(
        _strict_lower(E), with_totals, run_bound * ep * ep
    )
    tiles_t = _mirror_run_tiles(run_start_t, aligned_t, shift)

    # This shard's own experts: rows [first, first + G) of each table.
    def mine(table):
        rows_scratch[...] = table
        return rows_scratch[pl.ds(pl.multiple_of(first, SUBLANES), G), :]

    my_true = mine(true_t)
    my_aligned = mine(aligned_t)
    my_run_start = mine(run_start_t)
    my_recv_base = mine(recv_base_t)
    my_tiles = mine(tiles_t)
    lane_g = _iota((G, LANES), 1)
    on_ep = lane_g < ep
    my_run_start = jnp.where(on_ep, my_run_start, 0)
    head = my_recv_base[0:1, :]
    group_off = jnp.where(on_ep, my_recv_base - head, 0)
    slab_base = _pick_lane(my_recv_base - head, ep)  # [G, 1]
    my_recv_base = jnp.where(on_ep, my_recv_base, 0)
    expert_rows = jnp.sum(my_aligned, axis=1, keepdims=True)  # [G, 1]

    # Contribution offsets: the regions this shard sends, per destination in
    # destination order, each region its experts in order.
    out_total = jnp.sum(my_aligned, axis=0, keepdims=True)  # [1, 128]
    contrib_base = _exact_matmul(
        jnp.broadcast_to(out_total, (SUBLANES, LANES)),
        _strict_upper(LANES),
        run_bound * G,
    )[0:1, :]
    contrib_off = jnp.where(on_ep, contrib_base + group_off, 0)

    # The count table, closed by the expert kernel's scalar core.
    lane_c = _iota((1, LANES), 1)
    is_me = (lane_c == me).astype(jnp.int32)
    other = jnp.where(lane_c < ep, 1 - is_me, 0)
    col = functools.partial(jnp.sum, axis=0, keepdims=True)
    active = jnp.sum((expert_rows > 0).astype(jnp.int32), axis=0, keepdims=True)
    count_rows = {
        COUNT_SEND_ROWS: col(my_true) * other,
        COUNT_RECV_ROWS: is_me * (col(true_t) - col(my_true)),
        COUNT_SEND_ALIGNED_ROWS: col(my_aligned) * other,
        COUNT_RECV_ALIGNED_ROWS: is_me * (col(aligned_t) - col(my_aligned)),
        COUNT_VISITS: is_me * active,
        COUNT_SEND_MIRROR: col(my_tiles) * other,
        COUNT_RECV_MIRROR: is_me * (col(tiles_t) - col(my_tiles)),
        COUNT_SELF_MIRROR: is_me * col(my_tiles),
    }
    sub_c = _iota((N_COUNTS, LANES), 0)
    counts = jnp.zeros((N_COUNTS, LANES), jnp.int32)
    for row, value in count_rows.items():
        counts = jnp.where(sub_c == row, value, counts)
    counts_ref[...] = counts

    # The visit list: experts with rows, most rows first, ties by index.
    key = _iota((G, 1), 0) - expert_rows * G  # [G, 1], all distinct
    key_row = _column_to_row(key, LANES)  # [1, 128]
    real = _iota((1, LANES), 1) < G
    rank = jnp.sum(
        jnp.where(real & (key_row < key), 1, 0), axis=1, keepdims=True
    )  # [G, 1]
    rank_row = jnp.where(real, _column_to_row(rank, LANES), -1)
    perm = jnp.sum(
        jnp.where(rank_row == _iota((G, LANES), 0), _iota((G, LANES), 1), 0),
        axis=1,
        keepdims=True,
    )
    visit = jnp.minimum(perm, G - 1)

    # The five [G, ep] tables side by side, then placed 8 lanes apart.
    tables = jnp.concatenate(
        [
            my_run_start >> ROWBLK_SHIFT,
            jnp.where(on_ep, my_aligned, 0) >> ROWBLK_SHIFT,
            contrib_off >> ROWBLK_SHIFT,
            my_recv_base >> ROWBLK_SHIFT,
            jnp.where(on_ep, my_true, 0),
        ],
        axis=1,
    )  # [G, 5 * 128]
    src_lane = _iota((PACKED_TABLES * LANES, LANES), 0)
    dst_lane = _iota((PACKED_TABLES * LANES, LANES), 1)
    lane_in = src_lane & (LANES - 1)
    table_of = src_lane >> pow2_shift(LANES, "lanes")
    place = (
        (lane_in < ep) & (dst_lane == (table_of << pow2_shift(ep, "ep")) + lane_in)
    ).astype(jnp.float32)
    packed = _exact_matmul(tables, place, run_bound * ep * E)
    packed = jnp.where(lane_g == PACKED_ROWS_LANE, expert_rows, packed)
    packed = jnp.where(lane_g == PACKED_BASE_LANE, slab_base, packed)
    packed = jnp.where(lane_g == PACKED_VISIT_LANE, visit, packed)
    packed_ref[...] = packed

    # Earlier pairs with the same expert, for every pair on lanes: x @ upper.
    upper = _strict_upper(PL)

    # This shard's own pairs: the arrival row and the mirror element each
    # selection comes back on.
    my_pairs = _pick_sublane(pairs, me)  # [1, PL]
    hit = (my_pairs == expert_sub) & valid  # [E, PL]
    hit_f = hit.astype(jnp.float32)
    earlier = jnp.dot(hit_f, upper, preferred_element_type=jnp.float32).astype(
        jnp.int32
    )
    rank_in_expert = jnp.sum(jnp.where(hit, earlier, 0), axis=0, keepdims=True)
    recv_me = _pick_lane(recv_base_t, me)  # [E, 1]
    start_me = _pick_lane(run_start_t, me)
    mirror_base = ((recv_me >> shift) + 2 * _iota((E, 1), 0)) * s.tile_m + (
        start_me & (s.tile_m - 1)
    )
    pos = jnp.sum(jnp.where(hit, recv_me, 0), axis=0, keepdims=True) + rank_in_expert
    mirror = (
        jnp.sum(jnp.where(hit, mirror_base, 0), axis=0, keepdims=True) + rank_in_expert
    )
    # A pair routed nowhere (expert e_total) looks up nothing, so both tables
    # hold its rank among such pairs of its own routing block.
    nowhere = (my_pairs == E) & valid
    block_shift = pow2_shift(s.block, "routing block")
    same_block = (
        (_iota((PL, PL), 0) < _iota((PL, PL), 1))
        & ((_iota((PL, PL), 0) >> block_shift) == (_iota((PL, PL), 1) >> block_shift))
    ).astype(jnp.float32)
    nowhere_rank = jnp.dot(
        jnp.broadcast_to(nowhere.astype(jnp.float32), (SUBLANES, PL)),
        same_block,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)[0:1, :]
    pos = jnp.where(nowhere, nowhere_rank, pos)
    mirror = jnp.where(nowhere, nowhere_rank, mirror)
    sub_p = _iota((SUBLANES, PL), 0)
    pos_ref[...] = jnp.where(sub_p == 0, pos, jnp.where(sub_p == 1, mirror, 0))

    # Every shard's pairs on this shard's experts: the slab row each computes
    # on, and that row's token number and activation scale. The scatter is
    # [values, pairs] @ [slab rows, pairs].T, one pair per slab row.
    expert_g = _iota((G, PL), 0) + first
    token_sub = _iota((LANES, PL), 0)
    token_of_lane = ((_iota((1, PL), 1).astype(jnp.float32) + 0.5) * (1.0 / K)).astype(
        jnp.int32
    )
    spread = (token_sub == token_of_lane).astype(jnp.float32)  # [128, PL]
    slab_sub = _iota((s.scatter_rows, PL), 0)
    value_sub = _iota((SUBLANES, LANES), 0)
    gathered = jnp.zeros((SUBLANES, s.scatter_rows), jnp.int32)
    for src in range(ep):
        hit = (pairs[src : src + 1, :] == expert_g) & valid  # [G, PL]
        earlier = jnp.dot(
            hit.astype(jnp.float32), upper, preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        base = slab_base + _pick_lane(my_run_start, src)  # [G, 1]
        row = jnp.sum(jnp.where(hit, base + earlier, 0), axis=0, keepdims=True)
        on_me = jnp.sum(hit.astype(jnp.int32), axis=0, keepdims=True) > 0
        row = jnp.where(on_me, row, -1)  # [1, PL]
        # Per token (lanes): its number in bytes 0-1, its scale bits in 2-5.
        token = src * s.t_local + _iota((SUBLANES, LANES), 1)
        bits = ex_ref[src : src + 1, s.scale_at : s.scale_at + LANES]
        per_token = jnp.where(
            value_sub < 2,
            (token >> (8 * value_sub)) & 255,
            jnp.where(value_sub < 6, (bits >> (8 * (value_sub - 2))) & 255, 0),
        ).astype(jnp.float32)
        per_pair = jnp.dot(per_token, spread, preferred_element_type=jnp.float32)
        onto_row = (slab_sub == row).astype(jnp.float32)  # [rows, PL]
        gathered = gathered + lax.dot_general(
            per_pair,
            onto_row,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)
    byte = [gathered[k : k + 1, :] for k in range(6)]
    token_row = byte[0] + (byte[1] << 8)
    slab_lane = _iota((1, s.scatter_rows), 1)
    token_row = jnp.where(slab_lane < s.shard_stride, token_row, 0)
    token_row = jnp.clip(token_row, 0, max(s.ep * s.t_local - 1, 0))
    gather_ref[...] = token_row[:, : s.gather_rows]
    bits_row = byte[2] | (byte[3] << 8) | (byte[4] << 16) | (byte[5] << 24)
    bits_row = jnp.where(slab_lane < s.shard_stride, bits_row, 0)
    scale_row = pltpu.bitcast(bits_row, jnp.float32)
    for r in range(s.scale_rows):
        scale_ref[r : r + 1, :] = scale_row[:, r * LANES : (r + 1) * LANES]


def _exact_matmul_lhs01(lhs01, b, bound):
    """``lhs01 @ b`` for lhs01 in {0, 1} and integer b in [0, bound]."""

    def byte_product(k):
        part = ((b >> (8 * k)) & 255).astype(jnp.float32)
        return jnp.dot(lhs01, part, preferred_element_type=jnp.float32).astype(
            jnp.int32
        )

    out = byte_product(0)
    for k in range(1, _bytes_for(bound)):
        out = out + (byte_product(k) << (8 * k))
    return out


class PlanOutputs(NamedTuple):
    packed: jax.Array
    count_lanes: jax.Array
    kernel_tables: tuple
    token_gather: jax.Array
    scale_slab: jax.Array
    expert_rows: jax.Array
    slab_base: jax.Array
    visit: jax.Array
    counts: jax.Array
    pos: jax.Array
    mirror_pos: jax.Array


@functools.cache
def _build(shape: PlanShape, interpret: bool):
    s = shape
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        functools.partial(_plan_kernel, s),
        in_specs=[smem, vmem],
        out_specs=[vmem] * 5,
        out_shape=[
            jax.ShapeDtypeStruct((s.g_local, LANES), jnp.int32),
            jax.ShapeDtypeStruct((N_COUNTS, LANES), jnp.int32),
            jax.ShapeDtypeStruct((SUBLANES, s.pair_lanes), jnp.int32),
            jax.ShapeDtypeStruct((1, s.gather_rows), jnp.int32),
            jax.ShapeDtypeStruct((s.scale_rows, LANES), jnp.float32),
        ],
        scratch_shapes=[pltpu.VMEM((s.e_total, LANES), jnp.int32)],
        interpret=interpret,
        name="moe_routing_plan",
    )


def routing_plan(exchange, rank, shape: PlanShape, *, interpret=False) -> PlanOutputs:
    """Every routing table one shard's expert kernel and combine read.

    exchange is the gathered [ep, width] payload (layer._pack_routing_exchange
    rows), rank this shard's [1, 1] index. The outputs equal the sharded XLA
    plan's, except the scale slab's elements past the slab's last row, which
    nothing reads and which are zero here.
    """
    s = shape
    ep = s.ep
    packed, counts, pos, gather, scale = _build(s, interpret)(rank, exchange)

    def table(k):
        return packed[:, k * ep : (k + 1) * ep]

    kernel_tables = (table(0), table(1), table(2), table(3), table(4))
    return PlanOutputs(
        packed=packed,
        count_lanes=counts,
        kernel_tables=kernel_tables,
        token_gather=gather.reshape(s.gather_rows),
        scale_slab=scale,
        expert_rows=packed[:, PACKED_ROWS_LANE],
        slab_base=packed[:, PACKED_BASE_LANE],
        visit=packed[:, PACKED_VISIT_LANE],
        counts=counts[:, :ep],
        pos=pos[0, : s.pairs].reshape(s.t_local, s.topk),
        mirror_pos=pos[1, : s.pairs].reshape(s.t_local, s.topk),
    )
