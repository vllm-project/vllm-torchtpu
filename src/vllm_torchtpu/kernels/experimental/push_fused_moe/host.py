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
"""Host-side tables of the push-dispatch fused EP MoE.

Everything the push build shares with v2 is imported from
kernels/fused_moe/v2/host.py; this module holds only the dispatch tables and
the sharded routing plan that reuses the dispatch's run tables.
"""

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    ROWBLK,
    ShardRoutingTables,
    _lookup_per_pair,
    _over_sub_blocks,
    _rank_within_block,
    _refuse_bad_plan_args,
    _window_hist,
    _window_table,
    expert_visit_list,
    mirror_run_tiles,
    off_shard_slab_row,
    pow2_shift,
    rank_window,
)


def run_tables(rows_by_dest):
    """(run_rows, run_rows_aligned, run_start_aligned, expert_rows_aligned).

    The plan's per-(expert, source) run tables, from the exchanged
    [ep, e_total] pair counts alone -- the same arithmetic as v2's sharded
    plan, computed early so the dispatch tables need not wait for the plan.
    """
    run_rows = rows_by_dest.T
    run_rows_aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start_aligned = jnp.cumsum(run_rows_aligned, axis=1) - run_rows_aligned
    expert_rows_aligned = run_rows_aligned.sum(axis=1)
    return run_rows, run_rows_aligned, run_start_aligned, expert_rows_aligned


def _take_small(table, idx):
    """table[idx] for a small 1-D int table, as a compare-select-sum: TPU
    gathers lower to a per-element loop, a few us even at a few hundred
    elements, while this is one fused vector pass."""
    n = table.shape[0]
    hit = idx.reshape(-1)[:, None] == jnp.arange(n, dtype=jnp.int32)[None, :]
    out = jnp.sum(jnp.where(hit, table.astype(jnp.int32)[None, :], 0), axis=1)
    return out.astype(jnp.int32).reshape(idx.shape)


def local_sort_activations(my_pairs, my_counts, q_l, *, e_total, topk):
    """Sort this shard's rows `q_l` by global expert, tokens ascending within.

    `my_pairs` [t_local * topk] are this shard's selections (e_total = routes
    nowhere) and `my_counts` [e_total] their per-expert counts. Depends only
    on local routing, so it runs on the TensorCore while the routing
    exchange is in flight. Returns `(q_sorted, seg_start)`: row j of q_sorted
    is the j-th routed pair in (expert, token) order and seg_start [e_total]
    the first row of each expert's run.
    """
    n = my_pairs.shape[0]
    t_local = n // topk
    my_pairs = my_pairs.astype(jnp.int32)
    my_counts = my_counts.astype(jnp.int32)
    valid = my_pairs < e_total
    pid = jnp.where(valid, my_pairs, e_total)
    seg_end = jnp.cumsum(my_counts)
    seg_start_local = seg_end - my_counts
    sel = jnp.any(
        pid.reshape(t_local, topk)[:, :, None] == jnp.arange(e_total)[None, None, :],
        axis=1,
    )
    C = jnp.cumsum(sel.astype(jnp.int32), axis=0)  # [t_local, e_total]
    jj = jnp.arange(n, dtype=jnp.int32)
    ge_j = jnp.minimum(
        jnp.sum(seg_end[None, :] <= jj[:, None], axis=1).astype(jnp.int32),
        e_total - 1,
    )
    m_j = jj - _take_small(seg_start_local, ge_j)
    # The token of row j is the m_j-th token selecting expert ge_j: a count
    # over C's column ge_j, read through a one-hot matmul (exact in f32 at
    # these sizes, Precision.HIGHEST) rather than a gather.
    onehot = (ge_j[:, None] == jnp.arange(e_total)[None, :]).astype(jnp.float32)
    col = jnp.dot(
        onehot, C.T.astype(jnp.float32), precision=lax.Precision.HIGHEST
    ).astype(jnp.int32)  # [n, t_local] = C[:, ge_j]
    tok_j = jnp.minimum(
        jnp.sum(col <= m_j[:, None], axis=1).astype(jnp.int32), t_local - 1
    )
    q_sorted = q_l[tok_j]
    return q_sorted, seg_start_local


def push_dispatch_segments(
    expert_rows_aligned,
    run_start_aligned,
    rows_by_dest,
    me,
    seg_start_local,
    *,
    e_total,
    ep,
):
    """The dispatch segment table of shard `me`: (seg, meta, visit).

    Each (destination d, local expert e) run of this shard's routed rows is
    contiguous both in the expert-sorted rows (from `seg_start_local`, see
    local_sort_activations) and in d's expert slab (rows local_base[d, e] +
    run_start_aligned[d * g_local + e, me] onward), so it moves as ONE DMA.
    Segment entry v * ep + d is d's v-th visited expert:
    seg = [dest << 24 | expert << 16 | rows (S), source row (S),
    destination slab row (S)], S = ep * g_local. meta = [g_local + 1 unused,
    receive pair count per local expert (g_local), remote rows sent (1),
    rows sent (1)]. visit [ep, g_local] is every shard's visit list.
    """
    g_local = e_total // ep
    rows_sh = expert_rows_aligned.astype(jnp.int32).reshape(ep, g_local)
    visit, n_visit = jax.vmap(lambda r: expert_visit_list(r, g_local))(rows_sh)
    pos = jnp.arange(g_local, dtype=jnp.int32)[None, :]
    live = pos < n_visit.reshape(ep, 1)
    # Segment grid in (visit position, destination) order.
    vis_t = visit.T  # [g_local (v), ep (d)]
    mine = rows_by_dest.astype(jnp.int32)[me]  # [e_total] my pairs per expert
    hit_v = (
        visit[:, :, None] == jnp.arange(g_local, dtype=jnp.int32)[None, None, :]
    ) & live[:, :, None]  # [ep (d), g_local (v), g_local (e)]

    def _take_by_visit(tbl_2d):
        return jnp.sum(
            jnp.where(hit_v, tbl_2d.astype(jnp.int32)[:, None, :], 0), axis=2
        ).T  # [g_local (v), ep (d)]

    cnt = _take_by_visit(mine.reshape(ep, g_local))
    src_off = _take_by_visit(seg_start_local.reshape(ep, g_local))
    local_base = jnp.cumsum(rows_sh, axis=1) - rows_sh  # [ep, g_local]
    run_me = lax.dynamic_slice(
        run_start_aligned.astype(jnp.int32), (0, me), (e_total, 1)
    ).reshape(ep, g_local)
    dst_row = _take_by_visit(local_base + run_me)
    d_grid = jnp.broadcast_to(jnp.arange(ep, dtype=jnp.int32)[None, :], (g_local, ep))
    a = jnp.left_shift(d_grid, 24) | jnp.left_shift(vis_t, 16) | cnt
    seg = jnp.concatenate(
        [a.reshape(-1), src_off.reshape(-1), jnp.where(cnt > 0, dst_row, 0).reshape(-1)]
    ).astype(jnp.int32)
    recv = lax.dynamic_slice(
        rows_by_dest.astype(jnp.int32), (0, me * g_local), (ep, g_local)
    ).sum(axis=0)
    n_remote = jnp.sum(jnp.where(d_grid != me, cnt, 0)).astype(jnp.int32).reshape(1)
    n_sent = jnp.sum(cnt).astype(jnp.int32).reshape(1)
    meta = jnp.concatenate(
        [jnp.zeros((g_local + 1,), jnp.int32), recv, n_remote, n_sent]
    )
    return seg, meta, visit


def _rank_within_window(expert_blocks, window):
    """``#{j < i : e_j == e_i}`` over i's ``window``-wide sub-block.

    v2's helper, spelled as a [block, window, window] self-compare instead of
    a broadcast of the sub-block peers; same values.
    """
    n_blocks, block = expert_blocks.shape
    if window == block:
        return _rank_within_block(expert_blocks, block)
    eb = expert_blocks.reshape(n_blocks, block // window, window)
    lower = (
        jnp.arange(window, dtype=jnp.int32)[None, :]
        < jnp.arange(window, dtype=jnp.int32)[:, None]
    )
    return jnp.sum(
        ((eb[:, :, :, None] == eb[:, :, None, :]) & lower[None, None, :, :]).astype(
            jnp.int32
        ),
        axis=3,
    ).reshape(n_blocks, block)


def _lookup_per_sub_pair(table, expert_blocks, bins, window):
    """``table[b, i // window, expert_blocks[b, i]]`` -- one value per pair.

    v2's helper, spelled over the [block / window, window] sub-block grid;
    same values.
    """
    if table.shape[1] == 1:
        return _lookup_per_pair(table[:, 0], expert_blocks, bins)
    n_blocks, block = expert_blocks.shape
    eb = expert_blocks.reshape(n_blocks, block // window, window)
    return jnp.sum(
        jnp.where(
            eb[:, :, :, None] == bins[None, None, None, :],
            table[:, :, None, :],
            0,
        ),
        axis=3,
    ).reshape(n_blocks, block)


def build_routing_tables_sharded(
    me, gathered, runs, *, e_total, ep, t_local, block, tile_m, shard_stride
):
    """v2's `build_routing_tables_sharded` on the `gathered` path, taking
    the run tables (`run_tables(rows_by_dest)`) the dispatch already built
    instead of recomputing them. Every table is identical to v2's.
    """
    T, K = gathered.n_tokens, gathered.topk
    n = T * K
    g_local = e_total // ep
    _refuse_bad_plan_args(
        T, K, e_total=e_total, ep=ep, t_local=t_local, block=block, tile_m=tile_m
    )
    if shard_stride % tile_m:
        raise ValueError(
            f"the per-shard slab stride {shard_stride} is not a "
            f"whole number of {tile_m}-row tiles"
        )

    n_blocks = n // block
    blocks_per_dest = (t_local * K) // block
    expert_blocks = gathered.expert_blocks
    bins = jnp.arange(e_total, dtype=jnp.int32)
    window = rank_window(block, blocks_per_dest * block)

    my_blocks, my_hist = gathered.local_blocks, gathered.block_hist
    my_off = jnp.cumsum(my_hist, axis=0) - my_hist

    run_rows, run_rows_aligned, run_start_aligned, expert_rows_aligned = runs
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard
    expert_base = (
        local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] * shard_stride)
    ).reshape(e_total)
    region_rows = run_rows_aligned.reshape(ep, g_local, ep)
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
    recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
        ep, ep, g_local
    )
    recv_rows = rows_per_dest.sum(axis=1)

    # For a pair in this rank's token block the source prefix cancels from
    # the arrival row, leaving the destination receive base plus its offset
    # within the source's expert run.
    my_recv_base = lax.dynamic_slice(recv_base, (me, 0, 0), (1, ep, g_local)).reshape(
        e_total
    )
    pos_table = my_off + my_recv_base[None, :]
    my_rank = _rank_within_block(my_blocks, block)
    pos = (_lookup_per_pair(pos_table, my_blocks, bins) + my_rank).reshape(t_local, K)

    # The mirror position of the same pair: its offset within the run, past
    # the lane its source tile's sublane starts on.
    run_tiles = mirror_run_tiles(run_start_aligned, run_rows_aligned, tile_m)
    shift = pow2_shift(tile_m, "tile_m")
    my_run_start = lax.dynamic_slice(run_start_aligned, (0, me), (e_total, 1)).reshape(
        e_total
    )
    mirror_table = (
        my_off
        + (
            (
                jnp.right_shift(my_recv_base, shift)
                + 2 * jnp.arange(e_total, dtype=jnp.int32)
            )
            * tile_m
            + (my_run_start & (tile_m - 1))
        )[None, :]
    )
    mirror_pos = (_lookup_per_pair(mirror_table, my_blocks, bins) + my_rank).reshape(
        t_local, K
    )

    # Across all tokens, compute offsets only for this rank's local experts.
    first = me * g_local
    my_experts = first + jnp.arange(g_local, dtype=jnp.int32)
    win_hist = _window_hist(expert_blocks, my_experts, window)
    hist_mine = _over_sub_blocks(win_hist)
    by_source = hist_mine.reshape(ep, blocks_per_dest, g_local)
    off_in_source = jnp.cumsum(by_source, axis=1) - by_source
    run_base = (
        lax.dynamic_slice(expert_base, (first,), (g_local,))[None, :]
        + lax.dynamic_slice(run_start_aligned, (first, 0), (g_local, ep)).T
    )
    row_table = _window_table(
        (off_in_source + run_base[:, None, :]).reshape(n_blocks, g_local), win_hist
    )
    on_me = (expert_blocks >= first) & (expert_blocks < first + g_local)
    my_row = _lookup_per_sub_pair(
        row_table, expert_blocks, my_experts, window
    ) + _rank_within_window(expert_blocks, window)
    slab_row = jnp.where(
        on_me, my_row, jnp.int32(off_shard_slab_row(shard_stride, ep))
    ).reshape(-1)

    return ShardRoutingTables(
        pos=pos,
        mirror_pos=mirror_pos,
        run_tiles=run_tiles,
        slab_row=slab_row,
        run_rows=run_rows,
        run_rows_aligned=run_rows_aligned,
        run_start_aligned=run_start_aligned,
        region_rows=region_rows,
        recv_base=recv_base,
        recv_rows=recv_rows,
        expert_rows_aligned=expert_rows_aligned,
        expert_base=expert_base,
        rows_alloc=ep * shard_stride,
        n_tokens=T,
        topk=K,
    )
