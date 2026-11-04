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
"""Exact unsorted top-k over fp32 score rows on SparseCore.

Single-stage multi-subcore selection with VMEM_SHARED global histogram:
1. For any batch size B (including B = 1, 2, 4, 8, 16, 32) and N in 2048..256K,
   rows are partitioned into P slices across up to 16 subcores per SparseCore.
2. When P == 1, all 32 subcores independently find the top-k on their full row
   in lock-free, single-subcore mode.
3. When P > 1, groups of P subcores on the same SparseCore cooperatively build
   and refine a global radix histogram in ``VMEM_SHARED`` to find the exact
   global threshold and emit only the final k winners directly to HBM in a
   single stage.
4. Either way, a subcore fetches only the first ``row_lengths`` words of its
   row (its share of them when P > 1), not the full width N: N follows
   max-model-len, and rows are usually far shorter.

5. ``TPU_SC_TOPK_COOP_V2`` (P > 1 only): the refinement levels run as one
   runtime loop (one copy of their code: the SparseCore call's startup grows
   with program size), each level's peer reduce, scan and bucket search share
   one loop over the histogram, and the winners reach HBM without subcore 0
   pulling its peers one by one -- at P >= 4 every subcore scatters its winners into the
   group's staging slot at its prefix offset; at P = 2 the peer's copies are all
   in flight before the first wait. ``sparsecore_topk_dcp_merge_coop`` splits
   each DCP merge row over P subcores by rank segment instead of one subcore
   per row. All bitwise identical to the default paths.

Output rows are the top-k column indices in unspecified order, ``-1``
suffix-padded, matching an exact top-k over the given scores as a set (equal
boundary scores may be broken differently). ``-inf`` scores and positions at
or beyond ``row_lengths`` are never selected. NaNs are unsupported.
"""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu import envs

LANES = 16
NUM_BUCKETS = 256
# Half of a subcore's slice buffer, in 32-bit words (128KB of the 512KB tile
# memory). Parts up to this long are double-buffered across waves; a longer part
# (up to twice this) fills the whole buffer and is fetched single-buffered.
MAX_SLICE_WORDS = 32 * 1024
# Unroll factor for the loops that walk the whole resident slice.
SCAN_UNROLL = 8

L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
BITS_NEG_INF = int(np.array(-np.inf, np.float32).view(np.int32))
KEY_NEG_INF = int(np.int32(BITS_NEG_INF) ^ np.int32(0x7FFFFFFF))


def _cdiv(a, b):
    return (a + b - 1) // b


def _align_to(x, a):
    return _cdiv(x, a) * a


def _cdiv_dyn(x):
    return jnp.right_shift(x + (LANES - 1), 4)


def _topk_body(
    scores_hbm,  # i32[padded_b * p * slice_len] (fp32 score bits)
    lengths_hbm,  # i32[padded_b + LANES]; the DCP merge: i32[padded_b * LANES]
    out_hbm,  # i32[padded_b * k]
    keys_hbm,  # i32[padded_b * k] | None
    glob_vmem,
    rowbuf_vmem,
    keybuf_vmem,
    peer_hist_vmem,
    peer_idx_vmem,
    peer_key_vmem,
    shared_hist_vmem,
    shared_counts_vmem,
    shared_idx_vmem,
    shared_key_vmem,
    meta_vmem,
    slices_vmem,
    slice_sems,
    gather_sems=None,
    oidx_vmem=None,
    lidx_vmem=None,
    *,
    b: int,
    n: int,
    k: int,
    p: int = 1,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    return_scores: bool,
    local_idxs_hbm: Any = None,
    rank_major_dcp: int = 0,
    coop_v2: bool = False,
    merge_coop_dcp: int = 0,
):
    dcp_merge = local_idxs_hbm is not None
    if rank_major_dcp:
        # Unpadded DCP merge (see `_sc_topk_direct`): one wave at P = 1, the
        # row fetched as `rank_major_dcp` k-word pieces.
        assert dcp_merge and p == 1 and num_waves == 1
        assert n == rank_major_dcp * k and slice_len == n
    merge_coop = merge_coop_dcp > 0
    # TPU_SC_TOPK_COOP_V2: the winner gather at P > 1. "shared": each subcore
    # scatters its winners straight into the group's staging slot at its prefix
    # offset (P >= 4: fewer winners per subcore, so the single-word scatter is
    # cheap and the 2 (P - 1) serial peer copies go away); "async": subcore 0's
    # peer copies are all started before the first wait (P == 2, where each
    # subcore holds ~k / 2 winners and the scatter costs more than the copies).
    gather_mode = "serial"
    if coop_v2 and p > 1 and not merge_coop:
        gather_mode = "shared" if p >= 4 else "async"
    level_loop = coop_v2 or merge_coop
    # Same flags: the peer reduce, the scan and (refinement levels) the bucket
    # search run as one loop over the histogram (`reduce_scan`).
    fused_reduce = (coop_v2 or merge_coop) and p > 1
    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    groups_per_core = num_subcores // p
    groups_per_chip = num_sc_cores * groups_per_core
    if p == num_subcores:
        group_id = jnp.int32(0)
        s = sub
        group_base = jnp.int32(0)
    else:
        group_id = jnp.right_shift(sub, int(np.log2(p)))
        s = jnp.bitwise_and(sub, jnp.int32(p - 1))
        group_base = group_id * p
    col_base = s * slice_len

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def key_window(prefix, shift):
        """Scalar bounds of the key range ``key >> shift == prefix``."""
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound):
        def body(i):
            glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket, mask):
        """Accumulates one chunk's bucket counts into the histogram."""
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def sync_and_reduce_hist(slot: int, active_hist):
        """Syncs per-subcore histograms via VMEM_SHARED, reducing across P slices."""
        if p == 1:
            return

        @pl.when(active_hist)
        def _():
            dst_off = pl.multiple_of((slot * num_subcores + sub) * NUM_BUCKETS, LANES)
            pltpu.sync_copy(
                glob_vmem.at[pl.ds(0, NUM_BUCKETS)],
                shared_hist_vmem.at[pl.ds(dst_off, NUM_BUCKETS)],
            )

        plsc.subcore_barrier()

        @pl.when(active_hist)
        def _():
            src_off = pl.multiple_of(
                (slot * num_subcores + group_base) * NUM_BUCKETS, LANES
            )
            pltpu.sync_copy(
                shared_hist_vmem.at[pl.ds(src_off, p * NUM_BUCKETS)],
                peer_hist_vmem.at[pl.ds(0, p * NUM_BUCKETS)],
            )

            def init_hist(i):
                glob_vmem[pl.ds(i * LANES, LANES)] = peer_hist_vmem[
                    pl.ds(i * LANES, LANES)
                ]

            plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=8)(init_hist)

            def add_peer(t):
                def add_chunk(i):
                    glob_vmem[pl.ds(i * LANES, LANES)] = (
                        glob_vmem[pl.ds(i * LANES, LANES)]
                        + peer_hist_vmem[pl.ds(t * NUM_BUCKETS + i * LANES, LANES)]
                    )

                plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=8)(add_chunk)

            plsc.parallel_loop(1, p, unroll=1)(add_peer)

    def exchange_hist(slot, active_hist):
        """The histogram exchange alone: own histogram out, barrier, the group's
        p in."""

        @pl.when(active_hist)
        def _():
            dst_off = pl.multiple_of((slot * num_subcores + sub) * NUM_BUCKETS, LANES)
            pltpu.sync_copy(
                glob_vmem.at[pl.ds(0, NUM_BUCKETS)],
                shared_hist_vmem.at[pl.ds(dst_off, NUM_BUCKETS)],
            )

        plsc.subcore_barrier()

        @pl.when(active_hist)
        def _():
            src_off = pl.multiple_of(
                (slot * num_subcores + group_base) * NUM_BUCKETS, LANES
            )
            pltpu.sync_copy(
                shared_hist_vmem.at[pl.ds(src_off, p * NUM_BUCKETS)],
                peer_hist_vmem.at[pl.ds(0, p * NUM_BUCKETS)],
            )

    # `plsc.parallel_loop(..., carry=)` returns the final carry, though its
    # stubs type the decorated loop as returning None; the helpers below cast.
    def reduce_scan(bound, thresh=None) -> tuple[jax.Array, jax.Array]:
        """glob = running bucket counts of the sum of the group's p histograms,
        chunk by chunk (the sum stays in registers); with `thresh`, also the
        number of buckets whose running count is <= thresh (`find_bucket`).
        Returns (total, that number)."""

        def body(i, carry):
            run, below = carry
            v = peer_hist_vmem[pl.ds(i * LANES, LANES)]
            for t in range(1, p):
                v = v + peer_hist_vmem[pl.ds(t * NUM_BUCKETS + i * LANES, LANES)]
            c = plsc.cumsum(v) + run
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            if thresh is not None:
                below = below + plsc.all_reduce_population_count(c <= thresh)[0]
            return c[LANES - 1], below

        return cast(
            tuple[jax.Array, jax.Array],
            plsc.parallel_loop(0, bound, unroll=2, carry=(jnp.int32(0), jnp.int32(0)))(
                body
            ),  # pytype: disable=bad-argument-type
        )

    def scan_glob(bound) -> jax.Array:
        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return cast(
            jax.Array,
            plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
                body
            ),  # pytype: disable=bad-argument-type
        )

    def find_bucket(bound, thresh) -> jax.Array:
        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return cast(
            jax.Array,
            plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
                body
            ),  # pytype: disable=bad-argument-type
        )

    def _run_step(
        scores_vmem,
        len_slice_vmem,
        out_vmem: Any = None,
        keys_vmem: Any = None,
        local_idxs_vmem: Any = None,
        *,
        w: Any = None,
        row_len: Any = None,
        col_lo: Any = None,
        col_cap: Any = None,
        merge_owner: Any = None,
    ):
        """One wave. The wave loop passes the wave, its row length and this
        subcore's column range; the DCP merge's emit_pipeline reads them off
        the grid and the lengths block."""
        if w is None:
            w = pl.program_id(0)
        row = (
            w * (num_sc_cores * num_subcores) + flat
            if p == 1
            else w * groups_per_chip + core * groups_per_core + group_id
        )
        active = row < b
        if row_len is not None:
            row_eff_len = row_len
        elif dcp_merge:
            row_eff_len = jnp.int32(n)
        else:
            row_eff_len = jnp.minimum(vec_at(len_slice_vmem, 0), n)
        row_eff_len = jnp.where(active, row_eff_len, 0)
        lo = col_base if col_lo is None else col_lo

        if p == 1:
            i_hi = jnp.where(row_eff_len > 0, jnp.minimum(slice_len, row_eff_len), 0)
        else:
            cap = slice_len if col_cap is None else col_cap
            i_hi = jnp.clip(row_eff_len - lo, 0, cap)
        walk_chunks = _cdiv_dyn(i_hi)

        zero_hist(NUM_BUCKETS // LANES)

        @pl.when(i_hi > 0)
        def _():
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            def hist0_body(i):
                key = monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
                scores_vmem[pl.ds(i * LANES, LANES)] = key
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
                hist_add(bucket, key > neg_inf_key)

            plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

        if fused_reduce:
            exchange_hist(0, active)
            total, _ = reduce_scan(NUM_BUCKETS // LANES)
        else:
            sync_and_reduce_hist(0, active)
            total = scan_glob(NUM_BUCKETS // LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        def level(
            lvl_shift, slot, prefix, shift, c_hi, quota, cnt, done, lvl_mask=0xFF
        ):
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            zero_hist(hist_bound)
            active_walk = jnp.where(done, 0, walk_chunks)
            lo_excl, hi_incl = key_window(prefix, shift)

            def ref_body(i):
                key = scores_vmem[pl.ds(i * LANES, LANES)]
                match = (key > lo_excl) & (key <= hi_incl)
                bucket = jnp.bitwise_and(
                    jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                )
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

            if fused_reduce:
                # This level's histogram total is the previous level's boundary
                # bucket count `cnt`, so the bucket search shares the loop.
                exchange_hist(slot, ~done)
                sub_total = cnt
                _, b2 = reduce_scan(hist_bound, cnt - quota)
            else:
                sync_and_reduce_hist(slot, ~done)
                sub_total = scan_glob(hist_bound)
                b2 = find_bucket(hist_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
            c_above2 = sub_total - cum_at2
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - c_above2

            width = shift - lvl_shift
            new_prefix = jnp.left_shift(prefix, width) | b2
            new_done = done | (quota2 == cnt2) | (quota2 == 0)

            prefix = jnp.where(done, prefix, new_prefix)
            shift = jnp.where(done, shift, jnp.int32(lvl_shift))
            c_hi = jnp.where(done, c_hi, c_hi + c_above2)
            quota = jnp.where(done, quota, quota2)
            cnt = jnp.where(done, cnt, cnt2)
            return prefix, shift, c_hi, quota, cnt, new_done

        carry = (prefix, shift, c_hi, quota, cnt, done)
        if level_loop:
            # One copy of the level code; shifts 16, 8, 0 and exchange slots 1, 0, 1.
            assert REFINEMENT_LEVELS == ((16, 0xFF), (8, 0xFF), (0, 0xFF))

            def lvl_iter(i, c):
                return level(
                    jnp.int32(16) - jnp.int32(8) * i, jnp.bitwise_and(i + 1, 1), *c
                )

            carry = lax.fori_loop(0, len(REFINEMENT_LEVELS), lvl_iter, carry)
        else:
            for lvl_idx, (lvl_shift, lvl_mask) in enumerate(REFINEMENT_LEVELS, start=1):
                carry = level(lvl_shift, lvl_idx % 2, *carry, lvl_mask=lvl_mask)
        prefix, shift, c_hi, quota, cnt, done = carry

        if callable(merge_owner):
            # Cooperative merge: the rank and this row's local slots were fetched
            # in flight with the levels; wait for them only now.
            merge_owner = merge_owner()

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
            if return_scores:
                keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES,), BITS_NEG_INF, jnp.int32
                )

        if p == 1:
            fill_bound = (
                k // LANES if dcp_merge else jnp.where(total < k, k // LANES, 0)
            )
        elif merge_coop:
            fill_bound = jnp.where(merge_owner & active, k // LANES, 0)
        else:
            fill_bound = jnp.where((s == 0) & (total < k), k // LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)
        if gather_mode == "shared":
            # A short row's padding goes into the group's staging slot before the
            # counts barrier; the winners overwrite its prefix after it.
            @pl.when((s == 0) & (total < k) & active)
            def _():
                go = pl.multiple_of(
                    (jnp.bitwise_and(w, jnp.int32(1)) * num_subcores + group_base) * k,
                    LANES,
                )
                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, k)], shared_idx_vmem.at[pl.ds(go, k)]
                )
                if return_scores:
                    pltpu.sync_copy(
                        keybuf_vmem.at[pl.ds(0, k)], shared_key_vmem.at[pl.ds(go, k)]
                    )

        lo_excl, hi_incl = key_window(prefix, shift)
        if p == 1:
            my_take = jnp.where(keep_all, 0, quota)
            take_all_ties = keep_all | (my_take >= cnt)
            gate = jnp.where(take_all_ties, lo_excl, hi_incl)
            whole_bucket = take_all_ties | (my_take <= 0)
        else:
            last_b = jnp.bitwise_and(prefix, jnp.int32(0xFF))

            def tie_body(t, carry):
                tb, cs = carry
                c_t = vec_at(peer_hist_vmem, t * NUM_BUCKETS + last_b)
                return (
                    jnp.where(t < s, tb + c_t, tb),
                    jnp.where(t == s, c_t, cs),
                )

            ties_before, cnt_s = plsc.parallel_loop(
                0, p, unroll=1, carry=(jnp.int32(0), jnp.int32(0))
            )(tie_body)

            global_take_all = keep_all | (quota >= cnt)
            global_drop_all = (~keep_all) & (quota <= 0)
            my_take = jnp.where(
                global_take_all,
                cnt_s,
                jnp.where(
                    global_drop_all,
                    0,
                    jnp.clip(quota - ties_before, 0, cnt_s),
                ),
            )
            take_all_ties = global_take_all | ((my_take >= cnt_s) & (cnt_s > 0))
            gate = jnp.where(take_all_ties, lo_excl, hi_incl)
            whole_bucket = take_all_ties | (my_take <= 0)

        if dcp_merge:
            score_chunk_base = vec_at(len_slice_vmem, 0) * (k // LANES)
            emitter = active if merge_owner is None else (active & merge_owner)
            emit_chunks = jnp.where(emitter, k // LANES, 0)

            def count_prior_ties(i, rank):
                key = scores_vmem[pl.ds(i * LANES, LANES)]
                in_set = (key > lo_excl) & (key <= hi_incl)
                return rank + plsc.all_reduce_population_count(in_set)[0]

            prior_ties = plsc.parallel_loop(
                0,
                jnp.where((~whole_bucket) & emitter, score_chunk_base, 0),
                unroll=4,
                carry=jnp.int32(0),
                # pyrefly: ignore[bad-argument-type]
            )(count_prior_ties)
        else:
            score_chunk_base = 0
            emit_chunks = jnp.where(active, walk_chunks, 0)
            prior_ties = jnp.int32(0)

        def emit(i, woff, key, sel):
            if dcp_merge:
                col_offset = local_idxs_vmem[pl.ds(i * LANES, LANES)]
                sel = sel & (col_offset >= 0)
            else:
                col_offset = (
                    i * LANES + lane_iota if p == 1 else lo + i * LANES + lane_iota
                )
            plsc.store_compressed(
                rowbuf_vmem.at[pl.ds(woff, LANES)],
                col_offset,
                mask=sel,
            )
            if return_scores:
                plsc.store_compressed(
                    keybuf_vmem.at[pl.ds(woff, LANES)],
                    monotone_key(key),
                    mask=sel,
                )
            return woff + plsc.all_reduce_population_count(sel)[0]

        def emit_whole(i, woff):
            key = scores_vmem[pl.ds((score_chunk_base + i) * LANES, LANES)]
            return emit(i, woff, key, key > gate)

        m_whole = plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, emit_chunks, 0),
            unroll=SCAN_UNROLL,
            carry=jnp.int32(0),
        )(emit_whole)  # pytype: disable=bad-argument-type

        def emit_ranked(i, carry):
            woff, rank = carry
            key = scores_vmem[pl.ds((score_chunk_base + i) * LANES, LANES)]
            in_set = key > lo_excl
            strict = key > hi_incl
            tie_rank = (
                plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
            )
            sel = strict | (in_set & (tie_rank <= my_take))
            return emit(i, woff, key, sel), tie_rank[LANES - 1]

        m_ranked, _ = plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, 0, emit_chunks),
            unroll=4,
            carry=(jnp.int32(0), prior_ties),
        )(emit_ranked)

        if merge_coop:
            # Only the subcore holding this rank's own segment emits; it writes the
            # row itself (no winner gather).
            @pl.when(active & merge_owner)
            def _():
                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, k)],
                    out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
                )

            return
        if p == 1 and out_vmem is None:
            # The wave loop has no pipeline output block: write the row out.
            out_off = pl.multiple_of(row * k, LANES)
            pltpu.sync_copy(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(out_off, k)])
            if return_scores:
                pltpu.sync_copy(
                    keybuf_vmem.at[pl.ds(0, k)], keys_hbm.at[pl.ds(out_off, k)]
                )
        elif p == 1:

            def copy_out(i):
                out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]
                if return_scores:
                    keys_vmem[pl.ds(i * LANES, LANES)] = keybuf_vmem[
                        pl.ds(i * LANES, LANES)
                    ]

            plsc.parallel_loop(0, k // LANES, unroll=4)(copy_out)
        else:
            # pyrefly: ignore[bad-argument-type]
            m_s = jnp.where(whole_bucket, m_whole, m_ranked)
            w_slot = jnp.bitwise_and(w, jnp.int32(1))
            glob_vmem[pl.ds(0, LANES)] = jnp.broadcast_to(m_s, (LANES,))

            @pl.when(active)
            def _():
                cnt_off = pl.multiple_of((w_slot * num_subcores + sub) * LANES, LANES)
                pltpu.sync_copy(
                    glob_vmem.at[pl.ds(0, LANES)],
                    shared_counts_vmem.at[pl.ds(cnt_off, LANES)],
                )

            chunks_s = _cdiv_dyn(m_s)
            aligned_m_s = pl.multiple_of(chunks_s * LANES, LANES)
            should_write = active & (jnp.bool_(write_empty) | (row_eff_len > 0))

            def copies(pairs, sems):
                """Starts every (src, dst) copy, then waits for all of them."""
                cps = [
                    pltpu.make_async_copy(src, dst, sems.at[i])
                    for i, (src, dst) in enumerate(pairs)
                ]
                for c in cps:
                    c.start()
                for c in cps:
                    c.wait()

            if gather_mode == "shared":
                # Counts barrier, then every subcore scatters its m_s winners into
                # the group's staging slot (slot `group_base`, unused by the serial
                # gather) at its prefix offset; after the second barrier subcore 0
                # copies the slot out. Same order: the parts are in column order.
                grp_out = (w_slot * num_subcores + group_base) * k
                assert oidx_vmem is not None and gather_sems is not None
                plsc.subcore_barrier()

                @pl.when(active)
                def _():
                    grp_cnt_off = pl.multiple_of(
                        (w_slot * num_subcores + group_base) * LANES, LANES
                    )
                    pltpu.sync_copy(
                        shared_counts_vmem.at[pl.ds(grp_cnt_off, p * LANES)],
                        peer_hist_vmem.at[pl.ds(0, p * LANES)],
                    )

                    def before(t, acc):
                        return acc + jnp.where(
                            t < s, vec_at(peer_hist_vmem, t * LANES), 0
                        )

                    # pyrefly: ignore[bad-argument-type]
                    pfx = plsc.parallel_loop(0, p, unroll=1, carry=jnp.int32(0))(before)

                    def offsets(c):
                        oidx_vmem[pl.ds(c * LANES, LANES)] = (
                            # pyrefly: ignore[unsupported-operation]
                            grp_out + pfx + c * LANES + lane_iota
                        )

                    plsc.parallel_loop(0, chunks_s, unroll=4)(offsets)

                    @pl.when(m_s > 0)
                    def _():
                        dst_idx = oidx_vmem.at[pl.ds(0, m_s)]
                        pairs = [
                            (rowbuf_vmem.at[pl.ds(0, m_s)], shared_idx_vmem.at[dst_idx])
                        ]
                        if return_scores:
                            pairs.append(
                                (
                                    keybuf_vmem.at[pl.ds(0, m_s)],
                                    shared_key_vmem.at[dst_idx],
                                )
                            )
                        copies(pairs, gather_sems)

                plsc.subcore_barrier()

                @pl.when((s == 0) & should_write)
                def _():
                    out_off = pl.multiple_of(row * k, LANES)
                    go = pl.multiple_of(grp_out, LANES)
                    pairs = [
                        (
                            shared_idx_vmem.at[pl.ds(go, k)],
                            out_hbm.at[pl.ds(out_off, k)],
                        )
                    ]
                    if return_scores:
                        pairs.append(
                            (
                                shared_key_vmem.at[pl.ds(go, k)],
                                keys_hbm.at[pl.ds(out_off, k)],
                            )
                        )
                    copies(pairs, gather_sems)

                return

            @pl.when(active & (s > 0) & (chunks_s > 0))
            def _():
                base_off = pl.multiple_of((w_slot * num_subcores + sub) * k, LANES)
                pairs = [
                    (
                        rowbuf_vmem.at[pl.ds(0, aligned_m_s)],
                        shared_idx_vmem.at[pl.ds(base_off, aligned_m_s)],
                    )
                ]
                if return_scores:
                    pairs.append(
                        (
                            keybuf_vmem.at[pl.ds(0, aligned_m_s)],
                            shared_key_vmem.at[pl.ds(base_off, aligned_m_s)],
                        )
                    )
                if gather_mode == "async":
                    copies(pairs, gather_sems)
                else:
                    for src, dst in pairs:
                        pltpu.sync_copy(src, dst)

            plsc.subcore_barrier()

            @pl.when((s == 0) & should_write)
            def _():
                grp_cnt_off = pl.multiple_of(
                    (w_slot * num_subcores + group_base) * LANES, LANES
                )
                pltpu.sync_copy(
                    shared_counts_vmem.at[pl.ds(grp_cnt_off, p * LANES)],
                    peer_hist_vmem.at[pl.ds(0, p * LANES)],
                )

                all_true = jnp.ones((LANES,), dtype=jnp.bool_)

                def append(src_idx, src_key, woff, m_t):
                    full_chunks = jnp.right_shift(m_t, 4)

                    def append_full(c):
                        plsc.store_compressed(
                            rowbuf_vmem.at[pl.ds(woff + c * LANES, LANES)],
                            src_idx[pl.ds(c * LANES, LANES)],
                            mask=all_true,
                        )
                        if return_scores:
                            plsc.store_compressed(
                                keybuf_vmem.at[pl.ds(woff + c * LANES, LANES)],
                                src_key[pl.ds(c * LANES, LANES)],
                                mask=all_true,
                            )

                    plsc.parallel_loop(0, full_chunks, unroll=4)(append_full)

                    tail_rem = jnp.bitwise_and(m_t, jnp.int32(LANES - 1))
                    tail_sel = lane_iota < tail_rem
                    tail_off = woff + full_chunks * LANES
                    plsc.store_compressed(
                        rowbuf_vmem.at[pl.ds(tail_off, LANES)],
                        src_idx[pl.ds(full_chunks * LANES, LANES)],
                        mask=tail_sel,
                    )
                    if return_scores:
                        plsc.store_compressed(
                            keybuf_vmem.at[pl.ds(tail_off, LANES)],
                            src_key[pl.ds(full_chunks * LANES, LANES)],
                            mask=tail_sel,
                        )

                def peer_copy(t, dst_off, sem_base=None):
                    m_t = vec_at(peer_hist_vmem, t * LANES)
                    chunks_t = _cdiv_dyn(m_t)
                    aligned_m_t = pl.multiple_of(chunks_t * LANES, LANES)
                    peer_off = pl.multiple_of(
                        (w_slot * num_subcores + group_base + t) * k, LANES
                    )
                    pairs = [
                        (
                            shared_idx_vmem.at[pl.ds(peer_off, aligned_m_t)],
                            peer_idx_vmem.at[pl.ds(dst_off, aligned_m_t)],
                        )
                    ]
                    if return_scores:
                        pairs.append(
                            (
                                shared_key_vmem.at[pl.ds(peer_off, aligned_m_t)],
                                peer_key_vmem.at[pl.ds(dst_off, aligned_m_t)],
                            )
                        )
                    cps = None
                    if sem_base is not None:
                        assert gather_sems is not None
                        cps = [
                            pltpu.make_async_copy(
                                src, dst, gather_sems.at[sem_base + i]
                            )
                            for i, (src, dst) in enumerate(pairs)
                        ]
                    return m_t, chunks_t, pairs, cps

                if gather_mode == "async":
                    # Every peer's copies in flight (own semaphores) before the
                    # first wait, then appended in subcore order.
                    stride = k + LANES

                    def start_peer(t, c):
                        _, chunks_t, _, cps = peer_copy(
                            t, pl.multiple_of(t * stride, LANES), 2 * t
                        )

                        assert cps is not None

                        @pl.when(chunks_t > 0)
                        def _():
                            for cp in cps:
                                cp.start()

                        return c

                    lax.fori_loop(1, p, start_peer, jnp.int32(0))

                    def append_peer(t, woff):
                        dst = pl.multiple_of(t * stride, LANES)
                        m_t, chunks_t, _, cps = peer_copy(t, dst, 2 * t)
                        assert cps is not None

                        @pl.when(chunks_t > 0)
                        def _():
                            for cp in cps:
                                cp.wait()
                            append(
                                peer_idx_vmem.at[pl.ds(dst, stride)],
                                peer_key_vmem.at[pl.ds(dst, stride)]
                                if return_scores
                                else None,
                                woff,
                                m_t,
                            )

                        return woff + m_t

                    lax.fori_loop(1, p, append_peer, m_s)
                else:

                    def gather_peer(t, woff):
                        m_t, chunks_t, pairs, _ = peer_copy(t, 0)

                        @pl.when(chunks_t > 0)
                        def _():
                            for src, dst in pairs:
                                pltpu.sync_copy(src, dst)
                            append(peer_idx_vmem, peer_key_vmem, woff, m_t)

                        return woff + m_t

                    plsc.parallel_loop(1, p, unroll=1, carry=m_s)(
                        gather_peer
                    )  # pytype: disable=bad-argument-type

                out_off = pl.multiple_of(row * k, LANES)
                pairs = [(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(out_off, k)])]
                if return_scores:
                    pairs.append(
                        (keybuf_vmem.at[pl.ds(0, k)], keys_hbm.at[pl.ds(out_off, k)])
                    )
                if gather_mode == "async":
                    copies(pairs, gather_sems)
                else:
                    for src, dst in pairs:
                        pltpu.sync_copy(src, dst)

    cores_per_chip = num_sc_cores * num_subcores

    def _rank_major_merge(local_idxs_hbm):
        # Subcore `flat` owns row `flat` if `flat < b`. Its `n = dcp * k`
        # scores are rank-major in HBM -- rank r's list for every row, then
        # rank r + 1's (the flat candidate all-gather) -- so the row is `dcp`
        # k-word pieces at stride `b * k`, landed back to back in VMEM: the
        # same `n` words, in the same order, as the padded path's row. The
        # row's rank and local slots are only read at emission and travel on
        # their own semaphore, in flight with the score pieces. Subcores past
        # `b` read and write nothing (P = 1 has no barriers to keep).
        assert meta_vmem is not None and slices_vmem is not None
        assert slice_sems is not None and lidx_vmem is not None
        row = flat
        active_row = row < b
        num_dmas = jnp.where(active_row, rank_major_dcp, 0)
        meta_copies = (
            pltpu.make_async_copy(
                lengths_hbm.at[pl.ds(pl.multiple_of(row * LANES, LANES), LANES)],
                meta_vmem.at[pl.ds(0, LANES)],
                slice_sems.at[1],
            ),
            pltpu.make_async_copy(
                local_idxs_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
                lidx_vmem.at[pl.ds(0, k)],
                slice_sems.at[1],
            ),
        )

        def piece(r):
            return pltpu.make_async_copy(
                scores_hbm.at[pl.ds(pl.multiple_of((r * b + row) * k, LANES), k)],
                slices_vmem.at[pl.ds(pl.multiple_of(r * k, LANES), k)],
                slice_sems.at[0],
            )

        @pl.when(active_row)
        def _():
            for c in meta_copies:
                c.start()

        @pl.loop(0, num_dmas)
        def _(r):
            piece(r).start()

        @pl.loop(0, num_dmas)
        def _(r):
            piece(r).wait()

        @pl.when(active_row)
        def _():
            for c in meta_copies:
                c.wait()
            # No pipeline output block: `_run_step` writes the row out itself.
            _run_step(
                slices_vmem,
                meta_vmem,
                None,
                None,
                lidx_vmem,
                w=jnp.int32(0),
                row_len=jnp.int32(n),
            )

    if merge_coop:
        # Cooperative DCP merge (`sparsecore_topk_dcp_merge_coop`): one wave, p
        # subcores per row, subcore s holding rank segments [s * S, (s + 1) * S)
        # of the row (S = dcp / p segments of k words, at stride b * k in the flat
        # rank-major all-gather). The radix select is the cooperative path above;
        # only the subcore holding this rank's own segment emits. The rank and
        # this row's local slots are fetched by every subcore of the row in
        # flight with the segments and the levels, and waited for only before
        # emission.
        assert num_waves == 1 and p > 1 and dcp_merge
        segs = merge_coop_dcp // p
        row = core * groups_per_core + group_id
        active_row = row < b
        safe_row = jnp.minimum(row, b - 1)
        assert lidx_vmem is not None
        rank_copy = pltpu.make_async_copy(
            lengths_hbm.at[pl.ds(pl.multiple_of(safe_row * LANES, LANES), LANES)],
            meta_vmem.at[pl.ds(0, LANES)],
            slice_sems.at[1],
        )
        slots_copy = pltpu.make_async_copy(
            local_idxs_hbm.at[pl.ds(pl.multiple_of(safe_row * k, LANES), k)],
            lidx_vmem.at[pl.ds(0, k)],
            slice_sems.at[2],
        )

        def seg_copy(j):
            seg = s * segs + j
            return pltpu.make_async_copy(
                scores_hbm.at[pl.ds(pl.multiple_of((seg * b + row) * k, LANES), k)],
                slices_vmem.at[pl.ds(j * k, k)],
                slice_sems.at[0],
            )

        @pl.when(active_row)
        def _():
            for j in range(segs):
                seg_copy(j).start()
            rank_copy.start()
            slots_copy.start()
            for j in range(segs):
                seg_copy(j).wait()

        def owner():
            @pl.when(active_row)
            def _():
                rank_copy.wait()
                slots_copy.wait()

            rank = vec_at(meta_vmem, 0)
            own = active_row & (jnp.right_shift(rank, segs.bit_length() - 1) == s)
            # The owner's segment within its slice: read by the emission below
            # through `len_slice_vmem`, as the p = 1 merge reads its rank.
            meta_vmem[pl.ds(0, LANES)] = jnp.broadcast_to(
                jnp.bitwise_and(rank, jnp.int32(segs - 1)), (LANES,)
            )
            return own

        _run_step(
            slices_vmem,
            meta_vmem,
            None,
            None,
            lidx_vmem,
            w=jnp.int32(0),
            row_len=jnp.where(active_row, n, 0),
            col_lo=s * slice_len,
            col_cap=slice_len,
            merge_owner=owner,
        )
        return
    if dcp_merge:
        # Merge rows are full by construction (n = dcp_size * k), so there is
        # nothing to bound: they stream through emit_pipeline.
        assert p == 1 and not return_scores
        if rank_major_dcp:
            _rank_major_merge(local_idxs_hbm)
            return
        # A slice past MAX_SLICE_WORDS only fits in tile memory single-buffered.
        buf_count = 1 if (num_waves == 1 or slice_len > MAX_SLICE_WORDS) else 2
        in_specs = (
            pl.BlockSpec(
                (slice_len,),
                lambda w: (w * cores_per_chip + flat,),
                pipeline_mode=pl.Buffered(buf_count),
            ),
            pl.BlockSpec(
                (LANES,),
                lambda w: (w * cores_per_chip + flat,),
                pipeline_mode=pl.Buffered(buf_count),
            ),
        )
        out_specs = (
            pl.BlockSpec(
                (k,),
                lambda w: (w * cores_per_chip + flat,),
                pipeline_mode=pl.Buffered(buf_count),
            ),
        )

        def pipeline_step_merge(scores_vmem, rank_vmem, local_idxs_vmem, out_vmem):
            _run_step(scores_vmem, rank_vmem, out_vmem, None, local_idxs_vmem)

        pltpu.emit_pipeline(
            pipeline_step_merge,
            grid=(num_waves,),
            in_specs=(*in_specs, *out_specs),
            out_specs=out_specs,
        )(scores_hbm, lengths_hbm, local_idxs_hbm, out_hbm)
        return

    def write_empty_row(row):
        """What `_run_step` writes for a row with nothing to select."""

        def fill(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
            if return_scores:
                keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES,), BITS_NEG_INF, jnp.int32
                )

        plsc.parallel_loop(0, k // LANES, unroll=4)(fill)
        out_off = pl.multiple_of(row * k, LANES)
        pltpu.sync_copy(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(out_off, k)])
        if return_scores:
            pltpu.sync_copy(keybuf_vmem.at[pl.ds(0, k)], keys_hbm.at[pl.ds(out_off, k)])

    # Each row is split over its group's p subcores by its OWN length, not by
    # the width of the score matrix: rows are usually far shorter than
    # max-model-len, and a fixed n / p split leaves one subcore doing a short
    # row while the other p - 1 wait at the barriers. The split is contiguous
    # and in column order, so ties still go to the lowest columns. Each subcore
    # fetches just its part; at p == 1 that is the row's first row_length
    # words. Every subcore copies all the row lengths in once.
    pltpu.sync_copy(lengths_hbm, meta_vmem)
    _, half = _slice_buffer(num_waves, slice_len)

    def wave_info(wave):
        """Row length, busy flag and this subcore's column range."""
        core_first = wave * groups_per_chip + core * groups_per_core
        row_len = jnp.minimum(vec_at(meta_vmem, core_first + group_id), n)
        if p == 1:
            # A lone subcore has no barriers to keep in step with, so it skips
            # its own empty rows.
            busy = row_len > 0
        else:
            # The subcores of one SparseCore share barriers, so they skip a
            # wave only together: when none of the core's rows in it has
            # anything to select. Padded token rows are empty, and a large
            # bucket holds thousands of them.
            core_lens = meta_vmem[pl.ds(core_first, LANES)]
            live = (lane_iota < groups_per_core) & (core_lens > 0)
            busy = plsc.all_reduce_population_count(live)[0] > 0
        part = _cdiv_dyn(jnp.right_shift(row_len + (p - 1), p.bit_length() - 1))
        part = jnp.minimum(pl.multiple_of(part * LANES, LANES), slice_len)
        lo = pl.multiple_of(s * part, LANES)
        words = jnp.clip(row_len - lo, 0, part)
        words = pl.multiple_of(_cdiv_dyn(words) * LANES, LANES)
        return row_len, busy, part, lo, words

    def slice_copy(wave, off, lo, words):
        row = wave * groups_per_chip + core * groups_per_core + group_id
        src = pl.multiple_of(row * (p * slice_len) + lo, LANES)
        return pltpu.make_async_copy(
            scores_hbm.at[pl.ds(src, words)],
            slices_vmem.at[pl.ds(pl.multiple_of(off, LANES), words)],
            slice_sems.at[(off > 0).astype(jnp.int32)],
        )

    def start_fetch(wave, off):
        _, _, _, lo, words = wave_info(wave)

        @pl.when(words > 0)
        def _():
            slice_copy(wave, off, lo, words).start()

    def wave_body(wave, off):
        """One wave whose part sits at `off`; returns where the next one goes."""
        off = pl.multiple_of(off, LANES)
        row_len, busy, part, lo, words = wave_info(wave)
        has_next = wave + 1 < num_waves
        next_words = wave_info(jnp.minimum(wave + 1, num_waves - 1))[4]
        # Double-buffering is decided per wave from the parts' real lengths,
        # not once from the width n. The buffer is two halves of `half` words
        # (MAX_SLICE_WORDS once n is wide), and `off` says which half holds
        # this wave's part:
        #
        #   both parts <= half:  fetch the next part into the other half,
        #                        `half - off`, now, so its DMA overlaps this
        #                        wave's selection.
        #   either part > half:  a part that long needs the whole buffer, so
        #                        the next part is fetched into 0 only after
        #                        this wave is done, and its DMA is exposed.
        #
        # A static double buffer would need two full-width slices, which do
        # not fit tile memory once n > MAX_SLICE_WORDS; that is what used to
        # force p = 2 on 32K-64K-wide rows. Real rows are far shorter than n
        # (GLM-5.2 at 1M over DCP16: 3-7K of 64K words), so nearly every wave
        # takes the first branch. Only parts longer than MAX_SLICE_WORDS
        # (contexts past 512K at DCP16) run single-buffered. Each half has its
        # own DMA semaphore (see `slice_copy`), so overlapping fetches never
        # wait on each other's bytes.
        early = has_next & (words <= half) & (next_words <= half)
        next_off = jnp.where(early, half - off, 0)
        pl.when(early)(functools.partial(start_fetch, wave + 1, next_off))

        @pl.when(words > 0)
        def _():
            slice_copy(wave, off, lo, words).wait()

        pl.when(busy)(
            functools.partial(
                _run_step,
                slices_vmem.at[pl.ds(off, slice_len)],
                None,
                w=wave,
                row_len=row_len,
                col_lo=lo,
                col_cap=part,
            )
        )
        if write_empty:
            row = wave * groups_per_chip + core * groups_per_core + group_id
            pl.when(~busy & (row < b) & (s == 0))(
                functools.partial(write_empty_row, row)
            )
        pl.when(has_next & ~early)(functools.partial(start_fetch, wave + 1, next_off))
        return next_off

    start_fetch(jnp.int32(0), jnp.int32(0))
    lax.fori_loop(0, num_waves, wave_body, jnp.int32(0))


# (b, n, p) shapes on which TPU_SC_TOPK_COOP_V2 is proven bitwise on hardware:
# the DCP8 decode buckets T = 4 / 8 / 16 of the local top-k (n = 1M / 8), and
# the a2a merge's stage 2 at T = 64 / 128 (b = T / 8 rows of n = 8 * 2048).
COOP_V2_TESTED_SHAPES = frozenset(
    {
        (4, 131072, 8),
        (8, 131072, 4),
        (16, 131072, 2),
        (8, 16384, 4),
        (16, 16384, 2),
    }
)


def _slice_buffer(num_waves: int, slice_len: int) -> tuple[int, int]:
    """(words, half) of a subcore's slice buffer on the wave loop.

    Two halves of up to MAX_SLICE_WORDS let a part that fits one be fetched
    while the previous wave's part is selected; a part up to the whole buffer
    still fits, single-buffered. One wave needs no second half.
    """
    if num_waves == 1:
        return slice_len, slice_len
    half = min(slice_len, MAX_SLICE_WORDS)
    return max(2 * half, slice_len), half


def _sc_topk_direct(
    scores: jax.Array,  # f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    local_idxs: jax.Array | None = None,  # i32[b, k]
    p: int = 1,
    write_empty: bool = True,
    return_scores: bool = True,
    scheduling_group_id: int | None = None,
    rank_major_dcp: int = 0,
    coop_v2: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Single-stage SparseCore top-k on 32 subcores.

    When ``p == 1``, each subcore processes a full row independently.
    When ``p > 1``, groups of ``p`` subcores on the same SparseCore cooperatively
    process a row using a shared global histogram in ``VMEM_SHARED``.

    ``scores`` may be f32 or the same values already bitcast to i32; the kernel
    compares raw bits either way.

    ``rank_major_dcp`` (DCP merge only, ``b <= 32``): ``scores`` is the flat
    rank-major candidate all-gather ``i32[dcp * b * k]`` -- every row's list
    from rank 0, then from rank 1, ... -- rather than ``[b, dcp * k]``. Each
    subcore fetches its own row's ``dcp`` pieces, so the scores and
    ``local_idxs`` are neither padded to 32 rows nor relaid out from 2-D tiles,
    and only ``b * k`` slots are written. Same row contents, same output.
    """
    if rank_major_dcp:
        if local_idxs is None or p != 1 or scores.ndim != 1:
            raise ValueError("rank_major_dcp needs a flat DCP merge at p = 1")
        b = local_idxs.shape[0]
        n = rank_major_dcp * k
        if scores.shape != (rank_major_dcp * b * k,) or not rank_major_merge_fits(b):
            raise ValueError(
                f"rank_major_dcp={rank_major_dcp}: scores {scores.shape} for"
                f" {b} rows of k={k}"
            )
    else:
        b, n = scores.shape
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    groups_per_chip = 32 // p
    slice_len = _align_to(n // p, LANES)
    num_waves = _cdiv(b, groups_per_chip)
    # The rank-major merge fetches only its b rows: nothing is padded.
    padded_b = b if rank_major_dcp else num_waves * groups_per_chip
    pad_b = padded_b - b
    pad_n = slice_len * p - n
    if not rank_major_dcp and (pad_b > 0 or pad_n > 0):
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))

    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b))
    wave_loop = local_idxs is None
    if wave_loop:
        # One flat copy, padded for the 16-lane reads at its end.
        lengths_in = jnp.pad(lengths, (0, LANES))
    else:
        lengths_in = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1))).reshape(-1)

    extra_inputs = ()
    if local_idxs is not None:
        local_words = local_idxs.astype(jnp.int32)
        if pad_b > 0:
            local_words = jnp.pad(local_words, ((0, pad_b), (0, 0)), constant_values=-1)
        extra_inputs = (local_words.reshape(-1),)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    out_type = (slot_type, slot_type if return_scores else None)

    # TPU_SC_TOPK_COOP_V2's gather (see `_topk_body`): "shared" at p >= 4 needs
    # an offset list; "async" at p == 2 a peer buffer per subcore of the group.
    # Single-wave calls only (b * p <= 32: the decode buckets, T <= 16 at the
    # DCP8 per-rank width); multi-wave calls (mixed / prefill chunks) keep the
    # default path.
    # P = 16 (b <= 2 rows at this width) is excluded: that configuration hung the
    # device on hardware (tests/kernels/test_sc_topk_coop_v2.py, b = 1) and no
    # decode bucket reaches it at DCP8 (T >= 4 -> P <= 8).
    # Safety net: engage only on (b, n, p) shapes this path has run bitwise on
    # hardware (tests/kernels/test_sc_topk_coop_v2.py); every other shape keeps
    # the default path.
    coop_v2 = (
        coop_v2
        and 1 < p <= 8
        and wave_loop
        and num_waves == 1
        and (b, n, p) in COOP_V2_TESTED_SHAPES
    )
    shared_gather = coop_v2 and p >= 4
    async_gather = coop_v2 and p < 4
    peer_n = p if async_gather else 1
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),  # glob_vmem
        pltpu.VMEM((k + LANES,), jnp.int32),  # rowbuf_vmem
        pltpu.VMEM((k + LANES,), jnp.int32) if return_scores else None,  # keybuf_vmem
        pltpu.VMEM((p * NUM_BUCKETS + LANES,), jnp.int32)
        if p > 1
        else None,  # peer_hist_vmem
        pltpu.VMEM((peer_n * (k + LANES),), jnp.int32)
        if p > 1
        else None,  # peer_idx_vmem
        pltpu.VMEM((peer_n * (k + LANES),), jnp.int32)
        if (p > 1 and return_scores)
        else None,  # peer_key_vmem
        pltpu.VMEM_SHARED((2 * sc.num_subcores * NUM_BUCKETS,), jnp.int32)
        if p > 1
        else None,  # shared_hist_vmem
        pltpu.VMEM_SHARED((2 * sc.num_subcores * LANES,), jnp.int32)
        if p > 1
        else None,  # shared_counts_vmem
        pltpu.VMEM_SHARED((2 * sc.num_subcores * k,), jnp.int32)
        if p > 1
        else None,  # shared_idx_vmem
        pltpu.VMEM_SHARED((2 * sc.num_subcores * k,), jnp.int32)
        if (p > 1 and return_scores)
        else None,  # shared_key_vmem
        pltpu.VMEM((padded_b + LANES,), jnp.int32) if wave_loop else None,  # meta_vmem
        pltpu.VMEM((_slice_buffer(num_waves, slice_len)[0],), jnp.int32)
        if wave_loop
        else None,  # slices_vmem
        pltpu.SemaphoreType.DMA((2,)) if wave_loop else None,  # slice_sems
        pltpu.SemaphoreType.DMA((2 * p,)) if coop_v2 else None,  # gather_sems
        pltpu.VMEM((k + LANES,), jnp.int32) if shared_gather else None,  # oidx_vmem
    )
    if rank_major_dcp:
        scratch_types = (
            *scratch_types[:-5],
            pltpu.VMEM((LANES,), jnp.int32),  # meta_vmem: this row's rank
            pltpu.VMEM((slice_len,), jnp.int32),  # slices_vmem: the row
            pltpu.SemaphoreType.DMA((2,)),  # slice_sems: [scores, rank + slots]
            None,  # gather_sems
            None,  # oidx_vmem
            pltpu.VMEM((k,), jnp.int32),  # lidx_vmem: this row's local idxs
        )

    body_fn = functools.partial(
        _topk_body,
        b=b,
        n=n,
        k=k,
        p=p,
        slice_len=slice_len,
        num_waves=num_waves,
        write_empty=write_empty,
        return_scores=return_scores,
        rank_major_dcp=rank_major_dcp,
        coop_v2=coop_v2,
    )
    if local_idxs is not None:
        orig_body = body_fn

        def body_fn(scores, lengths, idx, *rest):
            return orig_body(scores, lengths, *rest, local_idxs_hbm=idx)

    # TPU_SC_TOPK_COST_BYTES: a cost hint for XLA's latency-hiding scheduler on
    # decode-sized calls (b <= 32), which otherwise sizes the SparseCore call's
    # window far below its ~10 us and leaves independent TensorCore work (q_b)
    # outside it. Scheduling only; same values.
    cost_bytes = envs.TPU_SC_TOPK_COST_BYTES
    cost = (
        pl.CostEstimate(flops=0, transcendentals=0, bytes_accessed=cost_bytes)
        if cost_bytes > 0 and b <= 32
        else None
    )
    out = pl.kernel(
        body_fn,
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        cost_estimate=cost,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p{p}{'_v2' if coop_v2 else ''}",
    )(words.reshape(-1), lengths_in, *extra_inputs)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    # pl.kernel and set_xla_metadata are untyped upstream; both return the
    # out_type pytree here.
    slots, keys = cast(tuple[jax.Array, jax.Array | None], out)
    slots_2d = slots.reshape(padded_b, k)[:b]
    if return_scores:
        assert keys is not None
        keys_2d = keys.reshape(padded_b, k)[:b]
        return slots_2d, keys_2d
    return slots_2d


def rank_major_merge_fits(b: int) -> bool:
    """Whether `sparsecore_topk_dcp_merge` takes ``rank_major`` scores for
    ``b`` rows: one wave of the 32 P = 1 subcores, ``b <= 32``. Wider calls
    keep the pipelined, double-buffered row read (``b > 32``)."""
    return b <= 32


def _pick_partition(b: int, n: int, k: int) -> int:
    """Picks the power-of-two partition factor P (1..16) per row to maximize

    subcore utilization across subcores while fitting in VMEM and VMEM_SHARED.
    """
    # A cooperative group must fit on one SparseCore's 16 subcores. A subcore
    # holds its part of a row in up to two MAX_SLICE_WORDS buffers, which one
    # part may fill alone (single-buffered): fewer subcores per row means more
    # rows per wave, and every cooperative wave pays a fixed synchronization
    # cost. That holds down to p == 1, which has no synchronization at all.
    p_min = min(16, _cdiv(n, 2 * MAX_SLICE_WORDS))
    p = 1
    while p < p_min:
        p *= 2

    max_p = min(16, 32 // b) if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "write_empty_rows",
        "scheduling_group_id",
        "return_scores",
        "coop_v2",
    ),
)
def sparsecore_topk(
    scores: jax.Array,  # f32[b, n] or i32[b, n] of f32 bits
    k: int,
    row_lengths: jax.Array | None = None,  # i32[b], defaults to n
    *,
    write_empty_rows: bool = True,
    scheduling_group_id: int | None = None,
    return_scores: bool = False,
    coop_v2: bool | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:  # i32[b, k] or (i32[b, k], i32[b, k])
    """Exact top-k indices per row, unsorted, -1 suffix-padded.

    ``coop_v2`` (None reads ``TPU_SC_TOPK_COOP_V2``): at ``p > 1``, the level
    loop and the gather described in the module docstring. Bitwise identical.

    When ``p > 1`` (for ``b < 32`` or ``n > 64K``), subcores on the same
    SparseCore cooperatively compute a global histogram in ``VMEM_SHARED`` to
    find the exact global threshold in a single stage without writing intermediate
    candidates back to HBM.

    ``scores`` may be f32 or i32 holding those f32 bits; the comparison is on
    raw bits either way.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype not in (jnp.float32, jnp.int32):
        raise ValueError(f"scores must be f32 or i32, got {scores.dtype}")
    b, n = scores.shape
    if n % LANES != 0:
        raise ValueError(f"{n=} must be a multiple of {LANES}")
    if k % LANES != 0 or not 0 < k <= 4096:
        raise ValueError(f"{k=} must be a positive multiple of {LANES}, at most 4096")

    if row_lengths is None:
        row_lengths = jnp.full((b,), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    if coop_v2 is None:
        coop_v2 = envs.TPU_SC_TOPK_COOP_V2
    p = _pick_partition(b, n, k)
    return _sc_topk_direct(
        scores,
        k,
        row_lengths,
        p=p,
        write_empty=write_empty_rows,
        return_scores=return_scores,
        scheduling_group_id=scheduling_group_id,
        coop_v2=coop_v2,
    )


@functools.partial(
    jax.jit,
    static_argnames=("k", "dcp_size", "scheduling_group_id", "rank_major"),
)
def sparsecore_topk_dcp_merge(
    scores: jax.Array,  # [b, dcp_size * k], or [dcp_size * b * k] if rank_major
    local_idxs: jax.Array,  # i32[b, k]
    k: int,
    *,
    dcp_rank: jax.Array,  # i32[]
    dcp_size: int,
    scheduling_group_id: int | None = None,
    rank_major: bool = False,
) -> jax.Array:  # i32[b, k]
    """Fused global top-k threshold selection, owner filter, and prefix compact.

    ``rank_major`` (``rank_major_merge_fits(b)``): ``scores`` is the flat
    candidate all-gather of each rank's ``[b * k]`` list, ``scores[r * b * k +
    t * k + j] == scores_2d[t, r * k + j]``. The kernel reads the rows in place,
    with no 32-row pad and no 2-D relayout of the scores or ``local_idxs``.
    Same result.
    """
    b = local_idxs.shape[0]
    want = (dcp_size * b * k,) if rank_major else (b, dcp_size * k)
    if scores.shape != want or local_idxs.shape != (b, k):
        raise ValueError(
            f"Expected scores {want} and local_idxs {(b, k)}, got"
            f" {scores.shape} and {local_idxs.shape}"
        )
    if rank_major and not rank_major_merge_fits(b):
        raise ValueError(f"rank_major merge needs b <= 32, got {b}")
    # T <= 16 (one wave of p >= 2 subcores per row); T > 16 stays below.
    if rank_major and envs.TPU_SC_TOPK_COOP_MERGE and coop_merge_fits(b, dcp_size, k):
        # pyrefly: ignore[missing-attribute]
        return sparsecore_topk_dcp_merge_coop.__wrapped__(
            scores,
            local_idxs,
            k,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
            scheduling_group_id=scheduling_group_id,
        )
    ranks = jnp.full((b,), jnp.reshape(dcp_rank.astype(jnp.int32), ()), jnp.int32)
    out = _sc_topk_direct(
        scores,
        k,
        ranks,
        local_idxs=local_idxs,
        p=1,
        write_empty=True,
        return_scores=False,
        scheduling_group_id=scheduling_group_id,
        rank_major_dcp=dcp_size if rank_major else 0,
    )
    assert isinstance(out, jax.Array)
    return out


def coop_merge_fits(b: int, dcp_size: int, k: int) -> bool:
    """Whether `sparsecore_topk_dcp_merge_coop` takes this merge: one wave of
    ``p = min(dcp_size, 32 // b)`` subcores per row with ``p >= 2`` (so
    ``b <= 16``), ``p`` dividing ``dcp_size``, both powers of two."""
    if b < 1 or b > 16 or k % LANES or dcp_size < 2 or dcp_size & (dcp_size - 1):
        return False
    return dcp_size % min(dcp_size, 32 // b) == 0


@functools.partial(
    jax.jit,
    static_argnames=("k", "dcp_size", "scheduling_group_id"),
)
def sparsecore_topk_dcp_merge_coop(
    # i32[dcp_size * b * k] (or f32 bits), rank-major flat all-gather
    scores: jax.Array,
    local_idxs: jax.Array,  # i32[b, k]
    k: int,
    *,
    dcp_rank: jax.Array,  # i32[]
    dcp_size: int,
    scheduling_group_id: int | None = None,
) -> jax.Array:  # i32[b, k]
    """`sparsecore_topk.sparsecore_topk_dcp_merge(rank_major=True)` with each row
    split over ``p = min(dcp_size, 32 // b)`` subcores by rank segment instead of
    one subcore per row (``TPU_SC_TOPK_COOP_MERGE``; ``coop_merge_fits``).

    Subcore s of a row's group holds rank segments ``[s * S, (s + 1) * S)``
    (``S = dcp_size / p``) and the group runs the cooperative radix select of
    `sparsecore_topk`'s ``p > 1`` path over the row's ``dcp_size * k`` scores;
    only the subcore holding ``dcp_rank``'s segment emits (ties before it: the
    peers' tie counts plus its own earlier segments'), and it writes the row
    itself. Same output, bit for bit: this rank's winners in its own slot
    order, resolved through ``local_idxs``, ``-1`` suffix-padded.
    """
    b = local_idxs.shape[0]
    if scores.shape != (dcp_size * b * k,) or local_idxs.shape != (b, k):
        raise ValueError(
            f"Expected scores {(dcp_size * b * k,)} and local_idxs {(b, k)}, got"
            f" {scores.shape} and {local_idxs.shape}"
        )
    if not coop_merge_fits(b, dcp_size, k):
        raise ValueError(
            "coop merge needs b <= 16 and a power-of-two dcp_size; "
            f"got {b=}, {dcp_size=}"
        )
    p = min(dcp_size, 32 // b)
    n = dcp_size * k
    slice_len = (dcp_size // p) * k
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    ranks = jnp.full((b,), jnp.reshape(dcp_rank.astype(jnp.int32), ()), jnp.int32)
    lengths_in = jnp.pad(ranks[:, None], ((0, 0), (0, LANES - 1))).reshape(-1)
    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),  # glob_vmem
        pltpu.VMEM((k + LANES,), jnp.int32),  # rowbuf_vmem
        None,  # keybuf_vmem
        pltpu.VMEM((p * NUM_BUCKETS + LANES,), jnp.int32),  # peer_hist_vmem
        None,  # peer_idx_vmem
        None,  # peer_key_vmem
        pltpu.VMEM_SHARED(
            (2 * sc.num_subcores * NUM_BUCKETS,), jnp.int32
        ),  # shared_hist
        None,  # shared_counts_vmem
        None,  # shared_idx_vmem
        None,  # shared_key_vmem
        pltpu.VMEM(
            (2 * LANES,), jnp.int32
        ),  # meta_vmem: rank, then the owner's segment
        pltpu.VMEM((slice_len + LANES,), jnp.int32),  # slices_vmem
        pltpu.SemaphoreType.DMA((3,)),  # slice_sems: [segments, rank, local slots]
        None,  # gather_sems
        None,  # oidx_vmem
        pltpu.VMEM((k + LANES,), jnp.int32),  # lidx_vmem
    )
    body = functools.partial(
        _topk_body,
        b=b,
        n=n,
        k=k,
        p=p,
        slice_len=slice_len,
        num_waves=1,
        write_empty=True,
        return_scores=False,
        merge_coop_dcp=dcp_size,
    )

    def body_fn(scores_hbm, lengths_hbm, idx_hbm, out_hbm, *rest):
        return body(
            scores_hbm, lengths_hbm, out_hbm, None, *rest, local_idxs_hbm=idx_hbm
        )

    out = pl.kernel(
        body_fn,
        out_type=jax.ShapeDtypeStruct((b * k,), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p{p}_coop",
        # pyrefly: ignore[bad-argument-count]
    )(words.reshape(-1), lengths_in, local_idxs.astype(jnp.int32).reshape(-1))
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    return cast(jax.Array, out).reshape(b, k)
