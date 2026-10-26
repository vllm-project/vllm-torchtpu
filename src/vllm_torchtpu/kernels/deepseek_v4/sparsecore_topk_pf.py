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
"""Exact SparseCore top-k of long score rows, pre-filtered by group maxima.

The StreamIndex scorer (TensorCore) also writes, per row, the maximum monotone
key of every GROUP = 16 consecutive score columns (`gm`, one word per group,
packed 128 groups per 2048 columns: packed lane `16 j + i` of gm row `r` holds
the group starting at column `2048 r + 128 i + 16 j`). Per row, this kernel:

1. reads the row's gm words (1/16 of the scores) and selects the exact k-th
   largest group maximum `tau` (radix select, as `sparsecore_topk`);
2. collects, in column order, every group whose maximum is >= tau (plus the
   group cut by the row length, which is excluded from step 1 because its
   maximum may include invisible columns);
3. gathers just those groups' 16 scores (64-byte rows) and runs the exact top-k
   over them, emitting columns in ascending order with ties to the lowest
   columns -- what `sparsecore_topk` emits.

Exact: the k largest group maxima are k distinct visible scores >= tau, so the
k-th largest score s_k >= tau, and every score >= s_k lies in a group whose
maximum is >= tau. The candidate set therefore holds every score >= s_k (ties
included), in column order, and the selection over it is the selection over
the row.

A row whose candidate groups exceed the buffer (`cap` groups: only possible
with massive exact ties at tau) is flagged; the caller recomputes it with the
full-row kernel. Rows shorter than `min_len` columns are left to the caller.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    BITS_NEG_INF,
    KEY_NEG_INF,
    MONO_MASK,
    sparsecore_topk,
)

LANES = 16
GROUP = 16  # columns per group (one 16-lane vector, a 64-byte gather row)
GM_ROW = 2048  # columns covered by one packed 128-lane gm row
NUM_BUCKETS = 256
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
NBUF = 8  # gather DMAs in flight
# Test-only fault injection (positive controls), read at trace time:
# tau_hi | nocut | order | cut_in_tau. Empty in production.
import os as _os
FAULT = _os.environ.get("TPU_KIDXM_FAULT", "")


def monotone_key(bits):
    m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
    return jnp.bitwise_xor(bits, m)


def gm_reference(scores: jax.Array) -> jax.Array:
    """[b, n // 128, 128] i32 score bits -> [b, n // GM_ROW, 128] packed group-max keys."""
    b, nsub, _ = scores.shape
    n = nsub * 128
    keys = monotone_key(scores.reshape(b, n // GM_ROW, 16, 8, GROUP))  # [.., i, j, e]
    m = keys.max(-1)  # [b, R, i(16 tiles), j(8 groups)]
    return m.transpose(0, 1, 3, 2).reshape(b, n // GM_ROW, 128)  # lane 16 j + i


def _pf_body(
    rows_hbm,  # i32[b * n // GROUP, GROUP]: score rows viewed as 16-column groups
    gm_hbm,  # i32[b * n // GROUP]: packed group-max keys
    lens_hbm,  # i32[padded_b + LANES]
    out_hbm,  # i32[padded_b * k]
    flag_hbm,  # i32[padded_b * LANES]
    meta_v,
    gm_v,
    hist_v,
    cpos_v,
    cidx_v,
    cval_v,
    rowbuf_v,
    flag_v,
    gsem,
    *,
    b,
    n,
    k,
    num_waves,
    cap,
):
    num_subcores = 16
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub
    iota = jnp.arange(LANES, dtype=jnp.int32)
    neg = jnp.int32(KEY_NEG_INF)
    gm_per_row = n // GROUP
    rows_per_row = n // GROUP

    pltpu.sync_copy(lens_hbm, meta_v)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def zero_hist(bound):
        def body(i):
            hist_v[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket, mask):
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(hist_v, (bucket,), dup, mask=last)

    def scan_hist(bound):
        def body(i, carry):
            c = plsc.cumsum(hist_v[pl.ds(i * LANES, LANES)]) + carry
            hist_v[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)

    def find_bucket(bound, thresh):
        def body(i, cnt):
            below = hist_v[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)

    def key_window(prefix, shift):
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg + 1) - 1, jnp.maximum(hi, neg)

    def radix_select(load_key, nvec, kk):
        """Exact boundary of the kk largest keys (> -inf) among nvec 16-lane vectors.

        Returns (lo_excl, hi_incl, quota, cnt, keep_all): the kk largest are every
        key > hi_incl plus `quota` of the `cnt` keys in (lo_excl, hi_incl]."""
        zero_hist(NUM_BUCKETS // LANES)

        def h0(i):
            key = load_key(i)
            hist_add(jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2), key > neg)

        plsc.parallel_loop(0, nvec, unroll=8)(h0)
        total = scan_hist(NUM_BUCKETS // LANES)
        keep_all = total <= kk
        b_star = find_bucket(NUM_BUCKETS // LANES, total - kk)
        cum_at = vec_at(hist_v, b_star)
        cum_lo = jnp.where(b_star > 0, vec_at(hist_v, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = kk - c_hi
        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)
        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            zero_hist(hist_bound)
            lo_excl, hi_incl = key_window(prefix, shift)

            def rb(i, lvl_shift=lvl_shift, lvl_mask=lvl_mask, lo_excl=lo_excl, hi_incl=hi_incl):
                key = load_key(i)
                match = (key > lo_excl) & (key <= hi_incl)
                hist_add(jnp.bitwise_and(jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)), match)

            plsc.parallel_loop(0, jnp.where(done, 0, nvec), unroll=8)(rb)
            sub_total = scan_hist(hist_bound)
            b2 = find_bucket(hist_bound, sub_total - quota)
            cum_at2 = vec_at(hist_v, b2)
            cum_lo2 = jnp.where(b2 > 0, vec_at(hist_v, jnp.maximum(b2 - 1, 0)), 0)
            c_above2 = sub_total - cum_at2
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - c_above2
            width = shift - lvl_shift
            new_prefix = jnp.left_shift(prefix, width) | b2
            new_done = done | (quota2 == cnt2) | (quota2 == 0)
            prefix = jnp.where(done, prefix, new_prefix)
            shift = jnp.where(done, shift, jnp.int32(lvl_shift))
            quota = jnp.where(done, quota, quota2)
            cnt = jnp.where(done, cnt, cnt2)
            done = new_done
        lo_excl, hi_incl = key_window(prefix, shift)
        return lo_excl, hi_incl, quota, cnt, keep_all

    def process(row, row_len):
        nblk = (row_len + GM_ROW - 1) // GM_ROW
        ngv = nblk * (128 // LANES)  # gm vectors
        # Whole gm rows up to the length (multiple of 128 words).
        src = pl.multiple_of(row * gm_per_row, 128)
        pltpu.sync_copy(
            gm_hbm.at[pl.ds(src, pl.multiple_of(nblk * 128, 128))],
            gm_v.at[pl.ds(0, pl.multiple_of(nblk * 128, 128))],
        )
        # Last gm row: groups not wholly visible are excluded from tau.
        last = (nblk - 1) * 128
        for v in range(128 // LANES):
            w = v * LANES + iota
            start = (nblk - 1) * GM_ROW + (w & 15) * 128 + jnp.right_shift(w, 4) * GROUP
            off = last + v * LANES
            vis = (start < row_len) if FAULT == "cut_in_tau" else (start + GROUP <= row_len)
            gm_v[pl.ds(off, LANES)] = jnp.where(vis, gm_v[pl.ds(off, LANES)], neg)
        lo_excl, hi_incl, quota, cnt, keep_all_g = radix_select(
            lambda i: gm_v[pl.ds(i * LANES, LANES)], ngv, k
        )
        # Every group with max >= tau: the k largest maxima are k distinct
        # visible scores, so the k-th largest score is >= tau. tau is the lower
        # bound of the final radix window (the exact key once fully resolved).
        tau = jnp.where(keep_all_g, neg + 1, (hi_incl + 1) if FAULT == "tau_hi" else (lo_excl + 1))
        del hi_incl, quota, cnt

        # Candidate groups in column order: gm row r, tile i, group j ->
        # column 2048 r + 128 i + 16 j; a vector covers tiles (2 ii, 2 ii + 1).
        jj = jnp.bitwise_and(iota, 7)
        ti = jnp.right_shift(iota, 3)
        if FAULT == "order":  # walk tiles in reverse within a gm row
            ti = 1 - ti
        cap_v = jnp.int32(cap - LANES)

        def walk(t, carry):
            c, total = carry
            r = jnp.right_shift(t, 3)
            i = 2 * jnp.bitwise_and(t, 7) + ti
            key = plsc.load_gather(gm_v, [r * 128 + 16 * jj + i])
            sel = key >= tau
            col = r * GM_ROW + i * 128 + jj * GROUP
            plsc.store_compressed(cpos_v.at[pl.ds(c, LANES)], col, mask=sel & (c <= cap_v))
            npop = plsc.all_reduce_population_count(sel)[0]
            return jnp.where(c <= cap_v, c + npop, c), total + npop

        c, total = plsc.parallel_loop(0, nblk * 8, unroll=2, carry=(jnp.int32(0), jnp.int32(0)))(walk)
        # The group cut by the row length, if any, is always a candidate.
        cut = (jnp.bitwise_and(row_len, GROUP - 1) != 0) & (FAULT != "nocut")
        cut_col = jnp.right_shift(row_len, 4) * GROUP
        plsc.store_compressed(
            cpos_v.at[pl.ds(c, LANES)],
            jnp.broadcast_to(cut_col, (LANES,)),
            mask=(iota == 0) & cut & (c <= cap_v),
        )
        c = jnp.where(cut & (c <= cap_v), c + 1, c)
        total = jnp.where(cut, total + 1, total)
        overflow = total > c
        flag_v[pl.ds(0, LANES)] = jnp.broadcast_to(overflow.astype(jnp.int32), (LANES,))
        pltpu.sync_copy(flag_v, flag_hbm.at[pl.ds(pl.multiple_of(row * LANES, LANES), LANES)])

        @pl.when(jnp.logical_not(overflow))
        def _():
            # Gather row indices; pad the last chunk with distinct valid rows.
            nch = (c + LANES - 1) // LANES
            base_row = row * rows_per_row

            def mk_idx(t):
                p = cpos_v[pl.ds(t * LANES, LANES)]
                valid = t * LANES + iota < c
                cidx_v[pl.ds(t * LANES, LANES)] = jnp.where(
                    valid, base_row + jnp.right_shift(p, 4), base_row + iota
                )

            plsc.parallel_loop(0, nch, unroll=2)(mk_idx)

            def g_desc(t, s):
                return pltpu.make_async_copy(
                    rows_hbm.at[cidx_v.at[pl.ds(pl.multiple_of(t * LANES, LANES), LANES)]],
                    cval_v.at[pl.ds(pl.multiple_of(t * LANES, LANES), LANES)],
                    gsem.at[s],
                )

            for s in range(NBUF):
                @pl.when(s < nch)
                def _():
                    g_desc(s, s).start()

            @pl.loop(0, (nch + NBUF - 1) // NBUF)
            def _(tp):
                for s in range(NBUF):
                    t = tp * NBUF + s

                    @pl.when(t < nch)
                    def _():
                        g_desc(t, s).wait()

                        @pl.when(t + NBUF < nch)
                        def _():
                            g_desc(t + NBUF, s).start()

            # Candidate row q holds columns cpos[q] .. cpos[q] + 15; columns at
            # or past the row length (the cut group) read as -inf.
            def load_key(q):
                col0 = vec_at(cpos_v, q)
                bits = cval_v[q, :]
                key = monotone_key(bits)
                return jnp.where(col0 + iota < row_len, key, neg)

            lo_excl, hi_incl, quota, cnt, keep_all = radix_select(load_key, c, k)
            my_take = jnp.where(keep_all, 0, quota)
            take_all_ties = keep_all | (my_take >= cnt)
            gate = jnp.where(take_all_ties, lo_excl, hi_incl)
            whole_bucket = take_all_ties | (my_take <= 0)

            def fill(i):
                rowbuf_v[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

            plsc.parallel_loop(0, jnp.where(keep_all, k // LANES, 0), unroll=4)(fill)

            def emit(q, woff, sel):
                col = vec_at(cpos_v, q) + iota
                plsc.store_compressed(rowbuf_v.at[pl.ds(woff, LANES)], col, mask=sel)
                return woff + plsc.all_reduce_population_count(sel)[0]

            def emit_whole(q, woff):
                return emit(q, woff, load_key(q) > gate)

            m_whole = plsc.parallel_loop(0, jnp.where(whole_bucket, c, 0), unroll=4, carry=jnp.int32(0))(emit_whole)
            del m_whole

            def emit_ranked(q, carry):
                woff, rank = carry
                key = load_key(q)
                in_set = key > lo_excl
                strict = key > hi_incl
                tie_rank = plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
                sel = strict | (in_set & (tie_rank <= my_take))
                return emit(q, woff, sel), tie_rank[LANES - 1]

            plsc.parallel_loop(0, jnp.where(whole_bucket, 0, c), unroll=4, carry=(jnp.int32(0), jnp.int32(0)))(emit_ranked)
            out_off = pl.multiple_of(row * k, LANES)
            pltpu.sync_copy(rowbuf_v.at[pl.ds(0, k)], out_hbm.at[pl.ds(out_off, k)])

    def wave_body(wave, carry):
        row = wave * 32 + flat
        row_len = jnp.minimum(vec_at(meta_v, row), n)
        busy = (row < b) & (row_len > 0)
        pl.when(busy)(functools.partial(process, row, row_len))
        return carry

    lax.fori_loop(0, num_waves, wave_body, jnp.int32(0))


@functools.partial(jax.jit, static_argnames=("k", "cap"))
def sparsecore_topk_pf_long(
    scores: jax.Array,  # i32[b, n // 128, 128] score bits
    gm: jax.Array,  # i32[b, n // GM_ROW, 128] packed group-max keys
    row_lengths: jax.Array,  # i32[b]; 0 = row not handled here
    *,
    k: int,
    cap: int = 4096,
) -> tuple[jax.Array, jax.Array]:
    """(i32[b, k] indices of the rows with length > 0, i32[b] overflow flags)."""
    b = scores.shape[0]
    n = math.prod(scores.shape[1:])
    assert n % GM_ROW == 0 and gm.shape == (b, n // GM_ROW, 128), (scores.shape, gm.shape)
    assert k % LANES == 0 and cap % LANES == 0 and cap >= k + LANES
    num_waves = -(-b // 32)
    padded_b = num_waves * 32
    lens = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b + LANES))
    sc = pltpu.get_tpu_info().sparse_core
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores, num_subcores=sc.num_subcores,
        core_axis_name="core", subcore_axis_name="subcore",
    )
    out, flags = pl.kernel(
        functools.partial(_pf_body, b=b, n=n, k=k, num_waves=num_waves, cap=cap),
        out_type=(
            jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
            jax.ShapeDtypeStruct((padded_b * LANES,), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM((padded_b + LANES,), jnp.int32),  # meta_v
            pltpu.VMEM((n // GROUP + LANES,), jnp.int32),  # gm_v
            pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),  # hist_v
            pltpu.VMEM((cap + 2 * LANES,), jnp.int32),  # cpos_v
            pltpu.VMEM((cap + 2 * LANES,), jnp.int32),  # cidx_v
            pltpu.VMEM((cap + 2 * LANES, GROUP), jnp.int32),  # cval_v
            pltpu.VMEM((k + LANES,), jnp.int32),  # rowbuf_v
            pltpu.VMEM((LANES,), jnp.int32),  # flag_v
            pltpu.SemaphoreType.DMA((NBUF,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False, disable_bounds_checks=True
        ),
        mesh=mesh,
        name=f"sc_topk_pf_b{b}_n{n}_k{k}",
    )(scores.reshape(b * n // GROUP, GROUP), gm.reshape(-1), lens)
    return out.reshape(padded_b, k)[:b], flags.reshape(padded_b, LANES)[:b, 0]


def sparsecore_topk_pf(
    scores: jax.Array,  # i32[b, n // 128, 128] score bits
    gm: jax.Array,  # i32[b, n // GM_ROW, 128]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    eligible: jax.Array | None = None,  # bool[b]: rows whose gm was written
    min_len: int | None = None,
    **topk_kwargs,
) -> jax.Array:
    """Exact top-k (as `sparsecore_topk`) using the group maxima for long rows."""
    if min_len is None:
        min_len = 16 * k + GROUP
    lens = row_lengths.astype(jnp.int32)
    long = lens >= min_len
    if eligible is not None:
        long = long & eligible
    pf, overflow = sparsecore_topk_pf_long(scores, gm, jnp.where(long, lens, 0), k=k)
    # One full-row call, after the prefilter, for every row it did not take: the
    # short / ineligible rows and any overflow row (no lax.cond: a conditional
    # around a SparseCore launch costs more than the launch, STRAT-TKD-01).
    took = long & (overflow == 0)
    rest = sparsecore_topk(scores, k, row_lengths=jnp.where(took, 0, lens), **topk_kwargs)
    return jnp.where(took[:, None], pf, rest)
