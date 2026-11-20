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
"""Single-subcore two-pass compacting SparseCore top-k for wide batches."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 16
SCAN_UNROLL = 8
MONO_MASK = 0x7FFFFFFF
BITS_NEG_INF = int(np.array(-np.inf, np.float32).view(np.int32))
KEY_NEG_INF = int(np.int32(BITS_NEG_INF) ^ np.int32(MONO_MASK))

_SC_SAFE_SUBCORE_WORDS = 126 * 1024
_MAX_P1_DBLBUF_WORDS = 56 * 1024
_MIN_DBLBUF_WORDS = 32 * 1024
_MAX_RESIDENT_P1_WORDS = 122880
_MIN_RESIDENT_COMPACT_WORDS = 49152
_RESIDENT_L0_BITS = 11
_STREAMING_L0_BITS = 12
_STREAMING_TILE_WORDS = 16 * 1024

CompactMode = Literal["resident", "streaming"]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
    return _cdiv(x, a) * a


def _cdiv_dyn(x):
    return jnp.right_shift(x + (LANES - 1), 4)


def _refine_levels_for(l0_bits: int) -> tuple[tuple[int, int, int], ...]:
    rem = 32 - l0_bits
    b1 = (rem + 1) // 2
    b2 = rem - b1
    return (
        (b2, (1 << b1) - 1, 1 << b1),
        (0, (1 << b2) - 1, 1 << b2),
    )


def _candidate_capacity(k: int, l0_bits: int) -> int:
    floor = 7168 if l0_bits <= 11 else 6144
    return max(floor, _align_to(k + 4096, LANES))


def _resident_vmem_plan(b: int, n: int, k: int) -> tuple[int, int, int, int] | None:
    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)
    padded_b = num_waves * 32
    l0_buckets = 1 << _RESIDENT_L0_BITS
    ref_levels = _refine_levels_for(_RESIDENT_L0_BITS)
    hist_words = max(l0_buckets, *(lv[2] for lv in ref_levels))
    cand_cap = _candidate_capacity(k, _RESIDENT_L0_BITS)
    fixed = (hist_words + LANES) + (k + LANES) + (cand_cap + LANES) + (padded_b + LANES)
    if fixed + slice_len > _SC_SAFE_SUBCORE_WORDS:
        return None
    avail_half = ((_SC_SAFE_SUBCORE_WORDS - fixed) // 2 // LANES) * LANES
    if num_waves == 1:
        buf_words, half = slice_len, slice_len
    else:
        max_half = min(_MAX_P1_DBLBUF_WORDS, max(_MIN_DBLBUF_WORDS, avail_half))
        half = min(slice_len, max_half)
        buf_words = max(2 * half, slice_len)
        if fixed + buf_words > _SC_SAFE_SUBCORE_WORDS:
            buf_words = slice_len
            half = min(slice_len, max(LANES, avail_half))
    return hist_words, cand_cap, buf_words, half


def _streaming_vmem_plan(b: int, k: int) -> tuple[int, int] | None:
    padded_b = _cdiv(b, 32) * 32
    l0_buckets = 1 << _STREAMING_L0_BITS
    ref_levels = _refine_levels_for(_STREAMING_L0_BITS)
    hist_words = max(l0_buckets, *(lv[2] for lv in ref_levels))
    cand_cap = _candidate_capacity(k, _STREAMING_L0_BITS)
    total = (
        (hist_words + LANES)
        + (k + LANES)
        + 2 * (cand_cap + LANES)
        + (padded_b + LANES)
        + 2 * _STREAMING_TILE_WORDS
    )
    if total > _SC_SAFE_SUBCORE_WORDS:
        return None
    return hist_words, cand_cap


def select_compact_p1_mode(
    b: int,
    n: int,
    k: int,
    *,
    return_scores: bool,
) -> CompactMode | None:
    """Return compact P=1 strategy for wide batches, or None for default path."""
    if b < 32 or return_scores:
        return None
    if n > _MAX_RESIDENT_P1_WORDS:
        return "streaming" if _streaming_vmem_plan(b, k) is not None else None
    if n >= _MIN_RESIDENT_COMPACT_WORDS:
        if _resident_vmem_plan(b, n, k) is not None:
            return "resident"
        if _streaming_vmem_plan(b, k) is not None:
            return "streaming"
    return None


def _make_radix_ops(glob_vmem, lane_iota, neg_inf_key):
    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        mask = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, mask)

    def key_window(prefix, shift):
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound):
        def z(i):
            glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=SCAN_UNROLL)(z)

    def hist_add(bucket, mask):
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def scan_glob(bound):
        def sbody(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(sbody)

    def find_bucket(bound, thresh):
        def fbody(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(fbody)

    return vec_at, monotone_key, key_window, zero_hist, hist_add, scan_glob, find_bucket


def _refine_and_emit_candidates(
    *,
    coff0,
    prefix0,
    l0_shift: int,
    quota,
    cnt,
    ref_levels: tuple[tuple[int, int, int], ...],
    cand_idx_vmem,
    rowbuf_vmem,
    glob_vmem,
    lane_iota,
    vec_at: Callable,
    key_window: Callable,
    zero_hist: Callable,
    hist_add: Callable,
    scan_glob: Callable,
    find_bucket: Callable,
    load_cand_key: Callable,
):
    cand_chunks = _cdiv_dyn(coff0)
    c_prefix = prefix0
    c_shift = jnp.int32(l0_shift)
    c_quota = quota
    c_cnt = cnt
    c_done = jnp.bool_(False)

    for lvl_shift, lvl_mask, lvl_buckets in ref_levels:
        hb = jnp.where(c_done, 0, lvl_buckets // LANES)
        zero_hist(hb)
        aw = jnp.where(c_done, 0, cand_chunks)
        lo_e, hi_i = key_window(c_prefix, c_shift)

        def c_ref(
            i,
            *,
            lo_e=lo_e,
            hi_i=hi_i,
            lvl_shift=lvl_shift,
            lvl_mask=lvl_mask,
        ):
            pos = i * LANES + lane_iota
            valid = pos < coff0
            idx_v = jnp.where(valid, cand_idx_vmem[pl.ds(i * LANES, LANES)], 0)
            key = load_cand_key(i, idx_v, valid)
            match = valid & (key > lo_e) & (key <= hi_i)
            bucket = jnp.bitwise_and(
                jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
            )
            hist_add(bucket, match)

        plsc.parallel_loop(0, aw, unroll=4)(c_ref)
        sub_tot = scan_glob(hb)
        b2 = find_bucket(hb, sub_tot - c_quota)
        ca2 = vec_at(glob_vmem, b2)
        cl2 = jnp.where(b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
        c_ab2 = sub_tot - ca2
        cnt2 = ca2 - cl2
        q2 = c_quota - c_ab2
        wbits = c_shift - jnp.int32(lvl_shift)
        npref = jnp.left_shift(c_prefix, wbits) | b2
        ndone = c_done | (q2 == cnt2) | (q2 == 0)
        c_prefix = jnp.where(c_done, c_prefix, npref)
        c_shift = jnp.where(c_done, c_shift, jnp.int32(lvl_shift))
        c_quota = jnp.where(c_done, c_quota, q2)
        c_cnt = jnp.where(c_done, c_cnt, cnt2)
        c_done = ndone

    lo_f, hi_f = key_window(c_prefix, c_shift)
    take_all_t = c_quota >= c_cnt
    gate_f = jnp.where(take_all_t, lo_f, hi_f)
    whole_f = take_all_t | (c_quota <= 0)

    def c_em_w(i, woff):
        pos = i * LANES + lane_iota
        valid = pos < coff0
        idx_v = jnp.where(valid, cand_idx_vmem[pl.ds(i * LANES, LANES)], 0)
        key = load_cand_key(i, idx_v, valid)
        sel = valid & (key > gate_f)
        plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)], idx_v, mask=sel)
        return woff + plsc.all_reduce_population_count(sel)[0]

    plsc.parallel_loop(
        0, jnp.where(whole_f, cand_chunks, 0), unroll=4, carry=jnp.int32(0)
    )(c_em_w)

    def c_em_r(i, carry):
        woff, rank = carry
        pos = i * LANES + lane_iota
        valid = pos < coff0
        idx_v = jnp.where(valid, cand_idx_vmem[pl.ds(i * LANES, LANES)], 0)
        key = load_cand_key(i, idx_v, valid)
        in_set = valid & (key > lo_f)
        strict = valid & (key > hi_f)
        tie_rank = (
            plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
        )
        sel = strict | (in_set & (tie_rank <= c_quota))
        plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)], idx_v, mask=sel)
        return woff + plsc.all_reduce_population_count(sel)[0], tie_rank[LANES - 1]

    plsc.parallel_loop(
        0,
        jnp.where(whole_f, 0, cand_chunks),
        unroll=4,
        carry=(jnp.int32(0), jnp.int32(0)),
    )(c_em_r)


def _run_resident_1comp(
    words: jax.Array,
    lengths_in: jax.Array,
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    padded_b: int,
    write_empty: bool,
    scheduling_group_id: int | None,
    mesh: plsc.VectorSubcoreMesh,
) -> jax.Array:
    plan = _resident_vmem_plan(b, n, k)
    assert plan is not None
    hist_words, cand_cap, buf_words, half = plan
    l0_bits = _RESIDENT_L0_BITS
    l0_shift = 32 - l0_bits
    l0_buckets = 1 << l0_bits
    ref_levels = _refine_levels_for(l0_bits)

    def body(
        scores_hbm,
        lengths_hbm,
        out_hbm,
        glob_vmem,
        rowbuf_vmem,
        cand_idx_vmem,
        meta_vmem,
        slices_vmem,
        slice_sems,
    ):
        flat = lax.axis_index("core") * 16 + lax.axis_index("subcore")
        lane_iota = jnp.arange(LANES, dtype=jnp.int32)
        neg_inf_key = jnp.int32(KEY_NEG_INF)
        (
            vec_at,
            monotone_key,
            key_window,
            zero_hist,
            hist_add,
            scan_glob,
            find_bucket,
        ) = _make_radix_ops(glob_vmem, lane_iota, neg_inf_key)

        def write_empty_row(row):
            def fill(i):
                rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

            plsc.parallel_loop(0, k // LANES, unroll=4)(fill)
            pltpu.sync_copy(
                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
            )

        def run_step(scores_vmem, wave, row_len):
            row = wave * 32 + flat
            active = row < b
            i_hi = jnp.where(active & (row_len > 0), jnp.minimum(slice_len, row_len), 0)
            walk_chunks = _cdiv_dyn(i_hi)
            zero_hist(l0_buckets // LANES)

            @pl.when(i_hi > 0)
            def _():
                tail = (walk_chunks - 1) * LANES
                scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                    tail + lane_iota < i_hi,
                    scores_vmem[pl.ds(tail, LANES)],
                    jnp.int32(BITS_NEG_INF),
                )

                def hist0(i):
                    key = monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
                    scores_vmem[pl.ds(i * LANES, LANES)] = key
                    bucket = jnp.right_shift(key, l0_shift) + jnp.int32(l0_buckets // 2)
                    hist_add(bucket, key > neg_inf_key)

                plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0)

            total = scan_glob(l0_buckets // LANES)
            keep_all = total <= k
            b_star = find_bucket(l0_buckets // LANES, total - k)
            cum_at = vec_at(glob_vmem, b_star)
            cum_lo = jnp.where(
                b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0
            )
            c_hi = total - cum_at
            cnt = cum_at - cum_lo
            quota = k - c_hi
            prefix0 = b_star - jnp.int32(l0_buckets // 2)
            done0 = keep_all | (quota == cnt)
            use_1comp = active & (~done0) & (c_hi + cnt <= cand_cap)

            @pl.when(use_1comp)
            def _fast_1comp():
                lo0, _ = key_window(prefix0, jnp.int32(l0_shift))

                def pass2_1comp(i, coff):
                    key = scores_vmem[pl.ds(i * LANES, LANES)]
                    sel = key > lo0
                    plsc.store_compressed(
                        cand_idx_vmem.at[pl.ds(coff, LANES)],
                        i * LANES + lane_iota,
                        mask=sel,
                    )
                    return coff + plsc.all_reduce_population_count(sel)[0]

                coff0 = plsc.parallel_loop(
                    0, walk_chunks, unroll=SCAN_UNROLL, carry=jnp.int32(0)
                )(pass2_1comp)

                _refine_and_emit_candidates(
                    coff0=coff0,
                    prefix0=prefix0,
                    l0_shift=l0_shift,
                    quota=quota,
                    cnt=cnt,
                    ref_levels=ref_levels,
                    cand_idx_vmem=cand_idx_vmem,
                    rowbuf_vmem=rowbuf_vmem,
                    glob_vmem=glob_vmem,
                    lane_iota=lane_iota,
                    vec_at=vec_at,
                    key_window=key_window,
                    zero_hist=zero_hist,
                    hist_add=hist_add,
                    scan_glob=scan_glob,
                    find_bucket=find_bucket,
                    load_cand_key=lambda _i, idx_v, valid: plsc.load_gather(
                        scores_vmem, [idx_v], mask=valid
                    ),
                )

            @pl.when(active & (~use_1comp))
            def _fallback():
                prefix = prefix0
                shift = jnp.int32(l0_shift)
                quota_f = quota
                cnt_f = cnt
                done_f = done0
                for lvl_shift, lvl_mask, lvl_buckets in ref_levels:
                    hb = jnp.where(done_f, 0, lvl_buckets // LANES)
                    zero_hist(hb)
                    aw = jnp.where(done_f, 0, walk_chunks)
                    lo_e, hi_i = key_window(prefix, shift)

                    def ref_b(
                        i,
                        *,
                        lo_e=lo_e,
                        hi_i=hi_i,
                        lvl_shift=lvl_shift,
                        lvl_mask=lvl_mask,
                    ):
                        key = scores_vmem[pl.ds(i * LANES, LANES)]
                        match = (key > lo_e) & (key <= hi_i)
                        bucket = jnp.bitwise_and(
                            jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                        )
                        hist_add(bucket, match)

                    plsc.parallel_loop(0, aw, unroll=SCAN_UNROLL)(ref_b)
                    sub_tot = scan_glob(hb)
                    b2 = find_bucket(hb, sub_tot - quota_f)
                    ca2 = vec_at(glob_vmem, b2)
                    cl2 = jnp.where(
                        b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0
                    )
                    c_ab2 = sub_tot - ca2
                    cnt2 = ca2 - cl2
                    q2 = quota_f - c_ab2
                    wbits = shift - jnp.int32(lvl_shift)
                    npref = jnp.left_shift(prefix, wbits) | b2
                    ndone = done_f | (q2 == cnt2) | (q2 == 0)
                    prefix = jnp.where(done_f, prefix, npref)
                    shift = jnp.where(done_f, shift, jnp.int32(lvl_shift))
                    quota_f = jnp.where(done_f, quota_f, q2)
                    cnt_f = jnp.where(done_f, cnt_f, cnt2)
                    done_f = ndone

                def fill_b(i):
                    rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                        (LANES,), -1, jnp.int32
                    )

                plsc.parallel_loop(0, jnp.where(total < k, k // LANES, 0), unroll=4)(
                    fill_b
                )

                lo_e, hi_i = key_window(prefix, shift)
                my_take = jnp.where(keep_all, 0, quota_f)
                take_all_t = keep_all | (my_take >= cnt_f)
                gate = jnp.where(take_all_t, lo_e, hi_i)
                whole_b = take_all_t | (my_take <= 0)

                def em_w(i, woff):
                    key = scores_vmem[pl.ds(i * LANES, LANES)]
                    sel = key > gate
                    plsc.store_compressed(
                        rowbuf_vmem.at[pl.ds(woff, LANES)],
                        i * LANES + lane_iota,
                        mask=sel,
                    )
                    return woff + plsc.all_reduce_population_count(sel)[0]

                plsc.parallel_loop(
                    0,
                    jnp.where(whole_b, walk_chunks, 0),
                    unroll=SCAN_UNROLL,
                    carry=jnp.int32(0),
                )(em_w)

                def em_r(i, carry):
                    woff, rank = carry
                    key = scores_vmem[pl.ds(i * LANES, LANES)]
                    in_set = key > lo_e
                    strict = key > hi_i
                    tie_rank = (
                        plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32))
                        + rank
                    )
                    sel = strict | (in_set & (tie_rank <= my_take))
                    plsc.store_compressed(
                        rowbuf_vmem.at[pl.ds(woff, LANES)],
                        i * LANES + lane_iota,
                        mask=sel,
                    )
                    return woff + plsc.all_reduce_population_count(sel)[0], tie_rank[
                        LANES - 1
                    ]

                plsc.parallel_loop(
                    0,
                    jnp.where(whole_b, 0, walk_chunks),
                    unroll=4,
                    carry=(jnp.int32(0), jnp.int32(0)),
                )(em_r)

            @pl.when(active)
            def _write():
                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, k)],
                    out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
                )

        pltpu.sync_copy(lengths_hbm, meta_vmem)

        def w_info(wave):
            row = wave * 32 + flat
            rl = jnp.clip(vec_at(meta_vmem, row), 0, n)
            return rl, rl > 0, pl.multiple_of(_cdiv_dyn(rl) * LANES, LANES)

        def s_copy(wave, off, ww):
            row = wave * 32 + flat
            return pltpu.make_async_copy(
                scores_hbm.at[pl.ds(pl.multiple_of(row * slice_len, LANES), ww)],
                slices_vmem.at[pl.ds(pl.multiple_of(off, LANES), ww)],
                slice_sems.at[(off > 0).astype(jnp.int32)],
            )

        def start_f(wave, off):
            _, _, ww = w_info(wave)

            @pl.when(ww > 0)
            def _():
                s_copy(wave, off, ww).start()

        def wave_b(wave, off):
            off = pl.multiple_of(off, LANES)
            rl, busy, ww = w_info(wave)
            has_next = wave + 1 < num_waves
            nw = w_info(jnp.minimum(wave + 1, num_waves - 1))[2]
            early = has_next & (ww <= half) & (nw <= half)
            next_off = jnp.where(early, half - off, 0)
            pl.when(early)(functools.partial(start_f, wave + 1, next_off))

            @pl.when(ww > 0)
            def _():
                s_copy(wave, off, ww).wait()

            pl.when(busy)(
                functools.partial(
                    run_step, slices_vmem.at[pl.ds(off, slice_len)], wave, rl
                )
            )
            if write_empty:
                row = wave * 32 + flat
                pl.when(~busy & (row < b))(functools.partial(write_empty_row, row))
            pl.when(has_next & ~early)(functools.partial(start_f, wave + 1, next_off))
            return next_off

        start_f(jnp.int32(0), jnp.int32(0))
        lax.fori_loop(0, num_waves, wave_b, jnp.int32(0))

    out = pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=(
            pltpu.VMEM((hist_words + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((cand_cap + LANES,), jnp.int32),
            pltpu.VMEM((padded_b + LANES,), jnp.int32),
            pltpu.VMEM((buf_words,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p1_c11",
    )(words.reshape(-1), lengths_in)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    return out.reshape(padded_b, k)[:b]


def _run_streaming_1comp(
    words: jax.Array,
    lengths_in: jax.Array,
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    padded_b: int,
    write_empty: bool,
    scheduling_group_id: int | None,
    mesh: plsc.VectorSubcoreMesh,
) -> jax.Array:
    plan = _streaming_vmem_plan(b, k)
    assert plan is not None
    hist_words, cand_cap = plan
    tile_words = _STREAMING_TILE_WORDS
    l0_bits = _STREAMING_L0_BITS
    l0_shift = 32 - l0_bits
    l0_buckets = 1 << l0_bits
    ref_levels = _refine_levels_for(l0_bits)

    def body(
        scores_hbm,
        lengths_hbm,
        out_hbm,
        glob_vmem,
        rowbuf_vmem,
        cand_key_vmem,
        cand_idx_vmem,
        meta_vmem,
        tiles_vmem,
        tile_sems,
    ):
        flat = lax.axis_index("core") * 16 + lax.axis_index("subcore")
        lane_iota = jnp.arange(LANES, dtype=jnp.int32)
        neg_inf_key = jnp.int32(KEY_NEG_INF)
        (
            vec_at,
            monotone_key,
            key_window,
            zero_hist,
            hist_add,
            scan_glob,
            find_bucket,
        ) = _make_radix_ops(glob_vmem, lane_iota, neg_inf_key)

        def tile_copy(row, t_idx, slot, row_len):
            t_start = pl.multiple_of(t_idx * tile_words, LANES)
            t_len = jnp.minimum(tile_words, row_len - t_start)
            t_aligned = pl.multiple_of(_cdiv_dyn(t_len) * LANES, LANES)
            return (
                pltpu.make_async_copy(
                    scores_hbm.at[
                        pl.ds(
                            pl.multiple_of(row * slice_len + t_start, LANES), t_aligned
                        )
                    ],
                    tiles_vmem.at[
                        pl.ds(pl.multiple_of(slot * tile_words, LANES), t_aligned)
                    ],
                    tile_sems.at[slot],
                ),
                t_start,
                t_len,
            )

        def stream_pass(row, row_len, n_tiles, tile_fn, init_carry):
            cp0, _, _ = tile_copy(row, jnp.int32(0), jnp.int32(0), row_len)
            cp0.start()

            def t_body(t_idx, carry):
                slot = jnp.bitwise_and(t_idx, jnp.int32(1))

                @pl.when(t_idx + 1 < n_tiles)
                def _pref():
                    cp_n, _, _ = tile_copy(row, t_idx + 1, jnp.int32(1) - slot, row_len)
                    cp_n.start()

                cp_c, t_start, t_len = tile_copy(row, t_idx, slot, row_len)
                cp_c.wait()
                buf = tiles_vmem.at[
                    pl.ds(pl.multiple_of(slot * tile_words, LANES), tile_words)
                ]
                t_chunks = _cdiv_dyn(t_len)
                tail = (t_chunks - 1) * LANES
                buf[pl.ds(tail, LANES)] = jnp.where(
                    tail + lane_iota < t_len,
                    buf[pl.ds(tail, LANES)],
                    jnp.int32(BITS_NEG_INF),
                )
                return tile_fn(buf, t_start, t_chunks, carry)

            return lax.fori_loop(0, n_tiles, t_body, init_carry)

        pltpu.sync_copy(lengths_hbm, meta_vmem)

        def wave_b(wave, _):
            row = wave * 32 + flat
            active = row < b
            row_len = jnp.where(active, jnp.clip(vec_at(meta_vmem, row), 0, n), 0)
            busy = row_len > 0

            @pl.when(busy)
            def _do_row():
                n_tiles = (row_len + tile_words - 1) // tile_words
                zero_hist(l0_buckets // LANES)

                def p1_tile(buf, _ts, t_chunks, _c):
                    def h0(i):
                        key = monotone_key(buf[pl.ds(i * LANES, LANES)])
                        bucket = jnp.right_shift(key, l0_shift) + jnp.int32(
                            l0_buckets // 2
                        )
                        hist_add(bucket, key > neg_inf_key)

                    plsc.parallel_loop(0, t_chunks, unroll=SCAN_UNROLL)(h0)
                    return jnp.int32(0)

                stream_pass(row, row_len, n_tiles, p1_tile, jnp.int32(0))
                total = scan_glob(l0_buckets // LANES)
                keep_all = total <= k
                b_star = find_bucket(l0_buckets // LANES, total - k)
                cum_at = vec_at(glob_vmem, b_star)
                cum_lo = jnp.where(
                    b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0
                )
                c_hi = total - cum_at
                cnt = cum_at - cum_lo
                quota = k - c_hi
                prefix0 = b_star - jnp.int32(l0_buckets // 2)
                done0 = keep_all | (quota == cnt)
                use_1comp = (~done0) & (c_hi + cnt <= cand_cap)

                @pl.when(use_1comp)
                def _fast_1comp():
                    lo0, _ = key_window(prefix0, jnp.int32(l0_shift))

                    def p2_tile(buf, t_start, t_chunks, carry):
                        def inner(i, coff):
                            key = monotone_key(buf[pl.ds(i * LANES, LANES)])
                            sel = key > lo0
                            plsc.store_compressed(
                                cand_key_vmem.at[pl.ds(coff, LANES)], key, mask=sel
                            )
                            plsc.store_compressed(
                                cand_idx_vmem.at[pl.ds(coff, LANES)],
                                t_start + i * LANES + lane_iota,
                                mask=sel,
                            )
                            return coff + plsc.all_reduce_population_count(sel)[0]

                        return plsc.parallel_loop(
                            0, t_chunks, unroll=SCAN_UNROLL, carry=carry
                        )(inner)

                    coff0 = stream_pass(row, row_len, n_tiles, p2_tile, jnp.int32(0))
                    _refine_and_emit_candidates(
                        coff0=coff0,
                        prefix0=prefix0,
                        l0_shift=l0_shift,
                        quota=quota,
                        cnt=cnt,
                        ref_levels=ref_levels,
                        cand_idx_vmem=cand_idx_vmem,
                        rowbuf_vmem=rowbuf_vmem,
                        glob_vmem=glob_vmem,
                        lane_iota=lane_iota,
                        vec_at=vec_at,
                        key_window=key_window,
                        zero_hist=zero_hist,
                        hist_add=hist_add,
                        scan_glob=scan_glob,
                        find_bucket=find_bucket,
                        load_cand_key=lambda i, _idx_v, _valid: cand_key_vmem[
                            pl.ds(i * LANES, LANES)
                        ],
                    )

                @pl.when(~use_1comp)
                def _fallback():
                    prefix = prefix0
                    shift = jnp.int32(l0_shift)
                    quota_f = quota
                    cnt_f = cnt
                    done_f = done0
                    for lvl_shift, lvl_mask, lvl_buckets in ref_levels:
                        hb = jnp.where(done_f, 0, lvl_buckets // LANES)
                        zero_hist(hb)
                        lo_e, hi_i = key_window(prefix, shift)

                        def ref_tile(
                            buf,
                            _ts,
                            t_chunks,
                            _c,
                            *,
                            lo_e=lo_e,
                            hi_i=hi_i,
                            lvl_shift=lvl_shift,
                            lvl_mask=lvl_mask,
                        ):
                            def rb(
                                i,
                                *,
                                lo_e=lo_e,
                                hi_i=hi_i,
                                lvl_shift=lvl_shift,
                                lvl_mask=lvl_mask,
                            ):
                                key = monotone_key(buf[pl.ds(i * LANES, LANES)])
                                match = (key > lo_e) & (key <= hi_i)
                                bucket = jnp.bitwise_and(
                                    jnp.right_shift(key, lvl_shift),
                                    jnp.int32(lvl_mask),
                                )
                                hist_add(bucket, match)

                            plsc.parallel_loop(0, t_chunks, unroll=SCAN_UNROLL)(rb)
                            return jnp.int32(0)

                        @pl.when(~done_f)
                        def _sr():
                            stream_pass(row, row_len, n_tiles, ref_tile, jnp.int32(0))

                        sub_tot = scan_glob(hb)
                        b2 = find_bucket(hb, sub_tot - quota_f)
                        ca2 = vec_at(glob_vmem, b2)
                        cl2 = jnp.where(
                            b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0
                        )
                        c_ab2 = sub_tot - ca2
                        cnt2 = ca2 - cl2
                        q2 = quota_f - c_ab2
                        wbits = shift - jnp.int32(lvl_shift)
                        npref = jnp.left_shift(prefix, wbits) | b2
                        ndone = done_f | (q2 == cnt2) | (q2 == 0)
                        prefix = jnp.where(done_f, prefix, npref)
                        shift = jnp.where(done_f, shift, jnp.int32(lvl_shift))
                        quota_f = jnp.where(done_f, quota_f, q2)
                        cnt_f = jnp.where(done_f, cnt_f, cnt2)
                        done_f = ndone

                    def fill_b(i):
                        rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                            (LANES,), -1, jnp.int32
                        )

                    plsc.parallel_loop(
                        0, jnp.where(total < k, k // LANES, 0), unroll=4
                    )(fill_b)

                    lo_e, hi_i = key_window(prefix, shift)
                    my_take = jnp.where(keep_all, 0, quota_f)
                    take_all_t = keep_all | (my_take >= cnt_f)
                    gate = jnp.where(take_all_t, lo_e, hi_i)
                    whole_b = take_all_t | (my_take <= 0)

                    def em_tile(buf, t_start, t_chunks, carry):
                        def inner(i, cc):
                            woff, rank = cc
                            key = monotone_key(buf[pl.ds(i * LANES, LANES)])
                            col = t_start + i * LANES + lane_iota
                            in_set = key > lo_e
                            strict = key > hi_i
                            tie_rank = (
                                plsc.cumsum(
                                    in_set.astype(jnp.int32) - strict.astype(jnp.int32)
                                )
                                + rank
                            )
                            sel = jnp.where(
                                whole_b,
                                key > gate,
                                strict | (in_set & (tie_rank <= my_take)),
                            )
                            plsc.store_compressed(
                                rowbuf_vmem.at[pl.ds(woff, LANES)], col, mask=sel
                            )
                            return woff + plsc.all_reduce_population_count(sel)[
                                0
                            ], tie_rank[LANES - 1]

                        return plsc.parallel_loop(0, t_chunks, unroll=4, carry=carry)(
                            inner
                        )

                    stream_pass(
                        row, row_len, n_tiles, em_tile, (jnp.int32(0), jnp.int32(0))
                    )

                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, k)],
                    out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
                )

            if write_empty:

                @pl.when((~busy) & active)
                def _empty():
                    def fill(i):
                        rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                            (LANES,), -1, jnp.int32
                        )

                    plsc.parallel_loop(0, k // LANES, unroll=4)(fill)
                    pltpu.sync_copy(
                        rowbuf_vmem.at[pl.ds(0, k)],
                        out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
                    )

            return jnp.int32(0)

        lax.fori_loop(0, num_waves, wave_b, jnp.int32(0))

    out = pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=(
            pltpu.VMEM((hist_words + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((cand_cap + LANES,), jnp.int32),
            pltpu.VMEM((cand_cap + LANES,), jnp.int32),
            pltpu.VMEM((padded_b + LANES,), jnp.int32),
            pltpu.VMEM((2 * tile_words,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p1_s12",
    )(words.reshape(-1), lengths_in)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    return out.reshape(padded_b, k)[:b]


def sparsecore_topk_compact_p1(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    mode: CompactMode,
    write_empty: bool = True,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Single-subcore two-pass compacting SparseCore top-k (indices only)."""
    b, n = scores.shape
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    words = (
        scores
        if scores.dtype == jnp.int32
        else lax.bitcast_convert_type(scores, jnp.int32)
    )
    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)
    padded_b = num_waves * 32
    pad_b = padded_b - b
    pad_n = slice_len - n
    if pad_b > 0 or pad_n > 0:
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))
    lengths_in = jnp.pad(jnp.pad(row_lengths.astype(jnp.int32), (0, pad_b)), (0, LANES))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    runner = _run_resident_1comp if mode == "resident" else _run_streaming_1comp
    return runner(
        words,
        lengths_in,
        b=b,
        n=n,
        k=k,
        slice_len=slice_len,
        num_waves=num_waves,
        padded_b=padded_b,
        write_empty=write_empty,
        scheduling_group_id=scheduling_group_id,
        mesh=mesh,
    )
