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
"""TPU7x SparseCore single-stage exact unsorted top-k selection over fp32 score rows."""

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

TPU7X_GENERATION = 7
SC_LANES = 16
NUM_BUCKETS = 256
MAX_SLICE_WORDS = 32 * 1024
MAX_TOPK = 4096
SCAN_UNROLL = 8
NUM_SC_CORES = 2
NUM_SUBCORES = 16
TOTAL_SUBCORES = NUM_SC_CORES * NUM_SUBCORES
MAX_N = 2 * NUM_SUBCORES * MAX_SLICE_WORDS
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
BITS_NEG_INF = int(np.array(-np.inf, np.float32).view(np.int32))
KEY_NEG_INF = int(np.int32(BITS_NEG_INF) ^ np.int32(MONO_MASK))


def _check_tpu7x(func_name: str) -> None:
    # Avoid calling pltpu.get_tpu_info() on bare CPU: raises ValueError when no
    # TPU backend or AbstractMesh is active.
    if (
        jax.default_backend() != "tpu"
        and jax.sharding.get_abstract_mesh().abstract_device is None
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x, got backend={jax.default_backend()!r}"
        )
    tpu_info = pltpu.get_tpu_info()
    sc = tpu_info.sparse_core
    if (
        tpu_info.generation != TPU7X_GENERATION
        or sc is None
        or sc.num_cores != NUM_SC_CORES
        or sc.num_subcores != NUM_SUBCORES
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x SparseCore "
            f"({NUM_SC_CORES}x{NUM_SUBCORES} subcores), got "
            f"generation={tpu_info.generation}, sparse_core={sc}"
        )


def _cdiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _align_to(value: int, alignment: int) -> int:
    return _cdiv(value, alignment) * alignment


def _cdiv_dyn(value: jax.Array) -> jax.Array:
    return jnp.right_shift(value + (SC_LANES - 1), 4)


def _topk_body(
    scores_hbm: Any,
    lengths_hbm: Any,
    out_hbm: Any,
    glob_vmem: Any,
    rowbuf_vmem: Any,
    peer_hist_vmem: Any,
    peer_idx_vmem: Any,
    shared_hist_vmem: Any,
    shared_counts_vmem: Any,
    shared_idx_vmem: Any,
    *,
    b: int,
    n: int,
    k: int,
    p: int = 1,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
) -> None:
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * NUM_SUBCORES + sub

    groups_per_core = NUM_SUBCORES // p
    groups_per_chip = NUM_SC_CORES * groups_per_core
    if p == NUM_SUBCORES:
        group_id = jnp.int32(0)
        s = sub
        group_base = jnp.int32(0)
    else:
        group_id = jnp.right_shift(sub, int(np.log2(p)))
        s = jnp.bitwise_and(sub, jnp.int32(p - 1))
        group_base = group_id * p
    col_base = s * slice_len

    lane_iota = jnp.arange(SC_LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def vec_at(ref: Any, idx: Any) -> jax.Array:
        return ref[pl.ds(idx, SC_LANES)][0]

    def monotone_key(bits: jax.Array) -> jax.Array:
        mask = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, mask)

    def key_window(prefix: jax.Array, shift: jax.Array) -> tuple[jax.Array, jax.Array]:
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound: Any) -> None:
        def body(i: jax.Array) -> None:
            glob_vmem[pl.ds(i * SC_LANES, SC_LANES)] = jnp.zeros((SC_LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket: jax.Array, mask: jax.Array) -> None:
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def sync_and_reduce_hist(slot: int, active_hist: jax.Array) -> None:
        if p == 1:
            return

        @pl.when(active_hist)
        def _() -> None:
            dst_off = pl.multiple_of(
                (slot * NUM_SUBCORES + sub) * NUM_BUCKETS, SC_LANES
            )
            pltpu.sync_copy(
                glob_vmem.at[pl.ds(0, NUM_BUCKETS)],
                shared_hist_vmem.at[pl.ds(dst_off, NUM_BUCKETS)],
            )

        plsc.subcore_barrier()

        @pl.when(active_hist)
        def _() -> None:
            src_off = pl.multiple_of(
                (slot * NUM_SUBCORES + group_base) * NUM_BUCKETS, SC_LANES
            )
            pltpu.sync_copy(
                shared_hist_vmem.at[pl.ds(src_off, p * NUM_BUCKETS)],
                peer_hist_vmem.at[pl.ds(0, p * NUM_BUCKETS)],
            )

            def init_hist(i: jax.Array) -> None:
                glob_vmem[pl.ds(i * SC_LANES, SC_LANES)] = peer_hist_vmem[
                    pl.ds(i * SC_LANES, SC_LANES)
                ]

            plsc.parallel_loop(0, NUM_BUCKETS // SC_LANES, unroll=8)(init_hist)

            def add_peer(t: jax.Array) -> None:
                def add_chunk(i: jax.Array) -> None:
                    glob_vmem[pl.ds(i * SC_LANES, SC_LANES)] = (
                        glob_vmem[pl.ds(i * SC_LANES, SC_LANES)]
                        + peer_hist_vmem[
                            pl.ds(t * NUM_BUCKETS + i * SC_LANES, SC_LANES)
                        ]
                    )

                plsc.parallel_loop(0, NUM_BUCKETS // SC_LANES, unroll=8)(add_chunk)

            plsc.parallel_loop(1, p, unroll=1)(add_peer)

    def scan_glob(bound: Any) -> jax.Array:
        def body(i: jax.Array, carry: jax.Array) -> jax.Array:
            c = plsc.cumsum(glob_vmem[pl.ds(i * SC_LANES, SC_LANES)]) + carry
            glob_vmem[pl.ds(i * SC_LANES, SC_LANES)] = c
            return c[SC_LANES - 1]

        return cast(
            Callable[..., jax.Array],
            plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0)),
        )(body)

    def find_bucket(bound: Any, thresh: jax.Array) -> jax.Array:
        def body(i: jax.Array, cnt: jax.Array) -> jax.Array:
            below = glob_vmem[pl.ds(i * SC_LANES, SC_LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return cast(
            Callable[..., jax.Array],
            plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0)),
        )(body)

    def _run_step(
        scores_vmem: Any,
        len_slice_vmem: Any,
        out_vmem: Any = None,
    ) -> None:
        w = pl.program_id(0)
        row = (
            w * TOTAL_SUBCORES + flat
            if p == 1
            else w * groups_per_chip + core * groups_per_core + group_id
        )
        active = row < b
        row_eff_len = jnp.minimum(vec_at(len_slice_vmem, 0), n)
        row_eff_len = jnp.where(active, row_eff_len, 0)

        if p == 1:
            i_hi = jnp.where(row_eff_len > 0, jnp.minimum(slice_len, row_eff_len), 0)
        else:
            i_hi = jnp.clip(row_eff_len - col_base, 0, slice_len)
        walk_chunks = _cdiv_dyn(i_hi)

        zero_hist(NUM_BUCKETS // SC_LANES)

        @pl.when(i_hi > 0)
        def _() -> None:
            tail = (walk_chunks - 1) * SC_LANES
            scores_vmem[pl.ds(tail, SC_LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, SC_LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            def hist0_body(i: jax.Array) -> None:
                key = monotone_key(scores_vmem[pl.ds(i * SC_LANES, SC_LANES)])
                scores_vmem[pl.ds(i * SC_LANES, SC_LANES)] = key
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
                hist_add(bucket, key > neg_inf_key)

            plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

        sync_and_reduce_hist(0, active)

        total = scan_glob(NUM_BUCKETS // SC_LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // SC_LANES, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_idx, (lvl_shift, lvl_mask) in enumerate(REFINEMENT_LEVELS, start=1):
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // SC_LANES)
            zero_hist(hist_bound)
            active_walk = jnp.where(done, 0, walk_chunks)
            lo_excl, hi_incl = key_window(prefix, shift)

            def ref_body(
                i: jax.Array,
                lvl_shift: int = lvl_shift,
                lvl_mask: int = lvl_mask,
                lo_excl: jax.Array = lo_excl,
                hi_incl: jax.Array = hi_incl,
            ) -> None:
                key = scores_vmem[pl.ds(i * SC_LANES, SC_LANES)]
                match = (key > lo_excl) & (key <= hi_incl)
                bucket = jnp.bitwise_and(
                    jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                )
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

            sync_and_reduce_hist(lvl_idx % 2, ~done)

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
            done = new_done

        def fill_body(i: jax.Array) -> None:
            rowbuf_vmem[pl.ds(i * SC_LANES, SC_LANES)] = jnp.full(
                (SC_LANES,), -1, jnp.int32
            )

        if p == 1:
            fill_bound = jnp.where(total < k, k // SC_LANES, 0)
        else:
            fill_bound = jnp.where((s == 0) & (total < k), k // SC_LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        lo_excl, hi_incl = key_window(prefix, shift)
        if p == 1:
            my_take = jnp.where(keep_all, 0, quota)
            take_all_ties = keep_all | (my_take >= cnt)
            gate = jnp.where(take_all_ties, lo_excl, hi_incl)
            whole_bucket = take_all_ties | (my_take <= 0)
        else:
            last_b = jnp.bitwise_and(prefix, jnp.int32(0xFF))

            def tie_body(
                t: jax.Array, carry: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array]:
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

        emit_chunks = jnp.where(active, walk_chunks, 0)

        def emit(i: jax.Array, woff: jax.Array, sel: jax.Array) -> jax.Array:
            col_offset = (
                i * SC_LANES + lane_iota
                if p == 1
                else col_base + i * SC_LANES + lane_iota
            )
            plsc.store_compressed(
                rowbuf_vmem.at[pl.ds(woff, SC_LANES)],
                col_offset,
                mask=sel,
            )
            return woff + plsc.all_reduce_population_count(sel)[0]

        def emit_whole(i: jax.Array, woff: jax.Array) -> jax.Array:
            key = scores_vmem[pl.ds(i * SC_LANES, SC_LANES)]
            return emit(i, woff, key > gate)

        m_whole = cast(
            Callable[..., jax.Array],
            plsc.parallel_loop(
                0,
                jnp.where(whole_bucket, emit_chunks, 0),
                unroll=SCAN_UNROLL,
                carry=jnp.int32(0),
            ),
        )(emit_whole)

        def emit_ranked(
            i: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            woff, rank = carry
            key = scores_vmem[pl.ds(i * SC_LANES, SC_LANES)]
            in_set = key > lo_excl
            strict = key > hi_incl
            tie_rank = (
                plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
            )
            sel = strict | (in_set & (tie_rank <= my_take))
            return emit(i, woff, sel), tie_rank[SC_LANES - 1]

        m_ranked, _ = plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, 0, emit_chunks),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_ranked)

        if p == 1:

            def copy_out(i: jax.Array) -> None:
                out_vmem[pl.ds(i * SC_LANES, SC_LANES)] = rowbuf_vmem[
                    pl.ds(i * SC_LANES, SC_LANES)
                ]

            plsc.parallel_loop(0, k // SC_LANES, unroll=4)(copy_out)
        else:
            m_s = jnp.where(whole_bucket, m_whole, m_ranked)
            w_slot = jnp.bitwise_and(w, jnp.int32(1))
            glob_vmem[pl.ds(0, SC_LANES)] = jnp.broadcast_to(m_s, (SC_LANES,))

            @pl.when(active)
            def _() -> None:
                cnt_off = pl.multiple_of(
                    (w_slot * NUM_SUBCORES + sub) * SC_LANES, SC_LANES
                )
                pltpu.sync_copy(
                    glob_vmem.at[pl.ds(0, SC_LANES)],
                    shared_counts_vmem.at[pl.ds(cnt_off, SC_LANES)],
                )

            chunks_s = _cdiv_dyn(m_s)
            aligned_m_s = pl.multiple_of(chunks_s * SC_LANES, SC_LANES)

            @pl.when(active & (s > 0) & (chunks_s > 0))
            def _() -> None:
                base_off = pl.multiple_of((w_slot * NUM_SUBCORES + sub) * k, SC_LANES)
                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, aligned_m_s)],
                    shared_idx_vmem.at[pl.ds(base_off, aligned_m_s)],
                )

            plsc.subcore_barrier()

            should_write = active & (jnp.bool_(write_empty) | (row_eff_len > 0))

            @pl.when((s == 0) & should_write)
            def _() -> None:
                grp_cnt_off = pl.multiple_of(
                    (w_slot * NUM_SUBCORES + group_base) * SC_LANES, SC_LANES
                )
                pltpu.sync_copy(
                    shared_counts_vmem.at[pl.ds(grp_cnt_off, p * SC_LANES)],
                    peer_hist_vmem.at[pl.ds(0, p * SC_LANES)],
                )

                all_true = jnp.ones((SC_LANES,), dtype=jnp.bool_)

                def gather_peer(t: jax.Array, woff: jax.Array) -> jax.Array:
                    m_t = vec_at(peer_hist_vmem, t * SC_LANES)
                    chunks_t = _cdiv_dyn(m_t)
                    aligned_m_t = pl.multiple_of(chunks_t * SC_LANES, SC_LANES)

                    @pl.when(chunks_t > 0)
                    def _() -> None:
                        peer_off = pl.multiple_of(
                            (w_slot * NUM_SUBCORES + group_base + t) * k, SC_LANES
                        )
                        pltpu.sync_copy(
                            shared_idx_vmem.at[pl.ds(peer_off, aligned_m_t)],
                            peer_idx_vmem.at[pl.ds(0, aligned_m_t)],
                        )

                        full_chunks = jnp.right_shift(m_t, 4)

                        def append_full(c: jax.Array) -> None:
                            plsc.store_compressed(
                                rowbuf_vmem.at[pl.ds(woff + c * SC_LANES, SC_LANES)],
                                peer_idx_vmem[pl.ds(c * SC_LANES, SC_LANES)],
                                mask=all_true,
                            )

                        plsc.parallel_loop(0, full_chunks, unroll=4)(append_full)

                        tail_rem = jnp.bitwise_and(m_t, jnp.int32(SC_LANES - 1))
                        tail_sel = lane_iota < tail_rem
                        tail_off = woff + full_chunks * SC_LANES
                        plsc.store_compressed(
                            rowbuf_vmem.at[pl.ds(tail_off, SC_LANES)],
                            peer_idx_vmem[pl.ds(full_chunks * SC_LANES, SC_LANES)],
                            mask=tail_sel,
                        )

                    return woff + m_t

                plsc.parallel_loop(1, p, unroll=1, carry=m_s)(gather_peer)

                out_off = pl.multiple_of(row * k, SC_LANES)
                pltpu.sync_copy(
                    rowbuf_vmem.at[pl.ds(0, k)],
                    out_hbm.at[pl.ds(out_off, k)],
                )

    # Avoid double-buffering slices > MAX_SLICE_WORDS: exceeds 512 KiB subcore VMEM.
    buf_count = 1 if (num_waves == 1 or slice_len > MAX_SLICE_WORDS) else 2
    in_specs = (
        pl.BlockSpec(
            (slice_len,),
            lambda w: (w * TOTAL_SUBCORES + flat,),
            pipeline_mode=pl.Buffered(buf_count),
        ),
        pl.BlockSpec(
            (SC_LANES,),
            lambda w: (w * TOTAL_SUBCORES + flat,),
            pipeline_mode=pl.Buffered(buf_count),
        ),
    )
    if p == 1:

        def pipeline_step_p1_idx(
            scores_vmem: Any, len_slice_vmem: Any, out_vmem: Any
        ) -> None:
            _run_step(scores_vmem, len_slice_vmem, out_vmem)

        out_specs = (
            pl.BlockSpec(
                (k,),
                lambda w: (w * TOTAL_SUBCORES + flat,),
                pipeline_mode=pl.Buffered(buf_count),
            ),
        )
        pltpu.emit_pipeline(
            pipeline_step_p1_idx,
            grid=(num_waves,),
            in_specs=in_specs,
            out_specs=out_specs,
        )(scores_hbm, lengths_hbm, out_hbm)
    else:

        def pipeline_step_coop(scores_vmem: Any, len_slice_vmem: Any) -> None:
            _run_step(scores_vmem, len_slice_vmem)

        pltpu.emit_pipeline(
            pipeline_step_coop,
            grid=(num_waves,),
            in_specs=in_specs,
            out_specs=(),
        )(scores_hbm, lengths_hbm)


def _sc_topk_direct(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    p: int = 1,
    write_empty: bool = True,
) -> jax.Array:
    b, n = scores.shape
    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    groups_per_chip = TOTAL_SUBCORES // p
    slice_len = _align_to(n // p, SC_LANES)
    num_waves = _cdiv(b, groups_per_chip)
    padded_b = num_waves * groups_per_chip
    pad_b = padded_b - b
    pad_n = slice_len * p - n
    if pad_b > 0 or pad_n > 0:
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))

    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, pad_b))
    if p > 1:
        lengths = jnp.repeat(lengths, p)
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, SC_LANES - 1)))

    mesh = plsc.VectorSubcoreMesh(
        num_cores=NUM_SC_CORES,
        num_subcores=NUM_SUBCORES,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + SC_LANES,), jnp.int32),
        pltpu.VMEM((k + SC_LANES,), jnp.int32),
        pltpu.VMEM((p * NUM_BUCKETS + SC_LANES,), jnp.int32) if p > 1 else None,
        pltpu.VMEM((k + SC_LANES,), jnp.int32) if p > 1 else None,
        pltpu.VMEM_SHARED((2 * NUM_SUBCORES * NUM_BUCKETS,), jnp.int32)
        if p > 1
        else None,
        pltpu.VMEM_SHARED((2 * NUM_SUBCORES * SC_LANES,), jnp.int32) if p > 1 else None,
        pltpu.VMEM_SHARED((2 * NUM_SUBCORES * k,), jnp.int32) if p > 1 else None,
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
    )
    kernel_fn: Callable[..., Any] = pl.kernel(
        body_fn,
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}_p{p}",
    )
    slots = cast(
        jax.Array,
        kernel_fn(words.reshape(-1), lengths_2d.reshape(-1)),
    )
    return slots.reshape(padded_b, k)[:b]


def _pick_partition(b: int, n: int, k: int) -> int:
    # Avoid p > NUM_SUBCORES: subcore_barrier cannot synchronize across SparseCores.
    p_min = min(NUM_SUBCORES, _cdiv(n, MAX_SLICE_WORDS))
    p = 1
    while p < p_min:
        p *= 2

    max_p = min(NUM_SUBCORES, TOTAL_SUBCORES // b) if b < TOTAL_SUBCORES else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= SC_LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p


@functools.partial(
    jax.jit,
    static_argnames=("k", "write_empty_rows"),
)
def sparsecore_topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array | None = None,
    *,
    write_empty_rows: bool = True,
) -> jax.Array:
    """Select exact top-k ``int32[B, k]`` column indices from ``scores: (B, N)``.

    Accepts ``f32`` or fp32-bitcast ``i32`` scores and optional ``row_lengths: (B,)``
    (``0 < N <= 1_048_576``, ``0 < k <= 4096``, multiples of 16). Excludes ``-inf``
    and columns ``>= row_lengths``; returns unspecified order suffix-padded ``-1``.
    ``write_empty_rows=False`` skips empty rows for ``p > 1`` (undefined output);
    ``p == 1`` still writes ``-1``.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype not in (jnp.float32, jnp.int32):
        raise ValueError(f"scores must be f32 or i32, got {scores.dtype}")
    b, n = scores.shape
    if n <= 0 or n % SC_LANES != 0:
        raise ValueError(f"{n=} must be a positive multiple of {SC_LANES}")
    if n > MAX_N:
        raise ValueError(f"{n=} must be <= {MAX_N} to fit in SparseCore subcore VMEM")
    if k % SC_LANES != 0 or not 0 < k <= MAX_TOPK:
        raise ValueError(
            f"{k=} must be a positive multiple of {SC_LANES}, at most {MAX_TOPK}"
        )
    _check_tpu7x("sparsecore_topk")

    if row_lengths is None:
        row_lengths = jnp.full((b,), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    p = _pick_partition(b, n, k)
    return _sc_topk_direct(
        scores,
        k,
        row_lengths,
        p=p,
        write_empty=write_empty_rows,
    )
