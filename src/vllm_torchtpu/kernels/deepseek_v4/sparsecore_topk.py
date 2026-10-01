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

Hierarchical 2-stage multi-subcore selection:
1. For any batch size B (including B = 1, 2, 4, 8, 16, 32) and N in 2048..256K,
   rows are partitioned into P sub-rows (where B * P <= 32) so that all 32
   subcores on the chip are utilized in parallel with zero inter-subcore
   synchronization.
2. Stage 1 is length-bounded and rebalanced: sub-rows have a traced width
   w = align16(cdiv(max(row_lengths), P)) (clamped to [16, N/P]); sub-row j of
   row r covers columns [j*w, j*w + w). Each subcore walks its sub-rows with a
   manual double-buffered DMA of exactly align16(sub-row length) words, so
   columns at or beyond every row's length are never read, and empty rows
   (e.g. decode padding rows) issue no DMA.
3. In Stage 2, candidate scores are merged on SparseCore to produce the exact
   global top-k column indices. The single-stage path (P == 1) and stage 2 use
   the same length-bounded kernel. When the P * k_p candidates fit one slice,
   stage 1 emits global column indices (per-sub-row base j*w added in-kernel)
   and stage 2 compacts them as a payload loaded alongside the keys, so its
   index output is already the global column (no XLA take_along_axis remap).

Output rows are the top-k column indices in unspecified order, ``-1``
suffix-padded, matching an exact top-k over the given scores as a set (equal
boundary scores may be broken differently). ``-inf`` scores and positions at
or beyond ``row_lengths`` are never selected. NaNs are unsupported.
"""

# Pallas SparseCore APIs used, each learned from an upstream example
# (jax @ce57049 = ce570497d55d, version 0.11.2):
#   * pl.kernel over plsc.VectorSubcoreMesh with scratch_types of pltpu.VMEM
#     and pltpu.SemaphoreType.DMA((n,)); manual make_async_copy .start()/.wait()
#     double buffering in a lax.fori_loop, pl.when-gated prefetch/wait, drain
#     after the loop (its 2-D .at[buf_idx] ring is replaced by a 1-D ring here):
#       tests/pallas/tpu_sparsecore_pallas_test.py:3122-3197@ce57049 (test_manual)
#   * traced-size DMA destination o_ref.at[pl.ds(0, traced_n)]:
#       tests/pallas/tpu_sparsecore_pallas_test.py:920-945@ce57049
#       (gather_1d_with_dynamically_sized_ref); contiguous traced-size DMA on
#       TC: tests/pallas/tpu_pallas_test.py:3532-3558@ce57049.
#   * rebuilding the descriptor (same traced size) for .wait():
#       jax/_src/pallas/mosaic/sc_lowering.py:531-600@ce57049 (wait_dma2 sized
#       by the src/dst memref types) + test_manual pattern above.
#   * pltpu.sync_copy = make_async_copy start+wait on a scoped DMA semaphore:
#       jax/_src/pallas/mosaic/helpers.py:25-50@ce57049
#   * scalar from VMEM on a vector subcore = lane 0 of a vector load
#       (ref[pl.ds(i, lanes)][0]): tests/pallas/tpu_sparsecore_pallas_test.py:
#       1530-1542@ce57049 (test_scalar_load_emulated).
#   * zero-size DMAs are never started (pl.when(ext > 0)), so every started DMA
#     is waited.
#   * plsc.store_compressed of a VMEM-loaded vector (the stage-2 index payload,
#     compacted with the same mask at the same traced offset as the keys):
#       tests/pallas/tpu_sparsecore_pallas_test.py:1278-1291, 1495-1510@ce57049;
#       signature and constraints (dtype(ref) == dtype(x), vector-sized VMEM
#       view): jax/_src/pallas/mosaic/sc_primitives.py:119-185@ce57049.

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 16
NUM_BUCKETS = 256
# Largest per-subcore resident slice, in 32-bit words (128KB; double-buffered
# to 256KB out of the 512KB tile memory).
MAX_SLICE_WORDS = 32 * 1024
# Unroll factor for the loops that walk the whole resident slice.
SCAN_UNROLL = 8

L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
BITS_NEG_INF = int(np.array(-np.inf, np.float32).view(np.int32))
KEY_NEG_INF = int(np.int32(BITS_NEG_INF) ^ np.int32(0x7FFFFFFF))

NUM_WORKERS = 32  # 2 SparseCores x 16 vector subcores (one chiplet)


def _cdiv(a, b):
    return (a + b - 1) // b


def _align_to(x, a):
    return _cdiv(x, a) * a


def _cdiv_dyn(x):
    return jnp.right_shift(x + (LANES - 1), 4)


def _align16_dyn(x):
    return jnp.bitwise_and(x + (LANES - 1), jnp.int32(-LANES))


MODE_SLOT = "slot"  # emit the column index within the row (direct path)
MODE_BASE = "base"  # emit base[row] + column index (stage 1: global column)
MODE_PAYLOAD = "payload"  # emit payload[off + column] (stage 2: carried index)


def _topk_rows_kernel(
    *refs,
    rows: int,
    k: int,
    slice_len: int,
    num_waves: int,
    tab_stride: int,
    mode: str = MODE_SLOT,
):
    # Inputs:  scores_hbm i32[total_words] flat fp32 score bits,
    #          lens_hbm / offs_hbm i32[NUM_WORKERS * tab_stride] per (worker, wave),
    #          [MODE_BASE]    bases_hbm i32[NUM_WORKERS * tab_stride],
    #          [MODE_PAYLOAD] payload_hbm i32[total_words] (same layout as scores).
    # Outputs: out_hbm, keys_hbm i32[padded_rows * k].
    # Scratch: glob, rowbuf, keybuf, sbuf (1-D ring), lens, offs, obuf, kbuf,
    #          in_sem, out_sem, then [MODE_BASE] bases_vmem i32[tab_stride],
    #          [MODE_PAYLOAD] pbuf_vmem i32[slice_len] + pay_sem DMA[1].
    refs = list(refs)
    scores_hbm, lens_hbm, offs_hbm = refs[:3]
    del refs[:3]
    bases_hbm = refs.pop(0) if mode == MODE_BASE else None
    payload_hbm = refs.pop(0) if mode == MODE_PAYLOAD else None
    (
        out_hbm,
        keys_hbm,
        glob_vmem,
        rowbuf_vmem,
        keybuf_vmem,
        sbuf_vmem,
        lens_vmem,
        offs_vmem,
        obuf_vmem,
        kbuf_vmem,
        in_sem,
        out_sem,
    ) = refs[:12]
    del refs[:12]
    bases_vmem = refs.pop(0) if mode == MODE_BASE else None
    if mode == MODE_PAYLOAD:
        pbuf_vmem, pay_sem = refs
    else:
        assert not refs, len(refs)

    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub
    nbuf = 1 if num_waves == 1 else 2

    # Per-worker metadata, one contiguous row each (traced offset, static size).
    pltpu.sync_copy(lens_hbm.at[pl.ds(flat * tab_stride, tab_stride)], lens_vmem)
    pltpu.sync_copy(offs_hbm.at[pl.ds(flat * tab_stride, tab_stride)], offs_vmem)
    if mode == MODE_BASE:
        pltpu.sync_copy(bases_hbm.at[pl.ds(flat * tab_stride, tab_stride)], bases_vmem)

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def key_window(prefix, shift):
        """Scalar bounds of the key range ``key >> shift == prefix``.

        Returns ``(lo_excl, hi_incl)`` so that a bucket test is two plain compares
        against loop-invariant scalars, with no per-element shift:

          ``key >> shift >= prefix``  <=>  ``key > lo_excl``
          ``key >> shift >  prefix``  <=>  ``key > hi_incl``

        Both bounds are floored at the ``-inf`` sentinel key, so ``key > lo_excl``
        also rejects ``-inf`` and the poisoned tail lanes. That lets every scan
        below drop its separate validity test.
        """
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound):
        def body(i):
            glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket, mask):
        """Accumulates one chunk's bucket counts into the histogram."""
        # ``dup`` counts occurrences inclusive of the element itself, so at a
        # ``last`` position it already holds the chunk's total for that bucket.
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def scan_glob(bound):
        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):
        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    def select_row(
        scores_vmem,
        eff_len,
        active,
        out_vmem,
        keys_vmem,
        idx_base=None,
        before_emit=None,
    ):
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi)

        zero_hist(NUM_BUCKETS // LANES)

        @pl.when(i_hi > 0)
        def _():
            # Poison the ragged lanes of the final chunk with -inf once, so that no
            # later scan has to re-test ``j < i_hi``.
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            # Convert to monotone keys in place and histogram the top byte in the
            # same pass, so the slice is read once instead of twice.
            def hist0_body(i):
                key = monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
                scores_vmem[pl.ds(i * LANES, LANES)] = key
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
                hist_add(bucket, key > neg_inf_key)

            plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

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

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            zero_hist(hist_bound)
            active_walk = jnp.where(done, 0, walk_chunks)
            lo_excl, hi_incl = key_window(prefix, shift)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                lo_excl=lo_excl,
                hi_incl=hi_incl,
            ):
                key = scores_vmem[pl.ds(i * LANES, LANES)]
                match = (key > lo_excl) & (key <= hi_incl)
                bucket = jnp.bitwise_and(
                    jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                )
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

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

        my_take = jnp.where(keep_all, 0, quota)

        if before_emit is not None:
            # Stage 2 payload: the payload row must have landed before the emit
            # passes read it (its DMA overlapped the histogram passes above).
            before_emit()

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
            # Padding slots must read back as -inf, the same value the
            # caller used to substitute for a -1 index.
            keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                (LANES,), BITS_NEG_INF, jnp.int32
            )

        fill_bound = jnp.where(total < k, k // LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        # The winning set is ``key >> shift >= prefix`` minus however many of the
        # boundary-bucket ties do not fit in the quota. Ranking those ties costs a
        # per-chunk cumsum plus a second serial carry, but it is only needed when
        # the quota splits the bucket -- i.e. when the boundary score is an exact
        # fp32 duplicate. Whenever the bucket is taken whole (which includes
        # ``keep_all``) or dropped whole, selection collapses to one compare
        # against a loop-invariant scalar.
        lo_excl, hi_incl = key_window(prefix, shift)
        take_all_ties = keep_all | (my_take >= cnt)
        gate = jnp.where(take_all_ties, lo_excl, hi_incl)
        whole_bucket = take_all_ties | (my_take <= 0)
        emit_chunks = jnp.where(active, walk_chunks, 0)

        def emit(i, woff, key, sel):
            if mode == MODE_PAYLOAD:
                # Compact the carried global column index (loaded from the
                # payload row at the same position as the key) instead of the
                # slot index, so stage 2's output IS the global index.
                val = pbuf_vmem[pl.ds(i * LANES, LANES)]
            elif mode == MODE_BASE:
                # Stage 1: global column = sub-row base j*w + local column.
                val = i * LANES + lane_iota + idx_base
            else:
                val = i * LANES + lane_iota
            plsc.store_compressed(
                rowbuf_vmem.at[pl.ds(woff, LANES)],
                val,
                mask=sel,
            )
            # monotone_key is an involution, so this recovers the original
            # fp32 bits. Same woff and same mask, so the two compactions
            # stay aligned slot for slot.
            plsc.store_compressed(
                keybuf_vmem.at[pl.ds(woff, LANES)],
                monotone_key(key),
                mask=sel,
            )
            return woff + plsc.all_reduce_population_count(sel)[0]

        def emit_whole(i, woff):
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            return emit(i, woff, key, key > gate)

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, emit_chunks, 0),
            unroll=SCAN_UNROLL,
            carry=jnp.int32(0),
        )(emit_whole)  # pytype: disable=bad-argument-type

        def emit_ranked(i, carry):
            woff, rank = carry
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            in_set = key > lo_excl
            strict = key > hi_incl
            # ``strict`` implies ``in_set``, so the difference is the tie mask.
            tie_rank = (
                plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
            )
            sel = strict | (in_set & (tie_rank <= my_take))
            return emit(i, woff, key, sel), tie_rank[LANES - 1]

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, 0, emit_chunks),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_ranked)

        def copy_out(i):
            out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]
            keys_vmem[pl.ds(i * LANES, LANES)] = keybuf_vmem[pl.ds(i * LANES, LANES)]

        plsc.parallel_loop(0, k // LANES, unroll=4)(copy_out)

    def row_len(wv):
        return vec_at(lens_vmem, wv)

    def in_copy(wv, slot, ext):
        off = vec_at(offs_vmem, wv)
        # 1-D ring: a 2-D (nbuf, slice_len) VMEM buffer is tiled (2,128) on SC
        # and a 1-row traced slice of it fails "Slice sizes along tiled
        # dimensions must be aligned to tiles".
        off = pl.multiple_of(off, LANES)
        ext = pl.multiple_of(ext, LANES)
        base = pl.multiple_of(slot * slice_len, LANES)
        return pltpu.make_async_copy(
            scores_hbm.at[pl.ds(off, ext)],
            sbuf_vmem.at[pl.ds(base, ext)],
            in_sem.at[slot],
        )

    def start_in(wv, slot):
        ext = _align16_dyn(row_len(wv))

        @pl.when(ext > 0)
        def _():
            in_copy(wv, slot, ext).start()

    def wait_in(wv, slot):
        ext = _align16_dyn(row_len(wv))

        @pl.when(ext > 0)
        def _():
            in_copy(wv, slot, ext).wait()

    def out_copies(wv, slot):
        row = wv * (num_sc_cores * num_subcores) + flat
        return (
            pltpu.make_async_copy(
                obuf_vmem.at[pl.ds(slot * k, k)],
                out_hbm.at[pl.ds(row * k, k)],
                out_sem.at[slot],
            ),
            pltpu.make_async_copy(
                kbuf_vmem.at[pl.ds(slot * k, k)],
                keys_hbm.at[pl.ds(row * k, k)],
                out_sem.at[slot],
            ),
        )

    def pay_copy(wv, ext):
        # Payload row wv: same traced word offset and extent as its scores,
        # into a single static buffer (consumed by row wv's emit before row
        # wv + 1's copy is started, so one buffer suffices).
        off = pl.multiple_of(vec_at(offs_vmem, wv), LANES)
        ext = pl.multiple_of(ext, LANES)
        return pltpu.make_async_copy(
            payload_hbm.at[pl.ds(off, ext)],
            pbuf_vmem.at[pl.ds(0, ext)],
            pay_sem.at[0],
        )

    def start_pay(wv):
        ext = _align16_dyn(row_len(wv))

        @pl.when(ext > 0)
        def _():
            pay_copy(wv, ext).start()

    def wait_pay(wv):
        ext = _align16_dyn(row_len(wv))

        @pl.when(ext > 0)
        def _():
            pay_copy(wv, ext).wait()

    start_in(0, 0)

    def wave(wv, carry):
        slot = 0 if nbuf == 1 else lax.rem(wv, 2)
        if mode == MODE_PAYLOAD:
            start_pay(wv)
        if nbuf == 2:
            # Prefetch the next row into the other slot (its previous content,
            # row wv-1, was consumed in the previous iteration).
            @pl.when(wv + 1 < num_waves)
            def _():
                start_in(wv + 1, 1 - slot)

        wait_in(wv, slot)

        # The out slot was last used by row wv - nbuf; drain it before reuse.
        @pl.when(wv >= nbuf)
        def _():
            for c in out_copies(wv - nbuf, slot):
                c.wait()

        row = wv * (num_sc_cores * num_subcores) + flat
        active = row < rows
        eff_len = jnp.where(active, row_len(wv), 0)
        select_row(
            sbuf_vmem.at[pl.ds(pl.multiple_of(slot * slice_len, LANES), slice_len)],
            eff_len,
            active,
            obuf_vmem.at[pl.ds(slot * k, k)],
            kbuf_vmem.at[pl.ds(slot * k, k)],
            idx_base=vec_at(bases_vmem, wv) if mode == MODE_BASE else None,
            before_emit=(lambda: wait_pay(wv)) if mode == MODE_PAYLOAD else None,
        )
        for c in out_copies(wv, slot):
            c.start()
        if nbuf == 1:

            @pl.when(wv + 1 < num_waves)
            def _():
                start_in(wv + 1, 0)

        return carry

    lax.fori_loop(0, num_waves, wave, jnp.int32(0))
    for i in range(min(nbuf, num_waves)):
        wv = num_waves - 1 - i
        for c in out_copies(wv, wv % nbuf):
            c.wait()


def worker_table(v, num_waves: int, tab_stride: int):
    """Row-indexed i32[rows] -> flat [NUM_WORKERS, tab_stride]: worker f's wave
    wv entry (at f*tab_stride + wv) is row wv*32 + f; padding rows read 0."""
    rows = v.shape[0]
    v = jnp.pad(v.astype(jnp.int32), (0, num_waves * NUM_WORKERS - rows))
    v = v.reshape(num_waves, NUM_WORKERS).T
    return jnp.pad(v, ((0, 0), (0, tab_stride - num_waves))).reshape(-1)


def _sc_topk_rows(
    words_flat: jax.Array,  # i32[total] flat fp32 bits
    k: int,
    row_lens: jax.Array,  # i32[rows], already clipped to [0, slice_len]
    row_offs: jax.Array,  # i32[rows], word offset of each row's first column
    *,
    slice_len: int,
    scheduling_group_id: int | None = None,
    name: str,
    row_bases: jax.Array | None = None,
    payload_flat: jax.Array | None = None,
    flat_out: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Selects top-k of each (offset, length) window of `words_flat` on 32 subcores.

    Row i reads words_flat[row_offs[i] : row_offs[i] + align16(row_lens[i])];
    the caller guarantees that range is in bounds and align16(len) <= slice_len.
    Returns (local indices, winner score bits), i32[rows, k] each.

    With `row_bases` (i32[rows]) the emitted index is row_bases[i] + local;
    with `payload_flat` (same layout as `words_flat`) it is
    payload_flat[row_offs[i] + local]. -1 padding is unchanged. `flat_out`
    returns the raw i32[padded_rows * k] buffers (rows padded to 32).
    """
    assert row_bases is None or payload_flat is None
    mode = (
        MODE_BASE
        if row_bases is not None
        else MODE_PAYLOAD
        if payload_flat is not None
        else MODE_SLOT
    )
    rows = row_lens.shape[0]
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    assert sc.num_cores * sc.num_subcores == NUM_WORKERS, sc
    assert slice_len % LANES == 0

    num_waves = _cdiv(rows, NUM_WORKERS)
    padded_rows = num_waves * NUM_WORKERS
    tab_stride = _align_to(num_waves, LANES) + LANES  # +LANES: vec_at reads 16

    lens_tab = worker_table(row_lens, num_waves, tab_stride)
    offs_tab = worker_table(row_offs, num_waves, tab_stride)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((padded_rows * k,), jnp.int32)
    nbuf = 1 if num_waves == 1 else 2
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((nbuf * slice_len,), jnp.int32),
        pltpu.VMEM((tab_stride,), jnp.int32),
        pltpu.VMEM((tab_stride,), jnp.int32),
        pltpu.VMEM((nbuf * k,), jnp.int32),
        pltpu.VMEM((nbuf * k,), jnp.int32),
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.SemaphoreType.DMA((nbuf,)),
    )
    extra_in = ()
    if mode == MODE_BASE:
        extra_in = (worker_table(row_bases, num_waves, tab_stride),)
        scratch_types += (pltpu.VMEM((tab_stride,), jnp.int32),)
    elif mode == MODE_PAYLOAD:
        extra_in = (payload_flat,)
        scratch_types += (
            pltpu.VMEM((slice_len,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        )
    out = pl.kernel(
        functools.partial(
            _topk_rows_kernel,
            rows=rows,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            tab_stride=tab_stride,
            mode=mode,
        ),
        out_type=(slot_type, slot_type),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=name,
    )(words_flat, lens_tab, offs_tab, *extra_in)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    slots, keys = out
    if flat_out:
        return slots, keys
    return slots.reshape(padded_rows, k)[:rows], keys.reshape(padded_rows, k)[:rows]


def _as_words(scores):
    return (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )


def _sc_topk_direct(
    scores: jax.Array,  # f32[b, n] or i32 bits
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    write_empty: bool = True,
    scheduling_group_id: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single-stage top-k; empty rows issue no DMA."""
    del write_empty
    b, n = scores.shape
    words = _as_words(scores)
    slice_len = _align_to(n, LANES)
    if slice_len != n:
        words = jnp.pad(words, ((0, 0), (0, slice_len - n)))
    lens = jnp.clip(row_lengths.astype(jnp.int32), 0, n)
    offs = jnp.arange(b, dtype=jnp.int32) * slice_len
    return _sc_topk_rows(
        words.reshape(-1),
        k,
        lens,
        offs,
        slice_len=slice_len,
        scheduling_group_id=scheduling_group_id,
        name=f"sc_topk_v2_direct_b{b}_n{n}_k{k}",
    )


def _pick_partition(b: int, n: int, k: int) -> int:
    """Picks the power-of-two partition factor P (1..32) per row to maximize

    subcore utilization across all 32 subcores while fitting in VMEM.
    """
    p_min = _cdiv(n, MAX_SLICE_WORDS)
    p = 1
    while p < p_min:
        p *= 2

    max_p = 32 // b if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p


def stage1_geometry(row_lengths, n: int, p: int):
    """Rebalanced stage-1 partition (pure jnp).

    Returns (w, sub_lens i32[b*p], sub_offs i32[b*p], col_offsets i32[p]):
    w = clip(align16(cdiv(max(len), p)), 16, n_p); sub-row j of row r covers
    columns [j*w, j*w + w) and reads sub_lens[r*p+j] = clip(len_r - j*w, 0, w)
    words from flat word offset r*n + j*w.
    """
    b = row_lengths.shape[0]
    n_p = n // p
    lens = jnp.clip(row_lengths.astype(jnp.int32), 0, n)
    max_len = jnp.max(lens)
    w = _align16_dyn((max_len + (p - 1)) // p)
    w = jnp.clip(w, LANES, n_p).astype(jnp.int32)
    col = jnp.arange(p, dtype=jnp.int32) * w
    sub_lens = jnp.clip(lens[:, None] - col[None, :], 0, w).reshape(b * p)
    sub_offs = (
        jnp.arange(b, dtype=jnp.int32)[:, None] * jnp.int32(n) + col[None, :]
    ).reshape(b * p)
    return w, sub_lens, sub_offs, col


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "write_empty_rows",
        "scheduling_group_id",
        "stage2_scheduling_group_id",
        "return_scores",
    ),
)
def sparsecore_topk(
    scores: jax.Array,  # f32[b, n] or i32[b, n] of f32 bits
    k: int,
    row_lengths: jax.Array | None = None,  # i32[b], defaults to n
    *,
    write_empty_rows: bool = True,
    scheduling_group_id: int | None = None,
    stage2_scheduling_group_id: int | None = None,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:  # i32[b, k] or (i32[b, k], i32[b, k])
    """Exact top-k indices per row, unsorted, -1 suffix-padded.

    Uses 2-stage hierarchical subcore selection when b < 32 or n > 64K to utilize
    all 32 subcores in parallel without inter-core barrier overhead. Stage 1
    reads only columns below ``max(row_lengths)`` (see the module docstring).

    On the two-stage path, stage 1 emits each winner's score alongside its index,
    feeding directly into stage 2 without an intermediate HBM score gather.

    ``scores`` may be f32 or i32 holding those f32 bits; the comparison is on
    raw bits either way.

    ``scheduling_group_id`` and ``stage2_scheduling_group_id`` place each stage
    in an XLA scheduling group, so each can overlap a different caller's
    TensorCore work. Stage 2 only exists on the two-stage path.
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

    # The kernel addresses the flat scores with i32 word offsets (plus up to
    # LANES words of tail padding); split the rows so every part stays in range.
    max_rows = (2**31 - 1 - LANES) // n // 32 * 32
    if b > max_rows > 0:
        parts = [
            sparsecore_topk(
                scores[r : r + max_rows],
                k,
                row_lengths[r : r + max_rows],
                write_empty_rows=write_empty_rows,
                scheduling_group_id=scheduling_group_id,
                stage2_scheduling_group_id=stage2_scheduling_group_id,
                return_scores=return_scores,
            )
            for r in range(0, b, max_rows)
        ]
        if return_scores:
            return (
                jnp.concatenate([idx for idx, _ in parts]),
                jnp.concatenate([key for _, key in parts]),
            )
        return jnp.concatenate(parts)

    p = _pick_partition(b, n, k)

    if p == 1:
        slots, keys = _sc_topk_direct(
            scores,
            k,
            row_lengths,
            write_empty=write_empty_rows,
            scheduling_group_id=scheduling_group_id,
        )
        if return_scores:
            return slots, keys
        return slots

    # Stage 1: rebalanced sub-rows of traced width w, length-bounded DMA.
    n_p = n // p
    k_p = min(k, n_p)
    slice_len = _align_to(n_p, LANES)
    words = _as_words(scores).reshape(-1)
    if slice_len != n_p:
        # align16(sub_len) may then run past the last row's end; pad once.
        words = jnp.pad(words, (0, LANES))

    _, sub_lens, sub_offs, offsets = stage1_geometry(row_lengths, n, p)

    # Stage 2: Merge the P * k_p candidates to select the final top-k
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))

    if p * k_p <= MAX_SLICE_WORDS and (p * k_p) % LANES == 0:
        # Stage 1 emits global column indices (base j*w added in-kernel;
        # -1 for empty slots) and stage 2 carries them as the compaction
        # payload, so stage 2's index output is already the global column.
        # Stage 1's flat outputs are consumed in place: row r's candidates
        # are words [r*p*k_p, (r+1)*p*k_p) of both buffers (sub-row r*p+j at
        # (r*p+j)*k_p), which is exactly the stage-2 row layout.
        sub_bases = jnp.broadcast_to(offsets[None, :], (b, p)).reshape(b * p)
        cand_idx_flat, cand_keys_flat = _sc_topk_rows(
            words,
            k_p,
            sub_lens,
            sub_offs,
            slice_len=slice_len,
            scheduling_group_id=scheduling_group_id,
            name=f"sc_topk_v3_s1_b{b * p}_n{n_p}_k{k_p}",
            row_bases=sub_bases,
            flat_out=True,
        )
        m = p * k_p
        final_indices, final_cand_scores = _sc_topk_rows(
            cand_keys_flat,
            k,
            cand_lengths,
            jnp.arange(b, dtype=jnp.int32) * jnp.int32(m),
            slice_len=m,
            scheduling_group_id=stage2_scheduling_group_id,
            name=f"sc_topk_v3_s2_b{b}_n{m}_k{k}",
            payload_flat=cand_idx_flat,
        )
        if return_scores:
            return final_indices, final_cand_scores
        return final_indices

    # Too wide for one stage-2 slice (p * k_p > MAX_SLICE_WORDS, e.g. b == 1
    # with p == 32): recurse, then remap candidate slots to columns in XLA.
    local_slots, cand_scores = _sc_topk_rows(
        words,
        k_p,
        sub_lens,
        sub_offs,
        slice_len=slice_len,
        scheduling_group_id=scheduling_group_id,
        name=f"sc_topk_v2_s1_b{b * p}_n{n_p}_k{k_p}",
    )
    local_indices = local_slots.reshape(b, p, k_p)
    cand_scores = cand_scores.reshape(b, p * k_p)

    # Map local indices to global column indices (offsets j*w, traced).
    global_cand_indices = jnp.where(
        local_indices < 0,
        -1,
        offsets[None, :, None] + local_indices,
    ).reshape(b, p * k_p)

    # Stage 2: Merge the P * k_p candidates to select the final top-k
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))

    if p * k_p > MAX_SLICE_WORDS:
        # Too wide for one slice; recurse so stage 2 partitions in turn. `k_p`
        # is `k` whenever `p > 1`, so `p * k_p < n` and this terminates.
        final_cand_slots, final_cand_scores = sparsecore_topk(
            cand_scores,
            k,
            row_lengths=cand_lengths,
            write_empty_rows=write_empty_rows,
            scheduling_group_id=stage2_scheduling_group_id,
            return_scores=True,
        )
    else:
        final_cand_slots, final_cand_scores = _sc_topk_direct(
            cand_scores,
            k,
            cand_lengths,
            write_empty=write_empty_rows,
            scheduling_group_id=stage2_scheduling_group_id,
        )

    # Map candidate slots back to original column indices
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots, axis=1)
    final_indices = jnp.where(final_cand_slots >= 0, final_indices, -1)

    if return_scores:
        return final_indices, final_cand_scores
    return final_indices
