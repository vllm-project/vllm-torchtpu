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
   rows are partitioned into P slices (where B * P <= 32) so that all 32 subcores
   on the chip are utilized in parallel with zero inter-subcore synchronization.
2. In Stage 1, all 32 subcores independently find local top-k candidates on their
   slice in lock-free, single-subcore mode.
3. In Stage 2, candidate scores are merged on SparseCore to produce the exact
   global top-k column indices.

Output rows are the top-k column indices in unspecified order, ``-1``
suffix-padded, matching an exact top-k over the given scores as a set (equal
boundary scores may be broken differently). ``-inf`` scores and positions at
or beyond ``row_lengths`` are never selected. NaNs are unsupported.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 16
NUM_BUCKETS = 256
# Largest per-subcore resident slice, in 32-bit words (256KB out of the 512KB
# tile memory; the histogram and small buffers use ~100KB more).
MAX_SLICE_WORDS = 64 * 1024

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
    scores_hbm,  # i32[b * n]                (fp32 score bits)
    lengths_hbm,  # i32[b_pad]
    out_hbm,  # i32[b * k]
    keys_hbm,  # i32[b * k], or None
    resident_vmem,  # i32[slice_len]
    priv_vmem,  # i32[16 * NUM_BUCKETS + LANES]  (lane-private histograms)
    glob_vmem,  # i32[NUM_BUCKETS + LANES]  (folded hist, then prefix sum)
    rowbuf_vmem,  # i32[k + LANES]           (winner indices for one row)
    keybuf_vmem,  # i32[k + LANES], or None
    len_vmem,  # i32[b_pad]
    sems,  # DMA[4]
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
):
    # Shared body of the two kernel entry points below. `keys_hbm` is
    # i32[b * k]: the fp32 score bits of each winner, paired slot for slot with
    # out_hbm, so the caller need not gather the scores back. It and its VMEM
    # staging buffer are None on the indices-only variant, which then traces
    # exactly as it did before the option existed.
    emit_keys = keys_hbm is not None
    assert (keybuf_vmem is not None) == emit_keys

    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    b_pad = len_vmem.shape[0]

    cp = pltpu.make_async_copy(lengths_hbm.at[pl.ds(0, b_pad)], len_vmem,
                               sems.at[0])
    cp.start()
    cp.wait()

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def zero_priv(bound):

        def body(i):
            priv_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES, ),
                                                           jnp.int32)

        plsc.parallel_loop(0, bound, unroll=4)(body)

    def fold_priv(bound):

        def body(i):
            off = i * LANES
            acc = jnp.zeros((LANES, ), jnp.int32)
            for m in range(LANES):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, LANES)]
            glob_vmem[pl.ds(off, LANES)] = acc

        plsc.parallel_loop(0, bound, unroll=2)(body)

    def scan_glob(bound):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def wave_body(w, _):
        row = w * (num_sc_cores * num_subcores) + flat
        active = row < b
        row_c = jnp.minimum(row, b - 1)
        eff_len = jnp.minimum(vec_at(len_vmem, row_c), n)
        eff_len = jnp.where(active, eff_len, 0)

        s = row_c * n
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi)

        @pl.when(i_hi > 0)
        def _():
            dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)

            def cvt(i):
                bits = resident_vmem[pl.ds(i * LANES, LANES)]
                resident_vmem[pl.ds(i * LANES, LANES)] = monotone_key(bits)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)

        def load_keys(i):
            key = resident_vmem[pl.ds(i * LANES, LANES)]
            j = i * LANES + lane_iota
            valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))
            return key, valid, j

        @pl.when(i_hi > 0)
        def _():
            zero_priv(NUM_BUCKETS)

            def hist0_body(i):
                key, valid, _ = load_keys(i)
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                    NUM_BUCKETS // 2)
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((LANES, ), jnp.int32),
                                       mask=valid)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            fold_priv(NUM_BUCKETS // LANES)

        @pl.when(i_hi == 0)
        def _():

            def zg(i):
                glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES, ),
                                                               jnp.int32)

            plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=4)(zg)

        total = scan_glob(NUM_BUCKETS // LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0,
                           vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            zero_priv(jnp.where(done, 0, NUM_BUCKETS))
            active_walk = jnp.where(done, 0, walk_chunks)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                prefix=prefix,
                shift=shift,
            ):
                key, valid, _ = load_keys(i)
                match = valid & (jnp.right_shift(key, shift) == prefix)
                bucket = jnp.bitwise_and(jnp.right_shift(key, lvl_shift),
                                         jnp.int32(lvl_mask))
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((LANES, ), jnp.int32),
                                       mask=match)

            plsc.parallel_loop(0, active_walk, unroll=4)(ref_body)
            fold_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            fold_priv(fold_bound)

            sub_total = scan_glob(fold_bound)
            b2 = find_bucket(fold_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0,
                                vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
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

        out_gate = active if write_empty else (active & (i_hi > 0))

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES, ), -1,
                                                            jnp.int32)
            if emit_keys:
                # Padding slots must read back as -inf, the same value the
                # caller used to substitute for a -1 index.
                keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES, ), BITS_NEG_INF, jnp.int32)

        fill_bound = jnp.where(out_gate & (total < k), k // LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        def emit_body(i, carry):
            woff, rank = carry
            key, valid, j = load_keys(i)
            hi = jnp.right_shift(key, shift)
            strict = valid & (hi > prefix)
            tie = valid & (hi == prefix)
            tie_rank = plsc.cumsum(tie.astype(jnp.int32)) + rank
            take = tie & (tie_rank <= my_take)
            sel = jnp.where(keep_all, valid, strict | take)
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)],
                                  j,
                                  mask=sel)
            if emit_keys:
                # monotone_key is an involution, so this recovers the original
                # fp32 bits. Same woff and same mask, so the two compactions
                # stay aligned slot for slot.
                plsc.store_compressed(keybuf_vmem.at[pl.ds(woff, LANES)],
                                      monotone_key(key),
                                      mask=sel)
            woff = woff + plsc.all_reduce_population_count(sel)[0]
            rank = tie_rank[LANES - 1]
            return woff, rank

        n_win, _ = plsc.parallel_loop(
            0,
            jnp.where(active, walk_chunks, 0),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_body)

        @pl.when(out_gate)
        def _():
            dma(
                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm.at[pl.ds(row_c * k, k)],
                2,
            )
            if emit_keys:
                dma(
                    keybuf_vmem.at[pl.ds(0, k)],
                    keys_hbm.at[pl.ds(row_c * k, k)],
                    3,
                )

        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)


# Pallas passes inputs, then outputs, then scratch, all positionally, so the
# extra key output and its staging buffer land part-way through that sequence
# rather than at the end. That is one arity per variant, hence one entry point
# per variant: each spells out the refs it is handed, and both delegate to
# `_topk_body`.
def _topk_kernel(
    scores_hbm,
    lengths_hbm,
    out_hbm,
    resident_vmem,
    priv_vmem,
    glob_vmem,
    rowbuf_vmem,
    len_vmem,
    sems,
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
):
    """Indices only."""
    _topk_body(
        scores_hbm,
        lengths_hbm,
        out_hbm,
        None,
        resident_vmem,
        priv_vmem,
        glob_vmem,
        rowbuf_vmem,
        None,
        len_vmem,
        sems,
        b=b,
        n=n,
        k=k,
        slice_len=slice_len,
        num_waves=num_waves,
        write_empty=write_empty,
    )


def _topk_kernel_keys(
    scores_hbm,
    lengths_hbm,
    out_hbm,
    keys_hbm,
    resident_vmem,
    priv_vmem,
    glob_vmem,
    rowbuf_vmem,
    keybuf_vmem,
    len_vmem,
    sems,
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
):
    """Indices plus each winner's score."""
    _topk_body(
        scores_hbm,
        lengths_hbm,
        out_hbm,
        keys_hbm,
        resident_vmem,
        priv_vmem,
        glob_vmem,
        rowbuf_vmem,
        keybuf_vmem,
        len_vmem,
        sems,
        b=b,
        n=n,
        k=k,
        slice_len=slice_len,
        num_waves=num_waves,
        write_empty=write_empty,
    )


def _sc_topk_direct(
    scores: jax.Array,  # f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    write_empty: bool = True,
    emit_keys: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Single-stage lock-free SparseCore top-k on 32 subcores.

    With ``emit_keys``, also returns the f32 score of each winner, paired
    slot-for-slot with the returned indices and ``-inf`` in the pad slots.
    """
    b, n = scores.shape
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    words = jax.lax.bitcast_convert_type(scores, jnp.int32)
    b_pad = _align_to(b, LANES)
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, b_pad - b))

    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((b * k, ), jnp.int32)
    out_type = (slot_type, slot_type) if emit_keys else slot_type

    scratch_types = (
        pltpu.VMEM((slice_len, ), jnp.int32),
        pltpu.VMEM((LANES * NUM_BUCKETS + LANES, ), jnp.int32),
        pltpu.VMEM((NUM_BUCKETS + LANES, ), jnp.int32),
        pltpu.VMEM((k + LANES, ), jnp.int32),
    )
    if emit_keys:
        scratch_types += (pltpu.VMEM((k + LANES, ), jnp.int32), )
    scratch_types += (
        pltpu.VMEM((b_pad, ), jnp.int32),
        pltpu.SemaphoreType.DMA((4, )),
    )

    # The kernel names itself after the variant that was built, so a profile
    # scope name is enough to tell which path actually ran.
    suffix = "_keys" if emit_keys else ""
    out = pl.kernel(
        functools.partial(
            _topk_kernel_keys if emit_keys else _topk_kernel,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}{suffix}",
    )(words.reshape(-1), lengths)
    if emit_keys:
        slots, keys = out
        return slots.reshape(b, k), jax.lax.bitcast_convert_type(
            keys.reshape(b, k), jnp.float32)
    return out.reshape(b, k)


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


@functools.partial(jax.jit,
                   static_argnames=("k", "write_empty_rows",
                                    "_reference_stage2_gather"))
def sparsecore_topk(
    scores: jax.Array,  # f32[b, n]
    k: int,
    row_lengths: jax.Array | None = None,  # i32[b], defaults to n
    *,
    write_empty_rows: bool = True,
    _reference_stage2_gather: bool = False,
) -> jax.Array:  # i32[b, k]
    """Exact top-k indices per row, unsorted, -1 suffix-padded.

  Uses 2-stage hierarchical subcore selection when b < 32 or n > 64K to utilize
  all 32 subcores in parallel without inter-core barrier overhead.

  On the two-stage path stage 1 emits each winner's score alongside its index,
  so stage 2 does not gather those scores back out of `scores`. Stage 1 has
  already read every one of them.

  `_reference_stage2_gather` is a test seam, not a tuning knob. It selects the
  earlier formulation, which performs that gather. The two return the same
  indices bit for bit; the differential test in
  `tests/kernels/test_dsv4_sparsecore_topk.py` is what establishes that, and it
  is the reason this branch is still here. No production caller sets it --
  `streamindex_topk` does not expose it.
  """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    b, n = scores.shape
    if n % LANES != 0:
        raise ValueError(f"{n=} must be a multiple of {LANES}")
    if k % LANES != 0 or not 0 < k <= 4096:
        raise ValueError(
            f"{k=} must be a positive multiple of {LANES}, at most 4096")

    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    p = _pick_partition(b, n, k)

    # Fast-path: single-stage execution when p == 1
    if p == 1:
        return _sc_topk_direct(scores,
                               k,
                               row_lengths,
                               write_empty=write_empty_rows)

    # Stage 1: Partition each row into P slices and find local top-k candidates
    n_p = n // p
    k_p = min(k, n_p)
    scores_p = scores.reshape(b * p, n_p)

    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)

    if _reference_stage2_gather:
        # Test-seam formulation: stage 1 returns indices only, and stage 2's
        # scores are gathered back below. `cand_scores = None` is what couples
        # the two halves, so neither can be changed without the other.
        cand_scores = None
        local_indices = _sc_topk_direct(scores_p,
                                        k_p,
                                        lengths_p,
                                        write_empty=True).reshape(b, p, k_p)
    else:
        local_slots, cand_scores = _sc_topk_direct(scores_p,
                                                   k_p,
                                                   lengths_p,
                                                   write_empty=True,
                                                   emit_keys=True)
        local_indices = local_slots.reshape(b, p, k_p)
        cand_scores = cand_scores.reshape(b, p * k_p)

    # Map local indices to global column indices
    global_cand_indices = jnp.where(
        local_indices < 0,
        -1,
        offsets[None, :, None] + local_indices,
    ).reshape(b, p * k_p)

    if cand_scores is None:
        # The gather the shipped path deletes. Reached only from the test seam.
        safe_cand_indices = jnp.maximum(global_cand_indices, 0)
        cand_scores = jnp.take_along_axis(scores, safe_cand_indices, axis=1)
        cand_scores = jnp.where(global_cand_indices >= 0, cand_scores,
                                -jnp.inf)

    # Stage 2: Merge the P * k_p candidates to select the final top-k
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))

    final_cand_slots = _sc_topk_direct(cand_scores,
                                       k,
                                       cand_lengths,
                                       write_empty=write_empty_rows)

    # Map candidate slots back to original column indices
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices,
                                        safe_slots,
                                        axis=1)
    final_indices = jnp.where(final_cand_slots >= 0, final_indices, -1)

    return final_indices
