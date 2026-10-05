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
   rows are partitioned into P slices (where B * P <= 32) so that all 32
   subcores
   on the chip are utilized in parallel with zero inter-subcore synchronization.
2. In Stage 1, all 32 subcores independently find local top-k candidates on
their
   slice in lock-free, single-subcore mode.
3. In Stage 2, candidate scores are merged on SparseCore to produce the exact
   global top-k column indices.

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

LANES = 16
NUM_BUCKETS = 256
# Largest per-subcore resident slice, in 32-bit words (128KB; double-buffered by
# emit_pipeline to 256KB out of the 512KB tile memory).
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
    scores_hbm,  # i32[padded_b, slice_len] (fp32 score bits)
    lengths_hbm,  # i32[padded_b, LANES], or [padded_b, 2 * LANES] with a payload
    payload_hbm,  # i32[padded_b, pay_w], or None
    out_hbm,  # i32[padded_b, k]
    keys_hbm,  # i32[padded_b, k]          (fp32 winner score bits)
    glob_vmem,  # i32[NUM_BUCKETS + LANES] (histogram, then its prefix sum)
    rowbuf_vmem,  # i32[k + LANES]          (winner indices for one row)
    keybuf_vmem,  # i32[k + LANES]          (winner score bits for one row)
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    pay_w: int = 0,
):
    # With a payload (`pay_w > 0`), out_hbm receives `payload[pos - pay_off]`
    # for each winner at row position `pos` instead of `pos` itself, and `-1`
    # when `pos` falls outside `[pay_off, pay_off + pay_w)`. `pay_off` is the
    # row's lane LANES of `lengths_hbm`. Selection, slot order and padding are
    # untouched, so this equals gathering the payload with the returned slots.
    #
    # `keys_hbm` is i32[padded_b, k]: the fp32 score bits of each winner, paired
    # slot for slot with out_hbm, so the caller need not gather the scores back.
    del write_empty

    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

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

    def pipeline_step(*refs):
        if pay_w:
            scores_vmem, len_slice_vmem, pay_vmem, out_vmem, keys_vmem = refs
            pay_off = vec_at(len_slice_vmem, LANES)
        else:
            scores_vmem, len_slice_vmem, out_vmem, keys_vmem = refs
        w = pl.program_id(0)
        row = w * (num_sc_cores * num_subcores) + flat
        active = row < b
        eff_len = jnp.minimum(vec_at(len_slice_vmem, 0), n)
        eff_len = jnp.where(active, eff_len, 0)

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
            pos = i * LANES + lane_iota
            if pay_w:
                # `pay_off` is LANES-aligned, so a chunk is wholly in or out.
                rel = i * LANES - pay_off
                vals = pay_vmem[pl.ds(jnp.clip(rel, 0, pay_w - LANES), LANES)]
                inside = (pos >= pay_off) & (pos < pay_off + pay_w)
                pos = jnp.where(inside, vals, jnp.int32(-1))
            plsc.store_compressed(
                rowbuf_vmem.at[pl.ds(woff, LANES)],
                pos,
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

    cores_per_chip = num_sc_cores * num_subcores
    buf_count = 1 if num_waves == 1 else 2

    def row_spec(width):
        return pl.BlockSpec(
            (width,),
            lambda w: (w * cores_per_chip + flat,),
            pipeline_mode=pl.Buffered(buf_count),
        )

    in_specs = (row_spec(slice_len), row_spec(2 * LANES if pay_w else LANES))
    ins = (scores_hbm, lengths_hbm)
    if pay_w:
        in_specs += (row_spec(pay_w),)
        ins += (payload_hbm,)
    out_specs = (row_spec(k), row_spec(k))
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_waves,),
        in_specs=in_specs,
        out_specs=out_specs,
    )(*ins, out_hbm, keys_hbm)


def _topk_kernel(
    *refs,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    pay_w: int = 0,
):
    """Indices (or payload values) plus each winner's score."""
    if pay_w:
        scores_hbm, lengths_hbm, payload_hbm, *rest = refs
    else:
        scores_hbm, lengths_hbm, *rest = refs
        payload_hbm = None
    _topk_body(
        scores_hbm,
        lengths_hbm,
        payload_hbm,
        *rest,
        b=b,
        n=n,
        k=k,
        slice_len=slice_len,
        num_waves=num_waves,
        write_empty=write_empty,
        pay_w=pay_w,
    )


# Tile memory a subcore may spend on pipeline buffers and scratch when a
# payload rides along. The unpayloaded kernel peaks near 350KB of the 512KB
# at MAX_SLICE_WORDS; keep a margin below the hard limit.
_PAYLOAD_TILE_BUDGET_BYTES = 448 * 1024


def payload_fits(b: int, n: int, k: int, pay_w: int) -> bool:
    """Whether `_sc_topk_direct` can carry an i32[b, pay_w] payload in VMEM."""
    if pay_w <= 0 or pay_w % LANES or k % LANES:
        return False
    buf = 1 if _cdiv(b, 32) == 1 else 2
    words = buf * (_align_to(n, LANES) + 2 * LANES + pay_w + 2 * k)
    words += NUM_BUCKETS + LANES + 2 * (k + LANES)
    return 4 * words <= _PAYLOAD_TILE_BUDGET_BYTES


def _sc_topk_direct(
    scores: jax.Array,  # f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    write_empty: bool = True,
    scheduling_group_id: int | None = None,
    payload: jax.Array | None = None,  # i32[b, pay_w]
    payload_offset: jax.Array | None = None,  # i32[b], LANES-aligned
) -> tuple[jax.Array, jax.Array]:
    """Single-stage lock-free SparseCore top-k on 32 subcores.

    With ``payload``, the first output holds ``payload[r, pos - off[r]]`` for
    each winner position ``pos`` (``-1`` outside ``[off[r], off[r] + pay_w)``)
    instead of ``pos``: the gather by the returned slots, fused into the emit.

    ``scores`` may be f32 or the same values already bitcast to i32; the kernel
    compares raw bits either way.

    Returns each winner's score as i32 bits, paired slot-for-slot with the
    returned indices, and the bits of ``-inf`` in the pad slots.
    """
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
    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)
    padded_b = num_waves * 32
    pad_b = padded_b - b
    pad_n = slice_len - n
    if pad_b > 0 or pad_n > 0:
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))

    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b))
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1)))
    pay_w = 0
    extra_inputs = ()
    if payload is not None:
        pay_w = payload.shape[1]
        if payload.shape[0] != b or payload.dtype != jnp.int32 or pay_w % LANES:
            raise ValueError(f"bad payload {payload.shape} {payload.dtype}")
        if payload_offset is None:
            payload_offset = jnp.zeros((b,), jnp.int32)
        offs = jnp.pad(payload_offset.astype(jnp.int32), (0, padded_b - b))
        offs_2d = jnp.pad(offs[:, None], ((0, 0), (0, LANES - 1)))
        lengths_2d = jnp.concatenate([lengths_2d, offs_2d], axis=1)
        extra_inputs = (jnp.pad(payload, ((0, pad_b), (0, 0))).reshape(-1),)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    out_type = (slot_type, slot_type)

    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
    )

    out = pl.kernel(
        functools.partial(
            _topk_kernel,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
            pay_w=pay_w,
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_direct{'_pay' if pay_w else ''}_b{b}_n{n}_k{k}",
    )(words.reshape(-1), lengths_2d.reshape(-1), *extra_inputs)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    slots, keys = cast(Any, out)  # pyrefly: ignore[not-iterable]
    slots_2d = slots.reshape(padded_b, k)[:b]
    keys_2d = keys.reshape(padded_b, k)[:b]
    return slots_2d, keys_2d


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


def _gather_payload(
    idx: jax.Array,  # i32[b, k], -1 padded positions
    payload: jax.Array,  # i32[b, pay_w]
    payload_offset: jax.Array,  # i32[b]
) -> jax.Array:
    """Reference/fallback for the fused payload: ``payload[pos - off]`` or -1."""
    pay_w = payload.shape[1]
    rel = idx - payload_offset[:, None]
    inside = (idx >= 0) & (rel >= 0) & (rel < pay_w)
    got = jnp.take_along_axis(payload, jnp.clip(rel, 0, pay_w - 1), axis=1)
    return jnp.where(inside, got, -1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "write_empty_rows",
        "scheduling_group_id",
        "stage2_scheduling_group_id",
        "return_scores",
        "fused_payload",
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
    payload: jax.Array | None = None,  # i32[b, pay_w]
    payload_offset: jax.Array | None = None,  # i32[b], multiple of LANES
    fused_payload: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:  # i32[b, k] or (i32[b, k], i32[b, k])
    """Exact top-k indices per row, unsorted, -1 suffix-padded.

    Uses 2-stage hierarchical subcore selection when b < 32 or n > 64K to utilize
    all 32 subcores in parallel without inter-core barrier overhead.

    On the two-stage path, stage 1 emits each winner's score alongside its index,
    feeding directly into stage 2 without an intermediate HBM score gather.

    ``scores`` may be f32 or i32 holding those f32 bits; the comparison is on
    raw bits either way.

    ``scheduling_group_id`` and ``stage2_scheduling_group_id`` place each stage
    in an XLA scheduling group, so each can overlap a different caller's
    TensorCore work. Stage 2 only exists on the two-stage path.

    ``payload`` replaces each returned index ``pos`` with
    ``payload[r, pos - payload_offset[r]]``, or ``-1`` when that falls outside
    the payload's width -- slot for slot, so the result is the selection
    gathered through the payload. ``fused_payload`` folds every such gather,
    including the two-stage path's own candidate-to-column map, into the
    SparseCore emit wherever the payload fits in tile memory, instead of
    running it as a separate (SparseCore-offloaded) XLA gather. Both settings
    return identical arrays.
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

    pay_w = 0
    if payload is not None:
        if payload.ndim != 2 or payload.shape[0] != b:
            raise ValueError(f"payload must be [{b}, w], got {payload.shape}")
        payload = payload.astype(jnp.int32)
        pay_w = payload.shape[1]
        if payload_offset is None:
            payload_offset = jnp.zeros((b,), jnp.int32)
        payload_offset = payload_offset.astype(jnp.int32)

    p = _pick_partition(b, n, k)

    # Fast-path: single-stage execution when p == 1
    if p == 1:
        fuse = payload is not None and fused_payload and payload_fits(b, n, k, pay_w)
        slots, keys = _sc_topk_direct(
            scores,
            k,
            row_lengths,
            write_empty=write_empty_rows,
            scheduling_group_id=scheduling_group_id,
            **(dict(payload=payload, payload_offset=payload_offset) if fuse else {}),
        )
        if payload is not None and not fuse:
            slots = _gather_payload(slots, payload, payload_offset)
        if return_scores:
            return slots, keys
        return slots

    # Stage 1: Partition each row into P slices and find local top-k candidates
    n_p = n // p
    k_p = min(k, n_p)
    scores_p = scores.reshape(b * p, n_p)

    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0, n_p).reshape(b * p)

    # A user payload rides through stage 1 when every slice's offset stays
    # LANES-aligned; otherwise stage 1 emits positions and the payload is
    # applied to the final selection instead.
    fuse1 = (
        payload is not None
        and fused_payload
        and n_p % LANES == 0
        and payload_fits(b * p, n_p, k_p, pay_w)
    )
    local_slots, cand_scores = _sc_topk_direct(
        scores_p,
        k_p,
        lengths_p,
        write_empty=True,
        scheduling_group_id=scheduling_group_id,
        **(
            dict(
                payload=jnp.repeat(payload, p, axis=0),
                payload_offset=(payload_offset[:, None] - offsets[None, :]).reshape(
                    b * p
                ),
            )
            if fuse1
            else {}
        ),
    )
    cand_scores = cand_scores.reshape(b, p * k_p)

    if fuse1:
        # Already the payload values (-1 padded): stage 2 just carries them.
        global_cand_indices = local_slots.reshape(b, p * k_p)
    else:
        # Map local indices to global column indices
        local_indices = local_slots.reshape(b, p, k_p)
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
        if fused_payload:
            final_indices, final_cand_scores = sparsecore_topk(
                cand_scores,
                k,
                row_lengths=cand_lengths,
                write_empty_rows=write_empty_rows,
                scheduling_group_id=stage2_scheduling_group_id,
                return_scores=True,
                payload=global_cand_indices,
                fused_payload=True,
            )
        else:
            final_cand_slots, final_cand_scores = sparsecore_topk(
                cand_scores,
                k,
                row_lengths=cand_lengths,
                write_empty_rows=write_empty_rows,
                scheduling_group_id=stage2_scheduling_group_id,
                return_scores=True,
            )
            final_indices = None
    else:
        fuse2 = fused_payload and payload_fits(b, p * k_p, k, p * k_p)
        final_cand_slots, final_cand_scores = _sc_topk_direct(
            cand_scores,
            k,
            cand_lengths,
            write_empty=write_empty_rows,
            scheduling_group_id=stage2_scheduling_group_id,
            **(dict(payload=global_cand_indices) if fuse2 else {}),
        )
        final_indices = final_cand_slots if fuse2 else None

    if final_indices is None:
        # Map candidate slots back to original column indices
        safe_slots = jnp.maximum(final_cand_slots, 0)
        final_indices = jnp.take_along_axis(global_cand_indices, safe_slots, axis=1)
        final_indices = jnp.where(final_cand_slots >= 0, final_indices, -1)

    if payload is not None and not fuse1:
        final_indices = _gather_payload(final_indices, payload, payload_offset)

    if return_scores:
        return final_indices, final_cand_scores
    return final_indices
