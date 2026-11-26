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

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_TILE_ROWS = 128
NUM_LOCAL_HEADS = 3
QK_PADDED_DIM = 256
V_HEAD_DIM = 128
CAUSAL_SUB_FRAG_ROWS = 256
BF16_FRAGMENT_QUERIES = 512
FP8_FRAGMENT_QUERIES = 256
BF16_ONES_ROWS = 8
FP8_ONES_ROWS = 32
FP8_PROB_SHIFT = 4.0
FP8_STALE_LIMIT = 4.75
BF16_STALE_LIMIT = 64.0
VMEM_LIMIT_BYTES = 64 * 1024**2
SENTINEL_ROW_BOUND = 1 << 30


def _broadcast_bool_where(mask: jax.Array, x: jax.Array, fill: jax.Array) -> jax.Array:
    int_mask = jnp.broadcast_to(mask.astype(jnp.int32), x.shape) != 0
    return jnp.where(int_mask, x, fill)


def select_mla_flash_tile_rows(
    tokens: int,
    tile_rows: int = DEFAULT_TILE_ROWS,
    arena: tuple[jax.Array, jax.Array] | None = None,
    *,
    a2a_block_rows: int | None = None,
    mla_flash_qtile: int = 1024,
    mla_flash_ktile: int = 2048,
    mla_flash_prefix_tiles: bool = True,
) -> tuple[int, int, int]:
    q_tile = mla_flash_qtile if tokens % mla_flash_qtile == 0 else tile_rows
    k_tile = (
        mla_flash_ktile
        if tokens % mla_flash_ktile == 0
        else (512 if tokens % 512 == 0 else tile_rows)
    )
    if q_tile == 1024 and tokens >= 4096 and tokens % 2048 == 0:
        q_tile = 2048
    if k_tile == 2048 and tokens % 1024 == 0:
        k_tile = 1024
    if (
        mla_flash_prefix_tiles
        and tokens < mla_flash_qtile
        and tokens % 256 == 0
        and (tokens <= BF16_FRAGMENT_QUERIES or tokens % BF16_FRAGMENT_QUERIES == 0)
    ):
        q_tile = tokens
    if (
        a2a_block_rows is not None
        and q_tile < a2a_block_rows <= 2048
        and tokens % a2a_block_rows == 0
        and (
            a2a_block_rows <= BF16_FRAGMENT_QUERIES
            or a2a_block_rows % BF16_FRAGMENT_QUERIES == 0
        )
    ):
        q_tile = a2a_block_rows
    p_tile = k_tile
    if (
        arena is not None
        and mla_flash_prefix_tiles
        and arena[0].ndim == 4
        and arena[0].shape[2] % mla_flash_ktile == 0
    ):
        p_tile = mla_flash_ktile
    return q_tile, k_tile, p_tile


def can_fuse_mla_flash_a2a_deep_prefix(
    tokens: int,
    num_ranks: int,
    stripe_rows: int,
    *,
    tile_rows: int = DEFAULT_TILE_ROWS,
    arena: tuple[jax.Array, jax.Array] | None = None,
    mla_flash_qtile: int = 1024,
    mla_flash_ktile: int = 2048,
    mla_flash_prefix_tiles: bool = True,
) -> bool:
    block_rows = num_ranks * stripe_rows
    if (
        stripe_rows <= 0
        or stripe_rows % 8 != 0
        or block_rows <= 0
        or tokens % block_rows != 0
    ):
        return False
    q_tile, _, _ = select_mla_flash_tile_rows(
        tokens,
        tile_rows,
        arena,
        a2a_block_rows=block_rows,
        mla_flash_qtile=mla_flash_qtile,
        mla_flash_ktile=mla_flash_ktile,
        mla_flash_prefix_tiles=mla_flash_prefix_tiles,
    )
    return tokens % q_tile == 0 and q_tile % block_rows == 0


def _append_ones_rows(
    v_tile: jax.Array,
    key_block: jax.Array,
    tile_rows: int,
    value_mask_fn: Callable[[jax.Array], jax.Array] | None,
) -> jax.Array:
    if v_tile.dtype == jnp.float8_e4m3fn:
        if value_mask_fn is not None:
            value_keys = key_block * tile_rows + lax.broadcasted_iota(
                jnp.int32, (1, tile_rows), 1
            )
            v_tile = jnp.where(
                value_mask_fn(value_keys),
                v_tile.astype(jnp.bfloat16),
                jnp.zeros((), jnp.bfloat16),
            ).astype(jnp.float8_e4m3fn)
        return jnp.concatenate(
            (v_tile, jnp.ones((FP8_ONES_ROWS, tile_rows), v_tile.dtype)), axis=0
        )
    if value_mask_fn is not None:
        value_keys = key_block * tile_rows + lax.broadcasted_iota(
            jnp.int32, (1, tile_rows), 1
        )
        v_tile = jnp.where(
            value_mask_fn(value_keys), v_tile, jnp.zeros((), v_tile.dtype)
        )
    return jnp.concatenate(
        (v_tile, jnp.ones((BF16_ONES_ROWS, tile_rows), v_tile.dtype)), axis=0
    )


def _compact_exact_step_in_place(
    q_buf,
    k_tile: jax.Array,
    v_with_ones: jax.Array,
    max_ref,
    acc_ref,
    *,
    fragment_queries: int,
    pshift: float,
    exp_bf16: bool,
    score_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> None:
    q_rows = q_buf.shape[0]
    frag = min(fragment_queries, q_rows)
    n_frags = q_rows // frag
    prob_dtype = v_with_ones.dtype
    acc_rows = acc_ref.shape[0]

    @pl.loop(0, n_frags)
    def _fold_fragment(qi: jax.Array) -> None:
        q_slice = q_buf[pl.ds(qi * frag, frag), :]
        if k_tile.dtype != q_slice.dtype:
            q_slice = q_slice.astype(k_tile.dtype)
        old_max = max_ref[:, pl.ds(qi * frag, frag)]
        old_acc = acc_ref[:, pl.ds(qi * frag, frag)]
        scores = lax.dot_general(
            k_tile,
            q_slice,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if score_mask_fn is not None:
            scores = jnp.where(score_mask_fn(qi), scores, -jnp.inf)
        new_max = jnp.maximum(old_max, jnp.max(scores, axis=0, keepdims=True))
        safe_max = jnp.where(new_max == -jnp.inf, 0.0, new_max)
        exponent = scores - safe_max if not pshift else scores - (safe_max - pshift)
        probs = (
            jnp.exp2(exponent.astype(jnp.bfloat16)).astype(prob_dtype)
            if exp_bf16
            else jnp.exp2(exponent).astype(prob_dtype)
        )
        weighted = lax.dot_general(
            v_with_ones,
            probs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:acc_rows]
        correction = jnp.exp2(old_max - safe_max)
        max_ref[:, pl.ds(qi * frag, frag)] = new_max
        acc_ref[:, pl.ds(qi * frag, frag)] = old_acc * correction + weighted


def _compact_exact_functional_step(
    q_buf,
    k_tile: jax.Array,
    v_with_ones: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    max_ref,
    acc_ref,
    *,
    fragment_queries: int,
    pshift: float,
    exp_bf16: bool,
    score_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    max_ref[...], acc_ref[...] = carry
    _compact_exact_step_in_place(
        q_buf,
        k_tile,
        v_with_ones,
        max_ref,
        acc_ref,
        fragment_queries=fragment_queries,
        pshift=pshift,
        exp_bf16=exp_bf16,
        score_mask_fn=score_mask_fn,
    )
    return max_ref[...], acc_ref[...]


def _exact_unrolled_step(
    q: jax.Array,
    k_tile: jax.Array,
    v_with_ones: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    fragment_queries: int,
    pshift: float,
    exp_bf16: bool,
) -> tuple[jax.Array, jax.Array]:
    maximum, accumulator = carry
    frag = min(fragment_queries, q.shape[0])
    maxima: list[jax.Array] = []
    accumulators: list[jax.Array] = []
    previous: tuple[slice, jax.Array, jax.Array] | None = None
    prob_dtype = v_with_ones.dtype
    acc_rows = accumulator.shape[0]

    def fold_weighted(
        sl: slice, probabilities: jax.Array, correction: jax.Array
    ) -> None:
        weighted = lax.dot_general(
            v_with_ones,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:acc_rows]
        accumulators.append(accumulator[:, sl] * correction + weighted)

    for qi in range(q.shape[0] // frag):
        sl = slice(qi * frag, (qi + 1) * frag)
        scores = lax.dot_general(
            k_tile,
            q[sl],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        new_maximum = jnp.maximum(
            maximum[:, sl], jnp.max(scores, axis=0, keepdims=True)
        )
        safe_maximum = jnp.where(new_maximum == -jnp.inf, 0.0, new_maximum)
        exponent = (
            scores - safe_maximum if not pshift else scores - (safe_maximum - pshift)
        )
        probabilities = (
            jnp.exp2(exponent.astype(jnp.bfloat16)).astype(prob_dtype)
            if exp_bf16
            else jnp.exp2(exponent).astype(prob_dtype)
        )
        maxima.append(new_maximum)
        if previous is not None:
            fold_weighted(*previous)
        previous = (sl, probabilities, jnp.exp2(maximum[:, sl] - safe_maximum))
    assert previous is not None
    fold_weighted(*previous)
    return jnp.concatenate(maxima, axis=1), jnp.concatenate(accumulators, axis=1)


def _stale_unrolled_step(
    q: jax.Array,
    k_tile: jax.Array,
    v_with_ones: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    fragment_queries: int,
    pshift: float,
    exp_bf16: bool,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    maximum, accumulator = carry
    frag = min(fragment_queries, q.shape[0])
    maxima: list[jax.Array] = []
    accumulators: list[jax.Array] = []
    gaps: list[jax.Array] = []
    previous: tuple[slice, jax.Array, jax.Array, jax.Array] | None = None
    prob_dtype = v_with_ones.dtype
    acc_rows = accumulator.shape[0]

    def fold_weighted(
        sl: slice,
        probabilities: jax.Array,
        reference: jax.Array,
        new_maximum: jax.Array,
    ) -> None:
        weighted = lax.dot_general(
            v_with_ones,
            probabilities,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:acc_rows]
        accumulators.append(
            (accumulator[:, sl] + weighted) * jnp.exp2(reference - new_maximum)
        )

    for qi in range(q.shape[0] // frag):
        sl = slice(qi * frag, (qi + 1) * frag)
        scores = lax.dot_general(
            k_tile,
            q[sl],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        reference = maximum[:, sl]
        block_maximum = jnp.max(scores, axis=0, keepdims=True)
        exponent = scores - reference if not pshift else scores - (reference - pshift)
        probabilities = (
            jnp.exp2(exponent.astype(jnp.bfloat16)).astype(prob_dtype)
            if exp_bf16
            else jnp.exp2(exponent).astype(prob_dtype)
        )
        new_maximum = jnp.maximum(reference, block_maximum)
        maxima.append(new_maximum)
        gaps.append(block_maximum - reference)
        if previous is not None:
            fold_weighted(*previous)
        previous = (sl, probabilities, reference, new_maximum)
    assert previous is not None
    fold_weighted(*previous)
    return (
        jnp.concatenate(maxima, axis=1),
        jnp.concatenate(accumulators, axis=1),
    ), jnp.concatenate(gaps, axis=1)


def _causal_diagonal_first_block(
    q: jax.Array,
    k0: jax.Array,
    v0_with_ones: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    pshift: float,
) -> tuple[jax.Array, jax.Array]:
    maximum, accumulator = carry
    k_tile_rows = k0.shape[0]
    acc_rows = accumulator.shape[0]
    n_sub = k_tile_rows // CAUSAL_SUB_FRAG_ROWS
    full_frag = min(BF16_FRAGMENT_QUERIES, k_tile_rows)
    n_full = k_tile_rows // full_frag
    new_max: list[jax.Array] = []
    new_acc: list[jax.Array] = []
    prev: tuple[slice, int, jax.Array, jax.Array] | None = None

    def flush(sl: slice, k_end: int, probs: jax.Array, corr: jax.Array) -> None:
        weighted = lax.dot_general(
            v0_with_ones[:, :k_end],
            probs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:acc_rows]
        new_acc.append(accumulator[:, sl] * corr + weighted)

    for qi in range(n_sub):
        sl = slice(qi * CAUSAL_SUB_FRAG_ROWS, (qi + 1) * CAUSAL_SUB_FRAG_ROWS)
        k_end = (qi + 1) * CAUSAL_SUB_FRAG_ROWS
        scores = lax.dot_general(
            k0[:k_end],
            q[sl],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        key_rows = lax.broadcasted_iota(jnp.int32, (k_end, 1), 0)
        query_rows = (
            qi * CAUSAL_SUB_FRAG_ROWS + jnp.arange(CAUSAL_SUB_FRAG_ROWS)[None, :]
        )
        scores = jnp.where(key_rows <= query_rows, scores, -jnp.inf)
        nm = jnp.maximum(maximum[:, sl], jnp.max(scores, axis=0, keepdims=True))
        sm = jnp.where(nm == -jnp.inf, 0.0, nm)
        exp_arg = scores - sm if not pshift else scores - (sm - pshift)
        probs = jnp.exp2(exp_arg).astype(jnp.bfloat16)
        new_max.append(nm)
        if prev is not None:
            flush(*prev)
        prev = (sl, k_end, probs, jnp.exp2(maximum[:, sl] - sm))

    for hi_idx in range(n_full):
        sl = slice(
            k_tile_rows + hi_idx * full_frag,
            k_tile_rows + (hi_idx + 1) * full_frag,
        )
        scores = lax.dot_general(
            k0,
            q[sl],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        nm = jnp.maximum(maximum[:, sl], jnp.max(scores, axis=0, keepdims=True))
        sm = jnp.where(nm == -jnp.inf, 0.0, nm)
        exp_arg = scores - sm if not pshift else scores - (sm - pshift)
        probs = jnp.exp2(exp_arg).astype(jnp.bfloat16)
        new_max.append(nm)
        assert prev is not None
        flush(*prev)
        prev = (sl, k_tile_rows, probs, jnp.exp2(maximum[:, sl] - sm))
    assert prev is not None
    flush(*prev)
    return jnp.concatenate(new_max, axis=1), jnp.concatenate(new_acc, axis=1)


def _causal_diagonal_second_block(
    q: jax.Array,
    k1: jax.Array,
    v1_with_ones: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    pshift: float,
) -> tuple[jax.Array, jax.Array]:
    maximum, accumulator = carry
    k_tile_rows = k1.shape[0]
    acc_rows = accumulator.shape[0]
    n_sub = k_tile_rows // CAUSAL_SUB_FRAG_ROWS
    new_max: list[jax.Array] = [maximum[:, :k_tile_rows]]
    new_acc: list[jax.Array] = [accumulator[:, :k_tile_rows]]
    prev: tuple[slice, int, jax.Array, jax.Array] | None = None

    def flush(sl: slice, k_end: int, probs: jax.Array, corr: jax.Array) -> None:
        weighted = lax.dot_general(
            v1_with_ones[:, :k_end],
            probs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:acc_rows]
        new_acc.append(accumulator[:, sl] * corr + weighted)

    for qi in range(n_sub):
        sl = slice(
            k_tile_rows + qi * CAUSAL_SUB_FRAG_ROWS,
            k_tile_rows + (qi + 1) * CAUSAL_SUB_FRAG_ROWS,
        )
        k_end = (qi + 1) * CAUSAL_SUB_FRAG_ROWS
        scores = lax.dot_general(
            k1[:k_end],
            q[sl],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        key_rows = lax.broadcasted_iota(jnp.int32, (k_end, 1), 0)
        query_rows = (
            qi * CAUSAL_SUB_FRAG_ROWS + jnp.arange(CAUSAL_SUB_FRAG_ROWS)[None, :]
        )
        scores = jnp.where(key_rows <= query_rows, scores, -jnp.inf)
        nm = jnp.maximum(maximum[:, sl], jnp.max(scores, axis=0, keepdims=True))
        sm = jnp.where(nm == -jnp.inf, 0.0, nm)
        exp_arg = scores - sm if not pshift else scores - (sm - pshift)
        probs = jnp.exp2(exp_arg).astype(jnp.bfloat16)
        new_max.append(nm)
        if prev is not None:
            flush(*prev)
        prev = (sl, k_end, probs, jnp.exp2(maximum[:, sl] - sm))
    assert prev is not None
    flush(*prev)
    return jnp.concatenate(new_max, axis=1), jnp.concatenate(new_acc, axis=1)


def _fold_prefix_sequences_forward(
    carry: tuple[jax.Array, jax.Array],
    q_buf,
    pq: jax.Array,
    segment_ref,
    layer: jax.Array,
    head: jax.Array,
    first_row: jax.Array,
    rows: jax.Array,
    live: jax.Array,
    prefix_k_hbm,
    prefix_v_hbm,
    pk_buf,
    pv_buf,
    psems,
    max_ref,
    acc_ref,
    *,
    prefix_tile_rows: int,
    q_tile_rows: int,
    count: int,
    pshift: float,
    fp8: bool,
) -> tuple[jax.Array, jax.Array]:
    def _assign_seq(seq_idx: jax.Array, cur_seq: jax.Array) -> jax.Array:
        off_s = segment_ref[seq_idx]
        end_s = off_s + segment_ref[count + seq_idx]
        return jnp.where((rows >= off_s) & (rows < end_s), seq_idx, cur_seq)

    row_sequence = lax.fori_loop(
        0, count, _assign_seq, jnp.full(rows.shape, -1, jnp.int32)
    )

    def _bound_active_seqs(
        seq_idx: jax.Array, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        first_s, end_s_idx = bounds
        off_s = segment_ref[seq_idx]
        end_s = off_s + segment_ref[count + seq_idx]
        meets = live & (off_s < first_row + q_tile_rows) & (end_s > first_row)
        return (
            jnp.where(meets, jnp.minimum(first_s, seq_idx), first_s),
            jnp.where(meets, jnp.maximum(end_s_idx, seq_idx + 1), end_s_idx),
        )

    first_seq, end_seq = lax.fori_loop(
        0, count, _bound_active_seqs, (jnp.int32(count), jnp.int32(0))
    )
    stale_limit = FP8_STALE_LIMIT if fp8 else BF16_STALE_LIMIT
    stream_frag = (
        FP8_FRAGMENT_QUERIES
        if (fp8 or prefix_tile_rows > 1024)
        else BF16_FRAGMENT_QUERIES
    )
    edge_frag = min(FP8_FRAGMENT_QUERIES, q_tile_rows)

    def make_prefix_copies(kb: jax.Array, slot: jax.Array):
        return (
            pltpu.make_async_copy(
                prefix_k_hbm.at[
                    layer, head, pl.ds(kb * prefix_tile_rows, prefix_tile_rows), :
                ],
                pk_buf.at[slot],
                psems.at[0, slot],
            ),
            pltpu.make_async_copy(
                prefix_v_hbm.at[
                    layer, head, :, pl.ds(kb * prefix_tile_rows, prefix_tile_rows)
                ],
                pv_buf.at[slot],
                psems.at[1, slot],
            ),
        )

    def visit_sequence(
        sequence: jax.Array, seq_carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        offset = segment_ref[sequence]
        end = offset + segment_ref[count + sequence]
        base = segment_ref[3 * count + sequence]
        size = jnp.where(
            live & (offset < first_row + q_tile_rows) & (end > first_row),
            segment_ref[4 * count + sequence],
            0,
        )
        lo_row, hi_row = base, base + size
        lo = base // prefix_tile_rows
        hi = jnp.where(
            size > 0, (hi_row + prefix_tile_rows - 1) // prefix_tile_rows, lo
        )
        pa = (lo_row + prefix_tile_rows - 1) // prefix_tile_rows
        pb = hi_row // prefix_tile_rows
        pa = jnp.where(pa < pb, pa, hi)
        pb = jnp.where(pa < pb, pb, hi)

        def prefix_value_mask(value_rows: jax.Array) -> jax.Array:
            return (value_rows >= lo_row) & (value_rows < hi_row)

        def fold_edge_side(
            edge_idx: jax.Array, edge_carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            edge_lo = jnp.where(edge_idx == 0, lo, pb)
            edge_hi = jnp.where(edge_idx == 0, pa, hi)

            def fold_edge_block(
                kb: jax.Array, cur_carry: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array]:
                for cp in make_prefix_copies(kb, jnp.int32(0)):
                    cp.start()
                for cp in make_prefix_copies(kb, jnp.int32(0)):
                    cp.wait()
                vw_edge = _append_ones_rows(
                    pv_buf[0], kb, prefix_tile_rows, prefix_value_mask
                )
                key_rows = kb * prefix_tile_rows + lax.broadcasted_iota(
                    jnp.int32, (prefix_tile_rows, 1), 0
                )

                def edge_score_mask(qi: jax.Array) -> jax.Array:
                    r_sl = first_row + qi * edge_frag + jnp.arange(edge_frag)

                    def _seq_of_slice(s: jax.Array, cur_s: jax.Array) -> jax.Array:
                        o_s = segment_ref[s]
                        e_s = o_s + segment_ref[count + s]
                        return jnp.where((r_sl >= o_s) & (r_sl < e_s), s, cur_s)

                    rs_sl = lax.fori_loop(
                        0,
                        count,
                        _seq_of_slice,
                        jnp.full((edge_frag,), -1, jnp.int32),
                    )
                    return (
                        (rs_sl[None, :] == sequence)
                        & (key_rows >= lo_row)
                        & (key_rows < hi_row)
                    )

                return _compact_exact_functional_step(
                    q_buf,
                    pk_buf[0],
                    vw_edge,
                    cur_carry,
                    max_ref,
                    acc_ref,
                    fragment_queries=edge_frag,
                    pshift=pshift,
                    exp_bf16=fp8,
                    score_mask_fn=edge_score_mask,
                )

            return lax.fori_loop(edge_lo, edge_hi, fold_edge_block, edge_carry)

        after_edges = lax.fori_loop(0, 2, fold_edge_side, seq_carry)

        def stream_whole_blocks(
            c_init: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            slot0 = pa & 1
            for cp in make_prefix_copies(pa, slot0):
                cp.start()
            for cp in make_prefix_copies(pa, slot0):
                cp.wait()

            @pl.when(pa + 1 < pb)
            def _prefetch_second() -> None:
                for cp in make_prefix_copies(pa + 1, 1 - slot0):
                    cp.start()

            pk0 = pk_buf[slot0]
            vw0 = _append_ones_rows(pv_buf[slot0], pa, prefix_tile_rows, None)
            max_ref[...], acc_ref[...] = _exact_unrolled_step(
                pq,
                pk0,
                vw0,
                c_init,
                fragment_queries=stream_frag,
                pshift=pshift,
                exp_bf16=fp8,
            )

            def forward_block_body(kb: jax.Array, dummy_carry: jax.Array) -> jax.Array:
                cur = (max_ref[...], acc_ref[...])
                slt = kb & 1
                for cp in make_prefix_copies(kb, slt):
                    cp.wait()

                @pl.when(kb + 1 < pb)
                def _prefetch_next() -> None:
                    for cp in make_prefix_copies(kb + 1, 1 - slt):
                        cp.start()

                pk = pk_buf[slt]
                vw = _append_ones_rows(pv_buf[slt], kb, prefix_tile_rows, None)
                stale, gap = _stale_unrolled_step(
                    pq,
                    pk,
                    vw,
                    cur,
                    fragment_queries=stream_frag,
                    pshift=pshift,
                    exp_bf16=fp8,
                )
                max_ref[...], acc_ref[...] = lax.cond(
                    jnp.max(gap) > stale_limit,
                    lambda: _exact_unrolled_step(
                        pq,
                        pk,
                        vw,
                        cur,
                        fragment_queries=stream_frag,
                        pshift=pshift,
                        exp_bf16=fp8,
                    ),
                    lambda: stale,
                )
                return dummy_carry

            lax.fori_loop(pa + 1, pb, forward_block_body, jnp.int32(0))
            return max_ref[...], acc_ref[...]

        streamed = lax.cond(
            pa < pb, stream_whole_blocks, lambda c_in: c_in, after_edges
        )
        own = row_sequence[None, :] == sequence
        return (
            _broadcast_bool_where(own, streamed[0], seq_carry[0]),
            _broadcast_bool_where(own, streamed[1], seq_carry[1]),
        )

    return lax.fori_loop(
        first_seq, jnp.maximum(first_seq, end_seq), visit_sequence, carry
    )


def _compute_flash_query_tile(
    length_ref,
    segment_ref,
    layer_ref,
    q_ref,
    k_hbm,
    v_hbm,
    prefix_k_hbm,
    prefix_v_hbm,
    k_buf,
    v_buf,
    sems,
    pk_buf,
    pv_buf,
    psems,
    acc_ref,
    max_ref,
    *,
    head: jax.Array,
    query_block: jax.Array,
    k_tile_rows: int,
    q_tile_rows: int,
    count: int,
    prefixed: bool,
    prefix_tile_rows: int,
) -> jax.Array:
    length = length_ref[0]
    first_row = query_block * q_tile_rows
    rows = first_row + jnp.arange(q_tile_rows)
    q_buf = q_ref.at[0]
    q = q_ref[0]

    def _scan_window_bounds(
        s: jax.Array, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        cur_lowest, cur_interior = bounds
        off_s = segment_ref[s]
        end_s = off_s + segment_ref[count + s]
        meets = (
            (off_s < first_row + q_tile_rows) & (end_s > first_row) & (end_s > off_s)
        )
        holds = (off_s <= first_row) & (first_row + q_tile_rows <= end_s)
        return (
            jnp.where(meets, jnp.minimum(cur_lowest, off_s), cur_lowest),
            jnp.where(holds, (off_s + k_tile_rows - 1) // k_tile_rows, cur_interior),
        )

    lowest, interior = lax.fori_loop(
        0,
        count,
        _scan_window_bounds,
        (jnp.int32(SENTINEL_ROW_BOUND), jnp.int32(SENTINEL_ROW_BOUND)),
    )
    first_block = jnp.where(lowest < SENTINEL_ROW_BOUND, lowest // k_tile_rows, 0)
    last_query_key = ((query_block + 1) * q_tile_rows + k_tile_rows - 1) // k_tile_rows
    last_real_key = (length + k_tile_rows - 1) // k_tile_rows
    count_blocks = jnp.where(
        (first_row < length) & (lowest < SENTINEL_ROW_BOUND),
        jnp.minimum(last_query_key, last_real_key),
        0,
    )

    carry = (
        jnp.full((1, q_tile_rows), -jnp.inf, jnp.float32),
        jnp.zeros((V_HEAD_DIM + BF16_ONES_ROWS, q_tile_rows), jnp.float32),
    )
    fp8 = prefixed and pk_buf is not None and pk_buf.dtype == jnp.float8_e4m3fn
    pshift = FP8_PROB_SHIFT if fp8 else 0.0
    if prefixed:
        pq = q.astype(jnp.float8_e4m3fn) if fp8 else q
        carry = _fold_prefix_sequences_forward(
            carry,
            q_buf,
            pq,
            segment_ref,
            layer_ref[0],
            head,
            first_row,
            rows,
            first_row < length,
            prefix_k_hbm,
            prefix_v_hbm,
            pk_buf,
            pv_buf,
            psems,
            max_ref,
            acc_ref,
            prefix_tile_rows=prefix_tile_rows,
            q_tile_rows=q_tile_rows,
            count=count,
            pshift=pshift,
            fp8=fp8,
        )

    max_ref[...], acc_ref[...] = carry
    wb = jnp.minimum(count_blocks, first_row // k_tile_rows)
    wa = jnp.minimum(wb, jnp.maximum(first_block, interior))

    def make_window_copies(kb: jax.Array, slot: jax.Array):
        return (
            pltpu.make_async_copy(
                k_hbm.at[0, head, pl.ds(kb * k_tile_rows, k_tile_rows), :],
                k_buf.at[slot],
                sems.at[0, slot],
            ),
            pltpu.make_async_copy(
                v_hbm.at[0, head, pl.ds(kb * k_tile_rows, k_tile_rows), :],
                v_buf.at[slot],
                sems.at[1, slot],
            ),
        )

    def window_real_rows(indices: jax.Array) -> jax.Array:
        def _mark_real(s: jax.Array, inside_i32: jax.Array) -> jax.Array:
            o_s = segment_ref[s]
            e_s = o_s + segment_ref[count + s]
            return jnp.where((indices >= o_s) & (indices < e_s), 1, inside_i32)

        return (
            lax.fori_loop(0, count, _mark_real, jnp.zeros(indices.shape, jnp.int32)) > 0
        )

    masked_frag = min(FP8_FRAGMENT_QUERIES, q_tile_rows)
    window_stream_frag = min(BF16_FRAGMENT_QUERIES, q_tile_rows)

    def run_masked_window_range(w_lo: jax.Array, w_hi: jax.Array) -> None:
        @pl.loop(w_lo, w_hi)
        def _fold_masked_block(kb: jax.Array) -> None:
            for cp in make_window_copies(kb, jnp.int32(0)):
                cp.start()
            for cp in make_window_copies(kb, jnp.int32(0)):
                cp.wait()
            vw = _append_ones_rows(v_buf[0].T, kb, k_tile_rows, window_real_rows)
            key_rows = kb * k_tile_rows + lax.broadcasted_iota(
                jnp.int32, (k_tile_rows, 1), 0
            )

            def window_causal_mask(qi: jax.Array) -> jax.Array:
                r_sl = first_row + qi * masked_frag + jnp.arange(masked_frag)

                def _row_start(s: jax.Array, cur_rs: jax.Array) -> jax.Array:
                    o_s = segment_ref[s]
                    e_s = o_s + segment_ref[count + s]
                    return jnp.where((r_sl >= o_s) & (r_sl < e_s), o_s, cur_rs)

                rs_sl = lax.fori_loop(0, count, _row_start, r_sl + 1)
                return (key_rows <= r_sl[None, :]) & (key_rows >= rs_sl[None, :])

            _compact_exact_step_in_place(
                q_buf,
                k_buf[0],
                vw,
                max_ref,
                acc_ref,
                fragment_queries=masked_frag,
                pshift=pshift,
                exp_bf16=False,
                score_mask_fn=window_causal_mask,
            )

    run_masked_window_range(first_block, wa)

    @pl.when(wa < wb)
    def _stream_window_blocks() -> None:
        w_slot0 = wa & 1
        for cp in make_window_copies(wa, w_slot0):
            cp.start()
        for cp in make_window_copies(wa, w_slot0):
            cp.wait()

        @pl.when(wa + 1 < wb)
        def _prefetch_win_second() -> None:
            for cp in make_window_copies(wa + 1, 1 - w_slot0):
                cp.start()

        wk0 = k_buf[w_slot0]
        vw_first = _append_ones_rows(v_buf[w_slot0].T, wa, k_tile_rows, None)
        max_ref[...], acc_ref[...] = _exact_unrolled_step(
            q,
            wk0,
            vw_first,
            (max_ref[...], acc_ref[...]),
            fragment_queries=window_stream_frag,
            pshift=pshift,
            exp_bf16=False,
        )

        @pl.loop(wa + 1, wb)
        def _window_block_body(kb: jax.Array) -> None:
            cur = (max_ref[...], acc_ref[...])
            slt = kb & 1
            for cp in make_window_copies(kb, slt):
                cp.wait()

            @pl.when(kb + 1 < wb)
            def _prefetch_win_next() -> None:
                for cp in make_window_copies(kb + 1, 1 - slt):
                    cp.start()

            wk = k_buf[slt]
            vw = _append_ones_rows(v_buf[slt].T, kb, k_tile_rows, None)
            stale, gap = _stale_unrolled_step(
                q,
                wk,
                vw,
                cur,
                fragment_queries=window_stream_frag,
                pshift=pshift,
                exp_bf16=False,
            )
            max_ref[...], acc_ref[...] = stale

            @pl.when(jnp.max(gap) > BF16_STALE_LIMIT)
            def _rerun_window_overflow() -> None:
                max_ref[...], acc_ref[...] = cur
                _compact_exact_step_in_place(
                    q_buf,
                    wk,
                    vw,
                    max_ref,
                    acc_ref,
                    fragment_queries=window_stream_frag,
                    pshift=pshift,
                    exp_bf16=False,
                )

    can_use_causal_diag = (
        q_tile_rows == 2 * k_tile_rows
        and k_tile_rows >= 512
        and k_tile_rows % CAUSAL_SUB_FRAG_ROWS == 0
    )
    if can_use_causal_diag:
        fast_diag_ready = (
            (interior < SENTINEL_ROW_BOUND)
            & (first_row + q_tile_rows <= length)
            & (wb + 2 == count_blocks)
        )

        def _run_fast_causal_diagonal_pair() -> None:
            for cp in make_window_copies(wb, jnp.int32(0)):
                cp.start()
            for cp in make_window_copies(wb + 1, jnp.int32(1)):
                cp.start()
            for cp in make_window_copies(wb, jnp.int32(0)):
                cp.wait()
            vw0 = _append_ones_rows(v_buf[0].T, wb, k_tile_rows, None)
            carry_mid = _causal_diagonal_first_block(
                q,
                k_buf[0],
                vw0,
                (max_ref[...], acc_ref[...]),
                pshift=pshift,
            )
            for cp in make_window_copies(wb + 1, jnp.int32(1)):
                cp.wait()
            vw1 = _append_ones_rows(v_buf[1].T, wb + 1, k_tile_rows, None)
            max_ref[...], acc_ref[...] = _causal_diagonal_second_block(
                q,
                k_buf[1],
                vw1,
                carry_mid,
                pshift=pshift,
            )

        lax.cond(
            fast_diag_ready,
            _run_fast_causal_diagonal_pair,
            lambda: run_masked_window_range(wb, count_blocks),
        )
    else:
        run_masked_window_range(wb, count_blocks)

    denominator = acc_ref[V_HEAD_DIM : V_HEAD_DIM + 1, :]
    safe_denom = jnp.where(denominator > 0, denominator, 1.0)
    return (acc_ref[:V_HEAD_DIM, :] / safe_denom).T.astype(jnp.bfloat16)


def _flash_prefixed_kernel_packed(
    length_ref,
    segment_ref,
    layer_ref,
    q_ref,
    k_hbm,
    v_hbm,
    prefix_k_hbm,
    prefix_v_hbm,
    out_ref,
    k_buf,
    v_buf,
    sems,
    pk_buf,
    pv_buf,
    psems,
    acc_ref,
    max_ref,
    *,
    k_tile_rows: int,
    q_tile_rows: int,
    count: int,
    prefix_tile_rows: int,
) -> None:
    head, query_block = pl.program_id(0), pl.program_id(1)
    out_ref[0] = _compute_flash_query_tile(
        length_ref,
        segment_ref,
        layer_ref,
        q_ref,
        k_hbm,
        v_hbm,
        prefix_k_hbm,
        prefix_v_hbm,
        k_buf,
        v_buf,
        sems,
        pk_buf,
        pv_buf,
        psems,
        acc_ref,
        max_ref,
        head=head,
        query_block=query_block,
        k_tile_rows=k_tile_rows,
        q_tile_rows=q_tile_rows,
        count=count,
        prefixed=True,
        prefix_tile_rows=prefix_tile_rows,
    )


def _flash_unprefixed_kernel_packed(
    length_ref,
    segment_ref,
    q_ref,
    k_hbm,
    v_hbm,
    out_ref,
    k_buf,
    v_buf,
    sems,
    acc_ref,
    max_ref,
    *,
    k_tile_rows: int,
    q_tile_rows: int,
    count: int,
) -> None:
    head, query_block = pl.program_id(0), pl.program_id(1)
    out_ref[0] = _compute_flash_query_tile(
        length_ref,
        segment_ref,
        None,
        q_ref,
        k_hbm,
        v_hbm,
        None,
        None,
        k_buf,
        v_buf,
        sems,
        None,
        None,
        None,
        acc_ref,
        max_ref,
        head=head,
        query_block=query_block,
        k_tile_rows=k_tile_rows,
        q_tile_rows=q_tile_rows,
        count=count,
        prefixed=False,
        prefix_tile_rows=k_tile_rows,
    )


def _emit_owner_a2a_step(
    tables_ref,
    tile_out: jax.Array,
    out_ref,
    o_send_buf,
    local_sems,
    send_sems,
    recv_sem,
    *,
    block: jax.Array,
    head: jax.Array,
    num_ranks: int,
    stripe_rows: int,
    blocks: int,
    heads: int,
    sub_blocks: int,
    sp_axis: str,
) -> None:
    step = block * heads + head
    slot = step & 1
    rank = lax.axis_index(sp_axis) if num_ranks > 1 else jnp.int32(0)

    def local_copy(blk, sb, hd, slt):
        owner = tables_ref[0, 2, rank]
        src = o_send_buf.at[
            slt, pl.ds((sb * num_ranks + owner) * stripe_rows, stripe_rows), :
        ]
        dst = out_ref.at[
            rank,
            pl.ds((blk * sub_blocks + sb) * stripe_rows, stripe_rows),
            pl.ds(hd * V_HEAD_DIM, V_HEAD_DIM),
        ]
        return pltpu.make_async_copy(src, dst, local_sems.at[slt])

    def remote_copy(blk, sb, hd, slt, peer):
        owner = tables_ref[0, 2, peer]
        src = o_send_buf.at[
            slt, pl.ds((sb * num_ranks + owner) * stripe_rows, stripe_rows), :
        ]
        dst = out_ref.at[
            rank,
            pl.ds((blk * sub_blocks + sb) * stripe_rows, stripe_rows),
            pl.ds(hd * V_HEAD_DIM, V_HEAD_DIM),
        ]
        return pltpu.make_async_remote_copy(
            src,
            dst,
            send_sems.at[slt],
            recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def wait_sub_block(p_blk, p_hd, p_slt, sb) -> None:
        local_copy(p_blk, sb, p_hd, p_slt).wait()
        if num_ranks > 1:

            @pl.loop(1, num_ranks)
            def _wait_peer(offset) -> None:
                remote_copy(
                    p_blk, sb, p_hd, p_slt, (rank + offset) % num_ranks
                ).wait_send()

    def wait_step(prev_step) -> None:
        p_blk = prev_step // heads
        p_hd = prev_step % heads
        p_slt = prev_step & 1
        if sub_blocks == 1:
            wait_sub_block(p_blk, p_hd, p_slt, 0)
        else:

            @pl.loop(0, sub_blocks)
            def _wait_loop(sb) -> None:
                wait_sub_block(p_blk, p_hd, p_slt, sb)

    @pl.when(step >= 2)
    def _wait_prev() -> None:
        wait_step(step - 2)

    o_send_buf[slot, :, :] = tile_out

    if num_ranks > 1:

        @pl.when(step == 0)
        def _wait_barrier() -> None:
            pl.semaphore_wait(pltpu.get_barrier_semaphore(), num_ranks)

    def start_sub_block(sb) -> None:
        local_copy(block, sb, head, slot).start()
        if num_ranks > 1:

            @pl.loop(1, num_ranks)
            def _start_peer(offset) -> None:
                remote_copy(block, sb, head, slot, (rank + offset) % num_ranks).start()

    if sub_blocks == 1:
        start_sub_block(0)
    else:

        @pl.loop(0, sub_blocks)
        def _start_loop(sb) -> None:
            start_sub_block(sb)

    total_steps = blocks * heads

    @pl.when(step == total_steps - 1)
    def _finish_all() -> None:
        if total_steps >= 2:
            wait_step(total_steps - 2)
        wait_step(total_steps - 1)
        if num_ranks > 1:
            dummy = pltpu.make_async_copy(
                o_send_buf.at[0, pl.ds(0, stripe_rows), :],
                out_ref.at[rank, pl.ds(0, stripe_rows), pl.ds(0, V_HEAD_DIM)],
                recv_sem,
            )

            @pl.loop(0, total_steps * sub_blocks * (num_ranks - 1))
            def _wait_recv(_i) -> None:
                dummy.wait()


def _signal_entry_barrier(step: jax.Array, num_ranks: int) -> None:
    if num_ranks > 1:

        @pl.when(step == 0)
        def _signal() -> None:
            barrier = pltpu.get_barrier_semaphore()
            for peer in range(num_ranks):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )


def _flash_a2a_prefixed_kernel_packed(
    tables_ref,
    length_ref,
    segment_ref,
    layer_ref,
    q_ref,
    k_hbm,
    v_hbm,
    prefix_k_hbm,
    prefix_v_hbm,
    out_ref,
    k_buf,
    v_buf,
    sems,
    pk_buf,
    pv_buf,
    psems,
    acc_ref,
    max_ref,
    o_send_buf,
    local_sems,
    send_sems,
    recv_sem,
    *,
    k_tile_rows: int,
    q_tile_rows: int,
    count: int,
    prefix_tile_rows: int,
    num_ranks: int,
    stripe_rows: int,
    blocks: int,
    heads: int,
    sub_blocks: int,
    sp_axis: str,
) -> None:
    block, head = pl.program_id(0), pl.program_id(1)
    _signal_entry_barrier(block * heads + head, num_ranks)
    tile_out = _compute_flash_query_tile(
        length_ref,
        segment_ref,
        layer_ref,
        q_ref,
        k_hbm,
        v_hbm,
        prefix_k_hbm,
        prefix_v_hbm,
        k_buf,
        v_buf,
        sems,
        pk_buf,
        pv_buf,
        psems,
        acc_ref,
        max_ref,
        head=head,
        query_block=block,
        k_tile_rows=k_tile_rows,
        q_tile_rows=q_tile_rows,
        count=count,
        prefixed=True,
        prefix_tile_rows=prefix_tile_rows,
    )
    _emit_owner_a2a_step(
        tables_ref,
        tile_out,
        out_ref,
        o_send_buf,
        local_sems,
        send_sems,
        recv_sem,
        block=block,
        head=head,
        num_ranks=num_ranks,
        stripe_rows=stripe_rows,
        blocks=blocks,
        heads=heads,
        sub_blocks=sub_blocks,
        sp_axis=sp_axis,
    )


def _flash_a2a_unprefixed_kernel_packed(
    tables_ref,
    length_ref,
    segment_ref,
    q_ref,
    k_hbm,
    v_hbm,
    out_ref,
    k_buf,
    v_buf,
    sems,
    acc_ref,
    max_ref,
    o_send_buf,
    local_sems,
    send_sems,
    recv_sem,
    *,
    k_tile_rows: int,
    q_tile_rows: int,
    count: int,
    num_ranks: int,
    stripe_rows: int,
    blocks: int,
    heads: int,
    sub_blocks: int,
    sp_axis: str,
) -> None:
    block, head = pl.program_id(0), pl.program_id(1)
    _signal_entry_barrier(block * heads + head, num_ranks)
    tile_out = _compute_flash_query_tile(
        length_ref,
        segment_ref,
        None,
        q_ref,
        k_hbm,
        v_hbm,
        None,
        None,
        k_buf,
        v_buf,
        sems,
        None,
        None,
        None,
        acc_ref,
        max_ref,
        head=head,
        query_block=block,
        k_tile_rows=k_tile_rows,
        q_tile_rows=q_tile_rows,
        count=count,
        prefixed=False,
        prefix_tile_rows=k_tile_rows,
    )
    _emit_owner_a2a_step(
        tables_ref,
        tile_out,
        out_ref,
        o_send_buf,
        local_sems,
        send_sems,
        recv_sem,
        block=block,
        head=head,
        num_ranks=num_ranks,
        stripe_rows=stripe_rows,
        blocks=blocks,
        heads=heads,
        sub_blocks=sub_blocks,
        sp_axis=sp_axis,
    )


def _validate_flash_operands(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    segments: jax.Array,
    *,
    count: int,
    arena: tuple[jax.Array, jax.Array] | None,
    p_tile_rows: int,
) -> tuple[int, int]:
    if query.ndim != 3 or query.shape[2] != QK_PADDED_DIM:
        raise ValueError("query must be [H,T,256]")
    heads, tokens = query.shape[:2]
    if heads != NUM_LOCAL_HEADS:
        raise ValueError("the window phase takes this rank's 3 heads")
    if key.shape != (NUM_LOCAL_HEADS, tokens, QK_PADDED_DIM) or value.shape != (
        NUM_LOCAL_HEADS,
        tokens,
        V_HEAD_DIM,
    ):
        raise ValueError("packed MLA keys/values are the window's rows")
    fields = 3 if arena is None else 5
    if segments.shape != (fields * count,):
        raise ValueError(f"segments must be [{fields} * count]")
    if arena is not None:
        arena_keys, arena_values = arena
        if (
            arena_keys.ndim != 4
            or arena_keys.shape[1:] != (heads, arena_keys.shape[2], QK_PADDED_DIM)
            or arena_values.shape
            != (*arena_keys.shape[:2], V_HEAD_DIM, arena_keys.shape[2])
            or arena_keys.shape[2] % p_tile_rows
        ):
            raise ValueError(
                f"arena must be [L,{heads},C,256] / [L,{heads},128,C] "
                f"with C a multiple of {p_tile_rows}"
            )
    return heads, tokens


def mla_flash_prefill_packed_deep_prefix(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    length: jax.Array | int,
    segments: jax.Array,
    *,
    count: int,
    tile: int = DEFAULT_TILE_ROWS,
    arena: tuple[jax.Array, jax.Array] | None = None,
    layer: jax.Array | int = 0,
    mla_flash_qtile: int = 1024,
    mla_flash_ktile: int = 2048,
    mla_flash_prefix_tiles: bool = True,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    if query.ndim != 3:
        raise ValueError("query must be [H,T,256]")
    q_tile, k_tile, p_tile = select_mla_flash_tile_rows(
        query.shape[1],
        tile,
        arena,
        mla_flash_qtile=mla_flash_qtile,
        mla_flash_ktile=mla_flash_ktile,
        mla_flash_prefix_tiles=mla_flash_prefix_tiles,
    )
    heads, tokens = _validate_flash_operands(
        query, key, value, segments, count=count, arena=arena, p_tile_rows=p_tile
    )
    length_arr = jnp.asarray(length, jnp.int32).reshape((1,))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars: tuple[jax.Array, ...] = (length_arr, segments.astype(jnp.int32))
    operands: tuple[jax.Array, ...] = (query, key[None], value[None])
    in_specs: tuple[pl.BlockSpec, ...] = (
        pl.BlockSpec((1, q_tile, QK_PADDED_DIM), lambda h, i, *_: (h, i, 0)),
        hbm,
        hbm,
    )
    if arena is not None:
        scalars += (jnp.asarray(layer, jnp.int32).reshape((1,)),)
        operands += (arena[0], arena[1])
        in_specs += (hbm, hbm)
        kernel_fn = functools.partial(
            _flash_prefixed_kernel_packed,
            k_tile_rows=k_tile,
            q_tile_rows=q_tile,
            count=count,
            prefix_tile_rows=p_tile,
        )
        prefix_scratch: tuple[object, ...] = (
            pltpu.VMEM((2, p_tile, QK_PADDED_DIM), arena[0].dtype),
            pltpu.VMEM((2, V_HEAD_DIM, p_tile), arena[1].dtype),
            pltpu.SemaphoreType.DMA((2, 2)),
        )
    else:
        kernel_fn = functools.partial(
            _flash_unprefixed_kernel_packed,
            k_tile_rows=k_tile,
            q_tile_rows=q_tile,
            count=count,
        )
        prefix_scratch = ()
    name = "kimi_mla_flash_prefill_packed" + ("_prefixed" if arena is not None else "")
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((heads, tokens, V_HEAD_DIM), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(heads, tokens // q_tile),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, q_tile, V_HEAD_DIM), lambda h, i, *_: (h, i, 0)),
            scratch_shapes=(
                pltpu.VMEM((2, k_tile, QK_PADDED_DIM), jnp.bfloat16),
                pltpu.VMEM((2, k_tile, V_HEAD_DIM), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2, 2)),
                *prefix_scratch,
                pltpu.VMEM((V_HEAD_DIM + BF16_ONES_ROWS, q_tile), jnp.float32),
                pltpu.VMEM((1, q_tile), jnp.float32),
            ),
        ),
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=name,
    )(*scalars, *operands)


def mla_flash_prefill_packed_a2a_deep_prefix(
    tables: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    length: jax.Array | int,
    segments: jax.Array,
    *,
    count: int,
    num_ranks: int,
    stripe: int,
    tile: int = DEFAULT_TILE_ROWS,
    arena: tuple[jax.Array, jax.Array] | None = None,
    layer: jax.Array | int = 0,
    collective_id: int = 31,
    sp_axis: str = "k3_sp",
    mla_flash_qtile: int = 1024,
    mla_flash_ktile: int = 2048,
    mla_flash_prefix_tiles: bool = True,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    if query.ndim != 3:
        raise ValueError("query must be [H,T,256]")
    tokens_cand = query.shape[1]
    if tables.ndim != 3 or tables.shape[1] < 3 or tables.shape[2] != num_ranks:
        raise ValueError(
            f"tables {tables.shape} incompatible with num_ranks={num_ranks}"
        )
    block_rows = num_ranks * stripe
    if (
        stripe <= 0
        or stripe % 8 != 0
        or block_rows <= 0
        or tokens_cand % block_rows != 0
    ):
        raise ValueError(
            f"tokens={tokens_cand} incompatible with "
            f"num_ranks={num_ranks}, stripe={stripe}"
        )
    q_tile, k_tile, p_tile = select_mla_flash_tile_rows(
        tokens_cand,
        tile,
        arena,
        a2a_block_rows=block_rows,
        mla_flash_qtile=mla_flash_qtile,
        mla_flash_ktile=mla_flash_ktile,
        mla_flash_prefix_tiles=mla_flash_prefix_tiles,
    )
    heads, tokens = _validate_flash_operands(
        query, key, value, segments, count=count, arena=arena, p_tile_rows=p_tile
    )
    if tokens % q_tile != 0 or q_tile % block_rows != 0:
        raise ValueError(
            f"q_tile={q_tile} incompatible with "
            f"tokens={tokens} and block_rows={block_rows}"
        )
    blocks = tokens // q_tile
    sub_blocks = q_tile // block_rows
    rows_per_rank = tokens // num_ranks
    length_arr = jnp.asarray(length, jnp.int32).reshape((1,))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars: tuple[jax.Array, ...] = (
        jnp.asarray(tables, jnp.int32),
        length_arr,
        segments.astype(jnp.int32),
    )
    operands: tuple[jax.Array, ...] = (query, key[None], value[None])
    in_specs: tuple[pl.BlockSpec, ...] = (
        pl.BlockSpec((1, q_tile, QK_PADDED_DIM), lambda b, h, *_: (h, b, 0)),
        hbm,
        hbm,
    )
    if arena is not None:
        scalars += (jnp.asarray(layer, jnp.int32).reshape((1,)),)
        operands += (arena[0], arena[1])
        in_specs += (hbm, hbm)
        kernel_fn = functools.partial(
            _flash_a2a_prefixed_kernel_packed,
            k_tile_rows=k_tile,
            q_tile_rows=q_tile,
            count=count,
            prefix_tile_rows=p_tile,
            num_ranks=num_ranks,
            stripe_rows=stripe,
            blocks=blocks,
            heads=heads,
            sub_blocks=sub_blocks,
            sp_axis=sp_axis,
        )
        prefix_scratch: tuple[object, ...] = (
            pltpu.VMEM((2, p_tile, QK_PADDED_DIM), arena[0].dtype),
            pltpu.VMEM((2, V_HEAD_DIM, p_tile), arena[1].dtype),
            pltpu.SemaphoreType.DMA((2, 2)),
        )
    else:
        kernel_fn = functools.partial(
            _flash_a2a_unprefixed_kernel_packed,
            k_tile_rows=k_tile,
            q_tile_rows=q_tile,
            count=count,
            num_ranks=num_ranks,
            stripe_rows=stripe,
            blocks=blocks,
            heads=heads,
            sub_blocks=sub_blocks,
            sp_axis=sp_axis,
        )
        prefix_scratch = ()
    name = "kimi_mla_flash_prefill_packed_a2a" + (
        "_prefixed" if arena is not None else ""
    )
    effective_collective_id = collective_id if num_ranks > 1 else None
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(
            (num_ranks, rows_per_rank, heads * V_HEAD_DIM), jnp.bfloat16
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(blocks, heads),
            in_specs=in_specs,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, k_tile, QK_PADDED_DIM), jnp.bfloat16),
                pltpu.VMEM((2, k_tile, V_HEAD_DIM), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2, 2)),
                *prefix_scratch,
                pltpu.VMEM((V_HEAD_DIM + BF16_ONES_ROWS, q_tile), jnp.float32),
                pltpu.VMEM((1, q_tile), jnp.float32),
                pltpu.VMEM((2, q_tile, V_HEAD_DIM), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            collective_id=effective_collective_id,
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=name,
    )(*scalars, *operands)
