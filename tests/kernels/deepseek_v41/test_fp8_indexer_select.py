# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 decode selection: ragged groups, padding rows, short batches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import (
    _select_fp8_indexer_jax,
    select_fp8_indexer_jax,
)

TOKENS, WIDTH, TOPK, PAGE = 288, 16384, 512, 256


def _pack(keys, layout):
    """Feature-major pages: field f of record r at row-word f * page_size + r."""
    values, scales, _ = quantize_index_rows(keys)
    bits = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
    bits = bits.astype(np.uint32).reshape(-1, 32, 4)
    words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
    fields = np.concatenate(
        (words.view(np.int32), np.asarray(scales).view(np.int32)[:, None]), axis=1
    )
    records = np.arange(len(keys))
    words_at = (records // PAGE * layout.page_stride * 128)[:, None]
    words_at = words_at + np.arange(PAGE_ROWS)[None, :] * PAGE
    words_at = words_at + (records % PAGE)[:, None]
    pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
    pool[words_at] = fields
    return jnp.asarray(pool.reshape(-1, 128))


def _select(rng, counts, padding, short):
    """Request-major rows, then `padding` rows of request -1; returns inputs too."""
    requests = np.concatenate(
        (np.repeat(np.arange(len(counts)), counts), [-1] * padding)
    )
    low, high = (0, TOPK) if short else (TOPK + 1, WIDTH)
    lengths = np.where(requests >= 0, rng.integers(low, high + 1, TOKENS), 0)
    lengths = jnp.asarray(lengths, jnp.int32)
    layout = PackedFP8Layout(512, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(3), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = rng.integers(0, layout.pages, (len(counts), WIDTH // PAGE)).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(TOKENS, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (TOKENS, 32)), jnp.float32)
    # As query_start_loc: trailing padding rows are uncounted.
    offsets = jnp.asarray(np.concatenate(([0], np.cumsum(counts))), jnp.int32)
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE)
    select = jax.jit(
        lambda q, w, n: select_fp8_indexer_jax(
            q,
            (pool,),
            jnp.asarray(table),
            w,
            n,
            jnp.empty((TOKENS, 0), jnp.int32),
            jnp.asarray(requests),
            n.max().reshape(1),
            offsets,
            layout_args,
            TOPK,
            8,
            False,
            False,
            True,
            WIDTH,
        )[0]
    )
    selected = np.asarray(select(q, weights, lengths))
    return selected, (q, pool, weights, lengths, jnp.asarray(table), offsets, layout)


@pytest.mark.parametrize("short", [False, True], ids=["long", "short_batch"])
def test_ragged_requests_select_top_scores(short):
    """Requests that are not whole six-query groups, then padding rows."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 selection requires TPU")
    rng, padding = np.random.default_rng(11), 6
    counts = rng.integers(1, 10, 64)
    counts = counts[np.cumsum(counts) <= TOKENS - padding]
    counts[-1] += TOKENS - padding - counts.sum()
    selected, (q, pool, weights, lengths, table, offsets, layout) = _select(
        rng, counts, padding, short
    )
    real, lengths = int(counts.sum()), np.asarray(lengths)
    assert (selected[real:] == -1).all()
    if short:
        # Every visible key, as in the reference short-context bypass.
        for row in range(real):
            ids = np.sort(selected[row][selected[row] >= 0])
            np.testing.assert_array_equal(ids, np.arange(lengths[row]))
        return
    scores = np.asarray(
        fp8_paged_mqa_logits(
            q[:real],
            pool,
            weights[:real],
            jnp.asarray(lengths[:real]),
            table,
            query_start_loc=offsets,
            max_model_len=WIDTH,
            q_scale=jnp.ones_like(weights[:real]),
            query_size=6,
            cache_layout=layout,
        )
    )
    for row in range(real):
        ids = selected[row][selected[row] >= 0]
        assert np.unique(ids).size == ids.size == TOPK
        assert (ids < lengths[row]).all()
        # Split groups may round differently from the reference tiling.
        kth = np.sort(scores[row, : lengths[row]])[-TOPK]
        tolerance = 2e-2 * np.abs(scores[row, : lengths[row]]).max()
        assert scores[row, ids].min() >= kth - tolerance


@pytest.mark.parametrize(
    "rows,score_width,padding_id",
    [(512, 524288, -1), (512, 524288, 8), (288, 65536, 8)],
)
def test_prefill_preserves_score_boundaries_and_padding(rows, score_width, padding_id):
    """Keep exact IDs for score-budget splits, shared-Q splits, and padding."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 selection requires TPU")
    live, visible_width = rows - 4, 65536
    rng = np.random.default_rng(33)
    layout = PackedFP8Layout(visible_width // PAGE, page_size=PAGE)
    keys = jnp.asarray(rng.normal(size=(visible_width, 128)), jnp.bfloat16)
    pool = _pack(keys, layout)
    table = jnp.pad(
        jnp.arange(layout.pages, dtype=jnp.int32)[None],
        ((0, 7), (0, 0)),
        constant_values=-1,
    )
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.float8_e4m3fn)
    # One weighted head produces many exact BF16 score ties at the cutoff.
    weights = jnp.zeros((rows, 32), jnp.float32).at[:, 0].set(1)
    lengths = jnp.where(jnp.arange(rows) < live, 49152, 0).astype(jnp.int32)
    requests = jnp.where(jnp.arange(rows) < live, 0, padding_id).astype(jnp.int32)
    offsets = jnp.array([0] + [live] * 8, jnp.int32)
    if rows == 288:
        # The old shared-Q scorer restarts at 96 rows, splitting request 1's
        # six-query group because that request begins at row 73.
        table = table.at[1].set(jnp.roll(table[0], 1))
        requests = jnp.where(jnp.arange(rows) < 73, 0, requests)
        requests = jnp.where(
            (jnp.arange(rows) >= 73) & (jnp.arange(rows) < live), 1, requests
        )
        offsets = jnp.array([0, 73] + [live] * 7, jnp.int32)
        weights = jnp.asarray(rng.uniform(0.01, 0.1, (rows, 32)), jnp.float32)

    def run(q, weights, lengths, *, reference):
        arguments = (
            q,
            (pool,),
            table,
            weights,
            lengths,
            jnp.empty((rows, 0), jnp.int32),
            requests,
            lengths.max()[None],
            offsets,
            (layout.pages, layout.page_stride, layout.page_offset, PAGE),
            TOPK,
            8,
            False,
            False,
            True,
        )
        if reference:
            if score_width == visible_width:
                return _select_fp8_indexer_jax(*arguments, score_width)[0]
            # The previous wrapper narrows monotone IDs, but not a -1 suffix.
            width = visible_width if padding_id >= 0 else score_width
            return _select_fp8_indexer_jax(*arguments, width, prefill_chunk_rows=256)[0]
        return select_fp8_indexer_jax(*arguments, score_width)[0]

    compiled = jax.jit(run, static_argnames=("reference",))
    expected = compiled(q, weights, lengths, reference=True)
    actual = compiled(q, weights, lengths, reference=False)
    np.testing.assert_array_equal(actual, expected)
    assert (np.asarray(actual)[live:] == -1).all()
