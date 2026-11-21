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
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    prepare_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    candidate_windows,
    fp8_gathered_mqa_logits,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import (
    _select_from_prepared_candidates,
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


def _select(rng, counts, padding, short, width):
    """Request-major rows, then `padding` rows of request -1; returns inputs too."""
    requests = np.concatenate(
        (np.repeat(np.arange(len(counts)), counts), [-1] * padding)
    )
    low, high = (0, TOPK) if short else (TOPK + 1, width)
    lengths = np.where(requests >= 0, rng.integers(low, high + 1, TOKENS), 0)
    lengths = jnp.asarray(lengths, jnp.int32)
    layout = PackedFP8Layout(max(512, width // PAGE), page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(3), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = rng.integers(0, layout.pages, (len(counts), width // PAGE)).astype(np.int32)
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
            width,
        )[0]
    )
    selected = np.asarray(select(q, weights, lengths))
    return selected, (q, pool, weights, lengths, jnp.asarray(table), offsets, layout)


# Four 64K key tiles: rows of a short query tile leave the later tiles unscored.
@pytest.mark.parametrize("width", [WIDTH, 4 * 65536], ids=["one_tile", "four_tiles"])
@pytest.mark.parametrize("short", [False, True], ids=["long", "short_batch"])
def test_ragged_requests_select_top_scores(short, width):
    """Requests that are not whole six-query groups, then padding rows."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 selection requires TPU")
    rng, padding = np.random.default_rng(11), 6
    counts = rng.integers(1, 10, 64)
    counts = counts[np.cumsum(counts) <= TOKENS - padding]
    counts[-1] += TOKENS - padding - counts.sum()
    selected, (q, pool, weights, lengths, table, offsets, layout) = _select(
        rng, counts, padding, short, width
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
            max_model_len=width,
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


def test_long_decode_row_beside_a_prefill_chunk_selects_top_scores():
    """A served mixed step: one long decode request and a 1,000-row chunk."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 selection requires TPU")
    rng = np.random.default_rng(13)
    tokens, width = 1024, 8 * 65536  # Wider than the four resident K tiles.
    counts = np.array([6, 1000])
    lengths = np.concatenate([150_000 + np.arange(6), 30_000 + np.arange(1000) // 2])
    lengths = np.concatenate([lengths, np.zeros(tokens - counts.sum(), int)])
    # Padding rows own the last request, as the attention layer clamps them.
    requests = np.repeat([0, 1], [6, tokens - 6])
    layout = PackedFP8Layout(1024, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(5), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = rng.integers(0, layout.pages, (2, width // PAGE)).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)
    offsets = jnp.asarray([0, 6, 1006], jnp.int32)
    n = jnp.asarray(lengths, jnp.int32)
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE)
    selected = np.asarray(
        jax.jit(select_fp8_indexer_jax, static_argnums=range(9, 16))(
            q,
            (pool,),
            jnp.asarray(table),
            weights,
            n,
            jnp.empty((tokens, 0), jnp.int32),
            jnp.asarray(requests),
            n.max().reshape(1),
            offsets,
            layout_args,
            TOPK,
            8,
            False,
            False,
            True,
            width,
        )[0]
    )
    # Reference scores in the selector's 256-row score tiles.
    rows = 2**27 // width
    bounds = offsets.at[-1].set(tokens)
    scores = np.concatenate(
        [
            np.asarray(
                fp8_paged_mqa_logits(
                    q[start : start + rows],
                    pool,
                    weights[start : start + rows],
                    n[start : start + rows],
                    jnp.asarray(table),
                    query_start_loc=jnp.clip(bounds - start, 0, rows),
                    max_model_len=width,
                    q_scale=jnp.ones_like(weights[start : start + rows]),
                    query_size=6,
                    cache_layout=layout,
                )
            )
            for start in range(0, tokens, rows)
        ]
    )
    for row in range(tokens):
        ids = selected[row][selected[row] >= 0]
        length = lengths[row]
        assert np.unique(ids).size == ids.size == min(TOPK, length)
        if length:
            rest = np.delete(scores[row, :length], ids)
            assert scores[row, ids].min() >= rest.max()


# Served decode: one request's six MTP rows padded to the 16-row bucket, or an
# idle rank whose rows are all padding.
@pytest.mark.parametrize("live", [6, 0], ids=["six_mtp_rows", "all_padding"])
def test_candidate_decode_scores_windows_and_selects_top_candidates(live):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 candidate scoring requires TPU")
    rng = np.random.default_rng(17)
    rows, candidates_per_row, block, width = 16, 2048, 8, 1 << 20
    layout = PackedFP8Layout(1024, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(7), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = rng.permutation(layout.pages).astype(np.int32)
    table = np.resize(table, (1, width // PAGE))
    seq = 127_927
    lengths = np.zeros(rows, np.int32)
    lengths[:live] = seq - live + 1 + np.arange(live)
    # Neighbouring rows share most candidate blocks, as published ones do.
    shared = rng.choice(seq // block, candidates_per_row, replace=False)
    candidates = np.full((rows, candidates_per_row), -1, np.int32)
    for row in range(live):
        ids = shared.copy()
        swap = rng.random(ids.size) < 0.1
        ids[swap] = rng.choice(seq // block, swap.sum())
        ids = np.unique(ids)
        extra = np.setdiff1d(np.arange(seq // block), ids)
        ids = np.concatenate([ids, rng.choice(extra, candidates_per_row - ids.size)])
        candidates[row] = np.sort(ids)
    requests = np.where(np.arange(rows) < live, 0, -1).astype(np.int32)
    prepared = jax.jit(prepare_candidates, static_argnums=(4, 5, 6))(
        (pool,),
        jnp.asarray(table),
        jnp.asarray(candidates),
        jnp.asarray(requests),
        layout,
        block,
        width,
    )
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (rows, 32)), jnp.float32)
    n = jnp.asarray(lengths)

    windows = candidate_windows(
        prepared.candidate_block_ids,
        prepared.block_spans,
        jnp.where(prepared.valid_queries, n, 0),
        block,
    )
    scores = np.asarray(
        fp8_gathered_mqa_logits(
            q,
            prepared.keys,
            weights,
            windows,
            query_start_loc=prepared.offsets,
            output_width=candidates_per_row * block,
        )
    )
    # Each row's two windows, in output columns from its union start.
    windows = np.asarray(windows)
    columns = np.arange(scores.shape[1]) + windows[:, :1]
    visible = ((columns >= windows[:, :1]) & (columns < windows[:, 2:3])) | (
        (columns >= windows[:, 1:2]) & (columns < windows[:, 3:4])
    )
    assert (scores[~visible] == -np.inf).all()
    assert np.isfinite(scores[visible]).all()
    assert (visible[:live].sum(1) > TOPK).all() and not visible[live:].any()

    selected = np.asarray(
        jax.jit(_select_from_prepared_candidates, static_argnums=range(4, 8))(
            q, weights, n, tuple(prepared), TOPK, block, True, width
        )
    )
    assert (selected[live:] == -1).all()
    if not live:
        return
    reference = np.asarray(
        fp8_paged_mqa_logits(
            q[:live],
            pool,
            weights[:live],
            n[:live],
            jnp.asarray(table),
            query_start_loc=jnp.asarray([0, live], jnp.int32),
            max_model_len=width,
            q_scale=jnp.ones_like(weights[:live]),
            query_size=6,
            cache_layout=layout,
        )
    )
    for row in range(live):
        tokens = (candidates[row][:, None] * block + np.arange(block)).ravel()
        tokens = tokens[tokens < lengths[row]]
        ids = selected[row]
        assert np.unique(ids).size == ids.size == TOPK
        assert np.isin(ids, tokens).all()
        kth = np.sort(reference[row, tokens])[-TOPK]
        tolerance = 2e-2 * np.abs(reference[row, tokens]).max()
        assert reference[row, ids].min() >= kth - tolerance
