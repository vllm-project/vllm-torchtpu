# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP8 MQA logits across page sizes, padded widths and ragged lengths."""

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


def _pack(keys, layout):
    """Feature-major pages: field f of record r at row-word f * page_size + r."""
    values, scales, _ = quantize_index_rows(keys)
    bits = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
    bits = bits.astype(np.uint32).reshape(-1, 32, 4)
    words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
    fields = np.concatenate(
        (words.view(np.int32), np.asarray(scales).view(np.int32)[:, None]), axis=1
    )
    size, records = layout.page_size, np.arange(len(keys))
    words_at = (records // size * layout.page_stride * 128)[:, None]
    words_at = (
        words_at + np.arange(PAGE_ROWS)[None, :] * size + (records % size)[:, None]
    )
    pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
    pool[words_at] = fields
    dequantized = values.astype(jnp.float32) * scales[:, None]
    return pool.reshape(-1, 128), dequantized


@pytest.mark.parametrize(
    "page_size,max_model_len",
    [(128, 8192), (128, 50000), (128, 98816), (256, 8192), (256, 98816)],
)
def test_paged_mqa_logits_matches_reference(page_size, max_model_len):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(7)
    counts = np.array([6, 5, 6])  # A ragged request leaves a partial query tile.
    # Near-full, empty-to-short, and mid-width histories; one invalid page.
    bases = np.array([max_model_len - 5, 0, max_model_len // 2 - 3])
    lengths = np.concatenate([b + np.arange(c) for b, c in zip(bases, counts)])
    owners = np.repeat(np.arange(len(counts)), counts)
    per_request = -(-max_model_len // page_size)
    layout = PackedFP8Layout(len(counts) * per_request, page_size=page_size)
    table = rng.permutation(layout.pages).reshape(len(counts), per_request)
    table[2, 1] = -1
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, dequantized = _pack(keys, layout)
    split = int(layout.pages * 0.6) * layout.page_stride
    tokens = len(lengths)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)

    result = fp8_paged_mqa_logits(
        q,
        (jnp.asarray(pool[:split]), jnp.asarray(pool[split:])),
        weights,
        jnp.asarray(lengths, jnp.int32),
        jnp.asarray(table, jnp.int32),
        query_start_loc=jnp.asarray(
            np.concatenate(([0], np.cumsum(counts))), jnp.int32
        ),
        max_model_len=max_model_len,
        q_scale=jnp.ones_like(weights),
        query_size=6,
        cache_layout=layout,
    )
    scores = np.asarray(result)

    position = np.arange(max_model_len)
    for request in range(len(counts)):
        rows = owners == request
        pages = table[request][position // page_size]
        logical = dequantized[np.maximum(pages, 0) * page_size + position % page_size]
        logits = jnp.einsum(
            "qhd,kd->qhk",
            q[rows].astype(jnp.float32),
            logical,
            precision=jax.lax.Precision.HIGHEST,
        )
        expected = np.asarray(
            (jnp.maximum(logits, 0) * weights[rows][..., None]).sum(1)
        )
        visible = (position[None] < lengths[rows][:, None]) & (pages >= 0)[None]
        actual = scores[rows]
        np.testing.assert_array_equal(np.isneginf(actual), ~visible)
        np.testing.assert_allclose(
            actual[visible], expected[visible], rtol=2e-2, atol=2e-2 * expected.max()
        )


def _calc_diff(x, y):
    """DeepGEMM's `calc_diff`: one minus the cosine-style similarity."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


# Mirrors vLLM tests/kernels/attention/test_deepgemm_attention.py
# `test_deepgemm_fp8_fp4_paged_mqa_logits`: BF16 references (unquantized q and
# K, signed weights), causal speculative rows and calc_diff < 1e-3.
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (2, 4)])
def test_paged_mqa_logits_matches_vllm_reference(batch_size, next_n, page_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(0)
    max_model_len, avg_kv, heads = 4096, 2048, 32
    layout = PackedFP8Layout(max_model_len * 2 * 64 // page_size, page_size=page_size)
    q = rng.normal(size=(batch_size * next_n, heads, 128)).astype(np.float32)
    kv = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    weights = rng.normal(size=(batch_size * next_n, heads)).astype(np.float32)
    context_lens = rng.integers(int(0.8 * avg_kv), int(1.2 * avg_kv), batch_size)
    per_request = max_model_len // page_size
    table = np.zeros((batch_size, per_request), np.int32)
    pool_ids = iter(rng.permutation(layout.pages))
    for i, length in enumerate(context_lens):
        for j in range(-(-length // page_size)):
            table[i, j] = next(pool_ids)
    # Row j of a request sees context_len - next_n + j + 1 keys.
    lengths = (context_lens[:, None] - next_n + 1 + np.arange(next_n)).reshape(-1)
    pool, _ = _pack(kv, layout)

    logits = fp8_paged_mqa_logits(
        jnp.asarray(q, jnp.bfloat16).astype(jnp.float8_e4m3fn),
        jnp.asarray(pool),
        jnp.asarray(weights),
        jnp.asarray(lengths, jnp.int32),
        jnp.asarray(table),
        query_start_loc=jnp.arange(batch_size + 1, dtype=jnp.int32) * next_n,
        max_model_len=max_model_len,
        q_scale=jnp.ones_like(weights),
        query_size=next_n,
        cache_layout=layout,
    )

    position = np.arange(max_model_len)
    keys = np.asarray(kv, np.float32)
    ref = np.zeros((batch_size * next_n, max_model_len), np.float32)
    for i in range(batch_size):
        rows = slice(i * next_n, (i + 1) * next_n)
        k = keys[table[i][position // page_size] * page_size + position % page_size]
        score = np.einsum(
            "qhd,kd->qhk", np.asarray(jnp.asarray(q[rows], jnp.bfloat16), np.float32), k
        )
        ref[rows] = (np.maximum(score, 0) * weights[rows][..., None]).sum(1)
    visible = position[None] < lengths[:, None]
    logits = np.asarray(logits)
    np.testing.assert_array_equal(np.isneginf(logits), ~visible)
    diff = _calc_diff(np.where(visible, logits, 0), np.where(visible, ref, 0))
    assert diff < 1e-3, f"{diff=}"


def test_unfilled_tiles_keep_every_visible_score():
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(3)
    page_size, max_model_len = 256, 262144  # Four 64K key tiles.
    counts = np.array([6, 6, 4])
    # One tile, the first block of the third tile, and empty rows.
    bases = np.array([1000, 2 * 65536 + 3, 0])
    lengths = np.concatenate([b + np.arange(c) for b, c in zip(bases, counts)])
    per_request = max_model_len // page_size
    layout = PackedFP8Layout(len(counts) * per_request, page_size=page_size)
    table = rng.permutation(layout.pages).reshape(len(counts), per_request)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, _ = _pack(keys, layout)
    tokens = len(lengths)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)

    def scores(fill_unscored):
        return np.asarray(
            fp8_paged_mqa_logits(
                q,
                jnp.asarray(pool),
                weights,
                jnp.asarray(lengths, jnp.int32),
                jnp.asarray(table, jnp.int32),
                query_start_loc=jnp.asarray(
                    np.concatenate(([0], np.cumsum(counts))), jnp.int32
                ),
                max_model_len=max_model_len,
                q_scale=jnp.ones_like(weights),
                query_size=6,
                cache_layout=layout,
                fill_unscored=fill_unscored,
            )
        )

    filled, unfilled = scores(True), scores(False)
    # Readers mask at the length but may load the rest of its 8-key block.
    for row, length in enumerate(lengths):
        end = -(-length // 8) * 8
        np.testing.assert_array_equal(unfilled[row, :end], filled[row, :end])
        assert np.isneginf(filled[row, length:]).all()


def test_resident_tiles_match_streamed_scores():
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(5)
    page_size, max_model_len = 256, 262144  # Four 64K tiles, all resident.
    # Requests span several query groups and grow, shrink and change tiles.
    counts = np.array([13, 1, 20, 7, 6])
    bases = np.array([65536 - 6, 200000, 1000, 131072 + 9, 0])
    lengths = np.concatenate([b + 2 * np.arange(c) for b, c in zip(bases, counts)])
    per_request = max_model_len // page_size
    layout = PackedFP8Layout(len(counts) * per_request, page_size=page_size)
    table = rng.permutation(layout.pages).reshape(len(counts), per_request)
    table[3, 300] = -1
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, _ = _pack(keys, layout)
    split = int(layout.pages * 0.6) * layout.page_stride
    tokens = len(lengths)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)

    def scores(reuse_keys):
        return np.asarray(
            fp8_paged_mqa_logits(
                q,
                (jnp.asarray(pool[:split]), jnp.asarray(pool[split:])),
                weights,
                jnp.asarray(lengths, jnp.int32),
                jnp.asarray(table, jnp.int32),
                query_start_loc=jnp.asarray(
                    np.concatenate(([0], np.cumsum(counts))), jnp.int32
                ),
                max_model_len=max_model_len,
                q_scale=jnp.ones_like(weights),
                query_size=6,
                cache_layout=layout,
                reuse_keys=reuse_keys,
            )
        )

    np.testing.assert_array_equal(scores(True), scores(False))


@pytest.mark.parametrize("weight_sign", [1, -1])
def test_block_maxima_match_scores(weight_sign):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(7)
    page_size, max_model_len, block = 256, 262144, 8
    counts = np.array([6, 6, 4])
    # Half and whole tiles, a short row, a row ending inside a block, empty rows.
    bases = np.array([1000, 2 * 65536 + 3, 0])
    lengths = np.concatenate([b + 5 * np.arange(c) for b, c in zip(bases, counts)])
    per_request = max_model_len // page_size
    layout = PackedFP8Layout(len(counts) * per_request, page_size=page_size)
    table = rng.permutation(layout.pages).reshape(len(counts), per_request)
    table[1, 600] = -1
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, _ = _pack(keys, layout)
    tokens = len(lengths)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    # Negative weights give negative and signed-zero scores.
    weights = jnp.asarray(
        weight_sign * rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32
    )

    def run(block_maxima):
        return fp8_paged_mqa_logits(
            q,
            jnp.asarray(pool),
            weights,
            jnp.asarray(lengths, jnp.int32),
            jnp.asarray(table, jnp.int32),
            query_start_loc=jnp.asarray(
                np.concatenate(([0], np.cumsum(counts))), jnp.int32
            ),
            max_model_len=max_model_len,
            q_scale=jnp.ones_like(weights),
            query_size=6,
            cache_layout=layout,
            fill_unscored=False,
            block_maxima=block_maxima,
        )

    plain = np.asarray(run(0))
    scores, maxima = (np.asarray(x) for x in run(block))
    assert maxima.shape == (tokens, max_model_len // block)
    for row, length in enumerate(lengths):
        blocks = -(-length // block)
        # Scores are unchanged; maxima cover every block a reader may load.
        np.testing.assert_array_equal(
            scores[row, : blocks * block], plain[row, : blocks * block]
        )
        expected = scores[row, : blocks * block].reshape(blocks, block).max(-1)
        np.testing.assert_array_equal(maxima[row, :blocks], expected)


@pytest.mark.parametrize(
    "rows,owned,max_model_len,reuse_keys,block",
    [
        # Decode bucket: one request with five drafts, then padding rows.
        (16, 6, 524288, False, 8),
        # Prefill chunk whose rows share resident keys.
        (256, 256, 262144, True, 0),
    ],
)
def test_null_page_tails_match_masked_tails(
    rows, owned, max_model_len, reuse_keys, block
):
    """Tables hold the null page past a request's rows; it scores as masked."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(11)
    page_size = 128
    layout = PackedFP8Layout(max_model_len // page_size + 64, page_size=page_size)
    words = rng.integers(0, 2**32, (layout.pages, layout.page_rows, 128), np.uint64)
    words = words.astype(np.uint32)
    octets = words.view(np.uint8)
    octets[(octets & 0x7F) == 0x7F] &= 0xFE  # No E4M3FN NaNs.
    words[:, PAGE_ROWS - 1] = np.float32(2.0**-6).view(np.uint32)
    pool = jnp.asarray(words.reshape(-1, 128).view(np.int32))
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(-0.1, 0.1, (rows, 32)), jnp.float32)
    offsets = jnp.asarray([0, owned, rows], jnp.int32)
    pages = rng.permutation(np.arange(1, layout.pages))[: max_model_len // page_size]

    def run(lengths, tail):
        visible = -(-lengths.max() // page_size)
        table = np.full((2, pages.size), tail, np.int32)
        table[0, :visible] = pages[:visible]
        return jax.tree.map(
            np.asarray,
            fp8_paged_mqa_logits(
                q,
                pool,
                weights,
                jnp.asarray(lengths, jnp.int32),
                jnp.asarray(table),
                query_start_loc=offsets,
                max_model_len=max_model_len,
                q_scale=jnp.ones_like(weights),
                query_size=6,
                cache_layout=layout,
                reuse_keys=reuse_keys,
                fill_unscored=False,
                block_maxima=block,
            ),
        )

    # Rows end early in, midway through and at the end of a tile.
    spread = 3 * np.arange(owned) if owned < rows else 293 * np.arange(rows)
    for base in (1, 30000, 65536 - 20, 65536 + 9000, 3 * 65536 + 40000):
        lengths = np.zeros(rows, np.int64)
        lengths[:owned] = np.minimum(base + spread, max_model_len)
        null, masked = run(lengths, 0), run(lengths, -1)
        null, masked = (x if block else (x,) for x in (null, masked))
        for row, length in enumerate(lengths):
            end = -(-length // 8) * 8
            np.testing.assert_array_equal(
                null[0][row, :end].view(np.int32), masked[0][row, :end].view(np.int32)
            )
            assert np.isfinite(null[0][row, :length]).all()
            if block:
                blocks = -(-length // block)
                np.testing.assert_array_equal(
                    null[1][row, :blocks].view(np.int32),
                    masked[1][row, :blocks].view(np.int32),
                )
