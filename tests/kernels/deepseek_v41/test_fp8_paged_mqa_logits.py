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


@pytest.mark.parametrize(
    "rows,width,page_size,one_request,unordered,reference_rows",
    [
        (129, 4096, 128, False, False, None),
        (257, 65568, 256, False, False, None),
        (513, 65568, 128, False, False, None),
        (512, 65568, 256, True, False, None),
        (512, 65536, 256, True, False, None),
        (512, 65536, 256, True, True, None),
        (512, 65536, 256, True, False, 256),
        (512, 65536, 256, True, True, 256),
    ],
)
def test_persistent_prefill_preserves_score_bits_and_topk(
    rows, width, page_size, one_request, unordered, reference_rows, monkeypatch
):
    """A window splits a six-row group; request boundaries reset the head tree."""
    if jax.default_backend() != "tpu":
        pytest.skip("Persistent packed FP8 scoring requires TPU")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import streaming_topk
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import prefill_topk

    rng = np.random.default_rng(17)
    page_count = -(-width // page_size)
    layout = PackedFP8Layout(page_count + 3, page_size=page_size)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, _ = _pack(keys, layout)
    split = (layout.pages // 2) * layout.page_stride
    banks = (jnp.asarray(pool[:split]), jnp.asarray(pool[split:]))
    table = np.stack([rng.permutation(layout.pages)[:page_count] for _ in range(3)])
    table[2, 1] = -1
    # An empty request between two nonempty ones also exercises offset lookup.
    boundary = rows // 2 + 1
    offsets = jnp.array([0, boundary, boundary, rows], jnp.int32)
    owners = jnp.where(jnp.arange(rows) < boundary, 0, 2).astype(jnp.int32)
    lengths = jnp.asarray(rng.integers(width // 3, width, size=rows), jnp.int32)
    lengths = lengths.at[0].set(0).at[-1].set(17)
    if one_request:
        table = table[:1]
        offsets = jnp.array([0, rows], jnp.int32)
        owners = jnp.zeros(rows, jnp.int32)
        lengths = jnp.array([25000, 45000, 65500, 0], jnp.int32)[
            (jnp.arange(rows) // 6) % 4
        ]
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, size=(rows, 32)), jnp.float32)
    table = jnp.asarray(table, jnp.int32)
    step = reference_rows or rows
    expected_scores = jnp.concatenate(
        [
            fp8_paged_mqa_logits(
                q[start : start + step],
                banks,
                weights[start : start + step],
                lengths[start : start + step],
                table,
                query_start_loc=jnp.clip(offsets - start, 0, min(step, rows - start)),
                max_model_len=width,
                q_scale=jnp.ones_like(weights[start : start + step]),
                query_size=6,
                cache_layout=layout,
            )
            for start in range(0, rows, step)
        ]
    )
    expected_ids = streaming_topk.streaming_topk(
        expected_scores,
        valid_lengths=lengths,
        k=512,
        exclude_neg_inf=True,
        reconstruct_threshold=2048,
        unordered=unordered,
    )
    original = prefill_topk.score_window
    captured = []

    def recording(*args, **kwargs):
        scores, state = original(*args, **kwargs)
        captured.append((scores, kwargs["start"], kwargs["rows"]))
        return scores, state

    def run(q, banks, weights, lengths, table):
        return prefill_topk.prefill_topk(
            q,
            banks,
            weights,
            lengths,
            table,
            offsets,
            owners,
            layout=layout,
            width=width,
            k=512,
            unordered=unordered,
            reference_rows=reference_rows,
        )

    def export(*args):
        return run(*args), tuple(captured)

    with monkeypatch.context() as patch:
        patch.setattr(prefill_topk, "score_window", recording)
        result, windows = jax.jit(export)(q, banks, weights, lengths, table)
    np.testing.assert_array_equal(result, expected_ids)
    reference = np.asarray(expected_scores).view(np.uint32)
    for scores, start, count in windows:
        start, count = int(start), int(count)
        owner = int(owners[start])
        group_start = int(offsets[owner])
        if reference_rows is not None:
            group_start = max(group_start, start // reference_rows * reference_rows)
        skip = (start - group_start) % 6
        actual = np.asarray(scores)[skip : skip + count, :width]
        visible = (
            np.arange(width)[None] < np.asarray(lengths)[start : start + count, None]
        )
        actual = np.where(visible, actual, np.float32(-np.inf))
        np.testing.assert_array_equal(
            actual.view(np.uint32), reference[start : start + count]
        )
    # Exporting scores changes their lifetime: also exercise the production graph.
    actual_ids = jax.jit(run)(q, banks, weights, lengths, table)
    np.testing.assert_array_equal(actual_ids, expected_ids)
