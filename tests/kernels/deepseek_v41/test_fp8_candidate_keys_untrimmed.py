# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate (sparse-logits) selection reads untrimmed gathered K correctly.

`gather_candidate_keys` returns its kernel output without slicing off the
rows its last gather steps round up to (`candidate_key_rows`); the scorer
addresses each group's region by its size (`candidate_region_rows`). Slicing
would copy the whole gathered K, which is 8.25 GiB for a 2,048-token prefill
chunk. This test runs the candidate selection with the untrimmed array and
with the array sliced to the regions, and requires the same selected indices.
TPU only (SparseCore gather).
"""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import candidates
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import (
    select_fp8_indexer_jax,
)

# 2 requests of 256 query rows; each row has 2,048 candidate blocks of 8 tokens
# (16,384 candidate tokens) out of a 65,536-token compressed context.
TOKENS, WIDTH, TOPK, PAGE, BLOCK, CANDIDATES = 512, 65536, 512, 256, 8, 2048


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


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    counts = [TOKENS // 2, TOKENS // 2]
    requests = np.repeat(np.arange(len(counts)), counts).astype(np.int32)
    lengths = rng.integers(WIDTH // 2, WIDTH + 1, TOKENS).astype(np.int32)
    layout = PackedFP8Layout(WIDTH // PAGE + 8, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(seed), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = np.stack(
        [rng.permutation(layout.pages)[: WIDTH // PAGE] for _ in counts]
    ).astype(np.int32)
    # Distinct candidate blocks per row, as published by the source layer.
    order = np.argsort(rng.random((TOKENS, WIDTH // BLOCK)), axis=1)
    cands = np.sort(order[:, :CANDIDATES], axis=1).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(TOKENS, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (TOKENS, 32)), jnp.float32)
    offsets = jnp.asarray(np.concatenate(([0], np.cumsum(counts))), jnp.int32)
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE)
    return q, pool, table, weights, lengths, cands, requests, offsets, layout_args


def _select(inputs):
    q, pool, table, weights, lengths, cands, requests, offsets, layout_args = inputs
    lengths = jnp.asarray(lengths)
    fn = jax.jit(
        lambda q, w, n, c: select_fp8_indexer_jax(
            q,
            (pool,),
            jnp.asarray(table),
            w,
            n,
            c,
            jnp.asarray(requests),
            n.max().reshape(1),
            offsets,
            layout_args,
            TOPK,
            BLOCK,
            False,
            True,
            True,
            WIDTH,
        )[0]
    )
    return np.asarray(fn(q, weights, lengths, jnp.asarray(cands)))


def test_untrimmed_candidate_keys_select_the_same_records():
    groups, blocks = TOKENS, 2 * CANDIDATES
    exact = groups * blocks // (PAGE_RECORDS // BLOCK) * PAGE_ROWS
    rows = candidates.candidate_key_rows(groups, blocks, BLOCK, PAGE, 1)
    assert rows > exact, "this shape should exercise the untrimmed K array"
    inputs = _inputs()
    untrimmed = _select(inputs)

    original = candidates.gather_candidate_keys

    def trimmed_gather(banks, ids, offsets, *, block, layout):
        keys = original(banks, ids, offsets, block=block, layout=layout)
        return keys[: ids.size // (PAGE_RECORDS // block) * PAGE_ROWS]

    candidates.gather_candidate_keys = trimmed_gather
    try:
        jax.clear_caches()
        trimmed = _select(inputs)
    finally:
        candidates.gather_candidate_keys = original
    assert untrimmed.shape == (TOKENS, TOPK)
    assert (untrimmed >= 0).any()
    assert np.array_equal(untrimmed, trimmed)


if __name__ == "__main__":
    test_untrimmed_candidate_keys_select_the_same_records()
    print("CANDIDATE_KEYS_TEST_OK")
