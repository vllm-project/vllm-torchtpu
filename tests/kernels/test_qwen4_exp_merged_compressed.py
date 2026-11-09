# SPDX-License-Identifier: Apache-2.0
"""Merged compressed keys (QWEN4_EXP_MERGE_COMPRESSED_K) address their region.

A merged layer's keys live after the carrier's ``N`` scheduler pages: block
``c``'s keys are column ``cols * N + c`` of that region (``cols`` blocks of keys
per page), bound as ``compressed_slot = -cols * N``. Reads, writes and the
StreamIndex table expansion must agree with each other, stay inside the
region, and hold the same keys an unmerged layer holds in its own pages.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    _compressed_expansion,
    compressed_kernel_pages,
    compressed_pairs,
    compressed_region,
    merged_compressed_slot,
    merged_span,
    write_compressed,
)

RATIO = 4


@pytest.fixture
def interpret(monkeypatch):
    """Run the Pallas writers in interpret mode (CPU-capable)."""
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
ROWS = 128 // RATIO  # compressed keys per 128-token kernel page


def _pool(pages, block_tokens):
    # [pages, block, kv-head slots, fp8 packing, lanes]: one 1,024-B row per token.
    return jnp.zeros((pages, block_tokens, 1, 4, 256), jnp.float8_e4m3fn)


def _geometry(block_tokens, sched_pages):
    page_bytes = block_tokens * 1024
    key_page_bytes = ROWS * 256 * 2
    logical = block_tokens // 128
    cols = page_bytes // (logical * key_page_bytes)
    region_pages = -(-sched_pages // cols)
    return page_bytes, key_page_bytes, logical, cols, region_pages


@pytest.mark.parametrize("block_tokens", [512, 3200])
def test_merged_reads_stay_in_the_region(block_tokens):
    n = 37
    page_bytes, key_bytes, logical, cols, region = _geometry(block_tokens, n)
    assert cols == 8
    pool = _pool(n + region, block_tokens)
    slot = merged_compressed_slot(cols * n)
    assert compressed_region(slot) == cols * n and merged_span(slot) == 1
    columns = jnp.arange(1, n, dtype=jnp.int32)
    pages = np.asarray(compressed_kernel_pages(columns, pool, slot, RATIO)).reshape(len(columns), logical)
    expect = (np.arange(1, n)[:, None] + cols * n) * logical + np.arange(logical)
    np.testing.assert_array_equal(pages, expect)
    start, end = pages.min() * key_bytes, (pages.max() + 1) * key_bytes
    assert start >= n * page_bytes, "keys must not touch the scheduler pages"
    assert end <= (n + region) * page_bytes, "keys must fit the appended pages"
    # Distinct blocks never share a key page.
    assert len(np.unique(pages)) == pages.size


@pytest.mark.parametrize("block_tokens", [512, 3200])
def test_expansion_matches_kernel_pages(block_tokens):
    n = 21
    _, _, logical, cols, region = _geometry(block_tokens, n)
    pool = _pool(n + region, block_tokens)
    page_bytes, key_bytes, _, _, _ = _geometry(block_tokens, n)
    physical = page_bytes // key_bytes
    for slot in (0, 3, merged_compressed_slot(cols * n)):
        if slot > 0 and (slot + 1) * logical > physical:
            continue
        expansion, stride, offset = _compressed_expansion(logical, physical, slot)
        assert expansion == logical
        columns = jnp.arange(n, dtype=jnp.int32)
        pages = np.asarray(compressed_kernel_pages(columns, pool, slot, RATIO)).reshape(n, logical)
        k = np.arange(logical)
        np.testing.assert_array_equal(pages, np.arange(n)[:, None] * stride + offset + k)


def _write(pool, table, slot, tokens, keys):
    """One request of ``tokens`` new tokens from position 0."""
    logical = jnp.arange(tokens, dtype=jnp.int32)
    req = jnp.zeros((tokens,), jnp.int32)
    complete = (logical % RATIO) == RATIO - 1
    starts = jnp.asarray([0, tokens], jnp.int32)
    context = jnp.asarray([0], jnp.int32)
    return write_compressed(pool, keys, complete, req, logical, starts, context, table, RATIO, slot)


@pytest.mark.parametrize("tokens", [16, 64])
def test_merged_write_is_read_back_like_unmerged(interpret, tokens):
    block_tokens, n = 512, 9
    _, _, logical, cols, region = _geometry(block_tokens, n)
    keys = jnp.asarray(np.random.default_rng(0).standard_normal((tokens, 256)), jnp.bfloat16)
    table = jnp.asarray([[5, 2, 7]], jnp.int32)
    # Unmerged: the layer owns whole pages; merged: the region after N pages.
    plain = _write(_pool(n, block_tokens), table, 0, tokens, keys)
    merged = _write(_pool(n + region, block_tokens), table, merged_compressed_slot(cols * n), tokens, keys)
    view = lambda pool, slot: np.asarray(  # noqa: E731
        pool.reshape(-1, 4, 256)[
            np.asarray(compressed_pairs(
                pool, keys, (jnp.arange(tokens) % RATIO) == RATIO - 1,
                jnp.zeros((tokens,), jnp.int32), jnp.arange(tokens, dtype=jnp.int32),
                jnp.asarray([0, tokens], jnp.int32), jnp.asarray([0], jnp.int32),
                table, RATIO, slot,
            )[0])
        ].astype(jnp.float32)
    )
    np.testing.assert_array_equal(view(plain, 0), view(merged, merged_compressed_slot(cols * n)))
    # The merged write touched only the region.
    assert not np.asarray(merged[:n].astype(jnp.float32)).any()
    assert np.asarray(merged[n:].astype(jnp.float32)).any()


def test_merged_pairs_follow_the_table():
    block_tokens, n = 512, 13
    _, _, _, cols, region = _geometry(block_tokens, n)
    pool = _pool(n + region, block_tokens)
    tokens = 2 * block_tokens
    logical = jnp.arange(tokens, dtype=jnp.int32)
    table = jnp.asarray([[4, 11]], jnp.int32)
    pair, write, _, _ = compressed_pairs(
        pool, jnp.zeros((tokens, 256), jnp.bfloat16), (logical % RATIO) == RATIO - 1,
        jnp.zeros((tokens,), jnp.int32), logical, jnp.asarray([0, tokens], jnp.int32),
        jnp.asarray([0], jnp.int32), table, RATIO, merged_compressed_slot(cols * n),
    )
    layer_pairs = block_tokens // (2 * RATIO)
    pair = np.asarray(pair)
    expect = (np.asarray(table)[0][np.arange(tokens) // block_tokens] + cols * n) * layer_pairs + (
        np.arange(tokens) % block_tokens
    ) // (2 * RATIO)
    np.testing.assert_array_equal(pair, expect)
    pairs_per_page = block_tokens * 4 // 4
    assert pair.min() >= n * pairs_per_page
    assert pair.max() < (n + region) * pairs_per_page


@pytest.mark.parametrize("span", [2])
def test_span_entry_owns_consecutive_columns(span):
    """PCP2: a main-KV ID covers ``span`` pages of tokens and its keys are
    columns ``span * c + i`` of the region, read and written alike."""
    block_tokens, n = 3200, 11
    _, _, logical, cols, _ = _geometry(block_tokens, n)
    region = -(-n * span // cols)
    pool = _pool(n + region, block_tokens)
    slot = merged_compressed_slot(cols * n, span)
    assert compressed_region(slot) == cols * n and merged_span(slot) == span
    table = jnp.asarray([[7, 2]], jnp.int32)
    tokens = 2 * span * block_tokens
    logical_ids = jnp.arange(tokens, dtype=jnp.int32)
    pair, _, _, _ = compressed_pairs(
        pool, jnp.zeros((tokens, 256), jnp.bfloat16), (logical_ids % RATIO) == RATIO - 1,
        jnp.zeros((tokens,), jnp.int32), logical_ids, jnp.asarray([0, tokens], jnp.int32),
        jnp.asarray([0], jnp.int32), table, RATIO, slot,
    )
    layer_pairs = block_tokens // (2 * RATIO)
    t = np.arange(tokens)
    entry = np.asarray(table)[0][t // (span * block_tokens)]
    column = entry * span + (t % (span * block_tokens)) // block_tokens + cols * n
    np.testing.assert_array_equal(np.asarray(pair), column * layer_pairs + (t % block_tokens) // (2 * RATIO))
    pairs_per_page = block_tokens * 4 // 4
    assert np.asarray(pair).max() < (n + region) * pairs_per_page
    # Kernel pages of an entry: span * logical consecutive pages of its columns.
    pages = np.asarray(compressed_kernel_pages(jnp.asarray([3], jnp.int32), pool, slot, RATIO))
    np.testing.assert_array_equal(pages, (3 * span + cols * n) * logical + np.arange(span * logical))
    expansion, stride, offset = _compressed_expansion(span * logical, 0, slot)
    assert (expansion, stride, offset) == (span * logical, span * logical, cols * n * logical)
