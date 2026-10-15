# SPDX-License-Identifier: Apache-2.0
"""QSA prefill boundaries, bounded metadata and native BF16 cache views."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import decode_indexer
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing

streamindex = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


@pytest.fixture
def tpu_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


def test_shared_causal_default_keeps_current_group_visible():
    args = (jnp.array([8]), jnp.array([0, 8]), jnp.array([0, 0, 1]), 8, 512, 4)
    np.testing.assert_array_equal(
        streamindex._effective_row_lengths(*args), [1, 1, 1, 1, 2, 2, 2, 2]
    )
    np.testing.assert_array_equal(
        streamindex._effective_row_lengths(*args, components=qsa_indexing(3)),
        [0, 0, 0, 1, 1, 1, 1, 2],
    )


@pytest.mark.parametrize("pages", [64, 2048, 65536])
def test_decode_page_table_fits_scalar_memory(tpu_info, pages):
    chunk = decode_indexer.decode_query_chunk_size(pages)
    assert 1 <= chunk <= 512 and chunk & (chunk - 1) == 0
    assert chunk * (pages + 2) * 4 <= (tpu_info.smem_capacity_bytes - 32768) // 2


def test_reject_page_row_that_cannot_fit_scalar_memory(tpu_info):
    with pytest.raises(ValueError, match="scalar-prefetch"):
        decode_indexer.decode_query_chunk_size(tpu_info.smem_capacity_bytes // 4)


def test_nonpooled_prefill_bounds_expansion_and_preserves_request_ownership(
    monkeypatch, tpu_info
):
    tokens, pages, requests = 8193, 2048, 3
    chunk_shapes = []

    def transport(q, weights, cache, lengths, table, starts, distribution, **kwargs):
        # Inspect the actual operands at the shared-kernel boundary; numerical
        # top-k is covered separately on TPU.
        chunk_shapes.append((q.shape[0], table.size))
        owners = table.reshape(q.shape[0], pages)[:, 0]
        return jnp.broadcast_to(
            jnp.where(lengths > 0, owners + lengths, -1)[:, None],
            (q.shape[0], kwargs["k"]),
        )

    monkeypatch.setattr(decode_indexer, "streamindex_topk", transport)
    ids = jnp.arange(tokens, dtype=jnp.int32) % requests
    lens = jnp.where(jnp.arange(tokens) % 17 == 0, 0, 262144).astype(jnp.int32)
    table = jnp.broadcast_to(jnp.arange(requests)[:, None] * 100, (requests, pages))
    actual = decode_indexer.select_decode_blocks.__wrapped__(
        jnp.zeros((tokens, 4, 128), jnp.bfloat16),
        jnp.zeros((1, 32, 1, 128), jnp.bfloat16),
        lens,
        table,
        ids,
    )
    expected = jnp.where(lens > 0, ids * 100 + lens, -1)
    np.testing.assert_array_equal(
        actual, jnp.broadcast_to(expected[:, None], actual.shape)
    )
    limit = decode_indexer.decode_query_chunk_size(pages)
    assert chunk_shapes and all(q <= limit and n == q * pages for q, n in chunk_shapes)


@pytest.mark.parametrize("heads,dim,ratio", [(1, 256, 4), (4, 64, 8), (8, 256, 4)])
@pytest.mark.parametrize("fp8", [False, True])
def test_single_prefill_supports_tp_head_counts_and_padded_dimensions(
    tpu_info, heads, dim, ratio, fp8
):
    shape = jax.ShapeDtypeStruct
    packing = 4 if fp8 else 2
    query = shape((8, heads * 2, dim), jnp.bfloat16)
    result, cache = jax.eval_shape(
        functools.partial(
            block_sparse_gqa_attention,
            ratio=ratio,
            kv_heads=heads,
            query_block=32,
            kv_tile=ratio * 32,
        ),
        query,
        query,
        shape((8, 16), jnp.int32),
        shape((8,), jnp.int32),
        shape((8, 3), jnp.int32),
        shape(
            (3, 128, (2 * heads + packing - 1) // packing, packing, max(128, dim)),
            jnp.float8_e4m3fn if fp8 else jnp.bfloat16,
        ),
        shape((3,), jnp.int32),
    )
    assert result == query
    assert cache.shape[0] == 3
