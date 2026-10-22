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
"""QSA prefill boundaries, bounded metadata and native BF16 cache views."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import decode_indexer, pooled_qsa
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


def test_bf16_indexer_retains_fp32_head_reduction(tpu_info):
    config = streamindex._BlockedPrefillConfig.create(
        bq_sz=128,
        bkv_sz=256,
        num_q_heads=4,
        q_dtype=jnp.bfloat16,
        seq_batch_size=1,
    )
    assert config.enabled
    assert not config.bf16_reduce
    assert config.wb_dtype == jnp.float32


@pytest.mark.parametrize("tokens", [512, 8192])
@pytest.mark.parametrize(
    "dtype,packing,view",
    [
        (jnp.bfloat16, 2, None),
        (jnp.uint8, 4, "bfloat16"),
        (jnp.float8_e4m3fn, 4, "bfloat16"),
    ],
)
def test_blocked_prefill_traces_native_and_pooled_bf16(tokens, dtype, packing, view):
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    shape = jax.ShapeDtypeStruct
    with jax.sharding.use_abstract_mesh(mesh):
        output = jax.eval_shape(
            functools.partial(
                streamindex.streamindex_topk,
                k=512,
                compression_ratio=4,
                num_kv_pages_per_block=(8, 8, 8),
                num_queries_per_block=(1, 128, 128),
                chunk_tokens=512,
                components=qsa_indexing(3),
                return_scores=True,
            ),
            shape((tokens, 4, 256), jnp.bfloat16),
            shape((tokens, 4), jnp.float32),
            shape((640, 16, packing, 256), dtype),
            shape((3,), jnp.int32),
            shape((960,), jnp.int32),
            shape((4,), jnp.int32),
            shape((3,), jnp.int32),
        )
    assert all(x.shape == (tokens, 512) and x.dtype == jnp.int32 for x in output)


@pytest.mark.parametrize("context", [0, 1, 2047, 2048, 262144 - 8192])
def test_completed_groups_for_full_prefill_with_idle_request(context):
    # An empty request between two live requests must not hide the last one.
    starts = np.array([0, 4096, 4096, 8191], np.int32)
    seq_lens = np.array([context + 4096, 0, context + 4095], np.int32)
    expected = np.concatenate(
        ((context + np.arange(1, 4097)) // 4, (context + np.arange(1, 4096)) // 4, [0])
    )
    actual = streamindex._effective_row_lengths(
        jnp.asarray(seq_lens),
        jnp.asarray(starts),
        jnp.array([0, 0, 3]),
        num_tokens=8192,
        num_positions=65536,
        compression_ratio=4,
        components=qsa_indexing(3),
    )
    np.testing.assert_array_equal(actual, expected)


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


def test_pooled_prefill_keeps_live_requests_after_empty_rows(monkeypatch):
    def transport(q, weights, cache, lengths, table, starts, distribution, **kwargs):
        assert kwargs["components"].schedule.causal_position_offset == 3
        np.testing.assert_array_equal(distribution, [0, 0, 3])
        return streamindex._effective_row_lengths(
            lengths,
            starts,
            distribution,
            len(q),
            65536,
            4,
            components=kwargs["components"],
        )

    monkeypatch.setattr(decode_indexer, "streamindex_topk", transport)
    actual = pooled_qsa.select_blocks(
        jnp.zeros((8, 4, 128), jnp.bfloat16),
        jnp.zeros((3, 128, 1, 4, 256), jnp.float8_e4m3fn),
        jnp.zeros(8, jnp.int32),
        jnp.zeros((3, 1), jnp.int32),
        jnp.array([0, 0, 0, 2, 2, 2, 2, 2]),
        jnp.array([0, 3, 3, 8]),
        jnp.array([0, 0, 2048]),
        ratio=4,
        block_topk=512,
    )
    np.testing.assert_array_equal(actual, [0, 0, 0, 512, 512, 512, 513, 513])


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


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [512, 8192])
@pytest.mark.parametrize("carrier", ["uint8", "float8_e4m3fn"])
def test_blocked_prefill_bf16_pool_view_matches_native_cache(tokens, carrier):
    rng = np.random.default_rng(1641)
    query = jnp.asarray(rng.normal(size=(tokens, 4, 256)), jnp.bfloat16)
    native = jnp.asarray(rng.normal(size=(640, 16, 2, 256)), jnp.bfloat16)
    starts = jnp.array([0, tokens // 2, tokens // 2, tokens], jnp.int32)
    lengths = jnp.array([tokens // 2, 0, 32769 + tokens // 2], jnp.int32)
    table = jnp.asarray(rng.permutation(640).reshape(2, 320)[[0, 0, 1]], jnp.int32)

    def encode(src, dst):
        dst.bitcast(jnp.bfloat16)[...] = src[...]

    packed = pl.pallas_call(
        encode,
        grid=(len(native),),
        in_specs=[pl.BlockSpec((1, 16, 2, 256), lambda p: (p, 0, 0, 0))],
        out_specs=pl.BlockSpec((1, 16, 4, 256), lambda p: (p, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(
            (len(native), 16, 4, 256), getattr(jnp, carrier)
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(native)
    before = np.asarray(packed).view(np.uint8).copy()

    def select(cache, **kwargs):
        return streamindex.streamindex_topk(
            query,
            jnp.full(query.shape[:2], 256**-0.5, jnp.float32),
            cache,
            lengths,
            table.reshape(-1),
            starts,
            jnp.array([0, 0, 3]),
            k=512,
            compression_ratio=4,
            num_kv_pages_per_block=(8, 8, 8),
            num_queries_per_block=(1, 128, 128),
            chunk_tokens=512,
            components=qsa_indexing(3),
            return_scores=True,
            **kwargs,
        )

    expected = select(native)
    actual = select(packed)
    for wanted, got in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
    visible = np.concatenate(
        (
            np.arange(1, tokens // 2 + 1) // 4,
            (32769 + np.arange(1, tokens // 2 + 1)) // 4,
        )
    )
    ids = np.asarray(actual[0])
    assert np.all((ids == -1) | ((ids >= 0) & (ids < visible[:, None])))
    np.testing.assert_array_equal(np.sum(ids >= 0, axis=1), np.minimum(visible, 512))
    np.testing.assert_array_equal(np.asarray(packed).view(np.uint8), before)
