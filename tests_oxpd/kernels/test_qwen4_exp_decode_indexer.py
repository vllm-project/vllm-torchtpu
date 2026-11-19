# SPDX-License-Identifier: Apache-2.0
"""BF16 decode selection: page ownership, completed groups and exact ties."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import (
    select_decode_blocks,
)
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing


def _inputs(heads=4, batch=3):
    rng = np.random.default_rng(51)
    query = jnp.asarray(rng.normal(size=(batch, heads, 128)), jnp.bfloat16)
    cache = jnp.asarray(rng.normal(size=(batch * 24, 32, 1, 128)), jnp.bfloat16)
    table = jnp.asarray(rng.permutation(batch * 24).reshape(batch, 24), jnp.int32)
    lengths = jnp.asarray(np.resize([0, 4 * 517 + 3, 4 * 768], batch), jnp.int32)
    return query, cache, lengths, table


def _reference(query, cache, lengths, table):
    query, cache = np.asarray(query, np.float32), np.asarray(cache, np.float32)
    keys = cache[np.asarray(table), :, 0].reshape(len(query), -1, 128)
    scores = np.maximum(np.einsum("bhd,bgd->bhg", query, keys), 0).sum(1)
    scores /= np.sqrt(np.float32(128))
    return np.where(
        np.arange(keys.shape[1]) < np.asarray(lengths)[:, None] // 4, scores, -np.inf
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_decode_blocks_match_exact_topk_set(ties, heads):
    args = list(_inputs(heads))
    if ties:
        args[0] = jnp.zeros_like(args[0])
    actual = np.asarray(
        select_decode_blocks(*args, jnp.arange(len(args[0]), dtype=jnp.int32))
    )
    scores = _reference(*args)
    for row, length in enumerate(np.asarray(args[2])):
        count = min(512, length // 4)
        wanted = np.argsort(-scores[row], kind="stable")[:count]
        np.testing.assert_array_equal(np.sort(actual[row, :count]), np.sort(wanted))
        np.testing.assert_array_equal(actual[row, count:], -1)
    assert actual.dtype == np.int32


def test_decode_indexer_rejects_quantized_index_cache():
    query, cache, lengths, table = _inputs()
    with pytest.raises(ValueError, match="BF16"):
        select_decode_blocks(
            query,
            cache.astype(jnp.float8_e4m3fn),
            lengths,
            table,
            jnp.arange(len(query), dtype=jnp.int32),
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", [7, 8, 9])
@pytest.mark.parametrize("ties", [False, True])
def test_decode_batch_boundary_preserves_idle_tail_and_page_ownership(batch, ties):
    """A full decode tile and its remainder must retain each row's causal limit."""
    query, cache, _, table = _inputs(batch=batch)
    lengths = jnp.array([0, 1, 2047, 2048, 2049, 2500, 3071, 3072, 2052][:batch])
    if ties:
        query = jnp.zeros_like(query)
    actual = np.asarray(
        select_decode_blocks(
            query, cache, lengths, table, jnp.arange(len(query), dtype=jnp.int32)
        )
    )
    scores = _reference(query, cache, lengths, table)
    for row, length in enumerate(np.asarray(lengths)):
        count = min(512, length // 4)
        wanted = np.argsort(-scores[row], kind="stable")[:count]
        np.testing.assert_array_equal(np.sort(actual[row, :count]), np.sort(wanted))
        np.testing.assert_array_equal(actual[row, count:], -1)


@pytest.mark.tpu_test
def test_streamindex_bf16_cache_preserves_scores_and_selection():
    """Reusing FP8 unpacking on BF16 bytes corrupts QSA retrieval scores."""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

    q, cache, lengths, table = _inputs()
    ids, scores = streamindex_topk(
        q,
        jnp.full(q.shape[:2], 128**-0.5, jnp.float32),
        cache.reshape(72, 16, 2, 128),
        lengths,
        table.reshape(-1),
        jnp.arange(4, dtype=jnp.int32),
        jnp.full((3,), 3, jnp.int32),
        components=qsa_indexing(),
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=8,
        num_queries_per_block=1,
        return_scores=True,
    )
    expected = _reference(q, cache, lengths, table)
    for row, length in enumerate(np.asarray(lengths)):
        count = min(512, length // 4)
        selected = np.asarray(ids[row, :count])
        wanted = np.argsort(-expected[row], kind="stable")[:count]
        np.testing.assert_array_equal(np.sort(selected), np.sort(wanted))
        np.testing.assert_allclose(
            np.asarray(scores[row, :count]).view(np.float32),
            expected[row, selected],
            rtol=2e-6,
            atol=2e-5,
        )
        np.testing.assert_array_equal(np.asarray(ids[row, count:]), -1)


@pytest.mark.tpu_test
@pytest.mark.parametrize("carrier", ["uint8", "float8_e4m3fn"])
@pytest.mark.parametrize("ties", [False, True])
def test_streamindex_bf16_pool_view_matches_native_cache(carrier, ties):
    """An FP8 pool must not quantize or numerically convert BF16 index keys.

    Noncontiguous page ownership, idle rows and decode batch remainders must
    retain exactly the native BF16 scores and winners, without modifying bytes.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

    query, cache, lengths, table = _inputs(batch=9)
    if ties:
        query = jnp.zeros_like(query)
    native = cache.reshape(cache.shape[0], 16, 2, 128)

    def encode(src, dst):
        dst.bitcast(jnp.bfloat16)[...] = src[...]

    packed = pl.pallas_call(
        encode,
        grid=(len(cache),),
        in_specs=[pl.BlockSpec((1, 16, 2, 128), lambda p: (p, 0, 0, 0))],
        out_specs=pl.BlockSpec((1, 16, 4, 128), lambda p: (p, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((len(cache), 16, 4, 128), getattr(jnp, carrier)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(native)
    before = np.asarray(packed).view(np.uint8).copy()

    def select(keys, **kwargs):
        return streamindex_topk(
            query,
            jnp.full(query.shape[:2], 128**-0.5, jnp.float32),
            keys,
            lengths,
            table.reshape(-1),
            jnp.arange(10, dtype=jnp.int32),
            jnp.full((3,), 9, jnp.int32),
            k=512,
            compression_ratio=4,
            num_kv_pages_per_block=8,
            num_queries_per_block=1,
            decode_req_batch_size=8,
            return_scores=True,
            components=qsa_indexing(),
            **kwargs,
        )

    expected = select(native)
    actual = select(packed)
    for wanted, got in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
    np.testing.assert_array_equal(np.asarray(packed).view(np.uint8), before)


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", [8, 9, 128])
@pytest.mark.parametrize("ties", [False, True])
def test_pooled_decode_uses_logical_width_without_changing_selection(batch, ties):
    """Real owner geometry and mixed decode contexts retain padded-path winners."""
    import jax

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import select_blocks

    rng = np.random.default_rng(53)
    pages = batch * 2 + 1
    query = jnp.asarray(rng.normal(size=(batch, 4, 128)), jnp.bfloat16)
    if ties:
        query = jnp.zeros_like(query)
    keys = np.asarray(
        jnp.asarray(rng.normal(size=(pages, 208, 2, 128)), jnp.bfloat16)
    ).view(np.uint16)
    raw = np.zeros((pages, 1664, 1, 4, 256), np.uint8)
    raw[:, :208, 0, 0::2, :128] = keys & 255
    raw[:, :208, 0, 1::2, :128] = keys >> 8
    pool = jax.device_put(raw.view(np.dtype(jnp.float8_e4m3fn)))
    owner_pages = rng.permutation(np.arange(1, pages)).reshape(batch, 2)
    table = jnp.asarray(np.tile(owner_pages, (1, 3)), jnp.int32)
    req = jnp.asarray(rng.permutation(batch), jnp.int32)
    lengths = np.resize([0, 1, 3, 4, 2047, 2048, 2049, 8192, 9216], batch)
    if batch == 128:
        lengths = 8286 + np.arange(batch) % 51
    lengths = jnp.asarray(lengths, jnp.int32)
    expected = streamindex_topk(
        jnp.pad(query, ((0, 0), (0, 0), (0, 128))),
        jnp.full(query.shape[:2], 128**-0.5, jnp.float32),
        pool.reshape(-1, 16, 4, 256),
        lengths,
        (table[req, :, None] * 104 + jnp.arange(13)).reshape(-1),
        jnp.arange(batch + 1, dtype=jnp.int32),
        jnp.full((3,), batch, jnp.int32),
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=8,
        num_queries_per_block=1,
        decode_req_batch_size=8,
        enable_early_exit=True,
        components=qsa_indexing(),
    )
    actual = jax.jit(select_blocks, static_argnames=("ratio", "block_topk"))(
        query,
        pool,
        lengths,
        table,
        req,
        jnp.arange(batch + 1, dtype=jnp.int32),
        lengths - 1,
        ratio=4,
        block_topk=512,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(pool).view(np.uint8), raw)


@pytest.mark.tpu_test
@pytest.mark.parametrize("carrier", [jnp.uint8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_bf16_view_wider_storage_preserves_score_bits_and_owner(carrier, ties, prefill):
    """Unused finite features must not affect retrieval or mutate the carrier."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

    rng = np.random.default_rng(53)
    batch, pages, width = (3 if prefill else 9), 24, 256
    query_length = 128 if prefill else 1
    query = jnp.asarray(rng.normal(size=(batch * query_length, 4, 128)), jnp.bfloat16)
    if ties:
        query = jnp.zeros_like(query)
    keys = jnp.asarray(rng.normal(size=(batch * pages, 16, 2, width)), jnp.bfloat16)
    keys = keys.at[..., 128:].set(jnp.finfo(jnp.bfloat16).max)

    def encode(src, dst):
        dst.bitcast(jnp.bfloat16)[...] = src[...]

    packed = pl.pallas_call(
        encode,
        grid=(len(keys),),
        in_specs=[pl.BlockSpec((1, 16, 2, width), lambda p: (p, 0, 0, 0))],
        out_specs=pl.BlockSpec((1, 16, 4, width), lambda p: (p, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((len(keys), 16, 4, width), carrier),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(keys)
    before = np.asarray(packed).view(np.uint8).copy()
    table = jnp.asarray(rng.permutation(batch * pages), jnp.int32)
    lengths = jnp.array(
        [3072, 3071, 0] if prefill else [0, 1, 3, 4, 2047, 2048, 2049, 3071, 3072],
        jnp.int32,
    )

    def select(q):
        return streamindex_topk(
            q,
            jnp.full(q.shape[:2], 128**-0.5, jnp.float32),
            packed,
            lengths,
            table,
            jnp.arange(batch + 1, dtype=jnp.int32) * query_length,
            jnp.array([0, 0, batch] if prefill else [batch] * 3, jnp.int32),
            k=512,
            compression_ratio=4,
            num_kv_pages_per_block=8,
            num_queries_per_block=query_length,
            decode_req_batch_size=8,
            components=qsa_indexing(),
            return_scores=True,
        )

    expected = select(jnp.pad(query, ((0, 0), (0, 0), (0, 128))))
    actual = select(query)
    for wanted, got in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
    np.testing.assert_array_equal(np.asarray(packed).view(np.uint8), before)
