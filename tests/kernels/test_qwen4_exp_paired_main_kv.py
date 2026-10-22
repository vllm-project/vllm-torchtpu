# SPDX-License-Identifier: Apache-2.0
"""Paired main KV pages: two layers' single FP8 KV heads share each row.

Layer slot ``s`` stores K and V at head slots ``2s`` and ``2s + 1``. The
writers and the sparse prefill run in Pallas interpret mode on CPU (the
legacy ``interpret=True`` evaluator, which bitcasts refs low byte first like
the TPU kernels). Batched RPA cannot run in either interpret mode, so decode
checks the head-offset plumbing against a reference RPA.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
    block_sparse_gqa_prefill,
)

PAGES, PAGE_TOKENS, WIDTH = 6, 512, 256
RATIO = 4


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def fp8(rng, shape, scale=1.0):
    return jnp.asarray(rng.normal(0, scale, shape), jnp.float8_e4m3fn)


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def rows_for(rng, tokens):
    def rand():
        return jnp.asarray(rng.normal(0, 1, (tokens, 1, WIDTH)), jnp.bfloat16)

    return qsa_rpa.pack_rows(rand(), rand(), fp8=True, k_scale=0.375, v_scale=2.5)


def step(starts, context, table):
    """Per-token pages, logical positions and validity of a ragged step."""
    starts = np.asarray(starts)
    tokens = starts[-1] if len(starts) else 0
    padded = -(-max(tokens, 1) // 8) * 8 if tokens % 128 else tokens
    page = np.full(padded, PAGES, np.int32)
    logical = np.zeros(padded, np.int32)
    for r in range(len(context)):
        positions = context[r] + np.arange(starts[r + 1] - starts[r])
        logical[starts[r] : starts[r + 1]] = positions
        page[starts[r] : starts[r + 1]] = np.asarray(table)[r, positions // PAGE_TOKENS]
    valid = np.arange(padded) < tokens
    return jnp.asarray(page), jnp.asarray(logical), jnp.asarray(valid)


WRITE_CASES = {
    # 128-row bulk tiles, including one that crosses a page boundary.
    "tiled_prefill": ([0, 128, 384], [0, 448]),
    # Unaligned spans: every tile takes the per-row read/merge/write path.
    "edges": ([0, 37, 100, 256], [5, 509, 130]),
    # A small decode step takes the gather/scatter path.
    "decode": ([0, 1, 2, 3], [3, 1022, 0]),
}


@pytest.mark.parametrize("case", sorted(WRITE_CASES))
def test_paired_writes_match_unpacked_rows_and_keep_partner(interpret, case):
    starts, context = WRITE_CASES[case]
    table = np.asarray([[3, 5], [1, 4], [2, 0]], np.int32)[: len(context)]
    page, logical, valid = step(starts, context, table)
    rng = np.random.default_rng(len(case))
    initial = fp8(rng, (PAGES, PAGE_TOKENS, 1, 4, WIDTH))
    layers = [rows_for(rng, len(page)) for _ in range(2)]
    write = jax.jit(pooled_qsa.write_main_cache, static_argnames=("slot",))

    paired = initial
    for slot, rows in enumerate(layers):
        paired = write(paired, rows, page, logical, valid, slot=slot)

    touched = np.zeros((PAGES, PAGE_TOKENS), bool)
    v = np.asarray(valid)
    touched[np.asarray(page)[v], np.asarray(logical)[v] % PAGE_TOKENS] = True
    expected = raw(initial).copy()
    for slot, rows in enumerate(layers):
        unpacked = raw(write(initial, rows, page, logical, valid))
        # The unpacked writer fills K, V and two zero padding slots.
        assert not unpacked[touched][:, :, 2:].any()
        np.testing.assert_array_equal(
            unpacked[~touched], raw(initial)[~touched], "unpacked touched extra rows"
        )
        part = slice(2 * slot, 2 * slot + 2)
        expected[:, :, :, part][touched] = unpacked[:, :, :, :2][touched]
    np.testing.assert_array_equal(raw(paired), expected)


def test_paired_writes_keep_last_duplicate(interpret):
    """A token written twice in one tile keeps the later row, as unpacked."""
    rng = np.random.default_rng(7)
    initial = fp8(rng, (PAGES, PAGE_TOKENS, 1, 4, WIDTH))
    tokens = 128
    page = jnp.asarray(np.where(np.arange(tokens) < 100, 2, PAGES), jnp.int32)
    logical = jnp.asarray(np.arange(tokens) % 40 + 7, jnp.int32)
    valid = jnp.arange(tokens) < 100
    rows = rows_for(rng, tokens)
    write = jax.jit(pooled_qsa.write_main_cache, static_argnames=("slot",))
    unpacked = raw(write(initial, rows, page, logical, valid))
    paired = raw(write(initial, rows, page, logical, valid, slot=1))
    touched = np.zeros((PAGES, PAGE_TOKENS), bool)
    touched[2, 7:47] = True
    np.testing.assert_array_equal(
        paired[touched][..., 2:, :], unpacked[touched][..., :2, :]
    )
    np.testing.assert_array_equal(paired[~touched], raw(initial)[~touched])
    np.testing.assert_array_equal(paired[..., :2, :], raw(initial)[..., :2, :])


def test_paired_writer_rejects_non_fp8_rows():
    pool = jnp.zeros((2, 128, 1, 2, WIDTH), jnp.bfloat16)
    rows = jnp.zeros((8, 1, 2, WIDTH), jnp.bfloat16)
    index = jnp.zeros((8,), jnp.int32)
    with pytest.raises(ValueError, match="FP8 head-slot"):
        pooled_qsa.write_main_cache(pool, rows, index, index, index > 0, slot=1)


def paired_pool(rng):
    """(paired pool, layer A alone, layer B alone), unpacked-layout copies."""
    a = fp8(rng, (PAGES, PAGE_TOKENS, 1, 2, WIDTH), 0.5)
    b = fp8(rng, (PAGES, PAGE_TOKENS, 1, 2, WIDTH), 0.5)
    zero = jnp.zeros_like(a)
    pair = jnp.concatenate((a, b), axis=3)
    return pair, jnp.concatenate((a, zero), axis=3), jnp.concatenate((b, zero), axis=3)


def sparse_inputs(rng, tokens, heads=3):
    lengths = jnp.asarray(np.arange(tokens) * 13 + 200, jnp.int32)
    blocks = jnp.asarray(rng.integers(0, 256, (tokens, 32)), jnp.int32)
    blocks = jnp.where(blocks * RATIO + RATIO <= lengths[:, None], blocks, -1)
    table = jnp.tile(jnp.asarray([[4, 1]], jnp.int32), (tokens, 1))
    query = jnp.asarray(rng.normal(0, 1, (tokens, heads, WIDTH)), jnp.bfloat16)
    return query, blocks, lengths, table


@pytest.mark.parametrize("query_block", [32, 64])
def test_sparse_prefill_reads_each_layers_head_slots(interpret, query_block):
    rng = np.random.default_rng(query_block)
    pair, alone_a, alone_b = paired_pool(rng)
    query, blocks, lengths, table = sparse_inputs(rng, 64)
    prefill = functools.partial(
        block_sparse_gqa_prefill,
        ratio=RATIO,
        kv_heads=1,
        query_block=query_block,
        kv_tile=128,
        k_scale=0.375,
        v_scale=2.5,
        bf16_io=True,
    )
    for offset, alone in ((0, alone_a), (1, alone_b)):
        expected = prefill(query, blocks, lengths, table, alone)
        actual = prefill(query, blocks, lengths, table, pair, kv_head_offset=offset)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    with pytest.raises(ValueError, match="head offset"):
        prefill(query, blocks, lengths, table, pair, kv_head_offset=2)


def test_sparse_prefill_wrapper_forwards_offset(interpret):
    rng = np.random.default_rng(3)
    pair, _, alone_b = paired_pool(rng)
    query, blocks, lengths, table = sparse_inputs(rng, 40)
    gate = jnp.asarray(rng.normal(0, 1, query.shape), jnp.bfloat16)
    # 40 tokens in two requests: the padded (unaligned) path.
    starts = jnp.asarray([0, 17, 40], jnp.int32)
    attend = functools.partial(
        block_sparse_gqa_attention, kv_heads=1, ratio=RATIO, k_scale=0.375
    )
    expected, _ = attend(query, gate, blocks, lengths, table, alone_b, starts)
    actual, cache = attend(
        query, gate, blocks, lengths, table, pair, starts, kv_head_offset=1
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert cache is pair


def reference_rpa(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    sm_scale,
    k_scale,
    v_scale,
    out_dtype,
    kv_layout,
    skip_kv_update,
    **_,
):
    """Decode-only HEAD_ALONG_SUBLANE RPA: combined head c = 2h (K), 2h + 1 (V)."""
    assert skip_kv_update
    kv_heads = keys.shape[1]
    per_kv = queries.shape[1] // kv_heads
    pages_per_seq = len(page_indices) // len(kv_lens)
    page_size = kv_cache.shape[1]
    combined = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
    assert combined.shape[2] >= 2 * kv_heads
    out = jnp.zeros(queries.shape, jnp.float32)
    for seq in range(len(kv_lens)):
        pages = page_indices[seq * pages_per_seq : (seq + 1) * pages_per_seq]
        cache = combined[pages].reshape(
            pages_per_seq * page_size, -1, combined.shape[-1]
        )
        key = cache[:, 0 : 2 * kv_heads : 2].astype(jnp.float32) * k_scale
        value = cache[:, 1 : 2 * kv_heads : 2].astype(jnp.float32) * v_scale
        visible = jnp.arange(len(cache)) < kv_lens[seq]
        row = jnp.minimum(cu_q_lens[seq], len(queries) - 1)
        q = queries[row].astype(jnp.float32).reshape(kv_heads, per_kv, -1)
        score = jnp.einsum("hqd,thd->hqt", q, key) * sm_scale
        score = jnp.where(visible[None, None], score, -jnp.inf)
        prob = jax.nn.softmax(score, axis=-1)
        prob = jnp.where(jnp.isnan(prob), 0, prob)
        result = jnp.einsum("hqt,thd->hqd", prob, value).reshape(-1, q.shape[-1])
        live = cu_q_lens[seq + 1] > cu_q_lens[seq]
        out = out.at[row].set(jnp.where(live, result, out[row]))
    return out.astype(out_dtype), kv_cache


def test_decode_rpa_reads_second_layer_through_head_offset(interpret, monkeypatch):
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    calls = []

    def recording(*args, **kwargs):
        calls.append((args[0].shape, args[1].shape))
        return reference_rpa(*args, **kwargs)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", recording)
    rng = np.random.default_rng(11)
    pair, alone_a, alone_b = paired_pool(rng)
    batch = 4
    q = jnp.asarray(rng.normal(0, 1, (batch, 3, WIDTH)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1, (batch, 3, WIDTH)), jnp.bfloat16)
    lengths = jnp.asarray([700, 33, 0, 1000], jnp.int32)
    blocks = jnp.asarray(rng.integers(0, 160, (batch, 16)), jnp.int32)
    blocks = jnp.where(blocks * RATIO + RATIO <= lengths[:, None], blocks, -1)
    table = jnp.asarray([[1, 3], [4, 0], [0, 0], [5, 2]], jnp.int32)
    attend = functools.partial(
        qsa_rpa.rpa_attention, kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5
    )
    expected_a, _ = attend(q, gate, blocks, lengths, table, alone_a)
    expected_b, _ = attend(q, gate, blocks, lengths, table, alone_b)
    actual_a, _ = attend(q, gate, blocks, lengths, table, pair)
    actual_b, cache = attend(q, gate, blocks, lengths, table, pair, kv_head_offset=1)
    np.testing.assert_array_equal(np.asarray(actual_a), np.asarray(expected_a))
    np.testing.assert_array_equal(np.asarray(actual_b), np.asarray(expected_b))
    assert not np.array_equal(np.asarray(expected_a), np.asarray(expected_b))
    assert cache.shape == pair.shape
    # The second layer presents two KV heads and three leading zero queries.
    assert calls[-1] == ((batch, 6, WIDTH), (batch, 2, WIDTH))
    assert calls[0] == ((batch, 3, WIDTH), (batch, 1, WIDTH))
