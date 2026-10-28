# SPDX-License-Identifier: Apache-2.0
"""Shared-stream decode indexer: identical selections to per-token decode.

With ``VLLM_TPU_QSA_DECODE_QUERY_BLOCK`` set, the decode/verify tokens of a
request share one stream over its compressed keys. The real StreamIndex
scoring kernel runs in TPU interpret mode on CPU; a JAX top-k stands in for
the SparseCore selector, so both paths are compared on identical score bits.
"""

import functools
import importlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

streamindex = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)
bref_override = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.bref_override"
)

RATIO = 4
BLOCK_TOPK = 16
POOL_PAGES, TOKENS_PER_PAGE, WIDTH = 24, 512, 256
# One table column holds 128 compressed groups (512 tokens).
TABLE_COLUMNS = 8


class ReferenceTopK:
    """JAX stand-in for the SparseCore selector (lowest IDs win ties)."""

    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None],
            scores,
            -jnp.inf,
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ReferenceTopK)


def _copy_in_whole_pages(self, src_ref, grid_indices):
    """Interpret-mode page DMA: same pages, without the flattened HBM view.

    Interpret mode cannot DMA from a reshaped ref. Every page index the kernel
    reads is a valid page here, so indexing whole pages moves the same rows.
    """
    cache_kv_hbm, page_indices_ref, meta_ref = src_ref
    p_id = grid_indices[0] + bref_override._step_offset(meta_ref, self.chunk_idx)
    seq = meta_ref.batch_tile_idx[p_id]
    kv_p_start = meta_ref.bkv_idx[p_id] * self.bkv_p
    slot = self.current_copy_in_slot
    vmem_dst = self.window_ref.at[slot]
    for batch_idx in range(self.seq_batch_size):
        for i in range(self.bkv_p):
            page = self.page_id(page_indices_ref, seq + batch_idx, kv_p_start, i)
            page = jnp.minimum(page, cache_kv_hbm.shape[0] - 1)
            pltpu.make_async_copy(
                cache_kv_hbm.at[page],
                vmem_dst.at[
                    batch_idx,
                    pl.ds(
                        i * self.page_size_per_kv_packing,
                        self.page_size_per_kv_packing,
                    ),
                ],
                self.sem_recvs.at[slot],
            ).start()


@pytest.fixture
def interpret(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        streamindex.pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    monkeypatch.setattr(
        bref_override.StreamIndexKVBufferedRef, "copy_in", _copy_in_whole_pages
    )


def capture_scores(monkeypatch):
    """Route selection through the reference top-k and record winner scores."""
    calls = []

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, return_scores=True)
        ids, bits = streamindex.streamindex_topk(
            *args, topk_backend=ReferenceTopK(), **kwargs
        )
        calls.append((kwargs["num_queries_per_block"], bits))
        return ids

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    return calls


def ragged_step(q_lens, context, tokens):
    """Per-token request rows and causal lengths of a ragged step."""
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    req = np.zeros(tokens, np.int32)
    lengths = np.zeros(tokens, np.int32)
    for r, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        req[begin:end] = r
        lengths[begin:end] = context[r] + np.arange(1, end - begin + 1)
    return starts, req, lengths


def inputs(seed, q_lens, context, tokens, data="normal"):
    """Gaussian data, or small integers whose q @ k is exact in any order.

    Integer data isolates the scoring expression from matmul blocking (XLA:CPU
    may block a GEMM by its row count; the MXU computes rows independently)
    and makes many exact ties, which must break the same way on both paths.
    "dyadic" also makes the head weight 64 ** -0.5 = 1/8 exact, so every
    score is exact whatever the reduction order or fusion.
    """
    rng = np.random.default_rng(seed)
    starts, req, lengths = ragged_step(q_lens, context, tokens)

    def sample(shape):
        if data in ("integer", "dyadic"):
            return jnp.asarray(rng.integers(-3, 4, size=shape), jnp.bfloat16)
        return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

    pool = sample((POOL_PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH))
    table = np.stack(
        [rng.permutation(POOL_PAGES)[:TABLE_COLUMNS] for _ in range(len(context))]
    )
    query = sample((tokens, 4, 64 if data == "dyadic" else 128))
    return dict(
        query=query,
        pool=pool,
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table, jnp.int32),
        req=jnp.asarray(req),
        starts=jnp.asarray(starts),
        context=jnp.asarray(context, jnp.int32),
    )


def select(args):
    return np.asarray(
        pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK)
    )


# (q_lens per request row, context per request row, padded tokens). The decode
# path requires tokens <= request rows. Contexts cover the early-exit bound,
# partial ring tails (context + i not a multiple of 4), several KV blocks and
# table columns, and idle rows between and after live requests.
CASES = {
    "verify_k3": (
        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3001, 2048, 1203] + [0] * 9,
        12,
    ),
    "ragged_1_to_4": (
        [1, 2, 0, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2999, 1022, 7, 4093 - 3, 517, 3500] + [0] * 10,
        16,
    ),
    "decode_prefix": (
        [1] * 9 + [0] * 3,
        [4000, 3, 64, 65, 66, 67, 999, 2048, 3001, 0, 0, 0],
        12,
    ),
    "short_and_long": ([4, 3, 0, 0, 0, 0, 0, 0], [5, 3000, 0, 0, 0, 0, 0, 0], 8),
}


def ulp_distance(a, b):
    """Max distance in float32 ULPs between same-signed score bit patterns."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    finite = (a != streamindex.BITS_NEG_INF) & (b != streamindex.BITS_NEG_INF)
    np.testing.assert_array_equal(a[~finite], b[~finite])
    return int(np.max(np.abs(a - b)[finite], initial=0))


# Integer data at the decode pass's 256-key block is bit-exact on XLA:CPU too.
# A wider KV block changes how XLA:CPU fuses the (identical) per-token
# reduction, so there the exact check uses dyadic data instead.
@pytest.mark.parametrize(
    "query_block,kv_block,data",
    [
        (4, 256, "integer"),
        (8, 256, "integer"),
        (4, 1024, "dyadic"),
        (8, 1024, "dyadic"),
        (4, 1024, "normal"),
        (8, 256, "normal"),
    ],
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_shared_stream_matches_per_token_scores_and_selection(
    monkeypatch, interpret, case, query_block, kv_block, data
):
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, data)
    calls = capture_scores(monkeypatch)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)

    monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    expected = select(args)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    actual = select(args)

    (bq_ref, bits_ref), (bq_new, bits_new) = calls
    assert bq_ref == 1 and bq_new == (1, query_block, query_block)
    np.testing.assert_array_equal(actual, expected)
    if data != "normal" or jax.default_backend() == "tpu":
        np.testing.assert_array_equal(np.asarray(bits_new), np.asarray(bits_ref))
    else:
        # XLA:CPU rounds an M=4*bq GEMM differently from M=4 in the last bits.
        assert ulp_distance(bits_new, bits_ref) <= 4

    # The selections are the exact completed-group top-k of each token.
    lengths = np.asarray(args["lengths"])
    for row, length in enumerate(lengths):
        count = min(BLOCK_TOPK, length // RATIO)
        live = actual[row][actual[row] >= 0]
        assert len(live) == count and np.all(live < length // RATIO)
        assert len(np.unique(live)) == count
    assert np.any(lengths // RATIO > BLOCK_TOPK)


@pytest.mark.parametrize("chunk", [4, 8])
def test_token_windows_cut_requests_and_keep_positions(monkeypatch, interpret, chunk):
    """Windows smaller than a step split requests across scan iterations."""
    q_lens, context, tokens = CASES["ragged_1_to_4"]
    args = inputs(5, q_lens, context, tokens)
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: chunk)
    monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    expected = select(args)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "256")
    np.testing.assert_array_equal(select(args), expected)


def test_request_window_trims_cut_requests():
    # Requests own tokens [0,3) [3,3) [3,7) [7,8); window [2, 6).
    starts = jnp.array([0, 3, 3, 7, 8], jnp.int32)
    context = jnp.array([10, 0, 20, 30], jnp.int32)
    ids, seq_lens, cu, dist = pooled_qsa._request_window(2, 4, starts, context)
    np.testing.assert_array_equal(ids[:2], [0, 2])
    # Request 0 keeps its last token (position 12); request 2 keeps 3 of its
    # 4 tokens and drops the one cut from its end (positions 20..22).
    np.testing.assert_array_equal(seq_lens, [13, 23, 0, 0])
    np.testing.assert_array_equal(cu, [0, 1, 4, 4, 4])
    np.testing.assert_array_equal(dist, [1, 1, 2])
    # A window past the last token owns no request.
    ids, seq_lens, cu, dist = pooled_qsa._request_window(8, 4, starts, context)
    np.testing.assert_array_equal(seq_lens, 0)
    np.testing.assert_array_equal(cu, 0)
    np.testing.assert_array_equal(dist, [0, 0, 0])


def test_rejects_unaligned_kv_block(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "96")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    args = inputs(0, [1, 0], [8, 0], 2)
    with pytest.raises(ValueError, match="DECODE_KV_BLOCK"):
        select(args)


def test_per_token_scores_disable_blocked_prefill_layout():
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    kwargs = dict(bq_sz=32, bkv_sz=256, num_q_heads=4, q_dtype=jnp.bfloat16)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pltpu, "get_tpu_info", lambda: info)
        blocked = streamindex._BlockedPrefillConfig.create(seq_batch_size=1, **kwargs)
        per_token = streamindex._BlockedPrefillConfig.create(
            seq_batch_size=1, per_token_scores=True, **kwargs
        )
    assert blocked.enabled and not per_token.enabled
    assert per_token.score_pitch == 256 // 128


@pytest.mark.tpu_test
@pytest.mark.parametrize("query_block,kv_block", [(4, 256), (4, 1024), (8, 1024)])
def test_tpu_shared_stream_is_bit_exact_with_sparsecore(
    monkeypatch, query_block, kv_block
):
    """Mosaic kernel and SparseCore top-k: same winners and score bits."""
    calls = []

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, return_scores=True)
        ids, bits = streamindex.streamindex_topk(*args, **kwargs)
        calls.append((ids, bits))
        return ids

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    q_lens = [4] * 8 + [1, 2, 3] + [0] * 29
    context = [4093, 2048, 3001, 1203, 4000, 517, 3500, 999, 3998, 2047, 1]
    args = inputs(7, q_lens, context + [0] * 29, 40)
    monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    select(args)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    select(args)
    (ids_ref, bits_ref), (ids_new, bits_new) = calls
    order_ref = np.argsort(np.asarray(ids_ref), axis=1)
    order_new = np.argsort(np.asarray(ids_new), axis=1)
    np.testing.assert_array_equal(
        np.take_along_axis(np.asarray(ids_new), order_new, 1),
        np.take_along_axis(np.asarray(ids_ref), order_ref, 1),
    )
    np.testing.assert_array_equal(
        np.take_along_axis(np.asarray(bits_new), order_new, 1),
        np.take_along_axis(np.asarray(bits_ref), order_ref, 1),
    )


@pytest.mark.parametrize("query_block,kv_block", [(4, 1024), (8, 256)])
def test_shared_stream_lowers_to_one_query_block_per_request(
    monkeypatch, query_block, kv_block
):
    """The flag replaces per-token streams with a ragged bq pass on TPU7x."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    shape = jax.ShapeDtypeStruct
    args = (
        shape((32, 4, 128), jnp.bfloat16),
        shape((64, 512, 1, 2, 256), jnp.bfloat16),
        shape((32,), jnp.int32),
        shape((64, 16), jnp.int32),
        shape((32,), jnp.int32),
        shape((65,), jnp.int32),
        shape((64,), jnp.int32),
    )
    select = jax.jit(
        functools.partial(pooled_qsa.select_blocks, ratio=RATIO, block_topk=512)
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        text = select.trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert f"StreamIdxTC-m-bq_{query_block}-bkvp_{kv_block // 32}" in text
    assert "StreamIdxTC-d-bq_1-bkvp_8" in text


@pytest.mark.parametrize("tokens,expect", [(32, 32), (64, 8)])
def test_single_token_block_only_on_single_token_steps(monkeypatch, tokens, expect):
    """VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK sizes the one-token decode pass on
    steps of at most one token per request; verify steps keep 256 keys."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "1024")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", "1024")
    shape = jax.ShapeDtypeStruct
    args = (
        shape((tokens, 4, 128), jnp.bfloat16),
        shape((64, 512, 1, 2, 256), jnp.bfloat16),
        shape((tokens,), jnp.int32),
        shape((32, 16), jnp.int32),
        shape((tokens,), jnp.int32),
        shape((33,), jnp.int32),
        shape((32,), jnp.int32),
    )
    select = jax.jit(
        functools.partial(
            pooled_qsa.select_blocks, ratio=RATIO, block_topk=512, max_query_len=4
        )
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        text = select.trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert f"StreamIdxTC-d-bq_1-bkvp_{expect}" in text


@pytest.mark.parametrize(
    "flags",
    [
        ("VLLM_TPU_QSA_INDEXER_HOIST_META",),
        ("VLLM_TPU_QSA_INDEXER_MIXED_ONLY",),
        ("VLLM_TPU_QSA_INDEXER_HOIST_META", "VLLM_TPU_QSA_INDEXER_MIXED_ONLY"),
    ],
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_indexer_step_flags_keep_the_selection(monkeypatch, interpret, case, flags):
    """Hoisting the step metadata and scoring every request in the mixed pass
    select exactly the blocks the default shared stream selects."""
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    # The reference top-k runs on CPU; unlike capture_scores, keep the
    # early-exit cond the hoist moves the step metadata out of.
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "256")
    expected = select(args)
    for flag in flags:
        monkeypatch.setenv(flag, "1")
    np.testing.assert_array_equal(select(args), expected)


@pytest.mark.parametrize("slot", [0, 1])
def test_pool_page_ids_match_the_expanded_table(slot):
    """Every kernel page, including reads past the last sequence's last page
    (clamped), resolves to the expanded table's page id."""
    seqs, entries, expansion, stride = 3, 5, 13, 104
    table = jnp.asarray(
        np.random.default_rng(0).permutation(64)[: seqs * entries], jnp.int32
    )
    expanded = (
        table[:, None] * stride + slot * expansion + jnp.arange(expansion)
    ).reshape(-1)
    pages_per_seq = entries * expansion

    ref = types.SimpleNamespace(
        pages_per_seq=pages_per_seq,
        table_expansion=(expansion, stride, slot * expansion),
    )
    block = 8  # not a multiple of the expansion: blocks start mid-entry
    for seq in range(seqs + 1):
        for start in range(0, pages_per_seq + 2 * expansion, block):
            for i in range(block):
                k = start + i
                want = expanded[min(seq * pages_per_seq + k, len(expanded) - 1)]
                got = bref_override.StreamIndexKVBufferedRef.page_id(
                    ref, table, jnp.int32(seq), jnp.int32(start), i
                )
                assert int(got) == int(want), (seq, k)


@pytest.mark.parametrize("chunk", [1 << 20, 4])
@pytest.mark.parametrize("slot", [0, 1, 7])
@pytest.mark.parametrize("kv_block", [256, 1024])
@pytest.mark.parametrize("case", sorted(CASES))
def test_pool_page_table_keeps_the_selection(
    monkeypatch, interpret, case, kv_block, slot, chunk
):
    """VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: the KV reader expanding pool-page
    entries reads the same kernel pages as the pre-expanded table, whatever
    the slot, the KV block, or the query window."""
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: chunk)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))

    def run():
        return np.asarray(
            pooled_qsa.select_blocks(
                **args, ratio=RATIO, block_topk=BLOCK_TOPK, slot=slot
            )
        )

    expected = run()
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "1")
    np.testing.assert_array_equal(run(), expected)


def test_mixed_only_lowers_without_decode_passes(monkeypatch):
    """VLLM_TPU_QSA_INDEXER_MIXED_ONLY drops the one-token decode passes."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "1024")
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1")
    shape = jax.ShapeDtypeStruct
    args = (
        shape((32, 4, 128), jnp.bfloat16),
        shape((64, 512, 1, 2, 256), jnp.bfloat16),
        shape((32,), jnp.int32),
        shape((64, 16), jnp.int32),
        shape((32,), jnp.int32),
        shape((65,), jnp.int32),
        shape((64,), jnp.int32),
    )
    select_fn = jax.jit(
        functools.partial(pooled_qsa.select_blocks, ratio=RATIO, block_topk=512)
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        text = select_fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "StreamIdxTC-m-bq_4-bkvp_32" in text
    assert "StreamIdxTC-d" not in text


@pytest.mark.parametrize("bounded", [False, True])
def test_sc_bounded_copy_flag_selects_the_backend(monkeypatch, bounded):
    """VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY hands the decode/verify indexer a
    bounded-copy SparseCore selector; off keeps streamindex's default."""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        SparseCoreTopK,
    )

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    seen = []

    class Stop(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append(kwargs.get("topk_backend"))
        raise Stop

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", capture)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY", "1" if bounded else "0")
    q_lens, context, tokens = CASES["verify_k3"]
    with pytest.raises(Stop):
        pooled_qsa.select_blocks(
            **inputs(0, q_lens, context, tokens, "integer"),
            ratio=RATIO,
            block_topk=BLOCK_TOPK,
            max_query_len=4,
        )
    assert seen == [SparseCoreTopK(bounded_copy=True) if bounded else None]
