# SPDX-License-Identifier: Apache-2.0
"""TP-sharded QSA indexer: the same selections as the replicated indexer.

Every simulated rank scores its interleaved slice of the compressed keys with
the real StreamIndex kernel (TPU interpret mode on CPU) and a JAX top-k in
place of the SparseCore selector; ``merge_candidates`` then combines all
ranks' candidates. The replicated indexer is the reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as mtp_rows
from test_qwen4_exp_mtp_draft_rows import rowwise_attention  # noqa: F401
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixture)
    BLOCK_TOPK,
    RATIO,
    ReferenceTopK,
    inputs,
    interpret,
    streamindex,
)

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import BITS_NEG_INF
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa


def adversarial_topk(bits, k):
    """Exact top-k as a set, but equal scores keep the *highest* slot.

    The SparseCore selector does not promise which equal boundary scores it
    keeps; the merge must restore the replicated indexer's lowest-position
    choice whatever the selector did.
    """
    scores = jax.lax.bitcast_convert_type(bits, jnp.float32)[:, ::-1]
    values, ids = jax.lax.top_k(scores, k)
    return jnp.where(values > -jnp.inf, scores.shape[1] - 1 - ids, -1)


def lowest_topk(bits, k):
    scores = jax.lax.bitcast_convert_type(bits, jnp.float32)
    values, ids = jax.lax.top_k(scores, k)
    return jnp.where(values > -jnp.inf, ids, -1)


@pytest.fixture
def reference_kernel(monkeypatch, interpret):  # noqa: F811
    def scored(*args, return_scores=False, **kwargs):
        kwargs.update(enable_early_exit=False)
        return streamindex.streamindex_topk(
            *args, topk_backend=ReferenceTopK(), return_scores=return_scores, **kwargs
        )

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)


def select(args, **kw):
    return np.asarray(
        pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, **kw)
    )


def sharded(args, size, topk=adversarial_topk, **kw):
    candidates = np.stack(
        [select(args, shard=(rank, size), **kw) for rank in range(size)]
    )
    tokens = candidates.shape[1]
    # [ranks, tokens, k, 2] -> [tokens, ranks * k, 2], as the gather lays out.
    merged = jnp.moveaxis(jnp.asarray(candidates), 0, 1).reshape(tokens, -1, 2)
    return candidates, np.asarray(
        pooled_qsa.merge_candidates(merged, BLOCK_TOPK, topk=topk)
    )


def assert_same_selection(actual, expected, lengths):
    """Same set per row, ``-1`` suffix, exactly the visible count."""
    for row, length in enumerate(np.asarray(lengths)):
        count = min(BLOCK_TOPK, length // RATIO)
        assert np.all(actual[row, count:] == -1), row
        assert np.all(actual[row, :count] >= 0), row
        np.testing.assert_array_equal(
            np.sort(actual[row, :count]), np.sort(expected[row, :count]), str(row)
        )


DECODE_CASES = {
    # (q_lens per request row, context per request row, padded tokens).
    "verify_k3": ([4, 4, 4, 0, 0, 0], [3001, 2048, 1203, 0, 0, 0], 12),
    "ragged_1_to_4": (
        [1, 2, 0, 3, 4, 1, 0, 0],
        [2999, 1022, 7, 4090, 517, 3500, 0, 0],
        12,
    ),
    # Budgets larger than the available blocks, and a long row.
    "short_and_long": ([1, 1, 1, 1, 0, 0], [5, 30, 61, 3000, 0, 0], 6),
}


@pytest.mark.parametrize(
    "case,query_block,kv_block,data,size",
    [
        (case, *mode, data, 8)
        for case in sorted(DECODE_CASES)
        for mode in ((0, 0), (4, 256), (4, 1024))
        for data in ("integer", "normal")
    ]
    # A TP size that does not divide the table width.
    + [("ragged_1_to_4", 4, 1024, "integer", 3)],
)
def test_sharded_decode_matches_replicated(
    monkeypatch, reference_kernel, case, query_block, kv_block, data, size
):
    q_lens, context, tokens = DECODE_CASES[case]
    args = inputs(len(case) + size, q_lens, context, tokens, data)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 4)
    if query_block:
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    else:
        monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    max_query_len = max(q_lens)
    expected = select(args, max_query_len=max_query_len)
    candidates, actual = sharded(args, size, max_query_len=max_query_len)
    assert_same_selection(actual, expected, args["lengths"])

    # Candidates are global positions this rank owns (one 32-row page of
    # every `size`), never more than the rank's own visible groups.
    rows = 128 // RATIO
    for rank, rank_candidates in enumerate(candidates):
        position = rank_candidates[..., 1]
        live = position >= 0
        assert np.all((position[live] // rows) % size == rank)
        assert np.all(rank_candidates[..., 0][~live] == BITS_NEG_INF)
    if data == "integer":
        # Small integers tie a lot: the reselection branch must have run.
        _, first = sharded(args, size, topk=lowest_topk, max_query_len=max_query_len)
        assert_same_selection(first, expected, args["lengths"])


def test_sharded_prefill_matches_replicated(reference_kernel):
    # More tokens than request rows: the ragged prefill pass, with an idle
    # row, a partial group tail and a request shorter than the budget.
    q_lens, context = [40, 0, 20, 4], [700, 0, 1500, 9]
    args = inputs(3, q_lens, context, 64, "integer")
    expected = select(args)
    _, actual = sharded(args, 8)
    assert_same_selection(actual, expected, args["lengths"])


@pytest.mark.parametrize("seed", range(20))
def test_merge_is_exact_topk_with_lowest_position_ties(seed):
    """Random rank lists, many exact ties, empty slots and short rows."""
    rng = np.random.default_rng(seed)
    ranks, k, tokens = 8, 16, 9
    rows = 32
    want, lists = [], []
    for token in range(tokens):
        visible = int(rng.integers(0, 600))
        scores = rng.integers(0, 4, size=visible).astype(np.float32)
        if seed % 3 == 0:
            scores = rng.normal(size=visible).astype(np.float32)
        order = sorted(range(visible), key=lambda g: (-scores[g], g))
        want.append(sorted(order[:k]))
        per_rank = []
        for rank in range(ranks):
            owned = [g for g in order if (g // rows) % ranks == rank][:k]
            # Each rank's own winners in a scrambled slot order.
            owned = list(rng.permutation(owned)) if owned else []
            bits = np.full(k, BITS_NEG_INF, np.int32)
            pos = np.full(k, -1, np.int32)
            bits[: len(owned)] = scores[owned].view(np.int32)
            pos[: len(owned)] = owned
            per_rank.append(np.stack([bits, pos], -1))
        lists.append(np.concatenate(per_rank))
    candidates = jnp.asarray(np.stack(lists))
    for topk in (adversarial_topk, lowest_topk):
        merged = np.asarray(pooled_qsa.merge_candidates(candidates, k, topk=topk))
        for token in range(tokens):
            count = len(want[token])
            assert np.all(merged[token, count:] == -1)
            np.testing.assert_array_equal(np.sort(merged[token, :count]), want[token])


def test_visible_blocks_is_selection_when_everything_fits():
    lengths = jnp.asarray([0, 3, 4, 7, 64, 65], jnp.int32)
    blocks = np.asarray(pooled_qsa.visible_blocks(lengths, 4, 16))
    for row, length in enumerate(np.asarray(lengths)):
        count = length // 4
        np.testing.assert_array_equal(blocks[row, :count], np.arange(count))
        assert np.all(blocks[row, count:] == -1)


def test_shard_pages_interleave_and_pad():
    pages = jnp.arange(2 * 10, dtype=jnp.int32).reshape(2, 10)
    local = np.asarray(pooled_qsa._shard_pages(pages, (3, 4), 2))
    # Columns 3 and 7 of each row; three local columns padded to four.
    np.testing.assert_array_equal(local, [[3, 7, 0, 0], [13, 17, 0, 0]])
    local = np.asarray(pooled_qsa._shard_pages(pages, (1, 4), 4))
    np.testing.assert_array_equal(local, [[1, 5, 9, 0], [11, 15, 19, 0]])


@pytest.fixture
def sharded_stand_ins(monkeypatch, rowwise_attention):  # noqa: F811
    """The draft-rows CPU stand-ins, with rank candidates and set attention.

    A rank's candidates are the stand-in blocks it owns, scored so that the
    merge must return exactly the stand-in's set. Attention sums the blocks,
    so it depends on the selected set and not on the merge's slot order.
    """
    stand_in = pooled_qsa.select_blocks

    def select_blocks(query, pool, lengths, table, req, starts, context, **kw):
        shard = kw.pop("shard", None)
        blocks = stand_in(query, pool, lengths, table, req, starts, context, **kw)
        if shard is None:
            return blocks
        rank, size = shard
        own = (blocks >= 0) & ((blocks // (128 // RATIO)) % size == rank)
        score = jnp.arange(blocks.shape[1], 0, -1, dtype=jnp.float32)
        bits = jax.lax.bitcast_convert_type(score, jnp.int32)
        return jnp.stack(
            [
                jnp.where(own, bits[None, :], BITS_NEG_INF),
                jnp.where(own, blocks, -1),
            ],
            -1,
        )

    def qsa_attention(q, gate, blocks, lengths, table, cache, starts, **kw):
        value = q.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
        total = jnp.sum(jnp.where(blocks >= 0, blocks, 0), axis=1)
        offset = total[:, None, None] + table[:, :1, None] + lengths[:, None, None]
        value = value + offset
        return jnp.where((lengths > 0)[:, None, None], value, 0), cache

    monkeypatch.setattr(pooled_qsa, "select_blocks", select_blocks)
    monkeypatch.setattr(pooled_qsa, "qsa_attention", qsa_attention)
    merge = pooled_qsa.merge_candidates
    monkeypatch.setattr(
        pooled_qsa,
        "merge_candidates",
        lambda candidates, k: merge(candidates, k, topk=adversarial_topk),
    )


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("rows", [None, [2, 6, -1], [1, 6, 6], [2, 6, 6, 6]])
def test_shard_steps_compose_to_the_pooled_step(sharded_stand_ins, rows, merged):
    """select -> gather -> attend equals the replicated step and rows step."""
    head, pool, tail = mtp_rows._inputs(False, mtp_rows.STARTS, mtp_rows.CONTEXT)
    # A budget the stand-in contexts exceed: neither step short-cuts.
    options = dict(mtp_rows.OPTIONS, token_topk=256)
    extra = () if rows is None else (jnp.asarray(rows, jnp.int32),)
    reference = (
        pooled_qsa.pooled_qsa_step if rows is None else pooled_qsa.pooled_qsa_rows_step
    )
    want_pool, want_out, want_selected = jax.jit(
        lambda *args: reference(*args, **options)
    )(*head, pool, *tail, *extra)

    select_step = (
        pooled_qsa.pooled_qsa_shard_select_step
        if rows is None
        else pooled_qsa.pooled_qsa_shard_select_rows_step
    )
    size = 4
    results = [
        jax.jit(
            lambda *args, rank=rank: select_step(
                *args, shard_rank=rank, shard_size=size, **options
            )
        )(*head, pool, *tail, *extra)
        for rank in range(size)
    ]
    new_pool, q, gate, lengths, req, starts, _ = results[0]
    np.testing.assert_array_equal(
        np.asarray(new_pool).view(np.uint8), np.asarray(want_pool).view(np.uint8)
    )
    gathered = jnp.stack([result[-1] for result in results])
    if merged:
        gathered = pooled_qsa.pooled_qsa_shard_merge(gathered, 4, 256)
    attend_step = (
        pooled_qsa.pooled_qsa_shard_attend_step
        if rows is None
        else pooled_qsa.pooled_qsa_shard_attend_rows_step
    )
    _, out, selected = jax.jit(
        lambda *args: attend_step(
            *args, kv_heads=2, compress_ratio=4, token_topk=256, merged=merged
        )
    )(q, gate, lengths, req, starts, tail[3], tail[1], new_pool, gathered, *extra)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want_out))
    for got, expected in zip(np.asarray(selected), np.asarray(want_selected)):
        np.testing.assert_array_equal(np.sort(got), np.sort(expected))
