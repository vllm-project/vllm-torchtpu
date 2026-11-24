# SPDX-License-Identifier: Apache-2.0
"""Rank-balanced token shard of the prefill indexer
(VLLM_TPU_QSA_INDEXER_BALANCED_SHARD).

The runner orders verify requests first, so the contiguous token shard gives
shard 0 every verify request. The balanced layout deals short requests over
the shards and fills the rest with the other rows; after the all-gather the
rows are put back in step order. Each shard's selection must be bitwise the
contiguous shard's (and the replicated indexer's). Real StreamIndex kernel in
TPU interpret mode on CPU, as test_qwen4_exp_indexer_token_shard.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as mtp_rows
from test_qwen4_exp_indexer_shard import (  # noqa: F401 (fixtures)
    assert_same_selection,
    reference_kernel,
    select,
)
from test_qwen4_exp_indexer_token_shard import token_sharded
from test_qwen4_exp_mtp_draft_rows import rowwise_attention  # noqa: F401
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixtures)
    inputs,
    interpret,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa


def _layout(starts, context, tokens, size):
    starts = jnp.asarray(starts, jnp.int32)
    context = jnp.asarray(context, jnp.int32)
    plan = pooled_qsa.balanced_token_shard_plan(starts, context, tokens, size)
    shards = [
        pooled_qsa.balanced_shard_rows(plan, starts, context, jnp.asarray([s], jnp.int32))
        for s in range(size)
    ]
    positions = pooled_qsa.balanced_shard_positions(plan, starts, tokens)
    return plan, shards, np.asarray(positions)


def _check_layout(starts, context, tokens, size):
    """Shards partition the step; every real row keeps its request and its
    position (context + offset); the positions invert the concatenation."""
    plan, shards, positions = _layout(starts, context, tokens, size)
    width = tokens // size
    starts_np, context_np = np.asarray(starts), np.asarray(context)
    rows = np.concatenate([np.asarray(r) for r, _, _ in shards])
    np.testing.assert_array_equal(np.sort(rows), np.arange(tokens))
    np.testing.assert_array_equal(rows[positions], np.arange(tokens))
    for rows_s, sub_starts, sub_context in shards:
        rows_s, sub_starts = np.asarray(rows_s), np.asarray(sub_starts)
        sub_context = np.asarray(sub_context)
        assert len(rows_s) == width and sub_starts[0] == 0
        assert np.all(np.diff(rows_s) > 0)  # step order inside a shard
        for i in range(len(context_np)):
            for p in range(sub_starts[i], sub_starts[i + 1]):
                row = rows_s[p]
                assert starts_np[i] <= row < starts_np[i + 1]
                assert sub_context[i] + (p - sub_starts[i]) == context_np[i] + (
                    row - starts_np[i]
                )
        assert np.all(rows_s[sub_starts[-1] :] >= starts_np[-1])
    return plan, shards


def _starts(q_lens):
    return np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)


def test_verify_first_step_spreads_the_verify_requests():
    # 11 verify requests of 4 rows first (the runner's order), one prefill
    # chunk, a padded tail: 8192 rows over 8 ranks.
    q_lens = [4] * 11 + [8000] + [0] * 20
    context = [136000 + 97 * i for i in range(11)] + [120000] + [0] * 20
    plan, shards = _check_layout(_starts(q_lens), context, 8192, 8)
    owner = np.asarray(plan.shard_of)[:11]
    assert np.asarray(plan.short)[:11].all()
    np.testing.assert_array_equal(np.bincount(owner, minlength=8), [2, 2, 2, 1, 1, 1, 1, 1])
    # Every shard holds a part of the prefill chunk.
    assert all(int(np.asarray(c)[11]) > 0 for c in np.asarray(plan.count))


def test_contiguous_fallback_is_token_shard_step():
    # One prefill request: nothing to balance, the contiguous layout stays.
    q_lens, context = [3000, 0, 0], [5000, 0, 0]
    starts = _starts(q_lens)
    plan, shards = _check_layout(starts, context, 4096, 8)
    assert not np.asarray(plan.short).any()
    for s, (rows, sub_starts, sub_context) in enumerate(shards):
        _, want_starts, want_context = pooled_qsa.token_shard_step(
            jnp.asarray(starts), jnp.asarray(context, jnp.int32),
            jnp.asarray([s], jnp.int32), 512,
        )
        np.testing.assert_array_equal(np.asarray(rows), np.arange(s * 512, (s + 1) * 512))
        np.testing.assert_array_equal(np.asarray(sub_starts), np.asarray(want_starts))
        np.testing.assert_array_equal(np.asarray(sub_context), np.asarray(want_context))


@pytest.mark.parametrize("seed", range(40))
def test_random_layouts(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.choice([2, 4, 8]))
    tokens = int(rng.choice([256, 512, 1024, 8192]))
    slots = int(rng.integers(2, min(32, tokens // size - 1)))
    kinds = rng.integers(0, 3, size=slots)
    q_lens = np.where(kinds == 0, 0, np.where(kinds == 1, rng.integers(1, 6, slots), 0))
    budget = tokens - int(q_lens.sum())
    long_ids = np.flatnonzero(kinds == 2)
    if len(long_ids):
        cuts = np.sort(rng.integers(0, budget + 1, size=len(long_ids)))
        q_lens[long_ids] = np.diff(np.concatenate([[0], cuts]))
    context = rng.integers(0, 200000, size=slots)
    _check_layout(_starts(q_lens), context, tokens, size)


def _balanced_sharded(args, size):
    tokens = len(args["query"])
    starts, context = args["starts"], args["context"]
    plan = pooled_qsa.balanced_token_shard_plan(starts, context, tokens, size)
    shards = []
    for s in range(size):
        rows, sub_starts, sub_context = pooled_qsa.balanced_shard_rows(
            plan, starts, context, jnp.asarray([s], jnp.int32)
        )
        part = dict(args, starts=sub_starts, context=sub_context)
        for name in ("query", "lengths", "req"):
            part[name] = args[name][rows]
        shards.append(select(part))
    gathered = jnp.asarray(np.concatenate(shards))
    return plan, np.asarray(pooled_qsa.token_shard_unpermute(gathered, starts, context, size))


@pytest.mark.parametrize(
    "q_lens,context,tokens,size",
    [
        # Verify-first: five verify requests, then two prefill chunks.
        ([4, 4, 4, 4, 4, 60, 30], [3000, 1700, 2500, 900, 2900, 1500, 260], 256, 4),
        # Verify requests first, an empty slot, two prefill chunks, padding.
        ([4, 4, 0, 4, 40, 70], [2999, 2100, 0, 1203, 700, 9], 128, 2),
    ],
)
def test_balanced_shards_match_contiguous_bitwise(
    reference_kernel,  # noqa: F811
    q_lens,
    context,
    tokens,
    size,
):
    args = inputs(5, q_lens, context, tokens, "integer")
    plan, actual = _balanced_sharded(args, size)
    assert np.asarray(plan.short).any(), "balanced layout not taken"
    today = token_sharded(args, size)
    np.testing.assert_array_equal(actual, today)
    assert_same_selection(actual, select(args), args["lengths"])


@pytest.mark.parametrize(
    "starts,context",
    [([0, 6, 7, 8], [1661, 1663, 1500]), (mtp_rows.STARTS, mtp_rows.CONTEXT)],
)
def test_balanced_token_shard_steps_compose_to_the_pooled_step(
    monkeypatch,
    rowwise_attention,  # noqa: F811
    starts,
    context,
):
    """Two one-row requests after a 6-row one: balanced over 2 shards (the
    stand-in indexer shifts its blocks when a shard's ragged batch disagrees
    with its rows); the draft-rows layout keeps the contiguous one."""
    from vllm_torchtpu import envs

    size = 2
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_INDEXER_BALANCED_SHARD", True, raising=False)
    head, pool, tail = mtp_rows._inputs(False, starts, context)
    plan = pooled_qsa.balanced_token_shard_plan(tail[0], tail[1], 8, size)
    assert bool(np.asarray(plan.short).any()) == (starts[1] == 6)
    options = dict(mtp_rows.OPTIONS, token_topk=256)
    want_pool, want_out, want_selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_step(*args, **options)
    )(*head, pool, *tail)
    results = [
        jax.jit(
            lambda *args: pooled_qsa.pooled_qsa_token_select_step(
                *args, shard_size=size, **options
            )
        )(*head, pool, *tail, jnp.asarray([s], jnp.int32))
        for s in range(size)
    ]
    new_pool, q, gate, lengths, req, starts, _ = results[0]
    gathered = pooled_qsa.token_shard_unpermute(
        jnp.concatenate([result[-1] for result in results]), tail[0], tail[1], size
    )
    _, out, selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_shard_attend_step(
            *args, kv_heads=2, compress_ratio=4, token_topk=256, merged=True
        )
    )(q, gate, lengths, req, starts, tail[3], tail[1], new_pool, gathered)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(want_selected))
