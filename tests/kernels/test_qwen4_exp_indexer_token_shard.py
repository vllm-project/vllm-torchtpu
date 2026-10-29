# SPDX-License-Identifier: Apache-2.0
"""Token-sharded prefill indexer (VLLM_TPU_QSA_INDEXER_TOKEN_SHARD).

Under prefill sequence parallelism every rank indexes the whole gathered
step. With the token shard each rank indexes one contiguous 1/size of the
rows (``token_shard_step``); the concatenated shards must select exactly the
replicated indexer's blocks. Real StreamIndex kernel in TPU interpret mode on
CPU, a JAX top-k in place of the SparseCore selector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as mtp_rows
from test_qwen4_exp_mtp_draft_rows import rowwise_attention  # noqa: F401
from test_qwen4_exp_indexer_shard import (  # noqa: F401 (fixtures)
    assert_same_selection,
    reference_kernel,
    select,
)
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixtures)
    inputs,
    interpret,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa


def token_sharded(args, size):
    tokens = len(args["query"])
    rows = tokens // size
    shards = []
    for s in range(size):
        base, starts, context = pooled_qsa.token_shard_step(
            args["starts"], args["context"], jnp.asarray([s], jnp.int32), rows
        )
        assert int(base) == s * rows
        part = dict(args, starts=starts, context=context)
        for name in ("query", "lengths", "req"):
            part[name] = args[name][s * rows : (s + 1) * rows]
        shards.append(select(part))
    return np.concatenate(shards)


@pytest.mark.parametrize(
    "q_lens,context,tokens,size",
    [
        # Requests crossing shard borders, an idle row, a partial group tail
        # and a request shorter than the budget; the bucket's tail is empty.
        ([40, 0, 20, 4], [700, 0, 1500, 9], 64, 8),
        # Leading one-token decodes (the decode prefix lands in shard 0) and
        # a shard boundary leaving a single row of a request in a shard.
        ([1, 1, 25, 37], [900, 33, 1203, 0], 64, 8),
        # A full bucket, every shard live, one long request.
        ([128], [2999], 128, 8),
        ([60, 68], [1500, 260], 128, 4),
    ],
)
def test_token_shards_match_replicated(
    reference_kernel, q_lens, context, tokens, size
):
    args = inputs(5, q_lens, context, tokens, "integer")
    expected = select(args)
    actual = token_sharded(args, size)
    assert_same_selection(actual, expected, args["lengths"])


def test_shard_step_layout():
    starts = jnp.asarray([0, 1, 2, 27, 64], jnp.int32)
    context = jnp.asarray([900, 33, 1203, 0], jnp.int32)
    base, sub, ctx = pooled_qsa.token_shard_step(
        starts, context, jnp.asarray([3], jnp.int32), 8
    )
    assert int(base) == 24
    np.testing.assert_array_equal(np.asarray(sub), [0, 0, 0, 3, 8])
    np.testing.assert_array_equal(np.asarray(ctx), [901, 34, 1225, 0])


@pytest.mark.parametrize("size", [2, 4, 8])
def test_token_shard_steps_compose_to_the_pooled_step(rowwise_attention, size):
    """select on every token shard -> gather -> attend equals the replicated
    step. The stand-in indexer shifts its blocks when the shard's ragged
    batch (starts, context) disagrees with the rows' lengths and requests."""
    head, pool, tail = mtp_rows._inputs(False, mtp_rows.STARTS, mtp_rows.CONTEXT)
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
    np.testing.assert_array_equal(
        np.asarray(new_pool).view(np.uint8), np.asarray(want_pool).view(np.uint8)
    )
    gathered = jnp.concatenate([result[-1] for result in results])
    _, out, selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_shard_attend_step(
            *args, kv_heads=2, compress_ratio=4, token_topk=256, merged=True
        )
    )(q, gate, lengths, req, starts, tail[3], tail[1], new_pool, gathered)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(want_selected))


@pytest.mark.parametrize(
    "min_tokens,max_num_seqs,active", [(1024, 32, True), (1024, 128, False), (4096, 256, True)]
)
def test_mode_is_decided_once_for_the_smallest_sp_bucket(
    monkeypatch, min_tokens, max_num_seqs, active
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    for name, value in (
        ("VLLM_TPU_QSA_INDEXER_TOKEN_SHARD", True),
        ("QWEN4_EXP_PREFILL_SP", True),
        ("QWEN4_EXP_PREFILL_SP_MIN_TOKENS", min_tokens),
    ):
        monkeypatch.setattr(envs, name, value, raising=False)
    assert Qwen4ExpQSA._token_shard_mode(8, True, max_num_seqs) is active
    assert Qwen4ExpQSA._token_shard_mode(1, True, max_num_seqs) is False
    assert Qwen4ExpQSA._token_shard_mode(8, False, max_num_seqs) is False
    monkeypatch.setattr(envs, "QWEN4_EXP_PREFILL_SP", False, raising=False)
    assert Qwen4ExpQSA._token_shard_mode(8, True, max_num_seqs) is False


VERIFY_STARTS, VERIFY_CONTEXT = [0, 4, 8, 8], [1661, 1663, 0]


@pytest.mark.parametrize(
    "starts,context",
    [
        # Two verify requests of four rows and an idle slot.
        (VERIFY_STARTS, VERIFY_CONTEXT),
        # A short verify request and a decode: windows hold partial requests.
        ([0, 3, 4, 4], [1661, 1663, 0]),
    ],
)
@pytest.mark.parametrize("size", [2, 4, 8])
def test_request_window_shards_compose_to_the_pooled_step(
    rowwise_attention, starts, context, size
):
    """VLLM_TPU_QSA_INDEXER_REQ_SHARD: on a verify-sized step each rank
    indexes a request window (``request_window_step``); gathering the windows
    and mapping them back to rows (``request_window_merge``) equals the
    replicated step bit for bit."""
    head, pool, tail = mtp_rows._inputs(False, starts, context)
    options = dict(mtp_rows.OPTIONS, token_topk=256, max_query_len=4)
    tokens, slots = len(head[0]), len(context)
    assert pooled_qsa.request_window_shard(tokens, slots, size, 4)
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
    new_pool, q, gate, lengths, req, starts_out, _ = results[0]
    np.testing.assert_array_equal(
        np.asarray(new_pool).view(np.uint8), np.asarray(want_pool).view(np.uint8)
    )
    window = pooled_qsa.request_window_shape(slots, size, 4)
    assert results[0][-1].shape[0] == window[1]
    gathered = jnp.concatenate([result[-1] for result in results])
    _, out, selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_shard_attend_step(
            *args,
            kv_heads=2,
            compress_ratio=4,
            token_topk=256,
            max_query_len=4,
            merged=True,
            request_window=window,
        )
    )(q, gate, lengths, req, starts_out, tail[3], tail[1], new_pool, gathered)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(want_selected))


def test_request_window_layout():
    starts = jnp.asarray([0, 4, 8, 12, 13, 13], jnp.int32)
    context = jnp.asarray([10, 20, 30, 40, 0], jnp.int32)
    table = jnp.arange(10, dtype=jnp.int32).reshape(5, 2)
    first, sub, ctx, rows, own = pooled_qsa.request_window_step(
        starts, context, table, jnp.asarray([1], jnp.int32), 2, 8
    )
    assert int(first) == 2
    np.testing.assert_array_equal(np.asarray(sub), [0, 4, 5])
    np.testing.assert_array_equal(np.asarray(ctx), [30, 40])
    np.testing.assert_array_equal(np.asarray(rows), [[4, 5], [6, 7]])
    np.testing.assert_array_equal(np.asarray(own(jnp.arange(16))), np.arange(8, 16))
    # The last window pads past the step's slots with empty requests.
    first, sub, ctx, rows, own = pooled_qsa.request_window_step(
        starts, context, table, jnp.asarray([2], jnp.int32), 2, 8
    )
    np.testing.assert_array_equal(np.asarray(sub), [0, 0, 0])
    np.testing.assert_array_equal(np.asarray(ctx), [0, 0])
    gathered = jnp.arange(24)[:, None] * jnp.ones((1, 3), jnp.int32)
    req = jnp.asarray([0] * 4 + [1] * 4 + [2] * 4 + [3], jnp.int32)
    merged = pooled_qsa.request_window_merge(gathered, req, starts, 2, 8)
    np.testing.assert_array_equal(np.asarray(merged[:, 0]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
