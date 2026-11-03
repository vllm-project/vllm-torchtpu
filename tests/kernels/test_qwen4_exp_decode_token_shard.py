# SPDX-License-Identifier: Apache-2.0
"""Token-sharded decode indexer (VLLM_TPU_QSA_DECODE_TOKEN_SHARD).

On the replicated backbone each DCP pair indexes one contiguous 1/size of a
decode / verify step's rows (``token_shard_step``); a shard holds at most one
row per request slot, so it takes the shared-stream decode indexer. The
concatenated shards must select exactly what the whole step selects, for the
plain and the compressed-DCP (owned groups, local lists) indexers. Real
StreamIndex kernel in TPU interpret mode on CPU, a JAX top-k in place of the
SparseCore selector.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixture)
    BLOCK_TOPK,
    RATIO,
    ReferenceTopK,
    inputs,
    interpret,
    streamindex,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

# (q_lens per request slot, context per slot, padded rows, shards). Verify
# steps (4 rows per request) with idle slots and an empty bucket tail; a
# shard boundary inside a request; one-token decodes.
CASES = {
    "verify_full": ([4] * 4 + [0] * 4, [3001, 2048, 1203, 517] + [0] * 4, 16, 4),
    "verify_idle_tail": ([4, 4, 0, 4, 0, 0, 0, 0], [999, 3500, 0, 64] + [0] * 4, 16, 4),
    "boundary_in_request": (
        [3, 4, 4, 1, 0, 0, 0, 0],
        [2999, 1022, 4000, 7] + [0] * 4,
        16,
        2,
    ),
    "decodes": ([1] * 8, [4000, 3, 64, 65, 999, 2048, 3001, 517], 8, 4),
}


@pytest.fixture
def reference_topk(monkeypatch, interpret):  # noqa: F811
    """The JAX top-k in place of every SparseCore selector (the compressed-DCP
    path picks its own backend, so override it rather than add one)."""

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, topk_backend=ReferenceTopK())
        return streamindex.streamindex_topk(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)


def select(args, dcp=None, max_query_len=4):
    out = pooled_qsa.select_blocks(
        **args, ratio=RATIO, block_topk=BLOCK_TOPK, max_query_len=max_query_len, dcp=dcp
    )
    return tuple(np.asarray(x) for x in out) if dcp else (np.asarray(out),)


def token_sharded(args, size, dcp=None):
    """Every shard's rows through ``token_shard_selection``, concatenated."""

    def run(query, *, lengths, req, starts, context, max_query_len):
        part = dict(
            args, query=query, lengths=lengths, req=req, starts=starts, context=context
        )
        out = pooled_qsa.select_blocks(
            **part,
            ratio=RATIO,
            block_topk=BLOCK_TOPK,
            max_query_len=max_query_len,
            dcp=dcp,
        )
        return out if dcp else (out,)

    shards = [
        pooled_qsa.token_shard_selection(
            run,
            args["query"],
            args["lengths"],
            args["req"],
            args["starts"],
            args["context"],
            4,
            (jnp.asarray([s], jnp.int32), size),
        )
        for s in range(size)
    ]
    return tuple(
        np.concatenate([np.asarray(x[i]) for x in shards])
        for i in range(len(shards[0]))
    )


@pytest.mark.parametrize("dcp_rank", [None, 0, 1])
@pytest.mark.parametrize("case", sorted(CASES))
def test_decode_token_shards_match_whole_step(monkeypatch, reference_topk, case, dcp_rank):
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "256")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    q_lens, context, tokens, size = CASES[case]
    # Integer data: q @ k is exact in any order, so the CPU GEMM's blocking by
    # row count cannot differ between the shard and the whole step.
    args = inputs(len(case), q_lens, context, tokens, "integer")
    dcp = None if dcp_rank is None else (jnp.asarray([dcp_rank], jnp.int32), 2)
    expected = select(args, dcp)
    actual = token_sharded(args, size, dcp)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize(
    "q_lens,width,aligned",
    [
        ([4, 4, 4, 4, 0, 0], 8, True),
        ([4, 4, 0, 4, 4, 0], 8, True),
        ([3, 4, 4, 1, 0, 0], 4, False),
        ([1, 1, 1, 1, 1, 1], 2, True),
        ([4, 2, 2, 0, 0, 0], 4, True),
        ([2, 4, 2, 0, 0, 0], 4, False),
    ],
)
def test_requests_within_shards(q_lens, width, aligned):
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    assert bool(pooled_qsa.requests_within_shards(starts, width)) == aligned
