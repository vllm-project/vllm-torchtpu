# SPDX-License-Identifier: Apache-2.0
"""Compressed-key DCP indexer (VLLM_TPU_QSA_COMPRESSED_DCP): the union of a
pair's kept owned selections is the replicated selection.

Each simulated rank holds only its owned compressed groups (blocks with
``(t // B) % 2 == rank``) in the page of a table entry covering ``2B`` tokens;
every other kernel page of its pool is NaN. The real StreamIndex kernel runs
in TPU interpret mode with a JAX top-k for the SparseCore selector; the pair
merge (``dcp_select``) uses a top-k that keeps the *highest* tied slots, so it
must restore the replicated lowest-id order itself.
"""

import jax
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
from vllm_torchtpu.kernels.qwen4_exp.dcp_select import keep_mask, merge_threshold


def adversarial_topk(bits, k):
    scores = jax.lax.bitcast_convert_type(bits, jnp.float32)[:, ::-1]
    values, ids = jax.lax.top_k(scores, k)
    return jnp.where(values > -jnp.inf, scores.shape[1] - 1 - ids, -1)


@pytest.fixture
def reference_kernel(monkeypatch, interpret):  # noqa: F811
    def scored(*args, return_scores=False, **kwargs):
        kwargs.update(enable_early_exit=False)
        return streamindex.streamindex_topk(
            *args, topk_backend=ReferenceTopK(), return_scores=return_scores, **kwargs
        )

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)


def dcp_pools(args):
    """Per-rank pools / table: entry e of request r owns page r * E + e, whose
    first kernel pages hold that rank's block 2e + rank; everything else NaN."""
    pool, table = np.asarray(args["pool"].astype(jnp.float32)), np.asarray(args["table"])
    reqs, cols = table.shape
    entries = -(-cols // 2)
    kernel_pages = pool.reshape(-1, 128 // RATIO // 2, *pool.shape[3:])
    per_page = kernel_pages.shape[0] // pool.shape[0]
    logical = pool.shape[1] // 128
    dcp_table = np.arange(reqs * entries, dtype=np.int32).reshape(reqs, entries)
    assert reqs * entries <= pool.shape[0]
    pools = []
    for rank in range(2):
        out = np.full_like(kernel_pages, np.nan)
        for r in range(reqs):
            for e in range(entries):
                col = 2 * e + rank
                if col >= cols:
                    continue
                src, dst = table[r, col] * per_page, dcp_table[r, e] * per_page
                out[dst : dst + logical] = kernel_pages[src : src + logical]
        pools.append(jnp.asarray(out.reshape(pool.shape), args["pool"].dtype))
    return pools, jnp.asarray(dcp_table)


def dcp_union(args, **kw):
    pools, dcp_table = dcp_pools(args)
    vals, pos = [], []
    for rank in range(2):
        v, p = pooled_qsa.select_blocks(
            **{**args, "pool": pools[rank], "table": dcp_table},
            ratio=RATIO,
            block_topk=BLOCK_TOPK,
            dcp=(jnp.array([rank], jnp.int32), 2),
            **kw,
        )
        assert not np.isnan(np.asarray(v)).any()
        vals.append(v)
        pos.append(p)
    vals, pos = jnp.stack(vals), jnp.stack(pos)
    thr_val, thr_pos = merge_threshold(vals, pos, BLOCK_TOPK, topk=adversarial_topk)
    kept = [np.asarray(keep_mask(vals[r], pos[r], thr_val, thr_pos)) for r in range(2)]
    return [
        sorted(int(x) for r in range(2) for x in np.asarray(pos[r, t])[kept[r][t]])
        for t in range(pos.shape[1])
    ], np.asarray(pos)


def check(args, **kw):
    expected = np.asarray(
        pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, **kw)
    )
    union, pos = dcp_union(args, **kw)
    block = 128  # B / ratio groups per owner block
    for row, length in enumerate(np.asarray(args["lengths"])):
        want = sorted(int(x) for x in expected[row] if x >= 0)
        assert union[row] == want, row
        assert len(want) == min(BLOCK_TOPK, length // RATIO), row
    for rank in range(2):
        owned = pos[rank][pos[rank] < np.iinfo(np.int32).max]
        assert np.all((owned // block) % 2 == rank)


DECODE_CASES = {
    "verify_k3": ([4, 4, 4, 0, 0, 0], [3001, 2048, 1203, 0, 0, 0], 12),
    "ragged_1_to_4": ([1, 2, 0, 3, 4, 1], [2999, 1022, 7, 4090, 517, 3500], 12),
    "short_and_long": ([1, 1, 1, 1, 0, 0], [5, 30, 61, 3000, 0, 0], 6),
}


@pytest.mark.parametrize("case", sorted(DECODE_CASES))
@pytest.mark.parametrize("data", ["integer", "normal"])
@pytest.mark.parametrize("query_block", [0, 4])
def test_dcp_decode_union_is_replicated(monkeypatch, reference_kernel, case, data, query_block):
    q_lens, context, tokens = DECODE_CASES[case]
    args = inputs(len(case), q_lens, context, tokens, data)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 4)
    if query_block:
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "1024")
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", "1024")
    else:
        monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    check(args, max_query_len=max(q_lens))


@pytest.mark.parametrize("data", ["integer", "normal"])
def test_dcp_prefill_union_is_replicated(reference_kernel, data):
    # A ragged prefill step: a decode row, then two chunks.
    q_lens, context = [1, 7, 4], [2100, 900, 3000]
    args = inputs(7, q_lens, context, 12, data)
    check(args, max_query_len=1)


def test_dcp_and_shard_are_exclusive(reference_kernel):
    args = inputs(1, [1], [100], 1)
    with pytest.raises(ValueError, match="exclusive"):
        pooled_qsa.select_blocks(
            **args, ratio=RATIO, block_topk=BLOCK_TOPK, shard=(0, 2),
            dcp=(jnp.array([0], jnp.int32), 2),
        )
