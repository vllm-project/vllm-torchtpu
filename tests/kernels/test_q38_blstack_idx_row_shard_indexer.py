# SPDX-License-Identifier: Apache-2.0
"""q38 BLSTACK-IDX (VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD): the owned-group
decode indexer of compressed-key DCP, run on contiguous token shards of a
decode / verify step, returns exactly the unsharded step's local lists.

Under ``VLLM_TPU_QSA_COMPRESSED_DCP`` each DCP rank holds only the compressed
groups of its owned blocks (``(t // B) % D == rank``). With TP = S * D the TP
ranks ``r`` and ``r + D`` own the same blocks, so the row shard splits the
step's rows S ways: shard ``s`` scores rows ``[s * T / S, (s + 1) * T / S)``
with ``pooled_qsa.token_shard_step``'s re-based request starts / contexts and
the decode indexer's ``max_query_len`` routing. Every row must keep the same
local top-k values (f32 score bits) and global group ids as in the unsharded
call, on every DCP rank, so the pair merge, the keep mask and the attention see
identical inputs.

The real StreamIndex scoring kernel runs in TPU interpret mode with a JAX
top-k standing in for the SparseCore selector (as
``test_qwen4_exp_indexer_dcp.py``). Each simulated rank's pool holds only its
owned kernel pages; every other page is NaN, so a read of a page the rank does
not own shows up as NaN. Cases cut requests at the shard boundary, leave a
whole shard as padding, mix 1- to 4-row requests and include contexts with
fewer complete groups than the top-k.
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


@pytest.fixture(autouse=True)
def release_compiled_programs():
    """Each case compiles its own interpret-mode programs; drop them after
    the test (XLA:CPU's JIT code mappings otherwise run out: 'LLVM
    compilation error: Cannot allocate memory', blidx-dev1 j-002d0dbc)."""
    yield
    import gc

    jax.clear_caches()
    gc.collect()


@pytest.fixture
def reference_kernel(monkeypatch, interpret):  # noqa: F811
    def scored(*args, return_scores=False, **kwargs):
        kwargs.update(enable_early_exit=False)
        kwargs.pop("topk_backend", None)  # the reference selector stands in
        return streamindex.streamindex_topk(
            *args, topk_backend=ReferenceTopK(), return_scores=return_scores, **kwargs
        )

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)


def dcp_pools(args, d):
    """Per-rank pools / table for DCP ``d``: entry e of request r owns page
    r * E + e, whose first kernel pages hold that rank's block d * e + rank;
    every other kernel page is NaN (the rank pools get as many pages as the
    table needs)."""
    pool, table = np.asarray(args["pool"].astype(jnp.float32)), np.asarray(args["table"])
    reqs, cols = table.shape
    entries = -(-cols // d)
    kernel_pages = pool.reshape(-1, 128 // RATIO // 2, *pool.shape[3:])
    per_page = kernel_pages.shape[0] // pool.shape[0]
    logical = pool.shape[1] // 128
    dcp_table = np.arange(reqs * entries, dtype=np.int32).reshape(reqs, entries)
    pages = max(pool.shape[0], reqs * entries)
    pools = []
    for rank in range(d):
        out = np.full((pages * per_page, *kernel_pages.shape[1:]), np.nan, np.float32)
        for r in range(reqs):
            for e in range(entries):
                col = d * e + rank
                if col >= cols:
                    continue
                src, dst = table[r, col] * per_page, dcp_table[r, e] * per_page
                out[dst : dst + logical] = kernel_pages[src : src + logical]
        pools.append(jnp.asarray(out.reshape(pages, *pool.shape[1:]), args["pool"].dtype))
    return pools, jnp.asarray(dcp_table)


def local_lists(args, pool, table, rank, d, max_query_len, rows=None, shard=None):
    """``select_blocks`` under ``dcp=(rank, d)``: the unsharded step, or token
    shard ``shard = (s, S)`` of it exactly as ``_pooled_qsa`` slices it under
    ``token_shard`` with ``token_shard_decode``."""
    query, lengths, req = args["query"], args["lengths"], args["req"]
    starts, context = args["starts"], args["context"]
    if shard is not None:
        s, size = shard
        width = len(query) // size
        base, starts, context = pooled_qsa.token_shard_step(
            starts, context, jnp.asarray([s], jnp.int32), width
        )

        def own(x):
            return jax.lax.dynamic_slice_in_dim(x, base, width)

        query, lengths, req = own(query), own(lengths), own(req)
    vals, pos = pooled_qsa.select_blocks(
        query, pool, lengths, table, req, starts, context, ratio=RATIO,
        block_topk=BLOCK_TOPK, max_query_len=max_query_len,
        dcp=(jnp.array([rank], jnp.int32), d),
    )
    return np.asarray(vals), np.asarray(pos)


def ulp_distance(a, b):
    """Max distance in float32 ULPs between finite scores; non-finite entries
    (unfilled slots) must match bit for bit."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ai, bi = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    finite = np.isfinite(a) & np.isfinite(b)
    np.testing.assert_array_equal(ai[~finite], bi[~finite])
    return int(np.max(np.abs(ai - bi)[finite], initial=0))


def check_row_shards(args, d, size, max_query_len, exact=True):
    """Shards concatenated == unsharded on every live row and rank: group ids
    always equal; score bits equal (``exact``) or within 4 float32 ULPs (Gaussian
    data on XLA:CPU, which may round an M = 1..3-row GEMM of a request cut by a
    shard boundary differently from the M = 4 one; the MXU computes rows
    independently, as the integer and dyadic cases show)."""
    pools, table = dcp_pools(args, d)
    live = int(np.asarray(args["starts"])[-1])
    assert len(args["query"]) % size == 0
    checked = 0
    for rank in range(d):
        want_v, want_p = local_lists(args, pools[rank], table, rank, d, max_query_len)
        parts = [
            local_lists(args, pools[rank], table, rank, d, max_query_len, shard=(s, size))
            for s in range(size)
        ]
        got_v = np.concatenate([p[0] for p in parts])
        got_p = np.concatenate([p[1] for p in parts])
        assert not np.isnan(want_v[:live]).any() and not np.isnan(got_v[:live]).any()
        if exact:
            np.testing.assert_array_equal(
                got_v[:live].view(np.int32), want_v[:live].view(np.int32),
                err_msg=f"rank {rank}: local top-k score bits differ",
            )
        else:
            ulp = ulp_distance(got_v[:live], want_v[:live])
            assert ulp <= 4, (rank, ulp)
            print(f"Q38_BLIDX NORMAL_ULP rank={rank} max_ulp={ulp} "
                  f"bit_equal={bool(np.array_equal(got_v[:live].view(np.int32), want_v[:live].view(np.int32)))}")
        np.testing.assert_array_equal(
            got_p[:live], want_p[:live], err_msg=f"rank {rank}: local group ids differ"
        )
        # Every kept local entry is an owned group (block (g * ratio) // 512 % d).
        owned = got_p[:live][got_p[:live] < np.iinfo(np.int32).max]
        assert np.all(((owned * RATIO) // 512) % d == rank)
        checked += live
    # The case is not vacuous: some row selects a full top-k.
    lengths = np.asarray(args["lengths"])[:live]
    assert np.any(lengths // RATIO > BLOCK_TOPK)
    return checked


# (q_lens per request slot, context per slot, padded tokens). Tokens are a
# multiple of the shard counts tested; the decode path needs tokens <= slots *
# max_query_len, so every case keeps idle slots for its padding.
CASES = {
    # 8 verify requests, shard boundaries on request boundaries.
    "verify_aligned": ([4] * 8, [3001, 2048, 1203, 4000, 64, 65, 3, 517], 32),
    # A request cut by the 2-shard boundary (rows 11..14 across 12), a 1-row tail.
    "verify_cut": ([3, 4, 4, 4, 4, 1, 0, 0], [2999, 1022, 4089, 777, 2050, 1500, 0, 0], 24),
    # The second shard is all padding.
    "padding_shard": ([4, 1, 0, 0, 0, 0], [3500, 2047, 0, 0, 0, 0], 16),
    # Ragged 1..4 rows with idle slots between, padding after.
    "ragged": ([1, 2, 0, 3, 4, 1, 0, 0], [2999, 1022, 7, 4090, 517, 3500, 0, 0], 16),
    # Contexts below one group, below the top-k and long, cut at 4 shards too.
    "short_and_long": ([4, 4, 4, 4, 0, 0, 0, 0], [0, 5, 61, 3000, 0, 0, 0, 0], 16),
}


def _decode_env(monkeypatch, query_block, mixed_only, kv_block=1024):
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 4)
    if query_block:
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", str(kv_block))
    else:
        monkeypatch.delenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", raising=False)
    if mixed_only:
        monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", raising=False)


# (data, DCP, query block, mixed-only, KV block): the served decode indexer
# shape (query block 4, mixed pass only, DCP 4) on integer, Gaussian and dyadic
# data, plus the decode passes and the per-token path at DCP 2 / 4. With the
# decode passes on (mixed-only off) a shard can start a window on a 1-row
# request, which then takes a decode pass instead of the mixed pass; XLA:CPU
# rounds that pass's 1024-key reduction of integer data differently in the last
# bit (blidx-dev2 j-c8822ecd: 4 of 176 scores, 1 ULP), the case
# test_qwen4_exp_shared_stream_indexer.py checks with dyadic data. So those
# variants use dyadic data at 1024 keys and integer data at 256 keys.
VARIANTS = [
    ("integer", 4, 4, True, 1024),
    ("normal", 4, 4, True, 1024),
    ("dyadic", 2, 4, True, 1024),
    ("dyadic", 2, 4, False, 1024),
    ("integer", 2, 4, False, 256),
    ("integer", 4, 0, False, 1024),
]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("data,d,query_block,mixed_only,kv_block", VARIANTS)
def test_row_shards_equal_unsharded_lists(
    monkeypatch, reference_kernel, case, data, d, query_block, mixed_only, kv_block
):
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case) + 7 * d, q_lens, context, tokens, data)
    _decode_env(monkeypatch, query_block, mixed_only, kv_block)
    n = check_row_shards(args, d, 2, max(q_lens), exact=data != "normal")
    print(f"Q38_BLIDX ROWSHARD_EQUAL case={case} data={data} d={d} shards=2 "
          f"qb={query_block} mixed_only={int(mixed_only)} kv={kv_block} rows_checked={n}")


@pytest.mark.parametrize("case", ["short_and_long", "verify_aligned"])
def test_four_row_shards_equal_unsharded_lists(monkeypatch, reference_kernel, case):
    # TP 16 over DCP 4 would cut the step 4 ways; the geometry is generic in S.
    q_lens, context, tokens = CASES[case]
    args = inputs(3, q_lens, context, tokens, "integer")
    _decode_env(monkeypatch, 4, True)
    n = check_row_shards(args, 4, 4, max(q_lens))
    print(f"Q38_BLIDX ROWSHARD_EQUAL case={case} data=integer d=4 shards=4 rows_checked={n}")


def test_decode_shard_takes_the_decode_indexer(monkeypatch, reference_kernel):
    """Routing control: a verify step's shard with the step's max_query_len
    (what ``token_shard_decode`` passes) reaches the shared-stream decode
    indexer on the shard's rows; with max_query_len 1 (the prefill token
    shard's routing, which the port replaces) it takes the ragged prefill
    indexer instead."""
    q_lens, context, tokens = CASES["verify_aligned"]
    args = inputs(11, q_lens, context, tokens, "integer")
    _decode_env(monkeypatch, 4, True)
    calls = []
    original = pooled_qsa._select_shared_stream

    def spy(*a, **k):
        calls.append(len(a[0]))
        return original(*a, **k)

    monkeypatch.setattr(pooled_qsa, "_select_shared_stream", spy)
    pools, table = dcp_pools(args, 2)
    local_lists(args, pools[0], table, 0, 2, 4, shard=(1, 2))
    assert calls == [16], calls  # the decode route, on the shard's 16 rows
    calls.clear()
    local_lists(args, pools[0], table, 0, 2, 1, shard=(1, 2))
    assert calls == [], calls  # max_query_len 1: the ragged prefill indexer
