# SPDX-License-Identifier: Apache-2.0
"""Main-KV DCP sparse prefill attends only the rank's page of each KV tile.

A DCP KV tile is one table entry (dcp_world pages of tokens); the rank holds
the page at token offset dcp_rank * page. Merging every rank's (out, lse)
must equal the replicated prefill over the same logical keys. TPU only: the
Pallas interpreter cannot run the kernel's bitcast FP8 cache loads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_prefill,
)

RATIO, PAGE, TOKENS = 4, 512, 512


@pytest.mark.tpu_test
@pytest.mark.parametrize("dcp", [2, 4])
def test_dcp_ranks_merge_to_the_replicated_prefill(dcp):
    rng = np.random.default_rng(dcp)
    entries = 3
    logical_tokens = entries * dcp * PAGE
    query = jnp.asarray(rng.normal(size=(TOKENS, 12, 256)), jnp.bfloat16)
    lengths = np.linspace(1, logical_tokens, TOKENS).astype(np.int32)
    blocks = np.full((TOKENS, 512), -1, np.int32)
    for r, n in enumerate(lengths):
        pick = rng.permutation(-(-n // RATIO))[:512]
        blocks[r, : len(pick)] = pick
    rows = rng.normal(size=(logical_tokens, 1, 4, 256)).astype(np.float32)
    # Replicated: logical page p in physical page p + 1 (page 0 is null).
    replicated = np.zeros((entries * dcp + 1, PAGE, 1, 4, 256), np.float32)
    replicated[1:] = rows.reshape(entries * dcp, PAGE, 1, 4, 256)
    table = np.broadcast_to(np.arange(1, entries * dcp + 1, dtype=np.int32), (TOKENS, entries * dcp))
    common = dict(ratio=RATIO, kv_heads=1, query_block=512, bf16_io=True)
    expect = block_sparse_gqa_prefill(
        query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table.copy()),
        jnp.asarray(replicated, jnp.float8_e4m3fn), kv_tile=PAGE, **common,
    )
    outs, lses = [], []
    for rank in range(dcp):
        # Rank r holds token t with (t // PAGE) % dcp == r at (entry t // (dcp*PAGE), t % PAGE).
        local = np.zeros((entries + 1, PAGE, 1, 4, 256), np.float32)
        for e in range(entries):
            page = e * dcp + rank
            local[e + 1] = rows[page * PAGE : (page + 1) * PAGE]
        local_table = np.broadcast_to(np.arange(1, entries + 1, dtype=np.int32), (TOKENS, entries))
        out, lse = block_sparse_gqa_prefill(
            query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(local_table.copy()),
            jnp.asarray(local, jnp.float8_e4m3fn), kv_tile=dcp * PAGE,
            dcp_rank=jnp.asarray([rank], jnp.int32), dcp_world=dcp, return_lse=True, **common,
        )
        outs.append(np.asarray(out, np.float32))
        lses.append(np.asarray(lse, np.float32))
    lse = np.stack(lses)
    top = np.max(lse, axis=0)
    weight = np.where(np.isneginf(lse), 0.0, np.exp(lse - top))
    merged = np.einsum("rth,rthd->thd", weight, np.stack(outs)) / weight.sum(0)[..., None]
    np.testing.assert_allclose(merged, np.asarray(expect, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.tpu_test
@pytest.mark.parametrize("kv_heads,query_heads", [(1, 12), (2, 8)])
def test_interleaved_pcp_ranks_merge_to_the_replicated_prefill(kv_heads, query_heads):
    """PCP on a TP1 stage: 3,200-row pages, units of 128 tokens alternating over
    two ranks (``dcp_interleave``), one 256-token KV tile holding 128 rows of each
    rank. Merging both ranks equals the replicated prefill."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import dcp_owner_row

    page, world, interleave = 3200, 2, 128
    rng = np.random.default_rng(7 + kv_heads)
    entries = 2
    logical_tokens = entries * world * page
    query = jnp.asarray(rng.normal(size=(TOKENS, query_heads, 256)), jnp.bfloat16)
    lengths = np.linspace(1, logical_tokens, TOKENS).astype(np.int32)
    blocks = np.full((TOKENS, 512), -1, np.int32)
    for r, n in enumerate(lengths):
        pick = rng.permutation(-(-n // RATIO))[:512]
        blocks[r, : len(pick)] = pick
    rows = rng.normal(size=(logical_tokens, 1, 4, 256)).astype(np.float32)
    replicated = np.zeros((entries * world + 1, page, 1, 4, 256), np.float32)
    replicated[1:] = rows.reshape(entries * world, page, 1, 4, 256)
    table = np.broadcast_to(
        np.arange(1, entries * world + 1, dtype=np.int32), (TOKENS, entries * world)
    )
    common = dict(ratio=RATIO, kv_heads=kv_heads, query_block=256, bf16_io=True)
    expect = block_sparse_gqa_prefill(
        query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table.copy()),
        jnp.asarray(replicated, jnp.float8_e4m3fn), kv_tile=256, **common,
    )
    t = np.arange(logical_tokens)
    owner, row = (np.asarray(x) for x in dcp_owner_row(t, page, world, interleave))
    outs, lses = [], []
    for rank in range(world):
        local = np.zeros((entries + 1, page, 1, 4, 256), np.float32)
        mine = owner == rank
        local[1 + t[mine] // (world * page), row[mine]] = rows[mine]
        local_table = np.broadcast_to(np.arange(1, entries + 1, dtype=np.int32), (TOKENS, entries))
        out, lse = block_sparse_gqa_prefill(
            query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(local_table.copy()),
            jnp.asarray(local, jnp.float8_e4m3fn), kv_tile=world * interleave,
            dcp_rank=jnp.asarray([rank], jnp.int32), dcp_world=world,
            dcp_interleave=interleave, return_lse=True, **common,
        )
        outs.append(np.asarray(out, np.float32))
        lses.append(np.asarray(lse, np.float32))
    lse = np.stack(lses)
    top = np.max(lse, axis=0)
    weight = np.where(np.isneginf(lse), 0.0, np.exp(lse - np.where(np.isneginf(top), 0, top)))
    total = weight.sum(0)
    merged = np.einsum("rth,rthd->thd", weight, np.stack(outs)) / np.where(total > 0, total, 1)[..., None]
    np.testing.assert_allclose(merged, np.asarray(expect, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "query_lens,contexts",
    [
        # A decode step: four one-token requests deep in their contexts, in a
        # 16-token bucket.
        ([1, 1, 1, 1], [4100, 9000, 12000, 6500]),
        # A prefill step holding two requests' chunks.
        ([256, 128], [3000, 7000]),
    ],
)
def test_interleaved_pcp_multi_request_steps(query_lens, contexts):
    """Several requests in one step through ``block_sparse_gqa_attention``
    (the request-aligning wrapper the served model calls), each with its own
    page table: merging both interleaved PCP ranks equals the replicated
    attention for every request's rows."""
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import dcp_owner_row

    page, world, interleave, heads, kv_heads = 3200, 2, 128, 12, 1
    rng = np.random.default_rng(11)
    reqs = len(query_lens)
    real = sum(query_lens)
    bucket = max(16, -(-real // 16) * 16)
    totals = [c + q for c, q in zip(contexts, query_lens)]
    # Replicated: request r's logical token t at page rep[r][t // page].
    rep_pages = [-(-n // page) for n in totals]
    rep_base = np.concatenate([[1], 1 + np.cumsum(rep_pages)])
    width = max(rep_pages)
    # PCP: a table entry covers world * page tokens; one local page per entry.
    loc_entries = [-(-n // (world * page)) for n in totals]
    loc_base = np.concatenate([[1], 1 + np.cumsum(loc_entries)])
    loc_width = max(loc_entries)
    replicated = np.zeros((rep_base[-1], page, 1, 4, 256), np.float32)
    locals_ = [np.zeros((loc_base[-1], page, 1, 4, 256), np.float32) for _ in range(world)]
    query = np.zeros((bucket, heads, 256), np.float32)
    blocks = np.full((bucket, 512), -1, np.int32)
    lengths = np.zeros((bucket,), np.int32)
    rep_table = np.zeros((bucket, width), np.int32)
    loc_table = np.zeros((bucket, loc_width), np.int32)
    starts = np.zeros((reqs + 1,), np.int32)
    row = 0
    for r, (q, c, n) in enumerate(zip(query_lens, contexts, totals)):
        starts[r + 1] = starts[r] + q
        kv = rng.normal(size=(n, 1, 4, 256)).astype(np.float32)
        t = np.arange(n)
        replicated[rep_base[r] + t // page, t % page] = kv
        owner, local_row = (np.asarray(x) for x in dcp_owner_row(t, page, world, interleave))
        for rank in range(world):
            mine = owner == rank
            locals_[rank][loc_base[r] + t[mine] // (world * page), local_row[mine]] = kv[mine]
        for i in range(q):
            position = c + i
            query[row] = rng.normal(size=(heads, 256))
            lengths[row] = position + 1
            pick = rng.permutation(-(-(position + 1) // RATIO))[:512]
            blocks[row, : len(pick)] = pick
            rep_table[row, : rep_pages[r]] = rep_base[r] + np.arange(rep_pages[r])
            loc_table[row, : loc_entries[r]] = loc_base[r] + np.arange(loc_entries[r])
            row += 1
    gate = jnp.zeros((bucket, heads, 256), jnp.bfloat16)
    common = dict(ratio=RATIO, kv_heads=kv_heads, query_block=256, return_lse=True)
    q = jnp.asarray(query, jnp.bfloat16)
    expect, expect_lse, _ = block_sparse_gqa_attention(
        q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(rep_table),
        jnp.asarray(replicated, jnp.float8_e4m3fn), jnp.asarray(starts),
        kv_tile=256, **common,
    )
    outs, lses = [], []
    for rank in range(world):
        out, lse, _ = block_sparse_gqa_attention(
            q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(loc_table),
            jnp.asarray(locals_[rank], jnp.float8_e4m3fn), jnp.asarray(starts),
            kv_tile=world * interleave,
            dcp=(jnp.asarray([rank], jnp.int32), world, interleave), **common,
        )
        outs.append(np.asarray(out, np.float32))
        lses.append(np.asarray(lse, np.float32))
    lse = np.stack(lses)
    top = np.max(lse, axis=0)
    weight = np.where(np.isneginf(lse), 0.0, np.exp(lse - np.where(np.isneginf(top), 0, top)))
    total = weight.sum(0)
    merged = np.einsum("rth,rthd->thd", weight, np.stack(outs)) / np.where(total > 0, total, 1)[..., None]
    live = lengths > 0
    np.testing.assert_allclose(
        merged[live], np.asarray(expect, np.float32)[live], rtol=2e-2, atol=2e-2
    )
