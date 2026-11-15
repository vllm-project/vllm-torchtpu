# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_ALIGN_GATHER: the block-aligned layout built by a row
gather is bit-identical to the ``.at[slots].set`` scatter layout.

Part 1 swaps the block-sparse core for a recorder and compares the aligned
query / blocks / lengths / page_table it receives, plus the wrapper output,
for ragged multi-request mixes in the 1024 / 384 / 256 buckets (and token
counts that are not a multiple of query_block), with
VLLM_TPU_SORTED_SEARCH_MASK off and on (all four combinations). Part 2 runs
the real kernel in interpret mode, on both DCP ranks.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import sorted_search
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp

COMBOS = list(itertools.product([False, True], [False, True]))  # (gather, sorted)


def _bits(x):
    x = np.asarray(x)
    if x.dtype == jnp.bfloat16:
        return x.view(np.uint16)
    if x.dtype == np.float32:
        return x.view(np.uint32)
    return x


def _assert_same(a, b):
    a, b = _bits(a), _bits(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def _set_flags(monkeypatch, gather, sorted_mask):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_ALIGN_GATHER", gather)
    monkeypatch.setattr(envs, "VLLM_TPU_SORTED_SEARCH_MASK", sorted_mask)
    jax.clear_caches()


@pytest.fixture
def recorder(monkeypatch):
    """Row-wise core that records the aligned inputs of the executed branch."""
    seen = []

    def core(query, blocks, lengths, page_table, cache, *, query_block, **kw):
        assert len(query) % query_block == 0

        def keep(*arrays):
            seen.append([np.asarray(a) for a in arrays])

        jax.debug.callback(keep, query, blocks, lengths, page_table)
        value = query.astype(jnp.float32) * (lengths[:, None, None] + 1)
        value = value + page_table[:, :1, None].astype(jnp.float32)
        value = value + blocks[:, :1, None].astype(jnp.float32) * 0.5
        if kw.get("return_lse"):
            lse = jnp.where(lengths[:, None] > 0, value[:, :, 0], -jnp.inf)
            return value, lse
        return value

    monkeypatch.setattr(bsp, "block_sparse_gqa_prefill", core)
    return seen


def _inputs(real, tokens, seed, heads=6, dim=16, budget=8):
    rng = np.random.default_rng(seed)
    starts = np.r_[0, np.cumsum(real)].astype(np.int32)
    assert starts[-1] <= tokens
    req = np.minimum(np.searchsorted(starts[1:], np.arange(tokens), side="right"), len(real) - 1)
    valid = np.arange(tokens) < starts[-1]
    lengths = np.where(valid, 100 + np.arange(tokens) - starts[req], 0).astype(np.int32)
    table = rng.integers(1, 1000, (tokens, 5)).astype(np.int32)
    blocks = rng.integers(-1, 50, (tokens, budget)).astype(np.int32)
    query = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
    # Signed zeros and NaN payloads must survive the gather bit for bit.
    query[::7, 0, 0] = -0.0
    query[3::11, 1, 1] = np.nan
    return (
        jnp.asarray(query, jnp.bfloat16),
        jnp.zeros((tokens, heads, dim), jnp.bfloat16),
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        jnp.asarray(starts),
    )


# (per-request real lengths, token bucket)
MIXES = [
    ([1024], 1024),  # one aligned request (aligned branch unless qb > 1024)
    ([1000], 1024),  # single chunk, tail padding
    ([5, 300, 0, 77, 1, 129, 400], 1024),  # ragged, an empty request
    ([1] * 24 + [500], 1024),  # many decode-like rows + one chunk
    ([33] * 30, 1024),  # every request wastes query_block - 1 rows
    ([17, 0, 0, 200, 3], 256),
    ([16] * 16, 256),
    ([255], 256),
    ([100, 101, 102], 384),
    ([1] * 32 + [300], 384),
    ([7, 250, 0], 300),  # bucket not a multiple of query_block
    ([999], 1000),
    ([3, 5, 11, 13, 17, 19, 23, 29, 31, 37], 200),
]


@pytest.mark.parametrize("query_block", [32, 256])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("mix", MIXES, ids=lambda m: f"{len(m[0])}req-{m[1]}tok")
def test_aligned_inputs_and_output_match(recorder, monkeypatch, mix, query_block, return_lse):
    real, tokens = mix
    query, gate, blocks, lengths, table, starts = _inputs(real, tokens, seed=len(real) + tokens)
    results = {}
    for gather, sorted_mask in COMBOS:
        _set_flags(monkeypatch, gather, sorted_mask)
        recorder.clear()
        out = jax.jit(
            lambda q, b, n, t, s: bsp.block_sparse_gqa_attention(
                q, gate, b, n, t, None, s, kv_heads=1, ratio=4,
                query_block=query_block, return_lse=return_lse,
            )
        )(query, blocks, lengths, table, starts)
        jax.effects_barrier()
        assert len(recorder) == 1, len(recorder)
        results[gather, sorted_mask] = (recorder[0], jax.tree.leaves(out[:2] if return_lse else out))
    base_inputs, base_out = results[False, False]
    for key, (inputs, out) in results.items():
        for a, b in zip(inputs, base_inputs):
            _assert_same(a, b)
        for a, b in zip(out, base_out):
            _assert_same(a, b)


@pytest.mark.parametrize("mix", MIXES[:6], ids=lambda m: f"{len(m[0])}req-{m[1]}tok")
def test_scatter_inverse_fallback_matches(recorder, monkeypatch, mix):
    """Above MAX_MASK_ELEMENTS the inverse map comes from an int32 scatter."""
    monkeypatch.setattr(sorted_search, "MAX_MASK_ELEMENTS", 1)
    real, tokens = mix
    query, gate, blocks, lengths, table, starts = _inputs(real, tokens, seed=3)
    got = []
    for gather in (False, True):
        _set_flags(monkeypatch, gather, False)
        recorder.clear()
        out = jax.jit(
            lambda q, s: bsp.block_sparse_gqa_attention(
                q, gate, blocks, lengths, table, None, s, kv_heads=1, ratio=4, query_block=32
            )
        )(query, starts)
        jax.effects_barrier()
        got.append((recorder[0], jax.tree.leaves(out)))
    for a, b in zip(got[0][0], got[1][0]):
        _assert_same(a, b)
    for a, b in zip(got[0][1], got[1][1]):
        _assert_same(a, b)


@pytest.mark.parametrize("max_tokens", [256, 288, 1024, 1056])
def test_aligned_sources_vs_scatter(max_tokens):
    """Direct check of the inverse map, including rows dropped past max_tokens."""
    rng = np.random.default_rng(max_tokens)
    for _ in range(40):
        nreq = int(rng.integers(1, 40))
        real = rng.integers(0, 80, nreq) * (rng.random(nreq) < 0.8)
        tokens = int(real.sum() + rng.integers(0, 40))
        tokens = max(tokens, 1)
        starts = np.r_[0, np.cumsum(real)].astype(np.int32)
        padded = (real + 31) // 32 * 32
        offsets = np.r_[0, np.cumsum(padded)].astype(np.int32)
        req = np.minimum(np.searchsorted(starts[1:], np.arange(tokens), side="right"), nreq - 1)
        valid = np.arange(tokens) < starts[-1]
        slots = np.where(valid, offsets[req] + np.arange(tokens) - starts[req], max_tokens)
        want = np.full(max_tokens, -1)
        keep = slots < max_tokens
        want[slots[keep]] = np.arange(tokens)[keep]
        for limit in (sorted_search.MAX_MASK_ELEMENTS, 1):
            sorted_search.MAX_MASK_ELEMENTS, old = limit, sorted_search.MAX_MASK_ELEMENTS
            try:
                source, filled = bsp._aligned_sources(
                    jnp.asarray(starts), jnp.asarray(offsets), jnp.asarray(real, jnp.int32),
                    tokens, max_tokens,
                )
            finally:
                sorted_search.MAX_MASK_ELEMENTS = old
            np.testing.assert_array_equal(np.where(filled, source, -1), want)


# ---- Part 2: the real kernel in interpret mode, DCP ranks 0 and 1 ----

PAGE, HEADS, DIM, RATIO, D = 512, 3, 256, 4, 2


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _kernel_case(rng, pieces, budget, extra_rows=0):
    lens = [n for _, n in pieces]
    tokens = sum(lens) + extra_rows
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    cols = max(c + n for c, n in pieces) // (D * PAGE) + 2
    table = np.zeros((tokens, cols), np.int32)
    free = rng.permutation(np.arange(len(pieces) * cols + 4))
    blocks = np.full((tokens, budget), -1, np.int32)
    lengths = np.zeros(tokens, np.int32)
    i = 0
    for r, (ctx, n) in enumerate(pieces):
        table[starts[r] : starts[r + 1]] = free[r * cols : (r + 1) * cols]
        for t in range(n):
            complete = (ctx + t + 1) // RATIO
            pick = np.sort(rng.choice(complete, size=min(budget, complete), replace=False))
            blocks[i, : len(pick)] = pick
            lengths[i] = ctx + t + 1
            i += 1
    cache = jnp.asarray(
        rng.normal(0, 1, (len(pieces) * cols + 4, PAGE, 1, 4, DIM)), jnp.float8_e4m3fn
    )
    query = jnp.asarray(rng.normal(0, 1, (tokens, HEADS, DIM)), jnp.bfloat16)
    return (query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache,
            jnp.asarray(starts))


@pytest.mark.parametrize("dcp_rank", [None, 0, 1])
@pytest.mark.parametrize(
    "pieces,extra",
    [([(700, 200), (2047, 56)], 0), ([(1100, 250)], 0), ([(900, 40), (300, 3), (1500, 90)], 123)],
)
def test_kernel_output_matches(interpret, monkeypatch, pieces, extra, dcp_rank):
    query, blocks, lengths, table, cache, starts = _kernel_case(
        np.random.default_rng(5), pieces, 24, extra
    )
    gate = jnp.zeros_like(query)
    kw = {}
    if dcp_rank is not None:
        kw = {"dcp": (jnp.asarray([dcp_rank], jnp.int32), D), "return_lse": True}
    results = []
    for gather, sorted_mask in COMBOS:
        _set_flags(monkeypatch, gather, sorted_mask)
        out = jax.jit(lambda q, c: bsp.block_sparse_gqa_attention(
            q, gate, blocks, lengths, table, c, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.5, v_scale=2.0, query_block=256, kv_tile=PAGE, query_tile=256,
            kv_head_offset=1, **kw,
        ))(query, cache)
        results.append(jax.tree.leaves(out[:2] if dcp_rank is not None else out))
    for got in results[1:]:
        for a, b in zip(got, results[0]):
            _assert_same(a, b)
