# SPDX-License-Identifier: Apache-2.0
"""Packed compressed keys: slot ``s`` of a shared page equals an unpacked page.

The Pallas writers run in TPU interpret mode on CPU. Interpret mode cannot
bitcast refs, so the carrier is the BF16 pool (two BF16 rows per pair), whose
pair addressing is the FP8 carrier's.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

RATIO = 4
PAGES, TOKENS_PER_PAGE, WIDTH = 7, 512, 256
# One layer's compressed keys for a manager page: 512 / 4 rows = 64 pairs.
LAYER_PAIRS = TOKENS_PER_PAGE // (2 * RATIO)
PAIRS_PER_PAGE = TOKENS_PER_PAGE
SLOTS = PAIRS_PER_PAGE // LAYER_PAIRS


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(
        pooled_qsa.pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )


def region(pool, slot):
    pairs = np.asarray(pool).reshape(PAGES, PAIRS_PER_PAGE, 2, WIDTH)
    return pairs[:, slot * LAYER_PAIRS : (slot + 1) * LAYER_PAIRS]


def with_region0(pool, slot):
    """``pool`` whose slot-0 bytes are its slot-``slot`` bytes."""
    pairs = np.asarray(pool).reshape(PAGES, PAIRS_PER_PAGE, 2, WIDTH).copy()
    pairs[:, :LAYER_PAIRS] = region(pool, slot)
    return jnp.asarray(pairs.reshape(np.shape(pool)))


def step(starts, context, tokens):
    """Per-token request/logical positions of a ragged step."""
    starts = np.asarray(starts)
    req = np.zeros(tokens, np.int32)
    logical = np.zeros(tokens, np.int32)
    for r in range(len(context)):
        span = range(starts[r], starts[r + 1])
        req[span.start : span.stop] = r
        logical[span.start : span.stop] = context[r] + np.arange(len(span))
    valid = np.arange(tokens) < starts[-1]
    complete = valid & (logical % RATIO == RATIO - 1)
    return (
        jnp.asarray(complete),
        jnp.asarray(req),
        jnp.asarray(logical),
        jnp.asarray(starts, jnp.int32),
        jnp.asarray(context, jnp.int32),
    )


CASES = {
    # 128-token multiples take the tiled page writer, page-crossing included.
    "tiled_prefill": ([0, 128, 384], [0, 384], 384),
    # Odd offsets start and end on half-written pairs (read/modify/write).
    "partial_pairs": ([0, 37, 100, 101], [4, 509, 3], 104),
    "decode": ([0, 1, 2, 3, 3], [3, 7, 1022, 0], 8),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_packed_write_matches_unpacked_per_layer(interpret, case):
    starts, context, tokens = CASES[case]
    complete, req, logical, starts, context = step(starts, context, tokens)
    rng = np.random.default_rng(len(case))
    initial = jnp.asarray(
        rng.normal(size=(PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH)), jnp.bfloat16
    )
    table = jnp.asarray([[3, 5, 0], [1, 6, 2], [4, 0, 0], [0, 0, 0]], jnp.int32)
    table = table[: len(context)]
    slots = (0, 1, SLOTS - 1)
    keys = [jnp.asarray(rng.normal(size=(tokens, 128)), jnp.bfloat16) for _ in slots]
    write = jax.jit(pooled_qsa.write_compressed, static_argnames=("ratio", "slot"))
    packed = initial
    for slot, k in zip(slots, keys):
        packed = write(
            packed,
            k,
            complete,
            req,
            logical,
            starts,
            context,
            table,
            ratio=RATIO,
            slot=slot,
        )
    for slot, k in zip(slots, keys):
        unpacked = write(
            with_region0(initial, slot),
            k,
            complete,
            req,
            logical,
            starts,
            context,
            table,
            ratio=RATIO,
        )
        np.testing.assert_array_equal(region(packed, slot), region(unpacked, 0))
        # The unpacked writer touches only slot 0 of its pages.
        for other in range(1, SLOTS):
            np.testing.assert_array_equal(
                region(unpacked, other), region(with_region0(initial, slot), other)
            )
    for slot in set(range(SLOTS)) - set(slots):
        np.testing.assert_array_equal(region(packed, slot), region(initial, slot))
    assert bool(complete.any())
    assert not np.array_equal(np.asarray(packed), np.asarray(initial))


def test_unpacked_write_is_unchanged_by_slot_zero(interpret):
    starts, context, tokens = CASES["partial_pairs"]
    complete, req, logical, starts, context = step(starts, context, tokens)
    rng = np.random.default_rng(3)
    pool = jnp.asarray(
        rng.normal(size=(PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH)), jnp.bfloat16
    )
    keys = jnp.asarray(rng.normal(size=(tokens, 128)), jnp.bfloat16)
    table = jnp.asarray([[3, 5, 0], [1, 6, 2], [4, 0, 0]], jnp.int32)
    args = (pool, keys, complete, req, logical, starts, context, table, RATIO)
    lower = jax.jit(pooled_qsa.write_compressed, static_argnums=(8, 9)).lower
    assert lower(*args).as_text() == lower(*args, 0).as_text()


def gather_pages(query, scale, cache, lengths, pages, *args, **kwargs):
    """Stand-in for the TPU top-k: expose exactly the key pages it was given."""
    return cache[pages]


@pytest.mark.parametrize("prefill", [False, True])
def test_packed_selection_reads_the_layer_slot(monkeypatch, interpret, prefill):
    monkeypatch.setattr(pooled_qsa, "streamindex_topk", gather_pages)
    # The decode chunk bound reads TPU scalar memory; one chunk suffices here.
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    rng = np.random.default_rng(11)
    pool = jnp.asarray(
        rng.normal(size=(PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH)), jnp.bfloat16
    )
    table = jnp.asarray([[3, 5], [1, 6], [4, 2]], jnp.int32)
    requests = len(table)
    tokens = 2 * requests if prefill else requests
    starts = jnp.arange(requests + 1, dtype=jnp.int32) * (tokens // requests)
    context = jnp.asarray([5, 600, 900], jnp.int32)
    req = jnp.repeat(jnp.arange(requests, dtype=jnp.int32), tokens // requests)
    query = jnp.asarray(rng.normal(size=(tokens, 4, 128)), jnp.bfloat16)
    lengths = context[req] + 1
    select = functools.partial(
        pooled_qsa.select_blocks,
        query,
        lengths=lengths,
        table=table,
        req=req,
        starts=starts,
        context=context,
        ratio=RATIO,
        block_topk=16,
    )
    for slot in (1, 3, SLOTS - 1):
        packed = select(pool=pool, slot=slot)
        unpacked = select(pool=with_region0(pool, slot))
        np.testing.assert_array_equal(np.asarray(packed), np.asarray(unpacked))
    np.testing.assert_array_equal(
        np.asarray(select(pool=pool)), np.asarray(select(pool=pool, slot=0))
    )
    with pytest.raises(AssertionError, match="slot exceeds"):
        select(pool=pool, slot=SLOTS)
