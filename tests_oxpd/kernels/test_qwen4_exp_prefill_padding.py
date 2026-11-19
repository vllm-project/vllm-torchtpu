# SPDX-License-Identifier: Apache-2.0
"""QSA prefill keeps mixed spec-decode steps in their own token bucket.

The block-sparse core is replaced by a row-wise stand-in that poisons every
row whose query block mixes requests, so the tests check routing and the
block-aligned layout (not attention numerics, which are covered on TPU).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import (
    block_sparse_gqa_prefill as bsp,
)
from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.request_types import (
    short_request_prefix_tokens,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

QUERY_BLOCK = 8
HEADS, DIM = 2, 4


def _rowwise(query, lengths, table):
    value = query.astype(jnp.float32) * (lengths[:, None, None] + 1)
    return value + table[:, :1, None].astype(jnp.float32)


def _gated(output, gate):
    return bf16(
        bf16(output).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )


@pytest.fixture
def executed(monkeypatch):
    """Row-wise core; records the token count of every executed call."""
    calls = []

    def core(query, blocks, lengths, page_table, cache, *, query_block, **_):
        assert len(query) % query_block == 0
        live = (lengths > 0).reshape(-1, query_block)
        tables = page_table.reshape(-1, query_block, page_table.shape[-1])
        same = jnp.all(tables == tables[:, :1], axis=-1)
        # The kernel uses one page-table row per query block.
        mixed = jnp.any(live & ~same, axis=1)
        jax.debug.callback(lambda n: calls.append(int(n)), len(query))
        output = _rowwise(query, lengths, page_table)
        poison = jnp.repeat(mixed, query_block)[:, None, None]
        return jnp.where(poison, jnp.nan, output)

    monkeypatch.setattr(bsp, "block_sparse_gqa_prefill", core)
    return calls


def _inputs(real, tokens, seed=0):
    rng = np.random.default_rng(seed)
    starts = np.r_[0, np.cumsum(real)].astype(np.int32)
    req = np.searchsorted(starts[1:], np.arange(tokens), side="right")
    valid = np.arange(tokens) < starts[-1]
    req = np.minimum(req, len(real) - 1)
    lengths = np.where(valid, 100 + np.arange(tokens) - starts[req], 0)
    table = np.where(valid, 10 * (req + 1), 0)[:, None].repeat(3, axis=1)
    query = rng.integers(-4, 5, (tokens, HEADS, DIM)).astype(np.float32)
    gate = rng.standard_normal((tokens, HEADS, DIM)).astype(np.float32)
    return (
        jnp.asarray(query, jnp.bfloat16),
        jnp.asarray(gate, jnp.bfloat16),
        jnp.zeros((tokens, 4), jnp.int32),
        jnp.asarray(lengths, jnp.int32),
        jnp.asarray(table, jnp.int32),
        jnp.asarray(starts),
    )


def _expected(query, gate, lengths, table):
    output = _gated(_rowwise(query, lengths, table), gate)
    return np.asarray(jnp.where((lengths > 0)[:, None, None], output, 0))


@pytest.mark.parametrize(
    "real,tokens,expected_tokens",
    [
        # A prefill chunk after an RPA-routed decode prefix; the block-aligned
        # layout fits the bucket instead of doubling it.
        ([0, 0, 0, 50, 0, 0, 0, 0], 64, 64),
        # Two chunks that still fit after alignment.
        ([20, 30, 0, 0], 64, 64),
        # Already aligned: no relayout.
        ([16, 32, 16, 0], 64, 64),
        # An unaligned bucket rounds up to the next block only.
        ([50, 0], 60, 64),
        # A few blocks over the bucket: the intermediate layout.
        ([9, 9, 9, 9, 9, 9, 0, 0], 64, 64 + 4 * QUERY_BLOCK),
        # Too many short requests: fall back to the worst-case layout.
        ([9, 9, 9, 9, 9, 9, 9, 0], 64, 120),
        ([0, 0, 0, 0], 64, 64),
    ],
)
def test_block_aligned_layout_uses_bucket_when_it_fits(
    executed, real, tokens, expected_tokens
):
    query, gate, blocks, lengths, table, starts = _inputs(real, tokens)
    output, _ = jax.jit(
        lambda *args: bsp.block_sparse_gqa_attention(
            *args, kv_heads=1, ratio=4, query_block=QUERY_BLOCK, kv_tile=128
        )
    )(query, gate, blocks, lengths, table, jnp.zeros((1,)), starts)
    output.block_until_ready()
    assert executed == [expected_tokens]
    np.testing.assert_array_equal(
        np.asarray(output), _expected(query, gate, lengths, table)
    )


@pytest.mark.parametrize(
    "lengths,capacity,tokens",
    [
        ([4, 4, 4, 50], 16, 12),
        ([1, 1, 128, 0], 8, 2),
        ([1, 4, 1, 0], 8, 6),
        ([4, 4, 4, 50], 10, 8),
        ([128, 1], 8, 0),
        ([0, 4], 8, 0),
        ([17, 4], 32, 0),
        ([16, 4], 32, 20),
        ([4, 4], 8, 8),
        ([0, 0], 8, 0),
    ],
)
def test_short_request_prefix_tokens(lengths, capacity, tokens):
    starts = jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32)
    assert int(short_request_prefix_tokens(starts, capacity)) == tokens


@pytest.mark.parametrize(
    "real,rpa_rows,prefill_tokens",
    [
        # MTP K=3 verify rows ahead of a prefill chunk: verify rows go to RPA,
        # the chunk keeps the bucket (previously everything was padded).
        ([4, 4, 4, 41], 12, 64),
        # One-token decode rows, as before.
        ([1, 1, 1, 40], 3, 64),
        # Verify rows beyond the RPA window (16 request slots) stay on
        # prefill; aligned, they still fit the bucket.
        ([4, 4, 4, 4, 4, 4, 30], 16, 64),
        # Here they do not: the intermediate layout.
        ([4, 4, 4, 4] + [1] * 9, 16, 96),
        # A long first request leaves every short row on prefill.
        ([17] + [1] * 15, 0, 176),
        # Pure prefill.
        ([30, 20], 0, 64),
        # A pure verify step runs on RPA alone.
        ([4, 4], 8, 64),
    ],
)
def test_mixed_step_routes_short_requests_to_rpa(
    monkeypatch, executed, real, rpa_rows, prefill_tokens
):
    rpa_live = []

    def rpa(q, gate, blocks, lengths, table, cache, **_):
        jax.debug.callback(lambda n: rpa_live.append(int(n)), jnp.sum(lengths > 0))
        output = _gated(_rowwise(q, lengths, table), gate)
        return jnp.where((lengths > 0)[:, None, None], output, 0), cache

    monkeypatch.setattr(qsa_rpa, "rpa_attention", rpa)
    monkeypatch.setattr(
        prefill_tiles,
        "get_sparse_prefill_tiles",
        lambda *a, **k: (QUERY_BLOCK, 128, None),
    )
    real = real + [0] * (16 - len(real))
    query, gate, blocks, lengths, table, starts = _inputs(real, 64, seed=1)
    output, _ = jax.jit(
        lambda *args: qsa_rpa.qsa_attention(*args, kv_heads=1, ratio=4)
    )(query, gate, blocks, lengths, table, jnp.zeros((1,)), starts)
    output.block_until_ready()
    assert rpa_live == [rpa_rows]
    assert executed == [prefill_tokens]
    np.testing.assert_array_equal(
        np.asarray(output), _expected(query, gate, lengths, table)
    )
