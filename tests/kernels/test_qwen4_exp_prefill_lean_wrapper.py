# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER: the DCP sparse prefill with the lean
metadata gives the same (out, lse) bits as the served wrapper, on both ranks,
for ascending and unordered selections, with and without the ``ascending``
hint (the hint only picks the ranged loop; interpret mode)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)

PAGE, HEADS, DIM, RATIO, D = 512, 3, 256, 4, 2


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def make_case(rng, pieces, budget, *, ascending, extra_cols):
    lens = [n for _, n in pieces]
    tokens = sum(lens)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    # Unused table entries past the context (the served table spans max_model_len).
    cols = max(c + n for c, n in pieces) // (D * PAGE) + 2 + extra_cols
    table = np.zeros((tokens, cols), np.int32)
    free = rng.permutation(np.arange(len(pieces) * cols + 4))
    blocks = np.full((tokens, budget), -1, np.int32)
    lengths = np.zeros(tokens, np.int32)
    i = 0
    for r, (ctx, n) in enumerate(pieces):
        table[starts[r] : starts[r + 1]] = free[r * cols : (r + 1) * cols]
        for t in range(n):
            complete = (ctx + t + 1) // RATIO
            pick = rng.choice(complete, size=min(budget, complete), replace=False)
            pick = np.sort(pick) if ascending else pick
            blocks[i, : len(pick)] = pick
            lengths[i] = ctx + t + 1
            i += 1
    cache = jnp.asarray(
        rng.normal(0, 1, (len(pieces) * cols + 4, PAGE, 1, 4, DIM)), jnp.float8_e4m3fn
    )
    query = jnp.asarray(rng.normal(0, 1, (tokens, HEADS, DIM)), jnp.bfloat16)
    return (
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
        jnp.asarray(starts),
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "pieces,ascending,hint,extra_cols",
    [
        ([(1500, 256)], True, True, 0),
        ([(1500, 256)], False, False, 9),
        ([(1500, 256)], False, True, 0),
        ([(700, 200), (2047, 56)], True, True, 9),
    ],
)
def test_lean_wrapper_matches_served(
    interpret, monkeypatch, rank, pieces, ascending, hint, extra_cols
):
    query, blocks, lengths, table, cache, starts = make_case(
        np.random.default_rng(2), pieces, 24, ascending=ascending, extra_cols=extra_cols
    )
    gate = jnp.zeros_like(query)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_BITSET_OWNED", False)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_BITSET_RANGED", False)

    def attend(lean):
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER", lean)
        jax.clear_caches()
        return jax.jit(
            lambda q, c: block_sparse_gqa_attention(
                q, gate, blocks, lengths, table, c, starts, kv_heads=1, ratio=RATIO,
                k_scale=0.5, v_scale=2.0, query_block=256, kv_tile=PAGE,
                query_tile=256, kv_head_offset=1,
                dcp=(jnp.asarray([rank], jnp.int32), D), return_lse=True,
                ascending=hint,
            )[:2]
        )(query, cache)

    want, got = attend(False), attend(True)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)),
            np.asarray(jax.lax.bitcast_convert_type(b, jnp.uint32)),
        )
