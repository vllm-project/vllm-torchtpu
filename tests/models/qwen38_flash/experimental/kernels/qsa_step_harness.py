# SPDX-License-Identifier: Apache-2.0
"""Pooled QSA step harness: one QSA layer over a random pool in Pallas
interpret mode on CPU (the legacy evaluator bitcasts refs low byte first like
the TPU kernels); the indexer and attention kernels, which cannot run in
interpret mode and only read the pool, are stubbed out by ``interpret``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa

PAGES, PAGE_ROWS, WIDTH = 40, 512, 256
HEAD_DIM, INDEX_DIM, HEADS, INDEX_HEADS = 256, 128, 2, 2
RATIO, TOPK, ROTARY = 4, 64, 64


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)

    def no_selection(query, *args, block_topk, **kwargs):
        return jnp.zeros((len(query), block_topk), jnp.int32)

    def no_attention(q, gate, blocks, lengths, table, pool, *args, **kwargs):
        return jnp.zeros(q.shape, jnp.bfloat16), pool

    monkeypatch.setattr(pooled_qsa, "select_blocks", no_selection)
    monkeypatch.setattr(pooled_qsa, "qsa_attention", no_attention)


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def make_step(rng, query_lens, contexts, dtype):
    """One pooled QSA layer's operands; every request owns distinct pages."""
    requests = len(contexts)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    tokens = int(starts[-1])
    padded = max(8, -(-tokens // 8) * 8)
    columns = 4
    free = iter(rng.permutation(np.arange(1, PAGES)))
    main = np.zeros((requests, columns), np.int32)
    comp = np.zeros((requests, columns), np.int32)
    ring = np.zeros((requests, 8), np.int32)
    for r in range(requests):
        if query_lens[r]:
            last = (contexts[r] + query_lens[r] - 1) // PAGE_ROWS
            for c in range(last + 1):
                main[r, c] = next(free)
                comp[r, c] = next(free)
            ring[r, 0] = next(free)
    positions = np.zeros(padded, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    if dtype == jnp.float8_e4m3fn:
        pool = jnp.asarray(rng.normal(0, 1, (PAGES, PAGE_ROWS, 1, 4, WIDTH)), dtype)
    else:
        pool = normal(PAGES, PAGE_ROWS, 1, 2, WIDTH)
    rope = jnp.pad(normal(4096, ROTARY), ((0, 0), (0, 128 - ROTARY)))
    args = (
        normal(padded, 2 * HEADS * HEAD_DIM),
        normal(padded, HEAD_DIM),
        normal(padded, HEAD_DIM),
        normal(padded, (INDEX_HEADS + 1) * INDEX_DIM),
        normal(HEAD_DIM),
        normal(HEAD_DIM),
        normal(INDEX_DIM),
        normal(INDEX_DIM),
        jnp.asarray(positions),
        rope,
        pool,
        jnp.asarray(starts),
        jnp.asarray(contexts, jnp.int32),
        jnp.asarray(ring.reshape(-1)),
        jnp.asarray(main.reshape(-1)),
        jnp.asarray(comp.reshape(-1)),
    )
    return args


def run(args, **options):
    step = jax.jit(
        pooled_qsa.pooled_qsa_step,
        static_argnames=(
            "compress_ratio",
            "token_topk",
            "norm_eps",
            "rotary_dim",
            "k_scale",
            "v_scale",
            "compressed_slot",
            "main_slot",
            "max_query_len",
        ),
    )
    return step(
        *args,
        compress_ratio=RATIO,
        token_topk=TOPK,
        norm_eps=1e-6,
        rotary_dim=ROTARY,
        **options,
    )


CASES = {
    # c8 MTP verify: four tokens per request, two idle request rows.
    "verify": ([4, 4, 4, 0, 4, 4, 0, 4], [37, 0, 509, 0, 1022, 6, 0, 2]),
    # Plain decode, contexts on every ring/pair phase.
    "decode": ([1] * 8, [0, 1, 2, 3, 4, 7, 511, 1023]),
    # Ragged verify (partially filled rows, a page crossing).
    "ragged": ([2, 4, 1, 3], [510, 3, 8, 126]),
}

LAYOUTS = {
    "fp8_paired_slot0": (jnp.float8_e4m3fn, {"main_slot": 0}),
    "fp8_paired_slot1_packed_k": (
        jnp.float8_e4m3fn,
        {"main_slot": 1, "compressed_slot": 2},
    ),
    "fp8_whole_rows": (jnp.float8_e4m3fn, {}),
    "bf16_whole_rows": (jnp.bfloat16, {"compressed_slot": 1}),
}
