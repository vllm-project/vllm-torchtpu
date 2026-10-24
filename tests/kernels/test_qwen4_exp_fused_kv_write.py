# SPDX-License-Identifier: Apache-2.0
"""The fused decode/verify QSA cache writer leaves the pool byte-identical.

``VLLM_TPU_QSA_FUSED_KV_WRITE`` replaces the main KV, compressed-key and
raw ring/tag writers (and the ring read) of a decode-sized pooled QSA step
with one in-place kernel. Both paths run one QSA layer over the same random
pool in Pallas interpret mode on CPU (the legacy evaluator bitcasts refs low
byte first like the TPU kernels); the indexer and attention kernels, which
cannot run in interpret mode and only read the pool, are stubbed out.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa

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


def run(monkeypatch, fused, args, **options):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", fused)
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


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_fused_write_matches_separate_writers(interpret, monkeypatch, case, layout):
    dtype, options = LAYOUTS[layout]
    if dtype == jnp.float8_e4m3fn:
        options = {**options, "k_scale": 0.375, "v_scale": 2.5}
    query_lens, contexts = CASES[case]
    rng = np.random.default_rng(len(case) * 7 + len(layout))
    args = make_step(rng, query_lens, contexts, dtype)
    options["max_query_len"] = 4
    base_pool, base_out, base_sel = run(monkeypatch, False, args, **options)
    pool, out, sel = run(monkeypatch, True, args, **options)
    assert not np.array_equal(raw(base_pool), raw(args[10])), "nothing written"
    np.testing.assert_array_equal(raw(pool), raw(base_pool))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(base_out))
    np.testing.assert_array_equal(np.asarray(sel), np.asarray(base_sel))


def test_fused_path_is_taken_for_decode_only(interpret, monkeypatch):
    calls = []
    original = fused_kv_write.fused_layer_write

    def recording(*args, **kwargs):
        calls.append(len(kwargs["main_rows"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", recording)
    rng = np.random.default_rng(3)
    decode = make_step(rng, [4, 4], [5, 9], jnp.float8_e4m3fn)
    run(monkeypatch, True, decode, main_slot=1, max_query_len=4)
    assert calls == [8]
    prefill = make_step(rng, [40, 4], [0, 9], jnp.float8_e4m3fn)
    run(monkeypatch, True, prefill, main_slot=1, max_query_len=4)
    assert calls == [8]


def test_duplicate_main_rows_keep_last_writer(interpret):
    """Repeated destinations keep the later row and the partner's bytes."""
    rng = np.random.default_rng(5)
    pool = jnp.asarray(rng.normal(0, 1, (4, PAGE_ROWS, 1, 4, WIDTH)), jnp.float8_e4m3fn)
    tokens = 16
    # pack_rows layout: K and V in head slots 0 and 1, zero padding slots.
    rows = rng.normal(0, 1, (tokens, 1, 4, WIDTH))
    rows[:, :, 2:] = 0
    rows = jnp.asarray(rows, jnp.float8_e4m3fn)
    page = jnp.full((tokens,), 2, jnp.int32)
    offset = jnp.asarray(np.arange(tokens) % 5 + 3, jnp.int32)
    valid = jnp.arange(tokens) < 12
    zeros = jnp.zeros((tokens,), jnp.int32)
    # Rows enter as an argument: XLA:CPU constant folding of FP8 bitcasts
    # is not byte-exact.
    write = jax.jit(
        lambda pool, rows: fused_kv_write.fused_layer_write(
            pool,
            main_rows=pooled_qsa.main_slot_rows(rows, 1),
            main_page=page,
            main_offset=offset,
            main_valid=valid,
            main_slot=1,
            comp_pair=zeros,
            comp_write=zeros,
            comp_mask=jnp.zeros((tokens, 2), jnp.int32),
            comp_values=jnp.zeros((tokens, 2, INDEX_DIM), jnp.bfloat16),
            ring=jnp.zeros((8, 1024), jnp.bfloat16),
            tags=jnp.zeros((8, 16), jnp.int32),
            ring_ids=jnp.zeros((8,), jnp.int32),
        )
    )
    expected = raw(pool).copy()
    for i in range(tokens):
        if valid[i]:
            expected[2, int(offset[i]), :, 2:] = raw(rows)[i, :, :2]
    np.testing.assert_array_equal(raw(write(pool, rows)), expected)


def test_rings_stay_on_separate_writer(interpret, monkeypatch):
    """Stopgap: the fused kernel leaves ring/tag writes to write_ring_and_tags.

    Serving hung on its first warmup forward when the fused kernel also wrote
    the rings; with the rings on the separate writer it serves.
    """
    fused_ids, ring_calls = [], []
    original_fused = fused_kv_write.fused_layer_write
    original_ring = pooled_qsa.write_ring_and_tags

    def fused(*args, **kwargs):
        fused_ids.append(len(kwargs["ring_ids"]))
        return original_fused(*args, **kwargs)

    def ring(pool, ring, tags, ids):
        ring_calls.append(ids)
        return original_ring(pool, ring, tags, ids)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", fused)
    monkeypatch.setattr(pooled_qsa, "write_ring_and_tags", ring)
    # Earlier cases compiled this step; retrace so the recorders are called.
    jax.clear_caches()
    rng = np.random.default_rng(11)
    args = make_step(rng, [4, 4], [5, 9], jnp.float8_e4m3fn)
    run(monkeypatch, True, args, main_slot=1, max_query_len=4)
    assert len(fused_ids) == 1
    assert len(ring_calls) == 1
