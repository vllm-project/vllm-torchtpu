# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-GLUE WRITE-ALIGN CPU proof (tests only).

At the PP8 x TP1 prefill pool geometry (3,200-token FP8 pages of whole rows, compressed slot 0, ratio 4, 14
request slots) the run-based compressed-key writer (VLLM_TPU_QSA_COMPRESSED_RUN_WRITE) and the batched main-KV
writer (VLLM_TPU_QSA_MAIN_KV_DMA_BATCH, q38 MKV-B8) must leave the pool byte-identical to the served tiled writers:
aligned steps, the P0 misaligned 6398/2 step, page crossings, odd contexts (partial pairs) and random served mixes.
Pallas interpret mode on CPU; the writers only move bytes and select halves, so the bytes are backend-independent.

Q38_PGLUE_MUTANT=M1..M6 applies one named mutant (M1 merge_skip, M2 run dest+1, M3 drop_last_run, M4 half_swap,
M5 cap_short in compressed_write.write_runs; M6 drop_last_packet in the batched main-KV writer). Under a mutant
the identity tests must fail and print a 'PGLUE_DIFF' line.
"""

import inspect
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import compressed_write, pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata

PAGE, WIDTH, RATIO, SLOTS, PAGES, COLS = 3200, 256, 4, 14, 34, 40
INDEX_DIM, HEAD_DIM, ROTARY = 128, 256, 64
NAMES = []

MUTANTS = {
    "M1": ("cw", "        jax.lax.fori_loop(0, count_ref[0], merge, 0)\n", "        pass\n"),
    "M2": ("cw", "out.at[pl.ds(dest + done, size)]", "out.at[pl.ds(dest + done + 1, size)]"),
    "M3": ("cw", "jax.lax.fori_loop(0, count_ref[1], issue, 0)", "jax.lax.fori_loop(0, count_ref[1] - 1, issue, 0)"),
    "M4": ("cw", "jnp.where(half == 0, mask_ref[i, 0], mask_ref[i, 1])", "jnp.where(half == 0, mask_ref[i, 1], mask_ref[i, 0])"),
    "M5": ("cw", "cap = tokens // (2 * ratio) + 2 * requests + 8", "cap = tokens // (2 * ratio)"),
    "M6": ("pq", "            for packet in range(tile // batch):\n", "            for packet in range(tile // batch - 1):\n"),
}


def _mutated(module, old, new, tag):
    src = inspect.getsource(module)
    assert src.count(old) == 1, f"mutant {tag}: pattern found {src.count(old)} times"
    mod = types.ModuleType(module.__name__ + "_" + tag)
    mod.__file__ = module.__file__
    exec(compile(src.replace(old, new), module.__file__ + ":" + tag, "exec"), mod.__dict__)
    return mod


@pytest.fixture(autouse=True)
def interpret(monkeypatch):
    original = pl.pallas_call

    def recording(*args, **kwargs):
        NAMES.append(kwargs.get("name"))
        return original(*args, **{**kwargs, "interpret": True})

    monkeypatch.setattr(pl, "pallas_call", recording)
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *a, **k: info)
    tag = os.environ.get("Q38_PGLUE_MUTANT", "")
    if tag:
        where, old, new = MUTANTS[tag]
        if where == "cw":
            monkeypatch.setattr(compressed_write, "write_runs", _mutated(compressed_write, old, new, tag).write_runs)
        else:
            monkeypatch.setattr(pooled_qsa, "write_main_cache", _mutated(pooled_qsa, old, new, tag).write_main_cache)
        print(f"PGLUE_MUTANT {tag} applied")
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", False)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DMA_BATCH", 0)
    yield


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


SHAPES = {  # name: (query lens, contexts, bucket)
    "S1_6400x1_aligned": ([6400], [86400], 6400),
    "S2_6398x2_p0slow": ([3199, 3199], [89600, 89600], 6400),
    "S3_6400x2_aligned": ([3200, 3200], [86400, 86400], 6400),
    "S4_3199x1": ([3199], [89600], 3200),
    "S5_3200x1": ([3200], [86400], 3200),
    "S6_8192_oddctx": ([3199, 4993], [89601, 3], 8192),
    "S7_pagecross_pad": ([6000], [1001], 6400),
    "S8_full_oddctx": ([3, 3197, 3200], [5, 89603, 86402], 6400),  # 801 writers > 6400 // 8 (M5 guard)
}
_rng = np.random.default_rng(38)
for _k in range(6):
    _bucket = [3200, 6400, 8192][_k % 3]
    _n = int(_rng.integers(2, 7))
    _cuts = np.sort(_rng.choice(np.arange(1, min(_bucket, 3200 * _n)), _n, replace=False))
    _lens = [int(x) for x in np.diff(np.concatenate([[0], _cuts]))]
    _lens = [min(x, 3200) for x in _lens]
    if _k % 2:
        _lens[0] = int(_rng.integers(1, 8))  # a decode-like request first: every later start is misaligned
    _ctx = [int(x) for x in _rng.integers(0, 100000, _n)]
    SHAPES[f"R{_k}_mix{_bucket}"] = (_lens, _ctx, _bucket)


def make(name, seed=0, share_main=False):
    lens, ctx, bucket = SHAPES[name] if isinstance(name, str) else name
    rng = np.random.default_rng(seed + 7)
    n = len(lens)
    lens = list(lens) + [0] * (SLOTS - n)
    ctx = list(ctx) + [0] * (SLOTS - n)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    assert starts[-1] <= bucket, (name, starts[-1], bucket)
    free = iter(rng.permutation(np.arange(1, PAGES)))
    main = np.zeros((SLOTS, COLS), np.int32)
    comp = np.zeros((SLOTS, COLS), np.int32)
    ring = np.zeros((SLOTS,), np.int32)
    for r in range(n):
        if lens[r]:
            for c in range(ctx[r] // PAGE, (ctx[r] + lens[r] - 1) // PAGE + 1):
                main[r, c] = next(free)
                comp[r, c] = next(free)
            ring[r] = next(free)
    if share_main:  # duplicate destinations: request 1 writes request 0's main page
        main[1] = main[0]
    starts_j, ctx_j = jnp.asarray(starts), jnp.asarray(ctx, jnp.int32)
    active = starts_j[1:] > starts_j[:-1]
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(bucket, starts_j, ctx_j, slots)
    pool = jnp.asarray(rng.normal(0, 1, (PAGES, PAGE, 1, 4, WIDTH)), jnp.float8_e4m3fn)
    keys = jnp.asarray(rng.normal(0, 1, (bucket, INDEX_DIM)), jnp.bfloat16)
    rows = jnp.asarray(rng.normal(0, 1, (bucket, 1, 4, WIDTH)), jnp.float8_e4m3fn)
    complete = valid & (logical % RATIO == RATIO - 1)
    page = jnp.where(valid, jnp.asarray(main)[req, logical // PAGE], PAGES)
    return dict(pool=pool, keys=keys, rows=rows, complete=complete, req=req, logical=logical, valid=valid,
                starts=starts_j, ctx=ctx_j, comp=jnp.asarray(comp), main=jnp.asarray(main), page=page,
                ring=jnp.asarray(ring), bucket=bucket)


def compressed(monkeypatch, s, run_write):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", run_write)
    jax.clear_caches()
    NAMES.clear()
    out = pooled_qsa.write_compressed(s["pool"], s["keys"], s["complete"], s["req"], s["logical"], s["starts"],
                                      s["ctx"], s["comp"], RATIO, 0)
    return raw(out), list(NAMES)


def main_kv(monkeypatch, s, batch):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DMA_BATCH", batch)
    jax.clear_caches()
    NAMES.clear()
    out = pooled_qsa.write_main_cache(s["pool"], s["rows"], s["page"], s["logical"], s["valid"])
    return raw(out), list(NAMES)


def same(tag, case, got, expected, before):
    eq = np.array_equal(got, expected)
    touched = int(np.count_nonzero(expected != before))
    print(f"PGLUE_ROW {tag} {case} {'SAME' if eq else 'DIFF'} touched_bytes={touched}")
    if not eq:
        print(f"PGLUE_DIFF {tag} {case} differing_bytes={int(np.count_nonzero(got != expected))}")
    assert touched > 0, f"{tag} {case}: the writer touched no byte (vacuous case)"
    assert eq, f"{tag} {case}: pool bytes differ"


@pytest.mark.parametrize("case", sorted(SHAPES))
def test_crw_pool_bytes_identical(monkeypatch, case):
    s = make(case)
    before = raw(s["pool"])
    expected, names_off = compressed(monkeypatch, s, False)
    got, names_on = compressed(monkeypatch, s, True)
    assert "qwen4_exp_compressed_kv_page_tiled_scatter" in names_off, names_off
    assert "qwen4_exp_compressed_kv_run_write" in names_on and "qwen4_exp_compressed_kv_page_tiled_scatter" not in names_on, names_on
    same("CRW", case, got, expected, before)


@pytest.mark.parametrize("case", sorted(SHAPES) + ["D1_dup_dest"])
def test_mkv_pool_bytes_identical(monkeypatch, case):
    s = make(([4, 4, 120], [0, 2, 1000], 128), share_main=True) if case == "D1_dup_dest" else make(case)
    before = raw(s["pool"])
    expected, names_off = main_kv(monkeypatch, s, 0)
    got, names_on = main_kv(monkeypatch, s, 8)
    assert "qwen4_exp_main_kv_universal_tiled_scatter" in names_off, names_off
    assert "qwen4_exp_main_kv_batched_ordered_scatter" in names_on, names_on
    same("MKV", case, got, expected, before)


@pytest.mark.parametrize("case", ["S1_6400x1_aligned", "S2_6398x2_p0slow"])
def test_layer_writes_identical(monkeypatch, case):
    """The whole layer's cache writes (main KV, compressed keys, rings, tags) through qsa_cache_writes."""
    s = make(case, seed=1)
    rng = np.random.default_rng(3)
    T = s["bucket"]
    k = jnp.asarray(rng.normal(0, 1, (T, 1, HEAD_DIM)), jnp.bfloat16)
    v = jnp.asarray(rng.normal(0, 1, (T, 1, HEAD_DIM)), jnp.bfloat16)
    rawk = jnp.asarray(rng.normal(0, 1, (T, INDEX_DIM)), jnp.bfloat16)
    nik = jnp.asarray(rng.normal(0, 0.1, (INDEX_DIM,)), jnp.bfloat16)
    trig = jnp.asarray(rng.normal(0, 1, (131072, 128)), jnp.bfloat16)
    positions = jnp.where(s["valid"], s["logical"], 0).astype(jnp.int32)

    def layer(run_write, batch):
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", run_write)
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DMA_BATCH", batch)
        jax.clear_caches()
        NAMES.clear()
        out = pooled_qsa.qsa_cache_writes(
            s["pool"], k, v, rawk, positions, nik, trig, s["starts"], s["ctx"], s["ring"], s["main"], s["comp"],
            compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY)
        return raw(out[0]), list(NAMES)

    before = raw(s["pool"])
    expected, n_off = layer(False, 0)
    got, n_on = layer(True, 8)
    assert "qwen4_exp_compressed_kv_run_write" in n_on and "qwen4_exp_main_kv_batched_ordered_scatter" in n_on, n_on
    same("LAYER", case, got, expected, before)


def test_controls(monkeypatch):
    """CTL1: off vs off equal (fresh traces). CTL2: one flipped byte in a written row is caught."""
    s = make("S2_6398x2_p0slow")
    a, _ = compressed(monkeypatch, s, False)
    b, _ = compressed(monkeypatch, s, False)
    ctl1 = np.array_equal(a, b)
    before = raw(s["pool"])
    idx = np.flatnonzero(a.reshape(-1) != before.reshape(-1))
    assert len(idx) > 0
    flipped = a.copy().reshape(-1)
    flipped[idx[len(idx) // 2]] ^= np.uint8(1)
    ctl2 = not np.array_equal(flipped.reshape(a.shape), a)
    print(f"PGLUE_CTL CTL1={'PASS' if ctl1 else 'FAIL'} CTL2={'PASS' if ctl2 else 'FAIL'}")
    assert ctl1 and ctl2


def test_flag_off_selects_served_writers(monkeypatch):
    """Flags unset: the served tiled writers (MKV-B8 keeps the measured table; the 3,200-row page misses it)."""
    from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import get_main_kv_dma_batch

    s = make("S5_3200x1")
    assert get_main_kv_dma_batch(s["pool"], s["rows"]) == 1
    for override, want in ((8, 8), (6, 4), (200, 128), (1, 1)):
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DMA_BATCH", override)
        assert get_main_kv_dma_batch(s["pool"], s["rows"]) == want, (override, want)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DMA_BATCH", 8)
    assert get_main_kv_dma_batch(s["pool"], s["keys"]) == 1  # a dtype mismatch keeps serial writes
    print("PGLUE_ENG SELECT PASS")
