# SPDX-License-Identifier: Apache-2.0
"""q38mm DSTOCK-B: byte-exact equivalence harness for the decode QSA cache writers (scaffolding for the lean writer).

Writers compared against the separate writers (VLLM_TPU_QSA_FUSED_KV_WRITE off, the reference):
  fused  VLLM_TPU_QSA_FUSED_KV_WRITE=1 (e03b38650 + ebbba8253, the KVF-mb1 KEEP)
  lean   VLLM_TPU_QSA_FUSED_KV_WRITE=1 + VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN=1 (the compacted-list writer of
         dstockb_leanwriter_design_DRAFT.md; skipped until fused_kv_write defines LEAN_ENV)

Every case runs ``qsa_cache_writes`` under the c48 served flags (as test_kvf_production_geometry.py) and requires the
pool bytes and (req, logical, valid) to equal the reference's. Cases add the geometries a compacting writer must get
right beyond the decode shape: no live rows, one 128-row request among 32 slots (full compressed pairs, no verify
runs), mixed full and partial pairs, single-row draft steps, every DCP rank and both packed slots.

Fault controls wrap ``pooled_qsa.fused_layer_write`` (the entry point both fused writers sit behind) with one
corruption each; the comparison must FAIL for each (the harness is sensitive whichever implementation is active), and
must PASS for the equivalence control (forcing the per-row main path, which writes the same bytes).
Pallas interpret mode on CPU.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kvf_production_geometry import (  # noqa: E402
    BASE_CTX,
    D,
    K_SCALE,
    OFFSETS,
    RATIO,
    ROTARY,
    V_SCALE,
    B,
    INDEX_DIM,
    WIDTH,
    interpret,  # noqa: F401 (fixture)
    random_fp8,
    raw,
)

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa  # noqa: E402

LEAN_ENV = "VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN"
LEAN_BUILT = hasattr(fused_kv_write, "LEAN_ENV")


def make_step(seed, query_lens, contexts=None, dcp_size=D):
    """One step over len(query_lens) request slots (DCP tables, dcp_size x B tokens per entry; untouched entries ->
    page 0)."""
    rng = np.random.default_rng(seed)
    q = np.asarray(query_lens, np.int32)
    n = len(q)
    ctx = np.asarray(contexts if contexts is not None else
                     [BASE_CTX + OFFSETS[r % len(OFFSETS)] for r in range(n)], np.int32)
    starts = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    tokens = max(8, -(-int(starts[-1]) // 8) * 8)
    span = dcp_size * B
    cols = int((ctx.max() + max(int(q.max()), 1)) // span) + 2
    pages = 1 + n * 5
    free = iter(1 + rng.permutation(pages - 1))
    main = np.zeros((n, cols), np.int32)
    comp = np.zeros((n, cols), np.int32)
    ring = np.zeros((n,), np.int32)
    for r in range(n):
        lo, hi = ctx[r] // span, (ctx[r] + max(int(q[r]), 1) - 1) // span
        for c in range(lo, hi + 1):
            main[r, c] = next(free)
            comp[r, c] = next(free)
        ring[r] = next(free)
    positions = np.zeros(tokens, np.int32)
    for r in range(n):
        positions[starts[r]:starts[r + 1]] = ctx[r] + np.arange(q[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    return dict(pool=random_fp8(rng, (pages, B, 1, 4, WIDTH)), k=normal(tokens, 1, WIDTH), v=normal(tokens, 1, WIDTH),
                raw=normal(tokens, INDEX_DIM), positions=jnp.asarray(positions), nik=normal(INDEX_DIM),
                trig=jnp.pad(normal(8192, ROTARY), ((0, 0), (0, 128 - ROTARY))), starts=jnp.asarray(starts),
                context=jnp.asarray(ctx), ring_ids=jnp.asarray(ring), main_table=jnp.asarray(main),
                compressed_table=jnp.asarray(comp))


def run_writes(monkeypatch, writer, step, *, rank, main_slot, compressed_slot, wrap=None, dcp_size=D):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", writer != "separate")
    if writer == "lean":
        monkeypatch.setenv(LEAN_ENV, "1")
    else:
        monkeypatch.delenv(LEAN_ENV, raising=False)
    calls = []
    inner = fused_kv_write.fused_layer_write

    def entry(*args, **kwargs):
        calls.append(len(kwargs["main_rows"]))
        if wrap is not None:
            args, kwargs = wrap(args, kwargs)
        return inner(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", entry)
    jax.clear_caches()
    s = step
    dcp = (jnp.asarray([rank], jnp.int32), dcp_size)
    out = pooled_qsa.qsa_cache_writes(
        s["pool"], s["k"], s["v"], s["raw"], s["positions"], s["nik"], s["trig"], s["starts"], s["context"],
        s["ring_ids"], s["main_table"], s["compressed_table"], compress_ratio=RATIO, norm_eps=1e-6,
        rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE, compressed_slot=compressed_slot, main_slot=main_slot,
        max_query_len=4, dcp=dcp, compressed_dcp=dcp)
    return [raw(out[0])] + [np.asarray(x) for x in out[1:]], calls


def assert_same(got, ref):
    diff = np.argwhere((got[0] != ref[0]).reshape(got[0].shape[0], got[0].shape[1], -1).any(-1))
    assert len(diff) == 0, f"{len(diff)} pool rows differ; first (page, row) {diff[:3].tolist()}"
    for a, b in zip(got[1:], ref[1:]):
        np.testing.assert_array_equal(a, b)


# (query rows per slot, main_slot, compressed_slot, rank[, dcp size]). All satisfy the fused gate tokens <= slots * 4.
CASES = {
    "c48_128rows_r0_slot1": ([4] * 32, 1, 3, 0),
    "c48_128rows_r1_slot0": ([4] * 32, 0, 0, 1),
    "c48_96rows": ([4] * 24, 1, 3, 1),
    "c48_64rows": ([4] * 16, 0, 3, 0),
    "idle_slots": ([4] * 13 + [0] * 3 + [4] * 16, 1, 3, 1),
    "no_live_rows": ([0] * 8, 1, 3, 0),
    "one_128row_request": ([128] + [0] * 31, 1, 3, 0),
    "mixed_full_partial": ([8, 4, 12, 4, 1, 3, 16, 4] + [0] * 24, 0, 3, 1),
    "draft_single_rows": ([1] * 32, -1, 0, 1),
    # PD O1 geometry (DCP-4 over TP8, 64 sequences x 4 verify rows = 256 rows).
    "pd_dcp4_256rows_r0": ([4] * 64, 0, 3, 0, 4),
    "pd_dcp4_256rows_r3": ([4] * 64, 1, 0, 3, 4),
}

WRITERS = ["fused", pytest.param("lean", marks=pytest.mark.skipif(not LEAN_BUILT, reason="lean writer not built yet"))]


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_writer_matches_separate(interpret, monkeypatch, case, writer):  # noqa: F811
    q, main_slot, comp_slot, rank, *rest = CASES[case]
    size = rest[0] if rest else D
    step = make_step(sum(map(ord, case)), q, dcp_size=size)
    kw = dict(rank=rank, main_slot=main_slot, compressed_slot=comp_slot, dcp_size=size)
    ref, ref_calls = run_writes(monkeypatch, "separate", step, **kw)
    got, calls = run_writes(monkeypatch, writer, step, **kw)
    assert ref_calls == [], "the reference must not use the fused entry point"
    assert len(calls) == 1, f"{writer}: fused entry point not taken ({calls})"
    if sum(q):
        assert not np.array_equal(ref[0], raw(step["pool"])), "nothing written"
    assert_same(got, ref)


def _first_live(mask):
    idx = np.flatnonzero(np.asarray(mask) != 0)
    return int(idx[0]) if len(idx) else None


def f_drop_comp_write(args, kw):
    i = _first_live(kw["comp_write"])
    kw["comp_write"] = kw["comp_write"].at[i].set(0)
    return args, kw


def f_swap_halves(args, kw):
    kw["comp_mask"] = kw["comp_mask"][:, ::-1]
    return args, kw


def f_main_offset(args, kw):
    i = _first_live(kw["main_valid"])
    kw["main_offset"] = kw["main_offset"].at[i].add(1)
    return args, kw


def f_drop_main_row(args, kw):
    i = _first_live(kw["main_valid"])
    kw["main_valid"] = kw["main_valid"].at[i].set(False)
    return args, kw


def f_value_bit(args, kw):
    i = _first_live(kw["comp_write"])
    v = kw["comp_values"]
    kw["comp_values"] = v.at[i, 0, 0].set(v[i, 0, 0] + jnp.asarray(1.0, v.dtype))
    return args, kw


def f_wrong_slot(args, kw):
    kw["main_slot"] = 1 - kw["main_slot"]
    return args, kw


def c_per_row_main(args, kw):
    """Equivalence control: no multi-row verify runs (every main row its own DMA): the same bytes."""
    kw["run_length"] = 1
    return args, kw


FAULTS = {"drop_comp_write": f_drop_comp_write, "swap_halves": f_swap_halves, "main_offset": f_main_offset,
          "drop_main_row": f_drop_main_row, "value_bit": f_value_bit, "wrong_slot": f_wrong_slot}


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_fault_is_detected(interpret, monkeypatch, fault, writer):  # noqa: F811
    q, main_slot, comp_slot, rank = CASES["c48_128rows_r0_slot1"]
    step = make_step(7, q)
    kw = dict(rank=rank, main_slot=main_slot, compressed_slot=comp_slot)
    ref, _ = run_writes(monkeypatch, "separate", step, **kw)
    got, calls = run_writes(monkeypatch, writer, step, wrap=FAULTS[fault], **kw)
    assert len(calls) == 1
    with pytest.raises(AssertionError):
        assert_same(got, ref)


@pytest.mark.parametrize("writer", WRITERS)
def test_equivalent_transform_passes(interpret, monkeypatch, writer):  # noqa: F811
    q, main_slot, comp_slot, rank = CASES["c48_128rows_r0_slot1"]
    step = make_step(9, q)
    kw = dict(rank=rank, main_slot=main_slot, compressed_slot=comp_slot)
    ref, _ = run_writes(monkeypatch, "separate", step, **kw)
    got, _ = run_writes(monkeypatch, writer, step, wrap=c_per_row_main, **kw)
    assert_same(got, ref)


# ---- direct calls of the writer entry point: rings, duplicate destinations, verify runs (fused vs lean) --------------

def _direct(seed, *, duplicates, runs, rings, slot=1):
    rng = np.random.default_rng(seed)
    tokens, requests = 16, 8
    pool = random_fp8(rng, (12, B, 1, 4, WIDTH))
    rows = rng.normal(0, 1, (tokens, 1, 4, WIDTH))
    rows[:, :, 2:] = 0
    rows = jnp.asarray(rows, jnp.float8_e4m3fn)
    page = jnp.full((tokens,), 2, jnp.int32)
    offset = jnp.asarray(np.arange(tokens) % 5 + 3 if duplicates else np.arange(tokens) + 7, jnp.int32)
    valid = jnp.asarray(np.arange(tokens) < 13)
    pairs_per_page = B * 4 // 4
    comp_pair = jnp.asarray(5 * pairs_per_page + rng.permutation(64)[:tokens], jnp.int32)
    comp_write = jnp.asarray(rng.integers(0, 2, tokens), jnp.int32)
    comp_mask = jnp.asarray(rng.integers(0, 2, (tokens, 2)), jnp.int32)
    comp_values = jnp.asarray(rng.normal(0, 1, (tokens, 2, INDEX_DIM)), jnp.bfloat16)
    ring = jnp.asarray(rng.normal(0, 1, (requests, 1024)), jnp.bfloat16)
    tags = jnp.asarray(rng.integers(0, 1000, (requests, 16)), jnp.int32)
    ids = np.zeros(requests, np.int32)
    if rings:
        ids[[0, 3, 6]] = [9, 10, 11]
    ring_ids = jnp.asarray(ids)
    kw = dict(main_slot=slot)
    if runs:
        kw.update(run_starts=jnp.minimum(jnp.arange(requests, dtype=jnp.int32) * 4, tokens),
                  run_lengths=jnp.full((requests,), 4, jnp.int32).at[4:].set(0), run_length=4)
    return pool, rows, page, offset, valid, comp_pair, comp_write, comp_mask, comp_values, ring, tags, ring_ids, kw


def _call(monkeypatch, lean, ops):
    pool, rows, page, offset, valid, cp, cw, cm, cv, ring, tags, ids, kw = ops
    if lean:
        monkeypatch.setenv(LEAN_ENV, "1")
    else:
        monkeypatch.delenv(LEAN_ENV, raising=False)
    jax.clear_caches()
    # Every array enters as an argument: XLA:CPU constant folding of FP8 bitcasts is not byte-exact.
    f = jax.jit(lambda pool, rows, page, offset, valid, cp, cw, cm, cv, ring, tags, ids: fused_kv_write.fused_layer_write(
        pool, main_rows=pooled_qsa.main_slot_rows(rows, kw["main_slot"]), main_page=page, main_offset=offset,
        main_valid=valid, comp_pair=cp, comp_write=cw, comp_mask=cm, comp_values=cv, ring=ring, tags=tags,
        ring_ids=ids, **kw))
    return raw(f(pool, rows, page, offset, valid, cp, cw, cm, cv, ring, tags, ids))


@pytest.mark.skipif(not LEAN_BUILT, reason="lean writer not built yet")
@pytest.mark.parametrize("duplicates,runs,rings", [(False, False, True), (True, False, True), (False, True, True),
                                                   (False, True, False), (True, True, False)])
def test_direct_lean_matches_fused(interpret, monkeypatch, duplicates, runs, rings):  # noqa: F811
    ops = _direct(11 + 2 * duplicates + 3 * runs + rings, duplicates=duplicates, runs=runs, rings=rings)
    before = raw(ops[0])
    fused = _call(monkeypatch, False, ops)
    lean = _call(monkeypatch, True, ops)
    assert not np.array_equal(fused, before), "nothing written"
    if rings:
        assert not np.array_equal(fused[9:12], before[9:12]), "ring pages not written"
    np.testing.assert_array_equal(lean, fused)


@pytest.mark.skipif(not LEAN_BUILT, reason="lean writer not built yet")
def test_lean_dispatch_only_when_flag_set(monkeypatch):
    calls = []
    real = fused_kv_write._lean_layer_write
    monkeypatch.setattr(fused_kv_write, "_lean_layer_write", lambda *a, **k: calls.append(1) or real(*a, **k))
    original_call = fused_kv_write.pl.pallas_call
    monkeypatch.setattr(fused_kv_write.pl, "pallas_call", lambda *a, **k: original_call(*a, **{**k, "interpret": True}))
    ops = _direct(5, duplicates=False, runs=False, rings=False)
    for value, want in ((None, 0), ("0", 0), ("1", 1)):
        calls.clear()
        if value is None:
            monkeypatch.delenv(LEAN_ENV, raising=False)
        else:
            monkeypatch.setenv(LEAN_ENV, value)
        pool, rows, page, offset, valid, cp, cw, cm, cv, ring, tags, ids, kw = ops
        fused_kv_write.fused_layer_write(
            pool, main_rows=pooled_qsa.main_slot_rows(rows, 1), main_page=page, main_offset=offset, main_valid=valid,
            comp_pair=cp, comp_write=cw, comp_mask=cm, comp_values=cv, ring=ring, tags=tags, ring_ids=ids, **kw)
        assert len(calls) == want, (value, calls)
    monkeypatch.setenv(LEAN_ENV, "yes")
    with pytest.raises(ValueError):
        fused_kv_write.lean_writer_enabled()
