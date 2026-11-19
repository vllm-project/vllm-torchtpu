# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL UNION: union RPA verify attention on the O1s attend (VLLM_TPU_QSA_DECODE_RPA_UNION_O1S), through
the REAL ``pooled_qsa.pooled_qsa_request_attend_dcp_step`` (the jax_op body the served stock verify step calls),
at served dims: main-KV DCP 4, 512-token fp8 pages, 512 selected groups a row, 12 query heads of a DCP group,
64 request slots, the v8 decode RPA env (compact schedule, slot reads, fast fetch, 512-token per-row tiles, 4 lanes,
128 queries a call). Ported from Benliang Wang <benliangw@google.com> (2ce02bb1a on
benliangw/q38fh-integ-pin-5b44702aea); his tree checks the kernels on TPU only (bench f15, a 1e-3 tolerance gate).

Kernels on CPU: see ``portbl_union_harness`` (metadata kernel = its own per-request code on numpy; RPA = an exact
key-set stand-in or a tile-faithful online-softmax stand-in). The selection-mask RPA kernel and the SparseCore
kernel themselves only lower on CPU (``test_portbl_union_lowering.py``); they are unproven until device.

Legs (one ``PBLU`` line each):
  KS   union on vs off, key-set stand-in: out, LSE, ``selected`` and the pool bit-identical at 80 / 96 / 128 / 192 /
       256 rows (each row attends exactly the keys it attends on the per-row path), with ENG counts.
  NB   union on vs off, tiled stand-in, all four DCP ranks, then ``qsa_rpa.dcp_combine`` + gate: max |d| of the f32
       partials and LSE, and of the gated bf16 output (max ulp, differing fraction). NOT bitwise by design.
  FB   steps the union cannot take (a request of 5 rows; live requests past the covered slots) fall back to the
       per-row branch on device: bit-identical to union off, no union kernel runs.
  Q1   a one-row-per-request program (max_query_len 1, the draft / non-spec shape) never traces the union.
  OT   the other DCP attend (``pooled_qsa_shard_attend_dcp_step``) never takes it; only the O1s attend passes
       ``o1s_target``.
"""

from __future__ import annotations

import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import rank_operand

import portbl_union_harness as H
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa, qsa_union_meta

K_SCALE, V_SCALE = 0.375, 2.5
SERVED = {
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_DECODE_RPA_FETCH": "fast",
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
}
UNION_KEY = "VLLM_TPU_QSA_DECODE_RPA_UNION_O1S"
ROWS = (80, 96, 128, 192, 256)
WANT_COVER = {80: (32, 1), 96: (32, 1), 128: (32, 1), 192: (64, 2), 256: (64, 2)}
STATIC = ("kv_heads", "compress_ratio", "token_topk", "dcp_size", "k_scale", "v_scale", "main_slot",
          "max_query_len", "tag")


def _attend_tagged(q, gate, lengths, req, starts, main_table, context, pool, gathered, dcp_rank, *, tag, **kw):
    del tag  # a static argument per (stand-in, flag): one trace each, reused over the DCP ranks
    return pooled_qsa.pooled_qsa_request_attend_dcp_step(
        q, gate, lengths, req, starts, main_table, context, pool, gathered, dcp_rank, **kw)


ATTEND = jax.jit(_attend_tagged, static_argnames=STATIC)


@pytest.fixture
def served(monkeypatch, interpret):
    for k, v in SERVED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(UNION_KEY, raising=False)
    monkeypatch.delenv("VLLM_TPU_RPA_SCHEDULE_HBM_O1S", raising=False)
    monkeypatch.setattr(qsa_union_meta, "union_sc_metadata_staged", H.sc_standin)
    qsa_rpa.UNION_TRACED.clear()
    H.reset_counts()
    jax.clear_caches()
    return monkeypatch


def run(mp, step, pool, rank, *, union, standin, max_query_len=4):
    mp.setattr(wrapper, "ragged_paged_attention", H.STANDINS[standin])
    if union:
        mp.setenv(UNION_KEY, "640")
    else:
        mp.delenv(UNION_KEY, raising=False)
    H.reset_counts()
    res = ATTEND(step["q"], step["gate"], step["lengths"], step["req"], step["starts"], step["main_table"],
                 step["context"], pool, step["gathered"], rank_operand(rank), kv_heads=1, compress_ratio=4,
                 token_topk=2048, dcp_size=4, k_scale=K_SCALE, v_scale=V_SCALE, main_slot=-1,
                 max_query_len=max_query_len, tag=(standin, union, max_query_len, step["q"].shape[0]))
    jax.effects_barrier()
    pool_out, out, lse, selected = (np.asarray(x) for x in res)
    return dict(pool=pool_out, out=out, lse=lse, selected=selected, counts=dict(H.COUNTS))


def _same(a, b):
    return bool(np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)))


def _union_sizes(step, rank):
    """Host reference (qsa_union_meta.union_reference): each request's union kv_len on ``rank``, and the most
    owned tokens of one row (the per-row kv_len)."""
    ref = qsa_union_meta.union_reference(step["blocks"], np.asarray(step["lengths"]), np.asarray(step["starts"]),
                                         np.asarray(step["main_table"])[np.asarray(step["req"])], rank)
    return max(e[2] for e in ref), max(max(e[4]) for e in ref)


@pytest.mark.parametrize("rows", ROWS)
def test_ks_union_attends_exactly_the_per_row_keys(served, rows):
    step = H.make_step(seed=rows, tokens=rows)
    pool = H.random_pool(rows, step["npages"])
    ranks = (0, 1, 2, 3) if rows == 128 else (rows // 16 % 4, 3 - rows // 16 % 4)
    for rank in ranks:
        off = run(served, step, pool, rank, union=False, standin="keyset")
        on = run(served, step, pool, rank, union=True, standin="keyset")
        eq = {k: _same(on[k], off[k]) for k in ("out", "lse", "selected", "pool")}
        live = np.isfinite(off["lse"][:, 0])
        union_kv, row_kv = _union_sizes(step, rank)
        print(f"PBLU KS rows {rows} rank {rank}: equal {eq}; live rows {int(live.sum())}/{step['live']}; "
              f"union calls {on['counts']['union_rpa']} metadata runs {on['counts']['sc']} per-row RPA "
              f"{on['counts']['rows_rpa']} | off: union {off['counts']['union_rpa']} metadata {off['counts']['sc']} "
              f"per-row {off['counts']['rows_rpa']}; max union kv_len {union_kv} max row kv_len {row_kv}")
        assert all(eq.values()), eq
        assert _same(on["pool"], np.asarray(pool)), "the union route wrote the pool"
        # ENG: the union branch ran (one metadata run, one RPA call per 32 covered requests), the per-row one not.
        covered, calls = WANT_COVER[rows]
        assert on["counts"] == {"sc": 1, "union_rpa": calls, "rows_rpa": 0}, on["counts"]
        assert off["counts"]["sc"] == 0 and off["counts"]["union_rpa"] == 0 and off["counts"]["rows_rpa"] >= 1
        # Not vacuous: live rows exist, and some union sequence spans several 640-token tiles and some row
        # several 512-token tiles on this rank.
        assert live.sum() > rows // 4
        assert union_kv > 640 and row_kv > 512, (union_kv, row_kv)
    assert qsa_rpa.UNION_TRACED.get((rows, 64)) == WANT_COVER[rows], qsa_rpa.UNION_TRACED


def _bf16_ulp(a, b):
    ia = np.asarray(jnp.asarray(a, jnp.bfloat16)).view(np.int16).astype(np.int32)
    ib = np.asarray(jnp.asarray(b, jnp.bfloat16)).view(np.int16).astype(np.int32)
    # Map sign-magnitude to a monotone integer line.
    ia = np.where(ia < 0, -32768 - ia, ia)
    ib = np.where(ib < 0, -32768 - ib, ib)
    return np.abs(ia - ib)


@pytest.mark.parametrize("rows", ROWS)
def test_nb_union_differs_only_by_sum_order(served, rows):
    step = H.make_step(seed=7 * rows, tokens=rows)
    pool = H.random_pool(7 * rows, step["npages"])
    parts = {False: [], True: []}
    worst = dict(dout=0.0, dlse=0.0, rel=0.0)
    for rank in range(4):
        off = run(served, step, pool, rank, union=False, standin="tiled")
        on = run(served, step, pool, rank, union=True, standin="tiled")
        assert _same(on["pool"], off["pool"]) and _same(on["selected"], off["selected"])
        live_off = np.isfinite(off["lse"]).all(axis=1)
        live_on = np.isfinite(on["lse"]).all(axis=1)
        assert np.array_equal(live_off, live_on), "union and per-row disagree on which rows own a key"
        assert np.isfinite(on["out"]).all() and np.isfinite(off["out"]).all()
        if live_off.any():
            d = np.abs(on["out"][live_off] - off["out"][live_off])
            scale = np.abs(off["out"][live_off]).reshape(live_off.sum(), -1).max(axis=1)
            rel = d.reshape(live_off.sum(), -1).max(axis=1) / np.maximum(scale, 1e-30)
            worst["dout"] = max(worst["dout"], float(d.max()))
            worst["rel"] = max(worst["rel"], float(rel.max()))
            worst["dlse"] = max(worst["dlse"], float(np.abs(on["lse"][live_off] - off["lse"][live_off]).max()))
        assert on["counts"]["union_rpa"] == WANT_COVER[rows][1] and on["counts"]["rows_rpa"] == 0
        parts[False].append(off)
        parts[True].append(on)
    gated = {}
    heads = step["q"].shape[1] // 4
    for flag in (False, True):
        out_pair = jnp.asarray(np.stack([p["out"] for p in parts[flag]]))
        lse_pair = jnp.asarray(np.stack([p["lse"] for p in parts[flag]]))
        gated[flag] = np.concatenate([
            np.asarray(qsa_rpa.dcp_combine(out_pair, lse_pair, step["gate"][:, r * heads : (r + 1) * heads],
                                           rank_operand(r)), np.float32)
            for r in range(4)], axis=1)
    live = np.asarray(step["lengths"]) > 0
    a, b = gated[True][live], gated[False][live]
    differ = float(np.mean(a != b))
    same_sign = np.signbit(a) == np.signbit(b)
    ulp = _bf16_ulp(a[same_sign], b[same_sign])
    flips = ~same_sign & (a != b)
    print(f"PBLU NB rows {rows}: partial f32 max|dout| {worst['dout']:.3e} max rel {worst['rel']:.3e} "
          f"max|dlse| {worst['dlse']:.3e}; gated bf16 ({a.size} values) max|d| {float(np.abs(a - b).max()):.3e} "
          f"max ulp (same sign) {int(ulp.max()) if ulp.size else 0}, near-zero sign flips {int(flips.sum())} "
          f"(max|d| {float(np.abs(a - b)[flips].max()) if flips.any() else 0.0:.3e}), differing fraction "
          f"{differ:.5f} (bitwise {differ == 0.0})")
    # His TPU gate (bench f15_compare): out within 1e-3 of the row's largest |out|, LSE within 1e-3.
    assert worst["rel"] <= 1e-3 and worst["dlse"] <= 1e-3, worst


@pytest.mark.parametrize("case", ["five_rows", "past_covered"])
def test_fb_untakeable_steps_fall_back_bitwise(served, case):
    if case == "five_rows":
        # Requests of 5 rows (precompile dummy steps): more than max_query_len rows a request.
        step = H.make_step(seed=5, tokens=80, rows_per_request=5, pattern_mix=["partial", "identical"])
    else:
        # 40 live requests of 2 rows in an 80-row program covering 32 request slots.
        step = H.make_step(seed=6, tokens=80, rows_per_request=2, pattern_mix=["partial", "identical"])
    pool = H.random_pool(5, step["npages"])
    for rank in (0, 2):
        off = run(served, step, pool, rank, union=False, standin="tiled")
        on = run(served, step, pool, rank, union=True, standin="tiled")
        eq = {k: _same(on[k], off[k]) for k in ("out", "lse", "selected", "pool")}
        print(f"PBLU FB {case} rank {rank}: equal {eq}; on counts {on['counts']}")
        assert all(eq.values()), eq
        assert on["counts"]["union_rpa"] == 0 and on["counts"]["sc"] == 0 and on["counts"]["rows_rpa"] >= 1
    assert (80, 64) in qsa_rpa.UNION_TRACED  # the program holds the route; the device picked the per-row branch


def test_q1_one_row_programs_never_trace_the_union(served):
    step = H.make_step(seed=11, tokens=96, slots=96, rows_per_request=1, max_query_len=1,
                       pattern_mix=["partial", "identical"])
    pool = H.random_pool(11, step["npages"])
    off = run(served, step, pool, 1, union=False, standin="keyset", max_query_len=1)
    on = run(served, step, pool, 1, union=True, standin="keyset", max_query_len=1)
    eq = {k: _same(on[k], off[k]) for k in ("out", "lse", "selected", "pool")}
    print(f"PBLU Q1 max_query_len 1: equal {eq}; traced {dict(qsa_rpa.UNION_TRACED)}; on counts {on['counts']}")
    assert all(eq.values()) and not qsa_rpa.UNION_TRACED
    assert on["counts"]["union_rpa"] == 0 and on["counts"]["sc"] == 0


def test_ot_other_dcp_attend_never_takes_the_union(served):
    step = H.make_step(seed=13, tokens=96)
    pool = H.random_pool(13, step["npages"])
    served.setattr(wrapper, "ragged_paged_attention", H.STANDINS["keyset"])
    served.setenv(UNION_KEY, "640")
    blocks = jnp.asarray(step["blocks"])
    res = jax.jit(pooled_qsa.pooled_qsa_shard_attend_dcp_step, static_argnames=(
        "kv_heads", "compress_ratio", "token_topk", "dcp_size", "k_scale", "v_scale", "main_slot",
        "max_query_len", "merged"))(
        step["q"], step["gate"], step["lengths"], step["req"], step["starts"], step["main_table"], step["context"],
        pool, blocks, rank_operand(0), kv_heads=1, compress_ratio=4, token_topk=2048, dcp_size=4,
        k_scale=K_SCALE, v_scale=V_SCALE, max_query_len=4, merged=True)
    jax.effects_barrier()
    del res
    print(f"PBLU OT shard attend with the key on: traced {dict(qsa_rpa.UNION_TRACED)} counts {dict(H.COUNTS)}")
    assert not qsa_rpa.UNION_TRACED and H.COUNTS["union_rpa"] == 0 and H.COUNTS["sc"] == 0
    # Only the O1s attend passes o1s_target=True.
    src = inspect.getsource(pooled_qsa)
    sites = [m.start() for m in re.finditer(r"o1s_target=True", src)]
    owner = inspect.getsource(pooled_qsa.pooled_qsa_request_attend_dcp_step)
    assert len(sites) == 1 and "o1s_target=True" in owner, sites
