# SPDX-License-Identifier: Apache-2.0
"""DCP-2 union-walk prefill (VLLM_TPU_QSA_PREFILL_UNION_DCP).

``qsa_union_dense.union_walk_prefill(dcp=..., return_lse=True)`` walks only the
rank's owned selected groups (``dcp_local_selection`` renumbers them into the
rank's owned group space, where the DCP table addresses like a replicated one)
and returns the (out f32 ungated, lse) partials of the DCP block-sparse kernel.

CPU (this VM): the owned-space renumbering against the contract's owner rule,
and trace-level checks of the routing (the flag sends DCP prefill to the
union walk; with it off every jaxpr is unchanged). TPU
(``DCP2_HARNESS_DEVICE=tpu``, also ``python3 test_dcp2_union_walk.py``): per
rank (out, lse) vs the f32 reference and vs the DCP block-sparse kernel,
combined gated output vs the replicated route, fallbacks bitwise equal to the
DCP block-sparse call, and (``--bench``) per-layer timings at an 8K chunk.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcp2_harness import (  # noqa: E402
    D,
    ON_TPU,
    TPU_TOL,
    check_partial,
    diff_report,
    make_case,
    owner_of_token,
    rank_operand,
    reference_attend,
    reference_metadata,
    selected_token_ids,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

B = 512
RATIO = 4
# The served union walk (c16 za7h args): group 128, KV tile 1024, cap 12288,
# min context 32768, lean. UNION_DCP on top.
UNION_ENV = {
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "128",
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": "1024",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": "12288",
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "32768",
    "VLLM_TPU_QSA_PREFILL_LEAN": "1",
    "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES": "1",
}


def _set_env(setenv, union_dcp, extra=None):
    for k, v in UNION_ENV.items():
        setenv(k, v)
    setenv("VLLM_TPU_QSA_PREFILL_UNION_DCP", "1" if union_dcp else "0")
    for k, v in (extra or {}).items():
        setenv(k, v)


# ----------------------------------------------------------------------------
# CPU: owned-space renumbering


def _owned_visible_tokens(blocks, lengths, rank, page_size=B, ratio=RATIO):
    """Per row, the global tokens the union walk attends after the renumbering.

    The walk's virtual causal mask keeps a selected (or tail) local group's
    key j iff local token ``4 * id + j < local length`` (ranks are monotone in
    ids); local ids map back to global ones by inverting the renumbering."""
    lb, ll = ud.dcp_local_selection(jnp.asarray(blocks), jnp.asarray(lengths), rank=rank_operand(rank),
                                    world=D, page_size=page_size, ratio=ratio)
    lb, ll = np.asarray(lb), np.asarray(ll)
    g_per_page = page_size // ratio

    def to_global(local):
        return ((local // g_per_page) * D + rank) * g_per_page + local % g_per_page

    rows = []
    for i in range(len(lengths)):
        ids = {int(x) for x in lb[i] if x >= 0} | ({int(ll[i]) // ratio} if ll[i] % ratio else set())
        toks = set()
        for local in ids:
            for j in range(ratio):
                if local * ratio + j < ll[i]:
                    toks.add(int(to_global(local)) * ratio + j)
        rows.append(toks)
    return rows


def test_local_selection_matches_owned_tokens():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([
        np.arange(1, 70),  # first page: rank 1 owns nothing
        rng.integers(1, 6000, 300),
        np.array([0, 0, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049]),
    ]).astype(np.int32)
    from dcp2_harness import random_blocks

    blocks = random_blocks(rng, lengths, RATIO, 96)
    expected = selected_token_ids(blocks, lengths, RATIO)
    for rank in range(D):
        got = _owned_visible_tokens(blocks, lengths, rank)
        empty = 0
        for i in range(len(lengths)):
            want = {int(t) for t in expected[i] if t >= 0 and owner_of_token(t, B) == rank}
            assert got[i] == want, (rank, i, int(lengths[i]), sorted(got[i] ^ want)[:8])
            empty += not want and lengths[i] > 0
        _, kv_lens, _, _ = reference_metadata(blocks, lengths, np.zeros((len(lengths), 8), np.int32),
                                              ratio=RATIO, page_size=B, dcp=(rank, D))
        assert sum(len(t) for t in got) == int(kv_lens.sum())
        if rank == 1:
            assert empty >= 69  # live rows owning nothing are covered


# ----------------------------------------------------------------------------
# Cases (TPU sizes; the union walk needs a step whose longest row is >= 32768)

# (prefix, chunk) per request. "mix": a long chunk past min context, an
# unaligned short piece, and a fresh 200-token piece whose rows own nothing on
# rank 1; then NaN padding rows past starts[-1] up to a 128-row multiple.
CASES = {
    "mix_pad": dict(chunks=[(36001, 3000), (700, 333), (0, 200)], budget=1024, pad_to=128),
    "long_4k": dict(chunks=[(40003, 4096)], budget=1024, pad_to=0),
}


def union_case(rng, chunks, *, budget, pad_to, kv_head_offset=0):
    lengths, req, seq = [], [], []
    for r, (prefix, count) in enumerate(chunks):
        lengths += list(range(prefix + 1, prefix + count + 1))
        req += [r] * count
        seq.append(prefix + count)
    real = len(lengths)
    pad = (-real) % pad_to if pad_to else 0
    nan_rows = list(range(real, real + pad))
    lengths += [0] * pad
    req += [len(chunks) - 1] * pad
    case = make_case(rng, row_lengths=lengths, row_req=req, seq_tokens=seq, budget=budget,
                     nan_rows=nan_rows, kv_head_offset=kv_head_offset)
    # Padding rows sit past the last request (the served prefill layout).
    starts = np.zeros(len(chunks) + 1, np.int32)
    starts[1:] = np.cumsum([c for _, c in chunks])
    case.starts = starts
    return case


def prefill_attend(case, half):
    """``_prefill_attention(dcp, return_lse)`` on query heads ``half`` (None: all)."""

    def attend(q, gate, blocks, lengths, table, cache, **kw):
        return qsa_rpa._prefill_attention(
            q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache,
            jnp.asarray(case.starts), return_lse=True, **kw)[:2]

    return attend


def run_rank(case, rank, setenv, union_dcp, extra=None):
    _set_env(setenv, union_dcp, extra)
    out, lse = prefill_attend(case, None)(
        case.q, case.gate, case.blocks, case.lengths, case.dcp_rows, case.rank_pools[rank],
        dcp=(rank_operand(rank), D), **case.opts)
    return np.asarray(out), np.asarray(lse)


def check_union_case(case, setenv, *, tol=TPU_TOL):
    """Union-walk DCP partials vs the f32 reference and the DCP block-sparse
    kernel per rank; merged + gated vs the replicated served route."""
    reports = {}
    union, sparse = [], []
    for r in range(D):
        dcp = (rank_operand(r), D)
        u = run_rank(case, r, setenv, True)
        s = run_rank(case, r, setenv, False)
        ref_out, ref_lse = reference_attend(case.q, case.gate, case.blocks, case.lengths,
                                            case.dcp_rows, case.rank_pools[r], dcp=dcp, **case.opts)
        _, kv_lens, _, _ = reference_metadata(case.blocks, case.lengths, case.dcp_rows,
                                              ratio=case.ratio, page_size=B, dcp=(r, D))
        reports[f"union_vs_ref{r}"] = check_partial(*u, ref_out, ref_lse, kv_lens, tol,
                                                    what=f"union rank {r} vs reference")
        reports[f"union_vs_sparse{r}"] = check_partial(*u, *s, kv_lens, tol,
                                                       what=f"union rank {r} vs block-sparse")
        # The walk ran: its online softmax order differs from the block-sparse kernel's.
        reports[f"walk_ran{r}"] = bool((u[0] != s[0]).any())
        reports[f"rows_owning_nothing{r}"] = int(((kv_lens == 0) & (case.lengths > 0)).sum())
        union.append(u)
        sparse.append(s)
    assert any(reports[f"walk_ran{r}"] for r in range(D)), "union walk output equals block-sparse bitwise"
    from test_dcp2_prefill import check_pregate, merge_pair, production_replicated
    from dcp2_harness import check_gated

    out_pair = jnp.stack([jnp.asarray(u[0]) for u in union])
    lse_pair = jnp.stack([jnp.asarray(u[1]) for u in union])
    _set_env(setenv, False)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    gated_rep = production_replicated(case)
    pre_rep = production_replicated(case, pregate=True)
    for r in range(D):
        gate_r = case.gate[:, r * case.heads : (r + 1) * case.heads]
        gated = qsa_rpa.dcp_combine(out_pair, lse_pair, gate_r, rank_operand(r))
        reports[f"pregate{r}"] = check_pregate(*merge_pair(out_pair, lse_pair, r, case.heads),
                                               *pre_rep[r], live, tol, what=f"union pre-gate {r}")
        reports[f"gated{r}"] = check_gated(gated, gated_rep[r], live, tol, what=f"union gated {r}")
    return reports


def check_fallback(case, setenv, extra):
    """A forced fallback (cap / min context) returns the DCP block-sparse call bitwise."""
    reports = {}
    for r in range(D):
        u = run_rank(case, r, setenv, True, extra)
        s = run_rank(case, r, setenv, False, extra)
        assert np.array_equal(u[0], s[0]) and np.array_equal(u[1], s[1]), f"rank {r} fallback differs"
        reports[f"fallback_bitwise{r}"] = True
    return reports


# ----------------------------------------------------------------------------
# CPU: trace-level routing checks


def _trace_case():
    rng = np.random.default_rng(5)
    # 1024 rows of one request on a 40K-token context (no pools: shapes only).
    rows, seq, page_cols = 1024, 41000, -(-41000 // B)
    q = jnp.zeros((rows, 2 * 3, 256), jnp.bfloat16)
    blocks = jnp.asarray(rng.integers(0, seq // RATIO, (rows, 256)), jnp.int32)
    lengths = jnp.arange(seq - rows + 1, seq + 1, dtype=jnp.int32)
    starts = jnp.asarray([0, rows], jnp.int32)
    full = jnp.zeros((rows, page_cols), jnp.int32)
    dcp_table = jnp.zeros((rows, -(-page_cols // D)), jnp.int32)
    cache = jnp.zeros((2 * page_cols + 4, B, 1, 4, 256), jnp.float8_e4m3fn)
    return q, blocks, lengths, starts, full, dcp_table, cache


def _tpu_info(monkeypatch):
    """Trace Mosaic kernel bodies on a CPU host (TPU7x info, as conftest's ``interpret``)."""
    if ON_TPU:
        return
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _jaxpr(dcp, slot=0):
    q, blocks, lengths, starts, full, dcp_table, cache = _trace_case()
    table = full if dcp is None else dcp_table

    def f(q, blocks, lengths, table, cache, starts):
        return qsa_rpa._prefill_attention(
            q, q, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.375, v_scale=2.5, **({"kv_head_offset": slot} if slot else {}),
            **({"dcp": (rank_operand(1), D), "return_lse": True} if dcp else {}))

    return str(jax.make_jaxpr(f)(q, blocks, lengths, table, cache, starts))


@pytest.mark.parametrize("slot", [0, 1])
def test_flag_routes_dcp_prefill_to_union_walk(monkeypatch, slot):
    _tpu_info(monkeypatch)
    _set_env(monkeypatch.setenv, True)
    on = _jaxpr(True, slot)
    assert "qsa_union_walk_g128_kv1024_lean_fast_pg_lse" in on
    assert (f"_slot{slot}" in on) == bool(slot)
    _set_env(monkeypatch.setenv, False)
    off = _jaxpr(True, slot)
    assert "qsa_union_walk" not in off
    # Variants the DCP walk does not support keep the block-sparse kernel.
    _set_env(monkeypatch.setenv, True, {"VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": "1"})
    assert "qsa_union_walk" not in _jaxpr(True, slot)


def test_flag_leaves_replicated_route_unchanged(monkeypatch):
    _tpu_info(monkeypatch)
    _set_env(monkeypatch.setenv, False)
    off = _jaxpr(None)
    _set_env(monkeypatch.setenv, True)
    on = _jaxpr(None)
    assert on == off
    assert "qsa_union_walk_g128_kv1024_lean_fast_pg" in on and "fast_pg_lse" not in on


# ----------------------------------------------------------------------------
# TPU

tpu = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")


@tpu
@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tpu_union_dcp_matches(monkeypatch, name, slot):
    spec = CASES[name]
    case = union_case(np.random.default_rng(len(name) + slot), spec["chunks"], budget=spec["budget"],
                      pad_to=spec["pad_to"], kv_head_offset=slot)
    print(name, slot, check_union_case(case, monkeypatch.setenv))


@tpu
@pytest.mark.parametrize("which", ["cap", "min_context"])
def test_tpu_union_dcp_fallback_is_block_sparse(monkeypatch, which):
    spec = CASES["mix_pad"]
    case = union_case(np.random.default_rng(7), spec["chunks"], budget=spec["budget"], pad_to=128)
    extra = ({"VLLM_TPU_QSA_PREFILL_UNION_CAP": "64"} if which == "cap"
             else {"VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "1000000"})
    print(which, check_fallback(case, monkeypatch.setenv, extra))


# ----------------------------------------------------------------------------
# Bench: per-layer union-walk DCP vs block-sparse DCP at an 8K chunk


def clustered_blocks(rng, lengths, budget, ratio=RATIO, hot=3):
    """Correlated selections: rows of a request draw from one shared candidate
    pool of ``hot * budget`` groups (plus their causal tail), as indexer
    selections of neighbouring rows overlap."""
    lengths = np.asarray(lengths)
    complete_max = int(lengths.max()) // ratio
    pool = np.sort(rng.choice(complete_max, min(complete_max, hot * budget), replace=False))
    blocks = np.full((len(lengths), budget), -1, np.int32)
    for i, n in enumerate(lengths):
        cand = pool[pool < int(n) // ratio]
        count = min(len(cand), budget)
        if count:
            blocks[i, :count] = rng.choice(cand, count, replace=False)
    return blocks


def _time(fn, iters=10):
    jax.block_until_ready(fn())
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def bench(chunk=8192, context=74000, budget=2048, selections=("random", "clustered")):
    results = {}
    for sel in selections:
        rng = np.random.default_rng(11)
        case = union_case(rng, [(context - chunk, chunk)], budget=budget, pad_to=0)
        if sel == "clustered":
            case.blocks = clustered_blocks(rng, case.lengths, budget)
        for r in range(D):
            args = (case.q, case.gate, jnp.asarray(case.blocks), jnp.asarray(case.lengths),
                    jnp.asarray(case.dcp_rows), case.rank_pools[r], jnp.asarray(case.starts))
            times, outs = {}, {}
            for arm, union_dcp in (("union", True), ("block_sparse", False)):
                _set_env(os.environ.__setitem__, union_dcp)
                fn = jax.jit(lambda *a: qsa_rpa._prefill_attention(
                    *a, dcp=(rank_operand(r), D), return_lse=True, **case.opts)[:2])
                times[arm] = _time(lambda: fn(*args))
                outs[arm] = [np.asarray(x) for x in fn(*args)]
            _, kv_lens, _, _ = reference_metadata(case.blocks, case.lengths, case.dcp_rows,
                                                  ratio=RATIO, page_size=B, dcp=(r, D))
            agree = check_partial(*outs["union"], *outs["block_sparse"], kv_lens, TPU_TOL,
                                  what=f"bench {sel} rank {r}")
            results[f"{sel}_rank{r}"] = dict(
                union_ms=round(times["union"], 3), block_sparse_ms=round(times["block_sparse"], 3),
                speedup=round(times["block_sparse"] / times["union"], 3), union_vs_sparse=agree)
            print("BENCH", sel, r, results[f"{sel}_rank{r}"], flush=True)
    return results


if __name__ == "__main__":
    # python3 test_dcp2_union_walk.py [--bench]  (DCP2_HARNESS_DEVICE=tpu for the TPU tests)
    class _Env:
        def setenv(self, k, v):
            os.environ[k] = v

    env = _Env()
    test_local_selection_matches_owned_tokens()
    print("PASS local selection", flush=True)
    failures = 0
    if ON_TPU:
        for name in sorted(CASES):
            for slot in (0, 1):
                try:
                    spec = CASES[name]
                    case = union_case(np.random.default_rng(len(name) + slot), spec["chunks"],
                                      budget=spec["budget"], pad_to=spec["pad_to"], kv_head_offset=slot)
                    print("PASS", name, slot, check_union_case(case, env.setenv), flush=True)
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print("FAIL", name, slot, repr(e)[:4000], flush=True)
        for which in ("cap", "min_context"):
            try:
                spec = CASES["mix_pad"]
                case = union_case(np.random.default_rng(7), spec["chunks"], budget=spec["budget"],
                                  pad_to=128)
                extra = ({"VLLM_TPU_QSA_PREFILL_UNION_CAP": "64"} if which == "cap"
                         else {"VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "1000000"})
                print("PASS fallback", which, check_fallback(case, env.setenv, extra), flush=True)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print("FAIL fallback", which, repr(e)[:4000], flush=True)
        if "--bench" in sys.argv:
            bench()
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)
