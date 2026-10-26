# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the mixed-step routing plan (and combine).

Three parts (write-tpu-kernels 1.3):
  oracle    -- spec.oracle_plan, first-principles numpy int64 (the plan is integer
               bookkeeping, so the oracle is exact, not float64-approximate);
               spec.oracle_combine, float64.
  native    -- spec.incumbent_plan / incumbent_combine, the production code path.
  faults    -- spec.FAULTS injected into the oracle; each must turn the comparison
               red. Catch rate is reported per cell class; a fault blind on every
               production class (served, served_pad) fails the gate.
Candidates register in IMPLS (after the freeze gate) and must match the oracle
exactly on every cell. Runs on CPU (Pallas in interpret mode) and on TPU.
"""

import collections
import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402



def _on_tpu():
    # Not at import: touching the backend during collection is refused by conftest.
    return jax.default_backend() == "tpu"

# name -> fn(cell, exchange, local_blocks, block_hist, me, interpret) -> dict
IMPLS = {
    "incumbent": lambda c, e, l, b, m, interpret: spec.incumbent_plan(c, e, l, b, m),
    "a1": lambda c, e, l, b, m, interpret: spec.candidate_plan(c, e, l, b, m, interpret),
}

EDGE = [
    ("mix32", "one_shard"),
    ("mix32", "unrouted"),
    ("mix64", "hot"),
    ("mix64", "served_pad"),
    ("mix256", "dup"),
    ("mix256", "one_shard"),
    ("mix128", "unrouted"),
    ("tw16", "uniform"),
    ("tw16", "unrouted"),
    ("tw4", "uniform"),
    ("tw4", "hot"),
]
PROD = [(c, k) for c in spec.PROD_CELLS for k in ("served", "served_pad", "uniform")]
FAST = os.environ.get("KPLAN_FAST") == "1"


def _case(cn, kind, seed=1):
    cell = spec.CELLS[cn]
    pairs = spec.make_pairs(cell, kind, seed)
    sb = spec.make_scale_bits(cell, seed)
    ex = spec.make_exchange(cell, pairs, sb)
    mes = sorted({0, cell.ep - 1, spec.busiest_shard(cell, pairs)})
    return cell, pairs, sb, ex, mes


def _run(impl, cell, pairs, ex, me):
    lb, bh = spec.local_inputs(cell, pairs, me)
    f = jax.jit(lambda e, l, b, m: IMPLS[impl](cell, e, l, b, m, not _on_tpu()))
    return jax.device_get(f(ex, lb, bh, jnp.int32(me)))


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("cn,kind", PROD + EDGE)
def test_plan_matches_oracle(impl, cn, kind):
    cell, pairs, sb, ex, mes = _case(cn, kind)
    for me in mes[:1] if FAST else mes:
        got = _run(impl, cell, pairs, ex, me)
        want = spec.oracle_plan(cell, pairs, sb, me)
        bad = spec.compare(want, got)
        assert not bad, f"{impl} {cn} {kind} me={me}: {bad}"


CLASSES = ("served", "served_pad", "uniform", "hot", "one_shard", "unrouted")


def test_fault_control_catch_rate():
    """Positive control: every fault must be caught on a production class."""
    rate = collections.defaultdict(dict)
    for kind in CLASSES:
        cn = "mix128" if kind in ("served", "served_pad", "uniform") else "mix32"
        cell, pairs, sb, ex, mes = _case(cn, kind)
        # Every shard the plan cells use: a fault invisible from one shard (no
        # routed pairs of its own, no experts with rows) is not blind.
        refs = {me: _run("incumbent", cell, pairs, ex, me) for me in mes}
        for f in spec.FAULTS:
            rate[f][kind] = any(
                bool(spec.compare(spec.oracle_plan(cell, pairs, sb, me, fault=f), refs[me]))
                for me in mes
            )
    lines = []
    blind_prod = []
    for f in spec.FAULTS:
        caught = [k for k in CLASSES if rate[f][k]]
        blind = [k for k in CLASSES if not rate[f][k]]
        lines.append(f"{f:13s} caught {len(caught)}/{len(CLASSES)} blind={blind}")
        if not (rate[f]["served"] or rate[f]["served_pad"]):
            blind_prod.append(f)
    print("\nFAULT CONTROL (mix128 served/served_pad/uniform, mix32 others)\n" + "\n".join(lines))
    assert not blind_prod, f"faults blind on every production class: {blind_prod}"


@pytest.mark.parametrize("cn", spec.PROD_CELLS)
def test_combine_native_vs_f64_and_faults(cn):
    if not _on_tpu():
        pytest.skip("combine reference runs the TPU path")
    cell = spec.CELLS[cn]
    pairs = spec.make_pairs(cell, "served_pad", 3)
    sb = spec.make_scale_bits(cell, 3)
    ex = spec.make_exchange(cell, pairs, sb)
    # Source 0: bucket padding sits in the tail sources, so source 0 routes.
    me = 0
    plan = _run("incumbent", cell, pairs, ex, me)
    arr, scl, w, routes = spec.make_combine_inputs(cell, pairs, me, 3)
    assert bool(np.asarray(routes).any()), "degenerate combine cell: no routed rows"
    pos, mir = jnp.asarray(plan["pos"]), jnp.asarray(plan["mirror_pos"])
    native = np.asarray(jax.jit(spec.incumbent_combine)(arr, scl, pos, mir, w, routes), np.float64)
    want = spec.oracle_combine(arr, scl, pos, mir, w, routes)
    assert np.abs(want).max() > 0, "degenerate combine cell: oracle output is zero"
    scale = np.maximum(np.abs(want).max(1, keepdims=True), 1e-30)
    err = np.abs(native - want) / scale
    tol = max(4 * float(err.max()), 2.0 ** -8)  # bf16 output rounding, calibrated on native
    print(f"\nCOMBINE {cn}: native max rel err {err.max():.3e}, tol {tol:.3e}")
    faults = {
        "pos+1": (pos + (pos < cell.recv_rows - 1), mir, w),
        "mirror_lane+1": (pos, mir + 1, w),
        "drop_k0": (pos, mir, w.at[:, 0].set(0.0)),
        "weight_missing": (pos, mir, jnp.ones_like(w)),
    }
    for name, (p2, m2, w2) in faults.items():
        bad = spec.oracle_combine(arr, scl, p2, m2, w2, routes)
        e2 = np.abs(bad - want) / scale
        assert e2.max() > tol, f"combine fault {name} not caught ({e2.max():.3e} <= {tol:.3e})"


@pytest.mark.parametrize("fault", list(spec.CANDIDATE_FAULTS))
def test_candidate_fault_caught(fault):
    """Positive control for the candidate: each kernel fault turns the oracle
    comparison red on a production cell (dup_nohi: on the duplicate-id edge
    cell, the only class where it can act)."""
    kw = spec.CANDIDATE_FAULTS[fault]
    cells = [("mix256", "dup")] if fault == "dup_nohi" else [
        ("mix128", "served"), ("mix128", "served_pad"), ("mix256", "served")]
    caught = []
    for cn, kind in cells:
        cell, pairs, sb, ex, mes = _case(cn, kind)
        for me in mes:
            lb, bh = spec.local_inputs(cell, pairs, me)
            f = jax.jit(lambda e, m: spec.candidate_plan(cell, e, None, None, m, not _on_tpu(), **kw))
            got = jax.device_get(f(ex, jnp.int32(me)))
            bad = spec.compare(spec.oracle_plan(cell, pairs, sb, me), got)
            if bad:
                caught.append((cn, kind, me, bad))
    print(f"\nCANDIDATE FAULT {fault}: caught on {len(caught)} (cell, shard) -> {caught[:3]}")
    assert caught, f"candidate fault {fault} not caught"


# ---------------------------------------------------------------- stage A2 combine

A2_CELLS = ("mix128", "mix256", "tw16", "tw4")


def _combine_case(cn, seed=3):
    cell = spec.CELLS[cn]
    pairs = spec.make_pairs(cell, "served_pad", seed)
    sb = spec.make_scale_bits(cell, seed)
    ex = spec.make_exchange(cell, pairs, sb)
    me = 0  # bucket padding sits in the tail sources, so source 0 routes
    plan = _run("incumbent", cell, pairs, ex, me)
    arr, scl, w, routes = spec.make_combine_inputs(cell, pairs, me, seed)
    assert bool(np.asarray(routes).any()), "degenerate combine cell: no routed rows"
    return cell, arr, scl, jnp.asarray(plan["pos"]), jnp.asarray(plan["mirror_pos"]), w, routes


@pytest.mark.parametrize("variant", list(spec.COMBINE_VARIANTS))
@pytest.mark.parametrize("cn", A2_CELLS)
def test_a2_combine_vs_incumbent_oracle_and_faults(cn, variant):
    """A2 candidate: bit-exact vs the incumbent (reported; the gate if it holds),
    within 4x native error of the f64 oracle, NaN-safe on unwritten rows, and
    every combine fault applied to the CANDIDATE caught."""
    if not _on_tpu():
        pytest.skip("the incumbent combine runs the TPU path")
    cell, arr, scl, pos, mir, w, routes = _combine_case(cn)
    cand = jax.jit(lambda *xs, **kw: spec.candidate_combine(*xs, variant=variant, **kw),
                   static_argnames=("opts",))
    inc = jax.jit(spec.incumbent_combine)
    native = np.asarray(inc(arr, scl, pos, mir, w, routes), np.float64)
    got = np.asarray(cand(arr, scl, pos, mir, w, routes), np.float64)
    want = spec.oracle_combine(arr, scl, pos, mir, w, routes)
    scale = np.maximum(np.abs(want).max(1, keepdims=True), 1e-30)
    nerr = float((np.abs(native - want) / scale).max())
    err = float((np.abs(got - want) / scale).max())
    tol = max(4 * nerr, 2.0 ** -8)
    nbits = int((got != native).sum())
    print(f"\nA2 {variant} {cn}: bitexact={nbits == 0} ndiff={nbits}/{got.size} "
          f"err {err:.3e} native {nerr:.3e} tol {tol:.3e}")
    assert err <= tol
    # NaN poison: every 5th token made unrouted (weight 0, as the router leaves a
    # padding row), and the arrival rows only those tokens read set to NaN.
    keep = (np.arange(cell.t_local) % 5 != 0)[:, None]
    r_nan = jnp.asarray(np.asarray(routes) & keep)
    w_nan = jnp.where(r_nan, w, 0.0)
    arr_nan = spec.nan_poison(arr, pos, r_nan)
    assert not np.isfinite(np.asarray(arr_nan.astype(jnp.float32))).all(), "poison cell is empty"
    g_nan = np.asarray(cand(arr_nan, scl, pos, mir, w_nan, r_nan), np.float64)
    n_nan = np.asarray(inc(arr_nan, scl, pos, mir, w_nan, r_nan), np.float64)
    assert np.isfinite(g_nan).all(), "NaN leaked from an unwritten row"
    np.testing.assert_array_equal(g_nan, n_nan)
    # input faults on the candidate
    faults = {
        "pos+1": (pos + (pos < cell.recv_rows - 1), mir, w),
        "mirror_lane+1": (pos, mir + 1, w),
        "drop_k0": (pos, mir, w.at[:, 0].set(0.0)),
        "weight_missing": (pos, mir, jnp.ones_like(w)),
    }
    caught = {}
    for name, (p2, m2, w2) in faults.items():
        bad = np.asarray(cand(arr, scl, p2, m2, w2, routes), np.float64)
        caught[name] = float((np.abs(bad - want) / scale).max()) > tol
    # kernel faults (k_order is visible only to the bit-exact check)
    for f in ("pick_lane", "row_off", "k_order"):
        bad = np.asarray(cand(arr, scl, pos, mir, w, routes, opts=(f,)), np.float64)
        caught[f] = (bool((bad != native).any()) if f == "k_order"
                     else float((np.abs(bad - want) / scale).max()) > tol)
    bad = np.asarray(cand(arr_nan, scl, pos, mir, w_nan, r_nan, opts=("route_mul",)), np.float64)
    caught["route_mul(nan)"] = not np.isfinite(bad).all()
    print(f"A2 FAULTS {variant} {cn}: " + " ".join(f"{k}={'caught' if v else 'BLIND'}" for k, v in caught.items()))
    blind = [k for k, v in caught.items() if not v and not (k == "k_order" and nbits)]
    assert not blind, f"faults not caught: {blind}"
    assert nbits == 0, f"not bit-exact vs incumbent ({nbits} elements differ)"
