# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Accuracy instrument for the moe_tp_decode row-gather campaign (CPU tier; the TPU tier is accuracy_tpu.py).

CPU tier, one rank, small hidden (spec.TP16_SMALL / EP16_SMALL: the routing statistics of GLM-5.2 — 256 experts,
top-8 — at h=1024), m in {80, 112, 192}, routing kinds indep / mtp0.8 / zipf0.6, n_valid in {m, m-13}:
- routing: the constructed scores route exactly as intended (kernel routing rule).
- algebra: the row-gathered algorithm (refs.rg_partial) equals the incumbent's arithmetic (refs.native_partial):
  rel L2 <= 1e-6 and the bitwise fraction reported (CPU matmul rows need not be M-independent; the kernel-level
  bitwise claim is a TPU fact checked in-job).
- oracle: native vs float64 oracle within the tests' TOLERANCE (rel L2 0.06, cos 0.998).
- fault controls F1..F8 on rg_partial: catch = fails the native 1e-3 gate, padding +0 or NaN; catch rate per routing
  kind and deployment is printed; required: F1 F2 F3 F6 F7 F8 caught on every cell, F4 on every EP cell (NA on TP),
  F5 caught on every cell whose max rows/expert exceeds the slab (r_slab=16) and blind elsewhere (reported, by design).

Faults: F1 drop the last gathered row; F2 scatter to row r+1; F3 coefficient from slot j+1; F4 EP compact accumulator
not re-initialised (stale previous expert); F5 overflow passes disabled (slots >= R dropped); F6 slot table from expert
column e+1; F7 gathered row scales not gathered (slot 0's for all); F8 slot 0 added twice (TP: scatter; EP: compact accumulate).
"""

import collections
import json

import jax
import jax.numpy as jnp
import numpy as np

try:
    import pytest
except ImportError:  # script mode on hosts without pytest
    pytest = None

try:
    from . import refs, spec
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refs  # noqa: E402
    import spec  # noqa: E402

CPU_MS = (80, 112, 192)
R_SLAB = 16


def _inputs(dep, m, kind, n_valid, rank=3):
    scores, bias, p = spec.scores_bias(m, kind, spec.seed_of(m, dep.name, kind))
    x = jax.random.normal(jax.random.key(spec.seed_of(m, dep.name, kind, 1)), (m, dep.h)).astype(jnp.bfloat16)
    w = refs.rank_weights(jax.random.key(1000 + rank), dep)
    coef = refs.routing_coef(jnp.asarray(scores), jnp.asarray(bias), spec.TOPK)
    coef_x = refs.routing_coef(jnp.asarray(scores), jnp.asarray(bias), spec.TOPK, exact=True)
    valid = (jnp.arange(m) < n_valid)[:, None]
    coef, coef_x = jnp.where(valid, coef, 0.0), jnp.where(valid, coef_x, 0.0)
    e0 = rank * dep.e_l if dep.ep else 0
    return x, coef, coef_x, e0, w, p


def _cells():
    for dep in (spec.TP16_SMALL, spec.EP16_SMALL):
        for m in CPU_MS:
            for kind in spec.KINDS:
                yield dep, m, kind


def test_routing_reproduces_the_constructed_picks():
    for m in (16, 112, 192):
        for kind in spec.KINDS + ("zipf1.0+mtp0.8",):
            scores, bias, p = spec.scores_bias(m, kind, spec.seed_of(m, "x", kind))
            coef = np.asarray(refs.routing_coef(jnp.asarray(scores), jnp.asarray(bias), spec.TOPK))
            assert ((coef != 0) == p).all(), (m, kind)
            assert (p.sum(1) == spec.TOPK).all()


def _parametrize(fn):
    if pytest is None:
        return fn
    return pytest.mark.parametrize("dep, m, kind", list(_cells()), ids=lambda v: getattr(v, "name", str(v)))(fn)


@_parametrize
def test_algebra_and_oracle(dep, m, kind):
    for n_valid in (m, m - 13):
        x, coef, coef_x, e0, w, p = _inputs(dep, m, kind, n_valid)
        nat = refs.native_partial(x, coef, e0, *w, routed_scale=spec.ROUTED_SCALE)
        rg = refs.rg_partial(x, coef, e0, *w, routed_scale=spec.ROUTED_SCALE, r_slab=R_SLAB)
        # partials: padding rows of a partial are combine(0, shared(x_pad)), not 0 — compare valid rows only.
        c = refs.compare(rg[:n_valid], nat[:n_valid], n_valid, ref=nat[:n_valid])
        rows = np.arange(0, n_valid, max(1, n_valid // 6))
        orc = refs.oracle_partial_f64(x, coef_x, e0, *w, routed_scale=spec.ROUTED_SCALE, rows=rows)
        o = refs.compare(np.asarray(nat[rows].astype(jnp.float32)), orc, len(rows))
        print("CASE", json.dumps(dict(dep=dep.name, m=m, kind=kind, n_valid=n_valid, rg_vs_native=c, native_vs_f64=o,
                                      drivers=spec.drivers(p, n_valid, dep, R_SLAB))))
        assert c["rel_l2_vs_ref"] <= 1e-6, c
        assert o["rel_l2"] <= refs.TOL_ORACLE["rel_l2"] and o["cos"] >= refs.TOL_ORACLE["cos"], o


def test_fault_controls_catch_rate():
    caught = collections.defaultdict(lambda: [0, 0])
    detail = []
    for dep, m, kind in _cells():
        n_valid = m
        x, coef, coef_x, e0, w, p = _inputs(dep, m, kind, n_valid)
        nat = refs.native_partial(x, coef, e0, *w, routed_scale=spec.ROUTED_SCALE)
        d = spec.drivers(p, n_valid, dep, R_SLAB)
        local = p[:, e0 : e0 + dep.e_l]
        max_local = int(local.sum(0).max())
        for f in refs.FAULTS:
            if f == "F4" and not dep.ep:
                continue
            got = refs.rg_partial(x, coef, e0, *w, routed_scale=spec.ROUTED_SCALE, fault=f, r_slab=R_SLAB)
            c = refs.compare(got, nat, n_valid, ref=nat)
            hit = (c["rel_l2_vs_ref"] > refs.TOL_NATIVE) or c["nan"] > 0
            caught[(f, dep.name, kind)][0] += int(hit)
            caught[(f, dep.name, kind)][1] += 1
            detail.append((f, dep.name, m, kind, max_local, hit, c["rel_l2_vs_ref"]))
            if f == "F5":
                assert hit == (max_local > R_SLAB), (f, dep.name, m, kind, max_local, c)
            else:
                assert hit, (f, dep.name, m, kind, c)
    print("CATCH", json.dumps({"|".join(k): f"{v[0]}/{v[1]}" for k, v in sorted(caught.items())}))
    for row in detail:
        print("FAULT", row)


if __name__ == "__main__":
    import sys
    import time

    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if sel in ("all", "routing"):
        test_routing_reproduces_the_constructed_picks()
        print("PASS routing", round(time.time() - t0, 1))
    if sel in ("all", "algebra") or sel.startswith("algebra:"):
        want = sel.split(":", 1)[1] if ":" in sel else None
        for dep, m, kind in _cells():
            if want and want != f"{dep.name}-{m}-{kind}":
                continue
            test_algebra_and_oracle(dep, m, kind)
            print("PASS algebra", dep.name, m, kind, round(time.time() - t0, 1))
    if sel in ("all", "faults"):
        test_fault_controls_catch_rate()
        print("PASS faults", round(time.time() - t0, 1))
