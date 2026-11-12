# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX on the real DCP block-sparse prefill
(DCP2_HARNESS_DEVICE=tpu; skipped elsewhere; also runs as plain python).

Per case, with the flag on:
- the full DCP-2 gate (``check_production_prefill``): gated <= 2 BF16 ulps and
  pre-gate within TPU_TOL against the replicated served route, no NaN;
- each rank's partial against the float32 reference (``check_partial``: NaN
  pools and NaN padding rows, rows owning nothing exactly 0 / -inf);
- its error against the float32 reference (max and p99, relative to each
  head vector's max) at most twice the flag-off kernel's.
The guard case gives a few rows of every query subtile 64x larger queries,
so the other rows' probabilities underflow below the shared stabilizer: the
fallback must fire for every block and reproduce the flag-off output bit
for bit, and the same case without the guard (``noguard``, the positive
control) must fail the partial check.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
    ON_TPU,
    TPU_TOL,
    check_partial,
    make_case,
    prefill_case,
    rank_operand,
    reference_attend,
    reference_metadata,
)
from test_dcp2_prefill import check_production_prefill, production_prefill_attend

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

FLAG = "VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX"
SERVED_ENV = {"VLLM_TPU_QSA_PREFILL_LEAN": "1", "VLLM_TPU_QSA_DCP_SPLIT_LSE": "1"}

PREFILL = {
    "c2048_p1234": [(1234, 2048)],
    "c2048_p5000": [(5000, 2048)],
    "c8192_p9003": [(9003, 8192)],
    # Fresh request: rank 1 owns nothing for rows 1..511.
    "ragged_mix": [(5000, 1000), (0, 700), (9003, 347), (2047, 1)],
}


def with_flag(value, fn):
    old = os.environ.get(FLAG)
    os.environ[FLAG] = value
    jax.clear_caches()  # the flag is read inside a jitted wrapper
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old
        jax.clear_caches()


def nan_padded_case(rng, chunks, pad):
    """prefill_case plus ``pad`` NaN-query padding rows of length 0."""
    lengths, req, seq = [], [], []
    for r, (prefix, count) in enumerate(chunks):
        lengths += list(range(prefix + 1, prefix + count + 1))
        req += [r] * count
        seq.append(prefix + count)
    nan_rows = list(range(len(lengths), len(lengths) + pad))
    lengths += [0] * pad
    req += [len(chunks) - 1] * pad
    return make_case(rng, row_lengths=lengths, row_req=req, seq_tokens=seq,
                     nan_rows=nan_rows, budget=512)


def kernel_partials(case):
    attend = production_prefill_attend(case)
    return [
        tuple(np.asarray(x) for x in attend(
            case.q, case.gate, case.blocks, case.lengths, case.dcp_rows,
            case.rank_pools[r], dcp=(rank_operand(r), D), **case.opts))
        for r in range(D)
    ]


def reference_partials(case):
    return [
        tuple(np.asarray(x) for x in reference_attend(
            case.q, case.gate, case.blocks, case.lengths, case.dcp_rows,
            case.rank_pools[r], dcp=(rank_operand(r), D), **case.opts))
        for r in range(D)
    ]


def error_stats(out, ref):
    """|out - ref| / max|ref head vector|, over rows with a finite reference."""
    out = np.asarray(out, np.float64)
    ref = np.asarray(ref, np.float64)
    scale = np.max(np.abs(ref), axis=-1, keepdims=True)
    live = np.broadcast_to(scale > 0, ref.shape)
    err = (np.abs(out - ref) / np.where(scale > 0, scale, 1))[live]
    if not err.size:
        return dict(max=0.0, p99=0.0)
    return dict(max=float(err.max()), p99=float(np.percentile(err, 99)))


def check_lazy_case(case):
    today = with_flag("", lambda: kernel_partials(case))
    lazy = with_flag("1", lambda: kernel_partials(case))
    ref = reference_partials(case)
    report = {}
    for r in range(D):
        _, kv_lens, _, _ = reference_metadata(
            case.blocks, case.lengths, case.dcp_rows, ratio=case.ratio,
            page_size=case.layout.page_size, dcp=(r, D))
        check_partial(*lazy[r], *ref[r], kv_lens, TPU_TOL, what=f"lazy rank {r}")
        e_today = error_stats(today[r][0], ref[r][0])
        e_lazy = error_stats(lazy[r][0], ref[r][0])
        floor = 2.0**-20  # below f32-level noise, never binding at kernel precision
        for k in ("max", "p99"):
            assert e_lazy[k] <= max(2 * e_today[k], floor), (
                f"rank {r} {k}: lazy {e_lazy[k]:.3g} > 2 x today {e_today[k]:.3g}")
        report[f"rank{r}"] = dict(today=e_today, lazy=e_lazy,
                                  bitwise_vs_today=float(np.mean(lazy[r][0] == today[r][0])))
    report["gate"] = with_flag("1", lambda: check_production_prefill(case))
    return report


@pytest.mark.parametrize("name", sorted(PREFILL))
def test_lazy_softmax_prefill(name):
    case = prefill_case(np.random.default_rng(len(name)), PREFILL[name], budget=512)
    print(name, check_lazy_case(case))


def test_lazy_softmax_nan_padding():
    case = nan_padded_case(np.random.default_rng(7), [(0, 700), (3001, 1300)], pad=48)
    print("nan_padding", check_lazy_case(case))


def guard_case():
    case = prefill_case(np.random.default_rng(5), [(5000, 2048)], budget=512)
    rows = np.arange(case.q.shape[0])
    big = (rows % 256) < 8  # every query subtile of every block
    case.q = case.q * jnp.where(jnp.asarray(big)[:, None, None], 64.0, 1.0).astype(case.q.dtype)
    return case


def test_lazy_softmax_guard_fires():
    case = guard_case()
    today = with_flag("", lambda: kernel_partials(case))
    lazy = with_flag("1", lambda: kernel_partials(case))
    for r in range(D):
        for a, b, what in ((lazy[r][0], today[r][0], "out"), (lazy[r][1], today[r][1], "lse")):
            assert np.array_equal(a, b, equal_nan=True), (
                f"guard rank {r} {what}: fallback differs from flag off "
                f"({np.mean(a == b):.6f} bitwise)")
    # Positive control: the same case without the guard must underflow.
    unguarded = with_flag("1,noguard", lambda: kernel_partials(case))
    ref = reference_partials(case)
    failed = 0
    for r in range(D):
        _, kv_lens, _, _ = reference_metadata(
            case.blocks, case.lengths, case.dcp_rows, ratio=case.ratio,
            page_size=case.layout.page_size, dcp=(r, D))
        try:
            check_partial(*unguarded[r], *ref[r], kv_lens, TPU_TOL, what=f"noguard rank {r}")
        except AssertionError:
            failed += 1
    assert failed == D, f"positive control: only {failed}/{D} ranks failed without the guard"
    print("guard", dict(bitwise_vs_today=True, noguard_failed_ranks=failed))


if __name__ == "__main__":
    for k, v in SERVED_ENV.items():
        os.environ.setdefault(k, v)
    tests = [(f"prefill[{n}]", lambda n=n: test_lazy_softmax_prefill(n)) for n in sorted(PREFILL)]
    tests += [("nan_padding", test_lazy_softmax_nan_padding),
              ("guard_fires", test_lazy_softmax_guard_fires)]
    only = sys.argv[1:]
    failures = 0
    for name, fn in tests:
        if only and name not in only:
            continue
        try:
            fn()
            print(json.dumps({"test": name, "ok": True}), flush=True)
        except Exception as e:  # noqa: BLE001 (plain-python runner)
            failures += 1
            print(json.dumps({"test": name, "ok": False, "err": f"{type(e).__name__}: {e}"[:2000]}),
                  flush=True)
    print(json.dumps({"failures": failures}))
    sys.exit(1 if failures else 0)
