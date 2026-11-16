# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_PREFILL_COMPARE on TPU (skipped elsewhere): hc_prefill_cmp
against the served prefill HC kernels (hc_prefill_fused at
TPU_QWEN4_HC_PREFILL_NORM=bf16ep64, gated_residual) on the same inputs, at
Flash's shapes and the prefill chunk sizes.

The kernels differ in the norm's rounding (the cmp kernel rounds once, as the
torch path does; the served one twice), the summation orders and the
sigmoid's implementation, so the gate is a tolerance, not equality. The
reference is float64 host math with the cmp kernel's rounding points. Per
output (mixed, injection):

* accuracy: cmp against the reference, max <= 2^-7 of max |ref| and
  l2 <= 1e-3 of ||ref||;
* no less accurate than served: cmp's l2 error at most twice served's (plus
  1e-5), cmp's max error at most max(served's, 2^-7 of max |ref|);
* against served: l2 <= 1e-2 of ||served||, and max at most served's own max
  error against the reference plus 2^-7 of max |served| (the served kernel's
  double rounding alone puts it 0.96% of max from the reference at 512 rows in
  interpret mode, so a fixed 1%-of-max line would gate on served's error).

The combined streams equal the served kernel's bit for bit, and a second call
returns the same bits. The fold's premise: the combine prologue gives the
combined streams of gated_residual bit for bit, and its mixed rows and
injection equal those of gated_residual followed by the mix without prologue
within max 2^-7 of max and l2 1e-3. They are not bit for bit on the TPU: the
compiled prologue kernel forms the sum of squares of its freshly rounded rows
with bf16 multiply-accumulates into f32 accumulators (vmac.bf16.f32), the mix
kernel with f32 multiplies and adds, so the variance can differ in its last
bits and move a norm rounding (CPU interpret mode gives equal bits).

Run it on a TPU host, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=python -m pytest -q -s
tests/kernels/test_qwen4_exp_hc_prefill_cmp_tpu.py"``. With P5HC_OUT set, one
JSON line per case is appended to $P5HC_OUT/numerics.jsonl.
"""

import json
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_cmp import hc_prefill_mix_cmp
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix


@pytest.fixture(autouse=True)
def _tpu_only():
    # At run time: a backend initialized during collection would claim the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")


BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 2560, 320, 1e-6, 256
WIDTH = BRANCHES * HIDDEN
# Prefill chunk rows: 4,096 is s10623's chunk (3,882 rows padded), 3,904 a
# partial last tile, 8,192 the prefill max-num-batched-tokens.
TOKENS = [512, 3904, 4096, 8192]


def _inputs(seed, tokens):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = (
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    )
    return residual, block, injection.astype(jnp.float32), weights


def _np(a):
    return np.asarray(jnp.asarray(a, jnp.float32), np.float64)


def _bf16(a):
    return a.astype(ml_dtypes.bfloat16).astype(np.float64)


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def _reference(h, weights):
    """float64 mix of the (already combined) hyper input ``h`` with the cmp
    kernel's bf16 rounding points: (mixed, injection)."""
    w, down, up, inject = (_np(a) for a in weights)
    norm = []
    for b in range(BRANCHES):
        v = h[:, b * HIDDEN : (b + 1) * HIDDEN]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + w[b * HIDDEN : (b + 1) * HIDDEN])))
    norm = np.concatenate(norm, axis=1)
    z = _bf16(_bf16(norm @ down.T) / BRANCHES)
    gate = _bf16(z * _sigmoid(z))
    acc = 0.0
    for b in range(BRANCHES):
        s = _sigmoid(_bf16(gate @ up[b * HIDDEN : (b + 1) * HIDDEN].T))
        acc = acc + s * norm[:, b * HIDDEN : (b + 1) * HIDDEN]
    return _bf16(acc / BRANCHES), 2.0 * _sigmoid(_bf16(norm @ inject.T) / BRANCHES)


def _metrics(actual, expected):
    d = np.abs(actual - expected)
    return dict(
        max_rel=float(d.max() / max(np.abs(expected).max(), 1e-30)),
        l2=float(np.linalg.norm(d) / max(np.linalg.norm(expected), 1e-30)),
        diff_frac=float((d > 0).mean()),
    )


@pytest.mark.parametrize("prologue", [True, False])
@pytest.mark.parametrize("tokens", TOKENS)
def test_tpu_cmp_matches_served(tokens, prologue):
    residual, block, injection, weights = _inputs(7 * tokens + int(prologue), tokens)
    partial = bool(tokens % TILE)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, partial=partial)
    extra = dict(block=block, prev_injection=injection) if prologue else {}
    served = hc_prefill_mix(residual, *weights, **extra, norm_impl="bf16ep64", **static)
    cmp = hc_prefill_mix_cmp(residual, *weights, **extra, **static)
    again = hc_prefill_mix_cmp(residual, *weights, **extra, **static)
    combined = _np(served[2]) if prologue else _np(residual)
    ref = _reference(combined, weights)

    record = dict(tokens=tokens, prologue=int(prologue))
    failures = []
    for i, name in enumerate(("mixed", "injection", "combined")):
        if i >= len(served):
            continue
        s, c = _np(served[i]), _np(cmp[i])
        if not np.isfinite(c).all():
            failures.append((name, "not finite"))
            continue
        if not np.array_equal(c, _np(again[i])):
            failures.append((name, "not deterministic"))
        if name == "combined":
            record[name] = dict(equal=bool(np.array_equal(c, s)))
            if not np.array_equal(c, s):
                failures.append((name, "combined streams differ", _metrics(c, s)))
            continue
        vs, cr, sr = _metrics(c, s), _metrics(c, ref[i]), _metrics(s, ref[i])
        record[name] = dict(vs_served=vs, cmp_vs_ref=cr, served_vs_ref=sr)
        if cr["max_rel"] > 2.0**-7 or cr["l2"] > 1e-3:
            failures.append((name, "tolerance against the float64 reference", cr))
        if cr["l2"] > 2.0 * sr["l2"] + 1e-5 or cr["max_rel"] > max(sr["max_rel"], 2.0**-7):
            failures.append((name, "less accurate than served", cr, sr))
        if vs["l2"] > 1e-2 or vs["max_rel"] > sr["max_rel"] + 2.0**-7:
            failures.append((name, "tolerance against served", vs, sr))
    if prologue:
        # The fold's premise: prologue = gated_residual, then the mix.
        sep = gated_residual_fn(BRANCHES, HIDDEN, TILE, partial)(residual, block, injection)
        mixed2, inj2 = hc_prefill_mix_cmp(sep, *weights, **static)
        if not np.array_equal(np.asarray(cmp[2]), np.asarray(sep)):
            failures.append(("fold", "combined streams differ from gated_residual"))
        fold = {}
        for name, a, b in (("mixed", cmp[0], mixed2), ("injection", cmp[1], inj2)):
            fold[name] = _metrics(_np(a), _np(b))
            fold[name]["sep_vs_ref"] = _metrics(_np(b), ref[0 if name == "mixed" else 1])
            if fold[name]["max_rel"] > 2.0**-7 or fold[name]["l2"] > 1e-3:
                failures.append(("fold", name, fold[name]))
        record["prologue_vs_separate"] = fold
    record["failures"] = [str(f) for f in failures]
    print("P5HC_NUM " + json.dumps(record, sort_keys=True))
    out = os.environ.get("P5HC_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "numerics.jsonl"), "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    assert not failures, failures
