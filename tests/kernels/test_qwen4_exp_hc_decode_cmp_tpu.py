# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_COMPARE on TPU (skipped elsewhere): hc_decode_cmp against
hc_decode at every decode bucket the cmp kernel takes.

The kernels share their rounding boundaries and differ in summation orders and
the sigmoid's implementation, so the gate is a bf16 tolerance, not equality:
per output, max |cmp - served| <= 2^-7 * max |served|, ||cmp - served||_2 <=
1e-3 * ||served||_2 and at most 2% of elements different (``_check_bf16`` of
test_qwen4_exp_hc_decode.py, the served kernel's own tolerance against the
torch path). Each kernel's distance to a float64 host reference with the same
bf16 rounding points is printed beside it, and the cmp kernel's relative L2
error must not exceed twice the served kernel's (plus 1e-5).

Run it on a TPU host, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=python -m pytest -q -s
tests/kernels/test_qwen4_exp_hc_decode_cmp_tpu.py"``.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_cmp import hc_decode_mix_cmp


@pytest.fixture(autouse=True)
def _tpu_only():
    # At run time: a backend initialized during collection would claim the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")


BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
# Decode buckets of the served and bench configs that the cmp kernel takes.
BUCKETS = [16, 32, 64, 96, 112, 128, 144, 160, 192, 208, 224, 240, 256]


def _inputs(seed, tokens, inject):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
    ]
    if inject:
        weights.append((0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16))
    return residual, block, injection.astype(jnp.float32), weights


def _np(a):
    return np.asarray(jnp.asarray(a, jnp.float32), np.float64)


def _bf16(a):
    return a.astype(ml_dtypes.bfloat16).astype(np.float64)


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def _reference(residual, block, injection, weights, prologue):
    """float64 math with the kernels' bf16 rounding points."""
    w, down, up = (_np(a) for a in weights[:3])
    h = _np(residual)
    if prologue:
        blk, inj = _np(block), _np(injection)
        h = np.concatenate(
            [
                _bf16(h[:, b * HIDDEN : (b + 1) * HIDDEN] + blk * inj[:, b : b + 1])
                for b in range(BRANCHES)
            ],
            axis=1,
        )
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
    mixed = _bf16(acc / BRANCHES)
    inj_out = None
    if len(weights) == 4:
        inj_out = 2.0 * _sigmoid(_bf16(norm @ _np(weights[3]).T) / BRANCHES)
    return mixed, inj_out, h


def _metrics(actual, expected):
    d = np.abs(actual - expected)
    return dict(
        max_rel=float(d.max() / max(np.abs(expected).max(), 1e-30)),
        l2=float(np.linalg.norm(d) / max(np.linalg.norm(expected), 1e-30)),
        diff_frac=float((d > 0).mean()),
    )


def _fmt(m):
    return f"max/scale {m['max_rel']:.2e} l2 {m['l2']:.2e} diff {m['diff_frac']:.4f}"


@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("prologue", [True, False])
@pytest.mark.parametrize("tokens", BUCKETS)
def test_tpu_cmp_matches_served(tokens, prologue, inject):
    _check(tokens, prologue, inject, None)


# The kernel's schedule options (copy lookahead MiB, overlap, one up dot):
# every form must give the default's numerics.
VARIANTS = [(4, True, False), (2, True, False), (8, True, True)]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "v%d%d%d" % v)
@pytest.mark.parametrize("tokens", [16, 64, 192, 256])
def test_tpu_cmp_variants_match_served(tokens, variant):
    la, ov, uw = variant
    _check(tokens, True, True, (("lookahead", la), ("overlap", ov), ("up_whole", uw)))


def _check(tokens, prologue, inject, opts):
    residual, block, injection, weights = _inputs(7 * tokens + 1, tokens, inject)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS)
    extra = dict(block=block, prev_injection=injection) if prologue else {}
    served = hc_decode_mix(residual, *weights, **extra, max_tile=512, **static)
    cmp = hc_decode_mix_cmp(residual, *weights, **extra, **static, _opts=opts)
    again = hc_decode_mix_cmp(residual, *weights, **extra, **static, _opts=opts)
    ref = _reference(residual, block, injection, weights, prologue)

    names = ["mixed", "injection", "combined"]
    lines, failures = [], []
    for i, name in enumerate(names):
        if i >= len(served) or served[i] is None:
            assert i >= len(cmp) or cmp[i] is None, name
            continue
        s, c, r = _np(served[i]), _np(cmp[i]), ref[i]
        assert np.isfinite(c).all(), name
        assert np.array_equal(c, _np(again[i])), (name, "not deterministic")
        vs = _metrics(c, s)
        cr, sr = _metrics(c, r), _metrics(s, r)
        lines.append(
            f"{name}: cmp-vs-served {_fmt(vs)} eq {np.mean(c == s):.6f} | "
            f"cmp-vs-ref l2 {cr['l2']:.2e} | served-vs-ref l2 {sr['l2']:.2e}"
        )
        if vs["max_rel"] > 2.0**-7 or vs["l2"] > 1e-3 or vs["diff_frac"] > 0.02:
            failures.append((name, "tolerance", vs))
        if cr["l2"] > 2.0 * sr["l2"] + 1e-5:
            failures.append((name, "less accurate than served", cr["l2"], sr["l2"]))
    print(
        f"HCCMP_NUM tokens {tokens} prologue {int(prologue)} inject {int(inject)} "
        f"opts {dict(opts) if opts else 'default'}: "
        + " || ".join(lines)
    )
    assert not failures, failures
