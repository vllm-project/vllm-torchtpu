# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU_FUSED_NORM_QKVA: the fused MLA decode prologue against a float64 truth.

Today's path is XLA add + RMSNorm, ``xla_quantized_matmul`` on the fp8
``[6144, 2624]`` weight, the split and the q_a / kv_a ``rms_norm``s. Holding the
kernel to that path bit for bit (y) or within 2 bf16 ulps (q_c, kv_c, k_pe)
fails on a v7x chip: the two paths accumulate the fp8 dot in different orders
and the XLA path is the one further from exact. So both paths are judged
against the same float64 evaluation of the op's math instead:

* new residual: bit for bit ``bf16(x + residual)``;
* y: every element within 2 bf16 ulps of the float64 add + RMSNorm rounded
  where the op casts, and no more off-truth elements than
  ``max(ceil(1e-4 x size), XLA's own count + 2)``. A float32 mean of squares can
  move the normalized value onto the other side of a bf16 midpoint, and times a
  weight below 1 that can cross a binade: 2 ulps (as test_fused_add_rmsnorm);
* q_c, kv_c, k_pe: relative L2 error to the float64 quantized math (per-row fp8
  activation quant with an exact division, dot, dequant, split, RMSNorms) at
  most ``REL_L2_MAX`` and at most ``RATIO_MAX`` x the XLA path's own error;
* above 16 rows, the fp8 q_c and its scales bit for bit ``quantize_tensor`` of
  the kernel's own q_c.

Controls perturb the kernel itself: run with kv_a eps 1e-2 it must fail the
kv_c rules (and only those), and with distinct q / kv eps it must pass while
the run with the two swapped fails. Every test runs in interpret mode on the
host and compiled on a TPU (``tpu_test``).

Truth-side casts round once, in float64 (``_to_bf16`` / ``_to_fp8``): ml_dtypes
before 0.7 casts float64 through float32 and can round twice next to a midpoint.
"""

import math

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_norm_qkva as K
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

H, Q, KV, ROPE = 6144, 2048, 512, 64
EPS, Q_EPS, KV_EPS = 1e-5, 1e-6, 1e-6
FP8_MAX = float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)
# Measured on a v7x chip for rows 4-64: kernel rel-L2 <= 0.0041, and <= 0.72 x
# the XLA path's at every row count and output.
REL_L2_MAX = 0.006
RATIO_MAX = 1.10
Y_ULPS_MAX = 2
Y_OFF_FRAC = 1e-4
Y_XLA_SLACK = 2


def _inputs(rows, seed=0):
    rng = np.random.default_rng(seed + rows)
    x = (rng.standard_normal((rows, H), dtype=np.float32) * 0.5).astype(jnp.bfloat16)
    r = (rng.standard_normal((rows, H), dtype=np.float32) * 2.0).astype(jnp.bfloat16)
    x[-1] = 0  # a padded row: amax 0 -> scale 1
    r[-1] = 0
    ln_w = (1 + 0.1 * rng.standard_normal(H, dtype=np.float32)).astype(jnp.bfloat16)
    w_q = (rng.standard_normal((H, Q + KV + ROPE), dtype=np.float32) * 0.3).astype(
        jnp.float8_e4m3fn
    )
    w_s = rng.uniform(0.002, 0.02, Q + KV + ROPE).astype(np.float32)
    qa_w = (1 + 0.1 * rng.standard_normal(Q, dtype=np.float32)).astype(jnp.bfloat16)
    kva_w = (1 + 0.1 * rng.standard_normal(KV, dtype=np.float32)).astype(jnp.bfloat16)
    return tuple(jnp.asarray(a) for a in (x, r, ln_w, w_q, w_s, qa_w, kva_w))


def _bits(a):
    return np.asarray(a).tobytes()


def _rne(x, mant_bits, min_exp):
    """x rounded once to ``mant_bits`` stored mantissa bits, ties to even,
    exactly in float64 (subnormal below 2^min_exp)."""
    x = np.asarray(x, np.float64)
    with np.errstate(all="ignore"):
        _, e = np.frexp(x)
        q = np.ldexp(1.0, np.maximum(e - 1, min_exp) - mant_bits)
        r = np.rint(x / q) * q
    return np.where(np.isfinite(x), r, x)


def _to_bf16(x):
    return _rne(x, 7, -126).astype(ml_dtypes.bfloat16)


def _to_fp8(x):
    return _rne(x, 3, -6).astype(ml_dtypes.float8_e4m3fn)


def _addnorm_truth(x, residual, ln_w):
    """(y, new residual) of add + RMSNorm in float64, rounded where the op casts."""
    s = np.asarray(x, np.float64) + np.asarray(residual, np.float64)
    n = s / np.sqrt((s * s).mean(axis=1, keepdims=True) + EPS)
    y = _to_bf16(_to_bf16(n).astype(np.float64) * np.asarray(ln_w, np.float64))
    return y, _to_bf16(s)


def _qkva_truth(y, w_q, w_s, qa_w, kva_w, q_eps, kv_eps):
    """(q_c, kv_c, k_pe) of the quantized math in float64, from the truth's y."""
    yf = np.asarray(y, np.float64)
    scale = np.abs(yf).max(axis=1, keepdims=True) / FP8_MAX
    scale = np.where(scale == 0, 1.0, scale)
    x_q = _to_fp8(np.clip(yf / scale, -FP8_MAX, FP8_MAX)).astype(np.float64)
    qkv = (
        (x_q @ np.asarray(w_q).astype(np.float64)) * scale * np.asarray(w_s, np.float64)
    )

    def rms(v, w, e):
        return (
            v
            / np.sqrt((v * v).mean(axis=1, keepdims=True) + e)
            * np.asarray(w, np.float64)
        )

    return (
        rms(qkv[:, :Q], qa_w, q_eps),
        rms(qkv[:, Q : Q + KV], kva_w, kv_eps),
        qkv[:, Q + KV :],
    )


def _rel_l2(got, truth):
    g = np.asarray(got, np.float64)
    return float(np.linalg.norm(g - truth) / max(np.linalg.norm(truth), 1e-30))


def _ulps(a, b):
    """Elementwise distance in bf16 steps; non-finite counts as huge."""

    def key(v):
        bits = np.asarray(v).astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int64)
        mag = bits & 0x7FFF
        return np.where(bits & 0x8000, -mag, mag)

    finite = np.isfinite(np.asarray(a, np.float64)) & np.isfinite(
        np.asarray(b, np.float64)
    )
    return np.where(finite, np.abs(key(a) - key(b)), 1 << 30)


def _on_tpu():
    return jax.devices()[0].platform == "tpu"


# The same checks run twice: on the host with the kernel in interpret mode (the
# CPU lane; there it computes what XLA does, so only the truth side and the
# controls are exercised), and compiled on a TPU (the TPU lane, where the
# thresholds above were measured).
BACKENDS = [
    pytest.param("interpret", id="interpret"),
    pytest.param("tpu", id="tpu", marks=pytest.mark.tpu_test),
]


def _require(backend):
    if backend == "tpu" and not _on_tpu():
        pytest.skip("the compiled kernel needs a TPU")
    if backend == "interpret" and _on_tpu():
        pytest.skip("interpret mode is a host-only check")


def _run(backend, args, rows, q_eps, kv_eps):
    tpu = backend == "tpu"
    return K.fused_norm_qkva(
        *args,
        eps=EPS,
        q_eps=q_eps,
        kv_eps=kv_eps,
        q_rank=Q,
        kv_rank=KV,
        emit_q8=rows > 16,
        hbm_pin=tpu,
        interpret=not tpu,
    )


def _qkva_errors(out, args, q_eps, kv_eps):
    y_truth, _ = _addnorm_truth(*args[:3])
    truth = _qkva_truth(y_truth, *args[3:], q_eps, kv_eps)
    return [_rel_l2(g, t) for g, t in zip(out[2:5], truth)]


def _failures(rows, args, got, q_eps, kv_eps):
    """Every rule the kernel outputs ``got`` break, judged at ``q_eps`` /
    ``kv_eps``; empty when they pass."""
    xla = K.reference(*args, eps=EPS, q_eps=q_eps, kv_eps=kv_eps, q_rank=Q, kv_rank=KV)
    bad = []
    y_truth, res_truth = _addnorm_truth(*args[:3])
    if _bits(got[1]) != _bits(res_truth):
        bad.append("new residual differs from bf16(x + residual)")
    y_off = _ulps(got[0], y_truth)
    allowed = max(
        math.ceil(Y_OFF_FRAC * y_off.size),
        int((_ulps(xla[0], y_truth) > 0).sum()) + Y_XLA_SLACK,
    )
    if y_off.max() > Y_ULPS_MAX:
        bad.append(f"y off by {y_off.max()} ulps > {Y_ULPS_MAX}")
    if (y_off > 0).sum() > allowed:
        bad.append(f"y: {(y_off > 0).sum()} off-truth elements > {allowed}")
    for name, k_err, x_err in zip(
        ("q_c", "kv_c", "k_pe"),
        _qkva_errors(got, args, q_eps, kv_eps),
        _qkva_errors(xla, args, q_eps, kv_eps),
    ):
        if k_err > REL_L2_MAX:
            bad.append(f"{name}: rel-L2 {k_err:.5f} > {REL_L2_MAX}")
        if k_err > RATIO_MAX * x_err:
            bad.append(f"{name}: rel-L2 {k_err:.5f} > {RATIO_MAX} x XLA's {x_err:.5f}")
    if rows > 16:
        q8, qs = quantize_tensor(got[2], jnp.float8_e4m3fn)
        if _bits(got[5]) != _bits(q8) or _bits(got[6]) != _bits(qs):
            bad.append("fp8 q_c differs from quantize_tensor(q_c)")
    return bad


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", [4, 8, 16, 32, 48, 64])
def test_matches_float64_truth(backend, rows):
    _require(backend)
    args = _inputs(rows)
    got = _run(backend, args, rows, Q_EPS, KV_EPS)
    assert _failures(rows, args, got, Q_EPS, KV_EPS) == []


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", [4, 64])
def test_planted_fault_is_caught(backend, rows):
    """The kernel run with kv_a eps 1e-2 must fail the kv_c rules and only them."""
    _require(backend)
    args = _inputs(rows)
    got = _run(backend, args, rows, Q_EPS, 1e-2)
    bad = _failures(rows, args, got, Q_EPS, KV_EPS)
    assert bad and all(b.startswith("kv_c") for b in bad), bad


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", [4, 64])
def test_distinct_eps_are_not_swapped(backend, rows):
    """With distinct q / kv eps the kernel passes, and the kernel run with the
    two swapped fails."""
    _require(backend)
    q_eps, kv_eps = 1e-3, 1e-6
    args = _inputs(rows)
    got = _run(backend, args, rows, q_eps, kv_eps)
    assert _failures(rows, args, got, q_eps, kv_eps) == []
    swapped = _run(backend, args, rows, kv_eps, q_eps)
    bad = _failures(rows, args, swapped, q_eps, kv_eps)
    assert any(b.startswith(("q_c", "kv_c")) for b in bad), bad
