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
* y: every element within 1 bf16 ulp of the float64 add + RMSNorm rounded where
  the op casts, and no more off-truth elements than
  ``max(ceil(1e-4 x size), XLA's own count + 2)`` (float32 sums land on bf16
  midpoints now and then, on either path);
* q_c, kv_c, k_pe: relative L2 error to the float64 quantized math (per-row fp8
  activation quant with an exact division, dot, dequant, split, RMSNorms) at
  most ``REL_L2_MAX`` and at most ``RATIO_MAX`` x the XLA path's own error;
* above 16 rows, the fp8 q_c and its scales bit for bit ``quantize_tensor`` of
  the kernel's own q_c.

A planted fault (kv_a eps 1e-2) must fail the q_c / kv_c check, and a variant
with distinct q / kv eps must catch the two eps swapped.

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


def _run(args, rows, q_eps, kv_eps):
    on_tpu = jax.devices()[0].platform == "tpu"
    return K.fused_norm_qkva(
        *args,
        eps=EPS,
        q_eps=q_eps,
        kv_eps=kv_eps,
        q_rank=Q,
        kv_rank=KV,
        emit_q8=rows > 16,
        hbm_pin=on_tpu,
        interpret=not on_tpu,
    )


def _qkva_errors(out, args, q_eps, kv_eps):
    y_truth, _ = _addnorm_truth(*args[:3])
    truth = _qkva_truth(y_truth, *args[3:], q_eps, kv_eps)
    return [_rel_l2(g, t) for g, t in zip(out[2:5], truth)]


def _check(rows, q_eps, kv_eps):
    args = _inputs(rows)
    got = _run(args, rows, q_eps, kv_eps)
    xla = K.reference(*args, eps=EPS, q_eps=q_eps, kv_eps=kv_eps, q_rank=Q, kv_rank=KV)

    y_truth, res_truth = _addnorm_truth(*args[:3])
    assert _bits(got[1]) == _bits(res_truth)  # new residual
    y_off = _ulps(got[0], y_truth)
    allowed = max(
        math.ceil(Y_OFF_FRAC * y_off.size),
        int((_ulps(xla[0], y_truth) > 0).sum()) + Y_XLA_SLACK,
    )
    assert y_off.max() <= 1, f"y off by {y_off.max()} ulps"
    assert (y_off > 0).sum() <= allowed, (
        f"y: {(y_off > 0).sum()} off-truth elements > {allowed}"
    )

    for name, k_err, x_err in zip(
        ("q_c", "kv_c", "k_pe"),
        _qkva_errors(got, args, q_eps, kv_eps),
        _qkva_errors(xla, args, q_eps, kv_eps),
    ):
        assert k_err <= REL_L2_MAX, f"{name}: rel-L2 {k_err:.5f} > {REL_L2_MAX}"
        assert k_err <= RATIO_MAX * x_err, (
            f"{name}: rel-L2 {k_err:.5f} > {RATIO_MAX} x XLA's {x_err:.5f}"
        )

    if rows > 16:
        q8, qs = quantize_tensor(got[2], jnp.float8_e4m3fn)
        assert _bits(got[5]) == _bits(q8)
        assert _bits(got[6]) == _bits(qs)
    return args, got


@pytest.mark.parametrize("rows", [4, 8, 16, 32, 48, 64])
def test_matches_float64_truth(rows):
    args, got = _check(rows, Q_EPS, KV_EPS)
    # A planted fault must fail the check: the same outputs judged against a
    # truth with kv_a eps 1e-2.
    q_err, kv_err, _ = _qkva_errors(got, args, Q_EPS, 1e-2)
    assert kv_err > REL_L2_MAX and q_err <= REL_L2_MAX


@pytest.mark.parametrize("rows", [4, 64])
def test_distinct_eps_are_not_swapped(rows):
    q_eps, kv_eps = 1e-3, 1e-6
    args, got = _check(rows, q_eps, kv_eps)
    swapped = _qkva_errors(got, args, kv_eps, q_eps)
    right = _qkva_errors(got, args, q_eps, kv_eps)
    assert swapped[0] > RATIO_MAX * right[0] or swapped[1] > RATIO_MAX * right[1]
