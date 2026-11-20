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
"""JAX bodies of the MiniMax-M3 sequence-parallel ops, on CPU: the fused SP
shard add + norm + e4m3 quantize kernel (interpret mode) against the unfused
chain model.py runs without TPU_MINIMAX_M3_SP_FUSED_NORM, and the e4m3 matmuls
(TPU_MINIMAX_M3_SP_E4M3_QKV, TPU_MINIMAX_M3_SP_A2A_E4M3) against f64."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch_tpu")

from vllm_torchtpu.models.minimax_m3 import quant, sp_o_proj  # noqa: E402
from vllm_torchtpu.models.minimax_m3.kernels import sp_norm  # noqa: E402

EPS = 1e-6


def _rel(a, ref):
    a = np.asarray(a, np.float64)
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def _unfused(shard, update, weight):
    """model.py's chain: bf16 residual sum, f32 norm rounded to bf16, per-row
    e4m3 with scale amax / 448 (x / scale)."""
    r = (shard.astype(jnp.float32) + update.astype(jnp.float32)).astype(jnp.bfloat16)
    rf = r.astype(jnp.float32)
    var = jnp.mean(rf * rf, axis=-1, keepdims=True)
    normed = (rf / jnp.sqrt(var + EPS) * weight).astype(jnp.bfloat16).astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(normed), -1, keepdims=True), 1e-12) / 448.0
    return r, (normed / scale).astype(jnp.float8_e4m3fn), scale


@pytest.mark.parametrize("rows", [8, 128, 256])
def test_add_norm_quant_matches_unfused_chain(rows):
    rng = np.random.default_rng(rows)
    hidden = 512
    shard = jnp.asarray(rng.standard_normal((rows, hidden)) * 3, jnp.bfloat16)
    update = jnp.asarray(rng.standard_normal((rows, hidden)), jnp.bfloat16)
    weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(hidden), jnp.float32)
    res, q, scale = sp_norm.add_norm_quant(shard, update, weight, eps=EPS, interpret=True)
    r_ref, q_ref, s_ref = _unfused(shard, update, weight)
    assert res.dtype == jnp.bfloat16 and q.dtype == jnp.float8_e4m3fn
    assert scale.shape == (rows, 1) and scale.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(res, np.float32), np.asarray(r_ref, np.float32))
    np.testing.assert_allclose(np.asarray(scale), np.asarray(s_ref), rtol=1e-5)
    qf, qr = np.asarray(q, np.float32), np.asarray(q_ref, np.float32)
    # x * (1 / scale) against x / scale: a rare element lands one e4m3 step off.
    assert np.mean(qf != qr) < 2e-3
    np.testing.assert_allclose(qf, qr, rtol=0.13, atol=2**-9)


def test_add_norm_quant_rejects_ragged_rows():
    x = jnp.zeros((192, 128), jnp.bfloat16)
    with pytest.raises(AssertionError):
        sp_norm.add_norm_quant(x, x, jnp.ones(128), eps=EPS, interpret=True)


def test_quantize_rows_pow2():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((32, 256)) * 10.0 ** rng.integers(-3, 3, (32, 1)), jnp.bfloat16)
    q, s = sp_o_proj._quantize_rows_pow2(x)
    s = np.asarray(s, np.float64)
    assert q.dtype == jnp.float8_e4m3fn and s.shape == (32, 1)
    # Powers of two up to exp2's rounding (XLA CPU's exp2 is ~1 ulp off).
    np.testing.assert_allclose(np.exp2(np.round(np.log2(s))), s, rtol=1e-6)
    assert np.abs(np.asarray(q, np.float32)).max() <= 448.0
    deq = np.asarray(q, np.float64) * s
    assert _rel(deq, np.asarray(x, np.float64)) < 0.04


def test_o_proj_e4m3_matches_f64():
    """Every rank's heads of T rows (e4m3 rows, power-of-two row scales)
    contracted against every rank's weight block: the sum over ranks of each
    block's matmul."""
    rng = np.random.default_rng(1)
    tp, t, k, n = 4, 16, 64, 96
    heads = jnp.asarray(rng.standard_normal((tp, t, k)) * 2.0 ** rng.integers(-4, 4, (tp, t, 1)), jnp.bfloat16)
    q, r = sp_o_proj._quantize_rows_pow2(heads)
    w = jnp.asarray(rng.standard_normal((tp, k, n)), jnp.float8_e4m3fn)
    s = jnp.asarray(2.0 ** rng.integers(-8, -4, n), jnp.bfloat16)
    out = sp_o_proj._o_proj_e4m3(q, r, w, s)
    assert out.shape == (t, n) and out.dtype == jnp.bfloat16
    qf = np.asarray(q, np.float64) * np.asarray(r, np.float64)
    ref = np.einsum("eti,eio->to", qf, np.asarray(w, np.float64)) * np.asarray(s, np.float64)
    # Rows shift onto the largest scale of their T row (exact but for
    # underflow), then the output rounds to bf16.
    assert _rel(out, ref) < 2e-2
    ref_bf16 = np.einsum("eti,eio->to", np.asarray(heads, np.float64), np.asarray(w, np.float64)) * np.asarray(s, np.float64)
    assert _rel(out, ref_bf16) < 6e-2


def test_e4m3_rows_matmul_matches_f64():
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.standard_normal((32, 256)) * 50, jnp.float8_e4m3fn)
    rs = jnp.asarray(rng.uniform(0.001, 0.01, (32, 1)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((256, 128)), jnp.float8_e4m3fn)
    ws = jnp.asarray(2.0 ** rng.integers(-8, -4, 128), jnp.bfloat16)
    out = quant._e4m3_rows_matmul(q, rs, w, ws)
    assert out.shape == (32, 128) and out.dtype == jnp.bfloat16
    ref = (np.asarray(q, np.float64) * np.asarray(rs, np.float64)) @ np.asarray(w, np.float64)
    ref *= np.asarray(ws, np.float64)
    assert _rel(out, ref) < 4e-3
