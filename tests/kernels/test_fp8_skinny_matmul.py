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
"""fp8_skinny: the decode (M <= 16) weight-streaming FP8 matmul (TPU_FP8_LINEAR_PALLAS).

Judged against the float64 product of the SAME quantized operands (x_q, x_scale from
`util.quantize_tensor`, the per-channel weight and scale). The tolerance is set by the
XLA reference's own error on the cell: XLA's fp8 x fp8 dot accumulates below f32 and
this kernel does not, so the kernel must land inside 2x the reference's error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny, util

F8 = jnp.float8_e4m3fn


def _x(m, k, real, seed=0):
    rng = np.random.default_rng(seed)
    gain = np.exp(rng.normal(0, 0.5, k)).astype(np.float32)
    gain[rng.choice(k, 8, replace=False)] *= 20.0
    x = rng.standard_normal((m, k), dtype=np.float32) * gain[None]
    x[real:] = 0.0
    return jnp.asarray(x, jnp.bfloat16)


def _w(k, n, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((k, n), dtype=np.float32) * 0.02
    ws = np.abs(w).max(axis=0) / 448.0
    wq = np.clip(w / ws[None], -448, 448)
    return jnp.asarray(wq, F8), jnp.asarray(ws, jnp.float32)


def _oracle(x, wq, ws):
    xq, xs = util.quantize_tensor(x, F8)
    a = np.asarray(xq).astype(np.float64)
    a = a * np.asarray(xs).astype(np.float64).reshape(-1, 1)
    b = np.asarray(wq).astype(np.float64) * np.asarray(ws).astype(np.float64)[None]
    return a @ b


def _row_err(y, ref):
    y = np.asarray(y).astype(np.float64)
    m = np.abs(ref).max(axis=1)
    d = np.abs(y - ref).max(axis=1)
    return float(np.where(m > 0, d / np.where(m > 0, m, 1), d).max())


def _tol(x, wq, ws, ref):
    y_xla = jax.jit(util.xla_quantized_matmul)(x, wq, ws)
    return max(2 * _row_err(y_xla, ref), 2.0**-8)


# GLM-5.2-FP8 per-rank decode sites (TP16, q-replicate) at their tuned tilings, other
# tilings, and the generic picker.
SITES = [
    (2048, 16384, None),
    (1024, 6144, None),
    (6144, 1536, None),
    (6144, 2688, None),
    (1024, 6144, ("n", 512, 2)),
    (6144, 1536, ("k", 768, 4)),
    (6144, 2688, ("n", 896, 3)),
    (512, 640, None),
]


@pytest.mark.parametrize("k,n,config", SITES)
@pytest.mark.parametrize("m,real", [(16, 4), (16, 16), (8, 8)])
def test_matches_f64_of_quantized_operands(k, n, config, m, real):
    x = _x(m, k, real)
    wq, ws = _w(k, n)
    ref = _oracle(x, wq, ws)

    def f(a, b, c):
        return fp8_skinny.fp8_skinny_matmul(a, b, c, config=config)

    y = jax.jit(f)(x, wq, ws)
    assert y.shape == (m, n) and y.dtype == x.dtype
    y32 = np.asarray(y, np.float32)
    assert np.isfinite(y32).all()
    assert np.all(y32[real:] == 0), "pad rows must stay exactly zero"
    assert _row_err(y, ref) <= _tol(x, wq, ws, ref)


@pytest.mark.parametrize("config", [("n", 512, 2), ("k", 256, 3)])
@pytest.mark.parametrize("edge", ["none", "zero_row", "outlier", "tiny"])
def test_quantize_bit_identical_to_quantize_block(config, edge):
    """Through W = I the output is bf16(x_q * x_scale): the in-kernel quantize must
    equal util.quantize_tensor bit for bit."""
    k = 1024
    x = np.asarray(_x(16, k, 16), np.float32)
    if edge == "zero_row":
        x[3] = 0.0
    elif edge == "outlier":
        x[0, 5] = 3.0e4
        x[1] = -np.abs(x[1])
    elif edge == "tiny":
        x[2] *= 1e-30
    x = jnp.asarray(x, jnp.bfloat16)
    eye = jnp.asarray(np.eye(k, dtype=np.float32), F8)
    ones = jnp.ones((k,), jnp.float32)
    y = jax.jit(lambda a: fp8_skinny.fp8_skinny_matmul(a, eye, ones, config=config))(x)
    xq, xs = util.quantize_tensor(x, F8)
    ref = (xq.astype(jnp.float32) * xs.reshape(-1, 1)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(ref, np.float32)
    )


def test_padded_copy_and_logical_width():
    """q_a + kv_a: 2624 outputs; the kernel runs on a copy padded to 2688 at load and
    the cut output matches the unpadded math."""
    k, n = 6144, 2624
    n_pad = fp8_skinny.padded_n(n)
    assert n_pad == 2688
    x = _x(16, k, 4)
    wq, ws = _w(k, n)
    wq_p = jnp.pad(wq.astype(jnp.float32), ((0, 0), (0, n_pad - n))).astype(F8)
    ws_p = jnp.pad(ws, (0, n_pad - n), constant_values=1.0)

    def f(a, b, c):
        return fp8_skinny.fp8_skinny_matmul(a, b, c, n_logical=n)

    y = jax.jit(f)(x, wq_p, ws_p)
    assert y.shape == (16, n)
    ref = _oracle(x, wq, ws)
    assert _row_err(y, ref) <= _tol(x, wq, ws, ref)


@pytest.mark.parametrize("config", [("n", 1024, 3), ("k", 256, 3)])
def test_stacked_weight_index(config):
    """The kernel reads w_stack[idx] itself (the benchmark's cold-weight form): each
    slice gives the same result as passing that weight alone."""
    k, n = 1024, 3072
    x = _x(16, k, 16)
    ws_list = [_w(k, n, seed=s) for s in (3, 4, 5)]
    stack = jnp.stack([w for w, _ in ws_list])
    for i, (wq, ws) in enumerate(ws_list):
        idx = jnp.array([i], jnp.int32)
        y = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws, config=config)
        y1 = fp8_skinny.fp8_skinny_matmul(x, wq, ws, config=config)
        np.testing.assert_array_equal(
            np.asarray(y, np.float32), np.asarray(y1, np.float32)
        )


def test_supports():
    wq, ws = _w(1024, 6144)
    assert fp8_skinny.supports(16, wq, ws)
    assert not fp8_skinny.supports(17, wq, ws)
    w2, s2 = _w(512, 640)  # correct, but untuned: stays on XLA
    assert not fp8_skinny.supports(16, w2, s2)
    assert not fp8_skinny.supports(16, wq[:, :2624], ws[:2624])  # pad N at load first
    assert not fp8_skinny.supports(16, wq, ws.reshape(1, 1, 1, -1))  # blockwise scale
    assert not fp8_skinny.supports(16, wq.astype(jnp.bfloat16), ws)
