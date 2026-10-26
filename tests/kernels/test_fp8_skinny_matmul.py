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
"""fp8_skinny: the decode weight-streaming FP8 matmul (TPU_FP8_LINEAR_PALLAS).

Judged against the float64 product of the SAME quantized operands (x_q, x_scale from
`util.quantize_tensor`, the per-channel weight and scale). XLA's fp8 x fp8 dot
accumulates below f32 and this kernel does not, so the kernel must land inside 2x
the XLA reference's own error on the cell.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny, util

F8 = jnp.float8_e4m3fn
SHAPES = sorted(fp8_skinny.TUNED)


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
    a = np.asarray(xq).astype(np.float64) * np.asarray(xs, np.float64).reshape(-1, 1)
    b = np.asarray(wq).astype(np.float64) * np.asarray(ws, np.float64)[None]
    return a @ b


def _row_err(y, ref):
    y = np.asarray(y).astype(np.float64)
    m = np.abs(ref).max(axis=1)
    d = np.abs(y - ref).max(axis=1)
    return float(np.where(m > 0, d / np.where(m > 0, m, 1), d).max())


def _run(x, wq, ws, **kw):
    return jax.jit(lambda a, b, c: fp8_skinny.fp8_skinny_matmul(a, b, c, **kw))(
        x, wq, ws
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("hbm_pin", [False, True])
@pytest.mark.parametrize("m,real", [(8, 7), (16, 14), (32, 32), (64, 56), (128, 112)])
def test_matches_f64_of_quantized_operands(k, n, hbm_pin, m, real):
    x = _x(m, k, real)
    wq, ws = _w(k, n)
    ref = _oracle(x, wq, ws)
    y = _run(x, wq, ws, hbm_pin=hbm_pin)
    assert y.shape == (m, n) and y.dtype == x.dtype
    y32 = np.asarray(y, np.float32)
    assert np.isfinite(y32).all()
    assert np.all(y32[real:] == 0), "pad rows must stay exactly zero"
    y_xla = jax.jit(util.xla_quantized_matmul)(x, wq, ws)
    assert _row_err(y, ref) <= max(2 * _row_err(y_xla, ref), 2.0**-8)


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n", SHAPES)
def test_placement_and_dma_split_are_bit_identical(k, n):
    """hbm_pin and the ring shape move bytes, not arithmetic (within one split)."""
    x = _x(16, k, 16)
    wq, ws = _w(k, n)
    split, t, _, _ = fp8_skinny.TUNED[(k, n)]
    ref = np.asarray(_run(x, wq, ws, config=(split, t, 2, 1)), np.float32)
    for kw in (
        dict(config=(split, t, 3, 4)),
        dict(config=(split, t, 2, 1), hbm_pin=True),
        dict(config=(split, t, 2, 1), hbm_pin=True, scale_dma=True),
    ):
        np.testing.assert_array_equal(
            np.asarray(_run(x, wq, ws, **kw), np.float32), ref
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("config", [("n", 512, 2, 1), ("k", 256, 3, 2)])
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
    y = _run(x, eye, jnp.ones((k,), jnp.float32), config=config)
    xq, xs = util.quantize_tensor(x, F8)
    want = (xq.astype(jnp.float32) * xs).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(want, np.float32)
    )
