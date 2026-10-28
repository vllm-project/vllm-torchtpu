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
"""Correctness tests for one-pass Pallas fused_add_rms_norm(_quant)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)

EPS = 1e-5
HIDDEN = 6144
FP8 = jnp.float8_e4m3fn


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("compiled Pallas kernel requires TPU")


def _inputs(rows: int, hidden: int = HIDDEN, seed: int = 0, zero_rows: tuple = ()):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = (jax.random.normal(k1, (rows, hidden), jnp.float32) * 0.7).astype(jnp.bfloat16)
    r = (jax.random.normal(k2, (rows, hidden), jnp.float32) * 4.0).astype(jnp.bfloat16)
    if zero_rows:
        idx = jnp.asarray(zero_rows)
        x = x.at[idx].set(-r[idx])
    w = (1 + 0.2 * jax.random.normal(k3, (hidden,), jnp.float32)).astype(jnp.bfloat16)
    return x, r, w


def _bits(a):
    arr = np.asarray(a)
    return arr.view({1: np.uint8, 2: np.int16, 4: np.int32}[arr.dtype.itemsize])


@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("rows", [256, 1024])
def test_fused_add_rms_norm_matches_reference(rows: int, has_residual: bool):
    _require_tpu()
    x, r, w = _inputs(rows, seed=rows + int(has_residual))
    res_in = r if has_residual else None
    got = kernel.fused_add_rms_norm(x, res_in, w, eps=EPS)
    ref = kernel.reference(x, res_in, w, EPS)
    if has_residual:
        assert isinstance(got, tuple) and isinstance(ref, tuple)
        assert int((_bits(got[1]) != _bits(ref[1])).sum()) == 0
        np.testing.assert_allclose(
            np.asarray(got[0], dtype=np.float32),
            np.asarray(ref[0], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32),
            np.asarray(ref, dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize("emit_y", [True, False])
@pytest.mark.parametrize("zero_rows", [(), (0, 3, 257, 1023)])
def test_fused_add_rms_norm_quant_bitexact(emit_y: bool, zero_rows: tuple):
    _require_tpu()
    rows = 1024
    x, r, w = _inputs(rows, seed=31 + int(emit_y), zero_rows=zero_rows)
    y_inc, res_inc = kernel.fused_add_rms_norm(x, r, w, eps=EPS)
    q_inc, s_inc = jax.jit(lambda t: quantize_tensor(t, FP8))(y_inc)
    y, q, s, res = jax.block_until_ready(
        kernel.fused_add_rms_norm_quant(x, r, w, eps=EPS, emit_y=emit_y)
    )
    assert int((_bits(res) != _bits(res_inc)).sum()) == 0
    assert int((_bits(q) != _bits(q_inc)).sum()) == 0
    assert int((_bits(s) != _bits(s_inc)).sum()) == 0
    if emit_y:
        assert y is not None
        assert int((_bits(y) != _bits(y_inc)).sum()) == 0
    else:
        assert y is None


@pytest.mark.parametrize("n_out", [2624, 4096])
def test_prequantized_matmul_bitexact(n_out: int):
    _require_tpu()
    rows = 1024
    x, r, w = _inputs(rows, seed=99 + n_out)
    kq, ks = jax.random.split(jax.random.PRNGKey(n_out))
    w_q = (jax.random.normal(kq, (HIDDEN, n_out), jnp.float32) * 64).astype(FP8)
    w_s = jax.random.uniform(ks, (n_out,), jnp.float32, 1e-4, 1e-3)
    y, _ = kernel.fused_add_rms_norm(x, r, w, eps=EPS)
    ref = jax.jit(xla_quantized_matmul)(y, w_q, w_s)
    _, q, s, _ = kernel.fused_add_rms_norm_quant(x, r, w, eps=EPS, emit_y=False)
    got = jax.jit(xla_prequantized_matmul)(q, s, w_q, w_s)
    assert got.dtype == ref.dtype == jnp.bfloat16
    assert int((_bits(got) != _bits(ref)).sum()) == 0
