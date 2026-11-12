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
"""TPU_FUSED_ADD_RMSNORM_QUANT: fused add + RMSNorm + per-row fp8 quant vs
today's two-pass path, bit for bit.

The incumbent is what production runs today: the Pallas
``fused_add_rms_norm`` followed by XLA ``quantize_tensor`` (an amax pass and a
convert pass) inside ``xla_quantized_matmul``. The fused kernel must give the
same residual, the same bf16 ``y`` (when it writes one), the same fp8 bytes and
the same f32 scale bits, and the prequantized matmul the same bf16 output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_add_rmsnorm as K
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
        pytest.skip("compiled Mosaic kernel needs a TPU")


def _inputs(rows, hidden=HIDDEN, seed=0, res_scale=4.0, zero_rows=()):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = (jax.random.normal(k1, (rows, hidden), jnp.float32) * 0.7).astype(jnp.bfloat16)
    r = (jax.random.normal(k2, (rows, hidden), jnp.float32) * res_scale).astype(
        jnp.bfloat16
    )
    if zero_rows:
        idx = jnp.asarray(zero_rows)
        x = x.at[idx].set(-r[idx])  # s = x + r = 0 exactly: y = 0, amax = 0
    w = (1 + 0.2 * jax.random.normal(k3, (hidden,), jnp.float32)).astype(jnp.bfloat16)
    return x, r, w


@jax.jit
def _quant(y):
    return quantize_tensor(y, FP8)


def _incumbent(x, r, w):
    y, res = K.fused_add_rms_norm(x, r, w, eps=EPS)
    q, s = _quant(y)
    return y, q, s, res


def _bits(a):
    a = np.asarray(a)
    return a.view({1: np.uint8, 2: np.int16, 4: np.int32}[a.dtype.itemsize])


def _mismatch(got, ref):
    """Element mismatch counts per output; y compared only when emitted."""
    y, q, s, res = got
    y0, q0, s0, r0 = ref
    out = {
        "res": int((_bits(res) != _bits(r0)).sum()),
        "q": int((_bits(q) != _bits(q0)).sum()),
        "scale": int((_bits(s) != _bits(s0)).sum()),
    }
    if y is not None:
        out["y"] = int((_bits(y) != _bits(y0)).sum())
    return out


# Cell classes: the served mixed-step token counts, plus edge cells.
CELLS = {
    "T512": dict(rows=512),
    "T4096": dict(rows=4096),
    "zero_rows": dict(rows=1024, zero_rows=(0, 3, 257, 1023)),
    "large_res": dict(rows=1024, res_scale=1500.0),
}


@pytest.mark.parametrize("emit_y", [True, False])
@pytest.mark.parametrize("cell", list(CELLS))
def test_tpu_bitexact_vs_two_pass(cell, emit_y):
    _require_tpu()
    kw = dict(CELLS[cell])
    rows = kw.pop("rows")
    x, r, w = _inputs(rows, seed=rows + 7 * int(emit_y), **kw)
    got = jax.block_until_ready(
        K.fused_add_rms_norm_quant(x, r, w, eps=EPS, emit_y=emit_y)
    )
    assert got[1].shape == (rows, HIDDEN) and got[1].dtype == FP8
    assert got[2].shape == (rows, 1) and got[2].dtype == jnp.float32
    mm = _mismatch(got, _incumbent(x, r, w))
    assert all(v == 0 for v in mm.values()), mm


@pytest.mark.parametrize("n_out", [2624, 4096])
@pytest.mark.parametrize("rows", [512, 4096])
def test_tpu_prequantized_matmul_bitexact(rows, n_out):
    """The consumer: q_a (fused_qkv_a, n 2624) and a dense gate_up width."""
    _require_tpu()
    x, r, w = _inputs(rows, seed=11 + rows)
    kq, ks = jax.random.split(jax.random.PRNGKey(n_out))
    w_q = (jax.random.normal(kq, (HIDDEN, n_out), jnp.float32) * 64).astype(FP8)
    w_s = jax.random.uniform(ks, (n_out,), jnp.float32, 1e-4, 1e-3)
    y, _ = K.fused_add_rms_norm(x, r, w, eps=EPS)
    ref = jax.jit(xla_quantized_matmul)(y, w_q, w_s)
    _, q, s, _ = K.fused_add_rms_norm_quant(x, r, w, eps=EPS, emit_y=False)
    got = jax.jit(xla_prequantized_matmul)(q, s, w_q, w_s)
    n_bad = int((_bits(got) != _bits(ref)).sum())
    assert got.dtype == ref.dtype == jnp.bfloat16 and n_bad == 0
