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
"""TPU tests for the fused attention-output / o_proj FP8 quantization kernel."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.quantized_matmul.attn_out_fp8 import (
    attn_out_fp8_matmul,
    quantize_attn_out,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

pytestmark = pytest.mark.tpu_test

# Per-rank (num_kv_heads, q_per_kv, head_dim): Gemma 4 31B sliding and
# global attention at TP=2 and TP=8, plus `(1, 2, 256)` covering the
# single-pair (`len(q_blocks) == 1`) kernel branch.
_LAYOUTS = [(8, 2, 256), (2, 8, 512), (2, 2, 256), (1, 4, 512), (1, 2, 256)]


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _attn_out(key, m, kv, qpk, d):
    """Attention output as the model sees it: head-major, then flattened."""
    o5 = jax.random.normal(key, (kv, m, qpk // 2, 2, d), jnp.float32) * 3.0
    o5 = o5.astype(jnp.bfloat16)
    return o5.reshape(kv, m, qpk, d).swapaxes(0, 1).reshape(m, kv * qpk * d)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("m", [32, 256, 2048])
def test_matches_xla_quantized_matmul_bit_exact(layout, m):
    _require_tpu()
    kv, qpk, d = layout
    k, n = kv * qpk * d, 5376
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(m + k), 3)
    x = _attn_out(k1, m, kv, qpk, d)
    w = (jax.random.normal(k2, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
    ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5) * 0.01

    ref = jax.jit(xla_quantized_matmul)(x, w, ws)
    got = jax.jit(lambda x, w, ws: attn_out_fp8_matmul(x, w, ws, kv, qpk, d))(x, w, ws)
    assert got.dtype == ref.dtype and got.shape == (m, n)
    assert bool(jnp.all(got == ref))


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_quantized_activation_and_scale(layout):
    _require_tpu()
    kv, qpk, d = layout
    m = 256
    x = _attn_out(jax.random.PRNGKey(0), m, kv, qpk, d)
    # An all-zero row gets scale 1, as in the reference, not NaN.
    x = x.at[3].set(0)
    x_q, scale = jax.jit(lambda x: quantize_attn_out(x, kv, qpk, d))(x)
    ref_q, ref_s = quantize_tensor(x, jnp.float8_e4m3fn)
    assert scale.shape == (m, 128)
    assert bool(jnp.all(scale == scale[:, :1]))
    assert bool(jnp.all(scale[:, :1] == ref_s.reshape(m, 1).astype(jnp.float32)))
    assert bool(jnp.all(x_q.astype(jnp.float32) == ref_q.astype(jnp.float32)))
