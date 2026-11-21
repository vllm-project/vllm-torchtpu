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
"""Tests for the fused attention-output / o_proj FP8 quantization kernel."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.attn_out_fp8 import (
    attn_out_fp8_matmul,
    can_use_attn_out_fp8,
    quantize_attn_out,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

pytestmark = pytest.mark.tpu_test

# Per-rank (num_kv_heads, q_per_kv, head_dim): Gemma 4 31B sliding and
# global attention at TP=2 and TP=8.
_LAYOUTS = [(8, 2, 256), (2, 8, 512), (2, 2, 256), (1, 4, 512)]


@pytest.fixture
def require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")


def _attn_out(key, m, kv, qpk, d):
    """Attention output as the model sees it: head-major, then flattened."""
    o5 = jax.random.normal(key, (kv, m, qpk // 2, 2, d), jnp.float32) * 3.0
    o5 = o5.astype(jnp.bfloat16)
    return o5.reshape(kv, m, qpk, d).swapaxes(0, 1).reshape(m, kv * qpk * d)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("m", [32, 256, 2048])
def test_matches_xla_quantized_matmul_bit_exact(require_tpu, layout, m):
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
def test_quantized_activation_and_scale(require_tpu, layout):
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


def test_can_use_attn_out_fp8():
    assert can_use_attn_out_fp8(256, 8, 2, 256)
    assert not can_use_attn_out_fp8(16, 8, 2, 256)  # below the token floor
    assert not can_use_attn_out_fp8(256, 8, 3, 256)  # odd q_per_kv
    assert not can_use_attn_out_fp8(256, 8, 2, 64)  # head_dim not lane-aligned
    assert not can_use_attn_out_fp8(200, 8, 2, 256)  # no row tile divides m


def test_tag_attn_out_projections():
    import torch

    from vllm_torchtpu.layers.adapter.linear_common import (
        ATTN_OUT_HEAD_LAYOUT_ATTR,
        tag_attn_out_projections,
    )

    class _FakeAttn(torch.nn.Module):
        def __init__(self, num_heads: int, num_kv_heads: int, head_size: int):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_size = head_size

    class _FakeProj(torch.nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.input_size_per_partition = in_features

    class _Block(torch.nn.Module):
        def __init__(self, attn, o_proj):
            super().__init__()
            self.attn = attn
            self.o_proj = o_proj

    good = _Block(_FakeAttn(16, 2, 512), _FakeProj(16 * 512))
    bad_dim = _Block(_FakeAttn(16, 2, 512), _FakeProj(4096))
    model = torch.nn.ModuleList([good, bad_dim])
    assert tag_attn_out_projections(model) == 1
    assert getattr(good.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 8, 512)
    assert not hasattr(bad_dim.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
