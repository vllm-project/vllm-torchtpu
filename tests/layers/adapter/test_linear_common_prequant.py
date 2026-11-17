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
"""The pre-quantized FP8 matmul: rows quantized per row once, then a matmul
that takes them, give the plain quantized matmul's output bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter import linear_common

pytestmark = pytest.mark.cpu_test

M, K, N = 16, 256, 128


def _weight(seed):
    w = jax.random.normal(jax.random.key(seed), (K, N), jnp.float32)
    scale = jnp.abs(w).max(axis=0) / float(jnp.finfo(jnp.float8_e4m3fn).max)
    return (w / scale).astype(jnp.float8_e4m3fn), scale


@pytest.mark.parametrize("seed", [0, 1])
def test_prequant_path_matches_the_quantized_matmul_bit_for_bit(seed):
    x = jax.random.normal(jax.random.key(10 + seed), (M, K), jnp.bfloat16)
    # An all-zero row: quantize_block maps its scale to 1.
    x = x.at[3].set(0)
    w_q, w_s = _weight(seed)
    x_q, x_s = linear_common._quantize_rows_jax(x)
    assert x_q.dtype == jnp.float8_e4m3fn and x_q.shape == (M, K)
    assert x_s.dtype == jnp.float32 and x_s.shape == (M, 1)
    got = linear_common._prequant_matmul_jax(x_q, x_s, w_q, w_s)
    want = xla_quantized_matmul(x, w_q, w_s)
    assert got.dtype == want.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


def test_ops_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    linear_common.prebuild_prequant_ops()
    with FakeTensorMode():
        x = torch.empty(2, M, K, dtype=torch.bfloat16)
        x_q, x_s = linear_common.quantize_rows(x)
        assert (x_q.shape, x_q.dtype) == ((2 * M, K), torch.float8_e4m3fn)
        assert (x_s.shape, x_s.dtype) == ((2 * M, 1), torch.float32)
        w_q = torch.empty(K, N, dtype=torch.float8_e4m3fn)
        w_s = torch.empty(N, dtype=torch.float32)
        out = linear_common.quantized_matmul_prequant(x_q, x_s, w_q, w_s)
        assert (out.shape, out.dtype) == ((2 * M, N), torch.bfloat16)


@pytest.mark.parametrize(
    "w_s_shape,x_k,match",
    [((1, N), K, "per-channel"), ((N,), K // 2, "must match weight hidden dim")],
)
def test_prequant_matmul_refuses_mismatched_operands(w_s_shape, x_k, match):
    x_q = torch.empty(M, x_k, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(M, 1, dtype=torch.float32)
    w_q = torch.empty(K, N, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=match):
        linear_common.quantized_matmul_prequant(
            x_q, x_s, w_q, torch.empty(w_s_shape, dtype=torch.float32)
        )
