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
"""TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS: unscaled e4m3 activations x e4m3 weights."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch_tpu")

from vllm_torchtpu.models.minimax_m3 import quant  # noqa: E402


def test_fp8_cast_matmul_close_to_bf16():
    k = jax.random.PRNGKey(0)
    x = (jax.random.normal(k, (64, 512)) * 3).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.fold_in(k, 1), (512, 256)).astype(
        jnp.float8_e4m3fn
    )
    s = jnp.full((256,), 2.0**-7, jnp.bfloat16)
    y = quant._fp8_cast_matmul(x, w, s)
    ref = (x.astype(jnp.float32) @ w.astype(jnp.float32)) * 2.0**-7
    err = np.sqrt(np.mean((np.asarray(y, np.float32) - ref) ** 2)) / np.sqrt(
        np.mean(np.asarray(ref) ** 2)
    )
    assert y.dtype == jnp.bfloat16 and y.shape == (64, 256)
    assert err < 0.05


def test_fp8_cast_saturates():
    x = jnp.full((8, 32), 1e4, jnp.bfloat16)
    w = jnp.ones((32, 8), jnp.float8_e4m3fn)
    y = quant._fp8_cast_matmul(x, w, jnp.ones((8,), jnp.bfloat16))
    np.testing.assert_allclose(np.asarray(y, np.float32), 448.0 * 32)


@pytest.mark.parametrize(
    "min_rows,rows,prefix,expected",
    [
        (0, 8192, "model.layers.3.self_attn.qkv_proj", False),
        (1024, 512, "model.layers.3.self_attn.qkv_proj", False),
        (1024, 8192, "model.layers.3.self_attn.qkv_proj", True),
        (1024, 8192, "model.layers.3.block_sparse_moe.shared_experts.gate_up_proj", False),
        (1024, 8192, "model.layers.3.block_sparse_moe.shared_experts.down_proj", True),
    ],
)
def test_fp8_cast_applies(monkeypatch, min_rows, rows, prefix, expected):
    monkeypatch.setenv("TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS", str(min_rows))
    monkeypatch.delenv("TPU_MINIMAX_M3_FP8_CAST_SKIP", raising=False)
    layer = types.SimpleNamespace(prefix=prefix)
    assert quant._fp8_cast_applies(layer, rows) is expected
