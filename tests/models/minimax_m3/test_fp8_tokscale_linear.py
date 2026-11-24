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
"""TPU_MINIMAX_M3_FP8_TOKSCALE_MIN_ROWS: per-token scaled e4m3 activations."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

pytest.importorskip("torch_tpu")

from vllm_torchtpu.models.minimax_m3 import quant  # noqa: E402


def _rel(y, ref):
    y = np.asarray(y, np.float32)
    return np.linalg.norm(y - ref) / np.linalg.norm(ref)


@pytest.mark.parametrize("rms", [1.0, 0.002])
def test_tok_matmul_close_at_any_magnitude(rms):
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, (64, 512)) * rms
    w = jax.random.normal(jax.random.fold_in(k, 1), (512, 256)).astype(
        jnp.float8_e4m3fn
    )
    s = jnp.full((256,), 2.0**-7, jnp.float32)
    m = jnp.max(jnp.abs(x), -1, keepdims=True)
    x8 = (x * (448.0 / m)).astype(jnp.float8_e4m3fn)
    y = quant._fp8_tok_matmul(x8, m / 448.0, w, s)
    ref = np.asarray((x @ w.astype(jnp.float32)) * 2.0**-7)
    assert y.dtype == jnp.bfloat16 and y.shape == (64, 256)
    assert _rel(y, ref) < 0.05
    # The unscaled cast loses small activations to e4m3 subnormals.
    if rms < 0.1:
        assert _rel(quant._fp8_cast_matmul(x.astype(jnp.bfloat16), w, s), ref) > 0.1


def test_tok_quant_e4m3_roundtrip():
    x = torch.randn(16, 384) * torch.logspace(-3, 2, 16)[:, None]
    x8, tok = quant.tok_quant_e4m3(x.to(torch.bfloat16))
    assert x8.dtype == torch.float8_e4m3fn and tok.shape == (16, 1)
    assert x8.float().abs().amax(-1).max() <= 448.0
    back = x8.float() * tok
    rel = (back - x).norm(dim=-1) / x.norm(dim=-1)
    assert rel.max() < 0.05


def test_fused_add_rms_norm_e4m3_tok_matches_bf16_norm():
    torch.manual_seed(0)
    x = torch.randn(32, 256).to(torch.bfloat16)
    res = (torch.randn(32, 256) * 5).to(torch.bfloat16)
    w = torch.randn(256) * 0.3
    (x8, tok), r = quant.fused_add_rms_norm_e4m3_tok(x, res, w, 1e-6)
    r_ref = (x.float() + res.float()).to(torch.bfloat16)
    torch.testing.assert_close(r, r_ref, rtol=0, atol=0)
    rf = r_ref.float()
    y = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * (w + 1.0)
    assert x8.float().abs().amax(-1).max() <= 448.0
    rel = ((x8.float() * tok) - y).norm() / y.norm()
    assert rel < 0.05


@pytest.mark.parametrize(
    "min_rows,rows,prefix,expected",
    [
        (0, 8192, "model.layers.3.self_attn.qkv_proj", False),
        (4096, 3072, "model.layers.3.self_attn.qkv_proj", False),
        (4096, 4096, "model.layers.3.self_attn.qkv_proj", True),
        (4096, 8192, "model.layers.3.block_sparse_moe.shared_experts.gate_up_proj", False),
        (4096, 8192, "model.layers.3.block_sparse_moe.shared_experts.down_proj", True),
    ],
)
def test_tokscale_applies(monkeypatch, min_rows, rows, prefix, expected):
    monkeypatch.setenv("TPU_MINIMAX_M3_FP8_TOKSCALE_MIN_ROWS", str(min_rows))
    monkeypatch.delenv("TPU_MINIMAX_M3_FP8_CAST_SKIP", raising=False)
    layer = types.SimpleNamespace(prefix=prefix)
    assert quant._tokscale_applies(layer, rows) is expected
