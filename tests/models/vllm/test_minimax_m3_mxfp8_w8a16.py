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
"""MiniMax M3 MXFP8 linears kept as e4m3 (TPU_MINIMAX_M3_MXFP8_W8A16)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.linear_common import (
    WEIGHT_FLIPPED_ATTR,
    _quantized_matmul_jax,
)
from vllm_torchtpu.models.minimax_m3.quant import (
    MiniMaxM3Mxfp8LinearMethod,
    MiniMaxM3Mxfp8W8A16LinearMethod,
    MiniMaxM3QuantConfig,
    dequantize_mxfp8,
    mxfp8_to_w8a16,
)

_CONFIG = {
    "quant_method": "mxfp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 32],
    "group_size": 16,
    "moe_quant_algo": "NVFP4",
    "moe_quant_format": "nvfp4-pack-quantized",
    "ignored_layers": ["language_model.model.layers.3.block_sparse_moe.gate"],
}


def _mxfp8(n: int, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP-style MXFP8 of a gaussian [n, k] weight (block 32 along k)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k // 32, 32, generator=g) * 0.02
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    e = torch.floor(torch.log2(amax)) - 8
    q = (w / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(n, k), (e[..., 0] + 127).to(torch.uint8)


@pytest.mark.cpu_test
@pytest.mark.parametrize("n,k", [(1280, 6144), (6144, 384), (64, 96)])
def test_w8a16_layout_dequantizes_to_the_bf16_weight(n, k):
    w8, s = _mxfp8(n, k)
    w_kn, scale = mxfp8_to_w8a16(w8, s)
    assert w_kn.dtype == torch.float8_e4m3fn and w_kn.shape == (k, n)
    assert scale.dtype == torch.float32 and scale.shape == (1, k // 32, 1, n)
    # What gmm_v2 does in VMEM: e4m3 -> bf16, times the bf16 block scale.
    rep = scale[0, :, 0, :].to(torch.bfloat16).repeat_interleave(32, dim=0)
    deq = w_kn.to(torch.bfloat16) * rep
    assert torch.equal(deq.view(torch.int16), dequantize_mxfp8(w8, s).t().view(torch.int16))


@pytest.mark.cpu_test
@pytest.mark.parametrize("env,draft,expect", [
    ("0", False, MiniMaxM3Mxfp8LinearMethod),
    ("1", False, MiniMaxM3Mxfp8W8A16LinearMethod),
    ("1", True, MiniMaxM3Mxfp8LinearMethod),
])
def test_flag_selects_w8a16_for_the_target_only(monkeypatch, env, draft, expect):
    monkeypatch.setenv("TPU_MINIMAX_M3_MXFP8_W8A16", env)
    config = dict(_CONFIG)
    if draft:
        config["ignored_layers"] = ["language_model.model.dspark.fc", *config["ignored_layers"]]
    qc = MiniMaxM3QuantConfig.from_config(config)
    layer = torch.nn.Module.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = qc._get_checkpoint_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
    assert type(method) is expect


@pytest.mark.cpu_test
def test_process_weights_is_idempotent():
    w8, s = _mxfp8(256, 512)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w8.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s.clone(), requires_grad=False)
    method = MiniMaxM3Mxfp8W8A16LinearMethod()
    method.process_weights_after_loading(layer)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR)
    first = layer.weight.data.clone()
    method.process_weights_after_loading(layer)  # reload paths re-enter
    assert torch.equal(layer.weight.data.view(torch.uint8), first.view(torch.uint8))
    assert layer.weight.shape == (512, 256)


def _on_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except RuntimeError:
        return False


@pytest.mark.skipif(not _on_tpu(), reason="gmm_v2 is a TPU kernel")
@pytest.mark.parametrize("m", [8, 48, 512])
@pytest.mark.parametrize("n,k", [(1280, 6144), (6144, 1024), (768, 6144), (6144, 384)])
def test_w8a16_matmul_matches_bf16_bitwise(m, n, k):
    w8, s = _mxfp8(n, k, seed=m)
    w_kn, scale = mxfp8_to_w8a16(w8, s)
    w_bf16 = dequantize_mxfp8(w8, s).t().contiguous()
    rng = np.random.default_rng(m)
    x = jnp.asarray(rng.standard_normal((m, k)), jnp.bfloat16)
    to_jax = lambda t: jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    out = _quantized_matmul_jax(x, to_jax(w_kn), jnp.asarray(scale.numpy()))
    ref = jnp.matmul(x, jnp.asarray(w_bf16.float().numpy(), jnp.bfloat16))
    np.testing.assert_array_equal(
        np.asarray(out.astype(jnp.float32)), np.asarray(ref.astype(jnp.float32))
    )
