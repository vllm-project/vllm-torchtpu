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
"""The projections that take TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER's rows."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import linear_common


def test_prequantized_rows_give_the_quantizing_matmul_exactly():
    """The fp8 gather changes no fp8 projection's result: the rows and scales
    each rank quantizes are the ones the projection would compute."""
    g = torch.Generator().manual_seed(3)
    x = jnp.asarray(torch.randn(16, 256, generator=g).numpy(), jnp.bfloat16)
    w_q = jnp.asarray(
        (0.05 * torch.randn(256, 96, generator=g)).numpy(), jnp.float8_e4m3fn
    )
    w_s = jnp.asarray((0.02 + 0.01 * torch.rand(96, generator=g)).numpy())
    x_q, x_s = linear_common._quantize_rows_jax(x)
    assert x_q.dtype == jnp.float8_e4m3fn and x_s.shape == (16, 1)
    np.testing.assert_array_equal(
        np.asarray(linear_common._prequant_matmul_jax(x_q, x_s, w_q, w_s), np.float32),
        np.asarray(linear_common._quantized_matmul_jax(x, w_q, w_s), np.float32),
    )


def _fp8_linear(n_in, n_out, keep_vllm_layout=False, bias=None):
    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8Config,
        VllmFp8LinearMethodTPU,
    )

    shape = (n_out, n_in) if keep_vllm_layout else (n_in, n_out)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(n_out), requires_grad=False)
    layer.bias, layer.skip_bias_add = bias, False
    layer.quant_method = VllmFp8LinearMethodTPU(
        VllmFp8Config.from_config({"activation_scheme": "dynamic"})
    )
    if keep_vllm_layout:
        setattr(layer, linear_common.KEEP_VLLM_LAYOUT_ATTR, True)
    return layer


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
def test_a_per_channel_fp8_projection_takes_the_gathered_rows(
    monkeypatch, keep_vllm_layout
):
    calls = []
    monkeypatch.setattr(
        linear_common,
        "quantized_matmul_prequant",
        lambda x_q, x_s, w, s: calls.append((x_q, x_s, w, s)) or torch.ones(4, 6),
    )
    layer = _fp8_linear(8, 6, keep_vllm_layout, bias=torch.full((6,), 2.0))
    x_q, x_s = torch.zeros(4, 8, dtype=torch.float8_e4m3fn), torch.ones(4, 1)
    out = linear_common.project_rows(layer, x_q, (x_q, x_s))
    torch.testing.assert_close(out, torch.full((4, 6), 3.0))
    ((got_q, got_s, w, s),) = calls
    assert got_q is x_q and got_s is x_s and s is layer.weight_scale
    assert w.shape == (8, 6)
    assert w.data_ptr() == layer.weight.data_ptr()


def test_another_projection_is_given_the_rows_dequantized():
    """The checkpoint leaves the GDN's b/a projection in bf16."""
    layer = Mock(return_value=(torch.ones(4, 2), None), spec=["quant_method"])
    x_q = torch.full((4, 8), 2.0).to(torch.float8_e4m3fn)
    x_s = torch.full((4, 1), 0.25)
    out = linear_common.project_rows(layer, x_q, (x_q, x_s))
    torch.testing.assert_close(out, torch.ones(4, 2))
    (rows,), _ = layer.call_args
    assert rows.dtype == torch.bfloat16
    torch.testing.assert_close(rows, torch.full((4, 8), 0.5, dtype=torch.bfloat16))
    # Without gathered rows the projection runs as before.
    x = torch.randn(4, 8)
    linear_common.project_rows(layer, x, None)
    assert layer.call_args.args == (x,)


def test_the_gdn_in_projections_take_the_gathered_rows(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op

    seen = []

    def project_rows(layer, x, prequant):
        seen.append((layer, x, prequant))
        return torch.zeros(x.shape[0], layer.out)

    monkeypatch.setattr(gdn_attention_op, "project_rows", project_rows)
    attn = gdn_attention_op.VllmGatedDeltaNetAttention.__new__(
        gdn_attention_op.VllmGatedDeltaNetAttention
    )
    torch.nn.Module.__init__(attn)
    attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_ba = (
        SimpleNamespace(out=n) for n in (96, 64, 32)
    )
    attn.head_v_dim, attn.num_v_heads, attn.tp_size = 16, 4, 1
    attn.kv_cache = None
    attn._has_active_pooled_pcp_state = lambda kv_cache: False
    attn._apply_output_projection = lambda core, z, n: core
    x = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
    prequant = (x, torch.ones(4, 1))
    attn.forward(x, prequant=prequant)
    assert {id(layer) for layer, _, _ in seen} == {
        id(attn.in_proj_qkv),
        id(attn.in_proj_z),
        id(attn.in_proj_ba),
    }
    assert all(got_x is x and got is prequant for _, got_x, got in seen)
