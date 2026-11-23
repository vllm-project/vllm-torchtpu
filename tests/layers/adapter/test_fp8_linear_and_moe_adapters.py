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
"""Unit tests for FP8 linear pre-quantized input, GeGLU, and MoE adapter gating."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu import tpu_info
from vllm_torchtpu.layers.adapter import linear_common, norm_quant
from vllm_torchtpu.layers.adapter.quantization import fp8, unquantized
from vllm_torchtpu.layers.core import fused_moe_gmm


def test_sortfree_onehot_routing_bit_identical_to_sorted_form() -> None:
    k1, k2, k3 = jax.random.split(jax.random.key(2026), 3)
    x = jax.random.normal(k1, (32, 64), dtype=jnp.bfloat16)
    ids = jax.random.randint(k2, (32, 8), -4, 32, dtype=jnp.int32)
    w = (
        jax.nn.sigmoid(jax.random.normal(k3, (32, 8), jnp.float32))
        .astype(jnp.bfloat16)
        .at[-4:]
        .set(0)
    )
    kw = dict(
        local_num_experts=32,
        topk=8,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=512,
        skip_padded_tokens=True,
    )
    out_sf = fused_moe_gmm.prepare_routed_gmm_inputs(
        x, ids, w, onehot_sortfree=True, **kw
    )
    out_std = fused_moe_gmm.prepare_routed_gmm_inputs(
        x, ids, w, onehot_sortfree=False, **kw
    )
    for a, b in zip(out_sf, out_std, strict=True):
        assert a is not None and b is not None
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_can_use_gather_gmm_rejects_block_scales_and_small_batches(monkeypatch) -> None:
    monkeypatch.setattr(tpu_info, "is_tpu_v7x", lambda: True)
    monkeypatch.setattr(fused_moe_gmm, "_tpu_has_fp8_matmul", lambda: True)
    monkeypatch.setattr(fused_moe_gmm, "get_vmem_capacity_bytes", lambda: 64 << 20)
    e, h, inter = 16, 512, 256
    w1 = jnp.zeros((e, h, 2 * inter), jnp.float8_e4m3fn)
    w2 = jnp.zeros((e, inter, h), jnp.float8_e4m3fn)
    s1_pc, s2_pc = jnp.ones((e, 1, 1, 2 * inter)), jnp.ones((e, 1, 1, h))
    s1_blk = jnp.ones((e, h // 128, 1, 2 * inter))
    x_lg, x_sm = jnp.zeros((1024, h), jnp.bfloat16), jnp.zeros((512, h), jnp.bfloat16)
    kw = dict(topk=8, rhs_quant_dtype=None, masked=True)

    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", False)
    assert not fused_moe_gmm.can_use_gather_gmm(
        x_lg, w1, w2, s1_pc, s2_pc, None, None, **kw
    )
    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", True)
    assert fused_moe_gmm.can_use_gather_gmm(
        x_lg, w1, w2, s1_pc, s2_pc, None, None, **kw
    )
    assert not fused_moe_gmm.can_use_gather_gmm(
        x_lg, w1, w2, s1_blk, s2_pc, None, None, **kw
    )
    assert not fused_moe_gmm.can_use_gather_gmm(
        x_sm, w1, w2, s1_pc, s2_pc, None, None, **kw
    )


def test_v7x_fastpath_gates_for_decode_gemv_and_lm_head(monkeypatch) -> None:
    monkeypatch.setattr(tpu_info, "is_tpu_v7x", lambda: True)
    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", False)
    lm_head = object.__new__(ParallelLMHead)
    assert (
        unquantized.decode_lm_head_method(lm_head) is None
        and fp8._default_decode_gemv_max_rows() == 0
    )
    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", True)
    assert isinstance(
        unquantized.decode_lm_head_method(lm_head), unquantized.VllmDecodeLMHeadMethod
    )
    assert fp8._default_decode_gemv_max_rows() == 64


def test_fp8_linear_quantized_input_and_decode_gemv_dispatch(monkeypatch) -> None:
    class _FakeFp8Linear(LinearBase):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.params_dtype = torch.bfloat16
            self.quant_method = object.__new__(fp8.VllmFp8LinearMethodTPU)
            self.quant_method.w8a16_max_rows = 4
            self.quant_method.decode_gemv_max_rows = 64
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros((4096, 1024), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "weight_scale", nn.Parameter(torch.ones((1024,)), requires_grad=False)
            )
            self.register_parameter("bias", None)

    linear = _FakeFp8Linear()
    assert not linear.quant_method.supports_quantized_input(linear, rows=4)
    assert linear.quant_method.supports_quantized_input(linear, rows=8)

    seen_dg: list[int] = []
    monkeypatch.setattr(
        fp8,
        "quantized_matmul",
        lambda x, w, ws, w8a16_max_rows=0, decode_gemv_max_rows=0: (
            seen_dg.append(decode_gemv_max_rows)
            or torch.zeros((x.shape[0], w.shape[1]), dtype=x.dtype)
        ),
    )
    out = linear.quant_method.apply(
        linear, torch.zeros((16, 4096), dtype=torch.bfloat16)
    )
    assert out.shape == (16, 1024) and seen_dg == [64]

    seen_pre_dg: list[int] = []
    monkeypatch.setattr(
        fp8,
        "prequantized_matmul",
        lambda xq, xs, w, ws, decode_gemv_max_rows=0: (
            seen_pre_dg.append(decode_gemv_max_rows)
            or torch.zeros((xq.shape[0], w.shape[1]), dtype=torch.bfloat16)
        ),
    )
    out_pq = linear.quant_method.apply_quantized_input(
        linear,
        torch.zeros((8, 4096), dtype=torch.float8_e4m3fn),
        torch.ones((8, 1), dtype=torch.float32),
    )
    assert out_pq.shape == (8, 1024) and seen_pre_dg == [64]


def test_gelu_quantize_fake_schema_and_norm_quant_op_cache() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        xq, xs = linear_common.gelu_tanh_and_mul_quantize(
            torch.empty((16, 4096), dtype=torch.bfloat16),
            torch.empty((2048, 4096), dtype=torch.float8_e4m3fn),
        )
        assert xq.shape == (16, 2048) and xq.dtype == torch.float8_e4m3fn
        assert xs.shape == (16, 1) and xs.dtype == torch.float32
    op1 = norm_quant.add_rms_norm_quantize_op(1e-6, torch.float8_e4m3fn)
    assert op1 is norm_quant.add_rms_norm_quantize_op(1e-6, torch.float8_e4m3fn)
    assert hasattr(unquantized, "VllmDecodeLMHeadMethod")
