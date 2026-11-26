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
"""Unit tests for FP8 linear, RMSNorm pre-quantization, and adapter gating."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu import tpu_info
from vllm_torchtpu.layers.adapter import (
    linear_common,
    norm_quant,
    rmsnorm,
    small_all_reduce_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMultiHeadLatentAttentionWrapper,
)
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


def test_v7x_fastpath_gates_and_pcp_small_all_reduce(monkeypatch) -> None:
    monkeypatch.setattr(tpu_info, "is_tpu_v7x", lambda: True)
    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", False)
    monkeypatch.setattr(small_all_reduce_op, "glm_dsa_v7x_active", lambda: False)
    lm_head = object.__new__(ParallelLMHead)
    assert (
        unquantized.decode_lm_head_method(lm_head) is None
        and fp8._default_decode_gemv_max_rows() == 0
        and not small_all_reduce_op.prebuild()
    )
    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", True)
    assert isinstance(
        unquantized.decode_lm_head_method(lm_head), unquantized.VllmDecodeLMHeadMethod
    )
    assert fp8._default_decode_gemv_max_rows() == 64
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(small_all_reduce_op, "ep_device_ids", lambda: tuple(range(8)))
    monkeypatch.setattr(small_all_reduce_op, "get_pipeline_stage", lambda: (0, 1))
    # Non-GLM v7x fastpaths (e.g., Gemma 4) must not prebuild small_all_reduce_op.
    assert not small_all_reduce_op.prebuild()
    monkeypatch.setattr(small_all_reduce_op, "glm_dsa_v7x_active", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    assert not small_all_reduce_op.prebuild()


def test_upstream_rmsnorm_not_globally_overridden_and_arm_prequant_works(
    monkeypatch,
) -> None:
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(cfg):
        assert type(RMSNorm(4096, eps=1e-6, dtype=torch.bfloat16)) is RMSNorm

    class _FakeFp8Linear(LinearBase):
        def __init__(self, *, skinny_n: int | None = None) -> None:
            nn.Module.__init__(self)
            self.quant_method = object.__new__(fp8.VllmFp8LinearMethodTPU)
            self.quant_method.w8a16_max_rows, self.quant_method.decode_gemv_max_rows = (
                0,
                64,
            )
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
            if skinny_n is not None:
                setattr(self, linear_common.FP8_PALLAS_N_ATTR, skinny_n)

    class _FakeMlaWrapper(VllmTPUMultiHeadLatentAttentionWrapper):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.q_lora_rank, self.fused_qkv_a_proj = 1536, _FakeFp8Linear()
            self.kimi_preprocess, self.indexer, self.skip_topk, self.g_proj = (
                None,
                object(),
                False,
                None,
            )

    class _FakeAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mla = _FakeMlaWrapper()

    class _FakeDenseMlp(nn.Module):
        def __init__(self, *, skinny_n: int | None = None) -> None:
            super().__init__()
            self.gate_up_proj = _FakeFp8Linear(skinny_n=skinny_n)

    class _FakeDecoderLayer(nn.Module):
        def __init__(
            self,
            *,
            norm_dtype: torch.dtype = torch.bfloat16,
            skinny_n: int | None = None,
        ) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(4096, eps=1e-6, dtype=norm_dtype)
            self.self_attn = _FakeAttn()
            self.post_attention_layernorm = RMSNorm(4096, eps=1e-6, dtype=norm_dtype)
            self.mlp = _FakeDenseMlp(skinny_n=skinny_n)

    with set_current_vllm_config(cfg):
        model = nn.ModuleList(
            [
                _FakeDecoderLayer(),
                _FakeDecoderLayer(norm_dtype=torch.float32, skinny_n=1024),
            ]
        )
        assert rmsnorm.arm_prequant(model) == 2
    layer = model[0]
    assert isinstance(layer.input_layernorm, rmsnorm.TpuRMSNorm)
    assert isinstance(layer.post_attention_layernorm, rmsnorm.TpuRMSNorm)
    assert isinstance(
        getattr(layer.self_attn.mla.fused_qkv_a_proj, linear_common.PREQUANT_ATTR),
        linear_common.PrequantSlot,
    )
    assert not hasattr(model[1].mlp.gate_up_proj, linear_common.PREQUANT_ATTR)

    monkeypatch.setattr(
        fp8,
        "prequantized_matmul",
        lambda xq, xs, w, ws, decode_gemv_max_rows=0: torch.zeros(
            (xq.shape[0], w.shape[1]), dtype=torch.bfloat16
        ),
    )
    linear = layer.mlp.gate_up_proj
    ml_slot = getattr(linear, linear_common.PREQUANT_ATTR)

    def _run_prequant(x_in: torch.Tensor) -> torch.Tensor:
        ml_slot.value = (
            torch.zeros((x_in.shape[0], 4096), dtype=torch.float8_e4m3fn),
            torch.ones((x_in.shape[0], 1), dtype=torch.float32),
        )
        return linear.quant_method.apply(linear, x_in)

    out_pq = _run_prequant(torch.ones((4, 4096), dtype=torch.bfloat16))
    assert out_pq.shape == (4, 1024) and ml_slot.value is None

    skinny_linear, seen_dg = model[1].mlp.gate_up_proj, []
    monkeypatch.setattr(
        fp8,
        "quantized_matmul_pallas",
        lambda x, w, ws, wk, sk, *, hbm_pin=False, decode_gemv_max_rows=0: (
            seen_dg.append(decode_gemv_max_rows)
            or torch.zeros((x.shape[0], w.shape[1]), dtype=x.dtype)
        ),
    )
    out32 = skinny_linear.quant_method.apply(
        skinny_linear, torch.zeros((32, 4096), dtype=torch.bfloat16)
    )
    assert out32.shape == (32, 1024) and seen_dg == [64]


def test_gelu_quantize_fake_schema_and_skinny_pin_and_lm_head() -> None:
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
    assert fp8.fp8_skinny_hbm_pin_selected("model.layers.3.self_attn.q_b_proj")
    assert fp8.fp8_skinny_hbm_pin_selected("model.layers.3.self_attn.o_proj")
    assert not fp8.fp8_skinny_hbm_pin_selected(
        "model.layers.3.self_attn.kv_a_proj_with_mqa"
    )
    assert not fp8.fp8_skinny_hbm_pin_selected("model.layers.3.mlp.down_proj")
    assert (small_all_reduce_op.MIN_CORES, small_all_reduce_op.max_rows()) == (8, 32)
    fallback = []
    t = torch.zeros((4, 16))
    assert (
        small_all_reduce_op.all_reduce_or(t, (0, 1), lambda x: fallback.append(1) or x)
        is t
        and fallback == [1]
        and hasattr(unquantized, "VllmDecodeLMHeadMethod")
    )
