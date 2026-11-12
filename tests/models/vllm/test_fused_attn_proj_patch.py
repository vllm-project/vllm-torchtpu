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
"""TPU_FUSED_ATTN_PROJ wiring: admission gates, fallbacks and the Gemma 4 swap.

The kernels' numerics are covered by tests/kernels/fused_attn_proj; these CPU
tests cover the code around them with stand-in modules.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import gemma4, qwen3_moe, qwen3_next

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_attn_proj as fap
from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackendImpl
from vllm_torchtpu.models.vllm import fused_attn_proj_patch as patch_mod

UPSTREAM_QWEN3_MOE = qwen3_moe.Qwen3MoeDecoderLayer.forward
UPSTREAM_QWEN3_NEXT = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_GEMMA4_ATTN = gemma4.Gemma4Attention.forward


# Patch application.


def _restore_forwards(monkeypatch):
    monkeypatch.setattr(qwen3_moe.Qwen3MoeDecoderLayer, "forward", UPSTREAM_QWEN3_MOE)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "forward", UPSTREAM_QWEN3_NEXT
    )
    monkeypatch.setattr(gemma4.Gemma4Attention, "forward", UPSTREAM_GEMMA4_ATTN)


def test_patch_skipped_with_flag_off(monkeypatch):
    _restore_forwards(monkeypatch)
    monkeypatch.setattr(envs, "TPU_FUSED_ATTN_PROJ", False)
    patch_mod.maybe_patch_fused_attn_proj()
    assert qwen3_moe.Qwen3MoeDecoderLayer.forward is UPSTREAM_QWEN3_MOE
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_QWEN3_NEXT
    assert gemma4.Gemma4Attention.forward is UPSTREAM_GEMMA4_ATTN


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_patch_applied(monkeypatch, sequence_parallel):
    _restore_forwards(monkeypatch)
    monkeypatch.setattr(envs, "TPU_FUSED_ATTN_PROJ", True)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", sequence_parallel)
    patch_mod.maybe_patch_fused_attn_proj()
    assert (
        qwen3_moe.Qwen3MoeDecoderLayer.forward is patch_mod._patched_qwen3_moe_forward
    )
    assert gemma4.Gemma4Attention.forward is patch_mod._patched_gemma4_attention_forward
    # Under sequence parallelism qwen3_next_patch owns this forward.
    want = (
        UPSTREAM_QWEN3_NEXT
        if sequence_parallel
        else patch_mod._patched_qwen3_next_forward
    )
    assert qwen3_next.Qwen3NextDecoderLayer.forward is want


# Patched forwards: fused path and fallback.


def _layer(fused_result=None, layer_type=None):
    attn = SimpleNamespace()
    if fused_result is not None:
        setattr(attn, fap._ATTR, lambda hidden, residual, positions: fused_result)
    return SimpleNamespace(
        self_attn=attn,
        linear_attn=attn,
        layer_type=layer_type,
        post_attention_layernorm=lambda out, res: (out + 1, res + 2),
        mlp=lambda h: h * 10,
    )


def test_qwen3_moe_forward_fused(monkeypatch):
    out, res = torch.ones(2), torch.zeros(2)
    layer = _layer(fused_result=(out, res))
    hidden, new_res = patch_mod._patched_qwen3_moe_forward(
        layer, torch.zeros(2), torch.zeros(2), torch.zeros(2)
    )
    torch.testing.assert_close(hidden, (out + 1) * 10)
    torch.testing.assert_close(new_res, res + 2)


def test_qwen3_moe_forward_falls_back(monkeypatch):
    calls = []
    monkeypatch.setattr(
        patch_mod, "_qwen3_moe_forward", lambda *a: calls.append(a) or "upstream"
    )
    layer = _layer()  # Not armed.
    assert patch_mod._patched_qwen3_moe_forward(layer, 1, 2, 3) == "upstream"
    assert calls == [(layer, 1, 2, 3)]


def test_fused_op_returning_none_falls_back(monkeypatch):
    """The op declines a step (first layer, profiling, PCP): upstream runs."""
    monkeypatch.setattr(patch_mod, "_qwen3_next_forward", lambda *a, **k: "upstream")
    layer = _layer(layer_type="full_attention")
    setattr(layer.self_attn, fap._ATTR, lambda hidden, residual, positions: None)
    assert patch_mod._patched_qwen3_next_forward(layer, 1, 2, 3) == "upstream"


# Admission gates.


def _rpa_impl(**overrides):
    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    attrs = dict(
        dcp_world_size=1,
        kv_sharing_target_layer_name=None,
        sinks=None,
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
    )
    attrs.update(overrides)
    for k, v in attrs.items():
        setattr(impl, k, v)
    return impl


def _attn(impl=None, head_dim=128, bias=None, tp_size=1, neox=True, mrope=None):
    rotary = SimpleNamespace(is_neox_style=neox)
    if mrope is not None:
        rotary.mrope_section, rotary.mrope_interleaved = [24, 20, 20], mrope
    return SimpleNamespace(
        attn=SimpleNamespace(impl=impl if impl is not None else _rpa_impl()),
        head_dim=head_dim,
        q_size=8 * head_dim,
        kv_size=2 * head_dim,
        qkv_proj=None,  # Read by the stubbed _linear only.
        o_proj=SimpleNamespace(bias=bias, tp_size=tp_size),
        rotary_emb=rotary,
    )


class _Reached(Exception):
    """Raised by the stubbed _linear: every gate before it passed."""


@pytest.mark.parametrize(
    "attn, reason",
    [
        (_attn(impl=SimpleNamespace()), "not batched RPA"),
        (_attn(head_dim=64), "head_dim 64"),
        (_attn(impl=_rpa_impl(dcp_world_size=2)), "DCP"),
        (_attn(impl=_rpa_impl(kv_sharing_target_layer_name="l0")), "KV sharing"),
        (_attn(impl=_rpa_impl(sinks=torch.zeros(4))), "sinks"),
        (_attn(impl=_rpa_impl(kv_cache_quantized_dtype=None)), "KV cache dtype"),
        (
            _attn(impl=_rpa_impl(kv_cache_quantized_dtype=torch.float8_e5m2)),
            "KV cache dtype",
        ),
        (_attn(bias=torch.zeros(4), tp_size=2), "o_proj bias"),
        (_attn(neox=False), "NeoX"),
        (_attn(mrope=False), "MRoPE"),
    ],
)
def test_gqa_gates(monkeypatch, attn, reason):
    monkeypatch.setattr(
        fap, "_linear", lambda *a, **k: (_ for _ in ()).throw(_Reached())
    )
    with pytest.raises(fap.Ineligible, match=reason):
        fap.FusedGqa.build(attn, None, has_residual=True, v_from_k=False, page_size=256)


@pytest.mark.parametrize(
    "attn",
    [
        _attn(),
        _attn(bias=torch.zeros(4), tp_size=1),  # A bias without TP is fine.
        _attn(mrope=True),  # Interleaved MRoPE (Qwen3.5).
    ],
)
def test_gqa_gates_admit(monkeypatch, attn):
    monkeypatch.setattr(
        fap, "_linear", lambda *a, **k: (_ for _ in ()).throw(_Reached())
    )
    with pytest.raises(_Reached):
        fap.FusedGqa.build(attn, None, has_residual=True, v_from_k=False, page_size=256)


def test_act_quant_block():
    fp8 = torch.empty(5376, 8, dtype=torch.float8_e4m3fn)
    assert fap._act_quant_block(fp8) == 768
    assert fap._act_quant_block(torch.empty(4096, 8, dtype=torch.float8_e4m3fn)) == 1024
    # bf16 weights don't quantize activations: any hidden size is fine.
    assert fap._act_quant_block(torch.empty(1152, 8, dtype=torch.bfloat16)) == 1024
    with pytest.raises(fap.Ineligible, match="multiple of 256"):
        fap._act_quant_block(torch.empty(1152, 8, dtype=torch.float8_e4m3fn))


def test_check_kernel_turns_config_asserts_into_ineligible():
    def kernel(x):
        raise AssertionError("no tile fits in VMEM")

    with pytest.raises(fap.Ineligible, match="fused_gdn: no tile fits in VMEM"):
        fap._check_kernel(
            "fused_gdn", kernel, torch.empty(128, 8, dtype=torch.bfloat16)
        )


# prepare_model / arm.


def test_prepare_model_only_walks_patched_layers():
    class Llama(torch.nn.Module):  # Has the attributes, but is not patched.
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.Identity()
            self.self_attn = torch.nn.Module()

    model = torch.nn.Sequential(Llama())
    assert fap.prepare_model(model, SimpleNamespace(), page_size=256) == 0
    assert not fap.is_fused(model[0].self_attn)


def _gemma_layer():
    layer = gemma4.Gemma4DecoderLayer.__new__(gemma4.Gemma4DecoderLayer)
    torch.nn.Module.__init__(layer)
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(torch.ones(8))
    layer.input_layernorm = norm
    layer.self_attn = torch.nn.Module()
    return layer, norm


def test_arm_swaps_gemma_norm_and_rearms(monkeypatch):
    layer, norm = _gemma_layer()
    model = torch.nn.Sequential(layer)
    keys = set(model.state_dict())
    seen_norms = []

    def prepare(model, hf_config, *, page_size):
        seen_norms.append(model[0].input_layernorm)
        setattr(model[0].self_attn, fap._ATTR, object())
        return 1

    monkeypatch.setattr(fap, "prepare_model", prepare)
    for _ in range(2):  # Arming again starts from the original norm.
        assert patch_mod.arm(model, SimpleNamespace(), page_size=256) == 1
        assert isinstance(layer.input_layernorm, patch_mod._NormInAttention)
        assert layer.input_layernorm.unfused[0] is norm
        assert layer.self_attn._norm_in_attention is layer.input_layernorm
        assert set(model.state_dict()) == keys
    assert seen_norms == [norm, norm]


def test_arm_leaves_unfused_gemma_layer_alone(monkeypatch):
    layer, norm = _gemma_layer()
    monkeypatch.setattr(fap, "prepare_model", lambda *a, **k: 0)
    patch_mod.arm(torch.nn.Sequential(layer), SimpleNamespace(), page_size=256)
    assert layer.input_layernorm is norm
