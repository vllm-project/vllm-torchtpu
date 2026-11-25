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
"""CPU unit tests for the model-agnostic ``weight_mapping`` manifest generator."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.rl._fakes import (
    ColumnParallelLinear,
    GdnColumnParallelLinear,
    GdnInterleavedColumnParallelLinear,
    Qwen2ForCausalLM,
    Qwen3_5MoeForCausalLM,
    ReplicatedLinear,
    RoutedExperts,
    build_qwen2_hf_weights,
    build_qwen35_hf_weights,
)
from vllm_torchtpu.rl.manifest_generator import (
    generate_manifest,
    generate_manifest_dict,
)
from vllm_torchtpu.rl.weight_conversion import (
    from_hf,
    invert_manifest,
    to_hf,
    validate_manifest,
)
from vllm_torchtpu.rl.weight_installer import slice_tensor_for_shard


def _rule_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        r["target_keys"][0]: r
        for r in manifest["rules"]
        if len(r["target_keys"]) == 1 and not r["target_keys"][0].startswith("__tmp__:")
    }


def _assert_round_trip(
    manifest_json: str, hf_weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    converted = from_hf(manifest_json, hf_weights)
    assert not any(k.startswith("__tmp__:") for k in converted)
    inv = invert_manifest(manifest_json)
    validate_manifest(inv)
    recovered = to_hf(json.dumps(inv), converted)
    assert set(recovered) == set(hf_weights)
    for k, orig in hf_weights.items():
        assert torch.equal(recovered[k], orig), f"Round-trip mismatch for {k}"
    return converted


@pytest.mark.parametrize(
    "tp_size,num_heads,num_kv_heads,tie_word_embeddings,padded_vocab",
    [
        (1, 8, 2, True, 64),
        (2, 8, 2, True, 64),
        (4, 8, 2, True, 64),  # GQA KV replication (num_kv_head_replicas=2)
        (2, 4, 4, False, 72),  # Pure MHA + untied lm_head + vocab padding
    ],
)
def test_qwen2_dense_parity_and_roundtrip(
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    tie_word_embeddings: bool,
    padded_vocab: int,
) -> None:
    model = Qwen2ForCausalLM(
        vocab_size=64,
        padded_vocab_size=padded_vocab,
        hidden_size=64,
        intermediate_size=128,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tp_size=tp_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    manifest_json = generate_manifest(model, tp_size=tp_size)
    manifest = json.loads(manifest_json)
    validate_manifest(manifest)
    assert manifest["stripes"] == {"tp_size": tp_size}
    assert manifest["expert_axis_grouped"] is False
    if tp_size > num_kv_heads:
        assert "repeat" in [
            next(iter(op)) for r in manifest["rules"] for op in r["ops"]
        ]
    assert manifest["tied_keys"] == (
        [["model.embed_tokens.weight", "lm_head.weight"]] if tie_word_embeddings else []
    )

    hf = build_qwen2_hf_weights(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tie_word_embeddings=tie_word_embeddings,
    )
    converted = _assert_round_trip(manifest_json, hf)
    rules = _rule_map(manifest)

    # Bit-exact parity across every TP rank via contiguous slicing
    kv_rep = max(1, tp_size // num_kv_heads)
    for r in range(tp_size):
        q_w = hf["model.layers.0.self_attn.q_proj.weight"].chunk(tp_size, dim=0)[r]
        k_w = hf["model.layers.0.self_attn.k_proj.weight"].chunk(
            tp_size // kv_rep, dim=0
        )[r // kv_rep]
        v_w = hf["model.layers.0.self_attn.v_proj.weight"].chunk(
            tp_size // kv_rep, dim=0
        )[r // kv_rep]
        ref_qkv = torch.cat([q_w, k_w, v_w], dim=0).transpose(0, 1).contiguous()
        act_qkv = slice_tensor_for_shard(
            "model.layers.0.self_attn.qkv_proj.weight",
            converted["model.layers.0.self_attn.qkv_proj.weight"],
            rules["model.layers.0.self_attn.qkv_proj.weight"]["sharding"],
            {"tp": (r, tp_size)},
        )
        assert torch.equal(act_qkv, ref_qkv)


@pytest.mark.parametrize(
    "tp_size,dp_size,ep_size,moe_tp_size,qkvz_flipped",
    [
        (1, 1, 1, 1, False),
        (2, 1, 1, 2, False),
        (4, 1, 1, 4, True),  # GDN K/Q replication (tp=4, num_k_heads=2)
        (2, 1, 2, 1, False),
        (2, 2, 2, 2, False),  # Hybrid EP=2 + MoE-TP=2
    ],
)
def test_qwen35_moe_2d_parity_and_roundtrip(
    tp_size: int, dp_size: int, ep_size: int, moe_tp_size: int, qkvz_flipped: bool
) -> None:
    model = Qwen3_5MoeForCausalLM(
        tp_size=tp_size,
        ep_size=ep_size,
        moe_tp_size=moe_tp_size,
        qkvz_flipped=qkvz_flipped,
    )
    manifest_json = generate_manifest(
        model,
        tp_size=tp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        expert_axis_grouped=False,
    )
    manifest = json.loads(manifest_json)
    assert manifest["expert_axis_grouped"] is False
    assert manifest["stripes"] == {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "moe_tp_size": moe_tp_size,
    }
    hf = build_qwen35_hf_weights()
    converted = _assert_round_trip(manifest_json, hf)
    rules = _rule_map(manifest)

    # Independent per-(ep, moe_tp)-rank value parity for RoutedExperts w13 & w2
    w13 = converted["model.layers.0.mlp.experts.w13_weight"]
    w2 = converted["model.layers.0.mlp.experts.w2_weight"]
    num_e, inter_total, hidden = 4, 192, 64
    e_per_ep = num_e // ep_size
    unpadded = inter_total // moe_tp_size
    aligned = ((unpadded + 127) // 128) * 128
    pad_amt = aligned - unpadded
    for ep_r in range(ep_size):
        for mtp_r in range(moe_tp_size):
            coords = {"ep": (ep_r, ep_size), "moe_tp": (mtp_r, moe_tp_size)}
            local_w13 = slice_tensor_for_shard(
                "model.layers.0.mlp.experts.w13_weight",
                w13,
                rules["model.layers.0.mlp.experts.w13_weight"]["sharding"],
                coords,
            )
            local_w2 = slice_tensor_for_shard(
                "model.layers.0.mlp.experts.w2_weight",
                w2,
                rules["model.layers.0.mlp.experts.w2_weight"]["sharding"],
                coords,
            )
            assert local_w13.shape == model.model.layers[0].mlp.experts.w13_weight.shape
            assert local_w2.shape == model.model.layers[0].mlp.experts.w2_weight.shape
            # Reference per VllmUnquantizedFusedMoEMethod PWAL:
            # slice per (ep_r, mtp_r), right-pad each half to 128, transpose.
            ref_w13_list, ref_w2_list = [], []
            for e in range(ep_r * e_per_ep, (ep_r + 1) * e_per_ep):
                g_e = hf[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
                u_e = hf[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
                d_e = hf[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]
                g_s = g_e[mtp_r * unpadded : (mtp_r + 1) * unpadded, :]
                u_s = u_e[mtp_r * unpadded : (mtp_r + 1) * unpadded, :]
                d_s = d_e[:, mtp_r * unpadded : (mtp_r + 1) * unpadded]
                if pad_amt:
                    z1 = torch.zeros(pad_amt, hidden, dtype=g_s.dtype)
                    g_s = torch.cat([g_s, z1], dim=0)
                    u_s = torch.cat([u_s, z1], dim=0)
                    z2 = torch.zeros(hidden, pad_amt, dtype=d_s.dtype)
                    d_s = torch.cat([d_s, z2], dim=1)
                ref_w13_list.append(torch.cat([g_s, u_s], dim=0).T)
                ref_w2_list.append(d_s.T)
            assert torch.equal(local_w13, torch.stack(ref_w13_list, dim=0))
            assert torch.equal(local_w2, torch.stack(ref_w2_list, dim=0))

    # Independent per-TP-rank value parity for GDN in_proj_qkvz (with K/Q replication
    # when tp_size=4 > num_k_heads=2) and 3D conv1d.
    qkvz = converted["model.layers.0.linear_attn.in_proj_qkvz.weight"]
    conv = converted["model.layers.0.linear_attn.conv1d.weight"]
    hf_qkv = hf["model.layers.0.linear_attn.in_proj_qkv.weight"]
    hf_z = hf["model.layers.0.linear_attn.in_proj_z.weight"]
    hf_conv = hf["model.layers.0.linear_attn.conv1d.weight"]
    q_w, k_w, v_w = torch.split(hf_qkv, [16, 16, 32], dim=0)
    cq, ck, cv = torch.split(hf_conv, [16, 16, 32], dim=0)
    k_rep = max(1, tp_size // 2)  # num_k_heads=2
    q_rep_w = q_w.reshape(2, 8, hidden).repeat_interleave(k_rep, dim=0)
    k_rep_w = k_w.reshape(2, 8, hidden).repeat_interleave(k_rep, dim=0)
    cq_rep = cq.reshape(2, 8, 1, 4).repeat_interleave(k_rep, dim=0)
    ck_rep = ck.reshape(2, 8, 1, 4).repeat_interleave(k_rep, dim=0)
    q_per = (2 * k_rep) // tp_size
    v_per = 32 // tp_size
    for r in range(tp_size):
        local_qkvz = slice_tensor_for_shard(
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            qkvz,
            rules["model.layers.0.linear_attn.in_proj_qkvz.weight"]["sharding"],
            {"tp": (r, tp_size)},
        )
        ref_qkvz = torch.cat(
            [
                q_rep_w[r * q_per : (r + 1) * q_per].reshape(-1, hidden),
                k_rep_w[r * q_per : (r + 1) * q_per].reshape(-1, hidden),
                v_w[r * v_per : (r + 1) * v_per],
                hf_z[r * v_per : (r + 1) * v_per],
            ],
            dim=0,
        )
        if qkvz_flipped:
            ref_qkvz = ref_qkvz.T
        assert torch.equal(local_qkvz, ref_qkvz)

        local_conv = slice_tensor_for_shard(
            "model.layers.0.linear_attn.conv1d.weight",
            conv,
            rules["model.layers.0.linear_attn.conv1d.weight"]["sharding"],
            {"tp": (r, tp_size)},
        )
        ref_conv = torch.cat(
            [
                cq_rep[r * q_per : (r + 1) * q_per].reshape(-1, 1, 4),
                ck_rep[r * q_per : (r + 1) * q_per].reshape(-1, 1, 4),
                cv[r * v_per : (r + 1) * v_per],
            ],
            dim=0,
        )
        assert torch.equal(local_conv, ref_conv)


def test_qwen35_moe_canonical_3d_stacked_roundtrip() -> None:
    """Default expert_axis_grouped=True emits 3D stacked expert keys (no stack op)."""
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=2, moe_tp_size=1)
    manifest_json = generate_manifest(model, tp_size=2, dp_size=1, ep_size=2)
    manifest = json.loads(manifest_json)
    assert manifest["expert_axis_grouped"] is True

    w13_rule = _rule_map(manifest)["model.layers.0.mlp.experts.w13_weight"]
    assert w13_rule["source_keys"] == [
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
    ]
    assert all("stack" not in op for op in w13_rule["ops"])

    hf_2d = build_qwen35_hf_weights()
    hf_3d = {k: v for k, v in hf_2d.items() if ".mlp.experts." not in k}
    for li in (0, 1):
        pfx = f"model.layers.{li}.mlp.experts"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            hf_3d[f"{pfx}.{proj}.weight"] = torch.stack(
                [hf_2d[f"{pfx}.{e}.{proj}.weight"] for e in range(4)], dim=0
            )
    _assert_round_trip(manifest_json, hf_3d)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gdn_interleaved_ba_parity_and_roundtrip(tp_size: int) -> None:
    m = nn.Module()
    m.in_proj_ba = GdnInterleavedColumnParallelLinear(
        input_size=16,
        num_kq_heads=2,
        num_v_heads=4,
        group_sizes=[2, 2],
        replicated_segments=[False, False],
        tp_size=tp_size,
        flipped=True,
    )
    manifest_json = generate_manifest(m, tp_size=tp_size)
    hf_w = torch.randn(8, 16, dtype=torch.bfloat16)
    converted = _assert_round_trip(manifest_json, {"in_proj_ba.weight": hf_w})
    sharding = _rule_map(json.loads(manifest_json))["in_proj_ba.weight"]["sharding"]
    # Reference: HF layout is [num_kq_heads=2, n_segs=2, seg_w=2, hidden=16];
    # each TP rank owns [b_local, a_local] transposed (flipped=True).
    grouped = hf_w.reshape(2, 2, 2, 16)
    b_all = grouped[:, 0].reshape(4, 16)
    a_all = grouped[:, 1].reshape(4, 16)
    seg_per_rank = 4 // tp_size
    for r in range(tp_size):
        local = slice_tensor_for_shard(
            "in_proj_ba.weight",
            converted["in_proj_ba.weight"],
            sharding,
            {"tp": (r, tp_size)},
        )
        ref = torch.cat(
            [
                b_all[r * seg_per_rank : (r + 1) * seg_per_rank],
                a_all[r * seg_per_rank : (r + 1) * seg_per_rank],
            ],
            dim=0,
        ).T
        assert torch.equal(local, ref)


@pytest.mark.parametrize("use_mapper", [False, True])
def test_qwen35_vlm_prefix_and_hf_to_vllm_mapper_roundtrip(use_mapper: bool) -> None:
    class Qwen3_5MoeForConditionalGeneration(nn.Module):
        def __init__(self, lm: nn.Module) -> None:
            super().__init__()
            self.language_model = lm
            if use_mapper:
                self.hf_to_vllm_mapper = SimpleNamespace(
                    orig_to_new_prefix={
                        "model.language_model.": "language_model.model.",
                        "lm_head.": "language_model.lm_head.",
                        "custom.embed.": "language_model.model.embed_tokens.",
                    }
                )

    vlm = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForCausalLM())
    manifest_json = generate_manifest(vlm, expert_axis_grouped=False)
    manifest = json.loads(manifest_json)
    src_keys = {k for r in manifest["rules"] for k in r["source_keys"]}
    assert (
        "custom.embed.weight"
        if use_mapper
        else "model.language_model.embed_tokens.weight"
    ) in src_keys
    assert "lm_head.weight" in src_keys

    hf = {}
    for k, v in build_qwen35_hf_weights().items():
        if use_mapper and k == "model.embed_tokens.weight":
            hf["custom.embed.weight"] = v
        elif k.startswith("model."):
            hf["model.language_model." + k[len("model.") :]] = v
        else:
            hf[k] = v
    _assert_round_trip(manifest_json, hf)


def test_model_agnostic_reflection_and_config_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer reflection works on unregistered Llama-like models, unquantized config,
    square experts, None optional submodules, and TORCH_TPU_DP_SIZE."""

    class LlamaForCausalLM(Qwen2ForCausalLM):
        pass

    m = LlamaForCausalLM()
    assert generate_manifest_dict(m) == json.loads(generate_manifest(m))

    # Unquantized model.quant_config allowed; quantized rejected
    class VllmUnquantizedConfig:
        @staticmethod
        def get_name() -> str:
            return "tpu-unquantized"

    m.quant_config = VllmUnquantizedConfig()
    assert generate_manifest_dict(m)["schema_version"] == "weight_mapping"
    m.quant_config = SimpleNamespace(get_name=lambda: "fp8")
    with pytest.raises(NotImplementedError, match="Quantization"):
        generate_manifest(m)

    # TORCH_TPU_DP_SIZE env override for DP/EP inference
    from vllm_torchtpu.rl.manifest_generator import _resolve_parallel

    pc = SimpleNamespace(
        tensor_parallel_size=2, data_parallel_size=1, enable_expert_parallel=True
    )
    m_dp = Qwen2ForCausalLM(tp_size=2)
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    assert _resolve_parallel(m_dp, pc, None, None, None) == (2, 1, 2)
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "2")
    assert _resolve_parallel(m_dp, pc, None, None, None) == (2, 2, 4)

    # Square routed experts: UnquantizedFusedMoEMethod vs ambiguous fail-closed
    class UnquantizedFusedMoEMethod(nn.Module):
        pass

    m_sq = Qwen3_5MoeForCausalLM(
        hidden_size=128, num_experts=2, moe_intermediate_size=128
    )
    exp = m_sq.model.layers[0].mlp.experts
    exp.w13_weight = nn.Parameter(
        torch.zeros(2, 256, 128, dtype=torch.bfloat16), requires_grad=False
    )
    exp.w2_weight = nn.Parameter(
        torch.zeros(2, 128, 128, dtype=torch.bfloat16), requires_grad=False
    )
    exp.quant_method = UnquantizedFusedMoEMethod()
    w2_rule = _rule_map(generate_manifest_dict(m_sq))[
        "model.layers.0.mlp.experts.w2_weight"
    ]
    assert all("transpose" not in op for op in w2_rule["ops"])
    exp.quant_method = None
    with pytest.raises(NotImplementedError, match="ambiguous square"):
        generate_manifest(m_sq)

    # Optional submodules set to None are skipped cleanly
    m_opt = Qwen3_5MoeForCausalLM(num_experts=2, moe_intermediate_size=64)
    m_opt.model.layers[0].self_attn = None
    m_opt.model.layers[0].mlp.shared_expert = None
    m_opt.model.layers[0].mlp.shared_expert_gate = None
    m_opt.model.layers[1].self_attn.q_norm = None
    assert generate_manifest_dict(m_opt)["schema_version"] == "weight_mapping"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model_config": SimpleNamespace(quantization="fp8")}, "Quantization"),
        ({"model_config": SimpleNamespace(quantization="awq")}, "Quantization"),
        (
            {
                "model_config": SimpleNamespace(
                    hf_config=SimpleNamespace(quantization_config={"a": 1})
                )
            },
            "Quantization",
        ),
        ({"model_config": SimpleNamespace(enable_lora=True)}, "LoRA"),
        ({"model_config": SimpleNamespace(enable_mtp=True)}, "MTP"),
        ({"parallel_config": SimpleNamespace(pipeline_parallel_size=2)}, "PP"),
        ({"parallel_config": SimpleNamespace(prefill_context_parallel_size=2)}, "PCP"),
        ({"parallel_config": SimpleNamespace(decode_context_parallel_size=2)}, "DCP"),
        ({"parallel_config": SimpleNamespace(enable_eplb=True)}, "EPLB"),
        (
            {
                "parallel_config": SimpleNamespace(
                    eplb_config=SimpleNamespace(num_redundant_experts=1)
                )
            },
            "EPLB",
        ),
    ],
)
def test_fail_closed_config_guards(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(NotImplementedError, match=match):
        generate_manifest(Qwen2ForCausalLM(), **kwargs)


def test_fail_closed_model_and_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    # Quantized dtype, method, scale param
    m = Qwen2ForCausalLM()
    m.model.layers[0].self_attn.o_proj.weight = nn.Parameter(
        torch.zeros(64, 64, dtype=torch.int8), requires_grad=False
    )
    with pytest.raises(NotImplementedError, match="Unsupported dtype"):
        generate_manifest(m)

    class VllmFp8LinearMethodTPU:
        pass

    m2 = Qwen2ForCausalLM()
    m2.model.layers[0].self_attn.o_proj.quant_method = VllmFp8LinearMethodTPU()
    with pytest.raises(NotImplementedError, match="Quantized method"):
        generate_manifest(m2)

    m_scale = nn.Module()
    layer = ReplicatedLinear(16, 16)
    layer.weight_scale = nn.Parameter(
        torch.ones(16, dtype=torch.float32), requires_grad=False
    )
    m_scale.proj = layer
    with pytest.raises(NotImplementedError, match="Quantization scale parameter"):
        generate_manifest(m_scale)

    # LoRA module, vision/MTP prefix, unrecognized nn.Linear, GQA head padding
    class ColumnParallelLinearWithLoRA(ColumnParallelLinear):
        pass

    m_lora = nn.Module()
    m_lora.proj = ColumnParallelLinearWithLoRA(16, 16)
    with pytest.raises(NotImplementedError, match="LoRA"):
        generate_manifest(m_lora)

    for attr in ("visual", "mtp"):
        mv = nn.Module()
        setattr(mv, attr, ReplicatedLinear(16, 16))
        with pytest.raises(NotImplementedError, match="vision/MTP"):
            generate_manifest(mv)

    m_unk = nn.Module()
    m_unk.unknown = nn.Linear(16, 16, bias=False)
    with pytest.raises(NotImplementedError, match="Unsupported layer"):
        generate_manifest(m_unk)

    m_gqa = Qwen2ForCausalLM(num_heads=8, num_kv_heads=2, tp_size=2)
    m_gqa.model.layers[0].self_attn.orig_total_num_heads = 7
    m_gqa.model.layers[0].self_attn.total_num_heads = 8
    with pytest.raises(NotImplementedError, match="zero-padding"):
        generate_manifest(m_gqa, tp_size=2)

    # RoutedExperts bias, swigluoai, non-divisible TP
    m_bias = Qwen3_5MoeForCausalLM(num_experts=2, moe_intermediate_size=64)
    m_bias.model.layers[0].mlp.experts = RoutedExperts(
        num_experts=2, hidden_size=64, intermediate_size=64, has_bias=True
    )
    with pytest.raises(NotImplementedError, match="bias"):
        generate_manifest(m_bias)

    m_swi = Qwen3_5MoeForCausalLM(num_experts=2, moe_intermediate_size=64)
    m_swi.model.layers[0].mlp.experts.activation = "swigluoai"
    with pytest.raises(NotImplementedError, match="swigluoai"):
        generate_manifest(m_swi)

    m_div = Qwen2ForCausalLM(intermediate_size=10, tp_size=1)
    m_div.model.layers[0].mlp.gate_up_proj.tp_size = 4
    with pytest.raises(ValueError, match="evenly divisible"):
        generate_manifest(m_div, tp_size=4)

    # Quantized GDN modules
    m_gdn = nn.Module()
    proj = GdnColumnParallelLinear(
        input_size=16, num_k_heads=2, num_v_heads=2, d_k=4, d_v=4, include_z=True
    )
    proj.quant_method = VllmFp8LinearMethodTPU()
    m_gdn.in_proj_qkvz = proj
    with pytest.raises(NotImplementedError, match="Quantized method"):
        generate_manifest(m_gdn)

    # Runtime guards: _tpu_dp_vocab, placement, fused shared, env var
    m_vocab = Qwen2ForCausalLM()
    m_vocab.model.embed_tokens._tpu_dp_vocab = True
    with pytest.raises(NotImplementedError, match="_tpu_dp_vocab"):
        generate_manifest(m_vocab)

    m_place = Qwen3_5MoeForCausalLM(num_experts=2, moe_intermediate_size=64)
    m_place.model.layers[0].mlp.experts.expert_placement_strategy = "round_robin"
    with pytest.raises(NotImplementedError, match="expert_placement_strategy"):
        generate_manifest(m_place)

    m_fused = Qwen3_5MoeForCausalLM(num_experts=2, moe_intermediate_size=64)
    m_fused.model.layers[0].mlp.experts.expert_map_manager = SimpleNamespace(
        num_fused_shared_experts=1
    )
    with pytest.raises(NotImplementedError, match="fused shared experts"):
        generate_manifest(m_fused)

    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", "round_robin")
    with pytest.raises(NotImplementedError, match="TPU_EXPERT_PLACEMENT"):
        generate_manifest(Qwen2ForCausalLM())
