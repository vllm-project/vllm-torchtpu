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
"""Qwen3.5 tests of `megakernel.adapter.weight_collector.load_vllm_weights`:
base-buffer shapes / expert aliasing on the Qwen3.5 geometry and value parity
with `pack_logical_weights`. The model-free tests of `weight_source` /
`weight_collector` are `tests/megakernel/adapter/test_weight_source.py`."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.megakernel.core as mk
from vllm_torchtpu.megakernel.adapter.weight_collector import (
    WEIGHT_FLIPPED_ATTR,
    extract_ecopy_layer,
    load_vllm_weights,
)
from vllm_torchtpu.megakernel.core import planner
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.core.pool.config import CacheExec
from vllm_torchtpu.megakernel.core.weights.ops import JaxArrayOps
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import TINY_TILES
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    jax_to_torch as _jax_to_torch,
)
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    make_synthetic_backbone as make_synthetic_qwen35_backbone,
)


@pytest.mark.parametrize("stack_e,fused_mlp", [(False, True), (True, False)])
def test_load_vllm_weights_shapes_base_buffers_and_expert_aliasing(
    stack_e: bool, fused_mlp: bool
) -> None:
    """Verify `load_vllm_weights` shapes, base buffers, and `ecopy` extraction."""
    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=68,
    )
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex = ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        stack_e=stack_e,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )
    # 1. Without `_ecopy` / `_ecopy_cores` / `expert_placement`, `extract_ecopy_layer`
    # must return `None` (no fabricated fallback placement).
    backbone_no_ecopy = make_synthetic_qwen35_backbone(
        g, ex, fused_mlp=fused_mlp, fused_qkv=not fused_mlp, e_cores=None
    )
    assert (
        extract_ecopy_layer(0, backbone_no_ecopy.layers[0], g, ex, torch.device("cpu"))
        is None
    )
    packed_no_ecopy = load_vllm_weights(
        qwen35.Qwen35, backbone_no_ecopy, ex, g=g, tp_rank=1
    )
    plan: planner.Plan = packed_no_ecopy["plan"]
    ecopy_off, ecopy_w = plan.smalls_layout["ecopy"]
    ecopy_zero = packed_no_ecopy["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w]
    assert torch.count_nonzero(ecopy_zero).item() == 0

    # 2. With explicit `_ecopy_cores`, `extract_ecopy_layer` encodes the placement.
    e_cores = mk.default_eplb_cores(g, ex.R)
    backbone = make_synthetic_qwen35_backbone(
        g, ex, fused_mlp=fused_mlp, fused_qkv=not fused_mlp, e_cores=e_cores
    )
    packed = load_vllm_weights(qwen35.Qwen35, backbone, ex, g=g, tp_rank=1)
    w_shapes = planner.weight_shapes(qwen35.qwen35_spec(g), ex, plan)

    for k in plan.w_names:
        val = packed[k]
        if k in plan.dynamic_kinds and not stack_e:
            assert isinstance(val, tuple) and len(val) == g.L
            for l_idx in range(g.L):
                t_l = val[l_idx]
                orig = (
                    backbone.layers[l_idx].mlp.experts.w13_weight
                    if k == "e13"
                    else backbone.layers[l_idx].mlp.experts.w2_weight
                )
                assert t_l.data_ptr() == orig.data_ptr()
                assert t_l.storage_offset() == 0
                assert t_l.ndim == 3
        else:
            exp_shape, _ = w_shapes[k]
            assert isinstance(val, torch.Tensor)
            assert not val._is_view()
            assert val.storage_offset() == 0
            assert tuple(val.shape) == exp_shape

    ecopy_slice = packed["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w]
    assert torch.any(ecopy_slice > 0.0)


@pytest.mark.parametrize("model_kind", ["qwen35", "synth_att2"])
@pytest.mark.parametrize("fused_mlp", [False, True])
@pytest.mark.parametrize("stack_e", [False, True])
def test_load_vllm_weights_value_parity(
    model_kind: str, fused_mlp: bool, stack_e: bool
) -> None:
    """Verify `load_vllm_weights` matches `pack_logical_weights`."""
    from vllm_torchtpu.megakernel.core.model_def import (
        Every,
        FromHF,
        LMHead,
        ModelDef,
        Schedule,
    )
    from vllm_torchtpu.megakernel.core.modules import GDN, GQA, Embedding, MoE, Norm

    if model_kind == "qwen35":
        model_def: type[ModelDef] = qwen35.Qwen35
        att_every = 4
    else:

        class SynthHybridParity(ModelDef):
            default_hf_config = {
                "model_type": "synth_hybrid",
                "architectures": ["SynthHybridForCausalLM"],
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "vocab_size": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 128,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 256,
                "shared_expert_intermediate_size": 256,
            }
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                vocab="vocab_size",
                eps="rms_norm_eps",
                rope_theta="rope_theta",
                rope_partial=("partial_rotary_factor", 1.0),
            )
            attn = GQA.bind(
                prefix="self_attn",
                heads="num_attention_heads",
                kv_heads="num_key_value_heads",
                head_dim="head_dim",
                qk_norm=True,
                output_gate=True,
            )
            linear = GDN.bind(
                prefix="linear_attn",
                k_heads="linear_num_key_heads",
                v_heads="linear_num_value_heads",
                k_dim="linear_key_head_dim",
                v_dim="linear_value_head_dim",
                conv_kernel="linear_conv_kernel_dim",
            )
            moe = MoE.bind(
                prefix="mlp",
                experts="num_experts",
                top_k="num_experts_per_tok",
                inter="moe_intermediate_size",
                shared_inter="shared_expert_intermediate_size",
                shared_gate=True,
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=linear),
                ffn=moe,
                pre_norm=Norm.bind("input_layernorm", zero_centered=True),
                post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            )
            embed = Embedding("model.embed_tokens")
            final_norm = Norm.bind("model.norm", zero_centered=True)
            lm_head = LMHead("lm_head", tied_to=None)

        model_def = SynthHybridParity
        att_every = 2

    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=att_every,
        H=256,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=68,
    )
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex = ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        stack_e=stack_e,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    j_ops = JaxArrayOps()
    hf = dict(mk.random_hf_weights(jax.random.key(314), g))
    b_bf16 = (
        jnp.transpose(hf["lin_b"], (0, 2, 1)).astype(jnp.float32)
        * hf["lin_b_s"][:, None, :]
    ).astype(jnp.bfloat16)
    a_bf16 = (
        jnp.transpose(hf["lin_a"], (0, 2, 1)).astype(jnp.float32)
        * hf["lin_a_s"][:, None, :]
    ).astype(jnp.bfloat16)
    b_qs = [j_ops.quantize_fp8(b_bf16[i]) for i in range(b_bf16.shape[0])]
    a_qs = [j_ops.quantize_fp8(a_bf16[i]) for i in range(a_bf16.shape[0])]
    hf["lin_b"] = jnp.stack([q for q, _ in b_qs], axis=0)
    hf["lin_b_s"] = jnp.stack([s for _, s in b_qs], axis=0)
    hf["lin_a"] = jnp.stack([q for q, _ in a_qs], axis=0)
    hf["lin_a_s"] = jnp.stack([s for _, s in a_qs], axis=0)
    e_cores = mk.default_eplb_cores(g, ex.R)
    w_logical = dict(mk.load_logical_weights(model_def, ex, hf, g=g, e_cores=e_cores))

    tp_rank = 1
    gdn_lay = GDNModule.lin_in_layout(g, ex)
    qkvz_dim = gdn_lay.QKV + gdn_lay.Z
    r_fp8_layers: list[jax.Array] = []
    r_s_layers: list[jax.Array] = []
    if fused_mlp:
        router_deq_layers: list[jax.Array] = []
        for l_idx in range(g.L):
            r_q, r_s = j_ops.quantize_fp8(w_logical["router"][l_idx])
            r_fp8_layers.append(r_q)
            r_s_layers.append(r_s)
            router_deq_layers.append(j_ops.dequantize(r_q, r_s, jnp.bfloat16))
        w_logical["router"] = jnp.stack(router_deq_layers, axis=0)

    w_mk_ref = mk.pack_logical_weights(model_def, ex, w_logical, g=g)

    layers: list[Any] = []
    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        if is_att:
            a_idx = l_idx // g.att_every
            w_att_in = _jax_to_torch(w_logical["att_in"][tp_rank, a_idx])
            s_att_in = _jax_to_torch(w_logical["att_in_s"][tp_rank, a_idx])
            w_att_out = _jax_to_torch(w_logical["att_out"][tp_rank, a_idx])
            s_att_out = _jax_to_torch(w_logical["att_out_s"][tp_rank, a_idx])
            qkv_proj = SimpleNamespace(
                weight=torch.nn.Parameter(w_att_in, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_att_in, requires_grad=False),
            )
            setattr(qkv_proj, WEIGHT_FLIPPED_ATTR, True)
            o_proj = SimpleNamespace(
                weight=torch.nn.Parameter(w_att_out, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_att_out, requires_grad=False),
            )
            setattr(o_proj, WEIGHT_FLIPPED_ATTR, True)
            self_attn = SimpleNamespace(
                qkv_proj=qkv_proj,
                o_proj=o_proj,
                q_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["qn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
                k_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["kn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
            )
            linear_attn = None
        else:
            o_gdn = l_idx - (l_idx + 1) // g.att_every
            w_lin_full = _jax_to_torch(w_logical["lin_in"][tp_rank, o_gdn])
            s_lin_full = _jax_to_torch(w_logical["lin_in_s"][tp_rank, o_gdn])
            w_qkvz = w_lin_full[:, :qkvz_dim].contiguous()
            s_qkvz = s_lin_full[:qkvz_dim].contiguous()
            vh = gdn_lay.vh
            b_hf = _jax_to_torch(
                jnp.transpose(
                    b_bf16[o_gdn, :, tp_rank * vh : (tp_rank + 1) * vh], (1, 0)
                )
            )
            a_hf = _jax_to_torch(
                jnp.transpose(
                    a_bf16[o_gdn, :, tp_rank * vh : (tp_rank + 1) * vh], (1, 0)
                )
            )
            w_ba = torch.cat([b_hf, a_hf], dim=0).contiguous()

            in_proj_qkvz = SimpleNamespace(
                weight=torch.nn.Parameter(w_qkvz, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_qkvz, requires_grad=False),
            )
            setattr(in_proj_qkvz, WEIGHT_FLIPPED_ATTR, True)
            in_proj_ba = SimpleNamespace(
                weight=torch.nn.Parameter(w_ba, requires_grad=False),
            )
            out_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out_s"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
            )
            setattr(out_proj, WEIGHT_FLIPPED_ATTR, True)
            conv_4_qkv = _jax_to_torch(w_logical["conv"][tp_rank, o_gdn]).reshape(
                4, gdn_lay.QKV
            )
            conv_w = conv_4_qkv.transpose(0, 1).unsqueeze(1).contiguous()
            linear_attn = SimpleNamespace(
                in_proj_qkvz=in_proj_qkvz,
                in_proj_ba=in_proj_ba,
                out_proj=out_proj,
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(conv_w, requires_grad=False)
                ),
                A_log=torch.nn.Parameter(
                    _jax_to_torch(w_logical["alog"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                dt_bias=torch.nn.Parameter(
                    _jax_to_torch(w_logical["dtb"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["gnorm"][tp_rank, o_gdn]),
                        requires_grad=False,
                    )
                ),
            )
            self_attn = None

        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2"][tp_rank, l_idx]), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13_s"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2_s"][tp_rank, l_idx]), requires_grad=False
            ),
            _ecopy_cores=e_cores[l_idx],
        )
        f_r = g.ISR(ex.R)
        if fused_mlp:
            sh13_r0 = _jax_to_torch(w_logical["sh13"][0, l_idx])
            sh13_r1 = _jax_to_torch(w_logical["sh13"][1, l_idx])
            w_gu_unsharded = torch.cat(
                [
                    sh13_r0[:, :f_r],
                    sh13_r1[:, :f_r],
                    sh13_r0[:, f_r:],
                    sh13_r1[:, f_r:],
                ],
                dim=-1,
            ).contiguous()
            s13_r0 = _jax_to_torch(w_logical["sh13_s"][0, l_idx])
            s13_r1 = _jax_to_torch(w_logical["sh13_s"][1, l_idx])
            s_gu_unsharded = torch.cat(
                [s13_r0[:f_r], s13_r1[:f_r], s13_r0[f_r:], s13_r1[f_r:]], dim=0
            ).contiguous()
            r_fp8_t = _jax_to_torch(r_fp8_layers[l_idx])
            r_s_t = _jax_to_torch(r_s_layers[l_idx])
            fused_gu_w = torch.cat(
                [r_fp8_t.view(torch.uint8), w_gu_unsharded.view(torch.uint8)], dim=-1
            ).view(torch.float8_e4m3fn)
            fused_gu_s = torch.cat([r_s_t, s_gu_unsharded], dim=0).unsqueeze(0)
            w_d_unsharded = torch.cat(
                [
                    _jax_to_torch(w_logical["sh2"][0, l_idx]),
                    _jax_to_torch(w_logical["sh2"][1, l_idx]),
                ],
                dim=0,
            ).contiguous()
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=fused_gu_w,
                _fused_gate_gu_fp8_scale=fused_gu_s,
                _fused_se_down_fp8_weight=w_d_unsharded,
                _fused_se_down_fp8_scale=_jax_to_torch(
                    w_logical["sh2_s"][tp_rank, l_idx]
                ).unsqueeze(0),
                _fused_se_gate_weight=_jax_to_torch(
                    w_logical["sh_gate"][tp_rank, l_idx]
                ).unsqueeze(0),
            )
        else:
            gu_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh13"][tp_rank, l_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh13_s"][tp_rank, l_idx]),
                    requires_grad=False,
                ),
            )
            setattr(gu_proj, WEIGHT_FLIPPED_ATTR, True)
            down_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh2"][tp_rank, l_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh2_s"][tp_rank, l_idx]),
                    requires_grad=False,
                ),
            )
            setattr(down_proj, WEIGHT_FLIPPED_ATTR, True)
            mlp = SimpleNamespace(
                experts=experts,
                gate=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["router"][l_idx])
                        .transpose(0, 1)
                        .contiguous(),
                        requires_grad=False,
                    )
                ),
                shared_expert=SimpleNamespace(
                    gate_up_proj=gu_proj, down_proj=down_proj
                ),
                shared_expert_gate=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["sh_gate"][tp_rank, l_idx]),
                        requires_grad=False,
                    )
                ),
            )
        layers.append(
            SimpleNamespace(
                layer_type="full_attention" if is_att else "linear_attention",
                input_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["in_norm"][l_idx]), requires_grad=False
                    )
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["post_norm"][l_idx]),
                        requires_grad=False,
                    )
                ),
                linear_attn=linear_attn,
                self_attn=self_attn,
                mlp=mlp,
            )
        )

    backbone = SimpleNamespace(
        layers=layers,
        norm=SimpleNamespace(
            weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["final_norm"]), requires_grad=False
            )
        ),
    )
    packed = load_vllm_weights(model_def, backbone, ex, g=g, tp_rank=tp_rank)
    plan: planner.Plan = packed["plan"]

    for k in plan.w_names:
        val_t = packed[k]
        ref_j = w_mk_ref[k]
        if k in plan.dynamic_kinds and not stack_e:
            for l_idx in range(g.L):
                ref_l = np.asarray(ref_j[l_idx])[tp_rank]
                # `pack_logical_weights` tiles `[EW, NT, TK, N]`; unstacked `[EW, K, N]`
                ref_3d = ref_l.reshape(val_t[l_idx].shape)
                np.testing.assert_array_equal(
                    val_t[l_idx].view(torch.uint8).numpy(),
                    ref_3d.view(np.uint8),
                    err_msg=f"mismatch in {k}[{l_idx}]",
                )
        else:
            ref_r = np.asarray(ref_j)[tp_rank]
            np.testing.assert_array_equal(
                val_t.view(torch.uint8).numpy(),
                ref_r.view(np.uint8),
                err_msg=f"mismatch in {k}",
            )
