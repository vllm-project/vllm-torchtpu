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
"""Unit tests for `megakernel_runtime.weight_collector`."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.pool.config import CacheExec
from vllm_torchtpu.megakernel.weights.ops import JaxArrayOps
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    WEIGHT_FLIPPED_ATTR,
    _orient_2d_kn,
    _own,
    _stack,
    extract_ecopy_layer,
    find_backbone,
    load_vllm_weights,
    reshard_layer_experts_ep_to_tp,
)


def _jax_to_torch(a: jax.Array) -> torch.Tensor:
    arr = np.asarray(a)
    if a.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def _make_fp8_linear(
    in_dim: int, out_dim: int, *, flipped: bool = True
) -> SimpleNamespace:
    if flipped:
        w = torch.randint(1, 120, (in_dim, out_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        w = torch.randint(1, 120, (out_dim, in_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(s, requires_grad=False),
    )
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def make_synthetic_qwen35_backbone(
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    *,
    fused_mlp: bool = True,
    fused_qkv: bool = False,
    e_cores: list[list[list[int]]] | None = None,
) -> SimpleNamespace:
    """Construct a synthetic Qwen3.5 backbone matching `g` and `ex`."""
    gdn_lay = GDNModule.lin_in_layout(g, ex)
    nh = g.nh(ex.R)
    nai = g.NAI(ex.R)
    q_dim = 2 * nh * g.HD
    kv_dim = (nai - q_dim) // 2
    ew = g.EW(ex.R)
    ir = g.IR(ex.R)
    layers = []

    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"

        if is_att:
            if fused_qkv:
                self_attn = SimpleNamespace(
                    qkv_proj=_make_fp8_linear(g.H, nai),
                    o_proj=_make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            else:
                self_attn = SimpleNamespace(
                    q_proj=_make_fp8_linear(g.H, q_dim),
                    k_proj=_make_fp8_linear(g.H, kv_dim, flipped=False),
                    v_proj=_make_fp8_linear(g.H, kv_dim),
                    o_proj=_make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            linear_attn = None
        else:
            in_qkvz = _make_fp8_linear(g.H, gdn_lay.QKV + gdn_lay.Z)
            in_ba = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((2 * gdn_lay.vh, g.H), dtype=torch.bfloat16) * 0.05,
                    requires_grad=False,
                )
            )
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l_idx}.linear_attn",
                in_proj_qkvz=in_qkvz,
                in_proj_ba=in_ba,
                out_proj=_make_fp8_linear(gdn_lay.Z, g.H),
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn((gdn_lay.QKV, 1, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                ),
                A_log=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.float32), requires_grad=False
                ),
                dt_bias=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.bfloat16), requires_grad=False
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(gdn_lay.VD, dtype=torch.float32),
                        requires_grad=False,
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        w13 = torch.randint(1, 120, (ew, g.H, 2 * ir), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w2 = torch.randint(1, 120, (ew, ir, g.H), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w13_s = torch.rand((ew, 1, 2 * ir), dtype=torch.float32) + 0.1
        w2_s = torch.rand((ew, 1, g.H), dtype=torch.float32) + 0.1
        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
        )
        if e_cores is not None:
            experts._ecopy_cores = e_cores[l_idx]

        f_r = g.ISR(ex.R)
        if fused_mlp:
            fused_gu_cols = g.E + 2 * g.IS
            fused_w = torch.randint(
                1, 120, (g.H, fused_gu_cols), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
            se_down_w = torch.randint(1, 120, (g.IS, g.H), dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            se_down_s = torch.rand((1, g.H), dtype=torch.float32) + 0.1
            se_gate_w = torch.randn((1, g.H), dtype=torch.bfloat16)
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=fused_w,
                _fused_gate_gu_fp8_scale=fused_s,
                _fused_se_down_fp8_weight=se_down_w,
                _fused_se_down_fp8_scale=se_down_s,
                _fused_se_gate_weight=se_gate_w,
            )
        else:
            gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((g.E, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            shared_expert = SimpleNamespace(
                gate_up_proj=_make_fp8_linear(g.H, 2 * f_r),
                down_proj=_make_fp8_linear(f_r, g.H, flipped=False),
            )
            shared_expert_gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((1, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            mlp = SimpleNamespace(
                experts=experts,
                gate=gate,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
            )

        layer = SimpleNamespace(
            layer_type=ltype,
            input_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            linear_attn=linear_attn,
            self_attn=self_attn,
            mlp=mlp,
        )
        layers.append(layer)

    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
        )
    )
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=norm)


def test_own_and_stack_allocate_base_buffers() -> None:
    """Verify `_own` and `_stack` return non-view base buffers without extra copy."""
    base = torch.arange(32, dtype=torch.float32).reshape(4, 8).clone()
    assert _own(base).data_ptr() == base.data_ptr()

    sliced = base[1:3, 2:6]
    assert sliced._is_view() or sliced.storage_offset() != 0

    owned = _own(sliced)
    assert not owned._is_view()
    assert owned.storage_offset() == 0
    assert torch.equal(owned, sliced)

    stacked = _stack([sliced, sliced])
    assert not stacked._is_view()
    assert stacked.storage_offset() == 0
    assert stacked.shape == (2, 2, 4)


def test_orient_2d_kn_square_orientation() -> None:
    """Verify square `[D, D]` weights respect `WEIGHT_FLIPPED_ATTR`."""
    d = 64
    w_raw = (
        torch.arange(d * d, dtype=torch.int32)
        .remainder(120)
        .add(1)
        .to(torch.uint8)
        .reshape(d, d)
        .view(torch.float8_e4m3fn)
    )
    w_fl = _orient_2d_kn(w_raw.clone(), True, "sq", {d}, {d})
    assert torch.equal(w_fl.view(torch.uint8), w_raw.view(torch.uint8))

    w_unfl = _orient_2d_kn(w_raw.clone(), False, "sq", {d}, {d})
    assert torch.equal(
        w_unfl.view(torch.uint8), w_raw.transpose(0, 1).contiguous().view(torch.uint8)
    )

    with pytest.raises(ValueError, match="Ambiguous square weight shape"):
        _orient_2d_kn(w_raw.clone(), None, "sq", {d}, {d})


def test_find_backbone_wrappers() -> None:
    """Verify `find_backbone` locates `(layers, norm)` across nested vLLM wrappers."""
    inner = SimpleNamespace(layers=[1], norm=object())
    for wrapped in (
        inner,
        SimpleNamespace(model=inner),
        SimpleNamespace(model=SimpleNamespace(model=inner)),
        SimpleNamespace(language_model=SimpleNamespace(model=inner)),
        SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(model=inner))
        ),
    ):
        assert find_backbone(wrapped) is inner

    with pytest.raises(ValueError, match="Cannot locate decoder backbone"):
        find_backbone(SimpleNamespace(foo=1))


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
    from vllm_torchtpu.megakernel.model_def import (
        Every,
        FromHF,
        LMHead,
        ModelDef,
        Schedule,
    )
    from vllm_torchtpu.megakernel.modules import GDN, GQA, Embedding, MoE, Norm

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


def test_reshard_layer_experts_ep_to_tp() -> None:
    """Verify `reshard_layer_experts_ep_to_tp` slices experts and updates MoE flags."""
    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
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
        moe_tp=True,
        e_slots=0,
    )
    ex_ep = ExecSpec(
        R=2, B=4, T=1, ctx=256, stack_e=False, tiles=TINY_TILES, interpret=True
    )
    g_ep = replace(g, moe_tp=False, e_slots=68)
    backbone = make_synthetic_qwen35_backbone(g_ep, ex_ep)
    mlp = backbone.layers[0].mlp
    mlp.experts.moe_config = SimpleNamespace(
        use_ep=True,
        is_sequence_parallel=True,
        moe_parallel_config=SimpleNamespace(
            use_ep=True,
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            sp_size=2,
            is_sequence_parallel=True,
        ),
    )

    with pytest.raises(ValueError, match="all_gather_fn must be provided"):
        reshard_layer_experts_ep_to_tp(mlp, tp_rank=1, g=g, ex=ex_ep)

    # Simulate 2-rank all_gather by repeating primary `ER=64` experts twice -> `E=128`
    reshard_layer_experts_ep_to_tp(
        mlp,
        tp_rank=1,
        g=g,
        ex=ex_ep,
        all_gather_fn=lambda t: torch.cat([t, t], dim=0),
    )
    ir = g.I // ex_ep.R
    assert tuple(mlp.experts.w13_weight.shape) == (g.E, g.H, 2 * ir)
    assert tuple(mlp.experts.w2_weight.shape) == (g.E, ir, g.H)
    assert not mlp.experts.w13_weight.data._is_view()
    assert not mlp.experts.w2_weight.data._is_view()
    assert mlp.experts.moe_config.use_ep is False
    assert mlp.experts.moe_config.moe_parallel_config.tp_size == 2
    assert mlp.experts.moe_config.moe_parallel_config.tp_rank == 1
    assert mlp.experts.moe_config.moe_parallel_config.ep_size == 1


def test_load_vllm_weights_does_not_touch_jax_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_vllm_weights` with `tiles=None` never calls `jax.devices()`."""
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
    ex = ExecSpec(R=2, B=4, T=1, ctx=2048, stack_e=False, tiles=None)
    backbone = make_synthetic_qwen35_backbone(g, ex)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("jax.devices() must not be called during load_vllm_weights")
        ),
    )
    packed = load_vllm_weights(qwen35.Qwen35, backbone, ex, g=g, tp_rank=1)
    assert set(packed["plan"].w_names) <= set(packed)


def test_routed_experts_wrapper_and_scale_inv_fallback() -> None:
    """Verify `_walk_attr` reaches `experts.routed_experts` with `*_scale_inv`."""
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
        stack_e=False,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )
    e_cores = mk.default_eplb_cores(g, ex.R)
    backbone = make_synthetic_qwen35_backbone(g, ex, e_cores=e_cores)
    for layer in backbone.layers:
        inner = layer.mlp.experts
        routed = SimpleNamespace(
            w13_weight=inner.w13_weight,
            w2_weight=inner.w2_weight,
            w13_weight_scale_inv=inner.w13_weight_scale,
            w2_weight_scale_inv=inner.w2_weight_scale,
            _ecopy_cores=inner._ecopy_cores,
        )
        layer.mlp.experts = SimpleNamespace(routed_experts=routed)

    packed = load_vllm_weights(qwen35.Qwen35, backbone, ex, g=g, tp_rank=0)
    for l_idx in range(g.L):
        routed = backbone.layers[l_idx].mlp.experts.routed_experts
        assert packed["e13"][l_idx].data_ptr() == routed.w13_weight.data_ptr()
        assert packed["e2"][l_idx].data_ptr() == routed.w2_weight.data_ptr()
    ecopy_off, ecopy_w = packed["plan"].smalls_layout["ecopy"]
    assert torch.any(packed["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w] > 0.0)


def test_load_vllm_weights_validation_errors() -> None:
    """Verify scale shape, expert shape, missing alias, and `g=None` behavior."""
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
        stack_e=False,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )

    # 1. Wrong 2D scale length raises ValueError.
    bb_2d = make_synthetic_qwen35_backbone(g, ex, fused_mlp=False)
    bb_2d.layers[0].linear_attn.out_proj.weight_scale = torch.nn.Parameter(
        torch.ones((17, 1), dtype=torch.float32), requires_grad=False
    )
    with pytest.raises(ValueError, match="weight_scale for 'linear_attn.out_proj'"):
        load_vllm_weights(qwen35.Qwen35, bb_2d, ex, g=g, tp_rank=0)

    # 2. Wrong 3D expert scale shape raises ValueError.
    bb_3d = make_synthetic_qwen35_backbone(g, ex, fused_mlp=True)
    bb_3d.layers[0].mlp.experts.w13_weight_scale = torch.nn.Parameter(
        torch.ones((68, 1, 7), dtype=torch.float32), requires_grad=False
    )
    with pytest.raises(ValueError, match="scale for 'mlp.experts.w13_weight'"):
        load_vllm_weights(qwen35.Qwen35, bb_3d, ex, g=g, tp_rank=0)

    # 3. Wrong expert EW (unpadded ER=64 when EW=68) raises ValueError on stack_e=False.
    bb_ew = make_synthetic_qwen35_backbone(g, ex, fused_mlp=True)
    bb_ew.layers[0].mlp.experts.w13_weight = torch.nn.Parameter(
        bb_ew.layers[0].mlp.experts.w13_weight[:64].contiguous(), requires_grad=False
    )
    bb_ew.layers[0].mlp.experts.w13_weight_scale = torch.nn.Parameter(
        bb_ew.layers[0].mlp.experts.w13_weight_scale[:64].contiguous(),
        requires_grad=False,
    )
    with pytest.raises(ValueError, match="Layer 0 weight 'e13' has shape"):
        load_vllm_weights(qwen35.Qwen35, bb_ew, ex, g=g, tp_rank=0)

    # 4. Missing alias lists all attempted vllm_aliases sources.
    bb_miss = make_synthetic_qwen35_backbone(g, ex, fused_mlp=False)
    delattr(bb_miss.layers[0].mlp, "gate")
    with pytest.raises(
        ValueError,
        match=r"\['mlp\._fused_gate_gu_fp8_weight', 'mlp\.gate\.weight'\]",
    ):
        load_vllm_weights(qwen35.Qwen35, bb_miss, ex, g=g, tp_rank=0)

    # 5. `load_vllm_weights(model_def, backbone, ex)` works without passing `g=`.
    class TinyQwen35(qwen35.Qwen35):
        default_hf_config = {
            **qwen35.Qwen35.default_hf_config,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "vocab_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        }

    g_default = TinyQwen35.geometry()
    bb_def = make_synthetic_qwen35_backbone(g_default, ex, fused_mlp=True)
    packed_no_g = load_vllm_weights(TinyQwen35, bb_def, ex, tp_rank=0)
    assert len(packed_no_g["e13"]) == 4


def test_load_vllm_weights_non_qwen_synth_hybrid_model() -> None:
    """Verify `load_vllm_weights` loads a non-Qwen `ModelDef` (`att_every=2`)."""
    from vllm_torchtpu.megakernel.model_def import (
        Every,
        FromHF,
        LMHead,
        ModelDef,
        Schedule,
    )
    from vllm_torchtpu.megakernel.modules import (
        GDN,
        GQA,
        Embedding,
        MoE,
        Norm,
    )

    class SynthHybridVLLM(ModelDef):
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
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
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

    g = SynthHybridVLLM.geometry(moe_tp=False, e_slots=68)
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex = ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        stack_e=False,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )
    backbone = make_synthetic_qwen35_backbone(g, ex, fused_mlp=True, fused_qkv=True)
    packed = load_vllm_weights(SynthHybridVLLM, backbone, ex, g=g, tp_rank=0)
    assert packed["att_in"].shape[0] == 2
    assert packed["lin_in"].shape[0] == 2
    for l_idx in range(g.L):
        assert (
            packed["e13"][l_idx].data_ptr()
            == backbone.layers[l_idx].mlp.experts.w13_weight.data_ptr()
        )
        assert (
            packed["e2"][l_idx].data_ptr()
            == backbone.layers[l_idx].mlp.experts.w2_weight.data_ptr()
        )


def _bf16_experts_geom() -> tuple[qwen35.Qwen35Geometry, ExecSpec]:
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
    ex = ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        stack_e=False,
        tiles=replace(TINY_TILES, bank=512 << 10),
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )
    return g, ex


def _to_bf16_experts(experts: Any) -> None:
    """Dequantize fp8 experts in place (as an unquantized checkpoint loads)."""
    for w in ("w13_weight", "w2_weight"):
        s = getattr(experts, f"{w}_scale").data.float()
        deq = (getattr(experts, w).data.float() * s).to(torch.bfloat16)
        setattr(experts, w, torch.nn.Parameter(deq, requires_grad=False))
        delattr(experts, f"{w}_scale")


def test_bf16_experts_load_without_scales() -> None:
    """bf16 routed experts load with no scale (same packed weights as their fp8
    source dequantized by the library); fp8 experts still require a scale."""
    import dataclasses

    class Bf16Experts(qwen35.Qwen35):
        name = "test_bf16_experts"
        moe = dataclasses.replace(
            qwen35.Qwen35.moe,
            options={
                **qwen35.Qwen35.moe.options,
                "weight_dtypes": {"e13": "bf16", "e2": "bf16"},
            },
        )
        schedule = dataclasses.replace(qwen35.Qwen35.schedule, ffn=moe)

    g, ex = _bf16_experts_geom()
    torch.manual_seed(0)
    backbone = make_synthetic_qwen35_backbone(g, ex)
    ref = load_vllm_weights(Bf16Experts, backbone, ex, g=g, tp_rank=1)
    for layer in backbone.layers:
        _to_bf16_experts(layer.mlp.experts)
    got = load_vllm_weights(Bf16Experts, backbone, ex, g=g, tp_rank=1)
    assert len(got["w_flat"]) == len(ref["w_flat"])
    for a, b in zip(got["w_flat"], ref["w_flat"], strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    # An fp8 expert weight without its scale is a clear error.
    torch.manual_seed(0)
    bb_fp8 = make_synthetic_qwen35_backbone(g, ex)
    delattr(bb_fp8.layers[0].mlp.experts, "w13_weight_scale")
    with pytest.raises(ValueError, match="w13_weight.* is missing its scale"):
        load_vllm_weights(qwen35.Qwen35, bb_fp8, ex, g=g, tp_rank=1)


def test_reshard_bf16_experts_ep_to_tp_without_scales() -> None:
    g, ex_ep = _bf16_experts_geom()
    g_tp = replace(g, moe_tp=True, e_slots=0)
    torch.manual_seed(0)
    fp8_mlp = make_synthetic_qwen35_backbone(g, ex_ep).layers[0].mlp
    torch.manual_seed(0)
    bf16_mlp = make_synthetic_qwen35_backbone(g, ex_ep).layers[0].mlp
    _to_bf16_experts(bf16_mlp.experts)

    def gather(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, t], dim=0)

    kw = dict(tp_rank=1, g=g_tp, ex=ex_ep, all_gather_fn=gather)
    reshard_layer_experts_ep_to_tp(fp8_mlp, **kw)
    reshard_layer_experts_ep_to_tp(bf16_mlp, **kw)
    ir = g.I // ex_ep.R
    bf16_ex, fp8_ex = bf16_mlp.experts, fp8_mlp.experts
    assert tuple(bf16_ex.w13_weight.shape) == (g.E, g.H, 2 * ir)
    assert tuple(bf16_ex.w2_weight.shape) == (g.E, ir, g.H)
    assert bf16_ex.w13_weight.dtype == torch.bfloat16
    assert not hasattr(bf16_ex, "w13_weight_scale")
    assert not hasattr(bf16_ex, "w2_weight_scale")
    # Same slices as the fp8 path, dequantized.
    for w, s in (("w13_weight", "w13_weight_scale"), ("w2_weight", "w2_weight_scale")):
        sc = getattr(fp8_ex, s).data.float().reshape(g.E, 1, -1)
        deq = (getattr(fp8_ex, w).data.float() * sc).to(torch.bfloat16)
        assert torch.equal(getattr(bf16_ex, w).data, deq)

    torch.manual_seed(0)
    bad = make_synthetic_qwen35_backbone(g, ex_ep).layers[0].mlp
    delattr(bad.experts, "w2_weight_scale")
    with pytest.raises(ValueError, match="missing w2_weight_scale"):
        reshard_layer_experts_ep_to_tp(bad, **kw)


def test_bridge_and_weight_collector_have_no_vllm_imports() -> None:
    """Verify `exported_bridge` and `weight_collector` are framework-portable.

    Neither module may import from `vllm` or `vllm_torchtpu` outside
    `vllm_torchtpu.megakernel`, so they can be extracted into a
    standalone library alongside `megakernel`.
    """
    import ast
    from pathlib import Path

    import vllm_torchtpu.megakernel_runtime.exported_bridge as bridge_mod
    import vllm_torchtpu.megakernel_runtime.weight_collector as wc_mod

    allowed_prefixes = ("vllm_torchtpu.megakernel",)
    for mod in (bridge_mod, wc_mod):
        path = Path(mod.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("vllm"):
                        assert name.startswith(allowed_prefixes), (
                            f"{path.name} has disallowed import {name!r}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mod_name = node.module or ""
                if mod_name.startswith("vllm"):
                    assert mod_name.startswith(allowed_prefixes), (
                        f"{path.name} has disallowed from-import {mod_name!r}"
                    )
