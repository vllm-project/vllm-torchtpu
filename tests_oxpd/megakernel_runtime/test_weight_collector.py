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

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.pool.config import CacheExec
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    WEIGHT_FLIPPED_ATTR,
    _own,
    _stack,
    extract_ecopy_layer,
    extract_kn_fp8_weight,
    find_backbone,
    pack_weights,
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
    gdn_lay = GDNBlock.lin_in_layout(g, ex)
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

        if fused_mlp:
            fused_gu_cols = g.E + 2 * g.IS
            fused_w = torch.randint(
                1, 120, (g.H, fused_gu_cols), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
            shared_expert = SimpleNamespace(
                down_proj=_make_fp8_linear(g.IS, g.H),
            )
            shared_expert_gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((1, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=fused_w,
                _fused_gate_gu_fp8_scale=fused_s,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
            )
        else:
            gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((g.E, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            shared_expert = SimpleNamespace(
                gate_up_proj=_make_fp8_linear(g.H, 2 * g.IS),
                down_proj=_make_fp8_linear(g.IS, g.H, flipped=False),
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


def test_extract_kn_fp8_weight_square_orientation() -> None:
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
    s_raw = torch.ones((d, 1), dtype=torch.float32)

    proj_flipped = SimpleNamespace(
        weight=torch.nn.Parameter(w_raw.clone(), requires_grad=False),
        weight_scale=torch.nn.Parameter(s_raw.clone(), requires_grad=False),
    )
    setattr(proj_flipped, WEIGHT_FLIPPED_ATTR, True)
    w_fl, _ = extract_kn_fp8_weight(proj_flipped, d, d)
    assert torch.equal(w_fl.view(torch.uint8), w_raw.view(torch.uint8))

    proj_unflipped = SimpleNamespace(
        weight=torch.nn.Parameter(w_raw.clone(), requires_grad=False),
        weight_scale=torch.nn.Parameter(s_raw.clone(), requires_grad=False),
    )
    setattr(proj_unflipped, WEIGHT_FLIPPED_ATTR, False)
    w_unfl, _ = extract_kn_fp8_weight(proj_unflipped, d, d)
    assert torch.equal(
        w_unfl.view(torch.uint8), w_raw.transpose(0, 1).contiguous().view(torch.uint8)
    )


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
def test_pack_weights_shapes_base_buffers_and_expert_aliasing(
    stack_e: bool, fused_mlp: bool
) -> None:
    """Verify `pack_weights` shapes, base buffers, and `ecopy` extraction."""
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
    packed_no_ecopy = pack_weights(backbone_no_ecopy, g, ex, tp_rank=1)
    plan: planner.Plan = packed_no_ecopy["plan"]
    ecopy_off, ecopy_w = plan.smalls_layout["ecopy"]
    ecopy_zero = packed_no_ecopy["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w]
    assert torch.count_nonzero(ecopy_zero).item() == 0

    # 2. With explicit `_ecopy_cores`, `extract_ecopy_layer` encodes the placement.
    e_cores = qwen35.default_eplb_cores(g, ex.R)
    backbone = make_synthetic_qwen35_backbone(
        g, ex, fused_mlp=fused_mlp, fused_qkv=not fused_mlp, e_cores=e_cores
    )
    packed = pack_weights(backbone, g, ex, tp_rank=1)
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


def test_pack_weights_value_parity_against_split_hf_weights() -> None:
    """Verify `pack_weights` matches `split_hf_weights` + `pack_inputs`."""
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
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    hf = qwen35.random_hf_weights(jax.random.key(314), g)
    e_cores = qwen35.default_eplb_cores(g, ex.R)
    w_logical = qwen35.split_hf_weights(hf, g, ex, e_cores=e_cores)
    pos_j, hid_l, _, st_l = qwen35.random_logical_inputs(jax.random.key(315), g, ex)
    _, _, w_mk_ref, _ = qwen35.pack_inputs(
        pos_j, hid_l, w_logical, st_l, g, ex, build_default_pool=True
    )

    tp_rank = 1
    gdn_lay = GDNBlock.lin_in_layout(g, ex)
    qkvz_dim = gdn_lay.QKV + gdn_lay.Z
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
            w_b = w_lin_full[:, gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            w_a = w_lin_full[:, gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            w_ba = torch.cat([w_b, w_a], dim=1).contiguous()
            s_b = s_lin_full[gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            s_a = s_lin_full[gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            s_ba = torch.cat([s_b, s_a], dim=0).contiguous()

            in_proj_qkvz = SimpleNamespace(
                weight=torch.nn.Parameter(w_qkvz, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_qkvz, requires_grad=False),
            )
            setattr(in_proj_qkvz, WEIGHT_FLIPPED_ATTR, True)
            in_proj_ba = SimpleNamespace(
                weight=torch.nn.Parameter(w_ba, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_ba, requires_grad=False),
            )
            setattr(in_proj_ba, WEIGHT_FLIPPED_ATTR, True)
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
        gu_proj = SimpleNamespace(
            weight=torch.nn.Parameter(w_gu_unsharded, requires_grad=False),
            weight_scale=torch.nn.Parameter(s_gu_unsharded, requires_grad=False),
        )
        setattr(gu_proj, WEIGHT_FLIPPED_ATTR, True)
        w_d_unsharded = torch.cat(
            [
                _jax_to_torch(w_logical["sh2"][0, l_idx]),
                _jax_to_torch(w_logical["sh2"][1, l_idx]),
            ],
            dim=0,
        ).contiguous()
        down_proj = SimpleNamespace(
            weight=torch.nn.Parameter(w_d_unsharded, requires_grad=False),
            weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh2_s"][tp_rank, l_idx]), requires_grad=False
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
            shared_expert=SimpleNamespace(gate_up_proj=gu_proj, down_proj=down_proj),
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
    packed = pack_weights(backbone, g, ex, tp_rank=tp_rank)
    plan: planner.Plan = packed["plan"]

    for k in plan.w_names:
        val_t = packed[k]
        ref_j = w_mk_ref[k]
        if k in plan.dynamic_kinds:
            for l_idx in range(g.L):
                ref_l = np.asarray(ref_j[l_idx])[tp_rank]
                # `pack_inputs` tiles `[EW, NT, TK, N]`; unstacked keeps `[EW, K, N]`
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


def test_pack_weights_does_not_touch_jax_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pack_weights` with `tiles=None` never calls `jax.devices()`."""
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
            AssertionError("jax.devices() must not be called during pack_weights")
        ),
    )
    packed = pack_weights(backbone, g, ex, tp_rank=1)
    assert set(packed["plan"].w_names) <= set(packed)


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
