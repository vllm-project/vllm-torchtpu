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
"""Standalone unit tests for slice L-10a1 (`weights/{eplb,ops,spec_pack}`)."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeAltMixerModule,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
)
from vllm_torchtpu.megakernel.decls import HFParamDecl, ParamTransform, VLLMAlias
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.norm import NormModule
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32
from vllm_torchtpu.megakernel.planner import plan, weight_shapes
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec, SwitchSlot
from vllm_torchtpu.megakernel.weights.eplb import (
    default_eplb_cores,
    encode_eplb_ecopy,
    resolve_mapping_ecopy,
)
from vllm_torchtpu.megakernel.weights.ops import (
    JaxArrayOps,
    apply_decl_pad,
    slice_alias_targets,
    target_axis_widths,
)
from vllm_torchtpu.megakernel.weights.spec_pack import (
    pack_spec_logical_weights,
    switch_branch_for_layer,
)


def test_eplb_cores_and_ecopy_encoding():
    """Verify `default_eplb_cores`, `encode_eplb_ecopy`, and `resolve_mapping_ecopy`."""
    g_ep = FakeGeometry(L=2, E=16, TOPK=2, moe_tp=False, e_slots=6)
    cores = default_eplb_cores(g_ep, R=4)
    assert len(cores) == 2 and len(cores[0]) == 4 and len(cores[0][0]) == 6
    # Rank 0 primary slots are 0..3; redundant slots come from other ranks
    assert cores[0][0][:4] == [0, 1, 2, 3]
    assert all(e >= 4 for e in cores[0][0][4:])

    ec = encode_eplb_ecopy(g_ep, R=4)
    assert ec.shape == (2, 16) and ec.dtype == F32
    assert jnp.any(ec > 0)

    # Fallback when moe_tp=True or R=1 pads extra slots with -1 (encoded as 0)
    g_tp = dataclasses.replace(g_ep, moe_tp=True, e_slots=18)
    cores_tp = default_eplb_cores(g_tp, R=4)
    assert cores_tp[0][0][16:] == [-1, -1]
    ec_tp = encode_eplb_ecopy(g_tp, R=4, e_cores=cores_tp)
    assert jnp.all(ec_tp == 0)

    # `resolve_mapping_ecopy` from `ecopy`, `e_cores`, or empty mapping
    assert resolve_mapping_ecopy({}, None, g_ep, R=4) is None
    np.testing.assert_array_equal(
        resolve_mapping_ecopy({"ecopy": ec}, None, g_ep, R=4), ec
    )
    np.testing.assert_array_equal(
        resolve_mapping_ecopy({"ecopy": ec}, 1, g_ep, R=4), ec[1]
    )
    np.testing.assert_array_equal(
        resolve_mapping_ecopy({"e_cores": cores}, 0, g_ep, R=4), ec[0]
    )


def test_jax_array_ops_and_decl_helpers():
    """Verify `JaxArrayOps` methods, `target_axis_widths`, and `apply_decl_pad`."""
    ops = JaxArrayOps()
    x = jnp.arange(12, dtype=F32).reshape(3, 4)
    np.testing.assert_array_equal(ops.slice_axis(x, 1, 1, 3), x[:, 1:3])
    np.testing.assert_array_equal(ops.concat([x[:, :2], x[:, 2:]], axis=1), x)
    assert ops.reshape(x, (2, 6)).shape == (2, 6)
    assert ops.transpose(x, (1, 0)).shape == (4, 3)
    assert ops.pad(x, [(0, 0), (0, 0)]) is x
    assert ops.pad(x, [(0, 2)]).shape == (3, 6)
    assert ops.cast(x, F32) is x
    assert ops.cast(x, BF16).dtype == BF16
    assert not ops.is_fp8(x)

    # Quantize / dequantize with a zero column
    w_in = x.at[:, 0].set(0.0)
    w_fp8, scale = ops.quantize_fp8(w_in)
    assert ops.is_fp8(w_fp8) and w_fp8.shape == (3, 4) and scale.shape == (4,)
    assert float(scale[0]) == 1.0
    w_deq = ops.dequantize(w_fp8, scale[None, :], BF16)
    assert w_deq.dtype == BF16 and w_deq.shape == (3, 4)

    z = ops.zeros((2, 3), I32)
    assert z.shape == (2, 3) and z.dtype == I32
    st = ops.stack_layers([z, z])
    assert st.shape == (2, 2, 3)
    assert ops.own(st).shape == (2, 2, 3)

    # `target_axis_widths` and `apply_decl_pad`
    d_sh = HFParamDecl(
        "w",
        "hf.w",
        (16, 32),
        FP8,
        transform=ParamTransform(shard_axis=1, pad=((0, 24),)),
    )
    assert target_axis_widths(d_sh, R=4, split_axis=-1) == (32, 8)
    assert target_axis_widths(d_sh, R=4, split_axis=0) == (16, 16)
    padded = apply_decl_pad(ops, jnp.zeros((16, 8), FP8), d_sh)
    assert padded.shape == (16, 24)

    d_kv = HFParamDecl(
        "k", "hf.k", (2, 16, 8), FP8, transform=ParamTransform(kv_repeat=True)
    )
    assert target_axis_widths(d_kv, R=4, split_axis=-1) == (16, 8)
    assert target_axis_widths(d_kv, R=4, split_axis=0) == (16, 16)


def test_slice_alias_targets_quant_fastpath_and_dequant():
    """Verify `slice_alias_targets` across quantization, fast-path, and dequant."""
    ops = JaxArrayOps()
    d_q = HFParamDecl(
        "q",
        "hf.q",
        (8, 16),
        FP8,
        transform=ParamTransform(shard_axis=1, fuse_into="att_in"),
    )
    d_k = HFParamDecl(
        "k",
        "hf.k",
        (8, 8),
        FP8,
        transform=ParamTransform(shard_axis=1, fuse_into="att_in"),
    )
    s_q = HFParamDecl(
        "q_s",
        "hf.q_s",
        (16,),
        F32,
        transform=ParamTransform(shard_axis=0, scale_for="q", fuse_into="att_in_s"),
    )
    s_k = HFParamDecl(
        "k_s",
        "hf.k_s",
        (8,),
        F32,
        transform=ParamTransform(shard_axis=0, scale_for="k", fuse_into="att_in_s"),
    )
    scale_decls = {"q_s": s_q, "k_s": s_k}
    fuse_groups = {"att_in": ("q", "k"), "att_in_s": ("q_s", "k_s")}

    # 1. Unsharded BF16 -> per-target slice + on-the-fly FP8 quantization
    alias_bf16 = VLLMAlias(
        "qkv",
        ("q", "k"),
        already_sharded=False,
        already_fp8=False,
        scale_targets=("q_s", "k_s"),
    )
    w_bf16 = jnp.ones((8, 24), dtype=BF16)
    res_q = slice_alias_targets(
        ops, alias_bf16, w_bf16, None, (d_q, d_k), scale_decls, R=2, tp_rank=1
    )
    assert res_q["q"].shape == (8, 8) and res_q["q"].dtype == FP8
    assert res_q["k"].shape == (8, 4) and res_q["k_s"].shape == (4,)

    # 2. Already-sharded FP8 fast path returns fused group directly
    alias_fp8 = VLLMAlias(
        "qkv",
        ("q", "k"),
        already_sharded=True,
        already_fp8=True,
        scale_targets=("q_s", "k_s"),
    )
    w_fp8 = jnp.ones((8, 12), dtype=FP8)
    s_fp8 = jnp.ones((12,), dtype=F32)
    res_fast = slice_alias_targets(
        ops,
        alias_fp8,
        w_fp8,
        s_fp8,
        (d_q, d_k),
        scale_decls,
        R=2,
        tp_rank=0,
        fuse_groups=fuse_groups,
    )
    assert set(res_fast) == {"att_in", "att_in_s"}
    assert res_fast["att_in"] is w_fp8

    # 3. FP8 source dequantized to BF16 target + `flatten_tail=True`
    d_flat = HFParamDecl(
        "conv",
        "hf.conv",
        (4, 8),
        BF16,
        transform=ParamTransform(flatten_tail=True),
    )
    alias_conv = VLLMAlias("conv", ("conv",), already_sharded=True, already_fp8=False)
    w_conv_3d = jnp.ones((8, 1, 4), dtype=F32)
    res_conv = slice_alias_targets(
        ops, alias_conv, w_conv_3d, None, (d_flat,), {}, R=1, tp_rank=0
    )
    assert res_conv["conv"].shape == (32,) and res_conv["conv"].dtype == BF16

    d_deq = HFParamDecl("w_bf16", "hf.w", (8, 4), BF16)
    alias_deq = VLLMAlias("w", ("w_bf16",), already_sharded=True, already_fp8=True)
    res_deq = slice_alias_targets(
        ops,
        alias_deq,
        jnp.ones((8, 4), dtype=FP8),
        jnp.full((4,), 2.0, dtype=F32),
        (d_deq,),
        {},
        R=1,
        tp_rank=0,
    )
    assert res_deq["w_bf16"].dtype == BF16
    np.testing.assert_allclose(res_deq["w_bf16"], 2.0)

    # Error checks
    with pytest.raises(ValueError, match="declared already_fp8=False"):
        slice_alias_targets(
            ops, alias_bf16, w_fp8, None, (d_q, d_k), scale_decls, R=2, tp_rank=0
        )
    with pytest.raises(ValueError, match="width .* != expected"):
        slice_alias_targets(
            ops,
            alias_fp8,
            jnp.ones((8, 10), dtype=FP8),
            s_fp8,
            (d_q, d_k),
            scale_decls,
            R=2,
            tp_rank=0,
        )


def test_switch_branch_and_pack_spec_logical_weights():
    """Verify `switch_branch_for_layer` and `pack_spec_logical_weights`."""
    gdn = FakeAltMixerModule()
    gqa = FakeMixerModule()
    ffn = FakeFFNModule()
    slot = SwitchSlot("mixer", branches=(gdn, gqa), branch_counts=(2, 1))
    assert [switch_branch_for_layer(slot, i) for i in range(4)] == [
        (gdn, 0),
        (gdn, 1),
        (gqa, 0),
        (gdn, 2),
    ]

    R = 2
    g = FakeGeometry(
        L=2, att_every=2, H=256, NH=4, NKV=2, I=128, IS=256, E=16, TOPK=2, e_slots=10
    )
    sw = SwitchSlot("mixer", branches=(gdn, gqa), branch_counts=(1, 1))
    pre_norm = NormModule("input_layernorm")
    post_norm = NormModule("post_attention_layernorm")
    final_norm = NormModule("model.norm")
    ms = ModelSpec(
        "pack_test",
        config=g,
        layer_type=LayerType("lt", slots=(sw, ffn)),
    )
    ex = ExecSpec(R=R, B=4, T=1, stack_e=True, tiles=TINY_TILES)
    p = plan(ms, ex)
    wsh = weight_shapes(ms, ex, p)

    nh, hd = g.NH // R, g.HD
    nai = (2 * nh + 2) * hd
    kh, vh = g.KH // R, g.VH // R
    qkv, z = 2 * kh * g.KD + vh * g.VD, vh * g.VD
    nli = qkv + z + 128
    isr = g.IS // R
    logical = {
        "in_norm": jnp.ones((g.L, g.H), dtype=F32),
        "post_norm": jnp.ones((R, g.L, g.H), dtype=F32),
        "final_norm": jnp.ones((g.H,), dtype=F32),
        "in_s": jnp.ones((R, g.L, max(nai, nli)), dtype=F32),
        "out_s": jnp.ones((g.L, g.H), dtype=F32),
        "qn": jnp.ones((1, hd), dtype=F32),
        "kn": jnp.ones((R, 1, hd), dtype=F32),
        "conv": jnp.ones((R, 1, 4 * qkv), dtype=F32),
        "gnorm": jnp.ones((1, g.VD), dtype=F32),
        "sh13_s": jnp.ones((R, g.L, 2 * isr), dtype=F32),
        "sh2_s": jnp.ones((R, g.L, g.H), dtype=F32),
        "sh_gate": jnp.ones((g.L, g.H), dtype=F32),
        "e_cores": default_eplb_cores(g, R),
        "att_in": jnp.zeros((1, g.H, nai), dtype=FP8),
        "att_out": jnp.zeros((R, 1, nh * hd, g.H), dtype=FP8),
        "lin_in": jnp.zeros((R, 1, g.H, nli), dtype=FP8),
        "lin_out": jnp.zeros((R, 1, z, g.H), dtype=FP8),
        "router": jnp.zeros((R, g.L, g.H, g.E), dtype=BF16),
        "sh13": jnp.zeros((R, g.L, g.H, 2 * isr), dtype=FP8),
        "sh2": jnp.zeros((R, g.L, isr, g.H), dtype=FP8),
        "e13": jnp.zeros((R, g.L, g.EW(R), g.H, 2 * g.I), dtype=FP8),
        "e2": jnp.zeros((R, g.L, g.EW(R), g.I, g.H), dtype=FP8),
        "es": jnp.zeros((R, g.L, g.EW(R), 1, 2 * g.I + g.H), dtype=F32),
    }

    packed = pack_spec_logical_weights(
        ms,
        ex,
        logical,
        pre_norm=pre_norm,
        post_norm=post_norm,
        final_norm=final_norm,
    )
    for k in p.w_names:
        assert packed[k].shape == (R, *wsh[k][0]), (k, packed[k].shape, wsh[k][0])
        assert packed[k].dtype == wsh[k][1]

    packed_merged = pack_spec_logical_weights(
        ms,
        ex,
        logical,
        pre_norm=pre_norm,
        post_norm=post_norm,
        final_norm=final_norm,
        merged_tiles=True,
    )
    assert packed_merged["es"].shape == (R, g.L, g.EW(R), 2 * g.I + g.H)

    bad_logical = dict(logical)
    bad_logical.pop("sh_gate")
    with pytest.raises(ValueError, match="Missing logical weight 'sh_gate'"):
        pack_spec_logical_weights(
            ms, ex, bad_logical, pre_norm=pre_norm, post_norm=post_norm
        )
