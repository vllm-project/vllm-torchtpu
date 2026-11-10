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
"""Unit tests for `weights/{eplb,ops,spec_pack}`."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeAltMixerModule,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
)
from vllm_torchtpu.megakernel.core.decls import (
    HFParamDecl,
    ParamTransform,
    ResolvedModuleGeometry,
    VLLMAlias,
)
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.norm import EmbeddingModule, NormModule
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8, I32
from vllm_torchtpu.megakernel.core.planner import plan, weight_shapes
from vllm_torchtpu.megakernel.core.spec import (
    LayerType,
    ModelSpec,
    SwitchSlot,
    WeightDecl,
)
from vllm_torchtpu.megakernel.core.weights.eplb import (
    default_eplb_cores,
    encode_eplb_ecopy,
    resolve_mapping_ecopy,
)
from vllm_torchtpu.megakernel.core.weights.ops import (
    JaxArrayOps,
    apply_decl_pad,
    slice_alias_targets,
    target_axis_widths,
)
from vllm_torchtpu.megakernel.core.weights.spec_pack import (
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
    with pytest.raises(ValueError, match="kv_repeat nkv=2 does not divide R=3"):
        target_axis_widths(d_kv, R=3, split_axis=-1)
    d_kv6 = HFParamDecl(
        "k6", "hf.k6", (6, 16, 8), FP8, transform=ParamTransform(kv_repeat=True)
    )
    with pytest.raises(ValueError, match="kv_repeat nkv=6 is not divisible by R=4"):
        target_axis_widths(d_kv6, R=4, split_axis=-1)
    with pytest.raises(ValueError, match="exceeds target_width=24"):
        apply_decl_pad(ops, jnp.zeros((16, 28), FP8), d_sh)


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

    # 4. `EmbeddingModule` (`already_sharded=True, shard_axis=0`): exact shard
    # `(vocab // R, H)` passes; a padded `VocabParallelEmbedding` shard
    # `(num_embeddings_padded // R, H)` with `num_embeddings_padded // R > vocab // R`
    # or `vocab % R != 0` declines by name.
    emb_mod = EmbeddingModule("model.embed_tokens")
    emb_decl = emb_mod.hf_params(
        g=ResolvedModuleGeometry(
            EmbeddingModule, "model.embed_tokens", {"vocab": 64, "H": 16}
        )
    )[0]
    emb_alias = emb_decl.vllm_aliases[0]
    res_emb = slice_alias_targets(
        ops,
        emb_alias,
        jnp.ones((32, 16), dtype=BF16),
        None,
        (emb_decl,),
        {},
        R=2,
        tp_rank=0,
    )
    assert res_emb["embed_tokens"].shape == (32, 16)
    with pytest.raises(
        ValueError,
        match=(
            r"Alias 'model\.embed_tokens\.weight' \(already_sharded=True\) "
            r"axis 0 width 40 != expected 32"
        ),
    ):
        slice_alias_targets(
            ops,
            emb_alias,
            jnp.ones((40, 16), dtype=BF16),
            None,
            (emb_decl,),
            {},
            R=2,
            tp_rank=0,
        )
    odd_emb_decl = emb_mod.hf_params(
        g=ResolvedModuleGeometry(
            EmbeddingModule, "model.embed_tokens", {"vocab": 65, "H": 16}
        )
    )[0]
    with pytest.raises(
        ValueError,
        match=r"HFParamDecl 'embed_tokens' axis 0 width 65 is not divisible by R=2",
    ):
        target_axis_widths(odd_emb_decl, R=2, split_axis=0)

    # 5. Unsharded `channel_splits` and positive trailing `shard_axis=1`
    d_cs = HFParamDecl(
        "qkvz",
        "hf.qkvz",
        (8, 24),
        FP8,
        transform=ParamTransform(channel_splits=(8, 16)),
    )
    s_cs = HFParamDecl(
        "qkvz_s",
        "hf.qkvz_s",
        (24,),
        F32,
        transform=ParamTransform(scale_for="qkvz", channel_splits=(8, 16)),
    )
    alias_cs = VLLMAlias(
        "in_proj_qkvz",
        ("qkvz",),
        scale_targets=("qkvz_s",),
        scale_source="in_proj_qkvz_s",
        shard_axis=1,
        already_sharded=False,
        already_fp8=True,
    )
    w_cs = jnp.arange(8 * 24, dtype=F32).reshape(8, 24).astype(FP8)
    s_cs_buf = jnp.arange(24, dtype=F32)
    for r in range(2):
        res_cs = slice_alias_targets(
            ops,
            alias_cs,
            w_cs,
            s_cs_buf,
            (d_cs,),
            {"qkvz_s": s_cs},
            R=2,
            tp_rank=r,
        )
        exp_w = jnp.concatenate(
            [w_cs[:, r * 4 : (r + 1) * 4], w_cs[:, 8 + r * 8 : 8 + (r + 1) * 8]],
            axis=1,
        )
        exp_s = jnp.concatenate(
            [s_cs_buf[r * 4 : (r + 1) * 4], s_cs_buf[8 + r * 8 : 8 + (r + 1) * 8]],
            axis=0,
        )
        np.testing.assert_array_equal(res_cs["qkvz"], exp_w)
        np.testing.assert_array_equal(res_cs["qkvz_s"], exp_s)


@pytest.mark.parametrize(
    ("nkv", "R"),
    [(2, 2), (2, 4), (2, 8), (4, 2)],
    ids=["nkv2_r2", "nkv2_r4_rep", "nkv2_r8_tp8_rep", "nkv4_r2_shard"],
)
def test_slice_alias_targets_kv_repeat_replication_and_sharding(nkv: int, R: int):
    """Verify `slice_alias_targets` replicates/shards `kv_repeat` heads across `R`."""
    ops = JaxArrayOps()
    H, HD = 8, 4
    NH = R * 2
    d_q = HFParamDecl(
        "q",
        "hf.q",
        (H, NH * HD),
        BF16,
        transform=ParamTransform(shard_axis=1),
    )
    d_k = HFParamDecl(
        "k",
        "hf.k",
        (nkv, H, HD),
        BF16,
        transform=ParamTransform(kv_repeat=True),
    )
    d_v = HFParamDecl(
        "v",
        "hf.v",
        (nkv, H, HD),
        BF16,
        transform=ParamTransform(kv_repeat=True),
    )
    alias_qkv = VLLMAlias(
        "qkv_proj",
        ("q", "k", "v"),
        shard_axis=-1,
        already_sharded=False,
        already_fp8=False,
    )
    q_full = jnp.arange(H * NH * HD, dtype=BF16).reshape(H, NH * HD)
    k_heads = [jnp.full((H, HD), float(10 + h), dtype=BF16) for h in range(nkv)]
    v_heads = [jnp.full((H, HD), float(20 + h), dtype=BF16) for h in range(nkv)]
    k_full = jnp.concatenate(k_heads, axis=-1)
    v_full = jnp.concatenate(v_heads, axis=-1)
    w_qkv = jnp.concatenate([q_full, k_full, v_full], axis=-1)

    for r in range(R):
        res = slice_alias_targets(
            ops, alias_qkv, w_qkv, None, (d_q, d_k, d_v), {}, R=R, tp_rank=r
        )
        if nkv <= R:
            kv_head = r // (R // nkv)
            np.testing.assert_array_equal(res["k"], k_heads[kv_head])
            np.testing.assert_array_equal(res["v"], v_heads[kv_head])
        else:
            heads_per_r = nkv // R
            exp_k = jnp.concatenate(
                k_heads[r * heads_per_r : (r + 1) * heads_per_r], axis=-1
            )
            exp_v = jnp.concatenate(
                v_heads[r * heads_per_r : (r + 1) * heads_per_r], axis=-1
            )
            np.testing.assert_array_equal(res["k"], exp_k)
            np.testing.assert_array_equal(res["v"], exp_v)


@pytest.mark.parametrize("stack_e", [True, False], ids=["stacked_e", "per_layer_e"])
def test_switch_branch_and_pack_spec_logical_weights(stack_e: bool):
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
    ex = ExecSpec(R=R, B=4, T=1, stack_e=stack_e, tiles=TINY_TILES)
    p = plan(ms, ex)
    wsh = weight_shapes(ms, ex, p)

    nh, hd = g.NH // R, g.HD
    nai = (2 * nh + 2) * hd
    kh, vh = g.KH // R, g.VH // R
    qkv, z = 2 * kh * g.KD + vh * g.VD, vh * g.VD
    nli = qkv + z + 128
    isr = g.IS // R
    in_s_w = max(nai, nli)
    ew = g.EW(R)
    logical: dict[str, Any] = {
        "in_norm": jnp.arange(g.L * g.H, dtype=F32).reshape(g.L, g.H) + 1.0,
        "post_norm": jnp.arange(R * g.L * g.H, dtype=F32).reshape(R, g.L, g.H) + 10.0,
        "final_norm": jnp.arange(g.H, dtype=F32) + 20.0,
        "in_s": jnp.arange(R * g.L * in_s_w, dtype=F32).reshape(R, g.L, in_s_w) + 30.0,
        "out_s": jnp.arange(g.L * g.H, dtype=F32).reshape(g.L, g.H) + 40.0,
        "qn": jnp.arange(hd, dtype=F32).reshape(1, hd) + 50.0,
        "kn": jnp.arange(R * hd, dtype=F32).reshape(R, 1, hd) + 60.0,
        "conv": jnp.arange(R * 4 * qkv, dtype=F32).reshape(R, 1, 4 * qkv) + 70.0,
        "gnorm": jnp.arange(g.VD, dtype=F32).reshape(1, g.VD) + 80.0,
        "sh13_s": jnp.arange(R * g.L * 2 * isr, dtype=F32).reshape(R, g.L, 2 * isr)
        + 90.0,
        "sh2_s": jnp.arange(R * g.L * g.H, dtype=F32).reshape(R, g.L, g.H) + 100.0,
        "sh_gate": jnp.arange(g.L * g.H, dtype=F32).reshape(g.L, g.H) + 110.0,
        "e_cores": default_eplb_cores(g, R),
        "att_in": jnp.zeros((1, g.H, nai), dtype=FP8),
        "att_out": jnp.zeros((R, 1, nh * hd, g.H), dtype=FP8),
        "lin_in": jnp.zeros((R, 1, g.H, nli), dtype=FP8),
        "lin_out": jnp.zeros((R, 1, z, g.H), dtype=FP8),
        "router": jnp.arange(R * g.L * g.H * g.E, dtype=F32)
        .reshape(R, g.L, g.H, g.E)
        .astype(BF16),
        "sh13": ((jnp.arange(R * g.L * g.H * 2 * isr, dtype=F32) % 7) - 3.0)
        .reshape(R, g.L, g.H, 2 * isr)
        .astype(FP8),
        "sh2": jnp.zeros((R, g.L, isr, g.H), dtype=FP8),
        # Omit the `R` axis on `e13` to test `R`-broadcasting on both `stacked`
        # and `per_layer` dynamic weights.
        "e13": ((jnp.arange(g.L * ew * g.H * 2 * g.I, dtype=F32) % 5) - 2.0)
        .reshape(g.L, ew, g.H, 2 * g.I)
        .astype(FP8),
        "e2": jnp.zeros((R, g.L, ew, g.I, g.H), dtype=FP8),
        "es": jnp.arange(R * g.L * ew * (2 * g.I + g.H), dtype=F32).reshape(
            R, g.L, ew, 1, 2 * g.I + g.H
        ),
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
        expected = wsh[k]
        if isinstance(expected, tuple) and hasattr(expected[0], "shape"):
            assert isinstance(packed[k], tuple) and len(packed[k]) == len(expected)
            for l_arr, spec_l in zip(packed[k], expected, strict=True):
                assert l_arr.shape == (R, *spec_l.shape)
                assert l_arr.dtype == spec_l.dtype
        else:
            assert packed[k].shape == (R, *expected[0]), (
                k,
                packed[k].shape,
                expected[0],
            )
            assert packed[k].dtype == expected[1]

    # Verify packed `smalls` values for a per-layer field (`post_norm`), `qn`,
    # and a SwitchSlot field (`in_s`), with gdn (lyr=0) and gqa (lyr=1) differing.
    o_pn, w_pn = p.smalls_layout["post_norm"]
    o_qn, w_qn = p.smalls_layout["qn"]
    o_ins, w_ins = p.smalls_layout["in_s"]
    for r in range(R):
        for lyr in range(g.L):
            np.testing.assert_array_equal(
                packed["smalls"][r, lyr, 0, o_pn : o_pn + w_pn],
                logical["post_norm"][r, lyr],
            )
            np.testing.assert_array_equal(
                packed["smalls"][r, lyr, 0, o_ins : o_ins + w_ins],
                logical["in_s"][r, lyr, :w_ins],
            )
        assert not jnp.array_equal(
            packed["smalls"][r, 0, 0, o_ins : o_ins + w_ins],
            packed["smalls"][r, 1, 0, o_ins : o_ins + w_ins],
        )
        np.testing.assert_array_equal(
            packed["smalls"][r, 1, 0, o_qn : o_qn + w_qn],
            logical["qn"][0],
        )
        np.testing.assert_array_equal(
            packed["smalls"][r, 0, 0, o_qn : o_qn + w_qn],
            jnp.zeros((w_qn,), dtype=F32),
        )
        assert not jnp.array_equal(
            packed["smalls"][r, 0, 0, o_qn : o_qn + w_qn],
            packed["smalls"][r, 1, 0, o_qn : o_qn + w_qn],
        )

    # Verify stacked (`sh13`) and stacked/per_layer (`e13`) streamed weights.
    np.testing.assert_array_equal(
        packed["sh13"].reshape(R, g.L, g.H, 2 * isr), logical["sh13"]
    )
    if stack_e:
        for r in range(R):
            np.testing.assert_array_equal(
                packed["e13"][r].reshape(g.L, ew, g.H, 2 * g.I), logical["e13"]
            )
    else:
        for lyr in range(g.L):
            for r in range(R):
                np.testing.assert_array_equal(
                    packed["e13"][lyr][r].reshape(ew, g.H, 2 * g.I),
                    logical["e13"][lyr],
                )

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

    # R=1 with 2 layers and an HBM `per_layer` weight whose input carries the R axis
    # `(1, 2, *lead_dims, K, N)`: element count equals `n_layers * per_layer_elems`,
    # so rank-based detection must slice along the layer axis `[:, i]` not `[i]`.
    class _PerLayerHBMFFN(FakeFFNModule):
        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [
                dataclasses.replace(wd, layout="per_layer") if wd.name == "es" else wd
                for wd in super().weights(g, ex)
            ]

    r1 = 1
    g_r1 = FakeGeometry(
        L=2,
        att_every=2,
        H=256,
        NH=2,
        NKV=2,
        KH=1,
        VH=1,
        I=128,
        IS=128,
        E=16,
        TOPK=2,
        e_slots=16,
    )
    ex_r1 = ExecSpec(R=r1, B=4, T=1, stack_e=stack_e, tiles=TINY_TILES)
    ms_r1 = ModelSpec(
        "pack_test_r1",
        config=g_r1,
        layer_type=LayerType("lt", slots=(sw, _PerLayerHBMFFN())),
    )
    nh1, hd1 = g_r1.NH // r1, g_r1.HD
    nai1 = (2 * nh1 + 2) * hd1
    kh1, vh1 = g_r1.KH // r1, g_r1.VH // r1
    qkv1, z1 = 2 * kh1 * g_r1.KD + vh1 * g_r1.VD, vh1 * g_r1.VD
    nli1 = qkv1 + z1 + 128
    isr1 = g_r1.IS // r1
    in_s_w1 = max(nai1, nli1)
    ew1 = g_r1.EW(r1)
    es_r1 = jnp.arange(r1 * g_r1.L * ew1 * (2 * g_r1.I + g_r1.H), dtype=F32).reshape(
        r1, g_r1.L, ew1, 1, 2 * g_r1.I + g_r1.H
    )
    logical_r1: dict[str, Any] = {
        "in_norm": jnp.ones((g_r1.L, g_r1.H), dtype=F32),
        "post_norm": jnp.ones((r1, g_r1.L, g_r1.H), dtype=F32),
        "final_norm": jnp.ones((g_r1.H,), dtype=F32),
        "in_s": jnp.ones((r1, g_r1.L, in_s_w1), dtype=F32),
        "out_s": jnp.ones((g_r1.L, g_r1.H), dtype=F32),
        "qn": jnp.ones((1, hd1), dtype=F32),
        "kn": jnp.ones((r1, 1, hd1), dtype=F32),
        "conv": jnp.ones((r1, 1, 4 * qkv1), dtype=F32),
        "gnorm": jnp.ones((1, g_r1.VD), dtype=F32),
        "sh13_s": jnp.ones((r1, g_r1.L, 2 * isr1), dtype=F32),
        "sh2_s": jnp.ones((r1, g_r1.L, g_r1.H), dtype=F32),
        "sh_gate": jnp.ones((g_r1.L, g_r1.H), dtype=F32),
        "att_in": jnp.zeros((1, g_r1.H, nai1), dtype=FP8),
        "att_out": jnp.zeros((r1, 1, nh1 * hd1, g_r1.H), dtype=FP8),
        "lin_in": jnp.zeros((r1, 1, g_r1.H, nli1), dtype=FP8),
        "lin_out": jnp.zeros((r1, 1, z1, g_r1.H), dtype=FP8),
        "router": jnp.zeros((r1, g_r1.L, g_r1.H, g_r1.E), dtype=BF16),
        "sh13": jnp.zeros((r1, g_r1.L, g_r1.H, 2 * isr1), dtype=FP8),
        "sh2": jnp.zeros((r1, g_r1.L, isr1, g_r1.H), dtype=FP8),
        "e13": jnp.zeros((r1, g_r1.L, ew1, g_r1.H, 2 * g_r1.I), dtype=FP8),
        "e2": jnp.zeros((r1, g_r1.L, ew1, g_r1.I, g_r1.H), dtype=FP8),
        "es": es_r1,
    }
    packed_r1 = pack_spec_logical_weights(
        ms_r1,
        ex_r1,
        logical_r1,
        pre_norm=pre_norm,
        post_norm=post_norm,
        final_norm=final_norm,
    )
    assert isinstance(packed_r1["es"], tuple) and len(packed_r1["es"]) == g_r1.L
    for lyr in range(g_r1.L):
        np.testing.assert_array_equal(packed_r1["es"][lyr], es_r1[:, lyr])

    bad_logical = dict(logical)
    bad_logical.pop("sh_gate")
    with pytest.raises(ValueError, match="Missing logical weight 'sh_gate'"):
        pack_spec_logical_weights(
            ms, ex, bad_logical, pre_norm=pre_norm, post_norm=post_norm
        )

    no_fn_logical = dict(logical)
    no_fn_logical.pop("final_norm")
    with pytest.raises(
        ValueError, match="Missing logical weight 'final_norm' for 'final_norm'"
    ):
        pack_spec_logical_weights(
            ms,
            ex,
            no_fn_logical,
            pre_norm=pre_norm,
            post_norm=post_norm,
            final_norm=final_norm,
        )
