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
"""Unit tests for `modules/moe/{params,decls,reference}`."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
)
from vllm_torchtpu.megakernel.decls import (
    ModuleBinding,
    ResolvedModuleGeometry,
)
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec
from vllm_torchtpu.megakernel.modules.moe.decls import (
    MOE_FIELD_ALIASES,
    MOE_GEOMETRY_FIELDS,
    MOE_GEOMETRY_METHODS,
    MOE_OPTIONS,
    MOE_PARAM_NAMES,
    MOE_WEIGHT_NAMES,
    moe_hf_params,
    resolve_moe_binding,
    validate_moe_options,
)
from vllm_torchtpu.megakernel.modules.moe.params import (
    MoEDebug,
    MoETunables,
    derive_moe_params,
)
from vllm_torchtpu.megakernel.modules.moe.reference import moe_reference
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, round_bf16
from vllm_torchtpu.megakernel.planner import plan
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


def test_moe_topk_packed_and_route_vec_compile_time_bounds():
    """Verify `MoETunables`, `MoEDebug`, and `derive_moe_params` compile-time checks."""
    field_names = {f.name for f in dataclasses.fields(MoETunables)}
    assert set(MoETunables.NUMERICS.keys()) == field_names
    dbg = MoEDebug()
    assert dbg.forced_routing is False and dbg.row_mask is False
    assert TINY_TILES.with_modules({"moe": MoETunables(pipe=False)}).module(
        "moe", MoETunables
    ) == MoETunables(pipe=False)
    assert DebugOptions(modules={"moe": MoEDebug(row_mask=True)}).modules[
        "moe"
    ] == MoEDebug(row_mask=True)

    base_tiles = dataclasses.replace(TINY_TILES, bank=512 << 10)
    g_ok = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        NH=16,
        NKV=8,
        I=256,
        IS=1024,
        E=512,
        TOPK=8,
        moe_tp=False,
        e_slots=72,
    )
    ex_ok = ExecSpec(R=8, B=4, T=1, tiles=base_tiles)
    ms_ok = ModelSpec(
        "moe_ok",
        config=g_ok,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    p_ok = plan(ms_ok, ex_ok)
    tt_ok = MoETunables(topk=True, route_vec=True)
    mp_ok = derive_moe_params(
        g_ok,
        ex_ok,
        p_ok,
        tt_ok,
        dbg,
        moe_tp=False,
        e_slots=72,
        module_name="fake_moe",
    )
    assert mp_ok.topk_packed is True and mp_ok.route_vec is True and mp_ok.ECOPY is True

    # Non-power-of-two E=384 <= 512 is valid for topk=True.
    g_e384 = dataclasses.replace(g_ok, E=384, e_slots=0)
    ms_e384 = ModelSpec(
        "moe_e384",
        config=g_e384,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    p_e384 = plan(ms_e384, ex_ok)
    mp_e384 = derive_moe_params(
        g_e384,
        ex_ok,
        p_e384,
        MoETunables(topk=True, route_vec=False),
        dbg,
        moe_tp=False,
        e_slots=0,
        module_name="fake_moe",
    )
    assert mp_e384.topk_packed is True

    # E > 512 overflows the 9-bit float32 mantissa pack when topk=True.
    for bad_e in (513, 1024):
        g_bad_topk = dataclasses.replace(g_ok, E=bad_e, e_slots=0)
        with pytest.raises(MegakernelConfigError, match="topk_packed"):
            derive_moe_params(
                g_bad_topk,
                ex_ok,
                p_ok,
                MoETunables(topk=True, route_vec=False),
                dbg,
                moe_tp=False,
                e_slots=0,
                module_name="fake_moe",
            )

    # `route_vec` resolves to False (quiet fallback) when preconditions are not met:
    # - `e_slots=80` (R*EXTRA+R+1 = 137 > 128)
    # - `e_slots=96` (R*EXTRA+R+1 = 265 > 128)
    # - `E=2048, B=4` (E // 128 + 1 = 17 > MR=8)
    # - `e_slots=0` (ecopy=False)
    tt_rvec = MoETunables(topk=False, route_vec=True)
    for fallback_g in (
        dataclasses.replace(g_ok, E=512, e_slots=80),
        dataclasses.replace(g_ok, E=512, e_slots=96),
        dataclasses.replace(g_ok, E=2048, e_slots=260),
        dataclasses.replace(g_ok, E=512, e_slots=0),
    ):
        mp_fb = derive_moe_params(
            fallback_g,
            ex_ok,
            p_ok,
            tt_rvec,
            dbg,
            moe_tp=False,
            e_slots=fallback_g.e_slots,
            module_name="fake_moe",
        )
        assert mp_fb.route_vec is False

    # Partial layer count or non-128-aligned H raises MegakernelConfigError
    p_partial = dataclasses.replace(p_ok, module_layer_counts={"fake_moe": 0})
    with pytest.raises(MegakernelConfigError, match="every layer"):
        derive_moe_params(
            g_ok,
            ex_ok,
            p_partial,
            tt_ok,
            dbg,
            moe_tp=False,
            e_slots=0,
            module_name="fake_moe",
        )
    with pytest.raises(MegakernelConfigError, match="multiple of 128"):
        derive_moe_params(
            dataclasses.replace(g_ok, H=192),
            ex_ok,
            p_ok,
            tt_ok,
            dbg,
            moe_tp=False,
            e_slots=0,
            module_name="fake_moe",
        )
    # row_mask with ecopy=True raises MegakernelConfigError
    with pytest.raises(MegakernelConfigError, match="identity expert placement"):
        derive_moe_params(
            g_ok,
            ex_ok,
            p_ok,
            tt_ok,
            MoEDebug(row_mask=True),
            moe_tp=False,
            e_slots=72,
            module_name="fake_moe",
        )
    # Invalid e_slots values raise MegakernelConfigError
    with pytest.raises(MegakernelConfigError, match="e_slots=-1 must be >= 0"):
        derive_moe_params(
            g_ok,
            ex_ok,
            p_ok,
            tt_ok,
            dbg,
            moe_tp=False,
            e_slots=-1,
            module_name="fake_moe",
        )
    with pytest.raises(MegakernelConfigError, match="must be 0 or >= ER=64"):
        derive_moe_params(
            g_ok,
            ex_ok,
            p_ok,
            tt_ok,
            dbg,
            moe_tp=False,
            e_slots=32,
            module_name="fake_moe",
        )
    with pytest.raises(
        MegakernelConfigError, match="moe_tp=True does not support redundant"
    ):
        derive_moe_params(
            g_ok,
            ex_ok,
            p_ok,
            tt_ok,
            dbg,
            moe_tp=True,
            e_slots=72,
            module_name="fake_moe",
        )


def test_moe_decls_options_binding_and_hf_params():
    """Verify MoE options, binding resolution, geometry methods, and HF params."""
    assert MOE_WEIGHT_NAMES == ("sh13", "sh2", "e13", "e2")
    assert "router" in MOE_PARAM_NAMES and "e_gate_proj" in MOE_PARAM_NAMES
    assert "H" in MOE_FIELD_ALIASES and "experts" in MOE_GEOMETRY_FIELDS
    assert MOE_OPTIONS["shared_gate"] is True

    validate_moe_options(
        {
            "shared_gate": True,
            "weight_dtypes": {"sh13": "bf16", "sh2": "fp8", "e13": "fp8", "e2": "fp8"},
        }
    )
    with pytest.raises(MegakernelConfigError, match="shared_gate=True"):
        validate_moe_options({"shared_gate": False})
    with pytest.raises(MegakernelConfigError, match="same weight dtype"):
        validate_moe_options({"weight_dtypes": {"e13": "bf16", "e2": "fp8"}})
    with pytest.raises(MegakernelConfigError, match="weight_dtypes"):
        validate_moe_options({"weight_dtypes": {"bad": "bf16"}})

    binding = ModuleBinding(
        module_cls=FakeFFNModule,
        prefix="mlp",
        options={"moe_tp": False, "e_slots": 72},
        rename={"router": "gate_router"},
    )
    raw_res = ResolvedModuleGeometry(
        module_cls=FakeFFNModule,
        prefix="mlp",
        values={
            "hidden": 256,
            "experts": 512,
            "top_k": 8,
            "inter": 256,
            "shared_inter": 1024,
            "eps": 1e-6,
        },
        options=binding.options,
        rename=binding.rename,
    )
    res = resolve_moe_binding(FakeFFNModule, binding, raw_res)
    assert res.H == 256 and res.E == 512 and res.TOPK == 8
    assert res.I == 256 and res.IS == 1024
    assert res.moe_tp is False and res.e_slots == 72
    assert MOE_GEOMETRY_METHODS["ER"](res, 8) == 64
    assert MOE_GEOMETRY_METHODS["ISR"](res, 8) == 128
    assert MOE_GEOMETRY_METHODS["IR"](res, 8) == 256
    assert MOE_GEOMETRY_METHODS["EW"](res, 8) == 72

    decls_fp8 = moe_hf_params("mlp", res)
    names_fp8 = {d.name for d in decls_fp8}
    assert "gate_router" in names_fp8 and "sh_gate" in names_fp8
    assert "e_gate_proj_s" in names_fp8 and "e2_s" in names_fp8

    decls_bf16 = moe_hf_params(
        "mlp",
        res,
        moe_tp=True,
        weight_dtypes={"sh13": "bf16", "sh2": "bf16", "e13": "bf16", "e2": "bf16"},
    )
    names_bf16 = {d.name for d in decls_bf16}
    assert "e_gate_proj_s" not in names_bf16 and "sh2_s" not in names_bf16


@pytest.mark.parametrize(
    ("L", "layer_idx"), [(1, 0), (2, 1)], ids=["single_l0", "stacked_l2_idx1"]
)
@pytest.mark.parametrize("w_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("moe_tp", [True, False], ids=["tp", "ep"])
def test_moe_reference_matches_direct_computation(
    moe_tp: bool, w_dtype: str, L: int, layer_idx: int
):
    """Compare `moe_reference` against unsharded dense top-k MoE + shared expert."""
    R, B, H, E, TOPK, IR, ISR = 2, 4, 128, 16, 2, 128, 128
    ER = E // R
    I = IR * R if moe_tp else IR  # noqa: E741
    IS = ISR * R
    g = FakeGeometry(
        L=L,
        att_every=1,
        H=H,
        E=E,
        TOPK=TOPK,
        I=I,
        IS=IS,
        moe_tp=moe_tp,
    )
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=TINY_TILES)
    is_fp8 = w_dtype == "fp8"
    dt = FP8 if is_fp8 else BF16

    keys = jax.random.split(
        jax.random.PRNGKey(90 + int(moe_tp) + 2 * int(is_fp8) + 4 * L), 13
    )
    x = jax.random.normal(keys[0], (B, H), dtype=BF16) * 0.3
    router_u = jax.random.normal(keys[1], (L, H, E), dtype=BF16) * 0.1
    sh_gate_u = jax.random.normal(keys[2], (L, H), dtype=F32) * 0.2
    sh_g_u = (jax.random.normal(keys[3], (L, H, IS), dtype=F32) * 0.3).astype(dt)
    sh_u_u = (jax.random.normal(keys[4], (L, H, IS), dtype=F32) * 0.3).astype(dt)
    sh_g_s_u = jax.random.uniform(keys[5], (L, IS), dtype=F32, minval=0.05, maxval=0.15)
    sh_u_s_u = jax.random.uniform(keys[6], (L, IS), dtype=F32, minval=0.05, maxval=0.15)
    sh2_u = (jax.random.normal(keys[7], (L, IS, H), dtype=F32) * 0.3).astype(dt)
    sh2_s_u = jax.random.uniform(keys[8], (L, H), dtype=F32, minval=0.05, maxval=0.15)
    eg_u = (jax.random.normal(keys[9], (L, E, H, I), dtype=F32) * 0.3).astype(dt)
    eu_u = (jax.random.normal(keys[10], (L, E, H, I), dtype=F32) * 0.3).astype(dt)
    eg_s_u = jax.random.uniform(
        keys[11], (L, E, I), dtype=F32, minval=0.05, maxval=0.15
    )
    eu_s_u = jax.random.uniform(
        keys[12], (L, E, I), dtype=F32, minval=0.05, maxval=0.15
    )
    e2_u = (
        jax.random.normal(jax.random.PRNGKey(133 + L), (L, E, I, H), dtype=F32) * 0.3
    ).astype(dt)
    e2_s_u = jax.random.uniform(
        jax.random.PRNGKey(144 + L), (L, E, H), dtype=F32, minval=0.05, maxval=0.15
    )

    # Shard unsharded weights into per-rank shapes expected by `moe_reference`
    sh13_ranks = []
    sh13_s_ranks = []
    sh2_ranks = []
    sh2_s_ranks = []
    e13_ranks = []
    e13_s_ranks = []
    e2_ranks = []
    e2_s_ranks = []
    for r in range(R):
        c0, c1 = r * ISR, (r + 1) * ISR
        sh13_ranks.append(
            jnp.concatenate([sh_g_u[:, :, c0:c1], sh_u_u[:, :, c0:c1]], axis=-1)
        )
        sh13_s_ranks.append(
            jnp.concatenate([sh_g_s_u[:, c0:c1], sh_u_s_u[:, c0:c1]], axis=-1)
        )
        sh2_ranks.append(sh2_u[:, c0:c1, :])
        sh2_s_ranks.append(sh2_s_u)
        if moe_tp:
            ic0, ic1 = r * IR, (r + 1) * IR
            e13_ranks.append(
                jnp.concatenate(
                    [eg_u[:, :, :, ic0:ic1], eu_u[:, :, :, ic0:ic1]], axis=-1
                )
            )
            e13_s_ranks.append(
                jnp.concatenate([eg_s_u[:, :, ic0:ic1], eu_s_u[:, :, ic0:ic1]], axis=-1)
            )
            e2_ranks.append(e2_u[:, :, ic0:ic1, :])
            e2_s_ranks.append(e2_s_u)
        else:
            e0, e1 = r * ER, (r + 1) * ER
            e13_ranks.append(jnp.concatenate([eg_u[:, e0:e1], eu_u[:, e0:e1]], axis=-1))
            e13_s_ranks.append(
                jnp.concatenate([eg_s_u[:, e0:e1], eu_s_u[:, e0:e1]], axis=-1)
            )
            e2_ranks.append(e2_u[:, e0:e1])
            e2_s_ranks.append(e2_s_u[:, e0:e1])

    sh13 = jnp.stack(sh13_ranks, axis=0)
    sh13_s = jnp.stack(sh13_s_ranks, axis=0)
    sh2 = jnp.stack(sh2_ranks, axis=0)
    sh2_s = jnp.stack(sh2_s_ranks, axis=0)
    e13 = jnp.stack(e13_ranks, axis=0)
    e13_s = jnp.stack(e13_s_ranks, axis=0)
    e2 = jnp.stack(e2_ranks, axis=0)
    e2_s = jnp.stack(e2_s_ranks, axis=0)

    if L == 1:
        params: dict[str, jax.Array] = {
            "router": router_u[0],
            "sh_gate": jnp.broadcast_to(sh_gate_u[0][None, :], (R, H)),
            "sh13": sh13[:, 0],
            "sh2": sh2[:, 0],
            "e13": e13[:, 0],
            "e2": e2[:, 0],
        }
        if is_fp8:
            params.update(
                {
                    "sh13_s": sh13_s[:, 0],
                    "sh2_s": sh2_s[:, 0],
                    "e13_s": e13_s[:, 0],
                    "e2_s": e2_s[:, 0],
                }
            )
    else:
        # Pass 2-D `sh_gate` of shape `(L, H)` with `L == R == 2` to verify
        # multi-layer disambiguation at `layer_idx=1`.
        params = {
            "router": router_u,
            "sh_gate": sh_gate_u,
            "sh13": sh13,
            "sh2": sh2,
            "e13": e13,
            "e2": e2,
        }
        if is_fp8:
            params.update(
                {
                    "sh13_s": sh13_s,
                    "sh2_s": sh2_s,
                    "e13_s": e13_s,
                    "e2_s": e2_s,
                }
            )

    ref_delta, ref_st = moe_reference(
        params, x, state="sentinel", g=g, ex=ex, layer_idx=layer_idx
    )
    assert ref_st == "sentinel"

    # Independent unsharded dense top-k MoE + gated shared expert computation
    x_f32 = x.astype(F32)
    xb_f32 = x_f32.astype(BF16).astype(F32)
    probs = jax.nn.softmax(xb_f32 @ router_u[layer_idx].astype(F32), axis=-1)
    top_vals, top_idx = lax.top_k(probs, TOPK)
    top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    sg = jax.nn.sigmoid(
        jnp.sum(x_f32 * sh_gate_u[layer_idx][None, :], axis=-1, keepdims=True)
    )
    w_sh_g = (
        sh_g_u[layer_idx].astype(F32) * sh_g_s_u[layer_idx][None, :]
        if is_fp8
        else sh_g_u[layer_idx].astype(F32)
    )
    w_sh_u = (
        sh_u_u[layer_idx].astype(F32) * sh_u_s_u[layer_idx][None, :]
        if is_fp8
        else sh_u_u[layer_idx].astype(F32)
    )
    mid_sh_full = (
        round_bf16(
            round_bf16(jax.nn.silu(round_bf16(xb_f32 @ w_sh_g)))
            * round_bf16(xb_f32 @ w_sh_u)
        )
        .astype(BF16)
        .astype(F32)
    )
    w_sh2_full = (
        sh2_u[layer_idx].astype(F32) * sh2_s_u[layer_idx][None, :]
        if is_fp8
        else sh2_u[layer_idx].astype(F32)
    )
    w_eg_full = (
        eg_u[layer_idx].astype(F32) * eg_s_u[layer_idx][:, None, :]
        if is_fp8
        else eg_u[layer_idx].astype(F32)
    )
    w_eu_full = (
        eu_u[layer_idx].astype(F32) * eu_s_u[layer_idx][:, None, :]
        if is_fp8
        else eu_u[layer_idx].astype(F32)
    )
    w_e2_full = (
        e2_u[layer_idx].astype(F32) * e2_s_u[layer_idx][:, None, :]
        if is_fp8
        else e2_u[layer_idx].astype(F32)
    )
    mid_e_all = (
        round_bf16(
            round_bf16(
                jax.nn.silu(round_bf16(jnp.einsum("bh,ehi->bei", xb_f32, w_eg_full)))
            )
            * round_bf16(jnp.einsum("bh,ehi->bei", xb_f32, w_eu_full))
        )
        .astype(BF16)
        .astype(F32)
    )
    # Sum rank partial outputs (matching the per-rank all-reduce BF16 rounding)
    rank_totals = []
    for r in range(R):
        c0, c1 = r * ISR, (r + 1) * ISR
        ysh_r = (mid_sh_full[:, c0:c1] @ w_sh2_full[c0:c1, :]) * sg
        if moe_tp:
            ic0, ic1 = r * IR, (r + 1) * IR
            ye_all_r = round_bf16(
                jnp.einsum(
                    "bei,eih->beh", mid_e_all[:, :, ic0:ic1], w_e2_full[:, ic0:ic1, :]
                )
            )
            ye_topk_r = jnp.take_along_axis(ye_all_r, top_idx[:, :, None], axis=1)
            yexp_r = jnp.sum(top_w[:, :, None] * ye_topk_r, axis=1)
        else:
            ye_all_full = round_bf16(jnp.einsum("bei,eih->beh", mid_e_all, w_e2_full))
            ye_topk = jnp.take_along_axis(ye_all_full, top_idx[:, :, None], axis=1)
            in_rank = (top_idx // ER) == r
            yexp_r = jnp.sum(
                jnp.where(in_rank[:, :, None], top_w[:, :, None] * ye_topk, 0.0),
                axis=1,
            )
        rank_totals.append((ysh_r + yexp_r).astype(BF16).astype(F32))

    expected = jnp.sum(jnp.stack(rank_totals, axis=0), axis=0)
    np.testing.assert_allclose(ref_delta, expected, rtol=1e-5, atol=1e-5)


def test_moe_reference_rename_and_missing_scales_error():
    """Verify `moe_reference` resolves `g.rename` and rejects FP8 without scales."""
    R, B, H, E, TOPK, IR, ISR = 2, 4, 128, 16, 2, 128, 128
    base_g = FakeGeometry(
        L=1, att_every=1, H=H, E=E, TOPK=TOPK, I=IR * R, IS=ISR * R, moe_tp=True
    )
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=TINY_TILES)
    keys = jax.random.split(jax.random.PRNGKey(201), 8)
    x = jax.random.normal(keys[0], (B, H), dtype=BF16) * 0.3
    router = jax.random.normal(keys[1], (H, E), dtype=BF16) * 0.1
    sh_gate = jax.random.normal(keys[2], (H,), dtype=F32) * 0.2
    sh_g = jax.random.normal(keys[3], (R, H, ISR), dtype=BF16) * 0.2
    sh_u = jax.random.normal(keys[4], (R, H, ISR), dtype=BF16) * 0.2
    sh2 = jax.random.normal(keys[5], (R, ISR, H), dtype=BF16) * 0.2
    eg = jax.random.normal(keys[6], (R, E, H, IR), dtype=BF16) * 0.2
    eu = jax.random.normal(keys[7], (R, E, H, IR), dtype=BF16) * 0.2
    e2 = jax.random.normal(jax.random.PRNGKey(202), (R, E, IR, H), dtype=BF16) * 0.2

    canon_params = {
        "router": router,
        "sh_gate": sh_gate,
        "sh13": jnp.concatenate([sh_g, sh_u], axis=-1),
        "sh2": sh2,
        "e13": jnp.concatenate([eg, eu], axis=-1),
        "e2": e2,
    }
    out_canon, _ = moe_reference(canon_params, x, g=base_g, ex=ex)

    rename = {
        "router": "gate_router",
        "sh_gate": "se_gate",
        "sh_gate_proj": "se_g",
        "sh_up_proj": "se_u",
        "sh2": "se_down",
        "e_gate_proj": "exp_g",
        "e_up_proj": "exp_u",
        "e2": "exp_down",
    }
    binding = ModuleBinding(
        module_cls=FakeFFNModule,
        prefix="mlp",
        options={"moe_tp": True, "e_slots": 0},
        rename=rename,
    )
    raw_res = ResolvedModuleGeometry(
        module_cls=FakeFFNModule,
        prefix="mlp",
        values={
            "hidden": H,
            "experts": E,
            "top_k": TOPK,
            "inter": IR * R,
            "shared_inter": ISR * R,
            "eps": 1e-6,
        },
        options=binding.options,
        rename=rename,
    )
    res_geom = resolve_moe_binding(FakeFFNModule, binding, raw_res)
    assert res_geom.zero_centered_norm is True
    renamed_params = {
        "gate_router": router,
        "se_gate": sh_gate,
        "se_g": sh_g,
        "se_u": sh_u,
        "se_down": sh2,
        "exp_g": eg,
        "exp_u": eu,
        "exp_down": e2,
    }
    out_renamed, _ = moe_reference(renamed_params, x, g=res_geom, ex=ex)
    np.testing.assert_array_equal(out_canon, out_renamed)

    with pytest.raises(MegakernelConfigError, match="requires 'sh13_s'"):
        moe_reference(
            {**canon_params, "sh13": canon_params["sh13"].astype(FP8)},
            x,
            g=base_g,
            ex=ex,
        )
