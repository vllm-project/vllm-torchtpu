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
"""Standalone unit tests for slice L-8a (`modules/moe/{params,decls,reference}`)."""

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
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
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


@pytest.mark.parametrize("moe_tp", [True, False], ids=["tp", "ep"])
def test_moe_reference_matches_direct_computation(moe_tp: bool):
    """Compare `moe_reference` against direct multi-rank router + shared + experts."""
    R, B, H, E, TOPK, IR, ISR = 2, 4, 128, 16, 2, 128, 128
    ER = E // R
    EW = E if moe_tp else ER
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        E=E,
        TOPK=TOPK,
        I=IR * R if moe_tp else IR,
        IS=ISR * R,
        moe_tp=moe_tp,
    )
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=TINY_TILES)

    keys = jax.random.split(jax.random.PRNGKey(90 + int(moe_tp)), 11)
    x = jax.random.normal(keys[0], (B, H), dtype=BF16) * 0.3
    router = jax.random.normal(keys[1], (H, E), dtype=BF16) * 0.1
    sh_gate = jax.random.normal(keys[2], (R, H), dtype=F32) * 0.2
    sh13 = (jax.random.normal(keys[3], (R, H, 2 * ISR), dtype=F32) * 0.3).astype(FP8)
    sh13_s = jax.random.uniform(
        keys[4], (R, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15
    )
    sh2 = (jax.random.normal(keys[5], (R, ISR, H), dtype=F32) * 0.3).astype(FP8)
    sh2_s = jax.random.uniform(keys[6], (R, H), dtype=F32, minval=0.05, maxval=0.15)
    e13 = (jax.random.normal(keys[7], (R, EW, H, 2 * IR), dtype=F32) * 0.3).astype(FP8)
    e13_s = jax.random.uniform(
        keys[8], (R, EW, 2 * IR), dtype=F32, minval=0.05, maxval=0.15
    )
    e2 = (jax.random.normal(keys[9], (R, EW, IR, H), dtype=F32) * 0.3).astype(FP8)
    e2_s = jax.random.uniform(keys[10], (R, EW, H), dtype=F32, minval=0.05, maxval=0.15)

    params = {
        "router": router,
        "sh_gate": sh_gate,
        "sh13": sh13,
        "sh13_s": sh13_s,
        "sh2": sh2,
        "sh2_s": sh2_s,
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    ref_delta, ref_st = moe_reference(params, x, state="sentinel", g=g, ex=ex)
    assert ref_st == "sentinel"

    x_moe = x.astype(F32)
    xb_moe = x_moe.astype(BF16).astype(F32)
    probs = jax.nn.softmax(xb_moe @ router.astype(F32), axis=-1)
    top_vals, top_idx = lax.top_k(probs, TOPK)
    top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    partials = []
    for r in range(R):
        sg = jax.nn.sigmoid(
            jnp.sum(x_moe * sh_gate[r][None, :], axis=-1, keepdims=True)
        )
        g13 = (xb_moe @ sh13[r].astype(F32)) * sh13_s[r]
        mid_sh = (
            round_bf16(
                round_bf16(jax.nn.silu(round_bf16(g13[:, :ISR])))
                * round_bf16(g13[:, ISR:])
            )
            .astype(BF16)
            .astype(F32)
        )
        part_r = (mid_sh @ sh2[r].astype(F32)) * sh2_s[r] * sg
        for b in range(B):
            for k in range(TOPK):
                e = int(top_idx[b, k])
                if not moe_tp and (e // ER) != r:
                    continue
                w_slot = e if moe_tp else (e - r * ER)
                h_e = (xb_moe[b : b + 1] @ e13[r, w_slot].astype(F32)) * e13_s[
                    r, w_slot
                ]
                mid_e = (
                    round_bf16(
                        round_bf16(jax.nn.silu(round_bf16(h_e[:, :IR])))
                        * round_bf16(h_e[:, IR:])
                    )
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = round_bf16((mid_e @ e2[r, w_slot].astype(F32)) * e2_s[r, w_slot])
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)
        partials.append(part_r.astype(BF16).astype(F32))

    expected = jnp.sum(jnp.stack(partials, axis=0), axis=0)
    np.testing.assert_allclose(ref_delta, expected, rtol=1e-5, atol=1e-5)
