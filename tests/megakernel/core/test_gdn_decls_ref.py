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
"""Unit tests for `modules/gdn` gate helpers, parameters, decls, and reference."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
)
from vllm_torchtpu.megakernel.core.decls import (
    ModuleBinding,
    RecurrentCacheDecl,
    ResolvedModuleGeometry,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.module import Module
from vllm_torchtpu.megakernel.core.modules.gdn.decls import (
    GDN_FIELD_ALIASES,
    GDN_GEOMETRY_FIELDS,
    GDN_GEOMETRY_METHODS,
    GDN_OPTIONS,
    GDN_PARAM_NAMES,
    GDN_WEIGHT_NAMES,
    bind_gdn_module,
    gdn_cache_decl,
    gdn_hf_params,
    resolve_gdn_binding,
    validate_gdn_options,
)
from vllm_torchtpu.megakernel.core.modules.gdn.gate import (
    GDN_GATE_ACTS,
    gdn_gate_act,
    validate_gdn_gate,
)
from vllm_torchtpu.megakernel.core.modules.gdn.params import (
    GDNTunables,
    checked_pool_layout,
    derive_params,
    gdn_lin_in_layout,
)
from vllm_torchtpu.megakernel.core.modules.gdn.reference import gdn_reference
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.core.planner import plan
from vllm_torchtpu.megakernel.core.pool import CacheExec, PoolBackend, PoolLayout
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec


def test_gdn_gate_act_and_validate_gdn_gate():
    """Verify `validate_gdn_gate` and `gdn_gate_act` for valid and invalid gates."""
    assert set(GDN_GATE_ACTS) == {"silu", "sigmoid"}
    assert validate_gdn_gate("silu") == "silu"
    assert validate_gdn_gate("sigmoid") == "sigmoid"
    assert gdn_gate_act("silu") is jax.nn.silu
    assert gdn_gate_act("sigmoid") is jax.nn.sigmoid

    x = jnp.array([-1.0, 0.0, 2.0], dtype=F32)
    np.testing.assert_allclose(gdn_gate_act("silu")(x), jax.nn.silu(x))
    np.testing.assert_allclose(gdn_gate_act("sigmoid")(x), jax.nn.sigmoid(x))

    bad_cases: tuple[Any, ...] = ("relu", "", None, 123)
    for bad in bad_cases:
        with pytest.raises(ValueError, match="'silu' or 'sigmoid'"):
            validate_gdn_gate(bad)
        with pytest.raises(MegakernelConfigError, match="'silu' or 'sigmoid'"):
            gdn_gate_act(bad, err_cls=MegakernelConfigError)


def test_gdn_tunables_lin_in_layout_and_derive_params():
    """Verify GDN tunables, `lin_in` layout, `derive_params`, and pool layout."""
    field_names = {f.name for f in dataclasses.fields(GDNTunables)}
    assert set(GDNTunables.NUMERICS.keys()) == field_names

    g = FakeGeometry(L=2, att_every=2, H=256, KH=4, VH=8, KD=128, VD=128)
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, wide_rows=4)
    ex = ExecSpec(B=4, T=1, R=2, ctx=256, stack_e=True, tiles=tiles)
    lay = gdn_lin_in_layout(g, ex)
    assert lay == gdn_lin_in_layout(g, 2)
    assert lay.kh == 2 and lay.vh == 4
    assert lay.QKV == 2 * 2 * 128 + 4 * 128
    assert lay.Z == 4 * 128
    assert lay.BA == lay.QKV + lay.Z and lay.A_OFF == lay.BA + 64
    assert lay.NLI == lay.BA + 128

    ms = ModelSpec(
        "gdn_p",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    p = plan(ms, ex)
    gp = derive_params(g, ex, p, GDNTunables(), "fake_gqa", gate="sigmoid")
    assert gp.GCH is False and gp.CB == 4 and gp.NCH == 1 and gp.gate == "sigmoid"

    # Chunked verify (`GCH=True`) when B=16, T=4, gdn_rows=8, gdn_bufs=2
    ex_v = ExecSpec(B=16, T=4, R=2, ctx=256, stack_e=True, tiles=tiles)
    p_v = plan(ms, ex_v)
    tt_gch = GDNTunables(gdn_rows=8, gdn_bufs=2, gdn_head_major=True)
    gp_v = derive_params(g, ex_v, p_v, tt_gch, "fake_gqa")
    assert gp_v.GCH is True and gp_v.CB == 8 and gp_v.NCH == 2 and gp_v.NRB == 2
    assert gp_v.head_major is True

    with pytest.raises(MegakernelConfigError, match="gdn_bufs >= 2"):
        derive_params(g, ex_v, p_v, GDNTunables(gdn_rows=8, gdn_bufs=1), "fake_gqa")
    with pytest.raises(MegakernelConfigError, match="'silu' or 'sigmoid'"):
        derive_params(g, ex, p, GDNTunables(), "fake_gqa", gate="bad_gate")
    with pytest.raises(MegakernelConfigError, match="gdn_rows must be a multiple of 8"):
        derive_params(g, ex_v, p_v, GDNTunables(gdn_rows=12, gdn_bufs=2), "fake_gqa")
    ex_v3 = ExecSpec(B=12, T=3, R=2, ctx=256, stack_e=True, tiles=tiles)
    p_v3 = plan(ms, ex_v3)
    with pytest.raises(MegakernelConfigError, match="must be a multiple of T"):
        derive_params(g, ex_v3, p_v3, GDNTunables(gdn_rows=8, gdn_bufs=2), "fake_gqa")
    g_bad_kh = dataclasses.replace(g, KH=3)
    with pytest.raises(MegakernelConfigError, match="divisible by R"):
        derive_params(g_bad_kh, ex, p, GDNTunables(), "fake_gqa")
    g_bad_vh = dataclasses.replace(g, KH=2, VH=256)
    with pytest.raises(MegakernelConfigError, match="vh <= 64"):
        derive_params(g_bad_vh, ex, p, GDNTunables(), "fake_gqa")
    g_bad_h = dataclasses.replace(g, H=192)
    with pytest.raises(MegakernelConfigError, match="multiple of 128"):
        derive_params(g_bad_h, ex, p, GDNTunables(), "fake_gqa")
    g_eq_hv = dataclasses.replace(g, KH=4, VH=4)
    with pytest.raises(
        MegakernelConfigError,
        match=r"gdn_head_major=True requires 2 \* kh <= vh.*kh=2, vh=2",
    ):
        derive_params(g_eq_hv, ex_v, p_v, tt_gch, "fake_gqa")

    # checked_pool_layout validation and derive_params pool-I/O branch (FUSED & GPR)
    with pytest.raises(MegakernelConfigError, match="needs a pool backend"):
        checked_pool_layout(p, ex, g_per_att=1, ssm_u32_rows=256, g=g)

    pl_bf16 = PoolLayout(
        num_pools=1,
        slots_per_pool=1,
        kh=4,
        vh=8,
        rec_pages_per_slot=4,
        conv_pages_per_slot=1,
        conv_taps=3,
        conv_dtype=BF16,
    )
    pl_f32 = PoolLayout(
        num_pools=1,
        slots_per_pool=1,
        kh=4,
        vh=8,
        rec_pages_per_slot=4,
        conv_pages_per_slot=1,
        conv_taps=3,
        conv_dtype=F32,
    )
    ex_pool = ExecSpec(
        B=4,
        T=1,
        R=2,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        cache=CacheExec(paged_kv=True, page_size=128, num_pages=32, fused_pool_io=True),
    )
    k_names = ("gather_pool_states", "scatter_pool_states")
    ms_pool_bf16 = dataclasses.replace(
        ms,
        cache_backend=PoolBackend(layout=pl_bf16, kernel_names=k_names),
    )
    p_pool_bf16 = plan(ms_pool_bf16, ex_pool)
    assert checked_pool_layout(p_pool_bf16, ex_pool, 1, 256, g) is pl_bf16
    with pytest.raises(MegakernelConfigError, match="slots_per_pool"):
        checked_pool_layout(p_pool_bf16, ex_pool, 2, 256, g)
    with pytest.raises(MegakernelConfigError, match="exceeds the pool layout"):
        checked_pool_layout(p_pool_bf16, ex_pool, 1, 1024, g)
    ms_pool_bad_kh = dataclasses.replace(
        ms,
        cache_backend=PoolBackend(
            layout=dataclasses.replace(pl_bf16, kh=2, vh=4),
            kernel_names=k_names,
        ),
    )
    p_pool_bad_kh = plan(ms_pool_bad_kh, ex_pool)
    with pytest.raises(MegakernelConfigError, match=r"KH=4.*kh=2"):
        checked_pool_layout(p_pool_bad_kh, ex_pool, 1, 256, g)

    gp_fused_bf16 = derive_params(g, ex_pool, p_pool_bf16, GDNTunables(), "fake_gqa")
    assert gp_fused_bf16.pool_layout is pl_bf16
    assert gp_fused_bf16.REC_PAGES == 4 and gp_fused_bf16.GDN_PAGES == 5
    assert gp_fused_bf16.SSM_U32_CAP == 512 and gp_fused_bf16.G_PER_ATT == 1
    assert gp_fused_bf16.conv_is_f32 is False and gp_fused_bf16.A_conv == 16
    assert gp_fused_bf16.NGI == 4 and gp_fused_bf16.NGO == 2

    ms_pool_f32 = dataclasses.replace(
        ms,
        cache_backend=PoolBackend(layout=pl_f32, kernel_names=k_names),
    )
    p_pool_f32 = plan(ms_pool_f32, ex_pool)
    gp_fused_f32 = derive_params(g, ex_pool, p_pool_f32, GDNTunables(), "fake_gqa")
    assert gp_fused_f32.pool_layout is pl_f32
    assert gp_fused_f32.REC_PAGES == 4 and gp_fused_f32.GDN_PAGES == 5
    assert gp_fused_f32.conv_is_f32 is True and gp_fused_f32.A_conv == 8
    assert gp_fused_f32.GCONV_ROWS == 2 * gp_fused_bf16.GCONV_ROWS

    # GPR requires verify (T > 1)
    ex_bad_gpr = dataclasses.replace(
        ex_pool,
        cache=CacheExec(paged_kv=True, page_size=128, num_pages=32, pool_gather=True),
    )
    with pytest.raises(MegakernelConfigError, match="paged chunked verify path"):
        derive_params(g, ex_bad_gpr, p_pool_bf16, GDNTunables(), "fake_gqa")


class _BoundGDNModule(Module):
    name = "gdn_test_bind"
    GEOMETRY_FIELDS = GDN_GEOMETRY_FIELDS
    FIELD_ALIASES = GDN_FIELD_ALIASES
    OPTIONS = GDN_OPTIONS
    PARAM_NAMES = GDN_PARAM_NAMES
    GEOMETRY_METHODS = GDN_GEOMETRY_METHODS

    @classmethod
    def _validate_options(cls, options):
        validate_gdn_options(options)

    @classmethod
    def bind(cls, prefix: str = "", **hf_fields_or_options):
        return bind_gdn_module(cls, prefix, **hf_fields_or_options)

    @classmethod
    def _resolve_binding(cls, binding, hf_config):
        return resolve_gdn_binding(
            cls, binding, super()._resolve_binding(binding, hf_config)
        )


class _SubBoundGDNModule(_BoundGDNModule):
    name = "gdn_test_bind_sub"


def test_gdn_decls_options_binding_and_hf_params():
    """Verify GDN options, binding resolution, HF parameter keys, and cache decl."""
    assert GDN_WEIGHT_NAMES == ("lin_in", "lin_out")
    assert "lin_q" in GDN_PARAM_NAMES and "conv" in GDN_PARAM_NAMES
    assert "H" in GDN_FIELD_ALIASES and "hidden" in GDN_GEOMETRY_FIELDS
    assert set(GDN_OPTIONS.keys()) == {"gate", "weight_dtypes"}

    validate_gdn_options(
        {"gate": "sigmoid", "weight_dtypes": {"lin_in": "bf16", "lin_out": "fp8"}}
    )
    with pytest.raises(MegakernelConfigError, match="gate"):
        validate_gdn_options({"gate": "bad"})
    with pytest.raises(MegakernelConfigError, match="weight_dtypes"):
        validate_gdn_options({"weight_dtypes": {"bad": "bf16"}})

    # bind(gate=...) routes gate as an option and validates at bind() time
    with pytest.raises(MegakernelConfigError, match="gate"):
        _BoundGDNModule.bind("linear_attn", gate="bad")

    bound_sig = _BoundGDNModule.bind(
        "linear_attn",
        hidden="hidden_size",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        gate="sigmoid",
    )
    assert bound_sig.options["gate"] == "sigmoid"
    assert "gate" not in bound_sig.explicit_fields
    res_from_bind = bound_sig.resolve(
        {
            "hidden_size": 256,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
        }
    )
    assert res_from_bind.gate == "sigmoid" and res_from_bind.gdn_gate == "sigmoid"

    # Subclass B(A) of a delegating module A must bind without infinite recursion
    sub_plain = _SubBoundGDNModule.bind(
        "linear_attn",
        hidden="hidden_size",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
    )
    assert sub_plain.module_cls is _SubBoundGDNModule
    sub_sig = _SubBoundGDNModule.bind(
        "linear_attn",
        hidden="hidden_size",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        gate="sigmoid",
    )
    assert sub_sig.module_cls is _SubBoundGDNModule
    assert sub_sig.options["gate"] == "sigmoid"

    # Explicit option in ModuleBinding beats the geometry default ("silu")
    binding = ModuleBinding(
        module_cls=FakeMixerModule,
        prefix="linear_attn",
        options={"gate": "sigmoid"},
        rename={"lin_q": "q_ssm"},
    )
    raw_res = ResolvedModuleGeometry(
        module_cls=FakeMixerModule,
        prefix="linear_attn",
        values={
            "hidden": 256,
            "k_heads": 4,
            "v_heads": 8,
            "k_dim": 128,
            "v_dim": 128,
            "conv_kernel": 4,
            "eps": 1e-6,
            "gdn_l2_eps": 1e-6,
            "gdn_gate": "silu",
        },
        options=binding.options,
        rename=binding.rename,
    )
    res = resolve_gdn_binding(FakeMixerModule, binding, raw_res)
    assert res.H == 256 and res.KH == 4 and res.VH == 8 and res.conv_taps == 3
    assert res.gate == "sigmoid" and res.gdn_gate == "sigmoid"
    assert res.options["gate"] == "sigmoid"
    assert GDN_GEOMETRY_METHODS["kh"](res, 2) == 2
    assert GDN_GEOMETRY_METHODS["vh"](res, 2) == 4

    bad_res = dataclasses.replace(raw_res, values={**raw_res.values, "conv_kernel": 5})
    with pytest.raises(MegakernelConfigError, match="conv_kernel == 4"):
        resolve_gdn_binding(FakeMixerModule, binding, bad_res)

    decls_fp8 = gdn_hf_params("linear_attn", res)
    names_fp8 = {d.name for d in decls_fp8}
    assert "q_ssm" in names_fp8 and "lin_q_s" in names_fp8 and "lin_out_s" in names_fp8
    decls_bf16 = gdn_hf_params(
        "linear_attn", res, weight_dtypes={"lin_in": "bf16", "lin_out": "bf16"}
    )
    names_bf16 = {d.name for d in decls_bf16}
    assert "lin_q_s" not in names_bf16 and "lin_out_s" not in names_bf16

    cdecl = gdn_cache_decl(res)
    assert isinstance(cdecl, RecurrentCacheDecl)
    assert cdecl.kh == 4 and cdecl.vh == 8 and cdecl.conv_taps == 3


def _numpy_recurrent_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    *,
    l2_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy port of HuggingFace `torch_recurrent_gated_delta_rule`."""
    q = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + l2_eps)
    k = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + l2_eps)
    q = q * (query.shape[-1] ** -0.5)
    NS, T, vh, VD = value.shape
    kh, KD = q.shape[2], q.shape[3]
    if vh > kh:
        q = np.repeat(q, vh // kh, axis=2)
        k = np.repeat(k, vh // kh, axis=2)

    core_attn_out = np.zeros((NS, T, vh, VD), dtype=np.float32)
    step_states = np.zeros((NS, T, vh, KD, VD), dtype=np.float32)
    last_recurrent_state = initial_state.astype(np.float32).copy()
    for i in range(T):
        q_t = q[:, i]
        k_t = k[:, i]
        v_t = value[:, i]
        g_t = np.exp(g[:, i])[..., None, None]
        beta_t = beta[:, i][..., None]
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = np.einsum("shk,shkv->shv", k_t, last_recurrent_state)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + np.einsum(
            "shk,shv->shkv", k_t, delta
        )
        core_attn_out[:, i] = np.einsum("shkv,shk->shv", last_recurrent_state, q_t)
        step_states[:, i] = last_recurrent_state
    return core_attn_out, step_states


@pytest.mark.parametrize("multi_layer", [False, True], ids=["single_l", "multi_l"])
@pytest.mark.parametrize("R", [1, 2], ids=["r1", "r2"])
@pytest.mark.parametrize(("T", "gate"), [(1, "silu"), (2, "sigmoid"), (4, "silu")])
def test_gdn_reference_matches_direct_computation(
    T: int, gate: str, R: int, multi_layer: bool
):
    """Compare `gdn_reference` against an independent NumPy port of HF GatedDeltaNet."""
    B = 4
    NGDN = 2 if multi_layer else 1
    gdn_idx = 1 if multi_layer else 0
    g = FakeGeometry(L=NGDN, att_every=1, H=256, KH=2 * R, VH=4 * R, KD=128, VD=128)
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, tiles=TINY_TILES)
    lay = gdn_lin_in_layout(g, ex)
    kh, vh, KD, VD = lay.kh, lay.vh, lay.KD, lay.VD
    QKV, Z, BA, A_OFF, NLI = lay.QKV, lay.Z, lay.BA, lay.A_OFF, lay.NLI

    keys = jax.random.split(
        jax.random.PRNGKey(50 + T + 7 * R + 13 * int(multi_layer)), 9
    )
    x = jax.random.normal(keys[0], (B, g.H), dtype=BF16) * 0.2
    w_in_4d = jax.random.normal(keys[1], (R, NGDN, g.H, NLI), dtype=BF16) * 0.1
    w_out_4d = jax.random.normal(keys[2], (R, NGDN, Z, g.H), dtype=BF16) * 0.1
    conv_w_3d = jax.random.normal(keys[3], (R, NGDN, 4 * QKV), dtype=F32) * 0.1
    alog_3d = jax.random.normal(keys[4], (R, NGDN, vh), dtype=F32) * 0.1 - 1.0
    dtb_3d = jax.random.normal(keys[5], (R, NGDN, vh), dtype=F32) * 0.1
    gnorm_3d = jax.random.normal(keys[6], (R, NGDN, VD), dtype=F32) * 0.05 + 1.0
    conv_st_all = jax.random.normal(keys[7], (R, NGDN, ex.NS, 3, QKV), dtype=F32) * 0.2
    rec_st_all = (
        jax.random.normal(keys[8], (R, NGDN, ex.NS, vh, KD, VD), dtype=BF16) * 0.2
    )

    if multi_layer:
        params = {
            "lin_in": w_in_4d,
            "lin_out": w_out_4d,
            "conv": conv_w_3d,
            "alog": alog_3d,
            "dtb": dtb_3d,
            "gnorm": gnorm_3d,
        }
        state_in: dict[str, jax.Array] = {"conv": conv_st_all, "rec": rec_st_all}
    else:
        params = {
            "lin_in": w_in_4d[:, 0],
            "lin_out": w_out_4d[:, 0],
            "conv": conv_w_3d[:, 0],
            "alog": alog_3d[:, 0],
            "dtb": dtb_3d[:, 0],
            "gnorm": gnorm_3d[:, 0],
        }
        state_in = {"conv": conv_st_all[:, 0], "rec": rec_st_all[:, 0]}

    if T == 1 and R == 1 and not multi_layer:
        with pytest.raises(MegakernelConfigError, match="FP8 'lin_in' requires"):
            gdn_reference(
                {**params, "lin_in": params["lin_in"].astype(FP8)},
                x,
                state_in,
                g=g,
                ex=ex,
                multi_layer=False,
            )
        with pytest.raises(MegakernelConfigError, match="FP8 'lin_out' requires"):
            gdn_reference(
                {**params, "lin_out": params["lin_out"].astype(FP8)},
                x,
                state_in,
                g=g,
                ex=ex,
                multi_layer=False,
            )
    ref_out, ref_st = gdn_reference(
        params,
        x,
        state_in,
        g=g,
        ex=ex,
        gdn_idx=gdn_idx,
        multi_layer=multi_layer,
        gate=gate,
    )

    if T == 1 and not multi_layer:
        conv_packed = jnp.swapaxes(state_in["conv"], 1, 2).reshape(R, 3 * B, QKV)
        rec_flat = state_in["rec"].reshape(R, B * vh, KD, VD)
        ref_out_p, _ = gdn_reference(
            params,
            x,
            {"conv": conv_packed, "rec": rec_flat},
            g=g,
            ex=ex,
            multi_layer=False,
            packed_conv=True,
            flat_rec=True,
            gate=gate,
        )
        np.testing.assert_allclose(ref_out_p, ref_out, rtol=0, atol=0)

    # Independent unsharded NumPy HF GatedDeltaNet oracle across all KH/VH heads
    NS = ex.NS
    x_np = np.asarray(x.astype(F32), dtype=np.float32)
    w_in_np = np.asarray(w_in_4d[:, gdn_idx].astype(F32), dtype=np.float32)
    w_out_np = np.asarray(w_out_4d[:, gdn_idx].astype(F32), dtype=np.float32)
    conv_w_np = np.asarray(conv_w_3d[:, gdn_idx], dtype=np.float32).reshape(R, 4, QKV)
    alog_np = np.asarray(alog_3d[:, gdn_idx], dtype=np.float32).reshape(R * vh)
    dtb_np = np.asarray(dtb_3d[:, gdn_idx], dtype=np.float32).reshape(R * vh)
    gnorm_np = np.asarray(gnorm_3d[:, gdn_idx], dtype=np.float32)
    conv_st_np = np.asarray(conv_st_all[:, gdn_idx], dtype=np.float32)
    rec_st_np = np.asarray(rec_st_all[:, gdn_idx].astype(F32), dtype=np.float32)

    q_all = np.zeros((NS, T, R * kh, KD), dtype=np.float32)
    k_all = np.zeros((NS, T, R * kh, KD), dtype=np.float32)
    v_all = np.zeros((NS, T, R * vh, VD), dtype=np.float32)
    z_all = np.zeros((NS, T, R * vh, VD), dtype=np.float32)
    g_all = np.zeros((NS, T, R * vh), dtype=np.float32)
    beta_all = np.zeros((NS, T, R * vh), dtype=np.float32)
    conv_next_r = []

    for r in range(R):
        proj_r = x_np @ w_in_np[r]
        qkv_r = proj_r[:, :QKV].reshape(NS, T, QKV)
        # Depthwise causal conv1d over the 4-tap window [hist_0, hist_1, hist_2, new...]
        win_r = np.concatenate([conv_st_np[r], qkv_r], axis=1)  # (NS, 3 + T, QKV)
        conv_pre = np.stack(
            [
                np.einsum(
                    "tkc,kc->tc",
                    np.stack([win_r[s, t : t + 4] for t in range(T)], axis=0),
                    conv_w_np[r],
                )
                for s in range(NS)
            ],
            axis=0,
        )
        u_r = conv_pre / (1.0 + np.exp(-conv_pre))
        if T == 1:
            conv_next_r.append(win_r[:, -3:, :].transpose(1, 0, 2).reshape(3 * B, QKV))
        else:
            conv_next_r.append(win_r)
        q_all[:, :, r * kh : (r + 1) * kh, :] = u_r[:, :, : kh * KD].reshape(
            NS, T, kh, KD
        )
        k_all[:, :, r * kh : (r + 1) * kh, :] = u_r[
            :, :, kh * KD : 2 * kh * KD
        ].reshape(NS, T, kh, KD)
        v_all[:, :, r * vh : (r + 1) * vh, :] = u_r[:, :, 2 * kh * KD :].reshape(
            NS, T, vh, VD
        )
        z_all[:, :, r * vh : (r + 1) * vh, :] = proj_r[:, QKV : QKV + Z].reshape(
            NS, T, vh, VD
        )
        b_r = proj_r[:, BA : BA + vh].reshape(NS, T, vh)
        a_r = proj_r[:, A_OFF : A_OFF + vh].reshape(NS, T, vh)
        beta_all[:, :, r * vh : (r + 1) * vh] = 1.0 / (1.0 + np.exp(-b_r))
        sp_r = np.logaddexp(0.0, a_r + dtb_np[r * vh : (r + 1) * vh])
        g_all[:, :, r * vh : (r + 1) * vh] = (
            -np.exp(alog_np[r * vh : (r + 1) * vh]) * sp_r
        )

    init_rec_full = rec_st_np.transpose(1, 0, 2, 3, 4).reshape(NS, R * vh, KD, VD)
    core_out, step_states = _numpy_recurrent_gated_delta_rule(
        q_all, k_all, v_all, g_all, beta_all, init_rec_full, l2_eps=g.gdn_l2_eps
    )
    # RMSNormGated + gate activation + unsharded out_proj (`(R * Z, H)`)
    rms = core_out / np.sqrt(
        np.mean(core_out * core_out, axis=-1, keepdims=True) + g.eps
    )
    gnorm_full = np.repeat(gnorm_np, vh, axis=0)  # (R * vh, VD)
    normed = rms * gnorm_full[None, None, :, :]
    gate_val = (
        z_all / (1.0 + np.exp(-z_all))
        if gate == "silu"
        else 1.0 / (1.0 + np.exp(-z_all))
    )
    gated_full = (normed * gate_val).reshape(B, R * Z)
    w_out_full = w_out_np.reshape(R * Z, g.H)
    expected_np = gated_full @ w_out_full

    np.testing.assert_allclose(np.asarray(ref_out), expected_np, rtol=2e-2, atol=3e-2)
    rec_key = "rec_ckpt" if T > 1 else "rec"
    expected_rec = (
        step_states.reshape(NS, T, R, vh, KD, VD)
        .transpose(2, 0, 1, 3, 4, 5)
        .reshape(R, B * vh, KD, VD)
    )
    np.testing.assert_allclose(
        np.asarray(ref_st[rec_key]), expected_rec, rtol=1e-2, atol=1e-2
    )
    conv_key = "conv_ckpt" if T > 1 else "conv"
    for r in range(R):
        if T == 1:
            np.testing.assert_allclose(
                np.asarray(ref_st[conv_key][r, : 3 * B]),
                conv_next_r[r],
                rtol=1e-2,
                atol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                np.asarray(ref_st[conv_key][r, :, : 3 + T]),
                conv_next_r[r],
                rtol=1e-2,
                atol=1e-2,
            )
    # Must-fail control: inverted output fails at the same tolerance
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(ref_out), -expected_np, rtol=2e-2, atol=3e-2
        )
