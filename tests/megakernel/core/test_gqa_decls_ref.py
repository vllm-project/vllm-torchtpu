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
"""Standalone unit tests for GQA parameters, declarations, and reference."""

from __future__ import annotations

import dataclasses

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
from vllm_torchtpu.megakernel.core import module_options
from vllm_torchtpu.megakernel.core.decls import (
    KVCacheDecl,
    ModuleBinding,
    ResolvedModuleGeometry,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gqa.decls import (
    GQA_FIELD_ALIASES,
    GQA_GEOMETRY_FIELDS,
    GQA_GEOMETRY_METHODS,
    GQA_OPTIONS,
    GQA_PARAM_NAMES,
    GQA_WEIGHT_NAMES,
    gqa_cache_decl,
    gqa_hf_params,
    resolve_gqa_binding,
    validate_gqa_options,
)
from vllm_torchtpu.megakernel.core.modules.gqa.params import (
    GQATunables,
    derive_params,
    gqa_scratch,
)
from vllm_torchtpu.megakernel.core.modules.gqa.reference import gqa_reference
from vllm_torchtpu.megakernel.core.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    rope_partial,
    round_bf16,
)
from vllm_torchtpu.megakernel.core.planner import plan
from vllm_torchtpu.megakernel.core.pool.config import CacheExec
from vllm_torchtpu.megakernel.core.pool.layout import PoolLayout
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec


def test_gqa_tunables_and_derive_params_validation():
    """Verify `GQATunables`, `derive_params` modes/errors, and `gqa_scratch`."""
    field_names = {f.name for f in dataclasses.fields(GQATunables)}
    assert set(GQATunables.NUMERICS.keys()) == field_names

    g = FakeGeometry(L=2, att_every=1, H=256, NH=4, NKV=1, HD=256, ROT=64)
    ms = ModelSpec(
        "gqa_p",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10)
    ex = ExecSpec(B=4, T=1, R=2, ctx=256, stack_e=True, tiles=tiles)
    p = plan(ms, ex)

    tt_split = GQATunables(kv_blk=128, nkv=2, split=True, pack=1, gqa_qv=True)
    ap = derive_params(g, ex, p, tt_split)
    assert ap.G == 2 and ap.SPLIT and ap.PACK and ap.PB == 4 and ap.QV
    assert ap.qrow(5) == (1, 2)
    scr_split = gqa_scratch(ex, ap)
    assert "kbuf" in scr_split and "pso" in scr_split and "rope" in scr_split

    # GQ mode requires T > 1, gq_pack=True, gqa_qv=True, SPLIT, QG % 16 == 0, HD == 256
    g_gq = FakeGeometry(L=2, att_every=1, H=512, NH=8, NKV=1, HD=256, ROT=64)
    ms_gq = ModelSpec(
        "gqa_gq",
        config=g_gq,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    ex_gq = ExecSpec(B=8, T=4, R=2, ctx=256, stack_e=True, tiles=tiles)
    p_gq = plan(ms_gq, ex_gq)
    tt_gq = GQATunables(kv_blk=128, nkv=2, split=True, gqa_qv=True, gq_pack=True)
    ap_gq = derive_params(g_gq, ex_gq, p_gq, tt_gq)
    assert ap_gq.GQ and ap_gq.MLW == 128 // ex_gq.NS
    scr_gq = gqa_scratch(ex_gq, ap_gq)
    assert "qxb" in scr_gq and "psend" in scr_gq and "pso" not in scr_gq

    # Paged KV + KVE (expert ring aliasing)
    tiles_kve = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=512 << 10, nbe=4)
    ex_paged = ExecSpec(
        B=4,
        T=1,
        R=2,
        ctx=256,
        stack_e=True,
        tiles=tiles_kve,
        cache=CacheExec(paged_kv=True, page_size=128, num_pages=16),
    )
    p_paged = plan(ms, ex_paged)
    ap_kve = derive_params(
        g,
        ex_paged,
        p_paged,
        GQATunables(kv_blk=128, nkv=2, kv_ering=True, pair_kv=True),
    )
    assert ap_kve.KVE and ap_kve.PAIR_KV and ap_kve.NKV == 4
    scr_kve = gqa_scratch(ex_paged, ap_kve)
    assert "kvbuf" not in scr_kve and "knew_o" in scr_kve and "wl_kp" in scr_kve

    # KVE raises if dynamic-ring owner does not run on every layer
    p_partial = dataclasses.replace(
        p_paged, module_layer_counts=dict.fromkeys(p_paged.module_layer_counts, 1)
    )
    with pytest.raises(MegakernelConfigError, match="kv_ering"):
        derive_params(
            g, ex_paged, p_partial, GQATunables(kv_blk=128, nkv=2, kv_ering=True)
        )

    with pytest.raises(MegakernelConfigError, match="nkv >= 2"):
        derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=1, pair_kv=False))
    with pytest.raises(MegakernelConfigError, match="gq_pack requires gqa_qv"):
        derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, gq_pack=True))
    with pytest.raises(MegakernelConfigError, match="kv_blk"):
        derive_params(g, ex, p, GQATunables(kv_blk=100, nkv=2))
    with pytest.raises(MegakernelConfigError, match="kv_sub"):
        derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, kv_sub=64))
    with pytest.raises(MegakernelConfigError, match="amask"):
        derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, amask=5))
    with pytest.raises(MegakernelConfigError, match="paged_kv requires amask"):
        derive_params(g, ex_paged, p_paged, GQATunables(kv_blk=128, nkv=2, amask=3))
    ex_bad_ps = dataclasses.replace(
        ex_paged, cache=CacheExec(paged_kv=True, page_size=64, num_pages=16)
    )
    with pytest.raises(MegakernelConfigError, match="page_size"):
        derive_params(g, ex_bad_ps, p_paged, GQATunables(kv_blk=128, nkv=2))
    g_bad_g = FakeGeometry(L=2, att_every=1, H=256, NH=12, NKV=2, HD=256, ROT=64)
    ex_r6 = ExecSpec(B=4, T=1, R=6, ctx=256, stack_e=True, tiles=tiles)
    with pytest.raises(MegakernelConfigError, match="power of two"):
        derive_params(g_bad_g, ex_r6, p, GQATunables(kv_blk=128, nkv=2, split=True))


def test_gqa_decls_options_binding_and_hf_params():
    """Verify GQA options, binding resolution, HF parameter keys, and cache decl."""
    assert GQA_WEIGHT_NAMES == ("att_in", "att_out")
    assert "q_proj" in GQA_PARAM_NAMES and "qn" in GQA_PARAM_NAMES
    assert "H" in GQA_FIELD_ALIASES and "hidden" in GQA_GEOMETRY_FIELDS
    assert GQA_OPTIONS["qk_norm"] is True

    validate_gqa_options(
        {
            "qk_norm": True,
            "output_gate": True,
            "weight_dtypes": {"att_in": "bf16", "att_out": "fp8"},
        }
    )
    with pytest.raises(MegakernelConfigError, match="qk_norm=True"):
        validate_gqa_options({"qk_norm": False})
    with pytest.raises(MegakernelConfigError, match="output_gate=True"):
        validate_gqa_options({"output_gate": False})
    with pytest.raises(MegakernelConfigError, match="weight_dtypes"):
        validate_gqa_options({"weight_dtypes": {"bad": "fp8"}})

    binding = ModuleBinding(
        module_cls=FakeMixerModule,
        prefix="self_attn",
        options={"qk_norm": True, "output_gate": True},
        rename={"q_proj": "q_custom"},
    )
    raw_res = ResolvedModuleGeometry(
        module_cls=FakeMixerModule,
        prefix="self_attn",
        values={
            "hidden": 256,
            "heads": 4,
            "kv_heads": 2,
            "head_dim": None,
            "rot": 0.5,
            "eps": 1e-6,
            "theta": 10000.0,
        },
        options=binding.options,
        rename=binding.rename,
    )
    resolved = resolve_gqa_binding(FakeMixerModule, binding, raw_res)
    assert resolved.HD == 64 and resolved.ROT == 32
    assert GQA_GEOMETRY_METHODS["nh"](resolved, 2) == 2
    assert GQA_GEOMETRY_METHODS["G"](resolved, 2) == 1
    assert GQA_GEOMETRY_METHODS["NAI"](resolved, 2) == (2 * 2 + 2) * 64

    # Whole-number partial_rotary_factor=1 (int) resolves to full HD (64),
    # and odd or >HD rotary dimensions raise MegakernelConfigError.
    res_int1 = resolve_gqa_binding(
        FakeMixerModule,
        binding,
        dataclasses.replace(raw_res, values={**raw_res.values, "rot": 1}),
    )
    assert res_int1.ROT == 64
    for bad_rot in (0, 33, 128, True):
        with pytest.raises(MegakernelConfigError, match="ROT|rot"):
            resolve_gqa_binding(
                FakeMixerModule,
                binding,
                dataclasses.replace(raw_res, values={**raw_res.values, "rot": bad_rot}),
            )

    decls_fp8 = gqa_hf_params("self_attn", resolved)
    names_fp8 = {d.name for d in decls_fp8}
    assert "q_custom" in names_fp8 and "q_proj_s" in names_fp8 and "qn" in names_fp8

    decls_bf16 = gqa_hf_params(
        "self_attn", resolved, weight_dtypes={"att_in": "bf16", "att_out": "bf16"}
    )
    names_bf16 = {d.name for d in decls_bf16}
    assert "q_proj_s" not in names_bf16 and "att_out_s" not in names_bf16

    cdecl_geom = gqa_cache_decl(resolved)
    assert cdecl_geom == KVCacheDecl(hd=64, kvh=1, page_size=128, page_dtype=FP8)
    play = PoolLayout(num_pools=1, slots_per_pool=0, hd=128, page_size=128)
    assert gqa_cache_decl(None, pool_layout=play).hd == 128


@pytest.mark.parametrize("use_fp8", [False, True], ids=["bf16_w", "fp8_w"])
@pytest.mark.parametrize("paged_kv", [False, True], ids=["direct_hbm", "paged_kv"])
@pytest.mark.parametrize("T", [1, 2], ids=["decode_t1", "verify_t2"])
def test_gqa_reference_matches_direct_computation(
    monkeypatch: pytest.MonkeyPatch, T: int, paged_kv: bool, use_fp8: bool
):
    """Compare `gqa_reference` against an independent computation across modes."""
    R, B, ctx = 2, 2 * T, 128
    L = 2 if use_fp8 else 1
    att_idx = 1 if use_fp8 else 0
    zc = not use_fp8
    g = FakeGeometry(
        L=L,
        att_every=1,
        H=256,
        NH=4,
        NKV=2,
        HD=128,
        ROT=32,
        zero_centered_norm=zc,
    )
    # "gqa" is registered by `gqa/params.py`; also test isolated custom name.
    assert TINY_TILES.module("gqa") == GQATunables()
    monkeypatch.setitem(module_options._TUNABLES, "custom_gqa_test", GQATunables)
    mod_name = "custom_gqa_test" if use_fp8 else "gqa"
    tiles = TINY_TILES.with_modules({mod_name: GQATunables(rope_fp32=paged_kv)})
    rope_round = not paged_kv
    cache = (
        CacheExec(paged_kv=True, page_size=128, num_pages=16)
        if paged_kv
        else CacheExec()
    )
    ex = ExecSpec(B=B, T=T, R=R, ctx=ctx, tiles=tiles, cache=cache)
    nh = g.NH // R
    nai = (2 * nh + 2) * g.HD
    koff = nh * 2 * g.HD

    keys = jax.random.split(
        jax.random.PRNGKey(7 + T + 10 * int(paged_kv) + 20 * int(use_fp8)), 9
    )
    x = jax.random.normal(keys[0], (B, g.H), dtype=BF16) * 0.5
    qn = jax.random.normal(keys[3], (R, L, g.HD), dtype=F32) * 0.2 + 1.0
    kn = jax.random.normal(keys[4], (R, L, g.HD), dtype=F32) * 0.2 + 1.0
    kc = jax.random.normal(keys[5], (R, L, ex.NS, ctx, g.HD), dtype=BF16) * 0.4
    vc = jax.random.normal(keys[6], (R, L, ex.NS, ctx, g.HD), dtype=BF16) * 0.4
    pos = jnp.array([4 + 5 * (b // T) + (b % T) for b in range(B)], dtype=I32)

    if use_fp8:
        w_in = jax.random.normal(keys[1], (R, L, g.H, nai), dtype=BF16).astype(FP8)
        w_out = jax.random.normal(keys[2], (R, L, nh * g.HD, g.H), dtype=BF16).astype(
            FP8
        )
        in_s = jax.random.uniform(
            keys[7], (R, L, nai), dtype=F32, minval=0.05, maxval=0.2
        )
        out_s = jax.random.uniform(
            keys[8], (R, L, g.H), dtype=F32, minval=0.05, maxval=0.2
        )
        params = {
            "att_in": w_in,
            "att_in_s": in_s,
            "att_out": w_out,
            "att_out_s": out_s,
            "qn": qn,
            "kn": kn,
        }
    else:
        w_in = jax.random.normal(keys[1], (R, g.H, nai), dtype=BF16) * 0.2
        w_out = jax.random.normal(keys[2], (R, nh * g.HD, g.H), dtype=BF16) * 0.2
        in_s = None
        out_s = None
        params = {"att_in": w_in, "att_out": w_out, "qn": qn[:, 0], "kn": kn[:, 0]}
        kc = kc[:, 0]
        vc = vc[:, 0]

    state = {"kc": kc, "vc": vc}
    delta, new_state = gqa_reference(
        params,
        x,
        state,
        g=g,
        ex=ex,
        pos=pos,
        att_idx=att_idx,
        layer_idx=att_idx,
        module_name=mod_name,
    )
    assert delta.shape == (B, g.H)
    assert new_state["kc"].shape == kc.shape

    # Independent computation across every row `b in range(B)` (including `tb=1`
    # for `T=2` causal in-chunk attention) and all ranks `r`.
    xb = x[:B].astype(BF16).astype(F32)
    half = g.ROT // 2
    inv = 1.0 / (g.theta ** (jnp.arange(half, dtype=F32)[None, :] * 2.0 / g.ROT))

    expected_delta = jnp.zeros((B, g.H), dtype=F32)
    for r in range(R):
        w_in_r = (w_in[r, att_idx] if use_fp8 else w_in[r]).astype(F32)
        w_out_r = (w_out[r, att_idx] if use_fp8 else w_out[r]).astype(F32)
        qn_r = qn[r, att_idx]
        kn_r = kn[r, att_idx]
        y_raw = xb @ w_in_r
        y_r = round_bf16(y_raw * in_s[r, att_idx] if in_s is not None else y_raw)
        qg_r = y_r[:, :koff].reshape(B, nh, 2 * g.HD)

        q_rows = []
        k_rows = []
        v_rows = []
        gate_rows = []
        for b in range(B):
            ang_b = pos[b].astype(F32) * inv
            cos_b, sin_b = round_bf16(jnp.cos(ang_b)), round_bf16(jnp.sin(ang_b))
            gate_rows.append(
                round_bf16(jax.nn.sigmoid(qg_r[b, :, g.HD :].reshape(nh * g.HD)))
            )
            q_b = round_bf16(
                rope_partial(
                    rmsnorm(qg_r[b, :, : g.HD], qn_r, g.eps, zero_centered=zc),
                    cos_b,
                    sin_b,
                    g.ROT,
                    round_intermediates=rope_round,
                )
                * (g.HD**-0.5)
            )
            k_b = rope_partial(
                rmsnorm(
                    y_r[b : b + 1, koff : koff + g.HD],
                    kn_r,
                    g.eps,
                    zero_centered=zc,
                ),
                cos_b,
                sin_b,
                g.ROT,
                round_intermediates=rope_round,
            )[0]
            v_b = y_r[b, koff + g.HD : koff + 2 * g.HD]
            if paged_kv:
                k_b = jnp.clip(k_b, -448.0, 448.0).astype(FP8).astype(F32)
                v_b = jnp.clip(v_b, -448.0, 448.0).astype(FP8).astype(F32)
            q_rows.append(q_b)
            k_rows.append(k_b)
            v_rows.append(v_b)

        kc_r = kc[r, att_idx] if use_fp8 else kc[r]
        vc_r = vc[r, att_idx] if use_fp8 else vc[r]
        if paged_kv:
            kc_r = jnp.clip(kc_r.astype(F32), -448.0, 448.0).astype(FP8).astype(BF16)
            vc_r = jnp.clip(vc_r.astype(F32), -448.0, 448.0).astype(FP8).astype(BF16)
        kc_out_r = new_state["kc"][r, att_idx] if use_fp8 else new_state["kc"][r]
        vc_out_r = new_state["vc"][r, att_idx] if use_fp8 else new_state["vc"][r]

        for sq in range(ex.NS):
            b0 = sq * T
            plen = int(pos[b0])
            exp_k_chunk = jnp.stack(k_rows[b0 : b0 + T], axis=0).astype(BF16)
            exp_v_chunk = jnp.stack(v_rows[b0 : b0 + T], axis=0).astype(BF16)
            np.testing.assert_allclose(kc_out_r[sq, plen : plen + T], exp_k_chunk)
            np.testing.assert_allclose(vc_out_r[sq, plen : plen + T], exp_v_chunk)

        yo_rows = []
        for b in range(B):
            sq, tb = divmod(b, T)
            plen = int(pos[sq * T])
            k_valid = jnp.concatenate(
                [
                    kc_r[sq, :plen].astype(F32),
                    jnp.stack(k_rows[sq * T : sq * T + tb + 1], axis=0),
                ],
                axis=0,
            )
            v_valid = jnp.concatenate(
                [
                    vc_r[sq, :plen].astype(F32),
                    jnp.stack(v_rows[sq * T : sq * T + tb + 1], axis=0),
                ],
                axis=0,
            )
            scores_b = q_rows[b].astype(BF16).astype(F32) @ k_valid.T
            probs_b = jax.nn.softmax(scores_b, axis=-1)
            att_b = round_bf16(probs_b @ v_valid).reshape(nh * g.HD)
            mix_b = round_bf16(att_b * gate_rows[b]).astype(BF16).astype(F32)
            yo_raw = mix_b @ w_out_r
            yo_b = (
                (yo_raw * out_s[r, att_idx] if out_s is not None else yo_raw)
                .astype(BF16)
                .astype(F32)
            )
            yo_rows.append(yo_b)
        expected_delta = expected_delta + jnp.stack(yo_rows, axis=0)

    np.testing.assert_allclose(delta, expected_delta, rtol=1e-2, atol=1e-2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(delta + 0.5, expected_delta, rtol=1e-2, atol=1e-2)
