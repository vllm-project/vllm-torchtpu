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
"""KDA options on `GDNModule`: per-key decay, bounded sigmoid decay, sigmoid gate."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.gdn.v3 import compute_kda
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_v3_cfg
from vllm_torchtpu.megakernel import kernel, load_logical_weights
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import (
    qwen35,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.modules.gdn.decls import gdn_hf_params
from vllm_torchtpu.megakernel.modules.gdn.reference import gdn_reference
from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel.pool.config import CacheExec
from vllm_torchtpu.megakernel.weights import MappingWeightSource, load_weights
from vllm_torchtpu.megakernel.weights.ops import JaxArrayOps, slice_alias_targets

R = 32
U_BF16 = 2.0**-8
CTX = 2048
POS0 = 1800
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540

_SAMPLE_HF_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 128,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def _tiny_gdn_geometry() -> qwen35.Qwen35Geometry:
    """4-layer GDN test geometry."""
    return qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=32,
        NKV=4,
        HD=128,
        ROT=32,
        KH=16,
        VH=32,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=2048,
        moe_tp=False,
    )


def _rms(x) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(x.astype(F32)))))


def _report(name: str, got, exp) -> None:
    err = jnp.abs(got.astype(F32) - exp.astype(F32))
    print(
        f"{name}: max_err={float(err.max()):.3e} rms_err={_rms(err):.3e} "
        f"rms_ref={_rms(exp):.3e}"
    )


def _max_abs(got, exp) -> float:
    return float(jnp.max(jnp.abs(got.astype(F32) - exp.astype(F32))))


def _layer_bound(exp) -> float:
    """1 output + 4 recurrence/norm/gate + 4 rank-sum BF16 rounds at the output peak."""
    return 9.0 * U_BF16 * max(1e-4, float(jnp.max(jnp.abs(exp.astype(F32)))))


def _step_bound(exp, n_layers: int) -> float:
    """8 BF16 rounds per layer at RMS scale, in quadrature over independent layers."""
    return n_layers**0.5 * 8.0 * U_BF16 * max(1e-4, _rms(exp))


def _check(name: str, got, exp, controls, bound: float) -> None:
    """Assert max-abs error <= bound and each control misses by > 3 * bound."""
    _report(name, got, exp)
    err = _max_abs(got, exp)
    assert err <= bound, f"{name}: max_abs={err:.4e} > bound={bound:.4e}"
    for label, bad in controls:
        _report(f"{name} vs {label}", bad, exp)
        miss = _max_abs(bad, exp)
        assert miss > 3.0 * bound, (
            f"{name}: {label} max_abs={miss:.4e} <= 3 * {bound:.4e}"
        )


def _swap(x, a: int, b: int):
    return x.at[a].set(x[b]).at[b].set(x[a])


def test_gdn_block_with_kda_key_decay_and_gate_in_process():
    """Verify GDNModule KDA per-key decay and sigmoid gate numerical parity."""
    gb_default = GDNModule(prefix="linear_attn")
    gb_bound = GDNModule.bind(prefix="linear_attn")
    assert dict(gb_default.options) == dict(gb_bound.options)
    gb_kda = GDNModule(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    assert gb_kda.options["decay_type"] == "key"

    g_base = _tiny_gdn_geometry()
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )

    class _KDAGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    class _HeadBSigGeom(qwen35.Qwen35Geometry):
        decay_type: str = "head"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    base_fields = {
        f.name: getattr(g_base, f.name) for f in g_base.__dataclass_fields__.values()
    }
    g_kda = _KDAGeom(**base_fields)
    g_head_bsig = _HeadBSigGeom(**base_fields)
    lay_kda = GDNModule.lin_in_layout(g_kda, ex8)
    assert lay_kda.vh * lay_kda.KD == lay_kda.A_DIM
    assert gb_kda.weights(g_base, ex8)[0].N == lay_kda.NLI
    assert gb_kda.smalls(g_base, ex8)[4].width == 128
    assert gb_kda.smalls(g_base, ex8)[5].width == lay_kda.A_DIM
    lay_sc = GDNModule.lin_in_layout(g_base, ex8)
    _, hid_l, w_sc, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(92), g_base, ex8
    )
    out_sc, _ = gdn_reference(w_sc, hid_l, st_l, g=g_base, ex=ex8, gdn_idx=0)
    out_head_bsig, _ = gdn_reference(
        w_sc, hid_l, st_l, g=g_head_bsig, ex=ex8, gdn_idx=0
    )

    a_off = lay_sc.A_OFF
    vh = lay_sc.vh
    kd = lay_kda.KD
    w_in_prefix = w_sc["lin_in"][..., :a_off]
    w_in_a_rep = jnp.repeat(w_sc["lin_in"][..., a_off : a_off + vh], kd, axis=-1)
    w_in_cat = jnp.concatenate([w_in_prefix, w_in_a_rep], axis=-1)
    pad_cols = lay_kda.NLI - w_in_cat.shape[-1]
    s_prefix = w_sc["lin_in_s"][..., :a_off]
    s_a_rep = jnp.repeat(w_sc["lin_in_s"][..., a_off : a_off + vh], kd, axis=-1)
    s_cat = jnp.concatenate([s_prefix, s_a_rep], axis=-1)
    w_kda = dict(w_sc)
    w_kda["lin_in"] = jnp.pad(w_in_cat, ((0, 0), (0, 0), (0, 0), (0, pad_cols)))
    w_kda["lin_in_s"] = jnp.pad(
        s_cat, ((0, 0), (0, 0), (0, pad_cols)), constant_values=0.01
    )
    w_kda["alog"] = w_sc["alog"]
    w_kda["dtb"] = jnp.repeat(w_sc["dtb"], kd, axis=-1)
    out_kda, _ = gdn_reference(w_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    w_bad_kda = dict(w_kda, alog=w_kda["alog"] + 3.0)
    out_bad_kda, _ = gdn_reference(w_bad_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    _check(
        "kda gdn_reference vs head bounded_sigmoid",
        out_kda,
        out_head_bsig,
        [("softplus/silu control", out_sc), ("perturbed alog", out_bad_kda)],
        _layer_bound(out_head_bsig),
    )
    devs = jax.devices("cpu")
    if len(devs) >= ex8.R:
        pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
        ref_hid_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_kda, st_l)
        ref_hid_bad_kda, _ = qwen35.reference_step(
            g_kda, ex8, pos, hid_l, w_bad_kda, st_l
        )
        mesh = Mesh(devs[: ex8.R], ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_kda, st_l, g_kda, ex8, mesh=mesh
        )
        step_kda, _ = qwen35.make_step(mesh, g_kda, ex8, donate_states=True)
        out_hid_kda, _ = step_kda(pos_mk, hid_mk, w_mk, st_mk)
        hid_kda_r = kernel.unfold(out_hid_kda, ex8.R)[:, : ex8.B]
        _check(
            "kda make_step vs reference_step",
            hid_kda_r[0],
            ref_hid_kda,
            [("perturbed alog", ref_hid_bad_kda), ("input hidden", hid_l)],
            _step_bound(ref_hid_kda, g_kda.L),
        )


def _kimi_k3_kda_oracle_gdn_layer(
    w_kda: dict[str, jax.Array],
    hid_l: jax.Array,
    st_l: dict[str, jax.Array],
    g_kda: Any,
    ex: ExecSpec,
    *,
    gdn_idx: int = 0,
    wrong_axis: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Float64 reference for one KDA recurrent layer."""

    R = ex.R
    B = ex.B
    H = g_kda.H
    kh = g_kda.kh(R)
    vh = g_kda.vh(R)
    KD = g_kda.KD
    VD = g_kda.VD
    lay = GDNModule.lin_in_layout(g_kda, ex)
    QKV, Z, BA, A_OFF = lay.QKV, lay.Z, lay.BA, lay.A_OFF
    a_dim = vh * KD
    expand = vh // kh

    # gdn_reference receives x directly (input norm is applied by Norm module)
    norm_x = np.asarray(hid_l[:B].astype(jnp.bfloat16), dtype=np.float32).astype(
        np.float64
    )

    lin_in_all = np.asarray(w_kda["lin_in"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_in_s_all = np.asarray(w_kda["lin_in_s"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    conv_w_all = np.asarray(w_kda["conv"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    alog_all = np.asarray(w_kda["alog"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    dtb_all = np.asarray(w_kda["dtb"][:, gdn_idx], dtype=np.float64)
    gnorm_all = np.asarray(w_kda["gnorm"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_out_all = np.asarray(w_kda["lin_out"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_out_s_all = np.asarray(w_kda["lin_out_s"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    conv_st_all = np.asarray(st_l["conv"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    rec_st_all = np.asarray(st_l["rec"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )

    decay_law = str(getattr(g_kda, "decay_law", "softplus"))
    gate_type = str(getattr(g_kda, "gate_type", "silu"))
    decay_bound = float(getattr(g_kda, "decay_bound", 5.0))

    rank_partials = []
    rec_next_ranks = []
    for r in range(R):
        w_in_r = lin_in_all[r].reshape(-1, lay.NLI)
        ybuf = (norm_x @ w_in_r) * lin_in_s_all[r][None, :]
        # Causal depthwise 4-tap conv1d + SiLU on QKV
        c_st = conv_st_all[r]  # [B, 3, QKV]
        cw = conv_w_all[r]  # [4 * QKV]
        taps = [cw[t * QKV : (t + 1) * QKV] for t in range(4)]
        conv_out = (
            c_st[:, 0, :] * taps[0]
            + c_st[:, 1, :] * taps[1]
            + c_st[:, 2, :] * taps[2]
            + ybuf[:, :QKV] * taps[3]
        )
        u = conv_out / (1.0 + np.exp(-conv_out))

        alog = alog_all[r, :vh]
        exp_alog = np.exp(alog)[:, None]
        dtb = dtb_all[r, :a_dim]
        gnw = gnorm_all[r, :VD]
        mix_rows = []
        rec_b_list = []
        for b in range(B):
            st_cur = (
                rec_st_all[r, b].copy()
                if rec_st_all.ndim == 5
                else rec_st_all[r, b * vh : (b + 1) * vh].copy()
            )  # [vh, KD, VD]
            u_b = u[b]
            q = u_b[: kh * KD].reshape(kh, KD)
            k = u_b[kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = u_b[2 * kh * KD : QKV].reshape(vh, VD)
            q_den = np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + g_kda.gdn_l2_eps)
            k_den = np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + g_kda.gdn_l2_eps)
            q = q * (KD**-0.5) / q_den
            k = k / k_den
            qc = np.repeat(q, expand, axis=0)[:, :, None]
            kc = np.repeat(k, expand, axis=0)[:, :, None]
            beta = 1.0 / (1.0 + np.exp(-ybuf[b, BA : BA + vh]))
            a_val = ybuf[b, A_OFF : A_OFF + a_dim]
            a_shifted = (a_val + dtb).reshape(vh, KD)
            if decay_law == "bounded_sigmoid":
                sig_arg = exp_alog * a_shifted
                decay_mat = -decay_bound * (1.0 / (1.0 + np.exp(-sig_arg)))
            else:
                decay_mat = -exp_alog * np.logaddexp(0.0, a_shifted)
            if wrong_axis:
                # Deliberate axis bug: broadcast KD decay across first KD cols of VD
                eg_bad = np.ones((vh, 1, VD), dtype=np.float64)
                eg_bad[:, 0, :KD] = np.exp(decay_mat)
                st_cur = st_cur * eg_bad
            else:
                eg = np.exp(decay_mat)[:, :, None]
                st_cur = st_cur * eg
            pred = np.sum(st_cur * kc, axis=1)
            delta = (v - pred) * beta[:, None]
            st_cur = st_cur + kc * delta[:, None, :]
            rec_b_list.append(st_cur)
            out_h = np.sum(st_cur * qc, axis=1)
            val = out_h / np.sqrt(
                np.mean(out_h * out_h, axis=-1, keepdims=True) + g_kda.eps
            )
            val = gnw[None, :] * val
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            gate_val = (
                (1.0 / (1.0 + np.exp(-zz)))
                if gate_type == "sigmoid"
                else (zz / (1.0 + np.exp(-zz)))
            )
            mix_rows.append((val * gate_val).reshape(Z))
        mix = np.stack(mix_rows, axis=0)
        w_out_r = lin_out_all[r].reshape(Z, H)
        yo = (mix @ w_out_r) * lin_out_s_all[r][None, :]
        rank_partials.append(yo)
        rec_next_ranks.append(np.concatenate(rec_b_list, axis=0))

    y_sum = np.sum(rank_partials, axis=0)
    return (
        jnp.asarray(y_sum, dtype=F32),
        jnp.asarray(np.stack(rec_next_ranks, axis=0), dtype=F32),
    )


def test_gdn_kda_step_matches_kimi_kda_step():
    """Verify rectangular KD=128 != VD=256 KDA step matches float64 reference."""
    # Reject non-positive decay_bound.
    with pytest.raises(MegakernelConfigError, match="decay_bound > 0"):
        GDNModule(decay_bound=0.0)
    with pytest.raises(MegakernelConfigError, match="decay_bound > 0"):
        GDNModule(decay_bound=-3.0)

    class _RectKDAGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    g_base = dc_replace(_tiny_gdn_geometry(), KD=128, VD=256)
    base_fields = {
        f.name: getattr(g_base, f.name) for f in g_base.__dataclass_fields__.values()
    }
    g_kda = _RectKDAGeom(**base_fields)
    assert g_kda.KD == 128 and g_kda.VD == 256 and g_kda.KD != g_kda.VD

    # Explicit module option overrides geometry attribute.
    mod_default = GDNModule(prefix="linear_attn")
    assert mod_default.resolved_options(g_kda)["decay_type"] == "key"
    mod_explicit_head = GDNModule(prefix="linear_attn", decay_type="head")
    assert mod_explicit_head.resolved_options(g_kda)["decay_type"] == "head"

    b_default = GDNModule.bind(prefix="linear_attn")
    m_from_default_bind: Any = b_default.resolve(_SAMPLE_HF_CONFIG).to_module()
    assert m_from_default_bind.resolved_options(g_kda)["decay_type"] == "key"

    b_explicit_head = GDNModule.bind(prefix="linear_attn", decay_type="head")
    m_from_explicit_bind: Any = b_explicit_head.resolve(_SAMPLE_HF_CONFIG).to_module()
    assert m_from_explicit_bind.resolved_options(g_kda)["decay_type"] == "head"

    # Per-key decay widens lin_a and dt_bias to VH * KD while A_log stays (VH,).
    decls = {d.name: d for d in gdn_hf_params("model.layers.0.linear_attn", g_kda)}
    a_dim = g_kda.VH * g_kda.KD
    assert decls["lin_a"].shape == (g_kda.H, a_dim)
    assert decls["lin_a"].transform.pad is None
    assert decls["alog"].shape == (g_kda.VH,) and decls["dtb"].shape == (a_dim,)
    head_decls = {
        d.name: d for d in gdn_hf_params("model.layers.0.linear_attn", g_base)
    }
    assert head_decls["lin_a"].shape == (g_base.H, g_base.VH)
    assert head_decls["alog"].shape == (g_base.VH,)
    ctor_decls = {
        d.name: d
        for d in GDNModule(prefix="linear_attn", decay_type="key").hf_params(
            "model.layers.0.linear_attn", g=g_base
        )
    }
    # Full decl parity pins slot pads (lin_b 128-lane slot), not only shapes.
    assert {n: (d.shape, d.hf_shape, d.transform) for n, d in ctor_decls.items()} == {
        n: (d.shape, d.hf_shape, d.transform) for n, d in decls.items()
    }
    assert ctor_decls["lin_a"].shape != head_decls["lin_a"].shape

    # Rectangular KD=128 != VD=256 with distinct per-key decay columns.
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    lay_sc = GDNModule.lin_in_layout(g_base, ex8)
    lay_kda = GDNModule.lin_in_layout(g_kda, ex8)
    assert lay_kda.vh * 128 == lay_kda.A_DIM

    _, hid_l, w_sc, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(2026), g_base, ex8
    )
    a_off = lay_sc.A_OFF
    a_dim = lay_kda.A_DIM
    rng = np.random.default_rng(2026)

    # Construct independent random per-key a columns (NOT repeated head scalars)
    w_in_prefix = np.asarray(w_sc["lin_in"][..., :a_off])
    R_cnt, L_cnt, H_dim, _ = w_in_prefix.shape
    a_rand_fp8 = jnp.asarray(
        rng.standard_normal((R_cnt, L_cnt, H_dim, a_dim), dtype=np.float32) * 0.5,
        dtype=w_sc["lin_in"].dtype,
    )
    pad_cols = lay_kda.NLI - (a_off + a_dim)
    w_in_kda = jnp.concatenate(
        [
            w_sc["lin_in"][..., :a_off],
            a_rand_fp8,
            jnp.zeros((R_cnt, L_cnt, H_dim, pad_cols), dtype=w_sc["lin_in"].dtype),
        ],
        axis=-1,
    )
    s_a_rand = jnp.asarray(
        np.abs(rng.standard_normal((R_cnt, L_cnt, a_dim), dtype=np.float32)) * 0.2
        + 0.05,
        dtype=F32,
    )
    s_in_kda = jnp.concatenate(
        [
            w_sc["lin_in_s"][..., :a_off],
            s_a_rand,
            jnp.ones((R_cnt, L_cnt, pad_cols), dtype=F32) * 0.01,
        ],
        axis=-1,
    )
    alog_kda = jnp.asarray(
        rng.uniform(-0.5, 1.5, size=(R_cnt, L_cnt, lay_kda.vh)).astype(np.float32),
        dtype=F32,
    )
    dtb_kda = jnp.asarray(
        rng.standard_normal((R_cnt, L_cnt, a_dim), dtype=np.float32) * 1.5,
        dtype=F32,
    )

    w_kda = dict(w_sc)
    w_kda["lin_in"] = w_in_kda
    w_kda["lin_in_s"] = s_in_kda
    w_kda["alog"] = alog_kda
    w_kda["dtb"] = dtb_kda

    out_ref, st_ref = gdn_reference(w_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    gb_kda = GDNModule(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    out_mod_ref, _ = gb_kda.reference(w_kda, hid_l, st_l, g=g_base, ex=ex8, gdn_idx=0)
    np.testing.assert_allclose(
        np.asarray(out_mod_ref, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    out_oracle, st_oracle = _kimi_k3_kda_oracle_gdn_layer(
        w_kda, hid_l, st_l, g_kda, ex8, gdn_idx=0, wrong_axis=False
    )
    out_wrong_axis, _ = _kimi_k3_kda_oracle_gdn_layer(
        w_kda, hid_l, st_l, g_kda, ex8, gdn_idx=0, wrong_axis=True
    )
    _check(
        "rectangular KD=128 != VD=256 gdn_reference vs float64 Kimi K3 oracle",
        out_ref,
        out_oracle,
        [("wrong VD-axis decay broadcast", out_wrong_axis)],
        _layer_bound(out_oracle),
    )
    rec_ref_out = (
        st_ref["rec_out"][:, 0]
        if "rec_out" in st_ref
        else (st_ref["rec"][:, 0] if st_ref["rec"].ndim == 5 else st_ref["rec"])
    )
    # BF16 k, v, beta, and decay-gate inputs: 4 rounds at the state peak.
    st_ref_f32 = np.asarray(rec_ref_out, dtype=np.float32)
    st_oracle_f32 = np.asarray(st_oracle, dtype=np.float32)
    st_bound = 4.0 * U_BF16 * max(1e-4, float(np.max(np.abs(st_oracle_f32))))
    st_err = float(np.max(np.abs(st_ref_f32 - st_oracle_f32)))
    print(f"rec state vs float64 oracle: max_abs={st_err:.3e} bound={st_bound:.3e}")
    assert st_err <= st_bound

    # Full 8-rank Pallas make_step vs reference_step on rectangular KD=128 != VD=256
    devs = jax.devices("cpu")
    assert len(devs) >= ex8.R
    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_kda, st_l)
    w_bad_kda = dict(w_kda, dtb=w_kda["dtb"] + 4.0)
    ref_hid_bad_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_bad_kda, st_l)
    mesh = Mesh(devs[: ex8.R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_kda, st_l, g_kda, ex8, mesh=mesh
    )
    step_kda, _ = qwen35.make_step(mesh, g_kda, ex8, donate_states=True)
    out_hid_kda, _ = step_kda(pos_mk, hid_mk, w_mk, st_mk)
    hid_kda_r = kernel.unfold(out_hid_kda, ex8.R)[:, : ex8.B]
    _check(
        "rectangular KD=128 != VD=256 make_step vs reference_step",
        hid_kda_r[0],
        ref_hid_kda,
        [("perturbed dtb", ref_hid_bad_kda), ("input hidden", hid_l)],
        _step_bound(ref_hid_kda, g_kda.L),
    )


def test_gdn_options_are_plain_values_with_one_resolver():
    """Requested options stay plain values; explicit > geometry > default."""
    b = GDNModule.bind(prefix="linear_attn")
    assert all(v is None for v in b.options.values())
    m = GDNModule(prefix="linear_attn", decay_type="key")
    assert type(m.options["decay_type"]) is str and m.options["decay_law"] is None
    assert m.resolved_options(None) == {
        "decay_type": "key",
        "decay_law": "softplus",
        "gate_type": "silu",
        "decay_bound": 5.0,
    }


def _production_compute_kda_layer_oracle(
    w_kda: dict[str, jax.Array],
    hid_l: jax.Array,
    st_l: dict[str, jax.Array],
    g_kda: Any,
    ex: ExecSpec,
    *,
    gdn_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-verify one GDNModule KDA layer via production `compute_kda`."""
    r_cnt, b_cnt, t_cnt, ns_cnt = ex.R, ex.B, ex.T, ex.NS
    kh = int(g_kda.kh(r_cnt))
    vh = int(g_kda.vh(r_cnt))
    kd = int(g_kda.KD)
    vd = int(g_kda.VD)
    lay = GDNModule.lin_in_layout(g_kda, ex)
    qkv, z_dim, ba_off, a_off = lay.QKV, lay.Z, lay.BA, lay.A_OFF
    a_dim = vh * kd
    decay_law = str(getattr(g_kda, "decay_law", "softplus"))
    gate_type = str(getattr(g_kda, "gate_type", "silu"))
    decay_bound = float(getattr(g_kda, "decay_bound", 5.0))
    gate_lb = -decay_bound if decay_law == "bounded_sigmoid" else None

    cfgs = gdn_v3_cfg.GDNConfig(
        mode=gdn_v3_cfg.GDNMode.BATCHED,
        dtypes=gdn_v3_cfg.Dtypes(
            act_in=jnp.bfloat16,
            act_out=jnp.bfloat16,
            compute=jnp.float32,
            recurrent_state=jnp.bfloat16,
            conv_state=jnp.bfloat16,
        ),
        batch_size=1,
        dim_size=qkv,
        kernel_size=4,
        tile_size=1,
        num_kq_heads=kh,
        num_v_heads=vh,
        kq_head_dim=kd,
        v_head_dim=vd,
        attention_mode=gdn_v3_cfg.AttentionMode.KDA,
        gate_lower_bound=gate_lb,
        window_size=t_cnt,
    )

    xb = hid_l[:b_cnt].astype(jnp.bfloat16).astype(F32)
    rank_partials: list[np.ndarray] = []
    rec_final_ranks: list[np.ndarray] = []

    for r in range(r_cnt):
        w_in_r = w_kda["lin_in"][r, gdn_idx].astype(F32)
        in_s_r = w_kda["lin_in_s"][r, gdn_idx].astype(F32)
        ybuf = (xb @ w_in_r) * in_s_r[None, :]

        cw = w_kda["conv"][r, gdn_idx].astype(F32)
        taps = [cw[tap * qkv : (tap + 1) * qkv] for tap in range(4)]
        c_st_raw = st_l["conv"][r, gdn_idx]
        c_st = (
            c_st_raw[: 3 * ns_cnt].reshape(ns_cnt, 3, qkv)
            if c_st_raw.ndim == 2
            else c_st_raw[:ns_cnt]
        ).astype(F32)

        u_rows: list[jax.Array] = []
        for s_idx in range(ns_cnt):
            hist = [c_st[s_idx, 0], c_st[s_idx, 1], c_st[s_idx, 2]]
            for tb in range(t_cnt):
                b_row = s_idx * t_cnt + tb
                x_cur = ybuf[b_row, :qkv]
                conv_val = (
                    hist[0] * taps[0]
                    + hist[1] * taps[1]
                    + hist[2] * taps[2]
                    + x_cur * taps[3]
                )
                u_rows.append(jax.nn.silu(conv_val))
                hist = [hist[1], hist[2], x_cur]
        u_all = jnp.stack(u_rows, axis=0)

        alog_r = w_kda["alog"][r, gdn_idx, :vh].astype(F32)
        dtb_r = w_kda["dtb"][r, gdn_idx, :a_dim].astype(F32)
        gnw_r = w_kda["gnorm"][r, gdn_idx, :vd].astype(jnp.bfloat16).astype(F32)
        rec_raw = st_l["rec"][r, gdn_idx]
        rec_init = (
            rec_raw[:ns_cnt]
            if rec_raw.ndim == 4
            else rec_raw[: ns_cnt * vh].reshape(ns_cnt, vh, kd, vd)
        ).astype(F32)

        mix_seq: list[jax.Array] = []
        rec_seq: list[np.ndarray] = []
        for s_idx in range(ns_cnt):
            sl = slice(s_idx * t_cnt, (s_idx + 1) * t_cnt)
            u_seq = u_all[sl]
            q_seq = u_seq[:, : kh * kd].reshape(t_cnt, kh, kd).swapaxes(0, 1)
            k_seq = (
                u_seq[:, kh * kd : 2 * kh * kd].reshape(t_cnt, kh, kd).swapaxes(0, 1)
            )
            v_seq = u_seq[:, 2 * kh * kd : qkv].reshape(t_cnt, vh, vd).swapaxes(0, 1)
            q_norm = (
                q_seq
                * jax.lax.rsqrt(
                    jnp.sum(q_seq * q_seq, axis=-1, keepdims=True) + g_kda.gdn_l2_eps
                )
                * (kd**-0.5)
            )
            k_norm = k_seq * jax.lax.rsqrt(
                jnp.sum(k_seq * k_seq, axis=-1, keepdims=True) + g_kda.gdn_l2_eps
            )
            b_seq = (
                jax.nn.sigmoid(ybuf[sl, ba_off : ba_off + vh])
                .swapaxes(0, 1)[:, :, None]
                .astype(jnp.bfloat16)
                .astype(F32)
            )
            a_seq = (
                ybuf[sl, a_off : a_off + a_dim].reshape(t_cnt, vh, kd).swapaxes(0, 1)
            )
            gating_log = compute_kda.activate_gate(
                a_seq,
                alog_r.reshape(vh, 1, 1),
                dtb_r.reshape(vh, 1, kd),
                cfgs,
            )
            gating_decay = jnp.exp(gating_log)
            out_seq, st_seq = compute_kda.recurrent_kda_per_seq(
                q_norm,
                k_norm,
                v_seq,
                gating_decay,
                b_seq,
                rec_init[s_idx],
                cfgs,
            )
            rec_seq.append(np.asarray(st_seq[-1], dtype=np.float64))
            for tb in range(t_cnt):
                out_h = out_seq[tb]
                rms_inv = jax.lax.rsqrt(
                    jnp.mean(out_h * out_h, axis=-1, keepdims=True) + g_kda.eps
                )
                val = out_h * rms_inv * gnw_r[None, :]
                zz = ybuf[s_idx * t_cnt + tb, qkv : qkv + z_dim].reshape(vh, vd)
                g_act = (
                    jax.nn.sigmoid(zz) if gate_type == "sigmoid" else jax.nn.silu(zz)
                )
                mix_seq.append((val * g_act).reshape(z_dim))

        mix_r = jnp.stack(mix_seq, axis=0).astype(jnp.bfloat16).astype(F32)
        w_out_r = w_kda["lin_out"][r, gdn_idx].astype(F32)
        out_s_r = w_kda["lin_out_s"][r, gdn_idx].astype(F32)
        part_r = ((mix_r @ w_out_r) * out_s_r[None, :]).astype(jnp.bfloat16).astype(F32)
        rank_partials.append(np.asarray(part_r, dtype=np.float64))
        rec_final_ranks.append(np.concatenate(rec_seq, axis=0))

    return np.sum(rank_partials, axis=0), np.stack(rec_final_ranks, axis=0)


def _build_kda_weights_for_geom(
    seed: int, g_base: Any, g_kda: Any, ex: ExecSpec
) -> tuple[jax.Array, dict[str, jax.Array], dict[str, jax.Array]]:
    """Construct random KDA logical inputs with 1D `(vh,)` `alog`."""
    _, hid_l, w_sc, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(seed), g_base, ex
    )
    lay_sc = GDNModule.lin_in_layout(g_base, ex)
    lay_kda = GDNModule.lin_in_layout(g_kda, ex)
    a_off = lay_sc.A_OFF
    a_dim = lay_kda.A_DIM
    rng = np.random.default_rng(seed + 100)
    r_cnt, l_cnt, h_dim, _ = w_sc["lin_in"].shape
    a_fp8 = jnp.asarray(
        rng.standard_normal((r_cnt, l_cnt, h_dim, a_dim), dtype=np.float32) * 0.35,
        dtype=w_sc["lin_in"].dtype,
    )
    pad_cols = lay_kda.NLI - (a_off + a_dim)
    w_in_kda = jnp.concatenate(
        [
            w_sc["lin_in"][..., :a_off],
            a_fp8,
            jnp.zeros((r_cnt, l_cnt, h_dim, pad_cols), dtype=w_sc["lin_in"].dtype),
        ],
        axis=-1,
    )
    s_a = jnp.asarray(
        np.abs(rng.standard_normal((r_cnt, l_cnt, a_dim), dtype=np.float32)) * 0.1
        + 0.02,
        dtype=F32,
    )
    s_in_kda = jnp.concatenate(
        [
            w_sc["lin_in_s"][..., :a_off],
            s_a,
            jnp.ones((r_cnt, l_cnt, pad_cols), dtype=F32) * 0.01,
        ],
        axis=-1,
    )
    alog_1d = jnp.asarray(
        rng.uniform(-0.8, 1.1, size=(r_cnt, l_cnt, lay_kda.vh)).astype(np.float32),
        dtype=F32,
    )
    dtb_2d = jnp.asarray(
        rng.standard_normal((r_cnt, l_cnt, a_dim), dtype=np.float32) * 1.1,
        dtype=F32,
    )
    w_kda = dict(
        w_sc,
        lin_in=w_in_kda,
        lin_in_s=s_in_kda,
        alog=alog_1d,
        dtb=dtb_2d,
    )
    return hid_l, w_kda, st_l


def test_kda_matches_production_compute_kda_and_alog_1d():
    """Verify 1D `(VH,)` `A_log`, loader roundtrip, and parity vs `compute_kda`."""
    u_bf16 = 2.0**-8

    class _KDASoftplusGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "softplus"
        gate_type: str = "silu"
        decay_bound: float = 5.0

    class _KDABoundedGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    def _as_geom(cls_g: Any, g_src: Any) -> Any:
        fields = {
            f.name: getattr(g_src, f.name) for f in g_src.__dataclass_fields__.values()
        }
        return cls_g(**fields)

    g_base = _tiny_gdn_geometry()
    g_kda = _as_geom(_KDABoundedGeom, g_base)

    # 1. 1D (VH,) A_log, (VH * KD,) dt_bias decls + 128-lane VMEM smalls
    decls = {d.name: d for d in gdn_hf_params("model.layers.0.linear_attn", g_kda)}
    a_dim_full = g_kda.VH * g_kda.KD
    assert decls["alog"].shape == (g_kda.VH,)
    assert decls["dtb"].shape == (a_dim_full,)
    assert decls["lin_a"].shape == (g_kda.H, a_dim_full)
    ex_decl = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    gb_kda = GDNModule(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    sm_map = {sf.name: sf.width for sf in gb_kda.smalls(g_base, ex_decl)}
    assert sm_map["alog"] == 128
    assert sm_map["dtb"] == g_base.vh(8) * g_base.KD

    # 2. load_logical_weights + load_weights + in_proj_ba alias
    hf_raw = dict(qwen35.random_hf_weights(jax.random.PRNGKey(77), g_base))
    rng_loader = np.random.default_rng(77)
    hf_raw["lin_a"] = jnp.asarray(
        rng_loader.standard_normal((g_kda.NGDN, g_kda.H, a_dim_full), dtype=np.float32)
        * 0.2,
        dtype=hf_raw["lin_a"].dtype,
    )
    hf_raw["lin_a_s"] = jnp.asarray(
        np.abs(rng_loader.standard_normal((g_kda.NGDN, a_dim_full), dtype=np.float32))
        * 0.02
        + 0.01,
        dtype=F32,
    )
    hf_raw["dtb"] = jnp.asarray(
        rng_loader.standard_normal((g_kda.NGDN, a_dim_full), dtype=np.float32) * 0.1,
        dtype=F32,
    )
    assert hf_raw["alog"].shape == (g_kda.NGDN, g_kda.VH)
    loaded_l = load_logical_weights(qwen35.Qwen35, ex_decl, hf_raw, g=g_kda)
    assert loaded_l["alog"].shape == (8, g_kda.NGDN, g_kda.vh(8))
    assert loaded_l["dtb"].shape == (8, g_kda.NGDN, g_kda.vh(8) * g_kda.KD)
    w_rank0 = load_weights(
        qwen35.Qwen35,
        ex_decl,
        MappingWeightSource(loaded_l),
        g=g_kda,
        tp_rank=0,
    )
    alog_off, alog_w = w_rank0["plan"].smalls_layout["alog"]
    assert alog_w == 128
    assert np.allclose(
        np.asarray(w_rank0["smalls"][0, 0, alog_off : alog_off + g_kda.vh(8)]),
        np.asarray(loaded_l["alog"][0, 0]),
    )

    # Widened A_log `(VH * KD,)` must be rejected by `load_logical_weights`
    hf_bad_alog = dict(
        hf_raw,
        alog=jnp.ones((g_kda.NGDN, a_dim_full), dtype=F32),
    )
    with pytest.raises(ValueError):
        load_logical_weights(qwen35.Qwen35, ex_decl, hf_bad_alog, g=g_kda)

    # Verify `slice_alias_targets` on `in_proj_ba` pre-pad slicing
    ba_decl_b = decls["lin_b"]
    ba_decl_a = decls["lin_a"]
    ba_alias = ba_decl_b.vllm_aliases[0]
    vh_r8 = g_kda.vh(8)
    ba_rank_buf = jnp.arange(g_kda.H * (vh_r8 + vh_r8 * g_kda.KD), dtype=F32).reshape(
        g_kda.H, vh_r8 + vh_r8 * g_kda.KD
    )
    sliced_ba = slice_alias_targets(
        JaxArrayOps(),
        ba_alias,
        ba_rank_buf,
        None,
        (ba_decl_b, ba_decl_a),
        {"lin_b_s": decls["lin_b_s"], "lin_a_s": decls["lin_a_s"]},
        R=8,
        tp_rank=0,
    )
    assert sliced_ba["lin_b"].shape == (g_kda.H, 128)
    assert sliced_ba["lin_a"].shape == (g_kda.H, vh_r8 * g_kda.KD)
    # Random columns: a shifted slice changes the FP8 bits, not just shapes.
    ops_j = JaxArrayOps()
    ba_rand = jax.random.normal(jax.random.PRNGKey(7), ba_rank_buf.shape, dtype=F32)
    sliced_rand = slice_alias_targets(
        ops_j,
        ba_alias,
        ba_rand,
        None,
        (ba_decl_b, ba_decl_a),
        {"lin_b_s": decls["lin_b_s"], "lin_a_s": decls["lin_a_s"]},
        R=8,
        tp_rank=0,
    )
    exp_b, _ = ops_j.quantize_fp8(ba_rand[:, :vh_r8])
    exp_a, _ = ops_j.quantize_fp8(ba_rand[:, vh_r8:])
    got_b = np.asarray(sliced_rand["lin_b"], dtype=np.float32)
    np.testing.assert_array_equal(got_b[:, :vh_r8], np.asarray(exp_b, np.float32))
    assert not np.any(got_b[:, vh_r8:])
    np.testing.assert_array_equal(
        np.asarray(sliced_rand["lin_a"], dtype=np.float32),
        np.asarray(exp_a, dtype=np.float32),
    )

    # 3. Odd B=3 across 3 seeds + Kimi-K3 vh=3 + KD=128 vs compute_kda
    devs = jax.devices("cpu")
    assert len(devs) >= 8
    for seed in (0, 17, 42):
        for g_cls in (_KDASoftplusGeom, _KDABoundedGeom):
            g_cur = _as_geom(g_cls, g_base)
            ex_odd = ExecSpec(
                B=3,
                T=1,
                R=8,
                ctx=256,
                tiles=dc_replace(
                    TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False
                ),
                interpret=True,
            )
            hid_l, w_kda, st_l = _build_kda_weights_for_geom(
                seed, g_base, g_cur, ex_odd
            )
            out_ref, _ = gdn_reference(
                w_kda, hid_l, st_l, g=g_cur, ex=ex_odd, gdn_idx=0
            )
            out_prod, st_prod = _production_compute_kda_layer_oracle(
                w_kda, hid_l, st_l, g_cur, ex_odd, gdn_idx=0
            )
            out_f64, _ = _kimi_k3_kda_oracle_gdn_layer(
                w_kda, hid_l, st_l, g_cur, ex_odd, gdn_idx=0, wrong_axis=False
            )
            out_bad_ax, _ = _kimi_k3_kda_oracle_gdn_layer(
                w_kda, hid_l, st_l, g_cur, ex_odd, gdn_idx=0, wrong_axis=True
            )
            scale = max(1e-4, float(np.max(np.abs(out_prod))))
            # 1 output round + 4 recurrence/norm/gate rounds + 4 rank-sum rounds
            max_abs_bound = 9.0 * u_bf16 * scale
            assert (
                float(np.max(np.abs(np.asarray(out_ref) - out_prod))) <= max_abs_bound
            )
            assert (
                float(np.max(np.abs(np.asarray(out_f64) - out_prod))) <= max_abs_bound
            )
            assert (
                float(np.max(np.abs(np.asarray(out_bad_ax) - out_prod)))
                > 3.0 * max_abs_bound
            )

    # Kimi-K3 TP32 local head ratio (vh=3, kh=3 at R=2) in make_step
    g_k3_base = dc_replace(
        g_base, L=4, att_every=4, NH=8, NKV=2, KH=6, VH=6, KD=128, VD=128, IS=512
    )
    g_k3 = _as_geom(_KDABoundedGeom, g_k3_base)
    ex_k3 = ExecSpec(
        B=3,
        T=1,
        R=2,
        ctx=256,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    assert g_k3.vh(2) == 3 and g_k3.kh(2) == 3
    hid_k3, w_k3, st_k3 = _build_kda_weights_for_geom(31, g_k3_base, g_k3, ex_k3)
    out_k3_ref, _ = gdn_reference(w_k3, hid_k3, st_k3, g=g_k3, ex=ex_k3, gdn_idx=0)
    out_k3_prod, _ = _production_compute_kda_layer_oracle(
        w_k3, hid_k3, st_k3, g_k3, ex_k3, gdn_idx=0
    )
    k3_bound = 9.0 * u_bf16 * max(1e-4, float(np.max(np.abs(out_k3_prod))))
    assert float(np.max(np.abs(np.asarray(out_k3_ref) - out_k3_prod))) <= k3_bound
    pos_k3 = 32 + jnp.arange(ex_k3.NS, dtype=I32)
    mesh2 = Mesh(devs[:2], ("tp",))
    ref_step_k3, _ = qwen35.reference_step(g_k3, ex_k3, pos_k3, hid_k3, w_k3, st_k3)
    p_mk, h_mk, w_mk, s_mk = qwen35.pack_inputs(
        pos_k3, hid_k3, w_k3, st_k3, g_k3, ex_k3, mesh=mesh2
    )
    step_k3, _ = qwen35.make_step(mesh2, g_k3, ex_k3, donate_states=True)
    got_k3, _ = step_k3(p_mk, h_mk, w_mk, s_mk)
    got_k3_h = np.asarray(kernel.unfold(got_k3, 2)[0, : ex_k3.B], dtype=np.float64)
    ref_k3_h = np.asarray(ref_step_k3, dtype=np.float64)
    step_k3_bound = 48.0 * u_bf16 * max(1e-3, float(np.sqrt(np.mean(ref_k3_h**2))))
    assert float(np.max(np.abs(got_k3_h - ref_k3_h))) <= step_k3_bound

    # 4. Chunked recurrence GCH=True (B=18) with HEAD_MAJOR=False/True
    g_hm_base = dc_replace(g_base, KH=8, VH=32)
    g_hm = _as_geom(_KDABoundedGeom, g_hm_base)
    mesh8 = Mesh(devs[:8], ("tp",))
    for hm_flag in (False, True):
        tiles_gch = dc_replace(
            TINY_TILES, bank=256 << 10, vmem_mib=64, hier=0, mb8=3, inplace=False
        ).with_modules({"gdn": {"gdn_rows": 8, "gdn_head_major": hm_flag}})
        ex_gch = ExecSpec(B=18, T=1, R=8, ctx=256, tiles=tiles_gch, interpret=True)
        assert ex_gch.WIDE and g_hm.kh(8) == 1 and g_hm.vh(8) == 4
        hid_gch, w_gch, st_gch = _build_kda_weights_for_geom(
            55, g_hm_base, g_hm, ex_gch
        )
        pos_gch = 40 + jnp.arange(ex_gch.NS, dtype=I32)
        ref_gch, ref_st_gch = qwen35.reference_step(
            g_hm, ex_gch, pos_gch, hid_gch, w_gch, st_gch
        )
        p_gch, h_gch, wm_gch, sm_gch = qwen35.pack_inputs(
            pos_gch, hid_gch, w_gch, st_gch, g_hm, ex_gch, mesh=mesh8
        )
        step_gch, info_gch = qwen35.make_step(mesh8, g_hm, ex_gch, donate_states=True)
        assert info_gch["plan"].module_params["gdn"].GCH is True
        assert info_gch["plan"].module_params["gdn"].head_major is hm_flag
        out_gch, out_st_gch = step_gch(p_gch, h_gch, wm_gch, sm_gch)
        got_gch = np.asarray(kernel.unfold(out_gch, 8)[0, : ex_gch.B], dtype=np.float64)
        exp_gch = np.asarray(ref_gch, dtype=np.float64)
        gch_bound = 48.0 * u_bf16 * max(1e-3, float(np.sqrt(np.mean(exp_gch**2))))
        assert float(np.max(np.abs(got_gch - exp_gch))) <= gch_bound
        got_rec_gch = np.asarray(
            kernel.unfold(out_st_gch["rec"], 8)[:, :, : ex_gch.B * 4], dtype=np.float64
        )
        exp_rec_gch = np.asarray(
            ref_st_gch["rec"][:, :, : ex_gch.B * 4], dtype=np.float64
        )
        # 3 stacked GDN layers x 12 BF16 rounds/layer
        assert float(np.max(np.abs(got_rec_gch - exp_rec_gch))) <= 36.0 * u_bf16 * max(
            1e-4, float(np.max(np.abs(exp_rec_gch)))
        )

    # 5. Speculative verify VER=True (T=4, B=8, NS=2) vs compute_kda
    tiles_ver = dc_replace(
        TINY_TILES, bank=512 << 10, vmem_mib=64, hier=0, mb8=1, inplace=False
    )
    ex_ver = ExecSpec(B=8, T=4, R=8, ctx=256, tiles=tiles_ver, interpret=True)
    assert ex_ver.VER and ex_ver.NS == 2
    hid_ver, w_ver, st_ver = _build_kda_weights_for_geom(88, g_base, g_kda, ex_ver)
    out_ver_ref, st_ver_ref = gdn_reference(
        w_ver, hid_ver, st_ver, g=g_kda, ex=ex_ver, gdn_idx=0
    )
    out_ver_prod, st_ver_prod = _production_compute_kda_layer_oracle(
        w_ver, hid_ver, st_ver, g_kda, ex_ver, gdn_idx=0
    )
    ver_bound = 12.0 * u_bf16 * max(1e-4, float(np.max(np.abs(out_ver_prod))))
    assert float(np.max(np.abs(np.asarray(out_ver_ref) - out_ver_prod))) <= ver_bound
    vh_ver = g_kda.vh(8)
    rec_ver_got = np.asarray(
        st_ver_ref["rec_ckpt"]
        .reshape(8, ex_ver.NS, ex_ver.T, vh_ver, g_kda.KD, g_kda.VD)[:, :, -1]
        .reshape(8, ex_ver.NS * vh_ver, g_kda.KD, g_kda.VD),
        dtype=np.float64,
    )
    rec_ver_bound = 12.0 * u_bf16 * max(1e-4, float(np.max(np.abs(st_ver_prod))))
    assert float(np.max(np.abs(rec_ver_got - st_ver_prod))) <= rec_ver_bound
    pos_ver = 64 + jnp.arange(ex_ver.B, dtype=I32)
    ref_step_ver, _ = qwen35.reference_step(
        g_kda, ex_ver, pos_ver, hid_ver, w_ver, st_ver
    )
    p_ver, h_ver, wm_ver, sm_ver = qwen35.pack_inputs(
        pos_ver, hid_ver, w_ver, st_ver, g_kda, ex_ver, mesh=mesh8
    )
    step_ver, _ = qwen35.make_step(mesh8, g_kda, ex_ver, donate_states=True)
    got_ver, _ = step_ver(p_ver, h_ver, wm_ver, sm_ver)
    got_ver_h = np.asarray(kernel.unfold(got_ver, 8)[0, : ex_ver.B], dtype=np.float64)
    exp_ver_h = np.asarray(ref_step_ver, dtype=np.float64)
    ver_step_bound = 48.0 * u_bf16 * max(1e-3, float(np.sqrt(np.mean(exp_ver_h**2))))
    assert float(np.max(np.abs(got_ver_h - exp_ver_h))) <= ver_step_bound


def test_kda_rejects_fused_pool_io_and_nonfinite_decay_bound():
    """Verify non-finite/bool `decay_bound` and `FUSED` KDA fail closed."""
    bad_dbs: tuple[Any, ...] = (
        0.0,
        -3.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    )
    for bad_db in bad_dbs:
        with pytest.raises(MegakernelConfigError, match="decay_bound > 0"):
            GDNModule(decay_bound=bad_db)

    with pytest.raises(MegakernelConfigError, match="decay_type"):
        GDNModule(decay_type="channel")
    with pytest.raises(MegakernelConfigError, match="decay_law"):
        GDNModule(decay_law="exp")
    with pytest.raises(MegakernelConfigError, match="gate_type"):
        GDNModule(gate_type="relu")

    g_base = _tiny_gdn_geometry()
    ex_fused = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        cache=CacheExec(paged_kv=True, fused_pool_io=True, page_size=128, num_pages=64),
        interpret=True,
    )
    for mod_bad in (
        GDNModule(prefix="linear_attn", decay_type="key"),
        GDNModule(prefix="linear_attn", decay_law="bounded_sigmoid"),
        GDNModule(prefix="linear_attn", gate_type="sigmoid"),
    ):
        with pytest.raises(MegakernelConfigError, match="fused_pool_io supports only"):
            mod_bad.smalls(g_base, ex_fused)
        with pytest.raises(MegakernelConfigError, match="fused_pool_io supports only"):
            mod_bad.outputs(g_base, ex_fused)

    mod_ok = GDNModule(prefix="linear_attn")
    assert mod_ok.outputs(g_base, ex_fused) == []
