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
"""NVFP4 routed expert support (`expert_quant='nvfp4'`) in `MoEModule`."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import types
from dataclasses import replace as dc_replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

import vllm_torchtpu
import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38_moe
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.modules.moe import MoEModule
from vllm_torchtpu.megakernel.modules.moe import experts as moe_experts
from vllm_torchtpu.megakernel.modules.moe.experts import _mm_etake_w, unpack_nvfp4_tile
from vllm_torchtpu.megakernel.modules.moe.reference import dequant_nvfp4_weight
from vllm_torchtpu.megakernel.ops import BF16, F32, I32
from vllm_torchtpu.megakernel_runtime.weight_collector import pack_expert_views_layer

TOL = 3e-2
CTX = 2048
POS0 = 1800

QWEN38_FULL_HF_CONFIG = {
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


def _tiny_qwen38_geometry() -> qwen38_moe.Qwen38MoeGeometry:
    """Tiny Qwen3.8-shaped geometry (`KH=16 < R=32`, `IS=2048 < 128*R`)."""
    return qwen38_moe.Qwen38MoeGeometry(
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


def _check(name: str, got, exp, controls) -> None:
    _report(name, got, exp)
    assert jnp.allclose(got, exp, atol=TOL, rtol=TOL), name
    for label, bad in controls:
        _report(f"{name} vs {label}", bad, exp)
        assert not jnp.allclose(bad, exp, atol=TOL, rtol=TOL), f"{name}: {label}"


def test_moe_block_with_nvfp4_routed_experts_in_process():
    """Verify MoEModule expert_quant='nvfp4' numerical dequant and step parity."""
    mb_default = MoEModule(prefix="mlp")
    mb_bound = MoEModule.bind(prefix="mlp")
    assert mb_default.options == mb_bound.options
    mb_nv = MoEModule(prefix="mlp", expert_quant="nvfp4")
    assert mb_nv.options["expert_quant"] == "nvfp4"
    g_from_hf = qwen38_moe.Qwen38Moe.derive_geometry(
        {**QWEN38_FULL_HF_CONFIG, "quantization_config": {"quant_algo": "NVFP4"}}
    )
    assert g_from_hf.expert_quant == "nvfp4"
    with pytest.raises(ValueError, match="unpack_nvfp4_tile requires"):
        unpack_nvfp4_tile(jnp.zeros((4, 256), dtype=jnp.uint8), 128)

    g_base = dc_replace(_tiny_qwen38_geometry(), I=256)  # e2 tile K = I // 2 >= 128
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=256 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    g_nv = dc_replace(g_base, expert_quant="nvfp4")
    _, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.PRNGKey(91), g_nv, ex8, model_def=qwen38_moe.Qwen38Moe
    )
    assert w_l["e13"].dtype == jnp.uint8
    n_log = 2 * g_nv.IR(ex8.R)
    w_tile = w_l["e13"][0, 0, 0]
    unp_got = unpack_nvfp4_tile(w_tile, n_log, interpret=True).astype(F32)
    deq_ref = dequant_nvfp4_weight(w_tile, n_log)
    w_bad = w_tile.at[:, n_log:].set(jnp.uint8(0))
    deq_bad = dequant_nvfp4_weight(w_bad, n_log)
    _check(
        "nvfp4 unpack vs dequant_ref",
        unp_got,
        deq_ref,
        [("zeroed block scale", deq_bad)],
    )

    w_f32_eq = dict(w_l)
    w_f32_eq["e13"] = jax.vmap(
        jax.vmap(jax.vmap(lambda t: dequant_nvfp4_weight(t, n_log)))
    )(w_l["e13"])
    w_f32_eq["e2"] = jax.vmap(
        jax.vmap(jax.vmap(lambda t: dequant_nvfp4_weight(t, g_nv.H)))
    )(w_l["e2"])
    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_nv, _ = qwen38_moe.reference_step(g_nv, ex8, pos, hid_l, w_l, st_l)
    assert ref_hid_nv.shape == (ex8.B, g_nv.H)
    ref_hid_f32, _ = qwen38_moe.reference_step(g_base, ex8, pos, hid_l, w_f32_eq, st_l)
    w_bad_s = dict(w_l, e13_s=jnp.zeros_like(w_l["e13_s"]))
    ref_hid_bad, _ = qwen38_moe.reference_step(g_nv, ex8, pos, hid_l, w_bad_s, st_l)
    _check(
        "nvfp4 reference_step vs f32 dequant control",
        ref_hid_nv,
        ref_hid_f32,
        [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
    )
    devs = jax.devices("cpu")
    if len(devs) >= ex8.R:
        mesh = Mesh(devs[: ex8.R], ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
            pos,
            hid_l,
            w_l,
            st_l,
            g_nv,
            ex8,
            mesh=mesh,
            model_def=qwen38_moe.Qwen38Moe,
        )
        step_nv, _ = kernel.make_step(
            mesh, g_nv.to_model_spec(R=ex8.R), ex8, donate_states=True
        )
        out_hid_nv, _ = step_nv(pos_mk, hid_mk, w_mk, st_mk)
        hid_nv_r = kernel.unfold(out_hid_nv, ex8.R)[:, : ex8.B]
        _check(
            "nvfp4 make_step vs reference_step",
            hid_nv_r[0],
            ref_hid_nv,
            [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
        )


def test_moe_nvfp4_rolled_multi_tile_interpret():
    """G1: Multi-tile rolled NVFP4 MoE (T13=2, tk_raw=128) in interpret mode."""

    class _Ref:
        def __init__(self, arr):
            self.arr = arr

        @staticmethod
        def _idx(k):
            if isinstance(k, tuple):
                return tuple(_Ref._idx(i) for i in k)
            if hasattr(k, "start") and hasattr(k, "size") and not isinstance(k, slice):
                return slice(int(k.start), int(k.start) + int(k.size))
            return k

        def __getitem__(self, k):
            return self.arr[self._idx(k)]

        def __setitem__(self, k, v):
            self.arr = self.arr.at[self._idx(k)].set(v)

    g_rolled = dataclasses.replace(
        _tiny_qwen38_geometry(),
        H=512,
        I=1024,
        moe_tp=False,
        expert_quant="nvfp4",
    )
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(
            TINY_TILES, bank=256 << 10, ebank=320 << 10, hier=0, mb8=1, inplace=False
        ),
        interpret=True,
    )
    mspec = qwen38_moe.qwen38_moe_spec(g_rolled, R=ex8.R)
    p = planner.plan(mspec, ex8)
    assert p.nt["e13"] == 2 and p.tk["e13"] == 128
    assert not p.module_params["moe"].unroll

    _, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.PRNGKey(191), g_rolled, ex8, model_def=qwen38_moe.Qwen38Moe
    )

    n_log = 2 * g_rolled.IR(ex8.R)
    w_tile_128 = w_l["e13"][0, 0, 0, :128]
    x_act = jax.random.normal(jax.random.PRNGKey(192), (ex8.MB, 256), dtype=BF16)
    acc_128 = _Ref(jnp.zeros((ex8.MB, n_log), dtype=F32))
    acc_64 = _Ref(jnp.zeros((ex8.MB, n_log), dtype=F32))
    m_stub = types.SimpleNamespace(
        mp=types.SimpleNamespace(expert_quant="nvfp4"),
        ex=ex8,
        scr={},
        stream=types.SimpleNamespace(
            mm=lambda a, b: jnp.matmul(a.astype(F32), b.astype(F32)),
        ),
    )
    orig_etake = moe_experts.etake
    moe_experts.etake = lambda m, kind, u: _Ref(
        w_tile_128 if u == 0 else w_tile_128[:64]
    )
    try:
        with jax.disable_jit():
            _mm_etake_w(m_stub, "e13", 0, _Ref(x_act), 0, 256, n_log, acc_128)
            _mm_etake_w(m_stub, "e13", 1, _Ref(x_act), 0, 128, n_log, acc_64)
    finally:
        moe_experts.etake = orig_etake
    ref_128 = jnp.matmul(x_act.astype(F32), dequant_nvfp4_weight(w_tile_128, n_log))
    ref_64 = jnp.matmul(
        x_act[:, :128].astype(F32), dequant_nvfp4_weight(w_tile_128[:64], n_log)
    )
    _check(
        "_mm_etake_w tk_raw=128 vs dequant_ref",
        acc_128.arr,
        ref_128,
        [("half-tile control", ref_64)],
    )
    _check(
        "_mm_etake_w tk_raw=64 vs dequant_ref",
        acc_64.arr,
        ref_64,
        [("full-tile control", ref_128)],
    )

    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_nv, _ = qwen38_moe.reference_step(g_rolled, ex8, pos, hid_l, w_l, st_l)
    w_bad_s = dict(w_l, e13_s=jnp.zeros_like(w_l["e13_s"]))
    ref_hid_bad, _ = qwen38_moe.reference_step(g_rolled, ex8, pos, hid_l, w_bad_s, st_l)

    devs = jax.devices("cpu")
    assert len(devs) >= ex8.R
    mesh = Mesh(devs[: ex8.R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g_rolled,
        ex8,
        mesh=mesh,
        model_def=qwen38_moe.Qwen38Moe,
    )
    step_nv, _ = kernel.make_step(
        mesh, g_rolled.to_model_spec(R=ex8.R), ex8, donate_states=True
    )
    out_hid_nv, _ = step_nv(pos_mk, hid_mk, w_mk, st_mk)
    hid_nv_r = kernel.unfold(out_hid_nv, ex8.R)[:, : ex8.B]
    _check(
        "nvfp4 rolled multi-tile make_step vs reference_step",
        hid_nv_r[0],
        ref_hid_nv,
        [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
    )


def test_pack_nvfp4_rejects_stock_prepared_layout():
    """G2: pack_expert_views_layer rejects stock requant_unpack_kmajor layout."""
    spec = importlib.util.find_spec("libtpu")
    if spec and spec.submodule_search_locations:
        os.environ.setdefault(
            "TPU_LIBRARY_PATH",
            os.path.join(spec.submodule_search_locations[0], "libtpu.so"),
        )

    gmm_path = (
        Path(vllm_torchtpu.__file__).resolve().parent
        / "layers"
        / "core"
        / "fused_moe_gmm.py"
    )
    gmm_spec = importlib.util.spec_from_file_location(
        "vllm_torchtpu_fused_moe_gmm", gmm_path
    )
    assert gmm_spec is not None and gmm_spec.loader is not None
    gmm_mod = importlib.util.module_from_spec(gmm_spec)
    gmm_spec.loader.exec_module(gmm_mod)
    requant_unpack_kmajor = gmm_mod.requant_unpack_kmajor

    g_nv = dc_replace(
        _tiny_qwen38_geometry(), H=256, I=256, moe_tp=False, expert_quant="nvfp4"
    )
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=256 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    ew = g_nv.EW(ex8.R)  # 16
    ir = g_nv.IR(ex8.R)  # 256
    H = g_nv.H  # 256

    rng = np.random.default_rng(77)
    w13_raw = rng.integers(0, 256, size=(ew, 2 * ir, H // 2), dtype=np.uint8)
    s13_raw = rng.integers(1, 60, size=(ew, 2 * ir, H // 16), dtype=np.uint8)
    w2_raw = rng.integers(0, 256, size=(ew, H, ir // 2), dtype=np.uint8)
    s2_raw = rng.integers(1, 60, size=(ew, H, ir // 16), dtype=np.uint8)

    w13_stock, s13_stock = requant_unpack_kmajor(
        jnp.asarray(w13_raw),
        jnp.asarray(s13_raw).view(jnp.float8_e4m3fn).astype(jnp.float32),
        256,
    )
    assert tuple(w13_stock.shape) == (ew, H, 2 * ir)
    assert tuple(s13_stock.shape) == (ew, H // 256, 1, 2 * ir)

    class _Experts:
        pass

    class _MLP:
        pass

    # 1. Stock-prepared tensor shapes (uint8 [ew, H, ir], 4D f32 scale)
    exp_stock = _Experts()
    exp_stock.w13_weight = torch.zeros((ew, H, ir), dtype=torch.uint8)
    exp_stock.w13_weight_scale = torch.from_numpy(np.asarray(s13_stock).copy())
    exp_stock.w13_weight_scale_2 = None
    exp_stock.w2_weight = torch.from_numpy(w2_raw)
    exp_stock.w2_weight_scale = torch.from_numpy(s2_raw)
    exp_stock.w2_weight_scale_2 = None
    mlp_stock = _MLP()
    mlp_stock.experts = exp_stock

    with pytest.raises(
        MegakernelConfigError,
        match="stock process_weights_after_loading already requantized; see unit B",
    ):
        pack_expert_views_layer(mlp_stock, g_nv, ex8, tk_e13=128, tk_e2=128)

    # 2. Raw weights/scales present, but w13_weight_scale_2 deleted (None)
    exp_no_gscale = _Experts()
    exp_no_gscale.w13_weight = torch.from_numpy(w13_raw)
    exp_no_gscale.w13_weight_scale = torch.from_numpy(s13_raw)
    exp_no_gscale.w13_weight_scale_2 = None
    exp_no_gscale.w2_weight = torch.from_numpy(w2_raw)
    exp_no_gscale.w2_weight_scale = torch.from_numpy(s2_raw)
    exp_no_gscale.w2_weight_scale_2 = None
    mlp_no_gscale = _MLP()
    mlp_no_gscale.experts = exp_no_gscale

    with pytest.raises(
        MegakernelConfigError,
        match="stock process_weights_after_loading already requantized; see unit B",
    ):
        pack_expert_views_layer(mlp_no_gscale, g_nv, ex8, tk_e13=128, tk_e2=128)
