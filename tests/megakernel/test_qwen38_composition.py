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
"""Qwen3.8-shaped (`KH=16 < R=32`, `IS=2048 < 128*R`) hybrid composition at 32 ranks."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import find_model, get_model
from vllm_torchtpu.megakernel.models import (
    qwen35,
    qwen35_profile,
    qwen38,
    qwen38_profile,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel_runtime import weight_collector

R = 32
TOL = 3e-2
CTX = 2048
POS0 = 1800
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540

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


def _tiny_qwen38_geometry() -> qwen35.Qwen35Geometry:
    """Tiny Qwen3.8-shaped geometry (`KH=16 < R=32`, `IS=2048 < 128*R`) at R=32."""
    return qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
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


def _tiny_qwen38_r8_geometry() -> qwen35.Qwen35Geometry:
    """8-rank geometry with identical replication factors (`kh_rep=2`, `sh_rep=2`)."""
    return qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
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


def _swap(x, a: int, b: int):
    return x.at[a].set(x[b]).at[b].set(x[a])


def test_qwen38_full_geometry_and_shared_expert_replication_at_r32():
    """Verify full Qwen3.8 geometry (`L=92, H=8192, KH=16, IS=2048`) plans at `R=32`."""
    assert find_model(QWEN38_FULL_HF_CONFIG) is qwen38.Qwen38
    assert find_model(qwen38_profile.DEFAULT_HF_CONFIG) is qwen38.Qwen38
    assert find_model(qwen35_profile.DEFAULT_HF_CONFIG) is qwen35.Qwen35
    assert get_model("qwen3_8") is qwen38.Qwen38
    assert qwen38.Qwen38.default_profile().name == "qwen38-baseline"
    assert qwen38.Qwen38.get_profile("tp32-dcp8").name == "qwen38-tp32-dcp8"

    g = qwen38.Qwen38.geometry(QWEN38_FULL_HF_CONFIG, moe_tp=False)
    ex = ExecSpec(B=4, T=1, R=R, ctx=CTX)
    g.check(ex)
    assert g.kh_rep(R) == 2
    assert g.kh(R) == 1
    assert g.vh(R) == 4
    assert g.sh_rep(R) == 2
    assert g.ISR(R) == 128
    assert g.ER(R) == 16

    ms = g.to_model_spec(R=R)
    assert ms.name == "qwen3_8"
    assert ms.layer_type.name == "qwen38_layer"
    p = planner.plan(ms, ex, fit_vmem=False)
    assert p.block_params["gdn"].kh == 1
    assert p.block_params["gdn"].vh == 4
    assert p.block_params["moe"].ISR == 128
    assert p.block_params["moe"].sh_rep == 2
    assert ms.cache_backend is not None
    play = ms.cache_backend.layout
    assert play.kh_per_rank(R) == 1
    assert play.vh_per_rank(R) == 4
    assert play.qkv_per_rank(R) == 768
    assert play.rec_pages_per_slot == 4
    play.validate(ex)

    # Verify pack_router_and_shared_layer slices by sh_rank = tp_rank // 2.
    class _FakeMLP:
        pass

    mlp = _FakeMLP()
    g_tiny = _tiny_qwen38_geometry()
    ex_tiny = ExecSpec(B=2, T=1, R=R, ctx=CTX)
    p_tiny = planner.plan(g_tiny.to_model_spec(), ex_tiny, fit_vmem=False)
    torch.manual_seed(0)
    mlp._fused_gate_gu_fp8_weight = torch.nn.Parameter(
        torch.randn(g_tiny.H, g_tiny.E + 2 * g_tiny.IS).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    mlp._fused_gate_gu_fp8_scale = torch.nn.Parameter(
        torch.rand(g_tiny.E + 2 * g_tiny.IS, dtype=torch.float32) * 0.02 + 0.01,
        requires_grad=False,
    )
    mlp._fused_se_down_fp8_weight = torch.nn.Parameter(
        torch.randn(g_tiny.IS, g_tiny.H).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    mlp._fused_se_down_fp8_scale = torch.nn.Parameter(
        torch.rand(g_tiny.H, dtype=torch.float32) * 0.02 + 0.01,
        requires_grad=False,
    )
    mlp._fused_se_gate_weight = torch.nn.Parameter(
        torch.randn(g_tiny.H, dtype=torch.float32),
        requires_grad=False,
    )
    rs0 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=0,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    rs1 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=1,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    rs2 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=2,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    assert torch.equal(rs0["sh13"].view(torch.uint8), rs1["sh13"].view(torch.uint8))
    assert torch.equal(rs0["sh2"].view(torch.uint8), rs1["sh2"].view(torch.uint8))
    assert not torch.equal(rs0["sh13"].view(torch.uint8), rs2["sh13"].view(torch.uint8))

    # Also verify 32-rank weight splitting, packing, and reference step on tiny Qwen3.8.
    k_in, k_w = jax.random.split(jax.random.PRNGKey(3800))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g_tiny, ex_tiny)
    w_l = qwen35.split_hf_weights(
        qwen35.random_hf_weights(k_w, g_tiny), g_tiny, ex_tiny
    )
    pos = POS0 + jnp.arange(ex_tiny.NS, dtype=I32)
    qk_cols = 2 * g_tiny.kh(R) * g_tiny.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])
    ref_hid, ref_st = qwen35.reference_step(g_tiny, ex_tiny, pos, hid_l, w_l, st_l)
    _, _, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_tiny, ex_tiny, mesh=None
    )
    assert jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][1])
    assert not jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][2])
    conv_swap_01_qk = (
        ref_st["conv"]
        .at[:, :, :, :qk_cols]
        .set(_swap(ref_st["conv"][:, :, :, :qk_cols], 0, 1))
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex_tiny.B, :qk_cols],
        ref_st["conv"][:, :, : 3 * ex_tiny.B, :qk_cols],
    )
    assert not jnp.allclose(
        _swap(ref_st["conv"], 0, 2), ref_st["conv"], atol=TOL, rtol=TOL
    )
    assert not jnp.allclose(hid_l, ref_hid, atol=TOL, rtol=TOL)


def test_qwen38_sub_rank_replication_step_in_process():
    """Run `make_step` on 8 CPU devices with `kh_rep=2` and `sh_rep=2`."""
    r8 = 8
    devs = jax.devices("cpu")
    assert len(devs) >= r8, f"Requires {r8} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_r8_geometry()
    assert g.kh_rep(r8) == 2 and g.sh_rep(r8) == 2
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_blocks({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=r8, ctx=1024, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3808))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g, ex)
    w_l = qwen35.split_hf_weights(qwen35.random_hf_weights(k_w, g), g, ex)
    pos = 900 + jnp.arange(ex.NS, dtype=I32)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:r8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], r8))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, r8)[:, : ex.B]
    for r in range(1, r8):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    qk_cols = 2 * g.kh(r8) * g.KD
    conv_got = kernel.unfold(out_st["conv"], r8)
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        ref_st["conv"][:, :, : 3 * ex.B, :qk_cols],
    )
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [
            ("ranks 0 and 2 swapped", _swap(conv_got, 0, 2)),
            ("input state", st_in["conv"]),
        ],
    )
    for k, a, b in (("rec", 0, 1), ("kc", 0, 4), ("vc", 0, 4)):
        got = kernel.unfold(out_st[k], r8)
        _check(
            k,
            got,
            ref_st[k],
            [
                (f"ranks {a} and {b} swapped", _swap(got, a, b)),
                ("input state", st_in[k]),
            ],
        )


def _step_qwen38_32_ranks_in_process() -> None:
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_geometry()
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_blocks({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=R, ctx=CTX, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3801))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g, ex)
    w_l = qwen35.split_hf_weights(qwen35.random_hf_weights(k_w, g), g, ex)
    pos = POS0 + jnp.arange(ex.NS, dtype=I32)

    # Ranks 0 and 1 share key/query head 0 and shared-expert slice 0; rank 2 does not.
    qk_cols = 2 * g.kh(R) * g.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])

    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], R))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, R)[:, : ex.B]
    for r in range(1, R):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    conv_got = kernel.unfold(out_st["conv"], R)
    # Positive control: swapping Q/K conv state between replica ranks 0 and 1 is exact.
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        ref_st["conv"][:, :, : 3 * ex.B, :qk_cols],
    )
    # Negative control: swapping Q/K conv state between ranks 0 and 2 fails.
    conv_swap_02 = _swap(conv_got, 0, 2)
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [("ranks 0 and 2 swapped", conv_swap_02), ("input state", st_in["conv"])],
    )

    for k, a, b in (("rec", 0, 1), ("kc", 0, 8), ("vc", 0, 8)):
        got = kernel.unfold(out_st[k], R)
        _check(
            k,
            got,
            ref_st[k],
            [
                (f"ranks {a} and {b} swapped", _swap(got, a, b)),
                ("input state", st_in[k]),
            ],
        )


@pytest.mark.slow
def test_qwen38_decode_with_32_ranks():
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, "qwen38_r32"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    print(out.stdout)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def test_moe_block_with_nvfp4_routed_experts_in_process():
    """Verify MoEBlock expert_quant='nvfp4' numerical dequantization and step parity."""
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.megakernel.blocks.moe import MoEBlock
    from vllm_torchtpu.megakernel.blocks.moe.experts import unpack_nvfp4_tile
    from vllm_torchtpu.megakernel.blocks.moe.reference import dequant_nvfp4_weight

    mb_default = MoEBlock(prefix="mlp")
    mb_bound = MoEBlock.bind(prefix="mlp")
    assert mb_default.options == mb_bound.options
    mb_nv = MoEBlock(prefix="mlp", expert_quant="nvfp4")
    assert mb_nv.options["expert_quant"] == "nvfp4"
    g_from_hf = qwen38.Qwen38.derive_geometry(
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
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(91), g_nv, ex8
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
    ref_hid_nv, _ = qwen35.reference_step(g_nv, ex8, pos, hid_l, w_l, st_l)
    assert ref_hid_nv.shape == (ex8.B, g_nv.H)
    ref_hid_f32, _ = qwen35.reference_step(g_base, ex8, pos, hid_l, w_f32_eq, st_l)
    w_bad_s = dict(w_l, e13_s=jnp.zeros_like(w_l["e13_s"]))
    ref_hid_bad, _ = qwen35.reference_step(g_nv, ex8, pos, hid_l, w_bad_s, st_l)
    _check(
        "nvfp4 reference_step vs f32 dequant control",
        ref_hid_nv,
        ref_hid_f32,
        [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
    )
    devs = jax.devices("cpu")
    if len(devs) >= ex8.R:
        mesh = Mesh(devs[: ex8.R], ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_l, st_l, g_nv, ex8, mesh=mesh
        )
        step_nv, _ = qwen35.make_step(mesh, g_nv, ex8, donate_states=True)
        out_hid_nv, _ = step_nv(pos_mk, hid_mk, w_mk, st_mk)
        hid_nv_r = kernel.unfold(out_hid_nv, ex8.R)[:, : ex8.B]
        _check(
            "nvfp4 make_step vs reference_step",
            hid_nv_r[0],
            ref_hid_nv,
            [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
        )


def test_gdn_block_with_kda_key_decay_and_gate_in_process():
    """Verify GDNBlock KDA per-key decay and sigmoid gate numerical parity."""
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
    from vllm_torchtpu.megakernel.blocks.gdn.reference import gdn_reference

    gb_default = GDNBlock(prefix="linear_attn")
    gb_bound = GDNBlock.bind(prefix="linear_attn")
    assert dict(gb_default.options) == dict(gb_bound.options)
    gb_kda = GDNBlock(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    assert gb_kda.options["decay_type"] == "key"

    g_base = _tiny_qwen38_geometry()
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
    lay_kda = GDNBlock.lin_in_layout(g_kda, ex8)
    assert lay_kda.vh * lay_kda.KD == lay_kda.A_DIM
    assert gb_kda.weights(g_base, ex8)[0].N == lay_kda.NLI
    assert gb_kda.smalls(g_base, ex8)[4].width == lay_kda.A_DIM
    lay_sc = GDNBlock.lin_in_layout(g_base, ex8)
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
    w_kda["alog"] = jnp.repeat(w_sc["alog"], kd, axis=-1)
    w_kda["dtb"] = jnp.repeat(w_sc["dtb"], kd, axis=-1)
    out_kda, _ = gdn_reference(w_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    w_bad_kda = dict(w_kda, alog=w_kda["alog"] + 3.0)
    out_bad_kda, _ = gdn_reference(w_bad_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    _check(
        "kda gdn_reference vs head bounded_sigmoid",
        out_kda,
        out_head_bsig,
        [("softplus/silu control", out_sc), ("perturbed alog", out_bad_kda)],
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
        )


if __name__ == "__main__":
    if sys.argv[1] == "qwen38_r32":
        _step_qwen38_32_ranks_in_process()
