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
"""Regression and contract tests for generic module/kernel generalizations."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class _SynthBaseModel(mk.ModelDef):
    """Compact synthetic hybrid `ModelDef` supporting `R=2` and `R=8`."""

    name = "b8_synth_base"
    hf_architectures = ()
    hf_model_type = ()
    supported_tp_sizes = (2, 8)
    default_hf_config = {
        "n_vocab": 256,
        "n_layers": 2,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_head": 256,
        "rotary_frac": 0.25,
        "theta": 10000.0,
        "norm_eps": 1e-6,
        "ssm_k_heads": 2,
        "ssm_v_heads": 4,
        "ssm_k_dim": 128,
        "ssm_v_dim": 128,
        "ssm_conv_kernel": 4,
        "n_routed_experts": 128,
        "top_k_experts": 2,
        "d_ff_routed": 256,
        "d_ff_shared": 256,
    }
    geometry = mk.FromHF(
        hidden="d_model",
        layers="n_layers",
        vocab="n_vocab",
        eps="norm_eps",
        rope_theta="theta",
        rope_partial=("rotary_frac", 0.25),
    )
    attn = mk.GQA.bind(
        prefix="self_attn",
        heads="n_heads",
        kv_heads="n_kv_heads",
        head_dim="d_head",
        qk_norm=True,
        output_gate=True,
    )
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
    )
    moe = mk.MoE.bind(
        prefix="mlp",
        experts="n_routed_experts",
        top_k="top_k_experts",
        inter="d_ff_routed",
        shared_inter="d_ff_shared",
        shared_gate=True,
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=mk.Norm.bind("input_layernorm", zero_centered=True),
        post_norm=mk.Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = mk.Embedding("model.embed_tokens")
    final_norm = mk.Norm.bind("model.norm", zero_centered=True)
    lm_head = mk.LMHead("lm_head")


class _SigmoidGDNModel(_SynthBaseModel):
    """Synthetic hybrid model binding GDN gated-norm activation to `sigmoid`."""

    name = "b8_sigmoid_gdn"
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        gate="sigmoid",
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=_SynthBaseModel.attn, rest=linear),
        ffn=_SynthBaseModel.moe,
        pre_norm=_SynthBaseModel.schedule.pre_norm,
        post_norm=_SynthBaseModel.schedule.post_norm,
    )


class _HFBoundGateGDNModel(_SynthBaseModel):
    """Synthetic hybrid model reading GDN gate activation from an HF config key."""

    name = "b8_hf_gate_gdn"
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        gate=("linear_gate_act", "silu"),
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=_SynthBaseModel.attn, rest=linear),
        ffn=_SynthBaseModel.moe,
        pre_norm=_SynthBaseModel.schedule.pre_norm,
        post_norm=_SynthBaseModel.schedule.post_norm,
    )


def test_gdn_gate_binding_and_validation() -> None:
    g_default = _SynthBaseModel.derive_geometry()
    assert g_default.gdn_gate == "silu"

    g_sig = _SigmoidGDNModel.derive_geometry()
    assert g_sig.gdn_gate == "sigmoid"
    # Direct binding.resolve(hf_config) lets hf_config override static binding option
    assert (
        _SigmoidGDNModel.linear.resolve(
            {
                "hidden_size": 256,
                **_SynthBaseModel.default_hf_config,
                "gdn_gate": "silu",
            }
        ).gate
        == "silu"
    )
    # hf_config_for_geometry preserves non-default gdn_gate even when not in
    # explicit_fields
    from vllm_torchtpu.megakernel.derive import hf_config_for_geometry

    cfg_from_sig = hf_config_for_geometry(
        _SynthBaseModel, dataclasses.replace(g_default, gdn_gate="sigmoid")
    )
    assert cfg_from_sig["gdn_gate"] == "sigmoid"
    cfg_back_silu = hf_config_for_geometry(
        _SigmoidGDNModel, dataclasses.replace(g_sig, gdn_gate="silu")
    )
    assert cfg_back_silu["gdn_gate"] == "silu"

    g_hf_default = _HFBoundGateGDNModel.derive_geometry()
    assert g_hf_default.gdn_gate == "silu"

    cfg_sig = {**_SynthBaseModel.default_hf_config, "linear_gate_act": "sigmoid"}
    g_hf_sig = _HFBoundGateGDNModel.derive_geometry(cfg_sig)
    assert g_hf_sig.gdn_gate == "sigmoid"
    shrunk_cfg = _HFBoundGateGDNModel.shrink_hf_config(cfg_sig, periods=1, R=2)
    assert shrunk_cfg["linear_gate_act"] == "sigmoid"

    with pytest.raises(mk.MegakernelConfigError, match="gate"):
        mk.GDNModule(gate="relu")

    with pytest.raises(mk.MegakernelConfigError, match="gate"):
        _HFBoundGateGDNModel.derive_geometry(
            {**_SynthBaseModel.default_hf_config, "linear_gate_act": "relu"}
        )


def test_gdn_sigmoid_kernel_matches_reference_all_paths() -> None:
    # B_per_T=8 at R=2, T_values=(1, 4) -> covers T=1 (GCH=True) and T=4 verify
    report = mk.verify_model(
        _SigmoidGDNModel,
        R=2,
        B_per_T=8,
        T_values=(1, 4),
        hf=False,
        approx_profiles=False,
    )
    report.assert_passed()

    # Unchunked GCH=False (B=4, split PallasWriter) and fused pool I/O path
    R, ctx = 2, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _SigmoidGDNModel.derive_geometry()
    spec = _SigmoidGDNModel.to_model_spec(g, R=R)
    tiles = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules(
        {
            "gqa": {"kv_blk": 256},
            "gdn": {"conv_stage": False, "gdn_head_major": True},
        }
    )

    ex = mk.ExecSpec(
        R=R,
        B=4,
        T=1,
        ctx=ctx,
        tiles=tiles,
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(11), g, ex, model_def=_SigmoidGDNModel
    )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    play = _SigmoidGDNModel.pool_layout(g, R=R)
    pt, ckpt, roff, _ = mk.pool.allocate_test_layout(play, ex)
    base_pools = [
        np.asarray(r).copy()
        for r in mk.pool.states_to_pool(
            play, ex, st_l, pt, ckpt[..., :1], jnp.maximum(roff, 0)
        )
    ]
    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    out_s, _ = step_s(
        pos,
        hid_mk,
        w_mk,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    out_f, _ = step_f(
        pos,
        hid_mk,
        w_mk,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    np.testing.assert_allclose(
        np.asarray(out_f[:4], dtype=np.float32),
        np.asarray(out_s[:4], dtype=np.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # Confirm sigmoid output matches kernel and differs from silu on identical inputs
    g_silu = _SynthBaseModel.derive_geometry()
    ref_sig, _ = _SigmoidGDNModel.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    ref_silu, _ = _SynthBaseModel.reference_step(
        g_silu, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    assert (
        float(
            jnp.max(
                jnp.abs(out_f[:4].astype(jnp.float32) - ref_sig.astype(jnp.float32))
            )
        )
        <= 0.032
    )
    diff = float(
        jnp.max(jnp.abs(ref_sig.astype(jnp.float32) - ref_silu.astype(jnp.float32)))
    )
    assert diff > 1e-4


class _H1280Model(_SynthBaseModel):
    """Synthetic hybrid model with `H=1280` (`1280 % 1024 != 0`, `CH = 640`)."""

    name = "b8_h1280"
    default_hf_config = {
        **_SynthBaseModel.default_hf_config,
        "d_model": 1280,
    }

    @classmethod
    def shrink(
        cls,
        hf_config_or_geom=None,
        *,
        periods: int = 2,
        R: int = 8,
        ex: mk.ExecSpec | None = None,
    ) -> mk.ModelGeometry:
        shrunk = dataclasses.replace(
            super().shrink(hf_config_or_geom, periods=periods, R=R, ex=ex),
            H=1280,
        )
        object.__setattr__(shrunk, "_model_def", cls)
        return shrunk


def test_moe_col_chunk_h2560() -> None:
    assert mk.col_chunk(4096) == 1024
    assert mk.col_chunk(2048) == 1024
    assert mk.col_chunk(2560) == 640
    assert mk.col_chunk(1920) == 640
    assert mk.col_chunk(1536) == 768
    assert mk.col_chunk(1280) == 640
    assert mk.col_chunk(640) == 640

    # H=2560 passes ModelGeometry.check and derives CH=640 in MoEParams
    cfg_2560 = {**_SynthBaseModel.default_hf_config, "d_model": 2560}
    g_2560 = _SynthBaseModel.derive_geometry(cfg_2560)
    ex_2560 = mk.ExecSpec(R=2, B=4, T=1, ctx=2048)
    g_2560.check(ex_2560)
    pl_2560 = mk.planner.plan(_SynthBaseModel.to_model_spec(g_2560, R=2), ex_2560)
    mp_2560 = pl_2560.module_params["moe"]
    assert mp_2560.CH == 640
    assert g_2560.H % mp_2560.CH == 0

    # End-to-end kernel vs reference parity with H=1280 > 1024 (min(1024, H)=1024
    # does not divide 1280; col_chunk(1280)=640 divides 1280 into 2 chunks)
    report = mk.verify_model(
        _H1280Model,
        R=2,
        B_per_T=2,
        T_values=(1,),
        hf=False,
        approx_profiles=False,
    )
    assert report.geometry.H == 1280
    report.assert_passed()
