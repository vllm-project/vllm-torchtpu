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
from vllm_torchtpu.megakernel.models import qwen35


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


class _Qwen35TinyBase(qwen35.Qwen35):
    """Tiny 2-layer `Qwen35` base for mixed `weight_dtypes` + `hf=True` checks."""

    name = "b8_qwen35_tiny_base"
    supported_tp_sizes = (2, 8)
    profiles = {}
    default_hf_config = {
        **qwen35.Qwen35.default_hf_config,
        "vocab_size": 256,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 128,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
        "layer_types": ["linear_attention", "full_attention"],
    }


class _Bf16DenseModel(_Qwen35TinyBase):
    """Tiny `Qwen35` with `bf16` dense projections and `fp8` routed experts."""

    name = "b8_bf16_dense"
    attn = dataclasses.replace(
        qwen35.Qwen35.attn,
        options={
            **qwen35.Qwen35.attn.options,
            "weight_dtypes": {"att_in": "bf16", "att_out": "bf16"},
        },
    )
    linear = dataclasses.replace(
        qwen35.Qwen35.linear,
        options={
            **qwen35.Qwen35.linear.options,
            "weight_dtypes": {"lin_in": "bf16", "lin_out": "bf16"},
        },
    )
    moe = dataclasses.replace(
        qwen35.Qwen35.moe,
        options={
            **qwen35.Qwen35.moe.options,
            "weight_dtypes": {"sh13": "bf16", "sh2": "bf16", "e13": "fp8_channel"},
        },
    )
    schedule = mk.Schedule(
        mixer=mk.FromHFLayerTypes(
            "layer_types", {"full_attention": attn, "linear_attention": linear}
        ),
        ffn=moe,
        pre_norm=qwen35.Qwen35.schedule.pre_norm,
        post_norm=qwen35.Qwen35.schedule.post_norm,
    )


class _GQAOnlyBf16Model(_Qwen35TinyBase):
    """Mixed policy: GQA `bf16` + GDN `fp8` + MoE `fp8` (shares `in_s`/`out_s`)."""

    name = "b8_gqa_only_bf16"
    attn = _Bf16DenseModel.attn
    schedule = mk.Schedule(
        mixer=mk.FromHFLayerTypes(
            "layer_types",
            {"full_attention": attn, "linear_attention": qwen35.Qwen35.linear},
        ),
        ffn=qwen35.Qwen35.moe,
        pre_norm=qwen35.Qwen35.schedule.pre_norm,
        post_norm=qwen35.Qwen35.schedule.post_norm,
    )


class _GDNOnlyBf16Model(_Qwen35TinyBase):
    """Mixed policy: GQA `fp8` + GDN `bf16` + MoE `fp8` (shares `in_s`/`out_s`)."""

    name = "b8_gdn_only_bf16"
    linear = _Bf16DenseModel.linear
    schedule = mk.Schedule(
        mixer=mk.FromHFLayerTypes(
            "layer_types",
            {"full_attention": qwen35.Qwen35.attn, "linear_attention": linear},
        ),
        ffn=qwen35.Qwen35.moe,
        pre_norm=qwen35.Qwen35.schedule.pre_norm,
        post_norm=qwen35.Qwen35.schedule.post_norm,
    )


class _Sh13OnlyBf16Model(_Qwen35TinyBase):
    """Mixed policy: MoE `sh13: bf16` + `sh2: fp8`."""

    name = "b8_sh13_only_bf16"
    moe = dataclasses.replace(
        qwen35.Qwen35.moe,
        options={**qwen35.Qwen35.moe.options, "weight_dtypes": {"sh13": "bf16"}},
    )
    schedule = dataclasses.replace(qwen35.Qwen35.schedule, ffn=moe)


class _Bf16ExpertsModel(_Qwen35TinyBase):
    """Policy with `bf16` routed experts (`e13: bf16, e2: bf16`)."""

    name = "b8_bf16_experts"
    moe = dataclasses.replace(
        qwen35.Qwen35.moe,
        options={
            **qwen35.Qwen35.moe.options,
            "weight_dtypes": {"e13": "bf16", "e2": "bf16"},
        },
    )
    schedule = dataclasses.replace(qwen35.Qwen35.schedule, ffn=moe)


def test_per_weight_dtype_policy() -> None:
    with pytest.raises(mk.MegakernelConfigError, match="unknown weight"):
        mk.GQAModule(weight_dtypes={"bad": "bf16"})
    with pytest.raises(mk.MegakernelConfigError, match="must be 'fp8' or 'bf16'"):
        mk.GDNModule(weight_dtypes={"lin_in": "int8"})
    with pytest.raises(mk.MegakernelConfigError, match="share the same weight dtype"):
        mk.MoEModule(weight_dtypes={"e13": "bf16", "e2": "fp8"})

    g = _Bf16DenseModel.shrink(periods=1, R=2)
    ex = mk.ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        tiles=mk.Tiles(
            bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
        ).with_modules({"gqa": {"kv_blk": 256}}),
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    spec = _Bf16DenseModel.to_model_spec(g, R=2)
    pl = mk.planner.plan(spec, ex)

    # Scale smalls are dropped when dense projections are bf16 passthrough
    for dropped in ("in_s", "out_s", "sh13_s", "sh2_s"):
        assert dropped not in pl.smalls_layout
    for retained in (
        "in_norm",
        "post_norm",
        "qn",
        "kn",
        "conv",
        "alog",
        "dtb",
        "gnorm",
        "sh_gate",
    ):
        assert retained in pl.smalls_layout
    for k_bf16 in ("att_in", "att_out", "lin_in", "lin_out", "sh13", "sh2"):
        assert jnp.dtype(pl.weight_decls[k_bf16].dtype) == jnp.dtype(jnp.bfloat16)
    for k_fp8 in ("e13", "e2"):
        assert jnp.dtype(pl.weight_decls[k_fp8].dtype) == jnp.dtype(jnp.float8_e4m3fn)

    # Mixed policies keep shared smalls names in smalls_layout while module params
    # track each module's own static scale policy independently
    pl_gqa_bf16 = mk.planner.plan(_GQAOnlyBf16Model.to_model_spec(g, R=2), ex)
    assert "in_s" in pl_gqa_bf16.smalls_layout and "out_s" in pl_gqa_bf16.smalls_layout
    assert not pl_gqa_bf16.module_params["gqa"].in_scaled
    assert not pl_gqa_bf16.module_params["gqa"].out_scaled
    assert pl_gqa_bf16.module_params["gdn"].in_scaled
    assert pl_gqa_bf16.module_params["gdn"].out_scaled

    pl_gdn_bf16 = mk.planner.plan(_GDNOnlyBf16Model.to_model_spec(g, R=2), ex)
    assert "in_s" in pl_gdn_bf16.smalls_layout and "out_s" in pl_gdn_bf16.smalls_layout
    assert pl_gdn_bf16.module_params["gqa"].in_scaled
    assert pl_gdn_bf16.module_params["gqa"].out_scaled
    assert not pl_gdn_bf16.module_params["gdn"].in_scaled
    assert not pl_gdn_bf16.module_params["gdn"].out_scaled

    pl_sh13_bf16 = mk.planner.plan(_Sh13OnlyBf16Model.to_model_spec(g, R=2), ex)
    assert "sh13_s" not in pl_sh13_bf16.smalls_layout
    assert "sh2_s" in pl_sh13_bf16.smalls_layout
    assert not pl_sh13_bf16.module_params["moe"].sh13_scaled
    assert pl_sh13_bf16.module_params["moe"].sh2_scaled

    # Loading an FP8 vLLM checkpoint into a bf16-policy model dequantizes via
    # slice_alias_targets
    g_fp8 = _Qwen35TinyBase.shrink(periods=1, R=2)
    w_hf_fp8 = mk.normalize_hf_for_vllm(
        _Qwen35TinyBase,
        ex,
        mk.random_hf_weights(jax.random.key(3), g_fp8, model_def=_Qwen35TinyBase),
        g=g_fp8,
    )
    src_fp8_vllm = mk.SyntheticVLLMWeightSource(_Qwen35TinyBase, ex, w_hf_fp8, g=g_fp8)
    loaded_from_fp8 = mk.load_weights(_Bf16DenseModel, ex, src_fp8_vllm, g=g, tp_rank=0)
    for k_bf16 in ("att_in", "att_out", "lin_in", "lin_out", "sh13", "sh2"):
        assert jnp.dtype(loaded_from_fp8[k_bf16].dtype) == jnp.dtype(jnp.bfloat16)
        max_abs = float(jnp.max(jnp.abs(loaded_from_fp8[k_bf16].astype(jnp.float32))))
        assert max_abs > 0.0

    try:
        import torch
        import transformers

        has_tf = hasattr(transformers, "Qwen3_5MoeForCausalLM") and hasattr(
            torch, "Tensor"
        )
    except ImportError:
        has_tf = False

    for model_cls, t_vals in (
        (_Bf16DenseModel, (1, 4)),
        (_GQAOnlyBf16Model, (1,)),
        (_GDNOnlyBf16Model, (1,)),
        (_Sh13OnlyBf16Model, (1,)),
        (_Bf16ExpertsModel, (1,)),
    ):
        report = mk.verify_model(
            model_cls,
            R=2,
            B_per_T=2,
            T_values=t_vals,
            seed=2,
            hf=True,
            approx_profiles=False,
        )
        report.assert_passed(allow_skipped=not has_tf)
        by_name = {c.name: c for c in report.checks}
        assert by_name["loader_parity"].status == "passed"
        assert by_name["kernel_vs_reference"].status == "passed"
        if has_tf:
            assert by_name["reference_vs_hf"].status == "passed"


def test_merged_tile_operand_layout() -> None:
    # Direct unit test of unpack_local_weight for rank-3, rank-4, and rank-5 shapes
    x3 = jnp.arange(2 * 128 * 64, dtype=jnp.bfloat16).reshape(2 * 128, 64)
    u3 = mk.kernel.unpack_local_weight(
        x3, (2, 128, 64), name="per_layer", allow_merged_tiles=True
    )
    assert u3.shape == (2, 128, 64)
    u3_b = mk.kernel.unpack_local_weight(
        x3[None], (2, 128, 64), name="per_layer", allow_merged_tiles=True
    )
    assert u3_b.shape == (2, 128, 64)

    x4 = jnp.arange(3 * 2 * 128 * 64, dtype=jnp.bfloat16).reshape(3, 2 * 128, 64)
    u4 = mk.kernel.unpack_local_weight(
        x4, (3, 2, 128, 64), name="stacked", allow_merged_tiles=True
    )
    assert u4.shape == (3, 2, 128, 64)

    x_es = jnp.arange(3 * 4 * 64, dtype=jnp.float32).reshape(3, 4, 64)
    u_es = mk.kernel.unpack_local_weight(
        x_es[None], (3, 4, 1, 64), name="es", allow_merged_tiles=True
    )
    assert u_es.shape == (3, 4, 1, 64)

    # End-to-end step parity: merged_tiles=True vs merged_tiles=False (default)
    R, ctx = 2, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _SynthBaseModel.derive_geometry()
    spec = _SynthBaseModel.to_model_spec(g, R=R)
    ex = mk.ExecSpec(
        R=R,
        B=4,
        T=1,
        ctx=ctx,
        tiles=mk.Tiles(
            bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
        ).with_modules({"gqa": {"kv_blk": 256}}),
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    pl = mk.planner.plan(spec, ex)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(19), g, ex, model_def=_SynthBaseModel
    )
    _, hid_mk, w_tiled, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    w_merged_host = mk.pack_logical_weights(
        _SynthBaseModel, ex, w_l, g=g, merged_tiles=True
    )
    # Default pack_logical_weights keeps tiled layout; merged_tiles=True merges nt*tk
    for k in pl.static_kinds:
        nt_k, tk_k, n_k = pl.nt[k], pl.tk[k], pl.kinds[k][1]
        assert w_tiled[k].shape[-3:] == (nt_k, tk_k, n_k)
        assert w_merged_host[k].shape[-2:] == (nt_k * tk_k, n_k)
        assert w_merged_host[k].ndim == w_tiled[k].ndim - 1
        np.testing.assert_array_equal(
            np.asarray(w_merged_host[k]).view(np.uint8),
            np.asarray(w_tiled[k]).reshape(w_merged_host[k].shape).view(np.uint8),
        )
    assert w_tiled["es"].ndim == 5
    assert w_merged_host["es"].ndim == 4
    np.testing.assert_array_equal(
        np.asarray(w_merged_host["es"]),
        np.asarray(w_tiled["es"]).reshape(w_merged_host["es"].shape),
    )

    # load_weights with merged_tiles=True produces rank-3 stacked static operands
    # that match reshaped default operands bit-for-bit across all TP ranks
    src = mk.MappingWeightSource(w_l)
    lw_def_ranks = [
        mk.load_weights(_SynthBaseModel, ex, src, g=g, tp_rank=r) for r in range(R)
    ]
    lw_mrg_ranks = [
        mk.load_weights(_SynthBaseModel, ex, src, g=g, tp_rank=r, merged_tiles=True)
        for r in range(R)
    ]
    lw_default, lw_merged = lw_def_ranks[0], lw_mrg_ranks[0]
    for k in pl.static_kinds:
        assert lw_default[k].ndim == 4
        assert lw_merged[k].ndim == 3
        np.testing.assert_array_equal(
            np.asarray(lw_merged[k]).view(np.uint8),
            np.asarray(lw_default[k]).reshape(lw_merged[k].shape).view(np.uint8),
        )
    assert lw_default["es"].ndim == 4
    assert lw_merged["es"].ndim == 3
    np.testing.assert_array_equal(
        np.asarray(lw_merged["es"]),
        np.asarray(lw_default["es"]).reshape(lw_merged["es"].shape),
    )

    w_merged = {
        k: (
            tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
        )
        for k, v in w_merged_host.items()
    }
    w_lw_merged = {
        k: (
            tuple(
                jax.device_put(
                    jnp.stack([lw_mrg_ranks[r][k][i] for r in range(R)], axis=0), tp
                )
                for i in range(len(lw_merged[k]))
            )
            if isinstance(lw_merged[k], tuple)
            else jax.device_put(
                jnp.stack([lw_mrg_ranks[r][k] for r in range(R)], axis=0), tp
            )
        )
        for k in pl.w_names
    }

    # DirectHBM make_step bit-exact between tiled, pack_logical_weights, and
    # load_weights(merged_tiles=True) operands
    step_d, _ = mk.make_step(mesh, spec, ex, donate_states=False)
    out_d_tiled, _ = step_d(pos, hid_mk, w_tiled, st_mk)
    out_d_merged, _ = step_d(pos, hid_mk, w_merged, st_mk)
    out_d_lw_merged, _ = step_d(pos, hid_mk, w_lw_merged, st_mk)
    np.testing.assert_array_equal(np.asarray(out_d_merged), np.asarray(out_d_tiled))
    np.testing.assert_array_equal(np.asarray(out_d_lw_merged), np.asarray(out_d_tiled))

    # PallasWriter make_pooled_step and make_flat_pooled_step bit-exact
    play = _SynthBaseModel.pool_layout(g, R=R)
    pt, ckpt, roff, _ = mk.pool.allocate_test_layout(play, ex)
    base_pools = [
        np.asarray(r).copy()
        for r in mk.pool.states_to_pool(
            play, ex, st_l, pt, ckpt[..., :1], jnp.maximum(roff, 0)
        )
    ]
    step_p, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    out_p_tiled, _ = step_p(
        pos,
        hid_mk,
        w_tiled,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    out_p_merged, _ = step_p(
        pos,
        hid_mk,
        w_merged,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    np.testing.assert_array_equal(np.asarray(out_p_merged), np.asarray(out_p_tiled))

    flat_step, _, _, _ = mk.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=False, fused_pool_io=False
    )
    w_flat_tiled = []
    w_flat_merged = []
    for k in pl.w_names:
        vt, vm = w_tiled[k], w_merged[k]
        w_flat_tiled.extend(vt if isinstance(vt, tuple) else (vt,))
        w_flat_merged.extend(vm if isinstance(vm, tuple) else (vm,))
    pools_sh = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    out_f_tiled = flat_step(pos, hid_mk, *w_flat_tiled, *pools_sh, pt, ckpt, roff)
    out_f_merged = flat_step(pos, hid_mk, *w_flat_merged, *pools_sh, pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_f_merged[0]), np.asarray(out_f_tiled[0])
    )


def test_configurable_collective_id(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mk.Tiles().collective_id == 0
    assert mk.ExecSpec().resolved_collective_id == 0
    assert mk.ExecSpec(tiles=mk.Tiles(collective_id=47)).resolved_collective_id == 47
    assert (
        mk.ExecSpec(
            tiles=mk.Tiles(collective_id=12), collective_id=47
        ).resolved_collective_id
        == 47
    )
    assert mk.exec_spec.tile_numerics()["collective_id"] == "exact"

    with pytest.raises(ValueError, match="collective_id"):
        mk.Tiles(collective_id=-1)
    with pytest.raises(ValueError, match="collective_id"):
        mk.ExecSpec(collective_id=-1)

    g = _SynthBaseModel.derive_geometry()
    spec = _SynthBaseModel.to_model_spec(g, R=2)
    base_tiles = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules({"gqa": {"kv_blk": 256}})
    seen: list[int | None] = []
    real_cp = mk.kernel.pltpu.CompilerParams

    def spy_cp(*args, **kwargs):
        seen.append(kwargs.get("collective_id"))
        return real_cp(*args, **kwargs)

    monkeypatch.setattr(mk.kernel.pltpu, "CompilerParams", spy_cp)

    _, info_def = mk.build(spec, mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=base_tiles))
    assert info_def["collective_id"] == 0
    assert seen[-1] == 0

    _, info_tiles = mk.build(
        spec,
        mk.ExecSpec(
            R=2,
            B=4,
            T=1,
            ctx=256,
            tiles=dataclasses.replace(base_tiles, collective_id=47),
        ),
    )
    assert info_tiles["collective_id"] == 47
    assert seen[-1] == 47

    _, info_ex = mk.build(
        spec,
        mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=base_tiles, collective_id=99),
    )
    assert info_ex["collective_id"] == 99
    assert seen[-1] == 99


def test_comm_params_derives_pc_hpc_pr() -> None:
    ex = mk.ExecSpec(R=8, B=4, T=1, ctx=2048)
    tiles = mk.Tiles()

    cp_4096 = mk.comm.comm_params(ex, tiles, 4096)
    assert (cp_4096.PC, cp_4096.HPC, cp_4096.PR) == (1024, 4, 16)

    cp_2560 = mk.comm.comm_params(ex, tiles, 2560)
    assert (cp_2560.PC, cp_2560.HPC, cp_2560.PR) == (640, 4, 16)

    # Custom secondary ring width (e.g. HCSublayer with width=128)
    cp_hc = mk.comm.comm_params(ex, tiles, 128)
    assert (cp_hc.PC, cp_hc.HPC, cp_hc.PR) == (128, 1, 16)
    scr = mk.comm.ring_scratch(cp_hc)
    assert "send" in scr and "recv" in scr and "ar_ssem" in scr and "ar_rsem" in scr

