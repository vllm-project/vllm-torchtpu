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
"""Model-free unit tests for `ModelDef`, `derive.py`, and `reference.py`."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from tests.megakernel.megakernel_fakes import TINY_TILES
from vllm_torchtpu.megakernel import (
    derive,
    geometry,
    model_def,
    module,
    pool,
    profile,
    reference,
    schedule,
    spec,
    validate,
)
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    get_model,
    list_models,
    register_model,
)
from vllm_torchtpu.megakernel.modules import (
    GDN,
    GQA,
    Embedding,
    GDNModule,
    GQAModule,
    MoE,
    MoEModule,
    Norm,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, round_bf16
from vllm_torchtpu.megakernel.pool import PoolBackend, with_cache
from vllm_torchtpu.megakernel.profile import DEFAULT_PROFILE, Profile
from vllm_torchtpu.megakernel.spec import ModelSpec, Module, SwitchSlot


@pytest.fixture()
def isolated_model_registry(monkeypatch: pytest.MonkeyPatch):
    """Snapshot and restore model and profile registries around mutation tests."""
    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))


def assert_modules_equal(ba: Module, bb: Module) -> None:
    """Assert structural equality of two `Module` instances including `prefix`."""
    assert type(ba) is type(bb)
    assert (
        ba.name,
        ba.reduction,
        ba.input_norm,
        ba.prefix,
        ba.options,
    ) == (
        bb.name,
        bb.reduction,
        bb.input_norm,
        bb.prefix,
        bb.options,
    )
    if isinstance(ba, GQAModule):
        assert isinstance(bb, GQAModule)
        assert (ba.pool_layout, ba.qk_norm, ba.output_gate) == (
            bb.pool_layout,
            bb.qk_norm,
            bb.output_gate,
        )
    elif isinstance(ba, GDNModule):
        assert isinstance(bb, GDNModule)
        assert ba.prefix == bb.prefix
    elif isinstance(ba, MoEModule):
        assert isinstance(bb, MoEModule)
        assert (
            ba.shared_gate,
            ba._explicit_moe_tp,
            ba._explicit_e_slots,
            ba.moe_tp,
            ba.e_slots,
        ) == (
            bb.shared_gate,
            bb._explicit_moe_tp,
            bb._explicit_e_slots,
            bb.moe_tp,
            bb.e_slots,
        )
    elif isinstance(ba, Norm):
        assert isinstance(bb, Norm)
        assert (ba.zero_centered, ba.eps, ba.param_name) == (
            bb.zero_centered,
            bb.eps,
            bb.param_name,
        )
    elif isinstance(ba, Embedding):
        assert isinstance(bb, Embedding)
        assert (ba.param_name, ba.dtype) == (bb.param_name, bb.dtype)


def assert_pool_backends_equal(ba: PoolBackend | None, bb: PoolBackend | None) -> None:
    """Assert structural equality of two `PoolBackend` instances including callbacks."""
    if ba is None or bb is None:
        assert ba is bb
        return
    assert type(ba) is type(bb) is PoolBackend
    assert (
        ba.layout,
        ba.kernel_names,
        ba.page_table_state_name,
        ba.supports_fused_io,
        ba.supports_pool_gather,
    ) == (
        bb.layout,
        bb.kernel_names,
        bb.page_table_state_name,
        bb.supports_fused_io,
        bb.supports_pool_gather,
    )
    for attr in ("gather_states", "scatter_states", "can_pool_gather"):
        fn_a = getattr(ba, attr)
        fn_b = getattr(bb, attr)
        assert callable(fn_a) and callable(fn_b)
        assert fn_a.__code__ is fn_b.__code__
        assert fn_a.__qualname__ == fn_b.__qualname__
        nl_a = inspect.getclosurevars(fn_a).nonlocals
        nl_b = inspect.getclosurevars(fn_b).nonlocals
        assert set(nl_a) == set(nl_b)
        assert nl_a["layout"] == nl_b["layout"]
        if "has_rec" in nl_a:
            assert nl_a["has_rec"] == nl_b["has_rec"]
        if "rec_chunk" in nl_a:
            assert nl_a["rec_chunk"] == nl_b["rec_chunk"]
        if "check" in nl_a:
            chk_a, chk_b = nl_a["check"], nl_b["check"]
            assert (
                getattr(chk_a, "__func__", chk_a),
                getattr(chk_a, "__self__", None),
            ) == (
                getattr(chk_b, "__func__", chk_b),
                getattr(chk_b, "__self__", None),
            )
    for ex_probe in (
        with_cache(ExecSpec(R=8, B=8, T=1, ctx=256), paged_kv=True),
        with_cache(
            ExecSpec(R=8, B=24, T=4, ctx=256),
            paged_kv=True,
            pool_gather="eligible",
        ),
    ):
        assert ba.can_pool_gather(ex_probe) == bb.can_pool_gather(ex_probe)


def assert_specs_equal(spec_a: ModelSpec, spec_b: ModelSpec) -> None:
    """Assert structural equality of two `ModelSpec` instances."""
    assert type(spec_a) is type(spec_b) is ModelSpec
    assert spec_a.name == spec_b.name
    assert type(spec_a.config) is type(spec_b.config)
    assert spec_a.config == spec_b.config
    assert_pool_backends_equal(spec_a.cache_backend, spec_b.cache_backend)
    assert spec_a.layer_type.name == spec_b.layer_type.name
    assert len(spec_a.layer_type.slots) == len(spec_b.layer_type.slots)
    for sa, sb in zip(spec_a.layer_type.slots, spec_b.layer_type.slots, strict=True):
        if isinstance(sa, SwitchSlot):
            assert isinstance(sb, SwitchSlot)
            assert sa.name == sb.name
            assert sa.branch_counts == sb.branch_counts
            assert sa.period == sb.period
            for ba, bb in zip(sa.branches, sb.branches, strict=True):
                assert_modules_equal(ba, bb)
        else:
            assert isinstance(sa, Module) and isinstance(sb, Module)
            assert_modules_equal(sa, sb)
    for ba, bb in zip(spec_a.all_modules(), spec_b.all_modules(), strict=True):
        assert_modules_equal(ba, bb)


class _DenseMLPModule(MoEModule):
    """Test-only dense-FFN override module (shared SwiGLU MLP, no routed experts)."""

    routed_experts = False

    def reference(
        self,
        params: Mapping[str, Any],
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        ex: ExecSpec,
        layer_idx: int = 0,
        moe_tp: bool | None = None,
    ) -> tuple[jax.Array, Any]:
        del moe_tp
        geom = self._require_geometry(g)
        R, B = ex.R, ex.B
        ISR = int(geom.IS) // R
        x_f32 = x[:B].astype(F32)
        xb = x_f32.astype(BF16).astype(F32)
        sh_gate_l = params["sh_gate"][:, layer_idx]
        sh13_l = params["sh13"][:, layer_idx]
        sh13_s_l = params["sh13_s"][:, layer_idx]
        sh2_l = params["sh2"][:, layer_idx]
        sh2_s_l = params["sh2_s"][:, layer_idx]
        partials = []
        for r in range(R):
            sg = jax.nn.sigmoid(
                jnp.sum(x_f32 * sh_gate_l[r][None, :], axis=-1, keepdims=True)
            )
            g13 = (xb @ sh13_l[r].astype(F32)) * sh13_s_l[r]
            mid = (
                round_bf16(
                    round_bf16(jax.nn.silu(round_bf16(g13[:, :ISR])))
                    * round_bf16(g13[:, ISR:])
                )
                .astype(BF16)
                .astype(F32)
            )
            part_r = (mid @ sh2_l[r].astype(F32)) * sh2_s_l[r] * sg
            partials.append(part_r.astype(BF16).astype(F32))
        return jnp.sum(jnp.stack(partials, axis=0), axis=0), state


def test_core_model_def_modules_have_no_qwen_literals_and_fit_line_budget():
    """Core `ModelDef`, `weights/`, `runtime/` have no Qwen names and <= 400 lines."""
    pkg_dir = Path(derive.__file__).parent
    weight_files = sorted((pkg_dir / "weights").glob("*.py"))
    runtime_files = sorted((pkg_dir / "runtime").glob("*.py"))
    core_files = (
        [
            Path(mod.__file__)
            for mod in (
                module,
                spec,
                derive,
                geometry,
                schedule,
                reference,
                validate,
                model_def,
            )
        ]
        + weight_files
        + runtime_files
    )
    for path in core_files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 400, f"{path.name} has {len(lines)} lines (> 400)"
        assert "fmt: skip" not in text, f"{path.name} contains fmt: skip"
        assert "noqa: E501" not in text, f"{path.name} contains noqa: E501"
        if path in weight_files or path in runtime_files:
            assert "noqa" not in text, f"{path.name} contains noqa"
        low = text.lower()
        assert "qwen" not in low, f"{path.name} must not mention Qwen"

    b1 = MoEModule(prefix="mlp")
    b2 = MoEModule(prefix="mlp")
    b3 = MoEModule(prefix="other")
    assert b1 != b2 and b1 != b3
    assert_modules_equal(b1, b2)
    with pytest.raises(AssertionError):
        assert_modules_equal(b1, b3)

    mg = geometry.ModelGeometry(
        L=8,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=256,
        ROT=64,
        KH=8,
        VH=16,
        KD=128,
        VD=128,
        E=128,
        TOPK=4,
        I=1024,
        IS=1024,
    )
    kv_decl = GQAModule(geometry=mg).cache_decl(mg)
    rec_decl = GDNModule(geometry=mg).cache_decl(mg)
    assert kv_decl is not None and rec_decl is not None
    play = pool.derive_pool_layout(([rec_decl] * 3 + [kv_decl]) * 2, R=8)
    pb1 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    pb2 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    pb3 = pool.make_pool_backend(
        play,
        check=dataclasses.replace(mg, L=16).check,
        rec_chunk=GDNModule.chunk_rows,
    )
    assert pb1 != pb2 and pb1 != pb3
    assert_pool_backends_equal(pb1, pb2)
    with pytest.raises(AssertionError):
        assert_pool_backends_equal(pb1, pb3)

    with pytest.raises(MegakernelConfigError, match="divisible by att_every"):
        dataclasses.replace(mg, L=5).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid B=3, T=2"):
        mg.check(ExecSpec(R=2, B=3, T=2, ctx=128))


def test_schedule_validation_errors_name_layer_index_and_value(isolated_model_registry):
    """Unknown layer types, bad mixer order, norm mismatches, and OOB overrides fail."""
    attn = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)

    sched_hf = Schedule(
        mixer=FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": gdn},
        ),
        ffn=moe,
        pre_norm=in_norm,
        post_norm=post_norm,
    )

    valid_hf_cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500_000.0,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "k_heads": 8,
        "v_heads": 16,
        "k_dim": 128,
        "v_dim": 128,
        "conv_kernel": 4,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "layer_types": ["linear_attention", "full_attention"],
    }
    with pytest.raises(MegakernelConfigError, match=r"layer index 3.*num_layers=2"):

        @register_model(name="bad_oob_model")
        class _BadOOBModel(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden=("hidden_size", 1024),
                layers=("num_hidden_layers", 4),
                eps=("rms_norm_eps", 1e-6),
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=in_norm,
                post_norm=post_norm,
                overrides={3: {"ffn": moe}},
            )
            final_norm = final_norm_b

    with pytest.raises(MegakernelConfigError, match=r"mamba2.*layer 1"):

        @register_model(name="bad_lt_model")
        class _BadLTModel(ModelDef):
            default_hf_config = {
                **valid_hf_cfg,
                "layer_types": ["linear_attention", "mamba2"],
            }
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = sched_hf
            final_norm = final_norm_b

    with pytest.raises(
        MegakernelConfigError, match=r"bad_missing_field_model.*num_attention_heads"
    ):

        @register_model(name="bad_missing_field_model")
        class _BadMissingFieldModel(ModelDef):
            default_hf_config = {
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6,
            }
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=in_norm,
                post_norm=post_norm,
            )
            final_norm = final_norm_b

    with pytest.raises(MegakernelConfigError, match=r"zero_centered=False"):

        @register_model(name="bad_norm_zc_model")
        class _BadNormZCModel(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=Norm.bind("input_layernorm", zero_centered=True),
                post_norm=Norm.bind("post_attention_layernorm", zero_centered=False),
            )
            final_norm = final_norm_b

    with pytest.raises(MegakernelConfigError, match=r"layer 0 slot 'pre_norm' is None"):

        @register_model(name="bad_missing_pre_norm")
        class _BadMissingPreNorm(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=None,
                post_norm=post_norm,
            )
            final_norm = final_norm_b

    with pytest.raises(
        MegakernelConfigError, match=r"layer 0 slot 'post_norm' is None"
    ):

        @register_model(name="bad_missing_post_norm")
        class _BadMissingPostNorm(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=in_norm,
                post_norm=None,
            )
            final_norm = final_norm_b

    with pytest.raises(MegakernelConfigError, match=r"final_norm is None"):

        @register_model(name="bad_missing_final_norm")
        class _BadMissingFinalNorm(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=in_norm,
                post_norm=post_norm,
            )
            final_norm = None

    @register_model(name="no_def_cfg_custom_eps")
    class _NoDefCfgCustomEps(ModelDef):
        geometry = FromHF(
            hidden="hidden_size",
            layers=("num_hidden_layers", 2),
            eps="rms_norm_eps",
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True, eps=1e-5),
            post_norm=Norm.bind(
                "post_attention_layernorm", zero_centered=True, eps=1e-5
            ),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True, eps=1e-5)

    with pytest.raises(
        MegakernelConfigError, match=r"layer 1 slot 'post_norm' is None"
    ):

        @register_model(name="bad_override_post_norm_none")
        class _BadOverridePostNormNone(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=in_norm,
                post_norm=post_norm,
                overrides={1: {"post_norm": None}},
            )
            final_norm = final_norm_b


def test_model_registry_lookups_duplicates_and_profile_integration(
    isolated_model_registry,
):
    """Test `register_model`, `get_model`, `find_model`, `replace`, and profiles."""
    attn = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)
    synth_prof = Profile(
        name="synth-default",
        model="synth_reg_test",
        numerics="exact",
        model_default=True,
    )

    @register_model(
        name="synth_reg_test",
        hf_architectures=("SynthForCausalLM", "SynthForConditionalGeneration"),
        hf_model_type=("synth_moe", "synth"),
    )
    class SynthRegModel(ModelDef):
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            eps="rms_norm_eps",
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            pre_norm=in_norm,
            post_norm=post_norm,
        )
        final_norm = final_norm_b
        profiles = {"default": synth_prof}

    assert (
        register_model(
            name="synth_reg_test",
            hf_architectures=("SynthForCausalLM", "SynthForConditionalGeneration"),
            hf_model_type=("synth_moe", "synth"),
        )(SynthRegModel)
        is SynthRegModel
    )

    assert "synth_reg_test" in list_models()
    assert get_model("synth_reg_test") is SynthRegModel
    assert get_model("SynthForCausalLM") is SynthRegModel
    assert get_model("synth_moe") is SynthRegModel
    assert (
        get_model({"architectures": ["SynthForConditionalGeneration"]}) is SynthRegModel
    )
    assert get_model({"text_config": {"model_type": "synth"}}) is SynthRegModel
    assert SynthRegModel.default_profile() is synth_prof
    assert SynthRegModel.get_profile("default") is synth_prof
    assert find_model("nonexistent_model") is None
    with pytest.raises(KeyError, match="Unknown megakernel model"):
        get_model("nonexistent_model")

    with pytest.raises(MegakernelConfigError, match="already registered"):

        @register_model(name="synth_reg_test")
        class _DuplicateModel(ModelDef):
            geometry = SynthRegModel.geometry._from_hf
            schedule = SynthRegModel.schedule
            final_norm = final_norm_b

    @register_model(
        name="synth_reg_test",
        hf_architectures=("SynthV2ForCausalLM",),
        hf_model_type="synth_v2",
        replace=True,
    )
    class SynthRegModelV2(ModelDef):
        geometry = SynthRegModel.geometry._from_hf
        schedule = SynthRegModel.schedule
        final_norm = final_norm_b

    assert get_model("synth_reg_test") is SynthRegModelV2
    assert get_model("SynthV2ForCausalLM") is SynthRegModelV2
    assert get_model("synth_v2") is SynthRegModelV2
    assert find_model("SynthForCausalLM") is None
    assert find_model("synth_moe") is None


def test_synthetic_model_def_every2_spec_reference_and_override_errors(
    isolated_model_registry,
):
    """Synthetic `ModelDef` (`Every(n=2)`) spec/pool derivation and `reference_step`."""
    synth_hf_cfg = {
        "model_type": "synth_hybrid",
        "architectures": ["SynthHybridForCausalLM"],
        "vocab_size": 64000,
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.25,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }

    @register_model(
        name="synth_hybrid",
        hf_architectures=("SynthHybridForCausalLM",),
        hf_model_type="synth_hybrid",
    )
    class SynthHybrid(ModelDef):
        default_hf_config = synth_hf_cfg
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            vocab="vocab_size",
            eps="rms_norm_eps",
            rope_theta="rope_theta",
            rope_partial=("partial_rotary_factor", 1.0),
        )
        attn = GQA.bind(
            prefix="self_attn",
            heads="num_attention_heads",
            kv_heads="num_key_value_heads",
            head_dim="head_dim",
            qk_norm=True,
            output_gate=True,
        )
        linear = GDN.bind(
            prefix="linear_attn",
            k_heads="linear_num_key_heads",
            v_heads="linear_num_value_heads",
            k_dim="linear_key_head_dim",
            v_dim="linear_value_head_dim",
            conv_kernel="linear_conv_kernel_dim",
        )
        moe = MoE.bind(
            prefix="mlp",
            experts="num_experts",
            top_k="num_experts_per_tok",
            inter="moe_intermediate_size",
            shared_inter="shared_expert_intermediate_size",
            shared_gate=True,
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=linear),
            ffn=moe,
            pre_norm=Norm.bind("attn_norm", zero_centered=True),
            post_norm=Norm.bind("ffn_norm", zero_centered=True),
        )
        embed = Embedding("model.embed_tokens")
        final_norm = Norm.bind("final_ln", zero_centered=True)
        lm_head = LMHead("lm_head", tied_to=None)

    assert SynthHybrid.default_profile() is DEFAULT_PROFILE
    g_full = SynthHybrid.geometry()
    assert (g_full.L, g_full.att_every, g_full.H, g_full.NH, g_full.E, g_full.TOPK) == (
        12,
        2,
        2048,
        4,
        256,
        2,
    )
    assert (g_full.NGDN, g_full.NATT) == (6, 6)

    ex = ExecSpec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        interpret=True,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10, nbd=6),
    )
    g_tiny = SynthHybrid.shrink(R=ex.R)
    assert (
        g_tiny.L,
        g_tiny.att_every,
        g_tiny.NH,
        g_tiny.KH,
        g_tiny.VH,
        g_tiny.TOPK,
        g_tiny.I,
        g_tiny.IS,
    ) == (4, 2, 4, 4, 8, 2, 512, 512)
    g_tiny.check(ex)

    g_unbound = dataclasses.replace(g_tiny)
    assert getattr(g_unbound, "_model_def", None) is None
    ms = SynthHybrid.to_model_spec(g_unbound, R=ex.R)
    assert getattr(g_unbound, "_model_def", None) is None
    assert ms.name == "synth_hybrid"
    mixer_slot, ffn_slot = ms.layer_type.slots
    assert isinstance(mixer_slot, SwitchSlot)
    assert mixer_slot.branch_counts == (1, 1) and mixer_slot.period == 2
    assert isinstance(ffn_slot, MoEModule)

    play = SynthHybrid.pool_layout(g_tiny, R=ex.R)
    assert play.num_pools == 2 and play.recurrent_slots_per_pool == 1

    pos = jnp.array([4, 8], dtype=I32)
    hid_l = jnp.ones((ex.B, g_tiny.H), dtype=BF16) * 0.1
    nh = g_tiny.NH // ex.R
    nai = (2 * nh + 2) * g_tiny.HD
    lay_gdn = GDNModule.lin_in_layout(g_tiny, ex)
    vh, qkv, n_lin = lay_gdn.vh, lay_gdn.QKV, lay_gdn.NLI
    ew = g_tiny.EW(ex.R)
    ir = g_tiny.I // ex.R if g_tiny.moe_tp else g_tiny.I
    isr = g_tiny.IS // ex.R
    w_zero: dict[str, Any] = {
        "attn_norm": jnp.zeros((g_tiny.L, g_tiny.H), dtype=F32),
        "ffn_norm": jnp.zeros((g_tiny.L, g_tiny.H), dtype=F32),
        "final_ln": jnp.zeros((g_tiny.H,), dtype=F32),
        "att_in": jnp.zeros((ex.R, g_tiny.NATT, g_tiny.H, nai), dtype=BF16),
        "att_out": jnp.zeros((ex.R, g_tiny.NATT, nh * g_tiny.HD, g_tiny.H), dtype=BF16),
        "in_s": jnp.ones((ex.R, g_tiny.NATT, nai), dtype=F32),
        "out_s": jnp.ones((ex.R, g_tiny.NATT, g_tiny.H), dtype=F32),
        "qn": jnp.zeros((ex.R, g_tiny.NATT, g_tiny.HD), dtype=F32),
        "kn": jnp.zeros((ex.R, g_tiny.NATT, g_tiny.HD), dtype=F32),
        "lin_in": jnp.zeros((ex.R, g_tiny.NGDN, g_tiny.H, n_lin), dtype=BF16),
        "lin_out": jnp.zeros((ex.R, g_tiny.NGDN, vh * g_tiny.VD, g_tiny.H), dtype=BF16),
        "lin_in_s": jnp.ones((ex.R, g_tiny.NGDN, n_lin), dtype=F32),
        "lin_out_s": jnp.ones((ex.R, g_tiny.NGDN, g_tiny.H), dtype=F32),
        "conv": jnp.zeros((ex.R, g_tiny.NGDN, 4 * qkv), dtype=F32),
        "alog": jnp.zeros((ex.R, g_tiny.NGDN, vh), dtype=F32),
        "dtb": jnp.zeros((ex.R, g_tiny.NGDN, vh), dtype=F32),
        "gnorm": jnp.ones((ex.R, g_tiny.NGDN, g_tiny.VD), dtype=F32),
        "router": jnp.zeros((ex.R, g_tiny.L, g_tiny.H, g_tiny.E), dtype=BF16),
        "sh_gate": jnp.zeros((ex.R, g_tiny.L, g_tiny.H), dtype=F32),
        "sh13": jnp.zeros((ex.R, g_tiny.L, g_tiny.H, 2 * isr), dtype=BF16),
        "sh13_s": jnp.ones((ex.R, g_tiny.L, 2 * isr), dtype=F32),
        "sh2": jnp.zeros((ex.R, g_tiny.L, isr, g_tiny.H), dtype=BF16),
        "sh2_s": jnp.ones((ex.R, g_tiny.L, g_tiny.H), dtype=F32),
        "e13": jnp.zeros((ex.R, g_tiny.L, ew, g_tiny.H, 2 * ir), dtype=BF16),
        "e13_s": jnp.ones((ex.R, g_tiny.L, ew, 2 * ir), dtype=F32),
        "e2": jnp.zeros((ex.R, g_tiny.L, ew, ir, g_tiny.H), dtype=BF16),
        "e2_s": jnp.ones((ex.R, g_tiny.L, ew, g_tiny.H), dtype=F32),
    }
    st_zero: dict[str, Any] = {
        **GQAModule(geometry=g_tiny).init_logical_states(
            jax.random.PRNGKey(0), g_tiny, ex, g_tiny.NATT
        ),
        **GDNModule(geometry=g_tiny).init_logical_states(
            jax.random.PRNGKey(1), g_tiny, ex, g_tiny.NGDN
        ),
    }
    ref_out, ref_st = SynthHybrid.reference_step(
        g_tiny, ex, pos, hid_l, w_zero, st_zero
    )
    assert ref_out.shape == (ex.B, g_tiny.H) and jnp.all(jnp.isfinite(ref_out))
    assert set(ref_st) == {"kc", "vc", "conv", "rec"}

    dense_mlp = _DenseMLPModule.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    alt_linear = GDN.bind(
        prefix="alt_linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )

    @register_model(name="synth_dense_ov")
    class SynthDenseOverride(SynthHybrid):
        schedule = Schedule(
            mixer=Every(n=2, last=SynthHybrid.attn, rest=SynthHybrid.linear),
            ffn=SynthHybrid.moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={0: {"ffn": dense_mlp}},
        )

    with pytest.raises(
        MegakernelConfigError, match=r"layer 0.*slot 'ffn'.*_DenseMLPModule"
    ):
        SynthDenseOverride.to_model_spec(g_tiny, R=ex.R)

    @register_model(name="synth_mixer_ov")
    class SynthMixerOverride(SynthHybrid):
        schedule = Schedule(
            mixer=Every(n=2, last=SynthHybrid.attn, rest=SynthHybrid.linear),
            ffn=SynthHybrid.moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={0: {"mixer": alt_linear}},
        )

    with pytest.raises(
        MegakernelConfigError, match=r"layer 0.*slot 'mixer'.*alt_linear_attn"
    ) as exc_info:
        SynthMixerOverride.to_model_spec(g_tiny, R=ex.R)
    assert "ModuleBinding(" not in str(exc_info.value)
    assert "FrozenDict(" not in str(exc_info.value)

    eplb_moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
        e_slots=32,
    )

    class _UnregisteredSynthOptOverride(SynthHybrid):
        schedule = Schedule(
            mixer=Every(n=2, last=SynthHybrid.attn, rest=SynthHybrid.linear),
            ffn=SynthHybrid.moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={1: {"ffn": eplb_moe}},
        )

    with pytest.raises(
        MegakernelConfigError,
        match=(
            r"_UnregisteredSynthOptOverride'.*layer 1"
            r".*e_slots=32 \(expected None\)"
        ),
    ) as opt_exc:
        _UnregisteredSynthOptOverride.to_model_spec(g_tiny, R=ex.R)
    assert "ModuleBinding(" not in str(opt_exc.value)
    assert "FrozenDict(" not in str(opt_exc.value)

    class _UnregisteredSynthMissingFinalNorm(SynthHybrid):
        final_norm = None

    for fn_call in (
        lambda: _UnregisteredSynthMissingFinalNorm.derive_geometry(synth_hf_cfg),
        lambda: _UnregisteredSynthMissingFinalNorm.to_model_spec(g_tiny, R=ex.R),
        lambda: _UnregisteredSynthMissingFinalNorm.reference_step(
            g_tiny, ex, pos, hid_l, {}, st_zero
        ),
    ):
        with pytest.raises(
            MegakernelConfigError,
            match=r"_UnregisteredSynthMissingFinalNorm.*final_norm is None",
        ) as fn_exc:
            fn_call()
        assert "layer " not in str(fn_exc.value)
