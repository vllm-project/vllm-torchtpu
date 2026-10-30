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
"""Unit and CPU interpret tests for `ModelDef`, `Schedule`, and `FromHF`."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import (
    derive,
    geometry,
    model_def,
    module,
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
from vllm_torchtpu.megakernel.ops import BF16, F32, rnd
from vllm_torchtpu.megakernel.profile import DEFAULT_PROFILE, Profile
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel import pool as qwen35_pool
from vllm_torchtpu.models.qwen3_5.megakernel.model import DEFAULT_HF_CONFIG, Qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import (
    PROFILE,
    QWEN35_MODEL_NAME,
    TINY_TILES,
)


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


def assert_modules_equal(ba: mk.Module, bb: mk.Module) -> None:
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


def assert_pool_backends_equal(
    ba: mk.PoolBackend | None, bb: mk.PoolBackend | None
) -> None:
    """Assert structural equality of two `PoolBackend` instances including callbacks."""
    if ba is None or bb is None:
        assert ba is bb
        return
    assert type(ba) is type(bb) is mk.PoolBackend
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
        mk.with_cache(ExecSpec(R=8, B=8, T=1, ctx=256), paged_kv=True),
        mk.with_cache(
            ExecSpec(R=8, B=24, T=4, ctx=256),
            paged_kv=True,
            pool_gather="eligible",
        ),
    ):
        assert ba.can_pool_gather(ex_probe) == bb.can_pool_gather(ex_probe)


def assert_specs_equal(spec_a: mk.ModelSpec, spec_b: mk.ModelSpec) -> None:
    """Assert structural equality of two `ModelSpec` instances."""
    assert type(spec_a) is type(spec_b) is mk.ModelSpec
    assert spec_a.name == spec_b.name
    assert type(spec_a.config) is type(spec_b.config)
    assert spec_a.config == spec_b.config
    assert_pool_backends_equal(spec_a.cache_backend, spec_b.cache_backend)
    assert spec_a.layer_type.name == spec_b.layer_type.name
    assert len(spec_a.layer_type.slots) == len(spec_b.layer_type.slots)
    for sa, sb in zip(spec_a.layer_type.slots, spec_b.layer_type.slots, strict=True):
        if isinstance(sa, mk.SwitchSlot):
            assert isinstance(sb, mk.SwitchSlot)
            assert sa.name == sb.name
            assert sa.branch_counts == sb.branch_counts
            assert sa.period == sb.period
            for ba, bb in zip(sa.branches, sb.branches, strict=True):
                assert_modules_equal(ba, bb)
        else:
            assert isinstance(sa, mk.Module) and isinstance(sb, mk.Module)
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
                rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
                .astype(BF16)
                .astype(F32)
            )
            part_r = (mid @ sh2_l[r].astype(F32)) * sh2_s_l[r] * sg
            partials.append(part_r.astype(BF16).astype(F32))
        return jnp.sum(jnp.stack(partials, axis=0), axis=0), state


def test_core_model_def_modules_have_no_qwen_literals_and_fit_line_budget():
    """Core `ModelDef`, `weights/`, `runtime/` have no Qwen names and <= 400 lines."""
    from vllm_torchtpu.megakernel import runtime as runtime_pkg
    from vllm_torchtpu.megakernel import weights as weights_pkg

    weights_dir = Path(weights_pkg.__file__).parent
    weight_files = sorted(weights_dir.glob("*.py"))
    assert len(weight_files) >= 4
    runtime_dir = Path(runtime_pkg.__file__).parent
    runtime_files = sorted(runtime_dir.glob("*.py"))
    assert len(runtime_files) >= 5
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

    # Module and PoolBackend equality remain standard (callbacks compared in helper)
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
    play = qwen35_pool.qwen35_pool_layout(qwen35.TINY, R=8)
    pb1 = mk.pool.make_pool_backend(
        play, check=mg.check, rec_chunk=GDNModule.chunk_rows
    )
    pb2 = mk.pool.make_pool_backend(
        play, check=mg.check, rec_chunk=GDNModule.chunk_rows
    )
    pb3 = mk.pool.make_pool_backend(
        play,
        check=dataclasses.replace(mg, L=16).check,
        rec_chunk=GDNModule.chunk_rows,
    )
    assert pb1 != pb2 and pb1 != pb3
    assert_pool_backends_equal(pb1, pb2)
    with pytest.raises(AssertionError):
        assert_pool_backends_equal(pb1, pb3)

    # ModelGeometry.check() raises MegakernelConfigError instead of bare AssertionError
    with pytest.raises(MegakernelConfigError, match="divisible by att_every"):
        dataclasses.replace(mg, L=5).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid B=3, T=2"):
        mg.check(ExecSpec(R=2, B=3, T=2, ctx=128))


def test_from_hf_strict_resolution_and_nested_text_config():
    """`FromHF` requires known keys, declared HF field names, and `text_config`."""
    with pytest.raises(MegakernelConfigError, match="rope_thta"):
        FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            eps="rms_norm_eps",
            rope_thta="rope_theta",
        )

    decl = FromHF(
        hidden="custom_hidden_size",
        layers="custom_num_layers",
        vocab=("custom_vocab_size", 32000),
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.5),
    )
    # Canonical name alone ("hidden") does not satisfy explicit "custom_hidden_size"
    with pytest.raises(ValueError, match="custom_hidden_size"):
        decl.resolve({"hidden": 2048, "custom_num_layers": 8, "rms_norm_eps": 1e-5})

    nested_cfg = {
        "model_type": " wrapper",
        "text_config": {
            "custom_hidden_size": 2048,
            "custom_num_layers": 8,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500_000.0},
        },
    }
    res = decl.resolve(nested_cfg)
    assert dict(res) == {
        "hidden": 2048,
        "layers": 8,
        "vocab": 32000,
        "eps": 1e-5,
        "rope_theta": 500_000.0,
        "rope_partial": 0.5,
    }


def test_schedule_validation_errors_name_layer_index_and_value(isolated_model_registry):
    """Unknown layer types, bad mixer order, norm mismatches, and OOB overrides fail."""
    attn = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)

    # 1. Layers without a module in Every / FromHFLayerTypes / Schedule / overrides
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*None"):
        Every(n=2, last=attn, rest=None)
    with pytest.raises(MegakernelConfigError, match=r"layer 1.*None"):
        Every(n=2, last=None, rest=gdn)  # type: ignore[arg-type]
    with pytest.raises(MegakernelConfigError, match=r"Every\(n=1\).*rest"):
        Every(n=1, last=attn, rest=gdn)
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*None"):
        FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": None},  # type: ignore[dict-item]
        )
    with pytest.raises(MegakernelConfigError, match=r"layer 1.*None"):
        Schedule(mixer=[gdn, None], ffn=moe)
    with pytest.raises(MegakernelConfigError, match=r"layer 2.*None"):
        Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            overrides={2: {"ffn": None}},
        )

    # 2. Reversed mixer order (GQA before GDN) fails closed naming layer 0, and
    #    an override BEFORE the first attention layer names the first bad layer
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*gqa"):
        Schedule(mixer=Every(n=4, last=gdn, rest=attn), ffn=moe).resolve({}, 8)
    with pytest.raises(MegakernelConfigError, match=r"layer 1 has 'gqa'") as mix_ov_exc:
        Schedule(
            mixer=Every(n=4, last=attn, rest=gdn),
            ffn=moe,
            overrides={1: {"mixer": attn}},
        ).resolve({}, 8)
    assert "ModuleBinding(" not in str(mix_ov_exc.value)
    assert "FrozenDict(" not in str(mix_ov_exc.value)
    with pytest.raises(MegakernelConfigError, match=r"layer 1 has 'gqa'"):
        Schedule(
            mixer=FromHFLayerTypes(
                field="layer_types",
                mapping={"full_attention": attn, "linear_attention": gdn},
            ),
            ffn=moe,
        ).resolve(
            {
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ]
            },
            8,
        )

    # 3. Unknown layer type in FromHFLayerTypes names the layer index and bad value
    sched_hf = Schedule(
        mixer=FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": gdn},
        ),
        ffn=moe,
        pre_norm=in_norm,
        post_norm=post_norm,
    )
    with pytest.raises(MegakernelConfigError, match=r"sliding_attention.*layer 2"):
        sched_hf.resolve(
            {
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ]
            },
            4,
        )

    # Irregular base layer_types with L % #attention == 0 uses max-match period
    # heuristic (3/4 layers match period=2 vs 2/4 for period=4, so layer 3 is flagged)
    with pytest.raises(
        MegakernelConfigError, match=r"layer 3.*expected 'gqa' for period=2"
    ):
        sched_hf.resolve(
            {
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                ]
            },
            4,
        )

    # 4. Override indices out of range (negative or >= num_layers)
    with pytest.raises(MegakernelConfigError, match=r"-1.*out of range"):
        Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            overrides={-1: {"ffn": moe}},
        )
    sched_oob = Schedule(
        mixer=Every(n=2, last=attn, rest=gdn),
        ffn=moe,
        overrides={6: {"ffn": moe}},
    )
    with pytest.raises(MegakernelConfigError, match=r"layer index 6.*num_layers=4"):
        sched_oob.resolve({}, 4)

    # 5. Registration-time validation: prefers default_hf_config over FromHF tuple
    #    default for static_l, derives geometry/pool_layout, and catches norm mismatches
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

    # 6. Missing pre_norm / post_norm / final_norm fails closed naming layer and slot
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

    # Registration without default_hf_config compares only zero_centered on norms
    # (does not fail on explicit non-1e-6 eps literals)
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

    # Idempotent re-registration of the same class
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

    # Duplicate registration by a different class fails closed
    with pytest.raises(MegakernelConfigError, match="already registered"):

        @register_model(name="synth_reg_test")
        class _DuplicateModel(ModelDef):
            geometry = SynthRegModel.geometry._from_hf
            schedule = SynthRegModel.schedule
            final_norm = final_norm_b

    # replace=True cleans up stale architecture and model_type keys
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


def test_synthetic_model_def_every2_dense_override_cpu_kernel_vs_reference(
    isolated_model_registry,
):
    """Synthetic `ModelDef` (`Every(n=2)`) CPU kernel vs ref + override fail-closed."""
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
    # shrink(R=2) preserves SynthHybrid's distinct NH=4, KH=4, VH=8, TOPK=2, I=IS=512
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

    # Calling to_model_spec on an unbound ModelGeometry does not mutate it
    g_unbound = dataclasses.replace(g_tiny)
    assert getattr(g_unbound, "_model_def", None) is None
    ms = SynthHybrid.to_model_spec(g_unbound, R=ex.R)
    assert getattr(g_unbound, "_model_def", None) is None
    assert ms.name == "synth_hybrid"
    mixer_slot, ffn_slot = ms.layer_type.slots
    assert isinstance(mixer_slot, mk.SwitchSlot)
    assert mixer_slot.branch_counts == (1, 1) and mixer_slot.period == 2
    assert isinstance(ffn_slot, MoEModule)

    play = SynthHybrid.pool_layout(g_tiny, R=ex.R)
    assert play.num_pools == 2 and play.recurrent_slots_per_pool == 1

    # Load HF weights via generic loaders on SynthHybrid (att_every=2 != 4, custom
    # norm prefixes attn_norm / ffn_norm / final_ln) and feed them into the CPU
    # kernel vs composed reference check.
    hf_synth = mk.random_hf_weights(jax.random.PRNGKey(42), g_tiny)
    assert {"attn_norm", "ffn_norm", "final_ln"} <= set(hf_synth)
    assert {"in_norm", "post_norm", "final_norm"}.isdisjoint(hf_synth)

    w_l_loaded = mk.load_logical_weights(SynthHybrid, ex, hf_synth, g=g_tiny)
    assert {"attn_norm", "ffn_norm", "final_ln"} <= set(w_l_loaded)
    w_mk_packed = mk.pack_logical_weights(SynthHybrid, ex, w_l_loaded, g=g_tiny)
    for r in range(ex.R):
        w_rank = mk.load_weights(
            SynthHybrid,
            ex,
            mk.MappingWeightSource(w_l_loaded),
            g=g_tiny,
            tp_rank=r,
        )
        assert set(w_rank) == set(w_mk_packed) | {"plan", "w_flat"}
        for k in w_mk_packed:
            if isinstance(w_mk_packed[k], tuple):
                for lyr, (got_l, exp_l) in enumerate(
                    zip(w_rank[k], w_mk_packed[k], strict=True)
                ):
                    np.testing.assert_array_equal(
                        np.asarray(got_l).view(np.uint8),
                        np.asarray(exp_l[r]).view(np.uint8),
                        err_msg=f"SynthHybrid rank {r} layer {lyr} mismatch on {k}",
                    )
            else:
                np.testing.assert_array_equal(
                    np.asarray(w_rank[k]).view(np.uint8),
                    np.asarray(w_mk_packed[k][r]).view(np.uint8),
                    err_msg=f"SynthHybrid rank {r} mismatch on {k}",
                )

    pos, hid_l, _, st_l = mk.random_logical_inputs(jax.random.PRNGKey(42), g_tiny, ex)
    ref_out, ref_st = SynthHybrid.reference_step(
        g_tiny, ex, pos, hid_l, w_l_loaded, st_l
    )

    devs = jax.devices("cpu")[: ex.R]
    mesh = jax.sharding.Mesh(np.asarray(devs), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l_loaded, st_l, g_tiny, ex, mesh=mesh
    )
    step_fn, _ = mk.make_step(mesh, ms, ex)
    out_mk, st_out_mk = step_fn(pos_mk, hid_mk, w_mk, st_mk)
    out_mk_unfolded = mk.unfold(out_mk, ex.R)
    for r in range(ex.R):
        out_r = out_mk_unfolded[r, : ex.B].astype(F32)
        ref_f32 = ref_out.astype(F32)
        max_diff = float(jnp.max(jnp.abs(out_r - ref_f32)))
        cos = float(
            jnp.sum(out_r * ref_f32)
            / (jnp.linalg.norm(out_r) * jnp.linalg.norm(ref_f32))
        )
        assert max_diff < 0.05, f"rank {r}: max_diff={max_diff}"
        assert cos > 0.9999, f"rank {r}: cos={cos}"

    kc_mk = mk.unfold(st_out_mk["kc"], ex.R)
    vc_mk = mk.unfold(st_out_mk["vc"], ex.R)
    conv_mk = mk.unfold(st_out_mk["conv"], ex.R)
    rec_mk = mk.unfold(st_out_mk["rec"], ex.R)
    assert jnp.allclose(
        kc_mk.astype(F32), ref_st["kc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(
        vc_mk.astype(F32), ref_st["vc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(conv_mk, ref_st["conv"], atol=1e-2, rtol=1e-2)
    assert jnp.allclose(
        rec_mk.astype(F32), ref_st["rec"].astype(F32), atol=5e-2, rtol=2e-2
    )

    # Per-layer overrides that the single-LayerType kernel cannot represent fail closed
    # in to_model_spec naming the layer, the slot, and the binding.
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
            g_tiny, ex, pos, hid_l, {}, st_l
        ),
    ):
        with pytest.raises(
            MegakernelConfigError,
            match=r"_UnregisteredSynthMissingFinalNorm.*final_norm is None",
        ) as fn_exc:
            fn_call()
        assert "layer " not in str(fn_exc.value)


def _independent_qwen35_pool_layout(
    g: qwen35.Qwen35Geometry, *, R: int = 8
) -> mk.PoolLayout:
    """Build expected `PoolLayout` from `cache_decl()` + `derive_pool_layout`."""
    kv_decl = GQAModule(geometry=g).cache_decl(g)
    assert kv_decl is not None
    if g.att_every == 1:
        decls: list[Any] = [kv_decl] * g.NATT
    else:
        rec_decl = GDNModule(geometry=g).cache_decl(g)
        assert rec_decl is not None
        decls = ([rec_decl] * (g.att_every - 1) + [kv_decl]) * g.NATT
    return mk.pool.derive_pool_layout(decls, R=R)


def _independent_qwen35_spec(
    g: qwen35.Qwen35Geometry, *, gdn_module: mk.Module | None = None, R: int = 8
) -> mk.ModelSpec:
    """Build expected `ModelSpec` independently of `Qwen35.to_model_spec`."""
    play = _independent_qwen35_pool_layout(g, R=R)
    gqa_mod = GQAModule(
        pool_layout=play, prefix="self_attn", qk_norm=True, output_gate=True
    )
    moe_mod = MoEModule(prefix="mlp", shared_gate=True)
    if g.att_every == 1:
        mixer = mk.SwitchSlot("mixer", branches=(gqa_mod,), branch_counts=(1,))
    else:
        gdn_mod = GDNModule(prefix="linear_attn") if gdn_module is None else gdn_module
        mixer = mk.SwitchSlot(
            "mixer", branches=(gdn_mod, gqa_mod), branch_counts=(g.att_every - 1, 1)
        )
    lt = mk.LayerType(name="qwen35_layer", slots=(mixer, moe_mod))
    backend = mk.pool.make_pool_backend(
        play, check=g.check, rec_chunk=GDNModule.chunk_rows
    )
    return mk.ModelSpec(
        name=QWEN35_MODEL_NAME,
        config=g,
        layer_type=lt,
        cache_backend=backend,
    )


def test_qwen35_model_def_line_budget_docstrings_and_runtime_registry(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """The Qwen3.5 model folder's `model.py` has no fmt: skip / noqa, keeps
    docstrings, tests driver."""
    import logging
    import types

    from vllm_torchtpu.megakernel_runtime import driver as driver_mod
    from vllm_torchtpu.models.qwen3_5.megakernel import adapter as qwen_adapter
    from vllm_torchtpu.models.qwen3_5.megakernel.profiles import (
        baseline as qwen35_profile,
    )

    def _declined(drv: Any) -> bool:
        # A checkpoint the template doesn't serve: `create` declines, so every
        # rank runs stock under a `DeclinedDriver`.
        return isinstance(drv, driver_mod.DeclinedDriver)

    monkeypatch.setattr(
        driver_mod,
        "logger",
        logging.getLogger("vllm_torchtpu.megakernel_runtime.driver"),
    )

    qwen_text = Path(qwen35.__file__).read_text(encoding="utf-8")
    qwen_lines = qwen_text.splitlines()
    # 200 lines plus the explicit `register()` (no import-time registration).
    assert len(qwen_lines) <= 230, f"model.py has {len(qwen_lines)} lines (> 230)"
    assert "fmt: skip" not in qwen_text, "qwen35.py contains fmt: skip"
    assert "noqa" not in qwen_text, "qwen35.py contains noqa"
    for mod in (qwen35_pool, qwen35_profile):
        p = Path(mod.__file__)
        mod_text = p.read_text(encoding="utf-8")
        n_lines = len(mod_text.splitlines())
        assert n_lines <= 400, f"{p.name} has {n_lines} lines (> 400)"
        assert "fmt: skip" not in mod_text, f"{p.name} contains fmt: skip"
        assert "noqa" not in mod_text, f"{p.name} contains noqa"

    # Docstrings and explicit keyword parameters are preserved on Qwen3.5 entry points
    for fn in (
        qwen35.Qwen35Geometry.to_model_spec,
        qwen35.qwen35_spec,
        qwen35.build,
        qwen35.make_step,
        qwen35.make_pooled_step,
    ):
        assert fn.__doc__ and fn.__doc__.strip(), f"{fn.__name__} missing docstring"
    assert "checkpoint slots" in (qwen35_pool.__doc__ or "")
    assert "donate_states" in inspect.signature(qwen35.make_step).parameters
    pooled_params = inspect.signature(qwen35.make_pooled_step).parameters
    for kw_name in ("donate_pools", "fused_pool_io", "max_b_low", "min_b_hi"):
        assert kw_name in pooled_params

    # Model registry and profile lookups (after the explicit registration)
    assert qwen35.register() is Qwen35
    assert get_model(QWEN35_MODEL_NAME) is Qwen35
    assert get_model("Qwen3_5MoeForConditionalGeneration") is Qwen35
    assert get_model("Qwen3_5ForConditionalGeneration") is Qwen35
    assert get_model("qwen3_5_moe") is Qwen35
    assert get_model("qwen3_5_moe_text") is Qwen35
    assert (
        get_model(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": DEFAULT_HF_CONFIG,
            }
        )
        is Qwen35
    )
    assert Qwen35.default_profile() is PROFILE
    assert Qwen35.get_profile("candidate30").name == "qwen35-candidate30"

    # ModelDef HF architectures and model_type registrations
    assert Qwen35.hf_architectures == (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    )
    assert Qwen35.hf_model_type == ("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text")
    assert Qwen35.supported_tp_sizes == (8,)
    assert Qwen35.display_name == "Qwen3.5"

    # Dense Qwen3.5 config fails closed with "Qwen35 supports MoE checkpoints only"
    dense_cfg = {k: v for k, v in DEFAULT_HF_CONFIG.items() if k != "num_experts"}
    with pytest.raises(
        MegakernelConfigError, match="Qwen35 supports MoE checkpoints only"
    ):
        Qwen35.derive_geometry(dense_cfg)

    # Unregistered subclass of Qwen35 with missing final_norm names the subclass
    class _UnregisteredMissingFinalNorm(Qwen35):
        final_norm = None

    for fn_call in (
        lambda: _UnregisteredMissingFinalNorm.derive_geometry(DEFAULT_HF_CONFIG),
        lambda: _UnregisteredMissingFinalNorm.to_model_spec(qwen35.TINY),
        lambda: _UnregisteredMissingFinalNorm.reference_step(
            qwen35.TINY, ExecSpec(R=2, B=2, T=1, ctx=128), None, None, {}, {}
        ),
    ):
        with pytest.raises(
            MegakernelConfigError,
            match=r"_UnregisteredMissingFinalNorm.*final_norm is None",
        ) as fn_exc:
            fn_call()
        assert "layer " not in str(fn_exc.value)

    # Runner hf_config validation: consistent fail-closed policy (log differing fields
    # and return False / maybe_create returns None, never raising on mismatched configs)
    def dict_to_namespace(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return types.SimpleNamespace(
                **{str(k): dict_to_namespace(v) for k, v in obj.items()}
            )
        if isinstance(obj, list):
            return [dict_to_namespace(x) for x in obj]
        return obj

    def _make_runner(hf_cfg: Any) -> Any:
        mc = types.SimpleNamespace(hf_config=hf_cfg)
        vc = types.SimpleNamespace(
            model_config=mc,
            parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
            cache_config=types.SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
            additional_config={"megakernel": {"enabled": True}},
        )
        return types.SimpleNamespace(
            model_config=mc, parallel_config=vc.parallel_config, vllm_config=vc
        )

    # 1) Realistic nested VLM config with nested rope_parameters succeeds (both dict
    #    and PretrainedConfig-like attribute-style SimpleNamespace)
    nested_rope_text = {
        k: v
        for k, v in DEFAULT_HF_CONFIG.items()
        if k not in ("rope_theta", "partial_rotary_factor")
    }
    nested_rope_text["rope_parameters"] = {
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
    }
    nested_vlm = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": nested_rope_text,
    }
    for valid_cfg in (
        nested_vlm,
        dict_to_namespace(nested_vlm),
    ):
        assert Qwen35.validate_hf_config(valid_cfg).eligible
        assert qwen_adapter.validate_hf_config(valid_cfg).eligible

    # 2) Different layer count (48 vs 60) logs differing fields (L, NGDN, NATT) and
    #    maybe_create returns None without raising (both dict and attribute-style)
    cfg_48l = {
        **nested_vlm,
        "text_config": {
            **nested_rope_text,
            "num_hidden_layers": 48,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
        },
    }
    for cand_48l in (cfg_48l, dict_to_namespace(cfg_48l)):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(cand_48l).eligible
            assert not qwen_adapter.validate_hf_config(cand_48l).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(cand_48l)))
        assert "L=48 (expected 60)" in caplog.text

    # 3) Missing required key logs only the missing key (never embeds the config dict)
    cfg_missing_theta = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            k: v
            for k, v in DEFAULT_HF_CONFIG.items()
            if k not in ("rope_theta", "partial_rotary_factor")
        },
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _declined(driver_mod.maybe_create(_make_runner(cfg_missing_theta)))
    assert "missing HF key 'rope_theta'" in caplog.text
    assert "num_attention_heads" not in caplog.text

    # 4) Different hidden_size (2048 vs 4096) logs H mismatch and returns None
    cfg_2048h = {
        **nested_vlm,
        "text_config": {**nested_rope_text, "hidden_size": 2048},
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _declined(driver_mod.maybe_create(_make_runner(cfg_2048h)))
    assert "H=2048 (expected 4096)" in caplog.text

    # 5) Dense config WITH geometry (has hidden_size, no num_experts) and WITHOUT
    #    geometry (minimal stub without hidden_size) both log "supports MoE checkpoints
    #    only" and return None without raising (both dict and attribute-style)
    dense_with_geom = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": dense_cfg,
    }
    dense_without_geom = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"model_type": "qwen3_5_text"},
    }
    for dense_cand in (
        dense_with_geom,
        dict_to_namespace(dense_with_geom),
        dense_without_geom,
        dict_to_namespace(dense_without_geom),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(dense_cand).eligible
            assert not qwen_adapter.validate_hf_config(dense_cand).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(dense_cand)))
        assert "Qwen35 supports MoE checkpoints only" in caplog.text

    # 6) Scaled RoPE (YaRN in rope_parameters or linear in rope_scaling) is rejected
    yarn_cfg = {
        **nested_vlm,
        "text_config": {
            **nested_rope_text,
            "rope_parameters": {
                **nested_rope_text["rope_parameters"],
                "rope_type": "yarn",
                "factor": 4.0,
            },
        },
    }
    linear_cfg = {
        **DEFAULT_HF_CONFIG,
        "rope_scaling": {"rope_type": "linear", "factor": 2.0},
    }
    for scaled_cfg in (
        yarn_cfg,
        dict_to_namespace(yarn_cfg),
        linear_cfg,
        dict_to_namespace(linear_cfg),
    ):
        with pytest.raises(MegakernelConfigError, match="unsupported .* rope_type"):
            Qwen35.derive_geometry(scaled_cfg)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(scaled_cfg).eligible
            assert not qwen_adapter.validate_hf_config(scaled_cfg).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(scaled_cfg)))
        assert "unsupported" in caplog.text and "rope_type" in caplog.text


def test_validate_runner_hf_config_frozen_qwen35_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Frozen Qwen3.5 `config.json` passes `validate_hf_config` (skippable)."""
    import json

    import frozen_utils

    from vllm_torchtpu.models.qwen3_5.megakernel import adapter as qwen_adapter

    frozen_cfg = frozen_utils.load_frozen_qwen35_config_json()
    for valid_cfg in (frozen_cfg, frozen_utils.dict_to_namespace(frozen_cfg)):
        assert Qwen35.validate_hf_config(valid_cfg).eligible
        assert qwen_adapter.validate_hf_config(valid_cfg).eligible

    # Verify SHA-256 mismatch on first candidate falls back to DEFAULT_FROZEN_DIR
    bad_dir = tmp_path / "bad_frozen"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{}", encoding="utf-8")
    good_dir = tmp_path / "good_frozen"
    good_dir.mkdir()
    orig_dir = Path(
        frozen_utils.os.environ.get(
            "MK_FROZEN_QWEN35_DIR", frozen_utils.DEFAULT_FROZEN_DIR
        )
    )
    (good_dir / "config.json").write_bytes((orig_dir / "config.json").read_bytes())
    assert (
        json.loads((good_dir / "config.json").read_text(encoding="utf-8")) == frozen_cfg
    )
    monkeypatch.setattr(frozen_utils, "DEFAULT_FROZEN_DIR", str(good_dir))
    assert frozen_utils.load_frozen_qwen35_config_json(bad_dir) == frozen_cfg
    monkeypatch.setattr(frozen_utils, "DEFAULT_FROZEN_DIR", str(bad_dir))
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        frozen_utils.load_frozen_qwen35_config_json(bad_dir)


def test_qwen35_model_def_derived_geometry_spec_and_pool_layout_equivalence():
    """Derived `Qwen35` geometry, `ModelSpec`, and `PoolLayout` match ground truth."""
    nested_vlm_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": dict(DEFAULT_HF_CONFIG),
    }
    expected_full = qwen35.Qwen35Geometry()
    for cfg in (DEFAULT_HF_CONFIG, nested_vlm_cfg):
        g_derived = Qwen35.derive_geometry(cfg)
        assert type(g_derived) is qwen35.Qwen35Geometry
        assert g_derived == expected_full
        for f in dataclasses.fields(qwen35.Qwen35Geometry):
            assert getattr(g_derived, f.name) == getattr(expected_full, f.name), f.name

    tiny_hf_cfg = {
        **DEFAULT_HF_CONFIG,
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 16,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    expected_tiny = qwen35.Qwen35Geometry(
        L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024
    )
    g_tiny_from_hf = Qwen35.derive_geometry(tiny_hf_cfg)
    assert type(g_tiny_from_hf) is qwen35.Qwen35Geometry
    assert g_tiny_from_hf == expected_tiny
    assert expected_tiny == qwen35.TINY
    for f in dataclasses.fields(qwen35.Qwen35Geometry):
        assert getattr(g_tiny_from_hf, f.name) == getattr(expected_tiny, f.name), f.name
        assert getattr(qwen35.TINY, f.name) == getattr(expected_tiny, f.name), f.name

    # shrink() is R-aware across R in {2, 4, 8}, including EPLB e_slots=80
    g_ep_eplb = dataclasses.replace(qwen35.FULL_EP, e_slots=80)
    for r_val in (2, 4, 8):
        for base_g in (qwen35.FULL, qwen35.FULL_EP, g_ep_eplb):
            g_shrunk = base_g.shrink(R=r_val)
            assert type(g_shrunk) is qwen35.Qwen35Geometry
            assert (g_shrunk.L, g_shrunk.att_every) == (8, 4)
            g_shrunk.check(ExecSpec(R=r_val, B=r_val, T=1, ctx=256))

    # Independent PoolLayout + ModelSpec equivalence across FULL, TINY, FULL_EP,
    # EPLB e_slots=72, and att_every=1
    att_only_tiny = dataclasses.replace(
        qwen35.TINY, att_every=1, KH=0, VH=0, KD=0, VD=0
    )
    att_only_hf_cfg = {**tiny_hf_cfg, "layer_types": ["full_attention"] * 4}
    eplb72 = dataclasses.replace(qwen35.FULL_EP, e_slots=72)
    spec_cases = [
        (qwen35.FULL, DEFAULT_HF_CONFIG, True, 0),
        (qwen35.TINY, tiny_hf_cfg, True, 0),
        (qwen35.FULL_EP, DEFAULT_HF_CONFIG, False, 0),
        (eplb72, DEFAULT_HF_CONFIG, False, 72),
        (att_only_tiny, att_only_hf_cfg, True, 0),
    ]
    for g, cfg, moe_tp_val, e_slots_val in spec_cases:
        expected_layout = _independent_qwen35_pool_layout(g, R=8)
        assert Qwen35.pool_layout(cfg, R=8) == expected_layout
        assert Qwen35.pool_layout(g, R=8) == expected_layout
        assert qwen35_pool.qwen35_pool_layout(g, R=8) == expected_layout

        expected_spec = _independent_qwen35_spec(g, R=8)
        spec_from_cfg = Qwen35.to_model_spec(
            cfg, moe_tp=moe_tp_val, e_slots=e_slots_val
        )
        spec_from_g = qwen35.qwen35_spec(g)
        assert_specs_equal(spec_from_cfg, expected_spec)
        assert_specs_equal(spec_from_g, expected_spec)

    g_att1 = dataclasses.replace(qwen35.TINY, att_every=1)
    assert qwen35_pool.qwen35_pool_layout(
        g_att1, R=8
    ) == _independent_qwen35_pool_layout(g_att1, R=8)
    assert_specs_equal(
        qwen35.qwen35_spec(g_att1), _independent_qwen35_spec(g_att1, R=8)
    )


@pytest.mark.parametrize(
    "t_val,moe_tp,pool_io",
    [
        (1, True, False),
        pytest.param(4, True, False, marks=pytest.mark.slow),
        (1, False, True),
    ],
)
def test_qwen35_model_def_composed_reference_matches_module_loop(
    t_val: int, moe_tp: bool, pool_io: bool
):
    """Composed `Qwen35.reference_step` is bitwise identical to explicit module loop."""
    g = dataclasses.replace(qwen35.TINY, moe_tp=moe_tp)
    ex = ExecSpec(R=2, B=2 * t_val, T=t_val, ctx=128)
    pos, hid, w_l, st_l = mk.random_logical_inputs(
        jax.random.PRNGKey(17 + t_val), g, ex
    )
    out_composed, st_composed = Qwen35.reference_step(
        g, ex, pos, hid, w_l, st_l, pool_io=pool_io
    )

    # Verify against an independent explicit layer loop (both out and final states)
    in_norm_mod = Norm("input_layernorm", geometry=g, zero_centered=True, eps=g.eps)
    post_norm_mod = Norm(
        "post_attention_layernorm", geometry=g, zero_centered=True, eps=g.eps
    )
    final_norm_mod = Norm("model.norm", geometry=g, zero_centered=True, eps=g.eps)
    gqa_mod = GQAModule(geometry=g)
    gdn_mod = GDNModule(geometry=g)
    moe_mod = MoEModule(geometry=g)
    lay_gdn = GDNModule.lin_in_layout(g, ex)
    cr, hr = mk.rup(3 * ex.B, 8), mk.rup(3 + ex.T, 8)
    conv_in = st_l["conv"].astype(BF16).astype(F32) if pool_io else st_l["conv"]
    rec_in = st_l["rec"].reshape(
        ex.R, g.NGDN, ex.NS, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    )
    step_state: dict[str, Any] = {
        "kc": st_l["kc"],
        "vc": st_l["vc"],
        "conv": conv_in,
        "rec": rec_in,
        "conv_out": (
            jnp.zeros((ex.R, g.NGDN, cr, lay_gdn.QKV), dtype=F32)
            if ex.T == 1
            else jnp.zeros((ex.R, g.NGDN, ex.NS, hr, lay_gdn.QKV), dtype=F32)
        ),
        "rec_out": jnp.zeros(
            (ex.R, g.NGDN, ex.B * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD), dtype=BF16
        ),
    }
    res = hid[: ex.B].astype(F32)
    for lyr in range(g.L):
        x_m, _ = in_norm_mod.reference(w_l["in_norm"][lyr], res, g=g)
        if (lyr + 1) % g.att_every == 0:
            d_m, step_state = gqa_mod.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                pos=pos,
                att_idx=lyr // g.att_every,
                layer_idx=lyr,
                paged_kv=False,
            )
        else:
            d_m, step_state = gdn_mod.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                gdn_idx=lyr - (lyr + 1) // g.att_every,
                layer_idx=lyr,
                multi_layer=True,
            )
        res = rnd(res + d_m)
        x_f, _ = post_norm_mod.reference(w_l["post_norm"][lyr], res, g=g)
        d_f, step_state = moe_mod.reference(
            w_l, x_f, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = rnd(res + d_f)
    out_hand, _ = final_norm_mod.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)
    conv_out = (
        step_state["conv_out"].astype(BF16).astype(F32)
        if pool_io
        else step_state["conv_out"]
    )
    rec_out = step_state["rec_out"]
    st_hand: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ex.VER:
        conv_pad = (
            jnp.pad(
                conv_in,
                ((0, 0), (0, 0), (0, 0), (0, hr - conv_in.shape[-2]), (0, 0)),
            )
            if conv_in.ndim == 5 and conv_in.shape[-2] < hr
            else conv_in
        )
        st_hand["conv"] = conv_pad
        st_hand["rec"] = rec_in.reshape(
            ex.R, g.NGDN, ex.NS * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
        )
        st_hand["conv_ckpt"] = conv_out
        st_hand["rec_ckpt"] = rec_out
    else:
        st_hand["conv"] = conv_out
        st_hand["rec"] = rec_out

    assert jnp.array_equal(out_composed, out_hand)
    assert set(st_composed) == set(st_hand)
    for k in st_hand:
        assert jnp.array_equal(st_composed[k], st_hand[k]), k
