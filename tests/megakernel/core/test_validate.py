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
"""Unit tests for `validate.py` (`validate_norms`, `validate_model_def`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllm_torchtpu.megakernel.core import validate
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.geometry import FromHF, ModelGeometry
from vllm_torchtpu.megakernel.core.module import Module
from vllm_torchtpu.megakernel.core.modules.gdn import GDN
from vllm_torchtpu.megakernel.core.modules.gqa import GQA
from vllm_torchtpu.megakernel.core.modules.moe import MoE, MoEModule
from vllm_torchtpu.megakernel.core.modules.norm import Norm
from vllm_torchtpu.megakernel.core.profile import Profile
from vllm_torchtpu.megakernel.core.schedule import Every, FromHFLayerTypes, Schedule
from vllm_torchtpu.megakernel.core.validate import (
    describe_binding,
    model_label,
    norm_zc_and_eps,
    validate_model_def,
    validate_model_hf_config,
    validate_norms,
)


class _StubAttnModule(Module):
    name = "gqa"
    cache_role = "kv"
    OPTIONS = {"dummy": None}


def test_validate_file_fits_line_budget_and_has_no_qwen_literals():
    """`validate.py` has <= 400 lines and no Qwen literals."""
    path = Path(validate.__file__)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) <= 400, f"{path.name} has {len(lines)} lines (> 400)"
    assert "fmt: skip" not in text, f"{path.name} contains fmt: skip"
    assert "noqa: E501" not in text, f"{path.name} contains noqa: E501"
    assert "qwen" not in text.lower(), f"{path.name} must not mention Qwen"


def test_describe_binding_and_norm_zc_and_eps():
    """`describe_binding`, `model_label`, and `norm_zc_and_eps` validate norms."""
    assert describe_binding(None) == "None"
    moe_a = MoE.bind("mlp", shared_gate=True)
    moe_b = MoE.bind("mlp", shared_gate=True, e_slots=32)
    assert "e_slots=32 (expected None)" in describe_binding(moe_b, expected=moe_a)

    norm_b = Norm.bind("input_layernorm", zero_centered=True, eps=1e-5)
    assert norm_zc_and_eps(norm_b, None, default_eps=1e-6) == (True, 1e-5)
    norm_inst = Norm("input_layernorm", zero_centered=False, eps=2e-5)
    assert norm_zc_and_eps(norm_inst, None, default_eps=1e-6) == (False, 2e-5)
    norm_bool_eps = Norm.bind("input_layernorm", zero_centered=True, eps=True)
    assert norm_zc_and_eps(norm_bool_eps, None, default_eps=1e-6) == (True, 1e-6)
    with pytest.raises(MegakernelConfigError, match="Expected NormModule binding"):
        norm_zc_and_eps(moe_a, None)


def test_validate_norms_and_validate_model_def():
    """`validate_norms` and `validate_model_def` reject bad norms and profiles."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)

    class _BaseStub:
        default_hf_config: Any = None
        geometry = FromHF(
            hidden="hidden_size",
            layers=("num_hidden_layers", 2),
            eps="rms_norm_eps",
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            pre_norm=in_norm,
            post_norm=post_norm,
        )
        final_norm: Any = final_norm_b
        pool_layout_override: Any = None
        profiles: Any = None

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

    assert model_label(_BaseStub, "custom_name") == "custom_name"
    validate_model_def(
        _BaseStub, "valid_stub", ("StubForCausalLM",), ("stub",), _BaseStub
    )

    class _BadZCStub(_BaseStub):
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=False),
        )

    with pytest.raises(MegakernelConfigError, match="zero_centered=False"):
        validate_model_def(_BadZCStub, "bad_zc", (), (), _BaseStub)

    class _BadFinalNormStub(_BaseStub):
        final_norm = None

    with pytest.raises(MegakernelConfigError, match="final_norm is None"):
        validate_model_def(_BadFinalNormStub, "bad_fn", (), (), _BaseStub)

    sched_res = _BaseStub.schedule.resolve({}, 2)
    with pytest.raises(MegakernelConfigError, match="disagreeing with model eps=1e-06"):

        class _BadEpsStub(_BaseStub):
            final_norm = Norm.bind("model.norm", zero_centered=True, eps=1e-5)

        validate_norms(_BadEpsStub, sched_res, None, eps=1e-6)

    class _BadProfStub(_BaseStub):
        profiles = {"default": Profile(name="p", model="other_model")}

    with pytest.raises(MegakernelConfigError, match="targets model 'other_model'"):
        validate_model_def(_BadProfStub, "this_model", (), (), _BaseStub)


def test_validate_model_def_hf_layer_types_without_default_cfg():
    """`FromHFLayerTypes` in any slot without `default_hf_config` does not crash."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)

    class _BaseStub:
        default_hf_config: Any = None
        geometry = FromHF(
            hidden="hidden_size",
            layers=("num_hidden_layers", 2),
            eps="rms_norm_eps",
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            pre_norm=in_norm,
            post_norm=post_norm,
        )
        final_norm: Any = final_norm_b
        pool_layout_override: Any = None
        profiles: Any = None

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

    sched_res = _BaseStub.schedule.resolve({}, 2)
    with pytest.raises(
        MegakernelConfigError, match="disagreeing with model zero_centered=False"
    ):
        validate_norms(_BaseStub, sched_res, None, expected_zc=False)

    with pytest.raises(MegakernelConfigError, match="invalid hf_architectures entry"):
        validate_model_def(_BaseStub, "ws_arch", (" StubArch ",), (), _BaseStub)
    with pytest.raises(MegakernelConfigError, match="invalid hf_model_type entry"):
        validate_model_def(_BaseStub, "ws_mt", (), ("stub ",), _BaseStub)

    hf_lt_ffn = FromHFLayerTypes("ffn_types", {"moe": moe})
    hf_lt_norm = FromHFLayerTypes("norm_types", {"rms": in_norm})
    for slot_kw in (
        {"ffn": hf_lt_ffn},
        {"pre_norm": hf_lt_norm},
        {"post_norm": hf_lt_norm},
    ):

        class _HFLayerTypesNoDefStub(_BaseStub):
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=slot_kw.get("ffn", moe),
                pre_norm=slot_kw.get("pre_norm", in_norm),
                post_norm=slot_kw.get("post_norm", post_norm),
            )

        validate_model_def(_HFLayerTypesNoDefStub, "hf_lt_no_def", (), (), _BaseStub)

    class _HFMixerBadNormNoDefStub(_BaseStub):
        schedule = Schedule(
            mixer=FromHFLayerTypes("layer_types", {"att": attn, "lin": gdn}),
            ffn=moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=False),
        )

    with pytest.raises(MegakernelConfigError, match="zero_centered=False"):
        validate_model_def(_HFMixerBadNormNoDefStub, "hf_mix_bad_zc", (), (), _BaseStub)

    class _BadDefCfgStub(_BaseStub):
        default_hf_config = {"hidden_size": 1024}

        @classmethod
        def derive_geometry(cls, hf_cfg: Any) -> Any:
            raise ValueError("broken hf_config")

    with pytest.raises(
        MegakernelConfigError,
        match=r"register_model\('bad_def'\): invalid default_hf_config: broken",
    ):
        validate_model_def(_BadDefCfgStub, "bad_def", (), (), _BaseStub)


def test_validate_model_hf_config_eligibility_checks():
    """`validate_model_hf_config` returns `EligibilityResult` for MoE and geometry."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp", experts="num_experts")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)

    base_cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-6,
        "num_experts": 128,
    }

    class _StubModel:
        default_hf_config = base_cfg
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
        final_norm = Norm.bind("model.norm", zero_centered=True)

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

        @classmethod
        def derive_geometry(cls, hf_cfg: Any) -> ModelGeometry:
            res = cls.geometry.resolve(hf_cfg)
            return ModelGeometry(
                L=int(res["layers"]),
                att_every=2,
                H=int(res["hidden"]),
                NH=8,
                NKV=2,
                HD=128,
                ROT=64,
                KH=8,
                VH=8,
                KD=128,
                VD=128,
                E=128,
                TOPK=2,
                I=512,
                IS=512,
            )

    assert validate_model_hf_config(_StubModel, None).eligible
    assert validate_model_hf_config(_StubModel, base_cfg).eligible

    dense_res = validate_model_hf_config(
        _StubModel, {k: v for k, v in base_cfg.items() if k != "num_experts"}
    )
    assert not dense_res.eligible and dense_res.code == "non_moe_checkpoint"

    mismatch_res = validate_model_hf_config(
        _StubModel, {**base_cfg, "hidden_size": 2048}
    )
    assert not mismatch_res.eligible and mismatch_res.code == "geometry_mismatch"
    assert "H=2048 (expected 1024)" in (mismatch_res.message or "")

    mg = _StubModel.derive_geometry(base_cfg)

    class _InstantiatedMoEWithGeomModel(_StubModel):
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=MoEModule(geometry=mg),
            pre_norm=in_norm,
            post_norm=post_norm,
        )

    assert validate_model_hf_config(_InstantiatedMoEWithGeomModel, base_cfg).eligible

    class _InstantiatedMoENoGeomModel(_StubModel):
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=MoEModule(),
            pre_norm=in_norm,
            post_norm=post_norm,
        )

    assert validate_model_hf_config(_InstantiatedMoENoGeomModel, base_cfg).eligible
    no_exp_res = validate_model_hf_config(
        _InstantiatedMoENoGeomModel,
        {k: v for k, v in base_cfg.items() if k != "num_experts"},
    )
    assert not no_exp_res.eligible and no_exp_res.code == "non_moe_checkpoint"

    class _NestedDefModel(_StubModel):
        default_hf_config = {
            "model_type": "stub",
            "num_attention_heads": 8,
            "text_config": {**base_cfg, "num_attention_heads": 8},
        }

        @classmethod
        def derive_geometry(cls, hf_cfg: Any) -> ModelGeometry:
            merged = dict(hf_cfg)
            if isinstance(merged.get("text_config"), dict):
                merged.update(merged["text_config"])
            res = cls.geometry.resolve(merged)
            return ModelGeometry(
                L=int(res["layers"]),
                att_every=2,
                H=int(res["hidden"]),
                NH=int(merged.get("num_attention_heads", 8)),
                NKV=2,
                HD=128,
                ROT=64,
                KH=8,
                VH=8,
                KD=128,
                VD=128,
                E=128,
                TOPK=2,
                I=512,
                IS=512,
            )

    partial_mismatch = validate_model_hf_config(
        _NestedDefModel, {"num_experts": 128, "num_attention_heads": 16}
    )
    assert (
        not partial_mismatch.eligible and partial_mismatch.code == "geometry_mismatch"
    )

    bad_type_res = validate_model_hf_config(
        _StubModel, {**base_cfg, "hidden_size": [1024]}
    )
    assert not bad_type_res.eligible and bad_type_res.code == "invalid_geometry"


def test_validate_multi_key_alias_field_specs_and_default_cfg_norms():
    """`validate` resolves `FIELD_ALIASES` and `(key, default)` tuples and calls
    `validate_norms` directly when `default_hf_config` is set."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    in_norm = Norm.bind(
        "input_layernorm", zero_centered=True, eps=("layer_norm_epsilon", 1e-5)
    )
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)

    # 1. `norm_zc_and_eps` with `(hf_key, default)` tuple spec + `"H"` alias.
    assert norm_zc_and_eps(in_norm, {"H": 1024}) == (True, 1e-5)
    assert norm_zc_and_eps(in_norm, {"H": 1024, "layer_norm_epsilon": 2e-5}) == (
        True,
        2e-5,
    )

    # 2. Unpinned `MoE.bind("mlp")` resolves `FIELD_ALIASES` (`"E"`, `"experts"`).
    class _AliasMoEModel:
        default_hf_config = {"hidden_size": 1024, "num_hidden_layers": 4, "E": 128}
        geometry = FromHF(
            hidden=("n_embd", 1024),
            layers=("n_layer", 4),
            eps=("layer_norm_epsilon", 1e-5),
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=MoE.bind("mlp"),
            pre_norm=in_norm,
            post_norm=post_norm,
        )
        final_norm = final_norm_b

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

        @classmethod
        def derive_geometry(cls, hf_cfg: Any) -> ModelGeometry:
            res = cls.geometry.resolve(hf_cfg)
            return ModelGeometry(
                L=int(res["layers"]),
                att_every=2,
                H=int(res["hidden"]),
                NH=8,
                NKV=2,
                HD=128,
                ROT=64,
                KH=8,
                VH=8,
                KD=128,
                VD=128,
                E=128,
                TOPK=2,
                I=512,
                IS=512,
            )

    assert validate_model_hf_config(_AliasMoEModel, {"E": 128}).eligible
    assert validate_model_hf_config(_AliasMoEModel, {"experts": 128}).eligible
    zero_alias = validate_model_hf_config(_AliasMoEModel, {"E": 0})
    assert not zero_alias.eligible and zero_alias.code == "non_moe_checkpoint"
    false_alias = validate_model_hf_config(_AliasMoEModel, {"E": False})
    assert not false_alias.eligible and false_alias.code == "non_moe_checkpoint"

    # 3. Tuple `(hf_key, default)` specs on `FromHF` and `MoE.bind(..., experts=...)`.
    class _TupleSpecModel(_AliasMoEModel):
        default_hf_config = {}
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=MoE.bind("mlp", experts=("n_routed_experts", 128)),
            pre_norm=in_norm,
            post_norm=post_norm,
        )

    assert validate_model_hf_config(_TupleSpecModel, {}).eligible
    tup_dense = validate_model_hf_config(_TupleSpecModel, {"n_routed_experts": 0})
    assert not tup_dense.eligible and tup_dense.code == "non_moe_checkpoint"
    tup_false = validate_model_hf_config(_TupleSpecModel, {"n_routed_experts": False})
    assert not tup_false.eligible and tup_false.code == "non_moe_checkpoint"
    tup_mismatch = validate_model_hf_config(_TupleSpecModel, {"n_embd": 2048})
    assert not tup_mismatch.eligible and tup_mismatch.code == "geometry_mismatch"

    # 4. Direct `validate_norms` check in `validate_model_def` when `default_hf_config`
    # is present (runs before `derive_geometry`).
    class _BadNormWithDefCfg:
        default_hf_config = {"n_embd": 1024, "n_layer": 2}
        geometry = _AliasMoEModel.geometry
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=MoE.bind("mlp", experts=("n_routed_experts", 128)),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=False),
        )
        final_norm = final_norm_b
        pool_layout_override = None
        profiles = None

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

        @classmethod
        def derive_geometry(cls, hf_cfg: Any) -> Any:
            raise AssertionError("validate_norms must fail before derive_geometry")

    with pytest.raises(MegakernelConfigError, match="zero_centered=False"):
        validate_model_def(_BadNormWithDefCfg, "bad_def_zc", (), (), _BadNormWithDefCfg)


def test_validate_partial_hf_config_merges_nested_rope_dicts():
    """Partial `hf_config` merges nested `rope_parameters` and `rope_scaling`
    mappings (both top-level and inside `text_config`) without dropping sibling keys."""
    gqa = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    final_norm_b = Norm.bind("model.norm", zero_centered=True)
    base_fields = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "num_experts": 128,
    }

    def _derive_from_gqa(cls: Any, hf_cfg: Any) -> ModelGeometry:
        merged = dict(hf_cfg)
        if isinstance(merged.get("text_config"), dict):
            merged.update(merged["text_config"])
        for rk in ("rope_parameters", "rope_scaling"):
            sub = merged.get(rk)
            if isinstance(sub, dict) and sub.get("rope_type") not in (None, "default"):
                raise MegakernelConfigError(f"unsupported {rk}")
        res = cls.geometry.resolve(merged)
        gqa_g = gqa.resolve(merged)
        return ModelGeometry(
            L=int(res["layers"]),
            att_every=2,
            H=int(res["hidden"]),
            NH=int(gqa_g.NH),
            NKV=int(gqa_g.NKV),
            HD=int(gqa_g.HD),
            ROT=int(gqa_g.ROT),
            KH=8,
            VH=8,
            KD=128,
            VD=128,
            E=128,
            TOPK=2,
            I=512,
            IS=512,
            theta=float(gqa_g.theta),
        )

    class _TopRopeModel:
        default_hf_config = {
            **base_fields,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        }
        geometry = FromHF(
            hidden="hidden_size", layers="num_hidden_layers", eps="rms_norm_eps"
        )
        schedule = Schedule(
            mixer=Every(n=2, last=gqa, rest=gdn),
            ffn=MoE.bind("mlp"),
            pre_norm=in_norm,
            post_norm=post_norm,
        )
        final_norm = final_norm_b

        @classmethod
        def _from_hf_decl(cls) -> FromHF:
            return cls.geometry

        derive_geometry = classmethod(_derive_from_gqa)

    class _TextCfgRopeModel(_TopRopeModel):
        default_hf_config = {
            "model_type": "stub",
            "text_config": {
                **base_fields,
                "rope_parameters": {
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
                "rope_scaling": {"rope_type": "default"},
            },
        }

    for model_cls in (_TopRopeModel, _TextCfgRopeModel):
        partial_ok = validate_model_hf_config(
            model_cls,
            {"num_experts": 128, "rope_parameters": {"rope_theta": 1_000_000.0}},
        )
        assert partial_ok.eligible, partial_ok.message

        nested_mismatch = validate_model_hf_config(
            model_cls,
            {"num_experts": 128, "rope_parameters": {"rope_theta": 500_000.0}},
        )
        assert (
            not nested_mismatch.eligible
            and nested_mismatch.code == "geometry_mismatch"
            and "theta=500000.0" in (nested_mismatch.message or "")
        )

        flat_mismatch = validate_model_hf_config(
            model_cls, {"num_experts": 128, "partial_rotary_factor": 0.5}
        )
        assert (
            not flat_mismatch.eligible
            and flat_mismatch.code == "geometry_mismatch"
            and "ROT=64 (expected 32)" in (flat_mismatch.message or "")
        )

        scaled_bad = validate_model_hf_config(
            model_cls, {"num_experts": 128, "rope_scaling": {"rope_type": "yarn"}}
        )
        assert not scaled_bad.eligible and scaled_bad.code == "invalid_geometry"
