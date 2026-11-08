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

from vllm_torchtpu.megakernel import validate
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.geometry import FromHF, ModelGeometry
from vllm_torchtpu.megakernel.module import Module
from vllm_torchtpu.megakernel.modules.gdn import GDN
from vllm_torchtpu.megakernel.modules.moe import MoE
from vllm_torchtpu.megakernel.modules.norm import Norm
from vllm_torchtpu.megakernel.profile import Profile
from vllm_torchtpu.megakernel.schedule import Every, Schedule
from vllm_torchtpu.megakernel.validate import (
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
