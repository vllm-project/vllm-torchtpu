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
"""Unit tests for `schedule.py` and `geometry.py`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import geometry, pool, schedule
from vllm_torchtpu.megakernel.decls import KVCacheDecl
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.geometry import FromHF, ModelGeometry
from vllm_torchtpu.megakernel.module import Module
from vllm_torchtpu.megakernel.modules.gdn import GDN, GDNModule
from vllm_torchtpu.megakernel.modules.norm import Norm
from vllm_torchtpu.megakernel.schedule import Every, FromHFLayerTypes, Schedule


class _StubAttnModule(Module):
    name = "gqa"
    cache_role = "kv"
    OPTIONS = {"dummy": None}


class _StubFFNModule(Module):
    name = "moe"
    OPTIONS = {"dummy": None}


def test_schedule_and_geometry_files_fit_line_budget_and_have_no_qwen_literals():
    """`schedule.py` and `geometry.py` have <= 400 lines and no Qwen literals."""
    for mod in (schedule, geometry):
        path = Path(mod.__file__)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 400, f"{path.name} has {len(lines)} lines (> 400)"
        assert "fmt: skip" not in text, f"{path.name} contains fmt: skip"
        assert "noqa: E501" not in text, f"{path.name} contains noqa: E501"
        assert "qwen" not in text.lower(), f"{path.name} must not mention Qwen"


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


def test_model_geometry_properties_shrink_check_and_pool_backend():
    """`ModelGeometry` properties, `shrink()`, `check()`, and `make_pool_backend`."""
    mg = ModelGeometry(
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
    assert (mg.NATT, mg.NGDN, mg.QKV(8)) == (2, 6, 512)
    assert mg.ER(8) == 16 and mg.EW(8) == 128
    mg_shrunk = mg.shrink(R=2)
    assert (mg_shrunk.L, mg_shrunk.att_every) == (8, 4)
    mg_shrunk.check(ExecSpec(R=2, B=2, T=1, ctx=128))

    rec_decl = GDNModule(geometry=mg).cache_decl(mg)
    assert rec_decl is not None
    kv_decl = KVCacheDecl(hd=mg.HD, kvh=1)
    play = pool.derive_pool_layout(([rec_decl] * 3 + [kv_decl]) * 2, R=8)
    pb1 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    pb2 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    assert pb1 != pb2
    assert pb1.layout == pb2.layout

    with pytest.raises(MegakernelConfigError, match="divisible by att_every"):
        dataclasses.replace(mg, L=5).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid B=3, T=2"):
        mg.check(ExecSpec(R=2, B=3, T=2, ctx=128))


def test_schedule_every_and_from_hf_layer_types_resolution_and_errors():
    """`Every`, `FromHFLayerTypes`, and `Schedule.resolve` validate mixer order."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = _StubFFNModule.bind("mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)

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
