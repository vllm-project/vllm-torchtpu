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
from typing import Any

import pytest

from vllm_torchtpu.megakernel.core import geometry, pool, schedule
from vllm_torchtpu.megakernel.core.decls import KVCacheDecl, RecurrentCacheDecl
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import FromHF, ModelGeometry
from vllm_torchtpu.megakernel.core.module import Module
from vllm_torchtpu.megakernel.core.modules.gdn import GDN, GDNModule
from vllm_torchtpu.megakernel.core.modules.norm import Norm
from vllm_torchtpu.megakernel.core.pool.config import CacheExec
from vllm_torchtpu.megakernel.core.schedule import Every, FromHFLayerTypes, Schedule


class _StubAttnModule(Module):
    name = "gqa"
    cache_role = "kv"
    OPTIONS = {"dummy": None}

    def __init__(
        self,
        *,
        geometry: Any = None,
        prefix: str = "self_attn",
        dummy: Any = None,
    ) -> None:
        self.geometry = geometry
        self.prefix = prefix
        self.dummy = dummy

    def cache_decl(self, g: Any = None) -> KVCacheDecl:
        geom = self._require_geometry(g)
        return KVCacheDecl(hd=int(geom.HD), kvh=int(geom.NKV))


class _NoCacheAttnModule(Module):
    name = "gqa"
    OPTIONS = {"dummy": None}

    def __init__(
        self,
        *,
        geometry: Any = None,
        prefix: str = "self_attn",
        dummy: Any = None,
    ) -> None:
        self.geometry = geometry
        self.prefix = prefix
        self.dummy = dummy


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
        "model_type": "wrapper",
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
    mg_copy = dataclasses.replace(mg)
    assert mg == mg_copy and hash(mg) == hash(mg_copy)
    for f in dataclasses.fields(ModelGeometry):
        cur = getattr(mg, f.name)
        if isinstance(cur, bool):
            alt: Any = not cur
        elif isinstance(cur, int):
            alt = cur + 1
        elif isinstance(cur, float):
            alt = cur * 2.0
        else:
            alt = "sigmoid" if cur == "silu" else "silu"
        mg_diff = dataclasses.replace(mg, **{f.name: alt})
        assert mg_diff != mg, f"ModelGeometry equality ignored field {f.name!r}"
        assert hash(mg_diff) != hash(mg), f"ModelGeometry hash ignored field {f.name!r}"

    mg_large = ModelGeometry(
        L=32,
        att_every=4,
        H=4096,
        NH=32,
        NKV=4,
        HD=256,
        ROT=64,
        KH=16,
        VH=64,
        KD=128,
        VD=128,
        E=512,
        TOPK=8,
        I=2048,
        IS=2048,
    )
    object.__setattr__(mg_large, "_extra_tag", "preserved")
    mg_shrunk = mg_large.shrink(R=2)
    assert getattr(mg_shrunk, "_extra_tag", None) == "preserved"
    assert (
        mg_shrunk.L,
        mg_shrunk.att_every,
        mg_shrunk.H,
        mg_shrunk.NH,
        mg_shrunk.NKV,
        mg_shrunk.KH,
        mg_shrunk.VH,
        mg_shrunk.E,
        mg_shrunk.TOPK,
        mg_shrunk.I,
        mg_shrunk.IS,
        mg_shrunk.e_slots,
    ) == (8, 4, 1024, 8, 2, 8, 16, 128, 4, 512, 512, 0)
    mg_shrunk.check(ExecSpec(R=2, B=2, T=1, ctx=128))
    mg_shrunk_ex = mg_large.shrink(ex=ExecSpec(R=4, B=4, T=1, ctx=128))
    assert (mg_shrunk_ex.NH, mg_shrunk_ex.NKV, mg_shrunk_ex.KH) == (8, 2, 8)
    mg_shrunk_ex.check(ExecSpec(R=4, B=4, T=1, ctx=128))

    # `moe_tp=True` with `e_slots == E` shrinks `e_slots` to `e_tiny`, while
    # `e_slots > E` is rejected by `check()` (`validate_moe_geometry`).
    mg_tp_slots = dataclasses.replace(mg_large, moe_tp=True, e_slots=512)
    mg_tp_slots.check(ExecSpec(R=8, B=8, T=1, ctx=128))
    mg_tp_slots_shrunk = mg_tp_slots.shrink(R=2)
    assert (mg_tp_slots_shrunk.E, mg_tp_slots_shrunk.e_slots) == (128, 128)
    mg_tp_slots_shrunk.check(ExecSpec(R=2, B=2, T=1, ctx=128))
    assert (
        dataclasses.replace(mg_large, moe_tp=True, e_slots=520).shrink(R=2).e_slots
        == 128
    )
    with pytest.raises(MegakernelConfigError, match="moe_tp=True does not support"):
        dataclasses.replace(mg, moe_tp=True, e_slots=136).check(
            ExecSpec(R=2, B=2, T=1, ctx=128)
        )

    # `moe_tp=False` / `e_slots` (valid EP + EPLB slots, `shrink()`, and `< ER` error).
    mg_ep = dataclasses.replace(mg_large, moe_tp=False, e_slots=80)
    assert (mg_ep.ER(8), mg_ep.EW(8), mg_ep.IR(8)) == (64, 80, 2048)
    mg_ep.check(ExecSpec(R=8, B=8, T=1, ctx=128))
    mg_ep_shrunk = mg_ep.shrink(R=2)
    assert (mg_ep_shrunk.E, mg_ep_shrunk.e_slots) == (128, 80)
    mg_ep_shrunk.check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="e_slots=8 must be 0 or >= ER=16"):
        dataclasses.replace(mg, moe_tp=False, e_slots=8).check(
            ExecSpec(R=8, B=8, T=1, ctx=128)
        )

    # Clamped `I=384` / `IS=384` rounds down to a valid multiple of `i_min` / `128 * R`.
    mg_384 = dataclasses.replace(mg, NKV=1, I=384, IS=384)
    mg_384.check(ExecSpec(R=1, B=1, T=1, ctx=128))
    for r_val in (1, 2, 4, 8):
        s_tp = mg_384.shrink(R=r_val)
        s_tp.check(ExecSpec(R=r_val, B=r_val, T=1, ctx=128))
        s_ep = dataclasses.replace(mg_384, moe_tp=False).shrink(R=r_val)
        s_ep.check(ExecSpec(R=r_val, B=r_val, T=1, ctx=128))
    assert (mg_384.shrink(R=2).I, mg_384.shrink(R=2).IS) == (256, 256)
    assert (
        dataclasses.replace(mg_384, moe_tp=False).shrink(R=2).I,
        dataclasses.replace(mg_384, moe_tp=False).shrink(R=2).IS,
    ) == (384, 256)

    # Attention-only (`att_every=1`, `NGDN == 0`) geometry and `shrink()`.
    mg_att_only = dataclasses.replace(mg_large, att_every=1, KH=0, VH=0, KD=0, VD=0)
    assert (mg_att_only.NGDN, mg_att_only.NATT) == (0, 32)
    mg_att_only.check(ExecSpec(R=8, B=8, T=1, ctx=128))
    s_att = mg_att_only.shrink(periods=3, R=2)
    assert (s_att.L, s_att.NGDN, s_att.KH, s_att.VH) == (3, 0, 0, 0)
    s_att.check(ExecSpec(R=2, B=2, T=1, ctx=128))

    rec_decl = GDNModule(geometry=mg).cache_decl(mg)
    assert rec_decl is not None
    kv_decl = KVCacheDecl(hd=mg.HD, kvh=1)
    play = pool.derive_pool_layout(([rec_decl] * 3 + [kv_decl]) * 2, R=8)
    pb1 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    pb2 = pool.make_pool_backend(play, check=mg.check, rec_chunk=GDNModule.chunk_rows)
    assert pb1 != pb2
    assert pb1.layout == pb2.layout

    # `paged_kv=True` valid and invalid configurations.
    mg.check(
        ExecSpec(
            R=2,
            B=2,
            T=1,
            ctx=256,
            cache=CacheExec(paged_kv=True, page_size=128, num_pages=8),
        )
    )
    for bad_cache, bad_ctx in (
        (CacheExec(paged_kv=True, page_size=64, num_pages=8), 256),
        (CacheExec(paged_kv=True, page_size=128, num_pages=8), 200),
        (CacheExec(paged_kv=True, page_size=128, num_pages=0), 256),
        (
            CacheExec(paged_kv=True, page_size=128, num_pages=8, max_pages_per_seq=1),
            256,
        ),
    ):
        with pytest.raises(MegakernelConfigError, match="Invalid paged_kv config"):
            mg.check(ExecSpec(R=2, B=2, T=1, ctx=bad_ctx, cache=bad_cache))

    # `check()` validation errors (MoE divisibility/bounds, `% 128` > 0, and GDN).
    with pytest.raises(MegakernelConfigError, match="divisible by att_every"):
        dataclasses.replace(mg, L=5).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid B=3, T=2"):
        mg.check(ExecSpec(R=2, B=3, T=2, ctx=128))
    with pytest.raises(MegakernelConfigError, match="divisible by R=8"):
        dataclasses.replace(mg, I=1028).check(ExecSpec(R=8, B=8, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="positive multiple of R=8"):
        dataclasses.replace(mg, IS=68).check(ExecSpec(R=8, B=8, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="positive multiple of 128"):
        dataclasses.replace(mg, IS=64).check(ExecSpec(R=8, B=8, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="positive multiple of 128"):
        dataclasses.replace(mg, H=0).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="gate"):
        dataclasses.replace(mg, gdn_gate="relu").check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid GDN heads"):
        dataclasses.replace(mg, KH=6).check(ExecSpec(R=4, B=4, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="Invalid GDN heads"):
        dataclasses.replace(mg, KH=0).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(MegakernelConfigError, match="GDN vh=128 exceeds 64"):
        dataclasses.replace(mg, KH=128, VH=256).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(
        MegakernelConfigError, match="GDN dimension .* positive multiple of 128"
    ):
        dataclasses.replace(mg, KD=64).check(ExecSpec(R=2, B=2, T=1, ctx=128))
    with pytest.raises(
        MegakernelConfigError, match="GDN dimension 0 must be a positive multiple"
    ):
        dataclasses.replace(mg, VD=0).check(ExecSpec(R=2, B=2, T=1, ctx=128))

    # `shrink()` argument validation errors (`periods` and `R`).
    for bad_periods in (0, -1, True):
        with pytest.raises(MegakernelConfigError, match="shrink periods"):
            mg.shrink(periods=bad_periods, R=2)  # type: ignore[arg-type]
    for bad_r in (0, 3, 16, True):
        with pytest.raises(MegakernelConfigError, match="shrink R"):
            mg.shrink(R=bad_r)  # type: ignore[arg-type]
    with pytest.raises(MegakernelConfigError, match="shrink R"):
        mg.shrink(ex=ExecSpec(R=16, B=16, T=1, ctx=128))


def test_schedule_every_and_from_hf_layer_types_resolution_and_errors():
    """`Every`, `FromHFLayerTypes`, and `Schedule.resolve` validate mixer order."""
    attn = _StubAttnModule.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    alt_gdn = GDN.bind("alt_linear_attn")
    moe = _StubFFNModule.bind("mlp")
    alt_moe = _StubFFNModule.bind("alt_mlp")
    in_norm = Norm.bind("input_layernorm", zero_centered=True)
    alt_in_norm = Norm.bind("alt_input_layernorm", zero_centered=True)
    post_norm = Norm.bind("post_attention_layernorm", zero_centered=True)
    alt_post_norm = Norm.bind("alt_post_attention_layernorm", zero_centered=True)

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
    exp_rec_decl = GDNModule(geometry=mg).cache_decl(mg)
    exp_kv_decl = KVCacheDecl(hd=mg.HD, kvh=mg.NKV)
    assert isinstance(exp_rec_decl, RecurrentCacheDecl)

    sched_every4 = Schedule(
        mixer=Every(n=4, last=attn, rest=gdn),
        ffn=moe,
        pre_norm=in_norm,
        post_norm=post_norm,
        overrides={
            2: {"ffn": alt_moe, "pre_norm": alt_in_norm, "post_norm": alt_post_norm},
            5: {"mixer": alt_gdn},
        },
    )
    res_every4 = sched_every4.resolve({}, 8)
    assert res_every4.num_layers == 8
    assert res_every4.att_every == 4
    assert res_every4.mixer_branch_indices == (0, 1, 2, 0, 3, 4, 5, 1)
    assert res_every4.mixer == (gdn, gdn, gdn, attn, gdn, alt_gdn, gdn, attn)
    assert res_every4.ffn == (moe, moe, alt_moe, moe, moe, moe, moe, moe)
    assert res_every4.pre_norm[2] == alt_in_norm and res_every4.pre_norm[0] == in_norm
    assert (
        res_every4.post_norm[2] == alt_post_norm
        and res_every4.post_norm[0] == post_norm
    )
    decls_every4 = res_every4.layer_cache_decls(mg)
    assert [type(d) for d in decls_every4] == (
        [RecurrentCacheDecl, RecurrentCacheDecl, RecurrentCacheDecl, KVCacheDecl] * 2
    )
    assert decls_every4 == ([exp_rec_decl] * 3 + [exp_kv_decl]) * 2

    sched_hf = Schedule(
        mixer=FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": gdn},
        ),
        ffn=moe,
        pre_norm=in_norm,
        post_norm=post_norm,
    )
    res_hf = sched_hf.resolve(
        {"layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2},
        8,
    )
    assert res_hf.num_layers == 8
    assert res_hf.att_every == 4
    assert res_hf.mixer_branch_indices == (0, 1, 2, 0, 3, 4, 5, 1)
    decls_hf = res_hf.layer_cache_decls(mg)
    assert decls_hf == ([exp_rec_decl] * 3 + [exp_kv_decl]) * 2

    res_no_cache = Schedule(
        mixer=Every(n=4, last=_NoCacheAttnModule.bind("self_attn"), rest=gdn),
        ffn=moe,
    ).resolve({}, 4)
    with pytest.raises(MegakernelConfigError, match=r"layer 3 has no cache_decl\(\)"):
        res_no_cache.layer_cache_decls(mg)

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
    sched_seq_norms = Schedule(
        mixer=[gdn, attn],
        ffn=[moe, alt_moe],
        pre_norm=[in_norm, None],
        post_norm=[None, post_norm],
    ).resolve({}, 2)
    assert sched_seq_norms.pre_norm == (in_norm, None)
    assert sched_seq_norms.post_norm == (None, post_norm)

    with pytest.raises(MegakernelConfigError, match=r"layer index 6.*num_layers=4"):
        sched_oob.resolve({}, 4)
