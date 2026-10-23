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
"""Unit tests for Megakernel Plan/Profile adapter and standalone reconciliation."""

from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import vllm_torchtpu.megakernel.roofline as mk_roof
from vllm_torchtpu.megakernel.roofline import (
    floor_for_profile,
    floor_from_plan,
    reconcile_plan_vs_standalone,
)
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    ep_busiest_experts,
    pass_costs,
)


@dataclass(frozen=True)
class _FakeWeightDecl:
    name: str
    owner: str
    K: int
    N: int
    dtype: str
    stream_ring: str = "static"
    lead_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class _FakeKVCacheDecl:
    hd: int
    kvh: int
    page_size: int
    page_dtype: str | None


@dataclass(frozen=True)
class _FakeRecurrentCacheDecl:
    kh: int
    vh: int
    KD: int
    VD: int
    rec_dtype: str | None
    conv_taps: int = 3
    conv_dtype: str | None = "bfloat16"


@dataclass(frozen=True)
class _FakeModule:
    name: str
    reduction: str | None
    _decl: object = None

    def cache_decl(self, config: Any = None) -> object:
        return self._decl


@dataclass(frozen=True)
class _FakeModelSpec:
    modules: tuple[_FakeModule, ...]
    config: Any = None

    def all_modules(self) -> tuple[_FakeModule, ...]:
        return self.modules


@dataclass(frozen=True)
class _FakePlan:
    module_layer_counts: dict[str, int]
    weight_decls: tuple[_FakeWeightDecl, ...]
    model_spec: _FakeModelSpec
    smalls_shape: tuple[tuple[int, int, int], str]
    module_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FakeExecSpec:
    B: int = 1
    T: int = 1
    R: int = 8
    ctx: int = 1024
    tiles: Any = None


QWEN35_GEOM = ModelGeometry(
    hidden=4096,
    num_layers=60,
    layer_types=tuple(["linear_attention"] * 45 + ["full_attention"] * 15),
    num_heads=32,
    num_kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    linear_k_heads=16,
    linear_v_heads=64,
    linear_k_dim=128,
    linear_v_dim=128,
    conv_kernel=4,
    num_experts=512,
    top_k=10,
    moe_inter=1024,
    shared_inter=1024,
    vocab=248320,
    mtp_layers=1,
)

_DEFAULT_MODULES = (
    _FakeModule(
        "gdn",
        "allreduce",
        _FakeRecurrentCacheDecl(
            kh=2, vh=8, KD=128, VD=128, rec_dtype="bfloat16", conv_dtype="bfloat16"
        ),
    ),
    _FakeModule(
        "gqa",
        "allreduce",
        _FakeKVCacheDecl(hd=256, kvh=1, page_size=128, page_dtype="float8_e4m3fn"),
    ),
    _FakeModule("moe", "allreduce", None),
)

QWEN35_TP8_PLAN = _FakePlan(
    module_layer_counts={"gdn": 45, "gqa": 15, "moe": 60},
    weight_decls=(
        _FakeWeightDecl("lin_in", "gdn", 4096, 2688, "float8_e4m3fn"),
        _FakeWeightDecl("lin_out", "gdn", 1024, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("att_in", "gqa", 4096, 2560, "float8_e4m3fn"),
        _FakeWeightDecl("att_out", "gqa", 1024, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("router", "moe", 4096, 512, "bfloat16"),
        _FakeWeightDecl("sh13", "moe", 4096, 256, "float8_e4m3fn"),
        _FakeWeightDecl("sh2", "moe", 128, 4096, "float8_e4m3fn"),
        _FakeWeightDecl("e13", "moe", 4096, 256, "float8_e4m3fn", "dynamic", (512,)),
        _FakeWeightDecl("e2", "moe", 128, 4096, "float8_e4m3fn", "dynamic", (512,)),
        _FakeWeightDecl("es", "moe", 1, 4352, "float32", "hbm", (512,)),
    ),
    model_spec=_FakeModelSpec(_DEFAULT_MODULES),
    smalls_shape=((60, 1, 30464), "float32"),
    module_params={"gqa": types.SimpleNamespace(SPLIT=True)},
)


def test_qwen35_tp8_reconciles_to_zero_without_plan_extras() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    rows = reconcile_plan_vs_standalone(
        QWEN35_TP8_PLAN,
        ex,
        QWEN35_GEOM,
        Parallel(tp=8, kv_split=4, kv_merge="exchange"),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(kv=1.0, state=2.0, conv=2.0),
        include_plan_extras=False,
    )
    by_mod = {r.module: r for r in rows}
    assert set(by_mod) == {"gqa", "gdn", "moe", "lm_head"}
    for r in rows:
        assert r.delta_hbm_bytes == pytest.approx(0.0, abs=1e-6), r
        assert r.delta_flops == pytest.approx(0.0, abs=1e-6), r


def test_qwen35_tp8_plan_extras_match_ba_pad_es_and_smalls() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    sf = floor_from_plan(
        QWEN35_TP8_PLAN,
        ex,
        geom=QWEN35_GEOM,
        chip=CHIPS["tpu7x"],
        include_plan_extras=True,
    )
    mod_map = {m.name: m for m in sf.modules}
    assert mod_map["gqa"].hbm_bytes == pytest.approx(222_167_040.0, rel=1e-6)
    assert mod_map["gdn"].hbm_bytes == pytest.approx(708_618_240.0, rel=1e-6)
    assert mod_map["moe"].hbm_bytes == pytest.approx(1_300_193_280.0, rel=1e-6)
    assert mod_map["smalls"].hbm_bytes == pytest.approx(7_311_360.0, rel=1e-6)


def test_megakernel_roofline_wrapper_passes_portability_ast_rules() -> None:
    tree = ast.parse(Path(mk_roof.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise AssertionError(f"Absolute import forbidden in megakernel: {node}")
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module == "__future__"


GEOM_128E = ModelGeometry(
    hidden=2048,
    num_layers=2,
    layer_types=("full_attention", "full_attention"),
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    attn_output_gate=False,
    linear_k_heads=0,
    linear_v_heads=0,
    linear_k_dim=0,
    linear_v_dim=0,
    conv_kernel=0,
    num_experts=128,
    top_k=8,
    moe_inter=2048,
    shared_inter=0,
    vocab=0,
    mtp_layers=0,
)

PLAN_128E = _FakePlan(
    module_layer_counts={"gqa": 2, "moe": 2},
    weight_decls=(
        _FakeWeightDecl("att_in", "gqa", 2048, 1024, "float8_e4m3fn"),
        _FakeWeightDecl("att_out", "gqa", 512, 2048, "float8_e4m3fn"),
        _FakeWeightDecl("router", "moe", 2048, 128, "bfloat16"),
        _FakeWeightDecl("e13", "moe", 2048, 4096, "float8_e4m3fn", "dynamic", (16,)),
        _FakeWeightDecl("e2", "moe", 2048, 2048, "float8_e4m3fn", "dynamic", (16,)),
    ),
    model_spec=_FakeModelSpec(
        (
            _FakeModule(
                "gqa",
                "allreduce",
                _FakeKVCacheDecl(
                    hd=128, kvh=1, page_size=128, page_dtype="float8_e4m3fn"
                ),
            ),
            _FakeModule("moe", "allreduce", None),
        )
    ),
    smalls_shape=((2, 1, 1024), "float32"),
)


def test_floor_for_profile_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_exec_mod = types.ModuleType("vllm_torchtpu.megakernel.exec_spec")
    fake_exec_mod.__dict__["ExecSpec"] = _FakeExecSpec
    fake_plan_mod = types.ModuleType("vllm_torchtpu.megakernel.planner")
    fake_plan_mod.__dict__["plan"] = lambda model_spec, ex, fit_vmem=False: PLAN_128E
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.megakernel.exec_spec", fake_exec_mod
    )
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.megakernel.planner", fake_plan_mod)

    class _FakeProfile:
        R = 8

        def exec_for(self, base: _FakeExecSpec, B: int, T: int) -> _FakeExecSpec:
            return _FakeExecSpec(B=B, T=T, R=base.R, ctx=base.ctx)

    sf = floor_for_profile(
        object(),
        _FakeProfile(),
        geom=GEOM_128E,
        B=1,
        S=1024,
        T=1,
        include_plan_extras=False,
    )
    sf_explicit = floor_from_plan(
        PLAN_128E,
        _FakeExecSpec(B=1, T=1, R=8, ctx=1024),
        geom=GEOM_128E,
        include_plan_extras=False,
    )
    assert sf.hbm_s == pytest.approx(sf_explicit.hbm_s, rel=1e-12)
    assert sf.mm_s == pytest.approx(sf_explicit.mm_s, rel=1e-12)


def test_adapter_ep_lead_dim_uses_busiest_rank() -> None:
    ep_plan = _FakePlan(
        module_layer_counts={"gdn": 45, "gqa": 15, "moe": 60},
        weight_decls=(
            _FakeWeightDecl("lin_in", "gdn", 4096, 2688, "float8_e4m3fn"),
            _FakeWeightDecl("lin_out", "gdn", 1024, 4096, "float8_e4m3fn"),
            _FakeWeightDecl("att_in", "gqa", 4096, 2560, "float8_e4m3fn"),
            _FakeWeightDecl("att_out", "gqa", 1024, 4096, "float8_e4m3fn"),
            _FakeWeightDecl("router", "moe", 4096, 512, "bfloat16"),
            _FakeWeightDecl("sh13", "moe", 4096, 256, "float8_e4m3fn"),
            _FakeWeightDecl("sh2", "moe", 128, 4096, "float8_e4m3fn"),
            _FakeWeightDecl(
                "e13", "moe", 4096, 2048, "float8_e4m3fn", "dynamic", (64,)
            ),
            _FakeWeightDecl("e2", "moe", 1024, 4096, "float8_e4m3fn", "dynamic", (64,)),
            _FakeWeightDecl("es", "moe", 1, 4352, "float32", "hbm", (64,)),
        ),
        model_spec=_FakeModelSpec(_DEFAULT_MODULES),
        smalls_shape=((60, 1, 30464), "float32"),
    )
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=90036)
    sf = floor_from_plan(ep_plan, ex, geom=QWEN35_GEOM, include_plan_extras=False)
    by_name = {m.name: m for m in sf.modules}
    std = {
        m.name: m
        for m in pass_costs(
            QWEN35_GEOM,
            Parallel(tp=8, moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=90036),
            Dtypes(state=2.0),
        )
    }
    assert ep_busiest_experts(512, 10, 1, 8) == pytest.approx(2.979897, abs=1e-6)
    assert by_name["moe"].hbm_bytes == pytest.approx(std["moe"].hbm_bytes, abs=1.0)


def test_adapter_raises_on_missing_cache_decl() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    bad_gqa = _FakePlan(
        module_layer_counts={"gqa": 15},
        weight_decls=(_FakeWeightDecl("att_in", "gqa", 4096, 2560, "float8_e4m3fn"),),
        model_spec=_FakeModelSpec((_FakeModule("gqa", "allreduce", None),)),
        smalls_shape=((15, 1, 1024), "float32"),
    )
    with pytest.raises(ValueError, match="gqa"):
        floor_from_plan(bad_gqa, ex, geom=QWEN35_GEOM)

    bad_gdn = _FakePlan(
        module_layer_counts={"gdn": 45},
        weight_decls=(_FakeWeightDecl("lin_in", "gdn", 4096, 2688, "float8_e4m3fn"),),
        model_spec=_FakeModelSpec(
            (
                _FakeModule(
                    "gdn",
                    "allreduce",
                    _FakeRecurrentCacheDecl(kh=2, vh=8, KD=128, VD=128, rec_dtype=None),
                ),
            )
        ),
        smalls_shape=((45, 1, 1024), "float32"),
    )
    with pytest.raises(ValueError, match="gdn"):
        floor_from_plan(bad_gdn, ex, geom=QWEN35_GEOM)


def test_adapter_derives_parallel_from_exec_spec() -> None:
    ex_rsag = _FakeExecSpec(
        B=4,
        T=4,
        R=8,
        ctx=1024,
        tiles=types.SimpleNamespace(rsag=True, hier=0),
    )
    sf = floor_from_plan(
        QWEN35_TP8_PLAN,
        ex_rsag,
        geom=QWEN35_GEOM,
        include_plan_extras=False,
    )
    colls = [c for m in sf.modules for c in m.collectives]
    assert any(c.name == "merge_xchg" for c in colls)
    out_ars = [c for c in colls if c.name == "out_ar"]
    assert out_ars and all(c.hops == 2 for c in out_ars)

    ex_hier = _FakeExecSpec(
        B=1,
        T=1,
        R=8,
        ctx=1024,
        tiles=types.SimpleNamespace(rsag=False, hier=1),
    )
    with pytest.raises(NotImplementedError, match="hier=1"):
        floor_from_plan(QWEN35_TP8_PLAN, ex_hier, geom=QWEN35_GEOM)


def test_adapter_lm_head_uses_dtypes() -> None:
    ex = _FakeExecSpec(B=1, T=1, R=8, ctx=1024)
    sf_bf16 = floor_from_plan(
        QWEN35_TP8_PLAN,
        ex,
        geom=QWEN35_GEOM,
        dt=Dtypes(lm_head=2.0),
        include_plan_extras=False,
    )
    sf_fp8 = floor_from_plan(
        QWEN35_TP8_PLAN,
        ex,
        geom=QWEN35_GEOM,
        dt=Dtypes(lm_head=1.0, act="a8"),
        include_plan_extras=False,
    )
    lm_bf16 = next(m for m in sf_bf16.modules if m.name == "lm_head")
    lm_fp8 = next(m for m in sf_fp8.modules if m.name == "lm_head")
    assert lm_bf16.hbm_bytes == pytest.approx(254_279_680.0)
    assert lm_fp8.hbm_bytes == pytest.approx(127_139_840.0)
    assert lm_bf16.bf16_flops > 0.0 and lm_bf16.fp8_flops == 0.0
    assert lm_fp8.fp8_flops > 0.0 and lm_fp8.bf16_flops == 0.0
