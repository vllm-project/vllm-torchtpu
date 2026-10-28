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
"""Unit tests for `exec_spec` (core tiles, tunables registry, ExecSpec) and the
model-agnostic default tile rule (`profiles.default`)."""

from __future__ import annotations

import ast
import dataclasses
import sys
import types
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import exec_spec, module_options
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.profiles import default as default_profile
from vllm_torchtpu.megakernel.profiles.default import (
    DEFAULT_CHIP,
    TPU_V7X,
    ChipBudget,
    chip_budget,
    default_tiles,
)


@dataclasses.dataclass(frozen=True)
class _TestTunables:
    """Test-only module tunables (registered under `_TEST_MODULE`)."""

    width: int = 128
    fast: bool = False
    NUMERICS = {"width": "approx", "fast": "exact"}


_TEST_MODULE = "test_exec_spec_module"


@dataclasses.dataclass(frozen=True)
class _OtherOptions:
    width: int = 128


@dataclasses.dataclass
class _MutableOptions:
    width: int = 128


@pytest.fixture(autouse=True)
def _test_tunables(monkeypatch):
    """Register `_TestTunables` and `_OtherOptions` for one test; restored after."""
    monkeypatch.setitem(module_options._TUNABLES, _TEST_MODULE, _TestTunables)
    monkeypatch.setitem(module_options._DEBUG, _TEST_MODULE, _OtherOptions)


def test_core_tile_numerics_cover_every_core_field():
    names = {f.name for f in dataclasses.fields(Tiles)} - {"modules"}
    assert set(exec_spec.TILE_NUMERICS) == names
    assert set(exec_spec.TILE_NUMERICS.values()) <= {"exact", "approx"}
    numerics = exec_spec.tile_numerics()
    assert {
        k: v for k, v in numerics.items() if "." not in k
    } == exec_spec.TILE_NUMERICS
    assert numerics[f"{_TEST_MODULE}.width"] == "approx"
    assert numerics[f"{_TEST_MODULE}.fast"] == "exact"


def test_tunables_registration_rejects_incomplete_or_conflicting_classes():
    @dataclasses.dataclass(frozen=True)
    class Partial:
        a: int = 0
        b: int = 1
        NUMERICS = {"a": "exact"}

    @dataclasses.dataclass(frozen=True)
    class Other:
        width: int = 128
        NUMERICS = {"width": "approx"}

    with pytest.raises(TypeError, match="missing"):
        exec_spec.register_tunables("test_partial", Partial)
    with pytest.raises(ValueError, match="already registered"):
        exec_spec.register_tunables(_TEST_MODULE, Other)
    exec_spec.register_tunables(_TEST_MODULE, _TestTunables)  # same class: no-op
    assert exec_spec.tunables_class(_TEST_MODULE) is _TestTunables


def test_tiles_module_tunables_are_canonical_and_hashable():
    t = Tiles()
    assert t.module(_TEST_MODULE) == _TestTunables()
    same = Tiles(modules={_TEST_MODULE: _TestTunables()})
    assert same == t and hash(same) == hash(t) and not same.modules
    t2 = t.with_modules({_TEST_MODULE: {"width": 256}})
    assert t2.module(_TEST_MODULE).width == 256 and t2 != t
    assert hash(t2) == hash(t.with_modules({_TEST_MODULE: {"width": 256}}))
    with pytest.raises(TypeError):
        t.with_modules({_TEST_MODULE: {"no_such_field": 1}})
    with pytest.raises(KeyError):
        t.with_modules({"no_such_module": {"x": 1}})


def test_exec_spec_imports_only_ops_and_profiles():
    """`exec_spec` and `module_options` depend only on stdlib, `ops` and `profiles`."""
    for mod_obj, allowed_rel in (
        (module_options, set()),
        (exec_spec, {"module_options", "ops", "profiles"}),
    ):
        path = Path(mod_obj.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 0:
                    assert mod.split(".")[0] in sys.stdlib_module_names
                else:
                    assert node.level == 1
                    roots = [mod.split(".")[0]] if mod else [a.name for a in node.names]
                    assert set(roots) <= allowed_rel, (path.name, roots)


def test_module_maps_check_entry_types():
    """`Tiles.modules` takes only the module's registered `Tunables` class;
    `DebugOptions.modules` takes only the module's registered `Debug` class."""
    with pytest.raises(TypeError, match="_TestTunables"):
        Tiles(modules={_TEST_MODULE: _OtherOptions(width=256)})
    with pytest.raises(TypeError, match="with_modules"):
        Tiles(modules={_TEST_MODULE: {"width": 256}})
    dbg = exec_spec.DebugOptions(modules={_TEST_MODULE: _OtherOptions(width=256)})
    assert dbg.modules[_TEST_MODULE].width == 256
    assert (
        exec_spec.DebugOptions(modules={_TEST_MODULE: _OtherOptions()})
        == exec_spec.DebugOptions()
    )
    with pytest.raises(TypeError, match="DebugOptions.modules"):
        exec_spec.DebugOptions(modules={_TEST_MODULE: {"width": 256}})
    with pytest.raises(TypeError, match="_OtherOptions"):
        exec_spec.DebugOptions(modules={_TEST_MODULE: _TestTunables(width=256)})
    with pytest.raises(KeyError, match="no Debug class registered"):
        exec_spec.DebugOptions(modules={"no_such_module": _OtherOptions(width=256)})
    # A mutable dataclass would make `ExecSpec` unhashable.
    with pytest.raises(TypeError, match="frozen"):
        exec_spec.DebugOptions(modules={_TEST_MODULE: _MutableOptions(width=256)})


def test_tiles_reject_unregistered_module_names():
    """A misspelled module name raises instead of leaving that module at defaults."""
    typo = f"{_TEST_MODULE}_typo"
    with pytest.raises(KeyError, match="no tunables registered"):
        Tiles(modules={typo: _TestTunables(width=256)})
    with pytest.raises(KeyError, match="no tunables registered"):
        Tiles().with_modules({typo: _TestTunables(width=256)})
    t = Tiles(modules={_TEST_MODULE: _TestTunables(width=256)})
    with pytest.raises(TypeError, match="_OtherOptions"):
        t.module(_TEST_MODULE, _OtherOptions)


def test_default_tile_rule_is_model_agnostic():
    """One tile set for every bucket, rank count and TP flag; chip budgets."""
    ref = default_tiles(1)
    assert ref.hier == 0  # flat-ring all-reduce, no topology-specific mode
    assert (ref.vmem_mib, ref.smem_mib) == (
        DEFAULT_CHIP.vmem_mib,
        DEFAULT_CHIP.smem_mib,
    )
    for R in (1, 2, 4, 8, 16):
        for moe_tp in (False, True):
            for B in range(1, 65):
                assert exec_spec.default_tiles(B, moe_tp=moe_tp, R=R) == ref
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    assert default_tiles(8, chip=small).vmem_mib == 16


def test_chip_budget_follows_device_kind_and_rejects_unsupported_tpu(monkeypatch):
    assert chip_budget("TPU7x") is TPU_V7X
    with pytest.raises(MegakernelConfigError, match="TPU v5 lite.*TPU7x"):
        chip_budget("TPU v5 lite")
    assert chip_budget("cpu") is DEFAULT_CHIP
    assert chip_budget() is DEFAULT_CHIP  # CPU interpret runs: the v7x fallback
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    monkeypatch.setitem(
        default_profile.CHIP_BUDGETS, default_profile._first_device_kind(), small
    )
    assert chip_budget() is small
    # The tile rule itself never queries devices (profiles run it at import).
    assert default_tiles(8) == default_tiles(8, chip=DEFAULT_CHIP)


def test_resolved_tiles_agree_with_resolve_tiles_for_tp_geometry():
    g_tp = types.SimpleNamespace(moe_tp=True)
    for B in (4, 8, 16, 32, 64):
        ex = ExecSpec(B=B, R=8)
        assert ex.resolved_tiles == ex.resolve_tiles(g_tp) == ex.resolve_tiles()
