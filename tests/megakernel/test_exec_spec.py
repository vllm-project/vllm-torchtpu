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

from vllm_torchtpu.megakernel import exec_spec
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec, Tiles
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
    """Test-only block tunables (registered under `_TEST_BLOCK`)."""

    width: int = 128
    fast: bool = False
    NUMERICS = {"width": "approx", "fast": "exact"}


_TEST_BLOCK = "test_exec_spec_block"


@pytest.fixture(autouse=True)
def _test_tunables(monkeypatch):
    """Register `_TestTunables` for one test; the registry is restored after."""
    monkeypatch.setitem(exec_spec._TUNABLES, _TEST_BLOCK, _TestTunables)


def test_core_tile_numerics_cover_every_core_field():
    names = {f.name for f in dataclasses.fields(Tiles)} - {"blocks"}
    assert set(exec_spec.TILE_NUMERICS) == names
    assert set(exec_spec.TILE_NUMERICS.values()) <= {"exact", "approx"}
    numerics = exec_spec.tile_numerics()
    assert {
        k: v for k, v in numerics.items() if "." not in k
    } == exec_spec.TILE_NUMERICS
    assert numerics[f"{_TEST_BLOCK}.width"] == "approx"
    assert numerics[f"{_TEST_BLOCK}.fast"] == "exact"


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
        exec_spec.register_tunables(_TEST_BLOCK, Other)
    exec_spec.register_tunables(_TEST_BLOCK, _TestTunables)  # same class: no-op
    assert exec_spec.tunables_class(_TEST_BLOCK) is _TestTunables


def test_tiles_block_tunables_are_canonical_and_hashable():
    t = Tiles()
    assert t.block(_TEST_BLOCK) == _TestTunables()
    same = Tiles(blocks={_TEST_BLOCK: _TestTunables()})
    assert same == t and hash(same) == hash(t) and not same.blocks
    t2 = t.with_blocks({_TEST_BLOCK: {"width": 256}})
    assert t2.block(_TEST_BLOCK).width == 256 and t2 != t
    assert hash(t2) == hash(t.with_blocks({_TEST_BLOCK: {"width": 256}}))
    with pytest.raises(TypeError):
        t.with_blocks({_TEST_BLOCK: {"no_such_field": 1}})
    with pytest.raises(KeyError):
        t.with_blocks({"no_such_block": {"x": 1}})


def test_exec_spec_imports_only_errors_ops_and_profiles():
    """`exec_spec.py` depends only on Python stdlib, `errors`, `ops`, `profiles`."""
    path = Path(exec_spec.__file__)
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
                assert set(roots) <= {"errors", "ops", "profiles"}, roots


@dataclasses.dataclass(frozen=True)
class _OtherOptions:
    width: int = 128


@dataclasses.dataclass
class _MutableOptions:
    width: int = 128


def test_block_maps_check_entry_types():
    """`Tiles.blocks` takes only the block's registered `Tunables` class;
    `DebugOptions.blocks` any frozen dataclass, both canonical."""
    with pytest.raises(TypeError, match="_TestTunables"):
        Tiles(blocks={_TEST_BLOCK: _OtherOptions(width=256)})
    with pytest.raises(TypeError, match="with_blocks"):
        Tiles(blocks={_TEST_BLOCK: {"width": 256}})
    dbg = exec_spec.DebugOptions(blocks={_TEST_BLOCK: _OtherOptions(width=256)})
    assert dbg.blocks[_TEST_BLOCK].width == 256
    assert (
        exec_spec.DebugOptions(blocks={_TEST_BLOCK: _OtherOptions()})
        == exec_spec.DebugOptions()
    )
    with pytest.raises(TypeError, match="DebugOptions.blocks"):
        exec_spec.DebugOptions(blocks={_TEST_BLOCK: {"width": 256}})
    # A mutable dataclass would make `ExecSpec` unhashable.
    with pytest.raises(TypeError, match="frozen"):
        exec_spec.DebugOptions(blocks={_TEST_BLOCK: _MutableOptions(width=256)})


def test_tiles_reject_unregistered_block_names():
    """A misspelled block name raises instead of leaving that block at defaults."""
    typo = f"{_TEST_BLOCK}_typo"
    with pytest.raises(KeyError, match="no tunables registered"):
        Tiles(blocks={typo: _TestTunables(width=256)})
    with pytest.raises(KeyError, match="no tunables registered"):
        Tiles().with_blocks({typo: _TestTunables(width=256)})
    t = Tiles(blocks={_TEST_BLOCK: _TestTunables(width=256)})
    with pytest.raises(TypeError, match="_OtherOptions"):
        t.block(_TEST_BLOCK, _OtherOptions)


def test_default_tile_rule_is_model_agnostic():
    """One tile set for every bucket, rank count and TP flag; chip budgets."""
    ref = default_tiles(1)
    # The rule is the `Tiles` field defaults, whose budgets are the default chip's.
    assert ref == Tiles()
    assert ref.hier == 0  # flat-ring all-reduce, no topology-specific mode
    assert (ref.vmem_mib, ref.smem_mib) == (
        DEFAULT_CHIP.vmem_mib,
        DEFAULT_CHIP.smem_mib,
    )
    for R in (1, 2, 4, 8, 16):
        for B in range(1, 65):
            assert exec_spec.default_tiles(B, R=R) == ref
            for moe_tp in (False, True):
                assert default_tiles(B, moe_tp=moe_tp, R=R) == ref
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    assert default_tiles(8, chip=small).vmem_mib == 16
    assert exec_spec.default_tiles(8, chip=small).vmem_mib == 16


def test_chip_budget_follows_device_kind_and_rejects_unknown_tpus(monkeypatch):
    assert chip_budget("TPU7x") is TPU_V7X
    with pytest.raises(MegakernelConfigError, match="TPU v6 lite"):
        chip_budget("TPU v6 lite")

    def first_device(platform: str, device_kind: str) -> None:
        dev = types.SimpleNamespace(platform=platform, device_kind=device_kind)
        monkeypatch.setattr(default_profile, "_first_device", lambda: dev)

    first_device("cpu", "cpu")
    assert chip_budget() is DEFAULT_CHIP  # CPU interpret runs: the v7x default
    first_device("tpu", "TPU7x")
    assert chip_budget() is TPU_V7X
    first_device("tpu", "TPU v6 lite")
    with pytest.raises(MegakernelConfigError, match="CHIP_BUDGETS"):
        chip_budget()
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    monkeypatch.setitem(default_profile.CHIP_BUDGETS, "TPU v6 lite", small)
    assert chip_budget() is small
    # The tile rule itself never queries devices (profiles run it at import).
    assert default_tiles(8) == default_tiles(8, chip=DEFAULT_CHIP)


def test_resolved_tiles_use_the_default_rule_only_when_unset():
    for B in (4, 8, 16, 32, 64):
        assert ExecSpec(B=B, R=8).resolved_tiles == default_tiles(B, R=8)
    t = Tiles(nbd=6)
    assert ExecSpec(tiles=t).resolved_tiles is t
    assert ExecSpec(tiles=t).resolve_tiles() is t


def test_debug_skip_is_canonical_and_hashable():
    a = ExecSpec(debug=DebugOptions(skip={"kv"}))
    b = ExecSpec(debug=DebugOptions(skip=frozenset({"kv"})))
    assert a.skip == frozenset({"kv"}) and a == b and hash(a) == hash(b)
    with pytest.raises(TypeError, match="collection of phase names"):
        DebugOptions(skip="kv")


@pytest.mark.parametrize(
    "kw", [{"B": 6, "T": 4}, {"B": 0}, {"T": 0}, {"R": 0}, {"ctx": 0}]
)
def test_exec_spec_rejects_invalid_shapes(kw):
    with pytest.raises(MegakernelConfigError):
        ExecSpec(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"hier": 4},
        {"war": 2},
        {"mb8": 3},
        {"nbd": 0},
        {"nbe": 0},
        {"vmem_mib": -1},
        {"smem_mib": -1},
        {"wide_rows": -1},
        {"dmm_unroll": -1},
        {"bank": 0},
        # Not a whole number of 4 KiB VMEM tiles: ring slot 1 would start
        # mid-tile.
        {"bank": (2 << 20) + 128},
        {"ebank": 3000},
    ],
)
def test_tiles_reject_out_of_range_fields(kw):
    (name,) = kw
    with pytest.raises(MegakernelConfigError, match=f"{name}="):
        Tiles(**kw)
    with pytest.raises(MegakernelConfigError, match=f"{name}="):
        dataclasses.replace(Tiles(), **kw)
