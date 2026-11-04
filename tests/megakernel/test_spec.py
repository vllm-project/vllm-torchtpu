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
"""Unit tests for `spec` (model declarations, `Module`, `SwitchSlot`, `ModelSpec`)."""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import decls, exec_spec, module_options
from vllm_torchtpu.megakernel import module as module_mod
from vllm_torchtpu.megakernel import spec as spec_mod
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.spec import (
    ExtraInput,
    LayerType,
    ModelSpec,
    Module,
    ScratchReq,
    SmallField,
    Spec,
    StateDecl,
    SwitchSlot,
    WeightDecl,
)


def test_spec_imports_only_exec_spec_lazily_or_under_type_checking():
    """`decls.py` is stdlib-only at runtime; `module.py` and `spec.py` only import
    `decls`/`module` at top level and `exec_spec` lazily / under `TYPE_CHECKING`."""
    for mod_obj, top_rel, all_rel in (
        (decls, set(), {"module"}),
        (module_mod, {"decls"}, {"decls", "exec_spec", "spec"}),
        (spec_mod, {"decls", "module"}, {"decls", "module"}),
    ):
        path = Path(mod_obj.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.level > 0:
                assert stmt.level == 1 and (stmt.module or "") in top_rel, (
                    path.name,
                    stmt.level,
                    stmt.module,
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 0:
                    assert mod.split(".")[0] in sys.stdlib_module_names
                else:
                    assert node.level == 1 and mod in all_rel, (
                        path.name,
                        node.level,
                        mod,
                    )


def test_spec_declaration_dataclasses_and_defaults():
    s = Spec(shape=(4, 128), dtype=BF16)
    assert s.shape == (4, 128) and s.dtype == BF16

    w = WeightDecl("proj", K=256, N=512, dtype=FP8)
    assert w.stream_ring == "static" and w.layout == "stacked" and w.lead_dims == ()

    st = StateDecl("cache_k", shape=(8, 128), dtype=BF16)
    assert (
        st.access == "r"
        and st.per_layer is True
        and st.memory_space == "hbm"
        and st.tupled is False
    )

    ex_in = ExtraInput("routing_override", memory_space="smem", shape=(4,), dtype=F32)
    assert ex_in.memory_space == "smem" and ex_in.shape == (4,)

    sm = SmallField("norm_w", width=256)
    assert sm.name == "norm_w" and sm.width == 256

    scr = ScratchReq("tmp_buf", spec=(16, 128))
    assert scr.phase == "persistent"


def test_module_subclass_tunables_registration_and_default_hooks(monkeypatch):
    tunables_copy = dict(module_options._TUNABLES)
    debug_copy = dict(module_options._DEBUG)
    monkeypatch.setattr(module_options, "_TUNABLES", tunables_copy)
    monkeypatch.setattr(module_options, "_DEBUG", debug_copy)

    @dataclasses.dataclass(frozen=True)
    class _CustomTunables:
        tile_k: int = 128
        unroll: bool = False
        NUMERICS = {"tile_k": "approx", "unroll": "exact"}

    @dataclasses.dataclass(frozen=True)
    class _CustomDebug:
        force_path: bool = False

    class _CustomModule(Module):
        name = "test_spec_custom_module"
        Tunables = _CustomTunables
        Debug = _CustomDebug

    assert exec_spec.tunables_class("test_spec_custom_module") is _CustomTunables
    mod = _CustomModule()
    ex = ExecSpec(B=4, T=1, R=2)
    assert mod.tunables(ex) == _CustomTunables()
    assert mod.debug_flags(ex) == _CustomDebug()
    ex_override = ExecSpec(
        B=4,
        T=1,
        R=2,
        tiles=Tiles().with_modules({"test_spec_custom_module": {"tile_k": 256}}),
        debug=exec_spec.DebugOptions(
            modules={"test_spec_custom_module": _CustomDebug(force_path=True)}
        ),
    )
    assert mod.tunables(ex_override).tile_k == 256
    assert mod.debug_flags(ex_override).force_path is True
    with pytest.raises(TypeError, match="_CustomDebug"):
        mod.debug_flags(
            ExecSpec(
                B=4,
                T=1,
                R=2,
                debug=exec_spec.DebugOptions(
                    modules={"test_spec_custom_module": _CustomTunables(tile_k=256)}
                ),
            )
        )

    plain = Module()
    with pytest.raises(AssertionError, match="declares no Tunables"):
        plain.tunables(ex)
    with pytest.raises(AssertionError, match="declares no Debug"):
        plain.debug_flags(ex)
    assert plain.weights(None, ex) == []
    assert plain.states(None, ex) == []
    assert plain.outputs(None, ex) == []
    assert plain.extra_inputs(None, ex) == []
    assert plain.smalls(None, ex) == []
    assert plain.shared_widths(None, ex) == {}
    assert plain.plan_params(None, ex, None) is None
    assert plain.scratch(None, ex, None) == {}
    assert plain.host_update_states(None, ex, None, {}, {}) == {}
    plain.prologue(None)
    plain.init_buffers(None)
    plain.on_layer_start(None, 0, 0)
    plain.epilogue(None)
    with pytest.raises(NotImplementedError):
        plain.emit(None, 0, 0)


def test_switch_slot_and_model_spec_all_modules():
    class _BranchA(Module):
        name = "branch_a"

    class _BranchB(Module):
        name = "branch_b"

    class _FFN(Module):
        name = "ffn_module"

    a, b, ffn = _BranchA(), _BranchB(), _FFN()
    sw_default = SwitchSlot("mixer", branches=(a, b))
    assert sw_default.branch_counts == (1, 1)
    assert sw_default.period == 2
    assert sw_default.active_branches == (a, b)
    assert sw_default.active_counts == (1, 1)

    sw_filtered = SwitchSlot("mixer", branches=(a, b), branch_counts=(0, 3))
    assert sw_filtered.period == 3
    assert sw_filtered.active_pairs == ((b, 3),)
    assert sw_filtered.active_branches == (b,)
    assert sw_filtered.active_counts == (3,)

    ms = ModelSpec(
        name="toy_model",
        config=None,
        layer_type=LayerType("toy_layer", slots=(sw_filtered, ffn, b)),
    )
    assert ms.all_modules() == [b, ffn]

    with pytest.raises(ValueError, match="distinct modules sharing name"):
        ModelSpec(
            name="dup_model",
            config=None,
            layer_type=LayerType("dup_layer", slots=(_BranchA(), _BranchA())),
        ).all_modules()

    with pytest.raises(AssertionError):
        SwitchSlot("bad", branches=(a, b), branch_counts=(1,))
    with pytest.raises(AssertionError):
        SwitchSlot("bad_zero", branches=(a, b), branch_counts=(0, 0))
