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
"""Unit tests for `planner`."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import (
    FakeAltMixerModule,
    FakeGeometry,
    FakeMixerModule,
    make_fake_model_spec,
)

from vllm_torchtpu.megakernel import module_options, planner
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import FP8
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    SmallField,
    SwitchSlot,
    WeightDecl,
)


def test_planner_is_load_bearing_single_branch(fake_ex: ExecSpec):
    """Verify `planner.plan` derives only weights/smalls/states from active modules."""
    ms = make_fake_model_spec(L=4, att_every=1, H=256)
    p = planner.plan(ms, fake_ex)

    # Only FakeMixerModule and FakeFFNModule are in the spec -> no alt-mixer decls
    assert p.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p.dynamic_kinds == ("e13", "e2")
    assert "lin_in" not in p.kinds
    assert "conv" not in p.smalls_layout
    assert "rec" not in p.s_names
    assert p.s_names == ("kc", "vc")
    assert p.o_names == ("hidden", "knew", "vnew")

    wsh = planner.weight_shapes(ms, fake_ex, p)
    ssh = planner.state_shapes(ms, fake_ex, p)
    osh = planner.kernel_out_shapes(ms, fake_ex, p)

    assert set(wsh.keys()) == set(p.w_names)
    assert set(ssh.keys()) == {"kc", "vc"}
    assert set(osh.keys()) == {"hidden", "knew", "vnew"}
    # `es` has leading `(L, EW, 1, 2*IR + H)` derived from WeightDecl.lead_dims=(EW,)
    assert wsh["es"][0] == (4, 64, 1, 2 * 256 + 256)


def test_planner_hybrid_switch_slot(fake_ex: ExecSpec):
    """Verify multi-branch `SwitchSlot` derives per-branch counts and schedules."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    p = planner.plan(ms, fake_ex)

    assert p.period == 4
    assert p.branch_counts == (3, 1)
    assert p.module_layer_counts["fake_gdn"] == 3
    assert p.module_layer_counts["fake_gqa"] == 1
    assert p.module_layer_counts["fake_moe"] == 4

    wsh = planner.weight_shapes(ms, fake_ex, p)
    assert wsh["lin_in"][0][0] == 3
    assert wsh["att_in"][0][0] == 1
    assert wsh["router"][0][0] == 4

    # Shared smalls field `in_s` takes max width across branches (NAI=512, NLI=640)
    _, in_s_w = p.smalls_layout["in_s"]
    assert in_s_w == 640

    # >2 branches in a SwitchSlot raises AssertionError at plan time
    class ThirdBranchModule(Module):
        name = "third"

    bad_sw = SwitchSlot(
        "mixer",
        branches=(FakeAltMixerModule(), FakeMixerModule(), ThirdBranchModule()),
        branch_counts=(1, 1, 2),
    )
    bad_ms = ModelSpec(
        name="bad_3branch",
        config=FakeGeometry(L=4, H=256),
        layer_type=LayerType("l", slots=(bad_sw,)),
    )
    with pytest.raises(AssertionError, match="supports at most 2 active branches"):
        planner.plan(bad_ms, fake_ex)


def test_planner_custom_module_extension(fake_ex: ExecSpec):
    """Verify adding a custom `Module` updates `plan` without editing `planner.py`."""

    class CustomMLPModule(Module):
        name = "custom_mlp"
        reduction = "allreduce"

        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [
                WeightDecl("mlp_up", K=g.H, N=512, dtype=FP8, stream_ring="static"),
                WeightDecl("mlp_down", K=512, N=g.H, dtype=FP8, stream_ring="static"),
            ]

        def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
            return [SmallField("mlp_scale", 512)]

        def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
            return {"acc": 512, "xb2": 512}

    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        name="custom_model",
        config=g,
        layer_type=LayerType(
            "custom_layer", slots=(FakeMixerModule(), CustomMLPModule())
        ),
    )
    p = planner.plan(ms, fake_ex)
    assert p.static_kinds == ("att_in", "att_out", "mlp_up", "mlp_down")
    assert p.dynamic_kinds == ()
    assert "mlp_scale" in p.smalls_layout
    assert p.shared_widths["acc"] >= 512 and p.shared_widths["xb2"] >= 512


def test_unsupported_stream_ring_is_config_error(
    fake_ex: ExecSpec, monkeypatch: pytest.MonkeyPatch
):
    """Unknown `WeightDecl.stream_ring` raises `MegakernelConfigError` at plan time."""
    orig = FakeMixerModule.weights

    def _weights(
        self: FakeMixerModule, g: FakeGeometry, ex: ExecSpec
    ) -> list[WeightDecl]:
        ws = orig(self, g, ex)
        return [dataclasses.replace(ws[0], stream_ring="bogus"), *ws[1:]]  # type: ignore[arg-type]

    monkeypatch.setattr(FakeMixerModule, "weights", _weights)
    with pytest.raises(MegakernelConfigError, match="Unsupported stream_ring"):
        planner.plan(make_fake_model_spec(), fake_ex)


def test_plan_is_pure_and_rejects_inactive_debug_module(
    fake_ex: ExecSpec, monkeypatch: pytest.MonkeyPatch
):
    """`plan` and `*_shapes` never fit tiles or query device budgets;
    `plan` rejects `DebugOptions.modules` targeting a module not in `model_spec`."""
    import jax

    from vllm_torchtpu.megakernel import exec_spec

    ms = make_fake_model_spec(L=4, att_every=1, H=256)
    ex_unset = dataclasses.replace(fake_ex, tiles=None)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("jax.devices() must not be called by planner")
        ),
    )
    p = planner.plan(ms, ex_unset)
    assert p.ex.tiles is None
    wsh = planner.weight_shapes(ms, ex_unset)
    ssh = planner.state_shapes(ms, ex_unset)
    osh = planner.kernel_out_shapes(ms, ex_unset)
    assert set(wsh) == set(p.w_names)
    assert set(ssh) == set(p.s_names)
    assert set(osh) == set(p.o_names)

    @dataclasses.dataclass(frozen=True)
    class _InactiveDebug:
        flag: bool = False

    monkeypatch.setitem(module_options._DEBUG, "fake_gdn", _InactiveDebug)
    ex_bad_dbg = dataclasses.replace(
        fake_ex,
        debug=exec_spec.DebugOptions(modules={"fake_gdn": _InactiveDebug(flag=True)}),
    )
    with pytest.raises(
        MegakernelConfigError, match="unknown or inactive module.*fake_gdn"
    ):
        planner.plan(ms, ex_bad_dbg)


def test_planner_transitive_imports_exclude_vmem_kernel_and_cache():
    """`planner.py`'s transitive intra-package imports never reach `vmem`,
    `kernel`, or `cache*`."""
    import ast
    from pathlib import Path

    pkg_root = Path(planner.__file__).resolve().parent
    queue = [pkg_root / "planner.py"]
    reachable: set[str] = set()

    while queue:
        path = queue.pop()
        rel = path.relative_to(pkg_root).as_posix()
        if rel in reachable or path.name == "__init__.py":
            continue
        reachable.add(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            if not base.is_relative_to(pkg_root):
                continue
            candidates: list[Path] = []
            if node.module:
                mod_path = base / node.module.replace(".", "/")
                candidates.append(mod_path.with_suffix(".py"))
                for alias in node.names:
                    candidates.append(mod_path / f"{alias.name}.py")
            else:
                for alias in node.names:
                    candidates.append(base / f"{alias.name}.py")
            for cand in candidates:
                if cand.is_file() and cand.name != "__init__.py":
                    queue.append(cand)

    assert {"planner.py", "spec.py", "exec_spec.py", "ops.py"} <= reachable
    forbidden = {
        r
        for r in reachable
        if r in {"vmem.py", "kernel.py", "stream.py", "comm.py"}
        or Path(r).name.startswith("cache")
    }
    assert not forbidden, f"planner transitively imports forbidden modules: {forbidden}"
