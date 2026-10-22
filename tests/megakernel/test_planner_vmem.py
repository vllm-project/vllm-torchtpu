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
"""Unit tests for `planner` and `vmem`."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest
from conftest import (
    TINY_TILES,
    FakeAltMixerBlock,
    FakeGeometry,
    FakeMixerBlock,
    make_fake_model_spec,
)
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import block_options, planner, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import F32, FP8
from vllm_torchtpu.megakernel.spec import (
    Block,
    LayerType,
    ModelSpec,
    ScratchReq,
    SmallField,
    SwitchSlot,
    WeightDecl,
)


def test_planner_is_load_bearing_single_branch(fake_ex: ExecSpec):
    """Verify `planner.plan` derives only weights/smalls/states from active blocks."""
    ms = make_fake_model_spec(L=4, att_every=1, H=256)
    p = planner.plan(ms, fake_ex)

    # Only FakeMixerBlock and FakeFFNBlock are in the spec -> no GDN declarations
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
    assert p.block_layer_counts["fake_gdn"] == 3
    assert p.block_layer_counts["fake_gqa"] == 1
    assert p.block_layer_counts["fake_moe"] == 4

    wsh = planner.weight_shapes(ms, fake_ex, p)
    assert wsh["lin_in"][0][0] == 3
    assert wsh["att_in"][0][0] == 1
    assert wsh["router"][0][0] == 4

    # Shared smalls field `in_s` takes max width across branches (NAI=512, NLI=640)
    _, in_s_w = p.smalls_layout["in_s"]
    assert in_s_w == 640

    # >2 branches in a SwitchSlot raises AssertionError at plan time
    class ThirdBranchBlock(Block):
        name = "third"

    bad_sw = SwitchSlot(
        "mixer",
        branches=(FakeAltMixerBlock(), FakeMixerBlock(), ThirdBranchBlock()),
        branch_counts=(1, 1, 2),
    )
    bad_ms = ModelSpec(
        name="bad_3branch",
        config=FakeGeometry(L=4, H=256),
        layer_type=LayerType("l", slots=(bad_sw,)),
    )
    with pytest.raises(AssertionError, match="supports at most 2 active branches"):
        planner.plan(bad_ms, fake_ex)


def test_planner_custom_block_extension(fake_ex: ExecSpec):
    """Verify adding a custom `Block` updates `plan` without editing `planner.py`."""

    class CustomMLPBlock(Block):
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

        def scratch(self, g: FakeGeometry, ex: ExecSpec, plan) -> dict[str, object]:
            return {"mlp_tmp": pltpu.VMEM((ex.MR, 512), F32)}

    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        name="custom_model",
        config=g,
        layer_type=LayerType(
            "custom_layer", slots=(FakeMixerBlock(), CustomMLPBlock())
        ),
    )
    p = planner.plan(ms, fake_ex)
    assert p.static_kinds == ("att_in", "att_out", "mlp_up", "mlp_down")
    assert p.dynamic_kinds == ()
    assert "mlp_scale" in p.smalls_layout

    cp, scr = vmem.build_scratch(ms, fake_ex, p)
    assert "mlp_tmp" in scr
    assert "epool" not in scr and "esem" not in scr
    assert cp.NACC >= 512 and cp.KX2 >= 512


def test_vmem_phase_overlay_and_conflict(fake_ex: ExecSpec):
    """Verify `ScratchReq.phase` overlays non-persistent buffers and checks budget."""
    ms_hybrid = make_fake_model_spec(L=4, att_every=4, H=256)
    cp, scr = vmem.build_scratch(ms_hybrid, fake_ex)

    # `branch_tmp` is declared by both FakeAltMixerBlock and FakeMixerBlock -> overlaid!
    assert "branch_tmp" in scr
    assert scr["branch_tmp"].shape == (fake_ex.MR * 2, 128)
    assert cp.phase_map["branch_tmp"] == "branch:gdn|branch:gqa"

    # Overlaying two SMEM phase buffers preserves SMEM memory space
    smem_scratch: dict[str, object] = {}
    smem_phases: dict[str, str] = {}
    vmem._merge_scratch(
        smem_scratch,
        smem_phases,
        [
            ScratchReq("smem_tmp", pltpu.SMEM((4,), jnp.int32), phase="branch:a"),
            ScratchReq("smem_tmp", pltpu.SMEM((8,), jnp.int32), phase="branch:b"),
        ],
    )
    assert smem_scratch["smem_tmp"].shape == (8,)
    assert smem_scratch["smem_tmp"].memory_space == pltpu.SMEM

    # Persistent name collision must raise MegakernelConfigError
    class CollidingBlock(Block):
        name = "colliding"

        def scratch(self, g, ex, plan):
            return [ScratchReq("kbuf", pltpu.VMEM((16, 128), F32), phase="persistent")]

    bad_ms = ModelSpec(
        name="bad",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType("bad_layer", slots=(FakeMixerBlock(), CollidingBlock())),
    )
    with pytest.raises(
        MegakernelConfigError, match="Conflicting scratch buffer 'kbuf'"
    ):
        vmem.build_scratch(bad_ms, fake_ex)

    # Exceeding tiles.vmem_mib raises MegakernelConfigError with per-buffer breakdown
    tiny_budget_ex = dataclasses.replace(
        fake_ex, tiles=dataclasses.replace(TINY_TILES, vmem_mib=1)
    )
    with pytest.raises(MegakernelConfigError, match="exceeds tiles.vmem_mib budget"):
        vmem.build_scratch(ms_hybrid, tiny_budget_ex)


def test_unsupported_stream_ring_is_config_error(
    fake_ex: ExecSpec, monkeypatch: pytest.MonkeyPatch
):
    """Unknown `WeightDecl.stream_ring` raises `MegakernelConfigError` at plan time."""
    orig = FakeMixerBlock.weights

    def _weights(
        self: FakeMixerBlock, g: FakeGeometry, ex: ExecSpec
    ) -> list[WeightDecl]:
        ws = orig(self, g, ex)
        return [dataclasses.replace(ws[0], stream_ring="bogus"), *ws[1:]]  # type: ignore[arg-type]

    monkeypatch.setattr(FakeMixerBlock, "weights", _weights)
    with pytest.raises(MegakernelConfigError, match="Unsupported stream_ring"):
        planner.plan(make_fake_model_spec(), fake_ex)


def test_plan_fit_vmem_false_and_inactive_debug_block(
    fake_ex: ExecSpec, monkeypatch: pytest.MonkeyPatch
):
    """`plan(..., fit_vmem=False)` and `weight_shapes` skip VMEM fitting;
    `plan` rejects `DebugOptions.blocks` targeting a block not in `model_spec`."""
    import jax

    from vllm_torchtpu.megakernel import exec_spec

    ms = make_fake_model_spec(L=4, att_every=1, H=256)
    ex_unset = dataclasses.replace(fake_ex, tiles=None)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("jax.devices() must not be called when fit_vmem=False")
        ),
    )
    p = planner.plan(ms, ex_unset, fit_vmem=False)
    assert p.ex.tiles is None
    wsh = planner.weight_shapes(ms, ex_unset)
    assert set(wsh) == set(p.w_names)

    @dataclasses.dataclass(frozen=True)
    class _InactiveDebug:
        flag: bool = False

    monkeypatch.setitem(block_options._DEBUG, "fake_gdn", _InactiveDebug)
    ex_bad_dbg = dataclasses.replace(
        fake_ex,
        debug=exec_spec.DebugOptions(blocks={"fake_gdn": _InactiveDebug(flag=True)}),
    )
    with pytest.raises(
        MegakernelConfigError, match="unknown or inactive block.*fake_gdn"
    ):
        planner.plan(ms, ex_bad_dbg)
