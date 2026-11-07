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
"""Unit tests for `vmem`."""

from __future__ import annotations

import dataclasses
import logging

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeGeometry,
    FakeMixerModule,
    make_fake_model_spec,
)
from vllm_torchtpu.megakernel import planner, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import F32, FP8
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    ScratchReq,
    SmallField,
    WeightDecl,
)


def test_vmem_phase_overlay_and_conflict(fake_ex: ExecSpec):
    """Verify `ScratchReq.phase` overlays, budgets, `build_scratch`, `fitted_plan`."""
    ms_hybrid = make_fake_model_spec(L=4, att_every=4, H=256)
    cp, scr = vmem.build_scratch(ms_hybrid, fake_ex)

    # `branch_tmp` is declared by FakeAltMixerModule and FakeMixerModule -> overlaid!
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
    class CollidingModule(Module):
        name = "colliding"

        def scratch(self, g, ex, plan):
            return [ScratchReq("kbuf", pltpu.VMEM((16, 128), F32), phase="persistent")]

    bad_ms = ModelSpec(
        name="bad",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType("bad_layer", slots=(FakeMixerModule(), CollidingModule())),
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

    # Custom module scratch & static-only stream layout (no epool/esem) + fitted_plan
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

        def scratch(self, g: FakeGeometry, ex: ExecSpec, plan) -> dict[str, object]:
            return {"mlp_tmp": pltpu.VMEM((ex.MR, 512), F32)}

    custom_ms = ModelSpec(
        name="custom_model",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType(
            "custom_layer", slots=(FakeMixerModule(), CustomMLPModule())
        ),
    )
    p = planner.plan(custom_ms, fake_ex)
    cp_custom, scr_custom = vmem.build_scratch(custom_ms, fake_ex, p)
    assert "mlp_tmp" in scr_custom
    assert "epool" not in scr_custom and "esem" not in scr_custom
    assert cp_custom.NACC >= 512 and cp_custom.KX2 >= 512

    ex_unset = dataclasses.replace(fake_ex, tiles=None)
    fp = vmem.fitted_plan(custom_ms, ex_unset)
    assert fp.ex.tiles is not None
    cp_reused, _ = vmem.build_scratch(custom_ms, ex_unset, fp)
    assert cp_reused.plan == fp

    # 1D VMEM buffer is counted with (1, n) tiled storage_size; unknown scratch raises.
    assert vmem._vmem_storage_bytes((128,), F32) == 8 * 128 * 4
    with pytest.raises(MegakernelConfigError, match="Unsupported scratch spec"):
        vmem._merge_scratch({}, {}, [ScratchReq("bad", object(), phase="persistent")])


def test_fit_default_tiles_on_fake_model_spec(monkeypatch, caplog):
    """`fit_default_tiles` shrinks ring depths into budget and logs once."""
    from vllm_torchtpu.megakernel import exec_spec as exec_spec_mod

    monkeypatch.setattr(vmem, "_LOGGED_FIT_KEYS", set())
    ms_large = make_fake_model_spec(L=60, att_every=4, H=4096, moe_tp=False)
    default_64 = dataclasses.replace(exec_spec_mod.default_tiles(64, R=2), vmem_mib=24)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: default_64)
    ex_64 = ExecSpec(B=64, T=1, R=2, ctx=4096)
    with caplog.at_level(logging.INFO, logger=vmem.__name__):
        fitted = vmem.fit_default_tiles(ms_large, ex_64)
        _ = vmem.fit_default_tiles(ms_large, ex_64)
    assert len(caplog.records) == 1
    assert "fit_default_tiles shrunk ring depths" in caplog.records[0].getMessage()
    assert fitted.tiles is not None
    assert (fitted.tiles.nbd, fitted.tiles.nbe) != (default_64.nbd, default_64.nbe)
    assert fitted.tiles.nbd <= default_64.nbd and fitted.tiles.nbe <= default_64.nbe
    cp, _ = vmem.build_scratch(ms_large, fitted)
    assert cp.vmem_bytes <= vmem.fit_vmem_budget_bytes(fitted.tiles)

    # Every bucket of the same model gets the same `nbe` (the minimum over its
    # buckets), while `nbd` can stay deeper on smaller buckets.
    fitted_4 = vmem.fit_default_tiles(ms_large, ExecSpec(B=4, T=1, R=2, ctx=4096))
    assert fitted_4.tiles is not None
    assert fitted_4.tiles.nbe == fitted.tiles.nbe
    assert fitted_4.tiles.nbd >= fitted.tiles.nbd

    ex_explicit = ExecSpec(B=64, T=1, R=2, ctx=4096, tiles=default_64)
    assert vmem.fit_default_tiles(ms_large, ex_explicit) is ex_explicit

    tight = dataclasses.replace(default_64, vmem_mib=4)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: tight)
    with pytest.raises(MegakernelConfigError, match="do not fit"):
        vmem.fit_default_tiles(ms_large, ExecSpec(B=4, T=1, R=2, ctx=4096))
