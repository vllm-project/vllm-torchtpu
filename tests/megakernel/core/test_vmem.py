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
from typing import Any

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
    make_fake_model_spec,
)
from vllm_torchtpu.megakernel.core import planner, vmem
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.ops import F32, FP8
from vllm_torchtpu.megakernel.core.spec import (
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
    smem_scratch: dict[str, Any] = {}
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

    # 1D VMEM buffer is counted with (1, n) tiled storage_size; unmarked 3D bf16
    # with shape[-2] == 2 is NOT guessed packed, while packed=True folds sublane words;
    # unknown scratch raises.
    assert vmem._vmem_storage_bytes((128,), F32) == 8 * 128 * 4
    assert vmem._vmem_storage_bytes((3, 2, 128), jnp.bfloat16) == 3 * 16 * 128 * 2
    assert vmem._vmem_storage_bytes((2, 64, 4, 128), FP8, packed=True) == 2 * 256 * 128
    with pytest.raises(MegakernelConfigError, match="Unsupported scratch spec"):
        vmem._merge_scratch({}, {}, [ScratchReq("bad", object(), phase="persistent")])


def test_packed_buffer_counts_agree_in_vmem_bytes_breakdown_and_ring_bytes(
    fake_ex: ExecSpec,
):
    """`vmem_bytes`, breakdown, and `_ring_bytes` agree on packed buffers."""
    import types

    class PackedKVModule(FakeMixerModule):
        name = "packed_kv_gqa"

        def plan_params(self, g: FakeGeometry, ex: ExecSpec, plan):
            del g, ex, plan
            return types.SimpleNamespace(KVE=True, BLK=1024)

        def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
            del ex, plan
            return [ScratchReq("kvbuf", pltpu.VMEM((2, 2, g.HD // 4, 4, 1024), FP8))]

    kve_ms = ModelSpec(
        name="packed_kv_model",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType(
            "packed_kv_layer", slots=(PackedKVModule(), FakeFFNModule())
        ),
    )
    kve_tiles = dataclasses.replace(TINY_TILES, nbe=2, vmem_mib=32)
    ex_kve = dataclasses.replace(fake_ex, interpret=True, tiles=kve_tiles)
    cp_kve, _ = vmem.build_scratch(kve_ms, ex_kve)

    packed_sh = (2, 2, 32, 4, 1024)
    assert cp_kve.vmem_shapes["epool"] == (packed_sh, FP8)
    assert cp_kve.vmem_shapes["kvbuf"] == (packed_sh, FP8)

    packed_sz = 2 * 2 * (32 * 4) * 1024  # 524,288 B = 0.50 MiB
    unpacked_sz = vmem._vmem_storage_bytes(packed_sh, FP8, packed=False)  # 4,194,304 B
    assert vmem._vmem_buffer_bytes("epool", packed_sh, FP8) == packed_sz
    assert vmem._vmem_buffer_bytes("kvbuf", packed_sh, FP8) == packed_sz
    assert unpacked_sz == 8 * packed_sz

    # 1. `vmem_bytes` uses the packed size for both `epool` and `kvbuf`.
    other_bytes = sum(
        vmem._vmem_buffer_bytes(k, sh, dt)
        for k, (sh, dt) in cp_kve.vmem_shapes.items()
        if k not in ("epool", "kvbuf")
    )
    assert cp_kve.vmem_bytes == other_bytes + 2 * packed_sz

    # 2. `_ring_bytes` uses the same packed size for `epool`.
    _, expert_bytes = vmem._ring_bytes(cp_kve)
    ering_bytes = sum(
        vmem._vmem_buffer_bytes(k, sh, dt)
        for k, (sh, dt) in cp_kve.vmem_shapes.items()
        if k.startswith("ering_")
    )
    assert expert_bytes - ering_bytes == packed_sz

    # 3. Over-budget breakdown reports 0.50 MiB (not 4.00 MiB unpacked) for both.
    ex_tight = dataclasses.replace(
        ex_kve, tiles=dataclasses.replace(kve_tiles, vmem_mib=1)
    )
    with pytest.raises(MegakernelConfigError) as exc_info:
        vmem.build_scratch(kve_ms, ex_tight)
    msg = str(exc_info.value)
    assert "epool=0.50MiB (2, 2, 32, 4, 1024) float8_e4m3fn" in msg
    assert "kvbuf=0.50MiB (2, 2, 32, 4, 1024) float8_e4m3fn" in msg


def test_fit_default_tiles_on_fake_model_spec(monkeypatch, caplog):
    """`fit_default_tiles` shrinks ring depths into budget and logs once."""
    from vllm_torchtpu.megakernel.core import exec_spec as exec_spec_mod

    monkeypatch.setattr(vmem, "_LOGGED_FIT_KEYS", set())
    ms_large = make_fake_model_spec(L=60, att_every=4, H=4096, moe_tp=False)
    default_32 = dataclasses.replace(exec_spec_mod.default_tiles(64, R=2), vmem_mib=32)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: default_32)
    ex_64 = ExecSpec(B=64, T=1, R=2, ctx=4096)
    with caplog.at_level(logging.INFO, logger=vmem.__name__):
        fitted = vmem.fit_default_tiles(ms_large, ex_64)
        _ = vmem.fit_default_tiles(ms_large, ex_64)
    assert len(caplog.records) == 1
    assert "fit_default_tiles shrunk ring depths" in caplog.records[0].getMessage()
    assert fitted.tiles is not None
    assert (fitted.tiles.nbd, fitted.tiles.nbe) != (default_32.nbd, default_32.nbe)
    assert fitted.tiles.nbd <= default_32.nbd and fitted.tiles.nbe <= default_32.nbe
    cp, _ = vmem.build_scratch(ms_large, fitted)
    assert cp.vmem_bytes <= vmem.fit_vmem_budget_bytes(fitted.tiles)

    # Every bucket of the same model — including small decode (B=4, T=1) and
    # max-batch MTP verify (B=128 > 64, T=4 > 2) — gets the exact same `nbe`,
    # while `nbd` can stay deeper on smaller buckets.
    fitted_4 = vmem.fit_default_tiles(ms_large, ExecSpec(B=4, T=1, R=2, ctx=4096))
    fitted_128_t4 = vmem.fit_default_tiles(
        ms_large, ExecSpec(B=128, T=4, R=2, ctx=4096)
    )
    assert fitted_4.tiles is not None and fitted_128_t4.tiles is not None
    assert fitted_4.tiles.nbe == fitted.tiles.nbe == fitted_128_t4.tiles.nbe
    assert fitted_4.tiles.nbd >= fitted.tiles.nbd >= fitted_128_t4.tiles.nbd

    # Even when the worst-case B=128 probe does not fit at (nbd=2, nbe=2),
    # `model_nbe` is still pinned to `MIN_RING_DEPTH` across all buckets that fit,
    # and `shrink_nbe=False` never lowers `nbe`.
    default_24 = dataclasses.replace(default_32, vmem_mib=24)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: default_24)
    fit_b4_24 = vmem.fit_default_tiles(ms_large, ExecSpec(B=4, T=1, R=2, ctx=4096))
    fit_b64_24 = vmem.fit_default_tiles(ms_large, ExecSpec(B=64, T=4, R=2, ctx=4096))
    assert fit_b4_24.tiles is not None and fit_b64_24.tiles is not None
    assert fit_b4_24.tiles.nbe == fit_b64_24.tiles.nbe == vmem.MIN_RING_DEPTH
    t_no_shrink, _, t_last_no_shrink = vmem._fit_ring_depths(
        ms_large,
        ExecSpec(B=64, T=1, R=2, ctx=4096),
        default_24,
        vmem.fit_vmem_budget_bytes(default_24),
        shrink_nbe=False,
    )
    assert t_no_shrink is None
    assert t_last_no_shrink.nbe == default_24.nbe
    assert t_last_no_shrink.nbd == vmem.MIN_RING_DEPTH

    ex_explicit = ExecSpec(B=64, T=1, R=2, ctx=4096, tiles=default_32)
    assert vmem.fit_default_tiles(ms_large, ex_explicit) is ex_explicit

    tight = dataclasses.replace(default_32, vmem_mib=8)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: tight)
    with pytest.raises(
        MegakernelConfigError,
        match=(
            r"do not fit the 2\.00 MiB VMEM scratch budget "
            r"\(8 MiB limit - 6 MiB headroom\)"
        ),
    ):
        vmem.fit_default_tiles(ms_large, ExecSpec(B=4, T=1, R=2, ctx=4096))

    # A TypeError raised inside build_scratch propagates rather than being
    # caught by the cache-key hashability check.
    def _bad_build_scratch(*args, **kwargs):
        raise TypeError("internal build_scratch bug")

    monkeypatch.setattr(vmem, "build_scratch", _bad_build_scratch)
    vmem._fit_ring_depths_cached.cache_clear()
    with pytest.raises(TypeError, match="internal build_scratch bug"):
        vmem._fit_ring_depths(ms_large, ex_64, default_32, 1 << 20, shrink_nbe=False)
