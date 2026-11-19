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
"""The default tiles fit VMEM for every model config; the fit step never overflows."""

from __future__ import annotations

import dataclasses

import pytest

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import exec_spec as exec_spec_mod
from vllm_torchtpu.megakernel import vmem
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.profiles import default as default_chip

# Every model in `models/`, tiny and full configs.
MODEL_CONFIGS = {
    "qwen35_full": qwen35.FULL,
    "qwen35_full_ep": qwen35.FULL_EP,
    "qwen35_tiny": qwen35.TINY,
    "qwen35_tiny_ep": qwen35.TINY_EP,
}
MODEL_SPECS = {"qwen35": qwen35.qwen35_spec}


def _buckets() -> list[tuple[int, int]]:
    return [(1, B) for B in range(1, 65)] + [(4, B) for B in range(4, 65, 4)]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_default_tiles_fit_vmem_for_every_model_config(name: str, paged: bool):
    """`fit_default_tiles` gives an in-budget plan on every bucket."""
    g = MODEL_CONFIGS[name]
    spec = MODEL_SPECS[name.split("_")[0]](g)
    shrunk = False
    for T, B in _buckets():
        ex = mk.ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=4096,
            cache=mk.CacheExec(paged_kv=paged, num_pages=65536 if paged else 4096),
        )
        fitted = vmem.fit_default_tiles(spec, ex)
        t = fitted.tiles
        assert t is not None
        default = mk.default_tiles(B, R=8)
        # Only the ring depths move, and only downwards.
        assert dataclasses.replace(t, nbd=default.nbd, nbe=default.nbe) == default
        assert t.nbd <= default.nbd and t.nbe <= default.nbe
        shrunk |= (t.nbd, t.nbe) != (default.nbd, default.nbe)
        core_plan, _ = vmem.build_scratch(spec, fitted)  # raises if over budget
        assert core_plan.vmem_bytes <= t.vmem_mib * (1 << 20)
    if name.startswith("qwen35_full"):
        assert shrunk  # the 64-row buckets need shallower rings


def test_fit_leaves_explicit_tiles_alone():
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    t = mk.default_tiles(64, R=8)
    ex = mk.ExecSpec(B=64, T=1, R=8, ctx=4096, tiles=t)
    assert vmem.fit_default_tiles(spec, ex) is ex
    with pytest.raises(mk.MegakernelConfigError, match="VMEM"):
        vmem.build_scratch(spec, ex)  # unfitted default rings overflow at B=64


def test_kernel_build_fits_unset_tiles():
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    ex = mk.ExecSpec(B=64, T=1, R=8, ctx=4096)
    _, info = mk.build(spec, ex)
    assert info["exec"].tiles == vmem.fit_default_tiles(spec, ex).tiles
    assert info["vmem_scratch_bytes"] <= info["exec"].tiles.vmem_mib * (1 << 20)


def test_fit_sizes_default_tiles_for_the_device_chip(monkeypatch):
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    ex = mk.ExecSpec(B=8, T=1, R=8, ctx=4096)
    chip = default_chip.ChipBudget(name="test_chip", vmem_mib=62, smem_mib=2)
    monkeypatch.setitem(
        default_chip.CHIP_BUDGETS, default_chip._first_device_kind(), chip
    )
    assert vmem.fit_default_tiles(spec, ex).tiles.smem_mib == 2


def test_plan_and_build_scratch_fit_unset_tiles_like_build():
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    ex = mk.ExecSpec(B=64, T=1, R=8, ctx=4096)
    fitted = vmem.fit_default_tiles(spec, ex)
    assert fitted.tiles != mk.default_tiles(64, R=8)  # the fit shrinks a ring
    core_plan, _ = vmem.build_scratch(spec, ex)  # unfitted rings would overflow
    assert core_plan.vmem_bytes == vmem.build_scratch(spec, fitted)[0].vmem_bytes
    assert mk.planner.plan(spec, ex) == mk.planner.plan(spec, fitted)


def test_fit_raises_instead_of_returning_an_over_budget_plan(monkeypatch):
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    tight = dataclasses.replace(mk.default_tiles(4), vmem_mib=4)
    monkeypatch.setattr(exec_spec_mod, "default_tiles", lambda B, **kw: tight)
    ex = mk.ExecSpec(B=4, T=1, R=8, ctx=4096)
    with pytest.raises(mk.MegakernelConfigError, match="do not fit"):
        vmem.fit_default_tiles(spec, ex)
    with pytest.raises(mk.MegakernelConfigError, match="do not fit"):
        mk.build(spec, ex)
