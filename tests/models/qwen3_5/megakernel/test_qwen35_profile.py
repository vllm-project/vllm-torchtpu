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
"""Unit tests for the Qwen3.5 baseline tile rule (`models.qwen35_profile`)."""

from __future__ import annotations

import dataclasses

import pytest

from tests.megakernel.test_moe_routing_ring import _simulate_expert_ring_no_collision
from vllm_torchtpu.megakernel import CacheExec, vmem
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.models.qwen3_5.megakernel.model import (
    Qwen35Geometry,
    qwen35_spec,
)
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import (
    DECODE_TILES,
    baseline_tiles,
)
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.candidate30 import (
    candidate30_tiles,
)


@pytest.mark.parametrize("R", [1, 8])
@pytest.mark.parametrize("moe_tp", [False, True])
def test_qwen35_baseline_tiles_ignore_moe_tp(R: int, moe_tp: bool):
    for B in range(1, 65):
        t = baseline_tiles(B, moe_tp=moe_tp, R=R)
        assert t == baseline_tiles(B, R=R)  # no moe_tp dependence
        gqa, moe = t.module("gqa"), t.module("moe")
        assert gqa.kv_sub == 0 and not moe.route_vec and not t.dense_acc_pair
        assert not moe.pipe_ep and moe.pipe  # PIPE stays on (TP single tile)
    assert Tiles(hier=1, mb8=2).with_modules({"gqa": {"pack": 1}}) == DECODE_TILES


@pytest.mark.parametrize("R", [2, 4])
def test_qwen35_baseline_non_v7x8_ranks_use_flat_ring(R: int):
    for B in (2, 4, 8, 16, 32, 64):
        assert baseline_tiles(B, R=R).hier == 0


def test_moe_deployment_ring_invariants():
    """Verify MoE ring schedules and PIPE / PIPE_EP / LAZY_MOE across real profiles."""
    seen_pipe = False
    seen_pipe_ep = False
    seen_lazy = False
    sample_plain_mp = None
    sample_pipe_mp = None
    sample_pipe_ep_mp = None
    for moe_tp, e_slots in ((False, 0), (False, 72), (True, 0)):
        g = Qwen35Geometry(L=4, moe_tp=moe_tp, e_slots=e_slots)
        ms = qwen35_spec(g)
        for T in (1, 4):
            for B in (1, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64):
                if B % T != 0:
                    continue
                for tile_fn in (baseline_tiles, candidate30_tiles):
                    tiles = tile_fn(B, moe_tp=moe_tp, R=8)
                    ex = ExecSpec(
                        R=8,
                        B=B,
                        T=T,
                        tiles=tiles,
                        cache=CacheExec(paged_kv=True),
                    )
                    cp, _ = vmem.build_scratch(ms, ex)
                    mp = cp.module("moe")
                    assert mp.NSS >= 3 + (mp.NBE - 1) // mp.TE
                    assert mp.NBE >= 2
                    assert not (mp.PIPE and mp.PIPE_EP)
                    if mp.PIPE:
                        assert mp.T13 == 1 and mp.T2 == 1
                        sample_pipe_mp = mp
                    elif mp.PIPE_EP:
                        assert mp.T13 == 2 * mp.T2
                        assert mp.NBE >= mp.TE and mp.NBE % mp.TE == 0
                        sample_pipe_ep_mp = mp
                    else:
                        sample_plain_mp = mp
                    if mp.LAZY_MOE:
                        assert cp.plan.ex.resolved_tiles.nbd >= mp.N_MOE
                    seen_pipe = seen_pipe or mp.PIPE
                    seen_pipe_ep = seen_pipe_ep or mp.PIPE_EP
                    seen_lazy = seen_lazy or mp.LAZY_MOE
                    _simulate_expert_ring_no_collision(mp, n_hit=8)
    assert seen_pipe and seen_pipe_ep and seen_lazy, (
        seen_pipe,
        seen_pipe_ep,
        seen_lazy,
    )
    assert sample_plain_mp is not None
    assert sample_pipe_mp is not None
    assert sample_pipe_ep_mp is not None
    # Negative self-tests: mutating NSS or NBE below the schedule requirements
    # or breaking PIPE_EP's NBE % TE == 0 invariant must fail the simulator.
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_plain_mp, NSS=sample_plain_mp.NSS - 2),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_mp, NSS=sample_pipe_mp.NSS - 2),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, NSS=sample_pipe_ep_mp.NSS - 1),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, T13=2, T2=1, TE=3, NBE=12, NSS=3),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="not prefetched before body_ep"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, NBE=sample_pipe_ep_mp.TE - 1),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="PIPE_EP requires"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(
                sample_pipe_ep_mp, NBE=sample_pipe_ep_mp.TE + 1, NSS=10
            ),
            n_hit=8,
        )
