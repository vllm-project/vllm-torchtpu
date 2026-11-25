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
"""Physics-only legality, HBM capacity, and solver invariant tests."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import pytest

from tests.roofline.conftest import LOCKED_FIXTURE_FILES, load_locked_bundle
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    TrafficScenario,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.memory import (
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    kv_bytes_per_core_parts,
)
from vllm_torchtpu.roofline.modules import Parallel
from vllm_torchtpu.roofline.solver import (
    SolveOptions,
    _candidate_decode_parallels,
    _candidate_prefill_parallels,
    _eval_decode_floor,
    _eval_pp_decode_cycle,
    _SolveContext,
)

if TYPE_CHECKING:
    from vllm_torchtpu.roofline.geometry import ModelGeometry

_MIN_EP_RANKS = 2


def _indep_valid_tp(geometry: ModelGeometry, tp: int, *, has_rec: bool) -> bool:
    """Check independent physical TP divisibility over attention and recurrent heads."""
    if geometry.num_heads > 0 and (
        tp > geometry.num_heads or geometry.num_heads % tp != 0
    ):
        return False
    if (
        geometry.num_kv_heads > 0
        and geometry.kv_lora_rank == 0
        and geometry.num_kv_heads % tp != 0
        and tp % geometry.num_kv_heads != 0
    ):
        return False
    if not (has_rec and geometry.linear_v_heads > 0 and geometry.linear_k_heads > 0):
        return True
    if geometry.linear_v_heads % tp != 0:
        return False
    unique_k_heads = min(geometry.linear_k_heads, tp)
    if geometry.linear_k_heads % unique_k_heads != 0 or tp % unique_k_heads != 0:
        return False
    return (geometry.linear_v_heads // tp) % (
        geometry.linear_k_heads // unique_k_heads
    ) == 0


def _indep_kv_splits(geometry: ModelGeometry, tp: int, *, has_kv: bool) -> list[int]:
    """Return independent physical KV-split divisors for a given TP degree."""
    if not has_kv:
        return [1]
    kv_group = (
        max(1, tp // geometry.num_kv_heads)
        if geometry.num_kv_heads > 0 and geometry.kv_lora_rank == 0
        else tp
    )
    return [d for d in range(1, kv_group + 1) if kv_group % d == 0]


def _indep_moe_opts(
    geometry: ModelGeometry, tp: int, ep_domain: int
) -> list[tuple[str, int]]:
    """Return independent physical (moe_mode, ep) options for one TP and EP domain."""
    if geometry.num_experts == 0:
        return [("tp_slice", 1)]
    opts: list[tuple[str, int]] = []
    if tp == 1 or geometry.moe_inter <= 0 or geometry.moe_inter % tp == 0:
        opts.append(("tp_slice", 1))
    for ep in range(_MIN_EP_RANKS, ep_domain + 1):
        if (
            ep_domain % ep != 0
            or ep > geometry.num_experts
            or geometry.num_experts % ep != 0
        ):
            continue
        if ep < tp and (
            tp % ep != 0
            or (geometry.moe_inter > 0 and geometry.moe_inter % (tp // ep) != 0)
        ):
            continue
        opts.append(("ep", ep))
    return opts


def _independent_physical_layouts(
    geometry: ModelGeometry,
    cores: int,
    *,
    is_decode: bool,
) -> set[tuple[int, int, int, int, int, int, str]]:
    """Independent brute-force enumeration of the physical legality predicate."""
    layouts: set[tuple[int, int, int, int, int, int, str]] = set()
    divs = [d for d in range(1, cores + 1) if cores % d == 0]
    has_kv = geometry.num_kv_heads > 0 or geometry.kv_lora_rank > 0
    has_rec = any(t in ("linear_attention", "kda") for t in geometry.layer_types)

    for tp in divs:
        if not _indep_valid_tp(geometry, tp, has_rec=has_rec):
            continue
        kv_splits = _indep_kv_splits(geometry, tp, has_kv=has_kv)
        rem_tp = cores // tp
        for dp in [d for d in divs if rem_tp % d == 0]:
            rem_dp = rem_tp // dp
            pp_candidates = (
                [rem_dp] if is_decode else [d for d in divs if rem_dp % d == 0]
            )
            for pp in [d for d in pp_candidates if d <= geometry.num_layers]:
                pcp = 1 if is_decode else rem_dp // pp
                moe_opts = _indep_moe_opts(geometry, tp, tp * dp * pcp)
                layouts.update(
                    (tp, dp, ep_val, pp, pcp, kvs, moe_mode)
                    for kvs in kv_splits
                    for moe_mode, ep_val in moe_opts
                )
    return layouts


def _par_key(parallel: Parallel) -> tuple[int, int, int, int, int, int, str]:
    """Extract canonical structural layout key from a Parallel instance."""
    return (
        parallel.tp,
        parallel.dp,
        parallel.ep,
        parallel.pp,
        parallel.pcp,
        parallel.kv_split,
        parallel.moe,
    )


def test_memory_fit_uses_full_hbm_and_logical_bytes() -> None:
    """Verify fit uses 100% physical HBM, logical KV, and no artificial caps."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    q35 = bundle.geometry
    dtypes = bundle.dtypes
    chip = CHIPS["tpu7x"]

    # Default SeqKvLayout uses logical unpadded KV bytes (physical_kv_layout=False).
    assert SeqKvLayout(1, 1024).physical_kv_layout is False

    # Long-context scenario where artificial page-table or reserve caps would block.
    scenario = TrafficScenario(
        name="long_ctx_physics",
        isl=65536,
        osl=1024,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    serving_plan = solve_serving_plan(
        q35,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4, hbm_reserve_gb=0.0),
    )
    assert serving_plan.hbm_reserve_gb == 0.0
    assert serving_plan.candidates
    for point in serving_plan.candidates:
        assert point.decode_pool.hbm_limit_gb == chip.hbm_bytes / 1e9
        parallel = Parallel(
            tp=point.decode_pool.tp,
            dp=point.decode_pool.dp,
            ep=point.decode_pool.ep,
            pp=point.decode_pool.pp,
            pcp=point.decode_pool.pcp,
            kv_split=point.decode_pool.kv_split,
            moe="ep" if point.decode_pool.ep > 1 else "tp_slice",
            moe_dispatch=point.decode_pool.moe_dispatch,
            residual=point.decode_pool.residual,
        )
        c1_b = hbm_footprint_per_core_bytes(
            q35,
            dtypes,
            parallel,
            SeqKvLayout(1, scenario.isl + scenario.osl),
            chip=chip,
        )
        assert c1_b <= chip.hbm_bytes + 1.0


def test_no_legal_layout_dropped_by_estimate() -> None:
    """Verify every HBM-feasible legal layout stays in candidates and is ordered."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    q35 = bundle.geometry
    dtypes = bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]

    # Tight SLO so some layouts meet SLO and others do not; none may be dropped.
    scenario = TrafficScenario(
        name="strict_slo_keep_all",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 8),
        kv_working_set_turns=1.0,
        ttft_slo_ms=500.0,
        tpot_slo_ms=15.0,
    )
    serving_plan = solve_serving_plan(
        q35,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4, sort_by="throughput"),
    )
    expected_fitting = {
        _par_key(parallel)
        for parallel in _candidate_decode_parallels(q35, topology)
        if hbm_footprint_per_core_bytes(
            q35,
            dtypes,
            parallel,
            SeqKvLayout(1, scenario.isl + scenario.osl),
            chip=chip,
        )
        <= float(chip.hbm_bytes)
    }
    assert expected_fitting
    for concurrency in scenario.concurrencies:
        at_c = [
            point
            for point in serving_plan.candidates
            if point.concurrency == concurrency
        ]
        seen = {
            (
                point.decode_pool.tp,
                point.decode_pool.dp,
                point.decode_pool.ep,
                point.decode_pool.pp,
                point.decode_pool.pcp,
                point.decode_pool.kv_split,
                "ep" if point.decode_pool.ep > 1 else "tp_slice",
            )
            for point in at_c
        }
        assert seen == expected_fitting
        assert any(point.is_pareto for point in at_c)
        # Verify SLO-met candidates precede unmet, ordered by throughput descending.
        keys = [(not point.slo_met, -point.token_throughput_per_chip) for point in at_c]
        assert keys == sorted(keys)


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("topo_name", ["2x2x1", "2x2x2"])
def test_basis_models_physics_legality_and_ordering_invariants(
    fixture_name: str,
    topo_name: str,
) -> None:
    """Verify physical legality, HBM fit, per-pool mode, and ordering across basis."""
    bundle = load_locked_bundle(fixture_name)
    geometry = bundle.geometry
    dtypes = bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES[topo_name]

    expected_decode = _independent_physical_layouts(
        geometry, topology.num_cores, is_decode=True
    )
    expected_prefill = _independent_physical_layouts(
        geometry, topology.num_cores, is_decode=False
    )
    dec_parallels = _candidate_decode_parallels(geometry, topology)
    assert all(p.pcp == 1 for p in dec_parallels)
    assert {_par_key(p) for p in dec_parallels} == expected_decode
    assert {
        _par_key(p) for p in _candidate_prefill_parallels(geometry, topology)
    } == expected_prefill

    scenario = TrafficScenario(
        name="sweep_check",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.25,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    splan_tput = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="mixed",
            topologies=(topology,),
            sort_by="throughput",
        ),
    )
    splan_inter = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="mixed",
            topologies=(topology,),
            sort_by="interactivity",
            top_k=3,
        ),
    )
    for point in splan_tput.candidates:
        assert point.decode_pool.hbm_used_gb <= point.decode_pool.hbm_limit_gb + 1e-6
        assert point.est_ttft_ms > 0.0
        assert point.est_tpot_ms > 0.0
    for concurrency in scenario.concurrencies:
        at_c_tput = [
            point_item
            for point_item in splan_tput.candidates
            if point_item.concurrency == concurrency
        ]
        at_c_inter = [
            point_item
            for point_item in splan_inter.candidates
            if point_item.concurrency == concurrency
        ]
        assert len(at_c_inter) == min(len(at_c_tput), 3)
        if at_c_tput:
            assert any(point_item.is_pareto for point_item in at_c_tput)
            assert [
                -point_item.token_throughput_per_chip for point_item in at_c_tput
            ] == sorted(
                -point_item.token_throughput_per_chip for point_item in at_c_tput
            )
            assert [
                -point_item.est_interactivity_tok_s for point_item in at_c_inter
            ] == sorted(
                -point_item.est_interactivity_tok_s for point_item in at_c_inter
            )

    disagg_default = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="disagg", topologies=(topology,)),
    )
    assert disagg_default.candidates == ()
    if splan_tput.candidates:
        assert disagg_default.prefill_pools
        assert disagg_default.decode_pools


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("topo_name", ["2x2x1", "2x2x2", "2x2x4"])
@pytest.mark.parametrize("batch", [1, 2, 5, 32])
def test_pp_decode_cycle_hard_and_serial_invariants(
    fixture_name: str,
    topo_name: str,
    batch: int,
) -> None:
    """Verify PP decode cycle hard floor <= serial estimate across locked models."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES[topo_name]
    scenario = TrafficScenario(
        name="pp-cycle-check",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(batch,),
        kv_working_set_turns=1.0,
    )
    eval_context = _SolveContext(
        geometry=bundle.geometry,
        dtypes=bundle.dtypes,
        chip=chip,
        scenario=scenario,
        mode="mixed",
        max_chips=topology.num_chips,
        max_instances=None,
        hbm_reserve_gb=0.0,
        hbm_limit_b=float(chip.hbm_bytes),
        hbm_limit_gb=float(chip.hbm_bytes) / 1e9,
        effective_accept_len=1.0,
        spec_k=0,
        ctx_full=scenario.isl + scenario.osl,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=scenario.isl,
        n_chunks=max(1, scenario.isl // scenario.prefill_chunk),
        c_max=batch,
        shared_prefix_tokens=0,
        collective_latency_s=0.0,
        sort_by="throughput",
    )
    pp_candidates = [
        p
        for p in _candidate_decode_parallels(bundle.geometry, topology)
        if p.pp > 1 and p.dp == 1 and p.kv_split == 1
    ]
    assert pp_candidates
    parallel = pp_candidates[0]
    serial_s, hard_s, _ = _eval_pp_decode_cycle(
        eval_context, topology, parallel, batch, need_hard_floor=True
    )
    par_pp1 = dc_replace(parallel, pp=1)
    floor_mb = _eval_decode_floor(
        eval_context, topology, par_pp1, max(1, batch // parallel.pp)
    )
    floor_full = _eval_decode_floor(eval_context, topology, par_pp1, batch)
    expected_hard = max(
        floor_mb.hard_floor_s,
        floor_full.hard_floor_s / float(parallel.pp),
    )
    assert hard_s == pytest.approx(expected_hard, rel=1e-12)
    assert hard_s <= serial_s + 1e-12
    if batch == 1:
        stage_c1 = _eval_decode_floor(eval_context, topology, parallel, 1)
        assert serial_s == pytest.approx(
            max(floor_full.serial_estimate_s, stage_c1.serial_estimate_s),
            rel=1e-12,
        )


@pytest.mark.parametrize("fixture_name", ["qwen3_5_397b_fp8.json", "glm_5_2_fp8.json"])
@pytest.mark.parametrize("pp_degree", [1, 2, 4, 8])
def test_kv_bytes_per_core_parts_matches_hbm_breakdown_across_pp(
    fixture_name: str,
    pp_degree: int,
) -> None:
    """Verify kv_bytes_per_core_parts scales identically with PP stage layers."""
    bundle = load_locked_bundle(fixture_name)
    ctx_tokens = 4096
    parallel = Parallel(tp=4, pp=pp_degree)
    per_tok, fixed_seq = kv_bytes_per_core_parts(
        bundle.geometry, bundle.dtypes, parallel, ctx_tokens, physical=False
    )
    breakdown = hbm_breakdown_per_core_bytes(
        bundle.geometry, bundle.dtypes, parallel, SeqKvLayout(1, ctx_tokens)
    )
    assert per_tok * float(ctx_tokens) + fixed_seq == pytest.approx(
        breakdown.kv, rel=1e-9
    )
