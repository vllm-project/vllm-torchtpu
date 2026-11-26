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

import json
import math
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.roofline import regen_golden, render_docs_tables
from tests.roofline.conftest import (
    LOCKED_FIXTURE_FILES,
    load_locked_bundle,
    load_locked_config,
)
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    FloorOptions,
    StepShape,
    TrafficScenario,
    bound_tpot_ms,
    bound_ttft_ms,
    build_solve_payload,
    dtypes_from_hf_config,
    floor_throughput_per_chip,
    from_hf_config,
    pool_capacity_bound,
    request_rate_bound,
    solve_serving_plan,
    step_floor,
)
from vllm_torchtpu.roofline.constants import SLO_VERDICT_RANK
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
_FULL_2X2X1_CORES = 8


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
        keys = [
            (-SLO_VERDICT_RANK[point.slo_verdict], -point.token_throughput_per_chip)
            for point in at_c
        ]
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
    payload_inter = build_solve_payload(splan_inter, chip=chip, top_k=3)
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
        payload_at_c_inter = [
            point_item
            for point_item in payload_inter["candidates"]
            if point_item["concurrency"] == concurrency
        ]
        assert len(at_c_inter) == len(at_c_tput)
        assert len(payload_at_c_inter) == min(len(at_c_tput), 3)
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


def test_bound_counts_decode_steps_after_first_token() -> None:
    """At osl=1 prefill emits the only output token so decode steps are zero."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    pref_par = Parallel(tp=8)
    dec_par = Parallel(tp=8)
    scenario = TrafficScenario(
        name="osl1-check",
        isl=2048,
        osl=1,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    b_ttft_s = (
        bound_ttft_ms(
            geometry,
            dtypes,
            chip,
            scenario,
            prefill_par=pref_par,
            prefill_topo=topology,
        )
        / 1000.0
    )
    rate_bound = request_rate_bound(
        geometry,
        dtypes,
        chip,
        scenario,
        decode_par=dec_par,
        decode_topo=topology,
        replica_batch_cap=1,
        concurrency=1,
    )
    assert rate_bound == pytest.approx(1.0 / b_ttft_s, rel=1e-6)


def test_capacity_ceiling_uses_shared_prefix_batch() -> None:
    """Shared-prefix batch cap raises ceiling above unshared HBM batch cap."""
    config = load_locked_config("qwen3_5_397b_fp8.json")
    geometry = from_hf_config(config, model_ref="Qwen/Qwen3.5-397B-A17B-FP8")
    dtypes = dtypes_from_hf_config(config)
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    scen_h90 = TrafficScenario(
        name="prefix-h90",
        isl=32768,
        osl=2048,
        prefix_hit_rate=0.90,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(64,),
        kv_working_set_turns=1.0,
    )
    p90 = solve_serving_plan(
        geometry, dtypes, chip, scen_h90, SolveOptions(mode="mixed", max_chips=4)
    )
    pt90 = next(
        point
        for point in p90.candidates
        if point.decode_pool.tp == 1
        and point.decode_pool.dp == _FULL_2X2X1_CORES
        and point.decode_pool.ep == _FULL_2X2X1_CORES
        and point.decode_pool.kv_split == 1
    )
    parallel = Parallel(
        tp=pt90.decode_pool.tp,
        dp=pt90.decode_pool.dp,
        ep=pt90.decode_pool.ep,
        kv_split=pt90.decode_pool.kv_split,
        moe="ep",
        moe_dispatch=pt90.decode_pool.moe_dispatch,
        ranks_per_chip=topology.cores_per_chip,
    )
    w_base = hbm_footprint_per_core_bytes(
        geometry, dtypes, parallel, SeqKvLayout(0, 0), chip=chip
    )
    c1 = hbm_footprint_per_core_bytes(
        geometry,
        dtypes,
        parallel,
        SeqKvLayout(1, scen_h90.isl + scen_h90.osl),
        chip=chip,
    )
    b_unshared = max(1, int((float(chip.hbm_bytes) - w_base) // (c1 - w_base)))
    unshared_ceil = floor_throughput_per_chip(
        geometry,
        dtypes,
        chip,
        scen_h90,
        decode_par=parallel,
        decode_topo=topology,
        replica_batch_cap=b_unshared,
        concurrency=64,
    )
    assert pt90.floor_throughput_per_chip > unshared_ceil * 1.15


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_disagg_prefill_batched_soundness_and_dp_scaling(fixture_name: str) -> None:
    """Disagg prefill ceiling stays sound for batched prompts and scales with dp."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    pref_dp1 = Parallel(tp=2, dp=1)
    pref_dp2 = Parallel(tp=2, dp=2)
    dec_par = Parallel(tp=8)
    for isl, hit in ((512, 0.0), (16384, 0.98)):
        scenario = TrafficScenario(
            name=f"disagg-pref-{isl}",
            isl=isl,
            osl=1,
            prefix_hit_rate=hit,
            accept_len=1.0,
            think_time_s=None,
            concurrencies=(16,),
            kv_working_set_turns=1.0,
        )
        capacity_dp1 = pool_capacity_bound(
            geometry,
            dtypes,
            chip,
            scenario,
            prefill_par=pref_dp1,
            prefill_topo=topology,
            decode_par=dec_par,
            decode_topo=topology,
            prefill_instances=1,
            decode_instances=1,
            replica_batch_cap=16,
        )
        capacity_dp2 = pool_capacity_bound(
            geometry,
            dtypes,
            chip,
            scenario,
            prefill_par=pref_dp2,
            prefill_topo=topology,
            decode_par=dec_par,
            decode_topo=topology,
            prefill_instances=1,
            decode_instances=1,
            replica_batch_cap=16,
        )
        # 8 prompts packed in one prefill step share weight reads once.
        batched_floor = step_floor(
            geometry,
            chip,
            pref_dp1,
            StepShape(
                batch=0,
                rows_per_seq=1,
                context=0,
                prefill_tokens=8 * isl,
                prefix_hit_rate=hit,
            ),
            FloorOptions(dtypes=dtypes, topology=topology),
        )
        batched_rate = 8.0 / batched_floor.hard_floor_s
        assert math.isfinite(capacity_dp1)
        assert capacity_dp1 >= batched_rate * (1.0 - 1e-5)
        assert capacity_dp2 == pytest.approx(2.0 * capacity_dp1, rel=1e-6)


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_mixed_osl_monotonic_and_near_full_hit_finite(fixture_name: str) -> None:
    """Mixed capacity is non-increasing in osl and finite at near-full prefix hit."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    parallel = Parallel(tp=8, ep=8, moe="ep")
    for accept_len in (1.0, 2.0, 3.0):
        caps = [
            pool_capacity_bound(
                geometry,
                dtypes,
                chip,
                TrafficScenario(
                    name=f"mixed-osl-{osl}",
                    isl=2048,
                    osl=osl,
                    prefix_hit_rate=0.5,
                    accept_len=accept_len,
                    think_time_s=None,
                    concurrencies=(16,),
                    kv_working_set_turns=1.0,
                ),
                decode_par=parallel,
                decode_topo=topology,
                replica_batch_cap=16,
            )
            for osl in (1, 2, 16)
        ]
        assert caps[0] >= caps[1] - 1e-6
        assert caps[1] >= caps[2] - 1e-6

    near_full_cap = pool_capacity_bound(
        geometry,
        dtypes,
        chip,
        TrafficScenario(
            name="near-full-hit",
            isl=4096,
            osl=1,
            prefix_hit_rate=0.9999,
            accept_len=1.0,
            think_time_s=None,
            concurrencies=(16,),
            kv_working_set_turns=1.0,
        ),
        decode_par=parallel,
        decode_topo=topology,
        replica_batch_cap=16,
    )
    assert math.isfinite(near_full_cap)
    assert near_full_cap > 0.0


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_pp_decode_sound_bound_and_capacity(fixture_name: str) -> None:
    """PP decode bound charges embed+lm_head once and matches max(T(B//pp), T(B)/pp)."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x2"]
    pp = 4
    par_pp4 = Parallel(tp=4, pp=pp, ep=4, moe="ep")
    par_pp1 = Parallel(tp=4, pp=1, ep=4, moe="ep")
    scenario = TrafficScenario(
        name="pp-decode-sound",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    ctx_mid = scenario.isl + scenario.osl // 2
    sf_b1_full = step_floor(
        geometry,
        chip,
        par_pp1,
        StepShape(batch=1, context=ctx_mid),
        FloorOptions(dtypes=dtypes, topology=topology),
    )
    bound_b1_ms = bound_tpot_ms(
        geometry,
        dtypes,
        chip,
        scenario,
        decode_par=par_pp4,
        decode_topo=topology,
        replica_batch_cap=1,
        concurrency=1,
    )
    assert bound_b1_ms == pytest.approx(sf_b1_full.hard_floor_s * 1000.0, rel=1e-6)

    # Also verify B > pp (B=16, pp=4) exercises max(T(B//pp), T(B)/pp) and capacity.
    batch_b = 16
    sf_mb = step_floor(
        geometry,
        chip,
        par_pp1,
        StepShape(batch=batch_b // pp, context=ctx_mid),
        FloorOptions(dtypes=dtypes, topology=topology),
    )
    sf_full = step_floor(
        geometry,
        chip,
        par_pp1,
        StepShape(batch=batch_b, context=ctx_mid),
        FloorOptions(dtypes=dtypes, topology=topology),
    )
    expected_pp_cycle_ms = (
        max(sf_mb.hard_floor_s, sf_full.hard_floor_s / float(pp)) * 1000.0
    )
    scen_b16 = TrafficScenario(
        name="pp-decode-b16",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(batch_b,),
        kv_working_set_turns=1.0,
        ttft_slo_ms=10_000.0,
    )
    bound_b16_ms = bound_tpot_ms(
        geometry,
        dtypes,
        chip,
        scen_b16,
        decode_par=par_pp4,
        decode_topo=topology,
        replica_batch_cap=batch_b,
        concurrency=batch_b,
    )
    assert bound_b16_ms > 0.0
    assert bound_b16_ms <= expected_pp_cycle_ms + 1e-6
    from vllm_torchtpu.roofline import pool_capacity_bound

    cap_rate = pool_capacity_bound(
        geometry,
        dtypes,
        chip,
        scen_b16,
        decode_par=par_pp4,
        decode_topo=topology,
        replica_batch_cap=batch_b,
    )
    assert math.isfinite(cap_rate)
    assert cap_rate > 0.0


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("topo_name", ["2x2x1", "2x2x2", "2x2x4"])
def test_physics_invariants_hold_for_every_candidate(
    fixture_name: str,
    topo_name: str,
) -> None:
    """Check floor<=est, ceiling>=tput, HBM fit, and batch monotonicity."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES[topo_name]
    scenario = TrafficScenario(
        name="invariant-sweep",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    mixed_plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", topologies=(topology,)),
    )
    disagg_opts = (
        SolveOptions(
            mode="disagg",
            max_chips=topology.num_chips * 2,
            topologies=(topology,),
            include_pairs=True,
        )
        if topo_name == "2x2x1"
        else SolveOptions(
            mode="disagg",
            max_chips=topology.num_chips * 2,
            topologies=(topology,),
            include_pairs=False,
        )
    )
    disagg_plan = solve_serving_plan(geometry, dtypes, chip, scenario, disagg_opts)
    for plan in (mixed_plan, disagg_plan):
        for point in plan.candidates:
            assert point.bound_tpot_ms <= point.est_tpot_ms + 1e-6
            assert point.bound_ttft_ms <= point.est_ttft_ms + 1e-6
            assert (
                point.token_throughput_per_chip
                <= point.floor_throughput_per_chip + 1e-4
            )
            assert (
                point.decode_pool.hbm_used_gb <= point.decode_pool.hbm_limit_gb + 1e-6
            )
            if point.prefill_pool is not None:
                assert (
                    point.prefill_pool.hbm_used_gb
                    <= point.prefill_pool.hbm_limit_gb + 1e-6
                )
        for concurrency in scenario.concurrencies:
            at_c = [
                point_item
                for point_item in plan.candidates
                if point_item.concurrency == concurrency
            ]
            if at_c:
                assert any(point_item.is_pareto for point_item in at_c)

    if topo_name in ("2x2x2", "2x2x4") and mixed_plan.candidates:
        assert any(
            point_item.decode_pool.pp > 1 for point_item in mixed_plan.candidates
        )
        assert all(
            point_item.decode_pool.pcp == 1 for point_item in mixed_plan.candidates
        )
        assert any(record.pool.pcp > 1 for record in disagg_plan.prefill_pools)

    # Batch monotonicity: decode step hard floor at b=16 >= b=1.
    par_ref = Parallel(tp=topology.num_cores, ep=topology.num_cores, moe="ep")
    sf_b1 = step_floor(
        geometry,
        chip,
        par_ref,
        StepShape(batch=1, rows_per_seq=1, context=1024),
        FloorOptions(dtypes=dtypes, topology=topology),
    )
    sf_b16 = step_floor(
        geometry,
        chip,
        par_ref,
        StepShape(batch=16, rows_per_seq=1, context=1024),
        FloorOptions(dtypes=dtypes, topology=topology),
    )
    assert sf_b16.hard_floor_s >= sf_b1.hard_floor_s - 1e-12


def test_golden_scenario_grid_matches() -> None:
    """Golden grid is compact (<500 KB), 72/72 populated, and matches live CLI."""
    golden_path = Path(__file__).resolve().parent / "golden" / "scenarios.json"
    assert golden_path.exists()
    assert golden_path.stat().st_size < 512_000
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    expected_count = (
        len(regen_golden.GOLDEN_CHECKPOINTS)
        * len(regen_golden.GOLDEN_SLICES)
        * len(regen_golden.GOLDEN_MODES)
        * len(regen_golden.GOLDEN_PRESETS)
    )
    assert data["scenario_count"] == expected_count
    assert len(data["scenarios"]) == expected_count
    for key, entry in data["scenarios"].items():
        assert entry["candidate_count"] > 0, f"Empty golden scenario: {key}"

    sample_cases = (
        ("qwen3_5_397b_fp8", "qwen3_5_397b_fp8.json", "2x2x1", 4, "mixed", "chat-8k1k"),
        (
            "qwen3_5_397b_nvfp4",
            "qwen3_5_397b_nvfp4.json",
            "2x2x2",
            8,
            "disagg",
            "coding-agent",
        ),
        ("glm_5_2_fp8", "glm_5_2_fp8.json", "2x2x2", 8, "mixed", "reasoning-1k8k"),
        ("glm_5_2_nvfp4", "glm_5_2_nvfp4.json", "2x2x4", 16, "disagg", "chat-8k1k"),
    )
    for ckpt_id, fixture_name, slice_name, max_chips, mode, preset in sample_cases:
        key = f"{ckpt_id}__{slice_name}__{mode}__{preset}"
        live = regen_golden.run_one_scenario(
            ckpt_id, fixture_name, slice_name, max_chips, (mode, preset)
        )
        assert live == data["scenarios"][key], f"Golden drift on {key}"


def test_docs_tables_match_generator() -> None:
    """Generated doc tables file matches render_docs_tables.py --check."""
    assert render_docs_tables.main(["--check"]) == 0
