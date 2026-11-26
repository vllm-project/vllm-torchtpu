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
"""Property, conservation, continuity, and isolation invariants for physics bounds."""

from __future__ import annotations

import ast
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import LOCKED_FIXTURE_FILES, load_locked_config
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    ChipSpec,
    Dtypes,
    ExpertActivationModel,
    ModelGeometry,
    Parallel,
    SliceTopology,
    SolveOptions,
    StepShape,
    TrafficScenario,
    build_solve_payload,
    cli,
    collective_cut_bound_s,
    count_layer_types,
    dtypes_from_hf_config,
    from_hf_config,
    pass_costs,
    solve_serving_plan,
    step_floor,
    torus_collective_s,
)
from vllm_torchtpu.roofline import memory as rf_memory
from vllm_torchtpu.roofline import solver as rf_solver
from vllm_torchtpu.roofline.collectives import submesh_for_group
from vllm_torchtpu.roofline.constants import UNIT_GIGA
from vllm_torchtpu.roofline.step import _draft_geom_for_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_torchtpu.roofline import ParetoPoint, PoolMeshPlan

_T = TypeVar("_T")
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_BASIS_FIXTURE_SPECS: tuple[tuple[str, str], ...] = tuple(
    (Path(fixture_name).stem, fixture_name) for fixture_name in LOCKED_FIXTURE_FILES
)
_LCG_MULT: int = 6364136223846793005
_LCG_INC: int = 1442695040888963407
_LCG_MASK: int = (1 << 64) - 1
_PROPERTY_SEED: int = 20261120
_STEP_SAMPLES: int = 10
_PLAN_SAMPLES: int = 4
_EPS_TOL: float = 1e-12
_EPS_MS: float = 1e-9
_EPS_TPUT: float = 1e-6
_EPS_RATE: float = 1e-6
_MATCH_REL_TOL: float = 1e-9
_CONTINUITY_REL_TOL: float = 1e-3
_TINY_THINK_TIME_S: float = 1e-6
_TINY_HIT_EPS: float = 1e-6
_ONE_GB: float = 1.0
_TEST_RESERVE_GB: float = 24.0
_DEFAULT_MAX_CHIPS: int = 8
_DISAGG_MAX_CHIPS: int = 8
_CONTINUITY_ISL: int = 4096
_CONTINUITY_OSL: int = 256
_CONTINUITY_CONC: int = 16
_RESERVE_CONC: int = 32
_ZERO_RESERVE_GB: float = 0.0
_HULL_GRID: tuple[int, ...] = (1, 16, 128)
_HULL_EXPERTS: tuple[float, ...] = (10.0, 64.0, 200.0)
_FORBIDDEN_BOUND_NAMES: frozenset[str] = frozenset(
    {"h_eff", "serial_estimate_s", "local_batch", "collective_latency_s"}
)
_ISOLATED_SOLVER_FUNCS: tuple[str, ...] = (
    "bound_ttft_ms",
    "envelope_ttft_ms",
    "bound_tpot_ms",
    "envelope_tpot_ms",
    "pool_capacity_bound",
    "request_rate_bound",
    "floor_throughput_per_chip",
    "_resolve_slo_verdict",
    "_apply_capacity_ceiling_verdict",
    "_mixed_capacity_work_bound",
    "_disagg_capacity_work_bound",
)
_SAMPLE_TOPOLOGIES: tuple[str, ...] = ("2x2x1", "2x2x2")
_SAMPLE_BATCHES: tuple[int, ...] = (1, 4, 16, 64)
_SAMPLE_PREFILL_TOKENS: tuple[int, ...] = (64, 512, 2048)
_SAMPLE_HIT_RATES: tuple[float, ...] = (0.0, 0.5, 0.95)
_SAMPLE_THINK_TIMES: tuple[float, ...] = (0.0, 0.25)
_SAMPLE_ACCEPT_LENS: tuple[float, ...] = (1.0, 2.5)
_SAMPLE_ISLS: tuple[int, ...] = (1024, 2048)
_SAMPLE_OSLS: tuple[int, ...] = (256, 512)
_SAMPLE_CONCURRENCIES: tuple[tuple[int, ...], ...] = ((1, 8), (4, 32))


def _assert_bound_signature_and_ast_isolation() -> None:
    """Verify physical bound helpers never accept or read estimate-only fields."""
    solver_path = Path(rf_solver.__file__).resolve()
    tree = ast.parse(solver_path.read_text(encoding="utf-8"))
    func_nodes = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _ISOLATED_SOLVER_FUNCS
    }
    assert set(func_nodes) == set(_ISOLATED_SOLVER_FUNCS)

    for func_name in _ISOLATED_SOLVER_FUNCS:
        fn_obj = getattr(rf_solver, func_name)
        sig = inspect.signature(fn_obj)
        overlap = set(sig.parameters) & _FORBIDDEN_BOUND_NAMES
        assert not overlap, f"{func_name} exposes estimate params: {overlap}"

        node = func_nodes[func_name]
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in _FORBIDDEN_BOUND_NAMES:
                pytest.fail(f"{func_name} references forbidden name {child.id!r}")
            if (
                isinstance(child, ast.Attribute)
                and child.attr in _FORBIDDEN_BOUND_NAMES
            ):
                pytest.fail(f"{func_name} accesses forbidden attribute {child.attr!r}")


def _assert_conservation_laws(
    models: tuple[tuple[ModelGeometry, Dtypes], ...],
    chip: ChipSpec,
) -> None:
    """Check layer-count and collective wire-byte conservation."""
    topology = TOPOLOGIES["2x2x1"]
    for geometry, dtypes in models:
        counts = count_layer_types(geometry)
        ignored_attn = {"ffn_only", "mlp", "dense_mlp", "moe", "mtp", "ple"}
        expected_attn_rec = sum(
            1 for tok in geometry.layer_types if tok not in ignored_attn
        )
        assert counts.attn_total + counts.gdn + counts.kda == expected_attn_rec
        n_moe_ffn, n_dense_ffn = geometry.ffn_layer_counts()
        assert n_moe_ffn + n_dense_ffn <= geometry.num_layers

        par_tp = Parallel(
            tp=chip.cores_per_chip * 2,
            dp=1,
            ep=chip.cores_per_chip * 2 if geometry.num_experts > 0 else 1,
            moe="ep" if geometry.num_experts > 0 else "tp_slice",
            ranks_per_chip=chip.cores_per_chip,
        )
        for shape in (
            StepShape(batch=4, rows_per_seq=1, context=512),
            StepShape(batch=0, rows_per_seq=0, prefill_tokens=512, prefix_hit_rate=0.0),
        ):
            costs = pass_costs(geometry, par_tp, shape, dtypes)
            for module in costs:
                for collective in module.collectives:
                    assert collective.wire_bytes >= 0.0
                    assert collective.payload >= 0.0
                    submesh = submesh_for_group(
                        collective.group, topology, chip.cores_per_chip
                    )
                    cut_s = collective_cut_bound_s(
                        collective.kind, collective.payload, submesh, chip, topology
                    )
                    torus_s = torus_collective_s(
                        collective.kind, collective.payload, submesh, chip, topology
                    )
                    assert cut_s <= torus_s + _EPS_TOL


def _pool_to_parallel(pool: PoolMeshPlan, topology: SliceTopology) -> Parallel:
    """Construct a Parallel dataclass matching an emitted PoolMeshPlan."""
    return Parallel(
        tp=pool.tp,
        dp=pool.dp,
        ep=pool.ep,
        pp=pool.pp,
        pcp=pool.pcp,
        kv_split=pool.kv_split,
        moe="ep" if pool.ep > 1 else "tp_slice",
        moe_dispatch=pool.moe_dispatch,
        residual=pool.residual,
        ranks_per_chip=topology.cores_per_chip,
        mesh_axes=pool.mesh_axes,
    )


def _point_key(point: ParetoPoint) -> tuple[object, ...]:
    """Return structural key matching pools across solves at the same concurrency."""
    pre = point.prefill_pool
    dec = point.decode_pool
    pre_key = (
        (
            pre.instances,
            pre.topology,
            pre.tp,
            pre.dp,
            pre.ep,
            pre.pp,
            pre.pcp,
            pre.kv_split,
        )
        if pre is not None
        else None
    )
    dec_key = (
        dec.instances,
        dec.topology,
        dec.tp,
        dec.dp,
        dec.ep,
        dec.pp,
        dec.pcp,
        dec.kv_split,
        dec.moe_dispatch,
    )
    return (point.mode, point.concurrency, pre_key, dec_key)


def _pool_b_cap_single(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    pool: PoolMeshPlan,
    scenario: TrafficScenario,
) -> int:
    """Compute physical single-replica steady-state HBM capacity cap."""
    topology = TOPOLOGIES[pool.topology]
    parallel = _pool_to_parallel(pool, topology)
    ctx_full = scenario.isl + scenario.osl
    usable_b = float(chip.hbm_bytes) - (_ZERO_RESERVE_GB * UNIT_GIGA)
    w_base_b = rf_memory.hbm_footprint_per_core_bytes(
        geometry, dtypes, parallel, rf_memory.SeqKvLayout(0, 0), chip=chip
    )
    hbm_c1 = rf_memory.hbm_footprint_per_core_bytes(
        geometry, dtypes, parallel, rf_memory.SeqKvLayout(1, ctx_full), chip=chip
    )
    per_seq_b = hbm_c1 - w_base_b
    if hbm_c1 > usable_b:
        kv_cap_seqs = 0
    elif per_seq_b > 0.0:
        kv_cap_seqs = int(max(0.0, usable_b - w_base_b) // per_seq_b)
    else:
        kv_cap_seqs = max(scenario.concurrencies, default=1)
    state_phys = rf_memory.hbm_breakdown_per_core_bytes(
        geometry, dtypes, parallel, rf_memory.SeqKvLayout(1, 0), chip=chip
    ).state
    h_rate = scenario.prefix_hit_rate
    shared_toks = h_rate * float(scenario.isl)
    avg_priv_toks = (1.0 - h_rate) * float(scenario.isl) + (
        float(max(0, scenario.osl - 1)) / 2.0
    )
    full_kv_per_tok, _ = rf_memory.kv_bytes_per_core_parts(
        geometry, dtypes, parallel, 0, physical=False, chip=chip
    )
    if geometry.sliding_window > 0:
        _, swa_per_tok = rf_memory.kv_bytes_per_core_parts(
            geometry, dtypes, parallel, 1, physical=False, chip=chip
        )
        swa_win = float(geometry.sliding_window)
        priv_swa = min(avg_priv_toks, swa_win)
        shared_swa = max(0.0, min(shared_toks, swa_win - priv_swa))
        shared_kv = shared_toks * full_kv_per_tok + shared_swa * swa_per_tok
        avg_priv_kv = avg_priv_toks * full_kv_per_tok + priv_swa * swa_per_tok
    else:
        shared_kv = shared_toks * full_kv_per_tok
        avg_priv_kv = avg_priv_toks * full_kv_per_tok
    avail = usable_b - w_base_b - shared_kv
    denom = state_phys + avg_priv_kv
    if avail < 0.0:
        steady_cap = 0
    elif denom <= 0.0:
        steady_cap = max(scenario.concurrencies, default=1)
    else:
        steady_cap = math.floor(avail / denom)
    return max(1, kv_cap_seqs, steady_cap)


class _DeterministicSampler:
    """Deterministic integer PRNG avoiding stdlib random warnings."""

    def __init__(self, seed: int) -> None:
        """Initialize sampler with a non-negative integer seed."""
        self._state = seed & _LCG_MASK

    def choice(self, seq: Sequence[_T]) -> _T:
        """Pick one element deterministically from a non-empty sequence."""
        self._state = (self._state * _LCG_MULT + _LCG_INC) & _LCG_MASK
        return seq[self._state % len(seq)]


def _valid_step_tp(geometry: ModelGeometry, tp: int) -> bool:
    """Return whether tp cleanly divides attention and linear-head dimensions."""
    if geometry.num_heads % tp != 0:
        return False
    kv_heads = geometry.num_kv_heads
    if kv_heads > 0 and kv_heads % tp != 0 and tp % kv_heads != 0:
        return False
    return not (
        geometry.linear_v_heads > 0
        and (geometry.linear_v_heads % tp != 0 or geometry.linear_k_heads % tp != 0)
    )


@pytest.mark.parametrize(("fixture_label", "filename"), _BASIS_FIXTURE_SPECS)
def test_physics_min_step_pruning_and_invariants_across_all_4_basis_fixtures(
    fixture_label: str,
    filename: str,
) -> None:
    """Verify unpadded bounds, ordering invariants, and impossible-SLO pruning."""
    config = load_locked_config(filename)
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x2"]
    parallel = Parallel(
        tp=8,
        dp=2,
        ep=16 if geometry.num_experts > 0 else 1,
        moe="ep" if geometry.num_experts > 0 else "tp_slice",
        moe_dispatch="ag_ep" if geometry.num_experts > 0 else "allreduce",
        ranks_per_chip=topology.cores_per_chip,
    )
    shape = StepShape(batch=4, rows_per_seq=1, context=1024)
    floor_result = step_floor(
        geometry,
        chip,
        parallel,
        shape,
        rf.FloorOptions(dtypes=dtypes, topology=topology),
    )
    unp = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, topology=topology),
    )
    assert max(unp) <= floor_result.resource_bound_s + _EPS_TOL, fixture_label

    scenario = TrafficScenario(
        name=f"physics-{fixture_label}",
        isl=1536,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
    )
    max_chips = 16
    plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=max_chips),
    )
    assert plan.candidates, fixture_label
    for point in plan.candidates:
        assert point.bound_ttft_ms <= point.envelope_ttft_ms + _EPS_MS
        assert point.envelope_ttft_ms <= point.est_ttft_ms + _EPS_MS
        assert point.bound_tpot_ms <= point.envelope_tpot_ms + _EPS_MS
        assert point.envelope_tpot_ms <= point.est_tpot_ms + _EPS_MS
        assert (
            point.token_throughput_per_chip
            <= point.floor_throughput_per_chip + _EPS_TPUT
        )

    min_tpot_bound = min(point_item.bound_tpot_ms for point_item in plan.candidates)
    impossible_scen = replace(
        scenario,
        name=f"physics-tight-{fixture_label}",
        tpot_slo_ms=min_tpot_bound * 0.5,
    )
    ruled_out_plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        impossible_scen,
        SolveOptions(mode="mixed", max_chips=max_chips),
    )
    assert ruled_out_plan.candidates
    assert ruled_out_plan.recommended is None
    assert ruled_out_plan.search is not None
    assert ruled_out_plan.search.not_ruled_out == 0
    assert all(
        point_item.slo_verdict == rf.SLO_VERDICT_RULED_OUT
        for point_item in ruled_out_plan.candidates
    )


def test_bound_ordering_holds_over_generated_inputs() -> None:
    """Verify isolation, conservation, step ordering, and rate-bound invariants."""
    _assert_bound_signature_and_ast_isolation()

    cfgs = tuple(load_locked_config(fname) for fname in LOCKED_FIXTURE_FILES)
    models = tuple((from_hf_config(c), dtypes_from_hf_config(c)) for c in cfgs)
    chip_base = CHIPS["tpu7x"]
    _assert_conservation_laws(models, chip_base)

    rng = _DeterministicSampler(_PROPERTY_SEED)
    for _ in range(_STEP_SAMPLES):
        geometry, dtypes = rng.choice((models[1], models[2], models[3]))
        topology = TOPOLOGIES[rng.choice(_SAMPLE_TOPOLOGIES)]
        cores = topology.num_cores
        tp_choices = [
            tp for tp in (1, 2, 4, 8) if tp <= cores and _valid_step_tp(geometry, tp)
        ]
        tp = rng.choice(tp_choices)
        dp = max(1, cores // tp)
        ep = tp * dp if geometry.num_experts > 0 else 1
        parallel = Parallel(
            tp=tp,
            dp=dp,
            ep=ep,
            moe="ep" if ep > 1 else "tp_slice",
            moe_dispatch="ag_ep" if ep > 1 else "allreduce",
            ranks_per_chip=topology.cores_per_chip,
        )
        is_decode = rng.choice((True, False))
        shape = (
            StepShape(
                batch=rng.choice(_SAMPLE_BATCHES),
                rows_per_seq=1,
                context=_CONTINUITY_ISL,
            )
            if is_decode
            else StepShape(
                batch=0,
                rows_per_seq=0,
                prefill_tokens=rng.choice(_SAMPLE_PREFILL_TOKENS),
                prefix_hit_rate=rng.choice(_SAMPLE_HIT_RATES),
            )
        )
        floor_result = step_floor(
            geometry,
            chip_base,
            parallel,
            shape,
            rf.FloorOptions(dtypes=dtypes, topology=topology),
        )
        assert floor_result.hard_floor_s <= floor_result.envelope_floor_s + _EPS_TOL
        assert (
            floor_result.envelope_floor_s <= floor_result.serial_estimate_s + _EPS_TOL
        )
        unp = rf.unpadded_strict_resource_times(
            geometry,
            chip_base,
            parallel,
            shape,
            rf.UnpaddedFloorOptions(dtypes=dtypes, topology=topology),
        )
        assert max(unp) <= floor_result.resource_bound_s + _EPS_TOL

    for _ in range(_PLAN_SAMPLES):
        geometry, dtypes = models[1]
        scenario = TrafficScenario(
            name="prop-sweep",
            isl=rng.choice(_SAMPLE_ISLS),
            osl=rng.choice(_SAMPLE_OSLS),
            prefix_hit_rate=rng.choice(_SAMPLE_HIT_RATES),
            accept_len=rng.choice(_SAMPLE_ACCEPT_LENS),
            think_time_s=rng.choice(_SAMPLE_THINK_TIMES),
            concurrencies=rng.choice(_SAMPLE_CONCURRENCIES),
            kv_working_set_turns=1.0,
        )
        mode_is_disagg = rng.choice((False, True))
        budget = _DISAGG_MAX_CHIPS if mode_is_disagg else _DEFAULT_MAX_CHIPS
        plan_res = solve_serving_plan(
            geometry,
            dtypes,
            chip_base,
            scenario,
            SolveOptions(
                mode="disagg" if mode_is_disagg else "mixed",
                max_chips=budget,
                include_pairs=mode_is_disagg,
            ),
        )
        assert plan_res.candidates
        for point in plan_res.candidates[:64]:
            assert point.bound_ttft_ms <= point.envelope_ttft_ms + _EPS_MS
            assert point.envelope_ttft_ms <= point.est_ttft_ms + _EPS_MS
            assert point.bound_tpot_ms <= point.envelope_tpot_ms + _EPS_MS
            assert point.envelope_tpot_ms <= point.est_tpot_ms + _EPS_MS
            assert (
                point.token_throughput_per_chip
                <= point.floor_throughput_per_chip + _EPS_TPUT
            )

            dec_topo = TOPOLOGIES[point.decode_pool.topology]
            dec_par = _pool_to_parallel(point.decode_pool, dec_topo)
            pre_pool = point.prefill_pool
            pre_topo = TOPOLOGIES[pre_pool.topology] if pre_pool is not None else None
            pre_par = (
                _pool_to_parallel(pre_pool, pre_topo)
                if pre_pool is not None and pre_topo is not None
                else None
            )
            b_cap = _pool_b_cap_single(
                geometry, dtypes, chip_base, point.decode_pool, scenario
            )
            cap = rf.pool_capacity_bound(
                geometry,
                dtypes,
                chip_base,
                scenario,
                decode_par=dec_par,
                decode_topo=dec_topo,
                replica_batch_cap=b_cap,
                prefill_par=pre_par,
                prefill_topo=pre_topo,
                prefill_instances=pre_pool.instances if pre_pool is not None else 1,
                decode_instances=point.decode_pool.instances,
                spec_k=plan_res.spec_k,
            )
            assert cap > 0.0
            x_p_cnt = pre_pool.instances if pre_pool is not None else 1
            b1_tpot_ms = rf.bound_tpot_ms(
                geometry,
                dtypes,
                chip_base,
                scenario,
                decode_par=dec_par,
                decode_topo=dec_topo,
                replica_batch_cap=b_cap,
                concurrency=1,
                spec_k=plan_res.spec_k,
                activation_model=plan_res.activation_model,
            )
            assert b1_tpot_ms > 0.0
            dec_variants = [
                par_v
                for par_v in rf_solver._candidate_residual_parallels(
                    geometry, replace(dec_par, residual="replicated")
                )
                if rf_memory.hbm_footprint_per_core_bytes(
                    geometry,
                    dtypes,
                    par_v,
                    rf_memory.SeqKvLayout(1, scenario.isl + scenario.osl),
                    chip=chip_base,
                )
                <= float(chip_base.hbm_bytes)
            ]
            pre_variants: list[Parallel | None] = (
                [
                    par_v
                    for par_v in rf_solver._candidate_residual_parallels(
                        geometry, replace(pre_par, residual="replicated")
                    )
                    if rf_memory.hbm_footprint_per_core_bytes(
                        geometry,
                        dtypes,
                        par_v,
                        rf_memory.SeqKvLayout(1, scenario.isl),
                        chip=chip_base,
                    )
                    <= float(chip_base.hbm_bytes)
                ]
                if pre_par is not None
                else [None]
            )
            rate_bounds = [
                rf.request_rate_bound(
                    geometry,
                    dtypes,
                    chip_base,
                    scenario,
                    prefill_par=pv,
                    prefill_topo=pre_topo,
                    decode_par=dv,
                    decode_topo=dec_topo,
                    replica_batch_cap=_pool_b_cap_single(
                        geometry,
                        dtypes,
                        chip_base,
                        replace(point.decode_pool, residual=dv.residual),
                        scenario,
                    ),
                    concurrency=point.concurrency,
                    prefill_instances=x_p_cnt,
                    decode_instances=point.decode_pool.instances,
                    spec_k=plan_res.spec_k,
                    activation_model=plan_res.activation_model,
                )
                for dv in dec_variants
                for pv in pre_variants
            ]
            expected_floors = [
                rf.floor_throughput_per_chip(
                    geometry,
                    dtypes,
                    chip_base,
                    scenario,
                    prefill_par=pv,
                    prefill_topo=pre_topo,
                    decode_par=dv,
                    decode_topo=dec_topo,
                    replica_batch_cap=_pool_b_cap_single(
                        geometry,
                        dtypes,
                        chip_base,
                        replace(point.decode_pool, residual=dv.residual),
                        scenario,
                    ),
                    concurrency=point.concurrency,
                    prefill_instances=x_p_cnt,
                    decode_instances=point.decode_pool.instances,
                    spec_k=plan_res.spec_k,
                    activation_model=plan_res.activation_model,
                    total_chips=point.total_chips,
                )
                for dv in dec_variants
                for pv in pre_variants
            ]
            assert point.floor_throughput_per_chip == pytest.approx(
                max(expected_floors), rel=_MATCH_REL_TOL
            )
            x_rate = (
                point.token_throughput_per_chip
                * float(point.total_chips)
                / float(scenario.isl + scenario.osl)
            )
            assert x_rate <= max(rate_bounds) + _EPS_RATE


def test_bound_is_continuous_at_zero_think_time_and_full_hit_rate() -> None:
    """Verify bounds vary smoothly as think_time_s -> 0+ and prefix_hit_rate -> 1-."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = dtypes_from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]

    for mode, budget in (("mixed", _DEFAULT_MAX_CHIPS), ("disagg", _DISAGG_MAX_CHIPS)):
        scen_zero = TrafficScenario(
            name="cont-zero-think",
            isl=_CONTINUITY_ISL,
            osl=_CONTINUITY_OSL,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(_CONTINUITY_CONC,),
            kv_working_set_turns=1.0,
        )
        scen_eps = replace(scen_zero, think_time_s=_TINY_THINK_TIME_S)
        plan_zero = solve_serving_plan(
            geometry,
            dtypes,
            chip,
            scen_zero,
            SolveOptions(
                mode=mode,
                max_chips=budget,
                include_pairs=mode == "disagg",
                prefill_topology="2x2x1" if mode == "disagg" else None,
                decode_topology="2x2x1" if mode == "disagg" else None,
                pd_ratio=(1, 1) if mode == "disagg" else None,
            ),
        )
        plan_eps = solve_serving_plan(
            geometry,
            dtypes,
            chip,
            scen_eps,
            SolveOptions(
                mode=mode,
                max_chips=budget,
                include_pairs=mode == "disagg",
                prefill_topology="2x2x1" if mode == "disagg" else None,
                decode_topology="2x2x1" if mode == "disagg" else None,
                pd_ratio=(1, 1) if mode == "disagg" else None,
            ),
        )
        zero_by_key = {_point_key(p): p for p in plan_zero.candidates}
        eps_by_key = {_point_key(p): p for p in plan_eps.candidates}
        assert zero_by_key
        assert set(zero_by_key) == set(eps_by_key)
        for key, pt_zero in zero_by_key.items():
            pt_eps = eps_by_key[key]
            assert pt_zero.bound_tpot_ms == pytest.approx(
                pt_eps.bound_tpot_ms, rel=_CONTINUITY_REL_TOL
            )
            assert pt_zero.bound_ttft_ms == pytest.approx(
                pt_eps.bound_ttft_ms, rel=_CONTINUITY_REL_TOL
            )
            assert pt_zero.floor_throughput_per_chip == pytest.approx(
                pt_eps.floor_throughput_per_chip, rel=_CONTINUITY_REL_TOL
            )
            assert pt_zero.min_decode_seqs == pytest.approx(
                pt_eps.min_decode_seqs, rel=_CONTINUITY_REL_TOL
            )

    scen_full = TrafficScenario(
        name="cont-full-hit",
        isl=_CONTINUITY_ISL,
        osl=_CONTINUITY_OSL,
        prefix_hit_rate=1.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(_CONTINUITY_CONC,),
        kv_working_set_turns=1.0,
    )
    scen_near = replace(scen_full, prefix_hit_rate=1.0 - _TINY_HIT_EPS)
    topology = TOPOLOGIES["2x2x1"]
    parallel = Parallel(
        tp=4,
        dp=2,
        ep=8,
        moe="ep",
        moe_dispatch="ag_ep",
        ranks_per_chip=topology.cores_per_chip,
    )
    ttft_full = rf.bound_ttft_ms(
        geometry, dtypes, chip, scen_full, prefill_par=parallel, prefill_topo=topology
    )
    ttft_near = rf.bound_ttft_ms(
        geometry, dtypes, chip, scen_near, prefill_par=parallel, prefill_topo=topology
    )
    env_full = rf.envelope_ttft_ms(
        geometry, dtypes, chip, scen_full, prefill_par=parallel, prefill_topo=topology
    )
    env_near = rf.envelope_ttft_ms(
        geometry, dtypes, chip, scen_near, prefill_par=parallel, prefill_topo=topology
    )
    assert ttft_full > 0.0
    assert ttft_full == pytest.approx(ttft_near, rel=_CONTINUITY_REL_TOL)
    assert env_full == pytest.approx(env_near, rel=_CONTINUITY_REL_TOL)

    plan_full = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_full,
        SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS),
    )
    plan_near = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_near,
        SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS),
    )
    full_by_key = {_point_key(p): p for p in plan_full.candidates}
    near_by_key = {_point_key(p): p for p in plan_near.candidates}
    assert full_by_key
    for key, pt_full in full_by_key.items():
        pt_near = near_by_key[key]
        assert pt_full.bound_ttft_ms == pytest.approx(
            pt_near.bound_ttft_ms, rel=_CONTINUITY_REL_TOL
        )
        assert pt_full.bound_tpot_ms == pytest.approx(
            pt_near.bound_tpot_ms, rel=_CONTINUITY_REL_TOL
        )


def test_supplied_constraint_never_lowers_a_bound() -> None:
    """Reject negative reserve constraints and ensure valid constraints raise bounds."""
    with pytest.raises(ValueError, match="hbm_reserve_gb"):
        SolveOptions(hbm_reserve_gb=-_ONE_GB)

    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = dtypes_from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="supplied-constraints",
        isl=_CONTINUITY_ISL,
        osl=_CONTINUITY_OSL,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(_RESERVE_CONC,),
        kv_working_set_turns=1.0,
    )
    base_plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS, hbm_reserve_gb=0.0),
    )
    res_plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="mixed", max_chips=_DEFAULT_MAX_CHIPS, hbm_reserve_gb=_TEST_RESERVE_GB
        ),
    )
    base_map = {_point_key(p): p for p in base_plan.candidates}
    res_map = {_point_key(p): p for p in res_plan.candidates}
    shared_keys = set(base_map) & set(res_map)
    assert shared_keys
    for key in shared_keys:
        base_pt = base_map[key]
        res_pt = res_map[key]
        assert res_pt.bound_tpot_ms >= base_pt.bound_tpot_ms - _EPS_MS
        assert res_pt.bound_ttft_ms >= base_pt.bound_ttft_ms - _EPS_MS
        assert res_pt.envelope_tpot_ms >= base_pt.envelope_tpot_ms - _EPS_MS
        assert res_pt.envelope_ttft_ms >= base_pt.envelope_ttft_ms - _EPS_MS
        assert (
            res_pt.floor_throughput_per_chip
            <= base_pt.floor_throughput_per_chip + _EPS_TPUT
        )

    pigeon_opts = SolveOptions(
        mode="mixed",
        max_chips=_DEFAULT_MAX_CHIPS,
        activation_model=ExpertActivationModel(regime="pigeonhole"),
    )
    hull_opts = SolveOptions(
        mode="mixed",
        max_chips=_DEFAULT_MAX_CHIPS,
        activation_model=ExpertActivationModel(
            regime="empirical_hull",
            decode_token_grid=_HULL_GRID,
            decode_active_experts=_HULL_EXPERTS,
            prefill_token_grid=_HULL_GRID,
            prefill_active_experts=_HULL_EXPERTS,
        ),
    )
    pigeon_plan = solve_serving_plan(geometry, dtypes, chip, scenario, pigeon_opts)
    hull_plan = solve_serving_plan(geometry, dtypes, chip, scenario, hull_opts)
    pigeon_map = {_point_key(p): p for p in pigeon_plan.candidates}
    hull_map = {_point_key(p): p for p in hull_plan.candidates}
    assert set(pigeon_map) == set(hull_map)
    for key, p_pt in pigeon_map.items():
        h_pt = hull_map[key]
        assert h_pt.bound_tpot_ms == pytest.approx(
            p_pt.bound_tpot_ms, rel=_MATCH_REL_TOL
        )
        assert h_pt.bound_ttft_ms == pytest.approx(
            p_pt.bound_ttft_ms, rel=_MATCH_REL_TOL
        )
        assert h_pt.envelope_tpot_ms >= h_pt.bound_tpot_ms - _EPS_MS
        assert h_pt.envelope_ttft_ms >= h_pt.bound_ttft_ms - _EPS_MS


def test_default_output_unchanged_without_supplied_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify default solve output and provenance when no supplied flags are set."""
    cfg_path = _FIXTURE_DIR / "qwen3_5_397b_fp8.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    rc_bare = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "8",
            "--concurrency",
            "1,8",
            "--json",
        ]
    )
    assert rc_bare == 0
    payload_bare = json.loads(capsys.readouterr().out)

    rc_zero = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "8",
            "--concurrency",
            "1,8",
            "--hbm-reserve-gb",
            "0",
            "--collective-latency-us",
            "0",
            "--json",
        ]
    )
    assert rc_zero == 0
    payload_zero = json.loads(capsys.readouterr().out)

    assert payload_bare["provenance"]["calibrated"] is False
    assert payload_bare["provenance"]["supplied_values"] == []
    assert payload_zero["provenance"]["calibrated"] is False
    assert payload_zero["provenance"]["supplied_values"] == []
    assert payload_bare["top_k"] == payload_zero["top_k"]
    assert (
        payload_bare["bound_families_by_concurrency"]
        == payload_zero["bound_families_by_concurrency"]
    )

    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    chip = CHIPS["tpu7x"]
    scenario = replace(rf.EXAMPLE_SCENARIOS["chat-8k1k"], concurrencies=(1, 8))
    direct_plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=_DISAGG_MAX_CHIPS),
    )
    direct_payload = build_solve_payload(
        direct_plan, chip=chip, dtypes=dtypes, model=str(cfg_path)
    )
    direct_json = json.loads(json.dumps(direct_payload))
    assert payload_bare["top_k"] == direct_json["top_k"]
    assert payload_bare["provenance"] == direct_json["provenance"]


def test_agentic_64k_preset_and_spec_zero_weights_strip_draft() -> None:
    """Verify agentic-64k preset and zero_weights draft-module stripping."""
    scenario = rf.build_example_scenario("agentic-64k")
    expected_isl = 65536
    expected_osl = 1024
    expected_spec = 3
    assert scenario.isl == expected_isl
    assert scenario.osl == expected_osl
    assert scenario.prefix_hit_rate == pytest.approx(0.88)
    assert scenario.spec_tokens == expected_spec

    geometry = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = rf.dtypes_from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]
    parallel = rf.Parallel(tp=8)
    shape = rf.StepShape(batch=4, rows_per_seq=1, context=1024)
    hbm_with_w, _, _, _ = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=1, zero_weights=False),
    )
    hbm_zero_w, _, _, _ = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=1, zero_weights=True),
    )
    hbm_no_spec_zero_w, _, _, _ = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=0, zero_weights=True),
    )
    draft_only_hbm_s, _, _, _ = rf.unpadded_strict_resource_times(
        _draft_geom_for_spec(geometry),
        chip,
        parallel,
        rf.StepShape(batch=4, rows_per_seq=1, context=1024, state_writes=1),
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=0, zero_weights=True),
    )
    assert hbm_zero_w < hbm_with_w
    assert hbm_zero_w == pytest.approx(hbm_no_spec_zero_w + draft_only_hbm_s, rel=1e-12)


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_spec_zero_weights_draft_vector_all_dtypes(fixture_name: str) -> None:
    """Speculative zero_weights matches 1-layer draft geometry scaled by spec_k."""
    config = load_locked_config(fixture_name)
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8, ep=8, moe="ep")
    spec_k = 3
    shape = StepShape(batch=4, rows_per_seq=1, context=1024)
    hbm_spec, _, vec_spec, _ = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=spec_k, zero_weights=True),
    )
    hbm_base, _, vec_base, _ = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=0, zero_weights=True),
    )
    hbm_draft, _, vec_draft, _ = rf.unpadded_strict_resource_times(
        _draft_geom_for_spec(geometry),
        chip,
        parallel,
        StepShape(batch=4, rows_per_seq=1, context=1024, state_writes=1),
        rf.UnpaddedFloorOptions(dtypes=dtypes, spec_k=0, zero_weights=True),
    )
    assert hbm_spec == pytest.approx(hbm_base + float(spec_k) * hbm_draft, rel=1e-12)
    assert vec_spec == pytest.approx(vec_base + float(spec_k) * vec_draft, rel=1e-12)


def test_solver_assumptions_match_defaults_and_chip_spec() -> None:
    """Solver assumptions match real defaults and reflect active chip hop latency."""
    from dataclasses import replace

    default_rows = {row["name"]: row for row in rf.get_solver_assumptions()}
    assert default_rows["collective_latency_us"]["value"] == 0.0
    assert (
        default_rows["ep_uniform_routing"]["value"] == rf.ExpertActivationModel().regime
    )
    assert default_rows["ici_hop_latency_us"]["value"] == pytest.approx(
        CHIPS["tpu7x"].ici_hop_latency_s / 1e-6
    )

    custom_chip = replace(CHIPS["tpu7x"], name="custom_hop", ici_hop_latency_s=2.5e-6)
    custom_rows = {row["name"]: row for row in rf.get_solver_assumptions(custom_chip)}
    assert custom_rows["ici_hop_latency_us"]["value"] == pytest.approx(2.5)
    assert custom_rows["collective_latency_us"]["value"] == 0.0


def test_recommended_is_always_pareto_when_slo_met_positive() -> None:
    """Whenever slo_met > 0, the recommended layout is strictly Pareto-optimal."""
    bundle_cfg = load_locked_config("glm_5_2_fp8.json")
    geometry = from_hf_config(bundle_cfg)
    dtypes = dtypes_from_hf_config(bundle_cfg)
    chip = CHIPS["tpu7x"]
    scenario = rf.build_example_scenario("coding-agent")
    scenario = replace(scenario, concurrencies=(1, 16, 64))
    plan = solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="disagg",
            max_chips=12,
            prefill_topology="2x2x1",
            decode_topology="2x2x2",
            include_pairs=True,
        ),
    )
    assert plan.search is not None
    assert plan.search.slo_met > 0
    assert plan.recommended is not None
    assert plan.recommended.is_pareto is True
