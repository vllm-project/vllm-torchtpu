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
"""Property, conservation, continuity, and isolation invariants for L1 bounds."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import (
    NEMOTRON_CFG,
    QWEN35_CFG,
    SMALL_HYBRID_MOE_CFG,
)
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
    hbm_breakdown_per_core_bytes,
    pass_costs,
    solve_serving_plan,
    step_floor,
    torus_collective_s,
)
from vllm_torchtpu.roofline import memory as rf_memory
from vllm_torchtpu.roofline import solver as rf_solver
from vllm_torchtpu.roofline.collectives import submesh_for_group
from vllm_torchtpu.roofline.constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    UNIT_GIGA,
)
from vllm_torchtpu.roofline.memory import SeqKvLayout
from vllm_torchtpu.roofline.partition import _pp_layer_prefixes, pp_stage_breakdowns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_torchtpu.roofline import ParetoPoint, PoolMeshPlan

_T = TypeVar("_T")
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
_CONSERVATION_REL_TOL: float = 1e-9
_CONTINUITY_REL_TOL: float = 1e-3
_TINY_THINK_TIME_S: float = 1e-6
_TINY_HIT_EPS: float = 1e-6
_ONE_GB: float = 1.0
_TEST_RESERVE_GB: float = 24.0
_DEFAULT_MAX_CHIPS: int = 4
_DISAGG_MAX_CHIPS: int = 8
_PP_STAGE_COUNT: int = 2
_CONTINUITY_ISL: int = 4096
_CONTINUITY_OSL: int = 256
_CONTINUITY_CONC: int = 16
_RESERVE_CONC: int = 32
_ZERO_RESERVE_GB: float = 0.0
_HULL_GRID: tuple[int, ...] = (1, 16, 128)
_HULL_EXPERTS: tuple[float, ...] = (4.0, 12.0, 16.0)
_SMALL_DENSE_CFG: dict[str, object] = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 32000,
    "tie_embeddings": False,
    "dtype": "bfloat16",
}
_FORBIDDEN_BOUND_NAMES: frozenset[str] = frozenset(
    {"h_eff", "serial_estimate_s", "b_local", "collective_latency_s"}
)
_ISOLATED_SOLVER_FUNCS: tuple[str, ...] = (
    "bound_ttft_ms",
    "envelope_ttft_ms",
    "bound_tpot_ms",
    "envelope_tpot_ms",
    "x_cap_pool",
    "x_bound",
    "floor_throughput_per_chip",
    "_resolve_slo_verdict",
    "_apply_capacity_ceiling_verdict",
    "_compute_mixed_capacity_bound",
    "_compute_disagg_capacity_bound",
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
_FP8_SMALL_CFG: dict[str, object] = {
    **_SMALL_DENSE_CFG,
    "quantization_config": {"quant_method": "fp8"},
}


def _assert_bound_signature_and_ast_isolation() -> None:
    """Verify Layer-1 bound helpers never accept or read estimate-only fields."""
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

    assert rf.X_cap_pool is rf.x_cap_pool
    assert rf.X_bound is rf.x_bound


def _assert_conservation_laws(
    models: tuple[tuple[ModelGeometry, Dtypes], ...],
    chip: ChipSpec,
) -> None:
    """Check layer-count, PP weight, and collective wire-byte conservation."""
    topo = TOPOLOGIES["2x2x1"]
    for geom, dt in models:
        counts = count_layer_types(geom)
        ignored_attn = {"ffn_only", "mlp", "dense_mlp", "moe", "mtp", "ple"}
        expected_attn_rec = sum(
            1 for tok in geom.layer_types if tok not in ignored_attn
        )
        assert counts.attn_total + counts.gdn + counts.kda == expected_attn_rec
        n_moe_ffn, n_dense_ffn = geom.ffn_layer_counts()
        assert n_moe_ffn + n_dense_ffn <= geom.num_layers

        if geom.num_layers >= _PP_STAGE_COUNT:
            par_pp = Parallel(
                tp=1, dp=1, ep=1, pp=_PP_STAGE_COUNT, ranks_per_chip=chip.cores_per_chip
            )
            part, best_bd, _last_tot, feasible = pp_stage_breakdowns(
                geom, dt, par_pp, (0, 0)
            )
            assert feasible
            assert sum(part) == geom.num_layers
            par1 = replace(par_pp, pp=1)
            pref = _pp_layer_prefixes(geom, dt, par1, SeqKvLayout(0, 0))
            whole_bd = hbm_breakdown_per_core_bytes(geom, dt, par1, SeqKvLayout(0, 0))
            pos = 0
            stage_weight_sum = 0.0
            for stage_idx, cnt in enumerate(part):
                nxt = pos + cnt
                ex_w = (pref.embed_b if stage_idx == 0 else 0.0) + (
                    pref.lm_b if stage_idx == _PP_STAGE_COUNT - 1 else 0.0
                )
                stage_weight_sum += (pref.weights[nxt] - pref.weights[pos]) + ex_w
                pos = nxt
            assert stage_weight_sum == pytest.approx(
                whole_bd.weights, rel=_CONSERVATION_REL_TOL
            )
            assert best_bd.weights <= whole_bd.weights * (1.0 + _CONSERVATION_REL_TOL)

        par_tp = Parallel(
            tp=chip.cores_per_chip * 2,
            dp=1,
            ep=chip.cores_per_chip * 2 if geom.num_experts > 0 else 1,
            moe="ep" if geom.num_experts > 0 else "tp_slice",
            ranks_per_chip=chip.cores_per_chip,
        )
        for shape in (
            StepShape(batch=4, rows_per_seq=1, context=512),
            StepShape(batch=0, rows_per_seq=0, prefill_tokens=512, prefix_hit_rate=0.0),
        ):
            costs = pass_costs(geom, par_tp, shape, dt)
            for mod in costs:
                for col in mod.collectives:
                    assert col.wire_bytes >= 0.0
                    assert col.payload >= 0.0
                    submesh = submesh_for_group(col.group, topo, chip.cores_per_chip)
                    cut_s = collective_cut_bound_s(
                        col.kind, col.payload, submesh, chip, topo
                    )
                    torus_s = torus_collective_s(
                        col.kind, col.payload, submesh, chip, topo
                    )
                    assert cut_s <= torus_s + _EPS_TOL


def _pool_to_parallel(pool: PoolMeshPlan, topo: SliceTopology) -> Parallel:
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
        ranks_per_chip=topo.cores_per_chip,
    )


def _point_key(pt: ParetoPoint) -> tuple[object, ...]:
    """Return structural key matching pools across solves at the same concurrency."""
    pre = pt.prefill_pool
    dec = pt.decode_pool
    pre_key = (
        (pre.instances, pre.topology, pre.tp, pre.dp, pre.ep, pre.pp, pre.pcp)
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
        dec.moe_dispatch,
    )
    return (pt.mode, pt.concurrency, pre_key, dec_key)


def _pool_b_cap_single(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    pool: PoolMeshPlan,
    scen: TrafficScenario,
) -> int:
    """Compute physical single-replica capacity cap used by isolated bounds."""
    topo = TOPOLOGIES[pool.topology]
    par = _pool_to_parallel(pool, topo)
    ctx_full = scen.isl + scen.osl
    usable_b = float(chip.hbm_bytes) * DEFAULT_GPU_MEMORY_UTILIZATION - (
        _ZERO_RESERVE_GB * UNIT_GIGA
    )
    w_base_b = rf_memory.hbm_footprint_per_core_bytes(
        geom, dtypes, par, rf_memory.SeqKvLayout(0, 0)
    )
    hbm_c1 = rf_memory.hbm_footprint_per_core_bytes(
        geom, dtypes, par, rf_memory.SeqKvLayout(1, ctx_full)
    )
    per_seq_b = hbm_c1 - w_base_b
    if hbm_c1 > usable_b:
        kv_cap_seqs = 0
    elif per_seq_b > 0.0:
        kv_cap_seqs = int((usable_b - w_base_b) // per_seq_b)
    else:
        kv_cap_seqs = max(scen.concurrencies, default=1)
    page_size = (
        scen.page_size
        if scen.page_size is not None
        else rf_memory.default_rpa_page_size(ctx_full)
    )
    smem_cap = rf_memory.smem_max_seqs(ctx_full, par.kv_split, page_size)
    max_seqs_cap = kv_cap_seqs if scen.max_num_seqs is None else scen.max_num_seqs
    return max(1, min(kv_cap_seqs, smem_cap, max_seqs_cap))


class _DeterministicSampler:
    """Deterministic integer PRNG avoiding stdlib random warnings."""

    def __init__(self, seed: int) -> None:
        """Initialize sampler with a non-negative integer seed."""
        self._state = seed & _LCG_MASK

    def choice(self, seq: Sequence[_T]) -> _T:
        """Pick one element deterministically from a non-empty sequence."""
        self._state = (self._state * _LCG_MULT + _LCG_INC) & _LCG_MASK
        return seq[self._state % len(seq)]


def _valid_step_tp(geom: ModelGeometry, tp: int) -> bool:
    """Return whether tp cleanly divides attention and linear-head dimensions."""
    if geom.num_heads % tp != 0:
        return False
    kv_heads = geom.num_kv_heads
    if kv_heads > 0 and kv_heads % tp != 0 and tp % kv_heads != 0:
        return False
    return not (
        geom.linear_v_heads > 0
        and (geom.linear_v_heads % tp != 0 or geom.linear_k_heads % tp != 0)
    )


def test_bound_ordering_holds_over_generated_inputs() -> None:
    """Verify L1 isolation, conservation, step ordering, and X_bound properties."""
    _assert_bound_signature_and_ast_isolation()

    cfgs = (
        QWEN35_CFG,
        SMALL_HYBRID_MOE_CFG,
        {**NEMOTRON_CFG, "dtype": "bfloat16"},
        _SMALL_DENSE_CFG,
    )
    models = tuple((from_hf_config(c), dtypes_from_hf_config(c)) for c in cfgs)
    chip_base = CHIPS["tpu7x"]
    _assert_conservation_laws(models, chip_base)

    rng = _DeterministicSampler(_PROPERTY_SEED)
    for _ in range(_STEP_SAMPLES):
        geom, dt = rng.choice((models[1], models[2], models[3]))
        topo = TOPOLOGIES[rng.choice(_SAMPLE_TOPOLOGIES)]
        cores = topo.num_cores
        tp_choices = [
            tp for tp in (1, 2, 4, 8) if tp <= cores and _valid_step_tp(geom, tp)
        ]
        tp = rng.choice(tp_choices)
        dp = max(1, cores // tp)
        ep = tp * dp if geom.num_experts > 0 else 1
        par = Parallel(
            tp=tp,
            dp=dp,
            ep=ep,
            moe="ep" if ep > 1 else "tp_slice",
            moe_dispatch="ag_ep" if ep > 1 else "allreduce",
            ranks_per_chip=topo.cores_per_chip,
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
        sf = step_floor(
            geom,
            chip_base,
            par,
            shape,
            rf.FloorOptions(dt=dt, topology=topo),
        )
        assert sf.hard_floor_s <= sf.envelope_floor_s + _EPS_TOL
        assert sf.envelope_floor_s <= sf.serial_estimate_s + _EPS_TOL
        unp = rf.unpadded_strict_resource_times(
            geom,
            chip_base,
            par,
            shape,
            rf.UnpaddedFloorOptions(dt=dt, topology=topo),
        )
        assert max(unp) <= sf.resource_bound_s + _EPS_TOL

    for _ in range(_PLAN_SAMPLES):
        geom, dt = models[1]
        scen = TrafficScenario(
            name="prop-sweep",
            isl=rng.choice(_SAMPLE_ISLS),
            osl=rng.choice(_SAMPLE_OSLS),
            prefix_hit_rate=rng.choice(_SAMPLE_HIT_RATES),
            accept_len=rng.choice(_SAMPLE_ACCEPT_LENS),
            think_time_s=rng.choice(_SAMPLE_THINK_TIMES),
            concurrencies=rng.choice(_SAMPLE_CONCURRENCIES),
            kv_working_set_turns=1.0,
        )
        mode_choice = rng.choice(("mixed", "disagg"))
        budget = _DISAGG_MAX_CHIPS if mode_choice == "disagg" else _DEFAULT_MAX_CHIPS
        plan_res = solve_serving_plan(
            geom,
            dt,
            chip_base,
            scen,
            SolveOptions(mode=mode_choice, max_chips=budget),
        )
        assert plan_res.candidates
        for pt in plan_res.candidates:
            assert pt.bound_ttft_ms <= pt.envelope_ttft_ms + _EPS_MS
            assert pt.envelope_ttft_ms <= pt.est_ttft_ms + _EPS_MS
            assert pt.bound_tpot_ms <= pt.envelope_tpot_ms + _EPS_MS
            assert pt.envelope_tpot_ms <= pt.est_tpot_ms + _EPS_MS
            assert (
                pt.token_throughput_per_chip <= pt.floor_throughput_per_chip + _EPS_TPUT
            )

            dec_topo = TOPOLOGIES[pt.decode_pool.topology]
            dec_par = _pool_to_parallel(pt.decode_pool, dec_topo)
            pre_pool = pt.prefill_pool
            pre_topo = TOPOLOGIES[pre_pool.topology] if pre_pool is not None else None
            pre_par = (
                _pool_to_parallel(pre_pool, pre_topo)
                if pre_pool is not None and pre_topo is not None
                else None
            )
            b_cap = _pool_b_cap_single(geom, dt, chip_base, pt.decode_pool, scen)
            cap = rf.x_cap_pool(
                geom,
                dt,
                chip_base,
                scen,
                decode_par=dec_par,
                decode_topo=dec_topo,
                b_cap_single=b_cap,
                prefill_par=pre_par,
                prefill_topo=pre_topo,
                x_p=pre_pool.instances if pre_pool is not None else 1,
                y_d=pt.decode_pool.instances,
                spec_k=plan_res.spec_tokens,
            )
            assert cap > 0.0
            x_p_cnt = pre_pool.instances if pre_pool is not None else 1
            b1_tpot_ms = rf.bound_tpot_ms(
                geom,
                dt,
                chip_base,
                scen,
                decode_par=dec_par,
                decode_topo=dec_topo,
                b_cap_single=b_cap,
                concurrency=1,
                spec_k=plan_res.spec_tokens,
                activation_model=plan_res.activation_model,
            )
            assert b1_tpot_ms > 0.0
            x_bnd = rf.x_bound(
                geom,
                dt,
                chip_base,
                scen,
                prefill_par=pre_par,
                prefill_topo=pre_topo,
                decode_par=dec_par,
                decode_topo=dec_topo,
                b_cap_single=b_cap,
                concurrency=pt.concurrency,
                x_p=x_p_cnt,
                y_d=pt.decode_pool.instances,
                spec_k=plan_res.spec_tokens,
                activation_model=plan_res.activation_model,
            )
            expected_floor = rf.floor_throughput_per_chip(
                geom,
                dt,
                chip_base,
                scen,
                prefill_par=pre_par,
                prefill_topo=pre_topo,
                decode_par=dec_par,
                decode_topo=dec_topo,
                b_cap_single=b_cap,
                concurrency=pt.concurrency,
                x_p=x_p_cnt,
                y_d=pt.decode_pool.instances,
                spec_k=plan_res.spec_tokens,
                activation_model=plan_res.activation_model,
                total_chips=pt.total_chips,
            )
            assert pt.floor_throughput_per_chip == pytest.approx(
                expected_floor, rel=_MATCH_REL_TOL
            )
            x_rate = (
                pt.token_throughput_per_chip
                * float(pt.total_chips)
                / float(scen.isl + scen.osl)
            )
            assert x_rate <= x_bnd + _EPS_RATE


def test_bound_is_continuous_at_zero_think_time_and_full_hit_rate() -> None:
    """Verify bounds vary smoothly as think_time_s -> 0+ and prefix_hit_rate -> 1-."""
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
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
            geom, dt, chip, scen_zero, SolveOptions(mode=mode, max_chips=budget)
        )
        plan_eps = solve_serving_plan(
            geom, dt, chip, scen_eps, SolveOptions(mode=mode, max_chips=budget)
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
    topo = TOPOLOGIES["2x2x1"]
    par = Parallel(
        tp=4,
        dp=2,
        ep=8,
        moe="ep",
        moe_dispatch="ag_ep",
        ranks_per_chip=topo.cores_per_chip,
    )
    ttft_full = rf.bound_ttft_ms(
        geom, dt, chip, scen_full, prefill_par=par, prefill_topo=topo
    )
    ttft_near = rf.bound_ttft_ms(
        geom, dt, chip, scen_near, prefill_par=par, prefill_topo=topo
    )
    env_full = rf.envelope_ttft_ms(
        geom, dt, chip, scen_full, prefill_par=par, prefill_topo=topo
    )
    env_near = rf.envelope_ttft_ms(
        geom, dt, chip, scen_near, prefill_par=par, prefill_topo=topo
    )
    assert ttft_full > 0.0
    assert ttft_full == pytest.approx(ttft_near, rel=_CONTINUITY_REL_TOL)
    assert env_full == pytest.approx(env_near, rel=_CONTINUITY_REL_TOL)

    plan_full = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_full,
        SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS),
    )
    plan_near = solve_serving_plan(
        geom,
        dt,
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

    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
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
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS, hbm_reserve_gb=0.0),
    )
    res_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
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
    pigeon_plan = solve_serving_plan(geom, dt, chip, scen, pigeon_opts)
    hull_plan = solve_serving_plan(geom, dt, chip, scen, hull_opts)
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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify default L1 solve output and provenance when no supplied flags are set."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_FP8_SMALL_CFG), encoding="utf-8")

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
            "4",
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
            "4",
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
        == (payload_zero["bound_families_by_concurrency"])
    )

    geom = from_hf_config(_FP8_SMALL_CFG)
    dt = dtypes_from_hf_config(_FP8_SMALL_CFG)
    chip = CHIPS["tpu7x"]
    scen = replace(rf.EXAMPLE_SCENARIOS["chat-8k1k"], concurrencies=(1, 8))
    direct_plan = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=_DEFAULT_MAX_CHIPS)
    )
    direct_payload = build_solve_payload(
        direct_plan, chip=chip, dtypes=dt, model=str(cfg_path)
    )
    direct_json = json.loads(json.dumps(direct_payload))
    assert payload_bare["top_k"] == direct_json["top_k"]
    assert payload_bare["provenance"] == direct_json["provenance"]
