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
"""Build versioned solve JSON payloads and ceiling curves from serving plans."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict, Unpack

from .chips import CHIPS, JAX_VERSION, chip_catalog, topologies_for_chip
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    PERCENT_SCALE,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    SOLVE_SCHEMA,
    UNIT_GIGA,
)
from .families import (
    attach_reference_report,
    build_bound_families_by_concurrency,
    ordered_candidates_by_concurrency,
    parse_layout,
    reference_layout_report,
    strict_rank,
)
from .schema import (
    CeilingRow,
    ParetoPoint,
    PoolFloorRecord,
    PoolStructKey,
    ProvenancePayload,
    ServingPlan,
    SolvePayload,
    SortBy,
    TopKEntry,
    _point_struct_key,
    candidate_sort_key,
    effective_slo_verdict,
    format_host_count,
    format_parallelism,
    format_submodule_parallelism,
    pool_label,
    pool_slice_summary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chips import ChipSpec, SliceTopology
    from .modules import Dtypes

__all__ = [
    "CeilingRow",
    "ProvenancePayload",
    "SolvePayload",
    "TopKEntry",
    "attach_reference_report",
    "build_bound_families_by_concurrency",
    "build_solve_payload",
    "emit_ceiling_rows",
    "format_host_count",
    "format_parallelism",
    "format_solve_json",
    "format_submodule_parallelism",
    "ordered_candidates_by_concurrency",
    "parse_layout",
    "pool_label",
    "pool_slice_summary",
    "reference_layout_report",
    "strict_rank",
    "top_layout_roles",
    "top_layouts",
]


class _BuildSolvePayloadOpts(TypedDict, total=False):
    """Optional keyword arguments for :func:`build_solve_payload`."""

    dtypes: Dtypes | None
    model: str | None
    top_k: int | None
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None


def slice_per_concurrency[T: (ParetoPoint, PoolFloorRecord)](
    items: Sequence[T],
    top_k: int | None,
) -> tuple[T, ...]:
    """Keep up to ``top_k`` ordered items per concurrency for presentation."""
    if top_k is None:
        return tuple(items)
    counts: dict[int, int] = {}
    sliced: list[T] = []
    for item in items:
        seen = counts.get(item.concurrency, 0)
        if seen < top_k:
            sliced.append(item)
            counts[item.concurrency] = seen + 1
    return tuple(sliced)


def _is_2d_dominated(
    point: ParetoPoint,
    peers: Sequence[ParetoPoint],
) -> bool:
    """Return True if a peer beats point on both interactivity and throughput."""
    return any(
        other is not point
        and other.est_interactivity_tok_s >= point.est_interactivity_tok_s
        and other.token_throughput_per_chip >= point.token_throughput_per_chip
        and (
            other.est_interactivity_tok_s > point.est_interactivity_tok_s
            or other.token_throughput_per_chip > point.token_throughput_per_chip
        )
        for other in peers
    )


def _metric_pair(point: ParetoPoint, sort_by: str) -> tuple[float, float]:
    """Return (primary, secondary) objective values for the active sort_by mode."""
    if sort_by == "interactivity":
        return (point.est_interactivity_tok_s, point.token_throughput_per_chip)
    return (point.token_throughput_per_chip, point.est_interactivity_tok_s)


def top_layouts(
    plan: ServingPlan,
    top_k: int | None = None,
) -> tuple[ParetoPoint, ...]:
    """Return distinct layouts ordered by plan.sort_by, optionally cut to k."""
    if top_k is not None and top_k <= 0:
        return ()
    slo_cands = [point_item for point_item in plan.candidates if point_item.slo_met]
    if not slo_cands:
        return ()
    best: dict[tuple[str, PoolStructKey | None, PoolStructKey | None], ParetoPoint] = {}
    for point in slo_cands:
        key = _point_struct_key(point)
        current_best = best.get(key)
        if current_best is None or _metric_pair(point, plan.sort_by) > _metric_pair(
            current_best, plan.sort_by
        ):
            best[key] = point
    ranked = sorted(
        best.values(),
        key=lambda point_item: candidate_sort_key(point_item, plan.sort_by),
    )
    if top_k is None:
        return tuple(ranked)
    if len(ranked) <= 1 or top_k < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return tuple(ranked[:top_k])
    selected = list(ranked[:top_k])
    opposite = max(
        slo_cands,
        key=lambda point_item: (
            (
                point_item.token_throughput_per_chip,
                point_item.est_interactivity_tok_s,
            )
            if plan.sort_by == "interactivity"
            else (
                point_item.est_interactivity_tok_s,
                point_item.token_throughput_per_chip,
            )
        ),
    )
    opp_key = _point_struct_key(opposite)
    sel_keys = [_point_struct_key(point_item) for point_item in selected]
    if opp_key not in sel_keys:
        selected[-1] = opposite
    elif sel_keys.index(opp_key) > 0:
        selected[sel_keys.index(opp_key)] = opposite
    selected.sort(key=lambda point_item: candidate_sort_key(point_item, plan.sort_by))
    return tuple(selected)


def top_layout_roles(
    top_pts: tuple[ParetoPoint, ...],
    frontier: Sequence[ParetoPoint] = (),
    *,
    sort_by: SortBy = "throughput",
) -> list[tuple[ParetoPoint, str, float]]:
    """Annotate selected Top-K points with role labels and Rank-1 metric margins."""
    is_interactivity = sort_by == "interactivity"
    top1_metric = (
        (
            top_pts[0].est_interactivity_tok_s
            if is_interactivity
            else top_pts[0].token_throughput_per_chip
        )
        if top_pts
        else 0.0
    )
    fastest_idx = (
        max(
            range(len(top_pts)),
            key=lambda i: (
                (
                    top_pts[i].token_throughput_per_chip,
                    top_pts[i].est_interactivity_tok_s,
                )
                if is_interactivity
                else (
                    top_pts[i].est_interactivity_tok_s,
                    top_pts[i].token_throughput_per_chip,
                )
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    eligible_frontier = [
        point_item
        for point_item in frontier
        if effective_slo_verdict(point_item) != SLO_VERDICT_RULED_OUT
    ]
    dominance_pool: tuple[ParetoPoint, ...] = (*top_pts, *eligible_frontier)
    out: list[tuple[ParetoPoint, str, float]] = []
    for idx, point in enumerate(top_pts):
        if idx == 0:
            role = "max_interactivity" if is_interactivity else "max_throughput"
        elif idx == fastest_idx:
            role = "max_throughput" if is_interactivity else "max_interactivity"
        elif _is_2d_dominated(point, dominance_pool):
            role = "off_frontier_alt"
        else:
            role = "balanced_knee"
        point_metric = (
            point.est_interactivity_tok_s
            if is_interactivity
            else point.token_throughput_per_chip
        )
        margin = round(
            PERCENT_SCALE * (point_metric - top1_metric) / max(1e-9, top1_metric),
            2,
        )
        out.append((point, role, margin))
    return out


def _resolve_single_host_topology(
    chips_per_host: int,
    chip: ChipSpec | None,
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None,
) -> str:
    """Return single-host slice topology name matching ``chips_per_host``."""
    if topologies is not None:
        topo_seq = (
            list(topologies.values())
            if isinstance(topologies, Mapping)
            else list(topologies)
        )
    elif chip is not None:
        topo_seq = list(topologies_for_chip(chip).values())
    else:
        topo_seq = list(topologies_for_chip("tpu7x").values())
    return next(
        (
            topology.name
            for topology in topo_seq
            if topology.num_chips == chips_per_host
        ),
        f"{chips_per_host}chip",
    )


def _compute_single_host_fit(
    plan: ServingPlan,
    *,
    chip: ChipSpec | None = None,
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None = None,
) -> dict[str, object]:
    """Compute single-host slice fit verdict, capacity, and status."""
    eff_chip = chip if chip is not None else CHIPS.get(plan.chip)
    chips_per_host = (
        eff_chip.chips_per_host if eff_chip is not None else DEFAULT_CHIPS_PER_HOST
    )
    cores_per_chip = (
        eff_chip.cores_per_chip if eff_chip is not None else DEFAULT_CORES_PER_CHIP
    )
    cores_per_host = chips_per_host * cores_per_chip
    host_topo = _resolve_single_host_topology(chips_per_host, eff_chip, topologies)
    host_desc = f"(1 host, {chips_per_host} chips, {cores_per_host} TensorCores)"
    host_points = [
        point_item
        for point_item in (*plan.frontier, *plan.candidates)
        if point_item.total_chips == chips_per_host
        and point_item.decode_pool.topology == host_topo
        and point_item.prefill_pool is None
        and effective_slo_verdict(point_item) != SLO_VERDICT_RULED_OUT
    ]
    if host_points:
        best_pt = max(
            host_points,
            key=lambda point: (
                SLO_VERDICT_RANK[effective_slo_verdict(point)],
                point.token_throughput_per_chip,
            ),
        )
        pool = best_pt.decode_pool
        return {
            "feasible": True,
            "status": "FEASIBLE",
            "reason": f"Fits single-host {host_topo} {host_desc}",
            "topology": host_topo,
            "chips_per_host": chips_per_host,
            "cores_per_host": cores_per_host,
            "hbm_used_gb": round(pool.hbm_used_gb, 2),
            "hbm_limit_gb": round(pool.hbm_limit_gb, 2),
            "best_parallelism": format_parallelism(asdict(pool)),
            "best_submodule_parallelism": format_submodule_parallelism(
                asdict(pool), has_moe=plan.has_moe
            ),
        }
    per_core_bytes = eff_chip.hbm_bytes if eff_chip else CHIPS["tpu7x"].hbm_bytes
    limit_gb = round(
        max(0.0, float(per_core_bytes) / float(UNIT_GIGA) - plan.hbm_reserve_gb),
        2,
    )
    if plan.mode == "disagg":
        reason = (
            f"Disaggregated mode selected; single-host {host_topo} requires mixed mode "
            f"{host_desc}"
        )
    elif 0 < plan.budget < chips_per_host:
        reason = (
            f"Chip budget ({plan.budget}) is below 1 host "
            f"({host_topo} requires {chips_per_host} chips, "
            f"{cores_per_host} TensorCores)"
        )
    else:
        reason = (
            f"No feasible {host_topo} layout within physical HBM, head-divisibility, "
            f"or SLO bounds {host_desc}"
        )
    return {
        "feasible": False,
        "status": "REJECTED",
        "reason": reason,
        "topology": host_topo,
        "chips_per_host": chips_per_host,
        "cores_per_host": cores_per_host,
        "hbm_used_gb": 0.0,
        "hbm_limit_gb": limit_gb,
        "best_parallelism": None,
        "best_submodule_parallelism": None,
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    **opts: Unpack[_BuildSolvePayloadOpts],
) -> SolvePayload:
    """Build versioned solve JSON payload with analytical provenance and pools."""
    dtypes = opts.get("dtypes")
    model = opts.get("model")
    top_k = opts.get("top_k")
    topologies = opts.get("topologies")
    top_pts = top_layouts(plan, top_k)
    top_entries: list[TopKEntry] = [
        {
            "frontier_role": role,
            "margin_vs_rank1_pct": margin,
            "point": asdict(point),
        }
        for point, role, margin in top_layout_roles(
            top_pts, plan.frontier, sort_by=plan.sort_by
        )
    ]
    pins_map = plan.pins or {}
    chip_dict: dict[str, object] = dict(asdict(chip).items())
    chip_dict["jax_version"] = JAX_VERSION
    chip_dict["catalog"] = chip_catalog(chip)
    target_chips = (
        plan.recommended.total_chips if plan.recommended is not None else plan.budget
    )
    visible_candidates = slice_per_concurrency(plan.candidates, top_k)
    visible_prefill = slice_per_concurrency(plan.prefill_pools, top_k)
    visible_decode = slice_per_concurrency(plan.decode_pools, top_k)
    payload: SolvePayload = {
        "model_layers": plan.model_layers,
        "chip": plan.chip,
        "scenario": asdict(plan.scenario),
        "mode": plan.mode,
        "recommended": (
            asdict(plan.recommended) if plan.recommended is not None else None
        ),
        "frontier": tuple(asdict(point) for point in plan.frontier),
        "warnings": plan.warnings,
        "candidates": tuple(asdict(point) for point in visible_candidates),
        "prefill_pools": tuple(asdict(record) for record in visible_prefill),
        "decode_pools": tuple(asdict(record) for record in visible_decode),
        "hbm_reserve_gb": plan.hbm_reserve_gb,
        "sort_by": plan.sort_by,
        "top_k_limit": top_k,
        "search": asdict(plan.search) if plan.search is not None else None,
        "schema": SOLVE_SCHEMA,
        "provenance": {
            "calibrated": False,
            "basis": (
                "Analytical step floors and uncalibrated serial-phase "
                "estimates; modeled, not measured"
            ),
        },
        "chip_spec": chip_dict,
        "model": model,
        "top_k": top_entries,
        "has_moe": plan.has_moe,
        "single_host_fit": _compute_single_host_fit(
            plan, chip=chip, topologies=topologies
        ),
        "target_chips": target_chips,
        "budget": plan.budget,
        "budget_source": plan.budget_source,
        "max_instances": plan.max_instances,
        "pins": plan.pins,
        "chip_budget": {
            "max_chips": (
                None
                if plan.budget_source == "max_instances"
                else (plan.budget if plan.budget > 0 else None)
            ),
            "budget_source": plan.budget_source,
            "max_instances": plan.max_instances,
            "prefill_topology": pins_map.get("prefill_topology"),
            "decode_topology": pins_map.get("decode_topology"),
            "pd_ratio": pins_map.get("pd_ratio", pins_map.get("pd")),
        },
    }
    ordered_by_conc = ordered_candidates_by_concurrency(plan, top_k)
    concurrency_top_k: dict[str, list[TopKEntry]] = {}
    for concurrency, conc_pts in ordered_by_conc.items():
        eligible_conc = tuple(
            pt
            for pt in conc_pts
            if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT and pt.slo_met
        )
        ruled_conc = [
            pt
            for pt in conc_pts
            if effective_slo_verdict(pt) == SLO_VERDICT_RULED_OUT or not pt.slo_met
        ]
        conc_frontier = [
            pt for pt in eligible_conc if not _is_2d_dominated(pt, eligible_conc)
        ]
        entries: list[TopKEntry] = [
            {
                "frontier_role": role,
                "margin_vs_rank1_pct": margin,
                "point": asdict(point),
            }
            for point, role, margin in top_layout_roles(
                eligible_conc, conc_frontier, sort_by=plan.sort_by
            )
        ]
        entries.extend(
            {
                "frontier_role": effective_slo_verdict(point),
                "margin_vs_rank1_pct": 0.0,
                "point": asdict(point),
            }
            for point in ruled_conc
        )
        concurrency_top_k[str(concurrency)] = entries
    payload["concurrency_top_k"] = concurrency_top_k
    payload["bound_families_by_concurrency"] = build_bound_families_by_concurrency(plan)
    if dtypes is not None:
        payload["dtypes"] = asdict(dtypes)
    return payload


def format_solve_json(payload: SolvePayload) -> str:
    """Serialize a ``SolvePayload`` mapping as a JSON string."""
    return json.dumps(payload, allow_nan=False)


def emit_ceiling_rows(
    plan: ServingPlan,
    *,
    top_k: int | None = None,
) -> list[CeilingRow]:
    """Export ordered candidate rows covering legal candidates in ``plan``."""
    visible = slice_per_concurrency(plan.candidates, top_k)
    return [
        {
            "est_interactivity_tok_s": point.est_interactivity_tok_s,
            "token_throughput_per_chip": point.token_throughput_per_chip,
            "output_throughput_per_chip": point.output_throughput_per_chip,
            "est_ttft_ms": point.est_ttft_ms,
            "est_tpot_ms": point.est_tpot_ms,
            "concurrency": point.concurrency,
            "total_chips": point.total_chips,
            "mode": point.mode,
            "binding_unit": point.binding,
            "slo_met": point.slo_met,
            "is_pareto": point.is_pareto,
            "decode_topology": point.decode_pool.topology,
            "decode_parallelism": format_parallelism(asdict(point.decode_pool)),
            "prefill_topology": (
                point.prefill_pool.topology if point.prefill_pool is not None else None
            ),
            "prefill_parallelism": (
                format_parallelism(asdict(point.prefill_pool))
                if point.prefill_pool is not None
                else None
            ),
        }
        for point in visible
    ]
