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
"""Hard and serial step floors for decode, prefill, mixed, and speculative cycles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

from .collectives import (
    CollectiveConfig,
    CutPlacementOptions,
    LinkCutLoad,
    _expand_mask_bytes,
    _has_finite_ici,
    _mask_lane_sum,
    _parent_axis_lanes,
    _valid_submesh_factorizations,
    collective_axis_cut_bytes,
    collective_cut_bound_s,
    collective_ici_hops,
    eval_link_subset_bound_s,
    make,
    min_collective_cut_bound_s,
    outer_submesh_for_group,
    submesh_for_group,
    torus_collective_s,
)
from .constants import (
    DRAFT_PROJECTION_HIDDEN_VECTORS,
    STEP_FLOOR_CACHE_MAXSIZE,
    UNIT_COLL_CACHE_MAXSIZE,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_TO_BF16_RATIO_BRACKET,
)
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    _MatmulBuckets,
    _route_matmul_flops,
    activation_wire_bytes,
    pass_costs,
)
from .occupancy import ExpertActivationModel

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology
    from .collectives import Collective
    from .geometry import ModelGeometry

type _CollKey = tuple[str, str, float, int, float, float, int]


@dataclass(frozen=True, slots=True)
class FloorOptions:
    """Optional dtype, latency, topology, and activation parameters for step_floor."""

    dtypes: Dtypes = field(default_factory=Dtypes)
    collective_latency_s: float = 0.0
    topology: SliceTopology | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )


@dataclass(frozen=True, slots=True)
class SpecShape:
    """Batch, context window, and speculative lookahead length for spec_cycle_floor."""

    batch: int
    context: int = 0
    spec_k: int = 0
    shared_prefix_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StepFloor:
    """Resource times, physical lower bound, and serial estimate for one step."""

    modules: tuple[ModuleCost, ...]
    hbm_s: float
    matmul_s: float
    vector_s: float
    ici_s: float
    chain_s: float
    hard_floor_s: float
    serial_estimate_s: float
    binding: str
    hbm_bound_s: float = 0.0
    matmul_bound_s: float = 0.0
    vector_bound_s: float = 0.0
    coll_bound_s: float = 0.0
    latency_assumed_s: float = 0.0
    envelope_floor_s: float = 0.0
    link_load: LinkCutLoad = field(default_factory=LinkCutLoad)

    @property
    def resource_bound_s(self) -> float:
        """Maximum time across HBM, matmul, vector, and collective cut bounds."""
        return max(
            self.hbm_bound_s,
            self.matmul_bound_s,
            self.vector_bound_s,
            self.coll_bound_s,
        )


@lru_cache(maxsize=UNIT_COLL_CACHE_MAXSIZE)
def _unit_coll_times(
    collective: Collective,
    topology: SliceTopology | None,
    ranks_per_chip: int,
    c_axes: tuple[int, int, int, int] | None,
    chip: ChipSpec,
) -> tuple[float, float, int]:
    """Compute assumed hop latency, analytic torus duration, and ICI hop count."""
    has_finite_ici = chip.ici_bw > 0.0 and math.isfinite(chip.ici_bw)
    unit_hop_lat_s = (
        chip.ici_hop_latency_s
        if (
            has_finite_ici
            and chip.ici_hop_latency_s > 0.0
            and math.isfinite(chip.ici_hop_latency_s)
        )
        else 0.0
    )
    c_hops = collective_ici_hops(collective, topology, ranks_per_chip, c_axes)
    hop_s = c_hops * unit_hop_lat_s
    submesh = submesh_for_group(collective.group, topology, ranks_per_chip, c_axes)
    anal_s = torus_collective_s(
        collective.kind, collective.payload, submesh, chip, topology
    )
    return (hop_s, anal_s, c_hops)


def _collective_partition_key(
    collective: Collective,
    parallel: Parallel | None,
) -> str:
    """Return canonical parallel-partition identity for bisection accounting."""
    if parallel is None:
        return ""
    name = collective.name
    if name == "pp_p2p":
        return "pp"
    if name.startswith("pcp_"):
        return "pcp"
    if name.startswith(("dcp_", "kv_")):
        return "dcp"
    if name.startswith(("dispatch_", "combine_")) or name == "ep_a2a":
        return "ep"
    if collective.group == parallel.tp and parallel.tp > 1:
        return "tp"
    return f"g{collective.group}"


def _vertex_only_push_load(
    collective: Collective,
    chip: ChipSpec,
    topology: SliceTopology | None,
) -> LinkCutLoad:
    """Return vertex-only off-chip pigeonhole push load with zero bisection."""
    payload = collective.payload
    if payload <= 0.0 or not _has_finite_ici(chip):
        return LinkCutLoad()
    if topology is not None:
        ax, ay, az = topology.chip_axes
        mask = (1 if ax > 1 else 0) | (2 if ay > 1 else 0) | (4 if az > 1 else 0)
        if mask == 0:
            return LinkCutLoad()
        lanes = _mask_lane_sum(_parent_axis_lanes(topology), mask)
    else:
        mask = 1
        lanes = 1.0
    if lanes <= 0.0:
        return LinkCutLoad()
    return LinkCutLoad(
        d2d_bytes=0.0,
        vertex_mask_bytes=_expand_mask_bytes(mask, payload),
        edge_mask_bytes=_expand_mask_bytes(mask, payload),
        partition_axis_bisect=(),
        isolated_cut_s=payload / (lanes * chip.ici_link_bw),
    )


def _pool_core_count(parallel: Parallel | None, fallback_group: int) -> int:
    """Return total active cores in the parallel pool (including EP)."""
    if parallel is None:
        return max(1, fallback_group)
    return max(
        1,
        parallel.tp * parallel.dp * parallel.pp * parallel.pcp,
        parallel.ep * parallel.pp,
    )


def _pp_inner_ranks(
    parallel: Parallel,
    topology: SliceTopology | None,
) -> int:
    """Return per-stage inner rank count for pipeline stage-boundary P2P."""
    inner = max(1, parallel.tp * parallel.dp * max(1, parallel.pcp))
    if topology is not None and parallel.pp > 0:
        inner = min(inner, max(1, topology.num_cores // parallel.pp))
    return max(1, inner)


def _pp_concurrent_groups_per_chip(
    parallel: Parallel,
    pp_intra_ranks: int,
    ranks_per_chip: int,
    *,
    off_chip: bool,
) -> int:
    """Return distinct PP activation streams crossing off-chip links per chip."""
    if not off_chip:
        return 1
    # pp_p2p carries the full activation tensor; sibling TP ranks on one chip
    # hold the same activation, while independent DP/PCP ranks carry distinct
    # batches or sequence chunks.
    distinct_streams = max(1, parallel.dp * max(1, parallel.pcp))
    inner_per_chip = max(1, ranks_per_chip // max(1, pp_intra_ranks))
    tp_per_chip = min(max(1, parallel.tp), inner_per_chip)
    streams_per_chip = max(1, inner_per_chip // tp_per_chip)
    return min(distinct_streams, streams_per_chip)


def _pp_submesh_permutations(
    pp_axes: tuple[int, int, int, int],
    topology: SliceTopology,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return distinct valid (x, y, z) permutations of pp_axes inside topology."""
    intra, ax, ay, az = pp_axes
    bx, by, bz = topology.chip_axes
    perms = (
        (ax, ay, az),
        (ax, az, ay),
        (ay, ax, az),
        (ay, az, ax),
        (az, ax, ay),
        (az, ay, ax),
    )
    valid: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for px, py, pz in perms:
        if (px, py, pz) in seen:
            continue
        if px <= bx and py <= by and pz <= bz:
            seen.add((px, py, pz))
            valid.append((intra, px, py, pz))
    return tuple(valid) if valid else (pp_axes,)


def _estimate_axes_for_collective(
    collective: Collective,
    topology: SliceTopology | None,
    parallel: Parallel | None,
    ranks_per_chip: int,
) -> tuple[int, int, int, int] | None:
    """Return submesh axes used for analytical torus/hop estimates."""
    if collective.name == "pp_p2p" and parallel is not None and parallel.pp > 1:
        inner = _pp_inner_ranks(parallel, topology)
        return outer_submesh_for_group(parallel.pp, inner, topology, ranks_per_chip)
    if (
        parallel is not None
        and parallel.mesh_axes is not None
        and collective.group == parallel.tp
    ):
        return parallel.mesh_axes
    return None


def _unit_coll_bound_load(
    collective: Collective,
    chip: ChipSpec,
    topology: SliceTopology | None,
    parallel: Parallel | None,
    ranks_per_chip: int,
) -> tuple[tuple[int, int, int, int] | None, LinkCutLoad]:
    """Return estimate axes and sound per-link-axis LinkCutLoad for one collective."""
    partition_key = _collective_partition_key(collective, parallel)
    c_axes = _estimate_axes_for_collective(
        collective, topology, parallel, ranks_per_chip
    )
    if collective.name in ("dispatch_vertex_push", "combine_vertex_push"):
        return c_axes, _vertex_only_push_load(collective, chip, topology)

    if collective.name == "pp_p2p" and parallel is not None and parallel.pp > 1:
        inner = _pp_inner_ranks(parallel, topology)
        pp_axes = c_axes or outer_submesh_for_group(
            parallel.pp, inner, topology, ranks_per_chip
        )
        off_chip = pp_axes[1] * pp_axes[2] * pp_axes[3] > 1
        conc = _pp_concurrent_groups_per_chip(
            parallel, pp_axes[0], ranks_per_chip, off_chip=off_chip
        )
        opts = CutPlacementOptions(
            topology=topology,
            ranks_per_chip=ranks_per_chip,
            concurrent_groups_per_chip=conc,
            partition_key=partition_key,
        )
        return pp_axes, collective_axis_cut_bytes(
            collective.kind, collective.payload, pp_axes, chip, opts
        )

    pool_cores = _pool_core_count(parallel, collective.group)
    if topology is None or pool_cores < topology.num_cores:
        opts = CutPlacementOptions(
            topology=topology,
            ranks_per_chip=ranks_per_chip,
            mesh_axes=c_axes,
            concurrent_groups_per_chip=1,
            partition_key=partition_key,
        )
        _, best_submesh = min_collective_cut_bound_s(
            collective.kind, collective.payload, collective.group, chip, opts
        )
        return c_axes, collective_axis_cut_bytes(
            collective.kind, collective.payload, best_submesh, chip, opts
        )

    num_groups = max(1, pool_cores // max(1, collective.group))
    axes_prod = (
        c_axes[0] * c_axes[1] * c_axes[2] * c_axes[3] if c_axes is not None else 0
    )
    bounds = (
        c_axes
        if (
            c_axes is not None
            and collective.group <= axes_prod
            and axes_prod % collective.group == 0
        )
        else (topology.cores_per_chip, *topology.chip_axes)
    )
    best_s = math.inf
    best_submesh = submesh_for_group(collective.group, topology, ranks_per_chip, c_axes)
    best_conc = 1
    for cand in _valid_submesh_factorizations(collective.group, bounds):
        intra, axis_x, axis_y, axis_z = cand
        off_chip = axis_x * axis_y * axis_z > 1
        concurrency = (
            min(num_groups, max(1, ranks_per_chip // intra)) if off_chip else 1
        )
        base_s = collective_cut_bound_s(
            collective.kind, collective.payload, cand, chip, topology
        )
        cand_s = float(concurrency) * base_s if off_chip else base_s
        if cand_s < best_s:
            best_s = cand_s
            best_submesh = cand
            best_conc = concurrency
    load_opts = CutPlacementOptions(
        topology=topology,
        ranks_per_chip=ranks_per_chip,
        concurrent_groups_per_chip=best_conc,
        partition_key=partition_key,
    )
    return c_axes, collective_axis_cut_bytes(
        collective.kind, collective.payload, best_submesh, chip, load_opts
    )


def _partition_mask_options(
    items: tuple[tuple[Collective, int], ...],
    partition_key: str,
    chip: ChipSpec,
    topology: SliceTopology,
    parallel: Parallel | None,
    ranks_per_chip: int,
) -> tuple[LinkCutLoad, ...]:
    """Return best aggregated LinkCutLoad per active-axis bitmask for one partition."""
    first_coll = items[0][0]
    group = first_coll.group
    if first_coll.name == "pp_p2p" and parallel is not None and parallel.pp > 1:
        inner = _pp_inner_ranks(parallel, topology)
        pp_axes = outer_submesh_for_group(parallel.pp, inner, topology, ranks_per_chip)
        candidates = _pp_submesh_permutations(pp_axes, topology)
        use_pp_conc = True
        num_groups = inner
    else:
        c_axes = _estimate_axes_for_collective(
            first_coll, topology, parallel, ranks_per_chip
        )
        axes_prod = (
            c_axes[0] * c_axes[1] * c_axes[2] * c_axes[3] if c_axes is not None else 0
        )
        bounds = (
            c_axes
            if (c_axes is not None and group <= axes_prod and axes_prod % group == 0)
            else (topology.cores_per_chip, *topology.chip_axes)
        )
        candidates = _valid_submesh_factorizations(group, bounds)
        if not candidates:
            candidates = (submesh_for_group(group, topology, ranks_per_chip, c_axes),)
        pool_cores = _pool_core_count(parallel, group)
        use_pp_conc = False
        num_groups = (
            max(1, pool_cores // max(1, group))
            if pool_cores >= topology.num_cores
            else 1
        )

    best_by_mask: dict[int, tuple[float, LinkCutLoad]] = {}
    for cand in candidates:
        intra, axis_x, axis_y, axis_z = cand
        off_chip = axis_x * axis_y * axis_z > 1
        if use_pp_conc and parallel is not None:
            conc = _pp_concurrent_groups_per_chip(
                parallel, intra, ranks_per_chip, off_chip=off_chip
            )
        elif off_chip:
            conc = (
                min(num_groups, max(1, ranks_per_chip // max(1, intra)))
                if num_groups > 1
                else 1
            )
        else:
            conc = 1
        mask = (
            (1 if axis_x > 1 else 0)
            | (2 if axis_y > 1 else 0)
            | (4 if axis_z > 1 else 0)
        )
        opts = CutPlacementOptions(
            topology=topology,
            ranks_per_chip=ranks_per_chip,
            concurrent_groups_per_chip=conc,
            partition_key=partition_key,
        )
        agg = LinkCutLoad()
        for coll, call_count in items:
            unit = collective_axis_cut_bytes(coll.kind, coll.payload, cand, chip, opts)
            agg = agg.add(unit.scaled(float(call_count)))
        cand_s = eval_link_subset_bound_s(agg, chip, topology)
        prev = best_by_mask.get(mask)
        if prev is None or cand_s < prev[0]:
            best_by_mask[mask] = (cand_s, agg)
    ordered = sorted(best_by_mask.values(), key=lambda pair: pair[0])
    return tuple(load for _, load in ordered)


@lru_cache(maxsize=UNIT_COLL_CACHE_MAXSIZE)
def _resolve_strict_link_load_cached(
    strict_items: tuple[tuple[Collective, int], ...],
    chip: ChipSpec,
    topology: SliceTopology | None,
    parallel: Parallel | None,
    ranks_per_chip: int,
) -> tuple[LinkCutLoad, float]:
    """Resolve joint minimum-bound LinkCutLoad across all strict collectives."""
    if not strict_items:
        return LinkCutLoad(), 0.0
    if topology is None:
        total = LinkCutLoad()
        for coll, call_count in strict_items:
            _, unit_load = _unit_coll_bound_load(
                coll, chip, None, parallel, ranks_per_chip
            )
            total = total.add(unit_load.scaled(float(call_count)))
        return total, eval_link_subset_bound_s(total, chip, None)

    fixed_load = LinkCutLoad()
    grouped: dict[tuple[str, int], list[tuple[Collective, int]]] = {}
    for coll, call_count in strict_items:
        if coll.name in ("dispatch_vertex_push", "combine_vertex_push"):
            unit_load = _vertex_only_push_load(coll, chip, topology)
            fixed_load = fixed_load.add(unit_load.scaled(float(call_count)))
            continue
        partition_key = _collective_partition_key(coll, parallel)
        if not partition_key:
            _, unit_load = _unit_coll_bound_load(
                coll, chip, topology, parallel, ranks_per_chip
            )
            fixed_load = fixed_load.add(unit_load.scaled(float(call_count)))
            continue
        grouped.setdefault((partition_key, coll.group), []).append((coll, call_count))

    option_groups: list[tuple[LinkCutLoad, ...]] = []
    for (partition_key, _), group_items in grouped.items():
        opts = _partition_mask_options(
            tuple(group_items),
            partition_key,
            chip,
            topology,
            parallel,
            ranks_per_chip,
        )
        if len(opts) == 1:
            fixed_load = fixed_load.add(opts[0])
        elif opts:
            option_groups.append(opts)

    if not option_groups:
        return fixed_load, eval_link_subset_bound_s(fixed_load, chip, topology)

    if len(option_groups) == 1:
        best_load = fixed_load.add(option_groups[0][0])
        best_s = eval_link_subset_bound_s(best_load, chip, topology)
        for opt in option_groups[0][1:]:
            cand_load = fixed_load.add(opt)
            cand_s = eval_link_subset_bound_s(cand_load, chip, topology)
            if cand_s < best_s:
                best_s = cand_s
                best_load = cand_load
        return best_load, best_s

    best_s = math.inf
    best_load = fixed_load

    def _search(idx: int, acc: LinkCutLoad) -> None:
        nonlocal best_s, best_load
        if idx == len(option_groups):
            cand_s = eval_link_subset_bound_s(acc, chip, topology)
            if cand_s < best_s:
                best_s = cand_s
                best_load = acc
            return
        for opt in option_groups[idx]:
            nxt = acc.add(opt)
            if eval_link_subset_bound_s(nxt, chip, topology) >= best_s:
                continue
            _search(idx + 1, nxt)

    _search(0, fixed_load)
    return best_load, best_s


def _dedup_module_collectives(
    modules: tuple[ModuleCost, ...],
    *,
    use_strict: bool = False,
) -> tuple[dict[_CollKey, int], list[tuple[_CollKey, Collective]]]:
    """Aggregate collective call counts across all modules in a step."""
    coll_counts: dict[_CollKey, int] = {}
    coll_samples: list[tuple[_CollKey, Collective]] = []
    for module in modules:
        if use_strict and module.strict_collectives is not None:
            entries = module.strict_collectives
        elif module.unique_collectives is not None:
            entries = module.unique_collectives
        else:
            entries = tuple(
                (collective_item, 1) for collective_item in module.collectives
            )
        for collective, m_cnt in entries:
            if collective.group <= 1 or m_cnt <= 0:
                continue
            collective_key = (
                collective.name,
                collective.kind,
                collective.payload,
                collective.group,
                collective.wire_bytes,
                collective.offchip,
                collective.hops,
            )
            if collective_key not in coll_counts:
                coll_counts[collective_key] = m_cnt
                coll_samples.append((collective_key, collective))
            else:
                coll_counts[collective_key] += m_cnt
    return coll_counts, coll_samples


def _div_peak(flops: float, peak: float, *, precision: str) -> float:
    """Divide precision FLOPs by chip peak, guarding unsupported precision."""
    if flops == 0.0:
        return 0.0
    if peak <= 0.0:
        msg = (
            f"chip does not support {precision} matrix compute "
            f"(peak={peak}, required_flops={flops})"
        )
        raise ValueError(msg)
    return flops / peak


def _precision_time_raw_s(
    fp8_flops: float,
    bf16_flops: float,
    int8_flops: float,
    fp4_flops: float,
    int4_flops: float,
    chip: ChipSpec,
    *,
    scale: float = 1.0,
) -> float:
    """Sum per-precision matrix times across all hardware precision peaks."""
    total = 0.0
    if fp8_flops != 0.0:
        total += _div_peak(fp8_flops * scale, chip.fp8_flops, precision="fp8")
    if bf16_flops != 0.0:
        total += _div_peak(bf16_flops * scale, chip.bf16_flops, precision="bf16")
    if int8_flops != 0.0:
        total += _div_peak(int8_flops * scale, chip.int8_flops, precision="int8")
    if fp4_flops != 0.0:
        total += _div_peak(fp4_flops * scale, chip.fp4_flops, precision="fp4")
    if int4_flops != 0.0:
        total += _div_peak(int4_flops * scale, chip.int4_flops, precision="int4")
    return total


def _precision_time_s(
    buckets: _MatmulBuckets,
    chip: ChipSpec,
    *,
    scale: float = 1.0,
) -> float:
    """Sum per-precision matrix times across all hardware precision peaks."""
    return _precision_time_raw_s(
        buckets.fp8,
        buckets.bf16,
        buckets.int8,
        buckets.fp4,
        buckets.int4,
        chip,
        scale=scale,
    )


def module_serial_s(
    module: ModuleCost,
    chip: ChipSpec,
    *,
    matmul_scale: float = 1.0,
) -> float:
    """Return single-module serial time across HBM, matmul, and vector units."""
    return max(
        module.hbm_bytes / chip.hbm_bw,
        _precision_time_raw_s(
            module.fp8_flops,
            module.bf16_flops,
            module.int8_flops,
            module.fp4_flops,
            module.int4_flops,
            chip,
            scale=matmul_scale,
        ),
        module.vector_flops / chip.vector_flops,
    )


def _sum_module_resources(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
    *,
    has_finite_ici: bool,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Sum per-module HBM, FLOPs, vector, ICI, and serial module execution times."""
    tot_hbm_b = 0.0
    tot_hbm_min_b = 0.0
    nom_fp8 = 0.0
    nom_bf16 = 0.0
    nom_int8 = 0.0
    nom_fp4 = 0.0
    nom_int4 = 0.0
    min_fp8 = 0.0
    min_bf16 = 0.0
    min_int8 = 0.0
    min_fp4 = 0.0
    min_int4 = 0.0
    tot_vector = 0.0
    tot_vector_min = 0.0
    tot_ici_b = 0.0
    mod_serial_s = 0.0
    hbm_bw = chip.hbm_bw
    vector_peak = chip.vector_flops
    for module in modules:
        mod_hbm = module.hbm_bytes
        tot_hbm_b += mod_hbm
        tot_hbm_min_b += module.strict_hbm_bytes
        m_fp8 = module.fp8_flops
        m_bf16 = module.bf16_flops
        m_int8 = module.int8_flops
        m_fp4 = module.fp4_flops
        m_int4 = module.int4_flops
        nom_fp8 += m_fp8
        nom_bf16 += m_bf16
        nom_int8 += m_int8
        nom_fp4 += m_fp4
        nom_int4 += m_int4
        min_fp8 += module.strict_fp8_flops
        min_bf16 += module.strict_bf16_flops
        min_int8 += module.strict_int8_flops
        min_fp4 += module.strict_fp4_flops
        min_int4 += module.strict_int4_flops
        mod_vec = module.vector_flops
        tot_vector += mod_vec
        tot_vector_min += module.strict_vector_flops
        if has_finite_ici:
            tot_ici_b += module.ici_bytes
        mod_serial_s += module_serial_s(module, chip)
    hbm_s = tot_hbm_b / hbm_bw
    matmul_s = _precision_time_raw_s(
        nom_fp8, nom_bf16, nom_int8, nom_fp4, nom_int4, chip
    )
    vector_s = tot_vector / vector_peak
    ici_s = tot_ici_b / chip.ici_bw if has_finite_ici else 0.0
    hbm_bound_s = tot_hbm_min_b / hbm_bw
    matmul_bound_s = _precision_time_raw_s(
        min_fp8, min_bf16, min_int8, min_fp4, min_int4, chip
    )
    # Strict lower bound uses the fast end of the public vector-to-BF16 bracket
    # (1/64) unless chip.vector_flops was explicitly overridden, ensuring
    # bound <= estimate across the entire public bracket.
    strict_vector_peak = (
        vector_peak
        if chip.vector_flops_overridden
        else max(vector_peak, chip.bf16_flops * VECTOR_TO_BF16_RATIO_BRACKET[1])
    )
    vector_bound_s = tot_vector_min / (
        VECTOR_MAX_ELEMENTS_PER_WORD * strict_vector_peak
    )
    return (
        hbm_s,
        matmul_s,
        vector_s,
        ici_s,
        hbm_bound_s,
        matmul_bound_s,
        vector_bound_s,
        mod_serial_s,
    )


def _sum_envelope_resources(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
) -> tuple[float, float]:
    """Sum per-module activation-envelope memory and matrix-compute spans."""
    total_bytes_envelope = 0.0
    env_fp8 = 0.0
    env_bf16 = 0.0
    env_int8 = 0.0
    env_fp4 = 0.0
    env_int4 = 0.0
    for module in modules:
        total_bytes_envelope += module.envelope_hbm_bytes
        env_fp8 += module.envelope_fp8_flops
        env_bf16 += module.envelope_bf16_flops
        env_int8 += module.envelope_int8_flops
        env_fp4 += module.envelope_fp4_flops
        env_int4 += module.envelope_int4_flops
    return (
        total_bytes_envelope / chip.hbm_bw,
        _precision_time_raw_s(env_fp8, env_bf16, env_int8, env_fp4, env_int4, chip),
    )


def combine_modules(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
    collective_latency_s: float = 0.0,
    *,
    topology: SliceTopology | None = None,
    parallel: Parallel | None = None,
) -> StepFloor:
    """Combine per-module resource costs into a StepFloor summary."""
    eff_parallel = parallel
    has_finite_ici = chip.ici_bw > 0.0 and math.isfinite(chip.ici_bw)
    (
        hbm_s,
        matmul_s,
        vector_s,
        ici_s,
        hbm_bound_s,
        matmul_bound_s,
        vector_bound_s,
        mod_serial_s,
    ) = _sum_module_resources(modules, chip, has_finite_ici=has_finite_ici)
    mesh_axes = eff_parallel.mesh_axes if eff_parallel is not None else None
    ranks_per_chip = (
        eff_parallel.ranks_per_chip
        if eff_parallel is not None and eff_parallel.ranks_per_chip is not None
        else chip.cores_per_chip
    )
    use_torus = topology is not None or mesh_axes is not None

    call_counts, coll_samples = _dedup_module_collectives(modules, use_strict=False)
    hop_lat_s = 0.0
    analytic_chain_s = 0.0
    user_chain_lat_s = 0.0
    for collective_key, collective in coll_samples:
        call_count = call_counts[collective_key]
        c_axes = _estimate_axes_for_collective(
            collective, topology, eff_parallel, ranks_per_chip
        )
        hop_s, anal_s, c_hops = _unit_coll_times(
            collective, topology, ranks_per_chip, c_axes, chip
        )
        hop_lat_s += call_count * hop_s
        analytic_chain_s += call_count * anal_s
        if collective_latency_s > 0.0:
            user_chain_lat_s += call_count * (c_hops * collective_latency_s)

    strict_counts, strict_samples = _dedup_module_collectives(modules, use_strict=True)
    strict_items = tuple(
        (collective, strict_counts[collective_key])
        for collective_key, collective in strict_samples
    )
    total_link_load, coll_bound_s = _resolve_strict_link_load_cached(
        strict_items, chip, topology, eff_parallel, ranks_per_chip
    )
    chain_s = analytic_chain_s + user_chain_lat_s

    hard_floor_s = max(hbm_bound_s, matmul_bound_s, vector_bound_s, coll_bound_s)
    serial_estimate_s = mod_serial_s + chain_s
    memory_envelope_s, matmul_envelope_s = _sum_envelope_resources(modules, chip)
    envelope_floor_s = max(
        memory_envelope_s,
        matmul_envelope_s,
        vector_bound_s,
        coll_bound_s,
    )
    if hard_floor_s == hbm_bound_s:
        binding = "hbm"
    elif hard_floor_s == matmul_bound_s:
        binding = "matmul"
    elif hard_floor_s == vector_bound_s:
        binding = "vector"
    elif use_torus:
        binding = "chain"
    else:
        binding = "ici"
    return StepFloor(
        modules=modules,
        hbm_s=hbm_s,
        matmul_s=matmul_s,
        vector_s=vector_s,
        ici_s=ici_s,
        chain_s=chain_s,
        hard_floor_s=hard_floor_s,
        serial_estimate_s=serial_estimate_s,
        binding=binding,
        hbm_bound_s=hbm_bound_s,
        matmul_bound_s=matmul_bound_s,
        vector_bound_s=vector_bound_s,
        coll_bound_s=coll_bound_s,
        latency_assumed_s=hop_lat_s + user_chain_lat_s,
        envelope_floor_s=envelope_floor_s,
        link_load=total_link_load,
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def step_floor(
    geometry: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    shape: StepShape,
    options: FloorOptions | None = None,
) -> StepFloor:
    """Compute the floor for one decode, prefill, or mixed forward pass."""
    opts = options or FloorOptions()
    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
    return combine_modules(
        pass_costs(
            geometry,
            parallel,
            shape,
            opts.dtypes,
            mxu_m_tile=chip.mxu_m_tile,
            mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
            mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
            activation_model=opts.activation_model,
            fp8_to_bf16_ratio=fp8_ratio,
            chip=chip,
        ),
        chip,
        collective_latency_s=opts.collective_latency_s,
        topology=opts.topology,
        parallel=parallel,
    )


@lru_cache(maxsize=64)
def _draft_geom_for_spec(geometry: ModelGeometry) -> ModelGeometry:
    """Build single-layer draft geometry for one speculative MTP pass."""
    draft_layer_type = (
        "dsa"
        if (geometry.kv_lora_rank > 0 and geometry.index_topk > 0)
        else ("mla" if geometry.kv_lora_rank > 0 else "full_attention")
    )
    use_dense_mtp = geometry.mtp_inter > 0
    return replace(
        geometry,
        num_layers=1,
        layer_types=(draft_layer_type,),
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        num_experts=0 if use_dense_mtp else geometry.num_experts,
        top_k=0 if use_dense_mtp else geometry.top_k,
        moe_inter=0 if use_dense_mtp else geometry.moe_inter,
        shared_inter=0 if use_dense_mtp else geometry.shared_inter,
        dense_inter=geometry.mtp_inter if use_dense_mtp else geometry.dense_inter,
        first_k_dense_replace=0,
        ffn_kinds=(
            ("dense",) if (use_dense_mtp or geometry.num_experts <= 0) else ("moe",)
        ),
        n_moe_layers=None,
        n_dense_layers=None,
        compress_ratios=(),
        indexer_types=(
            ("shared",)
            if (draft_layer_type == "dsa" and geometry.index_share_for_mtp_iteration)
            else ()
        ),
        mtp_layers=0,
        ple_dim=0,
        ple_layer_ids=(),
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def spec_cycle_floor(
    geometry: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    spec: SpecShape,
    options: FloorOptions | None = None,
) -> StepFloor:
    """Compute the floor for one speculative cycle (verify + k draft passes)."""
    opts = options or FloorOptions()
    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
    verify_shape = StepShape(
        batch=spec.batch,
        rows_per_seq=spec.spec_k + 1,
        context=spec.context,
        state_writes=1,
        shared_prefix_tokens=spec.shared_prefix_tokens,
    )
    verify_mods = list(
        pass_costs(
            geometry,
            parallel,
            verify_shape,
            opts.dtypes,
            mxu_m_tile=chip.mxu_m_tile,
            mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
            mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
            activation_model=opts.activation_model,
            fp8_to_bf16_ratio=fp8_ratio,
            chip=chip,
        )
    )
    if spec.spec_k <= 0:
        return combine_modules(
            tuple(verify_mods),
            chip,
            collective_latency_s=opts.collective_latency_s,
            topology=opts.topology,
            parallel=parallel,
        )

    draft_geom = _draft_geom_for_spec(geometry)
    draft_shape = StepShape(
        batch=spec.batch,
        rows_per_seq=1,
        context=spec.context,
        state_writes=1,
        shared_prefix_tokens=spec.shared_prefix_tokens,
    )
    draft_parallel = replace(parallel, pp=1)
    one_draft = pass_costs(
        draft_geom,
        draft_parallel,
        draft_shape,
        opts.dtypes,
        mxu_m_tile=chip.mxu_m_tile,
        mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
        activation_model=opts.activation_model,
        fp8_to_bf16_ratio=fp8_ratio,
        chip=chip,
    )

    fc_elems = DRAFT_PROJECTION_HIDDEN_VECTORS * float(
        geometry.hidden * geometry.hidden // parallel.tp
    )
    fc_buckets = _route_matmul_flops(
        spec.batch, fc_elems, opts.dtypes.w_aux, opts.dtypes.act, chip=chip
    )
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else chip.cores_per_chip
    )
    coll_cfg = CollectiveConfig(parallel.ar_algo, eff_ranks)
    draft_act_bytes = activation_wire_bytes(opts.dtypes.act)
    fc_unit = (
        make(
            "draft_fc_ag",
            "ag",
            float(spec.batch * geometry.hidden) * draft_act_bytes,
            parallel.tp,
            coll_cfg,
        )
        if parallel.tp > 1
        else None
    )
    draft_fc = ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * opts.dtypes.w_aux * spec.spec_k,
        fp8_flops=fc_buckets.fp8 * spec.spec_k,
        bf16_flops=fc_buckets.bf16 * spec.spec_k,
        fp4_flops=fc_buckets.fp4 * spec.spec_k,
        int8_flops=fc_buckets.int8 * spec.spec_k,
        int4_flops=fc_buckets.int4 * spec.spec_k,
        collectives=(fc_unit,) * spec.spec_k if fc_unit is not None else (),
        unique_collectives=((fc_unit, spec.spec_k),) if fc_unit is not None else (),
    )
    verify_mods.extend(
        ModuleCost(
            name=f"draft_{module.name}",
            hbm_bytes=module.hbm_bytes * spec.spec_k,
            fp8_flops=module.fp8_flops * spec.spec_k,
            bf16_flops=module.bf16_flops * spec.spec_k,
            fp4_flops=module.fp4_flops * spec.spec_k,
            int8_flops=module.int8_flops * spec.spec_k,
            int4_flops=module.int4_flops * spec.spec_k,
            collectives=module.collectives * spec.spec_k,
            hbm_bytes_min=(
                None
                if module.hbm_bytes_min is None
                else module.hbm_bytes_min * spec.spec_k
            ),
            unique_collectives=(
                None
                if module.unique_collectives is None
                else tuple(
                    (collective, cnt * spec.spec_k)
                    for collective, cnt in module.unique_collectives
                )
            ),
            fp8_flops_min=(
                None
                if module.fp8_flops_min is None
                else module.fp8_flops_min * spec.spec_k
            ),
            bf16_flops_min=(
                None
                if module.bf16_flops_min is None
                else module.bf16_flops_min * spec.spec_k
            ),
            fp4_flops_min=(
                None
                if module.fp4_flops_min is None
                else module.fp4_flops_min * spec.spec_k
            ),
            int8_flops_min=(
                None
                if module.int8_flops_min is None
                else module.int8_flops_min * spec.spec_k
            ),
            int4_flops_min=(
                None
                if module.int4_flops_min is None
                else module.int4_flops_min * spec.spec_k
            ),
            vector_flops=module.vector_flops * spec.spec_k,
            strict_vector_flops=module.strict_vector_flops * spec.spec_k,
            hbm_bytes_envelope=(
                None
                if module.hbm_bytes_envelope is None
                else module.hbm_bytes_envelope * spec.spec_k
            ),
            fp8_flops_envelope=(
                None
                if module.fp8_flops_envelope is None
                else module.fp8_flops_envelope * spec.spec_k
            ),
            bf16_flops_envelope=(
                None
                if module.bf16_flops_envelope is None
                else module.bf16_flops_envelope * spec.spec_k
            ),
            fp4_flops_envelope=(
                None
                if module.fp4_flops_envelope is None
                else module.fp4_flops_envelope * spec.spec_k
            ),
            int8_flops_envelope=(
                None
                if module.int8_flops_envelope is None
                else module.int8_flops_envelope * spec.spec_k
            ),
            int4_flops_envelope=(
                None
                if module.int4_flops_envelope is None
                else module.int4_flops_envelope * spec.spec_k
            ),
            strict_collectives=(
                None
                if module.strict_collectives is None
                else tuple(
                    (collective, cnt * spec.spec_k)
                    for collective, cnt in module.strict_collectives
                )
            ),
        )
        for module in one_draft
    )
    verify_mods.append(draft_fc)
    return combine_modules(
        tuple(verify_mods),
        chip,
        collective_latency_s=opts.collective_latency_s,
        topology=opts.topology,
        parallel=parallel,
    )
