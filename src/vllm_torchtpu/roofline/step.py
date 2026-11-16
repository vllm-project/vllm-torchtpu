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
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from .collectives import (
    CollectiveConfig,
    collective_cut_bound_s,
    collective_ici_hops,
    make,
    submesh_for_group,
    torus_collective_s,
)
from .constants import (
    ACTIVATION_BYTES_BF16,
    MATMUL_FLOP_PER_MAC,
    STEP_FLOOR_CACHE_MAXSIZE,
    UNIT_COLL_CACHE_MAXSIZE,
    VECTOR_MAX_ELEMENTS_PER_WORD,
)
from .geometry import ModelGeometry
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    _split_matmul_flops,
    pass_costs,
)

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology
    from .collectives import Collective

type _CollKey = tuple[str, float, int, float, float, int]


@dataclass(frozen=True, slots=True)
class FloorOptions:
    """Optional dtype, latency, and topology parameters for step_floor calls."""

    dt: Dtypes = field(default_factory=Dtypes)
    collective_latency_s: float = 0.0
    topology: SliceTopology | None = None


@dataclass(frozen=True, slots=True)
class SpecShape:
    """Batch, context window, and speculative lookahead length for spec_cycle_floor."""

    batch: int
    context: int = 0
    k: int = 0


@dataclass(frozen=True, slots=True)
class StepFloor:
    """Resource times, physical lower bound, and serial estimate for one step."""

    modules: tuple[ModuleCost, ...]
    hbm_s: float
    matmul_s: float
    vpu_s: float
    ici_s: float
    chain_s: float
    hard_floor_s: float
    serial_estimate_s: float
    binding: str
    lat_s: float = 0.0
    hbm_bound_s: float = 0.0
    matmul_bound_s: float = 0.0
    vpu_bound_s: float = 0.0
    coll_bound_s: float = 0.0
    latency_assumed_s: float = 0.0

    @property
    def resource_bound_s(self) -> float:
        """Maximum time across HBM, matmul, VPU, and collective cut bounds."""
        return max(
            self.hbm_bound_s,
            self.matmul_bound_s,
            self.vpu_bound_s,
            self.coll_bound_s,
        )

    @property
    def latency_bound_s(self) -> float:
        """Assumed hop-derived network latency in seconds (reported beside bound)."""
        return self.latency_assumed_s


@lru_cache(maxsize=UNIT_COLL_CACHE_MAXSIZE)
def _unit_coll_times(
    c: Collective,
    topology: SliceTopology | None,
    ranks_per_chip: int,
    c_axes: tuple[int, int, int, int] | None,
    chip: ChipSpec,
) -> tuple[float, float, float]:
    """Compute assumed hop latency, analytic duration, and cut bound."""
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
    c_hops = collective_ici_hops(c, topology, ranks_per_chip, c_axes)
    hop_s = c_hops * unit_hop_lat_s
    submesh = submesh_for_group(c.group, topology, ranks_per_chip, c_axes)
    cut_s = collective_cut_bound_s(c.kind, c.payload, submesh, chip, topology)
    anal_s = torus_collective_s(c.kind, c.payload, submesh, chip, topology)
    return (hop_s, anal_s, cut_s)


def _dedup_module_collectives(
    modules: tuple[ModuleCost, ...],
) -> tuple[dict[_CollKey, int], list[tuple[_CollKey, Collective]]]:
    """Aggregate collective call counts across all modules in a step."""
    coll_counts: dict[_CollKey, int] = {}
    coll_samples: list[tuple[_CollKey, Collective]] = []
    for m in modules:
        entries = (
            m.unique_collectives
            if m.unique_collectives is not None
            else tuple((c, 1) for c in m.collectives)
        )
        for c, m_cnt in entries:
            if c.group <= 1 or m_cnt <= 0:
                continue
            collective_key = (
                c.kind,
                c.payload,
                c.group,
                c.wire_bytes,
                c.offchip,
                c.hops,
            )
            if collective_key not in coll_counts:
                coll_counts[collective_key] = m_cnt
                coll_samples.append((collective_key, c))
            else:
                coll_counts[collective_key] += m_cnt
    return coll_counts, coll_samples


def _sum_module_resources(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
    *,
    has_finite_ici: bool,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Sum per-module HBM, FLOPs, VPU, ICI, and serial module execution times."""
    tot_hbm_b = 0.0
    tot_hbm_min_b = 0.0
    tot_fp8 = 0.0
    tot_bf16 = 0.0
    tot_fp8_min = 0.0
    tot_bf16_min = 0.0
    tot_vpu = 0.0
    tot_vpu_min = 0.0
    tot_ici_b = 0.0
    mod_serial_s = 0.0
    vpu_peak = chip.vpu_flops
    for m in modules:
        hbm_bytes = m.hbm_bytes
        fp8_flops = m.fp8_flops
        bf16_flops = m.bf16_flops
        vpu_flops = m.vpu_flops
        tot_hbm_b += hbm_bytes
        tot_hbm_min_b += m.strict_hbm_bytes
        tot_fp8 += fp8_flops
        tot_bf16 += bf16_flops
        tot_fp8_min += m.strict_fp8_flops
        tot_bf16_min += m.strict_bf16_flops
        tot_vpu += vpu_flops
        tot_vpu_min += m.strict_vpu_flops
        if has_finite_ici:
            tot_ici_b += m.ici_bytes
        mod_serial_s += max(
            hbm_bytes / chip.hbm_bw,
            fp8_flops / chip.fp8_flops + bf16_flops / chip.bf16_flops,
            vpu_flops / vpu_peak,
        )
    hbm_s = tot_hbm_b / chip.hbm_bw
    matmul_s = tot_fp8 / chip.fp8_flops + tot_bf16 / chip.bf16_flops
    vpu_s = tot_vpu / vpu_peak
    ici_s = tot_ici_b / chip.ici_bw if has_finite_ici else 0.0
    hbm_bound_s = tot_hbm_min_b / chip.hbm_bw
    matmul_bound_s = tot_fp8_min / chip.fp8_flops + tot_bf16_min / chip.bf16_flops
    # Strict vector counts are per element; a word holds up to
    # VECTOR_MAX_ELEMENTS_PER_WORD packed elements, so this rate bounds every dtype.
    vpu_bound_s = tot_vpu_min / (VECTOR_MAX_ELEMENTS_PER_WORD * vpu_peak)
    return (
        hbm_s,
        matmul_s,
        vpu_s,
        ici_s,
        hbm_bound_s,
        matmul_bound_s,
        vpu_bound_s,
        mod_serial_s,
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
        vpu_s,
        ici_s,
        hbm_bound_s,
        matmul_bound_s,
        vpu_bound_s,
        mod_serial_s,
    ) = _sum_module_resources(modules, chip, has_finite_ici=has_finite_ici)
    mesh_axes = eff_parallel.mesh_axes if eff_parallel is not None else None
    ranks_per_chip = (
        eff_parallel.ranks_per_chip if eff_parallel is not None else chip.cores_per_chip
    )
    use_torus = topology is not None or mesh_axes is not None

    call_counts, coll_samples = _dedup_module_collectives(modules)
    hop_lat_s = 0.0
    analytic_chain_s = 0.0
    coll_bound_s = 0.0
    user_chain_lat_s = 0.0
    for collective_key, c in coll_samples:
        call_count = call_counts[collective_key]
        c_axes = (
            mesh_axes
            if (eff_parallel is not None and c.group == eff_parallel.tp)
            else None
        )
        hop_s, anal_s, cut_s = _unit_coll_times(
            c, topology, ranks_per_chip, c_axes, chip
        )
        hop_lat_s += call_count * hop_s
        analytic_chain_s += call_count * anal_s
        coll_bound_s += call_count * cut_s
        if collective_latency_s > 0.0:
            user_chain_lat_s += call_count * (c.hops * collective_latency_s)
    chain_s = analytic_chain_s + user_chain_lat_s

    hard_floor_s = max(hbm_bound_s, matmul_bound_s, vpu_bound_s, coll_bound_s)
    serial_estimate_s = mod_serial_s + chain_s
    if hard_floor_s == hbm_bound_s:
        binding = "hbm"
    elif hard_floor_s == matmul_bound_s:
        binding = "matmul"
    elif hard_floor_s == vpu_bound_s:
        binding = "vpu"
    elif use_torus:
        binding = "chain"
    else:
        binding = "ici"
    return StepFloor(
        modules=modules,
        hbm_s=hbm_s,
        matmul_s=matmul_s,
        vpu_s=vpu_s,
        ici_s=ici_s,
        chain_s=chain_s,
        hard_floor_s=hard_floor_s,
        serial_estimate_s=serial_estimate_s,
        binding=binding,
        lat_s=hop_lat_s,
        hbm_bound_s=hbm_bound_s,
        matmul_bound_s=matmul_bound_s,
        vpu_bound_s=vpu_bound_s,
        coll_bound_s=coll_bound_s,
        latency_assumed_s=hop_lat_s,
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def step_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    shape: StepShape,
    options: FloorOptions | None = None,
) -> StepFloor:
    """Compute the floor for one decode, prefill, or mixed forward pass."""
    opts = options or FloorOptions()
    return combine_modules(
        pass_costs(
            geom,
            parallel,
            shape,
            opts.dt,
            mxu_m_tile=chip.mxu_m_tile,
            mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
            mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
        ),
        chip,
        collective_latency_s=opts.collective_latency_s,
        topology=opts.topology,
        parallel=parallel,
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def spec_cycle_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    spec: SpecShape,
    options: FloorOptions | None = None,
) -> StepFloor:
    """Compute the floor for one speculative cycle (verify + k draft passes)."""
    opts = options or FloorOptions()
    verify_shape = StepShape(
        batch=spec.batch,
        rows_per_seq=spec.k + 1,
        context=spec.context,
        state_writes=1,
    )
    verify_mods = list(
        pass_costs(
            geom,
            parallel,
            verify_shape,
            opts.dt,
            mxu_m_tile=chip.mxu_m_tile,
            mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
            mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
        )
    )
    if spec.k <= 0:
        return combine_modules(
            tuple(verify_mods),
            chip,
            collective_latency_s=opts.collective_latency_s,
            topology=opts.topology,
            parallel=parallel,
        )

    draft_layer_type = (
        "dsa"
        if (geom.kv_lora_rank > 0 and geom.index_topk > 0)
        else ("mla" if geom.kv_lora_rank > 0 else "full_attention")
    )
    use_dense_mtp = geom.mtp_inter > 0
    draft_geom = ModelGeometry(
        hidden=geom.hidden,
        num_layers=1,
        layer_types=(draft_layer_type,),
        num_heads=geom.num_heads,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        attn_output_gate=geom.attn_output_gate,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=geom.conv_kernel,
        num_experts=0 if use_dense_mtp else geom.num_experts,
        top_k=0 if use_dense_mtp else geom.top_k,
        moe_inter=0 if use_dense_mtp else geom.moe_inter,
        shared_inter=0 if use_dense_mtp else geom.shared_inter,
        vocab=geom.vocab,
        mtp_layers=geom.mtp_layers,
        dense_inter=geom.mtp_inter if use_dense_mtp else geom.dense_inter,
        q_lora_rank=geom.q_lora_rank,
        kv_lora_rank=geom.kv_lora_rank,
        qk_nope_head_dim=geom.qk_nope_head_dim,
        qk_rope_head_dim=geom.qk_rope_head_dim,
        v_head_dim=geom.v_head_dim,
        index_n_heads=geom.index_n_heads,
        index_head_dim=geom.index_head_dim,
        index_topk=geom.index_topk,
        sliding_window=geom.sliding_window,
        block_parallel=geom.block_parallel,
        mtp_inter=geom.mtp_inter,
        global_head_dim=geom.global_head_dim,
    )
    draft_shape = StepShape(
        batch=spec.batch, rows_per_seq=1, context=spec.context, state_writes=1
    )
    one_draft = pass_costs(
        draft_geom,
        parallel,
        draft_shape,
        opts.dt,
        mxu_m_tile=chip.mxu_m_tile,
        mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
    )

    fc_elems = MATMUL_FLOP_PER_MAC * float(geom.hidden * geom.hidden // parallel.tp)
    fc_fp8, fc_bf16 = _split_matmul_flops(
        spec.batch, fc_elems, opts.dt.w_aux, opts.dt.act
    )
    coll_cfg = CollectiveConfig(parallel.ar_algo, parallel.ranks_per_chip)
    fc_unit = (
        make(
            "draft_fc_ag",
            "ag",
            float(spec.batch * geom.hidden) * ACTIVATION_BYTES_BF16,
            parallel.tp,
            coll_cfg,
        )
        if parallel.tp > 1
        else None
    )
    draft_fc = ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * opts.dt.w_aux * spec.k,
        fp8_flops=fc_fp8 * spec.k,
        bf16_flops=fc_bf16 * spec.k,
        collectives=(fc_unit,) * spec.k if fc_unit is not None else (),
        unique_collectives=((fc_unit, spec.k),) if fc_unit is not None else (),
    )
    verify_mods.extend(
        ModuleCost(
            name=f"draft_{m.name}",
            hbm_bytes=m.hbm_bytes * spec.k,
            fp8_flops=m.fp8_flops * spec.k,
            bf16_flops=m.bf16_flops * spec.k,
            collectives=m.collectives * spec.k,
            hbm_bytes_min=(
                None if m.hbm_bytes_min is None else m.hbm_bytes_min * spec.k
            ),
            unique_collectives=(
                None
                if m.unique_collectives is None
                else tuple((c, cnt * spec.k) for c, cnt in m.unique_collectives)
            ),
            fp8_flops_min=(
                None if m.fp8_flops_min is None else m.fp8_flops_min * spec.k
            ),
            bf16_flops_min=(
                None if m.bf16_flops_min is None else m.bf16_flops_min * spec.k
            ),
            vpu_flops=m.vpu_flops * spec.k,
            strict_vpu_flops=m.strict_vpu_flops * spec.k,
        )
        for m in one_draft
    )
    verify_mods.append(draft_fc)
    return combine_modules(
        tuple(verify_mods),
        chip,
        collective_latency_s=opts.collective_latency_s,
        topology=opts.topology,
        parallel=parallel,
    )
