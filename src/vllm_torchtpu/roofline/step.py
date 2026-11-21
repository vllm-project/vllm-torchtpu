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
    collective_cut_bound_s,
    collective_ici_hops,
    make,
    submesh_for_group,
    torus_collective_s,
)
from .constants import (
    ACTIVATION_BYTES_BF16,
    DRAFT_PROJECTION_HIDDEN_VECTORS,
    FP8_BYTES,
    STEP_FLOOR_CACHE_MAXSIZE,
    UNIT_COLL_CACHE_MAXSIZE,
    VECTOR_MAX_ELEMENTS_PER_WORD,
)
from .geometry import ModelGeometry, layer_ffn_kinds
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    _MatmulBuckets,
    _route_matmul_flops,
    _strict_sub_byte_unpack_vector_flops,
    _sub_byte_unpack_vector_flops,
    pass_costs,
)
from .occupancy import ExpertActivationModel

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology
    from .collectives import Collective

type _CollKey = tuple[str, float, int, float, float, int]


@dataclass(frozen=True, slots=True)
class FloorOptions:
    """Optional dtype, latency, topology, and activation parameters for step_floor."""

    dt: Dtypes = field(default_factory=Dtypes)
    collective_latency_s: float = 0.0
    topology: SliceTopology | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )


@dataclass(frozen=True, slots=True)
class UnpaddedFloorOptions:
    """Optional dtype, topology, and spec options for unpadded strict ceilings."""

    dt: Dtypes = field(default_factory=Dtypes)
    topology: SliceTopology | None = None
    spec_k: int = 0
    zero_weights: bool = False


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


def _precision_time_s(
    buckets: _MatmulBuckets,
    chip: ChipSpec,
    *,
    scale: float = 1.0,
) -> float:
    """Sum per-precision matrix times across all hardware precision peaks."""
    pairs = (
        ("fp8", buckets.fp8, chip.fp8_flops),
        ("bf16", buckets.bf16, chip.bf16_flops),
        ("int8", buckets.int8, chip.int8_flops),
        ("fp4", buckets.fp4, chip.fp4_flops),
        ("int4", buckets.int4, chip.int4_flops),
    )
    return sum(
        _div_peak(flops * scale, peak, precision=name) for name, flops, peak in pairs
    )


def _module_nominal_buckets(module: ModuleCost) -> _MatmulBuckets:
    """Return nominal per-precision FLOP buckets for one module."""
    return _MatmulBuckets(
        fp8=module.fp8_flops,
        bf16=module.bf16_flops,
        int8=module.int8_flops,
        fp4=module.fp4_flops,
        int4=module.int4_flops,
    )


def _module_strict_buckets(module: ModuleCost) -> _MatmulBuckets:
    """Return strict lower-bound per-precision FLOP buckets for one module."""
    return _MatmulBuckets(
        fp8=module.strict_fp8_flops,
        bf16=module.strict_bf16_flops,
        int8=module.strict_int8_flops,
        fp4=module.strict_fp4_flops,
        int4=module.strict_int4_flops,
    )


def _module_envelope_buckets(module: ModuleCost) -> _MatmulBuckets:
    """Return activation-envelope per-precision FLOP buckets for one module."""
    return _MatmulBuckets(
        fp8=module.envelope_fp8_flops,
        bf16=module.envelope_bf16_flops,
        int8=module.envelope_int8_flops,
        fp4=module.envelope_fp4_flops,
        int4=module.envelope_int4_flops,
    )


def module_serial_s(
    module: ModuleCost,
    chip: ChipSpec,
    *,
    matmul_scale: float = 1.0,
    vector_scale: float = 1.0,
) -> float:
    """Return single-module serial time across HBM, matmul, and vector units."""
    return max(
        module.hbm_bytes / chip.hbm_bw,
        _precision_time_s(_module_nominal_buckets(module), chip, scale=matmul_scale),
        (module.vector_flops * vector_scale) / chip.vector_flops,
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
    tot_nom = _MatmulBuckets()
    tot_min = _MatmulBuckets()
    tot_vector = 0.0
    tot_vector_min = 0.0
    tot_ici_b = 0.0
    mod_serial_s = 0.0
    vector_peak = chip.vector_flops
    for m in modules:
        tot_hbm_b += m.hbm_bytes
        tot_hbm_min_b += m.strict_hbm_bytes
        tot_nom = tot_nom + _module_nominal_buckets(m)
        tot_min = tot_min + _module_strict_buckets(m)
        tot_vector += m.vector_flops
        tot_vector_min += m.strict_vector_flops
        if has_finite_ici:
            tot_ici_b += m.ici_bytes
        mod_serial_s += module_serial_s(m, chip)
    hbm_s = tot_hbm_b / chip.hbm_bw
    matmul_s = _precision_time_s(tot_nom, chip)
    vector_s = tot_vector / vector_peak
    ici_s = tot_ici_b / chip.ici_bw if has_finite_ici else 0.0
    hbm_bound_s = tot_hbm_min_b / chip.hbm_bw
    matmul_bound_s = _precision_time_s(tot_min, chip)
    # Strict vector counts are per element; a word holds up to
    # VECTOR_MAX_ELEMENTS_PER_WORD packed elements, so this rate bounds every dtype.
    vector_bound_s = tot_vector_min / (VECTOR_MAX_ELEMENTS_PER_WORD * vector_peak)
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
    total_bytes_envelope = sum(module.envelope_hbm_bytes for module in modules)
    tot_env = _MatmulBuckets()
    for module in modules:
        tot_env = tot_env + _module_envelope_buckets(module)
    return (
        total_bytes_envelope / chip.hbm_bw,
        _precision_time_s(tot_env, chip),
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
        c_hops = collective_ici_hops(c, topology, ranks_per_chip, c_axes)
        hop_lat_s += call_count * hop_s
        analytic_chain_s += call_count * anal_s
        coll_bound_s += call_count * cut_s
        if collective_latency_s > 0.0:
            user_chain_lat_s += call_count * (c_hops * collective_latency_s)
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
    )


def _scale_opt(val: float | None, k: int) -> float | None:
    """Scale an optional float metric by ``k`` draft steps."""
    return None if val is None else val * k


def _scale_draft_module(module: ModuleCost, k: int) -> ModuleCost:
    """Scale one draft module's per-step resources across ``k`` draft steps."""
    scaled_unique = (
        None
        if module.unique_collectives is None
        else tuple((c, cnt * k) for c, cnt in module.unique_collectives)
    )
    return ModuleCost(
        name=f"draft_{module.name}",
        hbm_bytes=module.hbm_bytes * k,
        fp8_flops=module.fp8_flops * k,
        bf16_flops=module.bf16_flops * k,
        fp4_flops=module.fp4_flops * k,
        int8_flops=module.int8_flops * k,
        int4_flops=module.int4_flops * k,
        collectives=module.collectives * k,
        hbm_bytes_min=_scale_opt(module.hbm_bytes_min, k),
        unique_collectives=scaled_unique,
        fp8_flops_min=_scale_opt(module.fp8_flops_min, k),
        bf16_flops_min=_scale_opt(module.bf16_flops_min, k),
        fp4_flops_min=_scale_opt(module.fp4_flops_min, k),
        int8_flops_min=_scale_opt(module.int8_flops_min, k),
        int4_flops_min=_scale_opt(module.int4_flops_min, k),
        vector_flops=module.vector_flops * k,
        strict_vector_flops=module.strict_vector_flops * k,
        hbm_bytes_envelope=_scale_opt(module.hbm_bytes_envelope, k),
        fp8_flops_envelope=_scale_opt(module.fp8_flops_envelope, k),
        bf16_flops_envelope=_scale_opt(module.bf16_flops_envelope, k),
        fp4_flops_envelope=_scale_opt(module.fp4_flops_envelope, k),
        int8_flops_envelope=_scale_opt(module.int8_flops_envelope, k),
        int4_flops_envelope=_scale_opt(module.int4_flops_envelope, k),
    )


def _draft_geom_for_spec(geom: ModelGeometry) -> ModelGeometry:
    """Construct a single-layer ModelGeometry representing one MTP draft pass."""
    draft_layer_type = (
        "dsa"
        if (geom.kv_lora_rank > 0 and geom.index_topk > 0)
        else ("mla" if geom.kv_lora_rank > 0 else "full_attention")
    )
    use_dense_mtp = geom.mtp_inter > 0
    return replace(
        geom,
        num_layers=1,
        layer_types=(draft_layer_type,),
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        num_experts=0 if use_dense_mtp else geom.num_experts,
        top_k=0 if use_dense_mtp else geom.top_k,
        moe_inter=0 if use_dense_mtp else geom.moe_inter,
        shared_inter=0 if use_dense_mtp else geom.shared_inter,
        dense_inter=geom.mtp_inter if use_dense_mtp else geom.dense_inter,
        first_k_dense_replace=0,
        ffn_kinds=(
            ("dense",) if (use_dense_mtp or geom.num_experts <= 0) else ("moe",)
        ),
        n_moe_layers=None,
        n_dense_layers=None,
        compress_ratios=(),
        mtp_layers=0,
        ple_dim=0,
    )


def _draft_fc_module(
    geom: ModelGeometry,
    parallel: Parallel,
    batch: int,
    dt: Dtypes,
    spec_k: int,
    *,
    chip: ChipSpec,
) -> ModuleCost:
    """Construct the ModuleCost for spec_k speculative draft projection passes."""
    fc_elems = DRAFT_PROJECTION_HIDDEN_VECTORS * float(
        geom.hidden * geom.hidden // parallel.tp
    )
    fc_buckets = _route_matmul_flops(batch, fc_elems, dt.w_aux, dt.act, chip=chip)
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else chip.cores_per_chip
    )
    coll_cfg = CollectiveConfig(parallel.ar_algo, eff_ranks)
    fc_unit = (
        make(
            "draft_fc_ag",
            "ag",
            float(batch * geom.hidden) * ACTIVATION_BYTES_BF16,
            parallel.tp,
            coll_cfg,
        )
        if parallel.tp > 1
        else None
    )
    return ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * dt.w_aux * spec_k,
        fp8_flops=fc_buckets.fp8 * spec_k,
        bf16_flops=fc_buckets.bf16 * spec_k,
        fp4_flops=fc_buckets.fp4 * spec_k,
        int8_flops=fc_buckets.int8 * spec_k,
        int4_flops=fc_buckets.int4 * spec_k,
        collectives=(fc_unit,) * spec_k if fc_unit is not None else (),
        unique_collectives=((fc_unit, spec_k),) if fc_unit is not None else (),
    )


def _strip_moe_routed_unpack(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    vector_m: float,
    *,
    chip: ChipSpec | None = None,
) -> float:
    """Remove per-step routed expert unpack vector FLOPs when stripping weights."""
    if dt.w_expert >= FP8_BYTES:
        return vector_m
    is_ep = parallel.moe == "ep" and parallel.ep > 1
    ep_ranks = parallel.ep if is_ep else parallel.tp
    n_m = layer_ffn_kinds(geom).count("moe")
    pp_s = (
        float(math.ceil(geom.num_layers / max(1, parallel.pp))) / float(geom.num_layers)
        if geom.num_layers > 0 and parallel.pp > 1
        else 1.0
    )
    ed_min = (
        float(math.ceil(min(geom.num_experts, geom.top_k) / ep_ranks))
        if is_ep
        else float(min(geom.num_experts, geom.top_k))
    )
    width = (
        geom.moe_inter // max(1, parallel.tp // max(1, ep_ranks))
        if (parallel.moe == "ep" and ep_ranks < parallel.tp)
        else (geom.moe_inter if parallel.moe == "ep" else geom.moe_inter // parallel.tp)
    )
    eh = geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
    r_elems_min = ed_min * float(n_m * pp_s * geom.ffn_num_matrices * eh * width)
    unpack_vector = _sub_byte_unpack_vector_flops(
        r_elems_min,
        dt.w_expert,
        chip=chip,
        weight_dtype=dt.w_expert_dtype,
        act=dt.act,
    )
    return max(0.0, vector_m - _strict_sub_byte_unpack_vector_flops(unpack_vector))


def _zero_weight_strict_hbm_vector(
    module: ModuleCost,
    zero_map: dict[str, ModuleCost],
    step_ctx: tuple[ModelGeometry, Parallel, Dtypes, ChipSpec],
) -> tuple[float, float]:
    """Return strict (hbm_bytes, vector_flops) with weight-loading work subtracted."""
    geom, parallel, dt, chip = step_ctx
    zero_mod = zero_map.get(module.name)
    if module.name in ("moe", "mlp", "lm_head"):
        hbm_m = 0.0
    else:
        zero_hbm = zero_mod.strict_hbm_bytes if zero_mod is not None else 0.0
        hbm_m = max(0.0, module.strict_hbm_bytes - zero_hbm)
    zero_vector = zero_mod.strict_vector_flops if zero_mod is not None else 0.0
    vector_m = max(0.0, module.strict_vector_flops - zero_vector)
    if module.name == "moe":
        vector_m = _strip_moe_routed_unpack(geom, parallel, dt, vector_m, chip=chip)
    return hbm_m, vector_m


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def unpadded_strict_resource_times(
    geom: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    shape: StepShape,
    options: UnpaddedFloorOptions | None = None,
) -> tuple[float, float, float, float]:
    """Return strict unpadded-linear (hbm_s, matmul_s, vector_s, coll_s) bounds."""
    opts = options or UnpaddedFloorOptions()
    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0

    def _eval_mods(
        target_geom: ModelGeometry, target_shape: StepShape
    ) -> tuple[ModuleCost, ...]:
        """Evaluate unpadded module resource costs for one step shape."""
        return pass_costs(
            target_geom,
            parallel,
            target_shape,
            opts.dt,
            mxu_m_tile=1,
            mxu_lhs_rows_bf16=1,
            mxu_lhs_rows_fp8=1,
            fp8_to_bf16_ratio=fp8_ratio,
            unpadded_linear=True,
            chip=chip,
        )

    mods: list[ModuleCost] = list(_eval_mods(geom, shape))
    if opts.spec_k > 0 and shape.batch > 0 and shape.prefill_tokens == 0:
        draft_geom = _draft_geom_for_spec(geom)
        draft_shape = StepShape(
            batch=shape.batch, rows_per_seq=1, context=shape.context, state_writes=1
        )
        one_draft = _eval_mods(draft_geom, draft_shape)
        mods.extend(_scale_draft_module(mod, opts.spec_k) for mod in one_draft)
        mods.append(
            _draft_fc_module(
                geom, parallel, shape.batch, opts.dt, opts.spec_k, chip=chip
            )
        )

    zero_map: dict[str, ModuleCost] = {}
    if opts.zero_weights:
        zero_shape = StepShape(batch=0, rows_per_seq=1, context=0, prefill_tokens=0)
        zero_map = {mod.name: mod for mod in _eval_mods(geom, zero_shape)}

    tot_hbm_min_b = 0.0
    tot_fp8_min = 0.0
    tot_bf16_min = 0.0
    tot_int8_min = 0.0
    tot_fp4_min = 0.0
    tot_int4_min = 0.0
    tot_vector_min = 0.0
    for mod in mods:
        if opts.zero_weights:
            hbm_m, vector_m = _zero_weight_strict_hbm_vector(
                mod, zero_map, (geom, parallel, opts.dt, chip)
            )
        else:
            hbm_m = mod.strict_hbm_bytes
            vector_m = mod.strict_vector_flops
        tot_hbm_min_b += hbm_m
        tot_fp8_min += mod.strict_fp8_flops
        tot_bf16_min += mod.strict_bf16_flops
        tot_int8_min += mod.strict_int8_flops
        tot_fp4_min += mod.strict_fp4_flops
        tot_int4_min += mod.strict_int4_flops
        tot_vector_min += vector_m

    sf_comb = combine_modules(
        tuple(mods),
        chip,
        collective_latency_s=0.0,
        topology=opts.topology,
        parallel=parallel,
    )
    matmul_s = _precision_time_s(
        _MatmulBuckets(
            fp8=tot_fp8_min,
            bf16=tot_bf16_min,
            int8=tot_int8_min,
            fp4=tot_fp4_min,
            int4=tot_int4_min,
        ),
        chip,
    )
    return (
        _div_peak(tot_hbm_min_b, chip.hbm_bw, precision="hbm"),
        matmul_s,
        _div_peak(
            tot_vector_min,
            VECTOR_MAX_ELEMENTS_PER_WORD * chip.vector_flops,
            precision="vector",
        ),
        sf_comb.coll_bound_s,
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
    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
    return combine_modules(
        pass_costs(
            geom,
            parallel,
            shape,
            opts.dt,
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
    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
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
            activation_model=opts.activation_model,
            fp8_to_bf16_ratio=fp8_ratio,
            chip=chip,
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

    draft_geom = _draft_geom_for_spec(geom)
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
        activation_model=opts.activation_model,
        fp8_to_bf16_ratio=fp8_ratio,
        chip=chip,
    )

    draft_fc = _draft_fc_module(geom, parallel, spec.batch, opts.dt, spec.k, chip=chip)
    verify_mods.extend(_scale_draft_module(m, spec.k) for m in one_draft)
    verify_mods.append(draft_fc)
    return combine_modules(
        tuple(verify_mods),
        chip,
        collective_latency_s=opts.collective_latency_s,
        topology=opts.topology,
        parallel=parallel,
    )
