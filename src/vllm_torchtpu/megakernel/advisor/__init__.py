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
"""First-principles fusion-scope advisor for TPU megakernel evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from math import ceil, gcd, log2
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from .. import errors
from ..advisor.models import (
    DEPLOYMENT_PRESET_SOURCES,
    DEPLOYMENT_PRESETS,
    MODEL_PRESET_SOURCES,
    MODEL_PRESETS,
    REGISTERED_MODEL_ALIASES,
    SUPPORTED_MODEL_TYPES,
    VERIFIED_MEGAKERNEL_MODEL_TYPES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    UnsupportedArchitectureError,
    model_profile_from_hf_config,
    resolve_model_profile,
    supports_hf_config,
)
from .chips import (
    BUILTIN_CHIP_SPECS,
    REQUIRED_CHIP_SPEC_KEYS,
    TPU6E_TOPOLOGIES,
    TPU7X_TOPOLOGIES,
    ChipSpec,
    FieldSource,
    load_chip_spec_file,
    resolve_chip_spec,
    verify_chip_spec_against_jax,
)
from .report import (
    format_advisor_summary,
    format_regime_table,
    render_advisor_html,
    write_advisor_html,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

    from .. import exec_spec, spec

# Where the kernel boundary falls: one call per operator, per layer, per group
# of N layers (a model preset or a pipeline stage), or one call per step.
FusionScope = Literal["per_op", "single_layer", "layer_group", "whole_step"]
Phase = Literal["decode", "verify", "prefill", "mixed"]
# Where routed tokens are grouped by expert and where the data lives.
MoEMode = Literal["masked_on_chip", "grouped_on_chip", "grouped_in_hbm", "dense_mlp"]
ProvenanceTag = Literal["DERIVED", "ASSUMED", "MEASURED"]
ObjectiveMode = Literal["step", "throughput", "both"]

_MIN_VMEM_MIB = 2
_DEFAULT_USABLE_VMEM_MIB = 62
_DOUBLE_BUFFER_SLOTS = 2
_VALID_DMA_BUFFERS = (2, 3)
_GROUPED_MOE_MODES = frozenset({"grouped_on_chip"})
_DEFAULT_LAUNCH_GAP_FLOOR_US = 1.0
_DEFAULT_LAUNCH_GAP_VAL_US = 2.0
_DEFAULT_LAUNCH_GAP_UPPER_US = 8.0
_MEASUREMENTS_SCHEMA = "advisor.measurements.v1"
_CHIPS_PER_HOST: int = 8
_COLLECTIVE_RING_THRESHOLD_BYTES: int = 16 * 1024
_DEFAULT_DCN_HOP_LATENCY_S: float = 5.0e-6

SUPPORTED_ONCHIP_FAMILIES: frozenset[str] = frozenset(
    {"gqa", "mla", "dsa", "swa", "gdn", "kda", "dense_mlp", "moe", "mtp", "ple"}
)

_DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}


class ProvenanceMetric(TypedDict):
    """Single output metric value paired with its provenance tag, source, and band."""

    value: float | int | bool
    unit: str
    tag: ProvenanceTag
    source: str
    band: list[float]


@dataclass(frozen=True)
class RingPlan:
    """On-chip VMEM ring-buffer slot counts and bank widths for weight streaming."""

    dense_slots: int = 2
    expert_slots: int = 2
    dense_slot_bytes: int = 2 << 20
    expert_slot_bytes: int = 2 << 20
    compiler_spill_reserve_bytes: int = 6 << 20


def default_double_buffer_ring_plan(dma_buffers: int = 2) -> RingPlan:
    """Return the default n-slot weight-streaming RingPlan."""
    return RingPlan(dense_slots=dma_buffers, expert_slots=dma_buffers)


@dataclass(frozen=True)
class MeasurementsData:
    """Calibrated runtime measurements loaded from an `advisor.measurements.v1` file."""

    source_desc: str
    launch_gap_p50_us: float | None = None
    launch_gap_p90_us: float | None = None
    launch_gap_n: int = 0
    collective_hop_us: float | None = None
    collective_n: int = 0
    step_coverage: float | None = None
    step_time_us: float | None = None
    dcn_bw_gbps: float | None = None
    ring_plan: RingPlan | None = None


def _parse_collectives_hop_us(coll_raw: object) -> tuple[float | None, int]:
    """Extract minimum per-hop collective latency and count from collectives."""
    if type(coll_raw) is not list or not coll_raw:
        return None, 0
    hops_list: list[float] = []
    coll_n = 0
    for item in coll_raw:
        if type(item) is dict and "latency_us" in item:
            grp = max(2, int(item.get("group_size", 2)))
            hops = max(1, round(log2(grp)))
            hops_list.append(float(item["latency_us"]) / hops)
            coll_n += int(item.get("n", 1))
    return (min(hops_list) if hops_list else None), coll_n


def _parse_ring_plan(rp_raw: object) -> RingPlan | None:
    """Parse an optional RingPlan mapping from measurements JSON."""
    if type(rp_raw) is not dict:
        return None
    return RingPlan(
        dense_slots=int(rp_raw.get("dense_slots", 2)),
        expert_slots=int(rp_raw.get("expert_slots", 2)),
        dense_slot_bytes=int(rp_raw.get("dense_slot_bytes", 2 << 20)),
        expert_slot_bytes=int(rp_raw.get("expert_slot_bytes", 2 << 20)),
        compiler_spill_reserve_bytes=int(
            rp_raw.get("compiler_spill_reserve_bytes", 6 << 20)
        ),
    )


def load_measurements(
    target: object,
) -> MeasurementsData:
    """Load and validate an `advisor.measurements.v1` JSON file or mapping."""
    if type(target) is MeasurementsData:
        return target
    if type(target) is dict:
        payload: Mapping[str, object] = target
        file_label = "inline_mapping"
    else:
        path = Path(str(target))
        if not path.is_file():
            msg = f"Missing measurements file: {path}"
            raise errors.MegakernelConfigError(msg)
        payload = json.loads(path.read_text("utf-8"))
        file_label = path.name
    if payload.get("schema") != _MEASUREMENTS_SCHEMA:
        msg = (
            f"Unsupported measurements schema {payload.get('schema')!r}; "
            f"expected {_MEASUREMENTS_SCHEMA!r}"
        )
        raise errors.MegakernelConfigError(msg)
    src_raw = payload.get("source")
    src_map: Mapping[str, object] = src_raw if type(src_raw) is dict else {}
    src_desc = f"{src_map.get('kind', 'trace')}:{src_map.get('file', file_label)}"

    lg_raw = payload.get("launch_gap_us")
    lg_map: Mapping[str, object] = lg_raw if type(lg_raw) is dict else {}
    lg_p50 = float(str(lg_map["p50"])) if "p50" in lg_map else None
    lg_p90 = (
        float(str(lg_map.get("p90", lg_p50 or 0.0))) if lg_p50 is not None else None
    )
    lg_n = int(str(lg_map.get("n", 0))) if lg_p50 is not None else 0

    coll_hop, coll_n = _parse_collectives_hop_us(payload.get("collectives"))

    cov_raw = payload.get("step_coverage")
    cov_map: Mapping[str, object] = cov_raw if type(cov_raw) is dict else {}
    cov_val = (
        float(str(cov_map["fused_eligible_fraction"]))
        if "fused_eligible_fraction" in cov_map
        else None
    )

    st_raw = payload.get("step_time_us")
    st_map: Mapping[str, object] = st_raw if type(st_raw) is dict else {}
    st_val = (
        float(str(st_map["p50"]))
        if "p50" in st_map
        else (float(str(st_raw)) if type(st_raw) in (int, float) else None)
    )
    dcn_raw = payload.get("dcn_bw_gbps")
    dcn_val = float(str(dcn_raw)) if type(dcn_raw) in (int, float) else None

    return MeasurementsData(
        source_desc=src_desc,
        launch_gap_p50_us=lg_p50,
        launch_gap_p90_us=lg_p90,
        launch_gap_n=lg_n,
        collective_hop_us=coll_hop,
        collective_n=coll_n,
        step_coverage=cov_val,
        step_time_us=st_val,
        dcn_bw_gbps=dcn_val,
        ring_plan=_parse_ring_plan(payload.get("ring_plan")),
    )


@dataclass(frozen=True)
class WorkloadSpec:
    """Serving request geometry, VMEM budget, and multi-buffering configuration."""

    phase: Phase = "decode"
    batch: int = 16
    isl: int = 4096
    osl: int = 1024
    context: int = 0
    prefix_hit_rate: float = 0.0
    spec_tokens: int = 0
    chunk_size: int = 0
    microbatches: int = 0
    tp: int = 8
    pp: int = 1
    dp: int = 1
    ep: int = 1
    kv_split: int = 1
    dcp: int = 1
    pcp: int = 1
    segment_layers: int = 0
    # Layers fused per kernel call for the layer_group scope, in call order
    # (empty: the model has no shipped layer grouping). Must sum to num_layers.
    layer_groups: tuple[int, ...] = ()
    moe_mode: MoEMode = "masked_on_chip"
    vmem_alias: bool = True
    vmem_fit_headroom: bool = False
    tiles: exec_spec.Tiles | None = None
    op_overhead_us: float = 0.0
    vmem_budget_mib: float | None = None
    dma_buffers: int = 2
    ring_plan: RingPlan | None = None
    objective: ObjectiveMode = "step"


@dataclass(frozen=True)
class ScopeEstimate:
    """Roofline and VMEM capacity estimate for one fusion scope."""

    scope: FusionScope
    feasible: bool
    reason: str
    ops_per_step: int
    vmem_static_mib: float
    t_vmem_max: int
    n_splits: int
    w_step_gb: float
    w_crit_gb: float | None
    act_hbm_mb: float
    moe_flop_factor: float
    roof_floor_us: float
    step_bound_us: float
    break_even_op_overhead_us: float = 0.0
    step_upper_us: float = 0.0

    @property
    def max_vmem_tokens(self) -> int:
        """Return maximum active tokens per step fitting in a single VMEM pass."""
        return self.t_vmem_max

    @property
    def num_vmem_splits(self) -> int:
        """Return the number of VMEM token-dimension passes per step."""
        return self.n_splits


@dataclass(frozen=True)
class AdvisorReport:
    """First-principles fusion-scope recommendation for a single workload."""

    model: str
    chip: str
    phase: Phase
    active_tokens: int
    effective_context: int
    vmem_budget_mib: float
    hbm_fits: bool
    hbm_used_gb: float
    hbm_capacity_gb: float
    ridge_dense_flop_per_byte: float
    step_arithmetic_intensity: float
    asymptotic_decode_intensity: float | None
    recommended_scope: FusionScope
    build_megakernel: bool
    rationale: str
    scopes: tuple[ScopeEstimate, ...]
    per_token_vmem_bytes: int = 0
    ridge_tokens: int = 0
    recommended_moe_mode: str = "dense_mlp"
    dma_buffers: int = 2
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorReport to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ParallelismRecommendation:
    """Auto-derived balanced TPU parallelism layout for bare-minimum queries."""

    tp: int
    ep: int
    dp: int
    pp: int
    topology: str
    hbm_used_gb: float
    hbm_capacity_gb: float
    w_step_gb: float
    auto_selected: bool = True
    dcp: int = 1
    pcp: int = 1
    # "deployment preset" when taken from DEPLOYMENT_PRESETS, else empty.
    source: str = ""
    layer_groups: tuple[int, ...] = ()

    def label(self) -> str:
        """Return the layout as `TP8 x DCP4 x EP8`, omitting degree-1 axes."""
        return ParallelLayout(
            tp=self.tp, ep=self.ep, dp=self.dp, pp=self.pp, dcp=self.dcp, pcp=self.pcp
        ).label()


@dataclass(frozen=True)
class RegimeEntry:
    """Evaluation summary for one canonical serving request regime."""

    label: str
    phase: Phase
    active_tokens: int
    effective_context: int
    step_arithmetic_intensity: float
    max_vmem_tokens: int
    num_vmem_splits: int
    recommended_scope: FusionScope
    recommended_moe_mode: str
    step_bound_us: float
    rationale: str
    layout: str = ""
    step_upper_us: float = 0.0
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    # Floor, upper, launches, and passes of every scope, for the report ledger.
    scopes: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class DisaggHandoffSpec:
    """Per-request KV and recurrent state wire payload and DCN transfer floor."""

    isl: int
    kv_wire_mb: float
    state_wire_mb: float
    total_wire_mb: float
    dcn_bw_gbps: float
    dcn_bw_gbps_bracket: tuple[float, float]
    dcn_xfer_ms: float
    dcn_xfer_bracket_ms: tuple[float, float]
    dcn_provenance: Literal["ASSUMED", "MEASURED"] = "ASSUMED"


@dataclass(frozen=True)
class AdvisorSweepReport:
    """Multi-regime crossover report across request regimes."""

    model: str
    chip: str
    parallelism: ParallelismRecommendation
    vmem_budget_mib: float
    vmem_static_mib: float
    per_token_vmem_bytes: int
    vmem_breakdown_bytes: dict[str, int]
    max_vmem_tokens: int
    ridge_dense_flop_per_byte: float
    ridge_tokens: int
    regimes: tuple[RegimeEntry, ...]
    tailoring_guide: tuple[str, ...]
    dma_buffers: int = 2
    single_layer_vmem_tokens: int = 0
    prefill_parallelism: ParallelismRecommendation | None = None
    saturation_parallelism: ParallelismRecommendation | None = None
    disagg_handoff: DisaggHandoffSpec | None = None
    chip_sources: Mapping[str, str] = field(default_factory=dict)
    model_sources: Mapping[str, str] = field(default_factory=dict)
    chip_spec: ChipSpec | None = None
    model_profile: ModelProfile | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorSweepReport to a JSON-compatible dictionary."""
        return asdict(self)


def fit_vmem_budget_bytes(
    tiles: exec_spec.Tiles | None = None,
    ring_plan: RingPlan | None = None,
    chip_spec: ChipSpec | None = None,
) -> int:
    """Return usable VMEM budget bytes after subtracting the RingPlan spill reserve.

    Args:
        tiles: Optional execution tiles specifying vmem_mib.
        ring_plan: Optional RingPlan reserve to deduct.
        chip_spec: Optional target ChipSpec to derive usable VMEM limit.

    Returns:
        Usable VMEM budget in bytes.
    """
    rp = ring_plan or default_double_buffer_ring_plan()
    if tiles is not None and getattr(tiles, "vmem_mib", None):
        vmem_mib = tiles.vmem_mib
    elif chip_spec is not None:
        vmem_mib = chip_spec.vmem_usable_bytes >> 20
    else:
        vmem_mib = _DEFAULT_USABLE_VMEM_MIB
    return max(0, (vmem_mib << 20) - rp.compiler_spill_reserve_bytes)


def compute_all_reduce_transfer_cost(
    message_bytes: float,
    effective_tp: int,
    chip_spec: ChipSpec,
    dcn_bw_bytes_per_s: float | None = None,
    collective_latency_s: float | None = None,
) -> tuple[float, float, int, float]:
    """Computes transfer time, latency, hop count, and volume for All-Reduce.

    For intra-host domains (effective_tp <= 8), models optical ICI collective.
    For multi-host domains (effective_tp > 8), models XLA 2-level hierarchical
    collective: intra-host reduce-scatter/all-gather over ICI, plus inter-host
    reduction over DCN.

    Args:
        message_bytes: Payload size transferred in the collective, in bytes.
        effective_tp: Effective tensor-parallel rank count (e.g. tp // pp).
        chip_spec: Hardware chip specification providing ICI bandwidth and base latency.

    Returns:
        A tuple of (transfer_time_s, base_latency_s, hop_count, nominal_volume_bytes).
    """
    if effective_tp <= 1 or message_bytes <= 0.0:
        return 0.0, 0.0, 0, 0.0

    nominal_volume_bytes = 2.0 * (effective_tp - 1) / effective_tp * message_bytes
    hop_count = round(math.log2(effective_tp))
    coll_lat = (
        chip_spec.collective_latency_s
        if collective_latency_s is None
        else collective_latency_s
    )

    if effective_tp <= _CHIPS_PER_HOST:
        transfer_time_s = nominal_volume_bytes / chip_spec.ici_egress_bw_bytes_per_s
        base_latency_s = hop_count * coll_lat
        return transfer_time_s, base_latency_s, hop_count, nominal_volume_bytes

    # Multi-host hierarchical collective: intra-host ICI + inter-host DCN
    num_hosts = math.ceil(effective_tp / _CHIPS_PER_HOST)

    # 1. Intra-host ICI: reduce-scatter (7/8 * S) + all-gather (7/8 * S) across 8 chips
    intra_volume_bytes = 2.0 * (_CHIPS_PER_HOST - 1) / _CHIPS_PER_HOST * message_bytes
    intra_transfer_time_s = intra_volume_bytes / chip_spec.ici_egress_bw_bytes_per_s
    intra_latency_s = 3 * coll_lat

    # 2. Inter-host DCN: all-reduce of partitioned 1/8th slice across num_hosts
    dcn_bw = (
        (chip_spec.dcn_bw_gbps / 8.0) * 1e9
        if dcn_bw_bytes_per_s is None
        else dcn_bw_bytes_per_s
    )
    inter_slice_bytes = message_bytes / float(_CHIPS_PER_HOST)
    inter_volume_bytes = 2.0 * (num_hosts - 1) / num_hosts * inter_slice_bytes
    inter_transfer_time_s = inter_volume_bytes / max(1.0, dcn_bw)
    inter_latency_s = round(math.log2(num_hosts)) * _DEFAULT_DCN_HOP_LATENCY_S

    total_transfer_time_s = intra_transfer_time_s + inter_transfer_time_s
    total_base_latency_s = intra_latency_s + inter_latency_s
    return total_transfer_time_s, total_base_latency_s, hop_count, nominal_volume_bytes


def compute_per_token_vmem_bytes(
    profile: ModelProfile, workload: WorkloadSpec
) -> dict[str, int]:
    """Compute per-token live VMEM bytes from tensor lifetimes and TP/EP sharding."""
    hidden = profile.hidden_size
    tp = max(1, workload.tp, workload.dcp if workload.tp == 1 else workload.tp)
    ep = max(1, workload.ep)
    kv_split = max(1, workload.kv_split, workload.dcp)

    # 1. On-chip residual buffer (F32 4*H), norm (BF16 2*H), collective staging (6*H)
    residual_buffer_bytes = 6 * hidden + (6 * hidden if tp > 1 else 0)

    # 2. Attention per_op per-token scratch (sharded across TP ranks)
    nh_local = max(1, ceil(profile.num_attention_heads / tp))
    if profile.kv_lora_rank > 0:
        attn_scratch_bytes = 2 * (
            profile.q_lora_rank + profile.kv_lora_rank + profile.qk_rope_head_dim
        ) + 6 * (nh_local * profile.head_dim)
    else:
        nkv_local = max(1, ceil(profile.num_kv_heads / max(1, tp // kv_split)))
        attn_scratch_bytes = 2 * (2 * nh_local + 2 * nkv_local) * profile.head_dim + (
            4 * nh_local * profile.head_dim
        )

    # 3. Linear attention (GDN / KDA) per-token scratch (sharded across TP ranks)
    if profile.num_linear_layers > 0 and profile.linear_value_heads > 0:
        kh_local = max(1, ceil(profile.linear_key_heads / tp))
        vh_local = max(1, ceil(profile.linear_value_heads / tp))
        linear_attn_scratch_bytes = 2 * (
            2 * kh_local * profile.linear_key_dim
            + 2 * vh_local * profile.linear_value_dim
        ) + 4 * (vh_local * profile.linear_value_dim)
    else:
        linear_attn_scratch_bytes = 0

    # 4. MoE / FFN per_op per-token scratch (sharded across TP and EP ranks)
    expert_hidden = profile.routed_expert_hidden_size or hidden
    if profile.num_moe_layers > 0 and profile.num_experts > 0:
        ep_tp = gcd(ep, tp)
        inter_local = ceil(profile.moe_intermediate_size / max(1, tp // ep_tp))
        shared_local = (
            ceil(profile.shared_expert_intermediate_size / tp)
            if profile.shared_expert_intermediate_size > 0
            else 0
        )
        router_scratch_bytes = 4 * profile.num_experts + 8 * profile.top_k
        is_grouped = workload.moe_mode in _GROUPED_MOE_MODES
        if is_grouped:
            # Streaming grouped MoE holds active expert FFN intermediates in fixed
            # tile scratch; at most _DOUBLE_BUFFER_SLOTS * ep streamed expert slices
            # (or 1 on single-rank EP=1) are live per token at once.
            eff_top_k = min(profile.top_k, _DOUBLE_BUFFER_SLOTS * ep) if ep > 1 else 1
            expert_tile_bytes = (
                8 * expert_hidden
                + ceil(eff_top_k * (6 * expert_hidden + 8 * inter_local) / ep)
                + 8 * shared_local
            )
        else:
            expert_tile_bytes = 16 * expert_hidden + 8 * (inter_local + shared_local)
        moe_scratch_bytes = (
            max(router_scratch_bytes + 2 * expert_hidden, expert_tile_bytes)
            if workload.vmem_alias
            else (router_scratch_bytes + expert_tile_bytes)
        )
    else:
        inter_local = ceil(profile.intermediate_size / tp)
        moe_scratch_bytes = 6 * hidden + 8 * inter_local

    per_op_scratch_bytes = (
        max(attn_scratch_bytes, linear_attn_scratch_bytes, moe_scratch_bytes)
        if workload.vmem_alias
        else (attn_scratch_bytes + linear_attn_scratch_bytes + moe_scratch_bytes)
    )
    total_per_token_bytes = residual_buffer_bytes + per_op_scratch_bytes
    return {
        "residual_buffer_bytes": residual_buffer_bytes,
        "attn_scratch_bytes": attn_scratch_bytes,
        "linear_attn_scratch_bytes": linear_attn_scratch_bytes,
        "moe_scratch_bytes": moe_scratch_bytes,
        "total_per_token_bytes": total_per_token_bytes,
        "b_spine": residual_buffer_bytes,
        "b_attn": attn_scratch_bytes,
        "b_lin": linear_attn_scratch_bytes,
        "b_moe": moe_scratch_bytes,
        "b_tok": total_per_token_bytes,
    }


def _validate_advise_inputs(
    profile: ModelProfile, workload: WorkloadSpec
) -> tuple[float, float, float, float]:
    """Validate ModelProfile and WorkloadSpec and return dtype byte widths."""
    num_layers = profile.num_layers
    tp, pp, dp, ep = workload.tp, workload.pp, workload.dp, workload.ep
    kv_split, dcp, pcp = workload.kv_split, workload.dcp, workload.pcp
    eff_tp = max(tp, dcp) if tp == 1 else tp
    pos_fields = (
        num_layers,
        profile.hidden_size,
        profile.num_attention_heads,
        profile.head_dim,
        workload.batch,
        tp,
        pp,
        dp,
        ep,
        kv_split,
        dcp,
        pcp,
        workload.isl,
    )
    nonneg_fields = (
        profile.num_attention_layers,
        profile.num_linear_layers,
        profile.num_moe_layers,
        workload.osl,
        workload.context,
        workload.spec_tokens,
        workload.chunk_size,
        profile.kv_lora_rank,
        workload.microbatches,
        workload.segment_layers,
        profile.q_lora_rank,
        profile.qk_rope_head_dim,
        profile.qk_nope_head_dim,
    )
    dtypes = (
        profile.weight_dtype,
        profile.expert_dtype,
        profile.kv_dtype,
        profile.state_dtype,
    )
    if min(*pos_fields) < 1 or workload.op_overhead_us < 0.0:
        msg = "Dimensions and parallelism must be >= 1"
        raise errors.MegakernelConfigError(msg)
    if workload.dma_buffers not in _VALID_DMA_BUFFERS:
        msg = f"dma_buffers must be 2 or 3, got {workload.dma_buffers!r}"
        raise errors.MegakernelConfigError(msg)
    if (
        workload.vmem_budget_mib is not None
        and workload.vmem_budget_mib < _MIN_VMEM_MIB
    ):
        msg = f"vmem_budget_mib must be >= {_MIN_VMEM_MIB}"
        raise errors.MegakernelConfigError(msg)
    parallel_ok = (
        min(*nonneg_fields) >= 0
        and (eff_tp * dp * pcp) % ep == 0
        and eff_tp % kv_split == 0
        and eff_tp % dcp == 0
        and pp <= num_layers
    )
    expert_ok = (
        profile.top_k == 0
        if profile.num_experts == 0
        else (
            1 <= profile.top_k <= profile.num_experts and profile.num_experts % ep == 0
        )
    )
    tiles = workload.tiles
    tiles_ok = 0.0 <= workload.prefix_hit_rate < 1.0 and not (
        tiles and tiles.vmem_mib < _MIN_VMEM_MIB
    )
    layers_ok = (
        profile.num_attention_layers + profile.num_linear_layers <= num_layers
        and profile.num_moe_layers <= num_layers
        and (
            not workload.layer_groups
            or (
                sum(workload.layer_groups) == num_layers
                and min(workload.layer_groups) >= 1
            )
        )
    )
    if not (
        parallel_ok
        and expert_ok
        and tiles_ok
        and layers_ok
        and all(d.lower() in _DTYPE_BYTES for d in dtypes)
    ):
        msg = "Invalid parallelism, layers, or dtype"
        raise errors.MegakernelConfigError(msg)
    bw_w, bw_e, bw_kv, bw_st = (_DTYPE_BYTES[d.lower()] for d in dtypes)
    return bw_w, bw_e, bw_kv, bw_st


def _scope_vmem_and_ops(
    scope: FusionScope,
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
) -> tuple[int, float, int, int, int, bool, str]:
    """Compute static VMEM, activation HBM bytes, op count, and token split count."""
    vmem_budget = ctx.vmem_budget
    single_layer_dense_staging = ctx.single_layer_dense_staging_bytes
    single_layer_expert_staging = ctx.single_layer_expert_staging_bytes
    whole_step_dense_staging = ctx.whole_step_dense_staging_bytes
    whole_step_expert_staging = ctx.whole_step_expert_staging_bytes
    gqa_scratch = ctx.gqa_scratch
    gdn_scratch = ctx.gdn_scratch
    ffn_scratch = ctx.ffn_scratch
    layers_local = ctx.layers_local
    num_segments = ctx.num_segments
    l_moe, l_attn, l_lin = ctx.l_moe, ctx.l_attn, ctx.l_lin
    active_tokens = ctx.active_tokens
    per_token_fused_bytes = ctx.per_token_fused_bytes
    token_residual_bytes = ctx.token_residual_bytes
    no_whole_step = ctx.no_whole_step
    plan_error = ctx.plan_error

    feasible, reason = True, "ok"
    preset_groups = ctx.num_layer_groups
    if scope == "layer_group" and preset_groups == 0 and workload.pp == 1:
        if num_segments == 1:
            feasible = False
            reason = (
                "No layer grouping: needs a model preset, pp > 1, or segment_layers"
            )
    elif scope == "whole_step" and no_whole_step:
        feasible = False
        reason = (
            plan_error
            or "Bounded by layer groups, single layers, or unmodeled layer family"
        )

    if scope == "per_op":
        static_vmem_bytes = max(1 << 20, single_layer_dense_staging)
        alpha = 4 * profile.hidden_size
        n_splits = 1
        t_max = max(active_tokens, (vmem_budget - static_vmem_bytes) // alpha)
        n_ops = 2 * layers_local
        act_bytes = 4.0 * layers_local * token_residual_bytes
        return static_vmem_bytes, act_bytes, n_ops, t_max, n_splits, feasible, reason

    is_single_layer = scope == "single_layer"
    is_single_or_preset_group = is_single_layer or (
        scope == "layer_group" and preset_groups > 0
    )
    mul = (preset_groups or num_segments) if scope == "layer_group" else 1
    dense_staging_bytes = (
        single_layer_dense_staging
        if is_single_or_preset_group
        else whole_step_dense_staging
    )
    expert_staging_bytes = (
        single_layer_expert_staging
        if is_single_or_preset_group
        else whole_step_expert_staging
    ) * (l_moe > 0)
    scratch_alias = (
        max(gqa_scratch, gdn_scratch, ffn_scratch)
        if is_single_layer
        else max(gqa_scratch + gdn_scratch, ffn_scratch)
    )
    moe_alignment_pad_bytes = (
        (min(12 << 20, expert_staging_bytes) if workload.tiles else (12 << 20))
        if l_moe > 0
        else (2 << 20)
    )
    pad_bytes = (
        0
        if is_single_or_preset_group
        else moe_alignment_pad_bytes + 2304 * profile.hidden_size * (l_moe == 0)
    )
    static_vmem_bytes = (
        dense_staging_bytes
        + expert_staging_bytes
        + (
            scratch_alias
            if workload.vmem_alias
            else gqa_scratch + gdn_scratch + ffn_scratch
        )
        + pad_bytes
    )
    act_bytes = 2.0 * (layers_local if is_single_layer else mul) * token_residual_bytes
    n_ops = (
        layers_local
        if is_single_layer
        else mul
        if scope == "layer_group" and preset_groups
        else (mul * (3 if ctx.eff_tp * workload.ep > 1 and l_attn + l_lin else 1))
    )
    alpha = per_token_fused_bytes
    t_max = (
        0
        if static_vmem_bytes + alpha > vmem_budget
        else max(1, (vmem_budget - static_vmem_bytes) // alpha)
    )
    n_splits = active_tokens if t_max == 0 else ceil(active_tokens / t_max)
    if t_max == 0:
        feasible, reason = False, "Weight buffers exceed VMEM"
    return static_vmem_bytes, act_bytes, n_ops, t_max, n_splits, feasible, reason


def _experts_hit(
    tokens_ep: int, top_k: int, num_experts: int, experts_local: int
) -> float:
    """Return expected distinct local experts hit by `tokens_ep` tokens."""
    miss = (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    return max(1.0, experts_local * (1.0 - miss))


def _moe_weights_and_padding(
    ctx: _ScopeContext,
    profile: ModelProfile,
    tile_tokens: int,
    *,
    tiled: bool,
    grouped: bool,
) -> tuple[float, float]:
    """Return weight bytes read per step and the MoE matmul padding factor."""
    experts_local, ep_dp = ctx.experts_local, ctx.ep_dp
    if experts_local == 0:
        return ctx.w_static, 1.0

    t_ep = tile_tokens * ep_dp
    e_hit = _experts_hit(t_ep, profile.top_k, profile.num_experts, experts_local)
    pairs = max(1.0, t_ep * profile.top_k * experts_local / profile.num_experts)
    avg_m = max(1.0, pairs / e_hit)
    hit = (
        _experts_hit(
            ctx.active_tokens * ep_dp,
            profile.top_k,
            profile.num_experts,
            experts_local,
        )
        if tiled
        else e_hit
    )
    w_step = ctx.w_static + ctx.l_moe * hit * ctx.exp_bytes
    if grouped:
        return w_step, ceil(avg_m / ctx.row_floor) * ctx.row_floor / avg_m
    rup_16 = ((t_ep + 15) // 16) * 16
    return w_step, max(1.0, (rup_16 * e_hit) / pairs)


def _evaluate_scopes(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    *,
    bf16_attention_upper: bool = True,
) -> tuple[list[ScopeEstimate], dict[FusionScope, float]]:
    """Evaluate every FusionScope against VMEM, HBM, and matmul bounds."""
    layers_local, l_moe, active_tokens = ctx.layers_local, ctx.l_moe, ctx.active_tokens
    token_residual_bytes, ep_dp = ctx.token_residual_bytes, ctx.ep_dp
    num_ar = ctx.num_ar
    lat_coll = ctx.lat_coll
    pp_stage_time, a2a_time = ctx.pp_stage_time, ctx.a2a_time
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    peak_w, peak_e, bw_hbm = ctx.peak_w, ctx.peak_e, ctx.bw_hbm
    kv_cache_read_bytes, recurrent_state_rw_bytes, first_slot_bytes = (
        ctx.kv_cache_read_bytes,
        ctx.recurrent_state_rw_bytes,
        ctx.first_slot_bytes,
    )

    estimates: list[ScopeEstimate] = []
    raw_step_map: dict[FusionScope, float] = {}
    sub_ops, sub_pure_us = 2 * layers_local, 0.0
    scopes_order: tuple[FusionScope, ...] = (
        "per_op",
        "single_layer",
        "layer_group",
        "whole_step",
    )

    for scope in scopes_order:
        (
            static_vmem_bytes,
            act_bytes,
            n_ops,
            t_max,
            n_splits,
            feasible,
            reason,
        ) = _scope_vmem_and_ops(scope, ctx, profile, workload)
        tiled = scope in ("whole_step", "layer_group") and n_splits > 1
        tile_tokens = ceil(active_tokens / max(1, n_splits))
        can_group_moe = not (scope in ("whole_step", "layer_group") and ep_dp > 1)
        grouped_moe = can_group_moe and (
            scope == "per_op"
            or workload.moe_mode in _GROUPED_MOE_MODES
            or tile_tokens * ep_dp >= ctx.row_floor
        )
        # In grouped MoE on whole_step/layer_group/per_op, token-expert sorting
        # spans the active call batch rather than a single VMEM sub-tile.
        moe_tokens = (
            active_tokens if grouped_moe and scope != "single_layer" else tile_tokens
        )
        w_step, phi = _moe_weights_and_padding(
            ctx,
            profile,
            moe_tokens,
            tiled=tiled,
            grouped=grouped_moe,
        )
        if tiled:
            act_bytes += 2.0 * layers_local * token_residual_bytes

        xla_ar_factor = 2 if scope == "per_op" and ctx.eff_tp > 1 else 1
        xla_hbm_coll_bytes = (
            2.0 * num_ar * token_residual_bytes
            if scope == "per_op" and ctx.eff_tp > 1
            else 0.0
        )
        t_comm = (
            num_ar
            * (ctx.ar_transfer_time_s + n_splits * xla_ar_factor * ctx.ar_base_lat_s)
            + xla_hbm_coll_bytes / bw_hbm
            + pp_stage_time
            + (
                l_moe * (a2a_time + 2 * n_splits * xla_ar_factor * lat_coll)
                if l_moe > 0 and ep_dp > 1
                else 0.0
            )
        )
        eff_state_bytes = kv_cache_read_bytes + recurrent_state_rw_bytes
        pipeline_fill_drain_s = (n_ops - 1) * (
            first_slot_bytes / bw_hbm + 2.0 * lat_coll
        )
        if tiled:
            tile_residual_bytes = tile_tokens * profile.hidden_size * 2.0
            pipeline_fill_drain_s += (
                (n_splits - 1)
                * layers_local
                * (tile_residual_bytes / bw_hbm + 2.0 * lat_coll)
            )
        t_matmul = flops_dense / peak_w + (phi * flops_moe) / peak_e
        t_matmul_upper = (
            (flops_dense - ctx.flops_kv) / peak_w
            + ctx.flops_kv / ctx.bf16_peak
            + (phi * flops_moe) / peak_e
            if bf16_attention_upper
            and workload.phase in ("prefill", "mixed")
            and ctx.flops_kv > 0
            else t_matmul
        )
        act_saved = 4.0 * layers_local * token_residual_bytes - act_bytes
        hbm_int = (1 if tiled else n_splits) * w_step + eff_state_bytes
        d_ops = sub_ops - n_ops
        roof_us = (
            max(t_matmul, (hbm_int + act_bytes) / bw_hbm)
            + t_comm
            + pipeline_fill_drain_s
        ) * 1e6
        base_ai = (flops_dense + flops_moe) / max(
            1.0, w_step + kv_cache_read_bytes + recurrent_state_rw_bytes
        )
        repass_spill_us = (
            (act_bytes + (n_splits - 1) * w_step) / bw_hbm * 1e6
            if base_ai >= peak_w / bw_hbm and n_splits > 1 and not tiled
            else 0.0
        )
        pure_us = roof_us + repass_spill_us
        if scope == "per_op":
            sub_pure_us = pure_us
        step_us = pure_us + n_ops * workload.op_overhead_us

        coll_extra_s = (
            num_ar * n_splits * xla_ar_factor * ctx.ar_intra_hops
            + (2 * l_moe * n_splits * xla_ar_factor if l_moe > 0 and ep_dp > 1 else 0)
        ) * max(0.0, ctx.lat_coll_upper - lat_coll)
        roof_upper_us = (
            max(t_matmul_upper, (hbm_int + act_bytes) / bw_hbm)
            + t_comm
            + coll_extra_s
            + pipeline_fill_drain_s
        ) * 1e6
        step_upper_us = max(
            step_us,
            roof_upper_us + repass_spill_us + n_ops * ctx.launch_gap_upper_us,
        )

        d_bytes = bw_hbm * max(0, d_ops) * workload.op_overhead_us * 1e-6
        break_even = (
            round(max(0.0, (pure_us - sub_pure_us) / d_ops), 3) if d_ops > 0 else 0.0
        )
        w_crit = (
            round(((act_saved + d_bytes) / (n_splits - 1)) / 1e9, 4)
            if n_splits > 1
            else None
        )
        raw_step_map[scope] = step_us
        estimates.append(
            ScopeEstimate(
                scope=scope,
                feasible=feasible,
                reason=reason,
                ops_per_step=n_ops,
                vmem_static_mib=round(static_vmem_bytes / (1 << 20), 3),
                t_vmem_max=t_max,
                n_splits=n_splits,
                w_step_gb=round(w_step / 1e9, 4),
                w_crit_gb=w_crit,
                act_hbm_mb=round(act_bytes / 1e6, 3),
                moe_flop_factor=round(phi, 3),
                roof_floor_us=round(roof_us, 2),
                step_bound_us=round(step_us, 2),
                break_even_op_overhead_us=break_even,
                step_upper_us=round(step_upper_us, 2),
            )
        )

    return estimates, raw_step_map


@dataclass(frozen=True)
class _ScopeContext:
    """Per-core weight, state, FLOP, collective, and VMEM terms for one workload."""

    vmem_budget: int
    single_layer_dense_staging_bytes: int
    single_layer_expert_staging_bytes: int
    whole_step_dense_staging_bytes: int
    whole_step_expert_staging_bytes: int
    first_slot_bytes: float
    gqa_scratch: int
    gdn_scratch: int
    ffn_scratch: int
    layers_local: int
    num_segments: int
    num_layer_groups: int
    l_moe: int
    l_attn: int
    l_lin: int
    active_tokens: int
    eff_ctx: int
    batch_local: int
    verify_tokens: int
    nh_local: int
    q_dim: int
    base_dense_params: float
    per_token_fused_bytes: int
    token_residual_bytes: float
    w_static: float
    exp_bytes: float
    experts_local: int
    ep_dp: int
    eff_tp: int
    row_floor: int
    num_ar: int
    ar_hops: int
    ar_intra_hops: int
    ar_inter_latency_s: float
    dcn_bytes_per_s: float
    ar_vol: float
    ar_transfer_time_s: float
    ar_base_lat_s: float
    egress_bw: float
    lat_coll: float
    lat_coll_upper: float
    launch_gap_upper_us: float
    pp_stage_time: float
    a2a_time: float
    flops_dense: float
    flops_kv: float
    flops_moe: float
    peak_w: float
    peak_e: float
    bf16_peak: float
    kv_cache_read_bytes: float
    recurrent_state_rw_bytes: float
    bw_hbm: float
    hbm_used: float
    no_whole_step: bool
    plan_error: str


def _vmem_budget_bytes(workload: WorkloadSpec, chip_spec: ChipSpec) -> int:
    """Return usable VMEM bytes: user budget, Tiles budget, or chip default."""
    vmem_usable = chip_spec.vmem_usable_bytes
    tiles = workload.tiles
    rp = workload.ring_plan or default_double_buffer_ring_plan(workload.dma_buffers)
    base_vmem_bytes = (
        int(workload.vmem_budget_mib * (1 << 20))
        if workload.vmem_budget_mib is not None
        else (tiles.vmem_mib << 20 if tiles else vmem_usable)
    )
    vmem_raw = (
        base_vmem_bytes - rp.compiler_spill_reserve_bytes * workload.vmem_fit_headroom
    )
    return min(vmem_usable, max(0, vmem_raw))


def _whole_step_staging(
    workload: WorkloadSpec, layer_dense_weight_bytes: int
) -> tuple[float, int, int]:
    """Return first DMA slot bytes and multi-layer dense and expert staging bytes."""
    tiles = workload.tiles
    if tiles is not None:
        dense_slot_bytes = tiles.bank
        dense_staging = tiles.nbd * tiles.bank
        expert_staging = tiles.nbe * tiles.ebank
    else:
        rp = workload.ring_plan or default_double_buffer_ring_plan(workload.dma_buffers)
        dense_slot_bytes = rp.dense_slot_bytes
        extra_dense_bytes = (
            rp.dense_slots - _DOUBLE_BUFFER_SLOTS
        ) * rp.dense_slot_bytes
        extra_expert_bytes = (
            rp.expert_slots - _DOUBLE_BUFFER_SLOTS
        ) * rp.expert_slot_bytes
        dense_staging = min(28 << 20, 2 * layer_dense_weight_bytes) + extra_dense_bytes
        expert_staging = (10 << 20) + extra_expert_bytes
    first_slot_bytes = float(
        min(dense_slot_bytes, max(1 << 18, layer_dense_weight_bytes // 4))
    )
    return first_slot_bytes, dense_staging, expert_staging


def _build_scope_context(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    dtypes_bw: tuple[float, float, float, float],
    plan_error: str,
    meas: MeasurementsData | None = None,
) -> _ScopeContext:
    """Assemble per-core weight, state, FLOP, and VMEM scope context."""
    bw_w, bw_e, bw_kv, bw_st = dtypes_bw
    num_layers, hidden, num_experts, top_k = (
        profile.num_layers,
        profile.hidden_size,
        profile.num_experts,
        profile.top_k,
    )
    tp, pp, dp, ep = workload.tp, workload.pp, workload.dp, workload.ep
    eff_tp, kv_split, spec_tok = (
        max(tp, workload.dcp) if tp == 1 else tp,
        max(workload.kv_split, workload.dcp),
        workload.spec_tokens,
    )
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1
    head_dim, kv_lora, q_lora, qk_rope = (
        profile.head_dim,
        profile.kv_lora_rank,
        profile.q_lora_rank,
        profile.qk_rope_head_dim,
    )

    uncached_isl = max(1, round(workload.isl * (1.0 - workload.prefix_hit_rate)))
    batch_local, chunk_tokens, layers_local = (
        ceil(workload.batch / dp),
        ceil((workload.chunk_size or uncached_isl) / eff_pcp),
        ceil(num_layers / pp),
    )
    eff_ctx = workload.context or (
        workload.isl + (0 if workload.phase == "prefill" else workload.osl // 2)
    )
    verify_tokens = (
        max(1, spec_tok) + 1
        if workload.phase == "verify"
        else (spec_tok + 1 if spec_tok > 0 else 1)
    )
    prefill_tokens = (
        ceil(workload.chunk_size / eff_pcp)
        if workload.chunk_size
        else ceil(
            batch_local * uncached_isl / max(1, (workload.microbatches or pp) * eff_pcp)
        )
    )
    decode_tokens = batch_local * (1 if workload.phase == "decode" else verify_tokens)
    active_tokens = (
        prefill_tokens
        if workload.phase == "prefill"
        else decode_tokens + chunk_tokens * (workload.phase == "mixed")
    )
    l_attn = min(
        layers_local, ceil(profile.num_attention_layers * layers_local / num_layers)
    )
    l_moe = min(layers_local, ceil(profile.num_moe_layers * layers_local / num_layers))
    l_lin = min(
        layers_local - l_attn,
        ceil(profile.num_linear_layers * layers_local / num_layers),
    )
    token_residual_bytes = active_tokens * hidden * 2.0

    qk_nope = profile.qk_nope_head_dim or head_dim
    kd, vd = profile.linear_key_dim, profile.linear_value_dim
    nh_local, kvh_local = (
        max(1, ceil(profile.num_attention_heads / eff_tp)),
        max(1, ceil(profile.num_kv_heads / max(1, eff_tp // kv_split))),
    )
    q_dim, q_tot = nh_local * head_dim, nh_local * (qk_nope + qk_rope)
    mla_proj_params = (
        q_lora * (hidden + q_tot) if q_lora else hidden * q_tot
    ) + hidden * (kv_lora + qk_rope + q_dim)
    attn_params = (
        mla_proj_params + kv_lora * nh_local * (qk_nope + head_dim)
        if kv_lora
        else hidden * 2 * (q_dim + kvh_local * head_dim)
    )
    kh_local = max(1, ceil(profile.linear_key_heads / eff_tp)) if l_lin > 0 else 0
    vh_local = max(1, ceil(profile.linear_value_heads / eff_tp)) if l_lin > 0 else 0
    z_dim, qkv_lin, ep_tp = (
        vh_local * vd,
        2 * kh_local * kd + vh_local * vd,
        gcd(ep, eff_tp),
    )
    linear_attn_params, experts_local = (
        hidden * (qkv_lin + 2 * z_dim) * (l_lin > 0),
        (num_experts // ep) * (l_moe > 0),
    )
    dense_ffn_params, shared_expert_params, ep_dp = (
        3 * hidden * max(1, ceil(profile.intermediate_size / eff_tp)),
        3 * hidden * ceil(profile.shared_expert_intermediate_size / eff_tp),
        ep // ep_tp,
    )
    i_exp_local = ceil(profile.moe_intermediate_size / (eff_tp // ep_tp)) * (l_moe > 0)
    routed_expert_params = (
        3 * (profile.routed_expert_hidden_size or hidden) * i_exp_local
    )
    base_dense_params = (
        l_attn * attn_params
        + l_lin * linear_attn_params
        + (layers_local - l_moe) * dense_ffn_params
        + l_moe * shared_expert_params
    )
    w_static, exp_bytes = (
        base_dense_params * bw_w + l_moe * hidden * num_experts * 2.0,
        routed_expert_params * bw_e,
    )

    kv_bytes_per_tok = (
        ((kv_lora + qk_rope) if kv_lora else 2.0 * kvh_local * head_dim)
        * bw_kv
        / kv_split
    )
    kv_cache_read_bytes = l_attn * batch_local * eff_ctx * kv_bytes_per_tok
    recurrent_state_rw_bytes = (
        2.0
        * l_lin
        * batch_local
        * (z_dim * kd + qkv_lin * profile.linear_conv_kernel)
        * bw_st
    )
    vmem_budget = _vmem_budget_bytes(workload, chip_spec)
    layer_dense_weight_bytes = int(
        (
            max(attn_params, linear_attn_params)
            + (shared_expert_params if l_moe > 0 else dense_ffn_params)
        )
        * bw_w
    )
    t_prefill_part = {"prefill": active_tokens, "mixed": chunk_tokens}.get(
        workload.phase, 0
    )
    d_mla = (active_tokens - t_prefill_part) * (
        2 * kv_lora + qk_rope
    ) + t_prefill_part * (qk_nope + qk_rope + head_dim)
    # Causal prefill attention: each of T new tokens attends to prefix + (1..T),
    # giving mean context prefix + (T + 1) / 2 instead of rectangular eff_ctx.
    prefix_ctx = max(0, eff_ctx - t_prefill_part)
    attn_ctx = (
        prefix_ctx + (t_prefill_part + 1) / 2.0
        if workload.phase == "prefill" and t_prefill_part > 1
        else float(eff_ctx)
    )
    flops_kv = (
        2.0
        * l_attn
        * attn_ctx
        * (nh_local * d_mla if kv_lora else 2.0 * active_tokens * q_dim)
    )
    flops_dense = (
        2.0 * active_tokens * (base_dense_params + l_moe * hidden * num_experts)
        + flops_kv
    )
    num_segments = (
        ceil(layers_local / workload.segment_layers)
        if 0 < workload.segment_layers < layers_local
        else 1
    )
    flops_moe = (
        2.0
        * active_tokens
        * ep_dp
        * l_moe
        * top_k
        * routed_expert_params
        * experts_local
        / num_experts
        if experts_local > 0
        else 0.0
    )

    buf_slots = workload.dma_buffers
    (
        first_slot_bytes,
        whole_step_dense_staging_bytes,
        whole_step_expert_staging_bytes,
    ) = _whole_step_staging(workload, layer_dense_weight_bytes)
    has_unmodeled_family = any(
        fam not in SUPPORTED_ONCHIP_FAMILIES for fam in profile.layer_families
    )
    dcn_bw_gbps = (
        meas.dcn_bw_gbps
        if meas is not None and meas.dcn_bw_gbps is not None
        else chip_spec.dcn_bw_gbps
    )
    dcn_bytes_per_s = (dcn_bw_gbps / 8.0) * 1e9
    ar_transfer_time_s, ar_base_lat_s, ar_hops, ar_vol = (
        compute_all_reduce_transfer_cost(
            token_residual_bytes,
            eff_tp,
            chip_spec,
            dcn_bw_bytes_per_s=dcn_bytes_per_s,
        )
    )
    ar_intra_hops = round(math.log2(eff_tp)) if eff_tp <= _CHIPS_PER_HOST else 3
    num_hosts = math.ceil(eff_tp / _CHIPS_PER_HOST)
    ar_inter_latency_s = (
        0.0
        if eff_tp <= _CHIPS_PER_HOST
        else round(math.log2(num_hosts)) * _DEFAULT_DCN_HOP_LATENCY_S
    )
    return _ScopeContext(
        vmem_budget=vmem_budget,
        single_layer_dense_staging_bytes=min(4 << 20, layer_dense_weight_bytes),
        single_layer_expert_staging_bytes=min(4 << 20, int(buf_slots * exp_bytes)),
        whole_step_dense_staging_bytes=whole_step_dense_staging_bytes,
        whole_step_expert_staging_bytes=whole_step_expert_staging_bytes,
        first_slot_bytes=first_slot_bytes,
        gqa_scratch=3072 * max(head_dim, kv_lora) if l_attn > 0 else 0,
        gdn_scratch=16 * min(vh_local, 8) * kd * vd,
        ffn_scratch=(
            (1024 * max(1, i_exp_local or hidden)) if l_moe > 0 else (2 << 20)
        ),
        layers_local=layers_local,
        num_segments=num_segments,
        num_layer_groups=(
            ceil(len(workload.layer_groups) * layers_local / num_layers)
            if workload.layer_groups
            else 0
        ),
        l_moe=l_moe,
        l_attn=l_attn,
        l_lin=l_lin,
        active_tokens=active_tokens,
        eff_ctx=eff_ctx,
        batch_local=batch_local,
        verify_tokens=verify_tokens,
        nh_local=nh_local,
        q_dim=q_dim,
        base_dense_params=base_dense_params,
        per_token_fused_bytes=compute_per_token_vmem_bytes(profile, workload)[
            "total_per_token_bytes"
        ],
        token_residual_bytes=token_residual_bytes,
        w_static=w_static,
        exp_bytes=exp_bytes,
        experts_local=experts_local,
        ep_dp=ep_dp,
        eff_tp=eff_tp,
        row_floor=chip_spec.lhs_row_floor(
            profile.expert_dtype if l_moe > 0 else profile.weight_dtype
        ),
        num_ar=profile.ar_slots_per_layer * layers_local,
        ar_hops=ar_hops,
        ar_intra_hops=ar_intra_hops,
        ar_inter_latency_s=ar_inter_latency_s,
        dcn_bytes_per_s=dcn_bytes_per_s,
        ar_vol=ar_vol,
        ar_transfer_time_s=ar_transfer_time_s,
        ar_base_lat_s=ar_base_lat_s,
        egress_bw=chip_spec.ici_egress_bw_bytes_per_s,
        lat_coll=chip_spec.collective_latency_s,
        lat_coll_upper=chip_spec.collective_latency_us_bracket[1] * 1e-6,
        launch_gap_upper_us=max(workload.op_overhead_us, _DEFAULT_LAUNCH_GAP_UPPER_US),
        pp_stage_time=(
            (
                token_residual_bytes / chip_spec.ici_link_bw_bytes_per_s
                + chip_spec.collective_latency_s
            )
            if pp > 1
            else 0.0
        ),
        a2a_time=(
            2.0
            * (ep_dp - 1)
            / ep_dp
            * max(1, top_k)
            * token_residual_bytes
            / chip_spec.ici_egress_bw_bytes_per_s
            if ep_dp > 1
            else 0.0
        ),
        flops_dense=flops_dense,
        flops_kv=flops_kv,
        flops_moe=flops_moe,
        peak_w=chip_spec.peak_flops(profile.weight_dtype),
        peak_e=chip_spec.peak_flops(profile.expert_dtype),
        bf16_peak=chip_spec.bf16_flops_per_s,
        kv_cache_read_bytes=kv_cache_read_bytes,
        recurrent_state_rw_bytes=recurrent_state_rw_bytes,
        bw_hbm=chip_spec.hbm_bw_bytes_per_s,
        hbm_used=(
            w_static
            + l_moe * experts_local * exp_bytes
            + kv_cache_read_bytes
            + 0.5 * recurrent_state_rw_bytes
        ),
        no_whole_step=(
            pp > 1
            or num_segments > 1
            or len(workload.layer_groups) > 1
            or num_layers == 1
            or bool(plan_error)
            or has_unmodeled_family
        ),
        plan_error=plan_error,
    )


_SETTLE_CMD = "Pass --measurements measurements.json (schema advisor.measurements.v1)"


def _resolve_calibrated_priors(
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    meas: MeasurementsData | None,
) -> tuple[
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, list[float], ProvenanceTag, str],
]:
    """Return calibrated or assumed priors for launch gap, collectives, and coverage."""
    if meas is not None and meas.launch_gap_p50_us is not None:
        lg_lo = meas.launch_gap_p50_us
        lg_hi = max(lg_lo, meas.launch_gap_p90_us or lg_lo)
        lg_info: tuple[float, float, float, ProvenanceTag, str] = (
            lg_lo,
            lg_lo,
            lg_hi,
            "MEASURED",
            f"{meas.source_desc} (n={meas.launch_gap_n})",
        )
    elif workload.op_overhead_us > 0.0:
        lg_val = workload.op_overhead_us
        lg_info = (
            lg_val,
            lg_val,
            max(lg_val, _DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "caller WorkloadSpec.op_overhead_us override",
        )
    else:
        lg_info = (
            _DEFAULT_LAUNCH_GAP_VAL_US,
            _DEFAULT_LAUNCH_GAP_FLOOR_US,
            _DEFAULT_LAUNCH_GAP_UPPER_US,
            "ASSUMED",
            "uncalibrated TPU host/device launch gap prior [1.0, 8.0] us",
        )

    if meas is not None and meas.collective_hop_us is not None:
        c_val = round(meas.collective_hop_us, 3)
        coll_info: tuple[float, float, float, ProvenanceTag, str] = (
            c_val,
            c_val,
            c_val,
            "MEASURED",
            f"{meas.source_desc} (n={meas.collective_n})",
        )
    else:
        c_lo, c_hi = chip_spec.collective_latency_us_bracket
        c_src = (chip_spec.sources or {}).get(
            "collective_latency_us_bracket", "uncalibrated ICI hop latency bracket"
        )
        coll_info = (
            round(chip_spec.collective_latency_s * 1e6, 3),
            c_lo,
            c_hi,
            "ASSUMED",
            c_src,
        )

    if meas is not None and meas.step_coverage is not None:
        cov_val = round(meas.step_coverage, 4)
        cov_info: tuple[float, list[float], ProvenanceTag, str] = (
            cov_val,
            [cov_val, cov_val],
            "MEASURED",
            f"{meas.source_desc} step_coverage",
        )
    elif workload.phase == "mixed":
        dec_tok = ctx.batch_local * max(1, ctx.verify_tokens)
        cov_val = round(min(1.0, dec_tok / max(1, ctx.active_tokens)), 4)
        cov_info = (
            cov_val,
            [0.0, 1.0],
            "ASSUMED",
            "decode token share on mixed step (settle via --measurements)",
        )
    else:
        cov_info = (
            1.0,
            [0.0, 1.0],
            "ASSUMED",
            "single-phase step coverage prior (settle via --measurements)",
        )
    return lg_info, coll_info, cov_info


# (launch_gap_us, collective_us_per_hop, attention dot dtype) of one corner.
_Corner = tuple[float, float, str]
_DOT_FP8 = "fp8"
_DOT_BF16 = "bf16"
# Knob reported when only an end-to-end A/B of the two scopes can settle a tie.
_AB_KNOB = "ab_step_time"
_CORNER_AXES: Mapping[str, int] = {
    "launch_gap_us": 0,
    "collective_latency_us": 1,
    "prefill_attn_dot_dtype": 2,
}


@dataclass(frozen=True)
class _AssumedBox:
    """Low and high value of each step-time input shared by every scope."""

    launch_gap_us: tuple[float, float]
    collective_s: tuple[float, float]
    attn_dots: tuple[str, ...]


def _assumed_box(
    ctx: _ScopeContext,
    workload: WorkloadSpec,
    priors: tuple[
        tuple[float, float, float, ProvenanceTag, str],
        tuple[float, float, float, ProvenanceTag, str],
        tuple[float, list[float], ProvenanceTag, str],
    ],
) -> _AssumedBox:
    """Build the input box: measured values when present, else the assumed band."""
    (_, lg_lo, lg_hi, lg_tag, _), (c_val, _, _, c_tag, _), _ = priors
    # Uncalibrated floor keeps the zero-launch roofline; a trace floor uses p50.
    lg_box = (
        (lg_lo, lg_hi)
        if lg_tag == "MEASURED"
        else (workload.op_overhead_us, ctx.launch_gap_upper_us)
    )
    coll_box = (
        (c_val * 1e-6, c_val * 1e-6)
        if c_tag == "MEASURED"
        else (ctx.lat_coll, ctx.lat_coll_upper)
    )
    dot_differs = workload.phase in ("prefill", "mixed") and ctx.flops_kv > 0
    dots = (_DOT_FP8, _DOT_BF16) if dot_differs else (_DOT_FP8,)
    return _AssumedBox(lg_box, coll_box, dots)


def _apply_box(
    ctx: _ScopeContext, workload: WorkloadSpec, box: _AssumedBox
) -> tuple[_ScopeContext, WorkloadSpec]:
    """Price scope floors at the box low corner and band tops at its high corner."""
    new_ar_base_lat_s = ctx.ar_intra_hops * box.collective_s[0] + ctx.ar_inter_latency_s
    new_ctx = replace(
        ctx,
        lat_coll=box.collective_s[0],
        lat_coll_upper=box.collective_s[1],
        ar_base_lat_s=new_ar_base_lat_s,
        launch_gap_upper_us=box.launch_gap_us[1],
    )
    return new_ctx, replace(workload, op_overhead_us=box.launch_gap_us[0])


def _evaluate_corners(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    box: _AssumedBox,
) -> dict[_Corner, dict[FusionScope, float]]:
    """Step time of every feasible scope at every corner of the assumed box.

    Every scope at a corner sees the same launch gap, collective latency, and
    attention dtype. Collective latency moves only the all-reduce and cross-DP
    MoE hop term, as in `step_upper_us`, so the all-low corner equals
    `step_bound_us` and the all-high corner equals `step_upper_us`.
    """
    table: dict[_Corner, dict[FusionScope, float]] = {}
    for lg in dict.fromkeys(box.launch_gap_us):
        for coll in dict.fromkeys(box.collective_s):
            ar_lat = ctx.ar_intra_hops * coll + ctx.ar_inter_latency_s
            for dot in box.attn_dots:
                est, _ = _evaluate_scopes(
                    replace(
                        ctx,
                        lat_coll=coll,
                        lat_coll_upper=coll,
                        ar_base_lat_s=ar_lat,
                        launch_gap_upper_us=lg,
                    ),
                    profile,
                    replace(workload, op_overhead_us=lg),
                    bf16_attention_upper=dot == _DOT_BF16,
                )
                table[(lg, round(coll * 1e6, 3), dot)] = {
                    e.scope: e.step_upper_us for e in est if e.feasible
                }
    return table


def _corner_winners(
    corners: Mapping[_Corner, Mapping[FusionScope, float]],
    ops: Mapping[FusionScope, int],
) -> dict[_Corner, FusionScope]:
    """Fastest scope at each corner; fewer launches break exact ties."""
    return {
        c: min(steps, key=lambda s: (steps[s], ops[s]))
        for c, steps in corners.items()
        if steps
    }


def _margin_swing(
    corners: Mapping[_Corner, Mapping[FusionScope, float]],
    best: FusionScope,
    challenger: FusionScope,
    axis: str,
) -> float:
    """Largest shift in the challenger-minus-best margin across one input's band."""
    idx = _CORNER_AXES[axis]
    pair = {best, challenger}
    swing = 0.0
    for c_a, steps_a in corners.items():
        for c_b, steps_b in corners.items():
            moved_only_axis = all(c_a[k] == c_b[k] for k in range(3) if k != idx)
            if c_a == c_b or not moved_only_axis:
                continue
            if pair <= steps_a.keys() and pair <= steps_b.keys():
                d_a = steps_a[challenger] - steps_a[best]
                d_b = steps_b[challenger] - steps_b[best]
                swing = max(swing, abs(d_a - d_b))
    return round(swing, 2)


def _decisive_input(
    winners: Mapping[_Corner, FusionScope], candidates: Sequence[str]
) -> str:
    """First candidate input that leaves one winner when held at either band end."""
    for name in candidates:
        idx = _CORNER_AXES[name]
        for value in {c[idx] for c in winners}:
            if len({w for c, w in winners.items() if c[idx] == value}) == 1:
                return name
    return candidates[0] if candidates else _AB_KNOB


def _collect_unresolved_assumptions(
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    bounds: tuple[
        int,
        float,
        tuple[float, float, ProvenanceTag],
        tuple[float, float, ProvenanceTag],
        ProvenanceTag,
    ],
    contest: tuple[
        Mapping[_Corner, Mapping[FusionScope, float]],
        FusionScope,
        FusionScope | None,
    ],
) -> list[dict[str, object]]:
    """Collect and sort unresolved ASSUMED inputs by step-time sensitivity.

    `sensitivity_us` of a scope input is how far its band moves the margin
    between the best scope and its challenger (the corner winner on a tie,
    else the runner-up). `affects` is `scope` for inputs that can change the
    recommended scope and `gain` for `step_coverage`, which scales gain only.
    """
    d_ops, b_val, (lg_lo, lg_hi, lg_tag), (coll_lo, coll_hi, coll_tag), cov_tag = bounds
    corners, best_scope, challenger = contest

    def scope_sens(axis: str, fallback_us: float) -> float:
        """Margin swing vs the challenger, else the per_op-relative estimate."""
        if challenger is None:
            return round(fallback_us, 2)
        return _margin_swing(corners, best_scope, challenger, axis)

    unresolved: list[dict[str, object]] = []
    if lg_tag == "ASSUMED":
        unresolved.append(
            {
                "name": "launch_gap_us",
                "band": [lg_lo, lg_hi],
                "unit": "us",
                "sensitivity_us": scope_sens("launch_gap_us", d_ops * (lg_hi - lg_lo)),
                "affects": "scope",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    if coll_tag == "ASSUMED" and ctx.eff_tp * workload.ep > 1:
        unresolved.append(
            {
                "name": "collective_latency_us",
                "band": [coll_lo, coll_hi],
                "unit": "us/hop",
                "sensitivity_us": scope_sens(
                    "collective_latency_us",
                    ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
                ),
                "affects": "scope",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    if (
        workload.phase in ("prefill", "mixed")
        and chip_spec.native_ocp_fp8
        and ctx.flops_kv > 0
    ):
        attn_sens_us = (ctx.flops_kv / ctx.bf16_peak - ctx.flops_kv / ctx.peak_w) * 1e6
        unresolved.append(
            {
                "name": "prefill_attn_dot_dtype",
                "band": [
                    round(ctx.bf16_peak / 1e12, 1),
                    round(ctx.peak_w / 1e12, 1),
                ],
                "unit": "TFLOP/s",
                "sensitivity_us": scope_sens(
                    "prefill_attn_dot_dtype", max(0.0, attn_sens_us)
                ),
                "affects": "scope",
                "how_to_settle": "Inspect XProf HLO attention dot precision",
            }
        )
    if cov_tag == "ASSUMED":
        unresolved.append(
            {
                "name": "step_coverage",
                "band": [0.0, 1.0],
                "unit": "ratio",
                "sensitivity_us": round(b_val, 2),
                "affects": "gain",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    unresolved.sort(key=lambda item: float(str(item["sensitivity_us"])), reverse=True)
    return unresolved


def _build_provenance_and_verdict(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    info: tuple[
        list[ScopeEstimate],
        ScopeEstimate,
        int,
        bool,
        MeasurementsData | None,
        Mapping[_Corner, Mapping[FusionScope, float]],
        float,
    ],
) -> tuple[
    dict[str, ProvenanceMetric],
    tuple[str, ...],
    str,
    str,
    tuple[dict[str, object], ...],
    tuple[_Corner, FusionScope] | None,
]:
    """Assemble provenance-tagged metrics, flags, confidence, and verdict."""
    estimates, best, ridge_tokens, hbm_fits, meas, corners, lg_floor = info
    per_op_sc = estimates[0]
    d_ops = max(0, per_op_sc.ops_per_step - best.ops_per_step)
    (
        (lg_val, lg_lo, lg_hi, lg_tag, lg_src),
        (coll_val, coll_lo, coll_hi, coll_tag, coll_src),
        (cov_val, cov_band, cov_tag, cov_src),
    ) = _resolve_calibrated_priors(workload, chip_spec, ctx, meas)

    # step_bound_us already carries ops * lg_floor; strip it before re-pricing.
    base_per_op_us = per_op_sc.step_bound_us - per_op_sc.ops_per_step * lg_floor
    base_best_us = best.step_bound_us - best.ops_per_step * lg_floor
    pure_diff_us = max(0.0, base_per_op_us - base_best_us)
    b_val = round(d_ops * lg_val + pure_diff_us, 2)
    b_lo = round(d_ops * lg_lo + pure_diff_us, 2)
    b_hi = round(
        d_ops * lg_hi
        + pure_diff_us
        + ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
        2,
    )
    b_tag: ProvenanceTag = (
        "MEASURED" if lg_tag == "MEASURED" and coll_tag == "MEASURED" else "ASSUMED"
    )

    denom_us = max(1e-6, base_per_op_us + per_op_sc.ops_per_step * lg_val)
    fg_val = round(min(1.0, max(0.0, b_val / denom_us)), 4)
    fg_lo = round(min(fg_val, max(0.0, b_lo / denom_us)), 4)
    fg_hi = round(max(fg_val, min(1.0, b_hi / denom_us)), 4)
    eg_val, eg_lo, eg_hi = (
        round(fg_val * cov_val, 4),
        round(fg_lo * cov_band[0], 4),
        round(fg_hi * cov_band[1], 4),
    )

    # Tokens finished per step across the slice, divided by physical chips.
    # active_tokens is per core: PCP splits the window, TP and DCP replicate it,
    # and DP cancels (each replica owns its own tokens and devices).
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1
    tokens_per_chip = (
        ctx.active_tokens
        * eff_pcp
        * chip_spec.devices_per_chip
        / max(1, ctx.eff_tp * workload.pp)
    )
    tput_hi = round(tokens_per_chip / max(1e-9, best.step_bound_us * 1e-6), 2)
    tput_lo = round(
        tokens_per_chip / max(1e-9, max(best.step_bound_us, best.step_upper_us) * 1e-6),
        2,
    )

    metrics: dict[str, ProvenanceMetric] = {
        "fits_hbm": {
            "value": hbm_fits,
            "unit": "bool",
            "tag": "DERIVED",
            "source": "hbm_used <= chip_spec.hbm_capacity_bytes",
            "band": [float(hbm_fits), float(hbm_fits)],
        },
        "tokens_per_core": {
            "value": ctx.active_tokens,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "WorkloadSpec batch * phase tokens / (dp * pcp)",
            "band": [float(ctx.active_tokens), float(ctx.active_tokens)],
        },
        "onchip_ceiling": {
            "value": best.t_vmem_max,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "(vmem_budget - static_vmem_bytes) // per_token_vmem_bytes",
            "band": [float(best.t_vmem_max), float(best.t_vmem_max)],
        },
        "compute_bound_at": {
            "value": ridge_tokens,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "(w_step + state_bytes) * ridge_flop_per_byte / flop_per_tok",
            "band": [float(ridge_tokens), float(ridge_tokens)],
        },
        "launch_gap_us": {
            "value": lg_val,
            "unit": "us",
            "tag": lg_tag,
            "source": lg_src,
            "band": [lg_lo, lg_hi],
        },
        "collective_latency_us": {
            "value": coll_val,
            "unit": "us/hop",
            "tag": coll_tag,
            "source": coll_src,
            "band": [coll_lo, coll_hi],
        },
        "step_floor_us": {
            "value": best.step_bound_us,
            "unit": "us",
            "tag": "DERIVED",
            "source": f"roofline floor for scope={best.scope}",
            "band": [best.step_bound_us, max(best.step_bound_us, best.step_upper_us)],
        },
        "throughput_tok_s_per_chip": {
            "value": tput_hi,
            "unit": "tok/s/chip",
            "tag": "DERIVED",
            "source": "tokens/core * pcp * devices_per_chip / (tp * pp) / floor",
            "band": [min(tput_lo, tput_hi), max(tput_lo, tput_hi)],
        },
        "boundary_cost_us": {
            "value": b_val,
            "unit": "us",
            "tag": b_tag,
            "source": "(per_op_ops - scope_ops) * launch_gap_us + collective_saving_us",
            "band": [min(b_lo, b_hi), max(b_lo, b_hi)],
        },
        "fusion_gain_max": {
            "value": fg_val,
            "unit": "ratio",
            "tag": "DERIVED",
            "source": "boundary_cost_us / per_op_step_us",
            "band": [fg_lo, fg_hi],
        },
        "step_coverage": {
            "value": cov_val,
            "unit": "ratio",
            "tag": cov_tag,
            "source": cov_src,
            "band": cov_band,
        },
        "expected_gain_max": {
            "value": eg_val,
            "unit": "ratio",
            "tag": "DERIVED",
            "source": "fusion_gain_max * step_coverage",
            "band": [eg_lo, eg_hi],
        },
    }

    flags_list = [
        flag
        for cond, flag in (
            (profile.architecture_unverified, "architecture_unverified"),
            (best.n_splits > 1, "multi_call_unmodeled"),
        )
        if cond
    ]
    feasible_others = [e for e in estimates if e.feasible and e.scope != best.scope]
    runner_up = (
        min(feasible_others, key=lambda e: (e.step_bound_us, e.ops_per_step))
        if feasible_others
        else None
    )
    # Every scope is priced at the same corner; a tie needs the winner to change.
    ops = {e.scope: e.ops_per_step for e in estimates}
    winners = _corner_winners(corners, ops) if hbm_fits else {}
    flip = next(((c, w) for c, w in winners.items() if w != best.scope), None)
    challenger = flip[1] if flip else (runner_up.scope if runner_up else None)
    unresolved = _collect_unresolved_assumptions(
        workload,
        chip_spec,
        ctx,
        (
            d_ops,
            b_val,
            (lg_lo, lg_hi, lg_tag),
            (coll_lo, coll_hi, coll_tag),
            cov_tag,
        ),
        (corners, best.scope, challenger),
    )
    scope_inputs = [str(u["name"]) for u in unresolved if u["affects"] == "scope"]
    is_tie = hbm_fits and flip is not None
    verdict = (
        f"tie(measure: {_decisive_input(winners, scope_inputs)})"
        if is_tie
        else _scope_verdict(best.scope if hbm_fits else "per_op")
    )
    confidence = (
        "validated_end_to_end"
        if meas is not None and meas.step_time_us is not None
        else ("bounded" if meas is not None else "prior_only")
    )
    return metrics, tuple(flags_list), verdict, confidence, tuple(unresolved), flip


def _format_advise_reason(
    best: ScopeEstimate,
    estimates: Sequence[ScopeEstimate],
    ridge_tokens: int,
    used_gb: float,
    cap_gb: float,
) -> str:
    """Format rationale: best scope, runner-up, and per_op."""
    if used_gb > cap_gb:
        return f"Model+cache ({used_gb:.2f} GB) > HBM ({cap_gb:.2f} GB)"
    others = [e for e in estimates if e.feasible and e.scope != best.scope]
    text = f"{best.scope} ({best.step_bound_us:.1f} us)"
    if others:
        ru = min(others, key=lambda e: (e.step_bound_us, e.ops_per_step))
        delta = ru.step_bound_us - best.step_bound_us
        pct = 100.0 * delta / max(1e-9, best.step_bound_us)
        text += (
            f" vs runner-up {ru.scope} ({ru.step_bound_us:.1f} us, "
            f"+{delta:.1f} us, {pct:.1f}%)"
        )
        if best.scope != "per_op" and ru.scope != "per_op":
            text += f"; per_op {estimates[0].step_bound_us:.1f} us"
    if best.scope == "per_op":
        text += f"; compute-bound at T={ridge_tokens}"
    return text


def _scope_verdict(scope: FusionScope) -> str:
    """Name the build decision: multi-layer megakernel, per-layer kernel, or ops."""
    if scope in ("whole_step", "layer_group"):
        return "megakernel_likely"
    if scope == "single_layer":
        return "fused_layer_likely"
    return "separate_likely"


def _flip_note(flip: tuple[_Corner, FusionScope] | None) -> str:
    """Name the scope that wins at the corner where the verdict flips."""
    if flip is None:
        return ""
    (lg, coll, dot), alt = flip
    return (
        f"; {alt} wins at launch gap {lg:.1f} us, collective "
        f"{coll:.2f} us/hop, {dot} attention"
    )


def advise(
    profile: ModelProfile | str | Mapping[str, object],
    workload: WorkloadSpec,
    chip: ChipSpec | str = "tpu7x",
    plan_error: str = "",
    **extra: object,
) -> AdvisorReport:
    """Evaluate fusion scopes from first-principles hardware and model bounds."""
    prof = resolve_model_profile(profile)
    meas_arg = extra.get("measurements")
    meas = load_measurements(meas_arg) if meas_arg is not None else None
    chip_spec_path = extra.get("chip_spec_path")
    cs_path = str(chip_spec_path) if chip_spec_path is not None else None
    eff_wl = workload
    if meas is not None and meas.ring_plan is not None and eff_wl.ring_plan is None:
        eff_wl = replace(eff_wl, ring_plan=meas.ring_plan)

    dtypes_bw = _validate_advise_inputs(prof, eff_wl)
    chip_spec = resolve_chip_spec(chip, chip_spec_path=cs_path)
    ctx = _build_scope_context(
        prof, eff_wl, chip_spec, dtypes_bw, plan_error, meas=meas
    )
    box = _assumed_box(
        ctx, eff_wl, _resolve_calibrated_priors(eff_wl, chip_spec, ctx, meas)
    )
    ctx, eval_wl = _apply_box(ctx, eff_wl, box)
    estimates, raw_step_map = _evaluate_scopes(ctx, prof, eval_wl)
    corners = _evaluate_corners(ctx, prof, eval_wl, box)

    hbm_used = ctx.hbm_used
    hbm_fits = hbm_used <= chip_spec.hbm_capacity_bytes
    candidates = [e for e in estimates if e.feasible] if hbm_fits else [estimates[0]]
    best = min(candidates, key=lambda e: (raw_step_map[e.scope], e.ops_per_step))
    used_gb, cap_gb = hbm_used / 1e9, chip_spec.hbm_capacity_bytes / 1e9
    ridge = ctx.peak_w / ctx.bw_hbm
    b_wa = best.w_step_gb * 1e9 * best.n_splits + best.act_hbm_mb * 1e6
    state_bytes = ctx.kv_cache_read_bytes + ctx.recurrent_state_rw_bytes
    kv_lora, qk_rope = prof.kv_lora_rank, prof.qk_rope_head_dim
    nh_local, q_dim = ctx.nh_local, ctx.q_dim
    v_q = nh_local * (2 * kv_lora + qk_rope) if kv_lora else 2.0 * q_dim
    tot_bytes = max(1.0, b_wa + state_bytes)
    active_tokens = ctx.active_tokens
    eff_ctx = ctx.eff_ctx
    l_moe, l_attn = ctx.l_moe, ctx.l_attn
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    v_p = (
        ctx.base_dense_params
        + l_moe * prof.hidden_size * prof.num_experts
        + flops_moe / (2.0 * active_tokens)
    )
    v_flops = 2.0 * ctx.batch_local * ctx.verify_tokens * (v_p + l_attn * eff_ctx * v_q)
    flop_per_tok = max(1.0, (flops_dense + flops_moe) / max(1, active_tokens))
    ridge_tokens = max(
        1,
        ceil((estimates[0].w_step_gb * 1e9 + state_bytes) * ridge / flop_per_tok),
    )
    rec_scope: FusionScope = best.scope if hbm_fits else "per_op"
    step_ai = round((flops_dense + best.moe_flop_factor * flops_moe) / tot_bytes, 2)
    asymp_ai = round(v_flops / state_bytes, 2) if state_bytes else None
    if l_moe == 0:
        rec_moe = "dense_mlp"
    elif rec_scope == "per_op":
        rec_moe = "grouped_in_hbm"
    else:
        t_per_ep = ceil(active_tokens / max(1, best.n_splits)) * ctx.ep_dp
        rec_moe = (
            "grouped_on_chip"
            if t_per_ep >= chip_spec.lhs_row_floor(prof.expert_dtype)
            else "masked_on_chip"
        )

    metrics, flags, verdict, confidence, unresolved, flip = (
        _build_provenance_and_verdict(
            prof,
            eff_wl,
            chip_spec,
            ctx,
            (
                estimates,
                best,
                ridge_tokens,
                hbm_fits,
                meas,
                corners,
                eval_wl.op_overhead_us,
            ),
        )
    )
    rationale = _format_advise_reason(
        best, estimates, ridge_tokens, used_gb, cap_gb
    ) + _flip_note(flip)
    if eff_wl.objective in ("throughput", "both"):
        tput_val = float(metrics["throughput_tok_s_per_chip"]["value"])
        rationale += f"; throughput={tput_val:.1f} tok/s/chip"
        dep = DEPLOYMENT_PRESETS.get(prof.name)
        if dep is not None and dep.saturation is not None and eff_wl.phase == "decode":
            cur_layout = ParallelLayout(
                tp=eff_wl.tp,
                ep=eff_wl.ep,
                dp=eff_wl.dp,
                pp=eff_wl.pp,
                dcp=eff_wl.dcp,
                pcp=eff_wl.pcp,
                layer_groups=eff_wl.layer_groups,
            )
            if cur_layout.label() != dep.saturation.label():
                comp = compare_parallelism_candidates(
                    prof,
                    (cur_layout, dep.saturation),
                    batch=max(1, eff_wl.batch // max(1, eff_wl.dp)),
                    context=eff_ctx,
                    chip=chip_spec,
                    objective=eff_wl.objective,
                )
                comp_verdict = str(comp["verdict"])
                if eff_wl.objective == "both" and comp_verdict.startswith(
                    "tie(objective_split:"
                ):
                    flags = (*flags, "objective_split")
                    verdict = comp_verdict
                    rationale += f"; {comp_verdict}"
                elif eff_wl.objective == "throughput":
                    rationale += (
                        f"; best_throughput_layout={comp['best_throughput_layout']}"
                    )

    return AdvisorReport(
        model=prof.name,
        chip=chip_spec.name,
        phase=eff_wl.phase,
        active_tokens=active_tokens,
        effective_context=eff_ctx,
        vmem_budget_mib=round(ctx.vmem_budget / (1 << 20), 3),
        hbm_fits=hbm_fits,
        hbm_used_gb=round(used_gb, 3),
        hbm_capacity_gb=round(cap_gb, 3),
        ridge_dense_flop_per_byte=round(ridge, 2),
        step_arithmetic_intensity=step_ai,
        asymptotic_decode_intensity=asymp_ai,
        recommended_scope=rec_scope,
        build_megakernel=rec_scope != "per_op",
        rationale=rationale,
        scopes=tuple(estimates),
        per_token_vmem_bytes=ctx.per_token_fused_bytes,
        ridge_tokens=ridge_tokens,
        recommended_moe_mode=rec_moe,
        dma_buffers=eff_wl.dma_buffers,
        verdict=verdict,
        confidence=confidence,
        flags=flags,
        metrics=metrics,
        unresolved_assumptions=unresolved,
    )


def compare_parallelism_candidates(
    profile: ModelProfile | str | Mapping[str, object],
    layouts: Sequence[ParallelLayout],
    *,
    batch: int = 16,
    chip: ChipSpec | str = "tpu7x",
    **opts: object,
) -> dict[str, object]:
    """Compare ParallelLayouts under step-latency and chip-throughput objectives."""
    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    context = _int_opt(opts, "context", 4096)
    objective = str(opts.get("objective", "both"))
    rows: list[dict[str, object]] = []
    for layout in layouts:
        wl = WorkloadSpec(
            phase="decode",
            batch=batch * layout.dp,
            isl=context,
            osl=0,
            context=context,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
            moe_mode="grouped_on_chip" if prof.num_experts > 0 else "dense_mlp",
        )
        rep = advise(prof, wl, chip_spec)
        step_us = float(rep.metrics["step_floor_us"]["value"])
        tput = float(rep.metrics["throughput_tok_s_per_chip"]["value"])
        rows.append(
            {
                "layout": layout.label(),
                "step_floor_us": step_us,
                "throughput_tok_s_per_chip": tput,
                "recommended_scope": rep.recommended_scope,
                "hbm_fits": rep.hbm_fits,
            }
        )
    feasible_rows = [r for r in rows if bool(r["hbm_fits"])] or rows
    best_step = min(feasible_rows, key=lambda r: float(str(r["step_floor_us"])))
    best_tput = max(
        feasible_rows, key=lambda r: float(str(r["throughput_tok_s_per_chip"]))
    )
    step_lbl = str(best_step["layout"])
    tput_lbl = str(best_tput["layout"])
    if objective == "step":
        verdict = step_lbl
    elif objective == "throughput":
        verdict = tput_lbl
    elif step_lbl != tput_lbl:
        verdict = f"tie(objective_split: step={step_lbl}, throughput={tput_lbl})"
    else:
        verdict = step_lbl
    return {
        "model": prof.name,
        "chip": chip_spec.name,
        "objective": objective,
        "best_step_layout": step_lbl,
        "best_throughput_layout": tput_lbl,
        "verdict": verdict,
        "candidates": rows,
    }


def recommend_parallelism(
    profile: ModelProfile,
    chip: ChipSpec | str = "tpu7x",
    *,
    dp: int = 1,
    pp: int = 1,
) -> ParallelismRecommendation:
    """Auto-derive the minimum balanced TPU parallelism mesh for a model and chip."""
    chip_spec = resolve_chip_spec(chip)
    fallback: ParallelismRecommendation | None = None
    for tp_cand in (1, 2, 4, 8, 16, 32, 64):
        if tp_cand * dp * pp < chip_spec.min_slice_cores:
            continue
        if profile.num_attention_heads % tp_cand != 0:
            continue
        if (
            profile.num_linear_layers > 0
            and profile.linear_value_heads > 0
            and profile.linear_value_heads % tp_cand != 0
        ):
            continue
        ep_cand = (
            gcd(tp_cand * dp, profile.num_experts) if profile.num_experts > 0 else 1
        )
        wl = WorkloadSpec(
            phase="decode", batch=16, tp=tp_cand, ep=ep_cand, dp=dp, pp=pp
        )
        rep = advise(profile, wl, chip_spec)
        if not rep.hbm_fits:
            continue
        sub = rep.scopes[0]
        total_cores = tp_cand * dp * pp
        topo = chip_spec.topologies.get(total_cores, f"{total_cores} cores")
        cand = ParallelismRecommendation(
            tp=tp_cand,
            ep=ep_cand,
            dp=dp,
            pp=pp,
            topology=topo,
            hbm_used_gb=rep.hbm_used_gb,
            hbm_capacity_gb=rep.hbm_capacity_gb,
            w_step_gb=sub.w_step_gb,
            auto_selected=True,
        )
        fallback = cand
        # Verify HBM capacity holds at the high-concurrency 32K regime (B=64, 32K)
        # and that the reference B=16 decode batch fits in one VMEM pass.
        rep_high = advise(
            profile,
            replace(wl, batch=64, isl=32768, context=32768),
            chip_spec,
        )
        whole_sc = next(s for s in rep.scopes if s.scope == "whole_step")
        if rep_high.hbm_fits and whole_sc.n_splits == 1:
            return cand
    if fallback is not None:
        return fallback
    return ParallelismRecommendation(
        tp=8,
        ep=gcd(8 * dp, profile.num_experts) if profile.num_experts > 0 else 1,
        dp=dp,
        pp=pp,
        topology=chip_spec.topologies.get(8 * dp * pp, f"{8 * dp * pp} cores"),
        hbm_used_gb=0.0,
        hbm_capacity_gb=round(chip_spec.hbm_capacity_bytes / 1e9, 3),
        w_step_gb=0.0,
        auto_selected=True,
    )


def _int_opt(opts: Mapping[str, object], key: str, default: int) -> int:
    """Extract an integer option from opts or return default."""
    val = opts.get(key, default)
    return int(float(str(val))) if type(val) in (int, float, str) else default


def _float_opt(opts: Mapping[str, object], key: str, default: float) -> float:
    """Extract a float option from opts or return default."""
    val = opts.get(key, default)
    return float(str(val)) if type(val) in (int, float, str) else default


def _layout_recommendation(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    layout: ParallelLayout,
    base: WorkloadSpec,
    source: str = "deployment preset",
) -> ParallelismRecommendation:
    """Evaluate a fixed layout's HBM fit at the reference decode workload."""
    base_rep = advise(
        prof,
        replace(
            base,
            phase="decode",
            batch=16,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
        ),
        chip_spec,
    )
    total_cores = layout.tp * layout.dp * layout.pp
    return ParallelismRecommendation(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        topology=chip_spec.topologies.get(total_cores, f"{total_cores} cores"),
        hbm_used_gb=base_rep.hbm_used_gb,
        hbm_capacity_gb=base_rep.hbm_capacity_gb,
        w_step_gb=base_rep.scopes[0].w_step_gb,
        auto_selected=False,
        dcp=layout.dcp,
        pcp=layout.pcp,
        source=source,
        layer_groups=layout.layer_groups,
    )


def _select_layouts(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    base: WorkloadSpec,
    requested: tuple[int | None, int | None, int, int, int, int],
    dep: DeploymentPreset | None,
) -> tuple[
    ParallelismRecommendation,
    ParallelismRecommendation | None,
    ParallelismRecommendation | None,
]:
    """Return decode, prefill, and saturation layouts: preset, user, or auto-pick."""
    tp, ep, dp, pp, dcp, pcp = requested
    if dep is not None:
        sat_rec = (
            _layout_recommendation(prof, chip_spec, dep.saturation, base)
            if dep.saturation is not None
            else None
        )
        dec_rec = (
            sat_rec
            if base.objective == "throughput" and sat_rec is not None
            else _layout_recommendation(prof, chip_spec, dep.decode, base)
        )
        return (
            dec_rec,
            _layout_recommendation(prof, chip_spec, dep.prefill, base),
            sat_rec,
        )
    if tp is not None:
        ep_val = (
            ep
            if ep is not None
            else (gcd(tp * dp, prof.num_experts) if prof.num_experts > 0 else 1)
        )
        layout = ParallelLayout(tp=tp, ep=ep_val, dp=dp, pp=pp, dcp=dcp, pcp=pcp)
        return (
            _layout_recommendation(prof, chip_spec, layout, base, source=""),
            None,
            None,
        )
    par = recommend_parallelism(prof, chip_spec, dp=dp, pp=pp)
    if ep is not None or dcp > 1 or pcp > 1:
        par = replace(par, ep=ep if ep is not None else par.ep, dcp=dcp, pcp=pcp)
    return par, None, None


def _compute_disagg_handoff(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    dep: DeploymentPreset | None,
    meas: MeasurementsData | None,
    isl: int = 4096,
) -> DisaggHandoffSpec | None:
    """Compute per-request KV + recurrent state wire bytes and DCN transfer floor."""
    if dep is None:
        return None
    bw_kv = _DTYPE_BYTES.get(prof.kv_dtype.lower(), 1.0)
    bw_st = _DTYPE_BYTES.get(prof.state_dtype.lower(), 2.0)
    kv_per_tok = (
        (prof.kv_lora_rank + prof.qk_rope_head_dim) * bw_kv
        if prof.kv_lora_rank
        else 2.0 * prof.num_kv_heads * prof.head_dim * bw_kv
    )
    kv_wire_bytes = prof.num_attention_layers * isl * kv_per_tok
    qkv_lin_full = (
        2 * prof.linear_key_heads * prof.linear_key_dim
        + prof.linear_value_heads * prof.linear_value_dim
    )
    z_dim_full = prof.linear_value_heads * prof.linear_value_dim
    state_wire_bytes = (
        prof.num_linear_layers
        * (z_dim_full * prof.linear_key_dim + qkv_lin_full * prof.linear_conv_kernel)
        * bw_st
    )
    total_wire_bytes = kv_wire_bytes + state_wire_bytes
    if meas is not None and meas.dcn_bw_gbps is not None:
        dcn_gbps = meas.dcn_bw_gbps
        dcn_bracket = (dcn_gbps, dcn_gbps)
        dcn_prov: Literal["ASSUMED", "MEASURED"] = "MEASURED"
    else:
        dcn_gbps = chip_spec.dcn_bw_gbps
        dcn_bracket = chip_spec.dcn_bw_gbps_bracket
        dcn_prov = "ASSUMED"
    dcn_bytes_per_s = (dcn_gbps / 8.0) * 1e9
    dcn_lo_bytes_per_s = (dcn_bracket[0] / 8.0) * 1e9
    dcn_hi_bytes_per_s = (dcn_bracket[1] / 8.0) * 1e9
    xfer_ms = round((total_wire_bytes / max(1.0, dcn_bytes_per_s)) * 1e3, 2)
    xfer_lo_ms = round((total_wire_bytes / max(1.0, dcn_hi_bytes_per_s)) * 1e3, 2)
    xfer_hi_ms = round((total_wire_bytes / max(1.0, dcn_lo_bytes_per_s)) * 1e3, 2)
    return DisaggHandoffSpec(
        isl=isl,
        kv_wire_mb=round(kv_wire_bytes / 1e6, 2),
        state_wire_mb=round(state_wire_bytes / 1e6, 2),
        total_wire_mb=round(total_wire_bytes / 1e6, 2),
        dcn_bw_gbps=dcn_gbps,
        dcn_bw_gbps_bracket=dcn_bracket,
        dcn_xfer_ms=xfer_ms,
        dcn_xfer_bracket_ms=(xfer_lo_ms, xfer_hi_ms),
        dcn_provenance=dcn_prov,
    )


def _with_layout(wl: WorkloadSpec, layout: ParallelLayout) -> WorkloadSpec:
    """Copy `wl` onto `layout`'s parallelism degrees and layer groups."""
    return replace(
        wl,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )


def _with_preset_prefill(
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    dep: DeploymentPreset | None,
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Move prefill and saturated decode regimes onto preset layouts and add windows."""
    if dep is None:
        return workloads
    pre = dep.prefill
    sat = dep.saturation
    decode_base = _with_layout(workloads[0][1], pre)
    windows = tuple(
        (
            f"Preset Prefill Window (W={win})",
            replace(
                decode_base,
                phase="prefill",
                batch=1,
                isl=win,
                chunk_size=win,
                moe_mode="grouped_on_chip",
            ),
        )
        for win in dep.prefill_windows
    )
    sat_batch = min(600, 256 * sat.dp) if sat is not None else 256
    moved = tuple(
        (
            (
                f"Decode Saturated (B={sat_batch} over DP{sat.dp}, T=1, 32K)"
                if sat is not None and lbl.startswith("Decode Saturated")
                else lbl
            ),
            _with_layout(wl, pre)
            if wl.phase == "prefill"
            else (
                replace(
                    _with_layout(wl, sat),
                    batch=min(600, wl.batch * sat.dp),
                )
                if sat is not None and lbl.startswith("Decode Saturated")
                else wl
            ),
        )
        for lbl, wl in workloads
    )
    return moved + windows


def advise_regimes(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str = "tpu7x",
    **opts: object,
) -> AdvisorSweepReport:
    """Evaluate the 12-regime Batch Size x Phase & Context matrix."""
    raw_tp = opts.get("tp")
    raw_ep = opts.get("ep")
    raw_vmem = opts.get("vmem_budget_mib")
    tp = int(raw_tp) if type(raw_tp) is int else None
    ep = int(raw_ep) if type(raw_ep) is int else None
    vmem_budget_mib = (
        float(str(raw_vmem)) if type(raw_vmem) in (int, float, str) else None
    )
    dp = _int_opt(opts, "dp", 1)
    pp = _int_opt(opts, "pp", 1)
    dcp = _int_opt(opts, "dcp", 1)
    pcp = _int_opt(opts, "pcp", 1)
    meas_opt = opts.get("measurements")
    cs_path_opt = opts.get("chip_spec_path")
    raw_obj = str(opts.get("objective", "step"))
    obj_mode: ObjectiveMode = (
        raw_obj if raw_obj in ("step", "throughput", "both") else "step"
    )
    base = WorkloadSpec(
        vmem_alias=bool(opts.get("vmem_alias", True)),
        vmem_fit_headroom=bool(opts.get("vmem_fit_headroom", False)),
        op_overhead_us=_float_opt(opts, "op_overhead_us", 0.0),
        vmem_budget_mib=vmem_budget_mib,
        dma_buffers=_int_opt(opts, "dma_buffers", 2),
        objective=obj_mode,
    )
    dma_buffers = base.dma_buffers

    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(
        chip,
        chip_spec_path=str(cs_path_opt) if cs_path_opt is not None else None,
    )
    explicit_layout = tp is not None or ep is not None or (dp, pp, dcp, pcp) != (1,) * 4
    dep = (
        DEPLOYMENT_PRESETS.get(prof.name)
        if bool(opts.get("deployment_preset", True)) and not explicit_layout
        else None
    )
    par, prefill_par, sat_par = _select_layouts(
        prof, chip_spec, base, (tp, ep, dp, pp, dcp, pcp), dep
    )
    meas_data = load_measurements(meas_opt) if meas_opt is not None else None
    disagg_handoff = _compute_disagg_handoff(prof, chip_spec, dep, meas_data)
    base = replace(
        base, tp=par.tp, ep=par.ep, dp=par.dp, pp=par.pp, dcp=par.dcp, pcp=par.pcp
    )
    # Draft tokens per verify step: the shipped decode preset, else MTP-3.
    k_spec = dep.decode.spec_tokens if dep is not None and dep.decode.spec_tokens else 3
    workloads = (
        (
            "Decode Small (B=4, T=1, 4K)",
            replace(
                base,
                phase="decode",
                batch=4,
                isl=3072,
                osl=2048,
            ),
        ),
        (
            "Decode Low (B=8, T=1, 4K)",
            replace(
                base,
                phase="decode",
                batch=8,
                isl=3072,
                osl=2048,
            ),
        ),
        (
            "Decode Med (B=16, T=1, 4K)",
            replace(
                base,
                phase="decode",
                batch=16,
                isl=3072,
                osl=2048,
            ),
        ),
        (
            "Decode High (B=64, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=64,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Decode Saturated (B=256, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=256,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify Small (B=4, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=4,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
            ),
        ),
        (
            f"Spec Verify Low (B=8, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=8,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify Med (B=16, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=16,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify High (B=32, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=32,
                spec_tokens=k_spec,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Mixed Decode+Prefill (B=16, C=256)",
            replace(
                base,
                phase="mixed",
                batch=16,
                isl=4096,
                osl=2048,
                chunk_size=256,
                prefix_hit_rate=0.90,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Chunked Prefill (B=1, T=256)",
            replace(
                base,
                phase="prefill",
                batch=1,
                isl=4096,
                chunk_size=256,
                prefix_hit_rate=0.9375,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Long Prefill (B=1, T=4096)",
            replace(
                base,
                phase="prefill",
                batch=1,
                isl=4096,
                moe_mode="grouped_on_chip",
            ),
        ),
    )

    workloads = _with_preset_prefill(workloads, dep)

    entries: list[RegimeEntry] = []
    reports: list[AdvisorReport] = []
    for label, wl in workloads:
        rep = advise(prof, wl, chip_spec, measurements=meas_opt)
        reports.append(rep)
        best_sc = next(s for s in rep.scopes if s.scope == rep.recommended_scope)
        entries.append(
            RegimeEntry(
                label=label,
                phase=wl.phase,
                active_tokens=rep.active_tokens,
                effective_context=rep.effective_context,
                step_arithmetic_intensity=rep.step_arithmetic_intensity,
                max_vmem_tokens=best_sc.t_vmem_max,
                num_vmem_splits=best_sc.n_splits,
                recommended_scope=rep.recommended_scope,
                recommended_moe_mode=rep.recommended_moe_mode,
                step_bound_us=best_sc.step_bound_us,
                rationale=rep.rationale,
                layout=ParallelLayout(
                    tp=wl.tp, ep=wl.ep, dp=wl.dp, pp=wl.pp, dcp=wl.dcp, pcp=wl.pcp
                ).label(),
                step_upper_us=best_sc.step_upper_us,
                verdict=rep.verdict,
                confidence=rep.confidence,
                flags=rep.flags,
                metrics=rep.metrics,
                unresolved_assumptions=rep.unresolved_assumptions,
                scopes=tuple(
                    {
                        "scope": s.scope,
                        "feasible": s.feasible,
                        "step_bound_us": s.step_bound_us,
                        "step_upper_us": s.step_upper_us,
                        "ops_per_step": s.ops_per_step,
                        "n_splits": s.n_splits,
                        "t_vmem_max": s.t_vmem_max,
                    }
                    for s in rep.scopes
                ),
            )
        )

    decode_rep = reports[2]
    vmem_parts = compute_per_token_vmem_bytes(prof, workloads[2][1])
    cross_sc = next(s for s in decode_rep.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in decode_rep.scopes if s.scope == "single_layer")
    ref_sc = cross_sc if cross_sc.feasible else one_l_sc

    t_ceil = ref_sc.t_vmem_max
    t_ridge = decode_rep.ridge_tokens
    step_mode = primary_step_mode(cross_sc)
    slope_kib = vmem_parts["total_per_token_bytes"] / 1024
    ridge_fb = decode_rep.ridge_dense_flop_per_byte
    guide = (
        f"Small/Med Decode & Spec Verify (T <= {t_ceil} active tokens): "
        f"use scope='{step_mode}' with moe_mode='masked_on_chip' (T < 32) "
        f"or 'grouped_on_chip' (T >= 32) to keep [T, H] in VMEM "
        f"({ref_sc.vmem_static_mib:.1f} MiB {dma_buffers}-slot weight staging + "
        f"{slope_kib:.1f} KiB/tok).",
        f"High-Batch Decode & Chunked Prefill ({t_ceil} < T <= {t_ridge} tokens): "
        "use scope='single_layer' or 'layer_group' "
        f"(single-layer VMEM ceiling T_max={one_l_sc.t_vmem_max}) "
        "so weights are not re-streamed across multi-pass token splits.",
        f"Saturated Decode & Compute-Bound Prefill (T > {t_ridge} tokens, "
        f"AI >= {ridge_fb:.0f} FLOP/B): use scope='per_op' "
        "with moe_mode='grouped_in_hbm': tokens are grouped by expert in HBM "
        "and each operator is its own call, so no VMEM is held for "
        "multi-layer weight staging.",
    )
    return AdvisorSweepReport(
        model=prof.name,
        chip=chip_spec.name,
        parallelism=par,
        vmem_budget_mib=decode_rep.vmem_budget_mib,
        vmem_static_mib=ref_sc.vmem_static_mib,
        per_token_vmem_bytes=vmem_parts["total_per_token_bytes"],
        vmem_breakdown_bytes=vmem_parts,
        max_vmem_tokens=t_ceil,
        ridge_dense_flop_per_byte=decode_rep.ridge_dense_flop_per_byte,
        ridge_tokens=t_ridge,
        regimes=tuple(entries),
        tailoring_guide=guide,
        dma_buffers=dma_buffers,
        single_layer_vmem_tokens=one_l_sc.t_vmem_max,
        prefill_parallelism=prefill_par,
        saturation_parallelism=sat_par,
        disagg_handoff=disagg_handoff,
        chip_sources=chip_spec.sources if chip_spec.sources is not None else {},
        model_sources=_build_model_sources(prof),
        chip_spec=chip_spec,
        model_profile=prof,
    )


def _build_model_sources(prof: ModelProfile) -> dict[str, str]:
    """Extract provenance sources for model architecture facts."""
    base_src = MODEL_PRESET_SOURCES.get(prof.name, f"Model config: {prof.name}")
    sources = {
        "config": base_src,
        "num_layers": f"{prof.num_layers} layers ({base_src})",
        "hidden_size": f"H={prof.hidden_size} ({base_src})",
        "attention_heads": (
            f"NH={prof.num_attention_heads}, NKV={prof.num_kv_heads}, "
            f"HD={prof.head_dim}"
        ),
        "dtypes": (
            f"weight={prof.weight_dtype}, expert={prof.expert_dtype}, "
            f"kv={prof.kv_dtype}"
        ),
    }
    if prof.num_experts > 0:
        sources["moe"] = (
            f"E={prof.num_experts}, TOPK={prof.top_k}, "
            f"I={prof.moe_intermediate_size} ({base_src})"
        )
    return sources


def primary_step_mode(cross_sc: ScopeEstimate) -> str:
    """Return whole_step if whole-step fusion is feasible, else single_layer."""
    return "whole_step" if cross_sc.feasible else "single_layer"


def advise_from_spec(
    model_spec: spec.ModelSpec,
    ex: exec_spec.ExecSpec,
    chip: ChipSpec | str = "tpu7x",
    moe_mode: MoEMode = "masked_on_chip",
    op_overhead_us: float = 0.0,
) -> AdvisorReport:
    """Bridge declarative ModelSpec + ExecSpec schedules via planner.plan."""
    import jax.numpy as jnp

    from .. import planner

    plan_err = ""
    ar_slots = 2
    is_periodic = True
    mlc: Mapping[str, int] = {}
    wd, ed, kv_dt = "fp8", "", "fp8"
    try:
        plan_obj = planner.plan(model_spec, ex)
        wds = tuple(plan_obj.weight_decls.values())
        mlc = plan_obj.module_layer_counts
        ar_slots = sum(x is not None for x in plan_obj.slot_ar_index)
        rings = ([w for w in wds if w.stream_ring == r] for r in ("static", "dynamic"))
        kv_states = [
            s
            for k, s in plan_obj.state_decls.items()
            if "dn" not in plan_obj.state_owners[k]
        ]
        wd, ed, kv_dt = [
            ("fp8" if jnp.dtype(x[0].dtype).itemsize == 1 else "bf16") if x else d
            for x, d in zip((*rings, kv_states), ("fp8", "", "fp8"), strict=True)
        ]
    except errors.MegakernelConfigError as exc:
        plan_err, is_periodic = str(exc), False

    cfg = vars(model_spec.config)
    num_layers = int(cfg["L"])
    hidden = int(cfg["H"])
    att_every = int(cfg.get("att_every") or 1)
    num_experts = int(cfg.get("E") or 0)
    num_attn = sum(v for k, v in mlc.items() if "gqa" in k or "att" in k)
    num_attn = min(
        num_layers,
        num_attn or (num_layers // att_every if att_every > 1 else num_layers),
    )
    prof = ModelProfile(
        name=model_spec.name,
        num_layers=num_layers,
        hidden_size=hidden,
        intermediate_size=int(cfg.get("I") or 4 * hidden),
        num_attention_heads=int(cfg.get("NH") or ex.R),
        num_kv_heads=int(cfg.get("NKV") or ex.R),
        head_dim=int(cfg.get("HD") or 128),
        num_linear_layers=max(0, num_layers - num_attn),
        linear_key_heads=int(cfg.get("KH") or 0),
        linear_value_heads=int(cfg.get("VH") or 0),
        num_experts=num_experts,
        top_k=int(cfg.get("TOPK") or 0) * (num_experts > 0),
        moe_intermediate_size=int(cfg.get("I") or 0),
        shared_expert_intermediate_size=int(cfg.get("IS") or 0),
        num_moe_layers=num_layers * (num_experts > 0),
        num_attention_layers=num_attn,
        linear_key_dim=int(cfg.get("KD") or 128),
        linear_value_dim=int(cfg.get("VD") or 128),
        is_periodic=is_periodic,
        ar_slots_per_layer=ar_slots,
        weight_dtype=wd,
        expert_dtype=ed or wd,
        kv_dtype=kv_dt,
    )
    eff_ep = ex.R if (num_experts > 0 and not bool(cfg.get("moe_tp", True))) else 1
    wl = WorkloadSpec(
        phase="verify" if ex.T > 1 else "decode",
        batch=ex.NS if ex.T > 1 else ex.B,
        isl=ex.ctx,
        osl=0,
        context=ex.ctx,
        spec_tokens=max(0, ex.T - 1),
        tp=ex.R,
        ep=eff_ep,
        moe_mode=moe_mode,
        tiles=ex.tiles,
        op_overhead_us=op_overhead_us,
    )
    return advise(prof, wl, chip, plan_err)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the megakernel advisor."""
    from . import cli as cli_mod

    return cli_mod.build_arg_parser()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the megakernel advisor CLI entrypoint."""
    from . import cli as cli_mod

    return int(cli_mod.main(argv))


__all__ = [
    "BUILTIN_CHIP_SPECS",
    "DEPLOYMENT_PRESETS",
    "DEPLOYMENT_PRESET_SOURCES",
    "MODEL_PRESETS",
    "MODEL_PRESET_SOURCES",
    "REGISTERED_MODEL_ALIASES",
    "REQUIRED_CHIP_SPEC_KEYS",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_ONCHIP_FAMILIES",
    "TPU6E_TOPOLOGIES",
    "TPU7X_TOPOLOGIES",
    "VERIFIED_MEGAKERNEL_MODEL_TYPES",
    "AdvisorReport",
    "AdvisorSweepReport",
    "ChipSpec",
    "DeploymentPreset",
    "DisaggHandoffSpec",
    "FieldSource",
    "FusionScope",
    "MeasurementsData",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "ParallelismRecommendation",
    "Phase",
    "ProvenanceMetric",
    "ProvenanceTag",
    "RegimeEntry",
    "RingPlan",
    "ScopeEstimate",
    "UnsupportedArchitectureError",
    "WorkloadSpec",
    "advise",
    "advise_from_spec",
    "advise_regimes",
    "build_arg_parser",
    "compute_all_reduce_transfer_cost",
    "compare_parallelism_candidates",
    "compute_per_token_vmem_bytes",
    "default_double_buffer_ring_plan",
    "fit_vmem_budget_bytes",
    "format_advisor_summary",
    "format_regime_table",
    "load_chip_spec_file",
    "load_measurements",
    "main",
    "model_profile_from_hf_config",
    "recommend_parallelism",
    "render_advisor_html",
    "resolve_chip_spec",
    "resolve_model_profile",
    "supports_hf_config",
    "verify_chip_spec_against_jax",
    "write_advisor_html",
]
