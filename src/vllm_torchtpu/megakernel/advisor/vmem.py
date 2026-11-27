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
"""VMEM floorplan, collective cost, core legality, and per-token memory modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, gcd, log2
from typing import TYPE_CHECKING

from ..core import errors
from .constants import (
    ARENA_ALIGN_RESERVE_BYTES,
    BF16_BYTES,
    CHIP_KIND_MAP,
    CORE_LIMITS,
    CORE_MODULE_FAMILIES,
    CORE_SUPPORTED_DTYPES,
    CORE_SUPPORTED_KV_DTYPES,
    CORE_SUPPORTED_STATE_DTYPES,
    DEFAULT_USABLE_VMEM_MIB,
    DOUBLE_BUFFER_SLOTS,
    DTYPE_BYTES,
    FP32_BYTES,
    FUSED_ELIGIBLE_PHASES,
    GROUPED_MOE_MODES,
    KV_ONLY_DTYPES,
    MIN_VMEM_MIB,
    MULTI_LAYER_FUSED_SCOPES,
    STRUCTURAL_LAYER_TAGS,
    SUBLAYER_BOUNDARIES_PER_LAYER,
    SUPPORTED_ONCHIP_FAMILIES,
    TURBOQUANT_SUPPORTED_PAGE_SIZES,
    VALID_DMA_BUFFERS,
    WORD_ALIGN_BYTES,
    canonical_kv_dtype,
)
from .models import has_dsa_profile
from .specs import (
    FusionScope,
    HostTableBreakSpec,
    RingPlan,
    SupportStatus,
    WorkloadSpec,
    default_double_buffer_ring_plan,
)

if TYPE_CHECKING:
    from ..advisor.models import ModelProfile
    from ..core.exec_spec import Tiles
    from .chips import ChipSpec
    from .roofline import _ScopeContext


@dataclass(frozen=True)
class _AllReduceTransferCost:
    """Single-slice ICI all-reduce volume, hop count, latency, and transfer duration."""

    transfer_time_s: float
    base_latency_s: float
    hop_count: int
    nominal_volume_bytes: float


@dataclass(frozen=True)
class _AdaptiveRingChoice:
    """Selected dense and expert DMA ring buffer counts and VMEM byte totals."""

    dense_slots: int
    expert_slots: int
    ring_bytes: int
    min_ring_bytes: int


@dataclass(frozen=True)
class _ScopeVmemBudget:
    """Per-scope static VMEM bytes, activation traffic, op count, and row tiling."""

    static_vmem_bytes: int
    act_bytes: float
    ops_per_step: int
    max_vmem_tokens: int
    num_vmem_splits: int
    feasible: bool
    reason: str
    selected_dense_slots: int
    selected_expert_slots: int


@dataclass(frozen=True)
class CoreLegalityContext:
    """Sharded execution counts checked against hardware and core limits."""

    active_tokens: int
    eff_tp: int
    ep_dp: int
    plan_error: str = ""


def effective_tp(workload: WorkloadSpec) -> int:
    """Return effective TP, promoting `dcp` when `tp == 1` and active PCP is 1."""
    if workload.tp == 1 and effective_pcp(workload) == 1:
        return max(1, workload.dcp)
    return max(1, workload.tp)


def effective_pcp(workload: WorkloadSpec) -> int:
    """Return active PCP ranks (`workload.pcp` in prefill/mixed, else 1)."""
    return max(1, workload.pcp) if workload.phase in ("prefill", "mixed") else 1


def fit_vmem_budget_bytes(
    tiles: Tiles | None = None,
    ring_plan: RingPlan | None = None,
    chip_spec: ChipSpec | None = None,
    *,
    vmem_budget_mib: float | None = None,
    apply_ring_reserve: bool = True,
) -> int:
    """Compute effective per-core VMEM budget in bytes after core headroom.

    Args:
        tiles: Optional core `Tiles` definition providing `vmem_mib`.
        ring_plan: Optional `RingPlan` carrying `compiler_spill_reserve_bytes`.
        chip_spec: Optional chip hardware specification.
        vmem_budget_mib: Explicit MiB override taking precedence over chip defaults.
        apply_ring_reserve: Whether to subtract `compiler_spill_reserve_bytes`.

    Returns:
        Usable VMEM byte budget available to ring buffers and token scratch.
    """
    selected_ring = ring_plan or default_double_buffer_ring_plan()
    if vmem_budget_mib is not None:
        fit_base = int(vmem_budget_mib * (1 << 20))
    else:
        if tiles is not None:
            if tiles.vmem_mib <= 0:
                return 0
            limit_bytes = tiles.vmem_mib << 20
        elif chip_spec is not None:
            limit_bytes = chip_spec.vmem_usable_bytes
        else:
            limit_bytes = DEFAULT_USABLE_VMEM_MIB << 20
        headroom_bytes = (
            CORE_LIMITS.fit_vmem_headroom_mib << 20
            if tiles is not None
            else (
                chip_spec.vmem_fit_headroom_bytes
                if chip_spec is not None
                else CORE_LIMITS.fit_vmem_headroom_mib << 20
            )
        )
        fit_base = max(0, limit_bytes - headroom_bytes)
    reserve = selected_ring.compiler_spill_reserve_bytes if apply_ring_reserve else 0
    vmem_raw = max(0, fit_base - reserve)
    if chip_spec is not None and tiles is None and vmem_budget_mib is None:
        return min(chip_spec.vmem_usable_bytes, vmem_raw)
    return vmem_raw


def compute_all_reduce_transfer_cost(
    message_bytes: float,
    effective_tp_ranks: int,
    chip_spec: ChipSpec,
    collective_latency_s: float | None = None,
) -> _AllReduceTransferCost:
    """Compute transfer time, latency, hop count, and volume for All-Reduce.

    Models single-slice optical ICI ring/torus reduce-scatter + all-gather
    across `effective_tp_ranks` ranks (`effective_tp_ranks > 1`). Each step
    moves one link's share over `ici_link_bw_bytes_per_s`.

    Returns:
        `_AllReduceTransferCost` with transfer duration, latency, hops, and bytes.
    """
    if effective_tp_ranks <= 1 or message_bytes <= 0.0:
        return _AllReduceTransferCost(0.0, 0.0, 0, 0.0)

    nominal_volume_bytes = (
        2.0 * (effective_tp_ranks - 1) / effective_tp_ranks * message_bytes
    )
    hop_count = round(log2(effective_tp_ranks))
    coll_lat = (
        chip_spec.collective_latency_s
        if collective_latency_s is None
        else collective_latency_s
    )
    transfer_time_s = nominal_volume_bytes / chip_spec.ici_link_bw_bytes_per_s
    base_latency_s = hop_count * coll_lat
    return _AllReduceTransferCost(
        transfer_time_s=transfer_time_s,
        base_latency_s=base_latency_s,
        hop_count=hop_count,
        nominal_volume_bytes=nominal_volume_bytes,
    )


def dsa_query_gather_dim(profile: ModelProfile) -> int:
    """Return per-head query width gathered across DCP ranks for DSA attention."""
    if profile.kv_lora_rank > 0:
        return profile.kv_lora_rank + profile.qk_rope_head_dim
    return profile.head_dim


def wide_dcp_kv_heads_per_rank(
    num_kv_heads: int, tp: int, dcp: int, dcp_span: str = "kv_heads"
) -> int:
    """Return local KV head groups stored per rank under standard or Wide-DCP.

    Under `dcp_span="tp_group"`, when `tp > num_kv_heads`, each KV head is
    replicated `kv_head_replicas = max(1, tp // num_kv_heads)` times across TP
    ranks before DCP shards sequences across replicas, giving
    `dcp_kv_heads = max(1, dcp // kv_head_replicas)` locally cached heads (with
    fractional ratios rounded via `ceil(num_kv_heads * dcp / tp)`).
    """
    if num_kv_heads <= 0:
        return 0
    eff_tp = max(1, tp)
    eff_dcp = max(1, dcp)
    if dcp_span == "tp_group":
        return min(num_kv_heads, max(1, ceil(num_kv_heads * eff_dcp / eff_tp)))
    kv_split_tp = min(eff_dcp, eff_tp)
    return max(1, ceil(num_kv_heads / max(1, eff_tp // kv_split_tp)))


def effective_host_table_hbm_resident(
    table: HostTableBreakSpec, workload: WorkloadSpec
) -> bool:
    """Return whether `table` is HBM-resident after applying workload overrides."""
    for name, is_hbm in workload.effective_host_table_hbm_overrides:
        if name == table.name:
            return bool(is_hbm)
    return table.hbm_resident


def active_fusion_break_layers(
    profile: ModelProfile, workload: WorkloadSpec
) -> tuple[int, ...]:
    """Return sorted 0-based layer indices that force fusion-break boundaries."""
    breaks: set[int] = set()
    for table in profile.host_tables:
        if not effective_host_table_hbm_resident(table, workload):
            for lyr in table.layers:
                if 0 < lyr < profile.num_layers:
                    breaks.add(lyr)
    if (
        workload.effective_execution == "ced"
        and profile.ced_boundary > 0
        and workload.phase in ("prefill", "mixed")
        and 0 < profile.ced_boundary < profile.num_layers
    ):
        breaks.add(profile.ced_boundary)
    return tuple(sorted(breaks))


def fusion_break_layer_groups(
    num_layers: int, break_layers: tuple[int, ...]
) -> tuple[int, ...]:
    """Partition `num_layers` at sorted `break_layers` into contiguous group sizes."""
    if not break_layers or num_layers <= 1:
        return ()
    valid_cuts = sorted({lyr for lyr in break_layers if 0 < lyr < num_layers})
    if not valid_cuts:
        return ()
    bounds = (0, *valid_cuts, num_layers)
    return tuple(hi - lo for lo, hi in pairwise(bounds))


def _dcp_divides_mesh(profile: ModelProfile, eff_tp: int, dcp: int, pcp: int) -> bool:
    """Return whether `dcp` satisfies the profile's `dcp_span` divisibility rule."""
    if profile.dcp_span == "tp_group":
        if eff_tp % dcp != 0:
            return False
        if profile.num_kv_heads > 0:
            replicas = max(1, eff_tp // profile.num_kv_heads)
            return replicas % dcp == 0 or dcp % replicas == 0
        return True
    return (eff_tp * pcp) % dcp == 0


def _attention_vmem_bytes(
    profile: ModelProfile, tp_ranks: int, dcp: int, kv_split: int
) -> tuple[int, int]:
    """Return `(attn_scratch_bytes, shared_indexer_spine_bytes)` per active token."""
    has_dsa = has_dsa_profile(profile)
    indexer_indices_bytes = (
        WORD_ALIGN_BYTES * ceil(profile.index_topk / kv_split) if has_dsa else 0
    )
    shared_spine = (
        indexer_indices_bytes
        if has_dsa and 0 < profile.num_indexer_layers < profile.num_attention_layers
        else 0
    )
    local_attn_heads = max(1, ceil(profile.num_attention_heads / tp_ranks))
    if profile.kv_lora_rank > 0:
        attn_bytes = 2 * (
            profile.q_lora_rank + profile.kv_lora_rank + profile.qk_rope_head_dim
        ) + 6 * (local_attn_heads * profile.head_dim)
    else:
        local_kv_heads = (
            wide_dcp_kv_heads_per_rank(profile.num_kv_heads, tp_ranks, dcp, "tp_group")
            if profile.dcp_span == "tp_group"
            else max(1, ceil(profile.num_kv_heads / max(1, tp_ranks // kv_split)))
        )
        attn_bytes = 2 * (
            2 * local_attn_heads + 2 * local_kv_heads
        ) * profile.head_dim + (4 * local_attn_heads * profile.head_dim)
    if profile.o_lora_rank > 0:
        attn_bytes += profile.o_groups * profile.o_lora_rank * BF16_BYTES
    if profile.sparse_attn is not None:
        sp = profile.sparse_attn
        attn_bytes += sp.index_heads * sp.index_dim * BF16_BYTES + (
            WORD_ALIGN_BYTES * ceil((sp.window_tokens + sp.select_tokens) / kv_split)
        )
    kv_norm = canonical_kv_dtype(profile.kv_dtype)
    if kv_norm in KV_ONLY_DTYPES:
        attn_bytes += 2 * profile.head_dim * FP32_BYTES
    if has_dsa:
        query_gather_dim = dsa_query_gather_dim(profile)
        attn_bytes += (
            BF16_BYTES
            * (
                profile.index_n_heads * profile.index_head_dim
                + profile.index_head_dim
                + profile.index_n_heads
            )
            + (0 if shared_spine else indexer_indices_bytes)
            + (
                BF16_BYTES * (dcp - 1) * local_attn_heads * query_gather_dim
                if dcp > 1
                else 0
            )
        )
    return attn_bytes, shared_spine


def _moe_or_dense_vmem_bytes(
    profile: ModelProfile, workload: WorkloadSpec, tp_ranks: int, ep: int
) -> int:
    """Return per-token MoE or dense FFN VMEM scratchpad bytes."""
    hidden = profile.hidden_size
    expert_hidden = profile.routed_expert_hidden_size or hidden
    if profile.num_moe_layers <= 0 or profile.num_experts <= 0:
        if profile.intermediate_size <= 0:
            return 0
        return 6 * hidden + 8 * ceil(profile.intermediate_size / tp_ranks)

    ep_tp = gcd(ep, tp_ranks)
    inter_local = ceil(profile.moe_intermediate_size / max(1, tp_ranks // ep_tp))
    shared_local = (
        ceil(profile.shared_expert_intermediate_size / tp_ranks)
        if profile.shared_expert_intermediate_size > 0
        else 0
    )
    router_bytes = 4 * profile.num_experts + 8 * profile.top_k
    if workload.moe_mode in GROUPED_MOE_MODES:
        eff_top_k = min(profile.top_k, DOUBLE_BUFFER_SLOTS * ep) if ep > 1 else 1
        tile_bytes = (
            8 * expert_hidden
            + ceil(eff_top_k * (6 * expert_hidden + 8 * inter_local) / ep)
            + 8 * shared_local
        )
    else:
        tile_bytes = 16 * expert_hidden + 8 * (inter_local + shared_local)
    if workload.vmem_alias:
        return max(router_bytes + 2 * expert_hidden, tile_bytes)
    return router_bytes + tile_bytes


def compute_per_token_vmem_bytes(
    profile: ModelProfile, workload: WorkloadSpec
) -> dict[str, int]:
    """Compute per-token live VMEM bytes from tensor lifetimes and sharding.

    Returns:
        Mapping of VMEM lifetime component byte counts per active token.
    """
    tp_ranks = effective_tp(workload)
    ep = max(1, workload.ep)
    dcp = max(1, workload.dcp)
    kv_split = max(1, workload.kv_split, workload.dcp)
    attn_bytes, shared_spine = _attention_vmem_bytes(profile, tp_ranks, dcp, kv_split)
    hc = profile.hyper_connection
    streams = max(1, hc.num_streams if hc is not None else profile.residual_streams)
    hc_scratch_bytes = (
        (
            2 * hc.lowrank_dim * BF16_BYTES
            if hc.kind == "lowrank"
            else 4 * streams * streams * FP32_BYTES
        )
        if hc is not None
        else 0
    )
    residual_bytes = (
        (4 * streams + 2) * profile.hidden_size
        + (6 * profile.hidden_size if tp_ranks > 1 else 0)
        + shared_spine
        + hc_scratch_bytes
    )
    if profile.num_linear_layers > 0 and profile.linear_value_heads > 0:
        local_key_heads = max(1, ceil(profile.linear_key_heads / tp_ranks))
        local_val_heads = max(1, ceil(profile.linear_value_heads / tp_ranks))
        linear_bytes = 2 * (
            2 * local_key_heads * profile.linear_key_dim
            + 2 * local_val_heads * profile.linear_value_dim
        ) + 4 * (local_val_heads * profile.linear_value_dim)
    else:
        linear_bytes = 0
    moe_bytes = _moe_or_dense_vmem_bytes(profile, workload, tp_ranks, ep)
    per_op_bytes = (
        max(attn_bytes, linear_bytes, moe_bytes)
        if workload.vmem_alias
        else (attn_bytes + linear_bytes + moe_bytes)
    )
    total_bytes = residual_bytes + per_op_bytes
    return {
        "residual_buffer_bytes": residual_bytes,
        "attn_scratch_bytes": attn_bytes,
        "linear_attn_scratch_bytes": linear_bytes,
        "moe_scratch_bytes": moe_bytes,
        "total_per_token_bytes": total_bytes,
    }


def is_slice_topology_supported(slice_cores: int, chip_spec: ChipSpec) -> bool:
    """Return True when slice_cores matches a built-in torus or valid multi-slice.

    Returns:
        Boolean indicating whether the slice topology is physically supported.
    """
    if (
        slice_cores < chip_spec.min_slice_cores
        or (slice_cores & (slice_cores - 1)) != 0
    ):
        return False
    if not chip_spec.topologies:
        return True
    max_topo = max(chip_spec.topologies)
    return slice_cores in chip_spec.topologies or (
        slice_cores > max_topo and slice_cores % chip_spec.min_slice_cores == 0
    )


def _scalar_range_issues(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None,
) -> list[tuple[str, str]]:
    """Collect scalar range, buffer count, and VMEM budget violations."""
    issues: list[tuple[str, str]] = []
    pos_fields = (
        profile.num_layers,
        profile.hidden_size,
        profile.num_attention_heads,
        profile.head_dim,
        workload.batch,
        workload.tp,
        workload.pp,
        workload.dp,
        workload.ep,
        workload.kv_split,
        workload.dcp,
        workload.pcp,
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
        workload.max_rows_per_call,
        workload.latent_proj_tp,
    )
    if min(*pos_fields) < 1 or min(*nonneg_fields) < 0 or workload.op_overhead_us < 0.0:
        return [("invalid_dimensions", "Dimensions and parallelism must be >= 1")]
    if workload.dma_buffers not in VALID_DMA_BUFFERS:
        issues.append(
            (
                "invalid_dma_buffers",
                f"dma_buffers must be 2 or 3, got {workload.dma_buffers!r}",
            )
        )
    if not math.isfinite(workload.hbm_utilization_cap) or not (
        0.0 < workload.hbm_utilization_cap <= 1.0
    ):
        issues.append(
            (
                "invalid_hbm_utilization_cap",
                "hbm_utilization_cap must be in (0, 1], "
                f"got {workload.hbm_utilization_cap!r}",
            )
        )
    if workload.vmem_budget_mib is not None and workload.vmem_budget_mib < MIN_VMEM_MIB:
        issues.append(
            ("invalid_vmem_budget", f"vmem_budget_mib must be >= {MIN_VMEM_MIB}")
        )
    if chip_spec is not None and workload.vmem_budget_mib is not None:
        max_usable_mib = chip_spec.vmem_usable_bytes / (1 << 20)
        if workload.vmem_budget_mib > max_usable_mib:
            phys_mib = chip_spec.vmem_capacity_bytes / (1 << 20)
            issues.append(
                (
                    "vmem_budget_exceeds_chip",
                    f"vmem_budget_mib ({workload.vmem_budget_mib}) exceeds usable VMEM "
                    f"ceiling ({max_usable_mib:.1f} MiB usable out of "
                    f"{phys_mib:.1f} MiB physical) on chip {chip_spec.name!r}",
                )
            )
    if not math.isfinite(workload.prefix_hit_rate) or not (
        0.0 <= workload.prefix_hit_rate < 1.0
    ):
        issues.append(
            ("invalid_prefix_hit_rate", "Invalid parallelism, layers, or dtype")
        )
    if workload.tiles is not None and workload.tiles.vmem_mib < MIN_VMEM_MIB:
        issues.append(
            ("invalid_tiles_vmem_mib", "Invalid parallelism, layers, or dtype")
        )
    return issues


def _coupling_and_layer_issues(
    profile: ModelProfile,
    workload: WorkloadSpec,
) -> list[tuple[str, str]]:
    """Collect batch coupling, parallel shorthand, layer group, and dtype errors."""
    issues: list[tuple[str, str]] = []
    if workload.batch < workload.dp or workload.batch % workload.dp != 0:
        issues.append(
            (
                "batch_not_divisible_by_dp",
                f"batch ({workload.batch}) must be >= dp ({workload.dp}) "
                "and divisible by dp",
            )
        )
    if workload.kv_split > 1 and profile.num_kv_heads == 0:
        issues.append(
            (
                "kv_split_without_kv_heads",
                "kv_split > 1 is invalid when model has zero KV heads",
            )
        )
    if workload.phase == "verify" and workload.spec_tokens < 1:
        issues.append(
            (
                "verify_without_spec_tokens",
                f"phase='verify' requires spec_tokens >= 1, got {workload.spec_tokens}",
            )
        )
    verify_tokens = max(1, workload.spec_tokens) + 1
    if workload.phase == "verify" and 0 < workload.max_rows_per_call < verify_tokens:
        issues.append(
            (
                "row_limit_below_request",
                f"max_rows_per_call ({workload.max_rows_per_call}) is below one verify "
                f"request ({verify_tokens} rows)",
            )
        )
    if workload.tp == 1 and workload.dcp > 1 and effective_pcp(workload) > 1:
        issues.append(
            (
                "dcp_with_pcp_requires_explicit_tp",
                "tp=1 shorthand with dcp > 1 is invalid when pcp > 1",
            )
        )
    kv_dt = canonical_kv_dtype(profile.kv_dtype)
    non_kv_dtypes = tuple(
        dt.lower()
        for dt in (
            profile.weight_dtype,
            profile.expert_dtype or profile.weight_dtype,
            profile.state_dtype,
            *((profile.act_dtype,) if profile.act_dtype else ()),
            *((profile.expert_act_dtype,) if profile.expert_act_dtype else ()),
        )
    )
    layers_bad = (
        profile.num_attention_layers + profile.num_linear_layers > profile.num_layers
        or profile.num_moe_layers > profile.num_layers
        or (
            bool(workload.layer_groups)
            and (
                sum(workload.layer_groups) != profile.num_layers
                or min(workload.layer_groups) < 1
            )
        )
    )
    dtypes_bad = kv_dt not in DTYPE_BYTES or any(
        dt not in DTYPE_BYTES or dt in KV_ONLY_DTYPES for dt in non_kv_dtypes
    )
    if dtypes_bad or layers_bad:
        issues.append(
            ("invalid_layers_or_dtype", "Invalid parallelism, layers, or dtype")
        )
    if (
        kv_dt == "turboquant_4bit_nc"
        and workload.kv_page_size not in TURBOQUANT_SUPPORTED_PAGE_SIZES
    ):
        issues.append(
            (
                "unsupported_kv_page_size",
                "kv_dtype='turboquant_4bit_nc' requires kv_page_size in "
                f"{TURBOQUANT_SUPPORTED_PAGE_SIZES}, got {workload.kv_page_size}",
            )
        )
    return issues


def _scalar_and_coupling_issues(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None,
) -> list[tuple[str, str]]:
    """Collect hard scalar, budget, coupling, layer, and dtype validation issues."""
    issues = _scalar_range_issues(profile, workload, chip_spec)
    if issues and issues[0][0] == "invalid_dimensions":
        return issues
    issues.extend(_coupling_and_layer_issues(profile, workload))
    return issues


def _topology_issue_pairs(
    tp_ranks: int,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None,
) -> list[tuple[str, bool]]:
    """Collect `(reason_tag, is_hard_error)` pairs for rank powers and slice shape."""
    pairs: list[tuple[str, bool]] = []
    if tp_ranks < 1 or (tp_ranks & (tp_ranks - 1)) != 0:
        pairs.append(("tp_not_power_of_two", True))
    if workload.dp < 1 or (workload.dp & (workload.dp - 1)) != 0:
        pairs.append(("dp_not_power_of_two", True))
    if workload.pcp < 1 or (workload.pcp & (workload.pcp - 1)) != 0:
        pairs.append(("pcp_not_power_of_two", True))
    if chip_spec is not None and not is_slice_topology_supported(
        tp_ranks * workload.dp * workload.pp * effective_pcp(workload), chip_spec
    ):
        pairs.append(("slice_topology_unsupported", False))
    return pairs


def _head_issue_pairs(
    profile: ModelProfile, workload: WorkloadSpec, tp_ranks: int
) -> list[tuple[str, bool]]:
    """Collect `(reason_tag, is_hard_error)` pairs for attention/linear head splits."""
    pairs: list[tuple[str, bool]] = []
    if tp_ranks > 0 and profile.num_attention_heads % tp_ranks != 0:
        pairs.append(("attn_heads_not_divisible_by_tp", False))
    kv_split = max(1, workload.kv_split)
    kv_tp = max(1, tp_ranks // kv_split)
    if tp_ranks % kv_split != 0:
        pairs.append(("kv_heads_not_divisible_by_tp", True))
    elif (profile.num_kv_heads == 0 and kv_split > 1) or (
        profile.num_kv_heads > 0
        and profile.num_kv_heads % kv_tp != 0
        and kv_tp % profile.num_kv_heads != 0
    ):
        pairs.append(("kv_heads_not_divisible_by_tp", False))
    if profile.num_linear_layers > 0:
        if (
            profile.linear_value_heads > 0
            and profile.linear_value_heads % tp_ranks != 0
        ):
            pairs.append(("linear_v_heads_not_divisible_by_tp", False))
        if (
            profile.linear_key_heads > 0
            and profile.linear_key_heads % tp_ranks != 0
            and tp_ranks % profile.linear_key_heads != 0
        ):
            pairs.append(("linear_k_heads_not_divisible_by_tp", False))
    return pairs


def _mesh_issue_pairs(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None,
) -> list[tuple[str, bool]]:
    """Collect `(reason_tag, is_hard_error)` pairs for sharding and topology."""
    tp_ranks = effective_tp(workload)
    pcp_ranks = effective_pcp(workload)
    pairs = _topology_issue_pairs(tp_ranks, workload, chip_spec)
    pairs.extend(_head_issue_pairs(profile, workload, tp_ranks))
    if not _dcp_divides_mesh(profile, tp_ranks, max(1, workload.dcp), pcp_ranks):
        pairs.append(("dcp_not_dividing_tp", True))
    if (
        profile.sparse_attn is not None
        and profile.sparse_attn.kind == "csa"
        and workload.dcp > 1
    ):
        pairs.append(("csa_dcp_unsupported", True))
    if profile.num_experts > 0 and (
        profile.num_experts % max(1, workload.ep) != 0
        or not (1 <= profile.top_k <= profile.num_experts)
    ):
        pairs.append(("experts_not_divisible_by_ep", True))
    if profile.num_experts == 0 and profile.top_k != 0:
        pairs.append(("experts_not_divisible_by_ep", True))
    if (tp_ranks * workload.dp * pcp_ranks) % max(1, workload.ep) != 0:
        pairs.append(("ep_exceeds_slice_ranks", True))
    if workload.pp > profile.num_layers:
        pairs.append(("pp_exceeds_layers", True))
    return pairs


def _mesh_divisibility_reasons(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None = None,
) -> list[str]:
    """Return non-raising sharding divisibility and topology violation codes."""
    return [tag for tag, _ in _mesh_issue_pairs(profile, workload, chip_spec)]


def collect_validation_reasons(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None = None,
) -> tuple[list[str], bool]:
    """Collect unified validation reason tags and whether any hard error failed."""
    scalar_issues = _scalar_and_coupling_issues(profile, workload, chip_spec)
    reasons = [tag for tag, _ in scalar_issues]
    has_hard = bool(scalar_issues)
    if "invalid_dimensions" in reasons:
        return reasons, True
    for tag, hard in _mesh_issue_pairs(profile, workload, chip_spec):
        if tag not in reasons:
            reasons.append(tag)
        has_hard = has_hard or hard
    return reasons, has_hard


def validate_advise_inputs(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec | None = None,
) -> None:
    """Validate ModelProfile and WorkloadSpec field ranges and physical constraints.

    Raises:
        MegakernelConfigError: If any scalar bound, coupling, layer, or dtype fails.
    """
    scalar_issues = _scalar_and_coupling_issues(profile, workload, chip_spec)
    if scalar_issues:
        raise errors.MegakernelConfigError(scalar_issues[0][1])
    for tag, hard in _mesh_issue_pairs(profile, workload, chip_spec):
        if hard:
            msg = f"Invalid parallelism, layers, or dtype ({tag})"
            raise errors.MegakernelConfigError(msg)


def exceeds_default_buckets(workload: WorkloadSpec, rows_per_call: int) -> bool:
    """Return whether one call is larger than the core's default row buckets.

    Returns:
        True when `rows_per_call` exceeds the phase's default compiled bucket top.
    """
    tops = {
        "decode": CORE_LIMITS.default_decode_bucket_top,
        "verify": CORE_LIMITS.default_verify_bucket_top,
    }
    return rows_per_call > tops.get(workload.phase, rows_per_call)


def evaluate_core_legality(
    scope: FusionScope,
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    legality_ctx: CoreLegalityContext,
) -> tuple[SupportStatus, tuple[str, ...]]:
    """Evaluate hardware and core-limit legality without runtime declarations.

    Returns:
        `(support_status, blockers)` classification across the 3 support states.
    """
    if scope == "per_op" or workload.batch <= 0:
        return ("baseline", ())
    families = [
        family
        for family in profile.layer_families
        if family not in STRUCTURAL_LAYER_TAGS
    ]
    family_gap = any(
        family in SUPPORTED_ONCHIP_FAMILIES and family not in CORE_MODULE_FAMILIES
        for family in families
    )
    dtype_gap = any(
        dtype.lower() not in CORE_SUPPORTED_DTYPES
        for dtype in (
            profile.weight_dtype,
            profile.expert_dtype or profile.weight_dtype,
            *((profile.expert_act_dtype,) if profile.expert_act_dtype else ()),
        )
    )
    kv_norm = canonical_kv_dtype(profile.kv_dtype)
    cache_dtype_gap = (
        profile.num_attention_layers > 0 and kv_norm not in CORE_SUPPORTED_KV_DTYPES
    ) or (
        profile.num_linear_layers > 0
        and profile.state_dtype.strip().lower() not in CORE_SUPPORTED_STATE_DTYPES
    )
    planner_bad = (
        bool(legality_ctx.plan_error)
        or not profile.is_periodic
        or profile.ar_slots_per_layer > CORE_LIMITS.max_ar_slots_per_layer
    )
    has_active_break = scope == "whole_step" and bool(
        active_fusion_break_layers(profile, workload)
    )
    checks = (
        (scope in ("single_layer", "layer_group"), "scope_without_core_counterpart"),
        (
            legality_ctx.active_tokens > CORE_LIMITS.max_batch_rows,
            "rows_exceed_core_max_batch_rows",
        ),
        (
            legality_ctx.eff_tp not in CORE_LIMITS.verified_ranks,
            "rank_count_outside_core_grid",
        ),
        (
            CHIP_KIND_MAP.get(chip_spec.name.lower()) not in CORE_LIMITS.chip_kinds,
            "chip_without_core_budget",
        ),
        (workload.pp > 1, "pipeline_parallel_not_in_core"),
        (effective_pcp(workload) > 1, "context_parallel_not_in_core"),
        (workload.phase not in FUSED_ELIGIBLE_PHASES, "phase_not_in_core"),
        (legality_ctx.ep_dp > 1, "all_to_all_not_in_core"),
        (family_gap, "family_not_in_core"),
        (has_active_break, "fusion_break"),
        (dtype_gap, "dtype_not_in_core"),
        (cache_dtype_gap, "kv_or_state_dtype_not_in_core"),
        (planner_bad, "planner_rejected"),
        (
            scope == "whole_step"
            and workload.max_rows_per_call == 0
            and exceeds_default_buckets(workload, legality_ctx.active_tokens),
            "row_limit_not_stated",
        ),
    )
    blockers = tuple(tag for cond, tag in checks if cond)
    if blockers:
        return ("outside_core_limits", blockers)
    return ("within_core_limits", ())


def _select_adaptive_ring(
    scope_context: _ScopeContext,
    workload: WorkloadSpec,
    fixed_non_ring_bytes: int,
) -> _AdaptiveRingChoice:
    """Select widest `(dense_slots, expert_slots)` fitting `active_tokens` in VMEM."""
    has_moe = scope_context.local_moe_layers > 0
    if workload.tiles is not None:
        ring_cfg: tuple[int, int, int, int] | None = (
            workload.tiles.nbd,
            workload.tiles.nbe,
            workload.tiles.bank,
            workload.tiles.ebank,
        )
    elif workload.ring_plan is not None:
        ring_plan = workload.ring_plan
        ring_cfg = (
            ring_plan.dense_slots,
            ring_plan.expert_slots,
            ring_plan.dense_slot_bytes,
            ring_plan.expert_slot_bytes,
        )
    else:
        ring_cfg = None
    if ring_cfg is not None:
        dense_slots, raw_expert_slots, dense_bank, expert_bank = ring_cfg
        expert_slots = raw_expert_slots if has_moe else 0
        ring_bytes = dense_slots * dense_bank + expert_slots * expert_bank
        return _AdaptiveRingChoice(dense_slots, expert_slots, ring_bytes, ring_bytes)
    min_dense_slots = workload.dma_buffers
    extra = min_dense_slots - DOUBLE_BUFFER_SLOTS
    max_dense_slots = CORE_LIMITS.dense_ring_slots + extra
    min_expert_slots = min_dense_slots if has_moe else 0
    max_expert_slots = (CORE_LIMITS.expert_ring_slots + extra) if has_moe else 0
    min_ring_bytes = (
        min_dense_slots * CORE_LIMITS.dense_slot_bytes
        + min_expert_slots * CORE_LIMITS.expert_slot_bytes
    )
    avail = (
        scope_context.vmem_budget
        - fixed_non_ring_bytes
        - scope_context.per_token_fused_bytes * scope_context.active_tokens
    )
    if avail < min_ring_bytes:
        return _AdaptiveRingChoice(
            min_dense_slots, min_expert_slots, min_ring_bytes, min_ring_bytes
        )
    best_key = (-1, -1, -1)
    best_choice = _AdaptiveRingChoice(
        min_dense_slots, min_expert_slots, min_ring_bytes, min_ring_bytes
    )
    for dense_slots in range(min_dense_slots, max_dense_slots + 1):
        for expert_slots in range(min_expert_slots, max_expert_slots + 1):
            ring_bytes = (
                dense_slots * CORE_LIMITS.dense_slot_bytes
                + expert_slots * CORE_LIMITS.expert_slot_bytes
            )
            if ring_bytes <= avail:
                balance = (
                    dense_slots - min_dense_slots
                    if not has_moe
                    else min(
                        dense_slots - min_dense_slots,
                        expert_slots - min_expert_slots,
                    )
                )
                cand_key = (ring_bytes, balance, dense_slots)
                if cand_key > best_key:
                    best_key = cand_key
                    best_choice = _AdaptiveRingChoice(
                        dense_slots, expert_slots, ring_bytes, min_ring_bytes
                    )
    return best_choice


def _scope_vmem_and_ops(
    scope: FusionScope,
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
) -> _ScopeVmemBudget:
    """Compute static VMEM, activation HBM bytes, ops, splits, and ring slots."""
    if scope == "per_op":
        static_vmem_bytes = max(1 << 20, scope_context.single_layer_dense_staging_bytes)
        alpha = FP32_BYTES * profile.hidden_size
        per_op_ok = static_vmem_bytes + alpha <= scope_context.vmem_budget
        return _ScopeVmemBudget(
            static_vmem_bytes=static_vmem_bytes,
            act_bytes=(
                4.0
                * max(1, profile.residual_streams)
                * scope_context.layers_local
                * scope_context.token_residual_bytes
            ),
            ops_per_step=SUBLAYER_BOUNDARIES_PER_LAYER * scope_context.layers_local,
            max_vmem_tokens=(
                (scope_context.vmem_budget - static_vmem_bytes) // alpha
                if per_op_ok
                else 0
            ),
            num_vmem_splits=1,
            feasible=per_op_ok,
            reason="ok" if per_op_ok else "per_op_vmem_overflow",
            selected_dense_slots=0,
            selected_expert_slots=0,
        )

    feasible, reason = True, "ok"
    preset_groups = scope_context.num_layer_groups
    if (
        scope == "layer_group"
        and preset_groups == 0
        and workload.pp == 1
        and scope_context.num_segments == 1
    ):
        feasible = False
        reason = "No layer grouping: needs a model preset, pp > 1, or segment_layers"
    elif scope == "whole_step" and scope_context.no_whole_step:
        feasible = False
        if scope_context.plan_error:
            reason = scope_context.plan_error
        elif scope_context.has_fusion_break:
            reason = "Fusion break at host-table or CED boundary"
        else:
            reason = "Bounded by layer groups, single layers, or unmodeled layer family"

    scratch_op_bytes = (
        max(
            scope_context.gqa_scratch,
            scope_context.gdn_scratch,
            scope_context.ffn_scratch,
        )
        if workload.vmem_alias
        else (
            scope_context.gqa_scratch
            + scope_context.gdn_scratch
            + scope_context.ffn_scratch
        )
    )
    is_single_layer = scope == "single_layer"
    if is_single_layer or (scope == "layer_group" and bool(preset_groups)):
        dense_slots = workload.dma_buffers
        expert_slots = workload.dma_buffers if scope_context.local_moe_layers > 0 else 0
        expert_staging = (
            scope_context.single_layer_expert_staging_bytes
            if scope_context.local_moe_layers > 0
            else 0
        )
        static_vmem_bytes = (
            scope_context.single_layer_dense_staging_bytes
            + expert_staging
            + scratch_op_bytes
        )
        min_static_bytes = static_vmem_bytes
    else:
        arena_reserve = ARENA_ALIGN_RESERVE_BYTES if workload.vmem_fit_headroom else 0
        ring_choice = _select_adaptive_ring(
            scope_context, workload, scratch_op_bytes + arena_reserve
        )
        dense_slots = ring_choice.dense_slots
        expert_slots = ring_choice.expert_slots
        static_vmem_bytes = ring_choice.ring_bytes + scratch_op_bytes + arena_reserve
        min_static_bytes = ring_choice.min_ring_bytes + scratch_op_bytes + arena_reserve

    group_count = (
        (preset_groups or scope_context.num_segments) if scope == "layer_group" else 1
    )
    has_collectives = scope_context.eff_tp * workload.ep > 1 and (
        scope_context.local_attn_layers + scope_context.local_linear_layers > 0
    )
    if is_single_layer:
        ops_per_step = scope_context.layers_local
    elif scope == "layer_group" and preset_groups:
        ops_per_step = group_count
    else:
        ops_per_step = group_count * (3 if has_collectives else 1)
    per_token_bytes = scope_context.per_token_fused_bytes
    max_vmem_tokens = (
        0
        if min_static_bytes + per_token_bytes > scope_context.vmem_budget
        else max(1, (scope_context.vmem_budget - min_static_bytes) // per_token_bytes)
    )
    if scope in MULTI_LAYER_FUSED_SCOPES and scope_context.single_call_rows:
        max_vmem_tokens = min(max_vmem_tokens, scope_context.single_call_rows)
    num_vmem_splits = (
        scope_context.active_tokens
        if max_vmem_tokens == 0
        else ceil(scope_context.active_tokens / max_vmem_tokens)
    )
    if max_vmem_tokens <= 0:
        feasible, reason = False, "Weight buffers exceed VMEM"
    act_bytes = (
        2.0
        * (scope_context.layers_local if is_single_layer else group_count)
        * scope_context.token_residual_bytes
        * max(1, profile.residual_streams)
    )
    return _ScopeVmemBudget(
        static_vmem_bytes=static_vmem_bytes,
        act_bytes=act_bytes,
        ops_per_step=ops_per_step,
        max_vmem_tokens=max_vmem_tokens,
        num_vmem_splits=num_vmem_splits,
        feasible=feasible,
        reason=reason,
        selected_dense_slots=dense_slots,
        selected_expert_slots=expert_slots,
    )
