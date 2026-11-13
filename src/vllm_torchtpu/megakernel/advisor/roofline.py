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
"""First-principles roofline, scope assessment, and runtime bounds for TPU advisor."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from math import ceil, gcd
from typing import TYPE_CHECKING

from ..advisor.models import ModelProfile, resolve_model_profile
from ..core import errors
from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    ATTN_TILE_TOKENS,
    BOUNDARY_SYNC_HOP_FACTOR,
    COMPILED_XLA_GAP_P50_US,
    COMPILED_XLA_GAP_P90_US,
    CORES_PER_HOST,
    DECLARED_MOE_LAYOUTS,
    DEFAULT_LAUNCH_GAP_FLOOR_US,
    DEFAULT_LAUNCH_GAP_UPPER_US,
    DEFAULT_LAUNCH_GAP_VAL_US,
    DENSE_FFN_TILE_CAP,
    DENSE_SLOT_DEFAULT_BYTES,
    DTYPE_BYTES,
    EXPERT_SLOT_DEFAULT_BYTES,
    FIRST_SLOT_LAYER_WEIGHT_DIVISOR,
    FP32_BYTES,
    GDN_STREAMED_HEADS_CAP,
    GROUPED_MOE_MODES,
    MIN_FIRST_DMA_SLOT_BYTES,
    PER_OP_COLLECTIVE_CALL_FACTOR,
    SUB_BYTE_EXPERT_DTYPES,
    SUBCHANNEL_BLOCK_K_THRESHOLD,
    SUBCHANNEL_VPU_OPS_PER_BLOCK,
    UNFUSED_EAGER_GAP_US,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from .specs import (
    FusionScope,
    MeasurementsData,
    ProvenanceTag,
    ScopeEstimate,
    WorkloadSpec,
    default_double_buffer_ring_plan,
)
from .support import evaluate_support, find_declaration, single_call_row_limit
from .vmem import (
    _mesh_divisibility_reasons,
    _scope_vmem_and_ops,
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _request_rows(workload: WorkloadSpec) -> int:
    """Return the rows one request adds to a decode or verify call."""
    return max(1, workload.spec_tokens) + 1 if workload.phase == "verify" else 1


def _row_limit_below_request(profile: ModelProfile, workload: WorkloadSpec) -> bool:
    """Return whether the single-call row limit cannot hold one whole request."""
    eff_tp = max(workload.tp, workload.dcp) if workload.tp == 1 else workload.tp
    row_limit = single_call_row_limit(profile.name, workload, eff_tp)
    return 0 < row_limit < _request_rows(workload)


def _coupling_violation_reasons(
    profile: ModelProfile, workload: WorkloadSpec
) -> list[str]:
    """Return physical coupling violation tags without raising exceptions."""
    reasons: list[str] = []
    if workload.batch < workload.dp or workload.batch % workload.dp != 0:
        reasons.append("batch_not_divisible_by_dp")
    if workload.kv_split > 1 and profile.num_kv_heads == 0:
        reasons.append("kv_split_without_kv_heads")
    if workload.phase == "verify" and workload.spec_tokens < 1:
        reasons.append("verify_without_spec_tokens")
    return reasons


def validate_workload(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str,
    workload: WorkloadSpec,
) -> tuple[str, ...]:
    """Validate sharding divisibility and serving HBM pool fit without raising."""
    resolved_profile = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    pos_vals = (
        resolved_profile.num_layers,
        resolved_profile.hidden_size,
        resolved_profile.num_attention_heads,
        resolved_profile.head_dim,
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
    nonneg_vals = (
        workload.osl,
        workload.context,
        workload.spec_tokens,
        workload.chunk_size,
        workload.microbatches,
        workload.segment_layers,
    )
    if min(*pos_vals) < 1 or min(*nonneg_vals) < 0 or workload.op_overhead_us < 0.0:
        return ("invalid_dimensions",)
    reasons = [
        *_mesh_divisibility_reasons(resolved_profile, workload, chip_spec),
        *_coupling_violation_reasons(resolved_profile, workload),
    ]
    if _row_limit_below_request(resolved_profile, workload):
        return (*reasons, "row_limit_below_request")
    if reasons and (
        "batch_not_divisible_by_dp" in reasons
        or "verify_without_spec_tokens" in reasons
    ):
        return tuple(reasons)
    scope_context = _build_scope_context(
        resolved_profile, workload, chip_spec, plan_error=""
    )
    if (
        scope_context.hbm_used
        > workload.hbm_utilization_cap * chip_spec.hbm_capacity_bytes
    ):
        reasons.append("hbm_pool_overflow")
    return tuple(reasons)


def _experts_hit(
    tokens_ep: int,
    top_k: int,
    num_experts: int,
    experts_local: int,
) -> float:
    """Return expected distinct local experts hit by `tokens_ep` tokens."""
    miss = (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    return max(1.0, experts_local * (1.0 - miss))


def _strict_moe_weights(scope_context: _ScopeContext, profile: ModelProfile) -> float:
    """Return strict lower-bound step weight bytes under perfectly balanced routing."""
    if scope_context.experts_local == 0:
        return scope_context.w_static
    e_strict = min(
        float(scope_context.experts_local),
        max(
            1.0,
            profile.top_k * scope_context.experts_local / max(1, profile.num_experts),
        ),
    )
    return (
        scope_context.w_static
        + scope_context.l_moe * e_strict * scope_context.exp_bytes
    )


def _moe_weights_and_padding(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    tile_tokens: int,
    *,
    tiled: bool,
    grouped: bool,
) -> tuple[float, float]:
    """Return weight bytes read per step and the MoE matmul padding factor."""
    experts_local, ep_dp = scope_context.experts_local, scope_context.ep_dp
    if experts_local == 0:
        return scope_context.w_static, 1.0

    t_ep = tile_tokens * ep_dp
    e_hit = _experts_hit(t_ep, profile.top_k, profile.num_experts, experts_local)
    pairs = max(1.0, t_ep * profile.top_k * experts_local / profile.num_experts)
    avg_m = max(1.0, pairs / e_hit)
    hit = (
        _experts_hit(
            scope_context.active_tokens * ep_dp,
            profile.top_k,
            profile.num_experts,
            experts_local,
        )
        if tiled
        else e_hit
    )
    w_step = (
        scope_context.w_static + scope_context.l_moe * hit * scope_context.exp_bytes
    )
    if grouped:
        return w_step, ceil(
            avg_m / scope_context.row_floor
        ) * scope_context.row_floor / avg_m
    rup_16 = ((t_ep + 15) // 16) * 16
    return w_step, max(1.0, (rup_16 * e_hit) / pairs)


def can_group_moe_in_scope(
    scope: FusionScope,
    eff_tp: int,
    ep_dp: int,
    moe_mode: str,
) -> bool:
    """Return whether `scope` can group routed expert tokens on chip or in HBM."""
    return (
        scope not in ("whole_step", "layer_group")
        or ep_dp == 1
        or (eff_tp * ep_dp <= CORES_PER_HOST and moe_mode in GROUPED_MOE_MODES)
    )


def _resolve_scope_moe_and_acts(
    scope: FusionScope,
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    n_splits: int,
) -> tuple[float, float, float, int, bool]:
    """Return w_step, phi, extra activation bytes, tile_tokens, and grouped_moe."""
    tiled = scope in ("whole_step", "layer_group") and n_splits > 1
    tile_tokens = ceil(scope_context.active_tokens / max(1, n_splits))
    can_group_moe = can_group_moe_in_scope(
        scope, scope_context.eff_tp, scope_context.ep_dp, workload.moe_mode
    )
    grouped_moe = can_group_moe and (
        scope == "per_op"
        or workload.moe_mode in GROUPED_MOE_MODES
        or tile_tokens * scope_context.ep_dp >= scope_context.row_floor
    )
    declaration = find_declaration(
        profile.name, scope, scope_context.eff_tp, workload.phase
    )
    declared_layout = str(declaration.get("moe_layout", "")) if declaration else ""
    if declared_layout in DECLARED_MOE_LAYOUTS:
        # A declared megakernel runs the expert layout its deployment states.
        grouped_moe = declared_layout == "grouped"
    moe_tokens = (
        scope_context.active_tokens
        if grouped_moe and scope != "single_layer"
        else tile_tokens
    )
    w_step, moe_pad_factor = _moe_weights_and_padding(
        scope_context,
        profile,
        moe_tokens,
        # One weight pass serves every row tile, so experts are counted per step.
        tiled=tiled or scope == "single_layer",
        grouped=grouped_moe,
    )
    if scope == "per_op":
        # A per-op kernel may run each expert on all rows (masked) or on its own
        # rows (grouped). The cheaper padding of the two applies.
        _, masked_phi = _moe_weights_and_padding(
            scope_context, profile, tile_tokens, tiled=False, grouped=False
        )
        moe_pad_factor = min(moe_pad_factor, masked_phi)
    extra_act = (
        2.0
        * max(1, profile.residual_streams)
        * scope_context.layers_local
        * scope_context.token_residual_bytes
        if tiled
        else 0.0
    )
    return w_step, moe_pad_factor, extra_act, tile_tokens, grouped_moe


def _compute_scope_collective_times(
    scope: FusionScope,
    scope_context: _ScopeContext,
    n_splits: int,
) -> tuple[float, float, int]:
    """Compute sync stage time, comm time, and xla ar factor in seconds."""
    num_ar = scope_context.num_ar
    lat_coll = scope_context.lat_coll
    l_moe = scope_context.l_moe
    ep_dp = scope_context.ep_dp
    pp_stage_time = scope_context.pp_stage_time
    a2a_time = scope_context.a2a_time
    bw_hbm = scope_context.bw_hbm

    xla_ar_factor = (
        PER_OP_COLLECTIVE_CALL_FACTOR
        if scope == "per_op" and scope_context.eff_tp > 1
        else 1
    )
    xla_hbm_coll_bytes = (
        2.0 * num_ar * scope_context.token_residual_bytes
        if scope == "per_op" and scope_context.eff_tp > 1
        else 0.0
    )
    is_fused_multi = scope in ("whole_step", "layer_group")
    overlapped_comm = is_fused_multi and n_splits == 1
    ar_and_pcp_s = (
        num_ar
        * (
            scope_context.ar_transfer_time_s
            + n_splits * xla_ar_factor * scope_context.ar_base_lat_s
        )
        + xla_hbm_coll_bytes / bw_hbm
        + scope_context.pcp_ring_time
    )
    a2a_sync_s = (
        l_moe * (a2a_time + 2 * n_splits * xla_ar_factor * lat_coll)
        if l_moe > 0 and ep_dp > 1
        else 0.0
    )
    dcn_sync_s = num_ar * n_splits * xla_ar_factor * scope_context.ar_inter_latency_s
    sync_stage_s = (
        (dcn_sync_s + pp_stage_time + a2a_sync_s)
        if overlapped_comm
        else (ar_and_pcp_s + pp_stage_time + a2a_sync_s)
        if is_fused_multi
        else 0.0
    )
    comm_s = (
        (ar_and_pcp_s - dcn_sync_s)
        if overlapped_comm
        else 0.0
        if is_fused_multi
        else (ar_and_pcp_s + pp_stage_time + a2a_sync_s)
    )
    return sync_stage_s, comm_s, xla_ar_factor


def _routed_chain(
    t_dense: float,
    routed_hbm_s: float,
    t_moe: float,
    t_vec_moe: float = 0.0,
) -> float:
    """Return dense compute time plus the routed-expert stage that follows it.

    A layer requests its routed-expert weights only after its router runs, so
    that stream overlaps the MoE matmuls but not the dense compute before them.
    """
    return t_dense + max(routed_hbm_s, t_moe, t_vec_moe)


def _core_time(hbm_s: float, chain_s: float, comm_s: float, *, fused: bool) -> float:
    """Return core step time for the HBM stream, causal chain, and collectives.

    A fused kernel overlaps the whole HBM stream with the chain, but each
    all-reduce waits for its layer's result, so collectives extend the chain.
    Separate kernels run their collectives after the streamed compute.
    """
    return max(hbm_s, chain_s + comm_s) if fused else max(hbm_s, chain_s) + comm_s


def _sub_byte_unpack_ops(profile: ModelProfile, routed_bytes_streamed: float) -> float:
    """Return vector ops that unpack routed expert weights stored below one byte."""
    expert_dtype = profile.expert_dtype.lower()
    if expert_dtype not in SUB_BYTE_EXPERT_DTYPES:
        return 0.0
    expert_bytes = DTYPE_BYTES.get(expert_dtype, 1.0)
    return 2.0 * max(0.0, routed_bytes_streamed / max(1e-6, expert_bytes))


def _compute_scope_matmuls(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    *args: object,
    **kwargs: object,
) -> tuple[float, float, float, float, float]:
    """Compute baseline/upper matmul times, vector op counts, and rate."""
    if args and isinstance(args[0], tuple):
        moe_pad_factor, routed_bytes_streamed, bf16_attention_upper = args[0]
    elif len(args) > 1:
        moe_pad_factor = float(str(args[0]))
        routed_bytes_streamed = float(str(args[1]))
        bf16_attention_upper = bool(kwargs.get("bf16_attention_upper", False))
    else:
        moe_pad_factor = 1.0
        routed_bytes_streamed = 0.0
        bf16_attention_upper = False
    chip_spec = scope_context.chip_spec
    vec_rate = chip_spec.vector_f32_ops_per_s or (chip_spec.bf16_flops_per_s / 128.0)
    eff_tp = max(1, scope_context.eff_tp)
    tokens = scope_context.active_tokens

    ffn_dense_inter_local = max(1, ceil(profile.intermediate_size / eff_tp))
    shared_inter_local = (
        ceil(profile.shared_expert_intermediate_size / eff_tp)
        if profile.shared_expert_intermediate_size > 0
        else 0
    )
    subchannel_vec_dense = (
        (
            SUBCHANNEL_VPU_OPS_PER_BLOCK
            * tokens
            * scope_context.base_dense_params
            / profile.weight_block_k
        )
        if 0 < profile.weight_block_k < SUBCHANNEL_BLOCK_K_THRESHOLD
        else 0.0
    )
    vec_ops_dense = (
        (
            8.0
            * max(1, profile.residual_streams)
            * tokens
            * (profile.hidden_size / eff_tp)
            * scope_context.layers_local
        )
        + (
            4.0
            * tokens
            * ffn_dense_inter_local
            * (scope_context.layers_local - scope_context.l_moe)
            + 4.0 * tokens * shared_inter_local * scope_context.l_moe
        )
        + subchannel_vec_dense
    )

    ep_tp = gcd(workload.ep, scope_context.eff_tp)
    i_exp_local = (
        ceil(profile.moe_intermediate_size / max(1, scope_context.eff_tp // ep_tp))
        * (scope_context.l_moe > 0)
        if profile.num_experts > 0
        else 0
    )
    moe_tokens_local = (
        tokens
        * scope_context.ep_dp
        * scope_context.experts_local
        / max(1, profile.num_experts)
    )
    subchannel_vec_moe = (
        ((moe_pad_factor * scope_context.flops_moe) / profile.weight_block_k)
        if 0 < profile.weight_block_k < SUBCHANNEL_BLOCK_K_THRESHOLD
        else 0.0
    )
    vec_ops_moe = (
        4.0 * moe_tokens_local * scope_context.l_moe * profile.top_k * i_exp_local
        + _sub_byte_unpack_ops(profile, routed_bytes_streamed)
        + subchannel_vec_moe
    )

    attn_is_bf16 = (
        profile.act_dtype.lower() != "fp8" or profile.kv_dtype.lower() != "fp8"
    )
    t_matmul_dense = (
        (scope_context.flops_dense - scope_context.flops_kv) / scope_context.peak_w
        + scope_context.flops_kv / scope_context.bf16_peak
        if attn_is_bf16 and scope_context.flops_kv > 0
        else scope_context.flops_dense / scope_context.peak_w
    )
    t_matmul_dense_upper = (
        (scope_context.flops_dense - scope_context.flops_kv) / scope_context.peak_w
        + scope_context.flops_kv / scope_context.bf16_peak
        if (
            attn_is_bf16
            or (bf16_attention_upper and workload.phase in ("prefill", "mixed"))
        )
        and scope_context.flops_kv > 0
        else t_matmul_dense
    )
    t_dense = max(t_matmul_dense, vec_ops_dense / vec_rate)
    t_dense_upper = max(t_matmul_dense_upper, vec_ops_dense / vec_rate)
    t_moe = (moe_pad_factor * scope_context.flops_moe) / scope_context.peak_e
    t_matmul = t_dense + t_moe
    t_matmul_upper = t_dense_upper + t_moe
    return t_matmul, t_matmul_upper, vec_ops_dense, vec_ops_moe, vec_rate


def _compute_achievable_upper_us(
    scope_context: _ScopeContext,
    workload: WorkloadSpec,
    flags: tuple[bool, bool, bool, int],
    times: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    vector_terms: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> float:
    """Compute achievable upper-bound step latency in microseconds."""
    overlapped_comm, bf16_attention_upper, _, n_ops = flags
    (
        moe_pad_factor,
        eta_moe,
        _,
        t_hbm_ach,
        comm_s,
        ar_extra_s,
        a2a_extra_s,
        sync_comm_extra_s,
        pipeline_fill_drain_s,
        repass_spill_us,
        step_ach_us,
        step_upper_us,
        routed_hbm_s,
    ) = times
    vec_ops_dense, vec_ops_moe, vec_rate = vector_terms
    eff = workload.efficiency
    t_matmul_dense_upper_ach = (
        (
            (
                (scope_context.flops_dense - scope_context.flops_kv)
                / scope_context.peak_w
            )
            / max(1e-3, eff.eta_dense)
            + (scope_context.flops_kv / scope_context.bf16_peak)
            / max(1e-3, eff.eta_dense)
        )
        if (
            scope_context.attn_is_bf16
            or (bf16_attention_upper and workload.phase in ("prefill", "mixed"))
        )
        and scope_context.flops_kv > 0
        else (scope_context.flops_dense / scope_context.peak_w)
        / max(1e-3, eff.eta_dense)
    )
    t_dense_upper_ach = max(
        t_matmul_dense_upper_ach,
        (vec_ops_dense / vec_rate) / max(1e-3, eff.eta_dense),
    )
    t_comm_upper_ach = (comm_s + ar_extra_s) / max(1e-3, eff.eta_ici)
    t_moe_ach = (
        (moe_pad_factor * scope_context.flops_moe) / scope_context.peak_e
    ) / max(1e-3, eta_moe)
    t_vec_moe_ach = (vec_ops_moe / vec_rate) / max(1e-3, eff.eta_dense)
    chain_upper_ach_s = _routed_chain(
        t_dense_upper_ach,
        routed_hbm_s / max(1e-3, eff.eta_hbm),
        t_moe_ach,
        t_vec_moe_ach,
    )
    ach_upper_core_s = (
        _core_time(
            t_hbm_ach, chain_upper_ach_s, t_comm_upper_ach, fused=overlapped_comm
        )
        + sync_comm_extra_s
        + a2a_extra_s / max(1e-3, eff.eta_ici)
    )
    return max(
        step_ach_us,
        step_upper_us,
        (ach_upper_core_s + pipeline_fill_drain_s) * 1e6
        + repass_spill_us
        + n_ops * scope_context.launch_gap_upper_us,
    )


def _compute_strict_step_floor(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    flags: tuple[bool, bool, bool, int, int],
    scalars: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """Compute strict balanced-routing weight bytes and step latency floor."""
    one_weight_pass, is_fused_multi, overlapped_comm, n_splits, n_ops = flags
    comm_s, pipeline_fill_drain_s, act_bytes, eff_state_bytes, base_ai, ar_saving = (
        scalars
    )
    w_step_strict = _strict_moe_weights(scope_context, profile)
    t_matmul_dense_strict = (
        (scope_context.flops_dense - scope_context.flops_kv) / scope_context.peak_w
        + scope_context.flops_kv / scope_context.bf16_peak
        if scope_context.attn_is_bf16 and scope_context.flops_kv > 0
        else scope_context.flops_dense / scope_context.peak_w
    )
    t_matmul_strict = (
        t_matmul_dense_strict + scope_context.flops_moe / scope_context.peak_e
    )
    weight_passes = 1 if one_weight_pass else n_splits
    hbm_int_strict = weight_passes * w_step_strict + eff_state_bytes
    hbm_time_strict_s = (hbm_int_strict + act_bytes) / scope_context.bw_hbm
    t_moe_strict = scope_context.flops_moe / scope_context.peak_e
    routed_strict_s = (
        weight_passes * (w_step_strict - scope_context.w_static) / scope_context.bw_hbm
    )
    _, _, vec_ops_dense, vec_ops_moe_strict, vec_rate = _compute_scope_matmuls(
        scope_context,
        profile,
        workload,
        (1.0, weight_passes * (w_step_strict - scope_context.w_static), False),
    )
    # Vector counts are per element. A 32-bit word holds up to
    # VECTOR_MAX_ELEMENTS_PER_WORD packed elements, and
    # VECTOR_SUB_BYTE_ELEMENTS_PER_WORD when it unpacks weights below one byte, so
    # these rates bound every dtype.
    strict_vec_rate = VECTOR_MAX_ELEMENTS_PER_WORD * vec_rate
    unpack_ops_strict = _sub_byte_unpack_ops(
        profile, weight_passes * (w_step_strict - scope_context.w_static)
    )
    t_vec_moe_strict = (
        vec_ops_moe_strict - unpack_ops_strict
    ) / strict_vec_rate + unpack_ops_strict / (
        VECTOR_SUB_BYTE_ELEMENTS_PER_WORD * vec_rate
    )
    t_dense_strict = max(
        t_matmul_strict - t_moe_strict, vec_ops_dense / strict_vec_rate
    )
    chain_strict_s = _routed_chain(
        t_dense_strict, routed_strict_s, t_moe_strict, t_vec_moe_strict
    )
    # The strict floor prices all-reduce transfer at the egress sum, the limit
    # for any algorithm. That transfer sits in comm_s, or in the sync stage of
    # the fill/drain term when a fused scope runs several passes.
    comm_in_core = overlapped_comm or not is_fused_multi
    comm_strict_s = comm_s - (ar_saving if comm_in_core else 0.0)
    fill_drain_strict_s = pipeline_fill_drain_s - (0.0 if comm_in_core else ar_saving)
    roof_core_strict_s = _core_time(
        hbm_time_strict_s, chain_strict_s, comm_strict_s, fused=is_fused_multi
    )
    repass_spill_strict_us = (
        (act_bytes + (n_splits - 1) * w_step_strict) / scope_context.bw_hbm * 1e6
        if base_ai >= scope_context.peak_w / scope_context.bw_hbm and weight_passes > 1
        else 0.0
    )
    step_floor_strict_us = (
        (roof_core_strict_s + fill_drain_strict_s) * 1e6
        + repass_spill_strict_us
        + n_ops * workload.op_overhead_us
    )
    return w_step_strict, step_floor_strict_us


def _compute_ach_upper_and_strict(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    flags_and_vectors: tuple[
        tuple[bool, bool, bool, bool, int, int], tuple[float, float, float]
    ],
    times: tuple[float, ...],
) -> tuple[float, float, float]:
    """Return step_ach_upper_us, w_step_strict, and step_floor_strict_us."""
    flags, vector_terms = flags_and_vectors
    (
        one_weight_pass,
        is_fused_multi,
        overlapped_comm,
        bf16_attention_upper,
        n_splits,
        n_ops,
    ) = flags
    (
        moe_pad_factor,
        eta_moe,
        t_comp_ach,
        t_hbm_ach,
        comm_s,
        ar_extra_s,
        a2a_extra_s,
        sync_comm_extra_s,
        pipeline_fill_drain_s,
        repass_spill_us,
        step_ach_us,
        step_upper_us,
        act_bytes,
        eff_state_bytes,
        base_ai,
        routed_hbm_s,
        ar_strict_saving_s,
    ) = times
    step_ach_upper_us = _compute_achievable_upper_us(
        scope_context,
        workload,
        (overlapped_comm, bf16_attention_upper, one_weight_pass, n_ops),
        (
            moe_pad_factor,
            eta_moe,
            t_comp_ach,
            t_hbm_ach,
            comm_s,
            ar_extra_s,
            a2a_extra_s,
            sync_comm_extra_s,
            pipeline_fill_drain_s,
            repass_spill_us,
            step_ach_us,
            step_upper_us,
            routed_hbm_s,
        ),
        vector_terms,
    )
    w_step_strict, step_floor_strict_us = _compute_strict_step_floor(
        scope_context,
        profile,
        workload,
        (one_weight_pass, is_fused_multi, overlapped_comm, n_splits, n_ops),
        (
            comm_s,
            pipeline_fill_drain_s,
            act_bytes,
            eff_state_bytes,
            base_ai,
            ar_strict_saving_s,
        ),
    )
    return step_ach_upper_us, w_step_strict, step_floor_strict_us


def _upper_collective_extras(
    scope_context: _ScopeContext, n_splits: int, xla_ar_factor: float
) -> tuple[float, float]:
    """Return extra all-reduce and all-to-all seconds at the upper latency.

    The upper bound prices every collective hop at the top of the latency
    bracket; these are the seconds that adds over the floor's latency.
    """
    extra_latency_s = max(0.0, scope_context.lat_coll_upper - scope_context.lat_coll)
    ar_extra_s = (
        scope_context.num_ar * n_splits * xla_ar_factor * scope_context.ar_intra_hops
    ) * extra_latency_s
    a2a_extra_s = (
        (2 * scope_context.l_moe * n_splits * xla_ar_factor)
        if scope_context.l_moe > 0 and scope_context.ep_dp > 1
        else 0
    ) * extra_latency_s
    return ar_extra_s, a2a_extra_s


def _compute_scope_achievable_step(
    scope_context: _ScopeContext,
    workload: WorkloadSpec,
    scope_flags: tuple[bool, bool, int, float],
    timing_inputs: tuple[float, float, float, float, float, float, float, float],
    vector_terms: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float, float, float]:
    """Compute achievable and upper-band step components for one FusionScope."""
    is_fused_multi, grouped_moe, n_ops, _ = scope_flags
    (
        hbm_time_s,
        routed_hbm_s,
        t_moe,
        comm_s,
        sync_stage_s,
        pipeline_fill_drain_s,
        repass_spill_us,
        t_matmul_upper,
    ) = timing_inputs
    vec_ops_dense, vec_ops_moe, vec_rate = vector_terms
    eff = workload.efficiency
    eta_moe = (
        eff.eta_moe_grouped
        if grouped_moe and scope_context.l_moe > 0
        else eff.eta_dense
    )
    t_matmul_dense_ach = (
        (
            (
                (scope_context.flops_dense - scope_context.flops_kv)
                / scope_context.peak_w
            )
            / max(1e-3, eff.eta_dense)
            + (scope_context.flops_kv / scope_context.bf16_peak)
            / max(1e-3, eff.eta_dense)
        )
        if scope_context.attn_is_bf16 and scope_context.flops_kv > 0
        else (scope_context.flops_dense / scope_context.peak_w)
        / max(1e-3, eff.eta_dense)
    )
    t_dense_ach = max(
        t_matmul_dense_ach,
        (vec_ops_dense / vec_rate) / max(1e-3, eff.eta_dense),
    )
    t_hbm_ach = hbm_time_s / max(1e-3, eff.eta_hbm)
    t_comm_ach = comm_s / max(1e-3, eff.eta_ici)
    sync_comm_extra_s = (
        sync_stage_s * (1.0 / max(1e-3, eff.eta_ici) - 1.0) if is_fused_multi else 0.0
    )
    t_moe_ach = t_moe / max(1e-3, eta_moe)
    t_vec_moe_ach = (vec_ops_moe / vec_rate) / max(1e-3, eff.eta_dense)
    chain_ach_s = _routed_chain(
        t_dense_ach,
        routed_hbm_s / max(1e-3, eff.eta_hbm),
        t_moe_ach,
        t_vec_moe_ach,
    )
    step_ach_us = (
        (
            _core_time(t_hbm_ach, chain_ach_s, t_comm_ach, fused=is_fused_multi)
            + sync_comm_extra_s
            + pipeline_fill_drain_s
        )
        * 1e6
        + repass_spill_us
        + n_ops * workload.op_overhead_us
    )
    t_dense_upper = t_matmul_upper - t_moe
    chain_upper_s = _routed_chain(
        t_dense_upper,
        routed_hbm_s,
        t_moe,
        vec_ops_moe / vec_rate,
    )
    return (
        eta_moe,
        t_dense_ach + t_moe_ach,
        t_hbm_ach,
        sync_comm_extra_s,
        step_ach_us,
        chain_upper_s,
    )


def _evaluate_single_scope(
    scope: FusionScope,
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    scope_eval_cfg: tuple[float, bool],
) -> tuple[ScopeEstimate, float, float]:
    """Evaluate one FusionScope against VMEM, HBM, matmul, and collective bounds."""
    sub_pure_us, bf16_attention_upper = scope_eval_cfg
    layers_local = scope_context.layers_local
    bw_hbm = scope_context.bw_hbm
    eff_state_bytes = (
        scope_context.kv_cache_read_bytes + scope_context.recurrent_state_rw_bytes
    )
    (
        static_vmem_bytes,
        act_bytes,
        n_ops,
        t_max,
        n_splits,
        feasible,
        reason,
        selected_dense_slots,
        selected_expert_slots,
    ) = _scope_vmem_and_ops(scope, scope_context, profile, workload)
    w_step, moe_pad_factor, extra_act, _, grouped_moe = _resolve_scope_moe_and_acts(
        scope, scope_context, profile, workload, n_splits
    )
    tiled = scope in ("whole_step", "layer_group") and n_splits > 1
    # A fused scope above its row limit and a single-layer kernel stream weights
    # once: their row tiles move the residual through HBM at every layer instead.
    one_weight_pass = tiled or scope == "single_layer"
    weight_passes = 1 if one_weight_pass else n_splits
    # That per-layer residual traffic already holds the call-boundary activations.
    act_bytes = max(act_bytes, extra_act)
    is_fused_multi = scope in ("whole_step", "layer_group")
    overlapped_comm = is_fused_multi and n_splits == 1
    sync_stage_s, comm_s, xla_ar_factor = _compute_scope_collective_times(
        scope, scope_context, n_splits
    )
    pipeline_fill_drain_s = (n_ops - 1) * (
        scope_context.first_slot_bytes / bw_hbm
        + BOUNDARY_SYNC_HOP_FACTOR * scope_context.lat_coll
    ) + sync_stage_s
    streams = max(1, profile.residual_streams)
    if tiled:
        # Each tile boundary adds a sync hop per layer. The residual bytes moved
        # there are already in act_bytes.
        pipeline_fill_drain_s += (
            (n_splits - 1)
            * 2
            * layers_local
            * BOUNDARY_SYNC_HOP_FACTOR
            * scope_context.lat_coll
        )
    (
        t_matmul,
        t_matmul_upper,
        vec_ops_dense,
        vec_ops_moe,
        vec_rate,
    ) = _compute_scope_matmuls(
        scope_context,
        profile,
        workload,
        (
            moe_pad_factor,
            weight_passes * (w_step - scope_context.w_static),
            bf16_attention_upper,
        ),
    )
    act_saved = (
        4.0 * streams * layers_local * scope_context.token_residual_bytes - act_bytes
    )
    hbm_int = weight_passes * w_step + eff_state_bytes
    hbm_time_s = (hbm_int + act_bytes) / bw_hbm
    d_ops = 2 * layers_local - n_ops
    # Routed-expert bytes are requested after each layer's router, so they
    # stream under neither the dense compute nor the collectives.
    routed_hbm_s = weight_passes * (w_step - scope_context.w_static) / bw_hbm
    t_moe = (moe_pad_factor * scope_context.flops_moe) / scope_context.peak_e
    t_dense = t_matmul - t_moe
    t_vec_moe = vec_ops_moe / vec_rate
    chain_s = _routed_chain(t_dense, routed_hbm_s, t_moe, t_vec_moe)
    roof_us = (
        _core_time(hbm_time_s, chain_s, comm_s, fused=is_fused_multi)
        + pipeline_fill_drain_s
    ) * 1e6
    base_ai = (scope_context.flops_dense + scope_context.flops_moe) / max(
        1.0, w_step + eff_state_bytes
    )
    repass_spill_us = (
        (act_bytes + (n_splits - 1) * w_step) / bw_hbm * 1e6
        if base_ai >= scope_context.peak_w / bw_hbm and weight_passes > 1
        else 0.0
    )
    pure_us = roof_us + repass_spill_us
    new_sub_pure_us = pure_us if scope == "per_op" else sub_pure_us
    step_us = pure_us + n_ops * workload.op_overhead_us
    (
        eta_moe,
        t_comp_ach,
        t_hbm_ach,
        sync_comm_extra_s,
        step_ach_us,
        chain_upper_s,
    ) = _compute_scope_achievable_step(
        scope_context,
        workload,
        (is_fused_multi, grouped_moe, n_ops, moe_pad_factor),
        (
            hbm_time_s,
            routed_hbm_s,
            t_moe,
            comm_s,
            sync_stage_s,
            pipeline_fill_drain_s,
            repass_spill_us,
            t_matmul_upper,
        ),
        (vec_ops_dense, vec_ops_moe, vec_rate),
    )
    ar_extra_s, a2a_extra_s = _upper_collective_extras(
        scope_context, n_splits, xla_ar_factor
    )
    roof_upper_us = (
        _core_time(
            hbm_time_s, chain_upper_s, comm_s + ar_extra_s, fused=overlapped_comm
        )
        + a2a_extra_s
        + pipeline_fill_drain_s
    ) * 1e6
    step_upper_us = max(
        step_us,
        roof_upper_us + repass_spill_us + n_ops * scope_context.launch_gap_upper_us,
    )
    step_ach_upper_us, w_step_strict, step_floor_strict_us = (
        _compute_ach_upper_and_strict(
            scope_context,
            profile,
            workload,
            (
                (
                    one_weight_pass,
                    is_fused_multi,
                    overlapped_comm,
                    bf16_attention_upper,
                    n_splits,
                    n_ops,
                ),
                (vec_ops_dense, vec_ops_moe, vec_rate),
            ),
            (
                moe_pad_factor,
                eta_moe,
                t_comp_ach,
                t_hbm_ach,
                comm_s,
                ar_extra_s,
                a2a_extra_s,
                sync_comm_extra_s,
                pipeline_fill_drain_s,
                repass_spill_us,
                step_ach_us,
                step_upper_us,
                act_bytes,
                eff_state_bytes,
                base_ai,
                routed_hbm_s,
                scope_context.num_ar
                * (
                    scope_context.ar_transfer_time_s
                    - scope_context.ar_transfer_strict_s
                ),
            ),
        )
    )
    support_status, blockers = evaluate_support(
        scope, profile, workload, scope_context.chip_spec, scope_context
    )
    d_bytes = bw_hbm * max(0, d_ops) * workload.op_overhead_us * 1e-6
    break_even = (
        round(max(0.0, (pure_us - new_sub_pure_us) / d_ops), 3) if d_ops > 0 else 0.0
    )
    w_crit = (
        round(((act_saved + d_bytes) / (n_splits - 1)) / 1e9, 4)
        if n_splits > 1
        else None
    )
    est = ScopeEstimate(
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
        moe_flop_factor=round(moe_pad_factor, 3),
        roof_floor_us=round(roof_us, 2),
        step_bound_us=round(step_us, 2),
        selected_dense_slots=selected_dense_slots,
        selected_expert_slots=selected_expert_slots,
        step_achievable_us=round(step_ach_us, 2),
        break_even_op_overhead_us=break_even,
        step_upper_us=round(step_upper_us, 2),
        compute_us=round((t_dense + max(t_moe, t_vec_moe)) * 1e6, 2),
        hbm_us=round(hbm_time_s * 1e6, 2),
        comm_us=round(comm_s * 1e6, 2),
        chain_us=round(chain_s * 1e6, 2),
        fill_drain_us=round(pipeline_fill_drain_s * 1e6, 2),
        repass_spill_us=round(repass_spill_us, 2),
        launch_us=round(n_ops * workload.op_overhead_us, 2),
        step_achievable_upper_us=round(step_ach_upper_us, 2),
        support=support_status,
        blockers=blockers,
        w_step_strict_gb=round(w_step_strict / 1e9, 5),
        step_floor_strict_us=round(step_floor_strict_us, 2),
    )
    return est, step_us, new_sub_pure_us


def _microbatch_requests(scope_context: _ScopeContext) -> tuple[int, int, int]:
    """Return requests per full call, the full call count, and leftover requests."""
    tokens_per_request = scope_context.active_tokens // scope_context.batch_local
    per_call = scope_context.single_call_rows // tokens_per_request
    return (
        per_call,
        scope_context.batch_local // per_call,
        scope_context.batch_local % per_call,
    )


def _sum_call_estimates(
    full: ScopeEstimate, full_calls: int, rest: ScopeEstimate | None
) -> ScopeEstimate:
    """Return one estimate for `full_calls` full calls plus an optional last call.

    Time, byte, and op fields add up. VMEM and ring fields describe one call, so
    they come from the full call, the largest one. The caller sets the support
    fields for the whole step.
    """
    calls = [full] * full_calls + ([rest] if rest is not None else [])
    return replace(
        full,
        feasible=all(c.feasible for c in calls),
        reason=next((c.reason for c in calls if not c.feasible), full.reason),
        ops_per_step=sum(c.ops_per_step for c in calls),
        n_splits=sum(c.n_splits for c in calls),
        w_step_gb=round(sum(c.w_step_gb for c in calls), 4),
        w_crit_gb=None,
        act_hbm_mb=round(sum(c.act_hbm_mb for c in calls), 3),
        roof_floor_us=round(sum(c.roof_floor_us for c in calls), 2),
        step_bound_us=round(sum(c.step_bound_us for c in calls), 2),
        step_achievable_us=round(sum(c.step_achievable_us for c in calls), 2),
        step_upper_us=round(sum(c.step_upper_us for c in calls), 2),
        compute_us=round(sum(c.compute_us for c in calls), 2),
        hbm_us=round(sum(c.hbm_us for c in calls), 2),
        comm_us=round(sum(c.comm_us for c in calls), 2),
        chain_us=round(sum(c.chain_us for c in calls), 2),
        fill_drain_us=round(sum(c.fill_drain_us for c in calls), 2),
        repass_spill_us=round(sum(c.repass_spill_us for c in calls), 2),
        launch_us=round(sum(c.launch_us for c in calls), 2),
        step_achievable_upper_us=round(
            sum(c.step_achievable_upper_us for c in calls), 2
        ),
        w_step_strict_gb=round(sum(c.w_step_strict_gb for c in calls), 5),
        step_floor_strict_us=round(sum(c.step_floor_strict_us for c in calls), 2),
    )


def _evaluate_whole_step_microbatched(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    scope_eval_cfg: tuple[float, bool],
) -> tuple[ScopeEstimate, float, float]:
    """Price whole_step above its single-call row limit as separate full calls.

    The kernel serves a larger batch as microbatches, and every call streams
    its weights again. Requests stay whole: full calls run first, then the
    remainder. Each call is an independent single-pass evaluation whose context
    keeps the parent's latency priors, which callers override per corner.
    """
    per_call, full_calls, rest_requests = _microbatch_requests(scope_context)
    parts: list[tuple[ScopeEstimate, float, float]] = []
    for requests in (per_call, rest_requests):
        if requests == 0:
            continue
        sub_wl = replace(workload, batch=requests * workload.dp)
        sub_ctx = replace(
            _build_scope_context(
                profile, sub_wl, scope_context.chip_spec, scope_context.plan_error
            ),
            lat_coll=scope_context.lat_coll,
            lat_coll_upper=scope_context.lat_coll_upper,
            ar_base_lat_s=scope_context.ar_base_lat_s,
            ar_inter_latency_s=scope_context.ar_inter_latency_s,
            launch_gap_upper_us=scope_context.launch_gap_upper_us,
        )
        parts.append(
            _evaluate_single_scope(
                "whole_step", sub_ctx, profile, sub_wl, scope_eval_cfg
            )
        )
    full_est, full_step_us, sub_pure_us = parts[0]
    rest_est, rest_step_us = (parts[1][0], parts[1][1]) if parts[1:] else (None, 0.0)
    est = _sum_call_estimates(full_est, full_calls, rest_est)
    d_ops = 2 * scope_context.layers_local - est.ops_per_step
    pure_us = est.roof_floor_us + est.repass_spill_us
    break_even = (
        round(max(0.0, (pure_us - sub_pure_us) / d_ops), 3) if d_ops > 0 else 0.0
    )
    # Support belongs to the whole step: its total rows are checked against the
    # rows a declaration serves through split calls.
    support_status, blockers = evaluate_support(
        "whole_step", profile, workload, scope_context.chip_spec, scope_context
    )
    return (
        replace(
            est,
            break_even_op_overhead_us=break_even,
            support=support_status,
            blockers=blockers,
        ),
        full_calls * full_step_us + rest_step_us,
        sub_pure_us,
    )


def _evaluate_scopes(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    *,
    bf16_attention_upper: bool = True,
) -> tuple[list[ScopeEstimate], dict[FusionScope, float]]:
    """Evaluate every FusionScope against VMEM, HBM, and matmul bounds."""
    scopes_order: tuple[FusionScope, ...] = (
        "per_op",
        "single_layer",
        "layer_group",
        "whole_step",
    )
    estimates: list[ScopeEstimate] = []
    raw_step_map: dict[FusionScope, float] = {}
    sub_pure_us = 0.0
    for scope in scopes_order:
        cfg = (sub_pure_us, bf16_attention_upper)
        est, step_us, sub_pure_us = (
            _evaluate_whole_step_microbatched(scope_context, profile, workload, cfg)
            if scope == "whole_step"
            and 0 < scope_context.single_call_rows < scope_context.active_tokens
            else _evaluate_single_scope(scope, scope_context, profile, workload, cfg)
        )
        estimates.append(est)
        raw_step_map[scope] = step_us
    return estimates, raw_step_map


@dataclass(frozen=True)
class _ScopeContext:
    """Per-core weight, state, FLOP, collective, and VMEM terms for one workload."""

    vmem_budget: int
    single_layer_dense_staging_bytes: int
    single_layer_expert_staging_bytes: int
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
    single_call_rows: int
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
    ar_transfer_strict_s: float
    ar_base_lat_s: float
    egress_bw: float
    lat_coll: float
    lat_coll_upper: float
    launch_gap_upper_us: float
    pp_stage_time: float
    pcp_ring_time: float
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
    attn_is_bf16: bool
    plan_error: str
    chip_spec: ChipSpec


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


def _first_slot_bytes(workload: WorkloadSpec, layer_dense_weight_bytes: int) -> float:
    """Return first weight DMA slot size in bytes for pipeline fill/drain."""
    tiles = workload.tiles
    if tiles is not None:
        dense_slot_bytes = tiles.bank
    else:
        rp = workload.ring_plan or default_double_buffer_ring_plan(workload.dma_buffers)
        dense_slot_bytes = rp.dense_slot_bytes
    return float(
        min(
            dense_slot_bytes,
            max(
                MIN_FIRST_DMA_SLOT_BYTES,
                layer_dense_weight_bytes // FIRST_SLOT_LAYER_WEIGHT_DIVISOR,
            ),
        )
    )


@dataclass(frozen=True)
class _LayerGeometry:
    """Request and layer geometry terms for layer weight accumulation."""

    eff_tp: int
    kv_split: int
    l_attn: int
    l_moe: int
    l_lin: int
    layers_local: int
    active_tokens: int
    chunk_tokens: int
    batch_local: int
    eff_ctx: int


@dataclass(frozen=True)
class _LayerWeightsAndFlops:
    """Layer static weights, expert bytes, state bytes, and FLOP terms."""

    w_static: float
    exp_bytes: float
    kv_cache_read_bytes: float
    recurrent_state_rw_bytes: float
    layer_dense_weight_bytes: int
    flops_kv: float
    flops_dense: float
    flops_moe: float
    base_dense_params: float
    nh_local: int
    q_dim: int
    experts_local: int
    ep_dp: int
    kv_bytes_per_tok: float
    gqa_scratch: int
    gdn_scratch: int
    ffn_scratch: int


def _compute_attention_params(
    profile: ModelProfile, geom: _LayerGeometry
) -> tuple[int, int, int, float]:
    """Compute local head counts, Q projection dim, and attention parameter count."""
    head_dim = profile.head_dim
    kv_lora = profile.kv_lora_rank
    qk_nope = profile.qk_nope_head_dim or head_dim
    qk_rope = profile.qk_rope_head_dim
    nh_local = max(1, ceil(profile.num_attention_heads / geom.eff_tp))
    kvh_local = max(
        1, ceil(profile.num_kv_heads / max(1, geom.eff_tp // geom.kv_split))
    )
    q_dim = nh_local * head_dim
    q_tot = nh_local * (qk_nope + qk_rope)
    mla_proj_params = (
        profile.q_lora_rank * (profile.hidden_size + q_tot)
        if profile.q_lora_rank
        else profile.hidden_size * q_tot
    ) + profile.hidden_size * (kv_lora + qk_rope + q_dim)
    attn_gate_params = (
        profile.hidden_size * q_dim
        if profile.has_attn_output_gate and not kv_lora
        else 0
    )
    attn_params = (
        mla_proj_params + kv_lora * nh_local * (qk_nope + head_dim)
        if kv_lora
        else profile.hidden_size * 2 * (q_dim + kvh_local * head_dim) + attn_gate_params
    )
    return nh_local, kvh_local, q_dim, attn_params


def _compute_flops_and_scratches(
    profile: ModelProfile,
    workload: WorkloadSpec,
    geom: _LayerGeometry,
    counts: tuple[int, int, int, int, int, float],
) -> tuple[float, float, float, int, int, int]:
    """Compute attention/dense/MoE FLOPs and per-op VMEM scratchpad sizes."""
    nh_local, q_dim, vh_local, i_exp_local, routed_expert_params, base_dense_params = (
        counts
    )
    l_attn, l_moe, l_lin = geom.l_attn, geom.l_moe, geom.l_lin
    active_tokens, chunk_tokens, eff_ctx = (
        geom.active_tokens,
        geom.chunk_tokens,
        geom.eff_ctx,
    )
    ep_dp = workload.ep // gcd(workload.ep, geom.eff_tp)
    experts_local = (profile.num_experts // workload.ep) * (l_moe > 0)
    kv_lora = profile.kv_lora_rank
    qk_rope = profile.qk_rope_head_dim
    head_dim = profile.head_dim
    qk_nope = profile.qk_nope_head_dim or head_dim
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
    ) * profile.kv_compress_factor
    flops_kv = (
        2.0
        * l_attn
        * attn_ctx
        * (nh_local * d_mla if kv_lora else 2.0 * active_tokens * q_dim)
    )
    flops_dense = (
        2.0
        * active_tokens
        * (base_dense_params + l_moe * profile.hidden_size * profile.num_experts)
        + flops_kv
    )
    flops_moe = (
        2.0
        * active_tokens
        * ep_dp
        * l_moe
        * profile.top_k
        * routed_expert_params
        * experts_local
        / profile.num_experts
        if experts_local > 0
        else 0.0
    )
    gqa_scratch = (
        2 * ATTN_TILE_TOKENS * (FP32_BYTES * max(head_dim, kv_lora))
        + 2 * ATTN_TILE_TOKENS * ATTN_TILE_TOKENS * FP32_BYTES
        if l_attn > 0
        else 0
    )
    gdn_scratch = (
        FP32_BYTES
        * min(vh_local, GDN_STREAMED_HEADS_CAP)
        * profile.linear_key_dim
        * profile.linear_value_dim
        if l_lin > 0
        else 0
    )
    ffn_scratch = (
        (2 * ATTN_TILE_TOKENS * i_exp_local * FP32_BYTES)
        if l_moe > 0
        else (2 * ATTN_TILE_TOKENS * DENSE_FFN_TILE_CAP * FP32_BYTES)
    )
    return flops_kv, flops_dense, flops_moe, gqa_scratch, gdn_scratch, ffn_scratch


def _compute_layer_weights_and_flops(
    profile: ModelProfile,
    workload: WorkloadSpec,
    geom: _LayerGeometry,
    dtypes_bw: tuple[float, float, float, float],
) -> _LayerWeightsAndFlops:
    """Compute per-layer static weights, cache state bytes, and FLOP counts."""
    bw_w, bw_e, bw_kv, bw_st = dtypes_bw
    hidden = profile.hidden_size
    nh_local, kvh_local, q_dim, attn_params = _compute_attention_params(profile, geom)
    kh_local = (
        max(1, ceil(profile.linear_key_heads / geom.eff_tp)) if geom.l_lin > 0 else 0
    )
    vh_local = (
        max(1, ceil(profile.linear_value_heads / geom.eff_tp)) if geom.l_lin > 0 else 0
    )
    kd, vd = profile.linear_key_dim, profile.linear_value_dim
    z_dim = vh_local * vd
    qkv_lin = 2 * kh_local * kd + vh_local * vd
    ep_tp = gcd(workload.ep, geom.eff_tp)
    linear_attn_params = hidden * (qkv_lin + 2 * z_dim) * (geom.l_lin > 0)
    experts_local = (profile.num_experts // workload.ep) * (geom.l_moe > 0)
    dense_ffn_params = (
        3 * hidden * max(1, ceil(profile.intermediate_size / geom.eff_tp))
    )
    shared_expert_params = (
        3 * hidden * ceil(profile.shared_expert_intermediate_size / geom.eff_tp)
    )
    ep_dp = workload.ep // ep_tp
    i_exp_local = ceil(profile.moe_intermediate_size / (geom.eff_tp // ep_tp)) * (
        geom.l_moe > 0
    )
    routed_expert_params = (
        3 * (profile.routed_expert_hidden_size or hidden) * i_exp_local
    )
    base_dense_params = (
        geom.l_attn * attn_params
        + geom.l_lin * linear_attn_params
        + (geom.layers_local - geom.l_moe) * dense_ffn_params
        + geom.l_moe * shared_expert_params
    )
    w_static = (
        base_dense_params * bw_w + geom.l_moe * hidden * profile.num_experts * 2.0
    )
    exp_bytes = routed_expert_params * bw_e
    kv_bytes_per_tok = (
        (
            (profile.kv_lora_rank + profile.qk_rope_head_dim)
            if profile.kv_lora_rank
            else 2.0 * kvh_local * profile.head_dim
        )
        * bw_kv
        * profile.kv_compress_factor
        / geom.kv_split
    )
    kv_cache_read_bytes = (
        geom.l_attn * geom.batch_local * geom.eff_ctx * kv_bytes_per_tok
    )
    recurrent_state_rw_bytes = (
        2.0
        * geom.l_lin
        * geom.batch_local
        * (z_dim * kd + qkv_lin * profile.linear_conv_kernel)
        * bw_st
    )
    layer_dense_weight_bytes = int(
        (
            max(attn_params, linear_attn_params)
            + (shared_expert_params if geom.l_moe > 0 else dense_ffn_params)
        )
        * bw_w
    )
    flops_kv, flops_dense, flops_moe, gqa_scratch, gdn_scratch, ffn_scratch = (
        _compute_flops_and_scratches(
            profile,
            workload,
            geom,
            (
                nh_local,
                q_dim,
                vh_local,
                i_exp_local,
                routed_expert_params,
                base_dense_params,
            ),
        )
    )
    return _LayerWeightsAndFlops(
        w_static=w_static,
        exp_bytes=exp_bytes,
        kv_cache_read_bytes=kv_cache_read_bytes,
        recurrent_state_rw_bytes=recurrent_state_rw_bytes,
        layer_dense_weight_bytes=layer_dense_weight_bytes,
        flops_kv=flops_kv,
        flops_dense=flops_dense,
        flops_moe=flops_moe,
        base_dense_params=base_dense_params,
        nh_local=nh_local,
        q_dim=q_dim,
        experts_local=experts_local,
        ep_dp=ep_dp,
        kv_bytes_per_tok=kv_bytes_per_tok,
        gqa_scratch=gqa_scratch,
        gdn_scratch=gdn_scratch,
        ffn_scratch=ffn_scratch,
    )


def _extract_slot_banks(workload: WorkloadSpec) -> tuple[int, int]:
    """Return dense and expert weight slot byte sizes for pipeline buffers."""
    tiles = workload.tiles
    if tiles is not None:
        return tiles.bank, tiles.ebank
    rp = workload.ring_plan
    if rp is not None:
        return rp.dense_slot_bytes, rp.expert_slot_bytes
    return DENSE_SLOT_DEFAULT_BYTES, EXPERT_SLOT_DEFAULT_BYTES


def _compute_ar_hops_and_latency(eff_tp: int) -> tuple[int, float]:
    """Return single-slice ICI all-reduce hops and zero inter-host latency."""
    if eff_tp <= 1:
        return 0, 0.0
    return round(math.log2(eff_tp)), 0.0


def _whole_request_row_limit(
    profile: ModelProfile, workload: WorkloadSpec, eff_tp: int
) -> int:
    """Return the single-call row limit in whole requests (0: none declared).

    A call holds whole requests, so the limit is floored to a multiple of the
    rows one request contributes. A limit below one request is a config error,
    whether the workload or a deployment preset declares it; flooring it to 0
    would read as no limit at all.
    """
    tokens_per_request = _request_rows(workload)
    row_limit = single_call_row_limit(profile.name, workload, eff_tp)
    if 0 < row_limit < tokens_per_request:
        msg = (
            f"single-call row limit ({row_limit}) is below one verify request "
            f"({tokens_per_request} rows)"
        )
        raise errors.MegakernelConfigError(msg)
    return (row_limit // tokens_per_request) * tokens_per_request


def _resolve_workload_token_geometry(
    profile: ModelProfile, workload: WorkloadSpec
) -> tuple[_LayerGeometry, _LayerWeightsAndFlops, float, int, int]:
    """Resolve sharded layer counts, active tokens, weights, and FLOPs."""
    bw_w = DTYPE_BYTES.get(profile.weight_dtype.lower(), 1.0)
    bw_e = DTYPE_BYTES.get(profile.expert_dtype.lower(), 1.0)
    bw_kv = DTYPE_BYTES.get(profile.kv_dtype.lower(), 1.0)
    bw_st = DTYPE_BYTES.get(profile.state_dtype.lower(), 2.0)
    eff_tp = max(workload.tp, workload.dcp) if workload.tp == 1 else workload.tp
    kv_split = max(workload.kv_split, workload.dcp)
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1
    uncached_isl = max(1, round(workload.isl * (1.0 - workload.prefix_hit_rate)))
    batch_local = ceil(workload.batch / workload.dp)
    chunk_tokens = ceil((workload.chunk_size or uncached_isl) / eff_pcp)
    layers_local = ceil(profile.num_layers / workload.pp)
    eff_ctx = workload.context or (
        workload.isl + (0 if workload.phase == "prefill" else workload.osl // 2)
    )
    spec_tok = workload.spec_tokens
    verify_tokens = (
        max(1, spec_tok) + 1
        if workload.phase == "verify"
        else (spec_tok + 1 if spec_tok > 0 else 1)
    )
    prefill_tokens = (
        ceil(workload.chunk_size / eff_pcp)
        if workload.chunk_size
        else ceil(
            batch_local * uncached_isl / max(1, (workload.microbatches or 1) * eff_pcp)
        )
    )
    decode_tokens = batch_local * (1 if workload.phase == "decode" else verify_tokens)
    active_tokens = (
        prefill_tokens
        if workload.phase == "prefill"
        else decode_tokens + chunk_tokens * (workload.phase == "mixed")
    )
    l_attn = min(
        layers_local,
        ceil(profile.num_attention_layers * layers_local / profile.num_layers),
    )
    l_moe = min(
        layers_local, ceil(profile.num_moe_layers * layers_local / profile.num_layers)
    )
    l_lin = min(
        layers_local - l_attn,
        ceil(profile.num_linear_layers * layers_local / profile.num_layers),
    )
    token_residual_bytes = active_tokens * profile.hidden_size * 2.0
    geom = _LayerGeometry(
        eff_tp=eff_tp,
        kv_split=kv_split,
        l_attn=l_attn,
        l_moe=l_moe,
        l_lin=l_lin,
        layers_local=layers_local,
        active_tokens=active_tokens,
        chunk_tokens=chunk_tokens,
        batch_local=batch_local,
        eff_ctx=eff_ctx,
    )
    layer_weights_flops = _compute_layer_weights_and_flops(
        profile, workload, geom, (bw_w, bw_e, bw_kv, bw_st)
    )
    return geom, layer_weights_flops, token_residual_bytes, eff_pcp, verify_tokens


def _build_scope_context(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    plan_error: str = "",
    meas: MeasurementsData | None = None,
) -> _ScopeContext:
    """Assemble per-core weight, state, FLOP, and VMEM scope context."""
    geom, layer_weights_flops, token_residual_bytes, eff_pcp, verify_tokens = (
        _resolve_workload_token_geometry(profile, workload)
    )
    layers_local = geom.layers_local
    vmem_budget = _vmem_budget_bytes(workload, chip_spec)
    num_segments = (
        ceil(layers_local / workload.segment_layers)
        if 0 < workload.segment_layers < layers_local
        else 1
    )
    buf_slots = workload.dma_buffers
    first_slot_bytes = _first_slot_bytes(
        workload, layer_weights_flops.layer_dense_weight_bytes
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
            geom.eff_tp,
            chip_spec,
            dcn_bw_bytes_per_s=dcn_bytes_per_s,
        )
    )
    ar_intra_hops, ar_inter_latency_s = _compute_ar_hops_and_latency(geom.eff_tp)
    dense_bank, expert_bank = _extract_slot_banks(workload)
    pcp_ring_time = (
        (eff_pcp - 1)
        * geom.l_attn
        * geom.chunk_tokens
        * (layer_weights_flops.kv_bytes_per_tok * geom.kv_split)
        / chip_spec.ici_link_bw_bytes_per_s
        if eff_pcp > 1 and geom.l_attn > 0
        else 0.0
    )
    return _ScopeContext(
        vmem_budget=vmem_budget,
        single_layer_dense_staging_bytes=min(
            buf_slots * dense_bank,
            buf_slots * layer_weights_flops.layer_dense_weight_bytes,
        ),
        single_layer_expert_staging_bytes=min(
            buf_slots * expert_bank, int(buf_slots * layer_weights_flops.exp_bytes)
        ),
        first_slot_bytes=first_slot_bytes,
        gqa_scratch=layer_weights_flops.gqa_scratch,
        gdn_scratch=layer_weights_flops.gdn_scratch,
        ffn_scratch=layer_weights_flops.ffn_scratch,
        layers_local=layers_local,
        num_segments=num_segments,
        num_layer_groups=(
            ceil(len(workload.layer_groups) * layers_local / profile.num_layers)
            if workload.layer_groups
            else 0
        ),
        l_moe=geom.l_moe,
        l_attn=geom.l_attn,
        l_lin=geom.l_lin,
        active_tokens=geom.active_tokens,
        single_call_rows=_whole_request_row_limit(profile, workload, geom.eff_tp),
        eff_ctx=geom.eff_ctx,
        batch_local=geom.batch_local,
        verify_tokens=verify_tokens,
        nh_local=layer_weights_flops.nh_local,
        q_dim=layer_weights_flops.q_dim,
        base_dense_params=layer_weights_flops.base_dense_params,
        per_token_fused_bytes=compute_per_token_vmem_bytes(profile, workload)[
            "total_per_token_bytes"
        ],
        token_residual_bytes=token_residual_bytes,
        w_static=layer_weights_flops.w_static,
        exp_bytes=layer_weights_flops.exp_bytes,
        experts_local=layer_weights_flops.experts_local,
        ep_dp=layer_weights_flops.ep_dp,
        eff_tp=geom.eff_tp,
        row_floor=chip_spec.lhs_row_floor(
            profile.expert_dtype if geom.l_moe > 0 else profile.weight_dtype
        ),
        num_ar=profile.ar_slots_per_layer * layers_local,
        ar_hops=ar_hops,
        ar_intra_hops=ar_intra_hops,
        ar_inter_latency_s=ar_inter_latency_s,
        dcn_bytes_per_s=dcn_bytes_per_s,
        ar_vol=ar_vol,
        ar_transfer_time_s=ar_transfer_time_s,
        ar_transfer_strict_s=ar_vol / chip_spec.ici_egress_bw_bytes_per_s,
        ar_base_lat_s=ar_base_lat_s,
        egress_bw=chip_spec.ici_egress_bw_bytes_per_s,
        lat_coll=chip_spec.collective_latency_s,
        lat_coll_upper=chip_spec.collective_latency_us_bracket[1] * 1e-6,
        launch_gap_upper_us=max(workload.op_overhead_us, DEFAULT_LAUNCH_GAP_UPPER_US),
        pp_stage_time=(
            (
                token_residual_bytes / chip_spec.ici_link_bw_bytes_per_s
                + chip_spec.collective_latency_s
            )
            if workload.pp > 1
            else 0.0
        ),
        pcp_ring_time=pcp_ring_time,
        a2a_time=(
            2.0
            * (layer_weights_flops.ep_dp - 1)
            / layer_weights_flops.ep_dp
            * min(max(1, profile.top_k), layer_weights_flops.ep_dp)
            * token_residual_bytes
            / chip_spec.ici_egress_bw_bytes_per_s
            if layer_weights_flops.ep_dp > 1
            else 0.0
        ),
        flops_dense=layer_weights_flops.flops_dense,
        flops_kv=layer_weights_flops.flops_kv,
        flops_moe=layer_weights_flops.flops_moe,
        peak_w=chip_spec.peak_flops(profile.weight_dtype, profile.act_dtype),
        peak_e=chip_spec.peak_flops(profile.expert_dtype, profile.act_dtype),
        bf16_peak=chip_spec.bf16_flops_per_s,
        attn_is_bf16=(
            profile.act_dtype.lower() != "fp8" or profile.kv_dtype.lower() != "fp8"
        ),
        kv_cache_read_bytes=layer_weights_flops.kv_cache_read_bytes,
        recurrent_state_rw_bytes=layer_weights_flops.recurrent_state_rw_bytes,
        bw_hbm=chip_spec.hbm_bw_bytes_per_s,
        hbm_used=(
            layer_weights_flops.w_static
            + geom.l_moe
            * layer_weights_flops.experts_local
            * layer_weights_flops.exp_bytes
            + layer_weights_flops.kv_cache_read_bytes
            + 0.5 * layer_weights_flops.recurrent_state_rw_bytes
        ),
        no_whole_step=(
            workload.pp > 1
            or num_segments > 1
            or len(workload.layer_groups) > 1
            or profile.num_layers == 1
            or bool(plan_error)
        ),
        plan_error=plan_error,
        chip_spec=chip_spec,
    )


def _is_calibrated_trace(meas: MeasurementsData | None) -> bool:
    """Return True if meas represents a calibrated empirical trace."""
    return meas is not None and meas.source_kind != "architectural_prior"


def _resolve_launch_gap_prior(
    workload: WorkloadSpec,
    meas: MeasurementsData | None,
) -> tuple[float, float, float, ProvenanceTag, str]:
    """Resolve launch gap bounds and provenance tag across dispatch regimes."""
    if workload.dispatch_regime == "zero_floor":
        return (
            0.0,
            0.0,
            0.0,
            "DERIVED",
            "zero_floor compiler-proof physics regime (0.0 us/boundary)",
        )
    if workload.dispatch_regime == "unfused_eager":
        lg_val = (
            workload.op_overhead_us
            if workload.op_overhead_us > 0.0
            else UNFUSED_EAGER_GAP_US
        )
        return (
            lg_val,
            lg_val,
            max(lg_val, DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "unfused_eager exposed dispatch sensitivity regime",
        )
    if meas is not None and meas.launch_gap_p50_us is not None:
        lg_lo = meas.launch_gap_p50_us
        lg_hi = max(lg_lo, meas.launch_gap_p90_us or lg_lo)
        is_cal = _is_calibrated_trace(meas)
        tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        n_sfx = f" (n={meas.launch_gap_n})" if is_cal else ""
        return (lg_lo, lg_lo, lg_hi, tag, f"{meas.source_desc}{n_sfx}")
    if workload.dispatch_regime == "compiled_xla":
        return (
            COMPILED_XLA_GAP_P50_US,
            COMPILED_XLA_GAP_P50_US,
            COMPILED_XLA_GAP_P90_US,
            "ASSUMED",
            (
                "compiled-XLA sublayer boundary prior "
                "(uncalibrated; run advisor calibrate)"
            ),
        )
    if workload.op_overhead_us > 0.0:
        lg_val = workload.op_overhead_us
        return (
            lg_val,
            lg_val,
            max(lg_val, DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "caller WorkloadSpec.op_overhead_us override",
        )
    return (
        DEFAULT_LAUNCH_GAP_VAL_US,
        DEFAULT_LAUNCH_GAP_FLOOR_US,
        DEFAULT_LAUNCH_GAP_UPPER_US,
        "ASSUMED",
        "uncalibrated TPU host/device launch gap prior [1.0, 8.0] us",
    )


def _resolve_calibrated_priors(
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    scope_context: _ScopeContext,
    meas: MeasurementsData | None,
) -> tuple[
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, list[float], ProvenanceTag, str],
]:
    """Return calibrated or assumed priors for launch gap, collectives, and coverage."""
    lg_info = _resolve_launch_gap_prior(workload, meas)

    if meas is not None and meas.collective_hop_us is not None:
        c_val = round(meas.collective_hop_us, 3)
        is_cal = _is_calibrated_trace(meas)
        c_tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        n_sfx = f" (n={meas.collective_n})" if is_cal else ""
        coll_info = (
            c_val,
            c_val,
            c_val,
            c_tag,
            f"{meas.source_desc}{n_sfx}",
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
        is_cal = _is_calibrated_trace(meas)
        cov_tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        cov_info = (
            cov_val,
            [cov_val, cov_val],
            cov_tag,
            f"{meas.source_desc} step_coverage",
        )
    elif workload.phase == "mixed":
        dec_tok = scope_context.batch_local * max(1, scope_context.verify_tokens)
        cov_val = round(min(1.0, dec_tok / max(1, scope_context.active_tokens)), 4)
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
