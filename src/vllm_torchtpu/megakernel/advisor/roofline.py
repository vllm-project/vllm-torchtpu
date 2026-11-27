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
"""First-principles physics roofline, scope assessment, and hardware bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from math import ceil, gcd
from typing import TYPE_CHECKING

from vllm_torchtpu.megakernel.core import errors

from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    AR_BYTE_MATCH_MAX_RATIO,
    AR_BYTE_MATCH_MIN_RATIO,
    ATTN_TILE_TOKENS,
    BF16_BYTES,
    BOUNDARY_SYNC_HOP_FACTOR,
    COLLECTIVE_OP_ALL_REDUCE,
    COLLECTIVE_PATH_IN_KERNEL,
    COLLECTIVE_PATH_XLA,
    COMPILED_XLA_GAP_P50_US,
    COMPILED_XLA_GAP_P90_US,
    CORES_PER_HOST,
    DCP_COLLECTIVES_PER_ATTN_LAYER,
    DCP_LSE_MERGE_VEC_OPS,
    DEFAULT_LAUNCH_GAP_UPPER_US,
    DEFAULT_LAUNCH_GAP_VAL_US,
    DENSE_FFN_TILE_CAP,
    DSA_OUTPUT_RESCALE_VEC_OPS,
    DSA_SOFTMAX_VEC_OPS,
    DSA_UNPACK_MASK_VEC_OPS,
    DTYPE_BYTES,
    FIRST_SLOT_LAYER_WEIGHT_DIVISOR,
    FP8_DTYPES,
    FP32_BYTES,
    GDN_STREAMED_HEADS_CAP,
    GROUPED_MOE_MODES,
    HBM_CACHELINE_BYTES,
    HC_SUBLAYERS_PER_LAYER,
    INDEXER_HEAD_REDUCE_VEC_OPS,
    INDEXER_RADIX_BUCKETS,
    INDEXER_RADIX_PASS_VEC_OPS,
    INDEXER_RADIX_PASSES,
    INDEXER_SCALE_VEC_OPS,
    INT4_INDEX_SCALE_OVERHEAD_BYTES,
    INT4_SCALE_PAGE_BYTES_PER_TOK,
    LSE_SCALARS_PER_HEAD,
    MEGACORE_CORES,
    MHC_DYNAMIC_PROJ_TERMS,
    MIN_FIRST_DMA_SLOT_BYTES,
    MULTI_LAYER_FUSED_SCOPES,
    PER_OP_COLLECTIVE_CALL_FACTOR,
    PER_OP_DSA_KV_ROUNDTRIP_FACTOR,
    PER_OP_INDEXER_SCORE_ROUNDTRIP_BYTES,
    RECURRENT_STATE_RW_PASSES,
    RMSNORM_RESIDUAL_VEC_OPS_PER_ELEM,
    SINKHORN_VEC_OPS_PER_ENTRY_ITER,
    SUB_BYTE_EXPERT_DTYPES,
    SUB_BYTE_UNPACK_VEC_OPS_PER_ELEM,
    SUBCHANNEL_BLOCK_K_THRESHOLD,
    SUBCHANNEL_VPU_OPS_PER_BLOCK,
    SUBLAYER_BOUNDARIES_PER_LAYER,
    SWIGLU_ACT_VEC_OPS_PER_ELEM,
    TURBOQUANT_DEFAULT_PAGE_SIZE,
    TURBOQUANT_HEADER_BYTES,
    TURBOQUANT_SUBLANE_WORDS,
    TURBOQUANT_SUPPORTED_PAGE_SIZES,
    TURBOQUANT_WORD_BYTES,
    UNFUSED_EAGER_GAP_US,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
    canonical_kv_dtype,
)
from .models import (
    ModelProfile,
    has_dsa_profile,
    hyper_connection_total_params,
    indexer_record_bytes,
    resolve_model_profile,
)
from .specs import (
    FusionScope,
    MeasurementsData,
    ScopeEstimate,
    SparseAttnSpec,
    SparseCacheOwner,
    WorkloadSpec,
    default_double_buffer_ring_plan,
)
from .vmem import (
    CoreLegalityContext,
    _scope_vmem_and_ops,
    _ScopeVmemBudget,
    active_fusion_break_layers,
    collect_validation_reasons,
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
    dsa_query_gather_dim,
    effective_host_table_hbm_resident,
    effective_pcp,
    effective_tp,
    evaluate_core_legality,
    fit_vmem_budget_bytes,
    fusion_break_layer_groups,
    validate_advise_inputs,
    wide_dcp_kv_heads_per_rank,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .specs import CollectiveCall


@dataclass(frozen=True)
class _CollectiveSummary:
    """Computed collective sync, comm, hop, and calibrated AR totals for a band."""

    sync_stage_s: float
    comm_s: float
    xla_ar_factor: int
    measured_ar_total_s: float
    has_measured_ar: bool
    eff_num_ar: int
    step_hops: int
    has_dcp_collectives: bool
    ar_strict_saving_s: float


@dataclass(frozen=True)
class _TimingPointResult:
    """Full timing breakdown for one roofline point prior to display rounding."""

    step_us: float
    roof_us: float
    hbm_s: float
    chain_s: float
    comm_s: float
    fill_drain_s: float
    launch_us: float
    compute_s: float
    routed_hbm_s: float
    mxu_s: float
    vec_s: float
    dsa_hbm_s: float


@dataclass(frozen=True)
class _RawStepMetrics:
    """Unrounded metrics accumulated without rounding drift across microbatches."""

    step_weight_bytes: float
    act_bytes: float
    roof_floor_us: float
    step_bound_us: float
    step_achievable_us: float
    step_upper_us: float
    compute_us: float
    hbm_us: float
    comm_us: float
    chain_us: float
    fill_drain_us: float
    launch_us: float
    step_achievable_upper_us: float
    step_weight_strict_bytes: float
    step_floor_strict_us: float


@dataclass(frozen=True)
class _ScopeEvalConfig:
    """Scope-evaluation control flags and baseline reference floor."""

    sub_pure_us: float = 0.0
    bf16_attention_upper: bool = True
    upper_boundary_fill: bool = True
    upper_per_op_coll_factor: bool = True
    measurements: MeasurementsData | None = None


@dataclass(frozen=True)
class _LayerParamParts:
    """Sharded per-layer head, parameter, and expert counts for a model slice."""

    nh_local: int
    kvh_local: int
    q_dim: int
    vh_local: int
    kh_local: int
    i_exp_local: int
    attn_params: float
    linear_attn_params: float
    dense_ffn_params: float
    shared_expert_params: float
    routed_expert_params: float
    latent_proj_params: float
    indexer_query_proj_params: float
    indexer_key_weight_proj_params: float
    experts_local: int
    ep_dp: int


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
    local_moe_layers: int
    local_attn_layers: int
    local_linear_layers: int
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
    measured_ar_xla_s: float | None = None
    measured_ar_in_kernel_s: float | None = None
    local_indexer_layers: int = 0
    has_dsa: bool = False
    is_sparse_gather: bool = False
    indexer_ctx_local: float = 0.0
    dsa_kv_gather_bytes: float = 0.0
    dsa_kv_strict_bytes: float = 0.0
    vec_ops_pre_dsa_extra: float = 0.0
    vec_ops_dsa: float = 0.0
    vec_ops_indexer_radix: float = 0.0
    dcp_query_all_gather_transfer_s: float = 0.0
    dcp_query_all_gather_hops: int = 0
    dcp_post_attn_transfer_s: float = 0.0
    dcp_post_attn_hops: int = 0
    dcp_hops: int = 0
    has_fusion_break: bool = False
    host_h2d_s: float = 0.0


@dataclass(frozen=True)
class _EvaluatedScopes:
    """Evaluated scope estimates, unrounded step times, and prepared context."""

    estimates: tuple[ScopeEstimate, ...]
    raw_step_map: dict[FusionScope, float]
    scope_context: _ScopeContext
    effective_workload: WorkloadSpec
    eval_workload: WorkloadSpec
    launch_gap_us: float
    launch_gap_upper_us: float
    measured_ar_xla_s: float | None
    measured_ar_in_kernel_s: float | None
    step_coverage: float | None
    collective_latency_s: float


def _align_up(value: float, unit: int) -> float:
    """Round non-negative `value` up to a multiple of positive integer `unit`."""
    return float(ceil(value / unit) * unit)


def _core_time(hbm_s: float, chain_s: float, comm_s: float, *, fused: bool) -> float:
    """Return core step time across HBM, causal compute chain, and collectives."""
    return max(hbm_s, chain_s + comm_s) if fused else max(hbm_s, chain_s) + comm_s


def _dense_attn_matmul_s(
    ctx: _ScopeContext,
    eta_dense: float,
    *,
    bf16_attn: bool,
    kv_only: bool = False,
) -> float:
    """Return dense or attention matmul duration in seconds at target efficiency."""
    use_bf16 = bf16_attn and ctx.flops_kv > 0
    kv_peak = ctx.bf16_peak if use_bf16 else ctx.peak_w
    if kv_only:
        return (ctx.flops_kv / kv_peak / eta_dense) if ctx.flops_kv > 0 else 0.0
    if use_bf16:
        return (
            (ctx.flops_dense - ctx.flops_kv) / ctx.peak_w + ctx.flops_kv / kv_peak
        ) / eta_dense
    return (ctx.flops_dense / ctx.peak_w) / eta_dense


def _is_calibrated_trace(meas: MeasurementsData | None) -> bool:
    """Return True when `meas` represents a calibrated empirical trace."""
    return meas is not None and meas.source_kind != "architectural_prior"


def _ar_call_matches(
    call: CollectiveCall,
    path: str,
    effective_tp_ranks: int,
    expected_bytes: float,
) -> bool:
    """Return True when a calibrated AR call matches op, path, ranks, and bytes."""
    if (
        call.op != COLLECTIVE_OP_ALL_REDUCE
        or call.path != path
        or call.group_size != effective_tp_ranks
    ):
        return False
    if expected_bytes <= 0.0:
        return call.bytes is None or call.bytes <= 0
    if call.bytes is None or call.bytes <= 0:
        return False
    min_match_bytes = AR_BYTE_MATCH_MIN_RATIO * expected_bytes
    max_match_bytes = AR_BYTE_MATCH_MAX_RATIO * expected_bytes
    return min_match_bytes <= call.bytes <= max_match_bytes


def _resolve_measured_all_reduce_s(
    meas: MeasurementsData | None,
    effective_tp_ranks: int,
    path: str,
    expected_bytes: float,
) -> float | None:
    """Resolve measured all-reduce call duration in seconds matching shape and bytes."""
    if meas is None or not _is_calibrated_trace(meas) or effective_tp_ranks <= 1:
        return None
    matched = [
        call
        for call in meas.collective_calls
        if _ar_call_matches(call, path, effective_tp_ranks, expected_bytes)
    ]
    if not matched:
        return None
    values = [
        call.mean_latency_us if call.mean_latency_us is not None else call.latency_us
        for call in matched
    ]
    total_n = sum(call.n for call in matched)
    combined_us = (
        sum(val * call.n for val, call in zip(values, matched, strict=True)) / total_n
        if total_n > 0
        else sum(values) / len(values)
    )
    return combined_us * 1e-6


def _experts_hit(
    tokens_ep: int, top_k: int, num_experts: int, experts_local: int
) -> float:
    """Return expected active experts on one rank under uniform routing."""
    miss = (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    return max(1.0, experts_local * (1.0 - miss))


def experts_hit_busiest(
    tokens_ep: int,
    top_k: int,
    num_experts: int,
    experts_local: int,
    owner_groups: int,
) -> float:
    """Return expected per-rank active experts on the busiest owner group.

    Models per-expert selection probability ``hit = 1 - (1 - min(1, top_k /
    num_experts)) ** tokens_ep``, treats the ``owner_groups`` disjoint expert
    groups of size ``E_g = num_experts // owner_groups`` as independent
    ``Binomial(E_g, hit)`` draws so ``E[max] = sum_{m=0}^{E_g - 1} (1 -
    F(m) ** owner_groups)`` in ``lgamma`` log space, and scales to per-rank
    expert equivalents by ``experts_local / E_g`` (floored at ``1.0``).

    Returns:
        Expected active experts on the busiest expert owner group.

    Raises:
        ValueError: When expert or owner group counts are non-positive or ordered wrong.
    """
    if num_experts <= 0 or experts_local <= 0 or owner_groups > num_experts:
        msg = (
            "experts_hit_busiest requires num_experts > 0, experts_local > 0, "
            f"and owner_groups <= num_experts; got num_experts={num_experts}, "
            f"experts_local={experts_local}, owner_groups={owner_groups}"
        )
        raise ValueError(msg)
    if owner_groups <= 1:
        return _experts_hit(tokens_ep, top_k, num_experts, experts_local)
    experts_per_group = num_experts // owner_groups
    hit = 1.0 - (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    if hit <= 0.0:
        return 1.0
    if hit >= 1.0:
        return max(1.0, float(experts_local))
    log_fact_total = math.lgamma(experts_per_group + 1)
    log_hit = math.log(hit)
    log_miss = math.log1p(-hit)
    cdf = 0.0
    expected_group_max = 0.0
    for count in range(experts_per_group):
        log_pmf = (
            log_fact_total
            - math.lgamma(count + 1)
            - math.lgamma(experts_per_group - count + 1)
            + count * log_hit
            + (experts_per_group - count) * log_miss
        )
        cdf = min(1.0, cdf + math.exp(log_pmf))
        expected_group_max += 1.0 - cdf**owner_groups
    return max(1.0, expected_group_max * (experts_local / experts_per_group))


def can_group_moe_in_scope(
    scope: FusionScope,
    effective_tp_ranks: int,
    ep_dp: int,
    moe_mode: str,
) -> bool:
    """Return whether `scope` can group routed expert tokens on chip or in HBM.

    Returns:
        True when tokens can be grouped within the chip or host topology.
    """
    return (
        scope not in MULTI_LAYER_FUSED_SCOPES
        or ep_dp == 1
        or (
            effective_tp_ranks * ep_dp <= CORES_PER_HOST
            and moe_mode in GROUPED_MOE_MODES
        )
    )


def compute_break_even_host_head_delta_us(
    scope_device_us: float,
    per_op_device_us: float,
) -> float:
    """Return host-dispatch delta (us) needed for a fused scope to break even.

    Returns:
        Non-negative latency saving threshold in microseconds.
    """
    return round(max(0.0, scope_device_us - per_op_device_us), 2)


def validate_workload(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str,
    workload: WorkloadSpec,
) -> tuple[str, ...]:
    """Validate scalar budgets, sharding divisibility, and serving HBM pool fit.

    Returns:
        Tuple of violation tags; empty when the workload fits hardware and HBM.
    """
    resolved_profile = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    reasons, has_hard = collect_validation_reasons(
        resolved_profile, workload, chip_spec
    )
    if has_hard:
        return tuple(reasons)
    scope_context = _build_scope_context(resolved_profile, workload, chip_spec)
    if (
        scope_context.hbm_used
        > workload.hbm_utilization_cap * chip_spec.hbm_capacity_bytes
    ):
        reasons.append("hbm_pool_overflow")
    return tuple(reasons)


def turboquant_slot_bytes(
    head_dim: int, kv_page_size: int = TURBOQUANT_DEFAULT_PAGE_SIZE
) -> float:
    """Return packed 4-bit TurboQuant KV bytes per token per KV head."""
    if kv_page_size not in TURBOQUANT_SUPPORTED_PAGE_SIZES:
        msg = (
            "turboquant_4bit_nc requires kv_page_size in "
            f"{TURBOQUANT_SUPPORTED_PAGE_SIZES}, got {kv_page_size}"
        )
        raise errors.MegakernelConfigError(msg)
    page_sz = max(1, kv_page_size)
    slot_size = head_dim + TURBOQUANT_HEADER_BYTES
    raw_words = ceil(page_sz * slot_size / TURBOQUANT_WORD_BYTES)
    aligned_words = (
        (raw_words + TURBOQUANT_SUBLANE_WORDS - 1) // TURBOQUANT_SUBLANE_WORDS
    ) * TURBOQUANT_SUBLANE_WORDS
    return float(aligned_words * TURBOQUANT_WORD_BYTES) / float(page_sz)


def compute_hadamard_kv_bf16_flops(
    kv_dtype: str,
    head_dims: tuple[int, int, int],
    active_tokens: int,
    l_attn: int,
) -> float:
    """Return MXU BF16 Hadamard rotation FLOPs for 4-bit KV cache dtypes."""
    nh_local, kvh_local, head_dim = head_dims
    dt = canonical_kv_dtype(kv_dtype)
    if dt == "int4_per_token_head":
        heads_rot = 2 * nh_local + 2 * kvh_local
    elif dt == "turboquant_4bit_nc":
        heads_rot = nh_local + kvh_local
    else:
        return 0.0
    return float(2 * head_dim * head_dim * heads_rot * active_tokens * l_attn)


def compute_hyper_connection_terms(
    profile: ModelProfile,
    layers_local: int,
    eff_tp: int,
    active_tokens: int,
) -> tuple[int, float, float]:
    """Return per-rank HC BF16 parameter count, matmul FLOPs, and Sinkhorn ops."""
    hc = profile.hyper_connection
    if hc is None or hc.num_streams <= 1 or layers_local <= 0:
        return 0, 0.0, 0.0
    tp = max(1, eff_tp)
    streams = hc.num_streams
    hidden_local = ceil(profile.hidden_size / tp)
    if hc.kind == "lowrank":
        rank = hc.lowrank_dim
        per_sub = 2 * streams * hidden_local * rank + streams * streams * hidden_local
        params_local = layers_local * HC_SUBLAYERS_PER_LAYER * per_sub
        if hc.has_final_mixer:
            params_local += 2 * streams * hidden_local * rank
        return params_local, 2.0 * float(params_local) * float(active_tokens), 0.0
    per_sub = (streams * streams + MHC_DYNAMIC_PROJ_TERMS * streams) * (
        streams * hidden_local
    )
    params_local = layers_local * HC_SUBLAYERS_PER_LAYER * per_sub
    flops = 2.0 * float(params_local) * float(active_tokens) + 2.0 * float(
        HC_SUBLAYERS_PER_LAYER * layers_local * active_tokens
    ) * float(streams * streams * hidden_local)
    vec_ops = float(
        HC_SUBLAYERS_PER_LAYER
        * layers_local
        * active_tokens
        * streams
        * streams
        * SINKHORN_VEC_OPS_PER_ENTRY_ITER
        * hc.sinkhorn_iters
    )
    return params_local, flops, vec_ops


def compute_attention_params(
    profile: ModelProfile, eff_tp: int, kv_split: int, dcp: int = 1
) -> tuple[int, int, int, float]:
    """Compute local head counts, Q projection dim, and attention parameter count."""
    head_dim = profile.head_dim
    kv_lora = profile.kv_lora_rank
    qk_nope = profile.qk_nope_head_dim or head_dim
    qk_rope = profile.qk_rope_head_dim
    local_attn_heads = max(1, ceil(profile.num_attention_heads / eff_tp))
    local_kv_heads = (
        wide_dcp_kv_heads_per_rank(profile.num_kv_heads, eff_tp, dcp, "tp_group")
        if profile.dcp_span == "tp_group"
        else max(1, ceil(profile.num_kv_heads / max(1, eff_tp // kv_split)))
    )
    local_kv_weight_heads = (
        max(1, ceil(profile.num_kv_heads / eff_tp))
        if profile.dcp_span == "tp_group"
        else local_kv_heads
    )
    q_dim = local_attn_heads * head_dim
    hidden = profile.hidden_size
    attn_gate_params = hidden * q_dim if profile.has_attn_output_gate else 0
    if profile.o_lora_rank > 0 and kv_lora == 0:
        q_in = (
            profile.q_lora_rank * hidden + q_dim * profile.q_lora_rank
            if profile.q_lora_rank > 0
            else q_dim * hidden
        )
        shared_mqa = bool(
            profile.sparse_attn is not None and profile.sparse_attn.shared_kv_mqa
        )
        wkv = (1 if shared_mqa else 2) * local_kv_weight_heads * head_dim * hidden
        wo_a = profile.o_lora_rank * q_dim
        wo_b = hidden * max(1, ceil(profile.o_groups * profile.o_lora_rank / eff_tp))
        return (
            local_attn_heads,
            local_kv_heads,
            q_dim,
            float(q_in + wkv + wo_a + wo_b + attn_gate_params),
        )
    total_q_dim = local_attn_heads * (qk_nope + qk_rope)
    mla_proj_params = (
        profile.q_lora_rank * (hidden + total_q_dim)
        if profile.q_lora_rank
        else hidden * total_q_dim
    ) + hidden * (kv_lora + qk_rope + q_dim)
    attn_params = (
        mla_proj_params
        + kv_lora * local_attn_heads * (qk_nope + head_dim)
        + attn_gate_params
        if kv_lora
        else hidden * 2 * (q_dim + local_kv_weight_heads * head_dim) + attn_gate_params
    )
    return local_attn_heads, local_kv_heads, q_dim, float(attn_params)


def _sparse_indexer_proj_params(
    sp: SparseAttnSpec, profile: ModelProfile, eff_tp: int, local_attn: int
) -> tuple[float, float]:
    """Return `(query_proj_params_per_layer, key_weight_bf16_params_per_layer)`."""
    if sp.index_heads <= 0 or sp.index_dim <= 0 or local_attn <= 0:
        return 0.0, 0.0
    if sp.kind in ("qsa", "msa"):
        proj = float(
            ceil((sp.index_heads + sp.index_kv_heads) * sp.index_dim / eff_tp)
            * profile.hidden_size
        )
        return (0.0, proj) if sp.kind == "qsa" else (proj, 0.0)
    q_params = float(
        ceil(sp.index_heads * sp.index_dim / eff_tp)
        * (profile.q_lora_rank or profile.hidden_size)
    )
    kw_params = float(
        round(
            (len(sp.index_owners) / max(1, profile.num_layers))
            * profile.hidden_size
            * ceil((sp.index_dim + sp.index_heads) / eff_tp)
        )
    )
    return q_params, kw_params


def _compute_layer_parts(
    profile: ModelProfile,
    sharding: tuple[int, int, int, int, int],
) -> _LayerParamParts:
    """Compute sharded layer head counts, FFN/MoE dimensions, and parameter totals."""
    tp_ranks, kv_split, ep, eff_latent_tp, dcp = sharding
    local_attn_heads, local_kv_heads, q_dim, attn_params = compute_attention_params(
        profile, tp_ranks, kv_split, dcp
    )
    hidden = profile.hidden_size
    has_linear = profile.num_linear_layers > 0
    local_key_heads = (
        max(1, ceil(profile.linear_key_heads / tp_ranks)) if has_linear else 0
    )
    local_val_heads = (
        max(1, ceil(profile.linear_value_heads / tp_ranks)) if has_linear else 0
    )
    z_dim = local_val_heads * profile.linear_value_dim
    qkv_proj_dim = 2 * local_key_heads * profile.linear_key_dim + z_dim
    linear_attn_params = float(hidden * (qkv_proj_dim + 2 * z_dim) * has_linear)
    has_moe = profile.num_moe_layers != 0 and profile.num_experts > 0
    experts_local = (profile.num_experts // ep) if has_moe else 0
    ep_tp = gcd(ep, tp_ranks)
    ep_dp = ep // ep_tp
    dense_ffn_params = (
        float(3 * hidden * ceil(profile.intermediate_size / tp_ranks))
        if profile.intermediate_size > 0
        else 0.0
    )
    shared_expert_params = (
        float(3 * hidden * ceil(profile.shared_expert_intermediate_size / tp_ranks))
        if profile.shared_expert_intermediate_size > 0
        else 0.0
    )
    local_moe_inter = (
        ceil(profile.moe_intermediate_size / max(1, tp_ranks // ep_tp))
        if has_moe
        else 0
    )
    routed_hidden = profile.routed_expert_hidden_size or hidden
    routed_expert_params = float(3 * routed_hidden * local_moe_inter)
    latent_proj_params = (
        (2.0 * hidden * profile.routed_expert_hidden_size) / eff_latent_tp
        if 0 < profile.routed_expert_hidden_size < hidden and has_moe
        else 0.0
    )
    idx_active = has_dsa_profile(profile)
    indexer_query_proj_params = (
        float(
            (profile.q_lora_rank or hidden)
            * profile.index_n_heads
            * profile.index_head_dim
        )
        if idx_active
        else 0.0
    )
    indexer_key_weight_proj_params = (
        float(hidden * (profile.index_head_dim + profile.index_n_heads))
        if idx_active
        else 0.0
    )
    return _LayerParamParts(
        nh_local=local_attn_heads,
        kvh_local=local_kv_heads,
        q_dim=q_dim,
        vh_local=local_val_heads,
        kh_local=local_key_heads,
        i_exp_local=local_moe_inter,
        attn_params=attn_params,
        linear_attn_params=linear_attn_params,
        dense_ffn_params=dense_ffn_params,
        shared_expert_params=shared_expert_params,
        routed_expert_params=routed_expert_params,
        latent_proj_params=latent_proj_params,
        indexer_query_proj_params=indexer_query_proj_params,
        indexer_key_weight_proj_params=indexer_key_weight_proj_params,
        experts_local=experts_local,
        ep_dp=ep_dp,
    )


def compute_decoder_param_count(profile: ModelProfile) -> int:
    """Return unsharded decoder-only parameter count using advisor physics math.

    Uses the exact same layer parameter builder as `_ScopeContextBuilder` at
    `tp=1, ep=1, dcp=1` so weight totals stay in sync with HuggingFace totals.

    Returns:
        Total unsharded decoder parameter count.
    """
    local_attn = profile.num_attention_layers
    local_moe = max(0, profile.num_moe_layers)
    local_lin = profile.num_linear_layers
    sparse = profile.sparse_attn
    local_indexer = (
        local_attn
        if (sparse is not None and sparse.index_dim > 0)
        else (
            min(local_attn, profile.num_indexer_layers)
            if has_dsa_profile(profile)
            else 0
        )
    )
    parts = _compute_layer_parts(profile, (1, 1, 1, 1, 1))
    idx_q_params, idx_kw_params = (
        _sparse_indexer_proj_params(sparse, profile, 1, local_attn)
        if sparse is not None
        else (parts.indexer_query_proj_params, parts.indexer_key_weight_proj_params)
    )
    base_dense = (
        local_attn * parts.attn_params
        + local_indexer * idx_q_params
        + local_lin * parts.linear_attn_params
        + (profile.num_layers - local_moe) * parts.dense_ffn_params
        + local_moe * (parts.shared_expert_params + parts.latent_proj_params)
    )
    total = (
        base_dense
        + local_moe * profile.num_experts * parts.routed_expert_params
        + local_moe * profile.hidden_size * profile.num_experts
        + local_indexer * idx_kw_params
        + hyper_connection_total_params(profile)
        + sum(tbl.dense_proj_params for tbl in profile.host_tables)
    )
    return round(total)


def _owner_record_bytes(
    owner: SparseCacheOwner,
    profile: ModelProfile,
    workload: WorkloadSpec,
    kv_cfg: tuple[str, float, int, int, bool],
) -> float:
    """Return per-record bytes on one rank for a KV or index `SparseCacheOwner`."""
    kv_dt, bw_kv, eff_tp, kv_group_heads, is_index = kv_cfg
    rep_mult = (
        float(kv_group_heads)
        if owner.replication == "kv_head_group"
        else (1.0 / max(1, eff_tp) if owner.replication == "tp_sharded" else 1.0)
    )
    if owner.elem_bytes is not None:
        return (
            owner.record_elems * owner.elem_bytes + owner.record_overhead_bytes
        ) * rep_mult
    if is_index:
        if kv_dt == "int4_per_token_head":
            return (
                owner.record_elems * 0.5 + INT4_INDEX_SCALE_OVERHEAD_BYTES
            ) * rep_mult + owner.record_overhead_bytes
        elem_w = 1.0 if kv_dt == "turboquant_4bit_nc" else bw_kv
        return owner.record_elems * elem_w * rep_mult + owner.record_overhead_bytes
    if kv_dt == "turboquant_4bit_nc":
        heads_rec = owner.record_elems / max(1, 2 * profile.head_dim)
        return (
            heads_rec
            * turboquant_slot_bytes(profile.head_dim, workload.kv_page_size)
            * rep_mult
            + owner.record_overhead_bytes
        )
    if kv_dt == "int4_per_token_head":
        return (
            owner.record_elems * 0.5 * rep_mult
            + INT4_SCALE_PAGE_BYTES_PER_TOK
            + owner.record_overhead_bytes
        )
    return owner.record_elems * bw_kv * rep_mult + owner.record_overhead_bytes


def _synthesize_homogeneous_owners(
    sp: SparseAttnSpec, profile: ModelProfile, l_attn: int
) -> tuple[tuple[SparseCacheOwner, ...], tuple[SparseCacheOwner, ...]]:
    """Return `(kv_owners, idx_owners)` for homogeneous QSA/MSA specs."""
    if sp.kv_owners or sp.index_owners:
        return sp.kv_owners, sp.index_owners
    if sp.kind not in ("qsa", "msa"):
        return (), ()
    kv_own = tuple(
        SparseCacheOwner(
            lyr, 1.0, 2 * profile.head_dim, None, 0.0, "kv_head_group", (lyr,)
        )
        for lyr in range(l_attn)
    )
    if sp.index_dim <= 0:
        return kv_own, ()
    ratio = float(max(1, sp.index_compress_ratio)) if sp.kind == "qsa" else 1.0
    elem_b = float(FP32_BYTES) if sp.kind == "qsa" else None
    idx_own = tuple(
        SparseCacheOwner(lyr, ratio, sp.index_dim, elem_b, 0.0, "per_rank", (lyr,))
        for lyr in range(l_attn)
    )
    return kv_own, idx_own


def _ced_token_scale(
    profile: ModelProfile,
    workload: WorkloadSpec,
    layers_local: int,
    active_tokens: int,
    chunk_tokens: int,
) -> float:
    """Return stack-average active token fraction under Chunked Encoder-Decoder."""
    if (
        workload.effective_execution != "ced"
        or profile.ced_boundary <= 0
        or workload.phase not in ("prefill", "mixed")
        or active_tokens <= 0
        or layers_local <= 0
    ):
        return 1.0
    t_prefill_part = active_tokens if workload.phase == "prefill" else chunk_tokens
    t_decode_part = max(0, active_tokens - t_prefill_part)
    win_cap = (
        profile.sparse_attn.window_tokens
        if (profile.sparse_attn is not None and profile.sparse_attn.window_tokens > 0)
        else t_prefill_part
    )
    dec_tokens = t_decode_part + min(t_prefill_part, win_cap)
    enc_layers = min(
        layers_local,
        ceil(profile.ced_boundary * layers_local / max(1, profile.num_layers)),
    )
    dec_layers = max(0, layers_local - enc_layers)
    return float(enc_layers * active_tokens + dec_layers * dec_tokens) / float(
        layers_local * active_tokens
    )


def _host_table_shard_divisor(shard_axis: str, eff_tp: int, dp: int) -> int:
    """Return per-core HBM shard divisor for a host or HBM lookup table."""
    tp = max(1, eff_tp)
    if shard_axis == "tp":
        return tp
    if shard_axis == "dp_local":
        return max(1, min(dp, max(1, CORES_PER_HOST // tp)))
    return max(1, min(tp * dp, CORES_PER_HOST))


def _compute_host_table_context_terms(
    profile: ModelProfile,
    workload: WorkloadSpec,
    rank_dims: tuple[int, int, int],
    pcie_h2d_bw_bytes_per_s: float,
) -> tuple[float, float, float]:
    """Return `(hbm_table_and_state_bytes, hbm_gather_read_bytes, host_h2d_s)`."""
    if not profile.host_tables:
        return 0.0, 0.0, 0.0
    eff_tp, batch_local, active_tokens = rank_dims
    hbm_resident_b = 0.0
    hbm_read_b = 0.0
    offloaded_h2d_b = 0.0
    for tbl in profile.host_tables:
        hbm_resident_b += float(tbl.state_bytes_per_request * batch_local)
        step_lookup_b = float(active_tokens) * tbl.h2d_bytes_per_token
        if effective_host_table_hbm_resident(tbl, workload):
            div = _host_table_shard_divisor(tbl.shard_axis, eff_tp, workload.dp)
            hbm_resident_b += float(tbl.table_bytes_total) / float(div)
            hbm_read_b += step_lookup_b
        else:
            offloaded_h2d_b += step_lookup_b
    return hbm_resident_b, hbm_read_b, offloaded_h2d_b / pcie_h2d_bw_bytes_per_s


def _kv_bytes_per_token(
    profile: ModelProfile,
    workload: WorkloadSpec,
    kvh_local: int,
    kv_bytes: float,
    tok_split: int,
) -> tuple[str, float]:
    """Return normalized `kv_dtype` and per-token KV cache bytes per layer."""
    kv_dt = canonical_kv_dtype(profile.kv_dtype)
    if profile.kv_lora_rank == 0 and kv_dt == "turboquant_4bit_nc":
        per_tok = (
            kvh_local
            * turboquant_slot_bytes(profile.head_dim, workload.kv_page_size)
            * profile.kv_compress_factor
            / tok_split
        )
        return kv_dt, per_tok
    if profile.kv_lora_rank == 0 and kv_dt == "int4_per_token_head":
        per_tok = (
            (2.0 * kvh_local * profile.head_dim * 0.5 + INT4_SCALE_PAGE_BYTES_PER_TOK)
            * profile.kv_compress_factor
            / tok_split
        )
        return kv_dt, per_tok
    raw_elems = (
        (profile.kv_lora_rank + profile.qk_rope_head_dim)
        if profile.kv_lora_rank
        else 2.0 * kvh_local * profile.head_dim
    )
    return kv_dt, raw_elems * kv_bytes * profile.kv_compress_factor / tok_split


class _ScopeContextBuilder:
    """Assembles per-core weights, FLOPs, VMEM scratches, and collectives."""

    def __init__(
        self,
        profile: ModelProfile,
        workload: WorkloadSpec,
        chip_spec: ChipSpec,
    ) -> None:
        """Initialize workload geometry, sharding counts, and layer parts."""
        self.profile = profile
        self.workload = workload
        self.chip_spec = chip_spec
        self.eff_tp = effective_tp(workload)
        self.kv_split = max(workload.kv_split, workload.dcp)
        self.eff_pcp = effective_pcp(workload)
        uncached_isl = max(1, round(workload.isl * (1.0 - workload.prefix_hit_rate)))
        self.batch_local = ceil(workload.batch / workload.dp)
        self.prefill_tokens_per_req = (
            workload.chunk_size or uncached_isl
            if workload.phase in ("prefill", "mixed")
            else 0
        )
        self.chunk_tokens = ceil((workload.chunk_size or uncached_isl) / self.eff_pcp)
        self.layers_local = ceil(profile.num_layers / workload.pp)
        self.eff_ctx = workload.context or (
            workload.isl + (0 if workload.phase == "prefill" else workload.osl // 2)
        )
        spec_tok = workload.spec_tokens
        self.verify_tokens = (
            max(1, spec_tok) + 1
            if workload.phase == "verify"
            else (spec_tok + 1 if spec_tok > 0 else 1)
        )
        prefill_tokens = (
            ceil(workload.chunk_size / self.eff_pcp)
            if workload.chunk_size
            else ceil(
                self.batch_local
                * uncached_isl
                / max(1, (workload.microbatches or 1) * self.eff_pcp)
            )
        )
        self.decode_tokens = (
            0
            if workload.phase == "prefill"
            else self.batch_local
            * (1 if workload.phase == "decode" else self.verify_tokens)
        )
        self.prefill_rows = (
            prefill_tokens
            if workload.phase == "prefill"
            else (self.chunk_tokens if workload.phase == "mixed" else 0)
        )
        self.active_tokens = self.decode_tokens + self.prefill_rows
        self.kv_read_batches = (
            min(
                float(self.batch_local),
                max(1.0, (prefill_tokens * self.eff_pcp) / max(1, uncached_isl)),
            )
            if workload.phase == "prefill"
            else float(self.batch_local)
        )
        self.local_attn = min(
            self.layers_local,
            ceil(profile.num_attention_layers * self.layers_local / profile.num_layers),
        )
        self.local_moe = min(
            self.layers_local,
            ceil(profile.num_moe_layers * self.layers_local / profile.num_layers),
        )
        self.local_lin = min(
            self.layers_local - self.local_attn,
            ceil(profile.num_linear_layers * self.layers_local / profile.num_layers),
        )
        self.local_indexer = (
            self.local_attn
            if (profile.sparse_attn is not None and profile.sparse_attn.index_dim > 0)
            else (
                min(
                    self.local_attn,
                    ceil(
                        profile.num_indexer_layers
                        * self.layers_local
                        / profile.num_layers
                    ),
                )
                if has_dsa_profile(profile)
                else 0
            )
        )
        cores_per_chip = chip_spec.cores_per_device * chip_spec.devices_per_chip
        eff_latent_tp = (
            workload.latent_proj_tp
            if workload.latent_proj_tp > 0
            else (
                MEGACORE_CORES
                if cores_per_chip >= MEGACORE_CORES and self.eff_tp >= MEGACORE_CORES
                else 1
            )
        )
        self.parts = _compute_layer_parts(
            profile,
            (
                self.eff_tp,
                self.kv_split,
                workload.ep,
                eff_latent_tp,
                max(1, workload.dcp),
            ),
        )
        self.token_residual_bytes = (
            self.active_tokens * profile.hidden_size * float(BF16_BYTES)
        )

    def _build_sparse_attn_terms(
        self, kv_bytes_per_elem: float, raw_attn_ctx: float
    ) -> tuple[dict[str, float | bool], dict[str, int | float]]:
        """Price QSA, CSA2+SWA, and MSA sparse caches, scans, and attended tokens."""
        profile, workload, parts = self.profile, self.workload, self.parts
        sp = profile.sparse_attn
        zero_dcp: dict[str, int | float] = {
            "query_s": 0.0,
            "query_hops": 0,
            "post_s": 0.0,
            "post_hops": 0,
            "dcp_hops": 0,
        }
        if sp is None:
            return {}, zero_dcp
        kv_dt = canonical_kv_dtype(profile.kv_dtype)
        cap_tok = sp.window_tokens + sp.select_tokens
        attn_ctx = min(raw_attn_ctx, float(cap_tok)) if cap_tok > 0 else raw_attn_ctx
        is_gather = cap_tok > 0 and raw_attn_ctx > float(cap_tok)
        tok_div = float(
            max(1, workload.dcp) if profile.dcp_span == "tp_group" else self.kv_split
        )
        pp_scale = (
            float(self.layers_local) / max(1, profile.num_layers)
            if (sp.kv_owners or sp.index_owners)
            else 1.0
        )
        kv_owners, idx_owners = _synthesize_homogeneous_owners(
            sp, profile, self.local_attn
        )
        kv_cfg = (kv_dt, kv_bytes_per_elem, self.eff_tp, parts.kvh_local, False)
        idx_cfg = (kv_dt, kv_bytes_per_elem, self.eff_tp, parts.kvh_local, True)
        win_l = (
            ceil(len(sp.window_layers) / max(1, workload.pp))
            if sp.window_layers
            else (self.local_attn if sp.window_bytes_per_token > 0 else 0)
        )
        swa_res = (
            self.batch_local
            * win_l
            * min(self.eff_ctx, sp.window_tokens)
            * (sp.window_bytes_per_token / max(1, self.eff_tp))
        )
        ring_res = float(self.batch_local * sp.kv_owner_ring_bytes_per_req) * pp_scale
        kv_res, kv_read_strict, kv_read_gather = self._accumulate_sparse_kv_owners(
            kv_owners, kv_cfg, (attn_ctx, tok_div, pp_scale), is_gather=is_gather
        )
        idx_res, idx_read, flops_idx = self._accumulate_sparse_index_owners(
            sp, idx_owners, idx_cfg, (raw_attn_ctx, tok_div, pp_scale)
        )
        idx_q_params, idx_kw_params = _sparse_indexer_proj_params(
            sp, profile, self.eff_tp, self.local_attn
        )
        return {
            "has_dsa": True,
            "is_sparse_gather": is_gather,
            "indexer_ctx_local": raw_attn_ctx / tok_div,
            "attn_ctx": attn_ctx,
            "indexer_query_proj_params": idx_q_params,
            "indexer_key_weight_proj_params": idx_kw_params,
            "indexer_k_read_bytes": idx_read,
            "indexer_k_resident_bytes": swa_res + ring_res + kv_res + idx_res,
            "dsa_kv_gather_bytes": swa_res + kv_read_gather,
            "dsa_kv_strict_bytes": swa_res + kv_read_strict,
            "flops_indexer_score": flops_idx,
            "vec_ops_pre_dsa_extra": 0.0,
            "vec_ops_dsa": 0.0,
            "vec_ops_indexer_radix": 0.0,
        }, zero_dcp

    def _accumulate_sparse_kv_owners(
        self,
        kv_owners: tuple[SparseCacheOwner, ...],
        kv_cfg: tuple[str, float, int, int, bool],
        scale_terms: tuple[float, float, float],
        *,
        is_gather: bool,
    ) -> tuple[float, float, float]:
        """Accumulate resident, strict-read, and gather-read sparse KV owner bytes."""
        attn_ctx, tok_div, pp_scale = scale_terms
        kv_res = kv_read_strict = kv_read_gather = 0.0
        for owner in kv_owners:
            rec_b = _owner_record_bytes(owner, self.profile, self.workload, kv_cfg)
            rec_tok = max(1.0, owner.record_tokens)
            kv_res += (
                self.batch_local
                * ((self.eff_ctx / rec_tok) / tok_div)
                * rec_b
                * pp_scale
            )
            sel_r = (attn_ctx / rec_tok) / tok_div
            users_n = max(1, len(owner.users))
            strict_b = users_n * self.batch_local * sel_r * rec_b * pp_scale
            kv_read_strict += strict_b
            kv_read_gather += (
                users_n
                * self.active_tokens
                * sel_r
                * _align_up(rec_b, HBM_CACHELINE_BYTES)
                * pp_scale
                if is_gather
                else strict_b
            )
        return kv_res, kv_read_strict, kv_read_gather

    def _accumulate_sparse_index_owners(
        self,
        sp: SparseAttnSpec,
        idx_owners: tuple[SparseCacheOwner, ...],
        idx_cfg: tuple[str, float, int, int, bool],
        scale_terms: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Accumulate resident, read, and scoring FLOP terms across index owners."""
        raw_attn_ctx, tok_div, pp_scale = scale_terms
        idx_res = idx_read = flops_idx = 0.0
        for owner in idx_owners:
            rec_b = _owner_record_bytes(owner, self.profile, self.workload, idx_cfg)
            rec_tok = max(1.0, owner.record_tokens)
            idx_res += (
                self.batch_local
                * ((self.eff_ctx / rec_tok) / tok_div)
                * rec_b
                * pp_scale
            )
            scan_r = raw_attn_ctx / rec_tok
            if owner.max_scan_records is not None:
                scan_r = min(scan_r, float(owner.max_scan_records))
            loc_scan = scan_r / tok_div
            users_n = max(1, len(owner.users))
            idx_read += users_n * self.batch_local * loc_scan * rec_b * pp_scale
            flops_idx += (
                users_n
                * 2.0
                * self.active_tokens
                * sp.index_heads
                * sp.index_dim
                * loc_scan
                * pp_scale
            )
        return idx_res, idx_read, flops_idx

    def _build_dsa_terms(
        self, kv_bytes_per_elem: float
    ) -> tuple[dict[str, float | bool], dict[str, int | float]]:
        """Compute DSA sparse traffic, indexer FLOP/vector ops, and DCP exchanges."""
        profile, workload, parts = self.profile, self.workload, self.parts
        raw_attn_ctx = (
            max(0, self.eff_ctx - self.prefill_tokens_per_req)
            + (self.prefill_tokens_per_req + 1) / 2.0
            if workload.phase == "prefill" and self.prefill_tokens_per_req > 1
            else float(self.eff_ctx)
        )
        if profile.sparse_attn is not None:
            return self._build_sparse_attn_terms(kv_bytes_per_elem, raw_attn_ctx)
        if not has_dsa_profile(profile):
            dsa_fields: dict[str, float | bool] = {
                "has_dsa": False,
                "is_sparse_gather": False,
                "indexer_ctx_local": 0.0,
                "attn_ctx": raw_attn_ctx * profile.kv_compress_factor,
                "indexer_k_read_bytes": 0.0,
                "indexer_k_resident_bytes": 0.0,
                "dsa_kv_gather_bytes": 0.0,
                "dsa_kv_strict_bytes": 0.0,
                "flops_indexer_score": 0.0,
                "vec_ops_pre_dsa_extra": 0.0,
                "vec_ops_dsa": 0.0,
                "vec_ops_indexer_radix": 0.0,
            }
            return dsa_fields, {
                "query_s": 0.0,
                "query_hops": 0,
                "post_s": 0.0,
                "post_hops": 0,
                "dcp_hops": 0,
            }

        kv_lora, qk_rope = profile.kv_lora_rank, profile.qk_rope_head_dim
        dcp = max(1, workload.dcp)
        is_sparse_gather = raw_attn_ctx > profile.index_topk
        kv_seq_split = max(1, self.kv_split, self.eff_pcp)
        active_kv_split = max(1, self.kv_split)
        active_idx_ctx_local = raw_attn_ctx / active_kv_split
        selected_topk = min(raw_attn_ctx, float(profile.index_topk))
        active_sel_topk = selected_topk / active_kv_split
        idx_record_bytes = float(indexer_record_bytes(profile.index_head_dim))
        indexer_k_read_bytes = (
            self.local_indexer
            * self.kv_read_batches
            * (raw_attn_ctx / kv_seq_split)
            * idx_record_bytes
        )
        indexer_k_resident_bytes = (
            self.local_indexer
            * self.batch_local
            * (self.eff_ctx / kv_seq_split)
            * idx_record_bytes
        )
        raw_kv_elems = (
            float(kv_lora + qk_rope)
            if kv_lora > 0
            else 2.0 * parts.kvh_local * profile.head_dim
        )
        dsa_entry_bytes = (
            _align_up(kv_lora * kv_bytes_per_elem, HBM_CACHELINE_BYTES)
            + _align_up(qk_rope * kv_bytes_per_elem, HBM_CACHELINE_BYTES)
            if kv_lora > 0
            else _align_up(raw_kv_elems * kv_bytes_per_elem, HBM_CACHELINE_BYTES)
        )
        dsa_kv_strict_bytes = (
            self.local_attn
            * self.kv_read_batches
            * (selected_topk / kv_seq_split)
            * raw_kv_elems
            * kv_bytes_per_elem
            * profile.kv_compress_factor
        )
        gather_blocks = float(self.decode_tokens) + (
            ceil(self.prefill_rows / ATTN_TILE_TOKENS) if self.prefill_rows > 0 else 0
        )
        dsa_kv_gather_bytes = (
            self.local_attn * gather_blocks * active_sel_topk * dsa_entry_bytes
            if is_sparse_gather
            else dsa_kv_strict_bytes
        )
        flops_indexer_score = (
            2.0
            * self.local_indexer
            * self.active_tokens
            * profile.index_n_heads
            * profile.index_head_dim
            * active_idx_ctx_local
        )
        vec_ops_radix = (
            INDEXER_RADIX_PASSES
            * INDEXER_RADIX_PASS_VEC_OPS
            * self.local_indexer
            * self.active_tokens
            * active_idx_ctx_local
            if is_sparse_gather
            else 0.0
        )
        vec_ops_pre_extra = (
            INDEXER_HEAD_REDUCE_VEC_OPS * profile.index_n_heads + INDEXER_SCALE_VEC_OPS
        ) * (
            self.local_indexer * self.active_tokens * active_idx_ctx_local
        ) + vec_ops_radix
        sel_topk_pad = _align_up(active_sel_topk, ATTN_TILE_TOKENS)
        kv_elem_pad = _align_up(raw_kv_elems, ATTN_TILE_TOKENS)
        heads_with_dcp = parts.nh_local * dcp
        dsa_value_dim = kv_lora if kv_lora > 0 else profile.head_dim
        unpack_ops = (
            self.local_attn
            * gather_blocks
            * DSA_UNPACK_MASK_VEC_OPS
            * sel_topk_pad
            * kv_elem_pad
            if is_sparse_gather
            else 0.0
        )
        vec_ops_dsa = unpack_ops + self.local_attn * self.active_tokens * (
            DSA_SOFTMAX_VEC_OPS * heads_with_dcp * sel_topk_pad
            + DSA_OUTPUT_RESCALE_VEC_OPS * heads_with_dcp * dsa_value_dim
            + (
                DCP_LSE_MERGE_VEC_OPS * dcp * parts.nh_local * dsa_value_dim
                if dcp > 1
                else 0.0
            )
        )
        dsa_fields = {
            "has_dsa": True,
            "is_sparse_gather": is_sparse_gather,
            "indexer_ctx_local": active_idx_ctx_local,
            "attn_ctx": selected_topk * profile.kv_compress_factor,
            "indexer_k_read_bytes": indexer_k_read_bytes,
            "indexer_k_resident_bytes": indexer_k_resident_bytes,
            "dsa_kv_gather_bytes": dsa_kv_gather_bytes,
            "dsa_kv_strict_bytes": dsa_kv_strict_bytes,
            "flops_indexer_score": flops_indexer_score,
            "vec_ops_pre_dsa_extra": vec_ops_pre_extra,
            "vec_ops_dsa": vec_ops_dsa,
            "vec_ops_indexer_radix": vec_ops_radix,
        }
        if dcp <= 1:
            return dsa_fields, {
                "query_s": 0.0,
                "query_hops": 0,
                "post_s": 0.0,
                "post_hops": 0,
                "dcp_hops": 0,
            }
        dcp_hops = round(math.log2(dcp))
        out_merge_dim = kv_lora if kv_lora > 0 else profile.head_dim
        egress_bw = self.chip_spec.ici_egress_bw_bytes_per_s
        query_bytes = (
            self.local_attn
            * (dcp - 1)
            * self.active_tokens
            * parts.nh_local
            * dsa_query_gather_dim(profile)
            * float(BF16_BYTES)
        )
        post_bytes = (
            self.local_attn
            * (dcp - 1)
            * self.active_tokens
            * parts.nh_local
            * (out_merge_dim * float(BF16_BYTES) + float(FP32_BYTES))
        ) + (
            self.local_indexer
            * (dcp - 1)
            * self.active_tokens
            * min(profile.index_topk, INDEXER_RADIX_PASSES * INDEXER_RADIX_BUCKETS)
            * float(FP32_BYTES)
            if is_sparse_gather
            else 0.0
        )
        return dsa_fields, {
            "query_s": query_bytes / egress_bw,
            "query_hops": self.local_attn * dcp_hops,
            "post_s": post_bytes / egress_bw,
            "post_hops": (
                self.local_attn + (self.local_indexer if is_sparse_gather else 0)
            )
            * dcp_hops,
            "dcp_hops": dcp_hops,
        }

    def build(
        self, plan_error: str = "", meas: MeasurementsData | None = None
    ) -> _ScopeContext:
        """Build immutable `_ScopeContext` across physics, VMEM, and collectives."""
        profile, workload, chip_spec, parts = (
            self.profile,
            self.workload,
            self.chip_spec,
            self.parts,
        )
        w_bytes = DTYPE_BYTES[profile.weight_dtype.lower()]
        e_bytes = DTYPE_BYTES[(profile.expert_dtype or profile.weight_dtype).lower()]
        kv_alias = canonical_kv_dtype(profile.kv_dtype)
        kv_bytes = DTYPE_BYTES[kv_alias]
        st_bytes = DTYPE_BYTES[profile.state_dtype.lower()]
        dsa, dsa_dcp = self._build_dsa_terms(kv_bytes)
        idx_q_params = float(
            dsa.get("indexer_query_proj_params", parts.indexer_query_proj_params)
        )
        idx_kw_params = float(
            dsa.get(
                "indexer_key_weight_proj_params", parts.indexer_key_weight_proj_params
            )
        )
        hc_params, hc_flops, hc_vec_ops = compute_hyper_connection_terms(
            profile, self.layers_local, self.eff_tp, self.active_tokens
        )
        tbl_proj_local = sum(
            ceil(tbl.dense_proj_params / self.eff_tp) for tbl in profile.host_tables
        )
        base_dense_params = (
            self.local_attn * parts.attn_params
            + self.local_indexer * idx_q_params
            + self.local_lin * parts.linear_attn_params
            + (self.layers_local - self.local_moe) * parts.dense_ffn_params
            + self.local_moe * (parts.shared_expert_params + parts.latent_proj_params)
        )
        static_weight_bytes = (
            base_dense_params * w_bytes
            + self.local_moe
            * profile.hidden_size
            * profile.num_experts
            * float(BF16_BYTES)
            + self.local_indexer * idx_kw_params * float(BF16_BYTES)
            + (hc_params + tbl_proj_local) * float(BF16_BYTES)
        )
        exp_bytes = parts.routed_expert_params * e_bytes
        tok_split = max(
            1,
            max(1, workload.dcp) if profile.dcp_span == "tp_group" else self.kv_split,
            self.eff_pcp,
        )
        kv_dt, kv_bytes_per_tok = _kv_bytes_per_token(
            profile, workload, parts.kvh_local, kv_bytes, tok_split
        )
        full_kv_resident = (
            self.local_attn * self.batch_local * self.eff_ctx * kv_bytes_per_tok
        )
        full_kv_read = (
            self.local_attn * self.kv_read_batches * self.eff_ctx * kv_bytes_per_tok
        )
        if profile.sparse_attn is not None:
            resident_kv_bytes = float(dsa["indexer_k_resident_bytes"])
        elif dsa["has_dsa"]:
            resident_kv_bytes = full_kv_resident + float(
                dsa["indexer_k_resident_bytes"]
            )
        else:
            resident_kv_bytes = full_kv_resident
        kv_cache_read_bytes = (
            float(dsa["indexer_k_read_bytes"]) + float(dsa["dsa_kv_gather_bytes"])
            if dsa["has_dsa"]
            else full_kv_read
        )
        z_dim = parts.vh_local * profile.linear_value_dim
        qkv_proj_dim = 2 * parts.kh_local * profile.linear_key_dim + z_dim
        recurrent_state_rw_bytes = (
            RECURRENT_STATE_RW_PASSES
            * self.local_lin
            * self.batch_local
            * (
                z_dim * profile.linear_key_dim
                + qkv_proj_dim * profile.linear_conv_kernel
            )
            * st_bytes
        )
        hc_per_layer_bf16 = (
            ceil(hc_params / max(1, self.layers_local)) * float(BF16_BYTES)
            if hc_params > 0
            else 0.0
        )
        layer_dense_weight_bytes = int(
            (
                max(
                    parts.attn_params
                    + (idx_q_params if self.local_indexer > 0 else 0.0),
                    parts.linear_attn_params,
                )
                + (
                    (parts.shared_expert_params + parts.latent_proj_params)
                    if self.local_moe > 0
                    else parts.dense_ffn_params
                )
            )
            * w_bytes
            + (idx_kw_params * float(BF16_BYTES) if self.local_indexer > 0 else 0.0)
            + hc_per_layer_bf16
        )
        kv_lora = profile.kv_lora_rank
        qk_nope = profile.qk_nope_head_dim or profile.head_dim
        mla_attn_dim = self.decode_tokens * (
            2 * kv_lora + profile.qk_rope_head_dim
        ) + self.prefill_rows * (qk_nope + profile.qk_rope_head_dim + profile.head_dim)
        hadamard_flops = compute_hadamard_kv_bf16_flops(
            kv_dt,
            (parts.nh_local, parts.kvh_local, profile.head_dim),
            self.active_tokens,
            self.local_attn,
        )
        flops_kv = (
            2.0
            * self.local_attn
            * float(dsa["attn_ctx"])
            * (
                parts.nh_local * mla_attn_dim
                if kv_lora
                else 2.0 * self.active_tokens * parts.q_dim
            )
            + hadamard_flops
        )
        flops_dense = (
            2.0
            * self.active_tokens
            * (
                base_dense_params
                + self.local_moe * profile.hidden_size * profile.num_experts
                + self.local_indexer * idx_kw_params
            )
            + flops_kv
            + float(dsa["flops_indexer_score"])
            + hc_flops
            + 2.0 * self.active_tokens * tbl_proj_local
        )
        flops_moe = (
            2.0
            * self.active_tokens
            * parts.ep_dp
            * self.local_moe
            * profile.top_k
            * parts.routed_expert_params
            * parts.experts_local
            / profile.num_experts
            if parts.experts_local > 0
            else 0.0
        )
        ced_scale = _ced_token_scale(
            profile, workload, self.layers_local, self.active_tokens, self.chunk_tokens
        )
        if ced_scale < 1.0:
            flops_kv *= ced_scale
            flops_dense *= ced_scale
            flops_moe *= ced_scale
        gqa_scratch = (
            2 * ATTN_TILE_TOKENS * (FP32_BYTES * max(profile.head_dim, kv_lora))
            + 2 * ATTN_TILE_TOKENS * ATTN_TILE_TOKENS * FP32_BYTES
            if self.local_attn > 0
            else 0
        )
        gdn_scratch = (
            FP32_BYTES
            * min(parts.vh_local, GDN_STREAMED_HEADS_CAP)
            * profile.linear_key_dim
            * profile.linear_value_dim
            if self.local_lin > 0
            else 0
        )
        ffn_scratch = (
            2 * ATTN_TILE_TOKENS * parts.i_exp_local * FP32_BYTES
            if self.local_moe > 0
            else (2 * ATTN_TILE_TOKENS * DENSE_FFN_TILE_CAP * FP32_BYTES)
        )
        ring_plan = workload.ring_plan or default_double_buffer_ring_plan(
            workload.dma_buffers
        )
        vmem_budget = fit_vmem_budget_bytes(
            tiles=workload.tiles,
            ring_plan=ring_plan,
            chip_spec=chip_spec,
            vmem_budget_mib=workload.vmem_budget_mib,
            apply_ring_reserve=bool(workload.vmem_fit_headroom),
        )
        dense_bank = (
            workload.tiles.bank
            if workload.tiles is not None
            else ring_plan.dense_slot_bytes
        )
        expert_bank = (
            workload.tiles.ebank
            if workload.tiles is not None
            else ring_plan.expert_slot_bytes
        )
        first_slot_bytes = float(
            min(
                dense_bank,
                max(
                    MIN_FIRST_DMA_SLOT_BYTES,
                    layer_dense_weight_bytes // FIRST_SLOT_LAYER_WEIGHT_DIVISOR,
                ),
            )
        )
        ar_cost = compute_all_reduce_transfer_cost(
            self.token_residual_bytes, self.eff_tp, chip_spec
        )
        ar_hops = round(math.log2(self.eff_tp)) if self.eff_tp > 1 else 0
        num_segments = (
            ceil(self.layers_local / workload.segment_layers)
            if 0 < workload.segment_layers < self.layers_local
            else 1
        )
        pcp_tokens = (
            self.active_tokens if workload.phase == "prefill" else self.chunk_tokens
        )
        pcp_ring_time = (
            (self.eff_pcp - 1)
            * self.local_attn
            * pcp_tokens
            * (kv_bytes_per_tok * tok_split)
            / chip_spec.ici_link_bw_bytes_per_s
            if self.eff_pcp > 1 and self.local_attn > 0
            else 0.0
        )
        row_limit = (
            workload.max_rows_per_call if workload.phase in ("decode", "verify") else 0
        )
        tokens_per_req = (
            max(1, workload.spec_tokens) + 1 if workload.phase == "verify" else 1
        )
        single_call_rows = (row_limit // tokens_per_req) * tokens_per_req
        eff_expert_dt = (
            (profile.expert_dtype or profile.weight_dtype)
            if self.local_moe > 0
            else profile.weight_dtype
        )
        buf_slots = workload.dma_buffers
        breaks = active_fusion_break_layers(profile, workload)
        has_fusion_break = bool(breaks)
        eff_layer_groups = workload.layer_groups or fusion_break_layer_groups(
            profile.num_layers, breaks
        )
        tbl_hbm_b, tbl_read_b, host_h2d_s = _compute_host_table_context_terms(
            profile,
            workload,
            (self.eff_tp, self.batch_local, self.active_tokens),
            chip_spec.pcie_h2d_bw_bytes_per_s,
        )
        return _ScopeContext(
            vmem_budget=vmem_budget,
            single_layer_dense_staging_bytes=min(
                buf_slots * dense_bank, buf_slots * layer_dense_weight_bytes
            ),
            single_layer_expert_staging_bytes=min(
                buf_slots * expert_bank, int(buf_slots * exp_bytes)
            ),
            first_slot_bytes=first_slot_bytes,
            gqa_scratch=gqa_scratch,
            gdn_scratch=gdn_scratch,
            ffn_scratch=ffn_scratch,
            layers_local=self.layers_local,
            num_segments=num_segments,
            num_layer_groups=(
                ceil(len(eff_layer_groups) * self.layers_local / profile.num_layers)
                if eff_layer_groups
                else 0
            ),
            local_moe_layers=self.local_moe,
            local_attn_layers=self.local_attn,
            local_linear_layers=self.local_lin,
            active_tokens=self.active_tokens,
            single_call_rows=single_call_rows,
            eff_ctx=self.eff_ctx,
            batch_local=self.batch_local,
            verify_tokens=self.verify_tokens,
            nh_local=parts.nh_local,
            q_dim=parts.q_dim,
            base_dense_params=base_dense_params,
            per_token_fused_bytes=compute_per_token_vmem_bytes(profile, workload)[
                "total_per_token_bytes"
            ],
            token_residual_bytes=self.token_residual_bytes,
            w_static=static_weight_bytes,
            exp_bytes=exp_bytes,
            experts_local=parts.experts_local,
            ep_dp=parts.ep_dp,
            eff_tp=self.eff_tp,
            row_floor=chip_spec.matmul_row_tile(eff_expert_dt),
            num_ar=(
                profile.ar_slots_per_layer * self.layers_local if self.eff_tp > 1 else 0
            ),
            ar_hops=ar_cost.hop_count,
            ar_intra_hops=ar_hops,
            ar_inter_latency_s=0.0,
            ar_transfer_time_s=ar_cost.transfer_time_s,
            ar_transfer_strict_s=ar_cost.nominal_volume_bytes
            / chip_spec.ici_egress_bw_bytes_per_s,
            ar_base_lat_s=ar_cost.base_latency_s,
            egress_bw=chip_spec.ici_egress_bw_bytes_per_s,
            lat_coll=chip_spec.collective_latency_s,
            lat_coll_upper=chip_spec.collective_latency_us_bracket[1] * 1e-6,
            launch_gap_upper_us=max(
                workload.op_overhead_us, DEFAULT_LAUNCH_GAP_UPPER_US
            ),
            pp_stage_time=(
                self.token_residual_bytes / chip_spec.ici_link_bw_bytes_per_s
                + chip_spec.collective_latency_s
                if workload.pp > 1
                else 0.0
            ),
            pcp_ring_time=pcp_ring_time,
            a2a_time=(
                2.0
                * (parts.ep_dp - 1)
                / parts.ep_dp
                * min(max(1, profile.top_k), parts.ep_dp)
                * self.token_residual_bytes
                / chip_spec.ici_egress_bw_bytes_per_s
                if parts.ep_dp > 1
                else 0.0
            ),
            flops_dense=flops_dense,
            flops_kv=flops_kv,
            flops_moe=flops_moe,
            peak_w=chip_spec.peak_flops(profile.weight_dtype, profile.act_dtype),
            peak_e=chip_spec.peak_flops(
                profile.expert_dtype or profile.weight_dtype,
                profile.expert_act_dtype or profile.act_dtype,
            ),
            bf16_peak=chip_spec.bf16_flops_per_s,
            attn_is_bf16=profile.act_dtype.lower() not in FP8_DTYPES
            or kv_dt not in FP8_DTYPES,
            kv_cache_read_bytes=kv_cache_read_bytes + tbl_read_b,
            recurrent_state_rw_bytes=recurrent_state_rw_bytes,
            bw_hbm=chip_spec.hbm_bw_bytes_per_s,
            hbm_used=(
                static_weight_bytes
                + self.local_moe * parts.experts_local * exp_bytes
                + resident_kv_bytes
                + recurrent_state_rw_bytes / RECURRENT_STATE_RW_PASSES
                + tbl_hbm_b
            ),
            no_whole_step=(
                workload.pp > 1
                or num_segments > 1
                or len(workload.layer_groups) > 1
                or has_fusion_break
                or profile.num_layers == 1
                or bool(plan_error)
            ),
            plan_error=plan_error,
            chip_spec=chip_spec,
            measured_ar_xla_s=_resolve_measured_all_reduce_s(
                meas, self.eff_tp, COLLECTIVE_PATH_XLA, self.token_residual_bytes
            ),
            measured_ar_in_kernel_s=_resolve_measured_all_reduce_s(
                meas, self.eff_tp, COLLECTIVE_PATH_IN_KERNEL, self.token_residual_bytes
            ),
            local_indexer_layers=self.local_indexer,
            has_dsa=bool(dsa["has_dsa"]),
            is_sparse_gather=bool(dsa["is_sparse_gather"]),
            indexer_ctx_local=float(dsa["indexer_ctx_local"]),
            dsa_kv_gather_bytes=float(dsa["dsa_kv_gather_bytes"]),
            dsa_kv_strict_bytes=float(dsa["dsa_kv_strict_bytes"]),
            vec_ops_pre_dsa_extra=float(dsa["vec_ops_pre_dsa_extra"]) + hc_vec_ops,
            vec_ops_dsa=float(dsa["vec_ops_dsa"]),
            vec_ops_indexer_radix=float(dsa["vec_ops_indexer_radix"]),
            dcp_query_all_gather_transfer_s=float(dsa_dcp["query_s"]),
            dcp_query_all_gather_hops=int(dsa_dcp["query_hops"]),
            dcp_post_attn_transfer_s=float(dsa_dcp["post_s"]),
            dcp_post_attn_hops=int(dsa_dcp["post_hops"]),
            dcp_hops=int(dsa_dcp["dcp_hops"]),
            has_fusion_break=has_fusion_break,
            host_h2d_s=host_h2d_s,
        )


def _build_scope_context(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    plan_error: str = "",
    meas: MeasurementsData | None = None,
) -> _ScopeContext:
    """Assemble `_ScopeContext` for `profile`, `workload`, and `chip_spec`."""
    return _ScopeContextBuilder(profile, workload, chip_spec).build(plan_error, meas)


class _ScopePricer:
    """Prices all five roofline bands and classifies the binding hardware unit."""

    def __init__(
        self,
        scope: FusionScope,
        scope_context: _ScopeContext,
        profile: ModelProfile,
        workload: WorkloadSpec,
    ) -> None:
        """Compute VMEM tiling, MoE padding, collectives, and vector rates per scope."""
        self.scope = scope
        self.profile = profile
        self.workload = workload
        self.ctx = scope_context
        self.expert_mode = (
            "none"
            if scope_context.local_moe_layers <= 0
            else ("ep" if workload.ep > 1 else "tp")
        )
        self.vmem_budget: _ScopeVmemBudget = _scope_vmem_and_ops(
            scope, self.ctx, profile, self.workload
        )
        self.n_ops = self.vmem_budget.ops_per_step
        self.n_splits = self.vmem_budget.num_vmem_splits
        self.is_fused_multi = scope in MULTI_LAYER_FUSED_SCOPES
        self.overlapped_comm = self.is_fused_multi and self.n_splits == 1
        self.tiled = self.is_fused_multi and self.n_splits > 1
        self.boundary_count = max(0, self.n_ops - 1)
        self.host_head_us = (
            workload.host_head_per_op_us
            if scope == "per_op"
            else workload.host_head_fused_us
        )

        # Resolve routed weight traffic, padding, and activation HBM traffic.
        tile_tokens = ceil(self.ctx.active_tokens / max(1, self.n_splits))
        can_group_moe = can_group_moe_in_scope(
            scope, self.ctx.eff_tp, self.ctx.ep_dp, self.workload.moe_mode
        )
        grouped_moe = can_group_moe and (
            scope == "per_op"
            or self.workload.moe_mode in GROUPED_MOE_MODES
            or tile_tokens * self.ctx.ep_dp >= self.ctx.row_floor
        )
        w_step, moe_pad = self._moe_weights_and_padding(
            tile_tokens,
            tiled=self.tiled or scope == "single_layer",
            grouped=grouped_moe,
        )
        if scope == "per_op":
            _, masked_pad = self._moe_weights_and_padding(
                tile_tokens, tiled=False, grouped=False
            )
            moe_pad = min(moe_pad, masked_pad)
        self.step_weights_bytes = w_step
        self.moe_pad_factor = moe_pad
        self.step_weights_strict_bytes = self._strict_moe_weights()
        extra_act = (
            2.0
            * max(1, profile.residual_streams)
            * self.ctx.layers_local
            * self.ctx.token_residual_bytes
            if self.tiled
            else 0.0
        )
        dsa_per_op_act = (
            PER_OP_INDEXER_SCORE_ROUNDTRIP_BYTES
            * self.ctx.local_indexer_layers
            * self.ctx.active_tokens
            * self.ctx.indexer_ctx_local
            + PER_OP_DSA_KV_ROUNDTRIP_FACTOR * self.ctx.dsa_kv_gather_bytes
            if scope == "per_op" and self.ctx.has_dsa and self.ctx.is_sparse_gather
            else 0.0
        )
        self.act_bytes = max(self.vmem_budget.act_bytes, extra_act) + dsa_per_op_act
        self.eff_state_bytes = (
            self.ctx.kv_cache_read_bytes + self.ctx.recurrent_state_rw_bytes
        )
        self.routed_ach_ratio = self._routed_achievable_ratio()
        self.vec_ops_dense, self.vec_rate = self._dense_vec_ops_and_rate()
        self.floor_coll = self._collective_summary(
            per_op_call_factor=1, allow_measured_ar=True
        )
        self.has_any_collectives = (
            (self.ctx.num_ar > 0 and self.ctx.eff_tp > 1)
            or self.ctx.a2a_time > 0
            or self.ctx.pcp_ring_time > 0
            or self.ctx.pp_stage_time > 0
            or self.ctx.dcp_hops > 0
            or self.floor_coll.has_dcp_collectives
        )
        self.indexer_extra_hops = (
            (INDEXER_RADIX_PASSES - 1)
            * self.ctx.local_indexer_layers
            * self.n_splits
            * self.ctx.dcp_hops
            if scope != "per_op"
            and self.ctx.has_dsa
            and self.ctx.is_sparse_gather
            and self.ctx.dcp_hops > 0
            else 0
        )

    def _strict_moe_weights(self) -> float:
        """Return strict lower-bound step weight bytes under balanced routing."""
        if self.ctx.experts_local == 0:
            return self.ctx.w_static
        e_strict = min(
            float(self.ctx.experts_local),
            max(
                1.0,
                self.profile.top_k
                * self.ctx.experts_local
                / max(1, self.profile.num_experts),
            ),
        )
        return (
            self.ctx.w_static
            + self.ctx.local_moe_layers * e_strict * self.ctx.exp_bytes
        )

    def _moe_weights_and_padding(
        self, tile_tokens: int, *, tiled: bool, grouped: bool
    ) -> tuple[float, float]:
        """Return step weight bytes read and the MoE matmul padding factor."""
        experts_local, ep_dp = self.ctx.experts_local, self.ctx.ep_dp
        if experts_local == 0:
            return self.ctx.w_static, 1.0
        tokens_ep = tile_tokens * ep_dp
        experts_hit = _experts_hit(
            tokens_ep, self.profile.top_k, self.profile.num_experts, experts_local
        )
        pairs = max(
            1.0,
            tokens_ep * self.profile.top_k * experts_local / self.profile.num_experts,
        )
        avg_m = max(1.0, pairs / experts_hit)
        hit = (
            _experts_hit(
                self.ctx.active_tokens * ep_dp,
                self.profile.top_k,
                self.profile.num_experts,
                experts_local,
            )
            if tiled
            else experts_hit
        )
        w_step = (
            self.ctx.w_static + self.ctx.local_moe_layers * hit * self.ctx.exp_bytes
        )
        if grouped:
            return w_step, ceil(avg_m / self.ctx.row_floor) * self.ctx.row_floor / avg_m
        padded_tokens_ep = ceil(tokens_ep / self.ctx.row_floor) * self.ctx.row_floor
        return w_step, max(1.0, (padded_tokens_ep * experts_hit) / pairs)

    def _routed_achievable_ratio(self) -> float:
        """Return busiest-rank to mean routed-expert load imbalance ratio."""
        if self.ctx.experts_local <= 0:
            return 1.0
        tokens_ep = self.ctx.active_tokens * self.ctx.ep_dp
        owner_groups = 1 if self.expert_mode == "tp" else self.workload.ep
        return experts_hit_busiest(
            tokens_ep,
            self.profile.top_k,
            self.profile.num_experts,
            self.ctx.experts_local,
            owner_groups,
        ) / _experts_hit(
            tokens_ep,
            self.profile.top_k,
            self.profile.num_experts,
            self.ctx.experts_local,
        )

    def _dense_vec_ops_and_rate(self) -> tuple[float, float]:
        """Return dense vector element ops per step and base word-op vector rate."""
        profile, ctx = self.profile, self.ctx
        chip_spec = ctx.chip_spec
        vec_rate = chip_spec.vector_f32_ops_per_s
        eff_tp = max(1, ctx.eff_tp)
        tokens = ctx.active_tokens
        dense_inter = (
            ceil(profile.intermediate_size / eff_tp)
            if profile.intermediate_size > 0
            else 0
        )
        shared_inter = (
            ceil(profile.shared_expert_intermediate_size / eff_tp)
            if profile.shared_expert_intermediate_size > 0
            else 0
        )
        weight_bytes = DTYPE_BYTES[profile.weight_dtype.lower()]
        subchannel_dense = (
            SUBCHANNEL_VPU_OPS_PER_BLOCK
            * tokens
            * ctx.base_dense_params
            / profile.weight_block_k
            if 0 < profile.weight_block_k < SUBCHANNEL_BLOCK_K_THRESHOLD
            and weight_bytes < float(BF16_BYTES)
            else 0.0
        )
        hc = profile.hyper_connection
        streams = max(1, hc.num_streams if hc is not None else profile.residual_streams)
        vec_ops_dense = (
            RMSNORM_RESIDUAL_VEC_OPS_PER_ELEM
            * streams
            * tokens
            * (profile.hidden_size / eff_tp)
            * ctx.layers_local
            + SWIGLU_ACT_VEC_OPS_PER_ELEM
            * tokens
            * dense_inter
            * (ctx.layers_local - ctx.local_moe_layers)
            + SWIGLU_ACT_VEC_OPS_PER_ELEM * tokens * shared_inter * ctx.local_moe_layers
            + subchannel_dense
            + ctx.vec_ops_pre_dsa_extra
            + ctx.vec_ops_dsa
        )
        return vec_ops_dense, vec_rate

    def _collective_summary(
        self, *, per_op_call_factor: int, allow_measured_ar: bool
    ) -> _CollectiveSummary:
        """Compute sync-stage, comm, xla-factor, and strict-saving collective terms."""
        ctx = self.ctx
        eff_num_ar = ctx.num_ar
        n_splits = self.n_splits
        measured_ar_s = (
            (
                ctx.measured_ar_xla_s
                if self.scope == "per_op"
                else ctx.measured_ar_in_kernel_s
            )
            if allow_measured_ar
            else None
        )
        has_measured_ar = measured_ar_s is not None and ctx.eff_tp > 1
        strict_ar_call_s = ctx.ar_transfer_strict_s + ctx.ar_base_lat_s
        if has_measured_ar and measured_ar_s is not None:
            clamped_ar_s = max(measured_ar_s, strict_ar_call_s)
            measured_ar_total_s = eff_num_ar * (
                clamped_ar_s + max(0, n_splits - 1) * ctx.ar_base_lat_s
            )
            ar_strict_saving_s = eff_num_ar * max(0.0, clamped_ar_s - strict_ar_call_s)
        else:
            measured_ar_total_s = 0.0
            ar_strict_saving_s = eff_num_ar * max(
                0.0, ctx.ar_transfer_time_s - ctx.ar_transfer_strict_s
            )

        xla_ar_factor = (
            per_op_call_factor if self.scope == "per_op" and ctx.eff_tp > 1 else 1
        )
        xla_hbm_coll_s = (
            (2.0 * eff_num_ar * ctx.token_residual_bytes) / ctx.bw_hbm
            if not has_measured_ar and self.scope == "per_op" and ctx.eff_tp > 1
            else 0.0
        )
        step_call_mult = per_op_call_factor if self.scope == "per_op" else 1
        step_transfer_s, step_hops, has_dcp = self._non_dsa_dcp_gather_terms(
            step_call_mult
        )
        if ctx.has_dsa and ctx.dcp_post_attn_hops > 0:
            step_hops += n_splits * xla_ar_factor * ctx.dcp_post_attn_hops
            step_transfer_s += ctx.dcp_post_attn_transfer_s
            has_dcp = True
        step_coll_s = step_transfer_s + step_hops * ctx.lat_coll

        ar_and_pcp_s = (
            (
                measured_ar_total_s
                if has_measured_ar
                else eff_num_ar
                * (
                    ctx.ar_transfer_time_s
                    + n_splits * xla_ar_factor * ctx.ar_base_lat_s
                )
                + xla_hbm_coll_s
            )
            + ctx.pcp_ring_time
            + step_coll_s
        )
        a2a_sync_s = (
            ctx.local_moe_layers
            * (ctx.a2a_time + 2 * n_splits * xla_ar_factor * ctx.lat_coll)
            if ctx.local_moe_layers > 0 and ctx.ep_dp > 1
            else 0.0
        )
        dcn_sync_s = (
            0.0
            if has_measured_ar
            else eff_num_ar * n_splits * xla_ar_factor * ctx.ar_inter_latency_s
        )
        if self.overlapped_comm:
            sync_stage_s = dcn_sync_s + ctx.pp_stage_time + a2a_sync_s
            comm_s = ar_and_pcp_s - dcn_sync_s
        else:
            sync_stage_s = 0.0
            comm_s = ar_and_pcp_s + ctx.pp_stage_time + a2a_sync_s
        return _CollectiveSummary(
            sync_stage_s=sync_stage_s,
            comm_s=comm_s,
            xla_ar_factor=xla_ar_factor,
            measured_ar_total_s=measured_ar_total_s,
            has_measured_ar=has_measured_ar,
            eff_num_ar=eff_num_ar,
            step_hops=step_hops,
            has_dcp_collectives=has_dcp,
            ar_strict_saving_s=ar_strict_saving_s,
        )

    def _non_dsa_dcp_gather_terms(self, step_call_mult: int) -> tuple[float, int, bool]:
        """Compute non-DSA cross-DCP gather transfer seconds and hops on egress ICI."""
        ctx, profile, workload = self.ctx, self.profile, self.workload
        has_non_dsa_dcp = (
            not (ctx.has_dsa and profile.sparse_attn is None)
            and ctx.local_attn_layers > 0
            and workload.dcp > 1
            and (workload.dcp & (workload.dcp - 1)) == 0
            and workload.phase in ("decode", "verify")
        )
        if not has_non_dsa_dcp:
            return 0.0, 0, False
        if profile.kv_lora_rank > 0:
            q_head_dim = profile.kv_lora_rank + profile.qk_rope_head_dim
            out_head_dim = profile.kv_lora_rank
        else:
            q_head_dim = (
                profile.qk_nope_head_dim + profile.qk_rope_head_dim
            ) or profile.head_dim
            out_head_dim = profile.head_dim
        bytes_per_head_tok = float(BF16_BYTES) * q_head_dim + float(FP32_BYTES) * (
            out_head_dim + LSE_SCALARS_PER_HEAD
        )
        wire_bytes = (
            ctx.local_attn_layers
            * (workload.dcp - 1)
            * ctx.active_tokens
            * ctx.nh_local
            * bytes_per_head_tok
        )
        dcp_hops = round(math.log2(workload.dcp))
        calls_per_layer = DCP_COLLECTIVES_PER_ATTN_LAYER * (
            step_call_mult if self.scope == "per_op" else 1
        )
        return (
            wire_bytes / ctx.egress_bw,
            ctx.local_attn_layers * calls_per_layer * self.n_splits * dcp_hops,
            True,
        )

    def _boundary_fill_drain_s(
        self,
        sync_stage_s: float,
        eta_pair: tuple[float, float],
        sync_hop_factor: float,
        *,
        include_first_slot_hbm: bool,
    ) -> float:
        """Return inter-call cold-slot fill, row-tile sync, and stage sync seconds."""
        eta_hbm, eta_ici = eta_pair
        first_slot_s = (
            (self.ctx.first_slot_bytes / self.ctx.bw_hbm) / eta_hbm
            if include_first_slot_hbm
            else 0.0
        )
        coll_hop_s = (
            (sync_hop_factor * self.ctx.lat_coll) / eta_ici
            if self.has_any_collectives
            else 0.0
        )
        per_call_s = first_slot_s + coll_hop_s
        tiled_sync_s = (
            (self.n_splits - 1) * 2 * self.ctx.layers_local * coll_hop_s
            if self.tiled
            else 0.0
        )
        return self.boundary_count * per_call_s + sync_stage_s + tiled_sync_s

    def _upper_collective_extras(self, coll: _CollectiveSummary) -> tuple[float, float]:
        """Return extra all-reduce, step, and all-to-all seconds at upper latency."""
        extra_lat_s = max(0.0, self.ctx.lat_coll_upper - self.ctx.lat_coll)
        ar_hops = (
            0
            if coll.has_measured_ar
            else coll.eff_num_ar
            * self.n_splits
            * coll.xla_ar_factor
            * self.ctx.ar_intra_hops
        )
        ar_extra_s = (ar_hops + coll.step_hops) * extra_lat_s
        a2a_extra_s = (
            (2 * self.ctx.local_moe_layers * self.n_splits * coll.xla_ar_factor)
            if self.ctx.local_moe_layers > 0 and self.ctx.ep_dp > 1
            else 0
        ) * extra_lat_s
        return ar_extra_s, a2a_extra_s

    def _dsa_stage_seconds(
        self,
        band_name: str,
        eta_triple: tuple[float, float, float],
        xla_ar_factor: int,
        lat_coll_s: float,
    ) -> tuple[float, float, float, float]:
        """Return pre-DSA and scattered-KV DSA `(total_s, mxu_s, vec_s, hbm_s)`."""
        if not self.ctx.has_dsa:
            return 0.0, 0.0, 0.0, 0.0
        eta_dense, eta_hbm, eta_ici = eta_triple
        is_strict = band_name == "strict"
        eff_vec_rate = (
            VECTOR_MAX_ELEMENTS_PER_WORD * self.vec_rate if is_strict else self.vec_rate
        )
        vec_pre_dsa = max(0.0, self.vec_ops_dense - self.ctx.vec_ops_dsa)
        if is_strict:
            vec_pre_dsa = max(0.0, vec_pre_dsa - self.ctx.vec_ops_indexer_radix)
        t_matmul_pre = (
            (self.ctx.flops_dense - self.ctx.flops_kv) / self.ctx.peak_w
        ) / eta_dense
        bf16_attn = self.ctx.attn_is_bf16 or (
            band_name in ("upper", "ach_upper")
            and self.workload.phase in ("prefill", "mixed")
        )
        t_matmul_dsa = _dense_attn_matmul_s(
            self.ctx, eta_dense, bf16_attn=bf16_attn, kv_only=True
        )
        vec_pre_s = (vec_pre_dsa / eff_vec_rate) / eta_dense
        vec_dsa_s = (self.ctx.vec_ops_dsa / eff_vec_rate) / eta_dense
        dcp_gather_s = (
            (
                self.ctx.dcp_query_all_gather_transfer_s
                + self.n_splits
                * xla_ar_factor
                * self.ctx.dcp_query_all_gather_hops
                * lat_coll_s
            )
            / eta_ici
            if self.ctx.dcp_query_all_gather_hops > 0
            else 0.0
        )
        dsa_bytes = (
            self.ctx.dsa_kv_strict_bytes if is_strict else self.ctx.dsa_kv_gather_bytes
        )
        dsa_hbm_s = (dsa_bytes / self.ctx.bw_hbm) / eta_hbm
        total_s = max(t_matmul_pre, vec_pre_s, dcp_gather_s) + max(
            t_matmul_dsa, vec_dsa_s, dsa_hbm_s
        )
        return total_s, t_matmul_pre + t_matmul_dsa, vec_pre_s + vec_dsa_s, dsa_hbm_s

    def _moe_vector_time_s(
        self, routed_bytes: float, routed_scale: float, *, strict: bool
    ) -> float:
        """Return MoE vector seconds summing base, unpack, and subchannel terms."""
        if self.ctx.local_moe_layers <= 0:
            return 0.0
        expert_dt = (self.profile.expert_dtype or self.profile.weight_dtype).lower()
        unpack_ops = (
            SUB_BYTE_UNPACK_VEC_OPS_PER_ELEM
            * max(0.0, routed_bytes / DTYPE_BYTES[expert_dt])
            if expert_dt in SUB_BYTE_EXPERT_DTYPES
            else 0.0
        )
        # Count 2x router softmax/top-k (2*E + 3*K) and combine (3*H) vector ops.
        base_ops = (
            2.0
            * self.ctx.local_moe_layers
            * self.ctx.active_tokens
            * (
                2 * self.profile.num_experts
                + 3 * self.profile.top_k
                + 3 * self.profile.hidden_size
            )
        )
        unpack_rate = VECTOR_SUB_BYTE_ELEMENTS_PER_WORD * self.vec_rate
        base_rate = (
            VECTOR_MAX_ELEMENTS_PER_WORD * self.vec_rate if strict else self.vec_rate
        )
        expert_bytes = DTYPE_BYTES[expert_dt]
        moe_block_k = self.profile.expert_weight_block_k or self.profile.weight_block_k
        if 0 < moe_block_k < SUBCHANNEL_BLOCK_K_THRESHOLD and expert_bytes < float(
            BF16_BYTES
        ):
            streamed = max(0.0, routed_bytes) / expert_bytes
            strat_a_s = streamed / (VECTOR_MAX_ELEMENTS_PER_WORD * self.vec_rate)
            strat_b_s = (routed_scale * self.ctx.flops_moe / moe_block_k) / base_rate
            subchannel_s = min(strat_a_s, strat_b_s)
        else:
            subchannel_s = 0.0
        return base_ops / base_rate + unpack_ops / unpack_rate + subchannel_s

    def _price_band(
        self,
        band_name: str,
        scope_cfg: _ScopeEvalConfig,
        upper_coll: _CollectiveSummary,
        ach_coll: _CollectiveSummary,
    ) -> _TimingPointResult:
        """Price one named band across HBM, compute chain, collectives, and launch."""
        is_strict = band_name == "strict"
        apply_eff = band_name in ("achievable", "ach_upper")
        is_upper = band_name in ("upper", "ach_upper")
        eff = self.workload.efficiency
        eta_dense = eff.eta_dense if apply_eff else 1.0
        eta_hbm = eff.eta_hbm if apply_eff else 1.0
        eta_ici = eff.eta_ici if apply_eff else 1.0
        eta_moe = (
            (eff.eta_moe_grouped if self.ctx.local_moe_layers > 0 else eff.eta_dense)
            if apply_eff
            else 1.0
        )
        routed_scale = self.routed_ach_ratio if apply_eff else 1.0
        w_used = (
            self.step_weights_strict_bytes if is_strict else self.step_weights_bytes
        )
        eff_state = (
            self.eff_state_bytes
            - self.ctx.dsa_kv_gather_bytes
            + self.ctx.dsa_kv_strict_bytes
            if is_strict and self.ctx.has_dsa
            else self.eff_state_bytes
        )
        routed_bytes = max(0.0, w_used - self.ctx.w_static) * routed_scale
        extra_routed = (
            max(0.0, w_used - self.ctx.w_static) * (routed_scale - 1.0)
            if apply_eff
            else 0.0
        )
        hbm_s = max(
            ((w_used + eff_state + self.act_bytes + extra_routed) / self.ctx.bw_hbm)
            / eta_hbm,
            self.ctx.host_h2d_s,
        )
        routed_hbm_s = (routed_bytes / self.ctx.bw_hbm) / eta_hbm

        dense_vec_rate = (
            VECTOR_MAX_ELEMENTS_PER_WORD * self.vec_rate if is_strict else self.vec_rate
        )
        vec_ops_eff = (
            max(0.0, self.vec_ops_dense - self.ctx.vec_ops_indexer_radix)
            if is_strict
            else self.vec_ops_dense
        )
        bf16_upper = scope_cfg.bf16_attention_upper and self.workload.phase in (
            "prefill",
            "mixed",
        )
        dense_mxu_s = _dense_attn_matmul_s(
            self.ctx,
            eta_dense,
            bf16_attn=self.ctx.attn_is_bf16 or (is_upper and bf16_upper),
        )
        dense_vec_s = (vec_ops_eff / dense_vec_rate) / eta_dense
        dense_stage_s = max(dense_mxu_s, dense_vec_s)

        pad_factor = 1.0 if is_strict else self.moe_pad_factor
        moe_mxu_s = (
            (pad_factor * routed_scale * self.ctx.flops_moe) / self.ctx.peak_e
        ) / eta_moe
        moe_vec_s = (
            self._moe_vector_time_s(routed_bytes, routed_scale, strict=is_strict)
            / eta_dense
        )

        lat_s = self.ctx.lat_coll_upper if is_upper else self.ctx.lat_coll
        xla_factor = (
            ach_coll.xla_ar_factor
            if apply_eff
            else (
                upper_coll.xla_ar_factor if is_upper else self.floor_coll.xla_ar_factor
            )
        )
        dsa_total_s, dsa_mxu_s, dsa_vec_s, dsa_hbm_s = self._dsa_stage_seconds(
            band_name, (eta_dense, eta_hbm, eta_ici), xla_factor, lat_s
        )
        chain_s = max(dense_stage_s, dsa_total_s) + max(
            routed_hbm_s, moe_mxu_s, moe_vec_s
        )
        comm_s, fill_drain_s, launch_us, host_us = self._band_collectives_and_launch(
            band_name, (eta_ici, eta_hbm), scope_cfg, (upper_coll, ach_coll)
        )
        core_s = _core_time(hbm_s, chain_s, comm_s, fused=self.overlapped_comm)
        roof_us = (core_s + fill_drain_s) * 1e6
        return _TimingPointResult(
            step_us=roof_us + launch_us + host_us,
            roof_us=roof_us,
            hbm_s=hbm_s,
            chain_s=chain_s,
            comm_s=comm_s,
            fill_drain_s=fill_drain_s,
            launch_us=launch_us,
            compute_s=dense_stage_s + max(moe_mxu_s, moe_vec_s),
            routed_hbm_s=routed_hbm_s,
            mxu_s=(dsa_mxu_s if self.ctx.has_dsa else dense_mxu_s) + moe_mxu_s,
            vec_s=(dsa_vec_s if self.ctx.has_dsa else dense_vec_s) + moe_vec_s,
            dsa_hbm_s=dsa_hbm_s,
        )

    def _band_collectives_and_launch(
        self,
        band_name: str,
        eta_pair: tuple[float, float],
        scope_cfg: _ScopeEvalConfig,
        coll_pair: tuple[_CollectiveSummary, _CollectiveSummary],
    ) -> tuple[float, float, float, float]:
        """Resolve `(comm_s, fill_drain_s, launch_us, host_us)` for one band."""
        eta_ici, eta_hbm = eta_pair
        upper_coll, ach_coll = coll_pair
        floor_fill = self._boundary_fill_drain_s(
            self.floor_coll.sync_stage_s, (1.0, 1.0), 0.0, include_first_slot_hbm=False
        )
        if band_name == "strict":
            return (
                max(0.0, self.floor_coll.comm_s - self.floor_coll.ar_strict_saving_s),
                floor_fill,
                self.n_ops * self.workload.op_overhead_us,
                0.0,
            )
        if band_name == "floor":
            return (
                self.floor_coll.comm_s,
                floor_fill,
                self.n_ops * self.workload.op_overhead_us,
                self.host_head_us,
            )
        is_upper = band_name in ("upper", "ach_upper")
        if band_name in ("achievable", "ach_upper"):
            ar_extra_ach, a2a_extra_ach = self._upper_collective_extras(ach_coll)
            unmeasured_comm = max(0.0, ach_coll.comm_s - ach_coll.measured_ar_total_s)
            extra_hops_s = self.indexer_extra_hops * (
                self.ctx.lat_coll_upper if is_upper else self.ctx.lat_coll
            )
            comm_s = (
                ach_coll.measured_ar_total_s
                + (unmeasured_comm + (ar_extra_ach if is_upper else 0.0) + extra_hops_s)
                / eta_ici
            )
            ach_fill = self._boundary_fill_drain_s(
                ach_coll.sync_stage_s,
                (eta_hbm, eta_ici),
                BOUNDARY_SYNC_HOP_FACTOR,
                include_first_slot_hbm=True,
            )
            sync_extra = (
                ach_coll.sync_stage_s * (1.0 / eta_ici - 1.0)
                if self.overlapped_comm
                else 0.0
            )
            a2a_extra = a2a_extra_ach / eta_ici if is_upper else 0.0
            if self.overlapped_comm:
                fill_drain_s = ach_fill + sync_extra + a2a_extra
            else:
                comm_s += a2a_extra
                fill_drain_s = ach_fill
            gap = (
                self.ctx.launch_gap_upper_us
                if is_upper
                else self.workload.op_overhead_us
            )
            return comm_s, fill_drain_s, self.n_ops * gap, self.host_head_us

        ar_extra_upper_s, a2a_extra_upper_s = self._upper_collective_extras(upper_coll)
        upper_fill = (
            self._boundary_fill_drain_s(
                upper_coll.sync_stage_s,
                (1.0, 1.0),
                BOUNDARY_SYNC_HOP_FACTOR,
                include_first_slot_hbm=True,
            )
            if scope_cfg.upper_boundary_fill
            else upper_coll.sync_stage_s
        )
        comm_s = upper_coll.comm_s + ar_extra_upper_s
        if self.overlapped_comm:
            fill_drain_s = upper_fill + a2a_extra_upper_s
        else:
            comm_s += a2a_extra_upper_s
            fill_drain_s = upper_fill
        return (
            comm_s,
            fill_drain_s,
            self.n_ops * self.ctx.launch_gap_upper_us,
            self.host_head_us,
        )

    def _classify_binding_unit(self, floor_pt: _TimingPointResult) -> str:
        """Classify dominant bottleneck unit from floor component durations."""
        core_s = _core_time(
            floor_pt.hbm_s,
            floor_pt.chain_s,
            floor_pt.comm_s,
            fused=self.overlapped_comm,
        )
        step_dev_s = core_s + floor_pt.fill_drain_s
        if self.host_head_us * 1e-6 > step_dev_s:
            return "host"
        if (floor_pt.launch_us + self.host_head_us) * 1e-6 > step_dev_s:
            return "launch"
        if not self.overlapped_comm and floor_pt.comm_s + floor_pt.fill_drain_s > max(
            floor_pt.hbm_s, floor_pt.chain_s
        ):
            return "comm"
        if self.overlapped_comm and (
            floor_pt.fill_drain_s
            > max(floor_pt.hbm_s, floor_pt.chain_s + floor_pt.comm_s)
            or (
                floor_pt.chain_s + floor_pt.comm_s > floor_pt.hbm_s
                and floor_pt.comm_s + floor_pt.fill_drain_s > floor_pt.chain_s
            )
        ):
            return "comm"
        if floor_pt.hbm_s >= floor_pt.chain_s + (
            floor_pt.comm_s if self.overlapped_comm else 0.0
        ):
            return "hbm"
        return max(
            (floor_pt.mxu_s, "matmul"),
            (floor_pt.vec_s, "vector"),
            (floor_pt.routed_hbm_s + floor_pt.dsa_hbm_s, "hbm"),
            key=lambda item: item[0],
        )[1]

    def price(
        self, scope_cfg: _ScopeEvalConfig
    ) -> tuple[ScopeEstimate, float, float, _RawStepMetrics]:
        """Price all five bands and return `(ScopeEstimate, step_us, sub_us, raw)`."""
        upper_coll = (
            self._collective_summary(
                per_op_call_factor=PER_OP_COLLECTIVE_CALL_FACTOR,
                allow_measured_ar=True,
            )
            if scope_cfg.upper_per_op_coll_factor
            else self.floor_coll
        )
        ach_coll = self._collective_summary(
            per_op_call_factor=PER_OP_COLLECTIVE_CALL_FACTOR, allow_measured_ar=True
        )
        strict_pt = self._price_band("strict", scope_cfg, upper_coll, ach_coll)
        floor_pt = self._price_band("floor", scope_cfg, upper_coll, ach_coll)
        upper_pt = self._price_band("upper", scope_cfg, upper_coll, ach_coll)
        ach_pt = self._price_band("achievable", scope_cfg, upper_coll, ach_coll)
        ach_upper_pt = self._price_band("ach_upper", scope_cfg, upper_coll, ach_coll)

        step_upper_us = max(floor_pt.step_us, upper_pt.step_us)
        step_ach_upper_us = max(ach_pt.step_us, step_upper_us, ach_upper_pt.step_us)
        new_sub_pure_us = (
            floor_pt.roof_us if self.scope == "per_op" else scope_cfg.sub_pure_us
        )
        raw = _RawStepMetrics(
            step_weight_bytes=self.step_weights_bytes,
            act_bytes=self.act_bytes,
            roof_floor_us=floor_pt.roof_us,
            step_bound_us=floor_pt.step_us,
            step_achievable_us=ach_pt.step_us,
            step_upper_us=step_upper_us,
            compute_us=floor_pt.compute_s * 1e6,
            hbm_us=floor_pt.hbm_s * 1e6,
            comm_us=floor_pt.comm_s * 1e6,
            chain_us=floor_pt.chain_s * 1e6,
            fill_drain_us=floor_pt.fill_drain_s * 1e6,
            launch_us=floor_pt.launch_us,
            step_achievable_upper_us=step_ach_upper_us,
            step_weight_strict_bytes=self.step_weights_strict_bytes,
            step_floor_strict_us=strict_pt.step_us,
        )
        estimate = _build_scope_estimate(
            self.scope,
            self.ctx,
            self.profile,
            self.workload,
            _EstimateBuildParams(
                raw=raw,
                vmem_budget=self.vmem_budget,
                moe_pad_factor=self.moe_pad_factor,
                sub_pure_us=new_sub_pure_us,
                expert_mode=self.expert_mode,
                binding_unit=self._classify_binding_unit(floor_pt),
            ),
        )
        return estimate, floor_pt.step_us, new_sub_pure_us, raw


@dataclass(frozen=True)
class _EstimateBuildParams:
    """Bundled unrounded metrics and classification fields for `ScopeEstimate`."""

    raw: _RawStepMetrics
    vmem_budget: _ScopeVmemBudget
    moe_pad_factor: float
    sub_pure_us: float
    expert_mode: str
    binding_unit: str
    host_head_us: float = 0.0
    legality_active_tokens: int | None = None


def _build_scope_estimate(
    scope: FusionScope,
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    params: _EstimateBuildParams,
) -> ScopeEstimate:
    """Assemble `ScopeEstimate` with core legality, critical weight, and break-even."""
    raw = params.raw
    vmem_budget = params.vmem_budget
    delta_ops = (
        SUBLAYER_BOUNDARIES_PER_LAYER * ctx.layers_local - vmem_budget.ops_per_step
    )
    break_even = (
        round(max(0.0, (raw.roof_floor_us - params.sub_pure_us) / delta_ops), 3)
        if delta_ops > 0
        else 0.0
    )
    act_saved = (
        4.0
        * max(1, profile.residual_streams)
        * ctx.layers_local
        * ctx.token_residual_bytes
        - raw.act_bytes
    )
    delta_hbm_bytes = ctx.bw_hbm * max(0, delta_ops) * workload.op_overhead_us * 1e-6
    critical_weight_gb = (
        round(
            ((act_saved + delta_hbm_bytes) / (vmem_budget.num_vmem_splits - 1)) / 1e9,
            4,
        )
        if vmem_budget.num_vmem_splits > 1 and params.host_head_us == 0.0
        else None
    )
    per_call_rows = (
        params.legality_active_tokens
        if params.legality_active_tokens is not None
        else ctx.active_tokens
    )
    support_status, blockers = evaluate_core_legality(
        scope,
        profile,
        workload,
        ctx.chip_spec,
        CoreLegalityContext(
            active_tokens=per_call_rows,
            eff_tp=ctx.eff_tp,
            ep_dp=ctx.ep_dp,
            plan_error=ctx.plan_error,
        ),
    )
    head_us = params.host_head_us or (
        workload.host_head_per_op_us
        if scope == "per_op"
        else workload.host_head_fused_us
    )
    return ScopeEstimate(
        scope=scope,
        feasible=vmem_budget.feasible,
        reason=vmem_budget.reason,
        ops_per_step=vmem_budget.ops_per_step,
        vmem_static_mib=round(vmem_budget.static_vmem_bytes / (1 << 20), 3),
        max_vmem_tokens=vmem_budget.max_vmem_tokens,
        num_vmem_splits=vmem_budget.num_vmem_splits,
        step_weight_gb=round(raw.step_weight_bytes / 1e9, 4),
        critical_weight_gb=critical_weight_gb,
        act_hbm_mb=round(raw.act_bytes / 1e6, 3),
        moe_flop_factor=round(params.moe_pad_factor, 3),
        roof_floor_us=round(raw.roof_floor_us, 2),
        step_bound_us=round(raw.step_bound_us + params.host_head_us, 2),
        selected_dense_slots=vmem_budget.selected_dense_slots,
        selected_expert_slots=vmem_budget.selected_expert_slots,
        step_achievable_us=round(raw.step_achievable_us + params.host_head_us, 2),
        break_even_op_overhead_us=break_even,
        step_upper_us=round(raw.step_upper_us + params.host_head_us, 2),
        compute_us=round(raw.compute_us, 2),
        hbm_us=round(raw.hbm_us, 2),
        comm_us=round(raw.comm_us, 2),
        chain_us=round(raw.chain_us, 2),
        fill_drain_us=round(raw.fill_drain_us, 2),
        launch_us=round(raw.launch_us, 2),
        step_achievable_upper_us=round(
            raw.step_achievable_upper_us + params.host_head_us, 2
        ),
        support=support_status,
        blockers=blockers,
        step_weight_strict_gb=round(raw.step_weight_strict_bytes / 1e9, 5),
        step_floor_strict_us=round(raw.step_floor_strict_us, 2),
        expert_mode=params.expert_mode,
        host_head_us=round(head_us, 2),
        ranking_metric="step_bound_us",
        binding_unit=params.binding_unit,
    )


def _evaluate_fused_scope_microbatched(
    scope: FusionScope,
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    scope_cfg: _ScopeEvalConfig,
) -> tuple[ScopeEstimate, float, float]:
    """Price a multi-layer fused scope above its single-call row limit."""
    tokens_per_req = scope_context.active_tokens // scope_context.batch_local
    per_call = scope_context.single_call_rows // tokens_per_req
    full_calls = scope_context.batch_local // per_call
    rest_reqs = scope_context.batch_local % per_call
    parts: list[tuple[ScopeEstimate, float, float, _RawStepMetrics]] = []
    for requests in (per_call, rest_reqs):
        if requests == 0:
            continue
        sub_wl = replace(workload, batch=requests * workload.dp, host_head_fused_us=0.0)
        built_sub = _build_scope_context(
            profile,
            sub_wl,
            scope_context.chip_spec,
            scope_context.plan_error,
            meas=scope_cfg.measurements,
        )
        sub_ctx = replace(
            built_sub,
            lat_coll=scope_context.lat_coll,
            lat_coll_upper=scope_context.lat_coll_upper,
            ar_base_lat_s=scope_context.ar_base_lat_s,
            ar_inter_latency_s=scope_context.ar_inter_latency_s,
            pp_stage_time=(
                0.0 if scope_context.pp_stage_time == 0.0 else built_sub.pp_stage_time
            ),
            launch_gap_upper_us=scope_context.launch_gap_upper_us,
        )
        parts.append(_ScopePricer(scope, sub_ctx, profile, sub_wl).price(scope_cfg))

    full_est, full_step_us, sub_pure_us, full_raw = parts[0]
    rest_est, rest_step_us, _, rest_raw = (
        parts[1] if parts[1:] else (None, 0.0, 0.0, None)
    )
    rest_multiplier = 1.0 if rest_raw is not None else 0.0
    zero_raw = full_raw if rest_raw is None else rest_raw
    summed_raw = _RawStepMetrics(
        step_weight_bytes=full_calls * full_raw.step_weight_bytes
        + rest_multiplier * zero_raw.step_weight_bytes,
        act_bytes=full_calls * full_raw.act_bytes
        + rest_multiplier * zero_raw.act_bytes,
        roof_floor_us=full_calls * full_raw.roof_floor_us
        + rest_multiplier * zero_raw.roof_floor_us,
        step_bound_us=full_calls * full_raw.step_bound_us
        + rest_multiplier * zero_raw.step_bound_us,
        step_achievable_us=full_calls * full_raw.step_achievable_us
        + rest_multiplier * zero_raw.step_achievable_us,
        step_upper_us=full_calls * full_raw.step_upper_us
        + rest_multiplier * zero_raw.step_upper_us,
        compute_us=full_calls * full_raw.compute_us
        + rest_multiplier * zero_raw.compute_us,
        hbm_us=full_calls * full_raw.hbm_us + rest_multiplier * zero_raw.hbm_us,
        comm_us=full_calls * full_raw.comm_us + rest_multiplier * zero_raw.comm_us,
        chain_us=full_calls * full_raw.chain_us + rest_multiplier * zero_raw.chain_us,
        fill_drain_us=full_calls * full_raw.fill_drain_us
        + rest_multiplier * zero_raw.fill_drain_us,
        launch_us=full_calls * full_raw.launch_us
        + rest_multiplier * zero_raw.launch_us,
        step_achievable_upper_us=full_calls * full_raw.step_achievable_upper_us
        + rest_multiplier * zero_raw.step_achievable_upper_us,
        step_weight_strict_bytes=full_calls * full_raw.step_weight_strict_bytes
        + rest_multiplier * zero_raw.step_weight_strict_bytes,
        step_floor_strict_us=full_calls * full_raw.step_floor_strict_us
        + rest_multiplier * zero_raw.step_floor_strict_us,
    )
    step_head_us = workload.host_head_fused_us
    if step_head_us > summed_raw.roof_floor_us:
        binding_unit = "host"
    elif summed_raw.launch_us + step_head_us > summed_raw.roof_floor_us:
        binding_unit = "launch"
    else:
        binding_unit = full_est.binding_unit
    calls = [full_est] * full_calls + ([rest_est] if rest_est is not None else [])
    combined_budget = _ScopeVmemBudget(
        static_vmem_bytes=0,
        act_bytes=summed_raw.act_bytes,
        ops_per_step=sum(call.ops_per_step for call in calls),
        max_vmem_tokens=full_est.max_vmem_tokens,
        num_vmem_splits=sum(call.num_vmem_splits for call in calls),
        feasible=all(call.feasible for call in calls),
        reason=next(
            (call.reason for call in calls if not call.feasible), full_est.reason
        ),
        selected_dense_slots=full_est.selected_dense_slots,
        selected_expert_slots=full_est.selected_expert_slots,
    )
    expert_mode = (
        "none"
        if scope_context.local_moe_layers <= 0
        else ("ep" if workload.ep > 1 else "tp")
    )
    combined = _build_scope_estimate(
        scope,
        scope_context,
        profile,
        workload,
        _EstimateBuildParams(
            raw=summed_raw,
            vmem_budget=combined_budget,
            moe_pad_factor=full_est.moe_flop_factor,
            sub_pure_us=sub_pure_us,
            expert_mode=expert_mode,
            binding_unit=binding_unit,
            host_head_us=step_head_us,
            legality_active_tokens=min(
                scope_context.active_tokens, scope_context.single_call_rows
            ),
        ),
    )
    return (
        replace(combined, vmem_static_mib=full_est.vmem_static_mib),
        full_calls * full_step_us + rest_step_us + step_head_us,
        sub_pure_us,
    )


def _evaluate_scopes(
    scope_context: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    *,
    upper_priors: tuple[bool, bool] = (True, True),
    measurements: MeasurementsData | None = None,
) -> tuple[list[ScopeEstimate], dict[FusionScope, float]]:
    """Evaluate every FusionScope against VMEM, HBM, and matmul bounds."""
    scopes_order: tuple[FusionScope, ...] = (
        "per_op",
        "single_layer",
        "layer_group",
        "whole_step",
    )
    raw_estimates: list[ScopeEstimate] = []
    raw_step_map: dict[FusionScope, float] = {}
    sub_pure_us = 0.0
    upper_boundary_fill, upper_per_op_coll_factor = upper_priors
    for scope in scopes_order:
        cfg = _ScopeEvalConfig(
            sub_pure_us=sub_pure_us,
            upper_boundary_fill=upper_boundary_fill,
            upper_per_op_coll_factor=upper_per_op_coll_factor,
            measurements=measurements,
        )
        if (
            scope in MULTI_LAYER_FUSED_SCOPES
            and 0 < scope_context.single_call_rows < scope_context.active_tokens
        ):
            scope_est, step_us, sub_pure_us = _evaluate_fused_scope_microbatched(
                scope, scope_context, profile, workload, cfg
            )
        else:
            scope_est, step_us, sub_pure_us, _ = _ScopePricer(
                scope, scope_context, profile, workload
            ).price(cfg)
        raw_estimates.append(scope_est)
        raw_step_map[scope] = step_us
    per_op_dev_us = raw_step_map["per_op"] - workload.host_head_per_op_us
    estimates = [
        scope_est
        if scope_est.scope == "per_op" or not scope_est.feasible
        else replace(
            scope_est,
            break_even_host_head_delta_us=compute_break_even_host_head_delta_us(
                raw_step_map[scope_est.scope] - workload.host_head_fused_us,
                per_op_dev_us,
            ),
        )
        for scope_est in raw_estimates
    ]
    return estimates, raw_step_map


def _resolve_scalar_launch_gap(
    workload: WorkloadSpec,
    meas: MeasurementsData | None,
) -> tuple[float, float, bool]:
    """Return `(floor_or_prior_us, upper_gap_us, apply_override)` for launch gap."""
    if workload.dispatch_regime == "zero_floor":
        return 0.0, 0.0, True
    if workload.dispatch_regime == "unfused_eager":
        val = (
            workload.op_overhead_us
            if workload.op_overhead_us > 0.0
            else UNFUSED_EAGER_GAP_US
        )
        return val, max(val, DEFAULT_LAUNCH_GAP_UPPER_US), True
    if meas is not None and meas.launch_gap_p50_us is not None:
        lo = meas.launch_gap_p50_us
        hi = max(lo, meas.launch_gap_p90_us or lo)
        return lo, hi, _is_calibrated_trace(meas)
    if workload.dispatch_regime == "compiled_xla":
        val = (
            workload.op_overhead_us
            if workload.op_overhead_us > 0.0
            else COMPILED_XLA_GAP_P50_US
        )
        return val, max(val, COMPILED_XLA_GAP_P90_US), True
    if workload.op_overhead_us > 0.0:
        return (
            workload.op_overhead_us,
            max(workload.op_overhead_us, DEFAULT_LAUNCH_GAP_UPPER_US),
            False,
        )
    return DEFAULT_LAUNCH_GAP_VAL_US, DEFAULT_LAUNCH_GAP_UPPER_US, False


def _prepare_scope_evaluation(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    *,
    plan_error: str = "",
    measurements: MeasurementsData | None = None,
) -> tuple[_ScopeContext, WorkloadSpec, float, float, float | None]:
    """Build context and apply calibrated dispatch/gap bounds without duplication."""
    scope_context = _build_scope_context(
        profile, workload, chip_spec, plan_error=plan_error, meas=measurements
    )
    gap_lo_us, gap_hi_us, apply_override = _resolve_scalar_launch_gap(
        workload, measurements
    )
    if apply_override:
        eval_workload = replace(workload, op_overhead_us=gap_lo_us)
        eval_ctx = replace(scope_context, launch_gap_upper_us=gap_hi_us)
    else:
        eval_workload = workload
        eval_ctx = scope_context
    step_coverage = (
        round(workload.step_coverage, 4)
        if workload.step_coverage is not None
        else (
            round(measurements.step_coverage, 4)
            if measurements is not None and measurements.step_coverage is not None
            else None
        )
    )
    return (
        eval_ctx,
        eval_workload,
        gap_lo_us,
        eval_ctx.launch_gap_upper_us,
        step_coverage,
    )


def _evaluate_scopes_with_context(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    *,
    plan_error: str = "",
    measurements: MeasurementsData | None = None,
) -> _EvaluatedScopes:
    """Validate inputs, prepare context, and evaluate all four fusion scopes."""
    effective_workload = workload
    if (
        measurements is not None
        and measurements.ring_plan is not None
        and effective_workload.ring_plan is None
    ):
        effective_workload = replace(
            effective_workload, ring_plan=measurements.ring_plan
        )
    validate_advise_inputs(profile, effective_workload, chip_spec)
    scope_context, eval_workload, gap_us, gap_upper_us, step_cov = (
        _prepare_scope_evaluation(
            profile,
            effective_workload,
            chip_spec,
            plan_error=plan_error,
            measurements=measurements,
        )
    )
    estimates_list, raw_step_map = _evaluate_scopes(
        scope_context, profile, eval_workload, measurements=measurements
    )
    return _EvaluatedScopes(
        estimates=tuple(estimates_list),
        raw_step_map=raw_step_map,
        scope_context=scope_context,
        effective_workload=effective_workload,
        eval_workload=eval_workload,
        launch_gap_us=gap_us,
        launch_gap_upper_us=gap_upper_us,
        measured_ar_xla_s=scope_context.measured_ar_xla_s,
        measured_ar_in_kernel_s=scope_context.measured_ar_in_kernel_s,
        step_coverage=step_cov,
        collective_latency_s=scope_context.lat_coll,
    )


def evaluate_scopes(
    profile: ModelProfile | str | Mapping[str, object],
    workload: WorkloadSpec,
    chip_spec: ChipSpec | str,
    measurements: MeasurementsData | None = None,
) -> tuple[ScopeEstimate, ...]:
    """Evaluate all four fusion scopes from first-principles hardware physics.

    Returns:
        Tuple of `ScopeEstimate` records ordered from `per_op` to `whole_step`.
    """
    resolved_profile = resolve_model_profile(profile)
    resolved_chip = resolve_chip_spec(chip_spec)
    return _evaluate_scopes_with_context(
        resolved_profile,
        workload,
        resolved_chip,
        measurements=measurements,
    ).estimates
