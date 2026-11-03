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

import importlib
from dataclasses import asdict, dataclass, replace
from math import ceil, gcd, log2
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import errors, exec_spec, ops, planner, spec
from vllm_torchtpu.megakernel.advisor.models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESETS,
    REGISTERED_MODEL_ALIASES,
    SUPPORTED_MODEL_TYPES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    UnsupportedArchitectureError,
    model_profile_from_hf_config,
    resolve_model_profile,
    supports_hf_config,
)
from vllm_torchtpu.megakernel.advisor.report import (
    format_advisor_summary,
    format_regime_table,
    render_advisor_html,
    write_advisor_html,
)
from vllm_torchtpu.megakernel.profiles import default as default_profiles

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

# Where the kernel boundary falls: one call per operator, per layer, per group
# of N layers (a model preset or a pipeline stage), or one call per step.
FusionScope = Literal["per_op", "single_layer", "layer_group", "whole_step"]
Phase = Literal["decode", "verify", "prefill", "mixed"]
# Where routed tokens are grouped by expert and where the data lives.
MoEMode = Literal["masked_on_chip", "grouped_on_chip", "grouped_in_hbm", "dense_mlp"]

FIT_VMEM_HEADROOM_MIB = 6
_WIDE_MATMUL_COLS = 256
_MIN_VMEM_MIB = 2
_MEGACORE_CORES = 2
_GROUPED_MOE_MIN_TOKENS = 32
_MAX_RECOMMENDED_STEP_WEIGHT_GB = 45.0
_VALID_DMA_BUFFERS = (2, 3)
_GROUPED_MOE_MODES = frozenset({"grouped_on_chip"})
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
_CHIP_ICI_SPECS: Mapping[str, tuple[int, float, float]] = {
    "v4": (1, 135.0, 45.0),
    "v5e": (0, 200.0, 50.0),
    "v5p": (1, 300.0, 100.0),
    "v6e": (0, 400.0, 100.0),
    "tpu7x": (0, 300.0, 100.0),
}
_TPU7X_TOPOLOGIES: Mapping[int, str] = {
    1: "1x1x1 (1 core)",
    2: "1x1x1 (1 chip, 2 cores)",
    4: "1x2x1 (2 chips, 4 cores)",
    8: "2x2x1 (4 chips, 8 cores)",
    16: "2x2x2 (8 chips, 16 cores)",
    32: "2x2x4 (16 chips, 32 cores)",
    64: "2x4x4 (32 chips, 64 cores)",
}


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one logical TPU device."""

    name: str
    cores_per_device: int
    tpu_info: pltpu.TpuInfo
    vmem_capacity_bytes: int
    vmem_usable_bytes: int
    hbm_capacity_bytes: int
    hbm_bw_bytes_per_s: float
    bf16_flops_per_s: float
    native_ocp_fp8: bool
    ici_egress_bw_bytes_per_s: float
    ici_link_bw_bytes_per_s: float
    collective_latency_s: float

    def peak_flops(self, dtype: str) -> float:
        """Return peak dense matrix FLOP/s for dtype on this logical device."""
        dt = dtype.lower()
        cores = self.cores_per_device
        info = self.tpu_info
        fp8_peak = dt in ("fp8", "nvfp4", "mxfp4", "fp4") and info.fp8_ops_per_second
        ops_s = fp8_peak or (
            info.int8_ops_per_second if dt == "int8" else info.bf16_ops_per_second
        )
        return float(ops_s) * cores * (0.5 if dt == "fp32" else 1.0)

    def lhs_row_floor(self, dtype: str) -> int:
        """Return the minimum unpadded LHS row tile height for dtype."""
        is_fp8 = (
            dtype.lower() in ("fp8", "nvfp4", "mxfp4", "fp4") and self.native_ocp_fp8
        )
        if is_fp8:
            return _GROUPED_MOE_MIN_TOKENS
        return 16 if self.tpu_info.mxu_column_size >= _WIDE_MATMUL_COLS else 8


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

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorSweepReport to a JSON-compatible dictionary."""
        return asdict(self)


def fit_vmem_budget_bytes(tiles: exec_spec.Tiles | None = None) -> int:
    """Return usable VMEM budget bytes after applying the 6 MiB compiler headroom."""
    target_tiles = tiles or default_profiles.TPU_V7X
    return max(0, target_tiles.vmem_mib - FIT_VMEM_HEADROOM_MIB) << 20


def resolve_chip_spec(
    chip: ChipSpec | str = "tpu7x",
    *,
    megacore: bool = False,
    ici_bw_gbps: float | None = None,
    collective_latency_us: float | None = None,
) -> ChipSpec:
    """Resolve hardware ceiling parameters from JAX Pallas tpu_info."""
    if isinstance(chip, ChipSpec):
        return chip
    raw = str(chip).strip().lower().removeprefix("tpu_")
    info = None
    if raw == "auto":
        try:
            info = pltpu.get_tpu_info()
            raw = info.chip_version.value.lower()
        except Exception as exc:
            msg = "chip='auto' requires TPU"
            raise errors.MegakernelConfigError(msg) from exc
    num_cores = int(info.num_cores) if info is not None else 1 + int(megacore)
    canon = "tpu7x" if raw in ("7x", "v7x", "tpu7x") else raw
    if canon not in _CHIP_ICI_SPECS or (megacore and not _CHIP_ICI_SPECS[canon][0]):
        msg = f"Invalid chip={chip!r} or Megacore"
        raise errors.MegakernelConfigError(msg)
    _, egress_gbps, link_gbps = _CHIP_ICI_SPECS[canon]
    chip_ver = pltpu.ChipVersion(canon.removeprefix("tpu"))
    tpu_info = info or pltpu.get_tpu_info_for_chip(chip_ver, num_cores)
    vmem_cap = int(tpu_info.vmem_capacity_bytes) * num_cores
    hbm_cap = int(tpu_info.hbm_capacity_bytes) * num_cores
    return ChipSpec(
        name=f"{canon}_megacore" if num_cores == _MEGACORE_CORES else canon,
        cores_per_device=num_cores,
        tpu_info=tpu_info,
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_cap - (2 << 20),
        hbm_capacity_bytes=hbm_cap,
        hbm_bw_bytes_per_s=float(tpu_info.mem_bw_bytes_per_second) * num_cores,
        bf16_flops_per_s=float(tpu_info.bf16_ops_per_second) * num_cores,
        native_ocp_fp8=bool(
            tpu_info.is_matmul_supported(jnp.float8_e4m3fn, jnp.float8_e4m3fn)
        ),
        ici_egress_bw_bytes_per_s=(
            egress_gbps * num_cores if ici_bw_gbps is None else ici_bw_gbps
        )
        * 1e9,
        ici_link_bw_bytes_per_s=link_gbps * 1e9,
        collective_latency_s=(
            1.0 if collective_latency_us is None else collective_latency_us
        )
        * 1e-6,
    )


def compute_per_token_vmem_bytes(
    profile: ModelProfile, workload: WorkloadSpec
) -> dict[str, int]:
    """Compute per-token live VMEM bytes from tensor lifetimes and TP/EP sharding."""
    hidden = profile.hidden_size
    tp = max(1, workload.tp)
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
            expert_tile_bytes = (
                8 * expert_hidden
                + ceil(profile.top_k * (6 * expert_hidden + 8 * inter_local) / ep)
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
        and (tp * dp * pcp) % ep == 0
        and tp % kv_split == 0
        and tp % dcp == 0
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
        reason = plan_error or "Bounded by layer groups, single layers, or non-periodic"

    if scope == "per_op":
        static_vmem_bytes = max(1 << 20, single_layer_dense_staging)
        alpha = 4 * profile.hidden_size
        n_splits = 1
        t_max = max(active_tokens, (vmem_budget - static_vmem_bytes) // alpha)
        n_ops = 2 * layers_local
        act_bytes = 4.0 * layers_local * token_residual_bytes
        return static_vmem_bytes, act_bytes, n_ops, t_max, n_splits, feasible, reason

    is_single_layer = scope == "single_layer"
    # Multi-layer calls per step for layer_group: the preset's groups (one call
    # each) or the pipeline-stage segments; whole_step is one call.
    mul = (preset_groups or num_segments) if scope == "layer_group" else 1
    dense_staging_bytes = (
        single_layer_dense_staging if is_single_layer else whole_step_dense_staging
    )
    expert_staging_bytes = (
        single_layer_expert_staging if is_single_layer else whole_step_expert_staging
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
        if is_single_layer
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
        else (mul * (3 if workload.tp * workload.ep > 1 and l_attn + l_lin else 1))
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


def _evaluate_scopes(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
) -> tuple[list[ScopeEstimate], dict[FusionScope, float]]:
    """Evaluate all four FusionScope options against VMEM, HBM, and matmul bounds."""
    layers_local, l_moe = ctx.layers_local, ctx.l_moe
    active_tokens = ctx.active_tokens
    token_residual_bytes = ctx.token_residual_bytes
    w_static, exp_bytes = ctx.w_static, ctx.exp_bytes
    experts_local, ep_dp = ctx.experts_local, ctx.ep_dp
    row_floor, num_ar, ar_hops = (
        ctx.row_floor,
        ctx.num_ar,
        ctx.ar_hops,
    )
    ar_vol, egress_bw, lat_coll = (
        ctx.ar_vol,
        ctx.egress_bw,
        ctx.lat_coll,
    )
    pp_stage_time, a2a_time = ctx.pp_stage_time, ctx.a2a_time
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    peak_w, peak_e, bw_hbm = (
        ctx.peak_w,
        ctx.peak_e,
        ctx.bw_hbm,
    )
    kv_cache_read_bytes = ctx.kv_cache_read_bytes
    recurrent_state_rw_bytes = ctx.recurrent_state_rw_bytes
    first_slot_bytes = ctx.first_slot_bytes

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
        w_step, phi = w_static, 1.0
        t_ep = ceil(active_tokens / max(1, n_splits)) * ep_dp
        if experts_local > 0:
            e_hit = max(
                1.0,
                experts_local
                * (1.0 - (1.0 - min(1.0, profile.top_k / profile.num_experts)) ** t_ep),
            )
            pairs = max(1.0, t_ep * profile.top_k * experts_local / profile.num_experts)
            avg_m = max(1.0, pairs / e_hit)
            w_step = w_static + l_moe * e_hit * exp_bytes
            is_binned = scope == "per_op" or workload.moe_mode in _GROUPED_MOE_MODES
            phi = (
                ceil(avg_m / row_floor) * row_floor / avg_m
                if is_binned
                else max(1.0, (ops.rup(t_ep, 16) * e_hit) / pairs)
            )

        # Out-of-kernel XLA collectives (per_op) execute 2-phase RS + AG on HBM
        # buffers, whereas in-kernel megakernel collectives push directly VMEM->VMEM.
        xla_ar_factor = 2 if scope == "per_op" and workload.tp > 1 else 1
        xla_hbm_coll_bytes = (
            2.0 * num_ar * token_residual_bytes
            if scope == "per_op" and workload.tp > 1
            else 0.0
        )
        t_comm = (
            num_ar
            * (ar_vol / egress_bw + n_splits * xla_ar_factor * ar_hops * lat_coll)
            + xla_hbm_coll_bytes / bw_hbm
            + pp_stage_time
            + (
                l_moe * (a2a_time + 2 * n_splits * xla_ar_factor * lat_coll)
                if l_moe > 0 and ep_dp > 1
                else 0.0
            )
        )
        # Pallas pipeline prologue fill + epilogue drain per separate kernel launch:
        # whole_step pays once/step; single_layer/per_op pay at each HLO boundary.
        # dma_buffers only sizes VMEM staging; state streams are priced at full bytes.
        eff_state_bytes = kv_cache_read_bytes + recurrent_state_rw_bytes
        pipeline_fill_drain_s = (n_ops - 1) * (
            first_slot_bytes / bw_hbm + 2.0 * lat_coll
        )
        t_matmul = flops_dense / peak_w + (phi * flops_moe) / peak_e
        act_saved = 4.0 * layers_local * token_residual_bytes - act_bytes
        hbm_int = n_splits * w_step + eff_state_bytes
        d_ops = sub_ops - n_ops
        roof_us = (
            max(t_matmul, (hbm_int + act_bytes) / bw_hbm)
            + t_comm
            + pipeline_fill_drain_s
        ) * 1e6
        base_ai = (flops_dense + flops_moe) / max(
            1.0, w_step + kv_cache_read_bytes + recurrent_state_rw_bytes
        )
        pure_us = (
            roof_us + (act_bytes + (n_splits - 1) * w_step) / bw_hbm * 1e6
            if base_ai >= peak_w / bw_hbm and n_splits > 1
            else (
                max(t_matmul, hbm_int / bw_hbm)
                + act_bytes / bw_hbm
                + t_comm
                + pipeline_fill_drain_s
            )
            * 1e6
        )
        if scope == "per_op":
            sub_pure_us = pure_us
        step_us = pure_us + n_ops * workload.op_overhead_us
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
    row_floor: int
    num_ar: int
    ar_hops: int
    ar_vol: float
    egress_bw: float
    lat_coll: float
    pp_stage_time: float
    a2a_time: float
    flops_dense: float
    flops_moe: float
    peak_w: float
    peak_e: float
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
    base_vmem_bytes = (
        int(workload.vmem_budget_mib * (1 << 20))
        if workload.vmem_budget_mib is not None
        else (tiles.vmem_mib << 20 if tiles else vmem_usable)
    )
    vmem_raw = (
        base_vmem_bytes - (FIT_VMEM_HEADROOM_MIB << 20) * workload.vmem_fit_headroom
    )
    return min(vmem_usable, max(0, vmem_raw))


def _whole_step_staging(
    workload: WorkloadSpec, layer_dense_weight_bytes: int
) -> tuple[float, int, int]:
    """Return first DMA slot bytes and multi-layer dense and expert staging bytes.

    ExecSpec.Tiles slot counts (nbd, nbe) and slot widths (bank, ebank) map to
    dense_buffer_slots, expert_buffer_slots, dense_slot_bytes, expert_slot_bytes.
    Not modeled yet: per-stream slot counts, e.g. 3 slots for recurrent-state
    read-update-write streams while large weight banks stay on 2 slots.
    """
    tiles = workload.tiles
    if tiles is not None:
        dense_slot_bytes = tiles.bank
        dense_staging = tiles.nbd * tiles.bank
        expert_staging = tiles.nbe * tiles.ebank
    else:
        dense_slot_bytes = 2 << 20
        extra_slot_bytes = (workload.dma_buffers - 2) * (2 << 20)
        dense_staging = min(28 << 20, 2 * layer_dense_weight_bytes) + extra_slot_bytes
        expert_staging = (10 << 20) + extra_slot_bytes
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
) -> _ScopeContext:
    """Assemble per-core weight, state, FLOP, and VMEM scope context."""
    bw_w, bw_e, bw_kv, bw_st = dtypes_bw
    num_layers, hidden = profile.num_layers, profile.hidden_size
    num_experts, top_k = profile.num_experts, profile.top_k
    tp, pp, dp, ep = workload.tp, workload.pp, workload.dp, workload.ep
    kv_split = max(workload.kv_split, workload.dcp)
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1
    spec_tok = workload.spec_tokens
    head_dim, kv_lora = profile.head_dim, profile.kv_lora_rank
    q_lora, qk_rope = profile.q_lora_rank, profile.qk_rope_head_dim

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
        max(1, ceil(profile.num_attention_heads / tp)),
        max(1, ceil(profile.num_kv_heads / (tp // kv_split))),
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
    kh_local = max(1, ceil(profile.linear_key_heads / tp)) if l_lin > 0 else 0
    vh_local = max(1, ceil(profile.linear_value_heads / tp)) if l_lin > 0 else 0
    z_dim, qkv_lin, ep_tp = (
        vh_local * vd,
        2 * kh_local * kd + vh_local * vd,
        gcd(ep, tp),
    )
    linear_attn_params, experts_local = (
        hidden * (qkv_lin + 2 * z_dim) * (l_lin > 0),
        (num_experts // ep) * (l_moe > 0),
    )
    dense_ffn_params, shared_expert_params, ep_dp = (
        3 * hidden * max(1, ceil(profile.intermediate_size / tp)),
        3 * hidden * ceil(profile.shared_expert_intermediate_size / tp),
        ep // ep_tp,
    )
    i_exp_local = ceil(profile.moe_intermediate_size / (tp // ep_tp)) * (l_moe > 0)
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
    flops_kv = (
        2.0
        * l_attn
        * eff_ctx
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
        row_floor=chip_spec.lhs_row_floor(
            profile.expert_dtype if l_moe > 0 else profile.weight_dtype
        ),
        num_ar=profile.ar_slots_per_layer * layers_local,
        ar_hops=round(log2(tp)),
        ar_vol=2.0 * (tp - 1) / tp * token_residual_bytes,
        egress_bw=chip_spec.ici_egress_bw_bytes_per_s,
        lat_coll=chip_spec.collective_latency_s,
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
        flops_moe=flops_moe,
        peak_w=chip_spec.peak_flops(profile.weight_dtype),
        peak_e=chip_spec.peak_flops(profile.expert_dtype),
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
            or num_layers == 1
            or bool(plan_error)
            or not profile.is_periodic
        ),
        plan_error=plan_error,
    )


def advise(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip: ChipSpec | str = "tpu7x",
    plan_error: str = "",
) -> AdvisorReport:
    """Evaluate fusion scopes from first-principles hardware and model bounds."""
    dtypes_bw = _validate_advise_inputs(profile, workload)
    chip_spec = resolve_chip_spec(chip)
    ctx = _build_scope_context(profile, workload, chip_spec, dtypes_bw, plan_error)
    estimates, raw_step_map = _evaluate_scopes(ctx, profile, workload)

    hbm_used = ctx.hbm_used
    hbm_fits = hbm_used <= chip_spec.hbm_capacity_bytes
    candidates = [e for e in estimates if e.feasible] if hbm_fits else [estimates[0]]
    best = min(candidates, key=lambda e: (raw_step_map[e.scope], e.ops_per_step))
    used_gb, cap_gb = hbm_used / 1e9, chip_spec.hbm_capacity_bytes / 1e9
    ridge = ctx.peak_w / ctx.bw_hbm
    b_wa = best.w_step_gb * 1e9 * best.n_splits + best.act_hbm_mb * 1e6
    state_bytes = ctx.kv_cache_read_bytes + ctx.recurrent_state_rw_bytes
    kv_lora, qk_rope = profile.kv_lora_rank, profile.qk_rope_head_dim
    nh_local, q_dim = ctx.nh_local, ctx.q_dim
    v_q = nh_local * (2 * kv_lora + qk_rope) if kv_lora else 2.0 * q_dim
    tot_bytes = max(1.0, b_wa + state_bytes)
    active_tokens = ctx.active_tokens
    eff_ctx = ctx.eff_ctx
    l_moe, l_attn = ctx.l_moe, ctx.l_attn
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    v_p = (
        ctx.base_dense_params
        + l_moe * profile.hidden_size * profile.num_experts
        + flops_moe / (2.0 * active_tokens)
    )
    v_flops = 2.0 * ctx.batch_local * ctx.verify_tokens * (v_p + l_attn * eff_ctx * v_q)
    sub_us = estimates[0].step_bound_us
    rationale = (
        f"{best.scope} ({best.step_bound_us:.1f} vs {sub_us:.1f} us)"
        if hbm_fits
        else f"Model+cache ({used_gb:.2f} GB) > HBM ({cap_gb:.2f} GB)"
    )
    rec_scope: FusionScope = best.scope if hbm_fits else "per_op"
    step_ai = round((flops_dense + best.moe_flop_factor * flops_moe) / tot_bytes, 2)
    asymp_ai = round(v_flops / state_bytes, 2) if state_bytes else None

    flop_per_tok = max(1.0, (flops_dense + flops_moe) / max(1, active_tokens))
    ridge_tokens = max(
        1,
        ceil((estimates[0].w_step_gb * 1e9 + state_bytes) * ridge / flop_per_tok),
    )
    if l_moe == 0:
        rec_moe = "dense_mlp"
    elif rec_scope == "per_op":
        rec_moe = "grouped_in_hbm"
    else:
        t_per_ep = ceil(active_tokens / max(1, best.n_splits)) * ctx.ep_dp
        rec_moe = (
            "grouped_on_chip"
            if t_per_ep >= _GROUPED_MOE_MIN_TOKENS
            else "masked_on_chip"
        )

    return AdvisorReport(
        model=profile.name,
        chip=chip_spec.name,
        phase=workload.phase,
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
        dma_buffers=workload.dma_buffers,
    )


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
        topo = _TPU7X_TOPOLOGIES.get(total_cores, f"{total_cores} cores")
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
        if (
            rep.hbm_used_gb <= 0.80 * rep.hbm_capacity_gb
            and sub.w_step_gb <= _MAX_RECOMMENDED_STEP_WEIGHT_GB
        ):
            return cand
    if fallback is not None:
        return fallback
    return ParallelismRecommendation(
        tp=8,
        ep=gcd(8 * dp, profile.num_experts) if profile.num_experts > 0 else 1,
        dp=dp,
        pp=pp,
        topology=_TPU7X_TOPOLOGIES.get(8 * dp * pp, f"{8 * dp * pp} cores"),
        hbm_used_gb=0.0,
        hbm_capacity_gb=round(chip_spec.hbm_capacity_bytes / 1e9, 3),
        w_step_gb=0.0,
        auto_selected=True,
    )


def _int_opt(opts: Mapping[str, object], key: str, default: int) -> int:
    """Extract an integer option from opts or return default."""
    val = opts.get(key, default)
    return int(val) if isinstance(val, (int, float, str)) else default


def _float_opt(opts: Mapping[str, object], key: str, default: float) -> float:
    """Extract a float option from opts or return default."""
    val = opts.get(key, default)
    return float(val) if isinstance(val, (int, float, str)) else default


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
        topology=_TPU7X_TOPOLOGIES.get(total_cores, f"{total_cores} cores"),
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
) -> tuple[ParallelismRecommendation, ParallelismRecommendation | None]:
    """Return decode and prefill layouts: preset, user-specified, or auto-pick."""
    tp, ep, dp, pp, dcp, pcp = requested
    if dep is not None:
        return (
            _layout_recommendation(prof, chip_spec, dep.decode, base),
            _layout_recommendation(prof, chip_spec, dep.prefill, base),
        )
    if tp is not None:
        ep_val = (
            ep
            if ep is not None
            else (gcd(tp * dp, prof.num_experts) if prof.num_experts > 0 else 1)
        )
        layout = ParallelLayout(tp=tp, ep=ep_val, dp=dp, pp=pp, dcp=dcp, pcp=pcp)
        return _layout_recommendation(prof, chip_spec, layout, base, source=""), None
    par = recommend_parallelism(prof, chip_spec, dp=dp, pp=pp)
    if ep is not None or dcp > 1 or pcp > 1:
        par = replace(par, ep=ep if ep is not None else par.ep, dcp=dcp, pcp=pcp)
    return par, None


def _with_preset_prefill(
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    dep: DeploymentPreset | None,
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Move prefill regimes onto the preset's prefill layout and add its windows."""
    if dep is None:
        return workloads
    pre = dep.prefill
    fields = {
        "tp": pre.tp,
        "ep": pre.ep,
        "dp": pre.dp,
        "pp": pre.pp,
        "dcp": pre.dcp,
        "pcp": pre.pcp,
        "layer_groups": pre.layer_groups,
    }
    decode_base = workloads[0][1]
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
                **fields,
            ),
        )
        for win in dep.prefill_windows
    )
    moved = tuple(
        (lbl, replace(wl, **fields) if wl.phase == "prefill" else wl)
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
    tp = int(raw_tp) if isinstance(raw_tp, int) else None
    ep = int(raw_ep) if isinstance(raw_ep, int) else None
    vmem_budget_mib = (
        float(raw_vmem) if isinstance(raw_vmem, (int, float, str)) else None
    )
    dp = _int_opt(opts, "dp", 1)
    pp = _int_opt(opts, "pp", 1)
    dcp = _int_opt(opts, "dcp", 1)
    pcp = _int_opt(opts, "pcp", 1)
    base = WorkloadSpec(
        vmem_alias=bool(opts.get("vmem_alias", True)),
        vmem_fit_headroom=bool(opts.get("vmem_fit_headroom", False)),
        op_overhead_us=_float_opt(opts, "op_overhead_us", 0.0),
        vmem_budget_mib=vmem_budget_mib,
        dma_buffers=_int_opt(opts, "dma_buffers", 2),
    )
    dma_buffers = base.dma_buffers

    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    explicit_layout = tp is not None or ep is not None or (dp, pp, dcp, pcp) != (1,) * 4
    dep = (
        DEPLOYMENT_PRESETS.get(prof.name)
        if bool(opts.get("deployment_preset", True)) and not explicit_layout
        else None
    )
    par, prefill_par = _select_layouts(
        prof, chip_spec, base, (tp, ep, dp, pp, dcp, pcp), dep
    )
    base = replace(
        base, tp=par.tp, ep=par.ep, dp=par.dp, pp=par.pp, dcp=par.dcp, pcp=par.pcp
    )
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
            "Spec Verify Small (B=4, K=3)",
            replace(
                base,
                phase="verify",
                batch=4,
                spec_tokens=3,
                isl=3072,
                osl=2048,
            ),
        ),
        (
            "Spec Verify Low (B=8, K=3)",
            replace(
                base,
                phase="verify",
                batch=8,
                spec_tokens=3,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Spec Verify Med (B=16, K=3)",
            replace(
                base,
                phase="verify",
                batch=16,
                spec_tokens=3,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Spec Verify High (B=32, K=3)",
            replace(
                base,
                phase="verify",
                batch=32,
                spec_tokens=3,
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
        rep = advise(prof, wl, chip_spec)
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
    )


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
    cli_mod = importlib.import_module("vllm_torchtpu.megakernel.advisor.cli")
    return cli_mod.build_arg_parser()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the megakernel advisor CLI entrypoint."""
    cli_mod = importlib.import_module("vllm_torchtpu.megakernel.advisor.cli")
    return int(cli_mod.main(argv))


__all__ = [
    "DEPLOYMENT_PRESETS",
    "FIT_VMEM_HEADROOM_MIB",
    "MODEL_PRESETS",
    "REGISTERED_MODEL_ALIASES",
    "SUPPORTED_MODEL_TYPES",
    "AdvisorReport",
    "AdvisorSweepReport",
    "ChipSpec",
    "DeploymentPreset",
    "FusionScope",
    "MoEMode",
    "ModelProfile",
    "ParallelLayout",
    "ParallelismRecommendation",
    "Phase",
    "RegimeEntry",
    "ScopeEstimate",
    "UnsupportedArchitectureError",
    "WorkloadSpec",
    "advise",
    "advise_from_spec",
    "advise_regimes",
    "build_arg_parser",
    "compute_per_token_vmem_bytes",
    "fit_vmem_budget_bytes",
    "format_advisor_summary",
    "format_regime_table",
    "main",
    "model_profile_from_hf_config",
    "recommend_parallelism",
    "render_advisor_html",
    "resolve_chip_spec",
    "resolve_model_profile",
    "supports_hf_config",
    "write_advisor_html",
]
