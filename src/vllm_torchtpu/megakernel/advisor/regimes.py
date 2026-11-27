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
"""Regime sweep, auto-parallelism recommendation, and cross-regime advisor matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, gcd
from typing import TYPE_CHECKING, Literal, get_args

from vllm_torchtpu.megakernel.core import errors

from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    CORE_LIMITS,
    DECODE_MED_REF_INDEX,
    DEFAULT_CHIP,
    DEFAULT_PREFILL_ISL,
    DEFAULT_PREFILL_OSL,
    DEFAULT_SPEC_TOKENS,
    DTYPE_BYTES,
    MAX_DCP_POWER_BITS,
    RECOMMEND_DEFAULT_BATCH,
    RECOMMEND_HIGH_BATCH,
    RECOMMEND_HIGH_CTX,
    SATURATION_BASE_BATCH,
    SATURATION_BATCH_CAP,
    TP_CANDIDATE_DEGREES,
    canonical_kv_dtype,
)
from .models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESET_SOURCES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    has_dsa_profile,
    indexer_record_bytes,
    resolve_model_profile,
)
from .specs import (
    AdvisorReport,
    AdvisorSweepReport,
    DisaggHandoffSpec,
    LayerGroupSearchReport,
    MeasurementsData,
    MoEMode,
    ParallelismRecommendation,
    Phase,
    RegimeEntry,
    ScopeEstimate,
    WorkloadSpec,
    align_batch_to_dp,
    format_parallel_layout_label,
    load_measurements,
)
from .verdict import advise, search_layer_groups
from .vmem import (
    _mesh_divisibility_reasons,
    compute_per_token_vmem_bytes,
    effective_pcp,
    effective_tp,
    is_slice_topology_supported,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "ParallelismSearchOptions",
    "RegimeSweepOptions",
    "advise_regimes",
    "recommend_parallelism",
]


@dataclass(frozen=True, slots=True)
class ParallelismSearchOptions:
    """Typed search constraints for `recommend_parallelism` and layout selection."""

    dp: int = 1
    pp: int = 1
    dcp: int = 0
    pcp: int = 1
    ep: int | None = None
    phase: Phase = "decode"
    tp: int | None = None

    def __post_init__(self) -> None:
        """Validate phase and axis lower bounds."""
        if self.phase not in get_args(Phase):
            msg = f"Unsupported phase {self.phase!r}; expected {get_args(Phase)}"
            raise errors.MegakernelConfigError(msg)
        if self.dcp < 0 or self.dp < 1 or self.pp < 1 or self.pcp < 1:
            msg = (
                f"Invalid search degrees: dp={self.dp}, pp={self.pp}, "
                f"dcp={self.dcp}, pcp={self.pcp}"
            )
            raise errors.MegakernelConfigError(msg)
        if self.ep is not None and self.ep < 1:
            msg = f"ep must be >= 1, got {self.ep}"
            raise errors.MegakernelConfigError(msg)
        if self.tp is not None and self.tp < 1:
            msg = f"tp must be >= 1, got {self.tp}"
            raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class RegimeSweepOptions:
    """Typed options for multi-regime advisor sweeps without dynamic `**opts`."""

    base_workload: WorkloadSpec | None = None
    layout: ParallelLayout | None = None
    tp: int | None = None
    ep: int | None = None
    dp: int = 1
    pp: int = 1
    dcp: int | None = None
    pcp: int = 1
    measurements: MeasurementsData | Mapping[str, object] | Path | str | None = None
    deployment_preset: bool = True


@dataclass(frozen=True, slots=True)
class _SelectedLayouts:
    """Resolved decode, prefill, and saturation recommendations for a sweep."""

    decode: ParallelismRecommendation
    prefill: ParallelismRecommendation | None = None
    saturation: ParallelismRecommendation | None = None


@dataclass(frozen=True, slots=True)
class _RegimeSpec:
    """Declarative specification for one canonical sweep regime."""

    template: str
    raw_batch: int
    phase: Phase
    isl: int
    osl: int | None = None
    context: int | None = None
    use_spec_tokens: bool = False
    chunk_size: int | None = None
    prefix_hit_rate: float | None = None
    moe_mode: MoEMode | None = None


def _reference_regime_specs() -> tuple[_RegimeSpec, ...]:
    """Return the canonical 12-regime matrix definitions."""
    high_ctx, prefill_isl = RECOMMEND_HIGH_CTX, DEFAULT_PREFILL_ISL
    grouped_moe: MoEMode = "grouped_on_chip"
    saturation_batch = SATURATION_BASE_BATCH
    regime_spec = _RegimeSpec
    spec_kwargs = {"use_spec_tokens": True}
    return (
        regime_spec("Decode Small (B={batch}, T=1, 4K)", 4, "decode", 3072, 2048),
        regime_spec("Decode Low (B={batch}, T=1, 4K)", 8, "decode", 3072, 2048),
        regime_spec("Decode Med (B={batch}, T=1, 4K)", 16, "decode", 3072, 2048),
        regime_spec(
            "Decode High (B={batch}, T=1, 32K)",
            64,
            "decode",
            high_ctx,
            2048,
            high_ctx,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Decode Saturated (B={batch}, T=1, 32K)",
            saturation_batch,
            "decode",
            high_ctx,
            2048,
            high_ctx,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Spec Verify Small (B={batch}, K={spec})",
            4,
            "verify",
            3072,
            2048,
            **spec_kwargs,
        ),
        regime_spec(
            "Spec Verify Low (B={batch}, K={spec})",
            8,
            "verify",
            3072,
            2048,
            **spec_kwargs,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Spec Verify Med (B={batch}, K={spec})",
            16,
            "verify",
            3072,
            2048,
            **spec_kwargs,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Spec Verify High (B={batch}, K={spec})",
            32,
            "verify",
            high_ctx,
            2048,
            high_ctx,
            **spec_kwargs,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Mixed Decode+Prefill (B={batch}, C=256)",
            16,
            "mixed",
            prefill_isl,
            2048,
            chunk_size=256,
            prefix_hit_rate=0.90,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Chunked Prefill (B={batch}, T=256)",
            1,
            "prefill",
            prefill_isl,
            osl=DEFAULT_PREFILL_OSL,
            chunk_size=256,
            prefix_hit_rate=0.9375,
            moe_mode=grouped_moe,
        ),
        regime_spec(
            "Long Prefill (B={batch}, T=4096)",
            1,
            "prefill",
            prefill_isl,
            osl=DEFAULT_PREFILL_OSL,
            moe_mode=grouped_moe,
        ),
    )


def _saturation_batch(raw_batch: int, dp: int) -> int:
    """Cap saturation batch at the sweep max while keeping `dp` alignment."""
    return align_batch_to_dp(min(SATURATION_BATCH_CAP, raw_batch), dp)


def _supports_tp_heads(profile: ModelProfile, tp_degree: int) -> bool:
    """Return True when `tp_degree` cleanly shards dense and linear heads."""
    heads = [profile.num_attention_heads]
    if profile.num_linear_layers > 0:
        if profile.linear_value_heads > 0:
            heads.append(profile.linear_value_heads)
        if profile.linear_key_heads > 0:
            heads.append(profile.linear_key_heads)
    return all(
        head_count >= tp_degree and head_count % tp_degree == 0 for head_count in heads
    )


def _resolve_ep(
    profile: ModelProfile, tp_degree: int, dp: int, pcp: int, explicit_ep: int | None
) -> int:
    """Resolve expert-parallel degree from explicit override or mesh/expert gcd."""
    if explicit_ep is not None:
        return explicit_ep
    return (
        gcd(tp_degree * dp * pcp, profile.num_experts) if profile.num_experts > 0 else 1
    )


def _slice_topology(
    chip_spec: ChipSpec, layout: ParallelLayout | WorkloadSpec, pcp_ranks: int
) -> str:
    """Return topology description for active slice cores."""
    total_cores = layout.tp * layout.dp * pcp_ranks * layout.pp
    return chip_spec.topologies.get(total_cores, f"{total_cores} cores")


def _build_parallelism_recommendation(
    layout: ParallelLayout | WorkloadSpec,
    advisor_report: AdvisorReport,
    topology: str,
    *,
    auto_selected: bool,
    source: str,
) -> ParallelismRecommendation:
    """Build a `ParallelismRecommendation` from a layout and evaluated report."""
    exec_mode = (
        layout.effective_execution
        if isinstance(layout, WorkloadSpec)
        else layout.execution
    )
    table_overrides = (
        layout.effective_host_table_hbm_overrides
        if isinstance(layout, WorkloadSpec)
        else layout.host_table_hbm_overrides
    )
    return ParallelismRecommendation(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        topology=topology,
        hbm_used_gb=advisor_report.hbm_used_gb,
        hbm_capacity_gb=advisor_report.hbm_capacity_gb,
        step_weight_gb=advisor_report.scopes[0].step_weight_gb,
        auto_selected=auto_selected,
        source=source,
        layer_groups=layout.layer_groups,
        execution=exec_mode,
        host_table_hbm_overrides=table_overrides,
    )


def _evaluate_parallelism_candidate(
    profile: ModelProfile, chip_spec: ChipSpec, workload: WorkloadSpec
) -> tuple[ParallelismRecommendation, bool] | None:
    """Evaluate HBM fit and 32K high-batch single-pass feasibility for one mesh."""
    if _mesh_divisibility_reasons(profile, workload, chip_spec):
        return None
    advisor_report = advise(profile, workload, chip_spec)
    if not advisor_report.hbm_fits:
        return None
    candidate_layout = _build_parallelism_recommendation(
        workload,
        advisor_report,
        _slice_topology(chip_spec, workload, effective_pcp(workload)),
        auto_selected=True,
        source="",
    )
    high_report = advise(
        profile,
        replace(
            workload,
            batch=align_batch_to_dp(RECOMMEND_HIGH_BATCH, workload.dp),
            isl=RECOMMEND_HIGH_CTX,
            context=RECOMMEND_HIGH_CTX,
        ),
        chip_spec,
    )
    whole_step = next(
        scope_est
        for scope_est in advisor_report.scopes
        if scope_est.scope == "whole_step"
    )
    return candidate_layout, (high_report.hbm_fits and whole_step.num_vmem_splits == 1)


def recommend_parallelism(
    profile: ModelProfile,
    chip: ChipSpec | str = DEFAULT_CHIP,
    *,
    options: ParallelismSearchOptions | None = None,
) -> ParallelismRecommendation:
    """Auto-derive the minimum balanced TPU parallelism mesh for a model and chip.

    Returns:
        Lowest-core valid `ParallelismRecommendation` fitting HBM.

    Raises:
        errors.MegakernelConfigError: If no supported topology fits in HBM.
    """
    search = options or ParallelismSearchOptions()
    chip_spec = resolve_chip_spec(chip)
    fallback: ParallelismRecommendation | None = None
    spec_tokens = DEFAULT_SPEC_TOKENS if search.phase in ("verify", "mixed") else 0
    eff_pcp = search.pcp if search.phase in ("prefill", "mixed") else 1
    tp_candidates = [
        tp_deg
        for tp_deg in TP_CANDIDATE_DEGREES
        if is_slice_topology_supported(
            tp_deg * search.dp * eff_pcp * search.pp, chip_spec
        )
        and _supports_tp_heads(profile, tp_deg)
    ]
    for tp_degree in tp_candidates:
        dcp_candidates = (
            tuple(
                1 << bit for bit in range(MAX_DCP_POWER_BITS) if (1 << bit) <= tp_degree
            )
            if search.dcp == 0
            else (search.dcp,)
        )
        ep_degree = _resolve_ep(profile, tp_degree, search.dp, eff_pcp, search.ep)
        for dcp_degree in dcp_candidates:
            workload = WorkloadSpec(
                phase=search.phase,
                batch=align_batch_to_dp(RECOMMEND_DEFAULT_BATCH, search.dp),
                spec_tokens=spec_tokens,
                tp=tp_degree,
                ep=ep_degree,
                dp=search.dp,
                pp=search.pp,
                dcp=dcp_degree,
                pcp=search.pcp,
                kv_split=dcp_degree,
            )
            evaluated = _evaluate_parallelism_candidate(profile, chip_spec, workload)
            if evaluated is None:
                continue
            candidate_layout, meets_preferred = evaluated
            if fallback is None:
                fallback = candidate_layout
            if meets_preferred:
                return candidate_layout
    if fallback is not None:
        return fallback
    msg = (
        f"No balanced parallelism layout fits in HBM for {profile.name} "
        f"on {chip_spec.name} (dp={search.dp}, pp={search.pp})"
    )
    raise errors.MegakernelConfigError(msg)


def _layout_recommendation(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    layout: ParallelLayout,
    base: WorkloadSpec,
    source: str = "deployment preset",
) -> ParallelismRecommendation:
    """Evaluate a fixed layout HBM fit at the reference decode workload."""
    effective_workload = replace(
        base.with_layout(layout),
        phase="decode",
        batch=align_batch_to_dp(RECOMMEND_DEFAULT_BATCH, layout.dp),
    )
    base_report = advise(profile, effective_workload, chip_spec)
    active_pcp = layout.pcp if source else effective_pcp(base)
    effective_layout = replace(
        layout,
        execution=effective_workload.effective_execution,
        host_table_hbm_overrides=effective_workload.effective_host_table_hbm_overrides,
    )
    return _build_parallelism_recommendation(
        effective_layout,
        base_report,
        _slice_topology(chip_spec, layout, active_pcp),
        auto_selected=False,
        source=source,
    )


def _select_layouts(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    base: WorkloadSpec,
    requested: ParallelismSearchOptions,
    layout_preset: DeploymentPreset | None,
) -> _SelectedLayouts:
    """Return decode, prefill, and saturation layouts: preset, user, or auto-pick."""
    if layout_preset is not None:
        sat_rec = (
            _layout_recommendation(profile, chip_spec, layout_preset.saturation, base)
            if layout_preset.saturation is not None
            else None
        )
        decode_rec = (
            sat_rec
            if base.objective == "throughput" and sat_rec is not None
            else _layout_recommendation(profile, chip_spec, layout_preset.decode, base)
        )
        return _SelectedLayouts(
            decode=decode_rec,
            prefill=_layout_recommendation(
                profile, chip_spec, layout_preset.prefill, base
            ),
            saturation=sat_rec,
        )
    if requested.tp is not None:
        eff_pcp = requested.pcp if requested.phase in ("prefill", "mixed") else 1
        ep_deg = _resolve_ep(profile, requested.tp, requested.dp, eff_pcp, requested.ep)
        layout = ParallelLayout(
            tp=requested.tp,
            ep=ep_deg,
            dp=requested.dp,
            pp=requested.pp,
            dcp=requested.dcp or 1,
            pcp=requested.pcp,
        )
        return _SelectedLayouts(
            decode=_layout_recommendation(profile, chip_spec, layout, base, source="")
        )
    return _SelectedLayouts(
        decode=recommend_parallelism(profile, chip_spec, options=requested)
    )


def _disagg_wire_payload_bytes(profile: ModelProfile, isl: int) -> tuple[float, float]:
    """Return `(kv_wire_bytes, state_wire_bytes)` per request for P/D handoff."""
    kv_dt = canonical_kv_dtype(profile.kv_dtype)
    kv_bytes = DTYPE_BYTES[kv_dt]
    state_bytes = DTYPE_BYTES[profile.state_dtype.lower()]
    kv_per_tok = (
        (profile.kv_lora_rank + profile.qk_rope_head_dim) * kv_bytes
        if profile.kv_lora_rank
        else 2.0 * profile.num_kv_heads * profile.head_dim * kv_bytes
    )
    kv_wire = (
        profile.num_attention_layers * isl * kv_per_tok * profile.kv_compress_factor
    )
    if has_dsa_profile(profile):
        kv_wire += (
            profile.num_indexer_layers
            * isl
            * indexer_record_bytes(profile.index_head_dim)
        )
    z_dim = profile.linear_value_heads * profile.linear_value_dim
    qkv_linear = 2 * profile.linear_key_heads * profile.linear_key_dim + z_dim
    state_wire = (
        profile.num_linear_layers
        * (z_dim * profile.linear_key_dim + qkv_linear * profile.linear_conv_kernel)
        * state_bytes
    )
    return kv_wire, state_wire


def _gbps_to_ms(total_bytes: float, bw_gbps: float) -> float:
    """Compute link transfer latency in milliseconds for `total_bytes`."""
    return round((total_bytes / max(1.0, (bw_gbps / 8.0) * 1e9)) * 1e3, 2)


def _compute_disagg_handoff(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    layout_preset: DeploymentPreset | None,
    measurements_data: MeasurementsData | None,
    isl: int = DEFAULT_PREFILL_ISL,
) -> DisaggHandoffSpec | None:
    """Compute per-request KV + recurrent state wire bytes and DCN transfer floor."""
    if layout_preset is None:
        return None
    kv_wire, state_wire = _disagg_wire_payload_bytes(profile, isl)
    total_wire = kv_wire + state_wire
    if measurements_data is not None and measurements_data.dcn_bw_gbps is not None:
        dcn_gbps = measurements_data.dcn_bw_gbps
        dcn_bracket, dcn_prov = (dcn_gbps, dcn_gbps), "MEASURED"
    else:
        dcn_gbps = chip_spec.dcn_bw_gbps
        dcn_bracket, dcn_prov = chip_spec.dcn_bw_gbps_bracket, "ASSUMED"
    provenance: Literal["ASSUMED", "MEASURED"] = dcn_prov
    return DisaggHandoffSpec(
        isl=isl,
        kv_wire_mb=round(kv_wire / 1e6, 2),
        state_wire_mb=round(state_wire / 1e6, 2),
        total_wire_mb=round(total_wire / 1e6, 2),
        dcn_bw_gbps=dcn_gbps,
        dcn_bw_gbps_bracket=dcn_bracket,
        dcn_xfer_ms=_gbps_to_ms(total_wire, dcn_gbps),
        dcn_xfer_bracket_ms=(
            _gbps_to_ms(total_wire, dcn_bracket[1]),
            _gbps_to_ms(total_wire, dcn_bracket[0]),
        ),
        dcn_provenance=provenance,
    )


def _with_preset_prefill(
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    layout_preset: DeploymentPreset | None,
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Move prefill and saturated decode regimes onto preset layouts and windows."""
    if layout_preset is None:
        return workloads
    prefill_layout, saturation_layout = layout_preset.prefill, layout_preset.saturation
    decode_base = workloads[0][1].with_layout(prefill_layout)
    windows = tuple(
        (
            f"Preset Prefill Window (W={window_tokens})",
            replace(
                decode_base,
                phase="prefill",
                batch=decode_base.dp,
                isl=window_tokens,
                osl=DEFAULT_PREFILL_OSL,
                chunk_size=window_tokens,
                layer_groups=prefill_layout.layer_groups,
                moe_mode="grouped_on_chip",
                moe_mode_explicit=True,
            ),
        )
        for window_tokens in layout_preset.prefill_windows
    )
    decode_dp = max(1, layout_preset.decode.dp)
    saturation_batch = (
        _saturation_batch(
            SATURATION_BASE_BATCH * saturation_layout.dp, saturation_layout.dp
        )
        if saturation_layout is not None
        else align_batch_to_dp(SATURATION_BASE_BATCH, decode_dp)
    )
    saturation_spec_tokens = layout_preset.decode.spec_tokens or DEFAULT_SPEC_TOKENS
    moved: list[tuple[str, WorkloadSpec]] = []
    for label, workload in workloads:
        if workload.phase == "prefill":
            moved.append(
                (
                    label,
                    replace(
                        workload.with_layout(prefill_layout, align_batch=True),
                        layer_groups=prefill_layout.layer_groups,
                    ),
                )
            )
        elif saturation_layout is not None and label.startswith("Decode Saturated"):
            entry_label = (
                f"Decode Saturated (B={saturation_batch} over "
                f"DP{saturation_layout.dp}, T=1, 32K)"
            )
            moved.append(
                (
                    entry_label,
                    replace(
                        workload.with_layout(saturation_layout),
                        batch=saturation_batch,
                        layer_groups=saturation_layout.layer_groups,
                    ),
                )
            )
        elif saturation_layout is not None and label.startswith("Spec Verify High"):
            moved.append((label, workload))
            entry_label = (
                f"Spec Verify Saturated (B={saturation_batch} over "
                f"DP{saturation_layout.dp}, K={saturation_spec_tokens})"
            )
            moved.append(
                (
                    entry_label,
                    replace(
                        workload.with_layout(saturation_layout),
                        phase="verify",
                        batch=saturation_batch,
                        spec_tokens=saturation_spec_tokens,
                        layer_groups=saturation_layout.layer_groups,
                        moe_mode="grouped_on_chip",
                        moe_mode_explicit=True,
                    ),
                )
            )
        elif (
            saturation_layout is None
            and decode_dp > 1
            and label.startswith("Decode Saturated")
        ):
            moved.append(
                (
                    label,
                    replace(
                        workload,
                        batch=align_batch_to_dp(workload.batch, decode_dp),
                    ),
                )
            )
        else:
            moved.append((label, workload))
    return tuple(moved) + windows


def _build_reference_workloads(
    base: WorkloadSpec, spec_tokens: int
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Build the standard 12 reference request regimes with DP-divisible batches."""
    built: list[tuple[str, WorkloadSpec]] = []
    for spec in _reference_regime_specs():
        batch_size = align_batch_to_dp(spec.raw_batch, base.dp)
        label = spec.template.format(batch=batch_size, spec=spec_tokens)
        workload = replace(
            base,
            phase=spec.phase,
            batch=batch_size,
            isl=spec.isl,
            osl=base.osl if spec.osl is None else spec.osl,
            context=base.context if spec.context is None else spec.context,
            spec_tokens=spec_tokens if spec.use_spec_tokens else base.spec_tokens,
            chunk_size=base.chunk_size if spec.chunk_size is None else spec.chunk_size,
            prefix_hit_rate=(
                base.prefix_hit_rate
                if spec.prefix_hit_rate is None
                else spec.prefix_hit_rate
            ),
            moe_mode=base.moe_mode if spec.moe_mode is None else spec.moe_mode,
            batch_explicit=True,
            moe_mode_explicit=(
                base.moe_mode_explicit if spec.moe_mode is None else True
            ),
        )
        built.append((label, workload))
    return tuple(built)


def _evaluate_sweep_workloads(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    measurements_input: MeasurementsData | Mapping[str, object] | Path | str | None,
) -> tuple[list[RegimeEntry], list[AdvisorReport]]:
    """Evaluate workloads across all regimes and assemble report entries."""
    entries: list[RegimeEntry] = []
    reports: list[AdvisorReport] = []
    for label, workload in workloads:
        advisor_report = advise(
            profile, workload, chip_spec, measurements=measurements_input
        )
        reports.append(advisor_report)
        recommended_scope = next(
            scope_est
            for scope_est in advisor_report.scopes
            if scope_est.scope == advisor_report.recommended_scope
        )
        workload_dict = asdict(
            replace(
                workload,
                context=advisor_report.effective_context,
                tiles=None,
                execution=workload.effective_execution,
                host_table_hbm_overrides=workload.effective_host_table_hbm_overrides,
            )
        )
        workload_dict["layer_groups"] = list(workload.layer_groups)
        entries.append(
            RegimeEntry(
                label=label,
                phase=workload.phase,
                active_tokens=advisor_report.active_tokens,
                effective_context=advisor_report.effective_context,
                step_arithmetic_intensity=advisor_report.step_arithmetic_intensity,
                max_vmem_tokens=recommended_scope.max_vmem_tokens,
                num_vmem_splits=recommended_scope.num_vmem_splits,
                recommended_scope=advisor_report.recommended_scope,
                recommended_moe_mode=advisor_report.recommended_moe_mode,
                step_bound_us=recommended_scope.step_bound_us,
                rationale=advisor_report.rationale,
                layout=format_parallel_layout_label(workload),
                step_upper_us=recommended_scope.step_upper_us,
                verdict=advisor_report.verdict,
                confidence=advisor_report.confidence,
                hbm_fits=advisor_report.hbm_fits,
                hbm_pool_overflow=advisor_report.hbm_pool_overflow,
                flags=advisor_report.flags,
                metrics=advisor_report.metrics,
                unresolved_assumptions=advisor_report.unresolved_assumptions,
                scopes=tuple(asdict(scope_est) for scope_est in advisor_report.scopes),
                workload=workload_dict,
                fastest_scope=advisor_report.fastest_scope,
                fastest_verdict=advisor_report.fastest_verdict,
                evidence=advisor_report.evidence,
                step_achievable_us=recommended_scope.step_achievable_us,
                step_achievable_upper_us=recommended_scope.step_achievable_upper_us,
                step_floor_strict_us=recommended_scope.step_floor_strict_us,
                blockers=advisor_report.core_blockers,
                gain_basis=advisor_report.gain_basis,
                ranking_metric=advisor_report.ranking_metric,
                binding_unit=advisor_report.binding_unit,
                core_support=advisor_report.core_support,
                core_blockers=advisor_report.core_blockers,
                core_ready=advisor_report.core_ready,
                core_ready_scope=advisor_report.core_ready_scope,
            )
        )
    return entries, reports


def _select_vmem_reference_scope(decode_report: AdvisorReport) -> ScopeEstimate:
    """Return deepest feasible scope among whole_step, layer_group, single_layer."""
    scopes = {scope_est.scope: scope_est for scope_est in decode_report.scopes}
    if scopes["whole_step"].feasible:
        return scopes["whole_step"]
    if scopes["layer_group"].feasible:
        return scopes["layer_group"]
    return scopes["single_layer"]


def _build_tailoring_guide(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    decode_report: AdvisorReport,
    vmem_parts: Mapping[str, int],
    base: WorkloadSpec,
) -> tuple[str, ...]:
    """Assemble the three-regime tailoring guidance rules for the sweep."""
    whole_step_scope = next(
        scope_est
        for scope_est in decode_report.scopes
        if scope_est.scope == "whole_step"
    )
    single_layer_scope = next(
        scope_est
        for scope_est in decode_report.scopes
        if scope_est.scope == "single_layer"
    )
    reference_scope = _select_vmem_reference_scope(decode_report)
    max_tokens, ridge_tokens = (
        reference_scope.max_vmem_tokens,
        decode_report.ridge_tokens,
    )
    per_token_kib = vmem_parts["total_per_token_bytes"] / 1024
    core_limits_note = (
        " (currently outside_core_limits)"
        if whole_step_scope.support != "within_core_limits"
        else ""
    )
    if profile.num_experts > 0:
        ep_per_dp_group = max(1, base.ep // gcd(base.ep, effective_tp(base)))
        expert_dtype = profile.expert_dtype or profile.weight_dtype
        row_tile_threshold = max(
            1, ceil(chip_spec.matmul_row_tile(expert_dtype) / ep_per_dp_group)
        )
        low_batch_moe_hint = (
            f"moe_mode='masked_on_chip' (T < {row_tile_threshold}) or "
            f"'grouped_on_chip' (T >= {row_tile_threshold})"
        )
        saturated_moe_hint = (
            "moe_mode='grouped_in_hbm': tokens are grouped by expert in HBM"
        )
    else:
        low_batch_moe_hint = saturated_moe_hint = "moe_mode='dense_mlp'"
    mid_regime_band = (
        f"({max_tokens} < T <= {ridge_tokens} tokens)"
        if max_tokens < ridge_tokens
        else (
            f"(T > {ridge_tokens} tokens, single-layer ceiling "
            f"max_vmem_tokens={single_layer_scope.max_vmem_tokens})"
        )
    )
    ridge_intensity = decode_report.ridge_dense_flop_per_byte
    mid_scope_hint = (
        "scope='single_layer'"
        if reference_scope.scope == "layer_group"
        else "scope='single_layer' or 'layer_group'"
    )
    return (
        f"Small/Med Decode & Spec Verify (T <= {max_tokens} active tokens): use "
        f"scope='{reference_scope.scope}'{core_limits_note} with {low_batch_moe_hint} "
        f"to keep [T, H] in VMEM ({reference_scope.vmem_static_mib:.1f} MiB "
        f"{base.dma_buffers}-slot weight staging + {per_token_kib:.1f} KiB/tok). "
        f"The megakernel core's default buckets stop at "
        f"{CORE_LIMITS.default_decode_bucket_top} decode rows and "
        f"{CORE_LIMITS.default_verify_bucket_top} verify rows; "
        "a larger call needs a custom bucket list.",
        f"High-Batch Decode & Chunked Prefill {mid_regime_band}: use "
        f"{mid_scope_hint} (single-layer VMEM ceiling "
        f"max_vmem_tokens={single_layer_scope.max_vmem_tokens}) so activations fit "
        "a single pass without multi-layer weight staging.",
        f"Saturated Decode & Compute-Bound Prefill (T > {ridge_tokens} tokens, "
        f"Arithmetic Intensity >= {ridge_intensity:.0f} FLOP/B): use scope='per_op' "
        f"with {saturated_moe_hint} and each operator is its own call, so no VMEM "
        "is held for multi-layer weight staging.",
    )


def _sublayer_labels(
    layouts: Mapping[str, ParallelismRecommendation | None],
) -> dict[str, str]:
    """Build pure physics attention/MoE sublayer labels per active role."""
    return {
        role: ParallelLayout(
            tp=recommendation.tp,
            ep=recommendation.ep,
            dp=recommendation.dp,
            pp=recommendation.pp,
            dcp=recommendation.dcp,
            pcp=recommendation.pcp,
        ).submodule_label()
        for role, recommendation in layouts.items()
        if recommendation is not None
    }


def _resolve_requested_layout(
    sweep_options: RegimeSweepOptions, phase: Phase
) -> ParallelismSearchOptions:
    """Extract requested 6-axis mesh options from `RegimeSweepOptions`."""
    layout_source = (
        sweep_options.layout if sweep_options.layout is not None else sweep_options
    )
    return ParallelismSearchOptions(
        tp=layout_source.tp,
        ep=layout_source.ep,
        dp=layout_source.dp,
        pp=layout_source.pp,
        dcp=0 if layout_source.dcp is None else layout_source.dcp,
        pcp=layout_source.pcp,
        phase=phase,
    )


def advise_regimes(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str = DEFAULT_CHIP,
    *,
    options: RegimeSweepOptions | None = None,
    measurements: MeasurementsData | Mapping[str, object] | Path | str | None = None,
) -> AdvisorSweepReport:
    """Evaluate the Batch Size x Phase & Context matrix across a model's layouts.

    Returns:
        Populated `AdvisorSweepReport` covering all evaluated request regimes.
    """
    sweep_options = options or RegimeSweepOptions()
    measurements_input = (
        measurements if measurements is not None else sweep_options.measurements
    )
    base_workload = sweep_options.base_workload or WorkloadSpec()
    model_profile = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    requested_layout = _resolve_requested_layout(sweep_options, base_workload.phase)
    explicit = sweep_options.layout is not None or (
        requested_layout != ParallelismSearchOptions(dcp=0, phase=base_workload.phase)
    )
    preset = (
        DEPLOYMENT_PRESETS.get(model_profile.name)
        if sweep_options.deployment_preset and not explicit
        else None
    )
    selected_layouts = _select_layouts(
        model_profile, chip_spec, base_workload, requested_layout, preset
    )
    measurements_data = (
        load_measurements(measurements_input, chip_name=chip_spec.name)
        if measurements_input
        else None
    )
    base_workload = replace(
        base_workload.with_layout(selected_layouts.decode), batch_explicit=True
    )
    spec_tokens = (
        preset.decode.spec_tokens
        if preset is not None and preset.decode.spec_tokens
        else DEFAULT_SPEC_TOKENS
    )
    workloads = _with_preset_prefill(
        _build_reference_workloads(base_workload, spec_tokens), preset
    )
    entries, reports = _evaluate_sweep_workloads(
        model_profile, chip_spec, workloads, measurements_input
    )
    decode_report = reports[DECODE_MED_REF_INDEX]
    vmem_parts = compute_per_token_vmem_bytes(
        model_profile, workloads[DECODE_MED_REF_INDEX][1]
    )
    reference_scope = _select_vmem_reference_scope(decode_report)
    return AdvisorSweepReport(
        model=model_profile.name,
        chip=chip_spec.name,
        parallelism=selected_layouts.decode,
        vmem_budget_mib=decode_report.vmem_budget_mib,
        vmem_static_mib=reference_scope.vmem_static_mib,
        per_token_vmem_bytes=vmem_parts["total_per_token_bytes"],
        vmem_breakdown_bytes=vmem_parts,
        max_vmem_tokens=reference_scope.max_vmem_tokens,
        ridge_dense_flop_per_byte=decode_report.ridge_dense_flop_per_byte,
        ridge_tokens=decode_report.ridge_tokens,
        regimes=tuple(entries),
        tailoring_guide=_build_tailoring_guide(
            model_profile, chip_spec, decode_report, vmem_parts, base_workload
        ),
        dma_buffers=base_workload.dma_buffers,
        vmem_reference_scope=reference_scope.scope,
        prefill_parallelism=selected_layouts.prefill,
        saturation_parallelism=selected_layouts.saturation,
        disagg_handoff=_compute_disagg_handoff(
            model_profile, chip_spec, preset, measurements_data
        ),
        chip_sources=chip_spec.sources if chip_spec.sources is not None else {},
        model_sources=_build_model_sources(model_profile),
        chip_spec=chip_spec,
        layer_group_search=_sweep_layer_group_search(
            model_profile,
            chip_spec,
            preset,
            selected_layouts,
            (base_workload, measurements_input),
        ),
        model_profile=model_profile,
        submodule_labels=_sublayer_labels(
            {
                "decode": selected_layouts.decode,
                "prefill": selected_layouts.prefill,
                "saturation": selected_layouts.saturation,
            }
        ),
    )


def _build_model_sources(profile: ModelProfile) -> dict[str, str]:
    """Extract provenance sources for model architecture facts."""
    base_source = MODEL_PRESET_SOURCES.get(
        profile.name, f"Model config: {profile.name}"
    )
    dtype_source = profile.dtype_source or "preset"
    act_clause = f", act={profile.act_dtype}" if profile.act_dtype else ""
    state_clause = (
        f", state={profile.state_dtype}" if profile.num_linear_layers > 0 else ""
    )
    block_clause = (
        f", block_k={profile.weight_block_k}" if profile.weight_block_k > 0 else ""
    )
    source_tag = (
        f"[source: {dtype_source} - WARNING: no quantization_config block; "
        "using default dtype]"
        if dtype_source.startswith("default")
        else f"(source: {dtype_source})"
    )
    dtype_summary = (
        f"weight={profile.weight_dtype}, expert={profile.expert_dtype}"
        f"{act_clause}, kv={profile.kv_dtype}{state_clause}{block_clause} "
        f"{source_tag}"
    )
    sources = {
        "config": base_source,
        "num_layers": f"{profile.num_layers} layers ({base_source})",
        "hidden_size": f"hidden={profile.hidden_size} ({base_source})",
        "attention_heads": (
            f"attn_heads={profile.num_attention_heads}, "
            f"kv_heads={profile.num_kv_heads}, head_dim={profile.head_dim}"
        ),
        "dtypes": dtype_summary,
    }
    if profile.num_experts > 0:
        sources["moe"] = (
            f"experts={profile.num_experts}, top_k={profile.top_k}, "
            f"moe_intermediate={profile.moe_intermediate_size} ({base_source})"
        )
    return sources


def _sweep_layer_group_search(
    model_profile: ModelProfile,
    chip_spec: ChipSpec,
    layout_preset: DeploymentPreset | None,
    selected: _SelectedLayouts,
    workload_and_measurements: tuple[
        WorkloadSpec, MeasurementsData | Mapping[str, object] | Path | str | None
    ],
) -> LayerGroupSearchReport:
    """Evaluate `search_layer_groups` on the sweep's prefill role (or decode role)."""
    base_workload, measurements = workload_and_measurements
    prefill_layout, decode_layout = selected.prefill, selected.decode
    if prefill_layout is not None:
        window_tokens = (
            layout_preset.prefill_windows[0]
            if layout_preset is not None and layout_preset.prefill_windows
            else DEFAULT_PREFILL_ISL
        )
        search_workload = replace(
            base_workload.with_layout(prefill_layout),
            phase="prefill",
            batch=align_batch_to_dp(1, prefill_layout.dp),
            isl=window_tokens,
            osl=DEFAULT_PREFILL_OSL,
            context=window_tokens,
            layer_groups=prefill_layout.layer_groups,
        )
    else:
        search_workload = replace(
            base_workload.with_layout(decode_layout),
            phase="decode",
            batch=align_batch_to_dp(max(1, base_workload.batch), decode_layout.dp),
        )
    return search_layer_groups(
        model_profile, search_workload, chip_spec, measurements=measurements
    )
