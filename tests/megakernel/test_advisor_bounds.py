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
"""First-principles physics, VMEM, collective, and core-sync oracle tests."""

from __future__ import annotations

import ast
import dataclasses
import functools
import itertools
import json
import math
import pkgutil
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.megakernel.advisor_test_helpers import minimal_valid_core_layout
from tests.megakernel.core.megakernel_fakes import TINY_TILES
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    roofline,
    spec_bridge,
    specs,
    vmem,
)
from vllm_torchtpu.megakernel.core import exec_spec as core_exec_spec
from vllm_torchtpu.megakernel.core import modules as core_modules
from vllm_torchtpu.megakernel.core.dma import vmem as core_vmem
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import Tiles
from vllm_torchtpu.megakernel.core.profiles import default as default_profiles
from vllm_torchtpu.megakernel.core.runtime import policy as core_runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_ADVISOR_DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
_REPORT_ROUND_TOL_US = 0.02
_UNBOUNDED_RATE = 1e30
_FUSED_SCOPES = frozenset({"whole_step", "layer_group"})
_MIXED_PREFILL_PHASES = frozenset({"prefill", "mixed"})
CASE_PHASES: tuple[specs.Phase, ...] = ("decode", "prefill", "verify", "mixed")
# 2-decimal rounding of reported microsecond fields (round(..., 2) in roofline.py).
ROUND_HALF_UNIT_US: float = 0.005
# 4-decimal rounding of step_weight_gb in roofline.py.
ROUND_HALF_UNIT_GB: float = 0.00005


def _scope_by_name(
    scopes: tuple[advisor.ScopeEstimate, ...],
    name: advisor.FusionScope,
) -> advisor.ScopeEstimate:
    """Return ScopeEstimate matching `name`.

    Returns:
        Matching `ScopeEstimate` entry.
    """
    return next(est for est in scopes if est.scope == name)


def _out_of_scope_params(
    slug: str, profile: advisor.ModelProfile, raw_config: dict[str, object]
) -> int:
    """Compute HuggingFace non-decoder parameter counts excluded from decoder scope.

    Returns:
        Total parameter count in embeddings, LM head, vision tower, and MTP layers.
    """
    cfg = raw_config.get("config", raw_config)
    assert isinstance(cfg, dict)
    text_config = cfg.get("text_config", cfg)
    assert isinstance(text_config, dict)
    vocab_size = int(text_config["vocab_size"])
    tied = bool(
        cfg.get("tie_word_embeddings", text_config.get("tie_word_embeddings", False))
    )
    extra = vocab_size * profile.hidden_size * (1 if tied else 2)
    vc = cfg.get("vision_config")
    if isinstance(vc, dict) and str(vc.get("model_type", "")) in (
        "qwen3_5_moe",
        "qwen4_exp",
    ):
        vh = int(vc["hidden_size"])
        vi = int(vc["intermediate_size"])
        vd = int(vc["depth"])
        in_ch = int(vc.get("in_channels", 3))
        pt = int(vc.get("temporal_patch_size", 2))
        ps = int(vc.get("patch_size", 16))
        sm = int(vc.get("spatial_merge_size", 2))
        oh = int(vc.get("out_hidden_size", profile.hidden_size))
        m_in = vh * sm * sm
        extra += (
            vd * (4 * vh * vh + 3 * vh * vi)
            + vh * in_ch * pt * ps * ps
            + m_in * m_in
            + m_in * oh
        )
    ple_ids = text_config.get("ple_layer_ids")
    ngram_base = int(text_config.get("ngram_vocab_size_base") or 0)
    if isinstance(ple_ids, list) and ple_ids and ngram_base > 0:
        div = max(1, int(text_config.get("make_ngram_vocab_size_divisible_by") or 1))
        ple_vocab = ((ngram_base + div - 1) // div) * div
        ple_dim = int(text_config.get("ple_embed_dim") or profile.hidden_size)
        extra += ple_vocab * ple_dim * len(ple_ids)
    mtp_layers = int(
        text_config.get("mtp_num_hidden_layers")
        or text_config.get("num_nextn_predict_layers")
        or 0
    )
    if mtp_layers:
        per_layer_dec = (
            roofline.compute_decoder_param_count(profile) / profile.num_layers
        )
        extra += round(
            mtp_layers * (per_layer_dec + 2 * profile.hidden_size * profile.hidden_size)
        )
    return extra


def test_hub_safetensors_param_closure_oracle() -> None:
    """Verify decoder params plus out-of-scope heads close Hub totals within 0.1%."""
    hub_data = json.loads(
        (_ADVISOR_DATA_DIR / "hub_param_totals.json").read_text(encoding="utf-8")
    )
    max_rel_err = 0.001
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        raw_config = json.loads(
            (_ADVISOR_DATA_DIR / "hf_configs" / f"{slug}.json").read_text(
                encoding="utf-8"
            )
        )
        decoder_params = roofline.compute_decoder_param_count(profile)
        predicted_total = decoder_params + _out_of_scope_params(
            slug, profile, raw_config
        )
        safetensors_total = int(hub_data["models"][slug]["safetensors_total"])
        rel_err = abs(predicted_total - safetensors_total) / safetensors_total
        assert rel_err < max_rel_err, (
            f"{slug}: decoder={decoder_params}, total={predicted_total}, "
            f"hub={safetensors_total}, rel_err={rel_err:.6f}"
        )


def _single_layer_profile(
    family: str,
) -> tuple[advisor.ModelProfile, advisor.WorkloadSpec]:
    """Build a 1-layer synthetic profile and decode/verify workload per layer family.

    Returns:
        `(ModelProfile, WorkloadSpec)` pair for `family`.
    """
    eff_unit = advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0)
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=1024,
        tp=1,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=eff_unit,
    )
    if family == "gqa":
        prof = models.finalize_model_profile(
            advisor.ModelProfile(
                name=f"oracle-{family}",
                num_layers=1,
                hidden_size=1024,
                num_attention_heads=8,
                num_kv_heads=4,
                head_dim=128,
                num_attention_layers=1,
                num_moe_layers=0,
                has_attn_output_gate=True,
                weight_dtype="bf16",
                expert_dtype="bf16",
                kv_dtype="bf16",
                state_dtype="fp32",
                act_dtype="bf16",
            )
        )
        return prof, wl
    return _single_layer_non_gqa(family, wl)


def _single_layer_non_gqa(
    family: str,
    wl: advisor.WorkloadSpec,
) -> tuple[advisor.ModelProfile, advisor.WorkloadSpec]:
    """Build non-GQA 1-layer profiles (`mla`, `gdn`, `kda`, `dense_mlp`, `moe`, `dsa`).

    Returns:
        `(ModelProfile, WorkloadSpec)` pair.
    """
    prof_name = f"oracle-{family}"
    if family == "mla":
        prof = models.finalize_model_profile(
            advisor.ModelProfile(
                name=prof_name,
                num_layers=1,
                hidden_size=1024,
                num_attention_heads=8,
                num_kv_heads=4,
                head_dim=128,
                num_attention_layers=1,
                num_moe_layers=0,
                kv_lora_rank=256,
                q_lora_rank=256,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                has_attn_output_gate=True,
                weight_dtype="bf16",
                expert_dtype="bf16",
                kv_dtype="bf16",
                state_dtype="fp32",
                act_dtype="bf16",
            )
        )
        return prof, wl
    if family in {"gdn", "kda"}:
        mtype = "kimi_linear" if family == "kda" else "qwen3_5_moe_text"
        prof = models.finalize_model_profile(
            advisor.ModelProfile(
                name=prof_name,
                num_layers=1,
                hidden_size=1024,
                num_attention_heads=8,
                num_kv_heads=4,
                head_dim=128,
                model_type=mtype,
                num_attention_layers=0,
                num_linear_layers=1,
                linear_key_heads=16,
                linear_value_heads=16,
                linear_key_dim=128,
                linear_value_dim=128,
                linear_conv_kernel=4,
                num_moe_layers=0,
                weight_dtype="bf16",
                expert_dtype="bf16",
                kv_dtype="bf16",
                state_dtype="fp32",
                act_dtype="bf16",
            )
        )
        return prof, dataclasses.replace(wl, batch=64)
    if family == "dense_mlp":
        prof = dataclasses.replace(
            models.finalize_model_profile(
                advisor.ModelProfile(
                    name=prof_name,
                    num_layers=1,
                    hidden_size=1024,
                    intermediate_size=2048,
                    num_attention_heads=8,
                    num_kv_heads=4,
                    head_dim=128,
                    num_attention_layers=1,
                    num_moe_layers=0,
                    layer_families=("dense_mlp",),
                    weight_dtype="bf16",
                    expert_dtype="bf16",
                    kv_dtype="bf16",
                    state_dtype="fp32",
                    act_dtype="bf16",
                )
            ),
            num_attention_layers=0,
        )
        return prof, wl
    if family == "moe":
        prof = dataclasses.replace(
            models.finalize_model_profile(
                advisor.ModelProfile(
                    name=prof_name,
                    num_layers=1,
                    hidden_size=1024,
                    num_attention_heads=8,
                    num_kv_heads=4,
                    head_dim=128,
                    num_attention_layers=1,
                    num_moe_layers=1,
                    num_experts=4,
                    top_k=1,
                    moe_intermediate_size=1024,
                    shared_expert_intermediate_size=512,
                    layer_families=("moe",),
                    weight_dtype="bf16",
                    expert_dtype="bf16",
                    kv_dtype="bf16",
                    state_dtype="fp32",
                    act_dtype="bf16",
                )
            ),
            num_attention_layers=0,
        )
        return prof, dataclasses.replace(wl, ep=1)
    if family == "dsa":
        prof = models.finalize_model_profile(
            advisor.ModelProfile(
                name=prof_name,
                num_layers=1,
                hidden_size=1024,
                num_attention_heads=8,
                num_kv_heads=4,
                head_dim=128,
                num_attention_layers=1,
                num_moe_layers=0,
                kv_lora_rank=256,
                q_lora_rank=256,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                index_topk=128,
                index_head_dim=64,
                index_n_heads=8,
                num_indexer_layers=1,
                weight_dtype="bf16",
                expert_dtype="bf16",
                kv_dtype="bf16",
                state_dtype="fp32",
                act_dtype="bf16",
            )
        )
        return prof, dataclasses.replace(wl, context=2048, tp=2, dcp=2)
    msg = f"Unknown family: {family}"
    raise ValueError(msg)


def _independent_family_weight_gb(
    family: str,
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
) -> float:
    """Compute 1-layer step weight GB strictly from projection dimensions."""
    hidden = profile.hidden_size
    tp_ranks = max(1, workload.tp)
    weight_bytes = constants.DTYPE_BYTES[profile.weight_dtype.lower()]
    if family == "gqa":
        query_dim = (profile.num_attention_heads // tp_ranks) * profile.head_dim
        kv_dim = (profile.num_kv_heads // tp_ranks) * profile.head_dim
        gate_dim = query_dim if profile.has_attn_output_gate else 0
        total_bytes = hidden * (2 * query_dim + 2 * kv_dim + gate_dim) * weight_bytes
        return total_bytes / 1e9
    if family in {"mla", "dsa"}:
        local_attn_heads = profile.num_attention_heads // tp_ranks
        q_lora, kv_lora = profile.q_lora_rank, profile.kv_lora_rank
        qk_nope, qk_rope = profile.qk_nope_head_dim, profile.qk_rope_head_dim
        q_proj = hidden * q_lora + q_lora * local_attn_heads * (qk_nope + qk_rope)
        kv_proj = hidden * (kv_lora + qk_rope) + kv_lora * local_attn_heads * (
            qk_nope + profile.head_dim
        )
        out_proj = local_attn_heads * profile.head_dim * hidden
        gate_proj = (
            hidden * local_attn_heads * profile.head_dim
            if profile.has_attn_output_gate
            else 0
        )
        total_bytes = (q_proj + kv_proj + out_proj + gate_proj) * weight_bytes
        if family == "dsa":
            idx_query = q_lora * profile.index_n_heads * profile.index_head_dim
            idx_key_weight = hidden * (profile.index_head_dim + profile.index_n_heads)
            total_bytes += (
                idx_query * weight_bytes + idx_key_weight * constants.BF16_BYTES
            )
        return total_bytes / 1e9
    if family in {"gdn", "kda"}:
        local_key_heads = profile.linear_key_heads // tp_ranks
        local_value_heads = profile.linear_value_heads // tp_ranks
        value_dim = local_value_heads * profile.linear_value_dim
        in_proj_dim = 2 * local_key_heads * profile.linear_key_dim + 2 * value_dim
        return hidden * (in_proj_dim + value_dim) * weight_bytes / 1e9
    if family == "dense_mlp":
        return (
            3.0 * hidden * (profile.intermediate_size // tp_ranks) * weight_bytes / 1e9
        )
    expert_bytes = constants.DTYPE_BYTES[profile.expert_dtype.lower()]
    shared_bytes = (
        3.0
        * hidden
        * (profile.shared_expert_intermediate_size // tp_ranks)
        * weight_bytes
    )
    router_bytes = hidden * profile.num_experts * constants.BF16_BYTES
    hit_prob = 1.0 - (1.0 - profile.top_k / profile.num_experts) ** workload.batch
    active_experts = profile.num_experts * hit_prob
    routed_bytes = (
        active_experts
        * 3.0
        * hidden
        * (profile.moe_intermediate_size // tp_ranks)
        * expert_bytes
    )
    return (shared_bytes + router_bytes + routed_bytes) / 1e9


@pytest.mark.parametrize(
    "family",
    ["gqa", "mla", "gdn", "kda", "dense_mlp", "moe", "dsa"],
)
def test_single_layer_physics_oracle_per_family(family: str) -> None:
    """Verify 1-layer per-family weight bounds, HBM floor, and scaling laws."""
    chip = advisor.resolve_chip_spec("tpu7x")
    base_profile, base_workload = _single_layer_profile(family)
    workload = dataclasses.replace(base_workload, batch=64)
    per_op = _scope_by_name(
        advisor.evaluate_scopes(base_profile, workload, chip), "per_op"
    )

    expected_weight_gb = _independent_family_weight_gb(family, base_profile, workload)
    assert per_op.step_weight_gb == pytest.approx(expected_weight_gb, abs=1e-4)
    min_weight_hbm_us = (per_op.step_weight_gb * 1e9 / chip.hbm_bw_bytes_per_s) * 1e6
    assert per_op.hbm_us >= min_weight_hbm_us - _REPORT_ROUND_TOL_US
    assert per_op.compute_us >= 2.0 * _REPORT_ROUND_TOL_US

    two_layer_profile = dataclasses.replace(
        base_profile,
        num_layers=2 * base_profile.num_layers,
        num_attention_layers=2 * base_profile.num_attention_layers,
        num_linear_layers=2 * base_profile.num_linear_layers,
        num_moe_layers=2 * base_profile.num_moe_layers,
        num_indexer_layers=2 * base_profile.num_indexer_layers,
    )
    two_layer = _scope_by_name(
        advisor.evaluate_scopes(two_layer_profile, workload, chip), "per_op"
    )
    assert two_layer.step_weight_gb == pytest.approx(
        2.0 * per_op.step_weight_gb, abs=2e-4
    )
    assert two_layer.hbm_us == pytest.approx(
        2.0 * per_op.hbm_us, abs=2.0 * _REPORT_ROUND_TOL_US
    )
    assert two_layer.compute_us == pytest.approx(
        2.0 * per_op.compute_us, abs=2.0 * _REPORT_ROUND_TOL_US
    )

    double_batch = _scope_by_name(
        advisor.evaluate_scopes(
            base_profile, dataclasses.replace(workload, batch=2 * workload.batch), chip
        ),
        "per_op",
    )
    assert double_batch.compute_us > per_op.compute_us + _REPORT_ROUND_TOL_US
    if family == "moe":
        dense_moe_profile = dataclasses.replace(
            base_profile, top_k=base_profile.num_experts
        )
        dense_moe_b1 = _scope_by_name(
            advisor.evaluate_scopes(dense_moe_profile, workload, chip), "per_op"
        )
        dense_moe_b2 = _scope_by_name(
            advisor.evaluate_scopes(
                dense_moe_profile,
                dataclasses.replace(workload, batch=2 * workload.batch),
                chip,
            ),
            "per_op",
        )
        assert dense_moe_b2.compute_us == pytest.approx(
            2.0 * dense_moe_b1.compute_us, abs=2.0 * _REPORT_ROUND_TOL_US
        )
    else:
        assert double_batch.compute_us == pytest.approx(
            2.0 * per_op.compute_us, abs=2.0 * _REPORT_ROUND_TOL_US
        )

    double_context = _scope_by_name(
        advisor.evaluate_scopes(
            base_profile,
            dataclasses.replace(workload, context=2 * workload.context),
            chip,
        ),
        "per_op",
    )
    if family in {"gqa", "mla", "dsa"}:
        assert double_context.hbm_us > per_op.hbm_us + _REPORT_ROUND_TOL_US
        assert double_context.compute_us > per_op.compute_us + _REPORT_ROUND_TOL_US
    else:
        assert abs(double_context.hbm_us - per_op.hbm_us) <= _REPORT_ROUND_TOL_US
        assert (
            abs(double_context.compute_us - per_op.compute_us) <= _REPORT_ROUND_TOL_US
        )


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
@pytest.mark.parametrize("phase", ["decode", "prefill", "verify", "mixed"])
def test_shipped_presets_all_scopes_and_phases_satisfy_physics_invariants(
    model_slug: str,
    phase: advisor.Phase,
) -> None:
    """Verify band order, floor subterms, and legality across presets and phases."""
    profile = advisor.resolve_model_profile(model_slug)
    preset = advisor.DEPLOYMENT_PRESETS[model_slug]
    layout = preset.prefill if phase in _MIXED_PREFILL_PHASES else preset.decode
    spec_tokens = preset.decode.spec_tokens or 3 if phase in ("verify", "mixed") else 0
    chunk_size = 1024 if phase in _MIXED_PREFILL_PHASES else 0
    workload = advisor.WorkloadSpec(
        phase=phase,
        batch=16,
        isl=4096,
        chunk_size=chunk_size,
        spec_tokens=spec_tokens,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    scopes = advisor.evaluate_scopes(profile, workload, "tpu7x")
    expected_scopes = ("per_op", "single_layer", "layer_group", "whole_step")
    assert tuple(est.scope for est in scopes) == expected_scopes
    for est in scopes:
        assert est.step_floor_strict_us <= est.step_bound_us + _REPORT_ROUND_TOL_US
        assert est.step_bound_us <= est.step_achievable_us + _REPORT_ROUND_TOL_US
        assert est.step_bound_us <= est.step_upper_us + _REPORT_ROUND_TOL_US
        assert (
            est.step_achievable_us
            <= est.step_achievable_upper_us + _REPORT_ROUND_TOL_US
        )
        assert est.step_upper_us <= est.step_achievable_upper_us + _REPORT_ROUND_TOL_US
        overlapped = est.scope in _FUSED_SCOPES and est.num_vmem_splits == 1
        core_us = (
            max(est.hbm_us, est.chain_us + est.comm_us)
            if overlapped
            else max(est.hbm_us, est.chain_us) + est.comm_us
        )
        assert core_us + est.fill_drain_us == pytest.approx(
            est.roof_floor_us, abs=_REPORT_ROUND_TOL_US
        )
        assert est.step_weight_gb > 0.0
        if est.scope == "whole_step":
            if phase not in constants.FUSED_ELIGIBLE_PHASES:
                assert "phase_not_in_core" in est.blockers
            if profile.num_linear_layers > 0 and profile.state_dtype.lower() != "bf16":
                assert "kv_or_state_dtype_not_in_core" in est.blockers
            if models.has_dsa_profile(profile):
                assert "family_not_in_core" in est.blockers


def test_kimi_k3_latent_proj_and_mla_gate_bytes() -> None:
    """Verify Kimi-K3 latent projections (TP2 default) and MLA output gate bytes."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    chip = advisor.resolve_chip_spec("tpu7x")
    tp_ranks = 32
    default_latent_tp = 2
    alt_latent_tp = 4
    wl = advisor.WorkloadSpec(
        phase="verify",
        batch=1,
        spec_tokens=7,
        context=4096,
        tp=tp_ranks,
        ep=tp_ranks,
        dcp=tp_ranks,
        vmem_fit_headroom=True,
    )
    k3_no_gate = dataclasses.replace(k3, has_attn_output_gate=False)
    wl_tp1 = dataclasses.replace(wl, latent_proj_tp=1)
    wl_tp4 = dataclasses.replace(wl, latent_proj_tp=alt_latent_tp)

    w_full = _scope_by_name(
        advisor.evaluate_scopes(k3, wl, chip), "whole_step"
    ).step_weight_gb
    w_no_gate = _scope_by_name(
        advisor.evaluate_scopes(k3_no_gate, wl, chip), "whole_step"
    ).step_weight_gb
    w_tp1 = _scope_by_name(
        advisor.evaluate_scopes(k3, wl_tp1, chip), "whole_step"
    ).step_weight_gb
    w_tp4 = _scope_by_name(
        advisor.evaluate_scopes(k3, wl_tp4, chip), "whole_step"
    ).step_weight_gb

    expected_gate_gb = (
        k3.num_attention_layers
        * k3.hidden_size
        * (k3.num_attention_heads // tp_ranks)
        * k3.head_dim
        * constants.BF16_BYTES
    ) / 1e9
    expected_latent_tp2_gb = (
        k3.num_moe_layers
        * 2
        * k3.hidden_size
        * k3.routed_expert_hidden_size
        * constants.BF16_BYTES
        / default_latent_tp
    ) / 1e9
    delta_tol_gb = 1e-4
    assert w_full - w_no_gate == pytest.approx(expected_gate_gb, abs=delta_tol_gb)
    assert w_tp1 - w_full == pytest.approx(expected_latent_tp2_gb, abs=delta_tol_gb)
    assert w_full - w_tp4 == pytest.approx(
        0.5 * expected_latent_tp2_gb, abs=delta_tol_gb
    )


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
def test_attention_flops_conserved_under_dcp_while_kv_bytes_shrink(
    model_slug: str,
) -> None:
    """Verify DCP conserves global attention FLOPs while shrinking KV HBM reads."""
    profile = advisor.resolve_model_profile(model_slug)
    chip = advisor.resolve_chip_spec("tpu7x")
    layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    tp = max(layout.tp, 8)
    ep = max(1, layout.ep)
    fast_io = dataclasses.replace(
        chip,
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
    )
    base_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=4,
        spec_tokens=3,
        tp=tp,
        ep=ep,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )

    def _eval_per_op(
        target_chip: advisor.ChipSpec, dcp: int, context: int
    ) -> advisor.ScopeEstimate:
        """Evaluate per_op scope for given DCP degree and context length."""
        return _scope_by_name(
            advisor.evaluate_scopes(
                profile,
                dataclasses.replace(base_wl, dcp=dcp, context=context),
                target_chip,
            ),
            "per_op",
        )

    sp = profile.sparse_attn
    if sp is not None and sp.kind == "csa":
        for dcp in (4, 8):
            with pytest.raises(MegakernelConfigError):
                _eval_per_op(chip, dcp, 32768)
        return

    cap_tok = (
        (sp.window_tokens + sp.select_tokens) if sp is not None else profile.index_topk
    )
    has_idx = models.has_dsa_profile(profile) or (sp is not None and sp.index_dim > 0)

    def _attn_compute_us(dcp: int) -> float:
        """Isolate attention matmul compute microseconds across context delta."""
        hi_ctx = 32768
        if has_idx and cap_tok > 0:
            c_lo = _eval_per_op(fast_io, dcp, 1).compute_us
            c_mid = _eval_per_op(fast_io, dcp, cap_tok).compute_us
            c_hi = _eval_per_op(fast_io, dcp, hi_ctx).compute_us
            idx_slope = (c_hi - c_mid) / (hi_ctx - cap_tok)
            return (c_mid - c_lo) - (cap_tok - 1) * idx_slope
        return (
            _eval_per_op(fast_io, dcp, hi_ctx).compute_us
            - _eval_per_op(fast_io, dcp, 1).compute_us
        )

    def _kv_hbm_delta_us(dcp: int) -> float:
        """Isolate KV-gather HBM microseconds across context delta under DCP."""
        hi_ctx = 32768
        if sp is not None and cap_tok > 0 and sp.index_dim > 0:
            e_lo = _eval_per_op(chip, dcp, 1)
            e_mid = _eval_per_op(chip, dcp, cap_tok)
            e_post = _eval_per_op(chip, dcp, cap_tok + 1)
            e_hi = _eval_per_op(chip, dcp, hi_ctx)
            score_act_us = (
                (e_hi.act_hbm_mb - e_post.act_hbm_mb)
                * 1e6
                / chip.hbm_bw_bytes_per_s
                * 1e6
            )
            idx_slope = (e_hi.hbm_us - e_post.hbm_us - score_act_us) / (
                hi_ctx - (cap_tok + 1)
            )
            return (e_mid.hbm_us - e_lo.hbm_us) - (cap_tok - 1) * idx_slope
        return (
            _eval_per_op(chip, dcp, hi_ctx).hbm_us - _eval_per_op(chip, dcp, 1).hbm_us
        )

    round_tol = 4.0 * _REPORT_ROUND_TOL_US
    base_compute_us = _attn_compute_us(1)
    base_hbm_delta_us = _kv_hbm_delta_us(1)
    for dcp in (4, 8):
        assert _attn_compute_us(dcp) == pytest.approx(base_compute_us, abs=round_tol)
        dcp_hbm_delta_us = _kv_hbm_delta_us(dcp)
        if profile.kv_lora_rank > 0:
            expected_kv_ratio = 1.0 / dcp
        elif profile.dcp_span == "tp_group":
            kvh_base = max(1, math.ceil(profile.num_kv_heads / tp))
            kvh_dcp = vmem.wide_dcp_kv_heads_per_rank(
                profile.num_kv_heads, tp, dcp, "tp_group"
            )
            expected_kv_ratio = (kvh_dcp / kvh_base) / dcp
        else:
            kvh_base = max(1, math.ceil(profile.num_kv_heads / tp))
            kvh_dcp = max(1, math.ceil(profile.num_kv_heads / max(1, tp // dcp)))
            expected_kv_ratio = (kvh_dcp / kvh_base) / dcp
        assert dcp_hbm_delta_us == pytest.approx(
            base_hbm_delta_us * expected_kv_ratio, abs=round_tol
        )


def test_tp_scales_weights_and_prefill_compute() -> None:
    """Verify 2x TP halves static weights and uncached dense prefill compute."""
    chip = advisor.resolve_chip_spec("tpu7x")
    sharded_dense = models.finalize_model_profile(
        advisor.ModelProfile(
            name="sharded-dense",
            num_layers=16,
            num_attention_layers=16,
            num_moe_layers=0,
            hidden_size=4096,
            intermediate_size=8192,
            num_attention_heads=32,
            num_kv_heads=16,
            head_dim=128,
            weight_dtype="fp8",
            kv_dtype="fp8",
            act_dtype="fp8",
        )
    )
    dec_wl = advisor.WorkloadSpec(phase="decode", batch=1, context=1, tp=4)
    w_tp4 = _scope_by_name(
        advisor.evaluate_scopes(sharded_dense, dec_wl, chip), "per_op"
    ).step_weight_gb
    w_tp8 = _scope_by_name(
        advisor.evaluate_scopes(sharded_dense, dataclasses.replace(dec_wl, tp=8), chip),
        "per_op",
    ).step_weight_gb
    assert w_tp8 == pytest.approx(0.5 * w_tp4, abs=1e-4)

    compute_chip = dataclasses.replace(
        chip, hbm_bw_bytes_per_s=_UNBOUNDED_RATE, vector_f32_ops_per_s=_UNBOUNDED_RATE
    )
    wl_pre4 = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=4096,
        chunk_size=4096,
        tp=4,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    pre4 = _scope_by_name(
        advisor.evaluate_scopes(sharded_dense, wl_pre4, compute_chip), "per_op"
    )
    pre8 = _scope_by_name(
        advisor.evaluate_scopes(
            sharded_dense, dataclasses.replace(wl_pre4, tp=8), compute_chip
        ),
        "per_op",
    )
    assert pre8.compute_us == pytest.approx(
        0.5 * pre4.compute_us, abs=_REPORT_ROUND_TOL_US
    )


def test_dtype_hbm_scaling_across_all_supported_dtypes() -> None:
    """Verify HBM weight read time scales linearly with every weight dtype size."""
    chip = advisor.resolve_chip_spec("tpu7x")
    base_dense = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dtype-dense",
            num_layers=16,
            num_attention_layers=16,
            num_moe_layers=0,
            hidden_size=4096,
            intermediate_size=8192,
            num_attention_heads=32,
            num_kv_heads=16,
            head_dim=128,
            weight_dtype="fp8",
            kv_dtype="fp8",
            act_dtype="bf16",
        )
    )
    fast_compute = dataclasses.replace(
        chip,
        bf16_flops_per_s=_UNBOUNDED_RATE,
        fp8_flops_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
    )
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=1,
        tp=4,
        dispatch_regime="zero_floor",
        op_overhead_us=0.0,
    )
    act_us = (
        (4.0 * base_dense.hidden_size * constants.BF16_BYTES)
        / chip.hbm_bw_bytes_per_s
        * 1e6
    )
    ref_est = _scope_by_name(
        advisor.evaluate_scopes(base_dense, wl, fast_compute), "whole_step"
    )
    ref_hbm_weights_us = ref_est.hbm_us - act_us
    ref_bytes_per_elem = constants.DTYPE_BYTES["fp8"]
    for dtype_name, bytes_per_elem in sorted(constants.DTYPE_BYTES.items()):
        prof_dt = dataclasses.replace(
            base_dense, weight_dtype=dtype_name, expert_dtype=dtype_name
        )
        if dtype_name in constants.KV_ONLY_DTYPES:
            with pytest.raises(MegakernelConfigError):
                advisor.evaluate_scopes(prof_dt, wl, fast_compute)
            continue
        est = _scope_by_name(
            advisor.evaluate_scopes(prof_dt, wl, fast_compute), "whole_step"
        )
        ratio = bytes_per_elem / ref_bytes_per_elem
        assert est.step_weight_gb == pytest.approx(
            ratio * ref_est.step_weight_gb, abs=1e-4
        )
        assert est.hbm_us - act_us == pytest.approx(
            ratio * ref_hbm_weights_us, abs=2.0 * _REPORT_ROUND_TOL_US
        )


def test_subchannel_vec_moe_bounded_by_predequant_or_unpadded() -> None:
    """Verify Kimi-K3 subchannel MoE vector cost selects Strategy B over Strategy A."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="verify",
        batch=1,
        spec_tokens=7,
        context=4096,
        tp=32,
        ep=32,
        dcp=32,
        vmem_fit_headroom=True,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
    )
    whole = _scope_by_name(advisor.evaluate_scopes(k3, wl, chip), "whole_step")
    assert whole.compute_us < whole.hbm_us
    assert whole.binding_unit != "vector"

    fast_mxu_hbm_chip = dataclasses.replace(
        chip,
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        bf16_flops_per_s=_UNBOUNDED_RATE,
        ici_link_bw_bytes_per_s=_UNBOUNDED_RATE,
        ici_egress_bw_bytes_per_s=_UNBOUNDED_RATE,
        collective_latency_s=0.0,
    )
    with_subchannel = _scope_by_name(
        advisor.evaluate_scopes(k3, wl, fast_mxu_hbm_chip), "whole_step"
    )
    without_subchannel = _scope_by_name(
        advisor.evaluate_scopes(
            dataclasses.replace(k3, weight_block_k=0), wl, fast_mxu_hbm_chip
        ),
        "whole_step",
    )
    subchannel_delta_us = with_subchannel.compute_us - without_subchannel.compute_us
    strategy_b_us = 84.19
    strategy_a_us = 316.45
    assert subchannel_delta_us == pytest.approx(strategy_b_us, abs=_REPORT_ROUND_TOL_US)
    assert subchannel_delta_us < strategy_a_us


def test_achievable_fused_overlaps_collectives() -> None:
    """Verify untiled whole_step overlaps AR collectives unlike per_op."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    tp_ranks = 8
    wl_base = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=4096,
        tp=tp_ranks,
        ep=1,
        dispatch_regime="zero_floor",
        op_overhead_us=0.0,
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    low_lat_chip = dataclasses.replace(chip, collective_latency_s=1e-8)
    high_lat_chip = dataclasses.replace(chip, collective_latency_s=2e-7)
    sc_low = advisor.evaluate_scopes(q35, wl_base, low_lat_chip)
    sc_high = advisor.evaluate_scopes(q35, wl_base, high_lat_chip)
    whole_low = _scope_by_name(sc_low, "whole_step")
    whole_high = _scope_by_name(sc_high, "whole_step")
    per_op_low = _scope_by_name(sc_low, "per_op")
    per_op_high = _scope_by_name(sc_high, "per_op")

    ar_hops = round(math.log2(tp_ranks))
    num_ar = q35.ar_slots_per_layer * q35.num_layers
    delta_lat_us = (
        high_lat_chip.collective_latency_s - low_lat_chip.collective_latency_s
    ) * 1e6
    max_whole_delta_us = (ar_hops + 1) * delta_lat_us + 2 * _REPORT_ROUND_TOL_US
    min_per_op_delta_us = (
        constants.PER_OP_COLLECTIVE_CALL_FACTOR * num_ar * ar_hops * delta_lat_us
    )
    assert whole_high.step_achievable_us < whole_high.hbm_us + whole_high.comm_us
    assert (
        whole_high.step_achievable_us - whole_low.step_achievable_us
        < max_whole_delta_us
    )
    assert (
        per_op_high.step_achievable_us - per_op_low.step_achievable_us
        > min_per_op_delta_us
    )


def _apply_sync_drift_patch(monkeypatch: pytest.MonkeyPatch, drift_kind: str) -> None:
    """Apply one simulated core drift mutation for `verify_core_limits_sync` tests."""
    orig_tiles = core_exec_spec.Tiles()
    if drift_kind == "tiles_ring_depth":
        alt_tiles = dataclasses.replace(orig_tiles, nbd=core_vmem.MIN_RING_DEPTH - 1)
        monkeypatch.setattr(
            spec_bridge.exec_spec,
            "Tiles",
            functools.partial(dataclasses.replace, alt_tiles),
        )
        return
    simple_patches: dict[str, tuple[object, str, object]] = {
        "verified_ranks": (
            spec_bridge,
            "CORE_LIMITS",
            dataclasses.replace(constants.CORE_LIMITS, verified_ranks=(1, 2, 4)),
        ),
        "bucket_low": (core_runtime, "MAX_BUCKET_ROWS", 16),
        "bucket_exceeds_cap": (
            spec_bridge,
            "CORE_LIMITS",
            dataclasses.replace(
                constants.CORE_LIMITS,
                default_verify_bucket_top=core_runtime.MAX_BUCKET_ROWS + 1,
            ),
        ),
        "core_module_families": (
            spec_bridge,
            "CORE_MODULE_FAMILIES",
            (*specs.CORE_MODULE_FAMILIES, "nonexistent_family"),
        ),
        "max_ar_slots_per_layer": (
            spec_bridge,
            "CORE_LIMITS",
            dataclasses.replace(
                constants.CORE_LIMITS,
                max_ar_slots_per_layer=constants.CORE_LIMITS.max_ar_slots_per_layer + 1,
            ),
        ),
        "min_ring_depth": (core_vmem, "MIN_RING_DEPTH", core_vmem.MIN_RING_DEPTH + 1),
        "chip_budgets": (
            default_profiles,
            "CHIP_BUDGETS",
            {
                **default_profiles.CHIP_BUDGETS,
                "TPU7x": dataclasses.replace(
                    default_profiles.CHIP_BUDGETS["TPU7x"],
                    vmem_mib=default_profiles.CHIP_BUDGETS["TPU7x"].vmem_mib + 1,
                ),
            },
        ),
        "chip_spec_vmem": (
            spec_bridge,
            "BUILTIN_CHIP_SPECS",
            {
                **spec_bridge.BUILTIN_CHIP_SPECS,
                "tpu7x": dataclasses.replace(
                    spec_bridge.BUILTIN_CHIP_SPECS["tpu7x"],
                    vmem_usable_bytes=(
                        default_profiles.CHIP_BUDGETS["TPU7x"].vmem_mib - 2
                    )
                    << 20,
                ),
            },
        ),
        "core_supported_dtypes": (
            spec_bridge,
            "CORE_SUPPORTED_DTYPES",
            frozenset((*constants.CORE_SUPPORTED_DTYPES, "bogus_dtype")),
        ),
        "kv_page_dtype": (
            spec_bridge,
            "CORE_KV_DTYPE_ALIASES",
            {**constants.CORE_KV_DTYPE_ALIASES, "fp8_e5m2": "float8_e5m2"},
        ),
        "recurrent_state_dtype": (
            spec_bridge,
            "CORE_STATE_DTYPE_ALIASES",
            {**constants.CORE_STATE_DTYPE_ALIASES, "fp32": "float32"},
        ),
    }
    if drift_kind in ("core_pool_page_dtype", "core_gdn_rec_dtype"):
        assert spec_bridge.pool_layout is not None
        assert spec_bridge.gdn_decls is not None
        assert spec_bridge.jnp is not None
        if drift_kind == "core_pool_page_dtype":
            fake_pool = types.SimpleNamespace(
                page_dtype=spec_bridge.jnp.bfloat16,
                rec_dtype=spec_bridge.jnp.bfloat16,
            )
            monkeypatch.setattr(
                spec_bridge.pool_layout, "PoolLayout", lambda: fake_pool
            )
        else:
            fake_decl = types.SimpleNamespace(rec_dtype=spec_bridge.jnp.float32)
            monkeypatch.setattr(
                spec_bridge.gdn_decls, "gdn_cache_decl", lambda _geom: fake_decl
            )
        return
    target, attr_name, value = simple_patches[drift_kind]
    monkeypatch.setattr(target, attr_name, value)


@pytest.mark.parametrize(
    ("drift_kind", "prefix"),
    [
        ("verified_ranks", "verified_ranks:"),
        ("tiles_ring_depth", "min_ring_depth:"),
        ("chip_spec_vmem", "chip_budgets:"),
        ("bucket_low", "default_bucket_tops_within_max_bucket_rows:"),
        ("bucket_exceeds_cap", "default_bucket_tops_within_max_bucket_rows:"),
        ("core_module_families", "core_module_families:"),
        ("max_ar_slots_per_layer", "max_ar_slots_per_layer:"),
        ("min_ring_depth", "min_ring_depth:"),
        ("chip_budgets", "chip_budgets:"),
        ("core_supported_dtypes", "core_supported_dtypes:"),
        ("kv_page_dtype", "kv_page_dtype:"),
        ("recurrent_state_dtype", "recurrent_state_dtype:"),
        ("core_pool_page_dtype", "kv_page_dtype:"),
        ("core_gdn_rec_dtype", "recurrent_state_dtype:"),
    ],
)
def test_verify_core_limits_sync_detects_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
    prefix: str,
) -> None:
    """Verify verify_core_limits_sync detects live limits, dtype, and budget drift."""
    assert spec_bridge.verify_core_limits_sync() == []
    _apply_sync_drift_patch(monkeypatch, drift_kind)
    msgs = spec_bridge.verify_core_limits_sync()
    assert any(m.startswith(prefix) for m in msgs)


def test_advisor_core_limits_match_core_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify spec_bridge sync check passes and allows MAX_BUCKET_ROWS headroom."""
    assert spec_bridge.verify_core_limits_sync() == []
    live_families = tuple(
        sorted(
            info.name
            for info in pkgutil.iter_modules(core_modules.__path__)
            if info.ispkg
        )
    )
    assert tuple(sorted(specs.CORE_MODULE_FAMILIES)) == live_families
    monkeypatch.setattr(core_runtime, "MAX_BUCKET_ROWS", 128)
    assert spec_bridge.verify_core_limits_sync() == []


def test_fit_budget_matches_core_planner() -> None:
    """Verify advisor fit_vmem_budget_bytes matches core planner headroom and bounds."""
    chip_7x = advisor.resolve_chip_spec("tpu7x")
    chip_6e = advisor.resolve_chip_spec("v6e")
    adv_default = vmem.fit_vmem_budget_bytes(
        chip_spec=chip_7x, apply_ring_reserve=False
    )
    core_default = core_vmem.fit_vmem_budget_bytes(Tiles())
    assert adv_default == core_default

    adv_tiny = vmem.fit_vmem_budget_bytes(
        tiles=TINY_TILES, chip_spec=chip_7x, apply_ring_reserve=False
    )
    assert adv_tiny == core_vmem.fit_vmem_budget_bytes(TINY_TILES)
    assert (
        vmem.fit_vmem_budget_bytes(
            tiles=Tiles(vmem_mib=0), chip_spec=chip_7x, apply_ring_reserve=False
        )
        == 0
    )
    expected_headroom = constants.CORE_LIMITS.fit_vmem_headroom_mib << 20
    assert (
        vmem.fit_vmem_budget_bytes(
            tiles=Tiles(vmem_mib=62), chip_spec=chip_6e, apply_ring_reserve=False
        )
        == (62 << 20) - expected_headroom
    )
    assert (
        vmem.fit_vmem_budget_bytes(
            tiles=Tiles(vmem_mib=100), chip_spec=chip_7x, apply_ring_reserve=False
        )
        == (100 << 20) - expected_headroom
    )

    wl_base = advisor.WorkloadSpec(
        phase="decode", batch=4, tp=8, ep=8, vmem_fit_headroom=False
    )
    wl_headroom = dataclasses.replace(wl_base, vmem_fit_headroom=True)
    whole_base = _scope_by_name(
        advisor.evaluate_scopes("qwen3.5-397b", wl_base, chip_7x), "whole_step"
    )
    whole_headroom = _scope_by_name(
        advisor.evaluate_scopes("qwen3.5-397b", wl_headroom, chip_7x), "whole_step"
    )
    assert whole_headroom.max_vmem_tokens < whole_base.max_vmem_tokens


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify exact closed-form per-token live VMEM bytes and max_vmem_tokens."""
    pk3 = advisor.resolve_model_profile("kimi-k3")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    expected_grp = {
        "residual_buffer_bytes": 86016,
        "attn_scratch_bytes": 6528,
        "linear_attn_scratch_bytes": 4608,
        "moe_scratch_bytes": 53248,
        "total_per_token_bytes": 139264,
    }
    assert parts_grp == expected_grp

    parts_msk = advisor.compute_per_token_vmem_bytes(
        pk3, dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    )
    expected_masked_total = 169472
    assert parts_msk["total_per_token_bytes"] == expected_masked_total

    parts_no_alias = advisor.compute_per_token_vmem_bytes(
        pk3, dataclasses.replace(wl_grp, vmem_alias=False)
    )
    expected_no_alias_moe = 56960
    assert parts_no_alias["moe_scratch_bytes"] == expected_no_alias_moe
    assert parts_no_alias["total_per_token_bytes"] == (
        86016 + 6528 + 4608 + expected_no_alias_moe
    )

    scopes = advisor.evaluate_scopes(pk3, wl_grp, "tpu7x")
    whole = _scope_by_name(scopes, "whole_step")
    single = _scope_by_name(scopes, "single_layer")
    min_ring_bytes = (
        2 * constants.CORE_LIMITS.dense_slot_bytes
        + 2 * constants.CORE_LIMITS.expert_slot_bytes
    )
    budget_bytes = vmem.fit_vmem_budget_bytes(
        chip_spec=advisor.resolve_chip_spec("tpu7x"), apply_ring_reserve=False
    )
    ffn_scratch = (
        2
        * constants.ATTN_TILE_TOKENS
        * pk3.moe_intermediate_size
        * constants.FP32_BYTES
    )
    expected_whole_tokens = (budget_bytes - min_ring_bytes - ffn_scratch) // parts_grp[
        "total_per_token_bytes"
    ]
    assert whole.max_vmem_tokens == expected_whole_tokens
    capped_wl = dataclasses.replace(
        wl_grp, max_rows_per_call=constants.CORE_LIMITS.max_batch_rows
    )
    capped_whole = _scope_by_name(
        advisor.evaluate_scopes(pk3, capped_wl, "tpu7x"), "whole_step"
    )
    assert capped_whole.max_vmem_tokens == min(
        constants.CORE_LIMITS.max_batch_rows, expected_whole_tokens
    )
    assert single.max_vmem_tokens >= whole.max_vmem_tokens
    headroom_scopes = advisor.evaluate_scopes(
        pk3, dataclasses.replace(wl_grp, vmem_fit_headroom=True), "tpu7x"
    )
    assert (
        _scope_by_name(headroom_scopes, "single_layer").max_vmem_tokens
        > _scope_by_name(headroom_scopes, "whole_step").max_vmem_tokens
    )


def test_vmem_budget_override_and_dma_buffers_triple_buffering() -> None:
    """Verify `--vmem-budget-mib` override and `--dma-buffers {2,3}` effects."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl_2 = advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2)
    cx_2 = _scope_by_name(advisor.evaluate_scopes(p397, wl_2, chip), "whole_step")
    cx_3 = _scope_by_name(
        advisor.evaluate_scopes(p397, dataclasses.replace(wl_2, dma_buffers=3), chip),
        "whole_step",
    )
    cx_vmem = _scope_by_name(
        advisor.evaluate_scopes(
            p397, dataclasses.replace(wl_2, vmem_budget_mib=58.0), chip
        ),
        "whole_step",
    )
    per_token_bytes = vmem.compute_per_token_vmem_bytes(p397, wl_2)[
        "total_per_token_bytes"
    ]
    slot_pair_mib = (
        constants.CORE_LIMITS.dense_slot_bytes + constants.CORE_LIMITS.expert_slot_bytes
    ) / (1 << 20)
    assert cx_3.vmem_static_mib - cx_2.vmem_static_mib == pytest.approx(slot_pair_mib)
    assert cx_3.selected_dense_slots == cx_2.selected_dense_slots + 1
    assert cx_3.selected_expert_slots == cx_2.selected_expert_slots + 1
    assert cx_3.max_vmem_tokens < cx_2.max_vmem_tokens
    extra_budget_bytes = 2 << 20
    expected_tok_gain = extra_budget_bytes // per_token_bytes
    assert (
        abs((cx_vmem.max_vmem_tokens - cx_2.max_vmem_tokens) - expected_tok_gain) <= 1
    )


@pytest.mark.parametrize(
    ("tp", "ep", "pcp", "accepted"),
    [(4, 8, 2, True), (4, 8, 1, False), (4, 16, 2, False)],
)
def test_expert_parallel_spans_context_parallel_ranks(
    tp: int, ep: int, pcp: int, *, accepted: bool
) -> None:
    """Verify `ep` may span `tp * dp * pcp` ranks and no wider."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp
    )
    if accepted:
        per_op = _scope_by_name(advisor.evaluate_scopes(p397, workload, chip), "per_op")
        base_ep4 = _scope_by_name(
            advisor.evaluate_scopes(
                p397, dataclasses.replace(workload, pcp=1, ep=4), chip
            ),
            "per_op",
        )
        assert per_op.feasible
        assert per_op.compute_us == pytest.approx(0.5 * base_ep4.compute_us, abs=2.0)
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.evaluate_scopes(p397, workload, chip)


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.evaluate_scopes(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


def test_whole_step_tiles_overflow_and_reads_weights_once() -> None:
    """Verify whole_step tiles past VMEM ceiling without re-reading weights."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    base_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=64,
        spec_tokens=3,
        tp=8,
        ep=8,
        dcp=4,
        moe_mode="grouped_on_chip",
        max_rows_per_call=1024,
        dispatch_regime="zero_floor",
    )
    untiled_scopes = advisor.evaluate_scopes(q35, base_wl, chip)
    untiled = _scope_by_name(untiled_scopes, "whole_step")
    tiled_scopes = advisor.evaluate_scopes(
        q35, dataclasses.replace(base_wl, vmem_budget_mib=18.0), chip
    )
    tiled = _scope_by_name(tiled_scopes, "whole_step")
    single_tiled = _scope_by_name(tiled_scopes, "single_layer")
    assert untiled.num_vmem_splits == 1
    assert tiled.num_vmem_splits > 1
    assert tiled.feasible
    assert tiled.step_weight_gb == untiled.step_weight_gb
    max_act_spill_us = 100.0
    max_step_ratio = 1.5
    assert tiled.hbm_us > untiled.hbm_us
    assert tiled.hbm_us - untiled.hbm_us < max_act_spill_us
    assert tiled.step_bound_us < max_step_ratio * single_tiled.step_bound_us


def test_unchunked_pp_prefill_preserves_window() -> None:
    """Verify unchunked, full-chunk, and microbatched PP prefill match physics."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    isl_tokens = 4096
    wl_single = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=isl_tokens,
        tp=1,
        pp=8,
        ep=1,
        chunk_size=0,
        microbatches=0,
    )
    wl_full_chunk = dataclasses.replace(wl_single, chunk_size=isl_tokens)
    wl_mb = dataclasses.replace(wl_single, batch=8, microbatches=8)
    scopes_single = advisor.evaluate_scopes(model_profile, wl_single, chip)
    per_single = _scope_by_name(scopes_single, "per_op")
    single_l = _scope_by_name(scopes_single, "single_layer")
    per_chunk = _scope_by_name(
        advisor.evaluate_scopes(model_profile, wl_full_chunk, chip), "per_op"
    )
    per_mb = _scope_by_name(
        advisor.evaluate_scopes(model_profile, wl_mb, chip), "per_op"
    )
    assert single_l.step_upper_us < per_single.step_upper_us
    assert per_chunk.compute_us == pytest.approx(
        per_single.compute_us, abs=_REPORT_ROUND_TOL_US
    )
    assert per_chunk.hbm_us == pytest.approx(
        per_single.hbm_us, abs=_REPORT_ROUND_TOL_US
    )
    assert per_mb.compute_us == pytest.approx(
        per_single.compute_us, abs=_REPORT_ROUND_TOL_US
    )

    scopes_half = advisor.evaluate_scopes(
        model_profile, dataclasses.replace(wl_single, isl=isl_tokens // 2), chip
    )
    per_half = _scope_by_name(scopes_half, "per_op")
    assert _scope_by_name(scopes_half, "layer_group").step_upper_us < (
        per_half.step_upper_us
    )
    assert (
        0.5 * per_single.compute_us - _REPORT_ROUND_TOL_US
        <= per_half.compute_us
        < per_single.compute_us
    )
    assert per_half.hbm_us > 0.5 * per_single.hbm_us

    attn_only, attn_base_wl = _single_layer_profile("gqa")
    attn_chip = dataclasses.replace(
        chip,
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
    )
    attn_full_wl = dataclasses.replace(
        attn_base_wl, phase="prefill", isl=isl_tokens, batch=1, tp=1
    )
    attn_full = _scope_by_name(
        advisor.evaluate_scopes(attn_only, attn_full_wl, attn_chip), "per_op"
    )
    attn_half = _scope_by_name(
        advisor.evaluate_scopes(
            attn_only, dataclasses.replace(attn_full_wl, isl=isl_tokens // 2), attn_chip
        ),
        "per_op",
    )
    assert attn_half.compute_us < 0.5 * attn_full.compute_us


def test_ffn_scratch_includes_fp32_bytes() -> None:
    """Verify non-aliased single_layer static VMEM adds FP32 attn and GDN buffers."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl_alias = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=8, ep=8, vmem_alias=True
    )
    wl_noalias = dataclasses.replace(wl_alias, vmem_alias=False)
    s_alias = _scope_by_name(
        advisor.evaluate_scopes(q35, wl_alias, chip), "single_layer"
    )
    s_noalias = _scope_by_name(
        advisor.evaluate_scopes(q35, wl_noalias, chip), "single_layer"
    )
    gqa_bytes = (
        2 * constants.ATTN_TILE_TOKENS * constants.FP32_BYTES * q35.head_dim
        + 2
        * constants.ATTN_TILE_TOKENS
        * constants.ATTN_TILE_TOKENS
        * constants.FP32_BYTES
    )
    gdn_bytes = (
        constants.FP32_BYTES
        * min(q35.linear_value_heads // 8, constants.GDN_STREAMED_HEADS_CAP)
        * q35.linear_key_dim
        * q35.linear_value_dim
    )
    ffn_bytes = (
        2
        * constants.ATTN_TILE_TOKENS
        * (q35.moe_intermediate_size // 1)
        * constants.FP32_BYTES
    )
    expected_delta_mib = (
        gqa_bytes + gdn_bytes + ffn_bytes - max(gqa_bytes, gdn_bytes, ffn_bytes)
    ) / (1 << 20)
    assert s_noalias.vmem_static_mib - s_alias.vmem_static_mib == pytest.approx(
        expected_delta_mib, abs=1e-3
    )


def test_physical_config_rejects_undersized_slice_and_invalid_combos() -> None:
    """Verify validate_advise_inputs and validate_workload emit exact reason tuples."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")

    dense_small, _ = _single_layer_profile("dense_mlp")
    assert roofline.validate_workload(
        dense_small,
        c7x,
        advisor.WorkloadSpec(phase="decode", batch=4, dp=1, tp=4, ep=1),
    ) == ("slice_topology_unsupported",)
    assert set(
        roofline.validate_workload(
            q35, c7x, advisor.WorkloadSpec(phase="decode", batch=4, dp=1, tp=4, ep=4)
        )
    ) == {"slice_topology_unsupported", "hbm_pool_overflow"}

    bad_batch_wl = advisor.WorkloadSpec(phase="decode", batch=4, dp=8, tp=8)
    with pytest.raises(MegakernelConfigError):
        vmem.validate_advise_inputs(q35, bad_batch_wl, c7x)
    assert "batch_not_divisible_by_dp" in roofline.validate_workload(
        q35, c7x, bad_batch_wl
    )

    reasons_v6e = roofline.validate_workload(
        q35, cv6e, advisor.WorkloadSpec(tp=8, batch=4)
    )
    assert "hbm_pool_overflow" in reasons_v6e


def test_per_op_vmem_overflow_marks_infeasible() -> None:
    """Verify static VMEM overflow on per_op marks the baseline infeasible."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    scopes = advisor.evaluate_scopes(
        q35,
        advisor.WorkloadSpec(
            phase="decode",
            tp=8,
            ep=8,
            batch=4,
            vmem_budget_mib=float(constants.MIN_VMEM_MIB),
        ),
        "tpu7x",
    )
    per_op = _scope_by_name(scopes, "per_op")
    assert (per_op.feasible, per_op.max_vmem_tokens, per_op.reason) == (
        False,
        0,
        "per_op_vmem_overflow",
    )


def test_gdn_linear_head_divisibility_reasons() -> None:
    """Verify linear attention head divisibility checks in validate_workload."""
    base_prof = advisor.resolve_model_profile("qwen3.5-397b")
    c7x = advisor.resolve_chip_spec("tpu7x")
    bad_v_prof = dataclasses.replace(
        base_prof, linear_value_heads=18, linear_key_heads=16
    )
    wl_tp16 = advisor.WorkloadSpec(phase="decode", batch=1, tp=16)
    reasons_v = roofline.validate_workload(bad_v_prof, c7x, wl_tp16)
    assert "linear_v_heads_not_divisible_by_tp" in reasons_v

    bad_k_prof = dataclasses.replace(
        base_prof, linear_value_heads=16, linear_key_heads=12
    )
    reasons_k = roofline.validate_workload(bad_k_prof, c7x, wl_tp16)
    assert "linear_k_heads_not_divisible_by_tp" in reasons_k

    rep_prof = dataclasses.replace(
        base_prof, linear_value_heads=128, linear_key_heads=16
    )
    reasons_rep = roofline.validate_workload(
        rep_prof, c7x, advisor.WorkloadSpec(phase="decode", batch=1, tp=32)
    )
    assert "linear_v_heads_not_divisible_by_tp" not in reasons_rep
    assert "linear_k_heads_not_divisible_by_tp" not in reasons_rep


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
def test_core_legality_gates(model_slug: str) -> None:
    """Verify core legality gates pp, pcp, phase, dtypes, family, rows, and ranks."""
    profile = advisor.resolve_model_profile(model_slug)
    preset_decode = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    native_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        tp=preset_decode.tp,
        ep=preset_decode.ep,
        dp=preset_decode.dp,
        pp=1,
        dcp=preset_decode.dcp,
        pcp=1,
        layer_groups=preset_decode.layer_groups,
    )
    native_whole = _scope_by_name(
        advisor.evaluate_scopes(profile, native_wl, "tpu7x"), "whole_step"
    )
    families = set(profile.layer_families) - set(constants.STRUCTURAL_LAYER_TAGS)
    family_gap = bool(
        (families & constants.SUPPORTED_ONCHIP_FAMILIES)
        - constants.CORE_MODULE_FAMILIES
    )
    cache_gap = (
        profile.num_attention_layers > 0
        and profile.kv_dtype.lower() not in constants.CORE_SUPPORTED_KV_DTYPES
    ) or (
        profile.num_linear_layers > 0
        and profile.state_dtype.lower() not in constants.CORE_SUPPORTED_STATE_DTYPES
    )
    assert ("family_not_in_core" in native_whole.blockers) is family_gap
    assert ("kv_or_state_dtype_not_in_core" in native_whole.blockers) is cache_gap
    if model_slug == "qwen3.5-397b":
        assert profile.state_dtype == "bf16"
        assert native_whole.support == "within_core_limits"
    eff_tp = vmem.effective_tp(native_wl)
    assert ("rank_count_outside_core_grid" in native_whole.blockers) is (
        eff_tp not in constants.CORE_LIMITS.verified_ranks
    )

    core_tp = next(
        (
            r
            for r in reversed(constants.CORE_LIMITS.verified_ranks)
            if r <= preset_decode.tp
            and profile.num_attention_heads % r == 0
            and (
                profile.num_linear_layers == 0
                or (
                    profile.linear_key_heads % r == 0
                    and profile.linear_value_heads % r == 0
                )
            )
            and (profile.num_moe_layers == 0 or profile.moe_intermediate_size % r == 0)
        ),
        1,
    )
    core_ep = (
        core_tp
        if (
            profile.num_experts > 0
            and profile.num_experts % core_tp == 0
            and preset_decode.ep > 1
        )
        else 1
    )
    core_w_dt = (
        profile.weight_dtype
        if profile.weight_dtype.lower() in constants.CORE_SUPPORTED_DTYPES
        else "bf16"
    )
    raw_e_dt = profile.expert_dtype or profile.weight_dtype
    core_e_dt = (
        raw_e_dt if raw_e_dt.lower() in constants.CORE_SUPPORTED_DTYPES else "bf16"
    )
    ok_profile = advisor.resolve_model_profile(
        model_slug,
        weight_dtype=core_w_dt,
        expert_dtype=core_e_dt,
        kv_dtype="fp8_e4m3fn",
        state_dtype="bf16",
    )
    ok_wl = advisor.WorkloadSpec(phase="decode", batch=4, tp=core_tp, ep=core_ep)
    ok_whole = _scope_by_name(
        advisor.evaluate_scopes(ok_profile, ok_wl, "tpu7x"), "whole_step"
    )
    planner_gap = (
        not ok_profile.is_periodic
        or ok_profile.ar_slots_per_layer > constants.CORE_LIMITS.max_ar_slots_per_layer
    )
    if not family_gap and not planner_gap:
        assert ok_whole.support == "within_core_limits"
        assert ok_whole.blockers == ()
    else:
        assert ok_whole.support == "outside_core_limits"
        assert ("family_not_in_core" in ok_whole.blockers) is family_gap
        assert "kv_or_state_dtype_not_in_core" not in ok_whole.blockers

    for bad_kv in ("fp8_e5m2", "bf16", "fp4"):
        bad_kv_prof = advisor.resolve_model_profile(
            model_slug, kv_dtype=bad_kv, state_dtype="bf16"
        )
        bad_kv_whole = _scope_by_name(
            advisor.evaluate_scopes(bad_kv_prof, ok_wl, "tpu7x"), "whole_step"
        )
        assert bad_kv_whole.support == "outside_core_limits"
        assert "kv_or_state_dtype_not_in_core" in bad_kv_whole.blockers

    if profile.num_linear_layers > 0:
        for bad_state in ("fp8", "fp32"):
            bad_st_prof = advisor.resolve_model_profile(
                model_slug, kv_dtype="fp8_e4m3fn", state_dtype=bad_state
            )
            bad_st_whole = _scope_by_name(
                advisor.evaluate_scopes(bad_st_prof, ok_wl, "tpu7x"), "whole_step"
            )
            assert "kv_or_state_dtype_not_in_core" in bad_st_whole.blockers

    gate_probes = (
        (dataclasses.replace(ok_wl, pp=2), "pipeline_parallel_not_in_core"),
        (
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=64, tp=core_tp, ep=core_ep, pcp=2
            ),
            "context_parallel_not_in_core",
        ),
        (
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=64, tp=core_tp, ep=core_ep
            ),
            "phase_not_in_core",
        ),
        (
            advisor.WorkloadSpec(
                phase="mixed", batch=4, isl=64, chunk_size=32, tp=core_tp, ep=core_ep
            ),
            "phase_not_in_core",
        ),
        (
            dataclasses.replace(ok_wl, batch=constants.CORE_LIMITS.max_batch_rows + 8),
            "rows_exceed_core_max_batch_rows",
        ),
    )
    for bad_wl, expected_blocker in gate_probes:
        gated = _scope_by_name(
            advisor.evaluate_scopes(ok_profile, bad_wl, "tpu7x"), "whole_step"
        )
        assert gated.support == "outside_core_limits"
        assert expected_blocker in gated.blockers

    if profile.num_attention_heads % 32 == 0 and (
        profile.num_linear_layers == 0
        or (profile.linear_key_heads % 32 == 0 and profile.linear_value_heads % 32 == 0)
    ):
        bad_rank_wl = advisor.WorkloadSpec(phase="decode", batch=1, tp=32, ep=1)
        if not roofline.validate_workload(ok_profile, "tpu7x", bad_rank_wl):
            bad_rank_whole = _scope_by_name(
                advisor.evaluate_scopes(ok_profile, bad_rank_wl, "tpu7x"),
                "whole_step",
            )
            assert "rank_count_outside_core_grid" in bad_rank_whole.blockers


def _eval_verify_whole_step(
    batch: int,
    max_rows_per_call: int,
    op_overhead_us: float = 0.0,
    state_dtype: str | None = None,
) -> advisor.ScopeEstimate:
    """Evaluate Qwen3.5-397B verify step and return whole_step ScopeEstimate.

    Returns:
        `ScopeEstimate` for `whole_step`.
    """
    profile = (
        advisor.resolve_model_profile("qwen3.5-397b", state_dtype=state_dtype)
        if state_dtype is not None
        else advisor.resolve_model_profile("qwen3.5-397b")
    )
    return _scope_by_name(
        advisor.evaluate_scopes(
            profile,
            advisor.WorkloadSpec(
                phase="verify",
                batch=batch,
                spec_tokens=3,
                context=4096,
                tp=8,
                dcp=4,
                ep=8,
                max_rows_per_call=max_rows_per_call,
                op_overhead_us=op_overhead_us,
            ),
            "tpu7x",
        ),
        "whole_step",
    )


def test_max_rows_per_call_clamps_single_call_rows() -> None:
    """Verify max_rows_per_call clamps single-call whole_step max_vmem_tokens."""
    explicit_cap = 32
    full = _eval_verify_whole_step(batch=1, max_rows_per_call=0)
    capped = _eval_verify_whole_step(batch=1, max_rows_per_call=explicit_cap)
    assert full.max_vmem_tokens > explicit_cap
    assert capped.max_vmem_tokens == explicit_cap


def test_rows_above_single_call_limit_priced_as_microbatches() -> None:
    """Verify 16 verify requests at cap=32 equals 2 equal 8-request calls."""
    row_cap = 32
    per_call_requests = 8
    two_calls = 2
    one_call = _eval_verify_whole_step(
        batch=per_call_requests, max_rows_per_call=row_cap, state_dtype="bf16"
    )
    two_call_est = _eval_verify_whole_step(
        batch=two_calls * per_call_requests,
        max_rows_per_call=row_cap,
        state_dtype="bf16",
    )
    assert two_call_est.ops_per_step == two_calls * one_call.ops_per_step
    assert two_call_est.num_vmem_splits == two_calls * one_call.num_vmem_splits
    assert two_call_est.step_bound_us == pytest.approx(
        two_calls * one_call.step_bound_us, abs=1e-4
    )
    assert two_call_est.step_weight_gb == pytest.approx(
        two_calls * one_call.step_weight_gb, abs=1e-4
    )
    assert two_call_est.support == "within_core_limits"
    assert two_call_est.blockers == ()


def test_max_rows_per_call_splits_whole_requests() -> None:
    """Verify verify microbatching rounds row limits down to whole requests."""
    capped_32 = _eval_verify_whole_step(batch=16, max_rows_per_call=32)
    capped_35 = _eval_verify_whole_step(batch=16, max_rows_per_call=35)
    assert capped_35.step_bound_us == capped_32.step_bound_us
    assert capped_35.ops_per_step == capped_32.ops_per_step
    assert capped_35.max_vmem_tokens == capped_32.max_vmem_tokens


def test_max_rows_per_call_below_one_request_is_config_error() -> None:
    """Verify max_rows_per_call smaller than one verify request raises config error."""
    with pytest.raises(MegakernelConfigError):
        _eval_verify_whole_step(batch=16, max_rows_per_call=3)


def test_validate_workload_names_row_limit_below_one_request() -> None:
    """Verify non-raising validate_workload reports row limit below one request."""
    rows_per_req = 4
    workload = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=rows_per_req - 1,
        context=4096,
        tp=8,
        dcp=4,
        ep=8,
        max_rows_per_call=rows_per_req - 1,
    )
    assert roofline.validate_workload("qwen3.5-397b", "tpu7x", workload) == (
        "row_limit_below_request",
    )
    one_req = dataclasses.replace(workload, max_rows_per_call=rows_per_req)
    assert roofline.validate_workload("qwen3.5-397b", "tpu7x", one_req) == ()


def test_rows_above_default_buckets_flagged() -> None:
    """Verify default bucket boundaries emit row_limit_not_stated cleanly."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    dec_top = constants.CORE_LIMITS.default_decode_bucket_top
    ver_top = constants.CORE_LIMITS.default_verify_bucket_top
    dec_ok = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(phase="decode", batch=dec_top, tp=8, ep=8),
            "tpu7x",
        ),
        "whole_step",
    )
    dec_over = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(phase="decode", batch=dec_top + 1, tp=8, ep=8),
            "tpu7x",
        ),
        "whole_step",
    )
    dec_capped = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="decode",
                batch=dec_top + 1,
                tp=8,
                ep=8,
                max_rows_per_call=constants.CORE_LIMITS.max_batch_rows,
            ),
            "tpu7x",
        ),
        "whole_step",
    )
    assert dec_ok.blockers == ()
    assert "row_limit_not_stated" in dec_over.blockers
    assert "row_limit_not_stated" not in dec_capped.blockers

    ver_ok = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="verify", batch=ver_top // 4, spec_tokens=3, tp=8, ep=8
            ),
            "tpu7x",
        ),
        "whole_step",
    )
    ver_over = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="verify", batch=(ver_top // 4) + 1, spec_tokens=3, tp=8, ep=8
            ),
            "tpu7x",
        ),
        "whole_step",
    )
    assert ver_ok.blockers == ()
    assert "row_limit_not_stated" in ver_over.blockers


def test_moe_padding_floor_and_mode_symmetry() -> None:
    """Verify per_op floor symmetry and batch=1..64 MoE padding progression."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    row_tile = float(chip.matmul_row_tile(k3.expert_dtype))
    m1 = _scope_by_name(
        advisor.evaluate_scopes(
            k3,
            advisor.WorkloadSpec(
                phase="decode", batch=1, tp=32, ep=32, moe_mode="masked_on_chip"
            ),
            chip,
        ),
        "per_op",
    )
    g1 = _scope_by_name(
        advisor.evaluate_scopes(
            k3,
            advisor.WorkloadSpec(
                phase="decode", batch=1, tp=32, ep=32, moe_mode="grouped_on_chip"
            ),
            chip,
        ),
        "per_op",
    )
    assert m1.moe_flop_factor == g1.moe_flop_factor == pytest.approx(row_tile)

    per16 = _scope_by_name(
        advisor.evaluate_scopes(
            k3,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32),
            chip,
        ),
        "per_op",
    )
    assert per16.moe_flop_factor < row_tile

    m64 = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="decode", batch=64, tp=8, ep=8, moe_mode="masked_on_chip"
            ),
            chip,
        ),
        "whole_step",
    )
    g64 = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="decode", batch=64, tp=8, ep=8, moe_mode="grouped_on_chip"
            ),
            chip,
        ),
        "whole_step",
    )
    assert m64.moe_flop_factor == g64.moe_flop_factor


def test_intra_host_dp_and_block_scaled_moe_consistency() -> None:
    """Verify intra-host DP groups MoE while DCN/masked paths stay ungrouped."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    compute_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"), hbm_bw_bytes_per_s=_UNBOUNDED_RATE
    )
    ws_grp = _scope_by_name(
        advisor.evaluate_scopes(
            prof,
            advisor.WorkloadSpec(
                phase="decode",
                batch=600,
                context=32768,
                tp=2,
                dp=4,
                ep=8,
                moe_mode="grouped_on_chip",
            ),
            compute_chip,
        ),
        "whole_step",
    )
    ws_msk = _scope_by_name(
        advisor.evaluate_scopes(
            prof,
            advisor.WorkloadSpec(
                phase="decode",
                batch=600,
                context=32768,
                tp=2,
                dp=4,
                ep=8,
                moe_mode="masked_on_chip",
            ),
            compute_chip,
        ),
        "whole_step",
    )
    dcn_scopes = advisor.evaluate_scopes(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        compute_chip,
    )
    ws_dcn = _scope_by_name(dcn_scopes, "whole_step")
    sl_dcn = _scope_by_name(dcn_scopes, "single_layer")
    assert ws_grp.moe_flop_factor == pytest.approx(2.731, abs=1e-3)
    assert ws_msk.moe_flop_factor == pytest.approx(51.882, abs=1e-3)
    assert ws_dcn.moe_flop_factor == pytest.approx(51.882, abs=1e-3)
    assert sl_dcn.moe_flop_factor == pytest.approx(2.731, abs=1e-3)
    assert sl_dcn.compute_us < ws_dcn.compute_us


@pytest.mark.parametrize("ranks", [2, 8, 32])
def test_in_slice_all_reduce_uses_ici_for_every_rank_count(ranks: int) -> None:
    """Verify evaluate_scopes prices ICI all-reduce across ring volume and hops."""
    chip = advisor.resolve_chip_spec("tpu7x")
    attn_only = models.finalize_model_profile(
        advisor.ModelProfile(
            name="ar-attn",
            num_layers=4,
            num_attention_layers=4,
            num_moe_layers=0,
            hidden_size=4096,
            intermediate_size=0,
            num_attention_heads=64,
            num_kv_heads=64,
            head_dim=64,
            weight_dtype="fp8",
            kv_dtype="fp8",
            act_dtype="fp8",
        )
    )
    batch = 4
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=batch,
        context=64,
        tp=ranks,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
    )
    whole = _scope_by_name(advisor.evaluate_scopes(attn_only, wl, chip), "whole_step")
    msg_bytes = batch * attn_only.hidden_size * constants.BF16_BYTES
    vol = 2.0 * (ranks - 1) / ranks * msg_bytes
    hops = round(math.log2(ranks))
    per_call_s = vol / chip.ici_link_bw_bytes_per_s + hops * chip.collective_latency_s
    num_ar = attn_only.ar_slots_per_layer * attn_only.num_layers
    assert whole.comm_us == pytest.approx(
        num_ar * per_call_s * 1e6, abs=_REPORT_ROUND_TOL_US
    )


def test_all_to_all_fanout_capped_by_destination_groups() -> None:
    """Verify MoE All-to-All fanout caps at ep_dp and doubles from top_k 2 to 4."""
    base_moe = models.finalize_model_profile(
        advisor.ModelProfile(
            name="a2a-moe",
            num_layers=2,
            num_attention_layers=2,
            num_moe_layers=2,
            hidden_size=2048,
            intermediate_size=0,
            moe_intermediate_size=1024,
            num_attention_heads=8,
            num_kv_heads=4,
            head_dim=128,
            num_experts=16,
            top_k=10,
            weight_dtype="fp8",
            expert_dtype="fp8",
            act_dtype="fp8",
        )
    )
    chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        fp8_flops_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
        collective_latency_s=0.0,
    )
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=64,
        context=16,
        tp=2,
        dp=4,
        ep=8,
        moe_mode="grouped_on_chip",
        dispatch_regime="zero_floor",
    )
    w_k10 = _scope_by_name(advisor.evaluate_scopes(base_moe, wl, chip), "whole_step")
    w_k4 = _scope_by_name(
        advisor.evaluate_scopes(dataclasses.replace(base_moe, top_k=4), wl, chip),
        "whole_step",
    )
    w_k2 = _scope_by_name(
        advisor.evaluate_scopes(dataclasses.replace(base_moe, top_k=2), wl, chip),
        "whole_step",
    )
    assert w_k10.fill_drain_us == w_k4.fill_drain_us
    ep_dp = wl.ep // wl.tp
    msg_bytes = (wl.batch // wl.dp) * base_moe.hidden_size * constants.BF16_BYTES
    expected_delta_us = (
        base_moe.num_moe_layers
        * 2.0
        * (ep_dp - 1)
        / ep_dp
        * 2
        * msg_bytes
        / chip.ici_egress_bw_bytes_per_s
        * 1e6
    )
    assert w_k4.fill_drain_us - w_k2.fill_drain_us == pytest.approx(
        expected_delta_us, abs=_REPORT_ROUND_TOL_US
    )


def test_strict_floor_not_above_uniform_estimate() -> None:
    """Verify exact strict lower-bound weights and latency sit below uniform bound."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    q35_fp32 = dataclasses.replace(q35, state_dtype="fp32")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    whole = _scope_by_name(advisor.evaluate_scopes(q35, wl, chip), "whole_step")
    whole_fp32 = _scope_by_name(
        advisor.evaluate_scopes(q35_fp32, wl, chip), "whole_step"
    )
    assert whole.step_weight_strict_gb == pytest.approx(2.17055, abs=1e-4)
    assert whole.step_floor_strict_us == pytest.approx(846.55, abs=_REPORT_ROUND_TOL_US)
    assert whole_fp32.step_floor_strict_us == pytest.approx(
        953.35, abs=_REPORT_ROUND_TOL_US
    )
    assert whole.step_weight_strict_gb < whole.step_weight_gb
    assert whole.step_floor_strict_us < whole.step_bound_us


def test_whole_step_bound_pays_collectives_after_router() -> None:
    """Verify slower collectives raise comm-bound whole_step floor one-for-one."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="verify", batch=1, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
    )
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_coll_us = 3.0
    slow_chip = dataclasses.replace(
        chip,
        collective_latency_s=slow_coll_us * 1e-6,
        collective_latency_us_bracket=(slow_coll_us, slow_coll_us),
    )
    fast = _scope_by_name(advisor.evaluate_scopes(q35, wl, chip), "whole_step")
    slow = _scope_by_name(advisor.evaluate_scopes(q35, wl, slow_chip), "whole_step")
    assert fast.num_vmem_splits == 1
    assert slow.num_vmem_splits == 1
    fast_core_us = fast.roof_floor_us - fast.fill_drain_us
    slow_core_us = slow.roof_floor_us - slow.fill_drain_us
    assert fast_core_us > fast.hbm_us
    assert slow.comm_us > fast.comm_us
    assert slow_core_us - fast_core_us == pytest.approx(
        slow.comm_us - fast.comm_us, abs=0.03
    )


def test_achievable_upper_band_never_inverts_below_achievable_or_floor() -> None:
    """Verify step_achievable_upper_us never inverts and tracks upper priors."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="verify", batch=16, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
    )
    scopes = advisor.evaluate_scopes(q35, wl, chip)
    for sc in scopes:
        assert sc.step_achievable_upper_us >= sc.step_achievable_us
        assert sc.step_achievable_upper_us >= sc.step_upper_us
        assert sc.step_achievable_upper_us >= sc.step_bound_us
    wider_chip = dataclasses.replace(
        chip,
        collective_latency_us_bracket=(
            chip.collective_latency_us_bracket[0],
            chip.collective_latency_us_bracket[1] + 5.0,
        ),
    )
    wide_scopes = advisor.evaluate_scopes(q35, wl, wider_chip)
    for base_sc, wide_sc in zip(scopes, wide_scopes, strict=True):
        assert wide_sc.step_bound_us == base_sc.step_bound_us
        assert wide_sc.step_achievable_us == base_sc.step_achievable_us
        assert wide_sc.step_upper_us > base_sc.step_upper_us
        assert wide_sc.step_achievable_upper_us > base_sc.step_achievable_upper_us
        assert wide_sc.step_achievable_upper_us > wide_sc.step_achievable_us
        assert wide_sc.step_achievable_upper_us > wide_sc.step_upper_us


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
@pytest.mark.parametrize("phase", ["decode", "verify", "prefill", "mixed"])
def test_scope_subterms_reconstruct_roof_floor(
    model_slug: str, phase: advisor.Phase
) -> None:
    """Verify chain_us with hbm_us and comm_us rebuilds every scope's roof_floor_us."""
    preset = advisor.DEPLOYMENT_PRESETS[model_slug]
    layout = preset.prefill if phase in _MIXED_PREFILL_PHASES else preset.decode
    workload = advisor.WorkloadSpec(
        phase=phase,
        batch=1 if phase == "prefill" else (4 if phase == "verify" else 16),
        isl=512 if phase in _MIXED_PREFILL_PHASES else 4096,
        context=4096,
        chunk_size=512 if phase in _MIXED_PREFILL_PHASES else 0,
        spec_tokens=preset.decode.spec_tokens or 3
        if phase in ("verify", "mixed")
        else 0,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    for sc in advisor.evaluate_scopes(model_slug, workload, "tpu7x"):
        core_us = (
            max(sc.hbm_us, sc.chain_us + sc.comm_us)
            if sc.scope in _FUSED_SCOPES and sc.num_vmem_splits == 1
            else max(sc.hbm_us, sc.chain_us) + sc.comm_us
        )
        assert sc.chain_us >= sc.compute_us - _REPORT_ROUND_TOL_US
        assert core_us + sc.fill_drain_us == pytest.approx(
            sc.roof_floor_us, abs=_REPORT_ROUND_TOL_US
        )


def test_vector_unit_fp4_dequant_floor() -> None:
    """Verify vector unit floor doubles under half vector rate and eta_dense."""
    model_profile = dataclasses.replace(
        advisor.resolve_model_profile("qwen3.5-397b"), expert_dtype="mxfp4"
    )
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=8,
        ep=8,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        fp8_flops_per_s=_UNBOUNDED_RATE,
        bf16_flops_per_s=_UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        ici_link_bw_bytes_per_s=_UNBOUNDED_RATE,
        ici_egress_bw_bytes_per_s=_UNBOUNDED_RATE,
        collective_latency_s=0.0,
        vector_f32_ops_per_s=1e12,
    )
    whole = _scope_by_name(
        advisor.evaluate_scopes(model_profile, workload, chip), "whole_step"
    )
    half_rate = _scope_by_name(
        advisor.evaluate_scopes(
            model_profile,
            workload,
            dataclasses.replace(chip, vector_f32_ops_per_s=5e11),
        ),
        "whole_step",
    )
    derated = _scope_by_name(
        advisor.evaluate_scopes(
            model_profile,
            dataclasses.replace(
                workload,
                efficiency=advisor.EfficiencySpec(1.0, 0.5, 1.0, 1.0),
            ),
            chip,
        ),
        "whole_step",
    )
    assert whole.binding_unit == "vector"
    assert half_rate.compute_us == pytest.approx(
        2.0 * whole.compute_us, abs=2.0 * _REPORT_ROUND_TOL_US
    )
    assert half_rate.step_floor_strict_us == pytest.approx(
        2.0 * whole.step_floor_strict_us, abs=2.0 * _REPORT_ROUND_TOL_US
    )
    assert derated.step_bound_us == pytest.approx(
        whole.step_bound_us, abs=_REPORT_ROUND_TOL_US
    )
    assert derated.step_achievable_us == pytest.approx(
        2.0 * whole.step_achievable_us, abs=2.0 * _REPORT_ROUND_TOL_US
    )


def test_mixed_precision_attention_qk_pv_executes_at_bf16_peak() -> None:
    """Verify attention QK^T/PV runs at bf16_peak whenever act or kv dtype is BF16."""
    chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
    )
    wl_prefill = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=2048,
        osl=0,
        tp=8,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    base = advisor.resolve_model_profile(
        "qwen3.5-397b", weight_dtype="fp8", act_dtype="bf16", kv_dtype="fp8"
    )
    bf16_kv = dataclasses.replace(base, kv_dtype="bf16")
    all_fp8 = advisor.resolve_model_profile(
        "qwen3.5-397b", weight_dtype="fp8", act_dtype="fp8", kv_dtype="fp8"
    )
    fp8_act_bf16_kv = dataclasses.replace(all_fp8, kv_dtype="bf16")

    c_mixed = _scope_by_name(
        advisor.evaluate_scopes(base, wl_prefill, chip), "per_op"
    ).compute_us
    c_bf16_kv = _scope_by_name(
        advisor.evaluate_scopes(bf16_kv, wl_prefill, chip), "per_op"
    ).compute_us
    c_all_fp8 = _scope_by_name(
        advisor.evaluate_scopes(all_fp8, wl_prefill, chip), "per_op"
    ).compute_us
    c_fp8_mixed_kv = _scope_by_name(
        advisor.evaluate_scopes(fp8_act_bf16_kv, wl_prefill, chip), "per_op"
    ).compute_us
    assert c_mixed == pytest.approx(c_bf16_kv, abs=_REPORT_ROUND_TOL_US)
    assert c_all_fp8 < c_fp8_mixed_kv < c_mixed


def test_subchannel_quantization_vector_bottleneck() -> None:
    """Verify subchannel block_k < 512 adds exact vector ops while >=512/BF16 add 0."""
    chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        fp8_flops_per_s=_UNBOUNDED_RATE,
        bf16_flops_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=1e11,
    )
    wl = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=2048,
        osl=0,
        tp=8,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
    )
    dense_small = models.finalize_model_profile(
        advisor.ModelProfile(
            name="subchan-dense",
            num_layers=4,
            num_attention_layers=4,
            num_moe_layers=0,
            hidden_size=1024,
            intermediate_size=2048,
            num_attention_heads=8,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            kv_dtype="fp8",
            act_dtype="bf16",
            weight_block_k=128,
        )
    )
    c_128 = _scope_by_name(
        advisor.evaluate_scopes(dense_small, wl, chip), "per_op"
    ).compute_us
    c_0 = _scope_by_name(
        advisor.evaluate_scopes(
            dataclasses.replace(dense_small, weight_block_k=0), wl, chip
        ),
        "per_op",
    ).compute_us
    c_512 = _scope_by_name(
        advisor.evaluate_scopes(
            dataclasses.replace(dense_small, weight_block_k=512), wl, chip
        ),
        "per_op",
    ).compute_us
    local_qkv = 1024 * 2 * (128 + 128)
    local_ffn = 3 * 1024 * (2048 // 8)
    base_dense_params = 4 * (local_qkv + local_ffn)
    expected_delta_us = (
        constants.SUBCHANNEL_VPU_OPS_PER_BLOCK
        * wl.isl
        * base_dense_params
        / dense_small.weight_block_k
        / chip.vector_f32_ops_per_s
        * 1e6
    )
    assert c_512 == pytest.approx(c_0, abs=_REPORT_ROUND_TOL_US)
    assert c_128 - c_0 == pytest.approx(expected_delta_us, abs=_REPORT_ROUND_TOL_US)

    bf16_blocked = dataclasses.replace(
        dense_small, weight_dtype="bf16", kv_dtype="bf16"
    )
    bf16_unblocked = dataclasses.replace(bf16_blocked, weight_block_k=0)
    assert _scope_by_name(
        advisor.evaluate_scopes(bf16_blocked, wl, chip), "per_op"
    ).compute_us == pytest.approx(
        _scope_by_name(
            advisor.evaluate_scopes(bf16_unblocked, wl, chip), "per_op"
        ).compute_us,
        abs=_REPORT_ROUND_TOL_US,
    )


def test_strict_vector_floor_uses_packed_element_rate() -> None:
    """Verify strict floor reduces dense vector time by exact 4x register packing."""
    dense_prof = models.finalize_model_profile(
        advisor.ModelProfile(
            name="vec-strict-dense",
            num_layers=4,
            num_attention_layers=4,
            num_moe_layers=0,
            hidden_size=4096,
            intermediate_size=8192,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="bf16",
            kv_dtype="bf16",
            act_dtype="bf16",
        )
    )
    vec_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        fp8_flops_per_s=_UNBOUNDED_RATE,
        bf16_flops_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=1e11,
    )
    whole = _scope_by_name(
        advisor.evaluate_scopes(
            dense_prof,
            advisor.WorkloadSpec(
                phase="decode",
                batch=16,
                tp=1,
                ep=1,
                op_overhead_us=0.0,
                dispatch_regime="zero_floor",
            ),
            vec_chip,
        ),
        "whole_step",
    )
    assert whole.step_floor_strict_us == pytest.approx(
        whole.step_bound_us / constants.VECTOR_MAX_ELEMENTS_PER_WORD,
        abs=_REPORT_ROUND_TOL_US,
    )


def _exact_binomial_busiest_reference(
    tokens_ep: int,
    top_k: int,
    num_experts: int,
    experts_local: int,
    owner_groups: int,
) -> float:
    """Compute exact independent-binomial busiest-group expectation via math.comb.

    Returns:
        Expected active experts on the busiest owner group.
    """
    hit = 1.0 - (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    if owner_groups <= 1:
        return max(1.0, experts_local * hit)
    experts_per_group = num_experts // owner_groups
    if hit >= 1.0:
        return max(1.0, float(experts_local))
    cdf = 0.0
    expected_group_max = 0.0
    for count in range(experts_per_group):
        pmf = (
            math.comb(experts_per_group, count)
            * (hit**count)
            * ((1.0 - hit) ** (experts_per_group - count))
        )
        cdf = min(1.0, cdf + pmf)
        expected_group_max += 1.0 - cdf**owner_groups
    return max(1.0, expected_group_max * (experts_local / experts_per_group))


@pytest.mark.parametrize(
    "model_slug",
    sorted(
        slug for slug, preset in advisor.MODEL_PRESETS.items() if preset.num_experts > 0
    ),
)
@pytest.mark.parametrize(
    ("tokens_ep", "owner_groups", "saturate_hit"),
    [
        (16, 1, False),
        (16, 8, False),
        (4, 8, True),
    ],
)
def test_experts_hit_busiest_matches_exact_binomial_order_statistic(
    model_slug: str,
    tokens_ep: int,
    owner_groups: int,
    *,
    saturate_hit: bool,
) -> None:
    """Verify experts_hit_busiest against exact math.comb order statistic."""
    profile = advisor.resolve_model_profile(model_slug)
    preset_ep = max(1, advisor.DEPLOYMENT_PRESETS[model_slug].decode.ep)
    ep_size = preset_ep if profile.num_experts % preset_ep == 0 else 1
    top_k = profile.num_experts if saturate_hit else profile.top_k
    experts_local = profile.num_experts // ep_size
    expected = _exact_binomial_busiest_reference(
        tokens_ep, top_k, profile.num_experts, experts_local, owner_groups
    )
    actual = roofline.experts_hit_busiest(
        tokens_ep, top_k, profile.num_experts, experts_local, owner_groups
    )
    experts_per_group = max(1, profile.num_experts // max(1, owner_groups))
    rel_tol = (
        4.0
        * experts_per_group
        * max(1.0, math.lgamma(experts_per_group + 1))
        * math.ulp(1.0)
    )
    assert actual == pytest.approx(expected, rel=rel_tol)


def test_per_op_achievable_prices_operator_kernel_boundaries() -> None:
    """Verify per_op ops_per_step equals 2 * num_layers and prices launch gaps."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    gap_us = 2.0
    per_op = _scope_by_name(
        advisor.evaluate_scopes(
            q35,
            advisor.WorkloadSpec(
                phase="decode",
                batch=1,
                tp=8,
                ep=8,
                op_overhead_us=gap_us,
                dispatch_regime="compiled_xla",
            ),
            "tpu7x",
        ),
        "per_op",
    )
    expected_ops = constants.SUBLAYER_BOUNDARIES_PER_LAYER * q35.num_layers
    assert per_op.ops_per_step == expected_ops
    assert per_op.launch_us == pytest.approx(expected_ops * gap_us)


def test_host_head_defaults_zero_and_computes_break_even_delta() -> None:
    """Verify default 0 host head, single-add split behavior, and break-even delta."""
    whole_dev_us = 14250.0
    per_op_dev_us = 12800.0
    assert (
        roofline.compute_break_even_host_head_delta_us(whole_dev_us, per_op_dev_us)
        == whole_dev_us - per_op_dev_us
    )
    split_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
        max_rows_per_call=32,
        dispatch_regime="zero_floor",
    )
    def_scopes = advisor.evaluate_scopes("qwen3.5-397b", split_wl, "tpu7x")
    per_op_def = _scope_by_name(def_scopes, "per_op")
    whole_def = _scope_by_name(def_scopes, "whole_step")
    assert per_op_def.host_head_us == 0.0
    assert whole_def.host_head_us == 0.0
    assert whole_def.step_bound_us > per_op_def.step_bound_us
    assert whole_def.break_even_host_head_delta_us == pytest.approx(
        whole_def.step_bound_us - per_op_def.step_bound_us, abs=_REPORT_ROUND_TOL_US
    )
    fused_head_us = 550.0
    eq_wl = dataclasses.replace(
        split_wl,
        host_head_per_op_us=fused_head_us + whole_def.break_even_host_head_delta_us,
        host_head_fused_us=fused_head_us,
        host_head_explicit=True,
    )
    eq_scopes = advisor.evaluate_scopes("qwen3.5-397b", eq_wl, "tpu7x")
    whole_eq = _scope_by_name(eq_scopes, "whole_step")
    per_op_eq = _scope_by_name(eq_scopes, "per_op")
    assert whole_eq.host_head_us == pytest.approx(fused_head_us, abs=1e-6)
    assert whole_eq.step_bound_us == pytest.approx(
        per_op_eq.step_bound_us, abs=_REPORT_ROUND_TOL_US
    )
    assert whole_eq.roof_floor_us == whole_def.roof_floor_us
    assert whole_eq.step_floor_strict_us == whole_def.step_floor_strict_us


def test_floors_use_low_band_factors_and_exclude_first_slot_hbm() -> None:
    """Verify floor excludes cold DMA slot refill that upper bands include."""
    small_prof = models.finalize_model_profile(
        advisor.ModelProfile(
            name="slot-oracle",
            num_layers=4,
            num_attention_layers=4,
            num_moe_layers=0,
            hidden_size=1024,
            intermediate_size=2048,
            num_attention_heads=8,
            num_kv_heads=4,
            head_dim=128,
            weight_dtype="bf16",
            kv_dtype="bf16",
            act_dtype="bf16",
        )
    )
    chip = advisor.resolve_chip_spec("tpu7x")
    lo_slot = advisor.RingPlan(dense_slots=2, expert_slots=2, dense_slot_bytes=1 << 18)
    hi_slot = advisor.RingPlan(dense_slots=2, expert_slots=2, dense_slot_bytes=1 << 19)
    wl_lo = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=64,
        tp=8,
        dispatch_regime="zero_floor",
        ring_plan=lo_slot,
    )
    wl_hi = dataclasses.replace(wl_lo, ring_plan=hi_slot)
    w_lo = _scope_by_name(
        advisor.evaluate_scopes(small_prof, wl_lo, chip), "whole_step"
    )
    w_hi = _scope_by_name(
        advisor.evaluate_scopes(small_prof, wl_hi, chip), "whole_step"
    )
    assert w_lo.fill_drain_us == 0.0
    assert w_lo.step_bound_us == w_hi.step_bound_us
    boundary_count = w_lo.ops_per_step - 1
    delta_bytes = (1 << 19) - (1 << 18)
    expected_upper_delta = boundary_count * delta_bytes / chip.hbm_bw_bytes_per_s * 1e6
    assert w_hi.step_upper_us - w_lo.step_upper_us == pytest.approx(
        expected_upper_delta, abs=_REPORT_ROUND_TOL_US
    )


def test_prefill_attention_flops_scale_linearly_with_batch() -> None:
    """Verify prefill attention FLOPs scale linearly with batch via evaluate_scopes."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    compute_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_bw_bytes_per_s=_UNBOUNDED_RATE,
        vector_f32_ops_per_s=_UNBOUNDED_RATE,
        ici_link_bw_bytes_per_s=_UNBOUNDED_RATE,
        ici_egress_bw_bytes_per_s=_UNBOUNDED_RATE,
        collective_latency_s=0.0,
    )
    wl_b1 = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=2048,
        tp=8,
        ep=8,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
        efficiency=advisor.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    p4 = _scope_by_name(
        advisor.evaluate_scopes(q35, dataclasses.replace(wl_b1, batch=4), compute_chip),
        "per_op",
    )
    p16 = _scope_by_name(
        advisor.evaluate_scopes(
            q35, dataclasses.replace(wl_b1, batch=16), compute_chip
        ),
        "per_op",
    )
    assert p4.moe_flop_factor == p16.moe_flop_factor == 1.0
    assert p16.compute_us == pytest.approx(4.0 * p4.compute_us, abs=0.1)

    dense_prof, dense_wl = _single_layer_profile("gqa")
    dense_prefill = dataclasses.replace(dense_wl, phase="prefill", isl=2048, tp=1)
    d1 = _scope_by_name(
        advisor.evaluate_scopes(dense_prof, dense_prefill, compute_chip), "per_op"
    )
    d4 = _scope_by_name(
        advisor.evaluate_scopes(
            dense_prof, dataclasses.replace(dense_prefill, batch=4), compute_chip
        ),
        "per_op",
    )
    assert d4.compute_us == pytest.approx(4.0 * d1.compute_us, abs=_REPORT_ROUND_TOL_US)


def test_kimi_k3_non_dsa_mla_dcp_gather_uses_absorbed_widths() -> None:
    """Verify non-DSA absorbed MLA DCP gather prices kv_lora + rope and kv_lora out."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl_dcp32 = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=4096,
        tp=32,
        ep=32,
        dcp=32,
        op_overhead_us=0.0,
        dispatch_regime="zero_floor",
    )
    wl_dcp1 = dataclasses.replace(wl_dcp32, dcp=1)
    whole_d32 = _scope_by_name(
        advisor.evaluate_scopes(k3, wl_dcp32, chip), "whole_step"
    )
    whole_d1 = _scope_by_name(advisor.evaluate_scopes(k3, wl_dcp1, chip), "whole_step")
    absorbed_bytes_per_head_tok = constants.BF16_BYTES * (
        k3.kv_lora_rank + k3.qk_rope_head_dim
    ) + constants.FP32_BYTES * (k3.kv_lora_rank + constants.LSE_SCALARS_PER_HEAD)
    unabsorbed_bytes_per_head_tok = constants.BF16_BYTES * (
        k3.qk_nope_head_dim + k3.qk_rope_head_dim
    ) + constants.FP32_BYTES * (k3.head_dim + constants.LSE_SCALARS_PER_HEAD)
    assert absorbed_bytes_per_head_tok > unabsorbed_bytes_per_head_tok
    local_attn = k3.num_attention_layers
    nh_local = math.ceil(k3.num_attention_heads / wl_dcp32.tp)
    expected_wire_bytes = (
        local_attn * (wl_dcp32.dcp - 1) * 1 * nh_local * absorbed_bytes_per_head_tok
    )
    expected_hops = (
        local_attn
        * constants.DCP_COLLECTIVES_PER_ATTN_LAYER
        * round(math.log2(wl_dcp32.dcp))
    )
    expected_comm_delta_us = (
        expected_wire_bytes / chip.ici_egress_bw_bytes_per_s
        + expected_hops * chip.collective_latency_s
    ) * 1e6
    assert expected_comm_delta_us == pytest.approx(263.84, abs=_REPORT_ROUND_TOL_US)
    assert whole_d32.comm_us - whole_d1.comm_us == pytest.approx(
        expected_comm_delta_us, abs=_REPORT_ROUND_TOL_US
    )


def _non_weight_hbm_us(est: advisor.ScopeEstimate, chip: advisor.ChipSpec) -> float:
    """Return HBM time in microseconds after subtracting streamed step weights.

    Returns:
        Activation, KV cache, and recurrent state HBM duration in microseconds.
    """
    weight_hbm_us = (est.step_weight_gb * 1e9 / chip.hbm_bw_bytes_per_s) * 1e6
    return est.hbm_us - weight_hbm_us


def test_pcp_sharding_kv_scaling_and_attention_conservation() -> None:
    """Verify PCP ring batch scaling, HBM overflow, and DSA/MLA 1/pcp scaling."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    glm = advisor.resolve_model_profile("glm-5.2")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl_pcp1 = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=32, ep=32, pcp=1
    )
    wl_pcp32 = dataclasses.replace(wl_pcp1, pcp=32)
    assert "hbm_pool_overflow" not in roofline.validate_workload(k3, chip, wl_pcp32)
    k3_c1 = _scope_by_name(advisor.evaluate_scopes(k3, wl_pcp1, chip), "per_op")
    k3_c32 = _scope_by_name(advisor.evaluate_scopes(k3, wl_pcp32, chip), "per_op")
    assert _non_weight_hbm_us(k3_c32, chip) * 32.0 == pytest.approx(
        _non_weight_hbm_us(k3_c1, chip), rel=0.05
    )
    k3_nomoe = dataclasses.replace(k3, num_moe_layers=0)
    k3_nm1 = _scope_by_name(advisor.evaluate_scopes(k3_nomoe, wl_pcp1, chip), "per_op")
    k3_nm32 = _scope_by_name(
        advisor.evaluate_scopes(k3_nomoe, wl_pcp32, chip), "per_op"
    )
    assert k3_nm32.compute_us * 32.0 == pytest.approx(k3_nm1.compute_us, rel=1e-3)

    glm_base = advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=16, ep=16)
    g1 = _scope_by_name(advisor.evaluate_scopes(glm, glm_base, chip), "per_op")
    glm_nomoe = dataclasses.replace(glm, num_moe_layers=0)
    gnm1 = _scope_by_name(advisor.evaluate_scopes(glm_nomoe, glm_base, chip), "per_op")
    for pcp_factor in (4, 16):
        gp = _scope_by_name(
            advisor.evaluate_scopes(
                glm, dataclasses.replace(glm_base, pcp=pcp_factor), chip
            ),
            "per_op",
        )
        gnmp = _scope_by_name(
            advisor.evaluate_scopes(
                glm_nomoe, dataclasses.replace(glm_base, pcp=pcp_factor), chip
            ),
            "per_op",
        )
        assert gnmp.compute_us == pytest.approx(gnm1.compute_us / pcp_factor, rel=0.01)
        assert _non_weight_hbm_us(gp, chip) == pytest.approx(
            _non_weight_hbm_us(g1, chip) / pcp_factor, rel=0.01
        )


def test_evaluate_scopes_applies_calibrated_launch_gap_priors() -> None:
    """Verify evaluate_scopes directly applies calibrated launch-gap measurements."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    meas = advisor.load_measurements(
        {
            "schema": "advisor.measurements.v1",
            "source": {"kind": "xprof_xplane", "desc": "trace", "file": "trace.pb"},
            "launch_gap_us": {"p50": 3.5, "p90": 9.5, "n": 40},
        },
        chip_name="tpu7x",
    )
    per_op = _scope_by_name(
        advisor.evaluate_scopes(q35, wl, "tpu7x", measurements=meas), "per_op"
    )
    assert per_op.launch_us == pytest.approx(per_op.ops_per_step * 3.5)
    assert per_op.step_upper_us - per_op.step_bound_us > per_op.ops_per_step * (
        9.5 - 3.5 - 1e-3
    )


def test_evaluate_scopes_applies_measurements_ring_plan() -> None:
    """Verify public evaluate_scopes applies measurements.ring_plan when unset."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    custom_ring = advisor.RingPlan(dense_slots=4, expert_slots=3)
    meas = advisor.load_measurements(
        {
            "schema": "advisor.measurements.v1",
            "source": {"kind": "xprof_xplane", "desc": "ring", "file": "ring.pb"},
            "ring_plan": {"dense_slots": 4, "expert_slots": 3},
        },
        chip_name="tpu7x",
    )
    from_meas = advisor.evaluate_scopes(q35, wl, chip, measurements=meas)
    from_explicit = advisor.evaluate_scopes(
        q35, dataclasses.replace(wl, ring_plan=custom_ring), chip
    )
    assert from_meas == from_explicit
    whole_meas = _scope_by_name(from_meas, "whole_step")
    assert whole_meas.selected_dense_slots == custom_ring.dense_slots
    assert whole_meas.selected_expert_slots == custom_ring.expert_slots


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
def test_tiled_fused_scope_attributes_collectives_and_binding_unit(
    model_slug: str,
) -> None:
    """Verify tiled fused scopes expose comm_us and classify collective binding."""
    preset = advisor.DEPLOYMENT_PRESETS[model_slug]
    layout = preset.decode
    wl = advisor.WorkloadSpec(
        phase="mixed",
        batch=16,
        isl=4096,
        chunk_size=1024,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    scopes = advisor.evaluate_scopes(model_slug, wl, "tpu7x")
    single = _scope_by_name(scopes, "single_layer")
    whole = _scope_by_name(scopes, "whole_step")
    assert whole.num_vmem_splits > 1
    assert single.comm_us > 0.0
    assert whole.comm_us == pytest.approx(single.comm_us, abs=_REPORT_ROUND_TOL_US)
    assert whole.fill_drain_us < whole.comm_us
    expected_floor = (
        max(whole.hbm_us, whole.chain_us) + whole.comm_us + whole.fill_drain_us
    )
    assert expected_floor == pytest.approx(
        whole.roof_floor_us, abs=_REPORT_ROUND_TOL_US
    )
    if whole.comm_us > max(whole.hbm_us, whole.chain_us):
        assert whole.binding_unit == "comm"


def test_glm52_dsa_prefill_continuous_across_topk_and_blocked() -> None:
    """Verify GLM-5.2 DSA prefill scales smoothly across topk and with 1/pcp."""
    glm = advisor.resolve_model_profile("glm-5.2")
    chip = advisor.resolve_chip_spec("tpu7x")
    base_wl = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4094, tp=16, ep=1, dcp=1
    )

    def _per_op(
        isl: int, pcp: int = 1, target_chip: advisor.ChipSpec = chip
    ) -> advisor.ScopeEstimate:
        """Evaluate GLM-5.2 per_op prefill estimate at `isl` and `pcp`."""
        return _scope_by_name(
            advisor.evaluate_scopes(
                glm, dataclasses.replace(base_wl, isl=isl, pcp=pcp), target_chip
            ),
            "per_op",
        )

    p_4094 = _per_op(4094)
    p_4095 = _per_op(4095)
    p_4096 = _per_op(4096)
    p_8192 = _per_op(8192)
    step_slope = p_4094.compute_us / 4094.0
    assert p_4095.compute_us - p_4094.compute_us == pytest.approx(
        step_slope, abs=step_slope
    )
    assert p_4096.hbm_us < p_4095.hbm_us * 2.0
    assert p_4096.compute_us < p_4095.compute_us * 2.0
    assert p_8192.hbm_us < p_4096.hbm_us * 4.0

    base_non_weight_hbm = _non_weight_hbm_us(p_4096, chip)
    p4_pcp = _scope_by_name(
        advisor.evaluate_scopes(
            glm, dataclasses.replace(base_wl, isl=4096, pcp=4), chip
        ),
        "per_op",
    )
    assert (
        p_4096.compute_us / 4.0 - _REPORT_ROUND_TOL_US
        <= p4_pcp.compute_us
        < p_4096.compute_us
    )
    for pcp_deg in (4, 16):
        scaled = _per_op(4096, pcp=pcp_deg)
        assert _non_weight_hbm_us(scaled, chip) == pytest.approx(
            base_non_weight_hbm / pcp_deg, rel=0.01
        )
    glm_nomoe = dataclasses.replace(glm, num_moe_layers=0)
    nomoe_4096 = _scope_by_name(
        advisor.evaluate_scopes(
            glm_nomoe, dataclasses.replace(base_wl, isl=4096, pcp=1), chip
        ),
        "per_op",
    )
    for pcp_deg in (4, 16):
        nomoe_scaled = _scope_by_name(
            advisor.evaluate_scopes(
                glm_nomoe, dataclasses.replace(base_wl, isl=4096, pcp=pcp_deg), chip
            ),
            "per_op",
        )
        assert nomoe_scaled.compute_us == pytest.approx(
            nomoe_4096.compute_us / pcp_deg, rel=0.01
        )


def test_calibrated_measured_ar_preserves_band_order_and_split_scaling() -> None:
    """Verify calibrated AR avoids band inversion, checks bytes, and scales hop only."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    wl = advisor.WorkloadSpec(
        phase="mixed", batch=16, isl=4096, chunk_size=1024, tp=32, ep=32, dcp=32
    )
    active_tokens = 16 * 1 + 1024
    ar_bytes = active_tokens * k3.hidden_size * constants.BF16_BYTES

    results_by_lat: dict[
        float, tuple[advisor.ScopeEstimate, advisor.ScopeEstimate]
    ] = {}
    for ar_latency_us in (145.0, 0.5):
        meas = advisor.load_measurements(
            {
                "schema": "advisor.measurements.v1",
                "source": {"kind": "xprof_xplane", "desc": "ar", "file": "ar.pb"},
                "collectives": [
                    {
                        "path": "xla",
                        "op": "all-reduce",
                        "group_size": 32,
                        "latency_us": ar_latency_us,
                        "n": 120,
                        "bytes": ar_bytes,
                    },
                    {
                        "path": "in_kernel",
                        "op": "all-reduce",
                        "group_size": 32,
                        "latency_us": ar_latency_us,
                        "n": 120,
                        "bytes": ar_bytes,
                    },
                ],
            },
            chip_name="tpu7x",
        )
        scopes = advisor.evaluate_scopes(k3, wl, "tpu7x", measurements=meas)
        per_op = _scope_by_name(scopes, "per_op")
        whole = _scope_by_name(scopes, "whole_step")
        assert whole.num_vmem_splits > 1
        for est in scopes:
            assert est.step_floor_strict_us <= est.step_bound_us + _REPORT_ROUND_TOL_US
            assert est.step_bound_us <= est.step_achievable_us + _REPORT_ROUND_TOL_US
            assert (
                est.step_achievable_us
                <= est.step_achievable_upper_us + _REPORT_ROUND_TOL_US
            )
            assert est.step_bound_us <= est.step_upper_us + _REPORT_ROUND_TOL_US
            assert (
                est.step_upper_us <= est.step_achievable_upper_us + _REPORT_ROUND_TOL_US
            )
        results_by_lat[ar_latency_us] = (per_op, whole)

    per_lo, whole_lo = results_by_lat[0.5]
    per_hi, whole_hi = results_by_lat[145.0]
    per_comm_delta = per_hi.comm_us - per_lo.comm_us
    whole_comm_delta = whole_hi.comm_us - whole_lo.comm_us
    whole_ach_delta = whole_hi.step_achievable_us - whole_lo.step_achievable_us
    assert per_comm_delta > 0.0
    assert (
        per_comm_delta - _REPORT_ROUND_TOL_US
        <= whole_comm_delta
        <= per_comm_delta + _REPORT_ROUND_TOL_US
    )
    assert 0.0 < whole_ach_delta < whole_hi.num_vmem_splits * per_comm_delta

    uncal = advisor.evaluate_scopes(k3, wl, "tpu7x")
    bad_bytes_meas = advisor.load_measurements(
        {
            "schema": "advisor.measurements.v1",
            "source": {"kind": "xprof_xplane", "desc": "bad", "file": "bad.pb"},
            "collectives": [
                {
                    "path": "in_kernel",
                    "op": "all-reduce",
                    "group_size": 32,
                    "latency_us": 5000.0,
                    "n": 60,
                    "bytes": 64,
                },
            ],
        },
        chip_name="tpu7x",
    )
    assert (
        advisor.evaluate_scopes(k3, wl, "tpu7x", measurements=bad_bytes_meas) == uncal
    )


def test_calibrated_byteless_ar_ignored_when_expected_bytes_positive() -> None:
    """Verify calibrated AR entries without byte size are skipped when bytes > 0."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32, dcp=32)
    uncal = advisor.evaluate_scopes(k3, wl, "tpu7x")
    for raw_bytes in (None, 0):
        byteless_meas = advisor.load_measurements(
            {
                "schema": "advisor.measurements.v1",
                "source": {"kind": "xprof_xplane", "desc": "byteless", "file": "b.pb"},
                "collectives": [
                    {
                        "path": "xla",
                        "op": "all-reduce",
                        "group_size": 32,
                        "latency_us": 5000.0,
                        "n": 60,
                        **({"bytes": raw_bytes} if raw_bytes is not None else {}),
                    },
                    {
                        "path": "in_kernel",
                        "op": "all-reduce",
                        "group_size": 32,
                        "latency_us": 5000.0,
                        "n": 60,
                        **({"bytes": raw_bytes} if raw_bytes is not None else {}),
                    },
                ],
            },
            chip_name="tpu7x",
        )
        assert (
            advisor.evaluate_scopes(k3, wl, "tpu7x", measurements=byteless_meas)
            == uncal
        )


def test_microbatched_host_head_reclassifies_binding_unit() -> None:
    """Verify microbatched whole_step reclassifies binding_unit when host-bound."""
    whole = _eval_verify_whole_step(batch=16, max_rows_per_call=32, op_overhead_us=0.0)
    wl = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        context=4096,
        tp=8,
        dcp=4,
        ep=8,
        max_rows_per_call=32,
        host_head_fused_us=whole.roof_floor_us * 2.0,
    )
    host_whole = _scope_by_name(
        advisor.evaluate_scopes("qwen3.5-397b", wl, "tpu7x"), "whole_step"
    )
    assert host_whole.binding_unit == "host"


@pytest.mark.parametrize(
    "workload",
    [
        advisor.WorkloadSpec(
            phase="decode", batch=16, tp=8, ep=8, vmem_budget_mib=999.0
        ),
        advisor.WorkloadSpec(phase="decode", batch=16, tp=3, ep=8),
        advisor.WorkloadSpec(
            phase="decode", batch=16, tp=8, ep=8, layer_groups=(10, 10)
        ),
    ],
)
def test_validate_workload_and_advise_inputs_agree_on_bounds(
    workload: advisor.WorkloadSpec,
) -> None:
    """Verify validate_workload and validate_advise_inputs reject invalid budgets."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    assert roofline.validate_workload(q35, chip, workload)
    with pytest.raises(MegakernelConfigError):
        vmem.validate_advise_inputs(q35, workload, chip)


def test_tiled_moe_padding_consistent_across_tiled_scopes() -> None:
    """Verify tiled single_layer, layer_group, and whole_step share MoE padding."""
    wl = advisor.WorkloadSpec(
        phase="prefill", batch=16, isl=4096, tp=1, ep=1, dp=1, pp=8
    )
    scopes = advisor.evaluate_scopes("qwen3.5-397b", wl, "tpu7x")
    single = _scope_by_name(scopes, "single_layer")
    group = _scope_by_name(scopes, "layer_group")
    whole = _scope_by_name(scopes, "whole_step")
    assert single.num_vmem_splits == whole.num_vmem_splits > 1
    assert whole.moe_flop_factor == pytest.approx(single.moe_flop_factor, rel=1e-6)
    assert group.moe_flop_factor == pytest.approx(single.moe_flop_factor, rel=1e-6)


def test_pcp_dcp_coupling_and_chunked_prefill_kv_reads() -> None:
    """Verify tp=1 with pcp>1 and dcp>1 fails validation and chunked KV scales."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    bad_wl = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=1024, tp=1, ep=1, dcp=4, pcp=4
    )
    assert roofline.validate_workload(q35, chip, bad_wl)
    with pytest.raises(MegakernelConfigError):
        vmem.validate_advise_inputs(q35, bad_wl, chip)

    def _prefill_kv_delta(batch: int) -> float:
        """Compute KV-read HBM delta between long and short context at fixed chunk."""
        base = advisor.WorkloadSpec(
            phase="prefill", batch=batch, isl=4096, chunk_size=1024, tp=8, ep=8
        )
        lo = _scope_by_name(
            advisor.evaluate_scopes(q35, dataclasses.replace(base, context=1024), chip),
            "per_op",
        )
        hi = _scope_by_name(
            advisor.evaluate_scopes(
                q35, dataclasses.replace(base, context=32768), chip
            ),
            "per_op",
        )
        return hi.hbm_us - lo.hbm_us

    assert _prefill_kv_delta(4) == pytest.approx(
        _prefill_kv_delta(1), abs=_REPORT_ROUND_TOL_US
    )


def test_single_core_boundary_fill_drain_has_zero_collective_hops() -> None:
    """Verify 1-core run pays zero ICI collective hop latency in boundary fill/drain."""
    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    base_chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(phase="decode", batch=4, tp=1, ep=1, dp=1, pp=1)
    lo_chip = dataclasses.replace(base_chip, collective_latency_s=1e-6)
    hi_chip = dataclasses.replace(base_chip, collective_latency_s=1e-4)
    w_lo = _scope_by_name(advisor.evaluate_scopes(q35, wl, lo_chip), "whole_step")
    w_hi = _scope_by_name(advisor.evaluate_scopes(q35, wl, hi_chip), "whole_step")
    assert w_lo.step_upper_us == pytest.approx(
        w_hi.step_upper_us, abs=_REPORT_ROUND_TOL_US
    )
    assert w_lo.step_achievable_us == pytest.approx(
        w_hi.step_achievable_us, abs=_REPORT_ROUND_TOL_US
    )


def case_layouts(slug: str, phase: str) -> tuple[tuple[str, dict[str, int]], ...]:
    """Return `(layout_name, axes)` triples for `preset`, `single_rank`, `tp_ep_only`.

    Returns:
        Tuple of `(name, axes)` pairs for the model preset and phase.
    """
    preset = advisor.DEPLOYMENT_PRESETS[slug]
    layout = preset.prefill if phase in _MIXED_PREFILL_PHASES else preset.decode
    tp_ep_layout = minimal_valid_core_layout(
        preset.decode, max_rank=max(preset.decode.tp, preset.decode.ep)
    )
    return (
        (
            "preset",
            {
                "tp": layout.tp,
                "ep": layout.ep,
                "dp": layout.dp,
                "pp": layout.pp,
                "dcp": layout.dcp,
                "pcp": layout.pcp,
            },
        ),
        ("single_rank", {"tp": 1, "ep": 1, "dp": 1, "pp": 1, "dcp": 1, "pcp": 1}),
        (
            "tp_ep_only",
            {
                "tp": tp_ep_layout.tp,
                "ep": tp_ep_layout.ep,
                "dp": tp_ep_layout.dp,
                "pp": 1,
                "dcp": 1,
                "pcp": 1,
            },
        ),
    )


def case_workload(
    phase: specs.Phase, axes: Mapping[str, int], **overrides: object
) -> advisor.WorkloadSpec:
    """Build deterministic `WorkloadSpec` for `phase`, `axes`, and `overrides`.

    Returns:
        Configured `WorkloadSpec` instance.
    """
    merged_axes = dict(axes)
    if phase in ("decode", "verify"):
        merged_axes["pcp"] = 1
    dp_ranks = merged_axes["dp"]
    defaults_by_phase: dict[str, dict[str, object]] = {
        "decode": {"batch": 16 * dp_ranks, "isl": 4096},
        "prefill": {"batch": 1 * dp_ranks, "isl": 1024},
        "verify": {"batch": 4 * dp_ranks, "isl": 4096, "spec_tokens": 3},
        "mixed": {"batch": 8 * dp_ranks, "isl": 1024, "chunk_size": 512},
    }
    params = dict(defaults_by_phase[phase])
    params.update(merged_axes)
    params.update(overrides)
    workload_ctor: Callable[..., advisor.WorkloadSpec] = advisor.WorkloadSpec
    return workload_ctor(phase=phase, **params)


def case_chips() -> tuple[tuple[str, advisor.ChipSpec], ...]:
    """Return the four chip variants swept by the invariant case grid.

    Returns:
        Tuple of `(chip_label, ChipSpec)` pairs.
    """
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    slow_coll_us = 100.0
    return (
        ("tpu7x", tpu7x),
        (
            "tpu7x_ici50",
            dataclasses.replace(tpu7x, ici_egress_bw_bytes_per_s=50.0e9),
        ),
        (
            "tpu7x_lat100",
            dataclasses.replace(
                tpu7x,
                collective_latency_s=slow_coll_us * 1e-6,
                collective_latency_us_bracket=(slow_coll_us, slow_coll_us),
            ),
        ),
        ("v6e", advisor.resolve_chip_spec("v6e")),
    )


def iter_case_grid() -> Iterator[
    tuple[str, str, advisor.ModelProfile, specs.Phase, str, advisor.WorkloadSpec]
]:
    """Yield `(label, slug, profile, phase, layout_name, workload)` across the grid.

    Includes all `(preset, phase, layout)` triples plus the deterministic
    `(batch in (1, 16, 128)) x (context in (0, 8192))` sweep on `preset`.

    Yields:
        Case tuple for each grid entry.
    """
    extra_batches = (1, 16, 128)
    extra_contexts = (0, 8192)
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        for phase in CASE_PHASES:
            layouts = case_layouts(slug, phase)
            for lay_name, axes in layouts:
                wl = case_workload(phase, axes)
                yield (f"{slug}/{phase}/{lay_name}", slug, profile, phase, lay_name, wl)
            preset_axes = layouts[0][1]
            for batch_per_dp, ctx_val in itertools.product(
                extra_batches, extra_contexts
            ):
                wl_extra = case_workload(
                    phase,
                    preset_axes,
                    batch=batch_per_dp * preset_axes["dp"],
                    context=ctx_val,
                )
                label = f"{slug}/{phase}/preset/b={batch_per_dp},ctx={ctx_val}"
                yield (label, slug, profile, phase, "preset", wl_extra)


def _numeric_scope_fields(estimate: advisor.ScopeEstimate) -> dict[str, float]:
    """Return all non-bool numeric field values from `estimate`.

    Returns:
        Mapping of field name to float value.
    """
    out: dict[str, float] = {}
    for field in dataclasses.fields(estimate):
        val = getattr(estimate, field.name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[field.name] = float(val)
    return out


def _compare_numeric_estimates(
    label: str,
    ests_a: tuple[advisor.ScopeEstimate, ...],
    ests_b: tuple[advisor.ScopeEstimate, ...],
    max_diff: float,
) -> list[str]:
    """Return violation messages where any numeric field differs past `max_diff`.

    Returns:
        List of mismatch descriptions.
    """
    issues: list[str] = []
    for ea, eb in zip(ests_a, ests_b, strict=True):
        fa = _numeric_scope_fields(ea)
        fb = _numeric_scope_fields(eb)
        if set(fa) != set(fb):
            issues.append(f"{label} {ea.scope}: numeric field key mismatch")
            continue
        diffs = [
            f"{name}: {fa[name]} vs {fb[name]}"
            for name in fa
            if abs(fa[name] - fb[name]) > max_diff
        ]
        if diffs:
            issues.append(f"{label} {ea.scope}: {'; '.join(diffs[:3])}")
    return issues


def _check_band_order(
    label: str,
    estimates: tuple[advisor.ScopeEstimate, ...],
    half_units: int = 2,
) -> list[str]:
    """Verify monotonicity across feasible scope estimates within `half_units * u`.

    Returns:
        List of band-order violation descriptions.
    """
    margin = half_units * ROUND_HALF_UNIT_US
    issues: list[str] = []
    for est in estimates:
        if not est.feasible:
            continue
        pairs = (
            ("strict", est.step_floor_strict_us, "bound", est.step_bound_us),
            ("bound", est.step_bound_us, "achievable", est.step_achievable_us),
            (
                "achievable",
                est.step_achievable_us,
                "achievable_upper",
                est.step_achievable_upper_us,
            ),
            ("bound", est.step_bound_us, "upper", est.step_upper_us),
            (
                "upper",
                est.step_upper_us,
                "achievable_upper",
                est.step_achievable_upper_us,
            ),
        )
        for lo_name, lo_val, hi_name, hi_val in pairs:
            if lo_val > hi_val + margin:
                issues.append(
                    f"{label} {est.scope}: {lo_name}={lo_val} > {hi_name}={hi_val}"
                )
    return issues


def _build_measured_ar(
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    latency_us: float,
) -> specs.MeasurementsData:
    """Build synthetic trace MeasurementsData with XLA and in-kernel all-reduce calls.

    Returns:
        `MeasurementsData` containing both all-reduce path measurements.
    """
    act_dtype = profile.act_dtype or profile.weight_dtype
    nbytes = int(
        profile.hidden_size * workload.batch * constants.DTYPE_BYTES[act_dtype]
    )
    sample_count = 10
    calls = tuple(
        specs.CollectiveCall(
            path=path,
            op=constants.COLLECTIVE_OP_ALL_REDUCE,
            group_size=workload.tp,
            latency_us=latency_us,
            n=sample_count,
            bytes=nbytes,
        )
        for path in (constants.COLLECTIVE_PATH_XLA, constants.COLLECTIVE_PATH_IN_KERNEL)
    )
    return specs.MeasurementsData(
        source_desc="harness", source_kind="trace", collective_calls=calls
    )


def _check_calibrated_band_case(
    label: str,
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    default_chip: advisor.ChipSpec,
) -> tuple[bool, list[str]]:
    """Check calibrated band ordering and upper-bound gap invariance at 50 and 500 us.

    Returns:
        `(ran_calibration, issues)` for this calibrated case.
    """
    low_lat_us = 50.0
    high_lat_us = 500.0
    gap_half_units = 4
    uncalibrated = advisor.evaluate_scopes(profile, workload, default_chip)
    est_lo = advisor.evaluate_scopes(
        profile,
        workload,
        default_chip,
        measurements=_build_measured_ar(profile, workload, low_lat_us),
    )
    est_hi = advisor.evaluate_scopes(
        profile,
        workload,
        default_chip,
        measurements=_build_measured_ar(profile, workload, high_lat_us),
    )
    if uncalibrated in (est_lo, est_hi):
        return False, []
    issues = _check_band_order(f"{label}/ar50", est_lo)
    issues.extend(_check_band_order(f"{label}/ar500", est_hi))
    for ea, eb in zip(est_lo, est_hi, strict=True):
        if not (ea.feasible and eb.feasible):
            continue
        gap_lo = ea.step_upper_us - ea.step_bound_us
        gap_hi = eb.step_upper_us - eb.step_bound_us
        if abs(gap_lo - gap_hi) > gap_half_units * ROUND_HALF_UNIT_US:
            issues.append(
                f"{label} {ea.scope}: gap {gap_lo:.2f} us at 50 us vs "
                f"{gap_hi:.2f} us at 500 us"
            )
    return True, issues


def test_band_order_across_chips_and_measurements() -> None:
    """Verify band ordering across chip variants and calibrated AR gap invariance."""
    chips_list = case_chips()
    default_chip = chips_list[0][1]
    violations: list[str] = []
    calibrated_checked = 0
    for label, _, profile, _, lay_name, wl in iter_case_grid():
        for chip_name, chip in chips_list:
            ests = advisor.evaluate_scopes(profile, wl, chip)
            violations.extend(_check_band_order(f"{label}/{chip_name}", ests))
        if lay_name in ("preset", "tp_ep_only") and wl.tp > 1:
            checked, case_issues = _check_calibrated_band_case(
                label, profile, wl, default_chip
            )
            calibrated_checked += int(checked)
            violations.extend(case_issues)
    assert calibrated_checked > 0, "Expected at least one calibrated AR case to run"
    assert not violations, (
        f"Band order violations ({len(violations)}): {violations[:5]}"
    )


def test_single_rank_ignores_collective_latency() -> None:
    """Verify single-rank ignores collective latency and all layouts are monotone."""
    base_chip = advisor.resolve_chip_spec("tpu7x")
    chip_lo = dataclasses.replace(
        base_chip,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 1.0),
    )
    chip_hi = dataclasses.replace(
        base_chip,
        collective_latency_s=100.0e-6,
        collective_latency_us_bracket=(100.0, 100.0),
    )
    max_diff = 2 * ROUND_HALF_UNIT_US
    monotone_fields = (
        "step_floor_strict_us",
        "step_bound_us",
        "step_achievable_us",
        "step_upper_us",
    )
    violations: list[str] = []
    for label, _, profile, _, lay_name, wl in iter_case_grid():
        ests_lo = advisor.evaluate_scopes(profile, wl, chip_lo)
        ests_hi = advisor.evaluate_scopes(profile, wl, chip_hi)
        if lay_name == "single_rank":
            violations.extend(
                _compare_numeric_estimates(label, ests_lo, ests_hi, max_diff)
            )
        for ea, eb in zip(ests_lo, ests_hi, strict=True):
            for field_name in monotone_fields:
                va = float(getattr(ea, field_name))
                vb = float(getattr(eb, field_name))
                if vb < va - max_diff:
                    violations.append(
                        f"{label} {ea.scope}: {field_name} fell {va} -> {vb}"
                    )
    assert not violations, (
        f"Collective latency violations ({len(violations)}): {violations[:5]}"
    )


def _check_scaled_per_op(
    label: str,
    unit_est: advisor.ScopeEstimate,
    full_est: advisor.ScopeEstimate,
    scale: int,
) -> list[str]:
    """Compare full vs `scale * unit` on `per_op` compute, HBM, and weight fields.

    Returns:
        List of scaling violation descriptions.
    """
    checks = (
        ("compute_us", (scale + 1) * ROUND_HALF_UNIT_US),
        ("hbm_us", (scale + 1) * ROUND_HALF_UNIT_US),
        ("step_weight_gb", (scale + 1) * ROUND_HALF_UNIT_GB),
    )
    issues: list[str] = []
    for field_name, tol_bound in checks:
        unit_val = float(getattr(unit_est, field_name))
        full_val = float(getattr(full_est, field_name))
        if abs(full_val - scale * unit_val) > tol_bound:
            issues.append(
                f"{label} per_op {field_name}: unit={unit_val} x {scale} vs "
                f"full={full_val}"
            )
    return issues


def _strip_non_homogeneous_primitives(
    profile: advisor.ModelProfile,
) -> advisor.ModelProfile:
    """Clear fixed per-stack metadata so homogeneous layer scaling holds linearly."""
    hc = (
        dataclasses.replace(profile.hyper_connection, has_final_mixer=False)
        if profile.hyper_connection is not None
        else None
    )
    sp = (
        dataclasses.replace(
            profile.sparse_attn,
            kv_owners=(),
            index_owners=(),
            window_layers=(),
            kv_owner_ring_bytes_per_req=0,
        )
        if profile.sparse_attn is not None
        else None
    )
    return dataclasses.replace(
        profile,
        host_tables=(),
        ced_boundary=0,
        hyper_connection=hc,
        sparse_attn=sp,
    )


def _check_gcd_scaling_for_preset(
    slug: str,
    profile: advisor.ModelProfile,
    default_chip: advisor.ChipSpec,
) -> list[str]:
    """Verify full stack equals `gcd` times repeating unit for `pp == 1` layouts.

    Returns:
        List of gcd-unit scaling violations.
    """
    homog_profile = _strip_non_homogeneous_primitives(profile)
    counts: dict[str, int] = {
        "num_layers": homog_profile.num_layers,
        "num_linear_layers": homog_profile.num_linear_layers,
        "num_attention_layers": homog_profile.num_attention_layers,
        "num_indexer_layers": homog_profile.num_indexer_layers,
    }
    if homog_profile.num_moe_layers > 0:
        counts["num_moe_layers"] = homog_profile.num_moe_layers
    layer_gcd = 0
    for count in counts.values():
        layer_gcd = math.gcd(layer_gcd, count)
    if layer_gcd <= 1:
        return []
    unit_profile = dataclasses.replace(
        homog_profile, **{name: val // layer_gcd for name, val in counts.items()}
    )
    issues: list[str] = []
    for phase in CASE_PHASES:
        for lay_name, axes in case_layouts(slug, phase):
            if axes["pp"] != 1:
                continue
            wl = case_workload(phase, axes)
            unit_est = _scope_by_name(
                advisor.evaluate_scopes(unit_profile, wl, default_chip), "per_op"
            )
            full_est = _scope_by_name(
                advisor.evaluate_scopes(homog_profile, wl, default_chip), "per_op"
            )
            issues.extend(
                _check_scaled_per_op(
                    f"{slug}/{phase}/{lay_name}", unit_est, full_est, layer_gcd
                )
            )
    return issues


def _check_one_layer_anchor_for_preset(
    slug: str,
    profile: advisor.ModelProfile,
    default_chip: advisor.ChipSpec,
) -> list[str]:
    """Verify N-layer homogeneous stacks scale cleanly from the 1-layer anchor.

    Returns:
        List of 1-layer anchor scaling violations.
    """
    homog_profile = _strip_non_homogeneous_primitives(profile)
    kinds: list[tuple[str, dict[str, int]]] = []
    if homog_profile.num_linear_layers > 0:
        kinds.append(
            (
                "linear",
                {
                    "num_linear_layers": 1,
                    "num_attention_layers": 0,
                    "num_indexer_layers": 0,
                },
            )
        )
    if homog_profile.num_attention_layers > 0:
        kinds.append(
            (
                "attention",
                {
                    "num_linear_layers": 0,
                    "num_attention_layers": 1,
                    "num_indexer_layers": 1
                    if homog_profile.num_indexer_layers > 0
                    else 0,
                },
            )
        )
    layer_count = homog_profile.num_layers
    issues: list[str] = []
    for phase in ("decode", "prefill"):
        tp_ep_axes = dict(case_layouts(slug, phase))["tp_ep_only"]
        wl = case_workload(phase, tp_ep_axes)
        for kind_name, base_fields in kinds:
            one_fields = dict(base_fields)
            if homog_profile.num_moe_layers != 0:
                one_fields["num_moe_layers"] = 1
            one_prof = dataclasses.replace(homog_profile, num_layers=1, **one_fields)
            stack_prof = dataclasses.replace(
                homog_profile,
                num_layers=layer_count,
                **{name: val * layer_count for name, val in one_fields.items()},
            )
            one_est = _scope_by_name(
                advisor.evaluate_scopes(one_prof, wl, default_chip), "per_op"
            )
            stack_est = _scope_by_name(
                advisor.evaluate_scopes(stack_prof, wl, default_chip), "per_op"
            )
            issues.extend(
                _check_scaled_per_op(
                    f"{slug}/{phase}/{kind_name}", one_est, stack_est, layer_count
                )
            )
    return issues


def test_homogeneous_stack_scales_with_layer_count() -> None:
    """Verify gcd repeating unit and 1-layer anchor stacks scale linearly."""
    default_chip = advisor.resolve_chip_spec("tpu7x")
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        violations.extend(_check_gcd_scaling_for_preset(slug, profile, default_chip))
        violations.extend(
            _check_one_layer_anchor_for_preset(slug, profile, default_chip)
        )
    assert not violations, (
        f"Layer scaling violations ({len(violations)}): {violations[:5]}"
    )


def test_microbatched_step_equals_sum_of_calls() -> None:
    """Verify batch=128 at cap=32 equals 4x batch=32 for pp=1 and pp=2."""
    default_chip = advisor.resolve_chip_spec("tpu7x")
    max_core_rank = 16
    single_batch = 32
    quad_batch = 128
    call_count = 4
    tol_bound = (call_count + 1) * ROUND_HALF_UNIT_US
    fields = ("step_bound_us", "hbm_us", "compute_us", "fill_drain_us")
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        core_layout = minimal_valid_core_layout(
            advisor.DEPLOYMENT_PRESETS[slug].decode, max_rank=max_core_rank
        )
        dp_val = core_layout.dp
        base_axes = {
            "tp": core_layout.tp,
            "ep": core_layout.ep,
            "dp": dp_val,
            "pp": 1,
            "dcp": 1,
            "pcp": 1,
        }
        for pp_size in (1, 2):
            axes = {**base_axes, "pp": pp_size}
            wl_one = case_workload("decode", axes, batch=single_batch * dp_val)
            wl_four = case_workload(
                "decode",
                axes,
                batch=quad_batch * dp_val,
                max_rows_per_call=single_batch,
            )
            w_one = _scope_by_name(
                advisor.evaluate_scopes(profile, wl_one, default_chip), "whole_step"
            )
            w_four = _scope_by_name(
                advisor.evaluate_scopes(profile, wl_four, default_chip), "whole_step"
            )
            for field_name in fields:
                v_one = float(getattr(w_one, field_name))
                v_four = float(getattr(w_four, field_name))
                if abs(v_four - call_count * v_one) > tol_bound:
                    violations.append(
                        f"{slug}/pp{pp_size} {field_name}: 1call={v_one} vs "
                        f"4calls={v_four}"
                    )
    assert not violations, (
        f"Microbatch additivity violations ({len(violations)}): {violations[:5]}"
    )


def test_dtype_aliases_price_identically() -> None:
    """Verify FP8 aliases and empty expert_dtype fallback price identically."""
    default_chip = advisor.resolve_chip_spec("tpu7x")
    max_diff = 2 * ROUND_HALF_UNIT_US
    weight_dtypes = sorted(
        {
            "bf16",
            *(p.weight_dtype for p in advisor.MODEL_PRESETS.values()),
            *(p.expert_dtype for p in advisor.MODEL_PRESETS.values() if p.expert_dtype),
        }
    )
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        alias_kw: dict[str, str] = {}
        if profile.kv_dtype == "fp8":
            alias_kw["kv_dtype"] = "fp8_e4m3fn"
        if profile.act_dtype == "fp8":
            alias_kw["act_dtype"] = "fp8_e4m3fn"
        if alias_kw:
            aliased_profile = dataclasses.replace(profile, **alias_kw)
            for phase in ("decode", "prefill"):
                preset_axes = dict(case_layouts(slug, phase))["preset"]
                wl = case_workload(phase, preset_axes)
                base_ests = advisor.evaluate_scopes(profile, wl, default_chip)
                alias_ests = advisor.evaluate_scopes(aliased_profile, wl, default_chip)
                violations.extend(
                    _compare_numeric_estimates(
                        f"{slug}/{phase}/{alias_kw}", base_ests, alias_ests, max_diff
                    )
                )
        if profile.num_experts <= 0:
            continue
        for phase in CASE_PHASES:
            preset_axes = dict(case_layouts(slug, phase))["preset"]
            wl = case_workload(phase, preset_axes)
            for wdt in weight_dtypes:
                empty_exp = dataclasses.replace(
                    profile, weight_dtype=wdt, expert_dtype=""
                )
                explicit_exp = dataclasses.replace(
                    profile, weight_dtype=wdt, expert_dtype=wdt
                )
                ea = advisor.evaluate_scopes(empty_exp, wl, default_chip)
                eb = advisor.evaluate_scopes(explicit_exp, wl, default_chip)
                violations.extend(
                    _compare_numeric_estimates(
                        f"{slug}/{phase}/w={wdt}", ea, eb, max_diff
                    )
                )
    assert not violations, (
        f"Dtype alias violations ({len(violations)}): {violations[:5]}"
    )


def test_core_legality_follows_rows_per_call() -> None:
    """Verify core row-cap legality compares effective per-call rows to core max."""
    default_chip = advisor.resolve_chip_spec("tpu7x")
    max_core_rank = 16
    max_rows = constants.CORE_LIMITS.max_batch_rows
    over_batch = 2 * max_rows
    over_cap = 8 * max_rows
    split_batch = 16
    split_spec = 3
    split_cap = 32
    row_blocker = "rows_exceed_core_max_batch_rows"
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = dataclasses.replace(
            advisor.resolve_model_profile(slug), state_dtype="bf16"
        )
        core_layout = minimal_valid_core_layout(
            advisor.DEPLOYMENT_PRESETS[slug].decode, max_rank=max_core_rank
        )
        dp_val = core_layout.dp
        core_axes = {
            "tp": core_layout.tp,
            "ep": core_layout.ep,
            "dp": dp_val,
            "pp": 1,
            "dcp": 1,
            "pcp": 1,
        }
        wl_over = case_workload(
            "decode",
            core_axes,
            batch=over_batch * dp_val,
            max_rows_per_call=over_cap,
        )
        whole_over = _scope_by_name(
            advisor.evaluate_scopes(profile, wl_over, default_chip), "whole_step"
        )
        if row_blocker not in whole_over.blockers:
            violations.append(
                f"{slug}: {over_batch} rows with cap={over_cap} missed {row_blocker}"
            )
        wl_split = case_workload(
            "verify",
            core_axes,
            batch=split_batch * dp_val,
            spec_tokens=split_spec,
            max_rows_per_call=split_cap,
        )
        whole_split = _scope_by_name(
            advisor.evaluate_scopes(profile, wl_split, default_chip), "whole_step"
        )
        if row_blocker in whole_split.blockers:
            violations.append(
                f"{slug}: microbatched cap={split_cap} tripped {row_blocker}"
            )
    assert not violations, (
        f"Core row-cap violations ({len(violations)}): {violations[:5]}"
    )


def test_phase_unused_axis_is_rejected_or_inert() -> None:
    """Verify unused PCP in decode/verify leaves validation and legality identical."""
    default_chip = advisor.resolve_chip_spec("tpu7x")
    base_meshes: tuple[tuple[int, int, int], ...] = ((4, 8, 8), (8, 8, 4), (1, 1, 4))
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        dec = advisor.DEPLOYMENT_PRESETS[slug].decode
        meshes = (*base_meshes, (dec.tp, dec.ep, dec.dcp))
        for phase in ("decode", "verify"):
            for tp_val, ep_val, dcp_val in meshes:
                axes = {
                    "tp": tp_val,
                    "ep": ep_val,
                    "dp": 1,
                    "pp": 1,
                    "dcp": dcp_val,
                    "pcp": 1,
                }
                wl_p1 = case_workload(phase, axes, pcp=1)
                wl_p2 = case_workload(phase, axes, pcp=2)
                res_p1 = vmem.collect_validation_reasons(profile, wl_p1, default_chip)
                res_p2 = vmem.collect_validation_reasons(profile, wl_p2, default_chip)
                if res_p1 != res_p2:
                    violations.append(
                        f"{slug}/{phase} tp={tp_val} ep={ep_val} dcp={dcp_val}: "
                        f"pcp=1 {res_p1} != pcp=2 {res_p2}"
                    )
                    continue
                if not res_p1[1]:
                    est_p1 = _scope_by_name(
                        advisor.evaluate_scopes(profile, wl_p1, default_chip),
                        "whole_step",
                    )
                    est_p2 = _scope_by_name(
                        advisor.evaluate_scopes(profile, wl_p2, default_chip),
                        "whole_step",
                    )
                    if (est_p1.support, est_p1.blockers) != (
                        est_p2.support,
                        est_p2.blockers,
                    ):
                        violations.append(
                            f"{slug}/{phase} tp={tp_val} ep={ep_val} dcp={dcp_val} "
                            f"legality: pcp=1 {(est_p1.support, est_p1.blockers)} "
                            f"!= pcp=2 {(est_p2.support, est_p2.blockers)}"
                        )
    assert not violations, (
        f"Unused axis validation violations ({len(violations)}): {violations[:5]}"
    )


def test_spec_bridge_no_import_error_fallback_and_no_dead_expert_mode_helper() -> None:
    """Verify spec_bridge omits ImportError fallbacks and roofline omits dead helper."""
    advisor_dir = Path(spec_bridge.__file__).resolve().parent
    spec_bridge_src = (advisor_dir / "spec_bridge.py").read_text(encoding="utf-8")
    assert "except ImportError" not in spec_bridge_src
    assert "MegakernelPlannerUnavailableError" not in spec_bridge_src
    assert not hasattr(spec_bridge, "MegakernelPlannerUnavailableError")
    assert not hasattr(advisor, "MegakernelPlannerUnavailableError")
    assert "MegakernelPlannerUnavailableError" not in advisor.__all__

    roofline_tree = ast.parse((advisor_dir / "roofline.py").read_text(encoding="utf-8"))
    defined_fns = {
        node.name
        for node in ast.walk(roofline_tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_scope_context_for_expert_mode" not in defined_fns
    assert not hasattr(roofline, "_scope_context_for_expert_mode")
