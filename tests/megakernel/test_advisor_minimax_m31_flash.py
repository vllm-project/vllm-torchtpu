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
"""Verification tests for the MiniMax-M3.1-Flash megakernel advisor preset."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants, models, roofline, vmem
from vllm_torchtpu.megakernel.core import errors

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "minimax-m3.1-flash.json"

BYTES_PER_GB = 1e9
GIB = float(1024**3)
UNBOUNDED_RATE = 1e30

EXPECTED_NUM_LAYERS = 60
EXPECTED_MOE_LAYERS = 57
EXPECTED_HIDDEN_SIZE = 6144
EXPECTED_DENSE_INTER = 12_288
EXPECTED_ATTN_HEADS = 64
EXPECTED_KV_HEADS = 4
EXPECTED_HEAD_DIM = 128
EXPECTED_NUM_EXPERTS = 128
EXPECTED_TOP_K = 4
EXPECTED_MOE_INTER = 3072
EXPECTED_SHARED_INTER = 3072
EXPECTED_EXPERT_BLOCK_K = 16
EXPECTED_ROUTED_GIB = 216.421875
EXPECTED_DECODER_PARAMS = 423_727_202_304

EXPECTED_MSA_WINDOW = 128
EXPECTED_MSA_SELECT = 2048
EXPECTED_MSA_ATTENDED_TOKENS = EXPECTED_MSA_WINDOW + EXPECTED_MSA_SELECT
EXPECTED_INDEX_HEADS = 4
EXPECTED_INDEX_KV_HEADS = 1
EXPECTED_INDEX_DIM = 128
EXPECTED_IDX_PROJ_PARAMS = 235_929_600

EXPECTED_INT4_G1 = 228.0
EXPECTED_INT4_G2 = 356.0
# Per-stored-token bytes at TP8 x DCP8: dcp_kv_heads = max(1, 8 // (8 // 4)) = 4
# (612.0 B) vs TP8 x DCP4 where dcp_kv_heads = max(1, 4 // (8 // 4)) = 2 (356.0 B).
EXPECTED_INT4_G4 = 612.0
EXPECTED_FP8_G1 = 384.0
EXPECTED_FP8_G2 = 640.0
EXPECTED_FP8_G4 = 1152.0
EXPECTED_BF16_G1 = 768.0
EXPECTED_BF16_G2 = 1280.0
EXPECTED_BF16_G4 = 2304.0
EXPECTED_KV_BYTE_TOL = 0.6

TP_4 = 4
TP_EP_8 = 8
DP_2 = 2
DCP_2 = 2
DCP_4 = 4
DCP_8 = 8
DCP_16 = 16
PCP_2 = 2
SPEC_TOKENS_7 = 7
PREFILL_WINDOW_2K = 2048
CTX_1K = 1024
CTX_2K = 2048
CTX_4K = 4096
CTX_8K = 8192
CTX_32K = 32768
BATCH_8 = 8
BATCH_16 = 16


def _scope_by_name(
    report: advisor.AdvisorReport, scope_name: advisor.FusionScope
) -> advisor.ScopeEstimate:
    """Return one ScopeEstimate from `report` matching `scope_name`.

    Returns:
        Matching `ScopeEstimate` entry.
    """
    return next(scope for scope in report.scopes if scope.scope == scope_name)


def _eval_scope_by_name(
    estimates: tuple[advisor.ScopeEstimate, ...],
    scope_name: advisor.FusionScope,
) -> advisor.ScopeEstimate:
    """Return one ScopeEstimate from `estimates` matching `scope_name`.

    Returns:
        Matching `ScopeEstimate` entry.
    """
    return next(est for est in estimates if est.scope == scope_name)


def test_mxfp8_dense_preserves_nvfp4_expert_block_k() -> None:
    """Verify mxfp8 dense quant keeps dense block_k=0 and NVFP4 expert block_k=16."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    assert resolved.name == "minimax-m3.1-flash"
    assert resolved.weight_dtype == "fp8"
    assert resolved.expert_dtype == "nvfp4"
    assert resolved.weight_block_k == 0
    assert resolved.expert_weight_block_k == EXPECTED_EXPERT_BLOCK_K

    prof_k0 = models.finalize_model_profile(replace(resolved, expert_weight_block_k=0))
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    vec_only_chip = replace(
        tpu7x,
        bf16_flops_per_s=UNBOUNDED_RATE,
        fp8_flops_per_s=UNBOUNDED_RATE,
        int8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
    )
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_16,
        context=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dcp=DCP_4,
        dispatch_regime="zero_floor",
    )
    per_op_k16 = _eval_scope_by_name(
        advisor.evaluate_scopes(resolved, workload, vec_only_chip), "per_op"
    )
    per_op_k0 = _eval_scope_by_name(
        advisor.evaluate_scopes(prof_k0, workload, vec_only_chip), "per_op"
    )
    assert per_op_k16.compute_us > per_op_k0.compute_us


def _assert_msa_kv_bytes_for_dtypes(
    preset: models.ModelProfile, chip: advisor.ChipSpec
) -> None:
    """Assert per-stored-token per-layer per-rank MSA bytes across KV dtypes."""
    prof_int4 = models.resolve_model_profile(
        "minimax-m3.1-flash", kv_dtype="int4_per_token_head"
    )
    prof_bf16 = models.resolve_model_profile("minimax-m3.1-flash", kv_dtype="bf16")
    expected_bytes_by_kv: dict[str, tuple[float, float, float]] = {
        "int4_per_token_head": (EXPECTED_INT4_G1, EXPECTED_INT4_G2, EXPECTED_INT4_G4),
        "fp8": (EXPECTED_FP8_G1, EXPECTED_FP8_G2, EXPECTED_FP8_G4),
        "bf16": (EXPECTED_BF16_G1, EXPECTED_BF16_G2, EXPECTED_BF16_G4),
    }
    for kv_prof, expected_triple in (
        (prof_int4, expected_bytes_by_kv["int4_per_token_head"]),
        (preset, expected_bytes_by_kv["fp8"]),
        (prof_bf16, expected_bytes_by_kv["bf16"]),
    ):
        measured_bytes: list[float] = []
        for dcp_deg in (DCP_2, DCP_4, DCP_8):
            wl_lo = advisor.WorkloadSpec(
                phase="decode",
                batch=BATCH_8,
                context=CTX_1K,
                tp=TP_EP_8,
                ep=TP_EP_8,
                dcp=dcp_deg,
            )
            wl_hi = replace(wl_lo, context=CTX_32K)
            delta_bytes = (
                advisor.advise(kv_prof, wl_hi, chip).hbm_used_gb
                - advisor.advise(kv_prof, wl_lo, chip).hbm_used_gb
            ) * BYTES_PER_GB
            measured_bytes.append(
                delta_bytes
                * dcp_deg
                / (BATCH_8 * EXPECTED_NUM_LAYERS * (CTX_32K - CTX_1K))
            )
        assert tuple(measured_bytes) == pytest.approx(
            expected_triple, abs=EXPECTED_KV_BYTE_TOL
        )


def test_minimax_m31_mxfp8_nvfp4_and_int4_kv_end_to_end() -> None:
    """Verify preset resolution, HF parity, Wide-DCP, 4-bit KV, and regime sweep."""
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    dep = models.DEPLOYMENT_PRESETS["minimax-m3.1-flash"]
    assert preset.sparse_attn is not None
    assert preset.sparse_attn.kind == "msa"
    assert preset.sparse_attn.window_tokens == EXPECTED_MSA_WINDOW
    assert preset.sparse_attn.select_tokens == EXPECTED_MSA_SELECT
    assert preset.sparse_attn.index_heads == EXPECTED_INDEX_HEADS
    assert preset.sparse_attn.index_kv_heads == EXPECTED_INDEX_KV_HEADS
    assert preset.sparse_attn.index_dim == EXPECTED_INDEX_DIM
    assert preset.dcp_span == "tp_group"
    assert preset.weight_dtype == "fp8"
    assert preset.expert_dtype == "nvfp4"
    assert preset.weight_block_k == 0
    assert preset.expert_weight_block_k == EXPECTED_EXPERT_BLOCK_K

    assert (
        dep.decode.tp,
        dep.decode.ep,
        dep.decode.dcp,
        dep.decode.spec_tokens,
    ) == (TP_EP_8, TP_EP_8, DCP_4, SPEC_TOKENS_7)
    assert (dep.prefill.tp, dep.prefill.ep, dep.prefill.dcp) == (
        TP_EP_8,
        TP_EP_8,
        1,
    )
    assert dep.saturation is not None
    assert (dep.saturation.tp, dep.saturation.ep, dep.saturation.dcp) == (
        TP_EP_8,
        TP_EP_8,
        DCP_8,
    )
    assert dep.prefill_windows == (PREFILL_WINDOW_2K,)

    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    assert resolved.name == "minimax-m3.1-flash"
    for alias in (
        "minimax-m3.1-flash",
        "minimax-m3.1-flash-fp8",
        "MiniMaxAI/MiniMax-M3.1-Flash",
    ):
        assert models.resolve_model_profile(alias).name == "minimax-m3.1-flash"

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    _assert_msa_kv_bytes_for_dtypes(preset, tpu7x)
    prof_int4 = models.resolve_model_profile(
        "minimax-m3.1-flash", kv_dtype="int4_per_token_head"
    )

    assert (
        vmem.wide_dcp_kv_heads_per_rank(EXPECTED_KV_HEADS, TP_EP_8, DCP_4, "tp_group")
        == DCP_2
    )
    assert (
        vmem.wide_dcp_kv_heads_per_rank(EXPECTED_KV_HEADS, TP_EP_8, DCP_8, "tp_group")
        == DCP_4
    )
    prof_bf16 = models.resolve_model_profile("minimax-m3.1-flash", kv_dtype="bf16")
    bf16_only_chip = replace(
        tpu7x,
        fp8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        vector_f32_ops_per_s=UNBOUNDED_RATE,
    )
    rot_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_16,
        context=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dcp=DCP_4,
        dispatch_regime="zero_floor",
    )
    int4_bf16_us = _eval_scope_by_name(
        advisor.evaluate_scopes(prof_int4, rot_wl, bf16_only_chip), "per_op"
    ).compute_us
    bf16_ref_us = _eval_scope_by_name(
        advisor.evaluate_scopes(prof_bf16, rot_wl, bf16_only_chip), "per_op"
    ).compute_us
    fp8_bf16_us = _eval_scope_by_name(
        advisor.evaluate_scopes(preset, rot_wl, bf16_only_chip), "per_op"
    ).compute_us
    assert int4_bf16_us > bf16_ref_us > fp8_bf16_us

    sweep = advisor.advise_regimes(preset, tpu7x)
    assert (
        sweep.parallelism.tp,
        sweep.parallelism.ep,
        sweep.parallelism.dcp,
    ) == (TP_EP_8, TP_EP_8, DCP_4)
    assert sweep.prefill_parallelism is not None
    assert (
        sweep.prefill_parallelism.tp,
        sweep.prefill_parallelism.ep,
        sweep.prefill_parallelism.dcp,
    ) == (TP_EP_8, TP_EP_8, 1)
    assert sweep.saturation_parallelism is not None
    assert (
        sweep.saturation_parallelism.tp,
        sweep.saturation_parallelism.ep,
        sweep.saturation_parallelism.dcp,
    ) == (TP_EP_8, TP_EP_8, DCP_8)
    prefill_win = next(
        entry
        for entry in sweep.regimes
        if entry.label == f"Preset Prefill Window (W={PREFILL_WINDOW_2K})"
    )
    assert prefill_win.hbm_fits is True
    decode_med = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Med")
    )
    assert decode_med.hbm_fits is True


def test_minimax_m31_flash_geometry_and_param_counts() -> None:
    """Verify MiniMax-M3.1-Flash geometry, NVFP4 routed size, and decoder params."""
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_moe_layers == EXPECTED_MOE_LAYERS
    assert preset.hidden_size == EXPECTED_HIDDEN_SIZE
    assert preset.intermediate_size == EXPECTED_DENSE_INTER
    assert preset.num_attention_heads == EXPECTED_ATTN_HEADS
    assert preset.num_kv_heads == EXPECTED_KV_HEADS
    assert preset.head_dim == EXPECTED_HEAD_DIM
    assert preset.num_experts == EXPECTED_NUM_EXPERTS
    assert preset.top_k == EXPECTED_TOP_K
    assert preset.moe_intermediate_size == EXPECTED_MOE_INTER
    assert preset.shared_expert_intermediate_size == EXPECTED_SHARED_INTER
    assert "msa" in preset.layer_families
    assert "dense_prefix_moe" in preset.layer_families

    routed_params = (
        preset.num_moe_layers
        * preset.num_experts
        * 3
        * preset.moe_intermediate_size
        * preset.hidden_size
    )
    routed_gib = routed_params * constants.DTYPE_BYTES["nvfp4"] / GIB
    assert routed_gib == pytest.approx(EXPECTED_ROUTED_GIB, rel=1e-5)

    sa = preset.sparse_attn
    assert sa is not None
    assert sa.window_tokens + sa.select_tokens == EXPECTED_MSA_ATTENDED_TOKENS
    idx_params = (
        preset.num_layers
        * (sa.index_heads + sa.index_kv_heads)
        * sa.index_dim
        * preset.hidden_size
    )
    assert idx_params == EXPECTED_IDX_PROJ_PARAMS
    assert roofline.compute_decoder_param_count(preset) == EXPECTED_DECODER_PARAMS


def test_minimax_m31_flash_wide_dcp_validation_and_comm() -> None:
    """Verify Wide-DCP validation rules, tp=4 dp=2 KV bytes, and DCP collectives."""
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    tpu7x = advisor.resolve_chip_spec("tpu7x")

    wl_tp4_lo = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_8,
        context=CTX_1K,
        tp=TP_4,
        dp=DP_2,
        ep=TP_EP_8,
        dcp=1,
    )
    wl_tp4_hi = replace(wl_tp4_lo, context=CTX_32K)
    tp4_delta_b = (
        advisor.advise(preset, wl_tp4_hi, tpu7x).hbm_used_gb
        - advisor.advise(preset, wl_tp4_lo, tpu7x).hbm_used_gb
    ) * BYTES_PER_GB
    tp4_local_batch = BATCH_8 // DP_2
    tp4_per_tok = tp4_delta_b / (
        tp4_local_batch * EXPECTED_NUM_LAYERS * (CTX_32K - CTX_1K)
    )
    assert tp4_per_tok == pytest.approx(EXPECTED_FP8_G1, abs=EXPECTED_KV_BYTE_TOL)

    dcp1_rep = advisor.advise(
        preset,
        advisor.WorkloadSpec(
            phase="decode", batch=BATCH_8, context=CTX_4K, tp=TP_EP_8, ep=TP_EP_8, dcp=1
        ),
        tpu7x,
    )
    dcp4_rep = advisor.advise(
        preset,
        advisor.WorkloadSpec(
            phase="decode",
            batch=BATCH_8,
            context=CTX_4K,
            tp=TP_EP_8,
            ep=TP_EP_8,
            dcp=DCP_4,
        ),
        tpu7x,
    )
    assert (
        _scope_by_name(dcp4_rep, "per_op").comm_us
        > _scope_by_name(dcp1_rep, "per_op").comm_us
    )

    bad_tp_group_wl = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        pcp=PCP_2,
        dcp=DCP_16,
    )
    reasons, has_hard = vmem.collect_validation_reasons(preset, bad_tp_group_wl, tpu7x)
    assert has_hard is True
    assert "dcp_not_dividing_tp" in reasons
    with pytest.raises(errors.MegakernelConfigError, match="dcp_not_dividing_tp"):
        vmem.validate_advise_inputs(preset, bad_tp_group_wl, tpu7x)

    glm = models.resolve_model_profile("glm-5.2")
    glm_wide = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        pcp=PCP_2,
        dcp=DCP_16,
    )
    assert (
        "dcp_not_dividing_tp"
        not in vmem.collect_validation_reasons(glm, glm_wide, tpu7x)[0]
    )


def test_minimax_m31_flash_core_readiness_and_cli_parity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify core readiness separation and CLI JSON parity for MiniMax-M3.1-Flash."""
    wl = advisor.WorkloadSpec(
        phase="verify",
        batch=BATCH_8,
        spec_tokens=SPEC_TOKENS_7,
        context=CTX_8K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dcp=DCP_4,
        batch_explicit=True,
    )
    rep = advisor.advise("minimax-m3.1-flash", wl, "tpu7x")
    assert rep.hbm_fits is True
    assert rep.active_tokens == BATCH_8 * (SPEC_TOKENS_7 + 1)
    assert rep.core_support == "outside_core_limits"
    assert rep.core_ready is False
    assert rep.core_ready_scope == "per_op"
    assert "family_not_in_core" in rep.core_blockers
    assert "dtype_not_in_core" in rep.core_blockers
    assert (
        _scope_by_name(rep, "whole_step").step_achievable_us
        < _scope_by_name(rep, "per_op").step_achievable_us
    )

    rc = advisor.main(
        [
            "--model",
            "minimax-m3.1-flash",
            "--phase",
            "verify",
            "--batch",
            str(BATCH_8),
            "--spec-tokens",
            str(SPEC_TOKENS_7),
            "--context",
            str(CTX_8K),
            "--tp",
            str(TP_EP_8),
            "--ep",
            str(TP_EP_8),
            "--dcp",
            str(DCP_4),
            "--kv-dtype",
            "int4_per_token_head",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    api_rep = advisor.advise(
        models.resolve_model_profile(
            "minimax-m3.1-flash", kv_dtype="int4_per_token_head"
        ),
        wl,
        "tpu7x",
    )
    assert cli_payload == json.loads(json.dumps(api_rep.to_dict()))

    sweep_rc = advisor.main(
        ["--model", "minimax-m3.1-flash", "--regime-sweep", "--json"]
    )
    assert sweep_rc == 0
    sweep_cli = json.loads(capsys.readouterr().out)
    sweep_api = json.loads(
        json.dumps(advisor.advise_regimes("minimax-m3.1-flash", "tpu7x").to_dict())
    )
    assert sweep_cli == sweep_api
