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
"""Verification tests for the DeepSeek-V4.1-Flash megakernel advisor preset."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants, models, roofline, vmem
from vllm_torchtpu.megakernel.core import errors

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "deepseek-v4.1-flash.json"

BYTES_PER_GB = 1e9
UNBOUNDED_RATE = 1e30

EXPECTED_NUM_LAYERS = 40
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_ATTN_HEADS = 64
EXPECTED_KV_HEADS = 1
EXPECTED_HEAD_DIM = 512
EXPECTED_NUM_EXPERTS = 384
EXPECTED_TOP_K = 6
EXPECTED_MOE_INTER = 2304
EXPECTED_Q_LORA_RANK = 1280
EXPECTED_QK_ROPE_DIM = 64
EXPECTED_O_LORA_RANK = 1024
EXPECTED_O_GROUPS = 8
EXPECTED_CED_BOUNDARY = 20
EXPECTED_RESIDUAL_STREAMS = 4
EXPECTED_WEIGHT_BLOCK_K = 32
EXPECTED_ATTN_LAYER_PARAMS = 126_615_552
EXPECTED_ROUTED_BYTES = 288_777_830_400.0
EXPECTED_DECODER_PARAMS = 550_396_231_680

EXPECTED_SWA_WINDOW = 128
EXPECTED_SWA_BYTES_PER_TOK = 1024.0
EXPECTED_SWA_FIXED_BYTES = 5_242_880
EXPECTED_SELECT_TOKENS = 512
EXPECTED_INDEX_HEADS = 32
EXPECTED_INDEX_DIM = 128
EXPECTED_LINEAR_KV_BYTES = 2148
EXPECTED_RING_STATE_BYTES = 12_288
EXPECTED_CANDIDATE_SCAN_CAP = 16_384
EXPECTED_CAPPED_INDEX_OWNERS = 4
EXPECTED_KV_OWNER_COUNT = 4
EXPECTED_INDEX_OWNER_COUNT = 8

EXPECTED_MHC_SINKHORN_ITERS = 20
EXPECTED_MHC_TOTAL_PARAMS = 39_321_600

EXPECTED_ENGRAM_H2D_BOTH = 12_672.0
EXPECTED_ENGRAM_H2D_FIRST = 6_336.0
EXPECTED_ENGRAM_L1_BYTES = 101_377_628_352
EXPECTED_ENGRAM_L14_BYTES = 101_380_404_048
EXPECTED_ENGRAM_L1_LAYER = 1
EXPECTED_ENGRAM_L14_LAYER = 14
EXPECTED_ENGRAM_GROUP_SIZES = (1, 13, 26)
EXPECTED_ENGRAM_TABLE_COUNT = 2
EXPECTED_HBM_FIRST_GROUP_COUNT = 2
EXPECTED_OFFLOAD_HBM_TOL_GB = 0.01

DP_8 = 8
EP_8 = 8
DP_2 = 2
DCP_4 = 4
SPEC_TOKENS_5 = 5
PREFILL_WINDOW_1K = 1024
CTX_4K = 4096
BATCH_8 = 8
BATCH_16 = 16
BATCH_256 = 256
SWA_SLOPE_STEP_TOL = 0.05


def _scope_by_name(
    report: advisor.AdvisorReport, scope_name: advisor.FusionScope
) -> advisor.ScopeEstimate:
    """Return one ScopeEstimate from `report` matching `scope_name`.

    Returns:
        Matching `ScopeEstimate` entry.
    """
    return next(scope for scope in report.scopes if scope.scope == scope_name)


def test_dsv41_sweep_uses_ced_execution_and_csa_rejects_dcp_gt_1() -> None:
    """Verify DeepSeek-V4.1-Flash sweeps propagate CED and CSA rejects dcp > 1."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    dep = models.DEPLOYMENT_PRESETS["deepseek-v4.1-flash"]
    assert (
        dep.decode.tp,
        dep.decode.dp,
        dep.decode.ep,
        dep.decode.spec_tokens,
        dep.decode.execution,
    ) == (1, DP_8, EP_8, SPEC_TOKENS_5, "ced")
    assert (
        dep.prefill.tp,
        dep.prefill.dp,
        dep.prefill.ep,
        dep.prefill.execution,
    ) == (1, DP_8, EP_8, "ced")
    assert dep.saturation is not None
    assert (
        dep.saturation.tp,
        dep.saturation.dp,
        dep.saturation.ep,
        dep.saturation.execution,
    ) == (1, DP_8, EP_8, "ced")
    assert dep.prefill_windows == (PREFILL_WINDOW_1K,)

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(preset, tpu7x)
    assert sweep.parallelism.execution == "ced"
    assert sweep.prefill_parallelism is not None
    assert sweep.prefill_parallelism.execution == "ced"
    assert sweep.saturation_parallelism is not None
    assert sweep.saturation_parallelism.execution == "ced"
    assert all(entry.workload["execution"] == "ced" for entry in sweep.regimes)

    prefill_win = next(
        entry
        for entry in sweep.regimes
        if entry.label == f"Preset Prefill Window (W={PREFILL_WINDOW_1K})"
    )
    assert prefill_win.hbm_fits is True
    full_prefill = advisor.advise(
        preset,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=DP_8,
            isl=PREFILL_WINDOW_1K,
            osl=constants.DEFAULT_PREFILL_OSL,
            chunk_size=PREFILL_WINDOW_1K,
            tp=1,
            dp=DP_8,
            ep=EP_8,
            execution="full",
            moe_mode="grouped_on_chip",
            moe_mode_explicit=True,
        ),
        tpu7x,
    )
    ced_per_op = next(sc for sc in prefill_win.scopes if sc["scope"] == "per_op")
    assert (
        float(str(ced_per_op["step_bound_us"]))
        < _scope_by_name(full_prefill, "per_op").step_bound_us
    )

    bad_dcp_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_8,
        context=CTX_4K,
        tp=DCP_4,
        dp=DP_2,
        ep=EP_8,
        dcp=DCP_4,
    )
    reasons, has_hard = vmem.collect_validation_reasons(preset, bad_dcp_wl, tpu7x)
    assert has_hard is True
    assert "csa_dcp_unsupported" in reasons
    with pytest.raises(errors.MegakernelConfigError, match="csa_dcp_unsupported"):
        vmem.validate_advise_inputs(preset, bad_dcp_wl, tpu7x)
    with pytest.raises(errors.MegakernelConfigError, match="csa_dcp_unsupported"):
        advisor.evaluate_scopes(preset, bad_dcp_wl, tpu7x)
    with pytest.raises(errors.MegakernelConfigError, match="csa_dcp_unsupported"):
        advisor.advise(preset, bad_dcp_wl, tpu7x)


def test_dsv41_flash_snapshot_matches_preset_and_geometry() -> None:
    """Verify authentic DeepSeek-V4.1-Flash snapshot, aliases, and parameter math."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    assert resolved.name == "deepseek-v4.1-flash"
    for alias in (
        "deepseek-v4.1-flash",
        "deepseek-v41-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
    ):
        assert models.resolve_model_profile(alias).name == "deepseek-v4.1-flash"

    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_moe_layers == EXPECTED_NUM_LAYERS
    assert preset.hidden_size == EXPECTED_HIDDEN_SIZE
    assert preset.intermediate_size == 0
    assert preset.num_attention_heads == EXPECTED_ATTN_HEADS
    assert preset.num_kv_heads == EXPECTED_KV_HEADS
    assert preset.head_dim == EXPECTED_HEAD_DIM
    assert preset.num_experts == EXPECTED_NUM_EXPERTS
    assert preset.top_k == EXPECTED_TOP_K
    assert preset.moe_intermediate_size == EXPECTED_MOE_INTER
    assert preset.shared_expert_intermediate_size == EXPECTED_MOE_INTER
    assert preset.q_lora_rank == EXPECTED_Q_LORA_RANK
    assert preset.qk_rope_head_dim == EXPECTED_QK_ROPE_DIM
    assert preset.o_lora_rank == EXPECTED_O_LORA_RANK
    assert preset.o_groups == EXPECTED_O_GROUPS
    assert preset.ced_boundary == EXPECTED_CED_BOUNDARY
    assert preset.residual_streams == EXPECTED_RESIDUAL_STREAMS
    assert preset.weight_dtype == "fp8"
    assert preset.expert_dtype == "mxfp4"
    assert preset.act_dtype == "fp8"
    assert preset.kv_dtype == "fp8"
    assert preset.weight_block_k == EXPECTED_WEIGHT_BLOCK_K

    attn_params = (
        preset.hidden_size * preset.q_lora_rank
        + preset.q_lora_rank * preset.num_attention_heads * preset.head_dim
        + preset.hidden_size * preset.num_kv_heads * preset.head_dim
        + preset.num_attention_heads * preset.head_dim * preset.o_lora_rank
        + preset.o_groups * preset.o_lora_rank * preset.hidden_size
    )
    assert attn_params == EXPECTED_ATTN_LAYER_PARAMS
    routed_params = (
        preset.num_moe_layers
        * preset.num_experts
        * 3
        * preset.moe_intermediate_size
        * preset.hidden_size
    )
    routed_gb = routed_params * constants.DTYPE_BYTES["mxfp4"] / BYTES_PER_GB
    assert routed_gb == pytest.approx(EXPECTED_ROUTED_BYTES / BYTES_PER_GB, rel=1e-6)
    assert roofline.compute_decoder_param_count(preset) == EXPECTED_DECODER_PARAMS


def test_dsv41_flash_csa_owners_and_mhc_sinkhorn() -> None:
    """Verify CSA2 owner conservation, candidate scan caps, and mHC Sinkhorn math."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    assert "csa" in preset.layer_families
    assert "mhc" in preset.layer_families
    assert "dsa" not in preset.layer_families
    assert models.has_dsa_profile(preset) is False

    sparse = preset.sparse_attn
    assert sparse is not None
    assert sparse.kind == "csa"
    assert sparse.window_tokens == EXPECTED_SWA_WINDOW
    assert sparse.window_bytes_per_token == EXPECTED_SWA_BYTES_PER_TOK
    assert len(sparse.window_layers) == EXPECTED_NUM_LAYERS
    assert sparse.select_tokens == EXPECTED_SELECT_TOKENS
    assert sparse.index_heads == EXPECTED_INDEX_HEADS
    assert sparse.index_kv_heads == EXPECTED_KV_HEADS
    assert sparse.index_dim == EXPECTED_INDEX_DIM
    assert sparse.shared_kv_mqa is True
    assert sparse.kv_owner_ring_bytes_per_req == EXPECTED_RING_STATE_BYTES
    assert len(sparse.kv_owners) == EXPECTED_KV_OWNER_COUNT
    assert len(sparse.index_owners) == EXPECTED_INDEX_OWNER_COUNT

    linear_kv = sum(
        (o.record_elems * o.elem_bytes + o.record_overhead_bytes) / o.record_tokens
        for o in (*sparse.kv_owners, *sparse.index_owners)
    )
    assert int(linear_kv) == EXPECTED_LINEAR_KV_BYTES
    swa_fixed = (
        len(sparse.window_layers) * sparse.window_tokens * sparse.window_bytes_per_token
    )
    assert int(swa_fixed) == EXPECTED_SWA_FIXED_BYTES
    capped_owners = [
        o.max_scan_records for o in sparse.index_owners if o.max_scan_records
    ]
    assert capped_owners == [EXPECTED_CANDIDATE_SCAN_CAP] * EXPECTED_CAPPED_INDEX_OWNERS

    hc = preset.hyper_connection
    assert hc is not None
    assert hc.kind == "mhc"
    assert hc.num_streams == EXPECTED_RESIDUAL_STREAMS
    assert hc.sinkhorn_iters == EXPECTED_MHC_SINKHORN_ITERS
    assert models.hyper_connection_total_params(preset) == EXPECTED_MHC_TOTAL_PARAMS

    no_hc = models.finalize_model_profile(
        replace(preset, layer_families=(), hyper_connection=None, residual_streams=1)
    )
    assert (
        roofline.compute_decoder_param_count(preset)
        - roofline.compute_decoder_param_count(no_hc)
        == EXPECTED_MHC_TOTAL_PARAMS
    )

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    fast_io = replace(
        tpu7x,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        bf16_flops_per_s=UNBOUNDED_RATE,
        fp8_flops_per_s=UNBOUNDED_RATE,
        int8_flops_per_s=UNBOUNDED_RATE,
    )
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_16,
        context=CTX_4K,
        tp=1,
        dp=DP_8,
        ep=EP_8,
        dispatch_regime="zero_floor",
        op_overhead_us=0.0,
    )
    assert _scope_by_name(advisor.advise(preset, wl, fast_io), "per_op").compute_us > (
        _scope_by_name(advisor.advise(no_hc, wl, fast_io), "per_op").compute_us
    )
    assert (
        vmem.compute_per_token_vmem_bytes(preset, wl)["residual_buffer_bytes"]
        > vmem.compute_per_token_vmem_bytes(no_hc, wl)["residual_buffer_bytes"]
    )


def test_dsv41_flash_engram_hbm_first_and_ced_fusion_breaks() -> None:
    """Verify Engram CPU offload vs HBM-first override and CED layer-20 break."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    assert len(preset.host_tables) == EXPECTED_ENGRAM_TABLE_COUNT
    eng_l1, eng_l14 = preset.host_tables
    assert eng_l1.name == "engram_l1"
    assert eng_l1.layers == (EXPECTED_ENGRAM_L1_LAYER,)
    assert eng_l1.table_bytes_total == EXPECTED_ENGRAM_L1_BYTES
    assert eng_l1.hbm_resident is False
    assert eng_l1.shard_axis == "dp_local"
    assert eng_l14.name == "engram_l14"
    assert eng_l14.layers == (EXPECTED_ENGRAM_L14_LAYER,)
    assert eng_l14.table_bytes_total == EXPECTED_ENGRAM_L14_BYTES
    assert eng_l14.hbm_resident is False

    chip = advisor.resolve_chip_spec("tpu7x")
    cpu_wl = advisor.WorkloadSpec(
        phase="decode", batch=BATCH_8, context=CTX_4K, tp=1, dp=DP_8, ep=EP_8
    )
    hbm_first_wl = replace(cpu_wl, host_table_hbm_overrides=(("engram_l1", True),))

    cpu_h2d = sum(
        tbl.h2d_bytes_per_token
        for tbl in preset.host_tables
        if not vmem.effective_host_table_hbm_resident(tbl, cpu_wl)
    )
    first_h2d = sum(
        tbl.h2d_bytes_per_token
        for tbl in preset.host_tables
        if not vmem.effective_host_table_hbm_resident(tbl, hbm_first_wl)
    )
    assert cpu_h2d == pytest.approx(EXPECTED_ENGRAM_H2D_BOTH, rel=1e-6)
    assert first_h2d == pytest.approx(EXPECTED_ENGRAM_H2D_FIRST, rel=1e-6)

    cpu_breaks = vmem.active_fusion_break_layers(preset, cpu_wl)
    assert cpu_breaks == (EXPECTED_ENGRAM_L1_LAYER, EXPECTED_ENGRAM_L14_LAYER)
    assert (
        vmem.fusion_break_layer_groups(preset.num_layers, cpu_breaks)
        == EXPECTED_ENGRAM_GROUP_SIZES
    )
    assert vmem.active_fusion_break_layers(preset, hbm_first_wl) == (
        EXPECTED_ENGRAM_L14_LAYER,
    )

    cpu_rep = advisor.advise(preset, cpu_wl, chip)
    hbm_first_rep = advisor.advise(preset, hbm_first_wl, chip)
    expected_l1_rank_gb = EXPECTED_ENGRAM_L1_BYTES / (DP_8 * BYTES_PER_GB)
    assert (hbm_first_rep.hbm_used_gb - cpu_rep.hbm_used_gb) == pytest.approx(
        expected_l1_rank_gb, abs=EXPECTED_OFFLOAD_HBM_TOL_GB
    )
    assert "fusion_break" in _scope_by_name(cpu_rep, "whole_step").blockers
    assert "fusion_break" in _scope_by_name(hbm_first_rep, "whole_step").blockers
    assert _scope_by_name(cpu_rep, "layer_group").ops_per_step == len(
        EXPECTED_ENGRAM_GROUP_SIZES
    )
    assert (
        _scope_by_name(hbm_first_rep, "layer_group").ops_per_step
        == EXPECTED_HBM_FIRST_GROUP_COUNT
    )

    pref_std = advisor.WorkloadSpec(
        phase="prefill",
        batch=BATCH_8,
        isl=PREFILL_WINDOW_1K,
        tp=1,
        dp=DP_8,
        ep=EP_8,
        execution="full",
    )
    pref_ced = replace(pref_std, execution="ced")
    assert EXPECTED_CED_BOUNDARY not in vmem.active_fusion_break_layers(
        preset, pref_std
    )
    assert EXPECTED_CED_BOUNDARY in vmem.active_fusion_break_layers(preset, pref_ced)
    std_rep = advisor.advise(preset, pref_std, chip)
    ced_rep = advisor.advise(preset, pref_ced, chip)
    assert (
        _scope_by_name(ced_rep, "per_op").step_bound_us
        < _scope_by_name(std_rep, "per_op").step_bound_us
    )


def test_dsv41_flash_dp8_batch_rounding_and_core_readiness() -> None:
    """Verify DP8 batch scaling, HBM fit, and core-readiness blocker separation."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="verify",
        batch=BATCH_8,
        spec_tokens=SPEC_TOKENS_5,
        context=CTX_4K,
        tp=1,
        dp=DP_8,
        ep=EP_8,
    )
    rep = advisor.advise(preset, wl, chip)
    assert rep.hbm_fits is True
    assert rep.active_tokens == SPEC_TOKENS_5 + 1
    assert rep.core_support == "outside_core_limits"
    assert rep.core_ready is False
    assert rep.core_ready_scope == "per_op"
    for expected_blocker in (
        "all_to_all_not_in_core",
        "family_not_in_core",
        "dtype_not_in_core",
    ):
        assert expected_blocker in rep.core_blockers
    whole_step = _scope_by_name(rep, "whole_step")
    assert "fusion_break" in whole_step.blockers

    sweep = advisor.advise_regimes(preset, chip)
    assert all(int(str(entry.workload["batch"])) % DP_8 == 0 for entry in sweep.regimes)
    sat_entry = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    assert sat_entry.workload["batch"] == constants.SATURATION_BATCH_CAP


def test_dsv41_flash_cli_and_sweep_parity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI JSON and regime-sweep JSON match Python APIs verbatim."""
    rc = advisor.main(
        [
            "--model",
            "deepseek-v4.1-flash",
            "--phase",
            "prefill",
            "--batch",
            str(BATCH_8),
            "--isl",
            str(PREFILL_WINDOW_1K),
            "--tp",
            "1",
            "--dp",
            str(DP_8),
            "--ep",
            str(EP_8),
            "--execution",
            "ced",
            "--host-table-hbm",
            "engram_l1=hbm",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    api_rep = advisor.advise(
        "deepseek-v4.1-flash",
        advisor.WorkloadSpec(
            phase="prefill",
            batch=BATCH_8,
            isl=PREFILL_WINDOW_1K,
            osl=constants.DEFAULT_PREFILL_OSL,
            tp=1,
            dp=DP_8,
            ep=EP_8,
            execution="ced",
            host_table_hbm_overrides=(("engram_l1", True),),
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert cli_payload == json.loads(json.dumps(api_rep.to_dict()))

    sweep_rc = advisor.main(
        ["--model", "deepseek-v4.1-flash", "--regime-sweep", "--json"]
    )
    assert sweep_rc == 0
    sweep_cli = json.loads(capsys.readouterr().out)
    sweep_api = json.loads(
        json.dumps(advisor.advise_regimes("deepseek-v4.1-flash", "tpu7x").to_dict())
    )
    assert sweep_cli == sweep_api


def test_advise_regimes_dsv41_preserves_explicit_full_execution() -> None:
    """Verify explicit execution='full' survives CED deployment presets."""
    sweep = advisor.advise_regimes(
        "deepseek-v4.1-flash",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(execution="full")
        ),
    )
    assert sweep.parallelism.execution == "full"
    assert sweep.prefill_parallelism is not None
    assert sweep.prefill_parallelism.execution == "full"
    assert sweep.regimes
    for entry in sweep.regimes:
        assert entry.workload["execution"] == "full"
