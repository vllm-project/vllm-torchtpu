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
"""Verification tests for the Qwen3.8-Flash-Next-FP8 megakernel advisor preset."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants, models, roofline, vmem

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "qwen3.8-flash-next.json"

BYTES_PER_GB = 1e9
UNBOUNDED_RATE = 1e30
ROUND_TOL_US = 0.02

EXPECTED_NUM_LAYERS = 48
EXPECTED_GDN_LAYERS = 36
EXPECTED_QSA_LAYERS = 12
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_ATTN_HEADS = 24
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_DIM = 256
EXPECTED_NUM_EXPERTS = 512
EXPECTED_TOP_K = 10
EXPECTED_MOE_INTER = 640
EXPECTED_WEIGHT_BLOCK_K = 128

EXPECTED_QSA_SELECT_TOKENS = 2051
EXPECTED_QSA_INDEX_HEADS = 4
EXPECTED_QSA_INDEX_KV_HEADS = 1
EXPECTED_QSA_INDEX_DIM = 128
EXPECTED_QSA_COMPRESS_RATIO = 4

EXPECTED_HC_STREAMS = 4
EXPECTED_HC_LOWRANK_DIM = 320
EXPECTED_HC_TOTAL_PARAMS = 639_631_360

EXPECTED_PLE_LAYER_IDX = 1
EXPECTED_PLE_TOTAL_BYTES = 51_200_245_760
EXPECTED_PLE_H2D_BYTES = 2560.0
EXPECTED_PLE_STATE_BYTES = 184_320
EXPECTED_PLE_DENSE_PROJ_PARAMS = 32_768_000

TP_EP_8 = 8
DCP_4 = 4
TP_2 = 2
DP_4 = 4
PP_8 = 8
SPEC_TOKENS_3 = 3
PREFILL_WINDOW_8K = 8192
CTX_4K = 4096
CTX_32K = 32768
CTX_128K = 131072
BATCH_8 = 8
BATCH_16 = 16
BATCH_64 = 64
MAX_ROWS_64 = 64
TWO_LAYER_GROUPS = 2
EXPECTED_OFFLOAD_HBM_TOL_GB = 0.01


def _scope_by_name(
    report: advisor.AdvisorReport, scope_name: advisor.FusionScope
) -> advisor.ScopeEstimate:
    """Return one ScopeEstimate from `report` matching `scope_name`.

    Returns:
        Matching `ScopeEstimate` entry.
    """
    return next(scope for scope in report.scopes if scope.scope == scope_name)


def test_qwen38_pp8_prefill_ple_offload_fits_hbm() -> None:
    """Verify PP8 prefill preset offloads PLE to CPU so per-rank HBM fits cleanly."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    dep = models.DEPLOYMENT_PRESETS["qwen3.8-flash-next"]
    assert dep.prefill.tp == 1
    assert dep.prefill.ep == 1
    assert dep.prefill.pp == PP_8
    assert dep.prefill.host_table_hbm_overrides == (("ple", False),)
    assert dep.prefill_windows == (PREFILL_WINDOW_8K,)

    base_prefill = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=PREFILL_WINDOW_8K,
        tp=dep.prefill.tp,
        ep=dep.prefill.ep,
        pp=dep.prefill.pp,
    )
    resident_wl = replace(base_prefill, host_table_hbm_overrides=(("ple", True),))
    offload_wl = replace(
        base_prefill, host_table_hbm_overrides=dep.prefill.host_table_hbm_overrides
    )

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    v6e = advisor.resolve_chip_spec("v6e")

    res_7x = advisor.advise(preset, resident_wl, tpu7x)
    off_7x = advisor.advise(preset, offload_wl, tpu7x)
    expected_ple_gb = EXPECTED_PLE_TOTAL_BYTES / BYTES_PER_GB
    assert (res_7x.hbm_used_gb - off_7x.hbm_used_gb) == pytest.approx(
        expected_ple_gb, abs=EXPECTED_OFFLOAD_HBM_TOL_GB
    )
    assert off_7x.hbm_fits is True
    assert off_7x.hbm_pool_overflow is False
    assert "fusion_break" in _scope_by_name(off_7x, "whole_step").blockers
    assert "fusion_break" not in _scope_by_name(res_7x, "whole_step").blockers

    res_v6e = advisor.advise(preset, resident_wl, v6e)
    off_v6e = advisor.advise(preset, offload_wl, v6e)
    assert res_v6e.hbm_fits is False
    assert off_v6e.hbm_fits is True


def test_qwen38_flash_snapshot_matches_preset_and_geometry() -> None:
    """Verify authentic HF snapshot, aliases, geometry, and dtypes match preset."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    assert resolved.name == "qwen3.8-flash-next"
    for alias in (
        "qwen3.8-flash",
        "qwen3.8-flash-next-fp8",
        "Qwen/Qwen3.8-Flash-Next-FP8",
    ):
        assert models.resolve_model_profile(alias).name == "qwen3.8-flash-next"

    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_linear_layers == EXPECTED_GDN_LAYERS
    assert preset.num_attention_layers == EXPECTED_QSA_LAYERS
    assert preset.hidden_size == EXPECTED_HIDDEN_SIZE
    assert preset.intermediate_size == 0
    assert preset.num_attention_heads == EXPECTED_ATTN_HEADS
    assert preset.num_kv_heads == EXPECTED_KV_HEADS
    assert preset.head_dim == EXPECTED_HEAD_DIM
    assert preset.num_experts == EXPECTED_NUM_EXPERTS
    assert preset.top_k == EXPECTED_TOP_K
    assert preset.moe_intermediate_size == EXPECTED_MOE_INTER
    assert preset.shared_expert_intermediate_size == EXPECTED_MOE_INTER
    assert preset.num_moe_layers == EXPECTED_NUM_LAYERS
    assert preset.has_attn_output_gate is True
    assert preset.weight_dtype == "bf16"
    assert preset.expert_dtype == "fp8"
    assert preset.act_dtype == "bf16"
    assert preset.expert_act_dtype == "fp8"
    assert preset.kv_dtype == "fp8"
    assert preset.state_dtype == "fp32"
    assert preset.weight_block_k == EXPECTED_WEIGHT_BLOCK_K
    assert models.default_serving_state_dtype(preset.model_type) == "fp32"


def test_qwen38_flash_sparse_hc_and_ple_specs() -> None:
    """Verify QSA sparse attention, low-rank HC, and PLE host-table specs."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    assert preset.sparse_attn is not None
    assert preset.sparse_attn.kind == "qsa"
    assert preset.sparse_attn.select_tokens == EXPECTED_QSA_SELECT_TOKENS
    assert preset.sparse_attn.index_heads == EXPECTED_QSA_INDEX_HEADS
    assert preset.sparse_attn.index_kv_heads == EXPECTED_QSA_INDEX_KV_HEADS
    assert preset.sparse_attn.index_dim == EXPECTED_QSA_INDEX_DIM
    assert preset.sparse_attn.index_compress_ratio == EXPECTED_QSA_COMPRESS_RATIO

    assert preset.hyper_connection is not None
    assert preset.hyper_connection.kind == "lowrank"
    assert preset.hyper_connection.num_streams == EXPECTED_HC_STREAMS
    assert preset.hyper_connection.lowrank_dim == EXPECTED_HC_LOWRANK_DIM
    assert preset.hyper_connection.has_final_mixer is True
    assert models.hyper_connection_total_params(preset) == EXPECTED_HC_TOTAL_PARAMS

    assert len(preset.host_tables) == 1
    ple = preset.host_tables[0]
    assert ple.name == "ple"
    assert ple.layers == (EXPECTED_PLE_LAYER_IDX,)
    assert ple.hbm_resident is True
    assert ple.shard_axis == "tp"
    assert ple.table_bytes_total == EXPECTED_PLE_TOTAL_BYTES
    assert ple.h2d_bytes_per_token == EXPECTED_PLE_H2D_BYTES
    assert ple.state_bytes_per_request == EXPECTED_PLE_STATE_BYTES
    assert ple.dense_proj_params == EXPECTED_PLE_DENSE_PROJ_PARAMS


def test_qwen38_qsa_attention_saturation_and_dcp() -> None:
    """Verify QSA sparse saturation slope and DCP KV cache sharding."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    assert "qsa" in preset.layer_families
    assert "dsa" not in preset.layer_families

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    fast_io = replace(
        tpu7x,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        vector_f32_ops_per_s=UNBOUNDED_RATE,
    )
    base_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_8,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dispatch_regime="zero_floor",
        op_overhead_us=0.0,
    )
    c_sat = _scope_by_name(
        advisor.advise(
            preset, replace(base_wl, context=EXPECTED_QSA_SELECT_TOKENS), fast_io
        ),
        "per_op",
    ).compute_us
    c_32k = _scope_by_name(
        advisor.advise(preset, replace(base_wl, context=CTX_32K), fast_io), "per_op"
    ).compute_us
    c_128k = _scope_by_name(
        advisor.advise(preset, replace(base_wl, context=CTX_128K), fast_io), "per_op"
    ).compute_us
    slope_lo = (c_32k - c_sat) / (CTX_32K - EXPECTED_QSA_SELECT_TOKENS)
    slope_hi = (c_128k - c_32k) / (CTX_128K - CTX_32K)
    slope_tol = (2.0 * ROUND_TOL_US) / (CTX_32K - EXPECTED_QSA_SELECT_TOKENS) + (
        2.0 * ROUND_TOL_US
    ) / (CTX_128K - CTX_32K)
    assert slope_lo == pytest.approx(slope_hi, abs=slope_tol)

    dcp1 = advisor.advise(preset, replace(base_wl, dcp=1, context=CTX_32K), tpu7x)
    dcp4 = advisor.advise(preset, replace(base_wl, dcp=DCP_4, context=CTX_32K), tpu7x)
    assert dcp4.hbm_used_gb < dcp1.hbm_used_gb
    assert _scope_by_name(dcp4, "per_op").comm_us > 0.0


def test_qwen38_lowrank_hc_and_decode_ple_residency() -> None:
    """Verify low-rank HC params/VMEM and decode PLE HBM residency vs offload."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    no_hc = models.finalize_model_profile(
        replace(preset, layer_families=(), hyper_connection=None, residual_streams=1)
    )
    assert (
        roofline.compute_decoder_param_count(preset)
        - roofline.compute_decoder_param_count(no_hc)
        == EXPECTED_HC_TOTAL_PARAMS
    )

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    resident_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=BATCH_8,
        context=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dcp=DCP_4,
    )
    assert (
        vmem.compute_per_token_vmem_bytes(preset, resident_wl)["residual_buffer_bytes"]
        > vmem.compute_per_token_vmem_bytes(no_hc, resident_wl)["residual_buffer_bytes"]
    )

    offload_wl = replace(resident_wl, host_table_hbm_overrides=(("ple", False),))
    resident_rep = advisor.advise(preset, resident_wl, tpu7x)
    offload_rep = advisor.advise(preset, offload_wl, tpu7x)

    resident_ws = _scope_by_name(resident_rep, "whole_step")
    offload_ws = _scope_by_name(offload_rep, "whole_step")
    assert resident_ws.feasible is True
    assert "fusion_break" not in resident_ws.blockers
    assert offload_ws.feasible is False
    assert "fusion_break" in offload_ws.blockers
    assert _scope_by_name(offload_rep, "layer_group").ops_per_step == TWO_LAYER_GROUPS

    expected_tp8_ple_gb = EXPECTED_PLE_TOTAL_BYTES / (TP_EP_8 * BYTES_PER_GB)
    assert (resident_rep.hbm_used_gb - offload_rep.hbm_used_gb) == pytest.approx(
        expected_tp8_ple_gb, abs=EXPECTED_OFFLOAD_HBM_TOL_GB
    )


def test_qwen38_fp32_gdn_state_and_core_readiness() -> None:
    """Verify FP32 GDN state and QSA/HC separate physics winner from core limits."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    low_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=BATCH_8,
        spec_tokens=SPEC_TOKENS_3,
        context=CTX_4K,
        tp=TP_EP_8,
        ep=TP_EP_8,
        dcp=DCP_4,
    )
    low_rep = advisor.advise(preset, low_wl, "tpu7x")
    low_ws = _scope_by_name(low_rep, "whole_step")
    low_po = _scope_by_name(low_rep, "per_op")
    assert low_rep.hbm_fits is True
    assert low_rep.recommended_scope == "whole_step"
    assert low_rep.core_support == "outside_core_limits"
    assert low_rep.core_ready is False
    assert low_rep.core_ready_scope == "per_op"
    assert "family_not_in_core" in low_rep.core_blockers
    assert "kv_or_state_dtype_not_in_core" in low_rep.core_blockers
    assert low_ws.num_vmem_splits == 1
    assert low_ws.step_achievable_us < low_po.step_achievable_us

    split_wl = replace(
        low_wl,
        batch=BATCH_64,
        max_rows_per_call=MAX_ROWS_64,
    )
    split_rep = advisor.advise(preset, split_wl, "tpu7x")
    split_ws = _scope_by_name(split_rep, "whole_step")
    split_po = _scope_by_name(split_rep, "per_op")
    assert split_ws.num_vmem_splits > 1
    assert split_ws.step_achievable_us > split_po.step_achievable_us
    assert split_rep.recommended_scope == "single_layer"


def test_qwen38_cli_and_regime_sweep_parity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify deployment preset topologies, CLI JSON output, and sweep propagation."""
    dep = models.DEPLOYMENT_PRESETS["qwen3.8-flash-next"]
    assert (dep.decode.tp, dep.decode.ep, dep.decode.dcp, dep.decode.spec_tokens) == (
        TP_EP_8,
        TP_EP_8,
        DCP_4,
        SPEC_TOKENS_3,
    )
    assert dep.saturation is not None
    assert (dep.saturation.tp, dep.saturation.dp, dep.saturation.ep) == (
        TP_2,
        DP_4,
        TP_EP_8,
    )

    sweep = advisor.advise_regimes("qwen3.8-flash-next", "tpu7x")
    assert sweep.saturation_parallelism is not None
    assert sweep.saturation_parallelism.dp == DP_4
    assert sweep.prefill_parallelism is not None
    assert sweep.prefill_parallelism.pp == PP_8
    assert sweep.prefill_parallelism.host_table_hbm_overrides == (("ple", False),)

    prefill_window = next(
        entry
        for entry in sweep.regimes
        if entry.label == f"Preset Prefill Window (W={PREFILL_WINDOW_8K})"
    )
    assert prefill_window.hbm_fits is True
    assert prefill_window.workload["host_table_hbm_overrides"] == (("ple", False),)

    sat_entry = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    assert sat_entry.workload["batch"] == constants.SATURATION_BATCH_CAP

    rc = advisor.main(
        [
            "--model",
            "qwen3.8-flash-next",
            "--phase",
            "decode",
            "--batch",
            str(BATCH_16),
            "--context",
            str(CTX_4K),
            "--tp",
            str(TP_EP_8),
            "--ep",
            str(TP_EP_8),
            "--dcp",
            str(DCP_4),
            "--host-table-hbm",
            "ple=cpu",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    api_rep = advisor.advise(
        "qwen3.8-flash-next",
        advisor.WorkloadSpec(
            phase="decode",
            batch=BATCH_16,
            context=CTX_4K,
            tp=TP_EP_8,
            ep=TP_EP_8,
            dcp=DCP_4,
            host_table_hbm_overrides=(("ple", False),),
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert cli_payload == json.loads(json.dumps(api_rep.to_dict()))


def test_advise_regimes_qwen38_preserves_explicit_ple_hbm_override() -> None:
    """Verify explicit PLE HBM residency override survives prefill preset routing."""
    sweep = advisor.advise_regimes(
        "qwen3.8-flash-next",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(
                host_table_hbm_overrides=(("ple", True),)
            )
        ),
    )
    assert sweep.prefill_parallelism is not None
    assert sweep.prefill_parallelism.host_table_hbm_overrides == (("ple", True),)
    prefill_entries = [
        entry for entry in sweep.regimes if entry.workload.get("phase") == "prefill"
    ]
    assert prefill_entries
    for entry in prefill_entries:
        assert entry.workload["host_table_hbm_overrides"] == (("ple", True),)
