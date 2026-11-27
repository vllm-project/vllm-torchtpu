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
"""Oracle conservation, CSA2 KV, Engram, and CED tests for DeepSeek-V4.1-Flash."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import load_locked_config
from vllm_torchtpu.roofline import cli
from vllm_torchtpu.roofline.constants import (
    BF16_BYTES,
    FP8_BYTES,
    FP8_UE8M0_32X32_BYTES,
    FP8_UE8M0_BYTES,
    MXFP4_BYTES,
)
from vllm_torchtpu.roofline.hf_loader import audit_hf_config
from vllm_torchtpu.roofline.memory import kv_bytes_per_core_parts
from vllm_torchtpu.roofline.modules import pass_costs
from vllm_torchtpu.roofline.partition import stage_slice_geometry

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_DSV41_FIXTURE = _FIXTURES_DIR / "deepseek_v4_1_flash.json"
_GIB = float(1024**3)
_UNIT_BYTE_TOL = 1e-6

# Ground-truth structural constants from deepseek-ai/DeepSeek-V4.1-Flash config.json.
_TRUNK_LAYERS = 40
_HIDDEN_SIZE = 5120
_VOCAB_SIZE = 129_280
_NUM_HEADS = 64
_HEAD_DIM = 512
_Q_LORA_RANK = 1280
_O_LORA_RANK = 1024
_O_GROUPS = 8
_MOE_INTER = 2304
_ROUTED_EXPERTS = 384
_SHARED_EXPERTS = 1
_TOP_K = 6
_INDEX_HEADS = 32
_INDEX_HEAD_DIM = 128
_INDEX_TOPK = 512
_SLIDING_WINDOW = 128
_HC_MULT = 4
_SWIGLU_MATS = 3

_KV_SOURCE_LAYERS = (2, 8, 14, 20)
_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
_CANDIDATE_SCAN_CAP = 2048 * 8
_CED_BOUNDARY = 20
_CSA2_INDEX_RECORD_BYTES = 132.0
_CSA2_ALIGNED_LATENT_BYTES = 640.0
_CSA2_SWA_BYTES_PER_TOK = float(_HEAD_DIM) * BF16_BYTES

_ENGRAM_LAYER_IDS = (1, 14)
_ENGRAM_ROWS_L1 = 384_006_168
_ENGRAM_ROWS_L14 = 384_016_682
_ENGRAM_HEAD_DIM = 256
_ENGRAM_N_HEADS = 8
_ENGRAM_NGRAM_ORDERS = 3
_ENGRAM_PROJ_WIDTH = _ENGRAM_NGRAM_ORDERS * _ENGRAM_N_HEADS * _ENGRAM_HEAD_DIM
_ENGRAM_H2D_PER_LAYER_BYTES = float(_ENGRAM_PROJ_WIDTH) * FP8_UE8M0_BYTES
_ENGRAM_TOTAL_H2D_BYTES = 2.0 * _ENGRAM_H2D_PER_LAYER_BYTES
_ENGRAM_DENSE_PROJ_PER_LAYER = (_HC_MULT + 1) * _HIDDEN_SIZE * _ENGRAM_PROJ_WIDTH
_ENGRAM_HOST_GIB_EXPECTED = 188.833
_ENGRAM_FIRST_RANK_GIB_EXPECTED = 11.802
_HOST_GIB_TOL = 0.002

_DRAFT_LAYERS = 3
_DRAFT_EXPERTS = 128
_DRAFT_TOP_K = 3
_DRAFT_BLOCK_SIZE = 5
_DRAFT_FC_IN_MULT = 3
_DRAFT_MARKOV_RANK = 256

_UNIT_DTYPES = rf.Dtypes(
    w_dense=1.0,
    w_expert=1.0,
    w_aux=1.0,
    kv=1.0,
    state=1.0,
    conv=1.0,
    router=1.0,
    lm_head=1.0,
)
_UNIT_PARALLEL = rf.Parallel(tp=1, pp=1, ep=1, moe="tp_slice")
_DP_LOCAL_RANKS = 8
_KV_TEST_CONTEXT = 2048
_KV_READ_CONTEXT = 65536
_KV_READ_BATCH = 4
_ALT_INDEX_HEAD_DIM = 64
_ALT_INDEX_RECORD_BYTES = 68.0
_UE8M0_BLOCK_SIDE = 32.0
_CED_SHORT_PREFILL = 64
_CED_MED_PREFILL = 4096
_CED_LONG_PREFILL = 8192
_PP4_BOUNDARIES = ((0, 10), (10, 20), (20, 30), (30, 40))


def _load_dsv41_config() -> dict[str, object]:
    """Load the locked DeepSeek-V4.1-Flash HuggingFace config JSON."""
    return json.loads(_DSV41_FIXTURE.read_text(encoding="utf-8"))


def _expected_csa2_layer_elems() -> float:
    """Return exact per-layer CSA2 attention weight elements (wq_a/b, wkv, wo_a/b)."""
    wq_a = _Q_LORA_RANK * _HIDDEN_SIZE
    wq_b = _NUM_HEADS * _HEAD_DIM * _Q_LORA_RANK
    wkv = _HEAD_DIM * _HIDDEN_SIZE
    wo_a = (_O_GROUPS * _O_LORA_RANK) * ((_NUM_HEADS // _O_GROUPS) * _HEAD_DIM)
    wo_b = _HIDDEN_SIZE * (_O_GROUPS * _O_LORA_RANK)
    return float(wq_a + wq_b + wkv + wo_a + wo_b)


def _expected_indexer_layer_elems() -> float:
    """Return exact per-layer DSA indexer weight elements."""
    return float(
        _Q_LORA_RANK * _INDEX_HEADS * _INDEX_HEAD_DIM
        + _HIDDEN_SIZE * _INDEX_HEAD_DIM
        + _HIDDEN_SIZE * _INDEX_HEADS
    )


def _assert_dsv41_specs(
    geometry: rf.ModelGeometry,
    dtypes: rf.Dtypes,
) -> None:
    """Verify CSA2 sparse attention, Engram host tables, and DSpark draft specs."""
    assert geometry.model_id == "deepseek_v41"
    assert geometry.num_layers == _TRUNK_LAYERS
    assert geometry.layer_types == ("csa2",) * _TRUNK_LAYERS
    assert len(geometry.compress_ratios) == _TRUNK_LAYERS
    assert dtypes.w_dense == pytest.approx(
        FP8_BYTES + FP8_BYTES / (_UE8M0_BLOCK_SIDE * _UE8M0_BLOCK_SIDE)
    )
    assert dtypes.w_dense == pytest.approx(FP8_UE8M0_32X32_BYTES)
    assert dtypes.w_expert == pytest.approx(MXFP4_BYTES)
    assert dtypes.lm_head == pytest.approx(BF16_BYTES)
    assert dtypes.act == "a8"

    sparse = geometry.sparse_attn
    assert sparse is not None
    assert sparse.kind == "csa2"
    assert sparse.kv_source_layer_ids == _KV_SOURCE_LAYERS
    assert sparse.index_source_layer_ids == _INDEX_SOURCE_LAYERS
    assert sparse.candidate_scan_cap == _CANDIDATE_SCAN_CAP
    assert sparse.index_record_bytes == pytest.approx(_CSA2_INDEX_RECORD_BYTES)
    assert sparse.ced_boundary == _CED_BOUNDARY
    assert sparse.swa_bytes_per_tok == pytest.approx(_CSA2_SWA_BYTES_PER_TOK)
    assert sparse.o_lora_rank == _O_LORA_RANK
    assert sparse.o_groups == _O_GROUPS

    assert len(geometry.host_tables) == len(_ENGRAM_LAYER_IDS)
    expected_rows = (_ENGRAM_ROWS_L1, _ENGRAM_ROWS_L14)
    for spec, layer_id, rows in zip(
        geometry.host_tables, _ENGRAM_LAYER_IDS, expected_rows, strict=True
    ):
        assert spec.layer_ids == (layer_id,)
        assert spec.elem_count == rows * _ENGRAM_HEAD_DIM
        assert spec.shard_axis == "dp_local"
        assert spec.default_hbm_resident is False
        assert spec.h2d_bytes_per_tok == pytest.approx(_ENGRAM_H2D_PER_LAYER_BYTES)
        assert spec.dense_proj_params == _ENGRAM_DENSE_PROJ_PER_LAYER
    total_h2d = sum(spec.h2d_bytes_per_tok for spec in geometry.host_tables)
    assert total_h2d == pytest.approx(_ENGRAM_TOTAL_H2D_BYTES)

    draft = geometry.draft
    assert draft is not None
    assert draft.kind == "dspark"
    assert draft.num_layers == _DRAFT_LAYERS
    assert draft.layer_kind == "csa2"
    assert draft.num_experts == _DRAFT_EXPERTS
    assert draft.top_k == _DRAFT_TOP_K
    assert draft.block_size == _DRAFT_BLOCK_SIZE
    assert draft.fc_in_mult == _DRAFT_FC_IN_MULT
    assert draft.markov_rank == _DRAFT_MARKOV_RANK


def _assert_dsv41_param_and_engram_conservation(
    geometry: rf.ModelGeometry,
    dtypes: rf.Dtypes,
    chip: rf.ChipSpec,
) -> None:
    """Verify exact oracle parameter sums, Engram GiB totals, and unit conservation."""
    params = rf.param_counts(geometry)
    csa2_layer = _expected_csa2_layer_elems()
    idx_layer = _expected_indexer_layer_elems()
    exp_mat = float(_SWIGLU_MATS * _MOE_INTER * _HIDDEN_SIZE)

    expected_main_attn = float(_TRUNK_LAYERS) * csa2_layer + (
        float(len(_INDEX_SOURCE_LAYERS)) * idx_layer
    )
    expected_main_routed = float(_TRUNK_LAYERS * _ROUTED_EXPERTS) * exp_mat
    expected_main_shared = float(_TRUNK_LAYERS * _SHARED_EXPERTS) * exp_mat
    expected_embed = float(_VOCAB_SIZE * _HIDDEN_SIZE)
    expected_engram_elems = float(
        (_ENGRAM_ROWS_L1 + _ENGRAM_ROWS_L14) * _ENGRAM_HEAD_DIM
    )
    expected_engram_proj = float(2 * _ENGRAM_DENSE_PROJ_PER_LAYER)
    expected_mtp = (
        float(_DRAFT_LAYERS) * csa2_layer
        + float(_DRAFT_LAYERS * _DRAFT_EXPERTS) * exp_mat
        + float(_DRAFT_LAYERS * _SHARED_EXPERTS) * exp_mat
        + float(_DRAFT_LAYERS * _DRAFT_EXPERTS * _HIDDEN_SIZE)
        + float(_DRAFT_LAYERS * (_HC_MULT**2) * _HIDDEN_SIZE)
        + float(_DRAFT_FC_IN_MULT * _HIDDEN_SIZE * _HIDDEN_SIZE)
        + float(2 * _VOCAB_SIZE * _DRAFT_MARKOV_RANK)
    )

    assert params.attention == pytest.approx(expected_main_attn)
    assert params.moe_routed_ffn == pytest.approx(expected_main_routed)
    assert params.moe_shared_ffn == pytest.approx(expected_main_shared)
    assert params.embedding == pytest.approx(expected_embed)
    assert params.lm_head == pytest.approx(expected_embed)
    assert params.host_tables == pytest.approx(
        expected_engram_elems + expected_engram_proj
    )
    assert params.mtp == pytest.approx(expected_mtp)

    # Aggregate checkpoint totals match manifest oracle (within omitted norms/scales).
    trunk_with_engram_proj = params.total_params - params.mtp - expected_engram_elems
    trunk_plus_engram = params.total_params - params.mtp
    inactive_trunk = float(_TRUNK_LAYERS * (_ROUTED_EXPERTS - _TOP_K)) * exp_mat
    trunk_active_excl_embed = trunk_with_engram_proj - inactive_trunk - expected_embed
    assert trunk_with_engram_proj / 1e9 == pytest.approx(551.83, abs=0.1)
    assert trunk_plus_engram / 1e9 == pytest.approx(748.44, abs=0.1)
    assert params.total_params / 1e9 == pytest.approx(762.67, abs=0.1)
    assert trunk_active_excl_embed / 1e9 == pytest.approx(16.08, abs=0.1)

    # Exact unit-dtype parameter conservation across all host-table residency modes.
    for residency in ("auto", "hbm", "host", "first"):
        offloaded = rf.offloaded_host_table_params(geometry, residency)
        unit_hbm = rf.hbm_breakdown_per_core_bytes(
            geometry,
            _UNIT_DTYPES,
            _UNIT_PARALLEL,
            rf.SeqKvLayout(0, 0, host_table_residency=residency),
            chip=chip,
        )
        assert abs((params.total_params - offloaded) - unit_hbm.weights) <= (
            _UNIT_BYTE_TOL
        )

    # Engram physical HBM footprint: 188.833 GiB total, 11.802 GiB/rank under first.
    hbm_all = rf.hbm_breakdown_per_core_bytes(
        geometry,
        dtypes,
        _UNIT_PARALLEL,
        rf.SeqKvLayout(0, 0, host_table_residency="hbm"),
        chip=chip,
    )
    hbm_host = rf.hbm_breakdown_per_core_bytes(
        geometry,
        dtypes,
        _UNIT_PARALLEL,
        rf.SeqKvLayout(0, 0, host_table_residency="host"),
        chip=chip,
    )
    engram_total_gib = (hbm_all.weights - hbm_host.weights) / _GIB
    assert engram_total_gib == pytest.approx(
        _ENGRAM_HOST_GIB_EXPECTED, abs=_HOST_GIB_TOL
    )

    par_dp8 = rf.Parallel(tp=1, dp=_DP_LOCAL_RANKS, ep=_DP_LOCAL_RANKS, moe="ep")
    hbm_first_dp8 = rf.hbm_breakdown_per_core_bytes(
        geometry,
        dtypes,
        par_dp8,
        rf.SeqKvLayout(0, 0, host_table_residency="first"),
        chip=chip,
    )
    hbm_host_dp8 = rf.hbm_breakdown_per_core_bytes(
        geometry,
        dtypes,
        par_dp8,
        rf.SeqKvLayout(0, 0, host_table_residency="host"),
        chip=chip,
    )
    engram_first_rank_gib = (hbm_first_dp8.weights - hbm_host_dp8.weights) / _GIB
    assert engram_first_rank_gib == pytest.approx(
        _ENGRAM_FIRST_RANK_GIB_EXPECTED, abs=_HOST_GIB_TOL
    )


def _assert_dsv41_kv_and_ced_soundness(
    geometry: rf.ModelGeometry,
    dtypes: rf.Dtypes,
    chip: rf.ChipSpec,
) -> None:
    """Verify CSA2 KV bytes (640B latent + 132B idx + 1024B SWA) and CED soundness."""
    fp8_kv_dtypes = replace(dtypes, kv=FP8_BYTES, kv_dtype="fp8")
    par_dp8 = rf.Parallel(tp=1, dp=_DP_LOCAL_RANKS, ep=_DP_LOCAL_RANKS, moe="ep")
    kv_lin, kv_fix = kv_bytes_per_core_parts(
        geometry,
        fp8_kv_dtypes,
        par_dp8,
        _KV_TEST_CONTEXT,
        physical=True,
        chip=chip,
    )
    expected_lin = (
        float(len(_KV_SOURCE_LAYERS)) * _CSA2_ALIGNED_LATENT_BYTES
        + float(len(_INDEX_SOURCE_LAYERS)) * _CSA2_INDEX_RECORD_BYTES
    )
    expected_fix = (
        float(_TRUNK_LAYERS + _DRAFT_LAYERS)
        * float(_SLIDING_WINDOW)
        * _CSA2_SWA_BYTES_PER_TOK
    )
    assert kv_lin == pytest.approx(expected_lin)
    assert kv_fix == pytest.approx(expected_fix)

    # Decode KV read bytes charge full CSA2 latent + scan + SWA (no ratio scaling).
    mla_step = next(
        mod
        for mod in pass_costs(
            geometry,
            par_dp8,
            rf.StepShape(batch=_KV_READ_BATCH, context=_KV_READ_CONTEXT),
            fp8_kv_dtypes,
            chip=chip,
        )
        if mod.name == "mla"
    )
    mla_zero = next(
        mod
        for mod in pass_costs(
            geometry,
            par_dp8,
            rf.StepShape(batch=0, rows_per_seq=0, context=0),
            fp8_kv_dtypes,
            chip=chip,
        )
        if mod.name == "mla"
    )
    expected_decode_kv_read = (
        float(
            len(_KV_SOURCE_LAYERS) * _KV_READ_BATCH * min(_KV_READ_CONTEXT, _INDEX_TOPK)
        )
        * _CSA2_ALIGNED_LATENT_BYTES
        + float(
            len(_INDEX_SOURCE_LAYERS)
            * _KV_READ_BATCH
            * min(_KV_READ_CONTEXT, _CANDIDATE_SCAN_CAP)
        )
        * _CSA2_INDEX_RECORD_BYTES
        + float(_TRUNK_LAYERS * _KV_READ_BATCH * min(_KV_READ_CONTEXT, _SLIDING_WINDOW))
        * _CSA2_SWA_BYTES_PER_TOK
    )
    assert (mla_step.hbm_bytes - mla_zero.hbm_bytes) == pytest.approx(
        expected_decode_kv_read
    )

    opts = rf.FloorOptions(dtypes=fp8_kv_dtypes)
    # Pure decode is invariant under execution="ced".
    dec_full = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(batch=4, context=_KV_TEST_CONTEXT, execution="full"),
        opts,
    )
    dec_ced = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(batch=4, context=_KV_TEST_CONTEXT, execution="ced"),
        opts,
    )
    assert dec_ced.hard_floor_s == pytest.approx(dec_full.hard_floor_s)

    # Short prefill <= sliding_window is invariant under CED.
    short_full = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_SHORT_PREFILL, execution="full"
        ),
        opts,
    )
    short_ced = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_SHORT_PREFILL, execution="ced"
        ),
        opts,
    )
    assert short_ced.hard_floor_s == pytest.approx(short_full.hard_floor_s)

    # Long prefill > sliding_window strictly lowers CED floor and scales monotonically.
    med_ced = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_MED_PREFILL, execution="ced"
        ),
        opts,
    )
    long_full = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_LONG_PREFILL, execution="full"
        ),
        opts,
    )
    long_ced = rf.step_floor(
        geometry,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_LONG_PREFILL, execution="ced"
        ),
        opts,
    )
    assert 0.0 < med_ced.hard_floor_s < long_ced.hard_floor_s < long_full.hard_floor_s
    assert long_ced.serial_estimate_s < long_full.serial_estimate_s

    # Post-boundary stage (layers 20..39) stays invariant across prefill lengths >= SWA.
    post_stage = stage_slice_geometry(geometry, _CED_BOUNDARY, _TRUNK_LAYERS)
    assert post_stage.sparse_attn is not None
    assert post_stage.sparse_attn.ced_post_only is True
    post_med = rf.step_floor(
        post_stage,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_MED_PREFILL, execution="ced"
        ),
        opts,
    )
    post_long = rf.step_floor(
        post_stage,
        chip,
        par_dp8,
        rf.StepShape(
            batch=0, rows_per_seq=0, prefill_tokens=_CED_LONG_PREFILL, execution="ced"
        ),
        opts,
    )
    assert post_med.hard_floor_s == pytest.approx(post_long.hard_floor_s)


def test_deepseek_v41_flash_oracle_and_ced_soundness() -> None:
    """Verify DeepSeek-V4.1 identity, audit, oracle params, Engram, KV, and CED."""
    raw_cfg = _load_dsv41_config()
    identity = rf.resolve_model_identity(
        raw_cfg, model_ref="deepseek-ai/DeepSeek-V4.1-Flash"
    )
    assert identity.canonical_id == "deepseek_v41"
    assert identity.model_type == "deepseek_v41_text"

    audit = audit_hf_config(raw_cfg, model_ref="deepseek-ai/DeepSeek-V4.1-Flash")
    assert audit.onboarding_tier == "tier0_ready"
    assert audit.unmapped_structural_keys == ()
    assert audit.blocking_reasons == ()

    geometry = rf.from_hf_config(raw_cfg, model_ref="deepseek-ai/DeepSeek-V4.1-Flash")
    dtypes = rf.dtypes_from_hf_config(raw_cfg)
    chip = rf.CHIPS["tpu7x"]

    _assert_dsv41_specs(geometry, dtypes)
    _assert_dsv41_param_and_engram_conservation(geometry, dtypes, chip)
    _assert_dsv41_kv_and_ced_soundness(geometry, dtypes, chip)

    # Non-128 index_head_dim computes CSA2 index_record_bytes dynamically.
    alt_cfg = {
        **raw_cfg,
        "text_config": {
            **dict(raw_cfg["text_config"]),
            "index_head_dim": _ALT_INDEX_HEAD_DIM,
        },
    }
    alt_geom = rf.from_hf_config(alt_cfg, model_ref="deepseek-ai/DeepSeek-V4.1-Flash")
    assert alt_geom.sparse_attn is not None
    assert alt_geom.sparse_attn.index_record_bytes == pytest.approx(
        _ALT_INDEX_RECORD_BYTES
    )


def test_deepseek_v41_pp_slicing_and_spec_cycle_invariants() -> None:
    """Verify PP4 stage conservation (zero-KV stage) and DSpark spec-cycle bounds."""
    raw_cfg = _load_dsv41_config()
    geometry = rf.from_hf_config(raw_cfg)
    dtypes = replace(rf.dtypes_from_hf_config(raw_cfg), kv=FP8_BYTES, kv_dtype="fp8")
    chip = rf.CHIPS["tpu7x"]
    par = rf.Parallel(tp=1, dp=_DP_LOCAL_RANKS, ep=_DP_LOCAL_RANKS, moe="ep")

    stages = [
        stage_slice_geometry(geometry, start, end) for start, end in _PP4_BOUNDARIES
    ]
    # Stage 3 (layers 30..39) holds zero KV sources and owns the DSpark draft module.
    assert stages[-1].sparse_attn is not None
    assert stages[-1].sparse_attn.kv_source_layer_ids == ()
    assert stages[-1].sparse_attn.index_source_layer_ids == (2, 6)
    assert all(stage.draft is None for stage in stages[:-1])
    assert stages[-1].draft == geometry.draft

    head_bytes = float(2 * geometry.vocab * geometry.hidden) * dtypes.lm_head
    for residency in ("auto", "hbm", "host", "first"):
        layout = rf.SeqKvLayout(
            1,
            _KV_TEST_CONTEXT,
            physical_kv_layout=True,
            host_table_residency=residency,
        )
        full_bd = rf.hbm_breakdown_per_core_bytes(
            geometry, dtypes, par, layout, chip=chip
        )
        stage_bds = [
            rf.hbm_breakdown_per_core_bytes(stage, dtypes, par, layout, chip=chip)
            for stage in stages
        ]
        sum_weights = (
            sum(bd.weights for bd in stage_bds) - float(len(stages) - 1) * head_bytes
        )
        assert sum_weights == pytest.approx(full_bd.weights)
        assert sum(bd.kv for bd in stage_bds) == pytest.approx(full_bd.kv)
        assert sum(bd.state for bd in stage_bds) == pytest.approx(full_bd.state)

    # Speculative DSpark cycle (spec_k=5): PCIe H2D stays out of hard_floor_s.
    opts = rf.FloorOptions(dtypes=dtypes)
    fl_host = rf.spec_cycle_floor(
        geometry,
        chip,
        par,
        rf.SpecShape(
            batch=4,
            context=_KV_TEST_CONTEXT,
            spec_k=_DRAFT_BLOCK_SIZE,
            execution="ced",
            host_table_residency="host",
        ),
        opts,
    )
    fl_first = rf.spec_cycle_floor(
        geometry,
        chip,
        par,
        rf.SpecShape(
            batch=4,
            context=_KV_TEST_CONTEXT,
            spec_k=_DRAFT_BLOCK_SIZE,
            execution="ced",
            host_table_residency="first",
        ),
        opts,
    )
    assert fl_host.pcie_s > fl_first.pcie_s > 0.0
    assert fl_host.hard_floor_s <= fl_first.hard_floor_s
    assert fl_host.hard_floor_s <= fl_host.envelope_floor_s <= fl_host.serial_estimate_s
    assert (
        fl_first.hard_floor_s <= fl_first.envelope_floor_s <= fl_first.serial_estimate_s
    )


def test_glm_rejects_csa2_sparse_attention_keys() -> None:
    """Verify non-CSA2 models fail closed when CSA2 sparse attention keys appear."""
    glm_cfg = dict(load_locked_config("glm_5_2_fp8.json"))
    glm_cfg["index_source_layer_ids"] = [0]
    glm_cfg["o_lora_rank"] = 512
    with pytest.raises(rf.UnsupportedArchitectureError, match="CSA2"):
        rf.from_hf_config(glm_cfg)


@pytest.mark.parametrize(
    ("overrides", "match_pattern"),
    [
        ({"kv_source_layer_ids": [40]}, "kv_source_layer_ids"),
        ({"index_source_layer_ids": [-1]}, "index_source_layer_ids"),
        ({"engram_layer_ids": [40]}, "engram_layer_ids"),
        ({"candidate_source_layer_id": 40}, "candidate_source_layer_id"),
        ({"engram_num_embeddings": [1000]}, "engram_num_embeddings"),
        ({"engram_head_dim": 0, "engram_n_heads": 0}, "engram_head_dim"),
        ({"compress_ratios": [-1.0] + [1.0] * (_TRUNK_LAYERS - 1)}, "compress_ratios"),
    ],
)
def test_deepseek_v41_fail_closed_csa2_and_engram_validation(
    overrides: dict[str, object],
    match_pattern: str,
) -> None:
    """Verify DeepSeek-V4.1 rejects out-of-range IDs, lengths, dims, and negatives."""
    raw_cfg = _load_dsv41_config()
    bad_text = {**dict(raw_cfg["text_config"]), **overrides}
    bad_cfg = {**raw_cfg, "text_config": bad_text}
    with pytest.raises(ValueError, match=match_pattern):
        rf.from_hf_config(bad_cfg)


def test_deepseek_v41_cli_ced_and_first_residency_solve_differences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --execution ced and --host-table-residency first change CLI outputs."""
    base_args = [
        "solve",
        "--config",
        str(_DSV41_FIXTURE),
        "--example-scenario",
        "chat-8k1k",
        "--mode",
        "mixed",
        "--max-chips",
        "16",
        "--concurrency",
        "1",
        "--json",
    ]
    assert cli.main(base_args) == 0
    base_payload = json.loads(capsys.readouterr().out)

    ced_args = [*base_args[:-1], "--execution", "ced", "--json"]
    assert cli.main(ced_args) == 0
    ced_payload = json.loads(capsys.readouterr().out)

    first_args = [*base_args[:-1], "--host-table-residency", "first", "--json"]
    assert cli.main(first_args) == 0
    first_payload = json.loads(capsys.readouterr().out)

    assert ced_payload["recommended"] != base_payload["recommended"]
    assert first_payload["recommended"] != base_payload["recommended"]
    ced_flags = {
        e["flag"]: e["value"] for e in ced_payload["provenance"]["supplied_values"]
    }
    first_flags = {
        e["flag"]: e["value"] for e in first_payload["provenance"]["supplied_values"]
    }
    assert ced_flags["--execution"] == "ced"
    assert first_flags["--host-table-residency"] == "first"
