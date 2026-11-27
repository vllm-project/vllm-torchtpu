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
"""Pure-physics unit tests for Phase 3 roofline primitives and provenance."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline import test_model_lockin
from tests.roofline.conftest import load_locked_config
from vllm_torchtpu.roofline import cli, emit, memory, schema
from vllm_torchtpu.roofline.constants import (
    BF16_BYTES,
    FP4_BYTES,
    FP8_BYTES,
    KV_DTYPE_BYTES,
    MXFP4_BYTES,
    NVFP4_BYTES,
    WEIGHT_DTYPE_BYTES,
)
from vllm_torchtpu.roofline.modules import kv_bytes_per_token_per_layer
from vllm_torchtpu.roofline.partition import stage_slice_geometry
from vllm_torchtpu.roofline.solver import _SolveContext, _steady_state_batch_cap_phys

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_UNIT_BYTE_TOL = 1e-6
_MXFP8_EXPECTED = 1.0 + 1.0 / 32.0
_UE8M0_32X32_EXPECTED = FP8_BYTES + FP8_BYTES / (32.0 * 32.0)
_UE8M0_128X128_EXPECTED = FP8_BYTES + FP8_BYTES / (128.0 * 128.0)
_MSA_FP8_H1 = 384.0
_MSA_FP8_H2 = 640.0
_MSA_FP8_H4 = 1152.0
_MSA_INT4_H1 = 228.0
_MSA_INT4_H2 = 356.0
_MSA_INT4_H4 = 612.0
_CSA2_IDX_BYTES = 132.0
_CSA2_SWA_BYTES = 1024.0
_QSA_MAIN_BYTES = 1024.0
_QSA_COMP_K_BYTES = 128.0
_QSA_TOTAL_BYTES = 1152.0
_HOST_TABLE_L0_ELEMS = 10_000_000
_HOST_TABLE_L1_ELEMS = 12_000_000
_HOST_TABLE_PROJ_ELEMS = 500_000
_HOST_TABLE_H2D_PER_LAYER = 6336.0
_HOST_TABLE_STATE_BYTES = 184_320.0
_UE8M0_SCALE_RATIO = 1.0 / 32.0
_QSA_FIXED_STATE_BYTES = 524_288.0
_TP_SHARDED_HOST_H2D_BYTES = 1024.0
_TP_SHARDED_HOST_ELEMS = 8_000_000
_INT4_PTH_SCALE_BYTES = 32.0
_MARKOV_RANK = 256
_MARKOV_SPEC_K = 7
_CSA2_COMPRESS_RATIO = 0.5
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


def _base_msa_geometry() -> rf.ModelGeometry:
    """Build a compact MSA geometry mirroring 4-KV-head sparse attention."""
    sparse = rf.SparseAttnSpec(
        kind="msa",
        kv_source_layer_ids=(0, 1, 2, 3),
        index_source_layer_ids=(0, 1, 2, 3),
        sparse_block_size=128,
        sparse_topk_blocks=16,
        sparse_local_block=1,
        dcp_span="tp",
    )
    return rf.ModelGeometry(
        hidden=1024,
        num_layers=4,
        layer_types=("sparse_gqa",) * 4,
        num_heads=64,
        num_kv_heads=4,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        dense_inter=2048,
        vocab=32000,
        mtp_layers=0,
        index_n_heads=4,
        index_head_dim=128,
        sparse_attn=sparse,
    )


def _base_csa2_geometry() -> rf.ModelGeometry:
    """Build a compact CSA2 geometry with o_lora_rank, SWA, and CED boundary."""
    sparse = rf.SparseAttnSpec(
        kind="csa2",
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 2, 3),
        candidate_scan_cap=16384,
        index_record_bytes=_CSA2_IDX_BYTES,
        ced_boundary=2,
        swa_bytes_per_tok=_CSA2_SWA_BYTES,
        o_lora_rank=256,
        o_groups=4,
    )
    return rf.ModelGeometry(
        hidden=1024,
        num_layers=4,
        layer_types=("csa2",) * 4,
        num_heads=16,
        num_kv_heads=1,
        head_dim=512,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=16,
        top_k=2,
        moe_inter=512,
        shared_inter=512,
        vocab=32000,
        mtp_layers=0,
        q_lora_rank=256,
        qk_rope_head_dim=64,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=512,
        sliding_window=128,
        sparse_attn=sparse,
    )


def _base_qsa_geometry() -> rf.ModelGeometry:
    """Build a compact QSA geometry with compressed keys and raw ring state."""
    sparse = rf.SparseAttnSpec(
        kind="qsa",
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 3),
        compressed_key_bytes_per_tok=_QSA_COMP_K_BYTES,
        fixed_state_page_bytes=524_288.0,
    )
    return rf.ModelGeometry(
        hidden=1024,
        num_layers=4,
        layer_types=(
            "linear_attention",
            "indexed_attention",
            "linear_attention",
            "indexed_attention",
        ),
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        attn_output_gate=True,
        linear_k_heads=8,
        linear_v_heads=16,
        linear_k_dim=128,
        linear_v_dim=128,
        conv_kernel=4,
        num_experts=16,
        top_k=2,
        moe_inter=512,
        shared_inter=512,
        vocab=32000,
        mtp_layers=0,
        qsa_budget=2048,
        index_n_heads=4,
        index_head_dim=128,
        sparse_attn=sparse,
    )


def test_sparse_attn_indexer_and_int4_kv_bytes() -> None:
    """Verify MSA int4/FP8 KV bytes, CSA2 indexer/SWA/o_lora, and QSA compression."""
    assert KV_DTYPE_BYTES[rf.INT4_PER_TOKEN_HEAD_KV_DTYPE] == pytest.approx(FP4_BYTES)
    assert WEIGHT_DTYPE_BYTES["mxfp8"] == pytest.approx(_MXFP8_EXPECTED)
    assert pytest.approx(_MXFP8_EXPECTED) == rf.MXFP8_BYTES
    assert pytest.approx(_MXFP8_EXPECTED) == rf.FP8_UE8M0_BYTES

    msa_geom = _base_msa_geometry()
    fp8_dtypes = rf.Dtypes(kv=FP8_BYTES, kv_dtype="fp8")
    int4_dtypes = rf.Dtypes(kv=FP4_BYTES, kv_dtype=rf.INT4_PER_TOKEN_HEAD_KV_DTYPE)
    chip = rf.CHIPS["tpu7x"]

    # MSA per-token per-layer bytes across TP8 / DCP{1,2,4,8}.
    expected_fp8 = [
        (rf.Parallel(tp=8, kv_split=1), _MSA_FP8_H1),
        (rf.Parallel(tp=8, kv_split=2), _MSA_FP8_H1),
        (rf.Parallel(tp=8, kv_split=4), _MSA_FP8_H2),
        (rf.Parallel(tp=8, kv_split=8), _MSA_FP8_H4),
    ]
    for parallel, expected_bytes in expected_fp8:
        actual = kv_bytes_per_token_per_layer(
            msa_geom, fp8_dtypes, parallel, "gqa", physical=True, chip=chip
        )
        assert actual == pytest.approx(expected_bytes)

    expected_int4 = [
        (rf.Parallel(tp=8, kv_split=1), _MSA_INT4_H1),
        (rf.Parallel(tp=8, kv_split=2), _MSA_INT4_H1),
        (rf.Parallel(tp=8, kv_split=4), _MSA_INT4_H2),
        (rf.Parallel(tp=8, kv_split=8), _MSA_INT4_H4),
    ]
    for parallel, expected_bytes in expected_int4:
        actual = kv_bytes_per_token_per_layer(
            msa_geom, int4_dtypes, parallel, "gqa", physical=True, chip=chip
        )
        assert actual == pytest.approx(expected_bytes)

    # Bounded sparse attention beats dense GQA KV traffic at long context.
    dense_geom = replace(msa_geom, sparse_attn=None, layer_types=("gqa",) * 4)
    long_shape = rf.StepShape(batch=4, rows_per_seq=1, context=32768)
    par_tp8 = rf.Parallel(tp=8, kv_split=1)
    msa_floor = rf.step_floor(
        msa_geom, chip, par_tp8, long_shape, rf.FloorOptions(dtypes=fp8_dtypes)
    )
    dense_floor = rf.step_floor(
        dense_geom, chip, par_tp8, long_shape, rf.FloorOptions(dtypes=fp8_dtypes)
    )
    assert msa_floor.hbm_s < dense_floor.hbm_s
    assert msa_floor.hard_floor_s <= msa_floor.envelope_floor_s
    assert msa_floor.envelope_floor_s <= msa_floor.serial_estimate_s

    # CSA2 132 B indexer + 1024 B SWA window.
    csa_geom = _base_csa2_geometry()
    csa_kv_bd = rf.hbm_breakdown_per_core_bytes(
        csa_geom, fp8_dtypes, _UNIT_PARALLEL, rf.SeqKvLayout(1, 2048), chip=chip
    )
    expected_csa_kv = (
        2.0 * (512.0 + 64.0) * FP8_BYTES * 2048.0
        + 3.0 * _CSA2_IDX_BYTES * 2048.0
        + 4.0 * 128.0 * _CSA2_SWA_BYTES
    )
    assert csa_kv_bd.kv == pytest.approx(expected_csa_kv)

    # QSA 1024 B main + 128 B compressed keys + fixed state page bytes.
    qsa_geom = _base_qsa_geometry()
    qsa_per_tok = kv_bytes_per_token_per_layer(
        qsa_geom, fp8_dtypes, rf.Parallel(tp=8), "gqa", physical=True, chip=chip
    )
    assert qsa_per_tok == pytest.approx(_QSA_TOTAL_BYTES)
    qsa_bd = rf.hbm_breakdown_per_core_bytes(
        qsa_geom,
        fp8_dtypes,
        rf.Parallel(tp=8),
        rf.SeqKvLayout(1, 1024, physical_kv_layout=True),
        chip=chip,
    )
    assert qsa_bd.kv == pytest.approx(
        2.0 * 1024.0 * (_QSA_MAIN_BYTES + _QSA_COMP_K_BYTES)
    )
    assert qsa_bd.state >= 2.0 * 524_288.0


def test_csa2_o_lora_param_and_tp4_hbm_conservation() -> None:
    """Verify CSA2 o_lora_rank and o_groups closed-form params and TP4 HBM weights."""
    chip = rf.CHIPS["tpu7x"]
    csa_geom = _base_csa2_geometry()
    csa_params = rf.param_counts(csa_geom)
    csa_unit_hbm = rf.hbm_breakdown_per_core_bytes(
        csa_geom, _UNIT_DTYPES, _UNIT_PARALLEL, rf.SeqKvLayout(0, 0), chip=chip
    )
    assert abs(csa_params.total_params - csa_unit_hbm.weights) <= _UNIT_BYTE_TOL
    csa_wq_a = csa_geom.hidden * csa_geom.q_lora_rank
    csa_wq_b_tp1 = csa_geom.q_lora_rank * csa_geom.num_heads * csa_geom.head_dim
    csa_wkv = csa_geom.hidden * csa_geom.head_dim
    o_lora = 256
    o_groups = 4
    n_idx_layers = 3
    csa_wo_a_tp1 = csa_geom.num_heads * csa_geom.head_dim * o_lora
    csa_wo_b_tp1 = o_groups * o_lora * csa_geom.hidden
    csa_idx = (
        csa_geom.q_lora_rank * csa_geom.index_n_heads * csa_geom.index_head_dim
        + csa_geom.hidden * csa_geom.index_head_dim
        + csa_geom.hidden * csa_geom.index_n_heads
    )
    expected_csa_attn_tp1 = float(
        csa_geom.num_layers
        * (csa_wq_a + csa_wq_b_tp1 + csa_wkv + csa_wo_a_tp1 + csa_wo_b_tp1)
        + n_idx_layers * csa_idx
    )
    assert csa_params.attention == pytest.approx(expected_csa_attn_tp1)

    tp4_par = rf.Parallel(tp=4, moe="tp_slice")
    nq_tp4 = csa_geom.num_heads // tp4_par.tp
    expected_csa_attn_tp4 = float(
        csa_geom.num_layers
        * (
            csa_wq_a
            + csa_geom.q_lora_rank * nq_tp4 * csa_geom.head_dim
            + csa_wkv
            + nq_tp4 * csa_geom.head_dim * o_lora
            + (o_groups * o_lora * csa_geom.hidden) / tp4_par.tp
        )
        + n_idx_layers * csa_idx
    )
    expected_csa_ffn_tp4 = float(
        csa_geom.num_layers
        * (
            (
                csa_geom.num_experts
                * csa_geom.ffn_num_matrices
                * csa_geom.hidden
                * csa_geom.moe_inter
                + csa_geom.ffn_num_matrices * csa_geom.hidden * csa_geom.shared_inter
            )
            / tp4_par.tp
            + csa_geom.hidden * csa_geom.num_experts
        )
    )
    expected_csa_head_tp4 = float(2 * csa_geom.vocab * csa_geom.hidden) / float(
        tp4_par.tp
    )
    csa_tp4_hbm = rf.hbm_breakdown_per_core_bytes(
        csa_geom, _UNIT_DTYPES, tp4_par, rf.SeqKvLayout(0, 0), chip=chip
    )
    assert memory.compute_w_attn(
        csa_geom, _UNIT_DTYPES, tp4_par, rf.count_layer_types(csa_geom)
    ) == pytest.approx(expected_csa_attn_tp4)
    assert csa_tp4_hbm.weights == pytest.approx(
        expected_csa_attn_tp4 + expected_csa_ffn_tp4 + expected_csa_head_tp4
    )


def test_host_table_residency_hbm_and_h2d_conservation() -> None:
    """Verify HostTableSpec dtype conservation, residency modes, and PCIe H2D bounds."""
    table_l0 = rf.HostTableSpec(
        name="engram_l0",
        layer_ids=(0,),
        elem_count=_HOST_TABLE_L0_ELEMS,
        scale_ratio=_UE8M0_SCALE_RATIO,
        table_dtype_role="w_dense",
        dense_proj_dtype_role="w_dense",
        default_hbm_resident=False,
        shard_axis="dp_local",
        h2d_bytes_per_tok=_HOST_TABLE_H2D_PER_LAYER,
        dense_proj_params=_HOST_TABLE_PROJ_ELEMS,
    )
    table_l2 = rf.HostTableSpec(
        name="engram_l2",
        layer_ids=(2,),
        elem_count=_HOST_TABLE_L1_ELEMS,
        scale_ratio=_UE8M0_SCALE_RATIO,
        table_dtype_role="w_dense",
        dense_proj_dtype_role="w_dense",
        default_hbm_resident=False,
        shard_axis="dp_local",
        h2d_bytes_per_tok=_HOST_TABLE_H2D_PER_LAYER,
        state_bytes_per_req=_HOST_TABLE_STATE_BYTES,
        dense_proj_params=_HOST_TABLE_PROJ_ELEMS,
    )
    geom = replace(_base_msa_geometry(), host_tables=(table_l0, table_l2))
    chip = rf.CHIPS["tpu7x"]
    params = rf.param_counts(geom)
    expected_host_params = float(
        _HOST_TABLE_L0_ELEMS + _HOST_TABLE_L1_ELEMS + 2 * _HOST_TABLE_PROJ_ELEMS
    )
    assert params.host_tables == pytest.approx(expected_host_params)

    # Unit dtype yields exact 1.0 B/elem conservation across all residency modes.
    for mode, expected_offloaded in (
        ("hbm", 0.0),
        ("host", float(_HOST_TABLE_L0_ELEMS + _HOST_TABLE_L1_ELEMS)),
        ("auto", float(_HOST_TABLE_L0_ELEMS + _HOST_TABLE_L1_ELEMS)),
        ("first", float(_HOST_TABLE_L1_ELEMS)),
    ):
        offloaded = rf.offloaded_host_table_params(geom, mode)
        assert offloaded == pytest.approx(expected_offloaded)
        hbm = rf.hbm_breakdown_per_core_bytes(
            geom,
            _UNIT_DTYPES,
            _UNIT_PARALLEL,
            rf.SeqKvLayout(0, 0, host_table_residency=mode),
            chip=chip,
        )
        assert abs((params.total_params - offloaded) - hbm.weights) <= _UNIT_BYTE_TOL

    # Real dtypes apply scale_ratio and dp_local host-core sharding (8 cores/host).
    fp8_dt = rf.Dtypes(w_dense=FP8_BYTES, w_aux=BF16_BYTES)
    par_dp8 = rf.Parallel(tp=1, dp=8)
    hbm_first = rf.hbm_breakdown_per_core_bytes(
        geom,
        fp8_dt,
        par_dp8,
        rf.SeqKvLayout(1, 0, host_table_residency="first"),
        chip=chip,
    )
    hbm_host = rf.hbm_breakdown_per_core_bytes(
        geom,
        fp8_dt,
        par_dp8,
        rf.SeqKvLayout(1, 0, host_table_residency="host"),
        chip=chip,
    )
    expected_l0_per_core = (
        float(_HOST_TABLE_L0_ELEMS) * (FP8_BYTES + _UE8M0_SCALE_RATIO) / 8.0
    )
    assert (hbm_first.weights - hbm_host.weights) == pytest.approx(expected_l0_per_core)
    assert hbm_host.state == pytest.approx(_HOST_TABLE_STATE_BYTES)

    # Offloaded PCIe H2D is chip-scoped and NEVER inflates hard_floor_s.
    shape_host = rf.StepShape(
        batch=16, rows_per_seq=1, context=1024, host_table_residency="host"
    )
    shape_hbm = rf.StepShape(
        batch=16, rows_per_seq=1, context=1024, host_table_residency="hbm"
    )
    opts = rf.FloorOptions(dtypes=fp8_dt)
    floor_host = rf.step_floor(geom, chip, par_dp8, shape_host, opts)
    floor_hbm = rf.step_floor(geom, chip, par_dp8, shape_hbm, opts)
    expected_pcie_s = (
        float(16 * chip.cores_per_chip)
        * (2.0 * _HOST_TABLE_H2D_PER_LAYER)
        / chip.pcie_bw
    )
    assert floor_host.pcie_s == pytest.approx(expected_pcie_s)
    assert floor_hbm.pcie_s == 0.0
    assert floor_host.hard_floor_s <= floor_hbm.hard_floor_s
    assert floor_host.hard_floor_s <= floor_host.envelope_floor_s
    assert floor_host.envelope_floor_s <= floor_host.serial_estimate_s
    # Dropping PCIe BW 100x must raise envelope/serial but NEVER change hard_floor_s.
    slow_pcie_chip = replace(chip, pcie_bw=chip.pcie_bw / 100.0)
    floor_slow = rf.step_floor(geom, slow_pcie_chip, par_dp8, shape_host, opts)
    assert floor_slow.hard_floor_s == pytest.approx(floor_host.hard_floor_s)
    assert floor_slow.envelope_floor_s > floor_host.hard_floor_s
    assert floor_slow.serial_estimate_s > floor_host.serial_estimate_s


def test_draft_spec_pp_last_stage_and_verify_cost() -> None:
    """Verify DraftSpec weights, KV dtype/offload skip, single-row verify, and PP."""
    draft_dspark = rf.DraftSpec(
        kind="dspark",
        num_layers=5,
        layer_kind="sliding_attention",
        dense_inter=2048,
        sliding_window=4096,
        block_size=7,
        fc_in_mult=5,
        fc_replicated=False,
        markov_rank=256,
        draft_kv_dtype="fp8",
        skip_draft_kv_on_offload=True,
        single_row_lm_head_verify=True,
    )
    geom_single_row = replace(_base_msa_geometry(), draft=draft_dspark)
    geom_multi_row = replace(
        geom_single_row, draft=replace(draft_dspark, single_row_lm_head_verify=False)
    )
    chip = rf.CHIPS["tpu7x"]

    # Parameter conservation with dspark shared fc + markov head.
    params = rf.param_counts(geom_single_row)
    unit_hbm = rf.hbm_breakdown_per_core_bytes(
        geom_single_row, _UNIT_DTYPES, _UNIT_PARALLEL, rf.SeqKvLayout(0, 0), chip=chip
    )
    assert abs(params.total_params - unit_hbm.weights) <= _UNIT_BYTE_TOL
    expected_fc = 5.0 * 1024.0 * 1024.0
    expected_markov = 2.0 * 32000.0 * 256.0
    assert params.mtp > expected_fc + expected_markov

    # Separate draft_kv_dtype="fp8" + skip_draft_kv_on_offload behavior.
    int4_dt = rf.Dtypes(kv=FP4_BYTES, kv_dtype=rf.INT4_PER_TOKEN_HEAD_KV_DTYPE)
    par_tp8 = rf.Parallel(tp=8, kv_split=1)
    bd_with_draft_kv = rf.hbm_breakdown_per_core_bytes(
        geom_single_row,
        int4_dt,
        par_tp8,
        rf.SeqKvLayout(1, 2048, kv_offload_enabled=False),
        chip=chip,
    )
    bd_skip_draft_kv = rf.hbm_breakdown_per_core_bytes(
        geom_single_row,
        int4_dt,
        par_tp8,
        rf.SeqKvLayout(1, 2048, kv_offload_enabled=True),
        chip=chip,
    )
    expected_draft_kv = 5.0 * 2048.0 * (2.0 * 1.0 * 128.0 * FP8_BYTES)
    assert (bd_with_draft_kv.kv - bd_skip_draft_kv.kv) == pytest.approx(
        expected_draft_kv
    )

    # single_row_lm_head_verify reduces verify lm_head FLOPs by (spec_k + 1)x.
    spec = rf.SpecShape(batch=4, context=2048, spec_k=7)
    opts = rf.FloorOptions(dtypes=int4_dt)
    fl_single = rf.spec_cycle_floor(geom_single_row, chip, par_tp8, spec, opts)
    fl_multi = rf.spec_cycle_floor(geom_multi_row, chip, par_tp8, spec, opts)
    lm_single = next(mod for mod in fl_single.modules if mod.name == "lm_head")
    lm_multi = next(mod for mod in fl_multi.modules if mod.name == "lm_head")
    assert lm_multi.bf16_flops == pytest.approx(8.0 * lm_single.bf16_flops)

    # PP stage slicing keeps DraftSpec strictly on the last stage (end == num_layers).
    stage0 = stage_slice_geometry(geom_single_row, 0, 2)
    stage1 = stage_slice_geometry(geom_single_row, 2, 4)
    assert stage0.draft is None
    assert stage1.draft == draft_dspark
    part, peak_bd, last_total, feasible = rf.pp_stage_breakdowns(
        geom_single_row,
        _UNIT_DTYPES,
        rf.Parallel(tp=1, pp=2),
        workload=(0, 0),
        chip=chip,
    )
    assert feasible
    assert sum(part) == geom_single_row.num_layers
    assert last_total == pytest.approx(peak_bd.total)


def test_hf_loader_mxfp8_ue8m0_int4_kv_and_large_ignore_lists(
    tmp_path: Path,
) -> None:
    """Verify MXFP8, FP8+UE8M0+FP4, int4_per_token_head, and ignore-list parsing."""
    mxfp8_cfg: dict[str, object] = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "group_size": 16,
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "kv_cache_dtype": "int4_per_token_head",
            "ignored_layers": ["lm_head", "model.layers.3.block_sparse_moe.gate"],
        },
    }
    mxfp8_dt = rf.dtypes_from_hf_config(mxfp8_cfg)
    assert mxfp8_dt.w_dense == pytest.approx(_MXFP8_EXPECTED)
    assert mxfp8_dt.w_dense_dtype == "mxfp8"
    assert mxfp8_dt.w_expert == pytest.approx(NVFP4_BYTES)
    assert mxfp8_dt.w_expert_dtype == "nvfp4"
    assert mxfp8_dt.kv == pytest.approx(FP4_BYTES)
    assert mxfp8_dt.kv_dtype == rf.INT4_PER_TOKEN_HEAD_KV_DTYPE

    ue8m0_cfg: dict[str, object] = {
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    }
    ue8m0_dt = rf.dtypes_from_hf_config(ue8m0_cfg)
    assert ue8m0_dt.w_dense == pytest.approx(_UE8M0_32X32_EXPECTED)
    assert ue8m0_dt.w_expert == pytest.approx(MXFP4_BYTES)

    ue8m0_128_cfg: dict[str, object] = {
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    }
    ue8m0_128_dt = rf.dtypes_from_hf_config(ue8m0_128_cfg)
    assert ue8m0_128_dt.w_dense == pytest.approx(_UE8M0_128X128_EXPECTED)

    large_ignore_list = [
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj"
        for layer_idx in range(48)
    ] + [
        f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv"
        for layer_idx in range(36)
    ]
    large_ignore_cfg: dict[str, object] = {
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_convert": ["ple.ple_embedding.ngram_embedding"],
            "modules_to_not_convert": ["lm_head", *large_ignore_list],
        },
    }
    large_ignore_dt = rf.dtypes_from_hf_config(large_ignore_cfg)
    assert large_ignore_dt.w_dense == pytest.approx(BF16_BYTES)
    assert large_ignore_dt.w_expert == pytest.approx(FP8_BYTES + 4.0 / (128.0 * 128.0))

    draft_json = tmp_path / "draft.json"
    draft_payload = {"num_hidden_layers": 5, "dspark_block_size": 7}
    draft_json.write_text(json.dumps(draft_payload), encoding="utf-8")
    assert rf.load_draft_hf_config(None) is None
    assert rf.load_draft_hf_config(draft_json) == draft_payload


def test_ced_prefill_replay_window_floor_soundness_and_pp_conservation() -> None:
    """Verify CED replay window lowers long-prefill floor and slices cleanly on PP."""
    geom = _base_csa2_geometry()
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=rf.FP8_UE8M0_BYTES, w_expert=MXFP4_BYTES)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    opts = rf.FloorOptions(dtypes=dtypes)

    # Pure decode is unaffected by execution="ced".
    dec_full = rf.step_floor(
        geom, chip, par, rf.StepShape(batch=4, context=2048, execution="full"), opts
    )
    dec_ced = rf.step_floor(
        geom, chip, par, rf.StepShape(batch=4, context=2048, execution="ced"), opts
    )
    assert dec_ced.hard_floor_s == pytest.approx(dec_full.hard_floor_s)

    # Prefill longer than sliding_window reduces compute/HBM on layers >= ced_boundary.
    pre_full = rf.step_floor(
        geom,
        chip,
        par,
        rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=8192, execution="full"),
        opts,
    )
    pre_ced = rf.step_floor(
        geom,
        chip,
        par,
        rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=8192, execution="ced"),
        opts,
    )
    assert 0.0 < pre_ced.hard_floor_s < pre_full.hard_floor_s
    assert pre_ced.serial_estimate_s < pre_full.serial_estimate_s

    # PP2 split at boundary puts replay-only layers on stage 1.
    stage0 = stage_slice_geometry(geom, 0, 2)
    stage1 = stage_slice_geometry(geom, 2, 4)
    assert stage0.sparse_attn is not None
    assert stage1.sparse_attn is not None
    assert stage0.sparse_attn.ced_post_only is False
    assert stage1.sparse_attn.ced_post_only is True
    sf_stage0 = rf.step_floor(
        stage0,
        chip,
        par,
        rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=8192, execution="ced"),
        opts,
    )
    sf_stage1 = rf.step_floor(
        stage1,
        chip,
        par,
        rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=8192, execution="ced"),
        opts,
    )
    assert sf_stage1.matmul_bound_s < sf_stage0.matmul_bound_s

    # CED split merges same-named modules once so zero_weights strips w_full once.
    ced_prefill_shape = rf.StepShape(
        batch=0, rows_per_seq=0, prefill_tokens=8192, execution="ced"
    )
    ced_mods = rf.pass_costs(geom, par, ced_prefill_shape, dtypes, chip=chip)
    ced_names = [mod.name for mod in ced_mods]
    assert len(ced_names) == len(set(ced_names))
    hbm_s, _, _, _ = rf.unpadded_strict_resource_buckets(
        geom,
        chip,
        par,
        ced_prefill_shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, zero_weights=True),
    )
    replay_toks = geom.sliding_window
    pre_kv = (
        float(8192) * float(geom.head_dim + geom.qk_rope_head_dim) * FP8_BYTES
        + 2.0 * float(8192) * _CSA2_SWA_BYTES
        + float(8192) * _CSA2_IDX_BYTES
    )
    post_kv = (
        float(replay_toks) * float(geom.head_dim + geom.qk_rope_head_dim) * FP8_BYTES
        + 2.0 * float(replay_toks) * _CSA2_SWA_BYTES
    )
    assert hbm_s == pytest.approx((pre_kv + post_kv) / chip.hbm_bw)


def test_pp_stage_slice_conservation_for_sparse_host_tables_and_draft() -> None:
    """Verify PP stage slicing rebases IDs and conserves weights, KV, and state."""
    table_l0 = rf.HostTableSpec(
        name="t0",
        layer_ids=(0,),
        elem_count=4_000_000,
        default_hbm_resident=False,
        shard_axis="tp",
        state_bytes_per_req=4096.0,
    )
    table_l3 = rf.HostTableSpec(
        name="t3",
        layer_ids=(3,),
        elem_count=6_000_000,
        default_hbm_resident=False,
        shard_axis="tp",
        state_bytes_per_req=8192.0,
    )
    draft = rf.DraftSpec(
        kind="mtp",
        num_layers=1,
        layer_kind="full_attention",
        num_experts=16,
        top_k=2,
        moe_inter=512,
        fc_in_mult=2,
    )
    geom = replace(_base_csa2_geometry(), host_tables=(table_l0, table_l3), draft=draft)
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=2, pp=1, moe="ep", ep=2)

    s0 = stage_slice_geometry(geom, 0, 2)
    s1 = stage_slice_geometry(geom, 2, 4)
    assert [tbl.layer_ids for tbl in s0.host_tables] == [(0,)]
    assert [tbl.layer_ids for tbl in s1.host_tables] == [(1,)]
    assert s0.sparse_attn is not None
    assert s1.sparse_attn is not None
    assert s0.sparse_attn.kv_source_layer_ids == (1,)
    assert s1.sparse_attn.kv_source_layer_ids == (1,)
    assert s0.sparse_attn.index_source_layer_ids == (1,)
    assert s1.sparse_attn.index_source_layer_ids == (0, 1)
    assert s0.draft is None
    assert s1.draft == draft

    head_bytes = float(2 * geom.vocab * geom.hidden) * dtypes.lm_head / float(par.tp)
    for mode in ("auto", "hbm", "host", "first"):
        layout = rf.SeqKvLayout(2, 1024, host_table_residency=mode)
        full_bd = rf.hbm_breakdown_per_core_bytes(geom, dtypes, par, layout, chip=chip)
        bd0 = rf.hbm_breakdown_per_core_bytes(s0, dtypes, par, layout, chip=chip)
        bd1 = rf.hbm_breakdown_per_core_bytes(s1, dtypes, par, layout, chip=chip)
        # Each sliced stage carries full embed+lm_head once; subtract one extra copy.
        assert (bd0.weights + bd1.weights - head_bytes) == pytest.approx(
            full_bd.weights
        )
        assert (bd0.kv + bd1.kv) == pytest.approx(full_bd.kv)
        assert (bd0.state + bd1.state) == pytest.approx(full_bd.state)


def test_cli_phase3_flags_fail_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --execution ced, --host-table-residency, and --prefill-draft fail."""
    cfg_path = _FIXTURES_DIR / "qwen3_5_397b_fp8.json"
    qwen_geom = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    parser = cli.build_parser()
    for bad_flag, bad_val, match_txt in (
        ("--execution", "ced", "CED"),
        ("--host-table-residency", "first", "host_tables"),
        ("--prefill-draft", None, "draft"),
    ):
        cli_args = [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "8",
            bad_flag,
        ]
        if bad_val is not None:
            cli_args.append(bad_val)
        parsed = parser.parse_args(cli_args)
        with pytest.raises(ValueError, match=match_txt):
            cli.validate_cli_geometry_flags(parsed, qwen_geom)
        assert cli.main(cli_args) == cli.EXIT_ERROR
        assert match_txt in capsys.readouterr().err

    # Explicit --kv-dtype clears draft_kv_dtype so CLI override wins.
    draft_geom = replace(
        qwen_geom,
        draft=rf.DraftSpec(
            kind="dspark",
            num_layers=2,
            layer_kind="sliding_attention",
            draft_kv_dtype="fp8",
        ),
    )
    kv_args = parser.parse_args(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "8",
            "--kv-dtype",
            "bf16",
        ]
    )
    resolved_geom = cli.validate_cli_geometry_flags(kv_args, draft_geom)
    assert resolved_geom.draft is not None
    assert resolved_geom.draft.draft_kv_dtype == ""


def test_emit_provenance_records_all_four_new_cli_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify Phase 3 CLI flags change solve outputs and record provenance."""
    cfg_path = _FIXTURES_DIR / "qwen3_5_397b_fp8.json"
    draft_path = tmp_path / "draft_cfg.json"
    draft_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "dspark_block_size": 5,
                "dspark_use_dense_ffn": True,
                "dspark_ffn_hidden_size": 4096,
                "dspark_target_layer_ids": [10, 30, 50],
                "layer_types": ["sliding_attention", "sliding_attention"],
                "sliding_window": 4096,
            }
        ),
        encoding="utf-8",
    )
    base_args = [
        "solve",
        "--config",
        str(cfg_path),
        "--example-scenario",
        "chat-8k1k",
        "--mode",
        "mixed",
        "--max-chips",
        "8",
        "--concurrency",
        "1",
        "--json",
    ]
    assert cli.main(base_args) == 0
    base_payload = json.loads(capsys.readouterr().out)
    override_args = [
        *base_args[:-1],
        "--draft-config",
        str(draft_path),
        "--kv-offload-gb",
        "64",
        "--prefill-draft",
        "--json",
    ]
    assert cli.main(override_args) == 0
    ov_payload = json.loads(capsys.readouterr().out)
    supplied = {
        entry["flag"]: entry for entry in ov_payload["provenance"]["supplied_values"]
    }
    assert supplied["--draft-config"]["value"] == str(draft_path)
    expected_offload_gb = 64.0
    assert supplied["--kv-offload-gb"]["value"] == pytest.approx(expected_offload_gb)
    assert supplied["--prefill-draft"]["value"] is True
    assert ov_payload["recommended"] != base_payload["recommended"]

    # Record --execution=ced and --host-table-residency=first on a CED+host geometry.
    ced_host_geom = replace(
        _base_csa2_geometry(),
        host_tables=(
            rf.HostTableSpec(
                name="tbl",
                layer_ids=(0,),
                elem_count=_HOST_TABLE_L0_ELEMS,
            ),
        ),
        draft=rf.DraftSpec(kind="dspark", num_layers=1, layer_kind="full_attention"),
    )
    ced_scen = replace(
        rf.build_example_scenario("chat-8k1k"),
        concurrencies=(1,),
        execution="ced",
        host_table_residency="first",
        prefill_draft=True,
    )
    ced_plan = rf.solve_serving_plan(
        ced_host_geom,
        rf.Dtypes(),
        rf.CHIPS["tpu7x"],
        ced_scen,
        rf.SolveOptions(mode="mixed", max_chips=4, draft_config_source=str(draft_path)),
    )
    ced_payload = emit.build_solve_payload(
        ced_plan, chip=rf.CHIPS["tpu7x"], dtypes=rf.Dtypes(), model="synthetic_ced"
    )
    ced_supplied = {
        entry["flag"]: entry for entry in ced_payload["provenance"]["supplied_values"]
    }
    assert ced_supplied["--execution"]["value"] == "ced"
    assert ced_supplied["--host-table-residency"]["value"] == "first"
    assert ced_supplied["--prefill-draft"]["value"] is True
    assert ced_supplied["--draft-config"]["value"] == str(draft_path)


def test_model_lockin_supports_optional_draft_fixture_and_digest(
    tmp_path: Path,
) -> None:
    """Verify manifest lock-in enforces optional draft_fixture SHA-256."""
    base_manifest = json.loads(
        (_FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    for entry in base_manifest["checkpoints"]:
        src = _FIXTURES_DIR / entry["fixture"]
        (tmp_path / entry["fixture"]).write_bytes(src.read_bytes())
        existing_draft = entry.get("draft_fixture")
        if existing_draft is not None:
            draft_src = _FIXTURES_DIR / existing_draft
            (tmp_path / existing_draft).write_bytes(draft_src.read_bytes())
    draft_bytes = json.dumps(
        {
            "num_hidden_layers": 2,
            "dspark_block_size": 5,
            "dspark_use_dense_ffn": True,
            "dspark_ffn_hidden_size": 4096,
            "dspark_target_layer_ids": [10, 30],
            "layer_types": ["sliding_attention", "sliding_attention"],
            "sliding_window": 4096,
        }
    ).encode("utf-8")
    draft_name = "qwen3_5_397b_fp8_dspark.json"
    (tmp_path / draft_name).write_bytes(draft_bytes)
    draft_digest = base64.b64encode(hashlib.sha256(draft_bytes).digest()).decode(
        "ascii"
    )
    base_manifest["checkpoints"][0]["draft_fixture"] = draft_name
    base_manifest["checkpoints"][0]["draft_sha256_b64"] = draft_digest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")

    test_model_lockin.verify_model_manifest(manifest_path, tmp_path)

    # Corrupting the draft digest must fail lock-in verification.
    base_manifest["checkpoints"][0]["draft_sha256_b64"] = "bad_digest=="
    manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    with pytest.raises(AssertionError):
        test_model_lockin.verify_model_manifest(manifest_path, tmp_path)


def test_one_based_ple_layer_ids_conversion_and_validation() -> None:
    """Verify 1-based HF ple_layer_ids converts to 0-based indices cleanly."""
    base_cfg = load_locked_config("qwen3_5_397b_fp8.json")
    text_cfg = dict(base_cfg["text_config"])
    text_cfg["ple_layer_ids"] = [2]
    cfg_valid = {**base_cfg, "text_config": text_cfg}
    geom = rf.from_hf_config(cfg_valid)
    assert geom.ple_layer_ids == (1,)

    for bad_ids in ([0], [61]):
        bad_text = dict(base_cfg["text_config"])
        bad_text["ple_layer_ids"] = bad_ids
        with pytest.raises(ValueError, match="ple_layer_ids"):
            rf.from_hf_config({**base_cfg, "text_config": bad_text})


def test_legacy_mtp_strips_target_host_tables_and_sparse_attn() -> None:
    """Verify legacy MTP (draft=None) drops target host_tables and sparse_attn."""
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    parallel = rf.Parallel(tp=2, pp=1, moe="ep", ep=2)
    host_tbl = rf.HostTableSpec(
        name="target_tbl",
        layer_ids=(0, 2),
        elem_count=_TP_SHARDED_HOST_ELEMS,
        default_hbm_resident=False,
        shard_axis="tp",
        h2d_bytes_per_tok=_TP_SHARDED_HOST_H2D_BYTES,
        dense_proj_params=_HOST_TABLE_PROJ_ELEMS,
    )
    mtp_geom = replace(
        _base_qsa_geometry(),
        mtp_layers=1,
        mtp_inter=512,
        host_tables=(host_tbl,),
        draft=None,
    )
    clean_geom = replace(mtp_geom, host_tables=())
    opts = rf.FloorOptions(dtypes=dtypes)

    target_only = rf.step_floor(
        mtp_geom,
        chip,
        parallel,
        rf.StepShape(
            batch=4, rows_per_seq=3, context=2048, host_table_residency="host"
        ),
        opts,
    )
    spec_floor = rf.spec_cycle_floor(
        mtp_geom,
        chip,
        parallel,
        rf.SpecShape(batch=4, context=2048, spec_k=2, host_table_residency="host"),
        opts,
    )
    host_mods = [
        mod for mod in spec_floor.modules if mod.name == f"host_table_{host_tbl.name}"
    ]
    assert len(host_mods) == 1
    assert spec_floor.pcie_s == pytest.approx(target_only.pcie_s)

    # Draft attention in legacy MTP uses plain GQA without QSA keys or indexer.
    draft_gqa = next(mod for mod in spec_floor.modules if mod.name == "draft_gqa")
    expected_draft_gqa = next(
        mod
        for mod in rf.step_floor(
            replace(
                clean_geom,
                num_layers=1,
                layer_types=("full_attention",),
                sparse_attn=None,
                mtp_layers=0,
            ),
            chip,
            parallel,
            rf.StepShape(batch=4, rows_per_seq=1, context=2048),
            opts,
        ).modules
        if mod.name == "gqa"
    )
    assert draft_gqa.hbm_bytes == pytest.approx(2.0 * expected_draft_gqa.hbm_bytes)
    assert draft_gqa.fp8_flops == pytest.approx(2.0 * expected_draft_gqa.fp8_flops)


def test_multilayer_host_table_first_residency_pp_conservation() -> None:
    """Verify multi-layer HostTableSpec under first residency avoids double scaling."""
    chip = rf.CHIPS["tpu7x"]
    table_two_layers = rf.HostTableSpec(
        name="shared_tbl",
        layer_ids=(0, 2),
        elem_count=_TP_SHARDED_HOST_ELEMS,
        default_hbm_resident=False,
        shard_axis="tp",
        h2d_bytes_per_tok=65_536.0,
        resident_on_first=None,
    )
    base_geom = replace(_base_msa_geometry(), host_tables=())
    tbl_geom = replace(base_geom, host_tables=(table_two_layers,))
    par_pp2 = rf.Parallel(tp=1, pp=2)
    pp_opts = rf.PpStageOptions(host_table_residency="first")

    base_part, base_peak, base_last, base_ok = rf.pp_stage_breakdowns(
        base_geom, _UNIT_DTYPES, par_pp2, workload=(0, 0), options=pp_opts, chip=chip
    )
    tbl_part, tbl_peak, tbl_last, tbl_ok = rf.pp_stage_breakdowns(
        tbl_geom, _UNIT_DTYPES, par_pp2, workload=(0, 0), options=pp_opts, chip=chip
    )
    assert base_ok
    assert tbl_ok
    assert base_part == (2, 2)
    assert tbl_part == (2, 2)
    half_table_bytes = float(_TP_SHARDED_HOST_ELEMS) * 0.5
    assert (tbl_peak.weights - base_peak.weights) == pytest.approx(half_table_bytes)
    assert tbl_last == pytest.approx(base_last)

    # Uniform pp=2 HBM breakdown keeps exact 1/2 table bytes on stage 0 (not 1/4).
    base_bd = rf.hbm_breakdown_per_core_bytes(
        base_geom,
        _UNIT_DTYPES,
        par_pp2,
        rf.SeqKvLayout(0, 0, host_table_residency="first"),
        chip=chip,
    )
    tbl_bd = rf.hbm_breakdown_per_core_bytes(
        tbl_geom,
        _UNIT_DTYPES,
        par_pp2,
        rf.SeqKvLayout(0, 0, host_table_residency="first"),
        chip=chip,
    )
    assert (tbl_bd.weights - base_bd.weights) == pytest.approx(half_table_bytes)

    # Uniform pp=2 step path matches stage 0 lookup HBM bytes under first residency.
    step_shape = rf.StepShape(batch=4, context=128, host_table_residency="first")
    unif_mods = rf.pass_costs(tbl_geom, par_pp2, step_shape, _UNIT_DTYPES, chip=chip)
    stage0_mods = rf.pass_costs(
        stage_slice_geometry(tbl_geom, 0, 2),
        rf.Parallel(tp=1, pp=1),
        step_shape,
        _UNIT_DTYPES,
        chip=chip,
    )
    unif_tbl = next(m for m in unif_mods if m.name == "host_table_shared_tbl")
    stage0_tbl = next(m for m in stage0_mods if m.name == "host_table_shared_tbl")
    assert unif_tbl.hbm_bytes == pytest.approx(stage0_tbl.hbm_bytes)


def test_sparse_gqa_empty_source_ids_qsa_kv_scaling_and_int4() -> None:
    """Verify empty sparse source IDs do not fall back and QSA state scales."""
    chip = rf.CHIPS["tpu7x"]
    fp8_dt = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    bf16_kv_dt = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=BF16_BYTES)
    int4_kv_dt = rf.Dtypes(
        w_dense=FP8_BYTES,
        w_expert=MXFP4_BYTES,
        kv=FP4_BYTES,
        kv_dtype=rf.INT4_PER_TOKEN_HEAD_KV_DTYPE,
    )
    par = rf.Parallel(tp=2, pp=1, moe="ep", ep=2)

    # Hybrid GDN + QSA model with QSA source IDs strictly on layers 2 and 3.
    hybrid_qsa = replace(
        _base_qsa_geometry(),
        layer_types=(
            "linear_attention",
            "linear_attention",
            "indexed_attention",
            "indexed_attention",
        ),
        sparse_attn=replace(
            _base_qsa_geometry().sparse_attn,
            kv_source_layer_ids=(2, 3),
            index_source_layer_ids=(2, 3),
        ),
    )
    stage0 = stage_slice_geometry(hybrid_qsa, 0, 2)
    stage1 = stage_slice_geometry(hybrid_qsa, 2, 4)
    head_bytes = (
        float(2 * hybrid_qsa.vocab * hybrid_qsa.hidden) * fp8_dt.lm_head / float(par.tp)
    )
    layout = rf.SeqKvLayout(2, 1024, physical_kv_layout=True)
    full_bd = rf.hbm_breakdown_per_core_bytes(
        hybrid_qsa, fp8_dt, par, layout, chip=chip
    )
    bd0 = rf.hbm_breakdown_per_core_bytes(stage0, fp8_dt, par, layout, chip=chip)
    bd1 = rf.hbm_breakdown_per_core_bytes(stage1, fp8_dt, par, layout, chip=chip)
    assert (bd0.weights + bd1.weights - head_bytes) == pytest.approx(full_bd.weights)
    assert (bd0.kv + bd1.kv) == pytest.approx(full_bd.kv)
    assert (bd0.state + bd1.state) == pytest.approx(full_bd.state)

    # Stage with GQA layers and empty sparse source IDs adds zero extra KV/state.
    empty_qsa_src = replace(
        hybrid_qsa,
        layer_types=("full_attention",) * 4,
        sparse_attn=replace(
            hybrid_qsa.sparse_attn,
            kv_source_layer_ids=(),
            index_source_layer_ids=(),
        ),
    )
    plain_gqa = replace(empty_qsa_src, sparse_attn=None)
    assert rf.hbm_breakdown_per_core_bytes(
        empty_qsa_src, fp8_dt, par, layout, chip=chip
    ) == pytest.approx(
        rf.hbm_breakdown_per_core_bytes(plain_gqa, fp8_dt, par, layout, chip=chip)
    )
    assert kv_bytes_per_token_per_layer(
        empty_qsa_src, fp8_dt, par, "gqa", physical=True, chip=chip
    ) == pytest.approx(
        kv_bytes_per_token_per_layer(
            plain_gqa, fp8_dt, par, "gqa", physical=True, chip=chip
        )
    )

    empty_msa_src = replace(
        _base_msa_geometry(),
        sparse_attn=replace(
            _base_msa_geometry().sparse_attn,
            kv_source_layer_ids=(),
            index_source_layer_ids=(),
        ),
    )
    assert kv_bytes_per_token_per_layer(
        empty_msa_src, fp8_dt, par, "gqa", physical=True, chip=chip
    ) == pytest.approx(float(2 * (4 // par.tp) * 128) * FP8_BYTES)
    msa_floor = rf.step_floor(
        empty_msa_src,
        chip,
        par,
        rf.StepShape(batch=2, context=32_768),
        rf.FloorOptions(dtypes=fp8_dt),
    )
    plain_msa_floor = rf.step_floor(
        replace(empty_msa_src, sparse_attn=None, layer_types=("gqa",) * 4),
        chip,
        par,
        rf.StepShape(batch=2, context=32_768),
        rf.FloorOptions(dtypes=fp8_dt),
    )
    msa_gqa = next(m for m in msa_floor.modules if m.name == "gqa")
    plain_gqa = next(m for m in plain_msa_floor.modules if m.name == "gqa")
    assert msa_gqa.fp8_flops == pytest.approx(plain_gqa.fp8_flops)
    assert msa_gqa.hbm_bytes == pytest.approx(plain_gqa.hbm_bytes)
    assert msa_gqa.hbm_bytes_min == pytest.approx(plain_gqa.hbm_bytes_min)
    assert msa_gqa.bf16_flops == pytest.approx(plain_gqa.bf16_flops)

    # QSA fixed_state_page_bytes scales linearly with dtypes.kv / FP8_BYTES.
    qsa_only = replace(
        _base_qsa_geometry(),
        layer_types=("indexed_attention",) * 2,
        num_layers=2,
        sparse_attn=replace(
            _base_qsa_geometry().sparse_attn,
            kv_source_layer_ids=(0, 1),
            index_source_layer_ids=(0, 1),
        ),
    )
    bd_fp8 = rf.hbm_breakdown_per_core_bytes(
        qsa_only, fp8_dt, par, rf.SeqKvLayout(2, 0), chip=chip
    )
    bd_bf16 = rf.hbm_breakdown_per_core_bytes(
        qsa_only, bf16_kv_dt, par, rf.SeqKvLayout(2, 0), chip=chip
    )
    assert bd_fp8.state == pytest.approx(2.0 * 2.0 * _QSA_FIXED_STATE_BYTES)
    assert bd_bf16.state == pytest.approx(2.0 * bd_fp8.state)

    # QSA decode KV read under int4_per_token_head matches _int4_pth_kv + extra.
    qsa_i4_par = rf.Parallel(tp=2, moe="ep", ep=2)
    qsa_i4_floor = rf.step_floor(
        qsa_only,
        chip,
        qsa_i4_par,
        rf.StepShape(batch=2, context=1024),
        rf.FloorOptions(dtypes=int4_kv_dt),
    )
    qsa_i4_zero = rf.step_floor(
        qsa_only,
        chip,
        qsa_i4_par,
        rf.StepShape(batch=0, rows_per_seq=0, context=0),
        rf.FloorOptions(dtypes=int4_kv_dt),
    )
    qsa_i4_gqa = next(mod for mod in qsa_i4_floor.modules if mod.name == "gqa")
    qsa_i4_gqa0 = next(mod for mod in qsa_i4_zero.modules if mod.name == "gqa")
    expected_i4_main_per_tok = (
        float(2 * qsa_only.head_dim * (qsa_only.num_kv_heads // 2)) * FP4_BYTES
        + _INT4_PTH_SCALE_BYTES
    )
    expected_qsa_i4_kv = 2.0 * float(2 * 1024) * expected_i4_main_per_tok
    assert (qsa_i4_gqa.hbm_bytes_min - qsa_i4_gqa0.hbm_bytes_min) == pytest.approx(
        expected_qsa_i4_kv
    )
    assert kv_bytes_per_token_per_layer(
        qsa_only, int4_kv_dt, rf.Parallel(tp=2), "gqa", physical=True, chip=chip
    ) == pytest.approx(expected_i4_main_per_tok + _QSA_COMP_K_BYTES)


def test_partial_frequency_msa_pp_conservation() -> None:
    """Verify partial-frequency MSA conserves KV, step HBM, and scores across PP."""
    chip = rf.CHIPS["tpu7x"]
    fp8_dt = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=2, pp=1, moe="ep", ep=2)
    partial_msa = replace(
        _base_msa_geometry(),
        sparse_attn=replace(
            _base_msa_geometry().sparse_attn,
            kv_source_layer_ids=(2, 3),
            index_source_layer_ids=(2, 3),
        ),
    )
    p_s0 = stage_slice_geometry(partial_msa, 0, 2)
    p_s1 = stage_slice_geometry(partial_msa, 2, 4)
    p_layout = rf.SeqKvLayout(4, 32_768, physical_kv_layout=True)
    p_full_bd = rf.hbm_breakdown_per_core_bytes(
        partial_msa, fp8_dt, par, p_layout, chip=chip
    )
    p_s0_bd = rf.hbm_breakdown_per_core_bytes(p_s0, fp8_dt, par, p_layout, chip=chip)
    p_s1_bd = rf.hbm_breakdown_per_core_bytes(p_s1, fp8_dt, par, p_layout, chip=chip)
    assert (p_s0_bd.kv + p_s1_bd.kv) == pytest.approx(p_full_bd.kv)
    for test_shape in (
        rf.StepShape(batch=2, context=32_768),
        rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=8_192),
    ):
        opts = rf.FloorOptions(dtypes=fp8_dt)
        f_full = next(
            m
            for m in rf.step_floor(partial_msa, chip, par, test_shape, opts).modules
            if m.name == "gqa"
        )
        f_s0 = next(
            m
            for m in rf.step_floor(p_s0, chip, par, test_shape, opts).modules
            if m.name == "gqa"
        )
        f_s1 = next(
            m
            for m in rf.step_floor(p_s1, chip, par, test_shape, opts).modules
            if m.name == "gqa"
        )
        plain_s0 = replace(p_s0, sparse_attn=None, layer_types=("gqa",) * 2)
        f_plain = next(
            m
            for m in rf.step_floor(plain_s0, chip, par, test_shape, opts).modules
            if m.name == "gqa"
        )
        assert f_s0.hbm_bytes == pytest.approx(f_plain.hbm_bytes)
        assert f_s0.bf16_flops == pytest.approx(f_plain.bf16_flops)
        assert (f_s0.hbm_bytes + f_s1.hbm_bytes) == pytest.approx(f_full.hbm_bytes)
        assert (f_s0.bf16_flops + f_s1.bf16_flops) == pytest.approx(f_full.bf16_flops)


def test_csa2_ignores_compress_ratios_in_decode_kv_read() -> None:
    """Verify CSA2 decode KV reads do not apply compress_ratios attenuation."""
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=2, moe="ep", ep=2)
    shape = rf.StepShape(batch=4, rows_per_seq=1, context=1024)
    opts = rf.FloorOptions(dtypes=dtypes)

    csa_no_ratio = replace(_base_csa2_geometry(), compress_ratios=())
    csa_with_ratio = replace(
        csa_no_ratio, compress_ratios=(_CSA2_COMPRESS_RATIO,) * csa_no_ratio.num_layers
    )
    floor_no_ratio = rf.step_floor(csa_no_ratio, chip, par, shape, opts)
    floor_with_ratio = rf.step_floor(csa_with_ratio, chip, par, shape, opts)
    mla_no_ratio = next(mod for mod in floor_no_ratio.modules if mod.name == "mla")
    mla_with_ratio = next(mod for mod in floor_with_ratio.modules if mod.name == "mla")
    assert mla_with_ratio.hbm_bytes == pytest.approx(mla_no_ratio.hbm_bytes)
    assert mla_with_ratio.hbm_bytes_min == pytest.approx(mla_no_ratio.hbm_bytes_min)


def test_dspark_markov_scaling_and_tp_sharded_host_table_pcie() -> None:
    """Verify dspark Markov FLOP/weight scaling and TP-sharded host-table PCIe."""
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(
        w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, w_aux=BF16_BYTES, lm_head=BF16_BYTES
    )
    par_tp2 = rf.Parallel(tp=2, moe="ep", ep=2)
    par_tp4 = rf.Parallel(tp=4, moe="ep", ep=4)

    dspark_markov = rf.DraftSpec(
        kind="dspark",
        num_layers=2,
        layer_kind="sliding_attention",
        dense_inter=512,
        sliding_window=128,
        block_size=_MARKOV_SPEC_K,
        fc_in_mult=2,
        markov_rank=_MARKOV_RANK,
    )
    geom_dspark = replace(_base_msa_geometry(), draft=dspark_markov)
    spec = rf.SpecShape(batch=4, context=1024, spec_k=_MARKOV_SPEC_K)
    opts = rf.FloorOptions(dtypes=dtypes)

    fl_k7 = rf.spec_cycle_floor(geom_dspark, chip, par_tp2, spec, opts)
    fl_k1 = rf.spec_cycle_floor(
        geom_dspark, chip, par_tp2, replace(spec, spec_k=1), opts
    )
    mk_k7 = next(mod for mod in fl_k7.modules if mod.name == "draft_markov")
    mk_k1 = next(mod for mod in fl_k1.modules if mod.name == "draft_markov")
    expected_elems = float(2 * geom_dspark.vocab * _MARKOV_RANK // par_tp2.tp)
    assert mk_k7.hbm_bytes == pytest.approx(expected_elems * BF16_BYTES)
    assert mk_k7.bf16_flops == pytest.approx(float(_MARKOV_SPEC_K) * mk_k1.bf16_flops)

    # Zero-weight unpadded floor strips draft_markov weight read completely.
    step_k7 = rf.StepShape(batch=4, rows_per_seq=1, context=1024)
    unp_with_wt = rf.unpadded_strict_resource_buckets(
        geom_dspark,
        chip,
        par_tp2,
        step_k7,
        rf.UnpaddedFloorOptions(
            dtypes=dtypes, spec_k=_MARKOV_SPEC_K, zero_weights=False
        ),
    )
    unp_zero_wt = rf.unpadded_strict_resource_buckets(
        geom_dspark,
        chip,
        par_tp2,
        step_k7,
        rf.UnpaddedFloorOptions(
            dtypes=dtypes, spec_k=_MARKOV_SPEC_K, zero_weights=True
        ),
    )
    assert unp_zero_wt[0] < unp_with_wt[0]
    assert unp_zero_wt[1] == pytest.approx(unp_with_wt[1])

    # TP-sharded host-table HBM lookup and PCIe H2D both scale as 1 / tp.
    tp_table = rf.HostTableSpec(
        name="tp_tbl",
        layer_ids=(0,),
        elem_count=_TP_SHARDED_HOST_ELEMS,
        default_hbm_resident=False,
        shard_axis="tp",
        h2d_bytes_per_tok=_TP_SHARDED_HOST_H2D_BYTES,
    )
    geom_tp_tbl = replace(_base_msa_geometry(), host_tables=(tp_table,))
    fl_tp2_host = rf.step_floor(
        geom_tp_tbl,
        chip,
        par_tp2,
        rf.StepShape(batch=8, context=1024, host_table_residency="host"),
        opts,
    )
    fl_tp4_host = rf.step_floor(
        geom_tp_tbl,
        chip,
        par_tp4,
        rf.StepShape(batch=8, context=1024, host_table_residency="host"),
        opts,
    )
    expected_tp2_pcie = (
        float(8 * chip.cores_per_chip)
        * (_TP_SHARDED_HOST_H2D_BYTES / 2.0)
        / chip.pcie_bw
    )
    assert fl_tp2_host.pcie_s == pytest.approx(expected_tp2_pcie)
    assert fl_tp2_host.pcie_s == pytest.approx(2.0 * fl_tp4_host.pcie_s)

    fl_tp2_hbm = rf.step_floor(
        geom_tp_tbl,
        chip,
        par_tp2,
        rf.StepShape(batch=8, context=1024, host_table_residency="hbm"),
        opts,
    )
    fl_tp4_hbm = rf.step_floor(
        geom_tp_tbl,
        chip,
        par_tp4,
        rf.StepShape(batch=8, context=1024, host_table_residency="hbm"),
        opts,
    )
    mod_tp2_hbm = next(
        mod for mod in fl_tp2_hbm.modules if mod.name == f"host_table_{tp_table.name}"
    )
    mod_tp4_hbm = next(
        mod for mod in fl_tp4_hbm.modules if mod.name == f"host_table_{tp_table.name}"
    )
    assert mod_tp2_hbm.hbm_bytes == pytest.approx(
        8.0 * (_TP_SHARDED_HOST_H2D_BYTES / 2.0)
    )
    assert mod_tp2_hbm.hbm_bytes == pytest.approx(2.0 * mod_tp4_hbm.hbm_bytes)


def test_empty_layer_ids_host_table_pp_conservation() -> None:
    """Verify HostTableSpec(layer_ids=()) conserves table weights across PP stages."""
    chip = rf.CHIPS["tpu7x"]
    global_tbl = rf.HostTableSpec(
        name="global_tbl",
        layer_ids=(),
        elem_count=_TP_SHARDED_HOST_ELEMS,
        default_hbm_resident=False,
        shard_axis="tp",
    )
    base_geom = replace(_base_msa_geometry(), host_tables=())
    tbl_geom = replace(base_geom, host_tables=(global_tbl,))
    par_pp1 = rf.Parallel(tp=1, pp=1)
    par_pp2 = rf.Parallel(tp=1, pp=2)
    head_bytes = float(2 * tbl_geom.vocab * tbl_geom.hidden) * _UNIT_DTYPES.lm_head

    s0 = stage_slice_geometry(tbl_geom, 0, 2)
    s1 = stage_slice_geometry(tbl_geom, 2, 4)
    assert s0.host_tables[0].elem_count + s1.host_tables[0].elem_count == (
        _TP_SHARDED_HOST_ELEMS
    )

    for mode, expected_total_tbl in (
        ("auto", 0.0),
        ("host", 0.0),
        ("hbm", float(_TP_SHARDED_HOST_ELEMS)),
        ("first", float(_TP_SHARDED_HOST_ELEMS)),
    ):
        layout = rf.SeqKvLayout(0, 0, host_table_residency=mode)
        full_bd = rf.hbm_breakdown_per_core_bytes(
            tbl_geom, _UNIT_DTYPES, par_pp1, layout, chip=chip
        )
        bd0 = rf.hbm_breakdown_per_core_bytes(
            s0, _UNIT_DTYPES, par_pp1, layout, chip=chip
        )
        bd1 = rf.hbm_breakdown_per_core_bytes(
            s1, _UNIT_DTYPES, par_pp1, layout, chip=chip
        )
        assert (bd0.weights + bd1.weights - head_bytes) == pytest.approx(
            full_bd.weights
        )
        _, base_peak, _, _ = rf.pp_stage_breakdowns(
            base_geom,
            _UNIT_DTYPES,
            par_pp2,
            workload=(0, 0),
            options=rf.PpStageOptions(host_table_residency=mode),
            chip=chip,
        )
        part, peak, last_tot, ok = rf.pp_stage_breakdowns(
            tbl_geom,
            _UNIT_DTYPES,
            par_pp2,
            workload=(0, 0),
            options=rf.PpStageOptions(host_table_residency=mode),
            chip=chip,
        )
        assert ok
        expected_part = (2, 2)
        assert part == expected_part
        assert (peak.weights - base_peak.weights) == pytest.approx(
            0.5 * expected_total_tbl
        )
        assert (peak.weights + last_tot) == pytest.approx(full_bd.weights)


def test_qsa_dcp_tp_span_kv_scaling() -> None:
    """Verify TP-spanning DCP on QSA keeps per-rank KV at 1/8 (not 1/64) of total."""
    chip = rf.CHIPS["tpu7x"]
    fp8_dt = rf.Dtypes(kv=FP8_BYTES, kv_dtype="fp8")
    qsa_8h = replace(
        _base_qsa_geometry(),
        num_layers=2,
        layer_types=("indexed_attention", "indexed_attention"),
        num_kv_heads=8,
        head_dim=128,
        sparse_attn=rf.SparseAttnSpec(
            kind="qsa",
            kv_source_layer_ids=(0, 1),
            index_source_layer_ids=(),
            dcp_span="tp",
        ),
    )
    tp1_par = rf.Parallel(tp=1, kv_split=1)
    tp8_par = rf.Parallel(tp=8, kv_split=8)
    bd_tp1 = rf.hbm_breakdown_per_core_bytes(
        qsa_8h, fp8_dt, tp1_par, rf.SeqKvLayout(1, 1024), chip=chip
    )
    bd_tp8 = rf.hbm_breakdown_per_core_bytes(
        qsa_8h, fp8_dt, tp8_par, rf.SeqKvLayout(1, 1024), chip=chip
    )
    dcp_factor = 8.0
    assert bd_tp8.kv == pytest.approx(bd_tp1.kv / dcp_factor)
    lin_tp1, _ = memory.kv_bytes_per_core_parts(
        qsa_8h, fp8_dt, tp1_par, 1024, physical=True, chip=chip
    )
    lin_tp8, _ = memory.kv_bytes_per_core_parts(
        qsa_8h, fp8_dt, tp8_par, 1024, physical=True, chip=chip
    )
    assert lin_tp8 == pytest.approx(lin_tp1 / dcp_factor)


def test_steady_state_batch_cap_accounts_for_draft_swa_kv() -> None:
    """Verify steady-state batch cap shrinks when draft has sliding_window > 0."""
    chip = rf.CHIPS["tpu7x"]
    draft_swa = rf.DraftSpec(
        kind="dspark",
        num_layers=4,
        layer_kind="sliding_attention",
        dense_inter=512,
        sliding_window=128,
        skip_draft_kv_on_offload=True,
    )
    geom = replace(
        _base_msa_geometry(),
        sliding_window=0,
        layer_types=("gqa",) * 4,
        sparse_attn=None,
        draft=draft_swa,
    )
    assert geom.sliding_window == 0
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=8)
    # Choose HBM limit that leaves headroom for ~10 sequences after weights+shared KV.
    hbm_limit_b = 50 * (1 << 20)
    base_scen = replace(
        rf.build_example_scenario("chat-8k1k"),
        isl=4096,
        osl=512,
        prefix_hit_rate=0.95,
        accept_len=2.0,
        concurrencies=(1, 16, 64),
    )

    def _ctx(scen: rf.TrafficScenario) -> _SolveContext:
        """Build minimal _SolveContext for steady-state physical batch cap checks."""
        return _SolveContext(
            geometry=geom,
            dtypes=dtypes,
            chip=chip,
            scenario=scen,
            mode="mixed",
            max_chips=4,
            max_instances=None,
            hbm_reserve_gb=0.0,
            hbm_limit_b=float(hbm_limit_b),
            hbm_limit_gb=float(hbm_limit_b) / 1e9,
            effective_accept_len=2.0,
            spec_k=1,
            ctx_full=scen.isl + scen.osl,
            ctx_mid=scen.isl + scen.osl // 2,
            eff_isl=max(1, round((1.0 - scen.prefix_hit_rate) * scen.isl)),
            n_chunks=1,
            c_max=64,
        )

    cap_with_draft_kv = _steady_state_batch_cap_phys(_ctx(base_scen), par)
    cap_skip_draft_kv = _steady_state_batch_cap_phys(
        _ctx(replace(base_scen, kv_offload_gb=1e-6)), par
    )
    assert 0 < cap_with_draft_kv < cap_skip_draft_kv


def test_unpadded_strict_buckets_include_prefill_draft() -> None:
    """Verify prefill_draft=True raises unpadded buckets with and without weights."""
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, w_aux=BF16_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=4)
    geom = replace(
        _base_msa_geometry(),
        draft=rf.DraftSpec(
            kind="dspark",
            num_layers=2,
            layer_kind="sliding_attention",
            dense_inter=512,
            sliding_window=128,
            block_size=3,
            fc_in_mult=2,
        ),
    )
    shape_off = rf.StepShape(
        batch=0, rows_per_seq=0, prefill_tokens=1024, prefill_draft=False
    )
    shape_on = replace(shape_off, prefill_draft=True)
    for zero_w in (False, True):
        opts = rf.UnpaddedFloorOptions(dtypes=dtypes, zero_weights=zero_w)
        b_off = rf.unpadded_strict_resource_buckets(geom, chip, par, shape_off, opts)
        b_on = rf.unpadded_strict_resource_buckets(geom, chip, par, shape_on, opts)
        assert b_on[0] > b_off[0]
        assert b_on[1] > b_off[1]


@pytest.mark.parametrize(
    ("block_size", "num_layers", "accept_len", "expected_k"),
    [
        (7, 3, 2.5, 7),
        (1, 5, 2.5, 5),
        (3, 2, 5.2, 5),
    ],
)
def test_resolve_spec_k_with_draft_spec(
    block_size: int,
    num_layers: int,
    accept_len: float,
    expected_k: int,
) -> None:
    """Verify resolve_spec_k picks max(block_size, num_layers, ceil(accept_len)-1)."""
    draft = rf.DraftSpec(
        kind="dspark",
        num_layers=num_layers,
        layer_kind="full_attention",
        block_size=block_size,
    )
    geom = replace(_base_msa_geometry(), draft=draft)
    scen = replace(
        rf.build_example_scenario("chat-8k1k"),
        accept_len=accept_len,
        spec_tokens=None,
    )
    assert schema.resolve_spec_k(geom, scen) == expected_k


def test_from_hf_config_draft_spec_variants_and_fail_closed() -> None:
    """Verify from_hf_config DraftSpec parsing across layouts and invalid inputs."""
    base_cfg = load_locked_config("qwen3_5_397b_fp8.json")
    dense_draft_cfg: dict[str, object] = {
        "num_hidden_layers": 5,
        "dspark_use_dense_ffn": True,
        "dspark_ffn_hidden_size": 4096,
        "dspark_target_layer_ids": [10, 30, 50],
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 4096,
        "dspark_block_size": 7,
        "dspark_markov_rank": 128,
    }
    dense_geom = rf.from_hf_config(base_cfg, draft_config=dense_draft_cfg)
    assert dense_geom.draft == rf.DraftSpec(
        kind="dspark",
        num_layers=5,
        layer_kind="sliding_attention",
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        dense_inter=4096,
        sliding_window=4096,
        block_size=7,
        fc_in_mult=3,
        fc_replicated=False,
        markov_rank=128,
        draft_kv_dtype="",
        skip_draft_kv_on_offload=False,
        single_row_lm_head_verify=False,
    )

    moe_draft_cfg: dict[str, object] = {
        "num_hidden_layers": 3,
        "dspark_use_dense_ffn": False,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 8,
        "dspark_target_layer_ids": [5, 15],
        "layer_types": ["full_attention"] * 3,
        "dspark_block_size": 4,
    }
    moe_geom = rf.from_hf_config(base_cfg, draft_config=moe_draft_cfg)
    assert moe_geom.draft == rf.DraftSpec(
        kind="dspark",
        num_layers=3,
        layer_kind="full_attention",
        num_experts=128,
        top_k=8,
        moe_inter=0,
        shared_inter=0,
        dense_inter=0,
        sliding_window=0,
        block_size=4,
        fc_in_mult=2,
        fc_replicated=False,
        markov_rank=0,
        draft_kv_dtype="",
        skip_draft_kv_on_offload=False,
        single_row_lm_head_verify=False,
    )

    text_cfg_draft = rf.from_hf_config(
        base_cfg, draft_config={"text_config": dense_draft_cfg}
    )
    assert text_cfg_draft.draft == dense_geom.draft

    dflash_draft = rf.from_hf_config(
        base_cfg,
        draft_config={
            "dflash_config": {
                "num_hidden_layers": 2,
                "target_layer_ids": [1, 2, 3, 4],
                "layer_types": ["swa", "swa"],
                "sliding_window": 256,
                "block_size": 6,
            }
        },
    )
    assert dflash_draft.draft == rf.DraftSpec(
        kind="dspark",
        num_layers=2,
        layer_kind="swa",
        sliding_window=256,
        block_size=6,
        fc_in_mult=4,
    )

    for empty_or_zero in ({}, {"num_hidden_layers": 0}):
        with pytest.raises(ValueError, match="num_hidden_layers"):
            rf.from_hf_config(base_cfg, draft_config=empty_or_zero)

    with pytest.raises(rf.UnsupportedArchitectureError, match="Unknown draft"):
        rf.from_hf_config(
            base_cfg,
            draft_config={"num_hidden_layers": 2, "layer_types": ["unknown_attn"]},
        )

    with pytest.raises(rf.UnsupportedArchitectureError, match="Mixed layer_types"):
        rf.from_hf_config(
            base_cfg,
            draft_config={
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "sliding_attention"],
            },
        )


def test_qsa_prefill_index_key_write_continuity_across_budget() -> None:
    """Verify QSA prefill HBM writes compressed index keys smoothly across budget."""
    chip = rf.CHIPS["tpu7x"]
    dtypes = rf.Dtypes(w_dense=FP8_BYTES, w_expert=MXFP4_BYTES, kv=FP8_BYTES)
    par = rf.Parallel(tp=8)
    geom = replace(
        _base_qsa_geometry(),
        num_layers=2,
        layer_types=("indexed_attention", "indexed_attention"),
        sparse_attn=replace(
            _base_qsa_geometry().sparse_attn,
            kv_source_layer_ids=(0, 1),
            index_source_layer_ids=(0, 1),
        ),
    )
    budget = geom.qsa_budget

    def _gqa_hbm(prefill_tokens: int) -> float:
        """Return GQA module HBM bytes for one prefill token count."""
        mods = rf.pass_costs(
            geom,
            par,
            rf.StepShape(batch=0, rows_per_seq=0, prefill_tokens=prefill_tokens),
            dtypes,
            chip=chip,
        )
        return next(mod.hbm_bytes for mod in mods if mod.name == "gqa")

    h_below_0 = _gqa_hbm(budget - 1)
    h_at = _gqa_hbm(budget)
    h_above_1 = _gqa_hbm(budget + 1)
    marginal_below = h_at - h_below_0
    marginal_above = h_above_1 - h_at
    # Physical 128B row packing on 2 KV rows pads to 4 * 256B = _QSA_MAIN_BYTES (1024B).
    expected_marginal = 2.0 * (_QSA_MAIN_BYTES + _QSA_COMP_K_BYTES)
    assert marginal_below == pytest.approx(expected_marginal)
    assert marginal_above == pytest.approx(expected_marginal)
