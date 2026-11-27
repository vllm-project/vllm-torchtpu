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
"""Pure-physics oracle and invariance tests for Qwen3.8-Flash-Next."""

from __future__ import annotations

from dataclasses import replace

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import load_locked_config, override_locked_config
from vllm_torchtpu.roofline.constants import BF16_BYTES, FP8_BYTES, UNIT_GIBI
from vllm_torchtpu.roofline.modules import kv_bytes_per_token_per_layer
from vllm_torchtpu.roofline.partition import scale_pp_stage_floor, stage_slice_geometry

_FIXTURE_NAME = "qwen3_8_flash_next_fp8.json"
_HF_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
_UNIT_BYTE_TOL = 1e-6

# Model dimensions from the authentic Qwen3.8-Flash-Next-FP8 HuggingFace config.
_NUM_LAYERS = 48
_FULL_ATTN_LAYERS = 12
_LINEAR_ATTN_LAYERS = 36
_PLE_LAYERS = 1
_MTP_LAYERS = 1
_HIDDEN_SIZE = 2560
_VOCAB_SIZE = 248320
_NUM_HEADS = 24
_NUM_KV_HEADS = 2
_HEAD_DIM = 256
_INDEXER_HEADS = 4
_INDEXER_HEAD_DIM = 128
_INDEXER_BUDGET = 2048
_INDEXER_COMPRESS_RATIO = 4
_GDN_K_HEADS = 16
_GDN_V_HEADS = 48
_GDN_HEAD_DIM = 128
_GDN_CONV_KERNEL = 4
_NUM_EXPERTS = 512
_TOP_K = 10
_MOE_INTER = 640
_SHARED_INTER = 640
_HC_COUNT = 4
_HC_LOWRANK = 320
_PLE_EMBED_DIM = 2560
_PLE_CONV_KERNEL = 4
_NGRAM_SIZE = 3
_PLE_SPLIT_PARTS = 128
_PLE_ROWS_PER_PART = 2_500_012
_PLE_SHARD_DIM = 160
_QSA_RING_ROWS = 8
_SWIGLU_MATS = 3
_TP8 = 8
_PP8 = 8

_EXPECTED_QSA_LAYER_IDS: tuple[int, ...] = tuple(
    idx for idx in range(_NUM_LAYERS) if (idx + 1) % 4 == 0
)
_PP8_PARTITIONS: tuple[tuple[int, ...], ...] = (
    (4, 5, 7, 7, 7, 7, 7, 4),
    (6, 6, 6, 6, 6, 6, 7, 5),
)

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


def _assert_param_conservation(geometry: rf.ModelGeometry) -> None:
    """Assert closed-form parameter and unit-dtype conservation for Qwen3.8."""
    params = rf.param_counts(geometry)
    expected_routed = float(
        _NUM_LAYERS * _NUM_EXPERTS * _SWIGLU_MATS * _MOE_INTER * _HIDDEN_SIZE
    )
    expected_shared = float(_NUM_LAYERS * _SWIGLU_MATS * _SHARED_INTER * _HIDDEN_SIZE)
    expected_router = float(_NUM_LAYERS * _HIDDEN_SIZE * _NUM_EXPERTS)
    expected_qsa_proj = float(
        _FULL_ATTN_LAYERS
        * (
            2 * _NUM_HEADS * _HEAD_DIM
            + 2 * _NUM_KV_HEADS * _HEAD_DIM
            + _NUM_HEADS * _HEAD_DIM
        )
        * _HIDDEN_SIZE
    )
    expected_qsa_idx = float(
        _FULL_ATTN_LAYERS * (_INDEXER_HEADS + 1) * _INDEXER_HEAD_DIM * _HIDDEN_SIZE
    )
    gdn_qkv = 2 * _GDN_K_HEADS * _GDN_HEAD_DIM + _GDN_V_HEADS * _GDN_HEAD_DIM
    gdn_z = _GDN_V_HEADS * _GDN_HEAD_DIM
    expected_rec = float(
        _LINEAR_ATTN_LAYERS
        * (
            _HIDDEN_SIZE * (gdn_qkv + gdn_z)
            + gdn_z * _HIDDEN_SIZE
            + _HIDDEN_SIZE * 2 * _GDN_V_HEADS
            + gdn_qkv * _GDN_CONV_KERNEL
        )
    )
    expected_emb = float(_VOCAB_SIZE * _HIDDEN_SIZE)
    expected_hc = float(_NUM_LAYERS * 2 * _HIDDEN_SIZE * _HC_LOWRANK * _HC_COUNT)
    expected_ple_table = float(_PLE_SPLIT_PARTS * _PLE_ROWS_PER_PART * _PLE_SHARD_DIM)
    expected_ple_proj = float(
        _PLE_EMBED_DIM * (_HIDDEN_SIZE * _HC_COUNT + _HIDDEN_SIZE)
        + (_HIDDEN_SIZE * _HC_COUNT) * _PLE_CONV_KERNEL
    )
    expected_mtp_routed = float(
        _MTP_LAYERS * _NUM_EXPERTS * _SWIGLU_MATS * _MOE_INTER * _HIDDEN_SIZE
    )
    expected_mtp = (
        (expected_qsa_proj + expected_qsa_idx) / float(_FULL_ATTN_LAYERS)
        + expected_mtp_routed
        + (expected_shared + expected_router) / float(_NUM_LAYERS)
        + expected_hc / float(_NUM_LAYERS)
        + float(2 * _HIDDEN_SIZE * _HIDDEN_SIZE)
    )

    assert params.moe_routed_ffn == pytest.approx(expected_routed)
    assert (params.moe_routed_ffn * FP8_BYTES) / UNIT_GIBI == pytest.approx(112.5)
    assert params.moe_shared_ffn == pytest.approx(expected_shared)
    assert params.router == pytest.approx(expected_router)
    assert params.attention == pytest.approx(expected_qsa_proj + expected_qsa_idx)
    assert params.linear_attention == pytest.approx(expected_rec)
    assert params.embedding == pytest.approx(expected_emb)
    assert params.lm_head == pytest.approx(expected_emb)
    assert params.hc == pytest.approx(expected_hc)
    assert params.host_tables == pytest.approx(expected_ple_table + expected_ple_proj)
    assert params.mtp == pytest.approx(expected_mtp)

    # Trunk, +PLE, +MTP totals match safetensors manifest within omitted norms/mixers.
    trunk_params = params.total_params - expected_ple_table - params.mtp
    assert trunk_params == pytest.approx(125.74e9, rel=5e-3)
    assert (trunk_params + expected_ple_table) == pytest.approx(176.94e9, rel=5e-3)
    assert params.total_params == pytest.approx(179.55e9, rel=5e-3)
    active_excl_embed = params.active_params_per_token - params.embedding
    expected_mtp_active = expected_mtp - float(
        _MTP_LAYERS * (_NUM_EXPERTS - _TOP_K) * _SWIGLU_MATS * _HIDDEN_SIZE * _MOE_INTER
    )
    expected_active_excl_embed = (
        expected_qsa_proj
        + expected_qsa_idx
        + expected_rec
        + expected_routed * (float(_TOP_K) / float(_NUM_EXPERTS))
        + expected_shared
        + expected_router
        + expected_hc
        + expected_ple_proj
        + expected_emb
        + expected_mtp_active
    )
    assert active_excl_embed == pytest.approx(expected_active_excl_embed)

    # Unit-dtype weight conservation holds across every host-table residency mode.
    chip = rf.CHIPS["tpu7x"]
    for mode in ("auto", "hbm", "host", "first"):
        offloaded = rf.offloaded_host_table_params(geometry, mode)
        hbm = rf.hbm_breakdown_per_core_bytes(
            geometry,
            _UNIT_DTYPES,
            _UNIT_PARALLEL,
            rf.SeqKvLayout(0, 0, host_table_residency=mode),
            chip=chip,
        )
        assert abs((params.total_params - offloaded) - hbm.weights) <= _UNIT_BYTE_TOL


def _assert_ple_oracle(
    geometry: rf.ModelGeometry,
    dtypes: rf.Dtypes,
) -> None:
    """Assert PLE 0-based layer ID, table/H2D/state bytes, and PCIe floor exclusion."""
    chip = rf.CHIPS["tpu7x"]
    counts = rf.count_layer_types(geometry)
    assert (counts.gdn, counts.gqa) == (_LINEAR_ATTN_LAYERS, _FULL_ATTN_LAYERS)

    # 1-based HF ple_layer_ids=[2] converts to 0-based index (1,).
    assert geometry.ple_layer_ids == (1,)
    assert len(geometry.host_tables) == _PLE_LAYERS
    ple_spec = geometry.host_tables[0]
    expected_ple_elems = _PLE_SPLIT_PARTS * _PLE_ROWS_PER_PART * _PLE_SHARD_DIM
    assert ple_spec.name == "ple"
    assert ple_spec.layer_ids == (1,)
    assert ple_spec.elem_count == expected_ple_elems
    assert ple_spec.shard_axis == "tp"
    assert ple_spec.h2d_bytes_per_tok == pytest.approx(
        float(_PLE_EMBED_DIM) * FP8_BYTES
    )
    expected_ple_state = (
        float((_PLE_CONV_KERNEL - 1) * _NGRAM_SIZE * (_HIDDEN_SIZE * _HC_COUNT))
        * BF16_BYTES
    )
    assert ple_spec.state_bytes_per_req == pytest.approx(expected_ple_state)

    # Pure FP8 PLE footprint: 47.684 GiB total, 5.960 GiB/core at TP8.
    fp8_ple_dt = replace(dtypes, w_expert=FP8_BYTES, kv=FP8_BYTES, kv_dtype="fp8")
    tp8_par = rf.Parallel(tp=_TP8, ep=_TP8, moe="ep")
    bd_hbm = rf.hbm_breakdown_per_core_bytes(
        geometry,
        fp8_ple_dt,
        tp8_par,
        rf.SeqKvLayout(1, 0, host_table_residency="hbm"),
        chip=chip,
    )
    bd_host = rf.hbm_breakdown_per_core_bytes(
        geometry,
        fp8_ple_dt,
        tp8_par,
        rf.SeqKvLayout(1, 0, host_table_residency="host"),
        chip=chip,
    )
    ple_total_gib = float(expected_ple_elems) * FP8_BYTES / UNIT_GIBI
    assert round(ple_total_gib, 4) == pytest.approx(47.6839)
    assert (bd_hbm.weights - bd_host.weights) / UNIT_GIBI == pytest.approx(
        ple_total_gib / float(_TP8), rel=1e-6
    )
    assert round((bd_hbm.weights - bd_host.weights) / UNIT_GIBI, 5) == pytest.approx(
        5.96049
    )

    # Offloaded PLE PCIe H2D stays out of hard_floor_s and skips legacy ple.
    shape_host = rf.StepShape(
        batch=8, rows_per_seq=1, context=4096, host_table_residency="host"
    )
    shape_hbm = replace(shape_host, host_table_residency="hbm")
    mods_host = rf.pass_costs(geometry, tp8_par, shape_host, fp8_ple_dt, chip=chip)
    mod_names = {mod.name for mod in mods_host}
    assert "host_table_ple" in mod_names
    assert "ple" not in mod_names
    fl_host = rf.step_floor(
        geometry, chip, tp8_par, shape_host, rf.FloorOptions(dtypes=fp8_ple_dt)
    )
    fl_hbm = rf.step_floor(
        geometry, chip, tp8_par, shape_hbm, rf.FloorOptions(dtypes=fp8_ple_dt)
    )
    assert fl_host.pcie_s > 0.0
    assert fl_hbm.pcie_s == 0.0
    assert fl_host.hard_floor_s <= fl_hbm.hard_floor_s
    assert fl_host.hard_floor_s <= fl_host.envelope_floor_s <= fl_host.serial_estimate_s


def _assert_qsa_and_pd_oracle(
    geometry: rf.ModelGeometry,
    dtypes: rf.Dtypes,
) -> None:
    """Assert QSA compressed KV/ring state, 1L hybrid MTP, and PD transfer bytes."""
    chip = rf.CHIPS["tpu7x"]
    counts = rf.count_layer_types(geometry)
    fp8_ple_dt = replace(dtypes, w_expert=FP8_BYTES, kv=FP8_BYTES, kv_dtype="fp8")

    # QSA sparse attention: 1024 B main + 64 B compressed K + 8-row raw ring.
    sparse = geometry.sparse_attn
    assert sparse is not None
    assert sparse.kind == "qsa"
    assert sparse.kv_source_layer_ids == _EXPECTED_QSA_LAYER_IDS
    assert sparse.index_source_layer_ids == _EXPECTED_QSA_LAYER_IDS
    expected_main_kv = float(2 * _NUM_KV_HEADS * _HEAD_DIM) * FP8_BYTES
    expected_comp_k = (
        float(_INDEXER_HEAD_DIM) * BF16_BYTES / float(_INDEXER_COMPRESS_RATIO)
    )
    assert sparse.compressed_key_bytes_per_tok == pytest.approx(expected_comp_k)
    assert sparse.fixed_state_page_bytes == pytest.approx(
        float(_QSA_RING_ROWS) * expected_main_kv
    )

    qsa_tp8_b = kv_bytes_per_token_per_layer(
        geometry, fp8_ple_dt, rf.Parallel(tp=_TP8), "gqa", physical=True, chip=chip
    )
    assert qsa_tp8_b == pytest.approx(expected_main_kv + expected_comp_k)

    # 1L hybrid MTP DraftSpec and PD transfer: 13,056 / 14,144 B/tok + 49 / 50 pages.
    draft = geometry.draft
    assert draft is not None
    assert draft.kind == "mtp"
    assert draft.num_layers == _MTP_LAYERS
    assert draft.layer_kind == "indexed_attention"
    assert draft.num_experts == _NUM_EXPERTS
    assert draft.top_k == _TOP_K
    assert draft.moe_inter == _MOE_INTER
    assert draft.shared_inter == _SHARED_INTER

    tp1_par = rf.Parallel(tp=1, pp=1)
    geom_no_draft = replace(geometry, draft=None, mtp_layers=0)
    pd_no_draft_b = rf.hbm_breakdown_per_core_bytes(
        geom_no_draft,
        fp8_ple_dt,
        tp1_par,
        rf.SeqKvLayout(1, 1, physical_kv_layout=False),
        chip=chip,
    ).kv
    pd_with_draft_b = rf.hbm_breakdown_per_core_bytes(
        geometry,
        fp8_ple_dt,
        tp1_par,
        rf.SeqKvLayout(1, 1, physical_kv_layout=False),
        chip=chip,
    ).kv
    assert pd_no_draft_b == pytest.approx(
        float(_FULL_ATTN_LAYERS) * (expected_main_kv + expected_comp_k)
    )
    assert pd_with_draft_b == pytest.approx(
        float(_FULL_ATTN_LAYERS + _MTP_LAYERS) * (expected_main_kv + expected_comp_k)
    )
    assert counts.gdn == _LINEAR_ATTN_LAYERS
    ple_spec = geometry.host_tables[0]
    expected_gdn_state = float(_LINEAR_ATTN_LAYERS) * (
        float(_GDN_V_HEADS * _GDN_HEAD_DIM * _GDN_HEAD_DIM) * fp8_ple_dt.state
        + float(
            (2 * _GDN_K_HEADS * _GDN_HEAD_DIM + _GDN_V_HEADS * _GDN_HEAD_DIM)
            * (_GDN_CONV_KERNEL - 1)
        )
        * fp8_ple_dt.conv
    )
    expected_qsa_ring_fp8 = (
        float(len(sparse.kv_source_layer_ids)) * sparse.fixed_state_page_bytes
    )
    expected_state_fp8 = (
        expected_gdn_state + ple_spec.state_bytes_per_req + expected_qsa_ring_fp8
    )
    kv_layout = rf.SeqKvLayout(1, 1, physical_kv_layout=False)
    state_no_draft = rf.hbm_breakdown_per_core_bytes(
        geom_no_draft, fp8_ple_dt, tp1_par, kv_layout, chip=chip
    ).state
    state_with_draft = rf.hbm_breakdown_per_core_bytes(
        geometry, fp8_ple_dt, tp1_par, kv_layout, chip=chip
    ).state
    assert state_no_draft == pytest.approx(expected_state_fp8)
    assert state_with_draft == pytest.approx(expected_state_fp8)

    # Switching --kv-dtype to bf16 scales QSA fixed_state_page_bytes by 2x.
    bf16_kv_dt = replace(fp8_ple_dt, kv=BF16_BYTES, kv_dtype="bf16")
    state_bf16_no_draft = rf.hbm_breakdown_per_core_bytes(
        geom_no_draft, bf16_kv_dt, tp1_par, kv_layout, chip=chip
    ).state
    state_bf16_with_draft = rf.hbm_breakdown_per_core_bytes(
        geometry, bf16_kv_dt, tp1_par, kv_layout, chip=chip
    ).state
    assert (state_bf16_no_draft - state_no_draft) == pytest.approx(
        expected_qsa_ring_fp8
    )
    assert (state_bf16_with_draft - state_with_draft) == pytest.approx(
        expected_qsa_ring_fp8
    )


def _assert_pp8_slicing(geometry: rf.ModelGeometry, dtypes: rf.Dtypes) -> None:
    """Assert PP8 stage slicing places PLE on stage 0 and MTP draft on stage 7."""
    chip = rf.CHIPS["tpu7x"]
    fp8_dt = replace(dtypes, kv=FP8_BYTES, kv_dtype="fp8")
    pp8_par = rf.Parallel(tp=1, pp=_PP8)
    opts = rf.FloorOptions(dtypes=fp8_dt)

    for partition in _PP8_PARTITIONS:
        assert len(partition) == _PP8
        assert sum(partition) == _NUM_LAYERS
        stages: list[rf.ModelGeometry] = []
        cursor = 0
        for count in partition:
            stages.append(stage_slice_geometry(geometry, cursor, cursor + count))
            cursor += count

        # Stage 0 holds 0-indexed PLE layer 1; middle stages hold neither PLE nor draft.
        assert stages[0].ple_layer_ids == (1,)
        assert len(stages[0].host_tables) == _PLE_LAYERS
        assert stages[0].host_tables[0].layer_ids == (1,)
        assert stages[0].draft is None
        for middle in stages[1:-1]:
            assert middle.ple_layer_ids == ()
            assert middle.host_tables == ()
            assert middle.draft is None
        assert stages[-1].ple_layer_ids == ()
        assert stages[-1].host_tables == ()
        assert stages[-1].draft == geometry.draft

        # QSA source layers sum cleanly across all 8 pipeline stages.
        qsa_total = sum(
            len(stage.sparse_attn.kv_source_layer_ids)
            for stage in stages
            if stage.sparse_attn is not None
        )
        assert qsa_total == _FULL_ATTN_LAYERS

        for prefill_draft in (False, True):
            shape = rf.StepShape(
                batch=0,
                rows_per_seq=0,
                prefill_tokens=4096,
                host_table_residency="host",
                prefill_draft=prefill_draft,
            )
            pp_floor = scale_pp_stage_floor(
                geometry, chip, pp8_par, (shape, opts), partition
            )
            assert pp_floor.hard_floor_s > 0.0
            assert pp_floor.hard_floor_s <= pp_floor.envelope_floor_s
            assert pp_floor.envelope_floor_s <= pp_floor.serial_estimate_s


def test_qwen38_flash_next_oracle_ple_and_pp8_slicing() -> None:
    """Verify Qwen3.8-Flash-Next audit, param/PLE/QSA/MTP/PD oracles, and PP8 splits."""
    config = load_locked_config(_FIXTURE_NAME)
    audit = rf.audit_hf_config(config, model_ref=_HF_REPO)
    assert audit.onboarding_tier == "tier0_ready"
    assert audit.model_type_status == "listed"
    assert audit.unmapped_structural_keys == ()
    assert audit.blocking_reasons == ()
    assert audit.geometry is not None
    assert audit.dtypes is not None
    assert audit.geometry.model_id == "qwen4_exp"
    assert audit.dtypes.w_dense == pytest.approx(BF16_BYTES)
    assert audit.dtypes.w_expert > FP8_BYTES
    assert audit.dtypes.act == "a8"

    _assert_param_conservation(audit.geometry)
    _assert_ple_oracle(audit.geometry, audit.dtypes)
    _assert_qsa_and_pd_oracle(audit.geometry, audit.dtypes)
    _assert_pp8_slicing(audit.geometry, audit.dtypes)


def test_qwen38_flash_next_scaling_and_serving_invariants() -> None:
    """Verify QSA long-context HBM savings, MTP toggle, and end-to-end solve bounds."""
    config = load_locked_config(_FIXTURE_NAME)
    geometry = rf.from_hf_config(config, model_ref=_HF_REPO)
    dtypes = replace(rf.dtypes_from_hf_config(config), kv=FP8_BYTES, kv_dtype="fp8")
    chip = rf.CHIPS["tpu7x"]
    tp8_par = rf.Parallel(tp=_TP8, ep=_TP8, moe="ep")

    # Bounded QSA context (2048) reads strictly less HBM KV than dense GQA at 32k.
    dense_geom = replace(geometry, sparse_attn=None)
    long_shape = rf.StepShape(
        batch=4, rows_per_seq=1, context=32768, host_table_residency="hbm"
    )
    qsa_floor = rf.step_floor(
        geometry, chip, tp8_par, long_shape, rf.FloorOptions(dtypes=dtypes)
    )
    dense_floor = rf.step_floor(
        dense_geom, chip, tp8_par, long_shape, rf.FloorOptions(dtypes=dtypes)
    )
    assert qsa_floor.hbm_s < dense_floor.hbm_s

    # Disabling MTP via HF override clears DraftSpec cleanly.
    no_mtp_cfg = override_locked_config(
        _FIXTURE_NAME, {"text_config": {"mtp_num_hidden_layers": 0}}
    )
    no_mtp_geom = rf.from_hf_config(no_mtp_cfg, model_ref=_HF_REPO)
    assert no_mtp_geom.draft is None
    assert no_mtp_geom.mtp_layers == 0

    # End-to-end mixed and disaggregated solve satisfy physical bound <= estimate.
    for residency in ("hbm", "host"):
        scenario = rf.TrafficScenario(
            name=f"qwen38_mixed_{residency}",
            isl=4096,
            osl=512,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(16,),
            kv_working_set_turns=1.0,
            host_table_residency=residency,
        )
        plan = rf.solve_serving_plan(
            geometry,
            dtypes,
            chip,
            scenario,
            rf.SolveOptions(
                max_chips=4,
                mode="mixed",
            ),
        )
        assert plan.candidates
        for candidate in plan.candidates:
            assert candidate.bound_ttft_ms <= candidate.est_ttft_ms + 1e-6
            assert candidate.bound_tpot_ms <= candidate.est_tpot_ms + 1e-6

    disagg_scenario = rf.TrafficScenario(
        name="qwen38_disagg_hbm",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=2.3,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
        host_table_residency="hbm",
        prefill_draft=True,
    )
    disagg_plan = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        disagg_scenario,
        rf.SolveOptions(
            max_chips=8,
            mode="disagg",
            include_pairs=True,
        ),
    )
    assert disagg_plan.candidates
    for candidate in disagg_plan.candidates:
        assert candidate.bound_ttft_ms <= candidate.est_ttft_ms + 1e-6
        assert candidate.bound_tpot_ms <= candidate.est_tpot_ms + 1e-6


_OVERRIDE_KV_HEADS = 4
_DOUBLED_PAGE_SCALE = 2.0
_INVALID_SPLIT_NGRAM_PARTS = 0
_INVALID_PLE_EMBED_DIM = 2561


def test_qwen38_flash_qsa_ring_page_scales_with_kv_heads() -> None:
    """Verify QSA fixed_state_page_bytes scales linearly with num_key_value_heads."""
    base_cfg = load_locked_config(_FIXTURE_NAME)
    base_geom = rf.from_hf_config(base_cfg, model_ref=_HF_REPO)
    doubled_cfg = override_locked_config(
        _FIXTURE_NAME,
        {"text_config": {"num_key_value_heads": _OVERRIDE_KV_HEADS}},
    )
    doubled_geom = rf.from_hf_config(doubled_cfg, model_ref=_HF_REPO)
    assert base_geom.sparse_attn is not None
    assert doubled_geom.sparse_attn is not None
    assert doubled_geom.sparse_attn.fixed_state_page_bytes == pytest.approx(
        _DOUBLED_PAGE_SCALE * base_geom.sparse_attn.fixed_state_page_bytes
    )


@pytest.mark.parametrize(
    ("text_patch", "match"),
    [
        ({"split_ngram_parts": _INVALID_SPLIT_NGRAM_PARTS}, "split_ngram_parts"),
        ({"ple_embed_dim": _INVALID_PLE_EMBED_DIM}, "divisible by"),
    ],
)
def test_qwen38_flash_fail_closed_ple_validation(
    text_patch: dict[str, object],
    match: str,
) -> None:
    """Verify non-positive split_ngram_parts and non-divisible ple_embed_dim fail."""
    bad_cfg = override_locked_config(_FIXTURE_NAME, {"text_config": text_patch})
    with pytest.raises(ValueError, match=match):
        rf.from_hf_config(bad_cfg, model_ref=_HF_REPO)
