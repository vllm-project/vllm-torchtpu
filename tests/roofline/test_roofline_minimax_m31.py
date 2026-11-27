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
"""Oracle and invariance tests for MiniMax-M3.1-Flash pure-physics roofline."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import (
    FIXTURES_DIR,
    load_locked_config,
    load_locked_draft_config,
)
from vllm_torchtpu.roofline import cli
from vllm_torchtpu.roofline.constants import (
    BF16_BYTES,
    BIN_MEBI,
    FP4_BYTES,
    FP8_BYTES,
    INT4_PER_TOKEN_HEAD_KV_DTYPE,
    MXFP8_BYTES,
    NVFP4_BYTES,
    UNIT_GIBI,
)
from vllm_torchtpu.roofline.geometry import slice_geometry_layers
from vllm_torchtpu.roofline.hf_loader import audit_hf_config
from vllm_torchtpu.roofline.modules import kv_bytes_per_token_per_layer

_MAIN_FIXTURE = "minimax_m3_1_flash.json"
_DRAFT_FIXTURE = "minimax_m3_1_flash_dspark.json"
_HF_REPO = "MiniMaxAI/MiniMax-M3.1-Flash"
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
_UNIT_PARALLEL = rf.Parallel(tp=1, pp=1, ep=1)
_UNIT_BYTE_TOL = 1e-6

# Independent closed-form MiniMax-M3.1-Flash geometry constants from HF config.
_NUM_LAYERS = 60
_DENSE_LAYERS = 3
_MOE_LAYERS = 57
_HIDDEN_SIZE = 6144
_VOCAB_SIZE = 200064
_NUM_HEADS = 64
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_INDEX_HEADS = 4
_INDEX_DIM = 128
_DENSE_INTER = 12288
_MOE_INTER = 3072
_SHARED_INTER = 3072
_NUM_EXPERTS = 128
_TOP_K = 4
_SPARSE_BLOCK_SIZE = 128
_SPARSE_TOPK_BLOCKS = 16
_SPARSE_LOCAL_BLOCK = 1
_DSPARK_LAYERS = 5
_DSPARK_DENSE_INTER = 12288
_DSPARK_SWA = 4096
_DSPARK_BLOCK_SIZE = 7
_DSPARK_TARGET_LAYERS = 5
_DSPARK_MARKOV_RANK = 256
_SWIGLU_MATS = 3
_HALF_LAYERS = _NUM_LAYERS // 2


def _load_mm31_configs() -> tuple[dict[str, object], dict[str, object]]:
    """Load authentic MiniMax-M3.1-Flash target and companion draft JSON configs."""
    main_cfg = load_locked_config(_MAIN_FIXTURE)
    draft_cfg = load_locked_draft_config(_MAIN_FIXTURE)
    assert draft_cfg is not None
    return main_cfg, draft_cfg


def _assert_trunk_oracle(
    trunk_geom: rf.ModelGeometry,
    chip: rf.ChipSpec,
) -> float:
    """Assert MiniMax-M3.1-Flash trunk geometry and parameter conservation."""
    assert trunk_geom.num_layers == _NUM_LAYERS
    assert trunk_geom.ffn_layer_counts() == (_MOE_LAYERS, _DENSE_LAYERS)
    assert trunk_geom.dense_inter == _DENSE_INTER
    assert trunk_geom.moe_inter == _MOE_INTER
    assert trunk_geom.shared_inter == _SHARED_INTER
    assert trunk_geom.index_n_heads == _INDEX_HEADS
    assert trunk_geom.index_head_dim == _INDEX_DIM
    assert trunk_geom.sparse_attn is not None
    assert trunk_geom.sparse_attn.kind == "msa"
    assert trunk_geom.sparse_attn.sparse_block_size == _SPARSE_BLOCK_SIZE
    assert trunk_geom.sparse_attn.sparse_topk_blocks == _SPARSE_TOPK_BLOCKS
    assert trunk_geom.sparse_attn.sparse_local_block == _SPARSE_LOCAL_BLOCK
    assert trunk_geom.sparse_attn.dcp_span == "tp"
    assert trunk_geom.sparse_attn.index_cache_sharded is False

    exp_main_attn = float(
        _NUM_LAYERS
        * (
            _NUM_HEADS * _HEAD_DIM
            + 2 * _NUM_KV_HEADS * _HEAD_DIM
            + _NUM_HEADS * _HEAD_DIM
        )
        * _HIDDEN_SIZE
    )
    exp_main_index = float(
        _NUM_LAYERS * (_INDEX_HEADS * _INDEX_DIM + _INDEX_DIM) * _HIDDEN_SIZE
    )
    exp_dense_ffn = float(_DENSE_LAYERS * _SWIGLU_MATS * _DENSE_INTER * _HIDDEN_SIZE)
    exp_shared_ffn = float(_MOE_LAYERS * _SWIGLU_MATS * _SHARED_INTER * _HIDDEN_SIZE)
    exp_routed_ffn = float(
        _MOE_LAYERS * _NUM_EXPERTS * _SWIGLU_MATS * _MOE_INTER * _HIDDEN_SIZE
    )
    exp_router = float(_MOE_LAYERS * _HIDDEN_SIZE * _NUM_EXPERTS)
    exp_embed = float(_VOCAB_SIZE * _HIDDEN_SIZE)
    exp_trunk_total = (
        exp_main_attn
        + exp_main_index
        + exp_dense_ffn
        + exp_shared_ffn
        + exp_routed_ffn
        + exp_router
        + 2.0 * exp_embed
    )
    exp_inactive_routed = float(
        _MOE_LAYERS * (_NUM_EXPERTS - _TOP_K) * _SWIGLU_MATS * _MOE_INTER * _HIDDEN_SIZE
    )

    trunk_params = rf.param_counts(trunk_geom)
    assert trunk_params.attention == pytest.approx(exp_main_attn + exp_main_index)
    assert trunk_params.dense_ffn == pytest.approx(exp_dense_ffn)
    assert trunk_params.moe_shared_ffn == pytest.approx(exp_shared_ffn)
    assert trunk_params.moe_routed_ffn == pytest.approx(exp_routed_ffn)
    assert trunk_params.router == pytest.approx(exp_router)
    assert trunk_params.embedding == pytest.approx(exp_embed)
    assert trunk_params.lm_head == pytest.approx(exp_embed)
    assert trunk_params.total_params == pytest.approx(exp_trunk_total)
    assert (trunk_params.active_params_per_token - trunk_params.embedding) == (
        pytest.approx(exp_trunk_total - exp_inactive_routed - exp_embed)
    )
    assert (exp_routed_ffn * NVFP4_BYTES) / UNIT_GIBI == pytest.approx(216.421875)
    trunk_unit_hbm = rf.hbm_breakdown_per_core_bytes(
        trunk_geom, _UNIT_DTYPES, _UNIT_PARALLEL, rf.SeqKvLayout(0, 0), chip=chip
    )
    assert abs(trunk_params.total_params - trunk_unit_hbm.weights) <= _UNIT_BYTE_TOL
    return exp_trunk_total


def _assert_dspark_and_msa_kv_oracle(
    full_geom: rf.ModelGeometry,
    dtypes: rf.Dtypes,
    chip: rf.ChipSpec,
    exp_trunk_total: float,
) -> None:
    """Assert companion DSpark conservation and FP8/int4 MSA KV byte formulas."""
    assert full_geom.draft is not None
    assert full_geom.draft.kind == "dspark"
    assert full_geom.draft.num_layers == _DSPARK_LAYERS
    assert full_geom.draft.dense_inter == _DSPARK_DENSE_INTER
    assert full_geom.draft.sliding_window == _DSPARK_SWA
    assert full_geom.draft.block_size == _DSPARK_BLOCK_SIZE
    assert full_geom.draft.fc_in_mult == _DSPARK_TARGET_LAYERS
    assert full_geom.draft.markov_rank == _DSPARK_MARKOV_RANK
    assert full_geom.draft.draft_kv_dtype == "fp8"
    assert full_geom.draft.skip_draft_kv_on_offload is True
    assert full_geom.draft.single_row_lm_head_verify is True

    exp_dspark_attn = float(
        _DSPARK_LAYERS
        * (
            _NUM_HEADS * _HEAD_DIM
            + 2 * _NUM_KV_HEADS * _HEAD_DIM
            + _NUM_HEADS * _HEAD_DIM
        )
        * _HIDDEN_SIZE
    )
    exp_dspark_mlp = float(
        _DSPARK_LAYERS * _SWIGLU_MATS * _DSPARK_DENSE_INTER * _HIDDEN_SIZE
    )
    exp_dspark_fc = float(_DSPARK_TARGET_LAYERS * _HIDDEN_SIZE * _HIDDEN_SIZE)
    exp_dspark_markov = float(2 * _VOCAB_SIZE * _DSPARK_MARKOV_RANK)
    exp_dspark_total = (
        exp_dspark_attn + exp_dspark_mlp + exp_dspark_fc + exp_dspark_markov
    )
    assert (exp_dspark_fc * BF16_BYTES) / BIN_MEBI == pytest.approx(360.0)

    full_params = rf.param_counts(full_geom)
    assert full_params.mtp == pytest.approx(exp_dspark_total)
    assert full_params.total_params == pytest.approx(exp_trunk_total + exp_dspark_total)
    full_unit_hbm = rf.hbm_breakdown_per_core_bytes(
        full_geom, _UNIT_DTYPES, _UNIT_PARALLEL, rf.SeqKvLayout(0, 0), chip=chip
    )
    assert abs(full_params.total_params - full_unit_hbm.weights) <= _UNIT_BYTE_TOL

    fp8_dt = replace(dtypes, kv=FP8_BYTES, kv_dtype="fp8", kv_source="flag")
    int4_dt = replace(
        dtypes,
        kv=FP4_BYTES,
        kv_dtype=INT4_PER_TOKEN_HEAD_KV_DTYPE,
        kv_source="flag",
    )
    kv_cases = (
        (rf.Parallel(tp=8, kv_split=1), 384.0, 228.0),
        (rf.Parallel(tp=8, kv_split=2), 384.0, 228.0),
        (rf.Parallel(tp=4, kv_split=1), 384.0, 228.0),
        (rf.Parallel(tp=8, kv_split=4), 640.0, 356.0),
        (rf.Parallel(tp=8, kv_split=8), 1152.0, 612.0),
    )
    for par, exp_fp8_b, exp_int4_b in kv_cases:
        actual_fp8 = kv_bytes_per_token_per_layer(
            full_geom, fp8_dt, par, "gqa", physical=True, chip=chip
        )
        actual_int4 = kv_bytes_per_token_per_layer(
            full_geom, int4_dt, par, "gqa", physical=True, chip=chip
        )
        assert actual_fp8 == pytest.approx(exp_fp8_b)
        assert actual_int4 == pytest.approx(exp_int4_b)


def test_minimax_m31_flash_oracle_and_int4_kv() -> None:
    """Verify exact MiniMax-M3.1-Flash param, MSA FP8/int4 KV, and dspark oracles."""
    main_cfg, draft_cfg = _load_mm31_configs()
    chip = rf.CHIPS["tpu7x"]

    identity = rf.resolve_model_identity(main_cfg, model_ref=_HF_REPO)
    assert identity.canonical_id == "minimax_m3_vl"
    assert identity.hf_repo == _HF_REPO

    audit = audit_hf_config(main_cfg, model_ref=_HF_REPO)
    assert audit.onboarding_tier == "tier0_ready"
    assert audit.unmapped_structural_keys == ()
    assert audit.blocking_reasons == ()

    dtypes = rf.dtypes_from_hf_config(main_cfg)
    assert dtypes.w_dense == pytest.approx(MXFP8_BYTES)
    assert dtypes.w_dense_dtype == "mxfp8"
    assert dtypes.w_expert == pytest.approx(NVFP4_BYTES)
    assert dtypes.w_expert_dtype == "nvfp4"
    assert dtypes.w_aux == pytest.approx(BF16_BYTES)
    assert dtypes.lm_head == pytest.approx(BF16_BYTES)
    assert dtypes.act == "a8"

    trunk_geom = rf.from_hf_config(main_cfg, model_ref=_HF_REPO)
    exp_trunk_total = _assert_trunk_oracle(trunk_geom, chip)
    full_geom = rf.from_hf_config(main_cfg, model_ref=_HF_REPO, draft_config=draft_cfg)
    _assert_dspark_and_msa_kv_oracle(full_geom, dtypes, chip, exp_trunk_total)


def test_minimax_m31_dcp4_dcp8_dspark_offload_and_bounds() -> None:
    """Verify TP8xDCP2/4/8 DSpark spec cycles, offload KV skip, and bound <= est."""
    main_cfg, draft_cfg = _load_mm31_configs()
    chip = rf.CHIPS["tpu7x"]
    topology = rf.TOPOLOGIES["2x2x1"]
    geom = rf.from_hf_config(main_cfg, model_ref=_HF_REPO, draft_config=draft_cfg)
    base_dt = rf.dtypes_from_hf_config(main_cfg)
    int4_dt = replace(
        base_dt,
        kv=FP4_BYTES,
        kv_dtype=INT4_PER_TOKEN_HEAD_KV_DTYPE,
        kv_source="flag",
    )
    spec = rf.SpecShape(batch=16, context=65536, spec_k=_DSPARK_BLOCK_SIZE)
    opts = rf.FloorOptions(dtypes=int4_dt, topology=topology)

    # TP8 with DCP=2, 4, and 8 must all evaluate cleanly without kv_split failure.
    for kv_split in (1, 2, 4, 8):
        par = rf.Parallel(tp=8, ep=8, moe="ep", kv_split=kv_split)
        fl = rf.spec_cycle_floor(geom, chip, par, spec, opts)
        assert fl.hard_floor_s > 0.0
        assert fl.hard_floor_s <= fl.serial_estimate_s + 1e-12

    # Offload skips draft SWA KV when skip_draft_kv_on_offload is active.
    par_dcp4 = rf.Parallel(tp=8, ep=8, moe="ep", kv_split=4)
    bd_no_offload = rf.hbm_breakdown_per_core_bytes(
        geom,
        int4_dt,
        par_dcp4,
        rf.SeqKvLayout(8, 8192, kv_offload_enabled=False),
        chip=chip,
    )
    bd_offload = rf.hbm_breakdown_per_core_bytes(
        geom,
        int4_dt,
        par_dcp4,
        rf.SeqKvLayout(8, 8192, kv_offload_enabled=True),
        chip=chip,
    )
    assert bd_no_offload.kv > bd_offload.kv

    # Solver enumerates TP8xDCP4 and TP8xDCP8 layouts cleanly on 2x2x1.
    scenario = rf.TrafficScenario(
        name="mm31_int4_dcp",
        isl=65536,
        osl=512,
        prefix_hit_rate=0.5,
        accept_len=4.0,
        kv_working_set_turns=1,
        concurrencies=(16, 48),
        kv_offload_gb=440.0,
        spec_tokens=_DSPARK_BLOCK_SIZE,
        think_time_s=0.0,
    )
    plan = rf.solve_serving_plan(
        geom,
        int4_dt,
        chip,
        scenario,
        rf.SolveOptions(
            mode="mixed",
            max_chips=4,
            decode_topology="2x2x1",
        ),
    )
    dcp_values = {pt.decode_pool.kv_split for pt in plan.candidates if pt.decode_pool}
    assert {2, 4, 8}.issubset(dcp_values)


def test_minimax_m31_pp_slicing_and_cli_inspect(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify PP stage slicing conserves MSA layers/dspark and CLI inspect passes."""
    main_cfg, draft_cfg = _load_mm31_configs()
    geom = rf.from_hf_config(main_cfg, model_ref=_HF_REPO, draft_config=draft_cfg)
    stage0 = slice_geometry_layers(geom, 0, _HALF_LAYERS)
    stage1 = slice_geometry_layers(geom, _HALF_LAYERS, _NUM_LAYERS)
    assert stage0.draft is None
    assert stage1.draft == geom.draft
    assert stage0.ffn_layer_counts() == (_HALF_LAYERS - _DENSE_LAYERS, _DENSE_LAYERS)
    assert stage1.ffn_layer_counts() == (_HALF_LAYERS, 0)
    assert stage0.sparse_attn is not None
    assert stage1.sparse_attn is not None
    assert len(stage0.sparse_attn.kv_source_layer_ids) == _HALF_LAYERS
    assert len(stage1.sparse_attn.kv_source_layer_ids) == _HALF_LAYERS

    rc = cli.main(
        [
            "inspect",
            "--config",
            str(FIXTURES_DIR / _MAIN_FIXTURE),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["onboarding_tier"] == "tier0_ready"
    assert payload["canonical_id"] == "minimax_m3_vl"
    assert payload["unmapped_structural_keys"] == []


def test_non_minimax_dense_swa_draft_defaults_and_cli_kv_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify non-MiniMax dense+SWA DSpark defaults and CLI --kv-dtype override."""
    main_cfg, _ = _load_mm31_configs()
    generic_draft = {
        "architectures": ["GenericDenseSwaDraftModel"],
        "num_hidden_layers": 2,
        "dspark_use_dense_ffn": True,
        "dspark_ffn_hidden_size": 4096,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "sliding_window": 128,
    }
    generic_geom = rf.from_hf_config(main_cfg, draft_config=generic_draft)
    assert generic_geom.draft is not None
    assert generic_geom.draft.draft_kv_dtype == ""
    assert generic_geom.draft.skip_draft_kv_on_offload is False
    assert generic_geom.draft.single_row_lm_head_verify is False

    # Explicit --kv-dtype bf16 on MiniMax overrides MiniMax's default fp8 draft KV.
    def _explain_draft_swa_hbm(kv_dtype: str | None) -> float:
        """Return draft_swa HBM bytes from CLI explain with optional --kv-dtype."""
        argv = [
            "explain",
            "--config",
            str(FIXTURES_DIR / _MAIN_FIXTURE),
            "--draft-config",
            str(FIXTURES_DIR / _DRAFT_FIXTURE),
            "--tp",
            "2",
            "--batch",
            "4",
            "--context",
            "1024",
            "--spec-window",
            "7",
            "--json",
        ]
        if kv_dtype is not None:
            argv.extend(["--kv-dtype", kv_dtype])
        rc = cli.main(argv)
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        return float(next(r["hbm_bytes"] for r in rows if r["name"] == "draft_swa"))

    assert _explain_draft_swa_hbm("bf16") > _explain_draft_swa_hbm(None)


@pytest.mark.parametrize(
    "missing_key",
    [
        "sparse_block_size",
        "sparse_topk_blocks",
        "sparse_num_index_heads",
        "sparse_index_dim",
    ],
)
def test_msa_sparse_config_requires_positive_dimensions(missing_key: str) -> None:
    """Verify MSA sparse_attention_config fails closed on missing or zero dims."""
    main_cfg, _ = _load_mm31_configs()
    text_cfg = dict(main_cfg["text_config"])
    sparse_cfg = dict(text_cfg["sparse_attention_config"])
    sparse_cfg[missing_key] = 0
    text_cfg["sparse_attention_config"] = sparse_cfg
    bad_cfg = {**main_cfg, "text_config": text_cfg}
    with pytest.raises(rf.UnsupportedArchitectureError, match=missing_key):
        rf.from_hf_config(bad_cfg)


@pytest.mark.parametrize(
    ("mutator_kind", "match_pattern"),
    [
        ("short_moe_freq", "moe_layer_freq"),
        ("invalid_moe_flag", "moe_layer_freq"),
        ("short_sparse_freq", "sparse_attention_freq"),
    ],
)
def test_minimax_m31_fail_closed_layer_frequencies(
    mutator_kind: str,
    match_pattern: str,
) -> None:
    """Verify moe_layer_freq and sparse_attention_freq lengths/values fail closed."""
    bad_cfg = load_locked_config(_MAIN_FIXTURE)
    text_cfg = dict(bad_cfg["text_config"])
    if mutator_kind == "short_moe_freq":
        text_cfg["moe_layer_freq"] = [1] * (_NUM_LAYERS - 1)
    elif mutator_kind == "invalid_moe_flag":
        bad_freq = list(text_cfg["moe_layer_freq"])
        bad_freq[0] = 2
        text_cfg["moe_layer_freq"] = bad_freq
    else:
        sparse_cfg = dict(text_cfg["sparse_attention_config"])
        sparse_cfg["sparse_attention_freq"] = [1] * (_NUM_LAYERS - 1)
        text_cfg["sparse_attention_config"] = sparse_cfg
    bad_cfg["text_config"] = text_cfg
    with pytest.raises(ValueError, match=match_pattern):
        rf.from_hf_config(bad_cfg)
