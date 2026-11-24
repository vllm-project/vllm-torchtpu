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
"""Algebraic invariance and regression tests for sparse, HC, and 4-bit KV physics."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    roofline,
    specs,
    support,
    vmem,
)
from vllm_torchtpu.megakernel.core import errors

STAGE_A_BASELINE_ORACLE: dict[
    str, dict[str, tuple[float, float, float, float, int]]
] = {
    "qwen3.5-397b": {
        "per_op": (4767.47, 1275.2, 9147.0, 49.884, 3328),
        "single_layer": (4754.71, 1262.45, 7361.65, 49.884, 414),
        "layer_group": (4301.33, 798.84, 7165.15, 49.884, 397),
        "whole_step": (4301.33, 798.84, 7165.15, 49.884, 112),
    },
    "kimi-k3": {
        "per_op": (7183.08, 3474.26, 18351.4, 50.159, 1901),
        "single_layer": (7171.42, 3451.07, 14786.21, 50.159, 290),
        "layer_group": (7171.42, 2144.38, 14484.0, 50.159, 278),
        "whole_step": (7171.42, 2144.38, 11385.89, 50.159, 128),
    },
    "glm-5.2": {
        "per_op": (7420.18, 2537.66, 12328.06, 48.394, 2218),
        "single_layer": (7284.37, 2401.85, 9914.31, 48.394, 295),
        "layer_group": (7122.54, 2051.8, 9652.49, 48.394, 283),
        "whole_step": (7122.54, 2051.8, 9652.49, 48.394, 283),
    },
}


def _scope_map(rep: advisor.AdvisorReport) -> dict[str, advisor.ScopeEstimate]:
    """Index an AdvisorReport's scope estimates by scope name."""
    return {sc.scope: sc for sc in rep.scopes}


def test_sparse_attn_owner_kv_conservation_and_swa_cap_monotonicity() -> None:
    """Verify owner-layer KV conservation, DSA isolation, and window/topk saturation."""
    cand_source_layer = 20
    kv_r2 = tuple(
        specs.SparseCacheOwner(
            layer=lyr,
            record_tokens=2.0,
            record_elems=512,
            elem_bytes=1.0,
            record_overhead_bytes=4.0,
            replication="per_rank",
            users=tuple(range(lyr, lyr + 6)),
        )
        for lyr in (2, 8, 14)
    )
    kv_r1 = (
        specs.SparseCacheOwner(
            layer=cand_source_layer,
            record_tokens=1.0,
            record_elems=512,
            elem_bytes=1.0,
            record_overhead_bytes=4.0,
            replication="per_rank",
            users=tuple(range(cand_source_layer, 40)),
        ),
    )
    idx_r2 = tuple(
        specs.SparseCacheOwner(
            layer=lyr,
            record_tokens=2.0,
            record_elems=128,
            elem_bytes=1.0,
            record_overhead_bytes=4.0,
            replication="per_rank",
            users=tuple(range(lyr, lyr + 6)),
        )
        for lyr in (2, 8, 14)
    )
    idx_r1 = tuple(
        specs.SparseCacheOwner(
            layer=lyr,
            record_tokens=1.0,
            record_elems=128,
            elem_bytes=1.0,
            record_overhead_bytes=4.0,
            replication="per_rank",
            users=tuple(range(lyr, min(40, lyr + 4))),
            max_scan_records=16384 if lyr > cand_source_layer else None,
        )
        for lyr in (20, 24, 28, 32, 36)
    )
    csa_spec = specs.SparseAttnSpec(
        kind="csa",
        window_tokens=128,
        window_bytes_per_token=1024.0,
        window_layers=tuple(range(40)),
        select_tokens=512,
        index_heads=32,
        index_kv_heads=1,
        index_dim=128,
        kv_owners=(*kv_r2, *kv_r1),
        index_owners=(*idx_r2, *idx_r1),
        shared_kv_mqa=True,
        kv_owner_ring_bytes_per_req=12288,
    )
    csa_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="synthetic-csa",
            num_layers=40,
            hidden_size=5120,
            num_attention_heads=64,
            num_kv_heads=1,
            head_dim=512,
            q_lora_rank=1280,
            o_lora_rank=1024,
            o_groups=8,
            index_topk=512,
            index_n_heads=32,
            index_head_dim=128,
            sparse_attn=csa_spec,
        )
    )
    assert not models.has_dsa_profile(csa_prof)
    assert "csa" in csa_prof.layer_families
    assert "dsa" not in csa_prof.layer_families
    assert csa_prof.index_topk == 0

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    ctx_lo = 1000
    ctx_hi = 11000
    expected_linear_b_per_tok = 2148.0
    delta_tokens = ctx_hi - ctx_lo
    ctx_obj_lo = roofline.build_scope_context(
        csa_prof,
        specs.WorkloadSpec(phase="decode", batch=8, tp=1, dp=8, context=ctx_lo),
        tpu7x,
    )
    ctx_obj_hi = roofline.build_scope_context(
        csa_prof,
        specs.WorkloadSpec(phase="decode", batch=8, tp=1, dp=8, context=ctx_hi),
        tpu7x,
    )
    slope = (ctx_obj_hi.hbm_used - ctx_obj_lo.hbm_used) / delta_tokens
    assert slope == pytest.approx(expected_linear_b_per_tok)
    unshared_forty_layer_b_per_tok = 40.0 * (516.0 + 132.0)
    assert slope < unshared_forty_layer_b_per_tok

    contexts = (128, 2048, 32768, 262144)
    swa_only_prof = models.finalize_model_profile(
        replace(
            csa_prof,
            sparse_attn=replace(
                csa_spec, select_tokens=0, kv_owners=(), index_owners=()
            ),
        )
    )
    qsa_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="synthetic-qsa",
            num_layers=12,
            hidden_size=2560,
            num_attention_heads=24,
            num_kv_heads=2,
            head_dim=256,
            sparse_attn=specs.SparseAttnSpec(
                kind="qsa",
                window_tokens=0,
                select_tokens=2051,
                index_heads=4,
                index_kv_heads=1,
                index_dim=128,
                index_compress_ratio=4,
            ),
        )
    )
    msa_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="synthetic-msa",
            num_layers=12,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=4,
            head_dim=128,
            dcp_span="tp_group",
            sparse_attn=specs.SparseAttnSpec(
                kind="msa",
                window_tokens=128,
                select_tokens=2048,
                index_heads=4,
                index_kv_heads=1,
                index_dim=128,
            ),
        )
    )
    for prof, sat_idx in ((swa_only_prof, 0), (qsa_prof, 2), (msa_prof, 2)):
        flops_seq = [
            roofline.build_scope_context(
                prof,
                specs.WorkloadSpec(phase="decode", batch=8, tp=8, context=ctx),
                tpu7x,
            ).flops_kv
            for ctx in contexts
        ]
        for prev_f, next_f in pairwise(flops_seq):
            assert next_f >= prev_f
        assert flops_seq[sat_idx] == pytest.approx(flops_seq[-1])
        if sat_idx > 0:
            assert flops_seq[0] < flops_seq[sat_idx]


def test_wide_dcp_kv_head_replication_and_pcp_divisibility() -> None:
    """Verify Wide-DCP KV-head group counts, pure-KV conservation, and PCP*TP rules."""
    expected_groups = {1: 1, 2: 1, 4: 2, 8: 4}
    for dcp_deg, exp_g in expected_groups.items():
        assert vmem.wide_dcp_kv_heads_per_rank(4, 8, dcp_deg, "tp_group") == exp_g
    expected_tp2_groups = 2
    assert vmem.wide_dcp_kv_heads_per_rank(4, 2, 1, "tp_group") == expected_tp2_groups
    assert vmem.wide_dcp_kv_heads_per_rank(4, 4, 1, "tp_group") == 1

    moe_schedule = [0, 0, 0, *([1] * 57)]
    mm_cfg = {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "moe_layer_freq": moe_schedule,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_local_block": 1,
            },
        },
        "quantization_config": {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "group_size": 16,
            "moe_quant_algo": "NVFP4",
        },
    }
    mm_int4 = models.finalize_model_profile(
        replace(
            models.parse_hf_config_profile(mm_cfg, name="mm31-test"),
            kv_dtype="int4_per_token_head",
        )
    )
    assert mm_int4.dcp_span == "tp_group"

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    stored_bytes_by_dcp: dict[int, float] = {}
    ctx_tokens = 4096
    ctx_zero = 1024
    num_layers = 60
    tp_deg = 8
    min_replicated_dcp = 2
    conserved_tp_pure_kv = 512.0
    const_idx_scale_per_stored_tok = 100.0
    for dcp_deg in (1, 2, 4, 8):
        wl_hi = specs.WorkloadSpec(
            phase="decode", batch=1, tp=tp_deg, ep=8, dcp=dcp_deg, context=ctx_tokens
        )
        wl_lo = replace(wl_hi, context=ctx_zero)
        kv_delta = (
            roofline.build_scope_context(mm_int4, wl_hi, tpu7x).hbm_used
            - roofline.build_scope_context(mm_int4, wl_lo, tpu7x).hbm_used
        )
        per_ctx_tok_layer_rank = kv_delta / (num_layers * (ctx_tokens - ctx_zero))
        per_stored_tok_layer_rank = per_ctx_tok_layer_rank * dcp_deg
        stored_bytes_by_dcp[dcp_deg] = per_stored_tok_layer_rank
        idx_scale_rank = const_idx_scale_per_stored_tok / dcp_deg
        pure_kv_rank = per_ctx_tok_layer_rank - idx_scale_rank
        if dcp_deg >= min_replicated_dcp:
            assert tp_deg * pure_kv_rank == pytest.approx(conserved_tp_pure_kv)

    expected_stored = {1: 228.0, 2: 228.0, 4: 356.0, 8: 612.0}
    for dcp_deg, exp_b in expected_stored.items():
        assert stored_bytes_by_dcp[dcp_deg] == pytest.approx(exp_b)
    assert stored_bytes_by_dcp[2] < stored_bytes_by_dcp[4] < stored_bytes_by_dcp[8]

    glm_prof = models.resolve_model_profile("glm-5.2")
    wide_pcp_wl = specs.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=8, ep=8, pcp=2, dcp=16
    )
    assert "dcp_not_dividing_tp" not in advisor.validate_workload(
        glm_prof, "tpu7x", wide_pcp_wl
    )
    rep_glm = advisor.advise(glm_prof, wide_pcp_wl, "tpu7x")
    assert rep_glm.scopes[0].feasible
    wide_pcp_ep16_wl = replace(wide_pcp_wl, ep=16)
    assert advisor.validate_workload(glm_prof, "tpu7x", wide_pcp_ep16_wl) == ()

    bad_pcp1_wl = specs.WorkloadSpec(phase="decode", batch=8, tp=8, ep=8, pcp=1, dcp=16)
    assert "dcp_not_dividing_tp" in advisor.validate_workload(
        glm_prof, "tpu7x", bad_pcp1_wl
    )
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(glm_prof, bad_pcp1_wl)

    bad_tp_group_wl = specs.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=8, ep=8, pcp=2, dcp=16
    )
    assert "dcp_not_dividing_tp" in advisor.validate_workload(
        mm_int4, "tpu7x", bad_tp_group_wl
    )
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(mm_int4, bad_tp_group_wl)


def test_hadamard_int4_and_turboquant_kv_bytes_and_vpu_scaling() -> None:
    """Verify int4/fp8/bf16 MSA cache bytes, TQLayout widths, and split Hadamard ops."""
    expected_tq = {
        (128, 64): 144.0,
        (128, 16): 192.0,
        (256, 64): 272.0,
        (256, 16): 320.0,
    }
    for (hdim, page_sz), exp_slot_b in expected_tq.items():
        assert roofline.turboquant_slot_bytes(hdim, page_sz) == pytest.approx(
            exp_slot_b
        )

    msa_owners_kv = tuple(
        specs.SparseCacheOwner(
            layer=lyr,
            record_tokens=1.0,
            record_elems=256,
            elem_bytes=None,
            record_overhead_bytes=0.0,
            replication="kv_head_group",
            users=(lyr,),
        )
        for lyr in range(60)
    )
    msa_owners_idx = tuple(
        specs.SparseCacheOwner(
            layer=lyr,
            record_tokens=1.0,
            record_elems=128,
            elem_bytes=None,
            record_overhead_bytes=0.0,
            replication="per_rank",
            users=(lyr,),
        )
        for lyr in range(60)
    )
    base_msa = models.finalize_model_profile(
        models.ModelProfile(
            name="msa-cache-check",
            num_layers=60,
            hidden_size=6144,
            num_attention_heads=64,
            num_kv_heads=4,
            head_dim=128,
            dcp_span="tp_group",
            sparse_attn=specs.SparseAttnSpec(
                kind="msa",
                window_tokens=128,
                select_tokens=2048,
                index_heads=4,
                index_kv_heads=1,
                index_dim=128,
                kv_owners=msa_owners_kv,
                index_owners=msa_owners_idx,
            ),
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    expected_table = {
        "int4_per_token_head": (228.0, 356.0, 612.0),
        "fp8": (384.0, 640.0, 1152.0),
        "bf16": (768.0, 1280.0, 2304.0),
    }
    for kv_dt, exp_triple in expected_table.items():
        prof_dt = models.finalize_model_profile(replace(base_msa, kv_dtype=kv_dt))
        got_list: list[float] = []
        for dcp_deg in (2, 4, 8):
            hbm_2k = roofline.build_scope_context(
                prof_dt,
                specs.WorkloadSpec(
                    phase="decode", batch=1, tp=8, dcp=dcp_deg, context=2048
                ),
                tpu7x,
            ).hbm_used
            hbm_1k = roofline.build_scope_context(
                prof_dt,
                specs.WorkloadSpec(
                    phase="decode", batch=1, tp=8, dcp=dcp_deg, context=1024
                ),
                tpu7x,
            ).hbm_used
            got_list.append((hbm_2k - hbm_1k) * dcp_deg / (60 * 1024))
        assert tuple(got_list) == pytest.approx(exp_triple)

    gqa_base = models.finalize_model_profile(
        models.ModelProfile(
            name="gqa-hadamard",
            num_layers=8,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp8",
        )
    )
    ops_fp8 = roofline.compute_hadamard_kv_vec_ops("fp8", 4, 1, 16, 8)
    ops_int4 = roofline.compute_hadamard_kv_vec_ops("int4_per_token_head", 4, 1, 16, 8)
    ops_tq = roofline.compute_hadamard_kv_vec_ops("turboquant_4bit_nc", 4, 1, 16, 8)
    expected_tq_ops = float(constants.HADAMARD_128_VEC_OPS_PER_HEAD * (4 + 1) * 16 * 8)
    expected_int4_ops = 2.0 * expected_tq_ops
    assert ops_fp8 == 0.0
    assert ops_tq == pytest.approx(expected_tq_ops)
    assert ops_int4 == pytest.approx(expected_int4_ops)

    # Verify TurboQuant slot holds both K and V (single slot per token per head, < FP8).
    gqa_tq = models.finalize_model_profile(
        replace(gqa_base, kv_dtype="turboquant_4bit_nc")
    )
    tq_wl_1k = specs.WorkloadSpec(phase="decode", batch=1, tp=8, dcp=1, context=1024)
    tq_wl_2k = specs.WorkloadSpec(phase="decode", batch=1, tp=8, dcp=1, context=2048)
    tq_delta = (
        roofline.build_scope_context(gqa_tq, tq_wl_2k, tpu7x).hbm_used
        - roofline.build_scope_context(gqa_tq, tq_wl_1k, tpu7x).hbm_used
    )
    fp8_delta = (
        roofline.build_scope_context(gqa_base, tq_wl_2k, tpu7x).hbm_used
        - roofline.build_scope_context(gqa_base, tq_wl_1k, tpu7x).hbm_used
    )
    expected_tq_per_tok_layer = roofline.turboquant_slot_bytes(128, 64)
    assert tq_delta / (8 * 1024) == pytest.approx(expected_tq_per_tok_layer)
    assert tq_delta < fp8_delta

    # Verify sparse attention with DCP > 1 pays cross-DCP collectives (> dcp=1).
    msa_dcp1 = advisor.advise(
        base_msa,
        specs.WorkloadSpec(phase="decode", batch=8, tp=8, dcp=1, context=4096),
        tpu7x,
    )
    msa_dcp4 = advisor.advise(
        base_msa,
        specs.WorkloadSpec(phase="decode", batch=8, tp=8, dcp=4, context=4096),
        tpu7x,
    )
    po_dcp1 = next(s for s in msa_dcp1.scopes if s.scope == "per_op")
    po_dcp4 = next(s for s in msa_dcp4.scopes if s.scope == "per_op")
    assert po_dcp4.comm_us > 0.0
    assert po_dcp4.comm_us > po_dcp1.comm_us

    wl = specs.WorkloadSpec(phase="decode", batch=16, tp=8, context=2048)
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(
            replace(gqa_base, weight_dtype="int4_per_token_head"), wl
        )


def test_hyper_connection_lowrank_and_mhc_sinkhorn_scaling() -> None:
    """Verify exact low-rank HC and mHC parameter counts and Sinkhorn scaling."""
    q38_hc_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="q38-hc",
            num_layers=48,
            hidden_size=2560,
            num_attention_heads=24,
            num_kv_heads=2,
            head_dim=256,
            weight_dtype="bf16",
            hyper_connection=specs.HyperConnectionSpec(
                kind="lowrank", num_streams=4, lowrank_dim=320, has_final_mixer=True
            ),
        )
    )
    expected_q38_hc_params = 639_631_360
    expected_streams = 4
    assert q38_hc_prof.residual_streams == expected_streams
    assert "hyper_connection" in q38_hc_prof.layer_families
    assert models.hyper_connection_total_params(q38_hc_prof) == expected_q38_hc_params

    dsv_mhc_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="dsv-mhc",
            num_layers=40,
            hidden_size=5120,
            num_attention_heads=64,
            num_kv_heads=1,
            head_dim=512,
            hyper_connection=specs.HyperConnectionSpec(
                kind="mhc", num_streams=4, sinkhorn_iters=20
            ),
        )
    )
    expected_mhc_layer_params = 983_040
    expected_mhc_total_params = 39_321_600
    assert "mhc" in dsv_mhc_prof.layer_families
    assert (
        models.hyper_connection_total_params(dsv_mhc_prof) // 40
        == expected_mhc_layer_params
    )
    assert (
        models.hyper_connection_total_params(dsv_mhc_prof) == expected_mhc_total_params
    )

    wl = specs.WorkloadSpec(phase="decode", batch=32, tp=8, context=1024)
    dsv_mhc_it5 = models.finalize_model_profile(
        replace(
            dsv_mhc_prof,
            hyper_connection=specs.HyperConnectionSpec(
                kind="mhc", num_streams=4, sinkhorn_iters=5
            ),
        )
    )
    dsv_mhc_s2 = models.finalize_model_profile(
        replace(
            dsv_mhc_prof,
            hyper_connection=specs.HyperConnectionSpec(
                kind="mhc", num_streams=2, sinkhorn_iters=20
            ),
        )
    )
    _, _, vec_20 = roofline.compute_hyper_connection_terms(dsv_mhc_prof, 40, 8, 32)
    _, _, vec_5 = roofline.compute_hyper_connection_terms(dsv_mhc_it5, 40, 8, 32)
    _, _, vec_s2 = roofline.compute_hyper_connection_terms(dsv_mhc_s2, 40, 8, 32)
    assert vec_20 > vec_5
    assert vec_20 > vec_s2
    assert (
        vmem.compute_per_token_vmem_bytes(dsv_mhc_prof, wl)["residual_buffer_bytes"]
        > vmem.compute_per_token_vmem_bytes(dsv_mhc_s2, wl)["residual_buffer_bytes"]
    )


def test_fusion_break_blocks_whole_step_and_partitions_layer_groups() -> None:
    """Verify host-table and CED fusion breaks, overrides, and plan_error priority."""
    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="dp_local",
    )
    eng_l14 = specs.HostTableBreakSpec(
        name="engram_l14",
        layers=(14,),
        table_bytes_total=101_380_404_048,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="dp_local",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="break-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            ced_boundary=20,
            sparse_attn=specs.SparseAttnSpec(
                kind="csa", window_tokens=128, select_tokens=512
            ),
            host_tables=(eng_l1, eng_l14),
        )
    )
    wl_dec = specs.WorkloadSpec(phase="decode", batch=8, tp=1, dp=8, context=1024)
    assert hash(wl_dec) == hash(
        specs.WorkloadSpec(phase="decode", batch=8, tp=1, dp=8, context=1024)
    )
    breaks = vmem.active_fusion_break_layers(prof, wl_dec)
    assert breaks == (1, 14)
    assert vmem.fusion_break_layer_groups(40, breaks) == (1, 13, 26)

    rep_dec = advisor.advise(prof, wl_dec, "tpu7x")
    sc_dec = _scope_map(rep_dec)
    assert not sc_dec["whole_step"].feasible
    assert sc_dec["whole_step"].reason == "Fusion break at host-table or CED boundary"
    assert "fusion_break" in sc_dec["whole_step"].blockers
    assert sc_dec["layer_group"].feasible
    expected_three_groups = 3
    assert sc_dec["layer_group"].ops_per_step == expected_three_groups

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    err_ctx = roofline.build_scope_context(
        prof, wl_dec, tpu7x, plan_error="custom_planner_failure"
    )
    ws_res = vmem.compute_scope_vmem_and_ops("whole_step", err_ctx, prof, wl_dec)
    assert not ws_res[5]
    assert ws_res[6] == "custom_planner_failure"
    sup_status, sup_blks = support.evaluate_support(
        "whole_step", prof, wl_dec, tpu7x, err_ctx
    )
    assert sup_status == "hypothetical"
    assert "fusion_break" in sup_blks

    wl_hbm_all = replace(
        wl_dec,
        host_table_hbm_overrides=(("engram_l1", True), ("engram_l14", True)),
    )
    assert vmem.active_fusion_break_layers(prof, wl_hbm_all) == ()
    rep_hbm = advisor.advise(prof, wl_hbm_all, "tpu7x")
    expected_added_gb = (eng_l1.table_bytes_total + eng_l14.table_bytes_total) / 8 / 1e9
    assert rep_hbm.hbm_used_gb - rep_dec.hbm_used_gb == pytest.approx(
        expected_added_gb, abs=1e-2
    )
    assert _scope_map(rep_hbm)["whole_step"].feasible
    assert "fusion_break" not in _scope_map(rep_hbm)["whole_step"].blockers

    wl_pref_full = specs.WorkloadSpec(
        phase="prefill",
        execution="full",
        batch=8,
        isl=4096,
        tp=1,
        dp=8,
        host_table_hbm_overrides=(("engram_l1", True), ("engram_l14", True)),
    )
    wl_pref_ced = replace(wl_pref_full, execution="ced")
    ced_breaks = vmem.active_fusion_break_layers(prof, wl_pref_ced)
    assert ced_breaks == (20,)
    assert vmem.fusion_break_layer_groups(40, ced_breaks) == (20, 20)
    rep_full = advisor.advise(prof, wl_pref_full, "tpu7x")
    rep_ced = advisor.advise(prof, wl_pref_ced, "tpu7x")
    assert not _scope_map(rep_ced)["whole_step"].feasible
    assert (
        _scope_map(rep_ced)["per_op"].compute_us
        < _scope_map(rep_full)["per_op"].compute_us
    )


def test_grouped_o_lora_moe_layer_freq_mamba_ssm_dtype_and_mixed_quant() -> None:
    """Verify o_lora/o_groups, moe_layer_freq, mamba_ssm_dtype, and mixed quant."""
    dsv_ratios = [0, 0, *([2] * 18), *([1] * 20), 0, 0, 0]
    dsv_cfg = {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "sliding_window": 128,
            "compress_ratios": dsv_ratios,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
        },
    }
    dsv_prof = models.parse_hf_config_profile(dsv_cfg, name="dsv41")
    expected_dsv_tp1_attn = 126_615_552.0
    expected_dsv_tp16_wo_a = 2_097_152.0
    expected_block_k_32 = 32
    assert dsv_prof.expert_dtype == "mxfp4"
    assert dsv_prof.weight_block_k == expected_block_k_32
    assert roofline.compute_attention_params(dsv_prof, 1, 1, 1)[3] == pytest.approx(
        expected_dsv_tp1_attn
    )
    tp16_with_olora = roofline.compute_attention_params(dsv_prof, 16, 1, 1)[3]
    tp16_zero_wo_a = (
        1280 * 5120 + 4 * 512 * 1280 + 1 * 512 * 5120 + 5120 * ((8 * 1024) // 16)
    )
    assert tp16_with_olora - tp16_zero_wo_a == pytest.approx(expected_dsv_tp16_wo_a)

    mm_cfg = {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "moe_layer_freq": [0, 0, 0, *([1] * 57)],
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_local_block": 1,
            },
        },
        "quantization_config": {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "group_size": 16,
            "moe_quant_algo": "NVFP4",
        },
    }
    mm_prof = models.parse_hf_config_profile(mm_cfg, name="mm31")
    expected_dense_inter = 12288
    expected_moe_inter = 3072
    expected_moe_layers = 57
    assert mm_prof.intermediate_size == expected_dense_inter
    assert mm_prof.moe_intermediate_size == expected_moe_inter
    assert mm_prof.shared_expert_intermediate_size == expected_moe_inter
    assert mm_prof.num_moe_layers == expected_moe_layers
    assert mm_prof.expert_dtype == "nvfp4"
    assert mm_prof.weight_dtype == "fp8"
    assert mm_prof.weight_block_k == 0

    q38_skip = [
        f"model.language_model.layers.{idx}.self_attn.q_proj" for idx in range(3, 48, 4)
    ]
    q38_layer_types = [
        tok
        for _ in range(12)
        for tok in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
    ]
    q38_cfg = {
        "text_config": {
            "model_type": "qwen4_exp_text",
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": q38_layer_types,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "output_gate_type": "sigmoid",
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "heads_per_ngram": 8,
            "ngram_size": 3,
            "split_ngram_parts": 128,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": q38_skip,
        },
    }
    q38_prof = models.parse_hf_config_profile(q38_cfg, name="q38")
    assert q38_prof.state_dtype == "fp32"
    assert q38_prof.weight_dtype == "bf16"
    assert q38_prof.act_dtype == "bf16"
    assert q38_prof.expert_dtype == "fp8"
    assert q38_prof.expert_act_dtype == "fp8"
    assert q38_prof.has_attn_output_gate
    assert len(q38_prof.host_tables) == 1
    assert q38_prof.host_tables[0].layers == (1,)
    assert q38_prof.host_tables[0].hbm_resident

    alias_prof = models.finalize_model_profile(replace(q38_prof, kv_dtype="bfloat16"))
    assert alias_prof.kv_dtype == "bf16"
    auto_prof = models.finalize_model_profile(replace(q38_prof, kv_dtype="auto"))
    assert auto_prof.kv_dtype == "bf16"


def test_stage_a_three_models_zero_price_drift() -> None:
    """Assert 0.00 us and 0.00 GB drift across all 3 shipped Stage A presets."""
    for slug, scope_oracle in STAGE_A_BASELINE_ORACLE.items():
        dep = models.DEPLOYMENT_PRESETS[slug]
        dec = dep.decode
        wl = specs.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=dec.tp,
            ep=dec.ep,
            dp=dec.dp,
            pp=dec.pp,
            dcp=dec.dcp,
            pcp=dec.pcp,
            spec_tokens=dec.spec_tokens,
            layer_groups=dec.layer_groups,
        )
        rep = advisor.advise(slug, wl, "tpu7x")
        for sc in rep.scopes:
            exp_bound, exp_strict, exp_ach, exp_hbm, exp_tmax = scope_oracle[sc.scope]
            assert sc.step_bound_us == exp_bound, f"{slug} {sc.scope} step_bound_us"
            assert sc.step_floor_strict_us == exp_strict, f"{slug} {sc.scope} strict"
            assert sc.step_achievable_us == exp_ach, f"{slug} {sc.scope} achievable"
            assert rep.hbm_used_gb == exp_hbm, f"{slug} {sc.scope} hbm_used_gb"
            assert sc.t_vmem_max == exp_tmax, f"{slug} {sc.scope} t_vmem_max"
