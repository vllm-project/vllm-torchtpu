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
"""Algebraic invariance and physics tests for sparse, HC, and 4-bit KV primitives."""

from __future__ import annotations

import dataclasses
import http.client
import json
import threading
from dataclasses import replace
from itertools import pairwise

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    cli,
    models,
    report,
    roofline,
    serve,
    specs,
    vmem,
    workbench,
)
from vllm_torchtpu.megakernel.core import errors

STAGE_A_BASELINE_ORACLE: dict[
    str, dict[str, tuple[float, float, float, float, int, int]]
] = {
    "qwen3.5-397b": {
        "per_op": (4738.01, 1275.24, 8100.37, 49.884, 3328, 1),
        "single_layer": (4725.26, 1262.49, 7322.38, 49.884, 414, 1),
        "layer_group": (4271.88, 798.88, 6582.73, 49.884, 414, 1),
        "whole_step": (4271.88, 798.88, 6582.73, 49.884, 414, 1),
    },
    "kimi-k3": {
        "per_op": (8233.12, 5058.56, 15138.48, 55.136, 1901, 1),
        "single_layer": (8198.53, 5023.97, 13221.39, 55.136, 290, 1),
        "layer_group": (5809.1, 3185.62, 9752.31, 55.136, 290, 1),
        "whole_step": (5809.1, 3185.62, 9752.31, 55.136, 290, 1),
    },
    "glm-5.2": {
        "per_op": (7420.18, 2537.66, 11038.63, 48.394, 2218, 1),
        "single_layer": (7284.37, 2401.85, 9914.31, 48.394, 295, 1),
        "layer_group": (7122.54, 2051.8, 9550.77, 48.394, 295, 1),
        "whole_step": (7122.54, 2051.8, 9550.77, 48.394, 295, 1),
    },
}

UNBOUNDED_RATE = 1e30
BYTES_PER_GB = 1e9
ROUND_HALF_UNIT_US = 0.005
HTTP_BAD_REQUEST = 400
HTTP_OK = 200


def _scope_map(
    estimates: tuple[advisor.ScopeEstimate, ...],
) -> dict[str, advisor.ScopeEstimate]:
    """Index ScopeEstimate records by scope name."""
    return {est.scope: est for est in estimates}


def test_turboquant_4bit_slot_size_no_double_count() -> None:
    """Verify packed 4-bit TurboQuant slot holds K+V+meta without 2x double count."""
    expected_tq = {
        (128, 64): 144.0,
        (128, 16): 192.0,
        (256, 64): 272.0,
        (256, 16): 320.0,
    }
    for (head_dim, page_size), expected_slot_b in expected_tq.items():
        assert roofline.turboquant_slot_bytes(head_dim, page_size) == pytest.approx(
            expected_slot_b
        )

    num_layers = 8
    batch_size = 16
    ctx_lo = 1024
    ctx_hi = 17408
    delta_slots = num_layers * batch_size * (ctx_hi - ctx_lo)
    gqa_fp8 = models.finalize_model_profile(
        models.ModelProfile(
            name="gqa-fp8",
            num_layers=num_layers,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp8",
        )
    )
    gqa_tq = models.finalize_model_profile(
        replace(gqa_fp8, name="gqa-tq", kv_dtype="turboquant_4bit_nc")
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    wl_lo = specs.WorkloadSpec(phase="decode", batch=batch_size, tp=8, context=ctx_lo)
    wl_hi = replace(wl_lo, context=ctx_hi)

    tq_delta_gb = (
        advisor.advise(gqa_tq, wl_hi, tpu7x).hbm_used_gb
        - advisor.advise(gqa_tq, wl_lo, tpu7x).hbm_used_gb
    )
    fp8_delta_gb = (
        advisor.advise(gqa_fp8, wl_hi, tpu7x).hbm_used_gb
        - advisor.advise(gqa_fp8, wl_lo, tpu7x).hbm_used_gb
    )
    per_slot_tq_bytes = (tq_delta_gb * BYTES_PER_GB) / delta_slots
    expected_slot_bytes = 144.0
    expected_fp8_slot_bytes = 256.0
    assert per_slot_tq_bytes == pytest.approx(expected_slot_bytes, abs=0.5)
    assert per_slot_tq_bytes < expected_fp8_slot_bytes
    assert tq_delta_gb < fp8_delta_gb


def test_expert_weight_block_k_overrides_zero_dense_block_k() -> None:
    """Verify nonzero expert_weight_block_k activates MoE subchannel VPU ops."""
    expert_block_k = 16
    prof_k0 = models.finalize_model_profile(
        models.ModelProfile(
            name="moe-subchannel-k0",
            num_layers=4,
            num_moe_layers=4,
            hidden_size=2048,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
            intermediate_size=0,
            moe_intermediate_size=1024,
            num_experts=16,
            top_k=2,
            weight_dtype="fp8",
            expert_dtype="nvfp4",
            weight_block_k=0,
            expert_weight_block_k=0,
        )
    )
    prof_k16 = models.finalize_model_profile(
        replace(
            prof_k0, name="moe-subchannel-k16", expert_weight_block_k=expert_block_k
        )
    )
    assert prof_k16.weight_block_k == 0
    assert prof_k16.expert_weight_block_k == expert_block_k

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    vec_only_chip = replace(
        tpu7x,
        bf16_flops_per_s=UNBOUNDED_RATE,
        fp8_flops_per_s=UNBOUNDED_RATE,
        int8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
    )
    wl = specs.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=8,
        ep=8,
        context=1024,
        dispatch_regime="zero_floor",
    )
    per_op_k0 = _scope_map(advisor.evaluate_scopes(prof_k0, wl, vec_only_chip))[
        "per_op"
    ]
    per_op_k16 = _scope_map(advisor.evaluate_scopes(prof_k16, wl, vec_only_chip))[
        "per_op"
    ]
    assert per_op_k16.compute_us > per_op_k0.compute_us


def test_sparse_attn_owner_kv_conservation_and_swa_cap_monotonicity() -> None:
    """Verify owner-layer KV conservation, DSA isolation, and window/topk saturation."""
    cand_source_layer = 20
    num_layers = 40
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
            users=tuple(range(cand_source_layer, num_layers)),
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
            users=tuple(range(lyr, min(num_layers, lyr + 4))),
            max_scan_records=16384 if lyr > cand_source_layer else None,
        )
        for lyr in (20, 24, 28, 32, 36)
    )
    csa_spec = specs.SparseAttnSpec(
        kind="csa",
        window_tokens=128,
        window_bytes_per_token=1024.0,
        window_layers=tuple(range(num_layers)),
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
            num_layers=num_layers,
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
    ctx_hi = 101000
    batch_size = 8
    expected_linear_b_per_tok = 2148.0
    delta_tokens = batch_size * (ctx_hi - ctx_lo)
    rep_lo = advisor.advise(
        csa_prof,
        specs.WorkloadSpec(
            phase="decode", batch=batch_size, tp=1, dp=8, context=ctx_lo
        ),
        tpu7x,
    )
    rep_hi = advisor.advise(
        csa_prof,
        specs.WorkloadSpec(
            phase="decode", batch=batch_size, tp=1, dp=8, context=ctx_hi
        ),
        tpu7x,
    )
    # dp=8 splits batch_size=8 -> batch_local=1 per rank.
    per_rank_delta_tokens = delta_tokens // 8
    slope = ((rep_hi.hbm_used_gb - rep_lo.hbm_used_gb) * BYTES_PER_GB) / (
        per_rank_delta_tokens
    )
    assert slope == pytest.approx(expected_linear_b_per_tok, abs=2.0)
    unshared_forty_layer_b_per_tok = float(num_layers * (516 + 132))
    assert slope < unshared_forty_layer_b_per_tok

    fast_io = replace(
        tpu7x, hbm_bw_bytes_per_s=UNBOUNDED_RATE, vector_f32_ops_per_s=UNBOUNDED_RATE
    )
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
    for prof, sat_cap in ((swa_only_prof, 128), (qsa_prof, 2051), (msa_prof, 2176)):
        compute_seq = [
            _scope_map(
                advisor.evaluate_scopes(
                    prof,
                    specs.WorkloadSpec(
                        phase="decode",
                        batch=8,
                        tp=8,
                        context=ctx,
                        dispatch_regime="zero_floor",
                    ),
                    fast_io,
                )
            )["per_op"].compute_us
            for ctx in contexts
        ]
        for prev_us, next_us in pairwise(compute_seq):
            assert next_us >= prev_us
        if prof is swa_only_prof:
            assert compute_seq[0] == pytest.approx(compute_seq[-1])
        else:
            c_sat = _scope_map(
                advisor.evaluate_scopes(
                    prof,
                    specs.WorkloadSpec(
                        phase="decode",
                        batch=8,
                        tp=8,
                        context=sat_cap,
                        dispatch_regime="zero_floor",
                    ),
                    fast_io,
                )
            )["per_op"].compute_us
            slope_mid_hi = (compute_seq[2] - c_sat) / (contexts[2] - sat_cap)
            slope_hi_max = (compute_seq[3] - compute_seq[2]) / (
                contexts[3] - contexts[2]
            )
            assert c_sat > compute_seq[0]
            slope_tol = (2.0 * ROUND_HALF_UNIT_US) / (contexts[2] - sat_cap) + (
                2.0 * ROUND_HALF_UNIT_US
            ) / (contexts[3] - contexts[2])
            assert slope_mid_hi == pytest.approx(slope_hi_max, abs=slope_tol)


def test_wide_dcp_kv_head_replication_and_pcp_divisibility() -> None:
    """Verify Wide-DCP KV-head group counts, pure-KV conservation, and PCP*TP rules."""
    expected_groups = {1: 1, 2: 1, 4: 2, 8: 4}
    for dcp_deg, exp_g in expected_groups.items():
        assert vmem.wide_dcp_kv_heads_per_rank(4, 8, dcp_deg, "tp_group") == exp_g
    expected_tp2_groups = 2
    assert vmem.wide_dcp_kv_heads_per_rank(4, 2, 1, "tp_group") == expected_tp2_groups
    assert vmem.wide_dcp_kv_heads_per_rank(4, 4, 1, "tp_group") == 1

    num_layers = 60
    mm_int4 = models.finalize_model_profile(
        models.ModelProfile(
            name="msa-int4-test",
            num_layers=num_layers,
            hidden_size=6144,
            num_attention_heads=64,
            num_kv_heads=4,
            head_dim=128,
            intermediate_size=12288,
            moe_intermediate_size=3072,
            shared_expert_intermediate_size=3072,
            num_experts=128,
            top_k=4,
            num_moe_layers=57,
            kv_dtype="int4_per_token_head",
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
    assert mm_int4.dcp_span == "tp_group"

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    stored_bytes_by_dcp: dict[int, float] = {}
    batch_size = 8
    ctx_tokens = 32768
    ctx_zero = 1024
    tp_deg = 8
    min_replicated_dcp = 2
    conserved_tp_pure_kv = 512.0
    const_idx_scale_per_stored_tok = 100.0
    for dcp_deg in (1, 2, 4, 8):
        wl_hi = specs.WorkloadSpec(
            phase="decode",
            batch=batch_size,
            tp=tp_deg,
            ep=8,
            dcp=dcp_deg,
            context=ctx_tokens,
        )
        wl_lo = replace(wl_hi, context=ctx_zero)
        kv_delta_bytes = (
            advisor.advise(mm_int4, wl_hi, tpu7x).hbm_used_gb
            - advisor.advise(mm_int4, wl_lo, tpu7x).hbm_used_gb
        ) * BYTES_PER_GB
        per_ctx_tok_layer_rank = kv_delta_bytes / (
            batch_size * num_layers * (ctx_tokens - ctx_zero)
        )
        per_stored_tok_layer_rank = per_ctx_tok_layer_rank * dcp_deg
        stored_bytes_by_dcp[dcp_deg] = per_stored_tok_layer_rank
        idx_scale_rank = const_idx_scale_per_stored_tok / dcp_deg
        pure_kv_rank = per_ctx_tok_layer_rank - idx_scale_rank
        if dcp_deg >= min_replicated_dcp:
            assert tp_deg * pure_kv_rank == pytest.approx(conserved_tp_pure_kv, abs=2.0)

    expected_stored = {1: 228.0, 2: 228.0, 4: 356.0, 8: 612.0}
    for dcp_deg, exp_b in expected_stored.items():
        assert stored_bytes_by_dcp[dcp_deg] == pytest.approx(exp_b, abs=0.5)
    assert stored_bytes_by_dcp[2] < stored_bytes_by_dcp[4] < stored_bytes_by_dcp[8]

    glm_prof = models.resolve_model_profile("glm-5.2")
    wide_pcp_wl = specs.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=8, ep=8, pcp=2, dcp=16
    )
    assert "dcp_not_dividing_tp" not in roofline.validate_workload(
        glm_prof, tpu7x, wide_pcp_wl
    )
    rep_glm = advisor.advise(glm_prof, wide_pcp_wl, tpu7x)
    assert rep_glm.scopes[0].feasible
    wide_pcp_ep16_wl = replace(wide_pcp_wl, ep=16)
    assert roofline.validate_workload(glm_prof, tpu7x, wide_pcp_ep16_wl) == ()

    bad_pcp1_wl = specs.WorkloadSpec(phase="decode", batch=8, tp=8, ep=8, pcp=1, dcp=16)
    assert "dcp_not_dividing_tp" in roofline.validate_workload(
        glm_prof, tpu7x, bad_pcp1_wl
    )
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(glm_prof, bad_pcp1_wl)

    bad_tp_group_wl = specs.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=8, ep=8, pcp=2, dcp=16
    )
    assert "dcp_not_dividing_tp" in roofline.validate_workload(
        mm_int4, tpu7x, bad_tp_group_wl
    )
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(mm_int4, bad_tp_group_wl)


def test_hadamard_int4_and_turboquant_kv_bytes_and_vpu_scaling() -> None:
    """Verify int4/fp8/bf16 MSA cache bytes, Hadamard VPU ops, and MSA collectives."""
    num_layers = 60
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
        for lyr in range(num_layers)
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
        for lyr in range(num_layers)
    )
    base_msa = models.finalize_model_profile(
        models.ModelProfile(
            name="msa-cache-check",
            num_layers=num_layers,
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
    batch_size = 8
    ctx_lo = 1024
    ctx_hi = 32768
    for kv_dt, exp_triple in expected_table.items():
        prof_dt = models.finalize_model_profile(replace(base_msa, kv_dtype=kv_dt))
        got_list: list[float] = []
        for dcp_deg in (2, 4, 8):
            hbm_hi = (
                advisor.advise(
                    prof_dt,
                    specs.WorkloadSpec(
                        phase="decode",
                        batch=batch_size,
                        tp=8,
                        dcp=dcp_deg,
                        context=ctx_hi,
                    ),
                    tpu7x,
                ).hbm_used_gb
                * BYTES_PER_GB
            )
            hbm_lo = (
                advisor.advise(
                    prof_dt,
                    specs.WorkloadSpec(
                        phase="decode",
                        batch=batch_size,
                        tp=8,
                        dcp=dcp_deg,
                        context=ctx_lo,
                    ),
                    tpu7x,
                ).hbm_used_gb
                * BYTES_PER_GB
            )
            got_list.append(
                (hbm_hi - hbm_lo)
                * dcp_deg
                / (batch_size * num_layers * (ctx_hi - ctx_lo))
            )
        assert tuple(got_list) == pytest.approx(exp_triple, abs=0.6)

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
    ops_fp8 = roofline.compute_hadamard_kv_bf16_flops("fp8", (4, 1, 128), 16, 8)
    ops_int4 = roofline.compute_hadamard_kv_bf16_flops(
        "int4_per_token_head", (4, 1, 128), 16, 8
    )
    ops_tq = roofline.compute_hadamard_kv_bf16_flops(
        "turboquant_4bit_nc", (4, 1, 128), 16, 8
    )
    expected_tq_flops = float(2 * 128 * 128 * (4 + 1) * 16 * 8)
    expected_int4_flops = 2.0 * expected_tq_flops
    assert ops_fp8 == 0.0
    assert ops_tq == pytest.approx(expected_tq_flops)
    assert ops_int4 == pytest.approx(expected_int4_flops)

    bf16_only_chip = replace(
        tpu7x,
        fp8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        vector_f32_ops_per_s=UNBOUNDED_RATE,
    )
    rot_wl = specs.WorkloadSpec(
        phase="decode", batch=2048, tp=8, context=1024, dispatch_regime="zero_floor"
    )
    gqa_bf16 = models.finalize_model_profile(replace(gqa_base, kv_dtype="bf16"))
    gqa_tq = models.finalize_model_profile(
        replace(gqa_base, kv_dtype="turboquant_4bit_nc")
    )
    gqa_int4 = models.finalize_model_profile(
        replace(gqa_base, kv_dtype="int4_per_token_head")
    )
    bf16_us_fp8 = _scope_map(advisor.evaluate_scopes(gqa_base, rot_wl, bf16_only_chip))[
        "per_op"
    ].compute_us
    bf16_us_bf16 = _scope_map(
        advisor.evaluate_scopes(gqa_bf16, rot_wl, bf16_only_chip)
    )["per_op"].compute_us
    bf16_us_tq = _scope_map(advisor.evaluate_scopes(gqa_tq, rot_wl, bf16_only_chip))[
        "per_op"
    ].compute_us
    bf16_us_int4 = _scope_map(
        advisor.evaluate_scopes(gqa_int4, rot_wl, bf16_only_chip)
    )["per_op"].compute_us
    assert bf16_us_int4 > bf16_us_tq > bf16_us_bf16 > bf16_us_fp8
    assert bf16_us_int4 - bf16_us_bf16 == pytest.approx(
        2.0 * (bf16_us_tq - bf16_us_bf16), abs=0.03
    )

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
    po_dcp1 = _scope_map(msa_dcp1.scopes)["per_op"]
    po_dcp4 = _scope_map(msa_dcp4.scopes)["per_op"]
    assert po_dcp4.comm_us > 0.0
    assert po_dcp4.comm_us > po_dcp1.comm_us

    wl = specs.WorkloadSpec(phase="decode", batch=16, tp=8, context=2048)
    with pytest.raises(errors.MegakernelConfigError):
        vmem.validate_advise_inputs(
            replace(gqa_base, weight_dtype="int4_per_token_head"), wl
        )


def test_hyper_connection_lowrank_and_mhc_sinkhorn_scaling() -> None:
    """Verify low-rank HC and mHC parameter counts, VMEM, and Sinkhorn scaling."""
    q38_base = models.finalize_model_profile(
        models.ModelProfile(
            name="q38-no-hc",
            num_layers=48,
            hidden_size=2560,
            num_attention_heads=24,
            num_kv_heads=2,
            head_dim=256,
            weight_dtype="bf16",
        )
    )
    q38_hc_prof = models.finalize_model_profile(
        replace(
            q38_base,
            name="q38-hc",
            layer_families=(),
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
    assert (
        roofline.compute_decoder_param_count(q38_hc_prof)
        - roofline.compute_decoder_param_count(q38_base)
        == expected_q38_hc_params
    )

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

    wl = specs.WorkloadSpec(
        phase="decode", batch=32, tp=8, context=1024, dispatch_regime="zero_floor"
    )
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
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    vec_only_chip = replace(
        tpu7x,
        bf16_flops_per_s=UNBOUNDED_RATE,
        fp8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
    )
    c_it20 = _scope_map(advisor.evaluate_scopes(dsv_mhc_prof, wl, vec_only_chip))[
        "per_op"
    ].compute_us
    c_it5 = _scope_map(advisor.evaluate_scopes(dsv_mhc_it5, wl, vec_only_chip))[
        "per_op"
    ].compute_us
    assert c_it20 > c_it5


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

    tpu7x = advisor.resolve_chip_spec("tpu7x")
    rep_dec = advisor.advise(prof, wl_dec, tpu7x)
    sc_dec = _scope_map(rep_dec.scopes)
    assert not sc_dec["whole_step"].feasible
    assert sc_dec["whole_step"].reason == "Fusion break at host-table or CED boundary"
    assert sc_dec["whole_step"].support == "outside_core_limits"
    assert "fusion_break" in sc_dec["whole_step"].blockers
    assert sc_dec["layer_group"].feasible
    expected_three_groups = 3
    assert sc_dec["layer_group"].ops_per_step == expected_three_groups

    rep_err = advisor.advise(prof, wl_dec, tpu7x, plan_error="custom_planner_failure")
    sc_err = _scope_map(rep_err.scopes)
    assert not sc_err["whole_step"].feasible
    assert sc_err["whole_step"].reason == "custom_planner_failure"
    assert "fusion_break" in sc_err["whole_step"].blockers
    assert "planner_rejected" in sc_err["whole_step"].blockers

    wl_hbm_all = replace(
        wl_dec,
        host_table_hbm_overrides=(("engram_l1", True), ("engram_l14", True)),
    )
    assert vmem.active_fusion_break_layers(prof, wl_hbm_all) == ()
    rep_hbm = advisor.advise(prof, wl_hbm_all, tpu7x)
    expected_added_gb = (
        (eng_l1.table_bytes_total + eng_l14.table_bytes_total) / 8 / BYTES_PER_GB
    )
    assert rep_hbm.hbm_used_gb - rep_dec.hbm_used_gb == pytest.approx(
        expected_added_gb, abs=1e-2
    )
    assert _scope_map(rep_hbm.scopes)["whole_step"].feasible
    assert "fusion_break" not in _scope_map(rep_hbm.scopes)["whole_step"].blockers

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
    rep_full = advisor.advise(prof, wl_pref_full, tpu7x)
    rep_ced = advisor.advise(prof, wl_pref_ced, tpu7x)
    assert not _scope_map(rep_ced.scopes)["whole_step"].feasible
    assert (
        _scope_map(rep_ced.scopes)["per_op"].compute_us
        < _scope_map(rep_full.scopes)["per_op"].compute_us
    )


def test_grouped_o_lora_moe_layer_freq_and_primitives() -> None:
    """Verify o_lora/o_groups, moe_layer_freq, qwen4_exp state dtype, and aliases."""
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
            "model_type": "glm_moe_dsa",
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
    dsv_prof = models.parse_hf_config_profile(dsv_cfg, name="dsv-prim")
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
        "model_type": "qwen3_5_moe",
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
    }
    mm_prof = models.parse_hf_config_profile(mm_cfg, name="msa-prim")
    expected_dense_inter = 12288
    expected_moe_inter = 3072
    expected_moe_layers = 57
    assert mm_prof.intermediate_size == expected_dense_inter
    assert mm_prof.moe_intermediate_size == expected_moe_inter
    assert mm_prof.shared_expert_intermediate_size == expected_moe_inter
    assert mm_prof.num_moe_layers == expected_moe_layers
    assert mm_prof.dcp_span == "tp_group"

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
            "model_type": "qwen3_5_moe_text",
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
    q38_prof = models.parse_hf_config_profile(q38_cfg, name="q38-prim")
    assert q38_prof.weight_dtype == "bf16"
    assert q38_prof.act_dtype == "bf16"
    assert q38_prof.expert_dtype == "fp8"
    assert q38_prof.expert_act_dtype == "fp8"
    assert q38_prof.has_attn_output_gate
    assert len(q38_prof.host_tables) == 1
    assert q38_prof.host_tables[0].layers == (1,)
    assert q38_prof.host_tables[0].hbm_resident

    q38_no_tbl = replace(q38_prof, host_tables=())
    assert (
        roofline.compute_decoder_param_count(q38_prof)
        - roofline.compute_decoder_param_count(q38_no_tbl)
        == q38_prof.host_tables[0].dense_proj_params
    )

    assert models.default_serving_state_dtype("qwen4_exp_text") == "fp32"
    assert models.default_serving_state_dtype("kimi_linear") == "fp32"
    assert models.default_serving_state_dtype("qwen3_5_moe_text") == "bf16"

    alias_prof = models.finalize_model_profile(replace(q38_prof, kv_dtype="bfloat16"))
    assert alias_prof.kv_dtype == "bf16"
    auto_prof = models.finalize_model_profile(replace(q38_prof, kv_dtype="auto"))
    assert auto_prof.kv_dtype == "bf16"


def test_stage_a_three_models_zero_price_drift() -> None:
    """Assert zero pricing drift across all 3 shipped Stage A presets."""
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
            (
                exp_bound,
                exp_strict,
                exp_ach,
                exp_hbm,
                exp_max_tokens,
                exp_splits,
            ) = scope_oracle[sc.scope]
            assert sc.step_bound_us == exp_bound, f"{slug} {sc.scope} step_bound_us"
            assert sc.step_floor_strict_us == exp_strict, f"{slug} {sc.scope} strict"
            assert sc.step_achievable_us == exp_ach, f"{slug} {sc.scope} achievable"
            assert rep.hbm_used_gb == exp_hbm, f"{slug} {sc.scope} hbm_used_gb"
            assert sc.max_vmem_tokens == exp_max_tokens, f"{slug} {sc.scope} max_vmem"
            assert sc.num_vmem_splits == exp_splits, f"{slug} {sc.scope} splits"


def _post_json(
    host: str, port: int, path: str, payload: dict[str, object]
) -> tuple[int, dict[str, object]]:
    """POST JSON payload to a loopback workbench endpoint and return status and body."""
    raw_body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=5.0)
    try:
        conn.request(
            "POST",
            path,
            body=raw_body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        return int(resp.status), json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def test_cli_and_serve_execution_and_host_table_wiring(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --execution, --host-table-hbm, serve HTTP JSON, and layout parsing."""
    parser = cli.build_parser()
    parsed = parser.parse_args(
        [
            "--model",
            "qwen3.5-397b",
            "--phase",
            "prefill",
            "--execution",
            "ced",
            "--host-table-hbm",
            "ple=false",
            "--host-table-hbm",
            "engram_l1=true",
        ]
    )
    assert parsed.execution == "ced"
    assert parsed.host_table_hbm == ["ple=false", "engram_l1=true"]

    rc = cli.main(
        [
            "--model",
            "qwen3.5-397b",
            "--phase",
            "prefill",
            "--execution",
            "ced",
            "--host-table-hbm",
            "ple=false",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["model"] == "qwen3.5-397b"

    with pytest.raises(errors.MegakernelConfigError):
        cli.main(["--model", "qwen3.5-397b", "--host-table-hbm", "bad_format"])

    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="dp_local",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="serve-break-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            ced_boundary=20,
            sparse_attn=specs.SparseAttnSpec(
                kind="csa", window_tokens=128, select_tokens=512
            ),
            host_tables=(eng_l1,),
        )
    )
    server = serve.create_workbench_server(model=prof, chip="tpu7x", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        bound_host, bound_port = (
            str(server.server_address[0]),
            int(server.server_address[1]),
        )
        status_ced, ced_body = _post_json(
            bound_host,
            bound_port,
            "/api/advise",
            {
                "phase": "prefill",
                "batch": 8,
                "isl": 4096,
                "tp": 1,
                "dp": 8,
                "execution": "ced",
                "host_table_hbm_overrides": {"engram_l1": True},
            },
        )
        assert status_ced == HTTP_OK
        ced_scopes = {item["scope"]: item for item in ced_body["report"]["scopes"]}
        assert not ced_scopes["whole_step"]["feasible"]
        expected_ced_two_groups = 2
        assert ced_scopes["layer_group"]["ops_per_step"] == expected_ced_two_groups

        status_full, full_body = _post_json(
            bound_host,
            bound_port,
            "/api/advise",
            {
                "phase": "prefill",
                "batch": 8,
                "isl": 4096,
                "tp": 1,
                "dp": 8,
                "execution": "full",
                "host_table_hbm_overrides": {"engram_l1": True},
            },
        )
        assert status_full == HTTP_OK
        full_scopes = {item["scope"]: item for item in full_body["report"]["scopes"]}
        assert full_scopes["whole_step"]["feasible"]
        assert ced_scopes["per_op"]["compute_us"] < full_scopes["per_op"]["compute_us"]

        status_bad_exec, _ = _post_json(
            bound_host, bound_port, "/api/advise", {"execution": "invalid_mode"}
        )
        assert status_bad_exec == HTTP_BAD_REQUEST
        status_bad_ov, _ = _post_json(
            bound_host,
            bound_port,
            "/api/advise",
            {"host_table_hbm_overrides": {"engram_l1": "not_bool"}},
        )
        assert status_bad_ov == HTTP_BAD_REQUEST
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    layout = models.parse_parallel_layout(
        {
            "tp": 1,
            "ep": 8,
            "dp": 8,
            "execution": "ced",
            "host_table_hbm_overrides": {"ple": False},
        }
    )
    assert layout.execution == "ced"
    assert layout.host_table_hbm_overrides == (("ple", False),)

    with pytest.raises(errors.MegakernelConfigError):
        models.parse_parallel_layout({"tp": 1, "ep": 1, "execution": "bogus"})


def test_with_layout_propagates_execution_and_host_table_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify align_batch_to_dp and with_layout propagate execution and overrides."""
    expected_dp_batch = 8
    assert specs.align_batch_to_dp(1, 8) == expected_dp_batch
    layout = advisor.ParallelLayout(
        tp=1,
        ep=8,
        dp=8,
        execution="ced",
        host_table_hbm_overrides=(("engram_l1", True),),
    )
    wl = specs.WorkloadSpec(phase="prefill", batch=1).with_layout(
        layout, align_batch=True
    )
    assert wl.batch == expected_dp_batch
    assert wl.effective_execution == "ced"
    assert wl.effective_host_table_hbm_overrides == (("engram_l1", True),)

    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="dp_local",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="ssot-prop-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            ced_boundary=20,
            sparse_attn=specs.SparseAttnSpec(
                kind="csa", window_tokens=128, select_tokens=512
            ),
            host_tables=(eng_l1,),
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    base_sweep = advisor.advise_regimes(
        prof,
        tpu7x,
        options=advisor.RegimeSweepOptions(tp=1, dp=8, ep=8),
    )
    ced_override_par = replace(
        base_sweep.parallelism,
        execution="ced",
        host_table_hbm_overrides=(("engram_l1", True),),
        layer_groups=(),
    )
    ced_override_sweep = replace(
        base_sweep,
        parallelism=ced_override_par,
        vmem_reference_scope="layer_group",
    )
    no_override_sweep = replace(
        base_sweep,
        parallelism=replace(ced_override_par, host_table_hbm_overrides=()),
        vmem_reference_scope="layer_group",
    )

    seen_tick_workloads: list[specs.WorkloadSpec] = []
    orig_eval_scopes = report.evaluate_scopes

    def _spy_evaluate_scopes(
        profile_arg: models.ModelProfile,
        workload_arg: specs.WorkloadSpec,
        chip_arg: advisor.ChipSpec,
    ) -> tuple[advisor.ScopeEstimate, ...]:
        """Record tick workloads evaluated by `_vmem_tick_series`."""
        seen_tick_workloads.append(workload_arg)
        return orig_eval_scopes(profile_arg, workload_arg, chip_arg)

    monkeypatch.setattr(report, "evaluate_scopes", _spy_evaluate_scopes)
    html_with_ov = report.render_advisor_html(ced_override_sweep)
    assert seen_tick_workloads
    assert all(tick_wl.effective_execution == "ced" for tick_wl in seen_tick_workloads)
    assert all(
        tick_wl.effective_host_table_hbm_overrides == (("engram_l1", True),)
        for tick_wl in seen_tick_workloads
    )
    html_without_ov = report.render_advisor_html(no_override_sweep)
    assert "Adaptive Weight Ring (2,0)" in html_without_ov
    assert "Adaptive Weight Ring (2,0)" not in html_with_ov

    dec_map = workbench.build_phase_map_payload(
        prof,
        tpu7x,
        ced_override_sweep,
        options=workbench.PhaseMapOptions(phase="decode"),
    )
    dec_cells = dec_map["cells"]
    assert isinstance(dec_cells, list)
    assert dec_cells[0]["recommended_scope"] == "whole_step"
    dec_no_ov_map = workbench.build_phase_map_payload(
        prof,
        tpu7x,
        no_override_sweep,
        options=workbench.PhaseMapOptions(phase="decode"),
    )
    dec_no_ov_cells = dec_no_ov_map["cells"]
    assert isinstance(dec_no_ov_cells, list)
    assert dec_no_ov_cells[0]["recommended_scope"] != "whole_step"

    pref_ced_map = workbench.build_phase_map_payload(
        prof,
        tpu7x,
        ced_override_sweep,
        options=workbench.PhaseMapOptions(phase="prefill"),
    )
    pref_full_sweep = replace(
        ced_override_sweep,
        parallelism=replace(ced_override_par, execution="full"),
    )
    pref_full_map = workbench.build_phase_map_payload(
        prof,
        tpu7x,
        pref_full_sweep,
        options=workbench.PhaseMapOptions(phase="prefill"),
    )
    pref_ced_cells = pref_ced_map["cells"]
    pref_full_cells = pref_full_map["cells"]
    assert isinstance(pref_ced_cells, list)
    assert isinstance(pref_full_cells, list)
    assert pref_ced_cells[0]["recommended_scope"] != "whole_step"
    assert pref_full_cells[0]["recommended_scope"] == "whole_step"

    dp32_sweep = replace(
        ced_override_sweep,
        parallelism=replace(ced_override_par, dp=32),
        regimes=(),
    )
    seed = workbench.build_workbench_seed(prof, tpu7x, dp32_sweep)
    seed_phase_map = seed["phase_map"]
    assert isinstance(seed_phase_map, dict)
    assert seed_phase_map["cells"]


def test_layer_group_honors_max_rows_per_call_and_reference_scope() -> None:
    """Verify layer_group microbatches at max_rows_per_call and sets ref scope."""
    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="tp",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="lg-row-cap-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            host_tables=(eng_l1,),
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(
        prof,
        tpu7x,
        options=advisor.RegimeSweepOptions(tp=8, dp=1),
    )
    row_cap = 32
    capped_sweep = advisor.advise_regimes(
        prof,
        tpu7x,
        options=advisor.RegimeSweepOptions(
            tp=8,
            dp=1,
            base_workload=specs.WorkloadSpec(max_rows_per_call=row_cap),
        ),
    )
    assert sweep.vmem_reference_scope == "layer_group"
    assert capped_sweep.vmem_reference_scope == "layer_group"
    assert capped_sweep.max_vmem_tokens == row_cap < sweep.max_vmem_tokens

    uncapped_wl = specs.WorkloadSpec(phase="decode", batch=64, tp=8, context=8192)
    capped_wl = replace(uncapped_wl, max_rows_per_call=row_cap)
    uncapped = _scope_map(advisor.evaluate_scopes(prof, uncapped_wl, tpu7x))
    capped = _scope_map(advisor.evaluate_scopes(prof, capped_wl, tpu7x))
    assert uncapped["layer_group"].feasible
    assert capped["layer_group"].max_vmem_tokens == row_cap
    assert (
        capped["layer_group"].num_vmem_splits > uncapped["layer_group"].num_vmem_splits
    )
    for scope_name in ("per_op", "single_layer"):
        assert (
            capped[scope_name].max_vmem_tokens == uncapped[scope_name].max_vmem_tokens
        )
        assert (
            capped[scope_name].num_vmem_splits == uncapped[scope_name].num_vmem_splits
        )
        assert capped[scope_name].step_bound_us == uncapped[scope_name].step_bound_us

    html_doc = report.render_advisor_html(capped_sweep)
    first_svg = html_doc.split("</svg>", 1)[0]
    assert "row cap" in first_svg


def test_with_layout_preserves_explicit_execution_and_merges_table_overrides() -> None:
    """Verify with_layout keeps explicit execution and merges table overrides."""
    ced_layout = advisor.ParallelLayout(tp=1, ep=8, dp=8, execution="ced")
    full_layout = advisor.ParallelLayout(tp=1, ep=8, dp=8, execution="full")
    ple_cpu_layout = advisor.ParallelLayout(
        tp=8, ep=8, dp=1, host_table_hbm_overrides=(("ple", False),)
    )
    empty_layout = advisor.ParallelLayout(tp=8, ep=8, dp=1)
    mixed_layout = advisor.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        host_table_hbm_overrides=(("ple", False), ("engram", True)),
    )

    assert (
        specs.WorkloadSpec(execution="full").with_layout(ced_layout).effective_execution
        == "full"
    )
    assert specs.WorkloadSpec().with_layout(ced_layout).effective_execution == "ced"
    assert (
        specs.WorkloadSpec()
        .with_layout(ced_layout)
        .with_layout(full_layout)
        .effective_execution
        == "full"
    )
    assert (
        dataclasses.replace(specs.WorkloadSpec(), batch=32)
        .with_layout(ced_layout)
        .effective_execution
        == "ced"
    )
    assert (
        replace(specs.WorkloadSpec(), execution="ced")
        .with_layout(full_layout)
        .effective_execution
        == "ced"
    )
    assert (
        replace(
            specs.WorkloadSpec().with_layout(ced_layout), execution="full"
        ).effective_execution
        == "full"
    )

    assert replace(
        specs.WorkloadSpec(), host_table_hbm_overrides=(("ple", True),)
    ).with_layout(ple_cpu_layout).effective_host_table_hbm_overrides == (("ple", True),)
    merged_wl = specs.WorkloadSpec(
        host_table_hbm_overrides=(("ple", True),)
    ).with_layout(mixed_layout)
    assert dict(merged_wl.effective_host_table_hbm_overrides) == {
        "ple": True,
        "engram": True,
    }
    assert (
        specs.WorkloadSpec()
        .with_layout(ple_cpu_layout)
        .with_layout(empty_layout)
        .effective_host_table_hbm_overrides
        == ()
    )
    assert (
        specs.WorkloadSpec(host_table_hbm_overrides=())
        .with_layout(ple_cpu_layout)
        .effective_host_table_hbm_overrides
        == ()
    )


def test_csa_ced_boundary_uses_half_num_layers_independently_of_candidate_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CSA CED boundary uses half-depth only for allowlisted model types."""
    monkeypatch.setattr(
        models,
        "SUPPORTED_MODEL_TYPES",
        models.SUPPORTED_MODEL_TYPES | {"deepseek_v41", "generic_csa"},
    )
    expected_layers = 40
    expected_half_boundary = 20
    expected_explicit_boundary = 12
    candidate_src = 18
    base_text: dict[str, object] = {
        "num_hidden_layers": expected_layers,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 129280,
        "kv_source_layer_ids": list(range(expected_layers)),
        "index_source_layer_ids": list(range(expected_layers)),
        "compress_ratios": [4] * expected_layers,
        "candidate_source_layer_id": candidate_src,
        "candidate_topk_blocks": 16,
        "candidate_block_size": 128,
        "index_topk": 2048,
        "index_n_heads": 8,
        "index_head_dim": 128,
    }
    dsv41_prof = models.parse_hf_config_profile(
        {"text_config": {**base_text, "model_type": "deepseek_v41"}},
        name="csa-dsv41-test",
    )
    assert dsv41_prof.ced_boundary == expected_half_boundary

    explicit_prof = models.parse_hf_config_profile(
        {
            "text_config": {
                **base_text,
                "model_type": "generic_csa",
                "ced_boundary": expected_explicit_boundary,
            }
        },
        name="csa-explicit-test",
    )
    assert explicit_prof.ced_boundary == expected_explicit_boundary

    unknown_prof = models.parse_hf_config_profile(
        {"text_config": {**base_text, "model_type": "generic_csa"}},
        name="csa-unknown-test",
    )
    assert unknown_prof.ced_boundary == 0


def test_turboquant_rejects_unsupported_kv_page_size() -> None:
    """Verify turboquant_4bit_nc accepts page sizes 16 and 64 and rejects 32."""
    head_dim = 128
    unsupported_page_size = 32
    supported_page_16 = 16
    supported_page_64 = 64

    assert roofline.turboquant_slot_bytes(head_dim, supported_page_16) > 0.0
    assert roofline.turboquant_slot_bytes(head_dim, supported_page_64) > 0.0
    with pytest.raises(errors.MegakernelConfigError):
        roofline.turboquant_slot_bytes(head_dim, unsupported_page_size)

    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="tq-page-size-test",
            num_layers=8,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=head_dim,
            kv_dtype="turboquant_4bit_nc",
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    bad_wl = specs.WorkloadSpec(
        phase="decode", batch=8, tp=8, context=2048, kv_page_size=unsupported_page_size
    )
    with pytest.raises(errors.MegakernelConfigError):
        advisor.evaluate_scopes(prof, bad_wl, tpu7x)

    for ok_page_size in (supported_page_16, supported_page_64):
        ok_wl = replace(bad_wl, kv_page_size=ok_page_size)
        assert advisor.evaluate_scopes(prof, ok_wl, tpu7x)


def test_ple_table_bytes_derived_from_hf_ngram_config() -> None:
    """Verify PLE table bytes are derived from HF n-gram vocab base, size, and heads."""
    expected_q38_ple_bytes = 51_200_245_760
    base_cfg: dict[str, object] = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 8,
            "hidden_size": 2560,
            "intermediate_size": 7680,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 248320,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_vocab_size_base": 20_000_000,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "make_ngram_vocab_size_divisible_by": 128,
        }
    }
    q38_prof = models.parse_hf_config_profile(base_cfg, name="ple-q38")
    assert q38_prof.host_tables[0].table_bytes_total == expected_q38_ple_bytes

    smaller_base_cfg: dict[str, object] = {
        "text_config": {
            **base_cfg["text_config"],  # type: ignore[dict-item]
            "ngram_vocab_size_base": 100_000,
        }
    }
    smaller_prof = models.parse_hf_config_profile(smaller_base_cfg, name="ple-small")
    assert smaller_prof.host_tables[0].table_bytes_total < expected_q38_ple_bytes

    wider_ngram_cfg: dict[str, object] = {
        "text_config": {
            **base_cfg["text_config"],  # type: ignore[dict-item]
            "ngram_size": 4,
            "heads_per_ngram": 4,
        }
    }
    wider_prof = models.parse_hf_config_profile(wider_ngram_cfg, name="ple-wider")
    assert wider_prof.host_tables[0].table_bytes_total != expected_q38_ple_bytes


def test_qsa_select_tokens_matches_non_divisible_budget() -> None:
    """Verify QSA selected token count rounds up to block boundary minus 1."""
    expected_select_tokens = 2051
    qsa_cfg: dict[str, object] = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 8,
            "hidden_size": 2560,
            "intermediate_size": 7680,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 248320,
            "indexer_budget": 2050,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
        }
    }
    prof = models.parse_hf_config_profile(qsa_cfg, name="qsa-nondiv-test")
    assert prof.sparse_attn is not None
    assert prof.sparse_attn.select_tokens == expected_select_tokens


def test_wide_dcp_kv_heads_per_rank_matches_sharding_rule() -> None:
    """Verify Wide-DCP KV heads per rank match kv_head_replicas across 8 pairs."""
    num_kv_heads = 4
    pairs = (
        (8, 4),
        (8, 8),
        (4, 4),
        (8, 2),
        (8, 1),
        (16, 8),
        (16, 16),
        (4, 2),
    )
    for tp_val, dcp_val in pairs:
        expected = max(1, dcp_val // max(1, tp_val // num_kv_heads))
        assert (
            vmem.wide_dcp_kv_heads_per_rank(num_kv_heads, tp_val, dcp_val, "tp_group")
            == expected
        )


def test_sparse_attn_dcp_uses_non_dsa_collective_hop_counts() -> None:
    """Verify non-DSA sparse attention with DCP > 1 counts DCP collective hops."""
    msa_prof = models.finalize_model_profile(
        models.ModelProfile(
            name="msa-dcp-hops-test",
            num_layers=4,
            hidden_size=2048,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
            dcp_span="tp_group",
            sparse_attn=specs.SparseAttnSpec(
                kind="msa",
                window_tokens=128,
                select_tokens=512,
                index_heads=4,
                index_dim=128,
            ),
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    wl_no_dcp = specs.WorkloadSpec(
        phase="decode", batch=8, tp=1, dcp=1, kv_split=1, context=4096
    )
    wl_dcp4 = specs.WorkloadSpec(
        phase="decode", batch=8, tp=1, dcp=4, kv_split=4, context=4096
    )
    rep_no_dcp = advisor.advise(msa_prof, wl_no_dcp, tpu7x)
    rep_dcp4 = advisor.advise(msa_prof, wl_dcp4, tpu7x)

    assert not any(
        item.get("name") == "collective_latency_us"
        for item in rep_no_dcp.unresolved_assumptions
    )
    assert any(
        item.get("name") == "collective_latency_us"
        for item in rep_dcp4.unresolved_assumptions
    )


def test_layer_group_tailoring_guide_recommends_single_layer() -> None:
    """Verify layer_group ref scope avoids repeating layer_group in mid hint."""
    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="tp",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="lg-guide-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            host_tables=(eng_l1,),
        )
    )
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(
        prof,
        tpu7x,
        options=advisor.RegimeSweepOptions(tp=8, dp=1),
    )
    assert sweep.vmem_reference_scope == "layer_group"
    mid_hint = sweep.tailoring_guide[1]
    assert "scope='single_layer' (" in mid_hint
    assert "or 'layer_group'" not in mid_hint


def test_pcie_h2d_bw_scales_offload_latency_and_validates_positive() -> None:
    """Verify host-table offload H2D time uses ChipSpec.pcie_h2d_bw_bytes_per_s."""
    tpu7x = advisor.resolve_chip_spec("tpu7x")
    expected_default_pcie_bw = 64.0e9
    assert tpu7x.pcie_h2d_bw_bytes_per_s == expected_default_pcie_bw

    eng_l1 = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="tp",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="pcie-bw-test",
            num_layers=8,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            host_tables=(eng_l1,),
        )
    )
    wl = specs.WorkloadSpec(
        phase="decode",
        batch=1024,
        tp=8,
        context=1024,
        dispatch_regime="zero_floor",
    )
    slow_chip = replace(
        tpu7x,
        bf16_flops_per_s=UNBOUNDED_RATE,
        fp8_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        vector_f32_ops_per_s=UNBOUNDED_RATE,
        pcie_h2d_bw_bytes_per_s=32.0e9,
    )
    fast_chip = replace(slow_chip, pcie_h2d_bw_bytes_per_s=64.0e9)
    slow_hbm_us = _scope_map(advisor.evaluate_scopes(prof, wl, slow_chip))[
        "per_op"
    ].hbm_us
    fast_hbm_us = _scope_map(advisor.evaluate_scopes(prof, wl, fast_chip))[
        "per_op"
    ].hbm_us
    assert slow_hbm_us > fast_hbm_us > 0.0
    assert slow_hbm_us == pytest.approx(2.0 * fast_hbm_us, rel=1e-3)

    with pytest.raises(errors.MegakernelConfigError):
        advisor.resolve_chip_spec(replace(tpu7x, pcie_h2d_bw_bytes_per_s=0.0))
