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
"""Independent analytic oracle tests for HBM bytes, FLOPs, and collectives."""

from __future__ import annotations

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    Dtypes,
    FloorOptions,
    ModelGeometry,
    Parallel,
    SpecShape,
    StepShape,
    dtypes_from_hf_config,
    pass_costs,
    spec_cycle_floor,
)
from vllm_torchtpu.roofline.collectives import Collective, CollectiveConfig, make
from vllm_torchtpu.roofline.constants import KV_DTYPE_BYTES
from vllm_torchtpu.roofline.modules import ModuleCost
from vllm_torchtpu.roofline.step import combine_modules


def _tiny_gqa_moe_geom(
    *,
    num_experts: int = 64,
    top_k: int = 4,
    hc_mult: int = 1,
    shared_inter: int = 256,
) -> ModelGeometry:
    """Build a small hand-verifiable GQA+MoE geometry."""
    return ModelGeometry(
        hidden=1024,
        num_layers=2,
        layer_types=("full_attention", "full_attention"),
        num_heads=16,
        num_kv_heads=2,
        head_dim=64,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=512,
        shared_inter=shared_inter,
        dense_inter=0,
        first_k_dense_replace=0,
        vocab=32000,
        mtp_layers=1,
        hc_mult=hc_mult,
    )


def test_step_collective_term_is_max_over_disjoint_links() -> None:
    """Verify combine_modules takes max across disjoint link classes, not sum."""
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    config = CollectiveConfig(algo="one_shot", ranks_per_chip=2)
    payload_d2d = 1.2e9
    payload_ici = 0.5e9
    mod_d2d = ModuleCost(
        name="tp_block",
        hbm_bytes=0.0,
        fp8_flops=0.0,
        bf16_flops=0.0,
        collectives=(make("out_ar", "ar", payload_d2d, 2, config),),
        unique_collectives=((make("out_ar", "ar", payload_d2d, 2, config), 1),),
    )
    mod_pp = ModuleCost(
        name="pp_handoff",
        hbm_bytes=0.0,
        fp8_flops=0.0,
        bf16_flops=0.0,
        collectives=(make("pp_p2p", "p2p", payload_ici, 2, config),),
        unique_collectives=((make("pp_p2p", "p2p", payload_ici, 2, config), 1),),
    )
    parallel = Parallel(tp=2, pp=2)
    floor_result = combine_modules(
        (mod_d2d, mod_pp), chip, topology=topology, parallel=parallel
    )
    # TP=2 runs intra-chip over D2D (600 GB/s -> 1.2e9 / 600e9 = 2.0 ms).
    # PP=2 runs inter-chip over x-axis ICI (100 GB/s -> 0.5e9 / 100e9 = 5.0 ms).
    # Disjoint link classes must combine via max (5.0 ms), never sum (7.0 ms).
    expected_d2d_s = payload_d2d / 600.0e9
    expected_ici_s = payload_ici / 100.0e9
    assert floor_result.coll_bound_s == pytest.approx(
        max(expected_d2d_s, expected_ici_s), rel=1e-9
    )
    assert floor_result.coll_bound_s < expected_d2d_s + expected_ici_s


def test_collective_bytes_equal_physics_minimum() -> None:
    """Verify every collective payload matches its sharding-forced physics minimum."""
    geometry = _tiny_gqa_moe_geom(num_experts=64, top_k=4, hc_mult=2, shared_inter=256)
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, w_aux=1.0, kv=1.0, act="a8")

    # 1. Replicated embed uses 1-phase all-gather, not 2-phase all-reduce.
    mods_rep = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=2),
            StepShape(batch=1, rows_per_seq=1, context=128),
            dtypes,
        )
    }
    embed_colls = mods_rep["embed"].collectives
    assert len(embed_colls) == 1
    assert embed_colls[0].kind == "ag"

    # 2. LM-head logits_ag exchanges per-row sampling scalars, not full vocab.
    lm_colls = {c.name: c for c in mods_rep["lm_head"].collectives}
    assert lm_colls["logits_ag"].payload == pytest.approx(8.0 * 8.0, rel=1e-12)
    # pp_p2p includes hc_mult=2.
    assert lm_colls["pp_p2p"].payload == pytest.approx(
        1.0 * 1024.0 * 2.0 * 2.0, rel=1e-12
    )

    # 3. SP entry_ag, tail_ag, draft_fc_ag use unpadded rows at act width (1 B).
    par_sp = Parallel(tp=8, ep=8, moe="ep", residual="sp", kv_split=4, kv_merge="ag_rs")
    mods_sp = {
        m.name: m
        for m in pass_costs(
            geometry, par_sp, StepShape(batch=1, rows_per_seq=1, context=256), dtypes
        )
    }
    gqa_colls: dict[str, list[Collective]] = {}
    for c in mods_sp["gqa"].collectives:
        gqa_colls.setdefault(c.name, []).append(c)
    assert gqa_colls["entry_ag"][0].payload == pytest.approx(
        1.0 * 1024.0 * 1.0, rel=1e-12
    )
    # dcp_lse_ag = g * dec_rows * nq * 4 (no extra g factor); dcp_out_rs is BF16 (2 B).
    num_q_heads = 16 // 8
    assert gqa_colls["dcp_lse_ag"][0].payload == pytest.approx(
        4.0 * 1.0 * num_q_heads * 4.0, rel=1e-12
    )
    assert gqa_colls["dcp_out_rs"][0].payload == pytest.approx(
        1.0 * 4.0 * num_q_heads * 64.0 * 2.0, rel=1e-12
    )
    # Local routing (64 // 8 = 8 >= top_k=4) + SP shared expert -> 0 MoE colls.
    assert mods_sp["moe"].collectives == ()
    # tail_ag is unpadded sampled_rows=1 at 1 B.
    lm_sp = {c.name: c for c in mods_sp["lm_head"].collectives}
    assert lm_sp["tail_ag"].payload == pytest.approx(1.0 * 1024.0 * 1.0, rel=1e-12)

    # draft_fc_ag uses unpadded batch=1 at 1 B under a8.
    cyc = spec_cycle_floor(
        geometry,
        CHIPS["tpu7x"],
        par_sp,
        SpecShape(batch=1, context=256, spec_k=1),
        FloorOptions(dtypes=dtypes),
    )
    draft_fc = next(m for m in cyc.modules if m.name == "draft_fc")
    assert draft_fc.collectives[0].payload == pytest.approx(
        1.0 * 1024.0 * 1.0, rel=1e-12
    )


def test_int8_kv_scheme_priced_at_one_byte() -> None:
    """Verify int8 KV schemes resolve to 1.0 byte with dtype label int8."""
    assert KV_DTYPE_BYTES["int8"] == pytest.approx(1.0, rel=1e-12)
    dt_map = dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "kv_cache_scheme": {"type": "int", "num_bits": 8},
            },
        }
    )
    assert dt_map.kv == pytest.approx(1.0, rel=1e-12)
    assert dt_map.kv_dtype == "int8"


def test_analytic_oracle_synthetic_gqa_moe_hbm_flops_and_shared_prefix_dedup() -> None:
    """Verify exact hand-derived HBM bytes, FLOPs, and shared-prefix KV dedup."""
    geometry = _tiny_gqa_moe_geom(num_experts=16, top_k=4, hc_mult=1, shared_inter=256)
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, w_aux=1.0, kv=1.0, lm_head=2.0, act="a8")
    tp = 4
    batch = 4
    ctx = 1024
    c_shared = 512
    mods = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=tp, ep=4, moe="ep", residual="sp"),
            StepShape(
                batch=batch, rows_per_seq=1, context=ctx, shared_prefix_tokens=c_shared
            ),
            dtypes,
        )
    }
    # GQA weights per layer at TP=4: Q (1024*256) + K,V (2*1024*64) + O (256*1024)
    # = 655,360 elems/layer -> 1,310,720 elems over 2 layers.
    # Logical KV/tok at TP=4 (kvh=max(1, 2//4)=1): 2 * 1 * 64 * 1 B = 128 B.
    # Deduplicated KV tokens = c_shared + batch * (ctx - c_shared) = 2560.
    exp_gqa_w = 2.0 * (1024.0 * 256.0 + 2.0 * 1024.0 * 64.0 + 256.0 * 1024.0)
    exp_gqa_kv_min = 2.0 * float(c_shared + batch * (ctx - c_shared)) * 128.0
    assert mods["gqa"].strict_hbm_bytes == pytest.approx(
        exp_gqa_w + exp_gqa_kv_min, rel=1e-12
    )
    # Without dedup, nominal KV reads batch*ctx = 4096 tokens > 2560 tokens.
    assert mods["gqa"].strict_hbm_bytes < mods["gqa"].hbm_bytes

    # Subgroup EP (tp=8, ep=2 -> tp_sub=4) keeps mandatory tp//ep reduction.
    mods_sub = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, ep=2, moe="ep", residual="sp"),
            StepShape(batch=2, rows_per_seq=1, context=128),
            dtypes,
        )
    }
    moe_sub_colls = mods_sub["moe"].collectives
    assert any(c.name == "combine_rs" and c.group == 8 // 2 for c in moe_sub_colls)

    # Local experts < top_k (16 // 8 = 2 < 4) requires ceil(4/2)-1 = 1 remote push.
    mods_push = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, ep=8, moe="ep", residual="sp"),
            StepShape(batch=8, rows_per_seq=1, context=128),
            dtypes,
        )
    }
    push_names = {c.name for c in mods_push["moe"].collectives}
    assert push_names == {"dispatch_push", "combine_push"}


def test_dcp_mla_rope_pcp_prefix_and_swa_window_collectives() -> None:
    """Verify MLA dcp_q_ag includes rope_d, pcp_kv_ag includes prefix/DSA, SWA caps."""
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, w_aux=1.0, kv=1.0, act="a8")
    dsa_geom = ModelGeometry(
        hidden=1024,
        num_layers=1,
        layer_types=("dsa",),
        num_heads=16,
        num_kv_heads=16,
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
        first_k_dense_replace=1,
        vocab=16000,
        mtp_layers=0,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=256,
    )
    # Decode DCP: dcp_q_ag carries kv_lora + rope_d = 320.
    dec_dsa = {
        m.name: m
        for m in pass_costs(
            dsa_geom,
            Parallel(tp=4, kv_split=4, kv_merge="ag_rs"),
            StepShape(batch=2, rows_per_seq=1, context=512),
            dtypes,
        )
    }["dsa"]
    q_ag = next(c for c in dec_dsa.collectives if c.name == "dcp_q_ag")
    assert q_ag.payload == pytest.approx(
        2.0 * 4.0 * 4.0 * (256.0 + 64.0) * 2.0, rel=1e-12
    )

    # Pure prefill DCP at h=0 emits 0 DCP colls; PCP+DCP shards KV by pcp*kv_split.
    pre_h0 = {
        m.name: m
        for m in pass_costs(
            dsa_geom,
            Parallel(tp=4, pcp=2, kv_split=2),
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=512,
                prefill_tokens=512,
                prefix_hit_rate=0.0,
            ),
            dtypes,
        )
    }["dsa"]
    assert not any(c.name.startswith("dcp_") for c in pre_h0.collectives)
    pcp_ag = next(c for c in pre_h0.collectives if c.name == "pcp_kv_ag")
    # Includes DSA index_head_dim=64 alongside kv_lora+rope_d=320 -> 384 B/tok.
    assert pcp_ag.payload == pytest.approx(
        512.0 * (256.0 + 64.0 + 64.0) * 1.0, rel=1e-12
    )

    # Prefill cached_prefix=256 + eff_prefill*pcp=256 -> pcp_kv_ag covers 512 toks.
    pre_h50 = {
        m.name: m
        for m in pass_costs(
            dsa_geom,
            Parallel(tp=4, pcp=2, kv_split=1),
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=512,
                prefill_tokens=512,
                prefix_hit_rate=0.5,
            ),
            dtypes,
        )
    }["dsa"]
    pcp_ag_h50 = next(c for c in pre_h50.collectives if c.name == "pcp_kv_ag")
    assert pcp_ag_h50.payload == pytest.approx(
        512.0 * (256.0 + 64.0 + 64.0) * 1.0, rel=1e-12
    )

    # SWA pcp_kv_ag caps transferred tokens at sliding_window=128.
    swa_geom = ModelGeometry(
        hidden=1024,
        num_layers=1,
        layer_types=("sliding_attention",),
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
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
        first_k_dense_replace=1,
        vocab=16000,
        mtp_layers=0,
        sliding_window=128,
    )
    swa_mod = {
        m.name: m
        for m in pass_costs(
            swa_geom,
            Parallel(tp=2, pcp=4),
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=1024,
                prefill_tokens=1024,
                prefix_hit_rate=0.0,
            ),
            dtypes,
        )
    }["swa"]
    swa_pcp = next(c for c in swa_mod.collectives if c.name == "pcp_kv_ag")
    # kvh=2, hd=64 -> 2*2*64*1 = 256 B/tok; capped at sliding_window=128 tokens.
    assert swa_pcp.payload == pytest.approx(128.0 * 256.0, rel=1e-12)
