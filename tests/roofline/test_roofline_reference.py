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
"""Independent zero-shared-code reference oracle tests for roofline modules."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    pass_costs,
)
from vllm_torchtpu.roofline.constants import (
    FP8_BYTES,
    MATMUL_FLOP_PER_MAC,
)

_GQA_SCORE_WIDTH_MULT = 4.0
_MLA_SCORE_WIDTH_MULT = 2.0
_LINEAR_REC_FLOP_MULT = 4.0
_GLU_MATMUL_COUNT = 3
_LATENT_MATMUL_COUNT = 2
_CAUSAL_HALF = 0.5
_REL_TOL = 1e-9

_BASE_GEOMETRY = ModelGeometry(
    hidden=2048,
    num_layers=2,
    layer_types=("full_attention", "full_attention"),
    num_heads=16,
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
    dense_inter=4096,
    first_k_dense_replace=2,
    vocab=32000,
    mtp_layers=0,
)


@dataclass(frozen=True, slots=True)
class _StrictExpectation:
    """Expected strict bytes and matrix FLOPs from the reference oracle."""

    hbm_bytes: float
    fp8_flops: float
    bf16_flops: float


def _is_fp8_matmul(weight_bytes: float, act: str) -> bool:
    """Return whether a matmul routes to FP8 matrix units."""
    return weight_bytes <= FP8_BYTES and act == "a8"


def _add_matmul_flops(
    rows: float,
    weight_elems: float,
    weight_bytes: float,
    act: str,
) -> tuple[float, float]:
    """Return (fp8_flops, bf16_flops) for an unpadded reference matmul."""
    flops = MATMUL_FLOP_PER_MAC * rows * weight_elems
    if _is_fp8_matmul(weight_bytes, act):
        return flops, 0.0
    return 0.0, flops


def _ref_causal_capped_pairs(
    cached_prefix: int,
    prefill_global: int,
    pcp: int,
    cap: int,
) -> float:
    """Closed-form reference causal pair count per PCP rank with optional window cap."""
    if prefill_global <= 0:
        return 0.0
    end_pos = cached_prefix + prefill_global
    if cap <= 0 or end_pos <= cap:
        global_pairs = float(prefill_global) * (
            float(cached_prefix) + _CAUSAL_HALF * float(prefill_global)
        )
    elif cached_prefix >= cap:
        global_pairs = float(prefill_global * cap)
    else:
        n_ramp = cap - cached_prefix
        n_cap = end_pos - cap
        global_pairs = float(n_ramp) * (
            float(cached_prefix) + _CAUSAL_HALF * float(n_ramp)
        ) + float(n_cap * cap)
    return global_pairs / float(pcp)


def _step_rows(shape: StepShape, pcp: int) -> tuple[int, int, int, int]:
    """Return (decode_rows, cached_prefix, eff_prefill, total_rows) per rank."""
    decode_rows = shape.batch * shape.rows_per_seq
    if shape.prefill_tokens > 0:
        eff_prefill = max(
            1,
            math.ceil(
                shape.prefill_tokens * max(0.0, 1.0 - shape.prefix_hit_rate) / pcp
            ),
        )
        cached_prefix = round(shape.prefill_tokens * shape.prefix_hit_rate)
    else:
        eff_prefill = 0
        cached_prefix = 0
    return decode_rows, cached_prefix, eff_prefill, decode_rows + eff_prefill


def _ref_gqa(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> _StrictExpectation:
    """Reference strict oracle for GQA attention layers."""
    decode_rows, cached_prefix, eff_prefill, rows = _step_rows(shape, par.pcp)
    tp = par.tp
    n_layers = float(geom.layer_types.count("full_attention"))
    hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    nq = geom.num_heads // tp
    kvh = geom.num_kv_heads // tp
    q_mult = 2 if geom.attn_output_gate else 1
    w_elems = n_layers * float(
        geom.hidden * (nq * q_mult * hd)
        + 2 * geom.hidden * (kvh * hd)
        + (nq * hd) * geom.hidden
    )
    kv_tokens = n_layers * (
        float(shape.batch * shape.context) / par.kv_split
        + float(cached_prefix + eff_prefill)
    )
    kv_bytes = kv_tokens * float(2 * kvh * hd) * dt.kv
    hbm_bytes = w_elems * dt.w_dense + kv_bytes
    fp8_f, bf16_f = _add_matmul_flops(float(rows), w_elems, dt.w_dense, dt.act)
    pre_pairs = float(eff_prefill) * (
        float(cached_prefix) + _CAUSAL_HALF * float(eff_prefill * par.pcp)
    )
    scores = (
        (float(decode_rows * shape.context) + pre_pairs)
        * _GQA_SCORE_WIDTH_MULT
        * float(hd * nq)
        * n_layers
    )
    if dt.kv <= FP8_BYTES:
        fp8_f += scores
    else:
        bf16_f += scores
    return _StrictExpectation(hbm_bytes=hbm_bytes, fp8_flops=fp8_f, bf16_flops=bf16_f)


def _ref_swa(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> _StrictExpectation:
    """Reference strict oracle for sliding-window attention layers."""
    decode_rows, cached_prefix, eff_prefill, rows = _step_rows(shape, par.pcp)
    tp = par.tp
    n_layers = float(geom.layer_types.count("sliding_attention"))
    nq = geom.num_heads // tp
    kvh = geom.num_kv_heads // tp
    hd = geom.head_dim
    sw_cap = geom.sliding_window if geom.sliding_window > 0 else shape.context
    eff_ctx = min(shape.context, sw_cap) if shape.context > 0 else 0
    w_elems = n_layers * float(
        geom.hidden * (nq * hd) + 2 * geom.hidden * (kvh * hd) + (nq * hd) * geom.hidden
    )
    pre_kv = float(min(cached_prefix, sw_cap) + eff_prefill) if eff_prefill > 0 else 0.0
    kv_tokens = float(shape.batch * eff_ctx) / par.kv_split + pre_kv
    hbm_bytes = (
        w_elems * dt.w_dense + n_layers * kv_tokens * float(2 * kvh * hd) * dt.kv
    )
    fp8_f, bf16_f = _add_matmul_flops(float(rows), w_elems, dt.w_dense, dt.act)
    pre_pairs = (
        _ref_causal_capped_pairs(cached_prefix, eff_prefill * par.pcp, par.pcp, sw_cap)
        if eff_prefill > 0
        else 0.0
    )
    scores = (
        (float(decode_rows * eff_ctx) + pre_pairs)
        * _GQA_SCORE_WIDTH_MULT
        * float(hd * nq)
        * n_layers
    )
    if dt.kv <= FP8_BYTES:
        fp8_f += scores
    else:
        bf16_f += scores
    return _StrictExpectation(hbm_bytes=hbm_bytes, fp8_flops=fp8_f, bf16_flops=bf16_f)


def _ref_mla_weights(
    geom: ModelGeometry,
    tp: int,
    *,
    is_sparse: bool,
) -> tuple[float, float, float]:
    """Return (base_w_elems, ukv_w_elems, total_w_elems) for MLA or DSA."""
    layer_tag = "dsa" if is_sparse else "mla"
    n_layers = float(geom.layer_types.count(layer_tag))
    nq = geom.num_heads // tp
    kv_lora = geom.kv_lora_rank
    nope_d = geom.qk_nope_head_dim
    rope_d = geom.qk_rope_head_dim
    v_d = geom.v_head_dim or nope_d
    q_w = (
        geom.hidden * geom.q_lora_rank + geom.q_lora_rank * nq * (nope_d + rope_d)
        if geom.q_lora_rank > 0
        else geom.hidden * nq * (nope_d + rope_d)
    )
    kv_a_w = geom.hidden * (kv_lora + rope_d)
    ukv_w_layer = float(kv_lora * nq * (nope_d + v_d))
    o_w = float((nq * v_d) * geom.hidden)
    idx_w = 0.0
    if is_sparse and geom.index_n_heads > 0 and geom.index_head_dim > 0:
        idx_heads = max(1, geom.index_n_heads // tp)
        q_in = geom.q_lora_rank if geom.q_lora_rank > 0 else geom.hidden
        idx_w = float(
            q_in * idx_heads * geom.index_head_dim
            + geom.hidden * geom.index_head_dim
            + geom.hidden * idx_heads
        )
    base_w_elems = n_layers * (float(q_w + kv_a_w) + o_w + idx_w)
    ukv_w_elems = n_layers * ukv_w_layer
    return (base_w_elems, ukv_w_elems, base_w_elems + ukv_w_elems)


def _ref_mla_or_dsa(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
    *,
    is_sparse: bool,
) -> _StrictExpectation:
    """Reference strict oracle for MLA (with or without RoPE) and DSA layers."""
    decode_rows, cached_prefix, eff_prefill, rows = _step_rows(shape, par.pcp)
    tp = par.tp
    layer_tag = "dsa" if is_sparse else "mla"
    n_layers = float(geom.layer_types.count(layer_tag))
    nq = geom.num_heads // tp
    kv_lora = geom.kv_lora_rank
    nope_d = geom.qk_nope_head_dim
    rope_d = geom.qk_rope_head_dim
    v_d = geom.v_head_dim or nope_d
    base_w_elems, ukv_w_elems, total_w_elems = _ref_mla_weights(
        geom, tp, is_sparse=is_sparse
    )

    att_ctx = (
        min(shape.context, geom.index_topk)
        if (is_sparse and geom.index_topk > 0 and shape.context > 0)
        else shape.context
    )
    kv_tokens = float(shape.batch * att_ctx) / par.kv_split + float(
        cached_prefix + eff_prefill
    )
    kv_bytes = n_layers * kv_tokens * float(kv_lora + rope_d) * dt.kv
    if is_sparse and geom.index_head_dim > 0 and shape.context > 0:
        idx_tokens = n_layers * float(shape.batch * shape.context) / par.kv_split
        kv_bytes += idx_tokens * float(geom.index_head_dim) * dt.kv
    hbm_bytes = total_w_elems * dt.w_dense + kv_bytes

    fp8_f, bf16_f = _add_matmul_flops(float(rows), base_w_elems, dt.w_dense, dt.act)
    dec_ukv_fp8, dec_ukv_bf16 = _add_matmul_flops(
        float(decode_rows), ukv_w_elems, dt.w_dense, dt.act
    )
    fp8_f += dec_ukv_fp8
    bf16_f += dec_ukv_bf16
    dec_scores = (
        float(decode_rows * att_ctx)
        * _MLA_SCORE_WIDTH_MULT
        * float(nq * (2 * kv_lora + rope_d))
        * n_layers
    )
    if dt.kv <= FP8_BYTES:
        fp8_f += dec_scores
    else:
        bf16_f += dec_scores

    if eff_prefill > 0:
        cap = geom.index_topk if (is_sparse and geom.index_topk > 0) else 0
        pairs = _ref_causal_capped_pairs(
            cached_prefix, eff_prefill * par.pcp, par.pcp, cap
        )
        abs_ukv_fp8, abs_ukv_bf16 = _add_matmul_flops(
            float(eff_prefill), ukv_w_elems, dt.w_dense, dt.act
        )
        abs_scores = (
            pairs
            * _MLA_SCORE_WIDTH_MULT
            * float(nq * (2 * kv_lora + rope_d))
            * n_layers
        )
        abs_fp8 = abs_ukv_fp8 + (abs_scores if dt.kv <= FP8_BYTES else 0.0)
        abs_bf16 = abs_ukv_bf16 + (0.0 if dt.kv <= FP8_BYTES else abs_scores)

        key_tokens = float(cached_prefix) / float(par.pcp) + float(eff_prefill)
        mha_ukv_fp8, mha_ukv_bf16 = _add_matmul_flops(
            key_tokens, ukv_w_elems, dt.w_dense, dt.act
        )
        mha_scores = (
            pairs
            * _MLA_SCORE_WIDTH_MULT
            * float(nq * (nope_d + rope_d + v_d))
            * n_layers
        )
        mha_fp8 = mha_ukv_fp8 + (mha_scores if dt.kv <= FP8_BYTES else 0.0)
        mha_bf16 = mha_ukv_bf16 + (0.0 if dt.kv <= FP8_BYTES else mha_scores)

        tpu7x_fp8_ratio = CHIPS["tpu7x"].fp8_flops / CHIPS["tpu7x"].bf16_flops
        abs_time = abs_fp8 / tpu7x_fp8_ratio + abs_bf16
        mha_time = mha_fp8 / tpu7x_fp8_ratio + mha_bf16
        if mha_time < abs_time:
            fp8_f += mha_fp8
            bf16_f += mha_bf16
        else:
            fp8_f += abs_fp8
            bf16_f += abs_bf16

    return _StrictExpectation(hbm_bytes=hbm_bytes, fp8_flops=fp8_f, bf16_flops=bf16_f)


def _ref_gdn(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> _StrictExpectation:
    """Reference strict oracle for GDN linear attention layers."""
    _, cached_prefix, eff_prefill, rows = _step_rows(shape, par.pcp)
    tp = par.tp
    n_layers = float(geom.layer_types.count("linear_attention"))
    kh = geom.linear_k_heads // tp
    vh = geom.linear_v_heads // tp
    qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
    z_r = vh * geom.linear_v_dim
    dense_elems = n_layers * float(geom.hidden * (qkv_r + z_r) + z_r * geom.hidden)
    aux_elems = n_layers * float(geom.hidden * (2 * vh) + qkv_r * geom.conv_kernel)
    rec_bytes = n_layers * float(vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
    conv_bytes = n_layers * float(qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
    dec_io = float(shape.batch * (1 + shape.state_writes))
    pre_io = float((1 if cached_prefix > 0 else 0) + 1) if eff_prefill > 0 else 0.0
    state_bytes = (dec_io + pre_io) * (rec_bytes + conv_bytes)
    hbm_bytes = dense_elems * dt.w_dense + aux_elems * dt.w_aux + state_bytes
    d_fp8, d_bf16 = _add_matmul_flops(float(rows), dense_elems, dt.w_dense, dt.act)
    a_fp8, a_bf16 = _add_matmul_flops(float(rows), aux_elems, dt.w_aux, dt.act)
    rec_flops = (
        float(rows)
        * _LINEAR_REC_FLOP_MULT
        * float(vh * geom.linear_k_dim * geom.linear_v_dim)
        * n_layers
    )
    return _StrictExpectation(
        hbm_bytes=hbm_bytes,
        fp8_flops=d_fp8 + a_fp8,
        bf16_flops=d_bf16 + a_bf16 + rec_flops,
    )


def _ref_kda(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> _StrictExpectation:
    """Reference strict oracle for KDA linear attention layers."""
    _, cached_prefix, eff_prefill, rows = _step_rows(shape, par.pcp)
    tp = par.tp
    n_layers = float(geom.layer_types.count("kda"))
    kh = (geom.linear_k_heads or geom.num_heads) // tp
    vh = (geom.linear_v_heads or geom.num_heads) // tp
    kd = geom.linear_k_dim or geom.head_dim
    vd = geom.linear_v_dim or geom.head_dim
    proj_r = 2 * kh * kd + 2 * vh * vd
    w_elems = n_layers * float(geom.hidden * proj_r + (vh * vd) * geom.hidden)
    rec_bytes = n_layers * float(vh * kd * vd) * dt.state
    dec_io = float(shape.batch * (1 + shape.state_writes))
    pre_io = float((1 if cached_prefix > 0 else 0) + 1) if eff_prefill > 0 else 0.0
    hbm_bytes = w_elems * dt.w_dense + (dec_io + pre_io) * rec_bytes
    fp8_f, bf16_f = _add_matmul_flops(float(rows), w_elems, dt.w_dense, dt.act)
    rec_flops = float(rows) * _LINEAR_REC_FLOP_MULT * float(vh * kd * vd) * n_layers
    return _StrictExpectation(
        hbm_bytes=hbm_bytes, fp8_flops=fp8_f, bf16_flops=bf16_f + rec_flops
    )


def _ref_moe(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> _StrictExpectation:
    """Reference strict oracle for routed MoE with optional latent projection."""
    decode_rows, _, eff_prefill, rows = _step_rows(shape, par.pcp)
    n_layers = float(geom.num_layers)
    tp = par.tp
    is_ep = par.moe == "ep" and par.ep > 1
    ep_ranks = par.ep if is_ep else tp
    pcp_per_ep = min(par.pcp, max(1, ep_ranks // tp)) if is_ep else 1
    dp_per_ep = min(par.dp, max(1, ep_ranks // (tp * pcp_per_ep))) if is_ep else 1
    global_rows = (
        (decode_rows + eff_prefill * pcp_per_ep) * dp_per_ep if is_ep else rows
    )
    ed_min = (
        float(math.ceil(min(geom.num_experts, geom.top_k) / ep_ranks))
        if is_ep
        else float(min(geom.num_experts, geom.top_k))
    )
    expert_hidden = (
        geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
    )
    width = (
        geom.moe_inter // max(1, tp // ep_ranks)
        if (is_ep and ep_ranks < tp)
        else (geom.moe_inter if is_ep else geom.moe_inter // tp)
    )
    routed_unit = n_layers * float(_GLU_MATMUL_COUNT * expert_hidden * width)
    shared_w = n_layers * float(
        _GLU_MATMUL_COUNT * geom.hidden * (geom.shared_inter // tp)
    )
    router_w = n_layers * float(geom.hidden * geom.num_experts)
    latent_w = (
        n_layers * float(_LATENT_MATMUL_COUNT * geom.hidden * expert_hidden) / tp
        if geom.routed_expert_hidden > 0
        else 0.0
    )
    hbm_bytes = (
        ed_min * routed_unit * dt.w_expert
        + (shared_w + latent_w) * dt.w_dense
        + router_w * dt.router
    )
    eff_div = float(max(tp * pcp_per_ep, ep_ranks) if is_ep else tp)
    routed_rows = global_rows if is_ep else rows
    routed_act_elems = (
        n_layers
        * float(geom.top_k * _GLU_MATMUL_COUNT * expert_hidden)
        * (float(geom.moe_inter) / eff_div)
    )
    lhs_floor = (
        CHIPS["tpu7x"].mxu_lhs_rows_fp8
        if _is_fp8_matmul(dt.w_expert, dt.act)
        else CHIPS["tpu7x"].mxu_lhs_rows_bf16
    )
    pad_strict = max(1.0, float(lhs_floor) / float(routed_rows))
    r_fp8, r_bf16 = _add_matmul_flops(
        float(routed_rows) * pad_strict, routed_act_elems, dt.w_expert, dt.act
    )
    l_fp8, l_bf16 = _add_matmul_flops(float(rows), latent_w, dt.w_dense, dt.act)
    s_fp8, s_bf16 = _add_matmul_flops(float(rows), shared_w, dt.w_dense, dt.act)
    g_fp8, g_bf16 = _add_matmul_flops(float(rows), router_w, dt.router, dt.act)
    return _StrictExpectation(
        hbm_bytes=hbm_bytes,
        fp8_flops=r_fp8 + l_fp8 + s_fp8 + g_fp8,
        bf16_flops=r_bf16 + l_bf16 + s_bf16 + g_bf16,
    )


def test_reference_calculator_matches_per_layer_family() -> None:
    """Independent reference oracle matches strict bytes and FLOPs per layer family."""
    cases: tuple[
        tuple[str, str, ModelGeometry, Parallel, StepShape, Dtypes, _StrictExpectation],
        ...,
    ] = (
        (
            "gqa_fp8kv",
            "gqa",
            geom_gqa := replace(
                _BASE_GEOMETRY,
                hidden=4096,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
            ),
            par_gqa := Parallel(tp=4),
            shape_gqa := StepShape(
                batch=2,
                rows_per_seq=1,
                context=1024,
                prefill_tokens=128,
                prefix_hit_rate=0.5,
            ),
            dt_gqa := Dtypes(w_dense=1.0, kv=1.0, act="a8"),
            _ref_gqa(geom_gqa, par_gqa, shape_gqa, dt_gqa),
        ),
        (
            "swa_pcp_cap",
            "swa",
            geom_swa := replace(
                _BASE_GEOMETRY,
                layer_types=("sliding_attention", "sliding_attention"),
                sliding_window=128,
            ),
            par_swa := Parallel(tp=2, pcp=2),
            shape_swa := StepShape(
                batch=1,
                rows_per_seq=1,
                context=1024,
                prefill_tokens=512,
                prefix_hit_rate=0.5,
            ),
            dt_swa := Dtypes(w_dense=2.0, kv=1.0, act="a16"),
            _ref_swa(geom_swa, par_swa, shape_swa, dt_swa),
        ),
        (
            "mla_with_rope",
            "mla",
            geom_mla := replace(
                _BASE_GEOMETRY,
                layer_types=("mla", "mla"),
                num_heads=32,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            par_mla := Parallel(tp=4),
            shape_mla := StepShape(
                batch=1,
                rows_per_seq=1,
                context=1024,
                prefill_tokens=512,
                prefix_hit_rate=0.0,
            ),
            dt_mla := Dtypes(w_dense=1.0, kv=1.0, act="a8"),
            _ref_mla_or_dsa(geom_mla, par_mla, shape_mla, dt_mla, is_sparse=False),
        ),
        (
            "mla_no_rope",
            "mla",
            geom_mla_nr := replace(
                _BASE_GEOMETRY,
                layer_types=("mla", "mla"),
                num_heads=16,
                q_lora_rank=0,
                kv_lora_rank=256,
                qk_nope_head_dim=128,
                qk_rope_head_dim=0,
                v_head_dim=128,
            ),
            par_mla_nr := Parallel(tp=2),
            shape_mla_nr := StepShape(
                batch=2,
                rows_per_seq=1,
                context=512,
                prefill_tokens=64,
                prefix_hit_rate=0.0,
            ),
            dt_mla_nr := Dtypes(w_dense=2.0, kv=2.0, act="a16"),
            _ref_mla_or_dsa(
                geom_mla_nr, par_mla_nr, shape_mla_nr, dt_mla_nr, is_sparse=False
            ),
        ),
        (
            "dsa_sparse",
            "dsa",
            geom_dsa := replace(
                _BASE_GEOMETRY,
                layer_types=("dsa", "dsa"),
                q_lora_rank=1024,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                index_n_heads=8,
                index_head_dim=64,
                index_topk=256,
            ),
            par_dsa := Parallel(tp=4),
            shape_dsa := StepShape(
                batch=2,
                rows_per_seq=1,
                context=2048,
                prefill_tokens=1024,
                prefix_hit_rate=0.5,
            ),
            dt_dsa := Dtypes(w_dense=1.0, kv=1.0, act="a8"),
            _ref_mla_or_dsa(geom_dsa, par_dsa, shape_dsa, dt_dsa, is_sparse=True),
        ),
        (
            "gdn_mixed",
            "gdn",
            geom_gdn := replace(
                _BASE_GEOMETRY,
                layer_types=("linear_attention", "linear_attention"),
                linear_k_heads=8,
                linear_v_heads=32,
                linear_k_dim=128,
                linear_v_dim=128,
                conv_kernel=4,
            ),
            par_gdn := Parallel(tp=4),
            shape_gdn := StepShape(
                batch=2,
                rows_per_seq=1,
                context=512,
                state_writes=1,
                prefill_tokens=128,
                prefix_hit_rate=0.5,
            ),
            dt_gdn := Dtypes(w_dense=1.0, w_aux=2.0, state=4.0, conv=2.0, act="a8"),
            _ref_gdn(geom_gdn, par_gdn, shape_gdn, dt_gdn),
        ),
        (
            "kda_prefill_unhit",
            "kda",
            geom_kda := replace(
                _BASE_GEOMETRY,
                layer_types=("kda", "kda"),
                linear_k_heads=16,
                linear_v_heads=16,
                linear_k_dim=128,
                linear_v_dim=128,
            ),
            par_kda := Parallel(tp=4),
            shape_kda := StepShape(
                batch=0,
                rows_per_seq=0,
                context=512,
                state_writes=0,
                prefill_tokens=64,
                prefix_hit_rate=0.0,
            ),
            dt_kda := Dtypes(w_dense=1.0, state=2.0, act="a8"),
            _ref_kda(geom_kda, par_kda, shape_kda, dt_kda),
        ),
        (
            "latent_routed_moe",
            "moe",
            geom_moe := replace(
                _BASE_GEOMETRY,
                num_experts=16,
                top_k=4,
                moe_inter=512,
                shared_inter=512,
                dense_inter=0,
                first_k_dense_replace=0,
                routed_expert_hidden=256,
            ),
            par_moe := Parallel(tp=2, moe="ep", ep=4),
            shape_moe := StepShape(batch=4, rows_per_seq=1, context=512),
            dt_moe := Dtypes(w_dense=2.0, w_expert=1.0, router=2.0, act="a8"),
            _ref_moe(geom_moe, par_moe, shape_moe, dt_moe),
        ),
    )
    for label, mod_name, geom, par, shape, dt, exp in cases:
        mods = {m.name: m for m in pass_costs(geom, par, shape, dt)}
        mod = mods[mod_name]
        assert mod.strict_hbm_bytes == pytest.approx(exp.hbm_bytes, rel=_REL_TOL), label
        assert mod.strict_fp8_flops == pytest.approx(exp.fp8_flops, rel=_REL_TOL), label
        assert mod.strict_bf16_flops == pytest.approx(exp.bf16_flops, rel=_REL_TOL), (
            label
        )
