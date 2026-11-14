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
"""Per-core HBM weight, KV cache, recurrent state, and SMEM page-table sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .constants import (
    BYTES_PER_PAGE_ENTRY,
    GLU_MATMUL_COUNT,
    MAX_CTX_TOKENS_FOR_SMALL_PAGE,
    PAGE_TABLE_BYTE_BUDGET,
    RPA_MAX_PAGE_TOKENS,
    RPA_MIN_PAGE_TOKENS,
    STEP_FLOOR_CACHE_MAXSIZE,
)
from .geometry import LayerCounts, count_layer_types
from .modules import kv_bytes_per_token_per_layer

if TYPE_CHECKING:
    from .geometry import ModelGeometry
    from .modules import Dtypes, Parallel


@dataclass(frozen=True, slots=True)
class HbmBreakdown:
    """Per-TensorCore HBM memory breakdown across weights, KV, and state."""

    weights: float
    kv: float
    state: float
    total: float


@dataclass(frozen=True, slots=True)
class SeqKvLayout:
    """Sequence count, context length, and KV packing layout for HBM sizing."""

    seqs_per_dp: int
    context: int = 0
    physical_kv_layout: bool = True


@dataclass(frozen=True, slots=True)
class KvFitInput:
    """Memory capacity constraints used to compute clamped prefix hit rate."""

    b_ws: int
    ctx_full: int
    hbm_limit_b: float
    w_base_b: float
    b_kv_phys_core: float
    offload_b_per_core: float
    prefix_hit_rate: float
    kv_working_set_turns: float


def _compute_mla_w_attn(
    geom: ModelGeometry,
    dt: Dtypes,
    tp: int,
    pp: int,
    counts: LayerCounts,
) -> float:
    """Compute per-core weight bytes for MLA and DSA attention layers."""
    if geom.kv_lora_rank <= 0:
        msg = "MLA layer requires config field 'kv_lora_rank'"
        raise ValueError(msg)
    if geom.qk_nope_head_dim <= 0:
        msg = "MLA layer requires config field 'qk_nope_head_dim'"
        raise ValueError(msg)
    if geom.qk_rope_head_dim < 0:
        msg = "MLA layer requires config field 'qk_rope_head_dim'"
        raise ValueError(msg)
    kv_lora = geom.kv_lora_rank
    nope_d = geom.qk_nope_head_dim
    rope_d = geom.qk_rope_head_dim
    v_d = geom.v_head_dim or nope_d
    if geom.q_lora_rank > 0:
        q_term_hidden = float(geom.hidden * geom.q_lora_rank) / pp
        q_term_proj = float(geom.q_lora_rank * geom.num_heads * (nope_d + rope_d)) / (
            tp * pp
        )
    else:
        q_term_hidden = 0.0
        q_term_proj = float(geom.hidden * geom.num_heads * (nope_d + rope_d)) / (
            tp * pp
        )
    w_attn = (
        float(counts.mla)
        * (
            q_term_hidden
            + float(geom.hidden * (kv_lora + rope_d)) / pp
            + q_term_proj
            + float(
                kv_lora * geom.num_heads * (nope_d + v_d)
                + geom.num_heads * v_d * geom.hidden
            )
            / (tp * pp)
        )
        * dt.w_dense
    )
    if counts.dsa > 0 and geom.index_n_heads > 0 and geom.index_head_dim > 0:
        w_attn += (
            float(counts.dsa)
            * float(
                geom.hidden * max(1, geom.index_n_heads // tp) * geom.index_head_dim
            )
            * dt.w_dense
            / pp
        )
    return w_attn


def compute_w_attn(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core attention projection and QKV weight bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    q_mult = 2 if geom.attn_output_gate else 1
    kv_div = min(tp, max(1, geom.num_kv_heads))
    w_attn = 0.0
    gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    for n_layers, hd in ((counts.gqa, gqa_hd), (counts.swa, geom.head_dim)):
        if n_layers <= 0:
            continue
        w_q_o = (
            float(n_layers)
            * float(
                geom.hidden * geom.num_heads * hd * q_mult
                + geom.num_heads * hd * geom.hidden
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv = (
            float(n_layers)
            * float(2 * geom.hidden * geom.num_kv_heads * hd)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o + w_kv
    if counts.mla > 0:
        w_attn += _compute_mla_w_attn(geom, dt, tp, pp, counts)
    return w_attn


def compute_w_rec(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core recurrent weight bytes for linear attention layers."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_rec = 0.0
    if counts.gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        z_r = vh * geom.linear_v_dim
        w_rec += (
            float(counts.gdn)
            * (
                float(geom.hidden * (qkv_r + z_r) + z_r * geom.hidden) * dt.w_dense
                + float(geom.hidden * 2 * vh + qkv_r * geom.conv_kernel) * dt.w_aux
            )
            / pp
        )
    if counts.kda > 0:
        kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        w_rec += (
            float(counts.kda)
            * float(geom.hidden * (2 * kh * kd + 2 * vh * vd) + vh * vd * geom.hidden)
            * dt.w_dense
            / pp
        )
    return w_rec


def compute_w_ffn(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
) -> tuple[float, float]:
    """Compute per-core dense and routed MoE FFN weight bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    ep = (
        parallel.ep
        if (parallel.moe == "ep" and parallel.ep > 1)
        else (tp if parallel.moe == "ep" else 1)
    )
    w_ffn = 0.0
    w_dense_ffn = 0.0
    n_moe_ffn, n_dense_ffn = geom.ffn_layer_counts()
    if n_dense_ffn > 0 and geom.dense_inter > 0:
        w_dense_ffn = (
            float(n_dense_ffn * GLU_MATMUL_COUNT * geom.hidden * geom.dense_inter)
            * dt.w_dense
            / (tp * pp)
        )
        w_ffn += w_dense_ffn
    if n_moe_ffn > 0:
        routed_div = ep * pp if parallel.moe == "ep" else tp * pp
        expert_hidden = (
            geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
        )
        w_ffn += (
            float(
                n_moe_ffn
                * geom.num_experts
                * GLU_MATMUL_COUNT
                * expert_hidden
                * geom.moe_inter
            )
            * dt.w_expert
            / routed_div
        )
        if geom.routed_expert_hidden > 0:
            w_ffn += (
                float(n_moe_ffn * 2 * geom.hidden * geom.routed_expert_hidden)
                * dt.w_dense
                / (tp * pp)
            )
        if geom.shared_inter > 0:
            isr_eff = (
                geom.shared_inter
                if parallel.residual == "sp"
                else max(1, geom.shared_inter // tp)
            )
            w_ffn += (
                float(n_moe_ffn * GLU_MATMUL_COUNT * geom.hidden * isr_eff)
                * dt.w_dense
                / pp
            )
        w_ffn += float(n_moe_ffn * geom.hidden * geom.num_experts) * dt.router / pp
    return w_ffn, w_dense_ffn


def _compute_w_mtp(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    w_attn_data: tuple[float, int],
    w_ffn_terms: tuple[float, float],
) -> float:
    """Compute per-core weight bytes for multi-token prediction layers."""
    if geom.mtp_layers <= 0:
        return 0.0
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_attn, n_attn_total = w_attn_data
    w_ffn, w_dense_ffn = w_ffn_terms
    mtp_fc = float(2 * geom.hidden * geom.hidden) * dt.w_aux / tp
    mtp_att = (w_attn / float(max(1, n_attn_total))) * float(pp)
    if geom.mtp_inter > 0:
        mtp_ffn = (
            float(GLU_MATMUL_COUNT * geom.hidden * geom.mtp_inter) * dt.w_dense / tp
        )
    elif geom.num_experts > 0:
        n_moe_ffn, _ = geom.ffn_layer_counts()
        n_moe_layers = max(1, n_moe_ffn)
        mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_layers)) * float(pp)
    else:
        mtp_ffn = (w_ffn / float(max(1, geom.num_layers))) * float(pp)
    return float(geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn) / float(pp)


def _compute_kv_and_state(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    counts: LayerCounts,
) -> tuple[float, float]:
    """Compute (kv_bytes, state_bytes) active on one core."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    if counts.mla > 0:
        if geom.kv_lora_rank <= 0:
            msg = "MLA layer requires config field 'kv_lora_rank'"
            raise ValueError(msg)
        if geom.qk_rope_head_dim < 0:
            msg = "MLA layer requires config field 'qk_rope_head_dim'"
            raise ValueError(msg)
    gqa_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "gqa", physical=layout.physical_kv_layout
    )
    swa_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "swa", physical=layout.physical_kv_layout
    )
    mla_kind = "dsa" if geom.index_topk > 0 else "mla"
    mla_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, mla_kind, physical=layout.physical_kv_layout
    )
    kv_div = parallel.kv_split * parallel.pcp * pp
    kv_gqa = (
        float(layout.seqs_per_dp * layout.context) * counts.eff_gqa * gqa_kv_b_per_tok
    )
    eff_swa_ctx = (
        min(layout.context, geom.sliding_window)
        if geom.sliding_window > 0
        else layout.context
    )
    kv_swa = float(layout.seqs_per_dp * eff_swa_ctx) * counts.eff_swa * swa_kv_b_per_tok
    kv_mla = (
        float(layout.seqs_per_dp * layout.context) * counts.eff_mla * mla_kv_b_per_tok
    )
    kv_bytes = (kv_gqa + kv_swa + kv_mla) / kv_div

    st_bytes = 0.0
    if counts.gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        st_bytes += (
            float(layout.seqs_per_dp * counts.gdn)
            * (
                float(vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
                + float(qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
            )
            / pp
        )
    if counts.kda > 0:
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        st_bytes += (
            float(layout.seqs_per_dp * counts.kda * vh * kd * vd) * dt.state / pp
        )
    return kv_bytes, st_bytes


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def hbm_breakdown_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
) -> HbmBreakdown:
    """Compute per-core static weight, active KV cache, and state byte totals."""
    counts = count_layer_types(geom)
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_attn = compute_w_attn(geom, dt, parallel, counts)
    w_rec = compute_w_rec(geom, dt, parallel, counts)
    w_ffn, w_dense_ffn = compute_w_ffn(geom, dt, parallel)
    w_head = float(2 * geom.vocab * geom.hidden) * dt.lm_head / (tp * pp)
    w_mtp = _compute_w_mtp(
        geom,
        dt,
        parallel,
        (w_attn, counts.attn_total),
        (w_ffn, w_dense_ffn),
    )
    w_hc = (
        float(geom.num_layers * (geom.hc_mult**2) * geom.hidden)
        * dt.w_dense
        / float(tp * pp)
        if geom.hc_mult > 1
        else 0.0
    )
    weights = w_attn + w_rec + w_ffn + w_head + w_mtp + w_hc
    kv, state = _compute_kv_and_state(geom, dt, parallel, layout, counts)
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=state,
        total=weights + kv + state,
    )


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
) -> float:
    """Compute total HBM footprint (weights + KV + state) in bytes on one core."""
    return hbm_breakdown_per_core_bytes(geom, dt, parallel, layout).total


def default_rpa_page_size(ctx_tokens: int) -> int:
    """Determine default RPA page size based on sequence token budget."""
    if ctx_tokens > MAX_CTX_TOKENS_FOR_SMALL_PAGE:
        return RPA_MIN_PAGE_TOKENS
    pow2 = 1 << (max(1, ctx_tokens) - 1).bit_length()
    return min(RPA_MAX_PAGE_TOKENS, max(RPA_MIN_PAGE_TOKENS, pow2 // 16))


def smem_max_seqs(ctx_tokens: int, kv_split: int, page_size: int) -> int:
    """Estimate maximum concurrent sequences fitting inside SMEM page tables."""
    pages_per_req = max(
        1,
        math.ceil((ctx_tokens / max(1, kv_split)) / max(1, page_size)),
    )
    return PAGE_TABLE_BYTE_BUDGET // pages_per_req // BYTES_PER_PAGE_ENTRY


def kv_capacity_and_hit_rate(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    b_res: int,
    kv_fit: KvFitInput,
) -> tuple[float, float]:
    """Compute physical KV token capacity and clamped prefix cache hit rate."""
    st_b = (
        hbm_footprint_per_core_bytes(geom, dtypes, parallel, SeqKvLayout(b_res, 0))
        - kv_fit.w_base_b
    )
    cap_tok = (
        (
            max(0.0, kv_fit.hbm_limit_b - kv_fit.w_base_b - st_b)
            + kv_fit.offload_b_per_core
        )
        / kv_fit.b_kv_phys_core
        if kv_fit.b_kv_phys_core > 0.0
        else math.inf
    )
    w_tok = max(0.0, kv_fit.kv_working_set_turns) * float(kv_fit.b_ws * kv_fit.ctx_full)
    h_eff = (
        kv_fit.prefix_hit_rate * min(1.0, cap_tok / max(1.0, w_tok))
        if kv_fit.prefix_hit_rate > 0.0 and w_tok > 0.0 and kv_fit.b_kv_phys_core > 0.0
        else kv_fit.prefix_hit_rate
    )
    return cap_tok, h_eff
