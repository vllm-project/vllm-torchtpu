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
"""Per-core HBM weight, KV cache, and recurrent state sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict, Unpack

from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    FP8_BYTES,
    STEP_FLOOR_CACHE_MAXSIZE,
)
from .geometry import (
    HostTableResidency,
    LayerCounts,
    count_layer_types,
    draft_layer_geometry,
    host_table_hbm_fraction,
)
from .modules import (
    Dtypes,
    Parallel,
    csa2_layer_weight_elems,
    draft_kv_dtypes,
    draft_parallel_for_geometry,
    dsa_indexer_layer_weight_elems,
    host_table_elem_bytes,
    kv_bytes_per_token_per_layer,
    kv_heads_on_rank,
    role_dtype_bytes,
    sparse_gqa_index_layer_weight_elems,
)

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .geometry import ModelGeometry

_HC_LOWRANK_DIRECTIONS: int = 2


@dataclass(frozen=True, slots=True)
class ParamBreakdown:
    """Unsharded model parameter count breakdown matching HBM weight terms."""

    attention: float
    linear_attention: float
    dense_ffn: float
    moe_routed_ffn: float
    moe_shared_ffn: float
    router: float
    embedding: float
    lm_head: float
    mtp: float
    hc: float
    total_params: float
    active_params_per_token: float
    host_tables: float = 0.0


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
    physical_kv_layout: bool = False
    host_table_residency: HostTableResidency = "auto"
    kv_offload_enabled: bool = False


@dataclass(frozen=True, slots=True)
class KvFitInput:
    """Memory capacity constraints used to compute clamped prefix hit rate."""

    working_set_batch: int
    ctx_full: int
    hbm_limit_b: float
    w_base_b: float
    b_kv_phys_core: float
    offload_b_per_core: float
    prefix_hit_rate: float
    kv_working_set_turns: float
    fixed_kv_per_seq_bytes: float = 0.0


def pp_stage_layer_scale(num_layers: int, pp: int) -> float:
    """Return heaviest pipeline stage layer fraction ceil(L / pp) / L."""
    if pp <= 1 or num_layers <= 0:
        return 1.0
    return float(math.ceil(num_layers / pp)) / float(num_layers)


def _compute_mla_w_attn(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    tp: int,
    pp: int,
    counts: LayerCounts,
) -> float:
    """Compute per-core weight bytes for MLA, DSA, and CSA2 attention layers."""
    sparse = geometry.sparse_attn
    if sparse is not None and sparse.kind == "csa2":
        scale = pp_stage_layer_scale(geometry.num_layers, pp) * dtypes.w_dense
        w_attn = float(counts.mla) * csa2_layer_weight_elems(geometry, tp) * scale
        n_idx = len(sparse.index_source_layer_ids)
        if n_idx > 0 and geometry.index_n_heads > 0 and geometry.index_head_dim > 0:
            w_attn += float(n_idx) * dsa_indexer_layer_weight_elems(geometry) * scale
        return w_attn
    if geometry.kv_lora_rank <= 0:
        msg = "MLA layer requires config field 'kv_lora_rank'"
        raise ValueError(msg)
    if geometry.qk_nope_head_dim <= 0:
        msg = "MLA layer requires config field 'qk_nope_head_dim'"
        raise ValueError(msg)
    if geometry.qk_rope_head_dim < 0:
        msg = "MLA layer requires config field 'qk_rope_head_dim'"
        raise ValueError(msg)
    kv_lora = geometry.kv_lora_rank
    nope_d = geometry.qk_nope_head_dim
    rope_d = geometry.qk_rope_head_dim
    v_head_dim = geometry.v_head_dim or nope_d
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    if geometry.q_lora_rank > 0:
        q_term_hidden = float(geometry.hidden * geometry.q_lora_rank) * stage_scale
        q_term_proj = (
            float(geometry.q_lora_rank * geometry.num_heads * (nope_d + rope_d))
            * stage_scale
            / tp
        )
    else:
        q_term_hidden = 0.0
        q_term_proj = (
            float(geometry.hidden * geometry.num_heads * (nope_d + rope_d))
            * stage_scale
            / tp
        )
    w_attn = (
        float(counts.mla)
        * (
            q_term_hidden
            + float(geometry.hidden * (kv_lora + rope_d)) * stage_scale
            + q_term_proj
            + float(
                kv_lora * geometry.num_heads * (nope_d + v_head_dim)
                + geometry.num_heads * v_head_dim * geometry.hidden
            )
            * stage_scale
            / tp
        )
        * dtypes.w_dense
    )
    if counts.dsa > 0:
        n_full_idx = (
            geometry.indexer_types.count("full")
            if geometry.indexer_types
            else counts.dsa
        )
        w_attn += (
            float(n_full_idx)
            * dsa_indexer_layer_weight_elems(geometry)
            * dtypes.w_dense
            * stage_scale
        )
    return w_attn


def compute_w_attn(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core attention projection and QKV weight bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    q_mult = 2 if geometry.attn_output_gate else 1
    num_kv_heads = kv_heads_on_rank(geometry, parallel)
    w_attn = 0.0
    gqa_hd = (
        geometry.global_head_dim if geometry.global_head_dim > 0 else geometry.head_dim
    )
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    for n_layers, hd in ((counts.gqa, gqa_hd), (counts.swa, geometry.head_dim)):
        if n_layers <= 0:
            continue
        w_q_o = (
            float(n_layers)
            * float(
                geometry.hidden * geometry.num_heads * hd * q_mult
                + geometry.num_heads * hd * geometry.hidden
            )
            * dtypes.w_dense
            * stage_scale
            / tp
        )
        w_kv = (
            float(n_layers)
            * float(2 * geometry.hidden * num_kv_heads * hd)
            * dtypes.w_dense
            * stage_scale
        )
        w_attn += w_q_o + w_kv
    sparse = geometry.sparse_attn
    if counts.gqa > 0 and sparse is not None and sparse.kind in ("msa", "qsa"):
        n_idx = len(sparse.index_source_layer_ids)
        w_attn += (
            float(n_idx)
            * sparse_gqa_index_layer_weight_elems(geometry, tp)
            * dtypes.w_dense
            * stage_scale
        )
    if counts.mla > 0:
        w_attn += _compute_mla_w_attn(geometry, dtypes, tp, pp, counts)
    return w_attn


def compute_w_rec(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core recurrent weight bytes for linear attention layers."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_rec = 0.0
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    if counts.gdn > 0:
        key_heads = max(1, geometry.linear_k_heads // tp)
        value_heads = max(1, geometry.linear_v_heads // tp)
        qkv_rank_width = (
            2 * key_heads * geometry.linear_k_dim + value_heads * geometry.linear_v_dim
        )
        z_rank_width = value_heads * geometry.linear_v_dim
        w_rec += (
            float(counts.gdn)
            * (
                float(
                    geometry.hidden * (qkv_rank_width + z_rank_width)
                    + z_rank_width * geometry.hidden
                )
                * dtypes.w_dense
                + float(
                    geometry.hidden * 2 * value_heads
                    + qkv_rank_width * geometry.conv_kernel
                )
                * dtypes.w_aux
            )
            * stage_scale
        )
    if counts.kda > 0:
        key_heads = max(1, (geometry.linear_k_heads or geometry.num_heads) // tp)
        value_heads = max(1, (geometry.linear_v_heads or geometry.num_heads) // tp)
        k_head_dim = geometry.linear_k_dim or geometry.head_dim
        v_head_dim = geometry.linear_v_dim or geometry.head_dim
        w_rec += (
            float(counts.kda)
            * float(
                geometry.hidden
                * (2 * key_heads * k_head_dim + 2 * value_heads * v_head_dim)
                + value_heads * v_head_dim * geometry.hidden
            )
            * dtypes.w_dense
            * stage_scale
        )
    return w_rec


def _compute_ffn_component_bytes(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
) -> tuple[float, float, float, float]:
    """Compute per-core (dense_ffn, moe_routed_ffn, moe_shared_ffn, router) bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    is_ep = parallel.moe == "ep" and parallel.ep > 1
    ep_ranks = parallel.ep if is_ep else tp
    w_dense_ffn = 0.0
    w_routed_ffn = 0.0
    w_shared_ffn = 0.0
    w_router = 0.0
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    n_moe_ffn, n_dense_ffn = geometry.ffn_layer_counts()
    if n_dense_ffn > 0 and geometry.dense_inter > 0:
        w_dense_ffn = (
            float(
                n_dense_ffn
                * geometry.ffn_num_matrices
                * geometry.hidden
                * geometry.dense_inter
            )
            * dtypes.w_dense
            * stage_scale
            / tp
        )
    if n_moe_ffn > 0:
        expert_div = ep_ranks if is_ep else 1
        tp_shard_div = (
            max(1, tp // ep_ranks)
            if (is_ep and ep_ranks < tp)
            else (1 if is_ep else tp)
        )
        expert_width = (
            float(geometry.moe_inter) / float(tp_shard_div)
            if tp_shard_div > 1
            else float(geometry.moe_inter)
        )
        expert_hidden = (
            geometry.routed_expert_hidden
            if geometry.routed_expert_hidden > 0
            else geometry.hidden
        )
        w_routed_ffn = (
            float(
                n_moe_ffn
                * geometry.num_experts
                * geometry.ffn_num_matrices
                * expert_hidden
                * expert_width
            )
            * dtypes.w_expert
            * stage_scale
            / expert_div
        )
        if geometry.routed_expert_hidden > 0:
            w_routed_ffn += (
                float(n_moe_ffn * 2 * geometry.hidden * geometry.routed_expert_hidden)
                * dtypes.w_dense
                * stage_scale
                / tp
            )
        if geometry.shared_inter > 0:
            shared_rank_width = (
                float(geometry.shared_inter)
                if parallel.residual == "sp"
                else float(geometry.shared_inter) / float(tp)
            )
            w_shared_ffn = (
                float(n_moe_ffn * geometry.ffn_num_matrices * geometry.hidden)
                * shared_rank_width
                * dtypes.w_dense
                * stage_scale
            )
        w_router = (
            float(n_moe_ffn * geometry.hidden * geometry.num_experts)
            * dtypes.router
            * stage_scale
        )
    return w_dense_ffn, w_routed_ffn, w_shared_ffn, w_router


def compute_w_ffn(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
) -> tuple[float, float]:
    """Compute per-core dense and routed MoE FFN weight bytes."""
    w_dense_ffn, w_routed_ffn, w_shared_ffn, w_router = _compute_ffn_component_bytes(
        geometry, dtypes, parallel
    )
    return w_dense_ffn + w_routed_ffn + w_shared_ffn + w_router, w_dense_ffn


def _compute_w_mtp(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    w_attn_data: tuple[float, int],
    w_ffn_terms: tuple[float, float],
) -> float:
    """Compute per-core weight bytes for MTP or DraftSpec layers."""
    if geometry.draft is not None:
        draft = geometry.draft
        draft_geom = draft_layer_geometry(geometry)
        draft_counts = count_layer_types(draft_geom)
        par1 = replace(parallel, pp=1)
        tp = max(1, parallel.tp)
        w_ffn_draft, _ = compute_w_ffn(draft_geom, dtypes, par1)
        fc_layers = 1 if draft.kind == "dspark" else draft.num_layers
        fc_div = 1.0 if draft.fc_replicated else float(tp)
        w_fc = (
            float(fc_layers * draft.fc_in_mult * geometry.hidden * geometry.hidden)
            * dtypes.w_aux
            / fc_div
        )
        w_markov = (
            float(2 * geometry.vocab * draft.markov_rank) * dtypes.lm_head / float(tp)
            if draft.markov_rank > 0
            else 0.0
        )
        return (
            compute_w_attn(draft_geom, dtypes, par1, draft_counts)
            + compute_w_rec(draft_geom, dtypes, par1, draft_counts)
            + w_ffn_draft
            + _compute_w_hc(draft_geom, dtypes, tp, 1)
            + w_fc
            + w_markov
        )
    if geometry.mtp_layers <= 0:
        return 0.0
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    unscale = 1.0 / pp_stage_layer_scale(geometry.num_layers, pp)
    w_attn, n_attn_total = w_attn_data
    w_ffn, w_dense_ffn = w_ffn_terms
    mtp_fc = float(2 * geometry.hidden * geometry.hidden) * dtypes.w_aux / tp
    if n_attn_total > 0:
        mtp_att = (w_attn / float(n_attn_total)) * unscale
    else:
        draft_layer_type = (
            "dsa"
            if (geometry.kv_lora_rank > 0 and geometry.index_topk > 0)
            else ("mla" if geometry.kv_lora_rank > 0 else "full_attention")
        )
        draft_attn_geom = replace(
            geometry,
            num_layers=1,
            layer_types=(draft_layer_type,),
            indexer_types=(
                ("shared",)
                if (
                    draft_layer_type == "dsa" and geometry.index_share_for_mtp_iteration
                )
                else ()
            ),
        )
        draft_counts = count_layer_types(draft_attn_geom)
        mtp_att = compute_w_attn(
            draft_attn_geom, dtypes, replace(parallel, pp=1), draft_counts
        )
    n_moe_ffn, n_dense_ffn = geometry.ffn_layer_counts()
    if geometry.mtp_inter > 0:
        mtp_ffn = (
            float(geometry.ffn_num_matrices * geometry.hidden * geometry.mtp_inter)
            * dtypes.w_dense
            / tp
        )
    elif geometry.num_experts > 0:
        if n_moe_ffn > 0:
            mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_ffn)) * unscale
        elif geometry.moe_inter > 0:
            draft_moe_geom = replace(
                geometry,
                num_layers=1,
                first_k_dense_replace=0,
                ffn_kinds=("moe",),
                n_moe_layers=None,
                n_dense_layers=None,
            )
            mtp_ffn, _ = compute_w_ffn(draft_moe_geom, dtypes, replace(parallel, pp=1))
        elif n_dense_ffn > 0:
            mtp_ffn = (w_dense_ffn / float(n_dense_ffn)) * unscale
        else:
            mtp_ffn = 0.0
    elif n_dense_ffn > 0:
        mtp_ffn = (w_dense_ffn / float(n_dense_ffn)) * unscale
    else:
        mtp_ffn = (w_ffn / float(max(1, geometry.num_layers))) * unscale
    return float(geometry.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn)


def _effective_dsa_kv_layers(
    geometry: ModelGeometry,
    counts: LayerCounts,
) -> tuple[float, float]:
    """Split effective MLA layers into latent-only and full-indexer DSA counts."""
    plain_eff_mla = max(0.0, counts.eff_mla - counts.eff_dsa)
    if counts.dsa > 0 and geometry.indexer_types:
        full_frac = float(geometry.indexer_types.count("full")) / float(counts.dsa)
        full_dsa = counts.eff_dsa * full_frac
        return plain_eff_mla + (counts.eff_dsa - full_dsa), full_dsa
    return plain_eff_mla, counts.eff_dsa


def offloaded_host_table_params(
    geometry: ModelGeometry, residency: HostTableResidency = "auto"
) -> float:
    """Return host-table parameter count excluded from TensorCore HBM weights."""
    return sum(
        float(spec.elem_count) * (1.0 - host_table_hbm_fraction(spec, idx, residency))
        for idx, spec in enumerate(geometry.host_tables)
    )


def _table_pp_scale(n_tbl: int, pp: int, fallback: float) -> float:
    """Return PP stage fraction for a table attached to ``n_tbl`` layers."""
    return (
        float(math.ceil(n_tbl / pp)) / float(n_tbl)
        if (pp > 1 and n_tbl > 0)
        else fallback
    )


def _compute_w_host_tables(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    residency: HostTableResidency,
    chip: ChipSpec | None = None,
) -> float:
    """Compute per-core HBM weight bytes for host-table storage and projections."""
    if not geometry.host_tables:
        return 0.0
    tp, pp = max(1, parallel.tp), max(1, parallel.pp)
    host_c = (
        chip.chips_per_host * chip.cores_per_chip
        if chip is not None
        else DEFAULT_CHIPS_PER_HOST * DEFAULT_CORES_PER_CHIP
    )
    pool_c = max(1, tp * max(1, parallel.dp) * max(1, parallel.pcp), parallel.ep)
    dp_div = float(min(pool_c, max(1, host_c)))
    base = pp_stage_layer_scale(geometry.num_layers, pp)
    total = 0.0
    for idx, spec in enumerate(geometry.host_tables):
        stage_scale = _table_pp_scale(len(spec.layer_ids), pp, base)
        shard_div = dp_div if spec.shard_axis == "dp_local" else float(tp)
        hbm_frac = host_table_hbm_fraction(spec, idx, residency)
        table_bytes = (
            float(spec.elem_count)
            * host_table_elem_bytes(spec, dtypes)
            * hbm_frac
            / shard_div
        )
        proj_bytes = (
            float(spec.dense_proj_params)
            * role_dtype_bytes(dtypes, spec.dense_proj_dtype_role)
            / float(tp)
        )
        is_first_resident = (
            residency == "first" and spec.resident_on_first is None and idx == 0
        )
        table_scale = 1.0 if is_first_resident else stage_scale
        total += table_bytes * table_scale + proj_bytes * stage_scale
    return total


def _compute_kv_and_state(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> tuple[float, float]:
    """Compute (kv_bytes, state_bytes) active on one core."""
    counts = count_layer_types(geometry)
    tp, pp = max(1, parallel.tp), max(1, parallel.pp)
    sparse = geometry.sparse_attn
    if counts.mla > 0 and not (sparse is not None and sparse.kind == "csa2"):
        if geometry.kv_lora_rank <= 0:
            msg = "MLA layer requires config field 'kv_lora_rank'"
            raise ValueError(msg)
        if geometry.qk_rope_head_dim < 0:
            msg = "MLA layer requires config field 'qk_rope_head_dim'"
            raise ValueError(msg)
    full_kv, fixed_kv = _kv_bytes_per_core_parts_cached(
        geometry,
        dtypes,
        parallel,
        layout.context,
        physical=layout.physical_kv_layout,
        chip=chip,
        kv_offload_enabled=layout.kv_offload_enabled,
    )
    kv_bytes = float(layout.seqs_per_dp) * (float(layout.context) * full_kv + fixed_kv)
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)

    st_bytes = 0.0
    if counts.gdn > 0:
        key_heads = max(1, geometry.linear_k_heads // tp)
        value_heads = max(1, geometry.linear_v_heads // tp)
        qkv_rank_width = (
            2 * key_heads * geometry.linear_k_dim + value_heads * geometry.linear_v_dim
        )
        st_bytes += (
            float(layout.seqs_per_dp * counts.gdn)
            * (
                float(value_heads * geometry.linear_k_dim * geometry.linear_v_dim)
                * dtypes.state
                + float(qkv_rank_width * max(0, geometry.conv_kernel - 1)) * dtypes.conv
            )
            * stage_scale
        )
    if counts.kda > 0:
        value_heads = max(1, (geometry.linear_v_heads or geometry.num_heads) // tp)
        k_head_dim = geometry.linear_k_dim or geometry.head_dim
        v_head_dim = geometry.linear_v_dim or geometry.head_dim
        st_bytes += (
            float(
                layout.seqs_per_dp * counts.kda * value_heads * k_head_dim * v_head_dim
            )
            * dtypes.state
            * stage_scale
        )
    if sparse is not None and sparse.fixed_state_page_bytes > 0.0 and counts.gqa > 0:
        n_qsa = float(len(sparse.kv_source_layer_ids))
        st_bytes += (
            float(layout.seqs_per_dp)
            * n_qsa
            * sparse.fixed_state_page_bytes
            * (dtypes.kv / FP8_BYTES)
            * stage_scale
        )
    for spec in geometry.host_tables:
        if spec.state_bytes_per_req > 0.0:
            sc = _table_pp_scale(len(spec.layer_ids), pp, stage_scale)
            st_bytes += float(layout.seqs_per_dp) * spec.state_bytes_per_req * sc
    return kv_bytes, st_bytes


def _compute_w_head(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    tp: int,
) -> tuple[float, float]:
    """Compute single-stage per-core (embedding, lm_head) weight bytes."""
    emb = float(geometry.vocab * geometry.hidden) * dtypes.lm_head / tp
    lm_head = 0.0 if geometry.tie_word_embeddings else emb
    return emb, lm_head


def _compute_w_hc(geometry: ModelGeometry, dtypes: Dtypes, tp: int, pp: int) -> float:
    """Compute per-core hyper-connection weight bytes."""
    if geometry.hc_mult <= 1:
        return 0.0
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    if geometry.hc_lowrank > 0:
        return (
            float(
                geometry.num_layers
                * _HC_LOWRANK_DIRECTIONS
                * geometry.hidden
                * geometry.hc_lowrank
                * geometry.hc_mult
            )
            * dtypes.w_dense
            * stage_scale
            / float(tp)
        )
    return (
        float(geometry.num_layers * (geometry.hc_mult**2) * geometry.hidden)
        * dtypes.w_dense
        * stage_scale
        / float(tp)
    )


_UNIT_DTYPES = Dtypes(
    w_dense=1.0,
    w_expert=1.0,
    w_aux=1.0,
    kv=1.0,
    state=1.0,
    conv=1.0,
    router=1.0,
    lm_head=1.0,
)
_UNIT_PARALLEL = Parallel(tp=1, pp=1, ep=1, moe="tp_slice")


def param_counts(geometry: ModelGeometry) -> ParamBreakdown:
    """Return unsharded model parameter counts aligned 1:1 with HBM weight bytes.

    Shares the exact per-component element calculations with
    :func:`hbm_breakdown_per_core_bytes`, intentionally omitting normalization
    weights, linear biases, PLE parameters, and KDA auxiliary parameters that
    HBM weight sizing does not model.
    """
    counts = count_layer_types(geometry)
    attention = compute_w_attn(geometry, _UNIT_DTYPES, _UNIT_PARALLEL, counts)
    linear_attention = compute_w_rec(geometry, _UNIT_DTYPES, _UNIT_PARALLEL, counts)
    dense_ffn, moe_routed_ffn, moe_shared_ffn, router = _compute_ffn_component_bytes(
        geometry, _UNIT_DTYPES, _UNIT_PARALLEL
    )
    w_ffn = dense_ffn + moe_routed_ffn + moe_shared_ffn + router
    mtp = _compute_w_mtp(
        geometry,
        _UNIT_DTYPES,
        _UNIT_PARALLEL,
        (attention, counts.attn_total),
        (w_ffn, dense_ffn),
    )
    embedding, lm_head = _compute_w_head(geometry, _UNIT_DTYPES, tp=1)
    hc = _compute_w_hc(geometry, _UNIT_DTYPES, tp=1, pp=1)
    host_tables = _compute_w_host_tables(geometry, _UNIT_DTYPES, _UNIT_PARALLEL, "hbm")
    total_params = (
        attention
        + linear_attention
        + dense_ffn
        + moe_routed_ffn
        + moe_shared_ffn
        + router
        + embedding
        + lm_head
        + mtp
        + hc
        + host_tables
    )
    inactive_routed = sum(float(spec.elem_count) for spec in geometry.host_tables)
    if geometry.num_experts > 0:
        expert_hidden = (
            geometry.routed_expert_hidden
            if geometry.routed_expert_hidden > 0
            else geometry.hidden
        )
        n_moe_ffn, _ = geometry.ffn_layer_counts()
        mtp_moe_layers = (
            geometry.mtp_layers
            if (
                geometry.draft is None
                and n_moe_ffn > 0
                and geometry.mtp_layers > 0
                and geometry.mtp_inter <= 0
            )
            else 0
        )
        inactive_experts = max(0, geometry.num_experts - geometry.top_k)
        inactive_routed += float(
            (n_moe_ffn + mtp_moe_layers)
            * inactive_experts
            * geometry.ffn_num_matrices
            * expert_hidden
            * geometry.moe_inter
        )
    if geometry.draft is not None:
        dg = draft_layer_geometry(geometry)
        if dg.num_experts > 0:
            d_moe, _ = dg.ffn_layer_counts()
            d_h = dg.routed_expert_hidden if dg.routed_expert_hidden > 0 else dg.hidden
            inactive_routed += float(
                d_moe
                * max(0, dg.num_experts - dg.top_k)
                * dg.ffn_num_matrices
                * d_h
                * dg.moe_inter
            )
    return ParamBreakdown(
        attention=attention,
        linear_attention=linear_attention,
        dense_ffn=dense_ffn,
        moe_routed_ffn=moe_routed_ffn,
        moe_shared_ffn=moe_shared_ffn,
        router=router,
        embedding=embedding,
        lm_head=lm_head,
        mtp=mtp,
        hc=hc,
        total_params=total_params,
        active_params_per_token=max(0.0, total_params - inactive_routed),
        host_tables=host_tables,
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def hbm_breakdown_per_core_bytes(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> HbmBreakdown:
    """Compute per-core static weight, active KV cache, and state byte totals."""
    counts = count_layer_types(geometry)
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_attn = compute_w_attn(geometry, dtypes, parallel, counts)
    w_rec = compute_w_rec(geometry, dtypes, parallel, counts)
    w_ffn, w_dense_ffn = compute_w_ffn(geometry, dtypes, parallel)
    w_emb, w_lm_head = _compute_w_head(geometry, dtypes, tp)
    w_mtp = _compute_w_mtp(
        geometry,
        dtypes,
        parallel,
        (w_attn, counts.attn_total),
        (w_ffn, w_dense_ffn),
    )
    # Stage 0 holds embeddings whole; last stage holds lm_head (tied or untied) + MTP.
    last_stage_head = w_emb if geometry.tie_word_embeddings else w_lm_head
    w_head_mtp = (
        w_emb + w_lm_head + w_mtp if pp == 1 else max(w_emb, last_stage_head + w_mtp)
    )
    w_hc = _compute_w_hc(geometry, dtypes, tp, pp)
    w_host_tables = _compute_w_host_tables(
        geometry, dtypes, parallel, layout.host_table_residency, chip=chip
    )
    weights = w_attn + w_rec + w_ffn + w_head_mtp + w_hc + w_host_tables
    kv, state = _compute_kv_and_state(geometry, dtypes, parallel, layout, chip=chip)
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=state,
        total=weights + kv + state,
    )


def hbm_footprint_per_core_bytes(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> float:
    """Compute total HBM footprint (weights + KV + state) in bytes on one core."""
    return hbm_breakdown_per_core_bytes(
        geometry, dtypes, parallel, layout, chip=chip
    ).total


class _KvPartsOptions(TypedDict, total=False):
    """Optional keyword arguments for :func:`kv_bytes_per_core_parts`."""

    physical: bool
    chip: ChipSpec | None
    kv_offload_enabled: bool


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def _kv_bytes_per_core_parts_cached(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    ctx_tokens: int,
    *,
    physical: bool,
    chip: ChipSpec | None,
    kv_offload_enabled: bool = False,
) -> tuple[float, float]:
    """Cached implementation of kv_bytes_per_core_parts over frozen value keys."""
    counts = count_layer_types(geometry)
    pp = max(1, parallel.pp)
    stage_scale = pp_stage_layer_scale(geometry.num_layers, pp)
    kv_div = float(parallel.kv_split * parallel.pcp) / stage_scale
    sparse = geometry.sparse_attn
    is_csa2 = sparse is not None and sparse.kind == "csa2"
    is_sparse_gqa = sparse is not None and sparse.kind in ("msa", "qsa")
    gqa_kv_b = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "gqa", physical=physical, chip=chip
    )
    swa_kv_b = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "swa", physical=physical, chip=chip
    )
    mla_kv_b = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "mla", physical=physical, chip=chip
    )
    dsa_kv_b = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "dsa", physical=physical, chip=chip
    )
    eff_swa_ctx = (
        min(max(0, ctx_tokens), geometry.sliding_window)
        if geometry.sliding_window > 0
        else max(0, ctx_tokens)
    )
    if is_csa2 and sparse is not None:
        n_kv = float(len(sparse.kv_source_layer_ids))
        n_idx = float(len(sparse.index_source_layer_ids))
        idx_b = sparse.index_record_bytes or max(0.0, dsa_kv_b - mla_kv_b)
        linear_part = counts.eff_gqa * gqa_kv_b + n_kv * mla_kv_b + n_idx * idx_b
        extra_fixed = (
            float(eff_swa_ctx * counts.mla) * sparse.swa_bytes_per_tok
            if (sparse.swa_bytes_per_tok > 0.0 and geometry.sliding_window > 0)
            else 0.0
        )
    elif is_sparse_gqa and sparse is not None:
        mla_only_layers, full_dsa_layers = _effective_dsa_kv_layers(geometry, counts)
        main_only_b = kv_bytes_per_token_per_layer(
            replace(geometry, sparse_attn=replace(sparse, index_source_layer_ids=())),
            dtypes,
            parallel,
            "gqa",
            physical=physical,
            chip=chip,
        )
        idx_b = max(0.0, gqa_kv_b - main_only_b)
        n_idx = float(len(sparse.index_source_layer_ids))
        linear_part = (
            counts.eff_gqa * main_only_b
            + n_idx * idx_b
            + mla_only_layers * mla_kv_b
            + full_dsa_layers * dsa_kv_b
        )
        extra_fixed = 0.0
    else:
        mla_only_layers, full_dsa_layers = _effective_dsa_kv_layers(geometry, counts)
        linear_part = (
            counts.eff_gqa * gqa_kv_b
            + mla_only_layers * mla_kv_b
            + full_dsa_layers * dsa_kv_b
        )
        extra_fixed = 0.0
    if geometry.sliding_window > 0:
        full_kv_per_tok = linear_part / kv_div
        fixed_kv_per_seq = (
            float(eff_swa_ctx) * counts.eff_swa * swa_kv_b + extra_fixed
        ) / kv_div
    else:
        full_kv_per_tok = (linear_part + counts.eff_swa * swa_kv_b) / kv_div
        fixed_kv_per_seq = 0.0
    draft = geometry.draft
    if draft is not None and not (
        draft.skip_draft_kv_on_offload and kv_offload_enabled
    ):
        draft_geom = draft_layer_geometry(geometry)
        d_lin, d_fix = _kv_bytes_per_core_parts_cached(
            draft_geom,
            draft_kv_dtypes(geometry, dtypes),
            draft_parallel_for_geometry(draft_geom, parallel),
            ctx_tokens,
            physical=physical,
            chip=chip,
        )
        full_kv_per_tok += d_lin
        fixed_kv_per_seq += d_fix
    return full_kv_per_tok, fixed_kv_per_seq


def kv_bytes_per_core_parts(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    ctx_tokens: int,
    **options: Unpack[_KvPartsOptions],
) -> tuple[float, float]:
    """Return (full_context_kv_per_token_bytes, fixed_window_kv_per_seq_bytes)."""
    physical = options.get("physical", False)
    chip = options.get("chip")
    kv_offload_enabled = options.get("kv_offload_enabled", False)
    return _kv_bytes_per_core_parts_cached(
        geometry,
        dtypes,
        parallel,
        ctx_tokens,
        physical=physical,
        chip=chip,
        kv_offload_enabled=kv_offload_enabled,
    )


class _KvCapacityOptions(TypedDict, total=False):
    """Optional keyword arguments for :func:`kv_capacity_and_hit_rate`."""

    fixed_kv_per_seq_bytes: float | None
    chip: ChipSpec | None


def kv_capacity_and_hit_rate(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    resident_batch: int,
    kv_fit: KvFitInput,
    **options: Unpack[_KvCapacityOptions],
) -> tuple[float, float]:
    """Compute KV token capacity and clamped prefix cache hit rate."""
    fixed_kv_per_seq_bytes = options.get("fixed_kv_per_seq_bytes")
    chip = options.get("chip")
    state_bytes = hbm_breakdown_per_core_bytes(
        geometry, dtypes, parallel, SeqKvLayout(resident_batch, 0), chip=chip
    ).state
    fixed_seq_bytes = max(
        0.0,
        kv_fit.fixed_kv_per_seq_bytes
        if fixed_kv_per_seq_bytes is None
        else fixed_kv_per_seq_bytes,
    )
    fixed_resident_bytes = float(max(0, resident_batch)) * fixed_seq_bytes
    available_bytes = (
        max(
            0.0,
            kv_fit.hbm_limit_b - kv_fit.w_base_b - state_bytes - fixed_resident_bytes,
        )
        + kv_fit.offload_b_per_core
    )
    kv_bytes_per_tok = kv_fit.b_kv_phys_core
    capacity_tokens = (
        available_bytes / kv_bytes_per_tok if kv_bytes_per_tok > 0.0 else math.inf
    )
    working_set_tokens = max(0.0, kv_fit.kv_working_set_turns) * float(
        kv_fit.working_set_batch * kv_fit.ctx_full
    )
    if (
        kv_fit.prefix_hit_rate > 0.0
        and working_set_tokens > 0.0
        and kv_bytes_per_tok > 0.0
    ):
        eff_hit_rate = kv_fit.prefix_hit_rate * min(
            1.0, capacity_tokens / max(1.0, working_set_tokens)
        )
    elif (
        kv_fit.prefix_hit_rate > 0.0
        and kv_bytes_per_tok <= 0.0
        and fixed_seq_bytes > 0.0
    ):
        working_set_seqs = max(0.0, kv_fit.kv_working_set_turns) * float(
            max(0, kv_fit.working_set_batch)
        )
        available_no_fixed = (
            max(0.0, kv_fit.hbm_limit_b - kv_fit.w_base_b - state_bytes)
            + kv_fit.offload_b_per_core
        )
        capacity_seqs = available_no_fixed / fixed_seq_bytes
        eff_hit_rate = (
            kv_fit.prefix_hit_rate
            * min(1.0, capacity_seqs / max(1.0, working_set_seqs))
            if working_set_seqs > 0.0
            else kv_fit.prefix_hit_rate
        )
    else:
        eff_hit_rate = kv_fit.prefix_hit_rate
    return capacity_tokens, eff_hit_rate
