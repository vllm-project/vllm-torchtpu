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
"""Per-module HBM byte, matmul FLOP, and collective payload equations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .collectives import Algo, Collective, make, padded_rows
from .geometry import ModelGeometry


@dataclass(frozen=True)
class Dtypes:
    """Element widths in bytes for each tensor role and activation peak."""

    w_dense: float = 1.0
    w_expert: float = 1.0
    w_aux: float = 2.0
    kv: float = 1.0
    state: float = 4.0
    conv: float = 2.0
    router: float = 2.0
    lm_head: float = 2.0
    act: Literal["a8", "a16"] = "a8"


@dataclass(frozen=True)
class Parallel:
    """Per-sublayer parallelism, collective layout, and rank placement."""

    tp: int
    kv_split: int = 1
    kv_merge: Literal["ag_rs", "exchange"] = "ag_rs"
    moe: Literal["tp_slice", "ep"] = "tp_slice"
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    residual: Literal["replicated", "sp"] = "replicated"
    ar_algo: Algo = "one_shot"
    ranks_per_chip: int = 2
    dp: int = 1
    ep: int = 1
    pp: int = 1
    pcp: int = 1
    mesh_axes: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.pcp > 1 and self.kv_split > 1:
            raise ValueError(
                f"Parallel.pcp={self.pcp} and Parallel.kv_split={self.kv_split} "
                "are mutually exclusive in one forward pass"
            )


@dataclass(frozen=True)
class StepShape:
    """Per-step batch size, token rows per sequence, context, and prefill tokens."""

    batch: int = 1
    rows_per_seq: int = 1
    context: int = 0
    state_writes: int = 1
    prefill_tokens: int = 0
    prefix_hit_rate: float = 0.0


@dataclass(frozen=True)
class ModuleCost:
    """Per-TensorCore resource work for one module class across one pass."""

    name: str
    hbm_bytes: float
    fp8_flops: float
    bf16_flops: float
    collectives: tuple[Collective, ...] = ()
    hbm_bytes_min: float | None = None

    @property
    def ici_bytes(self) -> float:
        return sum(c.wire_bytes * c.offchip for c in self.collectives)

    @property
    def n_collectives(self) -> int:
        return len(self.collectives)

    @property
    def chain_hops(self) -> int:
        return sum(c.hops for c in self.collectives)

    @property
    def strict_hbm_bytes(self) -> float:
        return self.hbm_bytes if self.hbm_bytes_min is None else self.hbm_bytes_min


def experts_touched(num_experts: int, top_k: int, rows: int) -> float:
    """Expected distinct routed experts touched under uniform routing."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0:
        return 0.0
    if top_k >= num_experts:
        return float(num_experts)
    return float(num_experts) * (1.0 - (1.0 - top_k / num_experts) ** rows)


def ep_busiest_experts(num_experts: int, top_k: int, rows: int, ranks: int) -> float:
    """Expected distinct routed experts on the busiest rank under uniform EP."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0 or ranks <= 0:
        return 0.0
    if ranks == 1:
        return experts_touched(num_experts, top_k, rows)
    n = num_experts // ranks
    if n <= 0:
        return float(math.ceil(min(num_experts, top_k) / ranks))
    p = 1.0 if top_k >= num_experts else 1.0 - (1.0 - top_k / num_experts) ** rows
    cdf = 0.0
    acc = 0.0
    for x in range(n):
        cdf += math.comb(n, x) * (p**x) * ((1.0 - p) ** (n - x))
        acc += 1.0 - min(1.0, cdf) ** ranks
    return acc


def _split_matmul_flops(
    rows: int,
    weight_elems: float,
    weight_bytes: float,
    act: str = "a8",
) -> tuple[float, float]:
    """Route quantized (<2 B) matmuls with a8 activations to FP8 peak, else bf16."""
    flops = 2.0 * rows * weight_elems
    if weight_bytes < 2.0 and act == "a8":
        return (flops, 0.0)
    return (0.0, flops)


def _embed_collectives(rows: int, hidden: int, par: Parallel) -> tuple[Collective, ...]:
    if par.tp <= 1 or hidden <= 0:
        return ()
    return (
        make(
            "embed_ar",
            "ar",
            float(rows * hidden * 2),
            par.tp,
            par.ar_algo,
            par.ranks_per_chip,
        ),
    )


def _attn_layer_collectives(
    batch: int,
    rows_per_seq: int,
    hidden: int,
    nq: int,
    head_dim: int,
    par: Parallel,
    *,
    is_full_attention: bool,
    extra_rows: int = 0,
) -> tuple[Collective, ...]:
    rows = batch * rows_per_seq + extra_rows
    mp = padded_rows(rows, par.tp, par.residual)
    s_bytes = float(mp * hidden * 2)
    colls: list[Collective] = []
    if par.tp > 1:
        if par.residual == "replicated":
            colls.append(
                make(
                    "out_ar",
                    "ar",
                    s_bytes,
                    par.tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
        else:
            colls.append(
                make(
                    "entry_ag",
                    "ag",
                    s_bytes,
                    par.tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "out_rs",
                    "rs",
                    s_bytes,
                    par.tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
    if is_full_attention and par.kv_split > 1:
        g = par.kv_split
        dec_rows = batch * rows_per_seq
        if par.kv_merge == "ag_rs":
            colls.append(
                make(
                    "dcp_q_ag",
                    "ag",
                    float(dec_rows * g * nq * head_dim * 2),
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "dcp_lse_ag",
                    "ag",
                    float(g * dec_rows * g * nq * 4),
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "dcp_out_rs",
                    "rs",
                    float(dec_rows * g * nq * head_dim * 4),
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
        else:
            per_peer = float((batch + 1) * rows_per_seq * nq * head_dim * 4)
            colls.append(
                make(
                    "merge_xchg",
                    "push",
                    (g - 1) * per_peer,
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
    if is_full_attention and par.pcp > 1 and extra_rows > 0:
        colls.append(
            make(
                "pcp_kv_ag",
                "ag",
                float(extra_rows * par.pcp * max(1, nq) * head_dim * 2),
                par.pcp,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
    return tuple(colls)


def _moe_layer_collectives(
    rows: int,
    hidden: int,
    num_experts: int,
    top_k: int,
    par: Parallel,
) -> tuple[Collective, ...]:
    ep_ranks = par.ep if par.ep > 1 else par.tp
    if par.tp <= 1 and ep_ranks <= 1:
        return ()
    tp = par.tp
    mp = padded_rows(rows, max(1, tp), par.residual)
    s_bytes = float(mp * hidden * 2)
    colls: list[Collective] = []

    if par.moe_dispatch == "ag_ep" or (
        par.moe == "ep" and ep_ranks > tp and par.moe_dispatch == "allreduce"
    ):
        global_rows = rows * max(1, par.dp)
        per_rank_rows = max(1, math.ceil(global_rows / max(1, ep_ranks)))
        colls.append(
            make(
                "dispatch_ag",
                "ag",
                float(global_rows * hidden),
                ep_ranks,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
        colls.append(
            make(
                "route_ag",
                "ag",
                float(ep_ranks * per_rank_rows * num_experts * 2),
                ep_ranks,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
        colls.append(
            make(
                "restore_ag",
                "ag",
                float(global_rows * hidden * 2),
                ep_ranks,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
        if par.residual == "replicated" and tp > 1:
            ar_rows = rows
            colls.append(
                make(
                    "moe_ar",
                    "ar",
                    float(ar_rows * hidden * 2),
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
        return tuple(colls)

    if par.moe_dispatch == "fused_rs":
        if tp > 1:
            if par.residual == "replicated":
                colls.append(
                    make(
                        "combine_rs",
                        "rs",
                        s_bytes,
                        tp,
                        par.ar_algo,
                        par.ranks_per_chip,
                    )
                )
                colls.append(
                    make(
                        "restore_ag",
                        "ag",
                        s_bytes,
                        tp,
                        par.ar_algo,
                        par.ranks_per_chip,
                    )
                )
            else:
                colls.append(
                    make(
                        "dispatch_ag",
                        "ag",
                        s_bytes,
                        tp,
                        par.ar_algo,
                        par.ranks_per_chip,
                    )
                )
                colls.append(
                    make(
                        "combine_rs",
                        "rs",
                        s_bytes,
                        tp,
                        par.ar_algo,
                        par.ranks_per_chip,
                    )
                )
        return tuple(colls)

    if tp <= 1 and ep_ranks <= 1:
        return ()
    disp_group = ep_ranks if par.moe_dispatch == "a2a_push" else tp
    push_wire = float(
        math.ceil((rows * max(1, par.dp)) / max(1, disp_group))
        * top_k
        * (disp_group - 1)
        / max(1, disp_group)
        * hidden
    )
    if par.residual == "replicated":
        if par.moe_dispatch == "allreduce":
            colls.append(
                make(
                    "moe_ar",
                    "ar",
                    s_bytes,
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
        else:
            colls.append(
                make(
                    "dispatch_push",
                    "push",
                    push_wire,
                    disp_group,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "combine_push",
                    "push",
                    push_wire,
                    disp_group,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            if tp > 1:
                colls.append(
                    make(
                        "restore_ag",
                        "ag",
                        s_bytes,
                        tp,
                        par.ar_algo,
                        par.ranks_per_chip,
                    )
                )
    else:
        if par.moe_dispatch == "allreduce":
            colls.append(
                make(
                    "dispatch_ag",
                    "ag",
                    s_bytes,
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "logits_ag",
                    "ag",
                    float(mp * num_experts * 2),
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "combine_rs",
                    "rs",
                    s_bytes,
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
        else:
            colls.append(
                make(
                    "dispatch_push",
                    "push",
                    push_wire,
                    disp_group,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "combine_push",
                    "push",
                    push_wire,
                    disp_group,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
    return tuple(colls)


def _lm_head_collectives(
    rows: int, hidden: int, vocab: int, par: Parallel
) -> tuple[Collective, ...]:
    if par.tp <= 1:
        return ()
    tp = par.tp
    mp = padded_rows(rows, tp, par.residual)
    s_bytes = float(mp * hidden * 2)
    colls: list[Collective] = []
    if par.residual == "sp" and hidden > 0:
        colls.append(
            make(
                "tail_ag",
                "ag",
                s_bytes,
                tp,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
    if vocab > 0:
        colls.append(
            make(
                "logits_ag",
                "ag",
                float(rows * vocab * 2),
                tp,
                par.ar_algo,
                par.ranks_per_chip,
            )
        )
    return tuple(colls)


def pass_costs(
    geom: ModelGeometry,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
) -> tuple[ModuleCost, ...]:
    """Compute per-module resource work for one decode, prefill, or mixed pass."""
    tp = par.tp
    if tp < 1:
        raise ValueError(f"Parallel.tp must be >= 1, got {tp}")
    kv_group = (
        max(1, tp // geom.num_kv_heads)
        if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
        else max(1, tp)
    )
    if par.kv_split < 1 or kv_group % par.kv_split != 0:
        raise ValueError(
            f"Parallel.kv_split={par.kv_split} must be >= 1 and divide"
            f" kv_group={kv_group}"
        )
    if par.residual == "sp" and tp == 1:
        raise ValueError("Parallel.residual='sp' requires tp > 1")
    if par.residual == "sp" and par.moe != "ep" and geom.num_experts > 0:
        raise ValueError("Parallel.residual='sp' requires Parallel.moe='ep'")
    if par.moe_dispatch in ("a2a_push", "ag_ep") and par.moe != "ep":
        raise ValueError(
            f"Parallel.moe_dispatch={par.moe_dispatch!r} requires Parallel.moe='ep'"
        )

    pp_scale = (
        float(math.ceil(geom.num_layers / max(1, par.pp))) / float(geom.num_layers)
        if geom.num_layers > 0 and par.pp > 1
        else 1.0
    )
    decode_rows = shape.batch * shape.rows_per_seq
    eff_prefill = (
        max(
            1,
            math.ceil(
                shape.prefill_tokens
                * max(0.0, 1.0 - shape.prefix_hit_rate)
                / max(1, par.pcp)
            ),
        )
        if shape.prefill_tokens > 0
        else 0
    )
    cached_prefix = (
        int(round(shape.prefill_tokens * shape.prefix_hit_rate))
        if shape.prefill_tokens > 0
        else 0
    )
    rows = decode_rows + eff_prefill

    n_gqa = 0
    n_mla = 0
    n_dsa = 0
    n_swa = 0
    n_gdn = 0
    n_kda = 0
    for tok in geom.layer_types:
        if tok in ("mla", "mla_attention"):
            n_mla += 1
        elif tok in ("dsa", "dsa_attention", "sparse_mla"):
            n_dsa += 1
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif tok == "gqa":
            n_gqa += 1
        elif tok in ("full_attention", "attention"):
            if geom.kv_lora_rank > 0 and geom.index_topk > 0:
                n_dsa += 1
            elif geom.kv_lora_rank > 0:
                n_mla += 1
            else:
                n_gqa += 1

    out: list[ModuleCost] = [
        ModuleCost(
            name="embed",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=_embed_collectives(rows, geom.hidden, par),
        )
    ]

    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
    q_mult = 2 if geom.attn_output_gate else 1
    q_out = geom.num_heads * geom.head_dim * q_mult

    # 1. GQA full attention
    if n_gqa > 0:
        n_eff = n_gqa * pp_scale
        att_w_elems = float(
            n_eff
            * (
                geom.hidden * (q_out // tp)
                + 2 * geom.hidden * kvh * geom.head_dim
                + (geom.num_heads // tp) * geom.head_dim * geom.hidden
            )
        )
        kv_dec_tokens = float(shape.batch * shape.context) / par.kv_split
        kv_pre_tokens = float(cached_prefix + eff_prefill)
        kv_bytes = (
            float(kvh * geom.head_dim * 2)
            * n_eff
            * (kv_dec_tokens + kv_pre_tokens)
            * dt.kv
        )
        att_fp8, att_bf16 = _split_matmul_flops(rows, att_w_elems, dt.w_dense, dt.act)
        dec_scores = float(
            decode_rows
            * 4
            * shape.context
            * geom.head_dim
            * (geom.num_heads // tp)
            * n_eff
        )
        pre_ctx_eff = cached_prefix + 0.5 * eff_prefill * par.pcp
        pre_scores = float(
            eff_prefill
            * 4
            * pre_ctx_eff
            * geom.head_dim
            * (geom.num_heads // tp)
            * n_eff
        )
        n_colls_layers = max(1, round(n_eff))
        att_colls = (
            _attn_layer_collectives(
                shape.batch,
                shape.rows_per_seq,
                geom.hidden,
                geom.num_heads // tp,
                geom.head_dim,
                par,
                is_full_attention=True,
                extra_rows=eff_prefill,
            )
            * n_colls_layers
        )
        out.append(
            ModuleCost(
                name="gqa",
                hbm_bytes=att_w_elems * dt.w_dense + kv_bytes,
                fp8_flops=att_fp8,
                bf16_flops=att_bf16 + dec_scores + pre_scores,
                collectives=att_colls,
            )
        )

    # 2. Sliding-window attention (SWA)
    if n_swa > 0:
        n_eff = n_swa * pp_scale
        sw_cap = geom.sliding_window if geom.sliding_window > 0 else shape.context
        eff_ctx = min(shape.context, sw_cap) if shape.context > 0 else 0
        swa_w_elems = float(
            n_eff
            * (
                geom.hidden * (q_out // tp)
                + 2 * geom.hidden * kvh * geom.head_dim
                + nq * geom.head_dim * geom.hidden
            )
        )
        kv_tokens = float(shape.batch * eff_ctx) / par.kv_split + float(
            min(cached_prefix + eff_prefill, sw_cap) if eff_prefill > 0 else 0
        )
        swa_kv_bytes = float(kvh * geom.head_dim * 2) * n_eff * kv_tokens * dt.kv
        swa_fp8, swa_bf16 = _split_matmul_flops(rows, swa_w_elems, dt.w_dense, dt.act)
        swa_scores = float(
            (decode_rows * eff_ctx + eff_prefill * min(sw_cap, max(1, eff_prefill)))
            * 4
            * geom.head_dim
            * nq
            * n_eff
        )
        swa_colls = _attn_layer_collectives(
            shape.batch,
            shape.rows_per_seq,
            geom.hidden,
            nq,
            geom.head_dim,
            par,
            is_full_attention=True,
            extra_rows=eff_prefill,
        ) * max(1, round(n_eff))
        out.append(
            ModuleCost(
                name="swa",
                hbm_bytes=swa_w_elems * dt.w_dense + swa_kv_bytes,
                fp8_flops=swa_fp8,
                bf16_flops=swa_bf16 + swa_scores,
                collectives=swa_colls,
            )
        )

    # 3 & 4. Multi-head Latent Attention (MLA) and Dynamic Sparse Attention (DSA)
    for fam_name, n_fam, is_sparse in (("mla", n_mla, False), ("dsa", n_dsa, True)):
        if n_fam <= 0:
            continue
        n_eff = n_fam * pp_scale
        if geom.kv_lora_rank <= 0:
            raise ValueError(
                f"{fam_name.upper()} layer requires config field 'kv_lora_rank'"
            )
        if geom.qk_nope_head_dim <= 0:
            raise ValueError(
                f"{fam_name.upper()} layer requires config field 'qk_nope_head_dim'"
            )
        if geom.qk_rope_head_dim <= 0:
            raise ValueError(
                f"{fam_name.upper()} layer requires config field 'qk_rope_head_dim'"
            )
        kv_lora = geom.kv_lora_rank
        nope_d = geom.qk_nope_head_dim
        rope_d = geom.qk_rope_head_dim
        v_d = geom.v_head_dim or nope_d
        if geom.q_lora_rank > 0:
            q_w = float(
                geom.hidden * geom.q_lora_rank
                + geom.q_lora_rank * nq * (nope_d + rope_d)
            )
        else:
            q_w = float(geom.hidden * nq * (nope_d + rope_d))
        mla_w_layer = (
            q_w
            + float(geom.hidden * (kv_lora + rope_d))
            + float(kv_lora * nq * (nope_d + v_d) + nq * v_d * geom.hidden)
        )
        if is_sparse and geom.index_n_heads > 0 and geom.index_head_dim > 0:
            mla_w_layer += float(
                geom.hidden * max(1, geom.index_n_heads // tp) * geom.index_head_dim
            )
        mla_w_elems = n_eff * mla_w_layer
        att_ctx = (
            min(shape.context, geom.index_topk)
            if (is_sparse and geom.index_topk > 0 and shape.context > 0)
            else shape.context
        )
        latent_width = kv_lora + rope_d
        kv_tokens = float(shape.batch * att_ctx) / par.kv_split + float(
            cached_prefix + eff_prefill
        )
        mla_kv_bytes = n_eff * kv_tokens * float(latent_width) * dt.kv
        if is_sparse and geom.index_head_dim > 0 and shape.context > 0:
            mla_kv_bytes += (
                n_eff
                * float(shape.batch * shape.context * geom.index_head_dim)
                * dt.kv
                / par.kv_split
            )
        mla_fp8, mla_bf16 = _split_matmul_flops(rows, mla_w_elems, dt.w_dense, dt.act)
        mla_scores = float(
            (
                decode_rows * att_ctx
                + eff_prefill * (cached_prefix + 0.5 * eff_prefill * par.pcp)
            )
            * 2
            * nq
            * (kv_lora + rope_d + kv_lora)
            * n_eff
        )
        mla_colls = _attn_layer_collectives(
            shape.batch,
            shape.rows_per_seq,
            geom.hidden,
            nq,
            kv_lora,
            par,
            is_full_attention=True,
            extra_rows=eff_prefill,
        ) * max(1, round(n_eff))
        out.append(
            ModuleCost(
                name=fam_name,
                hbm_bytes=mla_w_elems * dt.w_dense + mla_kv_bytes,
                fp8_flops=mla_fp8,
                bf16_flops=mla_bf16 + mla_scores,
                collectives=mla_colls,
            )
        )

    # 5. Gated DeltaNet (GDN) linear attention
    if n_gdn > 0:
        n_eff = n_gdn * pp_scale
        kh = max(1, geom.linear_k_heads // tp) if geom.linear_k_heads > 0 else 0
        vh = max(1, geom.linear_v_heads // tp) if geom.linear_v_heads > 0 else 0
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        z_r = vh * geom.linear_v_dim
        gdn_dense_elems = float(
            n_eff * (geom.hidden * (qkv_r + z_r) + z_r * geom.hidden)
        )
        gdn_aux_elems = float(
            n_eff * (geom.hidden * (2 * vh) + qkv_r * geom.conv_kernel)
        )
        rec_state = float(n_eff * vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
        conv_state = float(n_eff * qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
        seq_count = shape.batch + (1 if eff_prefill > 0 and shape.batch == 0 else 0)
        sw_count = (
            shape.state_writes if shape.batch > 0 else (1 if eff_prefill > 0 else 0)
        )
        gdn_state_bytes = float(seq_count * (1 + sw_count)) * (rec_state + conv_state)
        dense_fp8, dense_bf16 = _split_matmul_flops(
            rows, gdn_dense_elems, dt.w_dense, dt.act
        )
        aux_fp8, aux_bf16 = _split_matmul_flops(rows, gdn_aux_elems, dt.w_aux, dt.act)
        gdn_rec_flops = float(
            rows * 4 * vh * geom.linear_k_dim * geom.linear_v_dim * n_eff
        )
        gdn_colls = _attn_layer_collectives(
            shape.batch,
            shape.rows_per_seq,
            geom.hidden,
            geom.num_heads // tp,
            geom.head_dim,
            par,
            is_full_attention=False,
            extra_rows=eff_prefill,
        ) * max(1, round(n_eff))
        out.append(
            ModuleCost(
                name="gdn",
                hbm_bytes=(
                    gdn_dense_elems * dt.w_dense
                    + gdn_aux_elems * dt.w_aux
                    + gdn_state_bytes
                ),
                fp8_flops=dense_fp8 + aux_fp8,
                bf16_flops=dense_bf16 + aux_bf16 + gdn_rec_flops,
                collectives=gdn_colls,
            )
        )

    # 6. Kimi Delta Attention (KDA)
    if n_kda > 0:
        n_eff = n_kda * pp_scale
        kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        proj_r = 2 * kh * kd + 2 * vh * vd
        kda_w_elems = float(n_eff * (geom.hidden * proj_r + (vh * vd) * geom.hidden))
        kda_rec = float(n_eff * vh * kd * vd) * dt.state
        seq_count = shape.batch + (1 if eff_prefill > 0 and shape.batch == 0 else 0)
        sw_count = (
            shape.state_writes if shape.batch > 0 else (1 if eff_prefill > 0 else 0)
        )
        kda_state_bytes = float(seq_count * (1 + sw_count)) * kda_rec
        kda_fp8, kda_bf16 = _split_matmul_flops(rows, kda_w_elems, dt.w_dense, dt.act)
        kda_rec_flops = float(rows * 4 * vh * kd * vd * n_eff)
        kda_colls = _attn_layer_collectives(
            shape.batch,
            shape.rows_per_seq,
            geom.hidden,
            nq,
            vd,
            par,
            is_full_attention=False,
            extra_rows=eff_prefill,
        ) * max(1, round(n_eff))
        out.append(
            ModuleCost(
                name="kda",
                hbm_bytes=kda_w_elems * dt.w_dense + kda_state_bytes,
                fp8_flops=kda_fp8,
                bf16_flops=kda_bf16 + kda_rec_flops,
                collectives=kda_colls,
            )
        )

    # 7 & 8. Dense MLP and Routed MoE FFN layers
    if geom.num_experts > 0:
        n_dense_ffn = min(geom.num_layers, geom.first_k_dense_replace)
        n_moe_ffn = geom.num_layers - n_dense_ffn
        if n_dense_ffn > 0 and geom.dense_inter > 0:
            n_d_eff = n_dense_ffn * pp_scale
            mlp_inter = geom.dense_inter // tp
            mlp_elems = float(n_d_eff * 3 * geom.hidden * mlp_inter)
            mlp_fp8, mlp_bf16 = _split_matmul_flops(rows, mlp_elems, dt.w_dense, dt.act)
            d_par = Parallel(
                tp=par.tp,
                residual=par.residual,
                ar_algo=par.ar_algo,
                ranks_per_chip=par.ranks_per_chip,
                moe="ep" if par.residual == "sp" else "tp_slice",
            )
            mlp_colls = _moe_layer_collectives(rows, geom.hidden, 0, 0, d_par) * max(
                1, round(n_d_eff)
            )
            out.append(
                ModuleCost(
                    name="mlp",
                    hbm_bytes=mlp_elems * dt.w_dense,
                    fp8_flops=mlp_fp8,
                    bf16_flops=mlp_bf16,
                    collectives=mlp_colls,
                )
            )
        if n_moe_ffn > 0:
            n_m_eff = n_moe_ffn * pp_scale
            ep_ranks = par.ep if par.ep > 1 else tp
            pcp_per_ep = (
                min(max(1, par.pcp), max(1, ep_ranks // max(1, tp)))
                if par.ep > 1
                else 1
            )
            dp_per_ep = (
                min(max(1, par.dp), max(1, ep_ranks // max(1, tp * pcp_per_ep)))
                if par.ep > 1
                else 1
            )
            global_rows = (
                (decode_rows + eff_prefill * pcp_per_ep) * dp_per_ep
                if par.ep > 1
                else rows
            )
            if par.moe == "ep":
                ed = ep_busiest_experts(
                    geom.num_experts, geom.top_k, global_rows, ep_ranks
                )
                ed_min = (
                    float(math.ceil(geom.top_k / ep_ranks)) if global_rows > 0 else 0.0
                )
                width = (
                    geom.moe_inter // max(1, tp // max(1, ep_ranks))
                    if ep_ranks < tp
                    else geom.moe_inter
                )
            else:
                ed = experts_touched(geom.num_experts, geom.top_k, rows)
                ed_min = float(min(geom.num_experts, geom.top_k)) if rows > 0 else 0.0
                width = geom.moe_inter // tp
            routed_unit = float(n_m_eff * 3 * geom.hidden * width)
            routed_elems = ed * routed_unit
            routed_elems_min = ed_min * routed_unit
            isr_tp = max(1, geom.shared_inter // tp) if geom.shared_inter > 0 else 0
            isr_eff = geom.shared_inter if par.residual == "sp" else isr_tp
            shared_elems = float(n_m_eff * 3 * geom.hidden * isr_eff)
            router_elems = float(n_m_eff * geom.hidden * geom.num_experts)
            moe_w_bytes = (
                routed_elems * dt.w_expert
                + shared_elems * dt.w_dense
                + router_elems * dt.router
            )
            moe_w_bytes_min = (
                routed_elems_min * dt.w_expert
                + shared_elems * dt.w_dense
                + router_elems * dt.router
            )
            eff_moe_div = max(tp * pcp_per_ep, ep_ranks) if par.ep > 1 else tp
            routed_act_elems = float(
                n_m_eff * geom.top_k * 3 * geom.hidden * (geom.moe_inter / eff_moe_div)
            )
            routed_rows = global_rows if par.ep > 1 else rows
            sh_rows = math.ceil(rows / tp) if par.residual == "sp" else rows
            r_fp8, r_bf16 = _split_matmul_flops(
                routed_rows, routed_act_elems, dt.w_expert, dt.act
            )
            s_fp8, s_bf16 = _split_matmul_flops(
                sh_rows, shared_elems, dt.w_dense, dt.act
            )
            rt_fp8, rt_bf16 = _split_matmul_flops(rows, router_elems, dt.router, dt.act)
            moe_colls = _moe_layer_collectives(
                rows, geom.hidden, geom.num_experts, geom.top_k, par
            ) * max(1, round(n_m_eff))
            out.append(
                ModuleCost(
                    name="moe",
                    hbm_bytes=moe_w_bytes,
                    fp8_flops=r_fp8 + s_fp8 + rt_fp8,
                    bf16_flops=r_bf16 + s_bf16 + rt_bf16,
                    collectives=moe_colls,
                    hbm_bytes_min=moe_w_bytes_min,
                )
            )
    else:
        n_d_eff = geom.num_layers * pp_scale
        mlp_inter = geom.dense_inter // tp
        mlp_elems = float(n_d_eff * 3 * geom.hidden * mlp_inter)
        mlp_fp8, mlp_bf16 = _split_matmul_flops(rows, mlp_elems, dt.w_dense, dt.act)
        mlp_colls = _moe_layer_collectives(rows, geom.hidden, 0, 0, par) * max(
            1, round(n_d_eff)
        )
        out.append(
            ModuleCost(
                name="mlp",
                hbm_bytes=mlp_elems * dt.w_dense,
                fp8_flops=mlp_fp8,
                bf16_flops=mlp_bf16,
                collectives=mlp_colls,
            )
        )

    # 9. Per-Layer Embedding (PLE)
    if "ple" in geom.layer_types or geom.ple_dim > 0:
        if geom.ple_dim <= 0:
            raise ValueError("PLE layer requires config field 'ple_dim'")
        ple_d = geom.ple_dim
        n_ple = geom.num_layers * pp_scale
        ple_proj_elems = float(n_ple * geom.hidden * ple_d // tp)
        ple_lookup_bytes = float(n_ple * rows * ple_d) * dt.w_aux
        ple_fp8, ple_bf16 = _split_matmul_flops(rows, ple_proj_elems, dt.w_aux, dt.act)
        out.append(
            ModuleCost(
                name="ple",
                hbm_bytes=ple_proj_elems * dt.w_aux + ple_lookup_bytes,
                fp8_flops=ple_fp8,
                bf16_flops=ple_bf16,
            )
        )

    # Pipeline stage handoff collective when PP > 1
    pp_colls = (
        (
            make(
                "pp_p2p",
                "p2p",
                float(rows * geom.hidden * 2),
                par.pp,
                par.ar_algo,
                par.ranks_per_chip,
            ),
        )
        if par.pp > 1
        else ()
    )

    lm_elems = float(geom.vocab * geom.hidden // tp) / float(max(1, par.pp))
    lm_fp8, lm_bf16 = _split_matmul_flops(rows, lm_elems, dt.lm_head, dt.act)
    out.append(
        ModuleCost(
            name="lm_head",
            hbm_bytes=lm_elems * dt.lm_head,
            fp8_flops=lm_fp8,
            bf16_flops=lm_bf16,
            collectives=_lm_head_collectives(rows, geom.hidden, geom.vocab, par)
            + pp_colls,
        )
    )
    return tuple(out)
