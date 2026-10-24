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
"""Per-module HBM byte, MXU FLOP, and collective payload equations."""

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
    moe_dispatch: Literal["allreduce", "a2a_push"] = "allreduce"
    residual: Literal["replicated", "sp"] = "replicated"
    ar_algo: Algo = "one_shot"
    ranks_per_chip: int = 2


@dataclass(frozen=True)
class StepShape:
    """Per-step batch size, token rows per sequence, context, and state writes."""

    batch: int = 1
    rows_per_seq: int = 1
    context: int = 0
    state_writes: int = 1


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
        return 0.0
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
    """Route quantized (<=1 B) matmuls with a8 activations to FP8 peak, else bf16."""
    flops = 2.0 * rows * weight_elems
    if weight_bytes <= 1.0 and act == "a8":
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
) -> tuple[Collective, ...]:
    rows = batch * rows_per_seq
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
        if par.kv_merge == "ag_rs":
            colls.append(
                make(
                    "dcp_q_ag",
                    "ag",
                    float(rows * g * nq * head_dim * 2),
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "dcp_lse_ag",
                    "ag",
                    float(g * rows * g * nq * 4),
                    g,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "dcp_out_rs",
                    "rs",
                    float(rows * g * nq * head_dim * 4),
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
    return tuple(colls)


def _moe_layer_collectives(
    rows: int,
    hidden: int,
    num_experts: int,
    top_k: int,
    par: Parallel,
) -> tuple[Collective, ...]:
    if par.tp <= 1:
        return ()
    tp = par.tp
    mp = padded_rows(rows, tp, par.residual)
    s_bytes = float(mp * hidden * 2)
    push_wire = float(math.ceil(rows / tp) * top_k * (tp - 1) / tp * hidden)
    colls: list[Collective] = []
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
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "combine_push",
                    "push",
                    push_wire,
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
                    tp,
                    par.ar_algo,
                    par.ranks_per_chip,
                )
            )
            colls.append(
                make(
                    "combine_push",
                    "push",
                    push_wire,
                    tp,
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
    """Compute per-module resource work for one forward pass."""
    tp = par.tp
    if tp < 1:
        raise ValueError(f"Parallel.tp must be >= 1, got {tp}")
    kv_group = max(1, tp // geom.num_kv_heads) if geom.num_kv_heads > 0 else 1
    if par.kv_split < 1 or kv_group % par.kv_split != 0:
        raise ValueError(
            f"Parallel.kv_split={par.kv_split} must be >= 1 and divide"
            f" kv_group={kv_group}"
        )
    if par.residual == "sp" and tp == 1:
        raise ValueError("Parallel.residual='sp' requires tp > 1")
    if par.residual == "sp" and par.moe != "ep":
        raise ValueError("Parallel.residual='sp' requires Parallel.moe='ep'")
    if par.moe_dispatch == "a2a_push" and par.moe != "ep":
        raise ValueError("Parallel.moe_dispatch='a2a_push' requires Parallel.moe='ep'")

    rows = shape.batch * shape.rows_per_seq
    n_att = sum(1 for t in geom.layer_types if t == "full_attention")
    n_gdn = sum(1 for t in geom.layer_types if t == "linear_attention")

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
    nq = geom.num_heads // tp
    q_mult = 2 if geom.attn_output_gate else 1
    q_out = geom.num_heads * geom.head_dim * q_mult
    att_w_elems = float(
        n_att
        * (
            geom.hidden * (q_out // tp)
            + 2 * geom.hidden * kvh * geom.head_dim
            + nq * geom.head_dim * geom.hidden
        )
    )
    kv_bytes = (
        float(shape.batch * kvh * geom.head_dim * 2 * n_att * shape.context)
        * dt.kv
        / par.kv_split
    )
    att_fp8, att_bf16 = _split_matmul_flops(rows, att_w_elems, dt.w_dense, dt.act)
    att_scores = float(rows * 4 * shape.context * geom.head_dim * nq * n_att)
    if n_att > 0:
        att_colls = (
            _attn_layer_collectives(
                shape.batch,
                shape.rows_per_seq,
                geom.hidden,
                nq,
                geom.head_dim,
                par,
                is_full_attention=True,
            )
            * n_att
        )
        out.append(
            ModuleCost(
                name="gqa",
                hbm_bytes=att_w_elems * dt.w_dense + kv_bytes,
                fp8_flops=att_fp8,
                bf16_flops=att_bf16 + att_scores,
                collectives=att_colls,
            )
        )

    kh = max(1, geom.linear_k_heads // tp) if geom.linear_k_heads > 0 else 0
    vh = max(1, geom.linear_v_heads // tp) if geom.linear_v_heads > 0 else 0
    qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
    z_r = vh * geom.linear_v_dim
    gdn_dense_elems = float(n_gdn * (geom.hidden * (qkv_r + z_r) + z_r * geom.hidden))
    gdn_aux_elems = float(n_gdn * (geom.hidden * (2 * vh) + qkv_r * geom.conv_kernel))
    rec_state = float(n_gdn * vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
    conv_state = float(n_gdn * qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
    gdn_state_bytes = float(shape.batch * (1 + shape.state_writes)) * (
        rec_state + conv_state
    )
    dense_fp8, dense_bf16 = _split_matmul_flops(
        rows, gdn_dense_elems, dt.w_dense, dt.act
    )
    aux_fp8, aux_bf16 = _split_matmul_flops(rows, gdn_aux_elems, dt.w_aux, dt.act)
    if n_gdn > 0:
        gdn_colls = (
            _attn_layer_collectives(
                shape.batch,
                shape.rows_per_seq,
                geom.hidden,
                nq,
                geom.head_dim,
                par,
                is_full_attention=False,
            )
            * n_gdn
        )
        out.append(
            ModuleCost(
                name="gdn",
                hbm_bytes=(
                    gdn_dense_elems * dt.w_dense
                    + gdn_aux_elems * dt.w_aux
                    + gdn_state_bytes
                ),
                fp8_flops=dense_fp8 + aux_fp8,
                bf16_flops=dense_bf16 + aux_bf16,
                collectives=gdn_colls,
            )
        )

    if geom.num_experts > 0:
        if par.moe == "ep":
            ed = ep_busiest_experts(geom.num_experts, geom.top_k, rows, tp)
            ed_min = float(math.ceil(geom.top_k / tp)) if rows > 0 else 0.0
            width = geom.moe_inter
        else:
            ed = experts_touched(geom.num_experts, geom.top_k, rows)
            ed_min = float(min(geom.num_experts, geom.top_k)) if rows > 0 else 0.0
            width = geom.moe_inter // tp
        routed_unit = float(geom.num_layers * 3 * geom.hidden * width)
        routed_elems = ed * routed_unit
        routed_elems_min = ed_min * routed_unit
        isr_tp = max(128, geom.shared_inter // tp) if geom.shared_inter > 0 else 0
        isr_eff = geom.shared_inter if par.residual == "sp" else isr_tp
        shared_elems = float(geom.num_layers * 3 * geom.hidden * isr_eff)
        router_elems = float(geom.num_layers * geom.hidden * geom.num_experts)
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
        routed_act_elems = float(
            geom.num_layers * geom.top_k * 3 * geom.hidden * (geom.moe_inter // tp)
        )
        sh_rows = math.ceil(rows / tp) if par.residual == "sp" else rows
        r_fp8, r_bf16 = _split_matmul_flops(rows, routed_act_elems, dt.w_expert, dt.act)
        s_fp8, s_bf16 = _split_matmul_flops(sh_rows, shared_elems, dt.w_dense, dt.act)
        rt_fp8, rt_bf16 = _split_matmul_flops(rows, router_elems, dt.router, dt.act)
        moe_colls = (
            _moe_layer_collectives(rows, geom.hidden, geom.num_experts, geom.top_k, par)
            * geom.num_layers
        )
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
        mlp_inter = geom.dense_inter // tp
        mlp_elems = float(geom.num_layers * 3 * geom.hidden * mlp_inter)
        mlp_fp8, mlp_bf16 = _split_matmul_flops(rows, mlp_elems, dt.w_dense, dt.act)
        mlp_colls = (
            _moe_layer_collectives(rows, geom.hidden, 0, 0, par) * geom.num_layers
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

    lm_elems = float(geom.vocab * geom.hidden // tp)
    lm_fp8, lm_bf16 = _split_matmul_flops(rows, lm_elems, dt.lm_head, dt.act)
    out.append(
        ModuleCost(
            name="lm_head",
            hbm_bytes=lm_elems * dt.lm_head,
            fp8_flops=lm_fp8,
            bf16_flops=lm_bf16,
            collectives=_lm_head_collectives(rows, geom.hidden, geom.vocab, par),
        )
    )
    return tuple(out)
