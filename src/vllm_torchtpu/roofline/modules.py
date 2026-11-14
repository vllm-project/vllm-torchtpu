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
from typing import TYPE_CHECKING, Literal

from .collectives import Algo, Collective, CollectiveConfig, Kind, make, padded_rows
from .constants import (
    BF16_BYTES,
    DEFAULT_CORES_PER_CHIP,
    FP8_BYTES,
    HBM_MIN_PACKING_DIVISOR,
    HBM_PACKING_NUMERATOR,
    HEAD_DIM_PACKING_THRESHOLD,
    MATMUL_FLOP_PER_MAC,
    PHYSICAL_ALIGNMENT_ELEMS,
    SPARSE_MLA_SCALE_BYTES,
    SUBCHANNEL_BLOCK_K_FLOOR,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from .geometry import LayerCounts, count_layer_types, layer_ffn_kinds

if TYPE_CHECKING:
    from .geometry import ModelGeometry


@dataclass(frozen=True, slots=True)
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
    w_dense_source: Literal["quantization_config", "flag", "default"] = "default"
    w_expert_source: Literal["quantization_config", "flag", "default"] = "default"
    kv_source: Literal["quantization_config", "flag", "default"] = "default"
    w_dense_dtype: str = "bfloat16"
    w_expert_dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"


@dataclass(frozen=True, slots=True)
class Parallel:
    """Per-sublayer parallelism, collective layout, and rank placement."""

    tp: int
    kv_split: int = 1
    kv_merge: Literal["ag_rs", "exchange"] = "ag_rs"
    moe: Literal["tp_slice", "ep"] = "tp_slice"
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    residual: Literal["replicated", "sp"] = "replicated"
    ar_algo: Algo = "one_shot"
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP
    dp: int = 1
    ep: int = 1
    pp: int = 1
    pcp: int = 1
    mesh_axes: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        """Validate parallelism constraints."""
        if self.pcp > 1 and self.kv_split > 1:
            msg = (
                f"Parallel.pcp={self.pcp} and Parallel.kv_split={self.kv_split} "
                "are mutually exclusive in one forward pass"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StepShape:
    """Per-step batch size, token rows per sequence, context, and prefill tokens."""

    batch: int = 1
    rows_per_seq: int = 1
    context: int = 0
    state_writes: int = 1
    prefill_tokens: int = 0
    prefix_hit_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class ModuleCost:
    """Per-TensorCore resource work for one module class across one pass."""

    name: str
    hbm_bytes: float
    fp8_flops: float
    bf16_flops: float
    collectives: tuple[Collective, ...] = ()
    hbm_bytes_min: float | None = None
    unique_collectives: tuple[tuple[Collective, int], ...] | None = None
    fp8_flops_min: float | None = None
    bf16_flops_min: float | None = None
    vpu_flops: float = 0.0
    strict_vpu_flops: float = 0.0

    @property
    def ici_bytes(self) -> float:
        """Total inter-chip interconnect wire bytes across off-chip collectives."""
        if self.unique_collectives is not None:
            return sum(
                c.wire_bytes * c.offchip * call_count
                for c, call_count in self.unique_collectives
            )
        return sum(c.wire_bytes * c.offchip for c in self.collectives)

    @property
    def n_collectives(self) -> int:
        """Number of collective communications required by this module."""
        return len(self.collectives)

    @property
    def chain_hops(self) -> int:
        """Total network chain hops incurred by all collectives in this module."""
        return sum(c.hops for c in self.collectives)

    @property
    def strict_hbm_bytes(self) -> float:
        """Strict lower-bound HBM bytes when available, else nominal HBM bytes."""
        return self.hbm_bytes if self.hbm_bytes_min is None else self.hbm_bytes_min

    @property
    def strict_fp8_flops(self) -> float:
        """Unpadded lower-bound FP8 FLOPs when available, else nominal FP8 FLOPs."""
        return self.fp8_flops if self.fp8_flops_min is None else self.fp8_flops_min

    @property
    def strict_bf16_flops(self) -> float:
        """Unpadded lower-bound BF16 FLOPs when available, else nominal BF16 FLOPs."""
        return self.bf16_flops if self.bf16_flops_min is None else self.bf16_flops_min


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
    """Route quantized (<2 B) matmuls with a8 activations to FP8 peak, else bf16.

    On TPU v7x, symmetric MXU rate requires both inputs in fp8 (W8A8);
    mixed W8A16 (bf16 activations x fp8 weights) executes at the bf16 MXU rate.
    """
    flops = MATMUL_FLOP_PER_MAC * rows * weight_elems
    if weight_bytes < BF16_BYTES and act == "a8":
        return (flops, 0.0)
    return (0.0, flops)


def _subchannel_scale_vpu_flops(
    rows: int,
    weight_elems: float,
    weight_bytes: float,
    weight_block_size: tuple[int, int] = (1, 1),
) -> float:
    """Return VPU partial-sum scale multiplication FLOPs when reduction block < 512."""
    if weight_bytes >= BF16_BYTES or rows <= 0 or weight_elems <= 0.0:
        return 0.0
    block_k = (
        weight_block_size[1]
        if len(weight_block_size) > 1 and weight_block_size[1] > 1
        else (weight_block_size[0] if weight_block_size else 1)
    )
    if 1 < block_k < SUBCHANNEL_BLOCK_K_FLOOR:
        return float(MATMUL_FLOP_PER_MAC * rows * weight_elems) / float(block_k)
    return 0.0


def _sub_byte_unpack_vpu_flops(weight_elems: float, weight_bytes: float) -> float:
    """Vector ops to unpack weights stored below one byte per element."""
    return 2.0 * weight_elems if weight_bytes < 1.0 else 0.0


def _strict_sub_byte_unpack_vpu_flops(unpack_vpu_flops: float) -> float:
    """Scale an unpack count for the strict vector bound.

    The bound divides strict counts by VECTOR_MAX_ELEMENTS_PER_WORD. Weights below
    one byte pack VECTOR_SUB_BYTE_ELEMENTS_PER_WORD per word, so their unpack count
    is scaled by the ratio of the two.
    """
    return (
        unpack_vpu_flops
        * VECTOR_MAX_ELEMENTS_PER_WORD
        / VECTOR_SUB_BYTE_ELEMENTS_PER_WORD
    )


def _make_coll(
    name: str,
    kind: Kind,
    payload: float,
    group: int,
    parallel: Parallel,
) -> Collective:
    """Create a Collective using parallel config's algorithm and core packing."""
    return make(
        name,
        kind,
        payload,
        group,
        CollectiveConfig(algo=parallel.ar_algo, ranks_per_chip=parallel.ranks_per_chip),
    )


def _embed_collectives(
    rows: int, hidden: int, parallel: Parallel
) -> tuple[Collective, ...]:
    """Return all-reduce collectives for embedding when tensor parallel > 1."""
    if parallel.tp <= 1 or hidden <= 0:
        return ()
    return (
        _make_coll("embed_ar", "ar", float(rows * hidden * 2), parallel.tp, parallel),
    )


def _repeat_colls(
    base: tuple[Collective, ...], count: int
) -> tuple[tuple[Collective, ...], tuple[tuple[Collective, int], ...]]:
    """Repeat base collectives count times and build unique collective tally."""
    if not base or count <= 0:
        return (), ()
    return base * count, tuple((c, count) for c in base)


@dataclass(frozen=True, slots=True)
class _StepContext:
    """Internal evaluation context for one forward step."""

    rows: int
    decode_rows: int
    eff_prefill: int
    cached_prefix: int
    pp_scale: float
    context: int
    batch: int
    rows_per_seq: int
    state_writes: int
    mxu_m_tile: int = 1
    mxu_lhs_rows_bf16: int = 16
    mxu_lhs_rows_fp8: int = 32


@dataclass(frozen=True, slots=True)
class _AttnCollSpec:
    """Layer collective parameters for one attention or recurrence block."""

    batch: int
    rows_per_seq: int
    hidden: int
    nq: int
    head_dim: int
    is_full_attention: bool = True
    extra_rows: int = 0
    block_parallel: int = 1


@dataclass(frozen=True, slots=True)
class _MoeDispatchSpec:
    """Activation wire payload and expert routing dimensions for SP MoE dispatch."""

    s_bytes: float
    push_wire: float
    mp: int
    num_experts: int
    disp_group: int
    block_parallel: int = 1


@dataclass(frozen=True, slots=True)
class _MoeLayerSpec:
    """Layer dimensions for MoE/MLP collective derivation."""

    rows: int
    hidden: int
    num_experts: int
    top_k: int
    block_parallel: int = 1


def _attn_layer_collectives(
    spec: _AttnCollSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Return communication collectives for one attention layer."""
    rows = spec.batch * spec.rows_per_seq + spec.extra_rows
    mp = padded_rows(rows, parallel.tp, parallel.residual)
    s_bytes = float(mp * spec.hidden * 2)
    colls: list[Collective] = []
    if parallel.tp > 1:
        if parallel.residual == "replicated":
            colls.append(_make_coll("out_ar", "ar", s_bytes, parallel.tp, parallel))
        else:
            colls.append(_make_coll("entry_ag", "ag", s_bytes, parallel.tp, parallel))
            if not (spec.block_parallel > 1 and parallel.moe_dispatch == "allreduce"):
                colls.append(_make_coll("out_rs", "rs", s_bytes, parallel.tp, parallel))
    if spec.is_full_attention and parallel.kv_split > 1:
        g = parallel.kv_split
        dec_rows = spec.batch * spec.rows_per_seq
        if parallel.kv_merge == "ag_rs":
            colls.append(
                _make_coll(
                    "dcp_q_ag",
                    "ag",
                    float(dec_rows * g * spec.nq * spec.head_dim * 2),
                    g,
                    parallel,
                )
            )
            colls.append(
                _make_coll(
                    "dcp_lse_ag",
                    "ag",
                    float(g * dec_rows * g * spec.nq * 4),
                    g,
                    parallel,
                )
            )
            colls.append(
                _make_coll(
                    "dcp_out_rs",
                    "rs",
                    float(dec_rows * g * spec.nq * spec.head_dim * 4),
                    g,
                    parallel,
                )
            )
        else:
            per_peer = float(
                (spec.batch + 1) * spec.rows_per_seq * spec.nq * spec.head_dim * 4
            )
            colls.append(
                _make_coll("merge_xchg", "push", (g - 1) * per_peer, g, parallel)
            )
    if spec.is_full_attention and parallel.pcp > 1 and spec.extra_rows > 0:
        colls.append(
            _make_coll(
                "pcp_kv_ag",
                "ag",
                float(
                    spec.extra_rows * parallel.pcp * max(1, spec.nq) * spec.head_dim * 2
                ),
                parallel.pcp,
                parallel,
            )
        )
    return tuple(colls)


def _moe_ag_ep_collectives(
    spec: _MoeLayerSpec,
    ep_ranks: int,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Return collectives for ag_ep MoE dispatch."""
    tp = parallel.tp
    global_rows = spec.rows * max(1, parallel.dp)
    per_rank_rows = max(1, math.ceil(global_rows / max(1, ep_ranks)))
    colls: list[Collective] = [
        _make_coll(
            "dispatch_ag", "ag", float(global_rows * spec.hidden), ep_ranks, parallel
        ),
        _make_coll(
            "route_ag",
            "ag",
            float(ep_ranks * per_rank_rows * spec.num_experts * 2),
            ep_ranks,
            parallel,
        ),
        _make_coll(
            "restore_ag", "ag", float(global_rows * spec.hidden * 2), ep_ranks, parallel
        ),
    ]
    if parallel.residual == "replicated" and tp > 1 and spec.block_parallel <= 1:
        colls.append(
            _make_coll("moe_ar", "ar", float(spec.rows * spec.hidden * 2), tp, parallel)
        )
    return tuple(colls)


def _moe_fused_rs_collectives(
    s_bytes: float,
    tp: int,
    parallel: Parallel,
    *,
    block_parallel: int = 1,
) -> tuple[Collective, ...]:
    """Return collectives for fused_rs MoE dispatch."""
    if tp <= 1:
        return ()
    if parallel.residual == "replicated":
        return (
            _make_coll("combine_rs", "rs", s_bytes, tp, parallel),
            _make_coll("restore_ag", "ag", s_bytes, tp, parallel),
        )
    colls: list[Collective] = []
    if block_parallel <= 1:
        colls.append(_make_coll("dispatch_ag", "ag", s_bytes, tp, parallel))
    colls.append(_make_coll("combine_rs", "rs", s_bytes, tp, parallel))
    return tuple(colls)


def _moe_replicated_collectives(
    s_bytes: float,
    push_wire: float,
    disp_group: int,
    parallel: Parallel,
    *,
    block_parallel: int = 1,
) -> tuple[Collective, ...]:
    """Return collectives for replicated residual MoE dispatch."""
    tp = parallel.tp
    if parallel.moe_dispatch == "allreduce":
        if block_parallel <= 1:
            return (_make_coll("moe_ar", "ar", s_bytes, tp, parallel),)
        return ()
    colls: list[Collective] = [
        _make_coll("dispatch_push", "push", push_wire, disp_group, parallel),
        _make_coll("combine_push", "push", push_wire, disp_group, parallel),
    ]
    if tp > 1:
        colls.append(_make_coll("restore_ag", "ag", s_bytes, tp, parallel))
    return tuple(colls)


def _moe_sp_collectives(
    spec: _MoeDispatchSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Return collectives for SP residual MoE dispatch."""
    tp = parallel.tp
    if parallel.moe_dispatch == "allreduce":
        colls: list[Collective] = []
        if spec.block_parallel <= 1:
            colls.append(_make_coll("dispatch_ag", "ag", spec.s_bytes, tp, parallel))
        colls.extend(
            [
                _make_coll(
                    "logits_ag",
                    "ag",
                    float(spec.mp * spec.num_experts * 2),
                    tp,
                    parallel,
                ),
                _make_coll("combine_rs", "rs", spec.s_bytes, tp, parallel),
            ]
        )
        return tuple(colls)
    return (
        _make_coll("dispatch_push", "push", spec.push_wire, spec.disp_group, parallel),
        _make_coll("combine_push", "push", spec.push_wire, spec.disp_group, parallel),
    )


def _moe_layer_collectives(
    spec: _MoeLayerSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Derive collectives for one MoE layer."""
    ep_ranks = parallel.ep if parallel.ep > 1 else parallel.tp
    if parallel.tp <= 1 and ep_ranks <= 1:
        return ()
    tp = parallel.tp
    mp = padded_rows(spec.rows, max(1, tp), parallel.residual)
    s_bytes = float(mp * spec.hidden * 2)

    if parallel.moe_dispatch == "ag_ep" or (
        parallel.moe == "ep" and ep_ranks > tp and parallel.moe_dispatch == "allreduce"
    ):
        return _moe_ag_ep_collectives(spec, ep_ranks, parallel)

    if parallel.moe_dispatch == "fused_rs":
        return _moe_fused_rs_collectives(
            s_bytes, tp, parallel, block_parallel=spec.block_parallel
        )

    disp_group = ep_ranks if parallel.moe_dispatch == "a2a_push" else tp
    push_wire = float(
        math.ceil((spec.rows * max(1, parallel.dp)) / max(1, disp_group))
        * spec.top_k
        * (disp_group - 1)
        / max(1, disp_group)
        * spec.hidden
    )
    if parallel.residual == "replicated":
        return _moe_replicated_collectives(
            s_bytes, push_wire, disp_group, parallel, block_parallel=spec.block_parallel
        )
    return _moe_sp_collectives(
        _MoeDispatchSpec(
            s_bytes=s_bytes,
            push_wire=push_wire,
            mp=mp,
            num_experts=spec.num_experts,
            disp_group=disp_group,
            block_parallel=spec.block_parallel,
        ),
        parallel,
    )


def _lm_head_collectives(
    rows: int,
    hidden: int,
    vocab: int,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Return communication collectives for language model head."""
    if parallel.tp <= 1:
        return ()
    tp = parallel.tp
    mp = padded_rows(rows, tp, parallel.residual)
    s_bytes = float(mp * hidden * 2)
    colls: list[Collective] = []
    if parallel.residual == "sp" and hidden > 0:
        colls.append(_make_coll("tail_ag", "ag", s_bytes, tp, parallel))
    if vocab > 0:
        colls.append(
            _make_coll("logits_ag", "ag", float(rows * vocab * 2), tp, parallel)
        )
    return tuple(colls)


def _validate_pass_costs_config(geom: ModelGeometry, parallel: Parallel) -> None:
    """Validate parallelism and geometry constraints for pass_costs."""
    tp = parallel.tp
    if tp < 1:
        msg = f"Parallel.tp must be >= 1, got {tp}"
        raise ValueError(msg)
    kv_group = (
        max(1, tp // geom.num_kv_heads)
        if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
        else max(1, tp)
    )
    if parallel.kv_split < 1 or kv_group % parallel.kv_split != 0:
        msg = (
            f"Parallel.kv_split={parallel.kv_split} must be >= 1 and divide "
            f"kv_group={kv_group}"
        )
        raise ValueError(msg)
    if parallel.residual == "sp" and tp == 1:
        msg = "Parallel.residual='sp' requires tp > 1"
        raise ValueError(msg)
    if parallel.residual == "sp" and parallel.moe != "ep" and geom.num_experts > 0:
        msg = "Parallel.residual='sp' requires Parallel.moe='ep'"
        raise ValueError(msg)
    if parallel.moe_dispatch in ("a2a_push", "ag_ep") and parallel.moe != "ep":
        disp = parallel.moe_dispatch
        msg = f"Parallel.moe_dispatch={disp!r} requires Parallel.moe='ep'"
        raise ValueError(msg)


def _gqa_swa_kv_bytes_per_tok(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    head_dim: int,
    *,
    physical: bool,
) -> float:
    """Compute per-token per-layer KV bytes for GQA or SWA head geometry."""
    tp = max(1, parallel.tp)
    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    if not physical:
        return float(2 * kvh * head_dim) * dt.kv
    kv_packing = max(
        1,
        round(HBM_PACKING_NUMERATOR / max(HBM_MIN_PACKING_DIVISOR, dt.kv)),
    )
    padded_hd = (
        math.ceil(head_dim / PHYSICAL_ALIGNMENT_ELEMS) * PHYSICAL_ALIGNMENT_ELEMS
        if head_dim > 0
        else 0
    )
    kx2 = kvh if head_dim == HEAD_DIM_PACKING_THRESHOLD else 2 * kvh
    padded_kx2 = math.ceil(kx2 / kv_packing) * kv_packing if kx2 > 0 else 0
    return float(padded_kx2 * padded_hd) * dt.kv


def kv_bytes_per_token_per_layer(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layer_type: str = "gqa",
    *,
    physical: bool = True,
) -> float:
    """Return KV cache bytes stored per token for one layer of given attention type."""
    resolved = layer_type
    if resolved in ("full_attention", "attention"):
        if geom.kv_lora_rank > 0 and geom.index_topk > 0:
            resolved = "dsa"
        elif geom.kv_lora_rank > 0:
            resolved = "mla"
        else:
            resolved = "gqa"

    if resolved == "gqa":
        gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
        return _gqa_swa_kv_bytes_per_tok(geom, dt, parallel, gqa_hd, physical=physical)
    if resolved == "swa":
        return _gqa_swa_kv_bytes_per_tok(
            geom, dt, parallel, geom.head_dim, physical=physical
        )
    if resolved in ("mla", "dsa"):
        raw_latent_w = geom.kv_lora_rank + geom.qk_rope_head_dim
        latent_w = (
            math.ceil(raw_latent_w / PHYSICAL_ALIGNMENT_ELEMS)
            * PHYSICAL_ALIGNMENT_ELEMS
            if physical
            else raw_latent_w
        )
        mla_latent_b = float(latent_w) * dt.kv
        has_sparse = (
            resolved == "dsa" and geom.index_topk > 0 and geom.index_head_dim > 0
        )
        idx_scale = SPARSE_MLA_SCALE_BYTES if (physical and dt.kv <= FP8_BYTES) else 0.0
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv + idx_scale if has_sparse else 0.0
        )
        return mla_latent_b + dsa_idx_b

    msg = f"Unknown attention layer type: {layer_type}"
    raise ValueError(msg)


def _align_mxu(dim: int, eval_context: _StepContext) -> int:
    """Pad systolic column dimension to 128 lanes when MXU tiling is active."""
    if dim <= 0 or eval_context.mxu_m_tile <= 1:
        return dim
    return math.ceil(dim / PHYSICAL_ALIGNMENT_ELEMS) * PHYSICAL_ALIGNMENT_ELEMS


def _compute_gqa_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    counts: LayerCounts,
) -> ModuleCost:
    """Compute ModuleCost for GQA attention."""
    n_gqa = counts.gqa
    tp = parallel.tp
    n_eff = n_gqa * eval_context.pp_scale
    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    q_mult = 2 if geom.attn_output_gate else 1
    gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    q_out = geom.num_heads * gqa_hd * q_mult
    q_sharded = q_out // tp
    kv_sharded = kvh * gqa_hd
    out_sharded = (geom.num_heads // tp) * gqa_hd
    att_w_elems = float(
        n_eff
        * (
            geom.hidden * q_sharded
            + 2 * geom.hidden * kv_sharded
            + out_sharded * geom.hidden
        )
    )
    att_mxu_elems = float(
        n_eff
        * geom.hidden
        * (
            _align_mxu(q_sharded, eval_context)
            + 2 * _align_mxu(kv_sharded, eval_context)
            + _align_mxu(out_sharded, eval_context)
        )
    )
    gqa_kv_scale = (counts.kv_gqa / float(n_gqa)) if geom.compress_ratios else 1.0
    kv_dec_tokens = (
        float(eval_context.batch * eval_context.context)
        * gqa_kv_scale
        / parallel.kv_split
    )
    kv_pre_tokens = float(eval_context.cached_prefix + eval_context.eff_prefill)
    kv_tokens_total = n_eff * (kv_dec_tokens + kv_pre_tokens)
    b_kv_phys = kv_bytes_per_token_per_layer(geom, dt, parallel, "gqa", physical=True)
    b_kv_min = kv_bytes_per_token_per_layer(geom, dt, parallel, "gqa", physical=False)
    att_fp8, att_bf16 = _split_matmul_flops(
        eval_context.rows, att_mxu_elems, dt.w_dense, dt.act
    )
    att_fp8_min, att_bf16_min = _split_matmul_flops(
        eval_context.rows, att_w_elems, dt.w_dense, dt.act
    )
    dec_scores = float(
        eval_context.decode_rows
        * 4
        * eval_context.context
        * gqa_hd
        * (geom.num_heads // tp)
        * n_eff
    )
    pre_ctx_eff = (
        eval_context.cached_prefix + 0.5 * eval_context.eff_prefill * parallel.pcp
    )
    pre_scores = float(
        eval_context.eff_prefill
        * 4
        * pre_ctx_eff
        * gqa_hd
        * (geom.num_heads // tp)
        * n_eff
    )
    n_colls_layers = max(1, round(n_eff))
    att_colls, att_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geom.hidden,
                nq=geom.num_heads // tp,
                head_dim=gqa_hd,
                is_full_attention=True,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geom.block_parallel,
            ),
            parallel,
        ),
        n_colls_layers,
    )
    dense_unpack = _sub_byte_unpack_vpu_flops(att_w_elems, dt.w_dense)
    vpu_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * (q_sharded + 2 * kv_sharded)
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        )
        + _subchannel_scale_vpu_flops(
            eval_context.rows, att_w_elems, dt.w_dense, geom.weight_block_size
        )
        + dense_unpack
    )
    strict_vpu_flops = (
        vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
    )
    return ModuleCost(
        name="gqa",
        hbm_bytes=att_w_elems * dt.w_dense + kv_tokens_total * b_kv_phys,
        fp8_flops=att_fp8,
        bf16_flops=att_bf16 + dec_scores + pre_scores,
        collectives=att_colls,
        hbm_bytes_min=att_w_elems * dt.w_dense + kv_tokens_total * b_kv_min,
        fp8_flops_min=att_fp8_min,
        bf16_flops_min=att_bf16_min + dec_scores + pre_scores,
        unique_collectives=att_ucolls,
        vpu_flops=vpu_flops,
        strict_vpu_flops=strict_vpu_flops,
    )


def _compute_swa_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    *,
    n_swa: int,
) -> ModuleCost:
    """Compute ModuleCost for SWA attention."""
    tp = parallel.tp
    n_eff = n_swa * eval_context.pp_scale
    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
    q_mult = 2 if geom.attn_output_gate else 1
    q_out = geom.num_heads * geom.head_dim * q_mult
    sw_cap = geom.sliding_window if geom.sliding_window > 0 else eval_context.context
    eff_ctx = min(eval_context.context, sw_cap) if eval_context.context > 0 else 0
    q_sharded = q_out // tp
    kv_sharded = kvh * geom.head_dim
    out_sharded = nq * geom.head_dim
    swa_w_elems = float(
        n_eff
        * (
            geom.hidden * q_sharded
            + 2 * geom.hidden * kv_sharded
            + out_sharded * geom.hidden
        )
    )
    kv_tokens = float(eval_context.batch * eff_ctx) / parallel.kv_split + float(
        min(eval_context.cached_prefix + eval_context.eff_prefill, sw_cap)
        if eval_context.eff_prefill > 0
        else 0
    )
    b_kv_phys = kv_bytes_per_token_per_layer(geom, dt, parallel, "swa", physical=True)
    b_kv_min = kv_bytes_per_token_per_layer(geom, dt, parallel, "swa", physical=False)
    swa_fp8, swa_bf16 = _split_matmul_flops(
        eval_context.rows, swa_w_elems, dt.w_dense, dt.act
    )
    swa_scores = float(
        (
            eval_context.decode_rows * eff_ctx
            + eval_context.eff_prefill * min(sw_cap, max(1, eval_context.eff_prefill))
        )
        * 4
        * geom.head_dim
        * nq
        * n_eff
    )
    swa_colls, swa_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geom.hidden,
                nq=nq,
                head_dim=geom.head_dim,
                is_full_attention=True,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geom.block_parallel,
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vpu_flops(swa_w_elems, dt.w_dense)
    vpu_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * (q_sharded + 2 * kv_sharded)
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        )
        + _subchannel_scale_vpu_flops(
            eval_context.rows, swa_w_elems, dt.w_dense, geom.weight_block_size
        )
        + dense_unpack
    )
    strict_vpu_flops = (
        vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
    )
    return ModuleCost(
        name="swa",
        hbm_bytes=swa_w_elems * dt.w_dense + n_eff * kv_tokens * b_kv_phys,
        fp8_flops=swa_fp8,
        bf16_flops=swa_bf16 + swa_scores,
        collectives=swa_colls,
        hbm_bytes_min=swa_w_elems * dt.w_dense + n_eff * kv_tokens * b_kv_min,
        unique_collectives=swa_ucolls,
        vpu_flops=vpu_flops,
        strict_vpu_flops=strict_vpu_flops,
    )


def _validate_mla_geom(fam_name: str, geom: ModelGeometry) -> None:
    """Validate model geometry fields required for MLA or DSA."""
    if geom.kv_lora_rank <= 0:
        msg = f"{fam_name.upper()} layer requires config field 'kv_lora_rank'"
        raise ValueError(msg)
    if geom.qk_nope_head_dim <= 0:
        msg = f"{fam_name.upper()} layer requires config field 'qk_nope_head_dim'"
        raise ValueError(msg)
    if geom.qk_rope_head_dim < 0:
        msg = f"{fam_name.upper()} layer requires config field 'qk_rope_head_dim'"
        raise ValueError(msg)


def _mla_unpack_vpu_flops(
    base_vpu_flops: float,
    mla_w_elems: float,
    w_dense: float,
) -> tuple[float, float]:
    """Return nominal and strict VPU FLOPs including sub-byte unpack."""
    dense_unpack = _sub_byte_unpack_vpu_flops(mla_w_elems, w_dense)
    vpu_flops = base_vpu_flops + dense_unpack
    strict_vpu_flops = (
        vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
    )
    return vpu_flops, strict_vpu_flops


def _compute_mla_dsa_costs(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    counts: LayerCounts,
) -> list[ModuleCost]:
    """Compute ModuleCost for MLA and DSA attention families."""
    tp = parallel.tp
    nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
    out: list[ModuleCost] = []
    for fam_name, n_fam, kv_n_fam, is_sparse in (
        ("mla", counts.mla, counts.kv_mla, False),
        ("dsa", counts.dsa, counts.kv_dsa, True),
    ):
        if n_fam <= 0:
            continue
        n_eff = n_fam * eval_context.pp_scale
        _validate_mla_geom(fam_name, geom)
        kv_lora = geom.kv_lora_rank
        nope_d = geom.qk_nope_head_dim
        rope_d = geom.qk_rope_head_dim
        v_d = geom.v_head_dim or nope_d
        nq_raw = nq * (nope_d + rope_d)
        if geom.q_lora_rank > 0:
            q_w = float(geom.hidden * geom.q_lora_rank + geom.q_lora_rank * nq_raw)
        else:
            q_w = float(geom.hidden * nq_raw)
        mla_w_layer = (
            q_w
            + float(geom.hidden * (kv_lora + rope_d))
            + float(kv_lora * nq * (nope_d + v_d) + (nq * v_d) * geom.hidden)
        )
        if is_sparse and geom.index_n_heads > 0 and geom.index_head_dim > 0:
            idx_heads = max(1, geom.index_n_heads // tp)
            mla_w_layer += float(geom.hidden * idx_heads * geom.index_head_dim)
        mla_w_elems = n_eff * mla_w_layer
        att_ctx = (
            min(eval_context.context, geom.index_topk)
            if (is_sparse and geom.index_topk > 0 and eval_context.context > 0)
            else eval_context.context
        )
        raw_latent_w = kv_lora + rope_d
        b_latent_phys = (
            float(
                math.ceil(raw_latent_w / PHYSICAL_ALIGNMENT_ELEMS)
                * PHYSICAL_ALIGNMENT_ELEMS
            )
            * dt.kv
        )
        b_latent_min = float(raw_latent_w) * dt.kv
        has_idx = is_sparse and geom.index_topk > 0 and geom.index_head_dim > 0
        b_idx_phys = (
            float(geom.index_head_dim) * dt.kv
            + (SPARSE_MLA_SCALE_BYTES if dt.kv <= FP8_BYTES else 0.0)
            if has_idx
            else 0.0
        )
        b_idx_min = float(geom.index_head_dim) * dt.kv if has_idx else 0.0
        fam_kv_scale = (kv_n_fam / float(n_fam)) if geom.compress_ratios else 1.0
        kv_tokens = float(
            eval_context.batch * att_ctx
        ) * fam_kv_scale / parallel.kv_split + float(
            eval_context.cached_prefix + eval_context.eff_prefill
        )
        mla_kv_bytes = n_eff * kv_tokens * b_latent_phys
        mla_kv_bytes_min = n_eff * kv_tokens * b_latent_min
        if is_sparse and geom.index_head_dim > 0 and eval_context.context > 0:
            idx_tokens = (
                n_eff
                * float(eval_context.batch * eval_context.context)
                * fam_kv_scale
                / parallel.kv_split
            )
            mla_kv_bytes += idx_tokens * b_idx_phys
            mla_kv_bytes_min += idx_tokens * b_idx_min
        mla_fp8, mla_bf16 = _split_matmul_flops(
            eval_context.rows, mla_w_elems, dt.w_dense, dt.act
        )
        mla_scores = float(
            (
                eval_context.decode_rows * att_ctx
                + eval_context.eff_prefill
                * (
                    eval_context.cached_prefix
                    + 0.5 * eval_context.eff_prefill * parallel.pcp
                )
            )
            * 2
            * nq
            * (kv_lora + rope_d + kv_lora)
            * n_eff
        )
        mla_colls, mla_ucolls = _repeat_colls(
            _attn_layer_collectives(
                _AttnCollSpec(
                    batch=eval_context.batch,
                    rows_per_seq=eval_context.rows_per_seq,
                    hidden=geom.hidden,
                    nq=nq,
                    head_dim=kv_lora,
                    is_full_attention=True,
                    extra_rows=eval_context.eff_prefill,
                    block_parallel=geom.block_parallel,
                ),
                parallel,
            ),
            max(1, round(n_eff)),
        )
        base_vpu_flops = float(
            n_eff
            * (
                4.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * nq * rope_d
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        ) + _subchannel_scale_vpu_flops(
            eval_context.rows, mla_w_elems, dt.w_dense, geom.weight_block_size
        )
        vpu_flops, strict_vpu_flops = _mla_unpack_vpu_flops(
            base_vpu_flops, mla_w_elems, dt.w_dense
        )
        out.append(
            ModuleCost(
                name=fam_name,
                hbm_bytes=mla_w_elems * dt.w_dense + mla_kv_bytes,
                fp8_flops=mla_fp8,
                bf16_flops=mla_bf16 + mla_scores,
                collectives=mla_colls,
                hbm_bytes_min=mla_w_elems * dt.w_dense + mla_kv_bytes_min,
                unique_collectives=mla_ucolls,
                vpu_flops=vpu_flops,
                strict_vpu_flops=strict_vpu_flops,
            )
        )
    return out


def _compute_gdn_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    *,
    n_gdn: int,
) -> ModuleCost:
    """Compute ModuleCost for GDN linear attention."""
    tp = parallel.tp
    n_eff = n_gdn * eval_context.pp_scale
    kh = max(1, geom.linear_k_heads // tp) if geom.linear_k_heads > 0 else 0
    vh = max(1, geom.linear_v_heads // tp) if geom.linear_v_heads > 0 else 0
    qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
    z_r = vh * geom.linear_v_dim
    gdn_dense_elems = float(n_eff * (geom.hidden * (qkv_r + z_r) + z_r * geom.hidden))
    gdn_aux_elems = float(n_eff * (geom.hidden * (2 * vh) + qkv_r * geom.conv_kernel))
    rec_state = float(n_eff * vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
    conv_state = float(n_eff * qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
    seq_count = eval_context.batch + (
        1 if eval_context.eff_prefill > 0 and eval_context.batch == 0 else 0
    )
    sw_count = (
        eval_context.state_writes
        if eval_context.batch > 0
        else (1 if eval_context.eff_prefill > 0 else 0)
    )
    gdn_state_bytes = float(seq_count * (1 + sw_count)) * (rec_state + conv_state)
    dense_fp8, dense_bf16 = _split_matmul_flops(
        eval_context.rows, gdn_dense_elems, dt.w_dense, dt.act
    )
    aux_fp8, aux_bf16 = _split_matmul_flops(
        eval_context.rows, gdn_aux_elems, dt.w_aux, dt.act
    )
    gdn_rec_flops = float(
        eval_context.rows * 4 * vh * geom.linear_k_dim * geom.linear_v_dim * n_eff
    )
    gdn_colls, gdn_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geom.hidden,
                nq=geom.num_heads // tp,
                head_dim=geom.head_dim,
                is_full_attention=False,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geom.block_parallel,
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vpu_flops(gdn_dense_elems, dt.w_dense)
    vpu_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        )
        + _subchannel_scale_vpu_flops(
            eval_context.rows, gdn_dense_elems, dt.w_dense, geom.weight_block_size
        )
        + dense_unpack
    )
    strict_vpu_flops = (
        vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
    )
    return ModuleCost(
        name="gdn",
        hbm_bytes=(
            gdn_dense_elems * dt.w_dense + gdn_aux_elems * dt.w_aux + gdn_state_bytes
        ),
        fp8_flops=dense_fp8 + aux_fp8,
        bf16_flops=dense_bf16 + aux_bf16 + gdn_rec_flops,
        collectives=gdn_colls,
        unique_collectives=gdn_ucolls,
        vpu_flops=vpu_flops,
        strict_vpu_flops=strict_vpu_flops,
    )


def _compute_kda_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    *,
    n_kda: int,
) -> ModuleCost:
    """Compute ModuleCost for KDA linear attention."""
    tp = parallel.tp
    nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
    n_eff = n_kda * eval_context.pp_scale
    kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
    vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
    kd = geom.linear_k_dim or geom.head_dim
    vd = geom.linear_v_dim or geom.head_dim
    proj_r = 2 * kh * kd + 2 * vh * vd
    kda_w_elems = float(n_eff * (geom.hidden * proj_r + (vh * vd) * geom.hidden))
    kda_rec = float(n_eff * vh * kd * vd) * dt.state
    seq_count = eval_context.batch + (
        1 if eval_context.eff_prefill > 0 and eval_context.batch == 0 else 0
    )
    sw_count = (
        eval_context.state_writes
        if eval_context.batch > 0
        else (1 if eval_context.eff_prefill > 0 else 0)
    )
    kda_state_bytes = float(seq_count * (1 + sw_count)) * kda_rec
    kda_fp8, kda_bf16 = _split_matmul_flops(
        eval_context.rows, kda_w_elems, dt.w_dense, dt.act
    )
    kda_rec_flops = float(eval_context.rows * 4 * vh * kd * vd * n_eff)
    kda_colls, kda_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geom.hidden,
                nq=nq,
                head_dim=vd,
                is_full_attention=False,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geom.block_parallel,
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vpu_flops(kda_w_elems, dt.w_dense)
    vpu_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        )
        + _subchannel_scale_vpu_flops(
            eval_context.rows, kda_w_elems, dt.w_dense, geom.weight_block_size
        )
        + dense_unpack
    )
    strict_vpu_flops = (
        vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
    )
    return ModuleCost(
        name="kda",
        hbm_bytes=kda_w_elems * dt.w_dense + kda_state_bytes,
        fp8_flops=kda_fp8,
        bf16_flops=kda_bf16 + kda_rec_flops,
        collectives=kda_colls,
        unique_collectives=kda_ucolls,
        vpu_flops=vpu_flops,
        strict_vpu_flops=strict_vpu_flops,
    )


def _compute_routed_moe_cost(  # noqa: PLR0915
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
    *,
    n_m_eff: float,
) -> ModuleCost:
    """Compute ModuleCost for routed MoE FFN layer."""
    tp = parallel.tp
    ep_ranks = parallel.ep if parallel.ep > 1 else tp
    pcp_per_ep = (
        min(max(1, parallel.pcp), max(1, ep_ranks // max(1, tp)))
        if parallel.ep > 1
        else 1
    )
    dp_per_ep = (
        min(max(1, parallel.dp), max(1, ep_ranks // max(1, tp * pcp_per_ep)))
        if parallel.ep > 1
        else 1
    )
    global_rows = (
        (eval_context.decode_rows + eval_context.eff_prefill * pcp_per_ep) * dp_per_ep
        if parallel.ep > 1
        else eval_context.rows
    )
    if parallel.moe == "ep":
        ed = ep_busiest_experts(geom.num_experts, geom.top_k, global_rows, ep_ranks)
        ed_min = float(math.ceil(geom.top_k / ep_ranks)) if global_rows > 0 else 0.0
        width = (
            geom.moe_inter // max(1, tp // max(1, ep_ranks))
            if ep_ranks < tp
            else geom.moe_inter
        )
    else:
        ed = experts_touched(geom.num_experts, geom.top_k, eval_context.rows)
        ed_min = (
            float(min(geom.num_experts, geom.top_k)) if eval_context.rows > 0 else 0.0
        )
        width = geom.moe_inter // tp
    expert_hidden = (
        geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
    )
    routed_unit = float(n_m_eff * 3 * expert_hidden * width)
    routed_elems = ed * routed_unit
    routed_elems_min = ed_min * routed_unit
    isr_tp = max(1, geom.shared_inter // tp) if geom.shared_inter > 0 else 0
    isr_eff = geom.shared_inter if parallel.residual == "sp" else isr_tp
    shared_elems = float(n_m_eff * 3 * geom.hidden * isr_eff)
    shared_mxu_elems = float(
        n_m_eff * 3 * geom.hidden * _align_mxu(isr_eff, eval_context)
    )
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
    latent_proj_elems = (
        float(n_m_eff * 2 * geom.hidden * geom.routed_expert_hidden)
        if geom.routed_expert_hidden > 0
        else 0.0
    )
    latent_proj_sharded = (
        latent_proj_elems / tp if geom.routed_expert_hidden > 0 else 0.0
    )
    if geom.routed_expert_hidden > 0:
        moe_w_bytes += latent_proj_sharded * dt.w_dense
        moe_w_bytes_min += latent_proj_sharded * dt.w_dense
    eff_moe_div = max(tp * pcp_per_ep, ep_ranks) if parallel.ep > 1 else tp
    width_mxu = _align_mxu(width, eval_context)
    unsharded_moe_inter_mxu = width_mxu * (
        1
        if (parallel.moe == "ep" and ep_ranks >= tp)
        else (max(1, tp // max(1, ep_ranks)) if parallel.moe == "ep" else tp)
    )
    per_rank_moe_inter = float(unsharded_moe_inter_mxu) / float(eff_moe_div)
    routed_act_elems = float(
        n_m_eff * geom.top_k * 3 * expert_hidden * per_rank_moe_inter
    )
    routed_act_elems_min = float(
        n_m_eff * geom.top_k * 3 * expert_hidden * (geom.moe_inter / eff_moe_div)
    )
    routed_rows = global_rows if parallel.ep > 1 else eval_context.rows
    sh_rows = (
        math.ceil(eval_context.rows / tp)
        if parallel.residual == "sp"
        else eval_context.rows
    )
    r_fp8, r_bf16 = _split_matmul_flops(
        routed_rows, routed_act_elems, dt.w_expert, dt.act
    )
    r_fp8_min, r_bf16_min = _split_matmul_flops(
        routed_rows, routed_act_elems_min, dt.w_expert, dt.act
    )
    if geom.routed_expert_hidden > 0:
        latent_flops_elems = float(
            n_m_eff * 2 * geom.hidden * geom.routed_expert_hidden / tp
        )
        l_fp8, l_bf16 = _split_matmul_flops(
            eval_context.rows, latent_flops_elems, dt.w_dense, dt.act
        )
        r_fp8 += l_fp8
        r_bf16 += l_bf16
        r_fp8_min += l_fp8
        r_bf16_min += l_bf16
    if eval_context.mxu_m_tile > 1 and routed_rows > 0 and ed > 0.0 and geom.top_k > 0:
        ed_global = (
            experts_touched(geom.num_experts, geom.top_k, routed_rows)
            if parallel.moe == "ep"
            else ed
        )
        m_per_touched = float(routed_rows * geom.top_k) / max(1.0, ed_global)
        if m_per_touched > 0.0:
            m_pad = float(
                math.ceil(m_per_touched / eval_context.mxu_m_tile)
                * eval_context.mxu_m_tile
            )
            pad_mult = m_pad / m_per_touched
            r_fp8 *= pad_mult
            r_bf16 *= pad_mult
    lhs_floor = (
        eval_context.mxu_lhs_rows_fp8
        if dt.w_expert <= FP8_BYTES
        else eval_context.mxu_lhs_rows_bf16
    )
    if lhs_floor > 1 and routed_rows > 0 and ed_min > 0.0 and geom.top_k > 0:
        # Concentrated K-expert lower bound: all routed_rows tokens hit the
        # same top_k experts.
        m_pad_strict = float(math.ceil(routed_rows / lhs_floor) * lhs_floor)
        pad_mult_strict = m_pad_strict / float(routed_rows)
        r_fp8_min *= pad_mult_strict
        r_bf16_min *= pad_mult_strict
    s_fp8, s_bf16 = _split_matmul_flops(sh_rows, shared_mxu_elems, dt.w_dense, dt.act)
    s_fp8_min, s_bf16_min = _split_matmul_flops(
        sh_rows, shared_elems, dt.w_dense, dt.act
    )
    rt_fp8, rt_bf16 = _split_matmul_flops(
        eval_context.rows, router_elems, dt.router, dt.act
    )
    moe_colls, moe_ucolls = _repeat_colls(
        _moe_layer_collectives(
            _MoeLayerSpec(
                rows=eval_context.rows,
                hidden=geom.hidden,
                num_experts=geom.num_experts,
                top_k=geom.top_k,
                block_parallel=geom.block_parallel,
            ),
            parallel,
        ),
        max(1, round(n_m_eff)),
    )
    sh_vpu = float(n_m_eff * 3.0 * sh_rows * isr_eff) if isr_eff > 0 else 0.0
    routed_vpu = float(n_m_eff * 3.0 * routed_rows * per_rank_moe_inter * geom.top_k)
    routed_vpu_min = float(
        n_m_eff * 3.0 * routed_rows * (geom.moe_inter / eff_moe_div) * geom.top_k
    )
    fp4_unpack = float(2.0 * routed_elems) if dt.w_expert < 1.0 else 0.0
    fp4_unpack_min = float(2.0 * routed_elems_min) if dt.w_expert < 1.0 else 0.0
    mix_vpu = float(
        n_m_eff
        * (
            2.0 * eval_context.rows * geom.hidden
            + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
        )
    )
    sub_vpu = _subchannel_scale_vpu_flops(
        routed_rows, routed_act_elems, dt.w_expert, geom.weight_block_size
    ) + _subchannel_scale_vpu_flops(
        sh_rows, shared_elems, dt.w_dense, geom.weight_block_size
    )
    sub_vpu_min = _subchannel_scale_vpu_flops(
        routed_rows, routed_act_elems_min, dt.w_expert, geom.weight_block_size
    ) + _subchannel_scale_vpu_flops(
        sh_rows, shared_elems, dt.w_dense, geom.weight_block_size
    )
    shared_unpack = _sub_byte_unpack_vpu_flops(
        shared_elems + latent_proj_sharded, dt.w_dense
    )
    vpu_flops = sh_vpu + routed_vpu + fp4_unpack + shared_unpack + mix_vpu + sub_vpu
    strict_vpu_flops = (
        sh_vpu
        + routed_vpu_min
        + _strict_sub_byte_unpack_vpu_flops(fp4_unpack_min)
        + _strict_sub_byte_unpack_vpu_flops(shared_unpack)
        + mix_vpu
        + sub_vpu_min
    )
    return ModuleCost(
        name="moe",
        hbm_bytes=moe_w_bytes,
        fp8_flops=r_fp8 + s_fp8 + rt_fp8,
        bf16_flops=r_bf16 + s_bf16 + rt_bf16,
        collectives=moe_colls,
        hbm_bytes_min=moe_w_bytes_min,
        unique_collectives=moe_ucolls,
        fp8_flops_min=r_fp8_min + s_fp8_min + rt_fp8,
        bf16_flops_min=r_bf16_min + s_bf16_min + rt_bf16,
        vpu_flops=vpu_flops,
        strict_vpu_flops=strict_vpu_flops,
    )


def _compute_moe_mlp_costs(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
) -> list[ModuleCost]:
    """Compute ModuleCost for dense MLP and routed MoE layers."""
    tp = parallel.tp
    out: list[ModuleCost] = []
    kinds = layer_ffn_kinds(geom)
    n_moe_ffn = kinds.count("moe")
    n_dense_ffn = kinds.count("dense")
    if n_dense_ffn > 0 and geom.dense_inter > 0:
        n_d_eff = n_dense_ffn * eval_context.pp_scale
        mlp_inter = geom.dense_inter // tp
        mlp_elems = float(n_d_eff * 3 * geom.hidden * mlp_inter)
        mlp_mxu_elems = float(
            n_d_eff * 3 * geom.hidden * _align_mxu(mlp_inter, eval_context)
        )
        mlp_fp8, mlp_bf16 = _split_matmul_flops(
            eval_context.rows, mlp_mxu_elems, dt.w_dense, dt.act
        )
        mlp_fp8_min, mlp_bf16_min = _split_matmul_flops(
            eval_context.rows, mlp_elems, dt.w_dense, dt.act
        )
        d_par = (
            Parallel(
                tp=parallel.tp,
                residual=parallel.residual,
                ar_algo=parallel.ar_algo,
                ranks_per_chip=parallel.ranks_per_chip,
                moe="ep" if parallel.residual == "sp" else "tp_slice",
            )
            if geom.num_experts > 0
            else parallel
        )
        mlp_colls = _moe_layer_collectives(
            _MoeLayerSpec(
                rows=eval_context.rows,
                hidden=geom.hidden,
                num_experts=0,
                top_k=0,
                block_parallel=geom.block_parallel,
            ),
            d_par,
        ) * max(1, round(n_d_eff))
        mlp_swiglu = float(n_d_eff * 3.0 * eval_context.rows * mlp_inter)
        mlp_mix = float(
            n_d_eff
            * (
                2.0 * eval_context.rows * geom.hidden
                + 2.0 * eval_context.rows * geom.hidden * geom.hc_mult
            )
        )
        dense_unpack = _sub_byte_unpack_vpu_flops(mlp_elems, dt.w_dense)
        vpu_flops = (
            mlp_swiglu
            + mlp_mix
            + _subchannel_scale_vpu_flops(
                eval_context.rows, mlp_elems, dt.w_dense, geom.weight_block_size
            )
            + dense_unpack
        )
        strict_vpu_flops = (
            vpu_flops - dense_unpack + _strict_sub_byte_unpack_vpu_flops(dense_unpack)
        )
        out.append(
            ModuleCost(
                name="mlp",
                hbm_bytes=mlp_elems * dt.w_dense,
                fp8_flops=mlp_fp8,
                bf16_flops=mlp_bf16,
                collectives=mlp_colls,
                fp8_flops_min=mlp_fp8_min,
                bf16_flops_min=mlp_bf16_min,
                vpu_flops=vpu_flops,
                strict_vpu_flops=strict_vpu_flops,
            )
        )
    if n_moe_ffn > 0:
        n_m_eff = n_moe_ffn * eval_context.pp_scale
        out.append(
            _compute_routed_moe_cost(
                geom,
                parallel,
                dt,
                eval_context,
                n_m_eff=n_m_eff,
            )
        )
    return out


def _compute_ple_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
) -> ModuleCost | None:
    """Compute ModuleCost for Per-Layer Embedding (PLE)."""
    if "ple" not in geom.layer_types and geom.ple_dim <= 0:
        return None
    if geom.ple_dim <= 0:
        msg = "PLE layer requires config field 'ple_dim'"
        raise ValueError(msg)
    tp = parallel.tp
    ple_d = geom.ple_dim
    n_ple = geom.num_layers * eval_context.pp_scale
    ple_proj_elems = float(n_ple * geom.hidden * ple_d // tp)
    ple_lookup_bytes = float(n_ple * eval_context.rows * ple_d) * dt.w_aux
    ple_fp8, ple_bf16 = _split_matmul_flops(
        eval_context.rows, ple_proj_elems, dt.w_aux, dt.act
    )
    return ModuleCost(
        name="ple",
        hbm_bytes=ple_proj_elems * dt.w_aux + ple_lookup_bytes,
        fp8_flops=ple_fp8,
        bf16_flops=ple_bf16,
        vpu_flops=0.0,
        strict_vpu_flops=0.0,
    )


def _compute_lm_head_cost(
    geom: ModelGeometry,
    parallel: Parallel,
    dt: Dtypes,
    eval_context: _StepContext,
) -> ModuleCost:
    """Compute ModuleCost for LM head and PP handoff collectives."""
    tp = parallel.tp
    pp_colls = (
        (
            _make_coll(
                "pp_p2p",
                "p2p",
                float(eval_context.rows * geom.hidden * 2),
                parallel.pp,
                parallel,
            ),
        )
        if parallel.pp > 1
        else ()
    )
    lm_elems = float(geom.vocab * geom.hidden // tp) / float(max(1, parallel.pp))
    lm_fp8, lm_bf16 = _split_matmul_flops(
        eval_context.rows, lm_elems, dt.lm_head, dt.act
    )
    lm_all_colls = (
        _lm_head_collectives(eval_context.rows, geom.hidden, geom.vocab, parallel)
        + pp_colls
    )
    lm_colls, lm_ucolls = _repeat_colls(lm_all_colls, 1)
    vpu_flops = float(2.0 * eval_context.rows * geom.hidden)
    return ModuleCost(
        name="lm_head",
        hbm_bytes=lm_elems * dt.lm_head,
        fp8_flops=lm_fp8,
        bf16_flops=lm_bf16,
        collectives=lm_colls,
        unique_collectives=lm_ucolls,
        vpu_flops=vpu_flops,
        strict_vpu_flops=vpu_flops,
    )


def pass_costs(  # noqa: PLR0913
    geom: ModelGeometry,
    parallel: Parallel,
    shape: StepShape,
    dt: Dtypes,
    *,
    mxu_m_tile: int = 1,
    mxu_lhs_rows_bf16: int = 16,
    mxu_lhs_rows_fp8: int = 32,
) -> tuple[ModuleCost, ...]:
    """Compute per-module resource work for one decode, prefill, or mixed pass."""
    _validate_pass_costs_config(geom, parallel)

    pp_scale = (
        float(math.ceil(geom.num_layers / max(1, parallel.pp))) / float(geom.num_layers)
        if geom.num_layers > 0 and parallel.pp > 1
        else 1.0
    )
    decode_rows = shape.batch * shape.rows_per_seq
    eff_prefill = (
        max(
            1,
            math.ceil(
                shape.prefill_tokens
                * max(0.0, 1.0 - shape.prefix_hit_rate)
                / max(1, parallel.pcp)
            ),
        )
        if shape.prefill_tokens > 0
        else 0
    )
    cached_prefix = (
        round(shape.prefill_tokens * shape.prefix_hit_rate)
        if shape.prefill_tokens > 0
        else 0
    )
    rows = decode_rows + eff_prefill

    eval_context = _StepContext(
        rows=rows,
        decode_rows=decode_rows,
        eff_prefill=eff_prefill,
        cached_prefix=cached_prefix,
        pp_scale=pp_scale,
        context=shape.context,
        batch=shape.batch,
        rows_per_seq=shape.rows_per_seq,
        state_writes=shape.state_writes,
        mxu_m_tile=mxu_m_tile,
        mxu_lhs_rows_bf16=mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=mxu_lhs_rows_fp8,
    )

    counts = count_layer_types(geom)
    emb_colls, emb_ucolls = _repeat_colls(
        _embed_collectives(rows, geom.hidden, parallel), 1
    )
    out: list[ModuleCost] = [
        ModuleCost(
            name="embed",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=emb_colls,
            unique_collectives=emb_ucolls,
            vpu_flops=0.0,
            strict_vpu_flops=0.0,
        )
    ]

    if counts.gqa > 0:
        out.append(_compute_gqa_cost(geom, parallel, dt, eval_context, counts))

    if counts.swa > 0:
        out.append(
            _compute_swa_cost(geom, parallel, dt, eval_context, n_swa=counts.swa)
        )

    if counts.mla > 0 or counts.dsa > 0:
        out.extend(_compute_mla_dsa_costs(geom, parallel, dt, eval_context, counts))

    if counts.gdn > 0:
        out.append(
            _compute_gdn_cost(geom, parallel, dt, eval_context, n_gdn=counts.gdn)
        )

    if counts.kda > 0:
        out.append(
            _compute_kda_cost(geom, parallel, dt, eval_context, n_kda=counts.kda)
        )

    out.extend(_compute_moe_mlp_costs(geom, parallel, dt, eval_context))

    ple = _compute_ple_cost(geom, parallel, dt, eval_context)
    if ple is not None:
        out.append(ple)

    out.append(_compute_lm_head_cost(geom, parallel, dt, eval_context))

    return tuple(out)
