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
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Literal, TypedDict, Unpack

from .chips import CHIPS, ChipSpec
from .collectives import Algo, Collective, CollectiveConfig, Kind, make
from .constants import (
    ACTIVATION_BYTES_BF16,
    BF16_BYTES,
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_SSM_STATE_BYTES,
    FP8_BYTES,
    FP32_BYTES,
    GLU_MATMUL_COUNT,
    HBM_MIN_PACKING_DIVISOR,
    HBM_PACKING_NUMERATOR,
    HEAD_DIM_PACKING_THRESHOLD,
    MATMUL_FLOP_PER_MAC,
    NON_GATED_MATMUL_COUNT,
    SPARSE_MLA_SCALE_BYTES,
    SUBCHANNEL_BLOCK_K_FLOOR,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from .geometry import (
    LayerCounts,
    ModelGeometry,
    count_layer_types,
    layer_ffn_kinds,
)
from .occupancy import (
    ExpertActivationModel,
    ExpertDispatch,
    active_experts_envelope,
    bernoulli_expected_experts,
    ep_busiest_experts,
    min_experts_per_ep_shard,
)

_TWO_DIRECTIONS = 2
_GQA_SCORE_WIDTH_MULT = 4
# Distributed sampling gathers one (float32 value, int32 index) scalar pair per row.
_DIST_SAMPLING_SCALAR_BYTES: float = 2.0 * FP32_BYTES


@dataclass(frozen=True, slots=True)
class Dtypes:
    """Element widths in bytes for each tensor role and activation peak."""

    w_dense: float = FP8_BYTES
    w_expert: float = FP8_BYTES
    w_aux: float = BF16_BYTES
    kv: float = FP8_BYTES
    state: float = DEFAULT_SSM_STATE_BYTES
    conv: float = BF16_BYTES
    router: float = BF16_BYTES
    lm_head: float = BF16_BYTES
    act: Literal["a4", "a8", "a16"] = "a8"
    w_dense_source: Literal["quantization_config", "flag", "default"] = "default"
    w_expert_source: Literal["quantization_config", "flag", "default"] = "default"
    kv_source: Literal["quantization_config", "flag", "default"] = "default"
    state_source: Literal["quantization_config", "config", "flag", "default"] = (
        "default"
    )
    w_dense_dtype: str = "fp8"
    w_expert_dtype: str = "fp8"
    kv_dtype: str = "fp8"
    state_dtype: str = "float32"


def activation_wire_bytes(act: str) -> float:
    """Return activation wire element width in bytes for collective traffic.

    Both ``a4`` and ``a8`` map to ``FP8_BYTES`` (conservative public wire floor
    when FP4 collective wire dispatch is unverified), keeping ``a4`` floors
    at or below ``a8`` floors.
    """
    if act in ("a4", "a8"):
        return FP8_BYTES
    if act == "a16":
        return ACTIVATION_BYTES_BF16
    msg = f"Unsupported activation mode {act!r}; expected 'a4', 'a8', or 'a16'"
    raise ValueError(msg)


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
    ranks_per_chip: int | None = None
    dp: int = 1
    ep: int = 1
    pp: int = 1
    pcp: int = 1
    mesh_axes: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class StepShape:
    """Per-step batch size, token rows per sequence, context, and prefill tokens."""

    batch: int = 1
    rows_per_seq: int = 1
    context: int = 0
    state_writes: int = 1
    prefill_tokens: int = 0
    prefix_hit_rate: float = 0.0
    shared_prefix_tokens: int = 0


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
    vector_flops: float = 0.0
    strict_vector_flops: float = 0.0
    hbm_bytes_envelope: float | None = None
    fp8_flops_envelope: float | None = None
    bf16_flops_envelope: float | None = None
    fp4_flops: float = 0.0
    int8_flops: float = 0.0
    int4_flops: float = 0.0
    fp4_flops_min: float | None = None
    int8_flops_min: float | None = None
    int4_flops_min: float | None = None
    fp4_flops_envelope: float | None = None
    int8_flops_envelope: float | None = None
    int4_flops_envelope: float | None = None
    strict_collectives: tuple[tuple[Collective, int], ...] | None = None

    @property
    def ici_bytes(self) -> float:
        """Total inter-chip interconnect wire bytes across off-chip collectives."""
        if self.unique_collectives is not None:
            return sum(
                collective.wire_bytes * collective.offchip * call_count
                for collective, call_count in self.unique_collectives
            )
        return sum(
            collective.wire_bytes * collective.offchip
            for collective in self.collectives
        )

    @property
    def n_collectives(self) -> int:
        """Number of collective communications required by this module."""
        return len(self.collectives)

    @property
    def chain_hops(self) -> int:
        """Total network chain hops incurred by all collectives in this module."""
        return sum(collective.hops for collective in self.collectives)

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

    @property
    def strict_fp4_flops(self) -> float:
        """Unpadded lower-bound FP4 FLOPs when available, else nominal FP4 FLOPs."""
        return self.fp4_flops if self.fp4_flops_min is None else self.fp4_flops_min

    @property
    def strict_int8_flops(self) -> float:
        """Unpadded lower-bound INT8 FLOPs when available, else nominal INT8 FLOPs."""
        return self.int8_flops if self.int8_flops_min is None else self.int8_flops_min

    @property
    def strict_int4_flops(self) -> float:
        """Unpadded lower-bound INT4 FLOPs when available, else nominal INT4 FLOPs."""
        return self.int4_flops if self.int4_flops_min is None else self.int4_flops_min

    @property
    def envelope_hbm_bytes(self) -> float:
        """Activation-envelope byte count when available, else strict bytes."""
        return (
            self.strict_hbm_bytes
            if self.hbm_bytes_envelope is None
            else self.hbm_bytes_envelope
        )

    @property
    def envelope_fp8_flops(self) -> float:
        """Activation-envelope FP8 FLOPs when available, else strict FP8 FLOPs."""
        return (
            self.strict_fp8_flops
            if self.fp8_flops_envelope is None
            else self.fp8_flops_envelope
        )

    @property
    def envelope_bf16_flops(self) -> float:
        """Activation-envelope BF16 FLOPs when available, else strict BF16 FLOPs."""
        return (
            self.strict_bf16_flops
            if self.bf16_flops_envelope is None
            else self.bf16_flops_envelope
        )

    @property
    def envelope_fp4_flops(self) -> float:
        """Activation-envelope FP4 FLOPs when available, else strict FP4 FLOPs."""
        return (
            self.strict_fp4_flops
            if self.fp4_flops_envelope is None
            else self.fp4_flops_envelope
        )

    @property
    def envelope_int8_flops(self) -> float:
        """Activation-envelope INT8 FLOPs when available, else strict INT8 FLOPs."""
        return (
            self.strict_int8_flops
            if self.int8_flops_envelope is None
            else self.int8_flops_envelope
        )

    @property
    def envelope_int4_flops(self) -> float:
        """Activation-envelope INT4 FLOPs when available, else strict INT4 FLOPs."""
        return (
            self.strict_int4_flops
            if self.int4_flops_envelope is None
            else self.int4_flops_envelope
        )


def experts_touched(num_experts: int, top_k: int, rows: int) -> float:
    """Expected distinct routed experts touched under uniform routing."""
    return bernoulli_expected_experts(num_experts, top_k, rows)


@dataclass(frozen=True, slots=True)
class _MatmulBuckets:
    """Per-precision matrix-unit FLOP totals across supported hardware buckets."""

    fp8: float = 0.0
    bf16: float = 0.0
    int8: float = 0.0
    fp4: float = 0.0
    int4: float = 0.0

    def __add__(self, other: _MatmulBuckets) -> _MatmulBuckets:
        """Sum matching precision buckets elementwise."""
        return _MatmulBuckets(
            fp8=self.fp8 + other.fp8,
            bf16=self.bf16 + other.bf16,
            int8=self.int8 + other.int8,
            fp4=self.fp4 + other.fp4,
            int4=self.int4 + other.int4,
        )

    def scale(self, factor: float) -> _MatmulBuckets:
        """Multiply every precision bucket by a scalar padding factor."""
        return _MatmulBuckets(
            fp8=self.fp8 * factor,
            bf16=self.bf16 * factor,
            int8=self.int8 * factor,
            fp4=self.fp4 * factor,
            int4=self.int4 * factor,
        )


def _bucket_from_name(bucket: str, flops: float) -> _MatmulBuckets:
    """Construct a _MatmulBuckets placing ``flops`` in ``bucket``."""
    if bucket == "fp8":
        return _MatmulBuckets(fp8=flops)
    if bucket == "int8":
        return _MatmulBuckets(int8=flops)
    if bucket == "fp4":
        return _MatmulBuckets(fp4=flops)
    if bucket == "int4":
        return _MatmulBuckets(int4=flops)
    return _MatmulBuckets(bf16=flops)


class _RouteMatmulOpts(TypedDict, total=False):
    """Optional chip capability spec and weight dtype for matmul FLOP routing."""

    chip: ChipSpec | None
    weight_dtype: str


def _route_matmul_flops(
    rows: float,
    weight_elems: float,
    weight_bytes: float,
    act: str = "a8",
    **opts: Unpack[_RouteMatmulOpts],
) -> _MatmulBuckets:
    """Route a matmul to its capability-selected matrix-unit precision bucket."""
    chip = opts.get("chip")
    weight_dtype = opts.get("weight_dtype", "")
    eff_chip = chip if chip is not None else CHIPS["tpu7x"]
    bucket, _peak, _needs_unpack = eff_chip.matmul_peak_kind(
        weight_bytes, act, weight_dtype=weight_dtype
    )
    flops = MATMUL_FLOP_PER_MAC * rows * weight_elems
    return _bucket_from_name(bucket, flops)


def _subchannel_scale_vector_flops(
    rows: int,
    weight_elems: float,
    weight_bytes: float,
    weight_block_size: tuple[int, int] = (1, 1),
    *,
    block_k_floor: int = SUBCHANNEL_BLOCK_K_FLOOR,
) -> float:
    """Return vector partial-sum scale multiplication FLOPs below block_k_floor."""
    if weight_bytes >= BF16_BYTES or rows <= 0 or weight_elems <= 0.0:
        return 0.0
    block_k = (
        weight_block_size[1]
        if len(weight_block_size) > 1 and weight_block_size[1] > 1
        else (weight_block_size[0] if weight_block_size else 1)
    )
    if 1 < block_k < block_k_floor:
        return float(MATMUL_FLOP_PER_MAC * rows * weight_elems) / float(block_k)
    return 0.0


def _sub_byte_unpack_vector_flops(
    weight_elems: float,
    weight_bytes: float,
    *,
    chip: ChipSpec | None = None,
    weight_dtype: str = "",
    act: str = "a8",
) -> float:
    """Return vector unpack FLOPs only when chip capabilities require unpacking."""
    eff_chip = chip if chip is not None else CHIPS["tpu7x"]
    _bucket, _peak, needs_unpack = eff_chip.matmul_peak_kind(
        weight_bytes, act, weight_dtype=weight_dtype
    )
    return 2.0 * weight_elems if needs_unpack else 0.0


def _strict_sub_byte_unpack_vector_flops(unpack_vector_flops: float) -> float:
    """Scale an unpack count for the strict vector bound.

    The bound divides strict counts by VECTOR_MAX_ELEMENTS_PER_WORD. Weights below
    one byte pack VECTOR_SUB_BYTE_ELEMENTS_PER_WORD per word, so their unpack count
    is scaled by the ratio of the two.
    """
    return (
        unpack_vector_flops
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
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else DEFAULT_CORES_PER_CHIP
    )
    return make(
        name,
        kind,
        payload,
        group,
        CollectiveConfig(algo=parallel.ar_algo, ranks_per_chip=eff_ranks),
    )


def _embed_collectives(
    rows: int, hidden: int, parallel: Parallel
) -> tuple[Collective, ...]:
    """Return TP collectives for embedding lookup (RS under SP, 1-phase AG else)."""
    if parallel.tp <= 1 or hidden <= 0:
        return ()
    payload = float(rows * hidden * ACTIVATION_BYTES_BF16)
    if parallel.residual == "sp":
        return (_make_coll("embed_rs", "rs", payload, parallel.tp, parallel),)
    return (_make_coll("embed_ag", "ag", payload, parallel.tp, parallel),)


def _repeat_colls(
    base: tuple[Collective, ...], count: int
) -> tuple[tuple[Collective, ...], tuple[tuple[Collective, int], ...]]:
    """Repeat base collectives count times and build unique collective tally."""
    if not base or count <= 0:
        return (), ()
    return base * count, tuple([(collective, count) for collective in base])


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
    shared_prefix_tokens: int = 0
    mxu_m_tile: int = 1
    mxu_lhs_rows_bf16: int | None = None
    mxu_lhs_rows_fp8: int | None = None
    fp8_to_bf16_ratio: float = 0.0
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )
    chip: ChipSpec = field(default_factory=lambda: CHIPS["tpu7x"])


@dataclass(frozen=True, slots=True)
class _AttnCollSpec:
    """Layer collective parameters for one attention or recurrence block."""

    batch: int
    rows_per_seq: int
    hidden: int
    num_q_heads: int
    head_dim: int
    is_full_attention: bool = True
    extra_rows: int = 0
    block_parallel: int = 1
    kv_bytes_per_tok: float = 0.0
    act_bytes: float = ACTIVATION_BYTES_BF16
    q_head_dim: int = 0
    cached_prefix: int = 0
    sliding_window: int = 0


@dataclass(frozen=True, slots=True)
class _MoeLayerSpec:
    """Layer dimensions for MoE/MLP collective derivation."""

    rows: int
    hidden: int
    num_experts: int
    top_k: int
    block_parallel: int = 1
    decode_rows: int = 0
    eff_prefill: int = 0
    dispatch_bytes: float = ACTIVATION_BYTES_BF16
    shared_inter: int = 0
    has_tp_latent_proj: bool = False


def _dcp_collectives(
    spec: _AttnCollSpec,
    parallel: Parallel,
    kv_payload_per_tok: float,
) -> list[Collective]:
    """Return physics-minimum DCP collectives for decode and cached prefill."""
    g = parallel.kv_split
    dec_rows = spec.batch * spec.rows_per_seq
    q_hd = spec.q_head_dim if spec.q_head_dim > 0 else spec.head_dim
    out: list[Collective] = []
    if dec_rows > 0:
        if parallel.kv_merge == "ag_rs":
            out.append(
                _make_coll(
                    "dcp_q_ag",
                    "ag",
                    float(dec_rows * g * spec.num_q_heads * q_hd)
                    * ACTIVATION_BYTES_BF16,
                    g,
                    parallel,
                )
            )
            out.append(
                _make_coll(
                    "dcp_lse_ag",
                    "ag",
                    float(g * dec_rows * spec.num_q_heads) * FP32_BYTES,
                    g,
                    parallel,
                )
            )
            out.append(
                _make_coll(
                    "dcp_out_rs",
                    "rs",
                    float(dec_rows * g * spec.num_q_heads * spec.head_dim)
                    * ACTIVATION_BYTES_BF16,
                    g,
                    parallel,
                )
            )
        else:
            per_peer = float(dec_rows * spec.num_q_heads) * (
                float(spec.head_dim) * ACTIVATION_BYTES_BF16 + FP32_BYTES
            )
            out.append(
                _make_coll("merge_xchg", "push", (g - 1) * per_peer, g, parallel)
            )
    if spec.extra_rows > 0 and spec.cached_prefix > 0:
        eff_cached_toks = float(
            min(spec.cached_prefix, spec.sliding_window)
            if spec.sliding_window > 0
            else spec.cached_prefix
        ) / float(max(1, parallel.pcp))
        kv_gather_payload = eff_cached_toks * kv_payload_per_tok
        merge_out_payload = (
            float(spec.extra_rows * g * spec.num_q_heads * spec.head_dim)
            * ACTIVATION_BYTES_BF16
        )
        merge_lse_payload = float(g * spec.extra_rows * spec.num_q_heads) * FP32_BYTES
        if kv_gather_payload <= merge_out_payload + merge_lse_payload:
            out.append(_make_coll("dcp_kv_ag", "ag", kv_gather_payload, g, parallel))
        else:
            out.append(_make_coll("dcp_lse_ag", "ag", merge_lse_payload, g, parallel))
            out.append(_make_coll("dcp_out_rs", "rs", merge_out_payload, g, parallel))
    return out


def _attn_layer_collectives(
    spec: _AttnCollSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Return physics-minimum communication collectives for one attention layer."""
    rows = spec.batch * spec.rows_per_seq + spec.extra_rows
    s_res_bytes = float(rows * spec.hidden) * ACTIVATION_BYTES_BF16
    s_act_bytes = float(rows * spec.hidden) * spec.act_bytes
    colls: list[Collective] = []
    if parallel.tp > 1:
        if parallel.residual == "replicated":
            colls.append(_make_coll("out_ar", "ar", s_res_bytes, parallel.tp, parallel))
        else:
            colls.append(
                _make_coll("entry_ag", "ag", s_act_bytes, parallel.tp, parallel)
            )
            if not (spec.block_parallel > 1 and parallel.moe_dispatch == "allreduce"):
                colls.append(
                    _make_coll("out_rs", "rs", s_res_bytes, parallel.tp, parallel)
                )
    kv_payload_per_tok = (
        spec.kv_bytes_per_tok
        if spec.kv_bytes_per_tok > 0.0
        else float(max(1, spec.num_q_heads) * spec.head_dim) * ACTIVATION_BYTES_BF16
    )
    if spec.is_full_attention and parallel.kv_split > 1:
        colls.extend(_dcp_collectives(spec, parallel, kv_payload_per_tok))
    if spec.is_full_attention and parallel.pcp > 1 and spec.extra_rows > 0:
        total_kv_toks = float(spec.cached_prefix + spec.extra_rows * parallel.pcp)
        eff_pcp_toks = (
            min(total_kv_toks, float(spec.sliding_window))
            if spec.sliding_window > 0
            else total_kv_toks
        )
        colls.append(
            _make_coll(
                "pcp_kv_ag",
                "ag",
                eff_pcp_toks * kv_payload_per_tok,
                parallel.pcp,
                parallel,
            )
        )
    return tuple(colls)


def _ep_local_dims(parallel: Parallel, ep_ranks: int) -> tuple[int, int]:
    """Return (pcp_per_ep, dp_per_ep) subgroup factors inside one EP group."""
    if parallel.moe != "ep" or parallel.ep <= 1:
        return (1, 1)
    pcp_per_ep = min(max(1, parallel.pcp), max(1, ep_ranks // max(1, parallel.tp)))
    dp_per_ep = min(
        max(1, parallel.dp),
        max(1, ep_ranks // max(1, parallel.tp * pcp_per_ep)),
    )
    return (pcp_per_ep, dp_per_ep)


def _moe_ep_global_rows(
    spec: _MoeLayerSpec,
    parallel: Parallel,
    ep_ranks: int,
) -> int:
    """Return EP-group global token rows using exact pcp_per_ep and dp_per_ep."""
    pcp_per_ep, dp_per_ep = _ep_local_dims(parallel, ep_ranks)
    if parallel.moe != "ep" or parallel.ep <= 1:
        return spec.rows
    if spec.decode_rows > 0 or spec.eff_prefill > 0:
        return (spec.decode_rows + spec.eff_prefill * pcp_per_ep) * dp_per_ep
    return spec.rows * pcp_per_ep * dp_per_ep


def _tp_reduction_collectives(
    spec: _MoeLayerSpec,
    group: int,
    parallel: Parallel,
) -> list[Collective]:
    """Return mandatory TP partial-sum reduction collectives across group ranks."""
    if group <= 1:
        return []
    resident_batch = ACTIVATION_BYTES_BF16
    b_act = spec.dispatch_bytes
    if parallel.residual == "replicated":
        if spec.block_parallel <= 1:
            return [
                _make_coll(
                    "moe_ar",
                    "ar",
                    float(spec.rows * spec.hidden) * resident_batch,
                    group,
                    parallel,
                )
            ]
        return []
    colls: list[Collective] = []
    if spec.block_parallel <= 1:
        colls.append(
            _make_coll(
                "dispatch_ag",
                "ag",
                float(spec.rows * spec.hidden) * b_act,
                group,
                parallel,
            )
        )
    colls.append(
        _make_coll(
            "combine_rs",
            "rs",
            float(spec.rows * spec.hidden) * resident_batch,
            group,
            parallel,
        )
    )
    return colls


def _moe_layer_collectives(
    spec: _MoeLayerSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Derive physics-minimum collectives forced by MoE or dense MLP sharding."""
    tp = parallel.tp
    ep_ranks = (parallel.ep if parallel.ep > 1 else tp) if parallel.moe == "ep" else 1
    is_ep = parallel.moe == "ep" and ep_ranks > 1 and spec.num_experts > 0
    if tp <= 1 and ep_ranks <= 1:
        return ()
    if not is_ep:
        return tuple(_tp_reduction_collectives(spec, tp, parallel))

    colls: list[Collective] = []
    experts_per_rank = max(1, spec.num_experts // ep_ranks)
    remote_peers = max(0, math.ceil(spec.top_k / experts_per_rank) - 1)
    if remote_peers > 0:
        global_rows = _moe_ep_global_rows(spec, parallel, ep_ranks)
        local_rows = float(global_rows) / float(ep_ranks)
        push_elems = local_rows * float(remote_peers * spec.hidden)
        colls.append(
            _make_coll(
                "dispatch_push",
                "push",
                push_elems * spec.dispatch_bytes,
                ep_ranks,
                parallel,
            )
        )
        colls.append(
            _make_coll(
                "combine_push",
                "push",
                push_elems * ACTIVATION_BYTES_BF16,
                ep_ranks,
                parallel,
            )
        )

    tp_sub = max(1, tp // ep_ranks) if ep_ranks < tp else 1
    needs_full_tp = tp > 1 and (
        (parallel.residual == "replicated" and spec.shared_inter > 0)
        or spec.has_tp_latent_proj
    )
    reduce_group = tp if needs_full_tp else tp_sub
    colls.extend(_tp_reduction_collectives(spec, reduce_group, parallel))
    return tuple(colls)


def _moe_layer_strict_collectives(
    spec: _MoeLayerSpec,
    parallel: Parallel,
) -> tuple[Collective, ...]:
    """Derive strict chip-level pigeonhole MoE collectives for hard lower bounds."""
    tp = parallel.tp
    ep_ranks = (parallel.ep if parallel.ep > 1 else tp) if parallel.moe == "ep" else 1
    is_ep = parallel.moe == "ep" and ep_ranks > 1 and spec.num_experts > 0
    if tp <= 1 and ep_ranks <= 1:
        return ()
    if not is_ep:
        return tuple(_tp_reduction_collectives(spec, tp, parallel))

    colls: list[Collective] = []
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else DEFAULT_CORES_PER_CHIP
    )
    local_ranks = min(ep_ranks, max(1, eff_ranks))
    experts_per_rank = max(1, spec.num_experts // ep_ranks)
    experts_per_chip = experts_per_rank * local_ranks
    global_rows = _moe_ep_global_rows(spec, parallel, ep_ranks)
    local_rows = float(global_rows) / float(ep_ranks)

    # Intra-chip D2D pigeonhole push when top_k exceeds one TensorCore's experts.
    local_remote_peers = min(
        local_ranks - 1, max(0, math.ceil(spec.top_k / experts_per_rank) - 1)
    )
    if local_ranks > 1 and local_remote_peers > 0:
        d2d_elems = local_rows * float(local_remote_peers * spec.hidden)
        colls.append(
            _make_coll(
                "dispatch_d2d_push",
                "push",
                d2d_elems * spec.dispatch_bytes,
                local_ranks,
                parallel,
            )
        )
        colls.append(
            _make_coll(
                "combine_d2d_push",
                "push",
                d2d_elems * ACTIVATION_BYTES_BF16,
                local_ranks,
                parallel,
            )
        )

    # Off-chip vertex-only pigeonhole push only when top_k exceeds experts per chip.
    remote_chips = max(0, math.ceil(spec.top_k / experts_per_chip) - 1)
    if ep_ranks > local_ranks and remote_chips > 0:
        chip_push_elems = (
            float(local_ranks) * local_rows * float(remote_chips * spec.hidden)
        )
        colls.append(
            _make_coll(
                "dispatch_vertex_push",
                "push",
                chip_push_elems * spec.dispatch_bytes,
                ep_ranks,
                parallel,
            )
        )
        colls.append(
            _make_coll(
                "combine_vertex_push",
                "push",
                chip_push_elems * ACTIVATION_BYTES_BF16,
                ep_ranks,
                parallel,
            )
        )

    tp_sub = max(1, tp // ep_ranks) if ep_ranks < tp else 1
    needs_full_tp = tp > 1 and (
        (parallel.residual == "replicated" and spec.shared_inter > 0)
        or spec.has_tp_latent_proj
    )
    reduce_group = tp if needs_full_tp else tp_sub
    colls.extend(_tp_reduction_collectives(spec, reduce_group, parallel))
    return tuple(colls)


def _lm_head_collectives(
    rows: int,
    hidden: int,
    vocab: int,
    parallel: Parallel,
    *,
    act_bytes: float = ACTIVATION_BYTES_BF16,
) -> tuple[Collective, ...]:
    """Return physics-minimum communication collectives for language model head."""
    if parallel.tp <= 1:
        return ()
    tp = parallel.tp
    colls: list[Collective] = []
    if parallel.residual == "sp" and hidden > 0:
        colls.append(
            _make_coll("tail_ag", "ag", float(rows * hidden) * act_bytes, tp, parallel)
        )
    if vocab > 0:
        colls.append(
            _make_coll(
                "logits_ag",
                "ag",
                float(rows * tp) * _DIST_SAMPLING_SCALAR_BYTES,
                tp,
                parallel,
            )
        )
    return tuple(colls)


def _validate_pass_costs_config(geometry: ModelGeometry, parallel: Parallel) -> None:
    """Validate parallelism and geometry constraints for pass_costs."""
    tp = parallel.tp
    if tp < 1:
        msg = f"Parallel.tp must be >= 1, got {tp}"
        raise ValueError(msg)
    if geometry.num_heads > 0 and tp > geometry.num_heads:
        msg = f"Parallel.tp={tp} exceeds geometry.num_heads={geometry.num_heads}"
        raise ValueError(msg)
    kv_group = (
        max(1, tp // geometry.num_kv_heads)
        if geometry.num_kv_heads > 0 and geometry.kv_lora_rank == 0
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
    if parallel.residual == "sp" and parallel.moe != "ep" and geometry.num_experts > 0:
        msg = "Parallel.residual='sp' requires Parallel.moe='ep'"
        raise ValueError(msg)
    if parallel.moe_dispatch in ("a2a_push", "ag_ep") and parallel.moe != "ep":
        disp = parallel.moe_dispatch
        msg = f"Parallel.moe_dispatch={disp!r} requires Parallel.moe='ep'"
        raise ValueError(msg)


class _KvBytesOptions(TypedDict, total=False):
    """Optional keyword flags for :func:`kv_bytes_per_token_per_layer`."""

    physical: bool
    chip: ChipSpec | None


def _gqa_swa_kv_bytes_per_tok(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    head_dim: int,
    *,
    physical_align: int = 0,
) -> float:
    """Compute per-token per-layer KV bytes for GQA or SWA head geometry."""
    tp = max(1, parallel.tp)
    num_kv_heads = (
        max(1, geometry.num_kv_heads // tp) if geometry.num_kv_heads > 0 else 0
    )
    if physical_align <= 0:
        return float(2 * num_kv_heads * head_dim) * dtypes.kv
    eff_align = physical_align
    kv_packing = max(
        1,
        round(HBM_PACKING_NUMERATOR / max(HBM_MIN_PACKING_DIVISOR, dtypes.kv)),
    )
    padded_hd = math.ceil(head_dim / eff_align) * eff_align if head_dim > 0 else 0
    can_pack_kv_pair = (
        head_dim > 0
        and 2 * head_dim <= eff_align
        and (head_dim == HEAD_DIM_PACKING_THRESHOLD or 2 * head_dim == padded_hd)
    )
    kv_row_count = num_kv_heads if can_pack_kv_pair else 2 * num_kv_heads
    padded_kv_rows = (
        math.ceil(kv_row_count / kv_packing) * kv_packing if kv_row_count > 0 else 0
    )
    return float(padded_kv_rows * padded_hd) * dtypes.kv


@lru_cache(maxsize=4096)
def _kv_bytes_per_token_per_layer_cached(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    layer_type: str,
    physical: bool,
    align: int,
) -> float:
    """Cached implementation of kv_bytes_per_token_per_layer over value tuples."""
    resolved = layer_type
    if resolved in ("full_attention", "attention"):
        if geometry.kv_lora_rank > 0 and geometry.index_topk > 0:
            resolved = "dsa"
        elif geometry.kv_lora_rank > 0:
            resolved = "mla"
        else:
            resolved = "gqa"

    if resolved == "gqa":
        gqa_hd = (
            geometry.global_head_dim
            if geometry.global_head_dim > 0
            else geometry.head_dim
        )
        return _gqa_swa_kv_bytes_per_tok(
            geometry, dtypes, parallel, gqa_hd, physical_align=align if physical else 0
        )
    if resolved == "swa":
        return _gqa_swa_kv_bytes_per_tok(
            geometry,
            dtypes,
            parallel,
            geometry.head_dim,
            physical_align=align if physical else 0,
        )
    if resolved in ("mla", "dsa"):
        raw_latent_w = geometry.kv_lora_rank + geometry.qk_rope_head_dim
        latent_w = math.ceil(raw_latent_w / align) * align if physical else raw_latent_w
        mla_latent_b = float(latent_w) * dtypes.kv
        has_sparse = (
            resolved == "dsa"
            and geometry.index_topk > 0
            and geometry.index_head_dim > 0
        )
        idx_scale = (
            SPARSE_MLA_SCALE_BYTES if (physical and dtypes.kv <= FP8_BYTES) else 0.0
        )
        dsa_idx_b = (
            float(geometry.index_head_dim) * dtypes.kv + idx_scale
            if has_sparse
            else 0.0
        )
        return mla_latent_b + dsa_idx_b

    msg = f"Unknown attention layer type: {layer_type}"
    raise ValueError(msg)


def kv_bytes_per_token_per_layer(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    layer_type: str = "gqa",
    **options: Unpack[_KvBytesOptions],
) -> float:
    """Return KV cache bytes stored per token for one layer of given attention type."""
    physical = options.get("physical", True)
    chip = options.get("chip")
    align = (
        chip.physical_alignment_elems
        if chip is not None
        else CHIPS["tpu7x"].physical_alignment_elems
    )
    return _kv_bytes_per_token_per_layer_cached(
        geometry, dtypes, parallel, layer_type, physical, align
    )


def _align_mxu(dim: int, eval_context: _StepContext) -> int:
    """Pad systolic column dimension to physical alignment when tiling is active."""
    if dim <= 0 or eval_context.mxu_m_tile <= 1:
        return dim
    align = eval_context.chip.physical_alignment_elems
    return math.ceil(dim / align) * align


def _compute_gqa_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    counts: LayerCounts,
) -> ModuleCost:
    """Compute ModuleCost for GQA attention."""
    n_gqa = counts.gqa
    tp = parallel.tp
    n_eff = n_gqa * eval_context.pp_scale
    num_kv_heads = (
        max(1, geometry.num_kv_heads // tp) if geometry.num_kv_heads > 0 else 0
    )
    q_mult = 2 if geometry.attn_output_gate else 1
    gqa_hd = (
        geometry.global_head_dim if geometry.global_head_dim > 0 else geometry.head_dim
    )
    q_out = geometry.num_heads * gqa_hd * q_mult
    q_sharded = q_out // tp
    kv_sharded = num_kv_heads * gqa_hd
    out_sharded = (geometry.num_heads // tp) * gqa_hd
    att_w_elems = float(
        n_eff
        * (
            geometry.hidden * q_sharded
            + 2 * geometry.hidden * kv_sharded
            + out_sharded * geometry.hidden
        )
    )
    att_mxu_elems = float(
        n_eff
        * geometry.hidden
        * (
            _align_mxu(q_sharded, eval_context)
            + 2 * _align_mxu(kv_sharded, eval_context)
            + _align_mxu(out_sharded, eval_context)
        )
    )
    gqa_kv_scale = (counts.eff_gqa / float(n_gqa)) if geometry.compress_ratios else 1.0
    kv_split_div = float(max(1, parallel.kv_split))
    cp_div = float(max(1, parallel.pcp * parallel.kv_split))
    kv_dec_tokens = (
        float(eval_context.batch * eval_context.context) * gqa_kv_scale / kv_split_div
    )
    c_shared = min(eval_context.context, max(0, eval_context.shared_prefix_tokens))
    kv_dec_tokens_min = (
        float(c_shared + eval_context.batch * max(0, eval_context.context - c_shared))
        * gqa_kv_scale
        / kv_split_div
        if (eval_context.batch > 0 and c_shared > 0)
        else kv_dec_tokens
    )
    kv_pre_tokens = (
        float(eval_context.cached_prefix) / cp_div
        + float(eval_context.eff_prefill) / kv_split_div
        if eval_context.eff_prefill > 0
        else 0.0
    )
    kv_tokens_total = n_eff * (kv_dec_tokens + kv_pre_tokens)
    kv_tokens_min_total = n_eff * (kv_dec_tokens_min + kv_pre_tokens)
    b_kv_phys = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "gqa", physical=True, chip=eval_context.chip
    )
    b_kv_min = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "gqa", physical=False, chip=eval_context.chip
    )
    att_flops = _route_matmul_flops(
        eval_context.rows,
        att_mxu_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    att_min_flops = _route_matmul_flops(
        eval_context.rows,
        att_w_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    dec_scores = float(
        eval_context.decode_rows
        * 4
        * eval_context.context
        * gqa_hd
        * (geometry.num_heads // tp)
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
        * (geometry.num_heads // tp)
        * n_eff
    )
    n_colls_layers = max(1, round(n_eff))
    att_colls, att_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geometry.hidden,
                num_q_heads=geometry.num_heads // tp,
                head_dim=gqa_hd,
                is_full_attention=True,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geometry.block_parallel,
                kv_bytes_per_tok=float(_TWO_DIRECTIONS * num_kv_heads * gqa_hd)
                * dtypes.kv,
                act_bytes=activation_wire_bytes(dtypes.act),
                cached_prefix=eval_context.cached_prefix,
            ),
            parallel,
        ),
        n_colls_layers,
    )
    dense_unpack = _sub_byte_unpack_vector_flops(
        att_w_elems,
        dtypes.w_dense,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * (q_sharded + 2 * kv_sharded)
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        )
        + _subchannel_scale_vector_flops(
            eval_context.rows,
            att_w_elems,
            dtypes.w_dense,
            geometry.weight_block_size,
            block_k_floor=eval_context.chip.subchannel_block_k_floor,
        )
        + dense_unpack
    )
    strict_vector_flops = (
        vector_flops - dense_unpack + _strict_sub_byte_unpack_vector_flops(dense_unpack)
    )
    strict_attn_fp8 = dtypes.kv <= FP8_BYTES and eval_context.fp8_to_bf16_ratio > 0.0
    score_tot = dec_scores + pre_scores
    return ModuleCost(
        name="gqa",
        hbm_bytes=att_w_elems * dtypes.w_dense + kv_tokens_total * b_kv_phys,
        fp8_flops=att_flops.fp8,
        bf16_flops=att_flops.bf16 + score_tot,
        collectives=att_colls,
        hbm_bytes_min=att_w_elems * dtypes.w_dense + kv_tokens_min_total * b_kv_min,
        fp8_flops_min=att_min_flops.fp8 + (score_tot if strict_attn_fp8 else 0.0),
        bf16_flops_min=att_min_flops.bf16 + (0.0 if strict_attn_fp8 else score_tot),
        unique_collectives=att_ucolls,
        vector_flops=vector_flops,
        strict_vector_flops=strict_vector_flops,
        fp4_flops=att_flops.fp4,
        int8_flops=att_flops.int8,
        int4_flops=att_flops.int4,
        fp4_flops_min=att_min_flops.fp4,
        int8_flops_min=att_min_flops.int8,
        int4_flops_min=att_min_flops.int4,
    )


def _capped_causal_pairs(cached_prefix: int, prefill_global: int, cap: int) -> float:
    """Closed-form causal query-key pairs over [C + 1, C + P_g] capped by window."""
    if prefill_global <= 0:
        return 0.0
    cached_tokens = max(0, cached_prefix)
    p = prefill_global
    hi = cached_tokens + p
    if cap <= 0 or hi <= cap:
        return float(p) * (float(cached_tokens) + CAUSAL_MIDPOINT_HALF * float(p))
    if cached_tokens >= cap:
        return float(p * cap)
    n_ramp = cap - cached_tokens
    n_cap = hi - cap
    return float(n_ramp) * (
        float(cached_tokens) + CAUSAL_MIDPOINT_HALF * float(n_ramp)
    ) + float(n_cap * cap)


def _compute_swa_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    *,
    counts: LayerCounts,
) -> ModuleCost:
    """Compute ModuleCost for SWA attention."""
    n_swa = counts.swa
    tp = parallel.tp
    n_eff = n_swa * eval_context.pp_scale
    num_kv_heads = (
        max(1, geometry.num_kv_heads // tp) if geometry.num_kv_heads > 0 else 0
    )
    num_q_heads = max(1, geometry.num_heads // tp) if geometry.num_heads > 0 else 0
    q_mult = 2 if geometry.attn_output_gate else 1
    q_out = geometry.num_heads * geometry.head_dim * q_mult
    sw_cap = (
        geometry.sliding_window if geometry.sliding_window > 0 else eval_context.context
    )
    eff_ctx = min(eval_context.context, sw_cap) if eval_context.context > 0 else 0
    q_sharded = q_out // tp
    kv_sharded = num_kv_heads * geometry.head_dim
    out_sharded = num_q_heads * geometry.head_dim
    swa_w_elems = float(
        n_eff
        * (
            geometry.hidden * q_sharded
            + 2 * geometry.hidden * kv_sharded
            + out_sharded * geometry.hidden
        )
    )
    swa_kv_scale = (counts.eff_swa / float(n_swa)) if geometry.compress_ratios else 1.0
    kv_split_div = float(max(1, parallel.kv_split))
    cp_div = float(max(1, parallel.pcp * parallel.kv_split))
    pre_kv_tokens = (
        float(min(eval_context.cached_prefix, sw_cap)) / cp_div
        + float(eval_context.eff_prefill) / kv_split_div
        if eval_context.eff_prefill > 0
        else 0.0
    )
    dec_kv_tokens = float(eval_context.batch * eff_ctx) * swa_kv_scale / kv_split_div
    c_shared_swa = min(eff_ctx, max(0, eval_context.shared_prefix_tokens))
    dec_kv_tokens_min = (
        float(c_shared_swa + eval_context.batch * max(0, eff_ctx - c_shared_swa))
        * swa_kv_scale
        / kv_split_div
        if (eval_context.batch > 0 and c_shared_swa > 0)
        else dec_kv_tokens
    )
    kv_tokens = dec_kv_tokens + pre_kv_tokens
    kv_tokens_min = dec_kv_tokens_min + pre_kv_tokens
    b_kv_phys = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "swa", physical=True, chip=eval_context.chip
    )
    b_kv_min = kv_bytes_per_token_per_layer(
        geometry, dtypes, parallel, "swa", physical=False, chip=eval_context.chip
    )
    swa_flops = _route_matmul_flops(
        eval_context.rows,
        swa_w_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    pre_pairs = _capped_causal_pairs(
        eval_context.cached_prefix,
        eval_context.eff_prefill * parallel.pcp,
        sw_cap,
    ) / float(max(1, parallel.pcp))
    swa_scores = float(
        (eval_context.decode_rows * eff_ctx + pre_pairs)
        * _GQA_SCORE_WIDTH_MULT
        * geometry.head_dim
        * num_q_heads
        * n_eff
    )
    swa_colls, swa_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geometry.hidden,
                num_q_heads=num_q_heads,
                head_dim=geometry.head_dim,
                is_full_attention=True,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geometry.block_parallel,
                kv_bytes_per_tok=float(
                    _TWO_DIRECTIONS * num_kv_heads * geometry.head_dim
                )
                * dtypes.kv,
                act_bytes=activation_wire_bytes(dtypes.act),
                cached_prefix=eval_context.cached_prefix,
                sliding_window=geometry.sliding_window,
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vector_flops(
        swa_w_elems,
        dtypes.w_dense,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * (q_sharded + 2 * kv_sharded)
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        )
        + _subchannel_scale_vector_flops(
            eval_context.rows,
            swa_w_elems,
            dtypes.w_dense,
            geometry.weight_block_size,
            block_k_floor=eval_context.chip.subchannel_block_k_floor,
        )
        + dense_unpack
    )
    strict_vector_flops = (
        vector_flops - dense_unpack + _strict_sub_byte_unpack_vector_flops(dense_unpack)
    )
    strict_attn_fp8 = dtypes.kv <= FP8_BYTES and eval_context.fp8_to_bf16_ratio > 0.0
    return ModuleCost(
        name="swa",
        hbm_bytes=swa_w_elems * dtypes.w_dense + n_eff * kv_tokens * b_kv_phys,
        fp8_flops=swa_flops.fp8,
        bf16_flops=swa_flops.bf16 + swa_scores,
        collectives=swa_colls,
        hbm_bytes_min=swa_w_elems * dtypes.w_dense + n_eff * kv_tokens_min * b_kv_min,
        fp8_flops_min=swa_flops.fp8 + (swa_scores if strict_attn_fp8 else 0.0),
        bf16_flops_min=swa_flops.bf16 + (0.0 if strict_attn_fp8 else swa_scores),
        unique_collectives=swa_ucolls,
        vector_flops=vector_flops,
        strict_vector_flops=strict_vector_flops,
        fp4_flops=swa_flops.fp4,
        int8_flops=swa_flops.int8,
        int4_flops=swa_flops.int4,
        fp4_flops_min=swa_flops.fp4,
        int8_flops_min=swa_flops.int8,
        int4_flops_min=swa_flops.int4,
    )


def _validate_mla_geom(fam_name: str, geometry: ModelGeometry) -> None:
    """Validate model geometry fields required for MLA or DSA."""
    if geometry.kv_lora_rank <= 0:
        msg = f"{fam_name.upper()} layer requires config field 'kv_lora_rank'"
        raise ValueError(msg)
    if geometry.qk_nope_head_dim <= 0:
        msg = f"{fam_name.upper()} layer requires config field 'qk_nope_head_dim'"
        raise ValueError(msg)
    if geometry.qk_rope_head_dim < 0:
        msg = f"{fam_name.upper()} layer requires config field 'qk_rope_head_dim'"
        raise ValueError(msg)


def _mla_unpack_vector_flops(
    base_vector_flops: float,
    mla_w_elems: float,
    dtypes: Dtypes,
    chip: ChipSpec,
) -> tuple[float, float]:
    """Return nominal and strict vector FLOPs including capability-gated unpack."""
    dense_unpack = _sub_byte_unpack_vector_flops(
        mla_w_elems,
        dtypes.w_dense,
        chip=chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = base_vector_flops + dense_unpack
    strict_vector_flops = (
        vector_flops - dense_unpack + _strict_sub_byte_unpack_vector_flops(dense_unpack)
    )
    return vector_flops, strict_vector_flops


@dataclass(frozen=True, slots=True)
class _MlaStrictInputs:
    """Inputs for MLA/DSA strict prefill vs decode FLOP computation."""

    base_mla_w_elems: float
    ukv_w_elems: float
    dec_abs_scores: float
    pre_abs_scores: float
    pre_mha_scores: float


def _relative_bucket_time(
    buckets: _MatmulBuckets,
    chip: ChipSpec,
    fp8_to_bf16_ratio: float,
) -> float:
    """Normalize per-precision FLOP buckets to relative BF16 compute time."""
    bf16_peak = chip.bf16_flops if chip.bf16_flops > 0.0 else 1.0
    fp8_rate = fp8_to_bf16_ratio if fp8_to_bf16_ratio > 0.0 else 1.0
    pairs = (
        (buckets.bf16, 1.0),
        (buckets.fp8, fp8_rate),
        (buckets.int8, chip.int8_flops / bf16_peak),
        (buckets.fp4, chip.fp4_flops / bf16_peak),
        (buckets.int4, chip.int4_flops / bf16_peak),
    )
    return sum(flops / rate for flops, rate in pairs if flops > 0.0 and rate > 0.0)


def _mla_strict_flops(
    eval_context: _StepContext,
    parallel: Parallel,
    dtypes: Dtypes,
    strict_in: _MlaStrictInputs,
) -> _MatmulBuckets:
    """Compute strict per-precision matmul FLOP buckets for MLA or DSA."""
    strict_attn_fp8 = dtypes.kv <= FP8_BYTES and eval_context.fp8_to_bf16_ratio > 0.0
    score_bucket = "fp8" if strict_attn_fp8 else "bf16"
    strict_buckets = (
        _route_matmul_flops(
            eval_context.rows,
            strict_in.base_mla_w_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
        + _route_matmul_flops(
            eval_context.decode_rows,
            strict_in.ukv_w_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
        + _bucket_from_name(score_bucket, strict_in.dec_abs_scores)
    )

    if eval_context.eff_prefill <= 0:
        return strict_buckets

    abs_buckets = _route_matmul_flops(
        eval_context.eff_prefill,
        strict_in.ukv_w_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    ) + _bucket_from_name(score_bucket, strict_in.pre_abs_scores)

    key_tokens_per_rank = float(eval_context.cached_prefix) / float(
        max(1, parallel.pcp)
    ) + float(eval_context.eff_prefill)
    ukv_mha_flops = (
        float(MATMUL_FLOP_PER_MAC) * key_tokens_per_rank * strict_in.ukv_w_elems
    )
    ukv_bucket, _peak, _unpack = eval_context.chip.matmul_peak_kind(
        dtypes.w_dense, dtypes.act, weight_dtype=dtypes.w_dense_dtype
    )
    mha_buckets = _bucket_from_name(ukv_bucket, ukv_mha_flops) + _bucket_from_name(
        score_bucket, strict_in.pre_mha_scores
    )

    abs_time = _relative_bucket_time(
        abs_buckets, eval_context.chip, eval_context.fp8_to_bf16_ratio
    )
    mha_time = _relative_bucket_time(
        mha_buckets, eval_context.chip, eval_context.fp8_to_bf16_ratio
    )
    if mha_time < abs_time:
        return strict_buckets + mha_buckets
    return strict_buckets + abs_buckets


def _dsa_indexer_score_buckets(
    geometry: ModelGeometry,
    parallel: Parallel,
    eval_context: _StepContext,
    n_idx_eff: float,
) -> tuple[_MatmulBuckets, _MatmulBuckets]:
    """Compute estimate and strict lower-bound FLOP buckets for DSA indexer q*k."""
    if n_idx_eff <= 0.0 or geometry.index_n_heads <= 0 or geometry.index_head_dim <= 0:
        return _MatmulBuckets(), _MatmulBuckets()
    dec_active = (
        eval_context.context > geometry.index_topk and eval_context.decode_rows > 0
    )
    dec_rows = (
        eval_context.batch
        if (geometry.index_share_for_mtp_iteration and eval_context.rows_per_seq > 1)
        else eval_context.decode_rows
    )
    dec_pairs = float(dec_rows * eval_context.context) if dec_active else 0.0
    pre_global = eval_context.eff_prefill * parallel.pcp
    pre_active = (
        eval_context.eff_prefill > 0
        and (eval_context.cached_prefix + pre_global) > geometry.index_topk
    )
    pre_pairs = (
        _capped_causal_pairs(eval_context.cached_prefix, pre_global, 0)
        / float(max(1, parallel.pcp))
        if pre_active
        else 0.0
    )
    total_pairs = dec_pairs + pre_pairs
    if total_pairs <= 0.0:
        return _MatmulBuckets(), _MatmulBuckets()
    est_flops = (
        total_pairs
        * MATMUL_FLOP_PER_MAC
        * float(geometry.index_n_heads * geometry.index_head_dim)
        * n_idx_eff
    )
    strict_heads = max(1, geometry.index_n_heads // max(1, parallel.tp))
    strict_flops = (
        total_pairs
        * MATMUL_FLOP_PER_MAC
        * float(strict_heads * geometry.index_head_dim)
        * n_idx_eff
    )
    bucket = "fp8" if eval_context.chip.fp8_flops > 0.0 else "bf16"
    return _bucket_from_name(bucket, est_flops), _bucket_from_name(bucket, strict_flops)


def dsa_indexer_layer_weight_elems(geometry: ModelGeometry) -> float:
    """Return replicated DSA indexer weight elements per full indexer layer."""
    if geometry.index_n_heads <= 0 or geometry.index_head_dim <= 0:
        return 0.0
    q_in = geometry.q_lora_rank if geometry.q_lora_rank > 0 else geometry.hidden
    return float(
        q_in * geometry.index_n_heads * geometry.index_head_dim
        + geometry.hidden * geometry.index_head_dim
        + geometry.hidden * geometry.index_n_heads
    )


def _dsa_indexer_layer_weight_elems_strict(geometry: ModelGeometry, tp: int) -> float:
    """Return strict TP-sharded DSA indexer weight elements per full indexer layer."""
    if geometry.index_n_heads <= 0 or geometry.index_head_dim <= 0:
        return 0.0
    q_in = geometry.q_lora_rank if geometry.q_lora_rank > 0 else geometry.hidden
    strict_heads = max(1, geometry.index_n_heads // max(1, tp))
    return float(
        q_in * strict_heads * geometry.index_head_dim
        + geometry.hidden * geometry.index_head_dim
        + geometry.hidden * strict_heads
    )


def _mla_weights_and_kv(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    fam_info: tuple[int, float, bool, int],
) -> tuple[float, float, float, float, float, float, float, int]:
    """Return MLA/DSA weight, strict weight, KV, effective layer counts, and context."""
    n_fam, kv_n_fam, is_sparse, num_q_heads = fam_info
    n_eff = n_fam * eval_context.pp_scale
    kv_lora = geometry.kv_lora_rank
    nope_d = geometry.qk_nope_head_dim
    rope_d = geometry.qk_rope_head_dim
    v_head_dim = geometry.v_head_dim or nope_d
    nq_raw = num_q_heads * (nope_d + rope_d)
    q_w = (
        float(geometry.hidden * geometry.q_lora_rank + geometry.q_lora_rank * nq_raw)
        if geometry.q_lora_rank > 0
        else float(geometry.hidden * nq_raw)
    )
    mla_w_layer = (
        q_w
        + float(geometry.hidden * (kv_lora + rope_d))
        + float(
            kv_lora * num_q_heads * (nope_d + v_head_dim)
            + (num_q_heads * v_head_dim) * geometry.hidden
        )
    )
    idx_scale = (
        float(geometry.indexer_types.count("full")) / float(n_fam)
        if (is_sparse and geometry.indexer_types)
        else (1.0 if is_sparse else 0.0)
    )
    n_idx_eff = n_eff * idx_scale
    idx_w_elems = (
        n_idx_eff * dsa_indexer_layer_weight_elems(geometry)
        if (is_sparse and n_idx_eff > 0.0)
        else 0.0
    )
    idx_w_elems_strict = (
        n_idx_eff * _dsa_indexer_layer_weight_elems_strict(geometry, parallel.tp)
        if (is_sparse and n_idx_eff > 0.0)
        else 0.0
    )
    mla_w_elems = n_eff * mla_w_layer + idx_w_elems
    mla_w_elems_strict = n_eff * mla_w_layer + idx_w_elems_strict
    ukv_w_elems = n_eff * float(kv_lora * num_q_heads * (nope_d + v_head_dim))
    att_ctx = (
        min(eval_context.context, geometry.index_topk)
        if (is_sparse and geometry.index_topk > 0 and eval_context.context > 0)
        else eval_context.context
    )
    raw_latent_w = kv_lora + rope_d
    align = eval_context.chip.physical_alignment_elems
    b_latent_phys = float(math.ceil(raw_latent_w / align) * align) * dtypes.kv
    b_latent_min = float(raw_latent_w) * dtypes.kv
    has_idx = is_sparse and geometry.index_topk > 0 and geometry.index_head_dim > 0
    b_idx_phys = (
        float(geometry.index_head_dim) * dtypes.kv
        + (SPARSE_MLA_SCALE_BYTES if dtypes.kv <= FP8_BYTES else 0.0)
        if has_idx
        else 0.0
    )
    b_idx_min = float(geometry.index_head_dim) * dtypes.kv if has_idx else 0.0
    fam_kv_scale = (kv_n_fam / float(n_fam)) if geometry.compress_ratios else 1.0
    kv_split_div = float(max(1, parallel.kv_split))
    cp_div = float(max(1, parallel.pcp * parallel.kv_split))
    pre_kv_tokens = (
        float(eval_context.cached_prefix) / cp_div
        + float(eval_context.eff_prefill) / kv_split_div
        if eval_context.eff_prefill > 0
        else 0.0
    )
    dec_kv_tokens = float(eval_context.batch * att_ctx) * fam_kv_scale / kv_split_div
    c_shared_mla = min(att_ctx, max(0, eval_context.shared_prefix_tokens))
    dec_kv_tokens_min = (
        float(c_shared_mla + eval_context.batch * max(0, att_ctx - c_shared_mla))
        * fam_kv_scale
        / kv_split_div
        if (eval_context.batch > 0 and c_shared_mla > 0)
        else dec_kv_tokens
    )
    kv_tokens = dec_kv_tokens + pre_kv_tokens
    kv_tokens_min = dec_kv_tokens_min + pre_kv_tokens
    mla_kv_bytes = n_eff * kv_tokens * b_latent_phys
    mla_kv_bytes_min = n_eff * kv_tokens_min * b_latent_min
    if is_sparse and geometry.index_head_dim > 0 and n_idx_eff > 0.0:
        if eval_context.batch > 0 and eval_context.context > geometry.index_topk:
            idx_tokens = (
                n_idx_eff
                * float(eval_context.batch * eval_context.context)
                * fam_kv_scale
                / kv_split_div
            )
            c_shared_idx = min(
                eval_context.context, max(0, eval_context.shared_prefix_tokens)
            )
            idx_tokens_min = (
                n_idx_eff
                * float(
                    c_shared_idx
                    + eval_context.batch * max(0, eval_context.context - c_shared_idx)
                )
                * fam_kv_scale
                / kv_split_div
                if c_shared_idx > 0
                else idx_tokens
            )
            mla_kv_bytes += idx_tokens * b_idx_phys
            mla_kv_bytes_min += idx_tokens_min * b_idx_min
        pre_global = eval_context.eff_prefill * parallel.pcp
        if (
            eval_context.eff_prefill > 0
            and (eval_context.cached_prefix + pre_global) > geometry.index_topk
        ):
            mla_kv_bytes += n_idx_eff * pre_kv_tokens * b_idx_phys
            mla_kv_bytes_min += n_idx_eff * pre_kv_tokens * b_idx_min
    return (
        mla_w_elems,
        mla_w_elems_strict,
        ukv_w_elems,
        mla_kv_bytes,
        mla_kv_bytes_min,
        n_eff,
        n_idx_eff,
        att_ctx,
    )


def _compute_mla_dsa_costs(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    counts: LayerCounts,
) -> list[ModuleCost]:
    """Compute ModuleCost for MLA and DSA attention families."""
    tp = parallel.tp
    num_q_heads = max(1, geometry.num_heads // tp) if geometry.num_heads > 0 else 0
    out: list[ModuleCost] = []
    plain_mla = max(0, counts.mla - counts.dsa)
    plain_eff_mla = max(0.0, counts.eff_mla - counts.eff_dsa)
    for fam_name, n_fam, kv_n_fam, is_sparse in (
        ("mla", plain_mla, plain_eff_mla, False),
        ("dsa", counts.dsa, counts.eff_dsa, True),
    ):
        if n_fam <= 0:
            continue
        _validate_mla_geom(fam_name, geometry)
        kv_lora = geometry.kv_lora_rank
        nope_d = geometry.qk_nope_head_dim
        rope_d = geometry.qk_rope_head_dim
        v_head_dim = geometry.v_head_dim or nope_d
        (
            mla_w_elems,
            mla_w_elems_strict,
            ukv_w_elems,
            mla_kv_bytes,
            mla_kv_bytes_min,
            n_eff,
            n_idx_eff,
            att_ctx,
        ) = _mla_weights_and_kv(
            geometry,
            parallel,
            dtypes,
            eval_context,
            (n_fam, kv_n_fam, is_sparse, num_q_heads),
        )
        base_mla_w_elems_strict = mla_w_elems_strict - ukv_w_elems
        mla_flops = _route_matmul_flops(
            eval_context.rows,
            mla_w_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
        prefill_cap = (
            geometry.index_topk if (is_sparse and geometry.index_topk > 0) else 0
        )
        prefill_pairs = _capped_causal_pairs(
            eval_context.cached_prefix,
            eval_context.eff_prefill * parallel.pcp,
            prefill_cap,
        ) / float(max(1, parallel.pcp))
        dec_abs_scores = float(
            eval_context.decode_rows
            * att_ctx
            * MATMUL_FLOP_PER_MAC
            * num_q_heads
            * (_TWO_DIRECTIONS * kv_lora + rope_d)
            * n_eff
        )
        pre_abs_scores = float(
            prefill_pairs
            * MATMUL_FLOP_PER_MAC
            * num_q_heads
            * (_TWO_DIRECTIONS * kv_lora + rope_d)
            * n_eff
        )
        pre_mha_scores = float(
            prefill_pairs
            * MATMUL_FLOP_PER_MAC
            * num_q_heads
            * (nope_d + rope_d + v_head_dim)
            * n_eff
        )
        idx_est_buckets, idx_strict_buckets = _dsa_indexer_score_buckets(
            geometry, parallel, eval_context, n_idx_eff
        )
        mla_scores = dec_abs_scores + pre_abs_scores
        strict_buckets = _mla_strict_flops(
            eval_context,
            parallel,
            dtypes,
            _MlaStrictInputs(
                base_mla_w_elems=base_mla_w_elems_strict,
                ukv_w_elems=ukv_w_elems,
                dec_abs_scores=dec_abs_scores,
                pre_abs_scores=pre_abs_scores,
                pre_mha_scores=pre_mha_scores,
            ),
        )
        mla_colls, mla_ucolls = _repeat_colls(
            _attn_layer_collectives(
                _AttnCollSpec(
                    batch=eval_context.batch,
                    rows_per_seq=eval_context.rows_per_seq,
                    hidden=geometry.hidden,
                    num_q_heads=num_q_heads,
                    head_dim=kv_lora,
                    is_full_attention=True,
                    extra_rows=eval_context.eff_prefill,
                    block_parallel=geometry.block_parallel,
                    kv_bytes_per_tok=float(kv_lora + rope_d) * dtypes.kv
                    + (
                        float(geometry.index_head_dim) * dtypes.kv
                        if (
                            is_sparse
                            and geometry.index_topk > 0
                            and geometry.index_head_dim > 0
                        )
                        else 0.0
                    ),
                    act_bytes=activation_wire_bytes(dtypes.act),
                    q_head_dim=kv_lora + rope_d,
                    cached_prefix=eval_context.cached_prefix,
                ),
                parallel,
            ),
            max(1, round(n_eff)),
        )
        base_vector_flops = float(
            n_eff
            * (
                4.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * num_q_heads * rope_d
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        ) + _subchannel_scale_vector_flops(
            eval_context.rows,
            mla_w_elems,
            dtypes.w_dense,
            geometry.weight_block_size,
            block_k_floor=eval_context.chip.subchannel_block_k_floor,
        )
        vector_flops, strict_vector_flops = _mla_unpack_vector_flops(
            base_vector_flops, mla_w_elems, dtypes, eval_context.chip
        )
        out.append(
            ModuleCost(
                name=fam_name,
                hbm_bytes=mla_w_elems * dtypes.w_dense + mla_kv_bytes,
                fp8_flops=mla_flops.fp8 + idx_est_buckets.fp8,
                bf16_flops=mla_flops.bf16 + mla_scores + idx_est_buckets.bf16,
                collectives=mla_colls,
                hbm_bytes_min=mla_w_elems_strict * dtypes.w_dense + mla_kv_bytes_min,
                fp8_flops_min=strict_buckets.fp8 + idx_strict_buckets.fp8,
                bf16_flops_min=strict_buckets.bf16 + idx_strict_buckets.bf16,
                unique_collectives=mla_ucolls,
                vector_flops=vector_flops,
                strict_vector_flops=strict_vector_flops,
                fp4_flops=mla_flops.fp4,
                int8_flops=mla_flops.int8,
                int4_flops=mla_flops.int4,
                fp4_flops_min=strict_buckets.fp4,
                int8_flops_min=strict_buckets.int8,
                int4_flops_min=strict_buckets.int4,
            )
        )
    return out


def _compute_gdn_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    *,
    n_gdn: int,
) -> ModuleCost:
    """Compute ModuleCost for GDN linear attention."""
    tp = parallel.tp
    n_eff = n_gdn * eval_context.pp_scale
    key_heads = (
        max(1, geometry.linear_k_heads // tp) if geometry.linear_k_heads > 0 else 0
    )
    value_heads = (
        max(1, geometry.linear_v_heads // tp) if geometry.linear_v_heads > 0 else 0
    )
    qkv_rank_width = (
        2 * key_heads * geometry.linear_k_dim + value_heads * geometry.linear_v_dim
    )
    z_rank_width = value_heads * geometry.linear_v_dim
    gdn_dense_elems = float(
        n_eff
        * (
            geometry.hidden * (qkv_rank_width + z_rank_width)
            + z_rank_width * geometry.hidden
        )
    )
    gdn_aux_elems = float(
        n_eff
        * (geometry.hidden * (2 * value_heads) + qkv_rank_width * geometry.conv_kernel)
    )
    rec_state = (
        float(n_eff * value_heads * geometry.linear_k_dim * geometry.linear_v_dim)
        * dtypes.state
    )
    conv_state = (
        float(n_eff * qkv_rank_width * max(0, geometry.conv_kernel - 1)) * dtypes.conv
    )
    dec_state_io = float(eval_context.batch * (1 + eval_context.state_writes))
    pre_state_io = (
        float((1 if eval_context.cached_prefix > 0 else 0) + 1)
        if eval_context.eff_prefill > 0
        else 0.0
    )
    gdn_state_bytes = (dec_state_io + pre_state_io) * (rec_state + conv_state)
    dense_buckets = _route_matmul_flops(
        eval_context.rows,
        gdn_dense_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    aux_buckets = _route_matmul_flops(
        eval_context.rows,
        gdn_aux_elems,
        dtypes.w_aux,
        dtypes.act,
        chip=eval_context.chip,
    )
    gdn_buckets = dense_buckets + aux_buckets
    gdn_rec_flops = float(
        eval_context.rows
        * 4
        * value_heads
        * geometry.linear_k_dim
        * geometry.linear_v_dim
        * n_eff
    )
    gdn_colls, gdn_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geometry.hidden,
                num_q_heads=geometry.num_heads // tp,
                head_dim=geometry.head_dim,
                is_full_attention=False,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geometry.block_parallel,
                act_bytes=activation_wire_bytes(dtypes.act),
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vector_flops(
        gdn_dense_elems,
        dtypes.w_dense,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        )
        + _subchannel_scale_vector_flops(
            eval_context.rows,
            gdn_dense_elems,
            dtypes.w_dense,
            geometry.weight_block_size,
            block_k_floor=eval_context.chip.subchannel_block_k_floor,
        )
        + dense_unpack
    )
    strict_vector_flops = (
        vector_flops - dense_unpack + _strict_sub_byte_unpack_vector_flops(dense_unpack)
    )
    return ModuleCost(
        name="gdn",
        hbm_bytes=(
            gdn_dense_elems * dtypes.w_dense
            + gdn_aux_elems * dtypes.w_aux
            + gdn_state_bytes
        ),
        fp8_flops=gdn_buckets.fp8,
        bf16_flops=gdn_buckets.bf16 + gdn_rec_flops,
        collectives=gdn_colls,
        unique_collectives=gdn_ucolls,
        vector_flops=vector_flops,
        strict_vector_flops=strict_vector_flops,
        fp4_flops=gdn_buckets.fp4,
        int8_flops=gdn_buckets.int8,
        int4_flops=gdn_buckets.int4,
    )


def _compute_kda_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    *,
    n_kda: int,
) -> ModuleCost:
    """Compute ModuleCost for KDA linear attention."""
    tp = parallel.tp
    num_q_heads = max(1, geometry.num_heads // tp) if geometry.num_heads > 0 else 0
    n_eff = n_kda * eval_context.pp_scale
    key_heads = max(1, (geometry.linear_k_heads or geometry.num_heads) // tp)
    value_heads = max(1, (geometry.linear_v_heads or geometry.num_heads) // tp)
    k_head_dim = geometry.linear_k_dim or geometry.head_dim
    v_head_dim = geometry.linear_v_dim or geometry.head_dim
    proj_rank_width = 2 * key_heads * k_head_dim + 2 * value_heads * v_head_dim
    kda_w_elems = float(
        n_eff
        * (
            geometry.hidden * proj_rank_width
            + (value_heads * v_head_dim) * geometry.hidden
        )
    )
    kda_rec = float(n_eff * value_heads * k_head_dim * v_head_dim) * dtypes.state
    dec_state_io = float(eval_context.batch * (1 + eval_context.state_writes))
    pre_state_io = (
        float((1 if eval_context.cached_prefix > 0 else 0) + 1)
        if eval_context.eff_prefill > 0
        else 0.0
    )
    kda_state_bytes = (dec_state_io + pre_state_io) * kda_rec
    kda_buckets = _route_matmul_flops(
        eval_context.rows,
        kda_w_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    kda_rec_flops = float(
        eval_context.rows * 4 * value_heads * k_head_dim * v_head_dim * n_eff
    )
    kda_colls, kda_ucolls = _repeat_colls(
        _attn_layer_collectives(
            _AttnCollSpec(
                batch=eval_context.batch,
                rows_per_seq=eval_context.rows_per_seq,
                hidden=geometry.hidden,
                num_q_heads=num_q_heads,
                head_dim=v_head_dim,
                is_full_attention=False,
                extra_rows=eval_context.eff_prefill,
                block_parallel=geometry.block_parallel,
                act_bytes=activation_wire_bytes(dtypes.act),
            ),
            parallel,
        ),
        max(1, round(n_eff)),
    )
    dense_unpack = _sub_byte_unpack_vector_flops(
        kda_w_elems,
        dtypes.w_dense,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = (
        float(
            n_eff
            * (
                4.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        )
        + _subchannel_scale_vector_flops(
            eval_context.rows,
            kda_w_elems,
            dtypes.w_dense,
            geometry.weight_block_size,
            block_k_floor=eval_context.chip.subchannel_block_k_floor,
        )
        + dense_unpack
    )
    strict_vector_flops = (
        vector_flops - dense_unpack + _strict_sub_byte_unpack_vector_flops(dense_unpack)
    )
    return ModuleCost(
        name="kda",
        hbm_bytes=kda_w_elems * dtypes.w_dense + kda_state_bytes,
        fp8_flops=kda_buckets.fp8,
        bf16_flops=kda_buckets.bf16 + kda_rec_flops,
        collectives=kda_colls,
        unique_collectives=kda_ucolls,
        vector_flops=vector_flops,
        strict_vector_flops=strict_vector_flops,
        fp4_flops=kda_buckets.fp4,
        int8_flops=kda_buckets.int8,
        int4_flops=kda_buckets.int4,
    )


_DEFAULT_ACTIVATION_MODEL: ExpertActivationModel = ExpertActivationModel()


@dataclass(frozen=True, slots=True)
class _RoutedEnvelopeSpec:
    """Precomputed routed MoE terms needed for activation-envelope sizing."""

    ep_shards: int
    decode_tokens: int
    prefill_tokens: int
    routed_rows: int
    routed_unit: float
    non_routed_weight_bytes: float
    routed_base_flops: float
    lhs_floor: int


def _resolve_envelope_experts(
    geometry: ModelGeometry,
    eval_context: _StepContext,
    spec: _RoutedEnvelopeSpec,
) -> tuple[float, float]:
    """Resolve per-shard and global active expert counts across decode/prefill/mixed."""
    model = eval_context.activation_model
    has_decode = spec.decode_tokens > 0
    has_prefill = spec.prefill_tokens > 0
    if has_decode and has_prefill and model.prunes_on_envelope:
        # Mixed step under empirical hull: set-union monotonicity guarantees
        # |E_decode U E_prefill| >= max(|E_decode|, |E_prefill|).
        decode_dispatch = ExpertDispatch(
            num_experts=geometry.num_experts,
            experts_per_token=geometry.top_k,
            dispatched_tokens=spec.decode_tokens,
            ep_shards=spec.ep_shards,
            is_prefill=False,
            tokens_per_seq=max(1, eval_context.rows_per_seq),
        )
        prefill_dispatch = ExpertDispatch(
            num_experts=geometry.num_experts,
            experts_per_token=geometry.top_k,
            dispatched_tokens=spec.prefill_tokens,
            ep_shards=spec.ep_shards,
            is_prefill=True,
            tokens_per_seq=1,
        )
        return (
            max(
                min_experts_per_ep_shard(model, decode_dispatch),
                min_experts_per_ep_shard(model, prefill_dispatch),
            ),
            max(
                active_experts_envelope(model, decode_dispatch),
                active_experts_envelope(model, prefill_dispatch),
            ),
        )
    is_prefill = has_prefill and not has_decode
    tokens_per_seq = 1 if is_prefill else max(1, eval_context.rows_per_seq)
    dispatch = ExpertDispatch(
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        dispatched_tokens=spec.decode_tokens + spec.prefill_tokens,
        ep_shards=spec.ep_shards,
        is_prefill=is_prefill,
        tokens_per_seq=tokens_per_seq,
    )
    return (
        min_experts_per_ep_shard(model, dispatch),
        active_experts_envelope(model, dispatch),
    )


def _compute_routed_moe_envelope(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    eval_context: _StepContext,
    spec: _RoutedEnvelopeSpec,
) -> tuple[float, float]:
    """Compute activation-envelope routed MoE weight bytes and compute FLOPs."""
    experts_per_shard_envelope, total_active_envelope = _resolve_envelope_experts(
        geometry, eval_context, spec
    )
    moe_weight_bytes_envelope = (
        experts_per_shard_envelope * spec.routed_unit * dtypes.w_expert
        + spec.non_routed_weight_bytes
    )
    routed_flops_envelope = spec.routed_base_flops
    if (
        spec.lhs_floor > 1
        and spec.routed_rows > 0
        and experts_per_shard_envelope > 0.0
        and geometry.top_k > 0
    ):
        total_assignments = float(spec.routed_rows * geometry.top_k)
        envelope_padded_tiles = max(
            float(math.ceil(total_assignments / spec.lhs_floor)),
            total_active_envelope,
        )
        pad_multiplier_envelope = (
            envelope_padded_tiles * float(spec.lhs_floor)
        ) / total_assignments
        routed_flops_envelope *= pad_multiplier_envelope
    return moe_weight_bytes_envelope, routed_flops_envelope


@dataclass(frozen=True, slots=True)
class _RoutedMoeLayout:
    """Sharded expert dimensions, element counts, and HBM weight bytes."""

    tp: int
    is_ep: bool
    ep_ranks: int
    pcp_per_ep: int
    dp_per_ep: int
    active_experts: float
    min_active_experts: float
    routed_unit: float
    routed_elems: float
    routed_elems_min: float
    shared_rank_width: int
    shared_elems: float
    shared_mxu_elems: float
    router_elems: float
    moe_w_bytes: float
    moe_w_bytes_min: float
    latent_proj_sharded: float
    eff_moe_div: int
    per_rank_moe_inter: float
    routed_act_elems: float
    routed_act_elems_min: float
    routed_rows: int
    sh_rows: int


def _routed_moe_layout(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    *,
    n_m_eff: float,
) -> _RoutedMoeLayout:
    """Compute sharded widths, element counts, and HBM bytes for routed MoE."""
    tp = parallel.tp
    ep_ranks = (parallel.ep if parallel.ep > 1 else tp) if parallel.moe == "ep" else 1
    is_ep = parallel.moe == "ep" and ep_ranks > 1 and geometry.num_experts > 0
    pcp_per_ep, dp_per_ep = _ep_local_dims(parallel, ep_ranks if is_ep else tp)
    global_rows = (
        (eval_context.decode_rows + eval_context.eff_prefill * pcp_per_ep) * dp_per_ep
        if is_ep
        else eval_context.rows
    )
    if parallel.moe == "ep":
        active_experts = ep_busiest_experts(
            geometry.num_experts, geometry.top_k, global_rows, ep_ranks
        )
        min_active_experts = (
            float(math.ceil(min(geometry.num_experts, geometry.top_k) / ep_ranks))
            if global_rows > 0
            else 0.0
        )
        tp_sub = max(1, tp // max(1, ep_ranks)) if ep_ranks < tp else 1
        width_exact = float(geometry.moe_inter) / float(tp_sub)
    else:
        active_experts = experts_touched(
            geometry.num_experts, geometry.top_k, eval_context.rows
        )
        min_active_experts = (
            float(min(geometry.num_experts, geometry.top_k))
            if eval_context.rows > 0
            else 0.0
        )
        width_exact = float(geometry.moe_inter) / float(max(1, tp))
    width = math.ceil(width_exact)
    expert_hidden = (
        geometry.routed_expert_hidden
        if geometry.routed_expert_hidden > 0
        else geometry.hidden
    )
    ffn_mats = geometry.ffn_num_matrices
    routed_unit = float(n_m_eff * ffn_mats * expert_hidden) * width_exact
    routed_elems = active_experts * routed_unit
    routed_elems_min = min_active_experts * routed_unit
    shared_tp_width = (
        max(1, geometry.shared_inter // tp) if geometry.shared_inter > 0 else 0
    )
    shared_rank_width = (
        geometry.shared_inter if parallel.residual == "sp" else shared_tp_width
    )
    shared_elems = float(n_m_eff * ffn_mats * geometry.hidden * shared_rank_width)
    shared_mxu_elems = float(
        n_m_eff
        * ffn_mats
        * geometry.hidden
        * _align_mxu(shared_rank_width, eval_context)
    )
    router_elems = float(n_m_eff * geometry.hidden * geometry.num_experts)
    moe_w_bytes = (
        routed_elems * dtypes.w_expert
        + shared_elems * dtypes.w_dense
        + router_elems * dtypes.router
    )
    moe_w_bytes_min = (
        routed_elems_min * dtypes.w_expert
        + shared_elems * dtypes.w_dense
        + router_elems * dtypes.router
    )
    latent_proj_elems = (
        float(
            n_m_eff * _TWO_DIRECTIONS * geometry.hidden * geometry.routed_expert_hidden
        )
        if geometry.routed_expert_hidden > 0
        else 0.0
    )
    latent_proj_sharded = (
        latent_proj_elems / tp if geometry.routed_expert_hidden > 0 else 0.0
    )
    if geometry.routed_expert_hidden > 0:
        moe_w_bytes += latent_proj_sharded * dtypes.w_dense
        moe_w_bytes_min += latent_proj_sharded * dtypes.w_dense
    eff_moe_div = max(tp * pcp_per_ep, ep_ranks) if is_ep else tp
    width_mxu = _align_mxu(width, eval_context)
    unsharded_moe_inter_mxu = width_mxu * (
        1
        if (parallel.moe == "ep" and ep_ranks >= tp)
        else (max(1, tp // max(1, ep_ranks)) if parallel.moe == "ep" else tp)
    )
    per_rank_moe_inter = float(unsharded_moe_inter_mxu) / float(eff_moe_div)
    routed_act_elems = float(
        n_m_eff * geometry.top_k * ffn_mats * expert_hidden * per_rank_moe_inter
    )
    routed_act_elems_min = float(
        n_m_eff
        * geometry.top_k
        * ffn_mats
        * expert_hidden
        * (geometry.moe_inter / eff_moe_div)
    )
    routed_rows = global_rows if is_ep else eval_context.rows
    sh_rows = (
        math.ceil(eval_context.rows / tp)
        if parallel.residual == "sp"
        else eval_context.rows
    )
    return _RoutedMoeLayout(
        tp=tp,
        is_ep=is_ep,
        ep_ranks=ep_ranks,
        pcp_per_ep=pcp_per_ep,
        dp_per_ep=dp_per_ep,
        active_experts=active_experts,
        min_active_experts=min_active_experts,
        routed_unit=routed_unit,
        routed_elems=routed_elems,
        routed_elems_min=routed_elems_min,
        shared_rank_width=shared_rank_width,
        shared_elems=shared_elems,
        shared_mxu_elems=shared_mxu_elems,
        router_elems=router_elems,
        moe_w_bytes=moe_w_bytes,
        moe_w_bytes_min=moe_w_bytes_min,
        latent_proj_sharded=latent_proj_sharded,
        eff_moe_div=eff_moe_div,
        per_rank_moe_inter=per_rank_moe_inter,
        routed_act_elems=routed_act_elems,
        routed_act_elems_min=routed_act_elems_min,
        routed_rows=routed_rows,
        sh_rows=sh_rows,
    )


def _routed_moe_vector_flops(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    eval_context: _StepContext,
    layout: _RoutedMoeLayout,
    *,
    n_m_eff: float,
) -> tuple[float, float]:
    """Return nominal and strict vector FLOPs for a routed MoE layer."""
    _, _, routed_unpack = eval_context.chip.matmul_peak_kind(
        dtypes.w_expert, dtypes.act, weight_dtype=dtypes.w_expert_dtype
    )
    ffn_act_vector = (
        1.0
        if geometry.ffn_num_matrices == NON_GATED_MATMUL_COUNT
        else float(GLU_MATMUL_COUNT)
    )
    sh_vector = (
        float(n_m_eff * ffn_act_vector * layout.sh_rows * layout.shared_rank_width)
        if layout.shared_rank_width > 0
        else 0.0
    )
    routed_vector = float(
        n_m_eff
        * ffn_act_vector
        * layout.routed_rows
        * layout.per_rank_moe_inter
        * geometry.top_k
    )
    routed_vector_min = float(
        n_m_eff
        * ffn_act_vector
        * layout.routed_rows
        * (geometry.moe_inter / layout.eff_moe_div)
        * geometry.top_k
    )
    fp4_unpack = float(2.0 * layout.routed_elems) if routed_unpack else 0.0
    fp4_unpack_min = float(2.0 * layout.routed_elems_min) if routed_unpack else 0.0
    mix_vector = float(
        n_m_eff
        * (
            2.0 * eval_context.rows * geometry.hidden
            + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
        )
    )
    sub_vector = _subchannel_scale_vector_flops(
        layout.routed_rows,
        layout.routed_act_elems,
        dtypes.w_expert,
        geometry.weight_block_size,
        block_k_floor=eval_context.chip.subchannel_block_k_floor,
    ) + _subchannel_scale_vector_flops(
        layout.sh_rows,
        layout.shared_elems,
        dtypes.w_dense,
        geometry.weight_block_size,
        block_k_floor=eval_context.chip.subchannel_block_k_floor,
    )
    sub_vector_min = _subchannel_scale_vector_flops(
        layout.routed_rows,
        layout.routed_act_elems_min,
        dtypes.w_expert,
        geometry.weight_block_size,
        block_k_floor=eval_context.chip.subchannel_block_k_floor,
    ) + _subchannel_scale_vector_flops(
        layout.sh_rows,
        layout.shared_elems,
        dtypes.w_dense,
        geometry.weight_block_size,
        block_k_floor=eval_context.chip.subchannel_block_k_floor,
    )
    shared_unpack = _sub_byte_unpack_vector_flops(
        layout.shared_elems + layout.latent_proj_sharded,
        dtypes.w_dense,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
        act=dtypes.act,
    )
    vector_flops = (
        sh_vector + routed_vector + fp4_unpack + shared_unpack + mix_vector + sub_vector
    )
    strict_vector_flops = (
        sh_vector
        + routed_vector_min
        + _strict_sub_byte_unpack_vector_flops(fp4_unpack_min)
        + _strict_sub_byte_unpack_vector_flops(shared_unpack)
        + mix_vector
        + sub_vector_min
    )
    return vector_flops, strict_vector_flops


def _compute_routed_moe_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    *,
    n_m_eff: float,
) -> ModuleCost:
    """Compute ModuleCost for routed MoE FFN layer."""
    layout = _routed_moe_layout(
        geometry, parallel, dtypes, eval_context, n_m_eff=n_m_eff
    )
    routed_buckets = _route_matmul_flops(
        layout.routed_rows,
        layout.routed_act_elems,
        dtypes.w_expert,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_expert_dtype,
    )
    routed_min_buckets = _route_matmul_flops(
        layout.routed_rows,
        layout.routed_act_elems_min,
        dtypes.w_expert,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_expert_dtype,
    )
    routed_base_bucket, _, _ = eval_context.chip.matmul_peak_kind(
        dtypes.w_expert, dtypes.act, weight_dtype=dtypes.w_expert_dtype
    )
    routed_base_flops = (
        MATMUL_FLOP_PER_MAC * layout.routed_rows * layout.routed_act_elems_min
    )
    latent_buckets = _MatmulBuckets()
    if geometry.routed_expert_hidden > 0:
        latent_flops_elems = float(
            n_m_eff
            * _TWO_DIRECTIONS
            * geometry.hidden
            * geometry.routed_expert_hidden
            / layout.tp
        )
        latent_buckets = _route_matmul_flops(
            eval_context.rows,
            latent_flops_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
    if (
        eval_context.mxu_m_tile > 1
        and layout.routed_rows > 0
        and layout.active_experts > 0.0
        and geometry.top_k > 0
    ):
        ed_global = (
            experts_touched(geometry.num_experts, geometry.top_k, layout.routed_rows)
            if parallel.moe == "ep"
            else layout.active_experts
        )
        m_per_touched = float(layout.routed_rows * geometry.top_k) / max(1.0, ed_global)
        if m_per_touched > 0.0:
            padded_token_rows = float(
                math.ceil(m_per_touched / eval_context.mxu_m_tile)
                * eval_context.mxu_m_tile
            )
            pad_mult = padded_token_rows / m_per_touched
            routed_buckets = routed_buckets.scale(pad_mult)
    lhs_floor = (
        (
            eval_context.mxu_lhs_rows_bf16
            if eval_context.mxu_lhs_rows_bf16 is not None
            else eval_context.chip.mxu_lhs_rows_bf16
        )
        if routed_base_bucket == "bf16"
        else (
            eval_context.mxu_lhs_rows_fp8
            if eval_context.mxu_lhs_rows_fp8 is not None
            else eval_context.chip.mxu_lhs_rows_fp8
        )
    )
    if (
        lhs_floor > 1
        and layout.routed_rows > 0
        and layout.min_active_experts > 0.0
        and geometry.top_k > 0
    ):
        pad_mult_strict = max(1.0, float(lhs_floor) / float(layout.routed_rows))
        routed_min_buckets = routed_min_buckets.scale(pad_mult_strict)
    routed_buckets = routed_buckets + latent_buckets
    routed_min_buckets = routed_min_buckets + latent_buckets
    non_routed_weight_bytes = (
        layout.shared_elems * dtypes.w_dense
        + layout.router_elems * dtypes.router
        + (
            layout.latent_proj_sharded * dtypes.w_dense
            if geometry.routed_expert_hidden > 0
            else 0.0
        )
    )
    decode_tokens_for_envelope = (
        eval_context.decode_rows * layout.dp_per_ep
        if layout.is_ep
        else eval_context.decode_rows
    )
    prefill_tokens_for_envelope = (
        eval_context.eff_prefill * layout.pcp_per_ep * layout.dp_per_ep
        if layout.is_ep
        else eval_context.eff_prefill
    )
    moe_weight_bytes_envelope, routed_env_flops = _compute_routed_moe_envelope(
        geometry,
        dtypes,
        eval_context,
        _RoutedEnvelopeSpec(
            ep_shards=layout.ep_ranks if parallel.moe == "ep" else 1,
            decode_tokens=decode_tokens_for_envelope,
            prefill_tokens=prefill_tokens_for_envelope,
            routed_rows=layout.routed_rows,
            routed_unit=layout.routed_unit,
            non_routed_weight_bytes=non_routed_weight_bytes,
            routed_base_flops=routed_base_flops,
            lhs_floor=lhs_floor,
        ),
    )
    routed_env_buckets = (
        _bucket_from_name(routed_base_bucket, routed_env_flops) + latent_buckets
    )
    shared_buckets = _route_matmul_flops(
        layout.sh_rows,
        layout.shared_mxu_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    shared_min_buckets = _route_matmul_flops(
        layout.sh_rows,
        layout.shared_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    router_buckets = _route_matmul_flops(
        eval_context.rows,
        layout.router_elems,
        dtypes.router,
        dtypes.act,
        chip=eval_context.chip,
    )
    moe_spec = _MoeLayerSpec(
        rows=eval_context.rows,
        hidden=geometry.hidden,
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        block_parallel=geometry.block_parallel,
        decode_rows=eval_context.decode_rows,
        eff_prefill=eval_context.eff_prefill,
        dispatch_bytes=activation_wire_bytes(dtypes.act),
        shared_inter=geometry.shared_inter,
        has_tp_latent_proj=geometry.routed_expert_hidden > 0,
    )
    moe_layer_count = max(1, round(n_m_eff))
    moe_colls, moe_ucolls = _repeat_colls(
        _moe_layer_collectives(moe_spec, parallel),
        moe_layer_count,
    )
    _, moe_strict_ucolls = _repeat_colls(
        _moe_layer_strict_collectives(moe_spec, parallel),
        moe_layer_count,
    )
    vector_flops, strict_vector_flops = _routed_moe_vector_flops(
        geometry, dtypes, eval_context, layout, n_m_eff=n_m_eff
    )
    moe_total = routed_buckets + shared_buckets + router_buckets
    moe_min_total = routed_min_buckets + shared_min_buckets + router_buckets
    moe_env_total = routed_env_buckets + shared_min_buckets + router_buckets
    return ModuleCost(
        name="moe",
        hbm_bytes=layout.moe_w_bytes,
        fp8_flops=moe_total.fp8,
        bf16_flops=moe_total.bf16,
        collectives=moe_colls,
        hbm_bytes_min=layout.moe_w_bytes_min,
        unique_collectives=moe_ucolls,
        strict_collectives=moe_strict_ucolls,
        fp8_flops_min=moe_min_total.fp8,
        bf16_flops_min=moe_min_total.bf16,
        vector_flops=vector_flops,
        strict_vector_flops=strict_vector_flops,
        hbm_bytes_envelope=moe_weight_bytes_envelope,
        fp8_flops_envelope=moe_env_total.fp8,
        bf16_flops_envelope=moe_env_total.bf16,
        fp4_flops=moe_total.fp4,
        int8_flops=moe_total.int8,
        int4_flops=moe_total.int4,
        fp4_flops_min=moe_min_total.fp4,
        int8_flops_min=moe_min_total.int8,
        int4_flops_min=moe_min_total.int4,
        fp4_flops_envelope=moe_env_total.fp4,
        int8_flops_envelope=moe_env_total.int8,
        int4_flops_envelope=moe_env_total.int4,
    )


def _compute_moe_mlp_costs(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
) -> list[ModuleCost]:
    """Compute ModuleCost for dense MLP and routed MoE layers."""
    tp = parallel.tp
    out: list[ModuleCost] = []
    kinds = layer_ffn_kinds(geometry)
    n_moe_ffn = kinds.count("moe")
    n_dense_ffn = kinds.count("dense")
    if n_dense_ffn > 0 and geometry.dense_inter > 0:
        n_d_eff = n_dense_ffn * eval_context.pp_scale
        mlp_inter = geometry.dense_inter // tp
        ffn_mats = geometry.ffn_num_matrices
        ffn_act_vector = (
            1.0 if ffn_mats == NON_GATED_MATMUL_COUNT else float(GLU_MATMUL_COUNT)
        )
        mlp_elems = float(n_d_eff * ffn_mats * geometry.hidden * mlp_inter)
        mlp_mxu_elems = float(
            n_d_eff * ffn_mats * geometry.hidden * _align_mxu(mlp_inter, eval_context)
        )
        mlp_buckets = _route_matmul_flops(
            eval_context.rows,
            mlp_mxu_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
        mlp_min_buckets = _route_matmul_flops(
            eval_context.rows,
            mlp_elems,
            dtypes.w_dense,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
        )
        eff_ranks = (
            parallel.ranks_per_chip
            if parallel.ranks_per_chip is not None
            else eval_context.chip.cores_per_chip
        )
        d_par = (
            Parallel(
                tp=parallel.tp,
                residual=parallel.residual,
                ar_algo=parallel.ar_algo,
                ranks_per_chip=eff_ranks,
                moe="ep" if parallel.residual == "sp" else "tp_slice",
            )
            if geometry.num_experts > 0
            else parallel
        )
        mlp_colls, mlp_ucolls = _repeat_colls(
            _moe_layer_collectives(
                _MoeLayerSpec(
                    rows=eval_context.rows,
                    hidden=geometry.hidden,
                    num_experts=0,
                    top_k=0,
                    block_parallel=geometry.block_parallel,
                    dispatch_bytes=activation_wire_bytes(dtypes.act),
                ),
                d_par,
            ),
            max(1, round(n_d_eff)),
        )
        mlp_swiglu = float(n_d_eff * ffn_act_vector * eval_context.rows * mlp_inter)
        mlp_mix = float(
            n_d_eff
            * (
                2.0 * eval_context.rows * geometry.hidden
                + 2.0 * eval_context.rows * geometry.hidden * geometry.hc_mult
            )
        )
        dense_unpack = _sub_byte_unpack_vector_flops(
            mlp_elems,
            dtypes.w_dense,
            chip=eval_context.chip,
            weight_dtype=dtypes.w_dense_dtype,
            act=dtypes.act,
        )
        vector_flops = (
            mlp_swiglu
            + mlp_mix
            + _subchannel_scale_vector_flops(
                eval_context.rows,
                mlp_elems,
                dtypes.w_dense,
                geometry.weight_block_size,
                block_k_floor=eval_context.chip.subchannel_block_k_floor,
            )
            + dense_unpack
        )
        strict_vector_flops = (
            vector_flops
            - dense_unpack
            + _strict_sub_byte_unpack_vector_flops(dense_unpack)
        )
        out.append(
            ModuleCost(
                name="mlp",
                hbm_bytes=mlp_elems * dtypes.w_dense,
                fp8_flops=mlp_buckets.fp8,
                bf16_flops=mlp_buckets.bf16,
                collectives=mlp_colls,
                unique_collectives=mlp_ucolls,
                fp8_flops_min=mlp_min_buckets.fp8,
                bf16_flops_min=mlp_min_buckets.bf16,
                vector_flops=vector_flops,
                strict_vector_flops=strict_vector_flops,
                fp4_flops=mlp_buckets.fp4,
                int8_flops=mlp_buckets.int8,
                int4_flops=mlp_buckets.int4,
                fp4_flops_min=mlp_min_buckets.fp4,
                int8_flops_min=mlp_min_buckets.int8,
                int4_flops_min=mlp_min_buckets.int4,
            )
        )
    if n_moe_ffn > 0:
        n_m_eff = n_moe_ffn * eval_context.pp_scale
        out.append(
            _compute_routed_moe_cost(
                geometry,
                parallel,
                dtypes,
                eval_context,
                n_m_eff=n_m_eff,
            )
        )
    return out


def _compute_ple_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
) -> ModuleCost | None:
    """Compute ModuleCost for Per-Layer Embedding (PLE)."""
    if "ple" not in geometry.layer_types and geometry.ple_dim <= 0:
        return None
    if geometry.ple_dim <= 0:
        msg = "PLE layer requires config field 'ple_dim'"
        raise ValueError(msg)
    tp = parallel.tp
    ple_d = geometry.ple_dim
    ple_count = (
        len(geometry.ple_layer_ids) if geometry.ple_layer_ids else geometry.num_layers
    )
    n_ple = float(ple_count) * eval_context.pp_scale
    ple_proj_elems = float(n_ple * geometry.hidden * ple_d // tp)
    ple_lookup_bytes = float(n_ple * eval_context.rows * ple_d) * dtypes.w_aux
    ple_buckets = _route_matmul_flops(
        eval_context.rows,
        ple_proj_elems,
        dtypes.w_aux,
        dtypes.act,
        chip=eval_context.chip,
    )
    return ModuleCost(
        name="ple",
        hbm_bytes=ple_proj_elems * dtypes.w_aux + ple_lookup_bytes,
        fp8_flops=ple_buckets.fp8,
        bf16_flops=ple_buckets.bf16,
        vector_flops=0.0,
        strict_vector_flops=0.0,
        fp4_flops=ple_buckets.fp4,
        int8_flops=ple_buckets.int8,
        int4_flops=ple_buckets.int4,
    )


def _compute_hc_lowrank_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
) -> ModuleCost | None:
    """Compute ModuleCost for low-rank hyper-connection stream projections."""
    if geometry.hc_mult <= 1 or geometry.hc_lowrank <= 0 or geometry.num_layers <= 0:
        return None
    tp = max(1, parallel.tp)
    n_eff = float(geometry.num_layers) * eval_context.pp_scale
    hc_elems = (
        n_eff
        * float(
            _TWO_DIRECTIONS * geometry.hidden * geometry.hc_lowrank * geometry.hc_mult
        )
        / float(tp)
    )
    hc_buckets = _route_matmul_flops(
        eval_context.rows,
        hc_elems,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    return ModuleCost(
        name="hc_proj",
        hbm_bytes=hc_elems * dtypes.w_dense,
        fp8_flops=hc_buckets.fp8,
        bf16_flops=hc_buckets.bf16,
        vector_flops=0.0,
        strict_vector_flops=0.0,
        fp4_flops=hc_buckets.fp4,
        int8_flops=hc_buckets.int8,
        int4_flops=hc_buckets.int4,
    )


def _compute_lm_head_cost(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
) -> ModuleCost:
    """Compute ModuleCost for LM head and PP handoff collectives."""
    tp = parallel.tp
    pp_colls = (
        (
            _make_coll(
                "pp_p2p",
                "p2p",
                float(eval_context.rows * geometry.hidden * geometry.hc_mult)
                * ACTIVATION_BYTES_BF16,
                parallel.pp,
                parallel,
            ),
        )
        if parallel.pp > 1
        else ()
    )
    # Pure prefill charges 1 sampled row/req; mixed steps charge decode_rows
    # (runner discards logits for incomplete prefill chunks).
    sampled_rows = (
        eval_context.decode_rows
        if eval_context.decode_rows > 0
        else max(1, eval_context.batch)
    )
    # Last pipeline stage holds lm_head whole (sharded only across TP).
    lm_elems = float(geometry.vocab * geometry.hidden // tp)
    lm_buckets = _route_matmul_flops(
        sampled_rows,
        lm_elems,
        dtypes.lm_head,
        dtypes.act,
        chip=eval_context.chip,
    )
    lm_act_bytes = activation_wire_bytes(dtypes.act)
    lm_all_colls = (
        _lm_head_collectives(
            sampled_rows,
            geometry.hidden,
            geometry.vocab,
            parallel,
            act_bytes=lm_act_bytes,
        )
        + pp_colls
    )
    lm_colls, lm_ucolls = _repeat_colls(lm_all_colls, 1)
    vector_flops = float(2.0 * sampled_rows * geometry.hidden)
    return ModuleCost(
        name="lm_head",
        hbm_bytes=lm_elems * dtypes.lm_head,
        fp8_flops=lm_buckets.fp8,
        bf16_flops=lm_buckets.bf16,
        collectives=lm_colls,
        unique_collectives=lm_ucolls,
        vector_flops=vector_flops,
        strict_vector_flops=vector_flops,
        fp4_flops=lm_buckets.fp4,
        int8_flops=lm_buckets.int8,
        int4_flops=lm_buckets.int4,
    )


class PassCostOptions(TypedDict, total=False):
    """Optional configuration overrides for :func:`pass_costs`."""

    mxu_m_tile: int
    mxu_lhs_rows_bf16: int | None
    mxu_lhs_rows_fp8: int | None
    fp8_to_bf16_ratio: float | None
    activation_model: ExpertActivationModel
    chip: ChipSpec | None


def pass_costs(
    geometry: ModelGeometry,
    parallel: Parallel,
    shape: StepShape,
    dtypes: Dtypes,
    **options: Unpack[PassCostOptions],
) -> tuple[ModuleCost, ...]:
    """Compute per-module resource work for one decode, prefill, or mixed pass."""
    mxu_m_tile = options.get("mxu_m_tile", 1)
    chip = options.get("chip")
    eff_chip = chip if chip is not None else CHIPS["tpu7x"]
    raw_lhs_bf16 = options.get("mxu_lhs_rows_bf16")
    mxu_lhs_rows_bf16 = (
        eff_chip.mxu_lhs_rows_bf16 if raw_lhs_bf16 is None else raw_lhs_bf16
    )
    raw_lhs_fp8 = options.get("mxu_lhs_rows_fp8")
    mxu_lhs_rows_fp8 = eff_chip.mxu_lhs_rows_fp8 if raw_lhs_fp8 is None else raw_lhs_fp8
    raw_fp8_ratio = options.get("fp8_to_bf16_ratio")
    default_fp8_ratio = (
        eff_chip.fp8_flops / eff_chip.bf16_flops
        if eff_chip.fp8_flops > 0.0 and eff_chip.bf16_flops > 0.0
        else 0.0
    )
    fp8_to_bf16_ratio = default_fp8_ratio if raw_fp8_ratio is None else raw_fp8_ratio
    activation_model = options.get("activation_model", _DEFAULT_ACTIVATION_MODEL)
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else eff_chip.cores_per_chip
    )
    eff_parallel = (
        parallel
        if parallel.ranks_per_chip is not None
        else replace(parallel, ranks_per_chip=eff_ranks)
    )
    _validate_pass_costs_config(geometry, eff_parallel)

    pp_scale = (
        float(math.ceil(geometry.num_layers / max(1, eff_parallel.pp)))
        / float(geometry.num_layers)
        if geometry.num_layers > 0 and eff_parallel.pp > 1
        else 1.0
    )
    decode_rows = shape.batch * shape.rows_per_seq
    eff_prefill = (
        max(
            1,
            math.ceil(
                shape.prefill_tokens
                * max(0.0, 1.0 - shape.prefix_hit_rate)
                / max(1, eff_parallel.pcp)
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
        shared_prefix_tokens=max(0, shape.shared_prefix_tokens),
        mxu_m_tile=mxu_m_tile,
        mxu_lhs_rows_bf16=mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=mxu_lhs_rows_fp8,
        fp8_to_bf16_ratio=fp8_to_bf16_ratio,
        activation_model=activation_model,
        chip=eff_chip,
    )

    counts = count_layer_types(geometry)
    emb_colls, emb_ucolls = _repeat_colls(
        _embed_collectives(rows, geometry.hidden, eff_parallel), 1
    )
    out: list[ModuleCost] = [
        ModuleCost(
            name="embed",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=emb_colls,
            unique_collectives=emb_ucolls,
            vector_flops=0.0,
            strict_vector_flops=0.0,
        )
    ]

    if counts.gqa > 0:
        out.append(
            _compute_gqa_cost(geometry, eff_parallel, dtypes, eval_context, counts)
        )

    if counts.swa > 0:
        out.append(
            _compute_swa_cost(
                geometry, eff_parallel, dtypes, eval_context, counts=counts
            )
        )

    if counts.mla > 0 or counts.dsa > 0:
        out.extend(
            _compute_mla_dsa_costs(geometry, eff_parallel, dtypes, eval_context, counts)
        )

    if counts.gdn > 0:
        out.append(
            _compute_gdn_cost(
                geometry, eff_parallel, dtypes, eval_context, n_gdn=counts.gdn
            )
        )

    if counts.kda > 0:
        out.append(
            _compute_kda_cost(
                geometry, eff_parallel, dtypes, eval_context, n_kda=counts.kda
            )
        )

    out.extend(_compute_moe_mlp_costs(geometry, eff_parallel, dtypes, eval_context))

    ple = _compute_ple_cost(geometry, eff_parallel, dtypes, eval_context)
    if ple is not None:
        out.append(ple)
    hc_cost = _compute_hc_lowrank_cost(geometry, eff_parallel, dtypes, eval_context)
    if hc_cost is not None:
        out.append(hc_cost)

    out.append(_compute_lm_head_cost(geometry, eff_parallel, dtypes, eval_context))

    return tuple(out)
