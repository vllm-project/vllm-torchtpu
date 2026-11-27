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
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

if TYPE_CHECKING:
    from collections.abc import Sequence

from .chips import CHIPS, ChipSpec
from .collectives import Algo, Collective, CollectiveConfig, Kind, make
from .constants import (
    ACTIVATION_BYTES_BF16,
    BF16_BYTES,
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_SSM_STATE_BYTES,
    FP4_BYTES,
    FP8_BYTES,
    FP32_BYTES,
    GLU_MATMUL_COUNT,
    HBM_MIN_PACKING_DIVISOR,
    HBM_PACKING_NUMERATOR,
    HEAD_DIM_PACKING_THRESHOLD,
    INT4_INDEX_SCALE_BYTES,
    INT4_PER_TOKEN_HEAD_KV_DTYPE,
    INT4_PER_TOKEN_HEAD_SCALE_BYTES,
    KV_DTYPE_BYTES,
    MATMUL_FLOP_PER_MAC,
    NON_GATED_MATMUL_COUNT,
    SPARSE_MLA_SCALE_BYTES,
    SUBCHANNEL_BLOCK_K_FLOOR,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from .geometry import (
    ExecutionMode,
    HostTableResidency,
    HostTableSpec,
    LayerCounts,
    ModelGeometry,
    SparseAttnSpec,
    count_layer_types,
    host_table_hbm_fraction,
    layer_ffn_kinds,
    slice_geometry_layers,
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


def role_dtype_bytes(
    dtypes: Dtypes, role: Literal["w_expert", "w_dense", "w_aux"]
) -> float:
    """Return byte width for a named weight dtype role."""
    return {"w_expert": dtypes.w_expert, "w_aux": dtypes.w_aux}.get(
        role, dtypes.w_dense
    )


def is_unit_dtypes(dtypes: Dtypes) -> bool:
    """Return True when every tensor role byte width equals 1.0 (element mode)."""
    return (
        dtypes.w_dense == 1.0
        and dtypes.w_expert == 1.0
        and dtypes.w_aux == 1.0
        and dtypes.kv == 1.0
        and dtypes.state == 1.0
        and dtypes.conv == 1.0
        and dtypes.router == 1.0
        and dtypes.lm_head == 1.0
    )


def host_table_elem_bytes(spec: HostTableSpec, dtypes: Dtypes) -> float:
    """Return bytes per host-table element, yielding 1.0 under unit dtypes."""
    extra = 0.0 if is_unit_dtypes(dtypes) else max(0.0, spec.scale_ratio)
    return role_dtype_bytes(dtypes, spec.table_dtype_role) + extra


def draft_kv_dtypes(geometry: ModelGeometry, dtypes: Dtypes) -> Dtypes:
    """Return Dtypes with DraftSpec.draft_kv_dtype applied when non-unit."""
    draft = geometry.draft
    if draft is None or not draft.draft_kv_dtype or is_unit_dtypes(dtypes):
        return dtypes
    dt = draft.draft_kv_dtype
    return replace(dtypes, kv=KV_DTYPE_BYTES[dt], kv_dtype=dt)


def draft_parallel_for_geometry(
    draft_geom: ModelGeometry, parallel: Parallel
) -> Parallel:
    """Return PP=1 Parallel with kv_split clamped to divide draft kv_group."""
    tp = max(1, parallel.tp)
    is_tp_dcp = (
        draft_geom.sparse_attn is not None and draft_geom.sparse_attn.dcp_span == "tp"
    )
    kv_group = (
        max(1, tp // draft_geom.num_kv_heads)
        if draft_geom.num_kv_heads > 0
        and draft_geom.kv_lora_rank == 0
        and not is_tp_dcp
        else tp
    )
    eff_kv_split = (
        parallel.kv_split
        if kv_group % parallel.kv_split == 0
        else math.gcd(parallel.kv_split, kv_group)
    )
    return replace(parallel, pp=1, kv_split=max(1, eff_kv_split))


def kv_heads_on_rank(geometry: ModelGeometry, parallel: Parallel) -> int:
    """Return KV heads stored on one rank, honoring TP-spanning DCP."""
    if geometry.num_kv_heads <= 0:
        return 0
    tp = max(1, parallel.tp)
    if geometry.sparse_attn is not None and geometry.sparse_attn.dcp_span == "tp":
        return min(
            geometry.num_kv_heads,
            max(1, (geometry.num_kv_heads * max(1, parallel.kv_split)) // tp),
        )
    return max(1, geometry.num_kv_heads // tp)


def csa2_layer_weight_elems(geometry: ModelGeometry, tp: int) -> float:
    """Return per-core attention weight elements for one CSA2 layer."""
    eff_tp = max(1, tp)
    nq, h, hd = max(1, geometry.num_heads // eff_tp), geometry.hidden, geometry.head_dim
    q_r = geometry.q_lora_rank
    q_w = float(h * q_r + q_r * nq * hd if q_r > 0 else h * nq * hd)
    sparse = geometry.sparse_attn
    o_r = sparse.o_lora_rank if sparse is not None else 0
    o_g = sparse.o_groups if sparse is not None else 1
    o_w = (
        float(nq * hd * o_r) + float(o_g * o_r * h) / float(eff_tp)
        if o_r > 0
        else float(nq * hd * h)
    )
    return q_w + float(h * hd) + o_w


def sparse_gqa_index_layer_weight_elems(
    geometry: ModelGeometry, tp: int, *, strict: bool = False
) -> float:
    """Return per-core sparse GQA/MSA/QSA indexer weight elements per layer."""
    if geometry.index_n_heads <= 0 or geometry.index_head_dim <= 0:
        return 0.0
    heads = (
        max(1, geometry.index_n_heads // max(1, tp))
        if strict
        else geometry.index_n_heads
    )
    return float(geometry.hidden * (heads + 1) * geometry.index_head_dim)


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
    execution: ExecutionMode = "full"
    host_table_residency: HostTableResidency = "auto"
    prefill_draft: bool = False


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
    pcie_h2d_s: float = 0.0

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
    rows: float,
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
    unpadded_linear: bool = False
    chip: ChipSpec = field(default_factory=lambda: CHIPS["tpu7x"])
    execution: ExecutionMode = "full"
    host_table_residency: HostTableResidency = "auto"


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
    is_tp_dcp = (
        geometry.sparse_attn is not None and geometry.sparse_attn.dcp_span == "tp"
    )
    kv_group = (
        max(1, tp // geometry.num_kv_heads)
        if geometry.num_kv_heads > 0 and geometry.kv_lora_rank == 0 and not is_tp_dcp
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
    num_kv_heads = (
        kv_heads_on_rank(geometry, parallel) if geometry.num_kv_heads > 0 else 0
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


def _int4_pth_kv(head_dim: int, h_rank: int) -> float:
    """Return per-token 4-bit per-token-head KV bytes including 32B scale cache."""
    return (
        float(_TWO_DIRECTIONS * head_dim * h_rank) * FP4_BYTES
        + INT4_PER_TOKEN_HEAD_SCALE_BYTES
    )


def _msa_kv_bytes_per_tok(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    head_dim: int,
    sparse_attn: SparseAttnSpec,
) -> tuple[float, float]:
    """Return (main_kv_bytes, index_bytes) per token per layer for MSA."""
    h_rank = kv_heads_on_rank(geometry, parallel)
    is_i4 = dtypes.kv_dtype == INT4_PER_TOKEN_HEAD_KV_DTYPE
    main_b = (
        _int4_pth_kv(head_dim, h_rank)
        if is_i4
        else float(_TWO_DIRECTIONS * head_dim * h_rank) * dtypes.kv
    )
    if geometry.index_head_dim <= 0:
        return main_b, 0.0
    idx_raw = (
        float(geometry.index_head_dim) * FP4_BYTES + INT4_INDEX_SCALE_BYTES
        if is_i4
        else float(geometry.index_head_dim) * dtypes.kv
    )
    return main_b, idx_raw / (
        float(max(1, parallel.tp)) if sparse_attn.index_cache_sharded else 1.0
    )


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
    if resolved in ("sparse_gqa", "indexed_attention"):
        resolved = "gqa"
    elif resolved == "csa2":
        resolved = "dsa"
    elif resolved in ("full_attention", "attention"):
        if geometry.kv_lora_rank > 0 and geometry.index_topk > 0:
            resolved = "dsa"
        elif geometry.kv_lora_rank > 0:
            resolved = "mla"
        else:
            resolved = "gqa"

    sparse = geometry.sparse_attn
    if resolved == "gqa":
        gqa_hd = geometry.global_head_dim or geometry.head_dim
        if sparse is not None and sparse.kind == "msa":
            main_b, idx_b = _msa_kv_bytes_per_tok(
                geometry, dtypes, parallel, gqa_hd, sparse
            )
            return main_b + (idx_b if sparse.index_source_layer_ids else 0.0)
        extra = (
            sparse.compressed_key_bytes_per_tok
            if (
                sparse is not None
                and sparse.kind == "qsa"
                and sparse.index_source_layer_ids
            )
            else 0.0
        )
        if dtypes.kv_dtype == INT4_PER_TOKEN_HEAD_KV_DTYPE:
            return _int4_pth_kv(gqa_hd, kv_heads_on_rank(geometry, parallel)) + extra
        main_b = _gqa_swa_kv_bytes_per_tok(
            geometry, dtypes, parallel, gqa_hd, physical_align=align if physical else 0
        )
        return main_b + extra
    if resolved == "swa":
        if dtypes.kv_dtype == INT4_PER_TOKEN_HEAD_KV_DTYPE:
            return _int4_pth_kv(geometry.head_dim, kv_heads_on_rank(geometry, parallel))
        return _gqa_swa_kv_bytes_per_tok(
            geometry,
            dtypes,
            parallel,
            geometry.head_dim,
            physical_align=align if physical else 0,
        )
    if resolved in ("mla", "dsa"):
        is_csa2 = sparse is not None and sparse.kind == "csa2"
        latent_dim = geometry.kv_lora_rank or (geometry.head_dim if is_csa2 else 0)
        raw_latent_w = latent_dim + geometry.qk_rope_head_dim
        latent_w = math.ceil(raw_latent_w / align) * align if physical else raw_latent_w
        mla_latent_b = float(latent_w) * dtypes.kv
        has_sparse = resolved == "dsa" and (
            (geometry.index_topk > 0 and geometry.index_head_dim > 0) or is_csa2
        )
        idx_scale = (
            SPARSE_MLA_SCALE_BYTES if (physical and dtypes.kv <= FP8_BYTES) else 0.0
        )
        if (
            is_csa2
            and sparse is not None
            and sparse.index_record_bytes > 0.0
            and has_sparse
        ):
            dsa_idx_b = sparse.index_record_bytes
        else:
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


def _shared_dec_toks(
    eval_context: _StepContext, ctx: int, kv_div: float
) -> tuple[float, float]:
    """Return (nominal_decode_tokens, shared_prefix_min_decode_tokens)."""
    dec = float(eval_context.batch * ctx) / kv_div
    c_sh = min(ctx, max(0, eval_context.shared_prefix_tokens))
    dec_min = (
        float(c_sh + eval_context.batch * max(0, ctx - c_sh)) / kv_div
        if (eval_context.batch > 0 and c_sh > 0)
        else dec
    )
    return dec, dec_min


def _sparse_gqa_extras(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
    gqa_info: tuple[int, float, float, float],
) -> tuple[float, float, float, float, float, float, _MatmulBuckets, _MatmulBuckets]:
    """Compute weighted context, KV bytes, indexer weights, and FLOPs for MSA/QSA."""
    gqa_hd, n_eff, kv_div, pre_toks = gqa_info
    sparse, zero = geometry.sparse_attn, _MatmulBuckets()
    if sparse is None or sparse.kind not in ("msa", "qsa"):
        return (
            n_eff * float(eval_context.context),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            zero,
            zero,
        )
    budget = (
        (sparse.sparse_topk_blocks + sparse.sparse_local_block)
        * sparse.sparse_block_size
        if sparse.kind == "msa"
        else geometry.qsa_budget
    )
    ctx, cap = eval_context.context, sparse.candidate_scan_cap
    att_ctx = min(ctx, budget) if (budget > 0 and ctx > 0) else ctx
    scan_ctx = min(ctx, cap) if (cap > 0 and ctx > 0) else ctx
    if sparse.kind == "msa":
        main_p, idx_kv_b = _msa_kv_bytes_per_tok(
            geometry, dtypes, parallel, gqa_hd, sparse
        )
        main_m = main_p
    else:
        if dtypes.kv_dtype == INT4_PER_TOKEN_HEAD_KV_DTYPE:
            main_p = main_m = _int4_pth_kv(gqa_hd, kv_heads_on_rank(geometry, parallel))
        else:
            al = eval_context.chip.physical_alignment_elems
            main_p = _gqa_swa_kv_bytes_per_tok(
                geometry, dtypes, parallel, gqa_hd, physical_align=al
            )
            main_m = _gqa_swa_kv_bytes_per_tok(
                geometry, dtypes, parallel, gqa_hd, physical_align=0
            )
        idx_kv_b = sparse.compressed_key_bytes_per_tok
    n_sp = min(n_eff, float(len(sparse.kv_source_layer_ids)) * eval_context.pp_scale)
    n_dense, n_idx = (
        max(0.0, n_eff - n_sp),
        float(len(sparse.index_source_layer_ids)) * eval_context.pp_scale,
    )
    idx_we = n_idx * sparse_gqa_index_layer_weight_elems(geometry, parallel.tp)
    idx_ws = n_idx * sparse_gqa_index_layer_weight_elems(
        geometry, parallel.tp, strict=True
    )
    dec_sp_t, dec_sp_m = _shared_dec_toks(eval_context, att_ctx, kv_div)
    dec_full_t, dec_full_m = _shared_dec_toks(eval_context, ctx, kv_div)
    kv_p = (n_sp * dec_sp_t + n_dense * dec_full_t + n_eff * pre_toks) * main_p
    kv_m = (n_sp * dec_sp_m + n_dense * dec_full_m + n_eff * pre_toks) * main_m
    pre_g = eval_context.eff_prefill * parallel.pcp
    idx_dec = eval_context.batch > 0 and ctx > budget > 0
    idx_pre = (
        eval_context.eff_prefill > 0
        and budget > 0
        and (eval_context.cached_prefix + pre_g) > budget
    )
    if idx_dec and idx_kv_b > 0.0 and n_idx > 0.0:
        sc_t, sc_m = _shared_dec_toks(eval_context, scan_ctx, kv_div)
        kv_p, kv_m = kv_p + n_idx * sc_t * idx_kv_b, kv_m + n_idx * sc_m * idx_kv_b
    if eval_context.eff_prefill > 0 and idx_kv_b > 0.0 and n_idx > 0.0:
        idx_pre_toks = pre_toks if idx_pre else float(eval_context.eff_prefill) / kv_div
        kv_p, kv_m = (
            kv_p + n_idx * idx_pre_toks * idx_kv_b,
            kv_m + n_idx * idx_pre_toks * idx_kv_b,
        )
    idx_fe, idx_fs = zero, zero
    if geometry.index_n_heads > 0 and geometry.index_head_dim > 0 and n_idx > 0.0:
        pairs = (float(eval_context.decode_rows * scan_ctx) if idx_dec else 0.0) + (
            _capped_causal_pairs(eval_context.cached_prefix, pre_g, cap)
            / float(max(1, parallel.pcp))
            if idx_pre
            else 0.0
        )
        if pairs > 0.0:
            bkt = "fp8" if eval_context.chip.fp8_flops > 0.0 else "bf16"
            unit = pairs * MATMUL_FLOP_PER_MAC * float(geometry.index_head_dim) * n_idx
            sh = max(1, geometry.index_n_heads // max(1, parallel.tp))
            idx_fe = _bucket_from_name(bkt, unit * float(geometry.index_n_heads))
            idx_fs = _bucket_from_name(bkt, unit * float(sh))
    full_pre_pairs = float(eval_context.eff_prefill) * (
        eval_context.cached_prefix + 0.5 * pre_g
    )
    sp_pre_pairs = (
        _capped_causal_pairs(eval_context.cached_prefix, pre_g, budget)
        / float(max(1, parallel.pcp))
        if budget > 0
        else full_pre_pairs
    )
    return (
        n_sp * float(att_ctx) + n_dense * float(ctx),
        n_sp * sp_pre_pairs + n_dense * full_pre_pairs,
        kv_p,
        kv_m,
        idx_we,
        idx_ws,
        idx_fe,
        idx_fs,
    )


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
    num_kv_heads = kv_heads_on_rank(geometry, parallel)
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
    is_sparse_gqa = geometry.sparse_attn is not None and geometry.sparse_attn.kind in (
        "msa",
        "qsa",
    )
    dec_ctx_layers, sp_pre_pairs, sp_kv_p, sp_kv_m, idx_we, idx_ws, idx_fe, idx_fs = (
        _sparse_gqa_extras(
            geometry,
            parallel,
            dtypes,
            eval_context,
            (gqa_hd, n_eff, kv_split_div, kv_pre_tokens),
        )
    )
    att_w_elems_strict = att_w_elems + idx_ws
    att_w_elems += idx_we
    att_mxu_elems += idx_we
    gqa_kv_phys_bytes = sp_kv_p if is_sparse_gqa else kv_tokens_total * b_kv_phys
    gqa_kv_min_bytes = sp_kv_m if is_sparse_gqa else kv_tokens_min_total * b_kv_min
    idx_est_flops, idx_strict_flops = idx_fe, idx_fs
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
        att_w_elems_strict,
        dtypes.w_dense,
        dtypes.act,
        chip=eval_context.chip,
        weight_dtype=dtypes.w_dense_dtype,
    )
    dec_scores = float(
        eval_context.decode_rows
        * 4
        * gqa_hd
        * (geometry.num_heads // tp)
        * dec_ctx_layers
    )
    pre_ctx_eff = (
        eval_context.cached_prefix + 0.5 * eval_context.eff_prefill * parallel.pcp
    )
    pre_scores = (
        float(sp_pre_pairs * 4 * gqa_hd * (geometry.num_heads // tp))
        if is_sparse_gqa
        else float(
            eval_context.eff_prefill
            * 4
            * pre_ctx_eff
            * gqa_hd
            * (geometry.num_heads // tp)
            * n_eff
        )
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
        hbm_bytes=att_w_elems * dtypes.w_dense + gqa_kv_phys_bytes,
        fp8_flops=att_flops.fp8 + idx_est_flops.fp8,
        bf16_flops=att_flops.bf16 + score_tot + idx_est_flops.bf16,
        collectives=att_colls,
        hbm_bytes_min=att_w_elems_strict * dtypes.w_dense + gqa_kv_min_bytes,
        fp8_flops_min=(
            att_min_flops.fp8
            + idx_strict_flops.fp8
            + (score_tot if strict_attn_fp8 else 0.0)
        ),
        bf16_flops_min=(
            att_min_flops.bf16
            + idx_strict_flops.bf16
            + (0.0 if strict_attn_fp8 else score_tot)
        ),
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
    """Validate model geometry fields required for MLA, DSA, or CSA2."""
    is_csa2 = geometry.sparse_attn is not None and geometry.sparse_attn.kind == "csa2"
    if is_csa2 and geometry.head_dim > 0 and geometry.qk_rope_head_dim >= 0:
        return
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
    scan_cap = (
        geometry.sparse_attn.candidate_scan_cap
        if geometry.sparse_attn is not None
        else 0
    )
    scan_ctx = (
        min(eval_context.context, scan_cap)
        if scan_cap > 0 and eval_context.context > 0
        else eval_context.context
    )
    dec_pairs = float(dec_rows * scan_ctx) if dec_active else 0.0
    pre_global = eval_context.eff_prefill * parallel.pcp
    pre_active = (
        eval_context.eff_prefill > 0
        and (eval_context.cached_prefix + pre_global) > geometry.index_topk
    )
    pre_pairs = (
        _capped_causal_pairs(eval_context.cached_prefix, pre_global, scan_cap)
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
    sparse = geometry.sparse_attn
    is_csa2 = sparse is not None and sparse.kind == "csa2"
    kv_lora = (
        geometry.kv_lora_rank
        if geometry.kv_lora_rank > 0
        else (geometry.head_dim if is_csa2 else 0)
    )
    nope_d = geometry.qk_nope_head_dim or geometry.head_dim
    rope_d = geometry.qk_rope_head_dim
    v_head_dim = geometry.v_head_dim or nope_d
    nq_raw = num_q_heads * (nope_d + rope_d)
    q_w = (
        float(geometry.hidden * geometry.q_lora_rank + geometry.q_lora_rank * nq_raw)
        if geometry.q_lora_rank > 0
        else float(geometry.hidden * nq_raw)
    )
    mla_w_layer = (
        csa2_layer_weight_elems(geometry, parallel.tp)
        if is_csa2
        else (
            q_w
            + float(geometry.hidden * (kv_lora + rope_d))
            + float(
                kv_lora * num_q_heads * (nope_d + v_head_dim)
                + (num_q_heads * v_head_dim) * geometry.hidden
            )
        )
    )
    idx_scale = (
        float(geometry.indexer_types.count("full")) / float(n_fam)
        if (is_sparse and geometry.indexer_types)
        else (1.0 if is_sparse else 0.0)
    )
    n_idx_eff = (
        float(len(sparse.index_source_layer_ids)) * eval_context.pp_scale
        if (is_csa2 and sparse is not None)
        else n_eff * idx_scale
    )
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
    ukv_w_elems = (
        0.0 if is_csa2 else n_eff * float(kv_lora * num_q_heads * (nope_d + v_head_dim))
    )
    if (
        is_csa2
        and sparse is not None
        and not sparse.kv_source_layer_ids
        and geometry.sliding_window > 0
        and eval_context.context > 0
    ):
        att_ctx = min(eval_context.context, geometry.sliding_window)
    elif is_sparse and geometry.index_topk > 0 and eval_context.context > 0:
        att_ctx = min(eval_context.context, geometry.index_topk)
    else:
        att_ctx = eval_context.context
    raw_latent_w = kv_lora + rope_d
    align = eval_context.chip.physical_alignment_elems
    b_latent_phys = float(math.ceil(raw_latent_w / align) * align) * dtypes.kv
    b_latent_min = float(raw_latent_w) * dtypes.kv
    has_idx = (
        is_sparse
        and geometry.index_topk > 0
        and (
            geometry.index_head_dim > 0
            or (is_csa2 and sparse is not None and sparse.index_record_bytes > 0.0)
        )
    )
    b_idx_phys = (
        float(geometry.index_head_dim) * dtypes.kv
        + (SPARSE_MLA_SCALE_BYTES if dtypes.kv <= FP8_BYTES else 0.0)
        if has_idx
        else 0.0
    )
    b_idx_min = float(geometry.index_head_dim) * dtypes.kv if has_idx else 0.0
    if is_csa2 and sparse is not None and sparse.index_record_bytes > 0.0:
        b_idx_phys = sparse.index_record_bytes if has_idx else 0.0
        b_idx_min = b_idx_phys
    fam_kv_scale = (
        1.0
        if is_csa2
        else ((kv_n_fam / float(n_fam)) if geometry.compress_ratios else 1.0)
    )
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
    n_kv_eff = (
        float(len(sparse.kv_source_layer_ids)) * eval_context.pp_scale
        if (is_csa2 and sparse is not None)
        else n_eff
    )
    mla_kv_bytes = n_kv_eff * kv_tokens * b_latent_phys
    mla_kv_bytes_min = n_kv_eff * kv_tokens_min * b_latent_min
    has_csa_swa = (
        is_csa2
        and sparse is not None
        and sparse.swa_bytes_per_tok > 0.0
        and geometry.sliding_window > 0
    )
    if has_csa_swa and sparse is not None:
        swa_ctx = min(eval_context.context, geometry.sliding_window)
        swa_dec = float(eval_context.batch * swa_ctx) / kv_split_div
        swa_pre = (
            float(min(eval_context.cached_prefix, geometry.sliding_window)) / cp_div
            + float(eval_context.eff_prefill) / kv_split_div
            if eval_context.eff_prefill > 0
            else 0.0
        )
        swa_extra = n_eff * (swa_dec + swa_pre) * sparse.swa_bytes_per_tok
        mla_kv_bytes += swa_extra
        mla_kv_bytes_min += swa_extra
    if is_sparse and has_idx and n_idx_eff > 0.0:
        scan_cap = sparse.candidate_scan_cap if sparse is not None else 0
        scan_ctx = (
            min(eval_context.context, scan_cap)
            if scan_cap > 0 and eval_context.context > 0
            else eval_context.context
        )
        if eval_context.batch > 0 and eval_context.context > geometry.index_topk:
            idx_tokens = (
                n_idx_eff
                * float(eval_context.batch * scan_ctx)
                * fam_kv_scale
                / kv_split_div
            )
            c_shared_idx = min(scan_ctx, max(0, eval_context.shared_prefix_tokens))
            idx_tokens_min = (
                n_idx_eff
                * float(
                    c_shared_idx + eval_context.batch * max(0, scan_ctx - c_shared_idx)
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
    is_csa2 = geometry.sparse_attn is not None and geometry.sparse_attn.kind == "csa2"
    plain_mla = max(0, counts.mla - counts.dsa)
    plain_eff_mla = max(0.0, counts.eff_mla - counts.eff_dsa)
    for fam_name, n_fam, kv_n_fam, is_sparse in (
        ("mla", plain_mla, plain_eff_mla, is_csa2 and geometry.index_topk > 0),
        ("dsa", counts.dsa, counts.eff_dsa, True),
    ):
        if n_fam <= 0:
            continue
        _validate_mla_geom(fam_name, geometry)
        kv_lora = (
            geometry.kv_lora_rank
            if geometry.kv_lora_rank > 0
            else (geometry.head_dim if is_csa2 else 0)
        )
        nope_d = (
            geometry.qk_nope_head_dim
            if geometry.qk_nope_head_dim > 0
            else geometry.head_dim
        )
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
    sh_rows: int | float


def _routed_moe_min_elems(
    geometry: ModelGeometry,
    parallel: Parallel,
    n_m_eff: float,
    *,
    has_rows: bool,
) -> tuple[float, float, float]:
    """Compute (min_active_experts, routed_unit, routed_elems_min) for routed MoE."""
    tp = parallel.tp
    ep_ranks = (parallel.ep if parallel.ep > 1 else tp) if parallel.moe == "ep" else 1
    if parallel.moe == "ep":
        min_active_experts = (
            float(math.ceil(min(geometry.num_experts, geometry.top_k) / ep_ranks))
            if has_rows
            else 0.0
        )
        tp_sub = max(1, tp // max(1, ep_ranks)) if ep_ranks < tp else 1
        width_exact = float(geometry.moe_inter) / float(tp_sub)
    else:
        min_active_experts = (
            float(min(geometry.num_experts, geometry.top_k)) if has_rows else 0.0
        )
        width_exact = float(geometry.moe_inter) / float(max(1, tp))
    expert_hidden = (
        geometry.routed_expert_hidden
        if geometry.routed_expert_hidden > 0
        else geometry.hidden
    )
    routed_unit = (
        float(n_m_eff * geometry.ffn_num_matrices * expert_hidden) * width_exact
    )
    return min_active_experts, routed_unit, min_active_experts * routed_unit


def routed_moe_min_unpack_vector_flops(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    *,
    chip: ChipSpec | None = None,
    layer_scale: float = 1.0,
    active_rows: int = 1,
) -> float:
    """Return strict routed-MoE weight-unpack vector FLOPs from one shared formula."""
    if dtypes.w_expert >= FP8_BYTES or geometry.num_experts <= 0 or active_rows <= 0:
        return 0.0
    num_moe_layers = layer_ffn_kinds(geometry).count("moe")
    if num_moe_layers <= 0:
        return 0.0
    pp_s = (
        float(math.ceil(geometry.num_layers / max(1, parallel.pp)))
        / float(geometry.num_layers)
        if geometry.num_layers > 0 and parallel.pp > 1
        else 1.0
    )
    n_m_eff = float(num_moe_layers) * pp_s * layer_scale
    _, _, routed_elems_min = _routed_moe_min_elems(
        geometry, parallel, n_m_eff, has_rows=active_rows > 0
    )
    unpack_vector = _sub_byte_unpack_vector_flops(
        routed_elems_min,
        dtypes.w_expert,
        chip=chip,
        weight_dtype=dtypes.w_expert_dtype,
        act=dtypes.act,
    )
    return _strict_sub_byte_unpack_vector_flops(unpack_vector)


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
        tp_sub = max(1, tp // max(1, ep_ranks)) if ep_ranks < tp else 1
        width_exact = float(geometry.moe_inter) / float(tp_sub)
    else:
        active_experts = experts_touched(
            geometry.num_experts, geometry.top_k, eval_context.rows
        )
        width_exact = float(geometry.moe_inter) / float(max(1, tp))
    width = math.ceil(width_exact)
    expert_hidden = (
        geometry.routed_expert_hidden
        if geometry.routed_expert_hidden > 0
        else geometry.hidden
    )
    ffn_mats = geometry.ffn_num_matrices
    has_rows = (global_rows > 0) if parallel.moe == "ep" else (eval_context.rows > 0)
    min_active_experts, routed_unit, routed_elems_min = _routed_moe_min_elems(
        geometry, parallel, n_m_eff, has_rows=has_rows
    )
    routed_elems = active_experts * routed_unit
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
    sh_rows: int | float = (
        float(eval_context.rows) / float(tp)
        if (eval_context.unpadded_linear and parallel.residual == "sp")
        else (
            math.ceil(eval_context.rows / tp)
            if parallel.residual == "sp"
            else eval_context.rows
        )
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
    if any(spec.name == "ple" for spec in geometry.host_tables):
        return None
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


def _compute_host_table_costs(
    geometry: ModelGeometry,
    parallel: Parallel,
    dtypes: Dtypes,
    eval_context: _StepContext,
) -> list[ModuleCost]:
    """Compute dense projection, HBM lookup, and chip-scoped PCIe H2D costs."""
    if not geometry.host_tables:
        return []
    tp, pp = max(1, parallel.tp), max(1, parallel.pp)
    pcie_bw = eval_context.chip.pcie_bw
    has_pcie = pcie_bw > 0.0 and math.isfinite(pcie_bw)
    cores_per_chip = float(max(1, eval_context.chip.cores_per_chip))
    out: list[ModuleCost] = []
    for idx, spec in enumerate(geometry.host_tables):
        hbm_frac = host_table_hbm_fraction(spec, idx, eval_context.host_table_residency)
        offload_frac, num_table_layers = 1.0 - hbm_frac, len(spec.layer_ids)
        stage_scale = (
            float(math.ceil(num_table_layers / pp)) / float(num_table_layers)
            if (pp > 1 and num_table_layers > 0)
            else (eval_context.pp_scale if pp > 1 else 1.0)
        )
        proj_elems = float(spec.dense_proj_params) * stage_scale / float(tp)
        proj_bytes = role_dtype_bytes(dtypes, spec.dense_proj_dtype_role)
        proj_dtype = (
            dtypes.w_dense_dtype if spec.dense_proj_dtype_role == "w_dense" else ""
        )
        proj_flops = _route_matmul_flops(
            eval_context.rows,
            proj_elems,
            proj_bytes,
            dtypes.act,
            chip=eval_context.chip,
            weight_dtype=proj_dtype,
        )
        tp_div = float(tp) if spec.shard_axis == "tp" else 1.0
        is_first_resident = (
            eval_context.host_table_residency == "first"
            and spec.resident_on_first is None
            and idx == 0
        )
        lookup_scale = 1.0 if is_first_resident else stage_scale
        hbm_bytes = (
            proj_elems * proj_bytes
            + hbm_frac
            * float(eval_context.rows)
            * (spec.h2d_bytes_per_tok / tp_div)
            * lookup_scale
        )
        pcie_s = (
            offload_frac
            * float(eval_context.rows)
            * cores_per_chip
            * (spec.h2d_bytes_per_tok / tp_div)
            * stage_scale
            / pcie_bw
            if (offload_frac > 0.0 and spec.h2d_bytes_per_tok > 0.0 and has_pcie)
            else 0.0
        )
        out.append(
            ModuleCost(
                name=f"host_table_{spec.name}",
                hbm_bytes=hbm_bytes,
                hbm_bytes_min=hbm_bytes,
                fp8_flops=proj_flops.fp8,
                bf16_flops=proj_flops.bf16,
                fp4_flops=proj_flops.fp4,
                int8_flops=proj_flops.int8,
                int4_flops=proj_flops.int4,
                pcie_h2d_s=pcie_s,
            )
        )
    return out


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
    # unless DraftSpec specifies single-row LM-head verification per sequence.
    if (
        eval_context.decode_rows > 0
        and geometry.draft is not None
        and geometry.draft.single_row_lm_head_verify
    ):
        sampled_rows = eval_context.batch
    elif eval_context.decode_rows > 0:
        sampled_rows = eval_context.decode_rows
    else:
        sampled_rows = 0 if eval_context.rows == 0 else max(1, eval_context.batch)
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
    unpadded_linear: bool
    chip: ChipSpec | None


def _merge_opt_float(
    val_a: float | None,
    val_b: float | None,
    fallback_a: float,
    fallback_b: float,
) -> float | None:
    """Sum two optional lower-bound or envelope metrics when either is set."""
    if val_a is None and val_b is None:
        return None
    return fallback_a + fallback_b


def _merge_collective_counts(
    entries_a: tuple[tuple[Collective, int], ...] | None,
    entries_b: tuple[tuple[Collective, int], ...] | None,
    raw_a: tuple[Collective, ...],
    raw_b: tuple[Collective, ...],
) -> tuple[tuple[Collective, int], ...] | None:
    """Merge deduplicated (Collective, count) pairs in first-seen order."""
    if entries_a is None and entries_b is None:
        return None
    counts: dict[Collective, int] = {}
    seq_a = entries_a if entries_a is not None else tuple((c, 1) for c in raw_a)
    seq_b = entries_b if entries_b is not None else tuple((c, 1) for c in raw_b)
    for coll, cnt in (*seq_a, *seq_b):
        counts[coll] = counts.get(coll, 0) + cnt
    return tuple(counts.items())


def _merge_two_module_costs(first: ModuleCost, second: ModuleCost) -> ModuleCost:
    """Combine two same-named ModuleCost records across a split pass."""
    strict_a = (
        first.strict_collectives
        if first.strict_collectives is not None
        else first.unique_collectives
    )
    strict_b = (
        second.strict_collectives
        if second.strict_collectives is not None
        else second.unique_collectives
    )
    strict_colls = (
        None
        if (first.strict_collectives is None and second.strict_collectives is None)
        else _merge_collective_counts(
            strict_a, strict_b, first.collectives, second.collectives
        )
    )
    return ModuleCost(
        name=first.name,
        hbm_bytes=first.hbm_bytes + second.hbm_bytes,
        fp8_flops=first.fp8_flops + second.fp8_flops,
        bf16_flops=first.bf16_flops + second.bf16_flops,
        collectives=first.collectives + second.collectives,
        hbm_bytes_min=_merge_opt_float(
            first.hbm_bytes_min,
            second.hbm_bytes_min,
            first.strict_hbm_bytes,
            second.strict_hbm_bytes,
        ),
        unique_collectives=_merge_collective_counts(
            first.unique_collectives,
            second.unique_collectives,
            first.collectives,
            second.collectives,
        ),
        fp8_flops_min=_merge_opt_float(
            first.fp8_flops_min,
            second.fp8_flops_min,
            first.strict_fp8_flops,
            second.strict_fp8_flops,
        ),
        bf16_flops_min=_merge_opt_float(
            first.bf16_flops_min,
            second.bf16_flops_min,
            first.strict_bf16_flops,
            second.strict_bf16_flops,
        ),
        vector_flops=first.vector_flops + second.vector_flops,
        strict_vector_flops=first.strict_vector_flops + second.strict_vector_flops,
        hbm_bytes_envelope=_merge_opt_float(
            first.hbm_bytes_envelope,
            second.hbm_bytes_envelope,
            first.envelope_hbm_bytes,
            second.envelope_hbm_bytes,
        ),
        fp8_flops_envelope=_merge_opt_float(
            first.fp8_flops_envelope,
            second.fp8_flops_envelope,
            first.envelope_fp8_flops,
            second.envelope_fp8_flops,
        ),
        bf16_flops_envelope=_merge_opt_float(
            first.bf16_flops_envelope,
            second.bf16_flops_envelope,
            first.envelope_bf16_flops,
            second.envelope_bf16_flops,
        ),
        fp4_flops=first.fp4_flops + second.fp4_flops,
        int8_flops=first.int8_flops + second.int8_flops,
        int4_flops=first.int4_flops + second.int4_flops,
        fp4_flops_min=_merge_opt_float(
            first.fp4_flops_min,
            second.fp4_flops_min,
            first.strict_fp4_flops,
            second.strict_fp4_flops,
        ),
        int8_flops_min=_merge_opt_float(
            first.int8_flops_min,
            second.int8_flops_min,
            first.strict_int8_flops,
            second.strict_int8_flops,
        ),
        int4_flops_min=_merge_opt_float(
            first.int4_flops_min,
            second.int4_flops_min,
            first.strict_int4_flops,
            second.strict_int4_flops,
        ),
        fp4_flops_envelope=_merge_opt_float(
            first.fp4_flops_envelope,
            second.fp4_flops_envelope,
            first.envelope_fp4_flops,
            second.envelope_fp4_flops,
        ),
        int8_flops_envelope=_merge_opt_float(
            first.int8_flops_envelope,
            second.int8_flops_envelope,
            first.envelope_int8_flops,
            second.envelope_int8_flops,
        ),
        int4_flops_envelope=_merge_opt_float(
            first.int4_flops_envelope,
            second.int4_flops_envelope,
            first.envelope_int4_flops,
            second.envelope_int4_flops,
        ),
        strict_collectives=strict_colls,
        pcie_h2d_s=first.pcie_h2d_s + second.pcie_h2d_s,
    )


def _merge_module_costs(modules: Sequence[ModuleCost]) -> tuple[ModuleCost, ...]:
    """Merge same-named ModuleCost records in order of first appearance."""
    merged: dict[str, ModuleCost] = {}
    for mod in modules:
        existing = merged.get(mod.name)
        merged[mod.name] = (
            mod if existing is None else _merge_two_module_costs(existing, mod)
        )
    return tuple(merged.values())


def _ced_replay_shape(geometry: ModelGeometry, shape: StepShape) -> StepShape:
    """Cap prefill tokens to the CED sliding-window replay length."""
    uncached = max(
        0, round(shape.prefill_tokens * max(0.0, 1.0 - shape.prefix_hit_rate))
    )
    return replace(
        shape,
        prefill_tokens=min(uncached, geometry.sliding_window),
        prefix_hit_rate=0.0,
        execution="full",
    )


def pass_costs(
    geometry: ModelGeometry,
    parallel: Parallel,
    shape: StepShape,
    dtypes: Dtypes,
    **options: Unpack[PassCostOptions],
) -> tuple[ModuleCost, ...]:
    """Compute per-module resource work for one decode, prefill, or mixed pass."""
    sparse = geometry.sparse_attn
    if (
        shape.execution == "ced"
        and sparse is not None
        and shape.prefill_tokens > 0
        and geometry.sliding_window > 0
    ):
        replay = _ced_replay_shape(geometry, shape)
        if sparse.ced_post_only:
            return pass_costs(geometry, parallel, replay, dtypes, **options)
        if 0 < sparse.ced_boundary < geometry.num_layers:
            pre_g = slice_geometry_layers(geometry, 0, sparse.ced_boundary)
            post_g = slice_geometry_layers(
                geometry, sparse.ced_boundary, geometry.num_layers
            )
            pre_m = tuple(
                m
                for m in pass_costs(
                    pre_g, parallel, replace(shape, execution="full"), dtypes, **options
                )
                if m.name != "lm_head"
            )
            post_m = tuple(
                m
                for m in pass_costs(post_g, parallel, replay, dtypes, **options)
                if m.name != "embed"
            )
            return _merge_module_costs((*pre_m, *post_m))
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
    unpadded_linear = options.get("unpadded_linear", False)
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
    raw_uncached = (
        shape.prefill_tokens
        * max(0.0, 1.0 - shape.prefix_hit_rate)
        / max(1, eff_parallel.pcp)
    )
    eff_prefill = (
        (
            math.floor(raw_uncached)
            if unpadded_linear
            else max(1, math.ceil(raw_uncached))
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
        unpadded_linear=unpadded_linear,
        chip=eff_chip,
        execution=shape.execution,
        host_table_residency=shape.host_table_residency,
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
    out.extend(_compute_host_table_costs(geometry, eff_parallel, dtypes, eval_context))

    out.append(_compute_lm_head_cost(geometry, eff_parallel, dtypes, eval_context))

    return tuple(out)
