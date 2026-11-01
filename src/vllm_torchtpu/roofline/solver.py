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
"""Unconstrained single-pool mixed and xP/yD disaggregated TPU mesh solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape
from .step import StepFloor, spec_cycle_floor, step_floor

# vLLM v0.32.0 CacheConfig.gpu_memory_utilization default (https://github.com/vllm-project/vllm):
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92
EXAMPLE_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class TrafficScenario:
    """User-selected workload distribution and optional p90 SLO bounds."""

    name: str
    isl: int
    osl: int
    prefix_hit_rate: float
    accept_len: float
    think_time_s: float
    concurrencies: tuple[int, ...]
    kv_working_set_turns: float
    prefill_chunk: int = 2048
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    spec_tokens: int | None = None
    kv_offload_gb: float = 0.0
    page_size: int | None = None
    max_num_seqs: int | None = None

    def __post_init__(self) -> None:
        if self.isl <= 0 or self.osl <= 0:
            raise ValueError(f"isl and osl must be > 0, got {self.isl}, {self.osl}")
        if not 0.0 <= self.prefix_hit_rate <= 1.0:
            raise ValueError(
                f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            )
        if self.accept_len <= 0.0:
            raise ValueError(f"accept_len must be > 0, got {self.accept_len}")
        if self.think_time_s < 0.0:
            raise ValueError(f"think_time_s must be >= 0, got {self.think_time_s}")
        if not self.concurrencies or any(c <= 0 for c in self.concurrencies):
            raise ValueError(
                f"concurrencies must be positive ints, got {self.concurrencies}"
            )
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {self.page_size}")
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be > 0, got {self.max_num_seqs}")


EXAMPLE_SCENARIOS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="example:chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="example:reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="example:coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.92,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "custom": TrafficScenario(
        name="example:custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
}

EXAMPLE_SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}


def build_example_scenario(
    preset: str = "chat-8k1k",
    *,
    isl: int | None = None,
    osl: int | None = None,
    concurrencies: tuple[int, ...] | None = None,
    prefix_hit_rate: float | None = None,
    accept_len: float | None = None,
    think_time_s: float | None = None,
    prefill_chunk: int | None = None,
    slo_tier: str = "none",
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    kv_working_set_turns: float | None = None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from an explicitly labeled example scenario."""
    if preset not in EXAMPLE_SCENARIOS:
        raise ValueError(
            f"Unknown example scenario preset {preset!r}; "
            f"known: {sorted(EXAMPLE_SCENARIOS)}"
        )
    if slo_tier not in EXAMPLE_SLO_TIERS:
        raise ValueError(
            f"Unknown SLO tier {slo_tier!r}; known: {sorted(EXAMPLE_SLO_TIERS)}"
        )
    base = EXAMPLE_SCENARIOS[preset]
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=isl if isl is not None else base.isl,
        osl=osl if osl is not None else base.osl,
        concurrencies=(
            tuple(int(c) for c in concurrencies)
            if concurrencies is not None
            else base.concurrencies
        ),
        prefix_hit_rate=(
            prefix_hit_rate if prefix_hit_rate is not None else base.prefix_hit_rate
        ),
        accept_len=accept_len if accept_len is not None else base.accept_len,
        think_time_s=think_time_s if think_time_s is not None else base.think_time_s,
        prefill_chunk=prefill_chunk
        if prefill_chunk is not None
        else base.prefill_chunk,
        ttft_slo_ms=ttft_slo_ms if ttft_slo_ms is not None else tier_ttft,
        tpot_slo_ms=tpot_slo_ms if tpot_slo_ms is not None else tier_tpot,
        kv_working_set_turns=(
            kv_working_set_turns
            if kv_working_set_turns is not None
            else base.kv_working_set_turns
        ),
    )


@dataclass(frozen=True)
class HbmBreakdown:
    """Per-TensorCore HBM bytes split into weights, KV cache, and recurrent state."""

    weights: float
    kv: float
    state: float
    total: float


@dataclass(frozen=True)
class FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""

    hbm_ms: float
    matmul_ms: float
    ici_ms: float
    chain_ms: float
    hop_latency_ms: float
    hard_floor_ms: float
    binding: str


@dataclass(frozen=True)
class SearchStats:
    """Search funnel: (layout, concurrency) tuples inside the chip budget."""

    enumerated: int
    hbm_fit: int
    slo_met: int
    frontier: int


@dataclass(frozen=True)
class PoolMeshPlan:
    """Topology, parallelism, and HBM footprint for one serving pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: str
    chips_per_instance: int
    tp: int
    dp: int
    ep: int
    pp: int
    pcp: int
    kv_split: int
    mesh_axes: tuple[int, int, int, int]
    hbm_used_gb: float
    hbm_limit_gb: float
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0
    max_num_seqs: int | None = None
    max_model_len: int | None = None


@dataclass(frozen=True)
class ParetoPoint:
    """One feasible operating point on the p90 interactivity vs throughput frontier."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    p90_ttft_ms: float
    p90_tpot_ms: float
    p90_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool
    decode_floor: FloorTimes | None = None
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float = 0.0


@dataclass(frozen=True)
class ServingPlan:
    """Ranked serving plan with recommended point, Pareto frontier, and warnings."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ParetoPoint, ...] = ()
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    search: SearchStats | None = None


def hbm_breakdown_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
    *,
    physical_kv_layout: bool = True,
) -> HbmBreakdown:
    """Static model weights plus active KV/recurrent state bytes on one TensorCore."""
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    ep = par.ep if (par.moe == "ep" and par.ep > 1) else (tp if par.moe == "ep" else 1)

    n_gqa = 0
    n_swa = 0
    n_mla = 0
    n_dsa = 0
    n_gdn = 0
    n_kda = 0
    for tok in geom.layer_types:
        if tok in ("mla", "mla_attention"):
            n_mla += 1
        elif tok in ("dsa", "dsa_attention", "sparse_mla"):
            n_mla += 1
            n_dsa += 1
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif geom.kv_lora_rank > 0:
            n_mla += 1
            if geom.index_topk > 0:
                n_dsa += 1
        else:
            n_gqa += 1

    q_mult = 2 if geom.attn_output_gate else 1
    kv_div = min(tp, max(1, geom.num_kv_heads))
    w_attn = 0.0
    if n_gqa + n_swa > 0:
        w_q_o = (
            float(n_gqa + n_swa)
            * float(
                geom.hidden * geom.num_heads * geom.head_dim * q_mult
                + geom.num_heads * geom.head_dim * geom.hidden
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv = (
            float(n_gqa + n_swa)
            * float(2 * geom.hidden * geom.num_kv_heads * geom.head_dim)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o + w_kv
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            raise ValueError("MLA layer requires config field 'kv_lora_rank'")
        if geom.qk_nope_head_dim <= 0:
            raise ValueError("MLA layer requires config field 'qk_nope_head_dim'")
        if geom.qk_rope_head_dim <= 0:
            raise ValueError("MLA layer requires config field 'qk_rope_head_dim'")
        kv_lora = geom.kv_lora_rank
        nope_d = geom.qk_nope_head_dim
        rope_d = geom.qk_rope_head_dim
        v_d = geom.v_head_dim or nope_d
        if geom.q_lora_rank > 0:
            q_term_hidden = float(geom.hidden * geom.q_lora_rank) / pp
            q_term_proj = float(
                geom.q_lora_rank * geom.num_heads * (nope_d + rope_d)
            ) / (tp * pp)
        else:
            q_term_hidden = 0.0
            q_term_proj = float(geom.hidden * geom.num_heads * (nope_d + rope_d)) / (
                tp * pp
            )
        w_mla = (
            float(n_mla)
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
        w_attn += w_mla
        if n_dsa > 0 and geom.index_n_heads > 0 and geom.index_head_dim > 0:
            w_attn += (
                float(n_dsa)
                * float(
                    geom.hidden * max(1, geom.index_n_heads // tp) * geom.index_head_dim
                )
                * dt.w_dense
                / pp
            )

    w_rec = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        z_r = vh * geom.linear_v_dim
        w_rec += (
            float(n_gdn)
            * (
                float(geom.hidden * (qkv_r + z_r) + z_r * geom.hidden) * dt.w_dense
                + float(geom.hidden * 2 * vh + qkv_r * geom.conv_kernel) * dt.w_aux
            )
            / pp
        )
    if n_kda > 0:
        kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        w_rec += (
            float(n_kda)
            * float(geom.hidden * (2 * kh * kd + 2 * vh * vd) + vh * vd * geom.hidden)
            * dt.w_dense
            / pp
        )

    w_ffn = 0.0
    w_dense_ffn = 0.0
    if geom.num_experts > 0:
        n_dense_ffn = min(geom.num_layers, geom.first_k_dense_replace)
        n_moe_ffn = geom.num_layers - n_dense_ffn
        if n_dense_ffn > 0 and geom.dense_inter > 0:
            w_dense_ffn = (
                float(n_dense_ffn * 3 * geom.hidden * geom.dense_inter)
                * dt.w_dense
                / (tp * pp)
            )
            w_ffn += w_dense_ffn
        routed_div = ep * pp if par.moe == "ep" else tp * pp
        w_ffn += (
            float(n_moe_ffn * geom.num_experts * 3 * geom.hidden * geom.moe_inter)
            * dt.w_expert
            / routed_div
        )
        if geom.shared_inter > 0:
            isr_eff = (
                geom.shared_inter
                if par.residual == "sp"
                else max(1, geom.shared_inter // tp)
            )
            w_ffn += float(n_moe_ffn * 3 * geom.hidden * isr_eff) * dt.w_dense / pp
        w_ffn += float(n_moe_ffn * geom.hidden * geom.num_experts) * dt.router / pp
    else:
        w_ffn += (
            float(geom.num_layers * 3 * geom.hidden * geom.dense_inter)
            * dt.w_dense
            / (tp * pp)
        )

    w_head = float(2 * geom.vocab * geom.hidden) * dt.lm_head / (tp * pp)
    w_mtp = 0.0
    if geom.mtp_layers > 0:
        mtp_fc = float(2 * geom.hidden * geom.hidden) * dt.w_aux / tp
        mtp_att = (w_attn / float(max(1, n_gqa + n_swa + n_mla))) * float(pp)
        if geom.num_experts > 0:
            n_moe_layers = max(
                1, geom.num_layers - min(geom.num_layers, geom.first_k_dense_replace)
            )
            mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_layers)) * float(pp)
        else:
            mtp_ffn = (w_ffn / float(max(1, geom.num_layers))) * float(pp)
        w_mtp = float(geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn) / float(pp)

    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            raise ValueError("MLA layer requires config field 'kv_lora_rank'")
        if geom.qk_rope_head_dim <= 0:
            raise ValueError("MLA layer requires config field 'qk_rope_head_dim'")
    raw_latent_w = geom.kv_lora_rank + geom.qk_rope_head_dim
    if physical_kv_layout:
        kv_packing = max(1, int(round(4.0 / max(0.5, dt.kv))))
        padded_hd = math.ceil(geom.head_dim / 128) * 128 if geom.head_dim > 0 else 0
        kx2 = kvh if geom.head_dim == 64 else 2 * kvh
        padded_kx2 = math.ceil(kx2 / kv_packing) * kv_packing if kx2 > 0 else 0
        gqa_kv_b_per_tok = float(padded_kx2 * padded_hd) * dt.kv
        mla_latent_b = float(math.ceil(raw_latent_w / 128) * 128) * dt.kv
        # 4-byte FP32 scale per token for quantized sparse-MLA indexer KV cache
        # (`[head_dim fp8 | 4-byte scale | ...]`).
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv + (4.0 if dt.kv <= 1.0 else 0.0)
            if (geom.index_topk > 0 and geom.index_head_dim > 0)
            else 0.0
        )
    else:
        gqa_kv_b_per_tok = float(2 * kvh * geom.head_dim) * dt.kv
        mla_latent_b = float(raw_latent_w) * dt.kv
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv
            if (geom.index_topk > 0 and geom.index_head_dim > 0)
            else 0.0
        )
    kv_gqa = (
        float(seqs_per_dp * n_gqa * context)
        * gqa_kv_b_per_tok
        / (par.kv_split * par.pcp * pp)
    )
    sw_ctx = min(context, geom.sliding_window) if geom.sliding_window > 0 else context
    kv_swa = (
        float(seqs_per_dp * n_swa * sw_ctx)
        * gqa_kv_b_per_tok
        / (par.kv_split * par.pcp * pp)
    )
    kv_mla = (
        float(seqs_per_dp * n_mla * context)
        * (mla_latent_b + dsa_idx_b)
        / (par.kv_split * par.pcp * pp)
    )

    st_bytes = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        st_bytes += (
            float(seqs_per_dp * n_gdn)
            * (
                float(vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
                + float(qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
            )
            / pp
        )
    if n_kda > 0:
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        st_bytes += float(seqs_per_dp * n_kda * vh * kd * vd) * dt.state / pp

    weights = w_attn + w_rec + w_ffn + w_head + w_mtp
    kv = kv_gqa + kv_swa + kv_mla
    state = st_bytes
    # Same summation order as before so HBM-fit decisions do not move.
    total = (
        w_attn + w_rec + w_ffn + w_head + w_mtp + kv_gqa + kv_swa + kv_mla + st_bytes
    )
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=state,
        total=total,
    )


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
    *,
    physical_kv_layout: bool = True,
) -> float:
    """Static model weights plus active KV/recurrent state bytes on one TensorCore."""
    return hbm_breakdown_per_core_bytes(
        geom, dt, par, seqs_per_dp, context, physical_kv_layout=physical_kv_layout
    ).total


def _floor_times(sf: StepFloor) -> FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""
    return FloorTimes(
        hbm_ms=sf.hbm_s * 1e3,
        matmul_ms=sf.mm_s * 1e3,
        ici_ms=sf.ici_s * 1e3,
        chain_ms=sf.chain_s * 1e3,
        hop_latency_ms=sf.lat_s * 1e3,
        hard_floor_ms=sf.hard_floor_s * 1e3,
        binding=sf.binding,
    )


def _default_rpa_page_size(ctx_tokens: int) -> int:
    if ctx_tokens > 8192:
        return 16
    pow2 = 1 << (max(1, ctx_tokens) - 1).bit_length()
    return min(256, max(16, pow2 // 16))


def _smem_max_seqs(ctx_tokens: int, kv_split: int, page_size: int) -> int:
    pages_per_req = max(
        1, math.ceil((ctx_tokens / max(1, kv_split)) / max(1, page_size))
    )
    return 524_288 // pages_per_req // 4


def _valid_head_tp(geom: ModelGeometry, tp: int) -> bool:
    if geom.num_heads > 0 and (tp > geom.num_heads or geom.num_heads % tp != 0):
        return False
    if geom.num_kv_heads <= 0:
        return True
    return geom.num_kv_heads % tp == 0 or tp % geom.num_kv_heads == 0


def _has_gdn_layers(geom: ModelGeometry) -> bool:
    return any(
        tok in ("linear_attention", "gdn", "gated_deltanet", "mamba")
        for tok in geom.layer_types
    )


def _valid_gdn_degree(geom: ModelGeometry, deg: int) -> bool:
    if deg <= 1 or not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    if geom.linear_v_heads % deg != 0:
        return False
    uq = min(geom.linear_k_heads, deg)
    if geom.linear_k_heads % uq != 0 or deg % uq != 0:
        return False
    return (geom.linear_v_heads // deg) % (geom.linear_k_heads // uq) == 0


def _valid_raiden_gdn_pair(
    geom: ModelGeometry, par_p: Parallel, par_d: Parallel
) -> bool:
    if not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    deg_p = par_p.tp * par_p.pcp
    deg_d = par_d.tp * par_d.pcp
    if not (_valid_gdn_degree(geom, deg_p) and _valid_gdn_degree(geom, deg_d)):
        return False
    if par_p.pp > 1 and deg_p > 1 and deg_d > deg_p and deg_d % deg_p == 0:
        return True
    kq = geom.linear_k_heads
    if max(1, deg_p // kq) != max(1, deg_d // kq):
        return False
    return deg_p == 1 or deg_p % deg_d == 0


def _count_full_attn_layers(geom: ModelGeometry) -> int:
    non_full = {
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
    }
    return sum(1 for tok in geom.layer_types if tok not in non_full)


def _kv_capacity_and_hit_rate(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    b_res: int,
    b_ws: int,
    ctx_full: int,
    hbm_limit_b: float,
    w_base_b: float,
    b_kv_phys_core: float,
    offload_b_per_core: float,
    scenario: TrafficScenario,
) -> tuple[float, float]:
    st_b = hbm_footprint_per_core_bytes(geom, dtypes, par, b_res, 0) - w_base_b
    cap_tok = (
        (max(0.0, hbm_limit_b - w_base_b - st_b) + offload_b_per_core) / b_kv_phys_core
        if b_kv_phys_core > 0.0
        else math.inf
    )
    w_tok = max(0.0, scenario.kv_working_set_turns) * float(b_ws * ctx_full)
    h_eff = (
        scenario.prefix_hit_rate * min(1.0, cap_tok / max(1.0, w_tok))
        if scenario.prefix_hit_rate > 0.0 and w_tok > 0.0 and b_kv_phys_core > 0.0
        else scenario.prefix_hit_rate
    )
    return cap_tok, h_eff


def _divisors(n: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _candidate_decode_parallels(
    geom: ModelGeometry, topo: SliceTopology
) -> list[Parallel]:
    cores = topo.num_cores
    out: list[Parallel] = []
    for tp in _divisors(cores):
        if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp):
            continue
        dp = cores // tp
        kv_group = (
            max(1, tp // geom.num_kv_heads)
            if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
            else tp
        )
        kv_splits = _divisors(kv_group) if kv_group > 1 else (1,)
        moe_mode: Literal["tp_slice", "ep"] = (
            "ep"
            if geom.num_experts > 0 and cores > 1 and geom.num_experts >= cores
            else "tp_slice"
        )
        if geom.num_experts > 0 and moe_mode == "tp_slice" and dp > 1:
            continue
        ep_val = cores if moe_mode == "ep" else 1
        mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
        modes: list[
            tuple[
                Literal["tp_slice", "ep"],
                int,
                Literal["allreduce", "ag_ep", "a2a_push"],
            ]
        ]
        if moe_mode == "ep" and dp == 1:
            modes = [
                ("ep", ep_val, "allreduce"),
                ("ep", ep_val, "ag_ep"),
                ("ep", ep_val, "a2a_push"),
            ]
            # Kernel constraint:
            # src/vllm_torchtpu/kernels/experimental/moe_dense_expert.py:71
            # requires i % 128 == 0 (intermediate dimension aligned to 128).
            if geom.moe_inter // tp >= 128 and (geom.moe_inter // tp) % 128 == 0:
                modes.append(("tp_slice", 1, "allreduce"))
        elif moe_mode == "ep":
            modes = [
                ("ep", ep_val, "ag_ep"),
                ("ep", ep_val, "a2a_push"),
            ]
        else:
            modes = [("tp_slice", 1, "allreduce")]
        for kvs in kv_splits:
            for m_mode, m_ep, m_disp in modes:
                out.append(
                    Parallel(
                        tp=tp,
                        kv_split=kvs,
                        moe=m_mode,
                        moe_dispatch=m_disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=dp,
                        ep=m_ep,
                        pp=1,
                        pcp=1,
                        mesh_axes=mesh_ax,
                    )
                )
    return out


def _candidate_prefill_parallels(
    geom: ModelGeometry, topo: SliceTopology, *, for_disagg: bool = False
) -> list[Parallel]:
    cores = topo.num_cores
    n_full_attn = _count_full_attn_layers(geom)
    out: list[Parallel] = []
    for pp in _divisors(cores):
        if geom.num_layers < pp:
            continue
        if for_disagg and 0 < n_full_attn < geom.num_layers and pp > n_full_attn:
            continue
        rem = cores // pp
        for pcp in _divisors(rem):
            tp = rem // pcp
            if tp < 1 or (pp > 1 and tp > 1):
                continue
            if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp * pcp):
                continue
            if pcp > 1 and geom.num_experts > 0:
                ep_span = tp * pcp
                if geom.num_experts < ep_span or geom.num_experts % ep_span != 0:
                    continue
                moe_mode: Literal["tp_slice", "ep"] = "ep"
                ep_val = ep_span
            else:
                moe_mode = (
                    "ep"
                    if geom.num_experts > 0 and tp > 1 and geom.num_experts >= tp
                    else "tp_slice"
                )
                ep_val = tp if moe_mode == "ep" else 1
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
            dispatches: tuple[Literal["allreduce", "ag_ep"], ...] = (
                ("ag_ep",)
                if (moe_mode == "ep" and tp == 1)
                else (("allreduce", "ag_ep") if moe_mode == "ep" else ("allreduce",))
            )
            for disp in dispatches:
                out.append(
                    Parallel(
                        tp=tp,
                        kv_split=1,
                        moe=moe_mode,
                        moe_dispatch=disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=1,
                        ep=ep_val,
                        pp=pp,
                        pcp=pcp,
                        mesh_axes=mesh_ax,
                    )
                )
    return out


def _pareto_filter(points: list[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    if not points:
        return ()
    ordered = sorted(
        points,
        key=lambda p: (
            -p.p90_interactivity_tok_s,
            -p.token_throughput_per_chip,
            p.total_chips,
        ),
    )
    kept: list[ParetoPoint] = []
    best_tput = -1.0
    for pt in ordered:
        if pt.token_throughput_per_chip > best_tput * (1.0 + 1e-9):
            kept.append(pt)
            best_tput = pt.token_throughput_per_chip
    return tuple(kept)


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    mode: Literal["auto", "mixed", "disagg"] = "auto",
    max_chips: int | None = None,
    *,
    max_instances: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> ServingPlan:
    """Search single-pool mixed and unconstrained xP/yD disagg plans."""
    if not (0.0 < gpu_memory_utilization <= 1.0):
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )
    hbm_limit_b = float(gpu_memory_utilization) * float(chip.hbm_bytes)
    hbm_limit_gb = hbm_limit_b / 1e9
    eff_al = max(1.0, scenario.accept_len)
    spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    ctx_full = scenario.isl + scenario.osl
    ctx_mid = scenario.isl + scenario.osl // 2
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    page_size = (
        scenario.page_size
        if scenario.page_size is not None
        else _default_rpa_page_size(ctx_full)
    )
    c_max = max(scenario.concurrencies)

    warnings: list[str] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf

    topos = [
        t for t in TOPOLOGIES.values() if max_chips is None or t.num_chips <= max_chips
    ]

    enumerated = 0
    hbm_fit = 0

    # Precompute feasible prefill instances per topology and GDN compatibility class.
    prefill_options: list[
        tuple[SliceTopology, Parallel, float, float, float, HbmBreakdown]
    ] = []
    for topo in topos:
        best_by_class: dict[
            tuple[int, bool],
            tuple[SliceTopology, Parallel, float, float, float, HbmBreakdown],
        ] = {}
        for par in _candidate_prefill_parallels(geom, topo, for_disagg=True):
            hbm_bd = hbm_breakdown_per_core_bytes(geom, dtypes, par, 1, scenario.isl)
            hbm_b = hbm_bd.total
            if hbm_b > hbm_limit_b:
                continue
            sf_p = step_floor(
                geom,
                chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=scenario.isl,
                    prefix_hit_rate=scenario.prefix_hit_rate,
                ),
                dtypes,
                topology=topo,
            )
            ttft_ms = (
                sf_p.serial_floor_s
                * (1.0 + float(par.pp - 1) / float(max(1, n_chunks)))
                * 1e3
            )
            min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
            rate_p = 1.0 / max(1e-9, sf_p.serial_floor_s)
            cls_key = (par.tp * par.pcp, par.pp > 1)
            prev = best_by_class.get(cls_key)
            if prev is None or rate_p > prev[3]:
                best_by_class[cls_key] = (
                    topo,
                    par,
                    ttft_ms,
                    rate_p,
                    hbm_b / 1e9,
                    hbm_bd,
                )
        prefill_options.extend(best_by_class.values())

    candidates: list[ParetoPoint] = []

    # 1. Single-pool mixed candidates
    for topo in topos:
        for par in _candidate_decode_parallels(geom, topo):
            w_base_b = hbm_footprint_per_core_bytes(geom, dtypes, par, 0, 0)
            hbm_c1 = hbm_footprint_per_core_bytes(geom, dtypes, par, 1, ctx_full)
            per_seq_b = hbm_c1 - w_base_b
            b_hbm_max = (
                int((hbm_limit_b - w_base_b) // per_seq_b)
                if (hbm_c1 <= hbm_limit_b and per_seq_b > 0.0)
                else (c_max if hbm_c1 <= hbm_limit_b else 0)
            )
            b_smem_max = _smem_max_seqs(ctx_full, par.kv_split, page_size)
            if hbm_c1 <= hbm_limit_b and b_smem_max >= 1:
                sf_c1 = (
                    spec_cycle_floor(
                        geom, chip, par, 1, ctx_mid, spec_k, dtypes, topology=topo
                    )
                    if spec_k > 0
                    else step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(batch=1, context=ctx_mid),
                        dtypes,
                        topology=topo,
                    )
                )
                min_c1_tpot_ms = min(
                    min_c1_tpot_ms, sf_c1.serial_floor_s * 1e3 / eff_al
                )

            if mode not in ("auto", "mixed"):
                continue

            sf_ttft = step_floor(
                geom,
                chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=scenario.isl,
                    prefix_hit_rate=scenario.prefix_hit_rate,
                ),
                dtypes,
                topology=topo,
            )
            min_req_ttft_ms = min(min_req_ttft_ms, sf_ttft.serial_floor_s * 1e3)

            b_kv_phys_core = hbm_footprint_per_core_bytes(
                geom, dtypes, par, 1, 1
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par, 1, 0)
            offload_b_per_core = (
                float(scenario.kv_offload_gb) * 1e9 / float(max(1, topo.num_cores))
            )
            dec_steps_per_req = max(1.0, scenario.osl / eff_al)
            for c in scenario.concurrencies:
                enumerated += 1
                if b_hbm_max < 1 or b_smem_max < 1:
                    continue
                b_local = max(1, math.ceil(c / max(1, par.dp)))
                if scenario.think_time_s > 0.0:
                    for _ in range(2):
                        sf_dec_tmp = (
                            spec_cycle_floor(
                                geom,
                                chip,
                                par,
                                b_local,
                                ctx_mid,
                                spec_k,
                                dtypes,
                                topology=topo,
                            )
                            if spec_k > 0
                            else step_floor(
                                geom,
                                chip,
                                par,
                                StepShape(batch=b_local, context=ctx_mid),
                                dtypes,
                                topology=topo,
                            )
                        )
                        w_srv = (
                            sf_ttft.serial_floor_s
                            + dec_steps_per_req * sf_dec_tmp.serial_floor_s
                        )
                        c_act = float(c) * w_srv / (w_srv + scenario.think_time_s)
                        b_local = max(1, math.ceil(c_act / max(1, par.dp)))
                b_res = min(b_local, b_hbm_max, b_smem_max)
                if scenario.max_num_seqs is not None:
                    b_res = min(b_res, scenario.max_num_seqs)
                hbm_bd = hbm_breakdown_per_core_bytes(
                    geom, dtypes, par, b_res, ctx_full
                )
                hbm_b = hbm_bd.total
                while b_res > 1 and hbm_b > hbm_limit_b:
                    b_res -= 1
                    hbm_bd = hbm_breakdown_per_core_bytes(
                        geom, dtypes, par, b_res, ctx_full
                    )
                    hbm_b = hbm_bd.total
                if hbm_b > hbm_limit_b:
                    continue
                hbm_fit += 1
                cap_tok, h_eff = _kv_capacity_and_hit_rate(
                    geom,
                    dtypes,
                    par,
                    b_res,
                    b_local,
                    ctx_full,
                    hbm_limit_b,
                    w_base_b,
                    b_kv_phys_core,
                    offload_b_per_core,
                    scenario,
                )
                cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
                sf_ttft_c = (
                    step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(
                            batch=0,
                            rows_per_seq=0,
                            context=0,
                            state_writes=0,
                            prefill_tokens=scenario.isl,
                            prefix_hit_rate=h_eff,
                        ),
                        dtypes,
                        topology=topo,
                    )
                    if h_eff != scenario.prefix_hit_rate
                    else sf_ttft
                )
                eff_isl_c = max(1, math.ceil((1.0 - h_eff) * scenario.isl))
                n_chunks_c = max(
                    1, math.ceil(eff_isl_c / max(1, scenario.prefill_chunk))
                )
                chunk_tok_c = min(eff_isl_c, scenario.prefill_chunk)
                sf_dec = (
                    spec_cycle_floor(
                        geom, chip, par, b_res, ctx_mid, spec_k, dtypes, topology=topo
                    )
                    if spec_k > 0
                    else step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(batch=b_res, context=ctx_mid),
                        dtypes,
                        topology=topo,
                    )
                )
                sf_ver_only = (
                    step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(
                            batch=b_res,
                            rows_per_seq=spec_k + 1,
                            context=ctx_mid,
                            state_writes=1,
                        ),
                        dtypes,
                        topology=topo,
                    )
                    if spec_k > 0
                    else sf_dec
                )
                draft_serial_s = max(
                    0.0, sf_dec.serial_floor_s - sf_ver_only.serial_floor_s
                )
                sf_mix = step_floor(
                    geom,
                    chip,
                    par,
                    StepShape(
                        batch=b_res,
                        rows_per_seq=spec_k + 1,
                        context=ctx_mid,
                        state_writes=1,
                        prefill_tokens=chunk_tok_c,
                        prefix_hit_rate=0.0,
                    ),
                    dtypes,
                    topology=topo,
                )
                mix_serial_s = sf_mix.serial_floor_s + draft_serial_s
                total_req_work_s = (
                    dec_steps_per_req * sf_dec.serial_floor_s
                    + float(b_res * n_chunks_c) * mix_serial_s
                )
                res_cap = float(b_res * max(1, par.dp))
                c_served = min(float(c), res_cap)
                server_cap_rate = res_cap / max(1e-9, total_req_work_s)
                w_srv_final = (
                    sf_ttft_c.serial_floor_s + dec_steps_per_req * sf_dec.serial_floor_s
                )
                rate_req_s = (
                    min(
                        server_cap_rate,
                        float(c) / max(1e-9, w_srv_final + scenario.think_time_s),
                    )
                    if scenario.think_time_s > 0.0
                    else c_served / max(1e-9, total_req_work_s)
                )
                p_mix = min(
                    1.0,
                    float(b_res * n_chunks_c)
                    / float(max(1.0, dec_steps_per_req + float(b_res * n_chunks_c))),
                )
                p90_step_s = (
                    1.0 - p_mix
                ) * sf_dec.serial_floor_s + p_mix * mix_serial_s
                p90_tpot_ms = p90_step_s * 1e3 / eff_al
                c_demand = min(float(c), float(b_local * max(1, par.dp)))
                queue_wait_s = (
                    max(0.0, (c_demand / max(1e-9, res_cap)) - 1.0) * w_srv_final
                )
                p90_ttft_ms = (sf_ttft_c.serial_floor_s + queue_wait_s) * 1e3 + (
                    sf_mix.serial_floor_s * 1e3 if c > 1 else 0.0
                )
                interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
                out_tput = rate_req_s * float(scenario.osl) / float(topo.num_chips)
                tot_tput = (
                    rate_req_s
                    * float(scenario.isl + scenario.osl)
                    / float(topo.num_chips)
                )
                slo_ok = (
                    scenario.ttft_slo_ms is None or p90_ttft_ms <= scenario.ttft_slo_ms
                ) and (
                    scenario.tpot_slo_ms is None or p90_tpot_ms <= scenario.tpot_slo_ms
                )
                max_seqs_flag = b_res if b_res < b_local else scenario.max_num_seqs
                pool = PoolMeshPlan(
                    role="mixed",
                    instances=1,
                    topology=topo.name,
                    chips_per_instance=topo.num_chips,
                    tp=par.tp,
                    dp=par.dp,
                    ep=par.ep,
                    pp=par.pp,
                    pcp=par.pcp,
                    kv_split=par.kv_split,
                    mesh_axes=(
                        par.mesh_axes
                        if par.mesh_axes is not None
                        else submesh_for_group(par.tp, topo, topo.cores_per_chip)
                    ),
                    hbm_used_gb=hbm_b / 1e9,
                    hbm_limit_gb=hbm_limit_gb,
                    moe_dispatch=par.moe_dispatch,
                    hbm_weights_gb=hbm_bd.weights / 1e9,
                    hbm_kv_gb=hbm_bd.kv / 1e9,
                    hbm_state_gb=hbm_bd.state / 1e9,
                    max_num_seqs=max_seqs_flag,
                    max_model_len=ctx_full,
                )
                candidates.append(
                    ParetoPoint(
                        mode="mixed",
                        concurrency=c,
                        total_chips=topo.num_chips,
                        prefill_pool=None,
                        decode_pool=pool,
                        p90_ttft_ms=p90_ttft_ms,
                        p90_tpot_ms=p90_tpot_ms,
                        p90_interactivity_tok_s=interactivity,
                        output_throughput_per_chip=out_tput,
                        token_throughput_per_chip=tot_tput,
                        p_mixed=p_mix,
                        binding=f"dec:{sf_dec.binding},mix:{sf_mix.binding}"
                        if p_mix > 0.0
                        else sf_dec.binding,
                        slo_met=slo_ok,
                        decode_floor=_floor_times(sf_dec),
                        kv_cache_capacity_tokens=cap_tok_int,
                        effective_prefix_hit_rate=h_eff,
                    )
                )

    # 2. Unconstrained balanced and imbalanced xP/yD disaggregated candidates
    if mode in ("auto", "disagg") and prefill_options:
        prefill_eff_cache: dict[tuple[str, Parallel, float], tuple[float, float]] = {}
        for topo_d in topos:
            offload_b_per_core_d = (
                float(scenario.kv_offload_gb) * 1e9 / float(max(1, topo_d.num_cores))
            )
            for par_d in _candidate_decode_parallels(geom, topo_d):
                w_base_d = hbm_footprint_per_core_bytes(geom, dtypes, par_d, 0, 0)
                hbm_d1 = hbm_footprint_per_core_bytes(geom, dtypes, par_d, 1, ctx_full)
                per_seq_d = hbm_d1 - w_base_d
                b_hbm_max_d = (
                    int((hbm_limit_b - w_base_d) // per_seq_d)
                    if (hbm_d1 <= hbm_limit_b and per_seq_d > 0.0)
                    else (c_max if hbm_d1 <= hbm_limit_b else 0)
                )
                b_smem_max_d = _smem_max_seqs(ctx_full, par_d.kv_split, page_size)
                b_kv_phys_d = hbm_footprint_per_core_bytes(
                    geom, dtypes, par_d, 1, 1
                ) - hbm_footprint_per_core_bytes(geom, dtypes, par_d, 1, 0)
                for c in scenario.concurrencies:
                    dec_steps_per_req = max(1.0, scenario.osl / eff_al)
                    b_one = math.ceil(c / max(1, par_d.dp))
                    sf_d_one = (
                        spec_cycle_floor(
                            geom,
                            chip,
                            par_d,
                            b_one,
                            ctx_mid,
                            spec_k,
                            dtypes,
                            topology=topo_d,
                        )
                        if spec_k > 0
                        else step_floor(
                            geom,
                            chip,
                            par_d,
                            StepShape(batch=b_one, context=ctx_mid),
                            dtypes,
                            topology=topo_d,
                        )
                    )
                    rate_d_one = float(c) / (
                        dec_steps_per_req * max(1e-9, sf_d_one.serial_floor_s)
                    )

                    for (
                        topo_p,
                        par_p,
                        ttft_ms,
                        rate_p_one,
                        hbm_p_gb,
                        hbm_p_bd,
                    ) in prefill_options:
                        if not _valid_raiden_gdn_pair(geom, par_p, par_d):
                            continue
                        eff_max_inst = (
                            max_instances if max_instances is not None else 16
                        )
                        if max_chips is None:
                            ratio_xp = math.ceil(rate_d_one / max(1e-9, rate_p_one)) + 1
                            ratio_yd = math.ceil(rate_p_one / max(1e-9, rate_d_one)) + 1
                            max_xp = min(eff_max_inst, max(1, ratio_xp))
                            max_yd = min(eff_max_inst, max(1, ratio_yd))
                        else:
                            max_xp = min(
                                eff_max_inst, max(1, max_chips // topo_p.num_chips)
                            )
                            max_yd = min(
                                eff_max_inst, max(1, max_chips // topo_d.num_chips)
                            )
                        for x_p in range(1, max_xp + 1):
                            for y_d in range(1, max_yd + 1):
                                total_chips = (
                                    x_p * topo_p.num_chips + y_d * topo_d.num_chips
                                )
                                if max_chips is not None and total_chips > max_chips:
                                    continue
                                enumerated += 1
                                if b_hbm_max_d < 1 or b_smem_max_d < 1:
                                    continue
                                b_local = max(1, math.ceil(c / max(1, y_d * par_d.dp)))
                                if scenario.think_time_s > 0.0:
                                    for _ in range(2):
                                        sf_d_tmp = (
                                            spec_cycle_floor(
                                                geom,
                                                chip,
                                                par_d,
                                                b_local,
                                                ctx_mid,
                                                spec_k,
                                                dtypes,
                                                topology=topo_d,
                                            )
                                            if spec_k > 0
                                            else step_floor(
                                                geom,
                                                chip,
                                                par_d,
                                                StepShape(
                                                    batch=b_local, context=ctx_mid
                                                ),
                                                dtypes,
                                                topology=topo_d,
                                            )
                                        )
                                        w_srv = (
                                            ttft_ms * 1e-3
                                            + dec_steps_per_req
                                            * sf_d_tmp.serial_floor_s
                                        )
                                        c_act = (
                                            float(c)
                                            * w_srv
                                            / (w_srv + scenario.think_time_s)
                                        )
                                        b_local = max(
                                            1, math.ceil(c_act / max(1, y_d * par_d.dp))
                                        )
                                b_res = min(b_local, b_hbm_max_d, b_smem_max_d)
                                if scenario.max_num_seqs is not None:
                                    b_res = min(b_res, scenario.max_num_seqs)
                                hbm_bd_d = hbm_breakdown_per_core_bytes(
                                    geom, dtypes, par_d, b_res, ctx_full
                                )
                                hbm_d = hbm_bd_d.total
                                while b_res > 1 and hbm_d > hbm_limit_b:
                                    b_res -= 1
                                    hbm_bd_d = hbm_breakdown_per_core_bytes(
                                        geom, dtypes, par_d, b_res, ctx_full
                                    )
                                    hbm_d = hbm_bd_d.total
                                if hbm_d > hbm_limit_b:
                                    continue
                                hbm_fit += 1
                                cap_tok, h_eff = _kv_capacity_and_hit_rate(
                                    geom,
                                    dtypes,
                                    par_d,
                                    b_res,
                                    b_local,
                                    ctx_full,
                                    hbm_limit_b,
                                    w_base_d,
                                    b_kv_phys_d,
                                    offload_b_per_core_d,
                                    scenario,
                                )
                                cap_tok_int = (
                                    None if math.isinf(cap_tok) else int(cap_tok)
                                )
                                if h_eff < scenario.prefix_hit_rate:
                                    p_key = (topo_p.name, par_p, round(h_eff, 6))
                                    cached_p = prefill_eff_cache.get(p_key)
                                    if cached_p is None:
                                        eff_isl_p = max(
                                            1, math.ceil((1.0 - h_eff) * scenario.isl)
                                        )
                                        n_chunks_p = max(
                                            1,
                                            math.ceil(
                                                eff_isl_p
                                                / max(1, scenario.prefill_chunk)
                                            ),
                                        )
                                        sf_p_eff = step_floor(
                                            geom,
                                            chip,
                                            par_p,
                                            StepShape(
                                                batch=0,
                                                rows_per_seq=0,
                                                context=0,
                                                state_writes=0,
                                                prefill_tokens=scenario.isl,
                                                prefix_hit_rate=h_eff,
                                            ),
                                            dtypes,
                                            topology=topo_p,
                                        )
                                        ttft_p_eff = (
                                            sf_p_eff.serial_floor_s
                                            * (
                                                1.0
                                                + float(par_p.pp - 1)
                                                / float(max(1, n_chunks_p))
                                            )
                                            * 1e3
                                        )
                                        rate_p_eff = 1.0 / max(
                                            1e-9, sf_p_eff.serial_floor_s
                                        )
                                        cached_p = (ttft_p_eff, rate_p_eff)
                                        prefill_eff_cache[p_key] = cached_p
                                    ttft_eff_ms, rate_p_eff = cached_p
                                else:
                                    ttft_eff_ms = ttft_ms
                                    rate_p_eff = rate_p_one
                                sf_d = (
                                    sf_d_one
                                    if b_res == b_one
                                    else (
                                        spec_cycle_floor(
                                            geom,
                                            chip,
                                            par_d,
                                            b_res,
                                            ctx_mid,
                                            spec_k,
                                            dtypes,
                                            topology=topo_d,
                                        )
                                        if spec_k > 0
                                        else step_floor(
                                            geom,
                                            chip,
                                            par_d,
                                            StepShape(batch=b_res, context=ctx_mid),
                                            dtypes,
                                            topology=topo_d,
                                        )
                                    )
                                )
                                p90_tpot_ms = sf_d.serial_floor_s * 1e3 / eff_al
                                interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
                                res_cap_d = float(b_res * max(1, y_d * par_d.dp))
                                c_served_d = min(float(c), res_cap_d)
                                rate_d_cap = c_served_d / (
                                    dec_steps_per_req * max(1e-9, sf_d.serial_floor_s)
                                )
                                c_demand_d = min(
                                    float(c), float(b_local * max(1, y_d * par_d.dp))
                                )
                                queue_wait_d_ms = (
                                    max(0.0, (c_demand_d / max(1e-9, res_cap_d)) - 1.0)
                                    * (dec_steps_per_req * sf_d.serial_floor_s)
                                    * 1e3
                                )
                                p90_ttft_d_ms = ttft_eff_ms + queue_wait_d_ms
                                if scenario.think_time_s > 0.0:
                                    client_rate = float(c) / max(
                                        1e-9,
                                        ttft_eff_ms * 1e-3
                                        + dec_steps_per_req * sf_d.serial_floor_s
                                        + scenario.think_time_s,
                                    )
                                    coupled_rate = min(
                                        x_p * rate_p_eff, rate_d_cap, client_rate
                                    )
                                else:
                                    coupled_rate = min(x_p * rate_p_eff, rate_d_cap)
                                out_tput = (
                                    coupled_rate
                                    * float(scenario.osl)
                                    / float(total_chips)
                                )
                                tot_tput = (
                                    coupled_rate
                                    * float(scenario.isl + scenario.osl)
                                    / float(total_chips)
                                )
                                slo_ok = (
                                    scenario.ttft_slo_ms is None
                                    or p90_ttft_d_ms <= scenario.ttft_slo_ms
                                ) and (
                                    scenario.tpot_slo_ms is None
                                    or p90_tpot_ms <= scenario.tpot_slo_ms
                                )
                                max_seqs_d = (
                                    b_res if b_res < b_local else scenario.max_num_seqs
                                )
                                p_pool = PoolMeshPlan(
                                    role="prefill",
                                    instances=x_p,
                                    topology=topo_p.name,
                                    chips_per_instance=topo_p.num_chips,
                                    tp=par_p.tp,
                                    dp=par_p.dp,
                                    ep=par_p.ep,
                                    pp=par_p.pp,
                                    pcp=par_p.pcp,
                                    kv_split=par_p.kv_split,
                                    mesh_axes=(
                                        par_p.mesh_axes
                                        if par_p.mesh_axes is not None
                                        else submesh_for_group(
                                            par_p.tp, topo_p, topo_p.cores_per_chip
                                        )
                                    ),
                                    hbm_used_gb=hbm_p_gb,
                                    hbm_limit_gb=hbm_limit_gb,
                                    moe_dispatch=par_p.moe_dispatch,
                                    hbm_weights_gb=hbm_p_bd.weights / 1e9,
                                    hbm_kv_gb=hbm_p_bd.kv / 1e9,
                                    hbm_state_gb=hbm_p_bd.state / 1e9,
                                    max_model_len=ctx_full,
                                )
                                d_pool = PoolMeshPlan(
                                    role="decode",
                                    instances=y_d,
                                    topology=topo_d.name,
                                    chips_per_instance=topo_d.num_chips,
                                    tp=par_d.tp,
                                    dp=par_d.dp,
                                    ep=par_d.ep,
                                    pp=par_d.pp,
                                    pcp=par_d.pcp,
                                    kv_split=par_d.kv_split,
                                    mesh_axes=(
                                        par_d.mesh_axes
                                        if par_d.mesh_axes is not None
                                        else submesh_for_group(
                                            par_d.tp, topo_d, topo_d.cores_per_chip
                                        )
                                    ),
                                    hbm_used_gb=hbm_d / 1e9,
                                    hbm_limit_gb=hbm_limit_gb,
                                    moe_dispatch=par_d.moe_dispatch,
                                    hbm_weights_gb=hbm_bd_d.weights / 1e9,
                                    hbm_kv_gb=hbm_bd_d.kv / 1e9,
                                    hbm_state_gb=hbm_bd_d.state / 1e9,
                                    max_num_seqs=max_seqs_d,
                                    max_model_len=ctx_full,
                                )
                                candidates.append(
                                    ParetoPoint(
                                        mode="disagg",
                                        concurrency=c,
                                        total_chips=total_chips,
                                        prefill_pool=p_pool,
                                        decode_pool=d_pool,
                                        p90_ttft_ms=p90_ttft_d_ms,
                                        p90_tpot_ms=p90_tpot_ms,
                                        p90_interactivity_tok_s=interactivity,
                                        output_throughput_per_chip=out_tput,
                                        token_throughput_per_chip=tot_tput,
                                        p_mixed=0.0,
                                        binding=sf_d.binding,
                                        slo_met=slo_ok,
                                        decode_floor=_floor_times(sf_d),
                                        kv_cache_capacity_tokens=cap_tok_int,
                                        effective_prefix_hit_rate=h_eff,
                                    )
                                )

    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < "
            f"single-request floor {min_c1_tpot_ms:.2f} ms"
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < "
            f"single-request prefill floor {min_req_ttft_ms:.2f} ms"
        )

    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if candidates and not slo_candidates and not warnings:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")

    # Keep full frontier plus any distinct pool geometry points if needed.
    frontier = _pareto_filter(pool_for_frontier)
    recommended = (
        max(frontier, key=lambda p: p.token_throughput_per_chip) if frontier else None
    )
    return ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
        candidates=tuple(pool_for_frontier),
        gpu_memory_utilization=gpu_memory_utilization,
        search=SearchStats(
            enumerated=enumerated,
            hbm_fit=hbm_fit,
            slo_met=len(slo_candidates),
            frontier=len(frontier),
        ),
    )
