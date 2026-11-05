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
from functools import lru_cache
from statistics import NormalDist
from typing import Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape, _classify_layers
from .step import StepFloor, _combine_modules, spec_cycle_floor, step_floor

_STD_NORM = NormalDist()

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
    pp_layer_partition: tuple[int, ...] | None = None
    kv_offload_gb: float = 0.0


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
    sweep_candidates: tuple[ParetoPoint, ...] = ()
    winner_prefix_hit_range: tuple[float, float] | None = None
    prefix_hit_crossover_rate: float | None = None
    spec_tokens: int = 0


@lru_cache(maxsize=32768)
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
    kv_n_gqa = 0.0
    kv_n_swa = 0.0
    kv_n_mla = 0.0
    for idx, tok in enumerate(geom.layer_types):
        r_i = geom.compress_ratios[idx] if idx < len(geom.compress_ratios) else 1.0
        kv_ret = (1.0 / r_i) if r_i > 1.0 else r_i
        if tok in ("mla", "mla_attention"):
            n_mla += 1
            kv_n_mla += kv_ret
        elif tok in ("dsa", "dsa_attention", "sparse_mla"):
            n_mla += 1
            n_dsa += 1
            kv_n_mla += kv_ret
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
            kv_n_swa += kv_ret
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif tok == "ffn_only":
            continue
        elif geom.kv_lora_rank > 0:
            n_mla += 1
            kv_n_mla += kv_ret
            if geom.index_topk > 0:
                n_dsa += 1
        else:
            n_gqa += 1
            kv_n_gqa += kv_ret

    q_mult = 2 if geom.attn_output_gate else 1
    kv_div = min(tp, max(1, geom.num_kv_heads))
    w_attn = 0.0
    if n_gqa > 0:
        gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
        w_q_o = (
            float(n_gqa)
            * float(
                geom.hidden * geom.num_heads * gqa_hd * q_mult
                + geom.num_heads * gqa_hd * geom.hidden
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv = (
            float(n_gqa)
            * float(2 * geom.hidden * geom.num_kv_heads * gqa_hd)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o + w_kv
    if n_swa > 0:
        w_q_o_swa = (
            float(n_swa)
            * float(
                geom.hidden * geom.num_heads * geom.head_dim * q_mult
                + geom.num_heads * geom.head_dim * geom.hidden
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv_swa = (
            float(n_swa)
            * float(2 * geom.hidden * geom.num_kv_heads * geom.head_dim)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o_swa + w_kv_swa
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            raise ValueError("MLA layer requires config field 'kv_lora_rank'")
        if geom.qk_nope_head_dim <= 0:
            raise ValueError("MLA layer requires config field 'qk_nope_head_dim'")
        if geom.qk_rope_head_dim < 0:
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
        if geom.n_moe_layers is not None:
            n_moe_ffn = min(geom.num_layers, max(0, geom.n_moe_layers))
            n_dense_ffn = (
                geom.n_dense_layers
                if geom.n_dense_layers is not None
                else (geom.num_layers - n_moe_ffn)
            )
        else:
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
        if geom.mtp_inter > 0:
            mtp_ffn = float(3 * geom.hidden * geom.mtp_inter) * dt.w_dense / tp
        elif geom.num_experts > 0:
            n_moe_layers = (
                max(1, min(geom.num_layers, geom.n_moe_layers))
                if geom.n_moe_layers is not None
                else max(
                    1,
                    geom.num_layers - min(geom.num_layers, geom.first_k_dense_replace),
                )
            )
            mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_layers)) * float(pp)
        else:
            mtp_ffn = (w_ffn / float(max(1, geom.num_layers))) * float(pp)
        w_mtp = float(geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn) / float(pp)

    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            raise ValueError("MLA layer requires config field 'kv_lora_rank'")
        if geom.qk_rope_head_dim < 0:
            raise ValueError("MLA layer requires config field 'qk_rope_head_dim'")
    raw_latent_w = geom.kv_lora_rank + geom.qk_rope_head_dim
    gqa_hd_kv = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    if physical_kv_layout:
        kv_packing = max(1, int(round(4.0 / max(0.5, dt.kv))))
        padded_hd = math.ceil(geom.head_dim / 128) * 128 if geom.head_dim > 0 else 0
        padded_gqa_hd = math.ceil(gqa_hd_kv / 128) * 128 if gqa_hd_kv > 0 else 0
        kx2 = kvh if geom.head_dim == 64 else 2 * kvh
        gqa_kx2 = kvh if gqa_hd_kv == 64 else 2 * kvh
        padded_kx2 = math.ceil(kx2 / kv_packing) * kv_packing if kx2 > 0 else 0
        padded_gqa_kx2 = (
            math.ceil(gqa_kx2 / kv_packing) * kv_packing if gqa_kx2 > 0 else 0
        )
        gqa_kv_b_per_tok = float(padded_kx2 * padded_hd) * dt.kv
        gqa_full_kv_b_per_tok = float(padded_gqa_kx2 * padded_gqa_hd) * dt.kv
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
        gqa_full_kv_b_per_tok = float(2 * kvh * gqa_hd_kv) * dt.kv
        mla_latent_b = float(raw_latent_w) * dt.kv
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv
            if (geom.index_topk > 0 and geom.index_head_dim > 0)
            else 0.0
        )
    eff_n_gqa = kv_n_gqa if geom.compress_ratios else float(n_gqa)
    eff_n_swa = kv_n_swa if geom.compress_ratios else float(n_swa)
    eff_n_mla = kv_n_mla if geom.compress_ratios else float(n_mla)
    kv_gqa = (
        float(seqs_per_dp * context)
        * eff_n_gqa
        * gqa_full_kv_b_per_tok
        / (par.kv_split * par.pcp * pp)
    )
    sw_ctx = min(context, geom.sliding_window) if geom.sliding_window > 0 else context
    kv_swa = (
        float(seqs_per_dp * sw_ctx)
        * eff_n_swa
        * gqa_kv_b_per_tok
        / (par.kv_split * par.pcp * pp)
    )
    kv_mla = (
        float(seqs_per_dp * context)
        * eff_n_mla
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


_NON_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    {
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
)


def _count_full_attn_layers(geom: ModelGeometry) -> int:
    return sum(1 for tok in geom.layer_types if tok not in _NON_FULL_ATTN_TOKENS)


@lru_cache(maxsize=256)
def _ep_z_unit(ep: int) -> float:
    return (
        1.0 / math.sqrt(math.pi)
        if ep == 2
        else _STD_NORM.inv_cdf(1.0 - 1.0 / float(ep))
    )


def max_ep_load_ratio(total_assignments: int, ep: int) -> float:
    """Expected busiest-rank token-load multiplier under uniform EP routing."""
    if ep <= 1 or total_assignments <= 0:
        return 1.0
    z = _ep_z_unit(ep)
    ratio = 1.0 + z * math.sqrt(float(ep - 1) / float(max(1, total_assignments)))
    return min(float(ep), max(1.0, ratio))


def _scale_ep_moe_floor(
    sf: StepFloor,
    chip: ChipSpec,
    geom: ModelGeometry,
    par: Parallel,
    global_tokens: int,
    topology: SliceTopology | None,
) -> StepFloor:
    if par.ep <= 1 or geom.num_experts <= 0 or global_tokens <= 0:
        return sf
    ratio = max_ep_load_ratio(global_tokens * geom.top_k, par.ep)
    if ratio <= 1.0:
        return sf
    scaled = tuple(
        replace(m, fp8_flops=m.fp8_flops * ratio, bf16_flops=m.bf16_flops * ratio)
        if m.name in ("moe", "draft_moe")
        else m
        for m in sf.modules
    )
    mm_s = (
        sum(m.fp8_flops for m in scaled) / chip.fp8_flops
        + sum(m.bf16_flops for m in scaled) / chip.bf16_flops
    )
    use_torus = topology is not None or par.mesh_axes is not None
    resource_bound_s = max(sf.hbm_s, mm_s, sf.ici_s)
    hard_floor_s = max(resource_bound_s, sf.lat_s, sf.chain_s)
    serial_floor_s = (
        sum(
            max(
                m.hbm_bytes / chip.hbm_bw,
                m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops,
            )
            for m in scaled
        )
        + sf.chain_s
    )
    if hard_floor_s == sf.hbm_s:
        binding = "hbm"
    elif hard_floor_s == mm_s:
        binding = "matmul"
    elif hard_floor_s > sf.ici_s or use_torus or sf.lat_s > sf.ici_s:
        binding = "chain"
    else:
        binding = "ici"
    return StepFloor(
        modules=scaled,
        hbm_s=sf.hbm_s,
        mm_s=mm_s,
        ici_s=sf.ici_s,
        chain_s=sf.chain_s,
        hard_floor_s=hard_floor_s,
        serial_floor_s=serial_floor_s,
        binding=binding,
        lat_s=sf.lat_s,
    )


def _scaled_ep_serial_s(
    sf: StepFloor,
    chip: ChipSpec,
    geom: ModelGeometry,
    par: Parallel,
    global_tokens: int,
) -> float:
    if par.ep <= 1 or geom.num_experts <= 0 or global_tokens <= 0:
        return sf.serial_floor_s
    ratio = max_ep_load_ratio(global_tokens * geom.top_k, par.ep)
    if ratio <= 1.0:
        return sf.serial_floor_s
    mod_s = 0.0
    for m in sf.modules:
        mul = ratio if m.name in ("moe", "draft_moe") else 1.0
        hb_s = m.hbm_bytes / chip.hbm_bw
        mm_s = (m.fp8_flops * mul) / chip.fp8_flops + (
            m.bf16_flops * mul
        ) / chip.bf16_flops
        mod_s += hb_s if hb_s >= mm_s else mm_s
    return mod_s + sf.chain_s


def _scale_pp_stage_floor(
    sf: StepFloor,
    pp_part: tuple[int, ...] | None,
    num_layers: int,
    pp: int,
    chip: ChipSpec,
    topology: SliceTopology | None,
    par: Parallel,
) -> StepFloor:
    if pp <= 1 or not pp_part or num_layers <= 0:
        return sf
    even_max = max(1, math.ceil(num_layers / pp))
    actual_max = max(pp_part)
    if actual_max == even_max:
        return sf
    ratio = float(actual_max) / float(even_max)
    scaled = []
    for m in sf.modules:
        if m.name in ("embed", "lm_head", "p2p", "pp_p2p"):
            scaled.append(m)
            continue
        colls = []
        for c in m.collectives:
            if c.name.startswith("pp"):
                colls.append(c)
        non_pp = [c for c in m.collectives if not c.name.startswith("pp")]
        if non_pp:
            target_n = max(1, round(len(non_pp) * ratio))
            for idx in range(target_n):
                colls.append(non_pp[idx % len(non_pp)])
        scaled.append(
            replace(
                m,
                hbm_bytes=m.hbm_bytes * ratio,
                fp8_flops=m.fp8_flops * ratio,
                bf16_flops=m.bf16_flops * ratio,
                collectives=tuple(colls),
                unique_collectives=None,
            )
        )
    return _combine_modules(tuple(scaled), chip, topology=topology, par=par)


def _peak_prefill_logit_bytes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    chunk_rows: int,
    *,
    sampled_rows: int = 1,
) -> float:
    if (chunk_rows <= 0 and sampled_rows <= 0) or geom.vocab <= 0:
        return 0.0
    tp = max(1, par.tp)
    rows_local = math.ceil(max(0, chunk_rows) / max(1, par.pcp))
    n_samp = max(1, sampled_rows) if chunk_rows > 0 else max(0, sampled_rows)
    act_b = 2.0 if dtypes.act == "a16" else 1.0
    logit_b = max(act_b, dtypes.lm_head)
    return (
        float(n_samp) * (float(geom.vocab) / float(tp)) * logit_b
        + float(rows_local) * float(geom.hidden) * act_b
    )


def _disagg_kv_xfer_s(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    topo_p: SliceTopology,
    topo_d: SliceTopology,
    uncached_tokens: int,
) -> float:
    if chip.dcn_bw <= 0.0 or not math.isfinite(chip.dcn_bw):
        return 0.0
    bd_one = hbm_breakdown_per_core_bytes(
        geom,
        dtypes,
        Parallel(tp=1, pp=1),
        1,
        max(0, uncached_tokens),
        physical_kv_layout=False,
    )
    wire_b = bd_one.kv + bd_one.state
    denom = float(max(1, min(topo_p.num_chips, topo_d.num_chips))) * chip.dcn_bw
    return wire_b / denom


def _pp_stage_breakdowns(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    *,
    context: int,
    seqs_per_dp: int,
    disagg: bool = False,
    max_stage_layers: int | None = None,
) -> tuple[tuple[int, ...], HbmBreakdown, float, bool]:
    pp = max(1, par.pp)
    n_l = geom.num_layers
    if pp <= 1 or n_l <= 0:
        bd = hbm_breakdown_per_core_bytes(geom, dtypes, par, seqs_per_dp, context)
        return ((n_l,), bd, bd.total, True)
    par1 = replace(par, pp=1)
    tp = max(1, par.tp)
    w_embed = float(geom.vocab * geom.hidden) * dtypes.lm_head / float(tp)
    if geom.mtp_layers > 0:
        bd_with_mtp = hbm_breakdown_per_core_bytes(geom, dtypes, par1, 0, 0)
        bd_no_mtp = hbm_breakdown_per_core_bytes(
            replace(geom, mtp_layers=0), dtypes, par1, 0, 0
        )
        w_mtp_total = max(0.0, bd_with_mtp.weights - bd_no_mtp.weights)
    else:
        w_mtp_total = 0.0
    w_lm = w_embed + w_mtp_total
    w_head_both = 2.0 * w_embed
    layer_w: list[float] = []
    layer_kv: list[float] = []
    layer_st: list[float] = []
    is_full: list[int] = []
    for idx in range(n_l):
        cr_i = (geom.compress_ratios[idx],) if idx < len(geom.compress_ratios) else ()
        fkd = 1 if idx < geom.first_k_dense_replace else 0
        g_i = replace(
            geom,
            num_layers=1,
            layer_types=(geom.layer_types[idx],),
            compress_ratios=cr_i,
            first_k_dense_replace=fkd,
            mtp_layers=0,
        )
        bd_i = hbm_breakdown_per_core_bytes(g_i, dtypes, par1, seqs_per_dp, context)
        layer_w.append(max(0.0, bd_i.weights - w_head_both))
        layer_kv.append(bd_i.kv)
        layer_st.append(bd_i.state)
        is_full.append(0 if geom.layer_types[idx] in _NON_FULL_ATTN_TOKENS else 1)
    pref_w = [0.0] * (n_l + 1)
    pref_kv = [0.0] * (n_l + 1)
    pref_st = [0.0] * (n_l + 1)
    pref_full = [0] * (n_l + 1)
    for i in range(n_l):
        pref_w[i + 1] = pref_w[i] + layer_w[i]
        pref_kv[i + 1] = pref_kv[i] + layer_kv[i]
        pref_st[i + 1] = pref_st[i] + layer_st[i]
        pref_full[i + 1] = pref_full[i] + is_full[i]
    req_full = disagg and (0 < pref_full[n_l] < n_l) and pp <= pref_full[n_l]
    inf = math.inf
    dp_cost = [[inf] * (n_l + 1) for _ in range(pp + 1)]
    dp_prev = [[-1] * (n_l + 1) for _ in range(pp + 1)]
    dp_cost[0][0] = 0.0
    for s in range(1, pp + 1):
        extra = (w_embed if s == 1 else 0.0) + (w_lm if s == pp else 0.0)
        rem_stages = pp - s
        for b in range(s, n_l - rem_stages + 1):
            best_c = inf
            best_a = -1
            a_min = (
                max(s - 1, b - max_stage_layers)
                if max_stage_layers is not None
                else (s - 1)
            )
            for a in range(a_min, b):
                if dp_cost[s - 1][a] == inf:
                    continue
                if req_full and (pref_full[b] - pref_full[a]) < 1:
                    continue
                seg = (
                    (pref_w[b] - pref_w[a])
                    + (pref_kv[b] - pref_kv[a])
                    + (pref_st[b] - pref_st[a])
                    + extra
                )
                cand = max(dp_cost[s - 1][a], seg)
                if cand < best_c:
                    best_c = cand
                    best_a = a
            dp_cost[s][b] = best_c
            dp_prev[s][b] = best_a
    feasible = dp_prev[pp][n_l] >= 0
    if not feasible:
        base = n_l // pp
        rem = n_l % pp
        part = tuple(base + (1 if s < rem else 0) for s in range(pp))
    else:
        rev: list[int] = []
        cur = n_l
        for s in range(pp, 0, -1):
            prv = dp_prev[s][cur]
            rev.append(cur - prv)
            cur = prv
        part = tuple(reversed(rev))
    if feasible and n_l % pp == 0:
        unif = (n_l // pp,) * pp
        u_ok = True
        u_max = 0.0
        pos = 0
        for s, cnt in enumerate(unif):
            nxt = pos + cnt
            if req_full and (pref_full[nxt] - pref_full[pos]) < 1:
                u_ok = False
                break
            ex = (w_embed if s == 0 else 0.0) + (w_lm if s == pp - 1 else 0.0)
            u_max = max(
                u_max,
                (pref_w[nxt] - pref_w[pos])
                + (pref_kv[nxt] - pref_kv[pos])
                + (pref_st[nxt] - pref_st[pos])
                + ex,
            )
            pos = nxt
        if u_ok and u_max <= dp_cost[pp][n_l] * (1.0 + 1e-6):
            part = unif
    best_bd = HbmBreakdown(0.0, 0.0, 0.0, -1.0)
    last_tot = 0.0
    pos = 0
    for s, cnt in enumerate(part):
        nxt = pos + cnt
        ex_w = (w_embed if s == 0 else 0.0) + (w_lm if s == pp - 1 else 0.0)
        sw = (pref_w[nxt] - pref_w[pos]) + ex_w
        skv = pref_kv[nxt] - pref_kv[pos]
        sst = pref_st[nxt] - pref_st[pos]
        stot = sw + skv + sst
        if s == pp - 1:
            last_tot = stot
        if stot > best_bd.total:
            best_bd = HbmBreakdown(weights=sw, kv=skv, state=sst, total=stot)
        pos = nxt
    return (part, best_bd, last_tot, feasible)


def partition_pp_layers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    *,
    context: int,
    seqs_per_dp: int,
    disagg: bool = False,
) -> tuple[int, ...]:
    """Dynamic-programming contiguous PP layer partition minimizing max-stage HBM."""
    part, _, _, _ = _pp_stage_breakdowns(
        geom,
        dtypes,
        par,
        context=context,
        seqs_per_dp=seqs_per_dp,
        disagg=disagg,
    )
    return part


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
        tuple[
            SliceTopology,
            Parallel,
            float,
            float,
            float,
            HbmBreakdown,
            tuple[int, ...] | None,
        ]
    ] = []
    pre_chunk_rows = min(eff_isl, scenario.prefill_chunk)
    for topo in topos:
        best_by_class: dict[
            tuple[int, bool],
            tuple[
                SliceTopology,
                Parallel,
                float,
                float,
                float,
                HbmBreakdown,
                tuple[int, ...] | None,
            ],
        ] = {}
        for par in _candidate_prefill_parallels(geom, topo, for_disagg=True):
            logit_peak_b = _peak_prefill_logit_bytes(
                geom, dtypes, par, pre_chunk_rows, sampled_rows=1
            )
            if par.pp > 1:
                even_cap = max(1, math.ceil(geom.num_layers / par.pp))
                pp_part, hbm_bd, last_stage_b, pp_feas = _pp_stage_breakdowns(
                    geom,
                    dtypes,
                    par,
                    context=scenario.isl,
                    seqs_per_dp=1,
                    disagg=True,
                    max_stage_layers=even_cap,
                )
                if (
                    not pp_feas
                    or hbm_bd.total > hbm_limit_b
                    or (last_stage_b + logit_peak_b) > hbm_limit_b
                ):
                    pp_part, hbm_bd, last_stage_b, pp_feas = _pp_stage_breakdowns(
                        geom,
                        dtypes,
                        par,
                        context=scenario.isl,
                        seqs_per_dp=1,
                        disagg=True,
                        max_stage_layers=None,
                    )
                if not pp_feas:
                    continue
                pp_part_opt = pp_part if len(set(pp_part)) > 1 else None
            else:
                hbm_bd = hbm_breakdown_per_core_bytes(
                    geom, dtypes, par, 1, scenario.isl
                )
                last_stage_b = hbm_bd.total
                pp_part_opt = None
            hbm_b = hbm_bd.total
            if hbm_b > hbm_limit_b or (last_stage_b + logit_peak_b) > hbm_limit_b:
                continue
            sf_p = _scale_pp_stage_floor(
                _scale_ep_moe_floor(
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
                            prefix_hit_rate=scenario.prefix_hit_rate,
                        ),
                        dtypes,
                        topology=topo,
                    ),
                    chip,
                    geom,
                    par,
                    eff_isl,
                    topo,
                ),
                pp_part_opt,
                geom.num_layers,
                par.pp,
                chip,
                topo,
                par,
            )
            ttft_ms = (
                sf_p.serial_floor_s
                * (1.0 + float(par.pp - 1) / float(max(1, n_chunks)))
                * 1e3
            )
            if mode in ("auto", "mixed"):
                min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
            else:
                xfer_min_ms = (
                    _disagg_kv_xfer_s(geom, dtypes, chip, topo, topo, eff_isl) * 1e3
                )
                min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms + xfer_min_ms)
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
                    pp_part_opt,
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

            sf_ttft = _scale_ep_moe_floor(
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
                        prefix_hit_rate=scenario.prefix_hit_rate,
                    ),
                    dtypes,
                    topology=topo,
                ),
                chip,
                geom,
                par,
                eff_isl,
                topo,
            )
            min_req_ttft_ms = min(min_req_ttft_ms, sf_ttft.serial_floor_s * 1e3)

            b_kv_phys_core = hbm_footprint_per_core_bytes(
                geom, dtypes, par, 1, 1
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par, 1, 0)
            offload_b_per_core = (
                float(scenario.kv_offload_gb)
                * float(1 << 30)
                / float(max(1, par.tp * par.pp))
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
                logit_peak_mix_b = _peak_prefill_logit_bytes(
                    geom,
                    dtypes,
                    par,
                    pre_chunk_rows,
                    sampled_rows=b_res * max(1, spec_k + 1),
                )
                while b_res > 1 and (hbm_b + logit_peak_mix_b) > hbm_limit_b:
                    b_res -= 1
                    hbm_bd = hbm_breakdown_per_core_bytes(
                        geom, dtypes, par, b_res, ctx_full
                    )
                    hbm_b = hbm_bd.total
                    logit_peak_mix_b = _peak_prefill_logit_bytes(
                        geom,
                        dtypes,
                        par,
                        pre_chunk_rows,
                        sampled_rows=b_res * max(1, spec_k + 1),
                    )
                if (hbm_b + logit_peak_mix_b) > hbm_limit_b:
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
                _, h_hbm_only = _kv_capacity_and_hit_rate(
                    geom,
                    dtypes,
                    par,
                    b_res,
                    b_local,
                    ctx_full,
                    hbm_limit_b,
                    w_base_b,
                    b_kv_phys_core,
                    0.0,
                    scenario,
                )
                offload_read_b = (
                    max(0.0, h_eff - h_hbm_only) * float(scenario.isl) * b_kv_phys_core
                )
                pcie_core_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
                pcie_read_s = (
                    offload_read_b / pcie_core_bw
                    if offload_read_b > 0.0
                    and pcie_core_bw > 0.0
                    and math.isfinite(pcie_core_bw)
                    else 0.0
                )
                cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
                eff_isl_c = max(1, math.ceil((1.0 - h_eff) * scenario.isl))
                sf_ttft_c = (
                    _scale_ep_moe_floor(
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
                        ),
                        chip,
                        geom,
                        par,
                        eff_isl_c,
                        topo,
                    )
                    if h_eff != scenario.prefix_hit_rate
                    else sf_ttft
                )
                n_chunks_c = max(
                    1, math.ceil(eff_isl_c / max(1, scenario.prefill_chunk))
                )
                chunk_tok_c = min(eff_isl_c, scenario.prefill_chunk)
                sf_dec_raw = (
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
                sf_dec = _scale_ep_moe_floor(
                    sf_dec_raw,
                    chip,
                    geom,
                    par,
                    b_res * max(1, par.dp) * max(1, spec_k + 1),
                    topo,
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
                pre_ttft_s = sf_ttft_c.serial_floor_s + pcie_read_s
                w_srv_final = pre_ttft_s + dec_steps_per_req * sf_dec.serial_floor_s
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
                p90_ttft_ms = (pre_ttft_s + queue_wait_s) * 1e3 + (
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
                    kv_offload_gb=scenario.kv_offload_gb,
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
            for par_d in _candidate_decode_parallels(geom, topo_d):
                offload_b_per_core_d = (
                    float(scenario.kv_offload_gb)
                    * float(1 << 30)
                    / float(max(1, par_d.tp * par_d.pp))
                )
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
                        pp_part_p,
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
                                _, h_hbm_d = _kv_capacity_and_hit_rate(
                                    geom,
                                    dtypes,
                                    par_d,
                                    b_res,
                                    b_local,
                                    ctx_full,
                                    hbm_limit_b,
                                    w_base_d,
                                    b_kv_phys_d,
                                    0.0,
                                    scenario,
                                )
                                offload_read_d_b = (
                                    max(0.0, h_eff - h_hbm_d)
                                    * float(scenario.isl)
                                    * b_kv_phys_d
                                )
                                pcie_d_bw = chip.pcie_bw / float(
                                    max(1, chip.cores_per_chip)
                                )
                                pcie_d_ms = (
                                    (offload_read_d_b / pcie_d_bw) * 1e3
                                    if offload_read_d_b > 0.0
                                    and pcie_d_bw > 0.0
                                    and math.isfinite(pcie_d_bw)
                                    else 0.0
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
                                        sf_p_eff = _scale_pp_stage_floor(
                                            _scale_ep_moe_floor(
                                                step_floor(
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
                                                ),
                                                chip,
                                                geom,
                                                par_p,
                                                eff_isl_p,
                                                topo_p,
                                            ),
                                            pp_part_p,
                                            geom.num_layers,
                                            par_p.pp,
                                            chip,
                                            topo_p,
                                            par_p,
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
                                sf_d_raw = (
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
                                sf_d = _scale_ep_moe_floor(
                                    sf_d_raw,
                                    chip,
                                    geom,
                                    par_d,
                                    b_res * max(1, par_d.dp) * max(1, spec_k + 1),
                                    topo_d,
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
                                uncached_d_tok = max(
                                    1, math.ceil((1.0 - h_eff) * scenario.isl)
                                )
                                kv_xfer_ms = (
                                    _disagg_kv_xfer_s(
                                        geom,
                                        dtypes,
                                        chip,
                                        topo_p,
                                        topo_d,
                                        uncached_d_tok,
                                    )
                                    * 1e3
                                )
                                p90_ttft_d_ms = (
                                    ttft_eff_ms
                                    + pcie_d_ms
                                    + kv_xfer_ms
                                    + queue_wait_d_ms
                                )
                                if scenario.think_time_s > 0.0:
                                    client_rate = float(c) / max(
                                        1e-9,
                                        (ttft_eff_ms + pcie_d_ms + kv_xfer_ms) * 1e-3
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
                                    pp_layer_partition=pp_part_p,
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
                                    kv_offload_gb=scenario.kv_offload_gb,
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
    pruned_sweep = _prune_sweep_candidates(geom, dtypes, chip, scenario, candidates)
    base_plan = ServingPlan(
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
        sweep_candidates=pruned_sweep,
        spec_tokens=spec_k,
    )
    win_range, cross_rate = sweep_prefix_hit_crossovers(geom, dtypes, chip, base_plan)
    return replace(
        base_plan,
        winner_prefix_hit_range=win_range,
        prefix_hit_crossover_rate=cross_rate,
    )


def _pool_sig(pool: PoolMeshPlan | None) -> tuple | None:
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
        pool.pp_layer_partition,
    )


def _layout_sig(pt: ParetoPoint) -> tuple:
    return (pt.mode, _pool_sig(pt.prefill_pool), _pool_sig(pt.decode_pool))


def _pool_parallel(pool: PoolMeshPlan) -> tuple[SliceTopology, Parallel]:
    topo = TOPOLOGIES[pool.topology]
    moe_m: Literal["tp_slice", "ep"] = "ep" if pool.ep > 1 else "tp_slice"
    return (
        topo,
        Parallel(
            tp=pool.tp,
            dp=pool.dp,
            ep=pool.ep,
            pp=pool.pp,
            pcp=pool.pcp,
            kv_split=pool.kv_split,
            moe=moe_m,
            moe_dispatch=pool.moe_dispatch,
            ranks_per_chip=topo.cores_per_chip,
            mesh_axes=pool.mesh_axes,
        ),
    )


def _prune_sweep_candidates(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scen: TrafficScenario,
    cands: list[ParetoPoint] | tuple[ParetoPoint, ...],
) -> tuple[ParetoPoint, ...]:
    """Prune candidates strictly dominated across every prefix hit rate h in [0, 1]."""
    if len(cands) <= 1:
        return tuple(cands)
    eff_al = max(1.0, scen.accept_len)
    eff_al_s = eff_al * 1e-3
    inv_al_ms = 1e3 / eff_al
    dec_steps = max(1.0, scen.osl / eff_al)
    ctx_full = scen.isl + scen.osl
    seq_tok = float(ctx_full)
    w_turns = max(0.0, scen.kv_working_set_turns)
    has_w_turns = w_turns > 0.0
    has_offload = scen.kv_offload_gb > 0.0
    offload_bytes = float(scen.kv_offload_gb) * float(1 << 30)
    has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
    has_ttft_slo = scen.ttft_slo_ms is not None
    tpot_slo = scen.tpot_slo_ms
    has_think = scen.think_time_s > 0.0
    need_ttft = has_ttft_slo or has_think
    need_dcn_split = has_dcn and need_ttft
    _ceil = math.ceil
    topo_chips = {k: v.num_chips for k, v in TOPOLOGIES.items()}
    kv_b_cache: dict[tuple[int, int, int, int], float] = {}

    def _get_b_kv(pool: PoolMeshPlan) -> float:
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                geom, dtypes, par_k, 1, 1
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par_k, 1, 0)
            kv_b_cache[k] = val
        return val

    mixed_groups: dict[tuple, list[tuple[float, float, int, ParetoPoint]]] = {}
    disagg_groups: dict[
        tuple, list[tuple[float, float, float, float, int, int, ParetoPoint]]
    ] = {}
    _m_get = mixed_groups.get
    _d_get = disagg_groups.get
    for idx, pt in enumerate(cands):
        d_pool = pt.decode_pool
        if pt.mode == "mixed":
            d_dp = d_pool.dp if d_pool.dp > 1 else 1
            d_idp = d_pool.instances * d_dp
            b_local = _ceil(pt.concurrency / (d_idp if d_idp > 1 else 1))
            if b_local < 1:
                b_local = 1
            b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
            cap_tok = pt.kv_cache_capacity_tokens
            if has_w_turns and cap_tok is not None:
                w_tok = w_turns * float(b_local * ctx_full)
                cap_t = float(cap_tok)
                r_c = cap_t / (w_tok if w_tok > 1.0 else 1.0)
                cap_ratio = r_c if r_c < 1.0 else 1.0
                if has_offload and need_ttft:
                    b_kv = _get_b_kv(d_pool)
                    tp_pp = d_pool.tp * d_pool.pp
                    off_tok = (
                        (offload_bytes / float(tp_pp if tp_pp > 1 else 1)) / b_kv
                        if b_kv > 0.0
                        else 0.0
                    )
                    hbm_cap_t = cap_t - off_tok
                    if hbm_cap_t < 0.0:
                        hbm_cap_t = 0.0
                    r_h = hbm_cap_t / (w_tok if w_tok > 1.0 else 1.0)
                    hbm_ratio = r_h if r_h < 1.0 else 1.0
                    pcie_k = b_kv if cap_ratio > hbm_ratio else 0.0
                else:
                    hbm_ratio = cap_ratio
                    pcie_k = 0.0
            else:
                cap_ratio = 1.0
                hbm_ratio = 1.0
                pcie_k = 0.0
            res_cap = float(b_res * d_dp)
            conc_f = float(pt.concurrency)
            c_srv = conc_f if conc_f <= res_cap else res_cap
            if has_ttft_slo:
                dem_cap = float(b_local * d_dp)
                c_dem = conc_f if conc_f <= dem_cap else dem_cap
                over = max(0.0, (c_dem / max(1e-9, res_cap)) - 1.0)
                mix_pen = 1 if pt.concurrency > 1 else 0
            else:
                over = 0.0
                mix_pen = 0
            mkey = (
                d_pool.topology,
                d_pool.instances,
                d_pool.tp,
                d_pool.dp,
                d_pool.ep,
                d_pool.pp,
                d_pool.pcp,
                d_pool.kv_split,
                d_pool.moe_dispatch,
                d_pool.pp_layer_partition,
                pt.total_chips,
                b_res,
                cap_ratio,
                hbm_ratio,
                pcie_k,
                mix_pen,
            )
            primary = conc_f if has_think else c_srv
            m_grp = _m_get(mkey)
            if m_grp is None:
                m_grp = []
                mixed_groups[mkey] = m_grp
            m_grp.append((-primary, over, idx, pt))
        else:
            p_pool = pt.prefill_pool
            assert p_pool is not None
            d_dp = d_pool.dp if d_pool.dp > 1 else 1
            d_idp = d_pool.instances * d_dp
            if d_idp < 1:
                d_idp = 1
            b_local = 0
            if d_pool.max_num_seqs is not None:
                b_res = d_pool.max_num_seqs
            else:
                b_local = _ceil(pt.concurrency / d_idp)
                if b_local < 1:
                    b_local = 1
                b_res = b_local
            cap_tok = pt.kv_cache_capacity_tokens
            if has_w_turns and cap_tok is not None:
                if b_local == 0:
                    b_local = _ceil(pt.concurrency / d_idp)
                    if b_local < 1:
                        b_local = 1
                w_tok = w_turns * float(b_local * ctx_full)
                cap_t = float(cap_tok)
                r_c = cap_t / (w_tok if w_tok > 1.0 else 1.0)
                cap_ratio = r_c if r_c < 1.0 else 1.0
                if has_offload and need_ttft:
                    b_kv = _get_b_kv(d_pool)
                    tp_pp = d_pool.tp * d_pool.pp
                    off_tok = (
                        (offload_bytes / float(tp_pp if tp_pp > 1 else 1)) / b_kv
                        if b_kv > 0.0
                        else 0.0
                    )
                    hbm_cap_t = cap_t - off_tok
                    if hbm_cap_t < 0.0:
                        hbm_cap_t = 0.0
                    r_h = hbm_cap_t / (w_tok if w_tok > 1.0 else 1.0)
                    hbm_ratio = r_h if r_h < 1.0 else 1.0
                    pcie_k = b_kv if cap_ratio > hbm_ratio else 0.0
                else:
                    hbm_ratio = cap_ratio
                    pcie_k = 0.0
            else:
                cap_ratio = 1.0
                hbm_ratio = 1.0
                pcie_k = 0.0
            sf_dec_s = pt.p90_tpot_ms * eff_al_s
            dec_dur_s = dec_steps * sf_dec_s
            res_cap_d = float(b_res * d_idp)
            conc_f = float(pt.concurrency)
            c_srv_d = conc_f if conc_f <= res_cap_d else res_cap_d
            rate_d = c_srv_d / (dec_dur_s if dec_dur_s > 1e-9 else 1e-9)
            tpot_h = sf_dec_s * inv_al_ms
            inter_h = 1000.0 / (tpot_h if tpot_h > 1e-6 else 1e-6)
            tpot_ok = tpot_slo is None or tpot_h <= tpot_slo
            chips_f = float(pt.total_chips)
            p_cap = float(p_pool.instances) * seq_tok / chips_f
            tput_d_cap = rate_d * seq_tok / chips_f
            if not need_ttft:
                c_cap = 0.0
                dkey: tuple = (
                    p_pool.topology,
                    p_pool.tp,
                    p_pool.dp,
                    p_pool.ep,
                    p_pool.pp,
                    p_pool.pcp,
                    p_pool.kv_split,
                    p_pool.moe_dispatch,
                    p_pool.pp_layer_partition,
                    cap_ratio,
                    tpot_ok,
                )
            else:
                c_cap = conc_f * seq_tok / chips_f if has_think else 0.0
                if has_ttft_slo:
                    if b_local == 0:
                        b_local = _ceil(pt.concurrency / d_idp)
                        if b_local < 1:
                            b_local = 1
                    dem_cap_d = float(b_local * d_idp)
                    c_dem_d = conc_f if conc_f <= dem_cap_d else dem_cap_d
                    q_wait_d_ms = (
                        max(0.0, (c_dem_d / max(1e-9, res_cap_d)) - 1.0)
                        * dec_dur_s
                        * 1e3
                    )
                else:
                    q_wait_d_ms = 0.0
                think_dur_k = dec_dur_s if has_think else 0.0
                min_c = (
                    max(
                        1,
                        min(
                            topo_chips[p_pool.topology],
                            topo_chips[d_pool.topology],
                        ),
                    )
                    if need_dcn_split
                    else 0
                )
                dkey = (
                    p_pool.topology,
                    p_pool.tp,
                    p_pool.dp,
                    p_pool.ep,
                    p_pool.pp,
                    p_pool.pcp,
                    p_pool.kv_split,
                    p_pool.moe_dispatch,
                    p_pool.pp_layer_partition,
                    min_c,
                    cap_ratio,
                    hbm_ratio,
                    pcie_k,
                    tpot_ok,
                    q_wait_d_ms,
                    think_dur_k,
                )
            d_grp = _d_get(dkey)
            if d_grp is None:
                d_grp = []
                disagg_groups[dkey] = d_grp
            d_grp.append(
                (
                    -p_cap,
                    -tput_d_cap,
                    -c_cap,
                    -inter_h,
                    pt.total_chips,
                    idx,
                    pt,
                )
            )

    kept: list[tuple[int, ParetoPoint]] = []
    for m_items in mixed_groups.values():
        m_items.sort()
        min_over = math.inf
        for _neg_prim, over, idx, pt in m_items:
            if over < min_over:
                min_over = over
                kept.append((idx, pt))
    strict_band = 1.0 + 1e-6
    for d_items in disagg_groups.values():
        d_items.sort()
        if not has_think:
            skyline_2d: list[tuple[int, float, float, float]] = []
            s_ptr = 0
            max_dk_strict = -1.0
            for neg_p, neg_d, _neg_c, neg_i, chips, idx, pt in d_items:
                p_cap = -neg_p
                tput_d = -neg_d
                inter = -neg_i
                p_req = p_cap * strict_band
                n_sky = len(skyline_2d)
                while s_ptr < n_sky and skyline_2d[s_ptr][1] > p_req:
                    dk_s = skyline_2d[s_ptr][2]
                    if dk_s > max_dk_strict:
                        max_dk_strict = dk_s
                    s_ptr += 1
                if max_dk_strict > tput_d * strict_band:
                    continue
                start_k = 0 if max_dk_strict >= tput_d else s_ptr
                dominated = False
                for k in range(start_k, n_sky):
                    ck, pk, dk, ik = skyline_2d[k]
                    if pk >= p_cap and dk >= tput_d and ik >= inter and ck <= chips:
                        dominated = True
                        break
                if dominated:
                    continue
                kept.append((idx, pt))
                skyline_2d.append((chips, p_cap, tput_d, inter))
        else:
            skyline_3d: list[tuple[int, float, float, float, float]] = []
            s_ptr = 0
            max_dk_3d = -1.0
            max_cpk_3d = -1.0
            for neg_p, neg_d, neg_c, neg_i, chips, idx, pt in d_items:
                p_cap = -neg_p
                tput_d = -neg_d
                c_cap = -neg_c
                inter = -neg_i
                p_req = p_cap * strict_band
                d_req = tput_d * strict_band
                c_req = c_cap * strict_band
                n_sky = len(skyline_3d)
                while s_ptr < n_sky and skyline_3d[s_ptr][1] > p_req:
                    dk_s = skyline_3d[s_ptr][2]
                    cpk_s = skyline_3d[s_ptr][3]
                    if dk_s > max_dk_3d:
                        max_dk_3d = dk_s
                    if cpk_s > max_cpk_3d:
                        max_cpk_3d = cpk_s
                    s_ptr += 1
                start_k = 0 if (max_dk_3d >= tput_d and max_cpk_3d >= c_cap) else s_ptr
                dominated = False
                for k in range(start_k, n_sky):
                    ck, pk, dk, cpk, ik = skyline_3d[k]
                    if (pk > p_req and dk > d_req and cpk > c_req) or (
                        pk >= p_cap
                        and dk >= tput_d
                        and cpk >= c_cap
                        and ik >= inter
                        and ck <= chips
                    ):
                        dominated = True
                        break
                if dominated:
                    continue
                kept.append((idx, pt))
                skyline_3d.append((chips, p_cap, tput_d, c_cap, inter))
    kept.sort()
    return tuple(pt for _idx, pt in kept)


def _best_frontier_winner(
    scored: list[tuple[float, float, int, ParetoPoint]],
) -> ParetoPoint:
    ordered = sorted(scored, key=lambda x: (-x[1], -x[0], -x[2]))
    best_tput = -1.0
    best_pt = ordered[0][3]
    for tput, _inter, _neg_c, pt in ordered:
        if tput > best_tput * (1.0 + 1e-9):
            best_tput = tput
            best_pt = pt
    return best_pt


_EP_BUSIEST_CACHE: dict[tuple[int, int, int, int], float] = {}


def _fast_ep_busiest(num_experts: int, top_k: int, rows: int, ranks: int) -> float:
    """Active-tail Binomial CDF order statistic matching ep_busiest_experts."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0 or ranks <= 0:
        return 0.0
    if ranks == 1:
        return (
            float(num_experts)
            if top_k >= num_experts
            else float(num_experts) * (1.0 - (1.0 - top_k / num_experts) ** rows)
        )
    n = num_experts // ranks
    if n <= 0:
        return float(math.ceil(min(num_experts, top_k) / ranks))
    if top_k >= num_experts:
        return float(n)
    q_one = 1.0 - top_k / num_experts
    q = q_one**rows
    if q <= 1e-15:
        return float(n)
    key = (num_experts, top_k, rows, ranks)
    cached = _EP_BUSIEST_CACHE.get(key)
    if cached is not None:
        return cached
    p = 1.0 - q
    if p >= 0.5:
        pmf = p**n
        surv = pmf
        q_over_p = q / p
        acc = 0.0
        rem = n
        for y in range(n):
            cdf = 1.0 - surv
            if cdf <= 0.0:
                acc += float(rem)
                break
            c_r = cdf**ranks if cdf < 1.0 else 1.0
            if c_r < 1e-16:
                acc += float(rem)
                break
            acc += 1.0 - c_r
            rem -= 1
            pmf *= (n - y) / (y + 1) * q_over_p
            surv += pmf
        res = acc
    else:
        pmf = q**n
        cdf = 0.0
        p_over_q = p / q
        acc = 0.0
        for x in range(n):
            cdf += pmf
            if cdf >= 1.0:
                break
            if cdf < 1e-8:
                acc += 1.0
            else:
                term = 1.0 - cdf**ranks
                if term < 1e-16 and x > n * p:
                    break
                acc += term
            pmf *= (n - x) / (x + 1) * p_over_q
        res = acc
    if len(_EP_BUSIEST_CACHE) < 65536:
        _EP_BUSIEST_CACHE[key] = res
    return res


@lru_cache(maxsize=4096)
def _coll_spec(
    kind: str,
    group: int,
    algo: str,
    ranks_per_chip: int,
    c_axes: tuple[int, int, int, int] | None,
    chip: ChipSpec,
    topology: SliceTopology | None,
) -> tuple[float, float, bool]:
    """Return (lat_s, byte_slope_s, zero_when_empty) for one collective."""
    if group <= 1:
        return (0.0, 0.0, True)
    use_torus = topology is not None or c_axes is not None
    unit_hop_lat_s = (
        chip.ici_hop_latency_s
        if (
            chip.ici_bw > 0
            and math.isfinite(chip.ici_bw)
            and chip.ici_hop_latency_s > 0
            and math.isfinite(chip.ici_hop_latency_s)
        )
        else 0.0
    )
    if use_torus:
        submesh = submesh_for_group(group, topology, ranks_per_chip, c_axes)
        d2d, ax, ay, az = submesh
        grp = d2d * ax * ay * az
        if grp <= 1:
            return (0.0, 0.0, True)
        if ax * ay * az == 1:
            intra_bw = chip.ici_bw * chip.cores_per_chip
            xf = 1.0 if kind in ("push", "p2p") else (grp - 1) / grp
            mult = 2.0 if kind == "ar" else 1.0
            return (0.0, mult * (xf / intra_bw), True)
        ici_hops = 0
        lanes_sum = 0
        for axis_idx, extent in enumerate((ax, ay, az)):
            if extent <= 1:
                continue
            wrapped = (
                topology.axis_wrapped(axis_idx, extent)
                if topology is not None
                else False
            )
            if wrapped:
                ici_hops += extent // 2
                lanes_sum += 2
            else:
                ici_hops += extent - 1
                lanes_sum += 1
        contention = float(max(1, d2d))
        cap_bw = chip.ici_bw if d2d >= 2 else chip.ici_bw * chip.cores_per_chip
        raw_bw = lanes_sum * chip.ici_link_bw / contention
        eff_bw = (
            min(raw_bw, cap_bw) if math.isfinite(cap_bw) and cap_bw > 0.0 else raw_bw
        )
        lat_1p = ici_hops * unit_hop_lat_s
        inv_eff = 1.0 / eff_bw
        if kind == "p2p":
            return (lat_1p, inv_eff, True)
        xf_inv_eff = (1.0 if kind == "push" else (grp - 1) / grp) * inv_eff
        if kind == "ar":
            return (2.0 * lat_1p, 2.0 * xf_inv_eff, True)
        return (lat_1p, xf_inv_eff, True)
    offchip = max(0, group - min(group, ranks_per_chip)) / (group - 1)
    if kind in ("push", "p2p"):
        w_mul, hops = 1.0, 1
    elif kind == "ar":
        if algo == "one_shot":
            w_mul, hops = float(group - 1), 1
        elif algo == "two_shot":
            w_mul, hops = 2.0 * (group - 1) / group, 2
        else:
            w_mul, hops = 2.0 * (group - 1) / group, 2 * (group - 1)
    else:
        w_mul = (group - 1) / group
        hops = group - 1 if algo == "ring" else 1
    c_hops = hops if offchip > 0.0 else 0
    hop_s = c_hops * unit_hop_lat_s
    has_ici = chip.ici_bw > 0 and math.isfinite(chip.ici_bw)
    inv_ici = (w_mul * offchip / chip.ici_bw) if has_ici else 0.0
    return (hop_s, inv_ici, False)


def _price_coll(spec: tuple[float, float, bool], payload: float) -> float:
    if payload <= 0.0 and spec[2]:
        return 0.0
    bw_t = payload * spec[1]
    lat = spec[0]
    return lat if lat >= bw_t else bw_t


@lru_cache(maxsize=64)
def _make_draft_geom(geom: ModelGeometry) -> ModelGeometry:
    draft_layer_type = (
        "dsa"
        if (geom.kv_lora_rank > 0 and geom.index_topk > 0)
        else ("mla" if geom.kv_lora_rank > 0 else "full_attention")
    )
    use_dense_mtp = geom.mtp_inter > 0
    return ModelGeometry(
        hidden=geom.hidden,
        num_layers=1,
        layer_types=(draft_layer_type,),
        num_heads=geom.num_heads,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        attn_output_gate=geom.attn_output_gate,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=geom.conv_kernel,
        num_experts=0 if use_dense_mtp else geom.num_experts,
        top_k=0 if use_dense_mtp else geom.top_k,
        moe_inter=0 if use_dense_mtp else geom.moe_inter,
        shared_inter=0 if use_dense_mtp else geom.shared_inter,
        vocab=geom.vocab,
        mtp_layers=geom.mtp_layers,
        dense_inter=geom.mtp_inter if use_dense_mtp else geom.dense_inter,
        q_lora_rank=geom.q_lora_rank,
        kv_lora_rank=geom.kv_lora_rank,
        qk_nope_head_dim=geom.qk_nope_head_dim,
        qk_rope_head_dim=geom.qk_rope_head_dim,
        v_head_dim=geom.v_head_dim,
        index_n_heads=geom.index_n_heads,
        index_head_dim=geom.index_head_dim,
        index_topk=geom.index_topk,
        sliding_window=geom.sliding_window,
        block_parallel=geom.block_parallel,
        mtp_inter=geom.mtp_inter,
    )


@lru_cache(maxsize=64)
def _classify_layers_cached(
    geom: ModelGeometry,
) -> tuple[int, int, int, int, int, int, float, float, float, float]:
    return _classify_layers(
        geom.layer_types,
        geom.compress_ratios,
        geom.kv_lora_rank,
        geom.index_topk,
    )


class _FastPassEvaluator:
    """Pre-lowered serial-floor evaluator for one pool mesh."""

    def __init__(
        self,
        geom: ModelGeometry,
        dt: Dtypes,
        chip: ChipSpec,
        topology: SliceTopology | None,
        par: Parallel,
        pp_part: tuple[int, ...] | None = None,
    ) -> None:
        self.geom = geom
        self.dt = dt
        self.chip = chip
        self.topology = topology
        self.par = par
        tp = par.tp
        self.tp = tp
        self.tp_eff = tp if tp > 1 else 1
        self.pcp = max(1, par.pcp)
        self.pp = max(1, par.pp)
        self.dp = max(1, par.dp)
        self.kv_split = par.kv_split
        self.mxu_m_tile = chip.mxu_m_tile
        self._last_moe_ep_delta = 0.0
        self._draft_ev: _FastPassEvaluator | None = None
        inv_hbm = 1.0 / chip.hbm_bw
        inv_fp8 = 1.0 / chip.fp8_flops
        inv_bf16 = 1.0 / chip.bf16_flops
        if par.ep > 1 and geom.num_experts > 0 and geom.top_k > 0:
            self.ep_z = _ep_z_unit(par.ep)
            self.ep_m1_f = float(par.ep - 1)
            self.ep_cap_f = float(par.ep)
        else:
            self.ep_z = 0.0
            self.ep_m1_f = 0.0
            self.ep_cap_f = 1.0

        def _mm_rate(w_bytes: float) -> float:
            return inv_fp8 if (w_bytes < 2.0 and dt.act == "a8") else inv_bf16

        rate_dense = _mm_rate(dt.w_dense)
        rate_aux = _mm_rate(dt.w_aux)
        rate_expert = _mm_rate(dt.w_expert)
        rate_router = _mm_rate(dt.router)
        rate_lm = _mm_rate(dt.lm_head)

        pp_scale = (
            float(math.ceil(geom.num_layers / self.pp)) / float(geom.num_layers)
            if geom.num_layers > 0 and par.pp > 1
            else 1.0
        )
        if par.pp > 1 and pp_part and geom.num_layers > 0:
            even_max = max(1, math.ceil(geom.num_layers / par.pp))
            actual_max = max(pp_part)
            pp_ratio = (
                float(actual_max) / float(even_max) if actual_max != even_max else 1.0
            )
        else:
            pp_ratio = 1.0
        self.pp_ratio = pp_ratio
        self.even_pp_ratio = pp_ratio == 1.0
        self.need_full_extra = par.kv_split > 1 or par.pcp > 1

        (
            n_gqa,
            n_mla,
            n_dsa,
            n_swa,
            n_gdn,
            n_kda,
            kv_n_gqa,
            kv_n_mla,
            kv_n_dsa,
            kv_n_swa,
        ) = _classify_layers_cached(geom)

        kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
        nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
        nq_gqa = geom.num_heads // tp
        q_mult = 2 if geom.attn_output_gate else 1
        q_out = geom.num_heads * geom.head_dim * q_mult

        def _sp(kind: str, group: int) -> tuple[float, float, bool]:
            c_axes = par.mesh_axes if group == tp else None
            return _coll_spec(
                kind,
                group,
                par.ar_algo,
                par.ranks_per_chip,
                c_axes,
                chip,
                topology,
            )

        self.sp_ar_tp = _sp("ar", tp)
        self.sp_ag_tp = _sp("ag", tp)
        self.sp_rs_tp = _sp("rs", tp)
        self.sp_p2p_pp = _sp("p2p", par.pp)
        self.sp_ag_kv = _sp("ag", par.kv_split)
        self.sp_rs_kv = _sp("rs", par.kv_split)
        self.sp_push_kv = _sp("push", par.kv_split)
        self.sp_ag_pcp = _sp("ag", par.pcp)
        self.kv_zero_empty = (
            self.sp_ag_kv[2] and self.sp_rs_kv[2] and self.sp_push_kv[2]
        )

        self.has_gqa = n_gqa > 0
        self.gqa_dec_kv_factor = 0.0
        self.gqa_hb_base = 0.0
        self.gqa_kv_hs_unit = 0.0
        self.gqa_row_ms = 0.0
        self.gqa_score_ms_unit = 0.0
        self.gqa_c0 = 0
        self.gqa_nq = 0
        self.gqa_hd = 0
        if self.has_gqa:
            n_eff0 = n_gqa * pp_scale
            gqa_hd_eff = (
                geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
            )
            gqa_q_out_eff = geom.num_heads * gqa_hd_eff * q_mult
            w_elems = (
                float(
                    n_eff0
                    * (
                        geom.hidden * (gqa_q_out_eff // tp)
                        + 2 * geom.hidden * kvh * gqa_hd_eff
                        + nq_gqa * gqa_hd_eff * geom.hidden
                    )
                )
                * pp_ratio
            )
            gqa_kv_scale = (kv_n_gqa / float(n_gqa)) if geom.compress_ratios else 1.0
            self.gqa_dec_kv_factor = gqa_kv_scale / par.kv_split
            self.gqa_hb_base = w_elems * dt.w_dense * inv_hbm
            self.gqa_kv_hs_unit = (
                float(kvh * gqa_hd_eff * 2) * n_eff0 * dt.kv * pp_ratio * inv_hbm
            )
            self.gqa_row_ms = 2.0 * w_elems * rate_dense
            self.gqa_score_ms_unit = (
                float(4 * gqa_hd_eff * nq_gqa * n_eff0) * pp_ratio * inv_bf16
            )
            self.gqa_c0 = max(1, round(n_eff0))
            self.gqa_nq = nq_gqa
            self.gqa_hd = gqa_hd_eff

        self.has_swa = n_swa > 0
        self.swa_cap = 0
        self.swa_dec_kv_factor = 0.0
        self.swa_hb_base = 0.0
        self.swa_kv_hs_unit = 0.0
        self.swa_row_ms = 0.0
        self.swa_score_ms_unit = 0.0
        self.swa_c0 = 0
        self.swa_nq = 0
        self.swa_hd = 0
        if self.has_swa:
            n_eff0 = n_swa * pp_scale
            self.swa_cap = geom.sliding_window
            w_elems = (
                float(
                    n_eff0
                    * (
                        geom.hidden * (q_out // tp)
                        + 2 * geom.hidden * kvh * geom.head_dim
                        + nq * geom.head_dim * geom.hidden
                    )
                )
                * pp_ratio
            )
            swa_kv_scale = (kv_n_swa / float(n_swa)) if geom.compress_ratios else 1.0
            self.swa_dec_kv_factor = swa_kv_scale / par.kv_split
            self.swa_hb_base = w_elems * dt.w_dense * inv_hbm
            self.swa_kv_hs_unit = (
                float(kvh * geom.head_dim * 2) * n_eff0 * dt.kv * pp_ratio * inv_hbm
            )
            self.swa_row_ms = 2.0 * w_elems * rate_dense
            self.swa_score_ms_unit = (
                float(4 * geom.head_dim * nq * n_eff0) * pp_ratio * inv_bf16
            )
            self.swa_c0 = max(1, round(n_eff0))
            self.swa_nq = nq
            self.swa_hd = geom.head_dim

        self.mla_fams: list[
            tuple[
                bool,
                float,
                float,
                float,
                float,
                float,
                float,
                int,
                int,
                int,
            ]
        ] = []
        for _fam_name, n_fam, kv_n_fam, is_sparse in (
            ("mla", n_mla, kv_n_mla, False),
            ("dsa", n_dsa, kv_n_dsa, True),
        ):
            if n_fam <= 0:
                continue
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
            n_eff0 = n_fam * pp_scale
            w_elems = n_eff0 * mla_w_layer * pp_ratio
            fam_kv_scale = (kv_n_fam / float(n_fam)) if geom.compress_ratios else 1.0
            latent_width = kv_lora + rope_d
            kv_hs_u = n_eff0 * float(latent_width) * dt.kv * pp_ratio * inv_hbm
            idx_hs_u = (
                n_eff0
                * float(geom.index_head_dim)
                * fam_kv_scale
                * dt.kv
                / par.kv_split
                * pp_ratio
                * inv_hbm
                if (is_sparse and geom.index_head_dim > 0)
                else 0.0
            )
            score_ms_u = (
                float(2 * nq * (kv_lora + rope_d + kv_lora) * n_eff0)
                * pp_ratio
                * inv_bf16
            )
            self.mla_fams.append(
                (
                    is_sparse,
                    w_elems * dt.w_dense * inv_hbm,
                    2.0 * w_elems * rate_dense,
                    fam_kv_scale / par.kv_split,
                    kv_hs_u,
                    idx_hs_u,
                    score_ms_u,
                    max(1, round(n_eff0)),
                    nq,
                    kv_lora,
                )
            )

        self.has_gdn = n_gdn > 0
        self.gdn_hb_base = 0.0
        self.gdn_state_hs_unit = 0.0
        self.gdn_row_ms = 0.0
        self.gdn_c0 = 0
        if self.has_gdn:
            n_eff0 = n_gdn * pp_scale
            kh = max(1, geom.linear_k_heads // tp) if geom.linear_k_heads > 0 else 0
            vh = max(1, geom.linear_v_heads // tp) if geom.linear_v_heads > 0 else 0
            qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
            z_r = vh * geom.linear_v_dim
            d_elems = (
                float(n_eff0 * (geom.hidden * (qkv_r + z_r) + z_r * geom.hidden))
                * pp_ratio
            )
            a_elems = (
                float(n_eff0 * (geom.hidden * (2 * vh) + qkv_r * geom.conv_kernel))
                * pp_ratio
            )
            rec_state = (
                float(n_eff0 * vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
            )
            conv_state = float(n_eff0 * qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
            self.gdn_hb_base = (d_elems * dt.w_dense + a_elems * dt.w_aux) * inv_hbm
            self.gdn_state_hs_unit = (rec_state + conv_state) * pp_ratio * inv_hbm
            self.gdn_row_ms = (
                2.0 * d_elems * rate_dense
                + 2.0 * a_elems * rate_aux
                + float(4 * vh * geom.linear_k_dim * geom.linear_v_dim * n_eff0)
                * pp_ratio
                * inv_bf16
            )
            self.gdn_c0 = max(1, round(n_eff0))

        self.has_kda = n_kda > 0
        self.kda_hb_base = 0.0
        self.kda_state_hs_unit = 0.0
        self.kda_row_ms = 0.0
        self.kda_c0 = 0
        self.kda_nq = 0
        self.kda_vd = 0
        if self.has_kda:
            n_eff0 = n_kda * pp_scale
            kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
            vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
            kd = geom.linear_k_dim or geom.head_dim
            vd = geom.linear_v_dim or geom.head_dim
            proj_r = 2 * kh * kd + 2 * vh * vd
            w_elems = (
                float(n_eff0 * (geom.hidden * proj_r + (vh * vd) * geom.hidden))
                * pp_ratio
            )
            self.kda_hb_base = w_elems * dt.w_dense * inv_hbm
            self.kda_state_hs_unit = (
                float(n_eff0 * vh * kd * vd) * dt.state * pp_ratio * inv_hbm
            )
            self.kda_row_ms = (
                2.0 * w_elems * rate_dense
                + float(4 * vh * kd * vd * n_eff0) * pp_ratio * inv_bf16
            )
            self.kda_c0 = max(1, round(n_eff0))
            self.kda_nq = max(1, geom.num_heads // tp) if geom.num_heads > 0 else 0
            self.kda_vd = vd

        self.has_dense_ffn = False
        self.mlp_hs = 0.0
        self.mlp_row_ms = 0.0
        self.mlp_c0 = 0
        self.mlp_d_par = par
        self.mlp_sp_ag_ep: tuple[float, float, bool] = (0.0, 0.0, True)
        self.mlp_sp_push: tuple[float, float, bool] = (0.0, 0.0, True)
        self.has_moe_ffn = False
        self.ep_ranks = 1
        self.pcp_per_ep = 1
        self.dp_per_ep = 1
        self.routed_hs_unit = 0.0
        self.moe_fixed_hs = 0.0
        self.routed_row_ms = 0.0
        self.shared_row_ms = 0.0
        self.router_row_ms = 0.0
        self.moe_c0 = 0
        self.sp_ag_ep: tuple[float, float, bool] = (0.0, 0.0, True)
        self.sp_push_disp: tuple[float, float, bool] = (0.0, 0.0, True)
        if geom.num_experts > 0:
            if geom.n_moe_layers is not None:
                n_moe_ffn = min(geom.num_layers, max(0, geom.n_moe_layers))
                n_dense_ffn = (
                    geom.n_dense_layers
                    if geom.n_dense_layers is not None
                    else (geom.num_layers - n_moe_ffn)
                )
            else:
                n_dense_ffn = min(geom.num_layers, geom.first_k_dense_replace)
                n_moe_ffn = geom.num_layers - n_dense_ffn
            if n_dense_ffn > 0 and geom.dense_inter > 0:
                self.has_dense_ffn = True
                n_d_eff0 = n_dense_ffn * pp_scale
                mlp_inter = geom.dense_inter // tp
                m_elems = float(n_d_eff0 * 3 * geom.hidden * mlp_inter) * pp_ratio
                self.mlp_hs = m_elems * dt.w_dense * inv_hbm
                self.mlp_row_ms = 2.0 * m_elems * rate_dense
                self.mlp_c0 = max(1, round(n_d_eff0))
                self.mlp_d_par = Parallel(
                    tp=par.tp,
                    residual=par.residual,
                    ar_algo=par.ar_algo,
                    ranks_per_chip=par.ranks_per_chip,
                    moe="ep" if par.residual == "sp" else "tp_slice",
                )
                mlp_ep_r = (
                    self.mlp_d_par.ep if self.mlp_d_par.ep > 1 else self.mlp_d_par.tp
                )
                self.mlp_sp_ag_ep = _sp("ag", mlp_ep_r)
                mlp_dg = (
                    mlp_ep_r
                    if self.mlp_d_par.moe_dispatch == "a2a_push"
                    else self.mlp_d_par.tp
                )
                self.mlp_sp_push = _sp("push", mlp_dg)
            if n_moe_ffn > 0:
                self.has_moe_ffn = True
                n_m_eff0 = n_moe_ffn * pp_scale
                self.ep_ranks = par.ep if par.ep > 1 else tp
                self.pcp_per_ep = (
                    min(self.pcp, max(1, self.ep_ranks // max(1, tp)))
                    if par.ep > 1
                    else 1
                )
                self.dp_per_ep = (
                    min(
                        self.dp,
                        max(1, self.ep_ranks // max(1, tp * self.pcp_per_ep)),
                    )
                    if par.ep > 1
                    else 1
                )
                width = (
                    geom.moe_inter // max(1, tp // max(1, self.ep_ranks))
                    if (par.moe == "ep" and self.ep_ranks < tp)
                    else (geom.moe_inter if par.moe == "ep" else geom.moe_inter // tp)
                )
                self.routed_hs_unit = (
                    float(n_m_eff0 * 3 * geom.hidden * width)
                    * dt.w_expert
                    * pp_ratio
                    * inv_hbm
                )
                isr_tp = max(1, geom.shared_inter // tp) if geom.shared_inter > 0 else 0
                isr_eff = geom.shared_inter if par.residual == "sp" else isr_tp
                shared_elems = float(n_m_eff0 * 3 * geom.hidden * isr_eff) * pp_ratio
                router_elems = (
                    float(n_m_eff0 * geom.hidden * geom.num_experts) * pp_ratio
                )
                self.moe_fixed_hs = (
                    shared_elems * dt.w_dense + router_elems * dt.router
                ) * inv_hbm
                eff_moe_div = (
                    max(tp * self.pcp_per_ep, self.ep_ranks) if par.ep > 1 else tp
                )
                routed_act_elems = (
                    float(
                        n_m_eff0
                        * geom.top_k
                        * 3
                        * geom.hidden
                        * (geom.moe_inter / eff_moe_div)
                    )
                    * pp_ratio
                )
                self.routed_row_ms = 2.0 * routed_act_elems * rate_expert
                self.shared_row_ms = 2.0 * shared_elems * rate_dense
                self.router_row_ms = 2.0 * router_elems * rate_router
                self.moe_c0 = max(1, round(n_m_eff0))
                self.sp_ag_ep = _sp("ag", self.ep_ranks)
                moe_dg = self.ep_ranks if par.moe_dispatch == "a2a_push" else tp
                self.sp_push_disp = _sp("push", moe_dg)
        else:
            self.has_dense_ffn = True
            n_d_eff0 = geom.num_layers * pp_scale
            mlp_inter = geom.dense_inter // tp
            m_elems = float(n_d_eff0 * 3 * geom.hidden * mlp_inter) * pp_ratio
            self.mlp_hs = m_elems * dt.w_dense * inv_hbm
            self.mlp_row_ms = 2.0 * m_elems * rate_dense
            self.mlp_c0 = max(1, round(n_d_eff0))
            self.mlp_d_par = par
            mlp_ep_r = par.ep if par.ep > 1 else tp
            self.mlp_sp_ag_ep = _sp("ag", mlp_ep_r)
            mlp_dg = mlp_ep_r if par.moe_dispatch == "a2a_push" else tp
            self.mlp_sp_push = _sp("push", mlp_dg)

        self.has_ple = "ple" in geom.layer_types or geom.ple_dim > 0
        self.ple_hb_base = 0.0
        self.ple_lookup_hs = 0.0
        self.ple_row_ms = 0.0
        if self.has_ple:
            n_ple0 = geom.num_layers * pp_scale
            ple_elems = float(n_ple0 * geom.hidden * geom.ple_dim // tp) * pp_ratio
            self.ple_hb_base = ple_elems * dt.w_aux * inv_hbm
            self.ple_lookup_hs = (
                float(n_ple0 * geom.ple_dim) * dt.w_aux * pp_ratio * inv_hbm
            )
            self.ple_row_ms = 2.0 * ple_elems * rate_aux

        lm_elems = float(geom.vocab * geom.hidden // tp) / float(self.pp)
        self.lm_hs = lm_elems * dt.lm_head * inv_hbm
        self.lm_row_ms = 2.0 * lm_elems * rate_lm
        fc_elems = float(2 * geom.hidden * geom.hidden // tp)
        self.fc_hs_unit = fc_elems * dt.w_aux * inv_hbm
        self.fc_row_ms_unit = 2.0 * fc_elems * rate_aux
        sum_c0 = (
            self.gqa_c0
            + self.swa_c0
            + self.gdn_c0
            + self.kda_c0
            + sum(m_fam[7] for m_fam in self.mla_fams)
        )
        self.sum_attn_c0 = sum_c0

    def _scale_chain_times(self, unit_times: list[float], c0: int) -> float:
        if not unit_times:
            return 0.0
        if self.even_pp_ratio:
            s = 0.0
            for t in unit_times:
                s += t
            return c0 * s
        b_len = len(unit_times)
        target_n = max(1, round((b_len * c0) * self.pp_ratio))
        q, r = divmod(target_n, b_len)
        tot = 0.0
        for j, t in enumerate(unit_times):
            tot += (q + (1 if j < r else 0)) * t
        return tot

    def _full_attn_extra_s(
        self,
        batch: int,
        rows_per_seq: int,
        nq_c: int,
        hd_c: int,
        extra_rows: int,
        c0: int,
    ) -> float:
        par = self.par
        s = 0.0
        if par.kv_split > 1 and (rows_per_seq > 0 or not self.kv_zero_empty):
            g = par.kv_split
            dec_rows = batch * rows_per_seq
            if par.kv_merge == "ag_rs":
                s += (
                    _price_coll(
                        self.sp_ag_kv,
                        float(dec_rows * g * nq_c * hd_c * 2),
                    )
                    + _price_coll(
                        self.sp_ag_kv,
                        float(g * dec_rows * g * nq_c * 4),
                    )
                    + _price_coll(
                        self.sp_rs_kv,
                        float(dec_rows * g * nq_c * hd_c * 4),
                    )
                )
            else:
                per_peer = float((batch + 1) * rows_per_seq * nq_c * hd_c * 4)
                s += _price_coll(self.sp_push_kv, (g - 1) * per_peer)
        if par.pcp > 1 and extra_rows > 0:
            s += _price_coll(
                self.sp_ag_pcp,
                float(extra_rows * par.pcp * max(1, nq_c) * hd_c * 2),
            )
        return c0 * s

    def _attn_chain_s(
        self,
        batch: int,
        rows_per_seq: int,
        nq_c: int,
        hd_c: int,
        is_full: bool,
        extra_rows: int,
        c0: int,
        tp_ar_t: float,
        tp_ag_t: float,
        tp_rs_t: float,
    ) -> float:
        par = self.par
        tp = self.tp
        unit: list[float] = []
        if tp > 1:
            if par.residual == "replicated":
                unit.append(tp_ar_t)
            else:
                unit.append(tp_ag_t)
                if self.geom.block_parallel <= 1:
                    unit.append(tp_rs_t)
        if is_full and par.kv_split > 1:
            g = par.kv_split
            dec_rows = batch * rows_per_seq
            if par.kv_merge == "ag_rs":
                unit.append(
                    _price_coll(
                        self.sp_ag_kv,
                        float(dec_rows * g * nq_c * hd_c * 2),
                    )
                )
                unit.append(
                    _price_coll(
                        self.sp_ag_kv,
                        float(g * dec_rows * g * nq_c * 4),
                    )
                )
                unit.append(
                    _price_coll(
                        self.sp_rs_kv,
                        float(dec_rows * g * nq_c * hd_c * 4),
                    )
                )
            else:
                per_peer = float((batch + 1) * rows_per_seq * nq_c * hd_c * 4)
                unit.append(_price_coll(self.sp_push_kv, (g - 1) * per_peer))
        if is_full and par.pcp > 1 and extra_rows > 0:
            unit.append(
                _price_coll(
                    self.sp_ag_pcp,
                    float(extra_rows * par.pcp * max(1, nq_c) * hd_c * 2),
                )
            )
        return self._scale_chain_times(unit, c0)

    def _moe_chain_s(
        self,
        rows: int,
        num_experts: int,
        top_k: int,
        mpar: Parallel,
        c0: int,
        mp: int,
        tp_ar_t: float,
        tp_ag_t: float,
        tp_rs_t: float,
        row_ar_t: float,
        sp_ag_e: tuple[float, float, bool],
        sp_push: tuple[float, float, bool],
    ) -> float:
        ep_ranks = mpar.ep if mpar.ep > 1 else mpar.tp
        tp = mpar.tp
        if tp <= 1 and ep_ranks <= 1:
            return 0.0
        hidden = self.geom.hidden
        bp = self.geom.block_parallel
        if self.even_pp_ratio:
            if mpar.moe_dispatch == "ag_ep" or (
                mpar.moe == "ep" and ep_ranks > tp and mpar.moe_dispatch == "allreduce"
            ):
                dp_v = mpar.dp if mpar.dp > 1 else 1
                global_rows = rows * dp_v
                per_rank_rows = (
                    (global_rows + ep_ranks - 1) // ep_ranks
                    if ep_ranks > 1
                    else global_rows
                )
                if per_rank_rows < 1:
                    per_rank_rows = 1
                s = 0.0
                if ep_ranks > 1:
                    s += (
                        _price_coll(sp_ag_e, float(global_rows * hidden))
                        + _price_coll(
                            sp_ag_e,
                            float(ep_ranks * per_rank_rows * num_experts * 2),
                        )
                        + _price_coll(sp_ag_e, float(global_rows * hidden * 2))
                    )
                if mpar.residual == "replicated" and tp > 1 and bp <= 1:
                    s += row_ar_t
                return c0 * s
            if mpar.moe_dispatch == "fused_rs":
                return c0 * (tp_rs_t + tp_ag_t) if tp > 1 else 0.0
            disp_group = ep_ranks if mpar.moe_dispatch == "a2a_push" else tp
            if mpar.residual == "replicated":
                if mpar.moe_dispatch == "allreduce":
                    return c0 * tp_ar_t if (bp <= 1 and tp > 1) else 0.0
                s = 0.0
                if disp_group > 1:
                    dp_v = mpar.dp if mpar.dp > 1 else 1
                    r_d = (rows * dp_v + disp_group - 1) // disp_group
                    push_wire = float(
                        r_d * top_k * (disp_group - 1) / disp_group * hidden
                    )
                    s += 2.0 * _price_coll(sp_push, push_wire)
                if tp > 1:
                    s += tp_ag_t
                return c0 * s
            if mpar.moe_dispatch == "allreduce":
                if tp <= 1:
                    return 0.0
                s = (
                    (tp_ag_t if bp <= 1 else 0.0)
                    + _price_coll(self.sp_ag_tp, float(mp * num_experts * 2))
                    + tp_rs_t
                )
                return c0 * s
            if disp_group > 1:
                dp_v = mpar.dp if mpar.dp > 1 else 1
                r_d = (rows * dp_v + disp_group - 1) // disp_group
                push_wire = float(r_d * top_k * (disp_group - 1) / disp_group * hidden)
                return c0 * 2.0 * _price_coll(sp_push, push_wire)
            return 0.0

        unit: list[float] = []
        if mpar.moe_dispatch == "ag_ep" or (
            mpar.moe == "ep" and ep_ranks > tp and mpar.moe_dispatch == "allreduce"
        ):
            global_rows = rows * max(1, mpar.dp)
            per_rank_rows = max(1, math.ceil(global_rows / max(1, ep_ranks)))
            if ep_ranks > 1:
                unit.append(_price_coll(sp_ag_e, float(global_rows * hidden)))
                unit.append(
                    _price_coll(
                        sp_ag_e,
                        float(ep_ranks * per_rank_rows * num_experts * 2),
                    )
                )
                unit.append(_price_coll(sp_ag_e, float(global_rows * hidden * 2)))
            if mpar.residual == "replicated" and tp > 1 and bp <= 1:
                unit.append(row_ar_t)
            return self._scale_chain_times(unit, c0)
        if mpar.moe_dispatch == "fused_rs":
            if tp > 1:
                if mpar.residual == "replicated":
                    unit.append(tp_rs_t)
                    unit.append(tp_ag_t)
                else:
                    unit.append(tp_ag_t)
                    unit.append(tp_rs_t)
            return self._scale_chain_times(unit, c0)
        disp_group = ep_ranks if mpar.moe_dispatch == "a2a_push" else tp
        push_wire = float(
            math.ceil((rows * max(1, mpar.dp)) / max(1, disp_group))
            * top_k
            * (disp_group - 1)
            / max(1, disp_group)
            * hidden
        )
        if mpar.residual == "replicated":
            if mpar.moe_dispatch == "allreduce":
                return (
                    self._scale_chain_times([tp_ar_t], c0)
                    if (bp <= 1 and tp > 1)
                    else 0.0
                )
            if disp_group > 1:
                pt = _price_coll(sp_push, push_wire)
                unit.append(pt)
                unit.append(pt)
            if tp > 1:
                unit.append(tp_ag_t)
        elif mpar.moe_dispatch == "allreduce":
            if tp <= 1:
                return 0.0
            if bp <= 1:
                unit.append(tp_ag_t)
            unit.append(_price_coll(self.sp_ag_tp, float(mp * num_experts * 2)))
            unit.append(tp_rs_t)
        elif disp_group > 1:
            pt = _price_coll(sp_push, push_wire)
            unit.append(pt)
            unit.append(pt)
        return self._scale_chain_times(unit, c0)

    def eval_pass(
        self,
        batch: int,
        rows_per_seq: int,
        context: int,
        state_writes: int,
        prefill_tokens: int,
        prefix_hit_rate: float,
        ep_load_tokens: int = 0,
    ) -> float:
        geom = self.geom
        par = self.par
        tp = self.tp
        decode_rows = batch * rows_per_seq
        eff_prefill = (
            max(
                1,
                math.ceil(prefill_tokens * max(0.0, 1.0 - prefix_hit_rate) / self.pcp),
            )
            if prefill_tokens > 0
            else 0
        )
        cached_prefix = (
            int(round(prefill_tokens * prefix_hit_rate)) if prefill_tokens > 0 else 0
        )
        rows = decode_rows + eff_prefill

        if ep_load_tokens > 0 and self.ep_z > 0.0:
            r_ep = 1.0 + self.ep_z * math.sqrt(
                self.ep_m1_f / float(ep_load_tokens * geom.top_k)
            )
            ep_mul = self.ep_cap_f if r_ep > self.ep_cap_f else r_ep
        else:
            ep_mul = 1.0

        mp = (
            rows
            if par.residual == "replicated"
            else ((rows + self.tp_eff - 1) // self.tp_eff) * self.tp_eff
        )
        if tp > 1:
            s_bytes = float(mp * geom.hidden * 2)
            if par.residual == "replicated":
                tp_ar_t = _price_coll(self.sp_ar_tp, s_bytes)
                row_ar_t = tp_ar_t
            else:
                tp_ar_t = 0.0
                row_ar_t = _price_coll(self.sp_ar_tp, float(rows * geom.hidden * 2))
            tp_ag_t = _price_coll(self.sp_ag_tp, s_bytes)
            tp_rs_t = (
                _price_coll(self.sp_rs_tp, s_bytes)
                if (par.residual == "sp" or par.moe_dispatch == "fused_rs")
                else 0.0
            )
            chain_s = row_ar_t if geom.hidden > 0 else 0.0
        else:
            tp_ar_t = 0.0
            row_ar_t = 0.0
            tp_ag_t = 0.0
            tp_rs_t = 0.0
            chain_s = 0.0

        mod_s = 0.0
        even_pp = self.even_pp_ratio
        if even_pp and tp > 1 and self.sum_attn_c0 > 0:
            if par.residual == "replicated":
                chain_s += self.sum_attn_c0 * tp_ar_t
            else:
                chain_s += self.sum_attn_c0 * (
                    tp_ag_t + (tp_rs_t if geom.block_parallel <= 1 else 0.0)
                )

        if self.has_gqa:
            kv_tok = float(batch * context) * self.gqa_dec_kv_factor + float(
                cached_prefix + eff_prefill
            )
            hs = self.gqa_hb_base + self.gqa_kv_hs_unit * kv_tok
            ms = rows * self.gqa_row_ms + self.gqa_score_ms_unit * (
                decode_rows * context
                + eff_prefill * (cached_prefix + 0.5 * eff_prefill * self.pcp)
            )
            mod_s += hs if hs >= ms else ms
            if not even_pp:
                chain_s += self._attn_chain_s(
                    batch,
                    rows_per_seq,
                    self.gqa_nq,
                    self.gqa_hd,
                    True,
                    eff_prefill,
                    self.gqa_c0,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                )
            elif self.need_full_extra:
                chain_s += self._full_attn_extra_s(
                    batch,
                    rows_per_seq,
                    self.gqa_nq,
                    self.gqa_hd,
                    eff_prefill,
                    self.gqa_c0,
                )

        if self.has_swa:
            sw_cap = self.swa_cap if self.swa_cap > 0 else context
            eff_ctx = min(context, sw_cap) if context > 0 else 0
            kv_tok = float(batch * eff_ctx) * self.swa_dec_kv_factor + float(
                min(cached_prefix + eff_prefill, sw_cap) if eff_prefill > 0 else 0
            )
            hs = self.swa_hb_base + self.swa_kv_hs_unit * kv_tok
            ms = rows * self.swa_row_ms + self.swa_score_ms_unit * (
                decode_rows * eff_ctx + eff_prefill * min(sw_cap, max(1, eff_prefill))
            )
            mod_s += hs if hs >= ms else ms
            if not even_pp:
                chain_s += self._attn_chain_s(
                    batch,
                    rows_per_seq,
                    self.swa_nq,
                    self.swa_hd,
                    True,
                    eff_prefill,
                    self.swa_c0,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                )
            elif self.need_full_extra:
                chain_s += self._full_attn_extra_s(
                    batch,
                    rows_per_seq,
                    self.swa_nq,
                    self.swa_hd,
                    eff_prefill,
                    self.swa_c0,
                )

        for (
            is_sparse,
            hb_base,
            row_ms,
            dec_kv_fac,
            kv_hs_u,
            idx_hs_u,
            score_ms_u,
            c0_m,
            m_nq,
            m_kvlora,
        ) in self.mla_fams:
            att_ctx = (
                min(context, geom.index_topk)
                if (is_sparse and geom.index_topk > 0 and context > 0)
                else context
            )
            kv_tok = float(batch * att_ctx) * dec_kv_fac + float(
                cached_prefix + eff_prefill
            )
            hs = hb_base + kv_hs_u * kv_tok
            if idx_hs_u > 0.0 and context > 0:
                hs += idx_hs_u * float(batch * context)
            ms = rows * row_ms + score_ms_u * (
                decode_rows * att_ctx
                + eff_prefill * (cached_prefix + 0.5 * eff_prefill * self.pcp)
            )
            mod_s += hs if hs >= ms else ms
            if not even_pp:
                chain_s += self._attn_chain_s(
                    batch,
                    rows_per_seq,
                    m_nq,
                    m_kvlora,
                    True,
                    eff_prefill,
                    c0_m,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                )
            elif self.need_full_extra:
                chain_s += self._full_attn_extra_s(
                    batch,
                    rows_per_seq,
                    m_nq,
                    m_kvlora,
                    eff_prefill,
                    c0_m,
                )

        if self.has_gdn:
            seq_c = batch + (1 if eff_prefill > 0 and batch == 0 else 0)
            sw_c = state_writes if batch > 0 else (1 if eff_prefill > 0 else 0)
            hs = self.gdn_hb_base + float(seq_c * (1 + sw_c)) * self.gdn_state_hs_unit
            ms = rows * self.gdn_row_ms
            mod_s += hs if hs >= ms else ms
            if not even_pp:
                chain_s += self._attn_chain_s(
                    batch,
                    rows_per_seq,
                    geom.num_heads // tp,
                    geom.head_dim,
                    False,
                    eff_prefill,
                    self.gdn_c0,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                )

        if self.has_kda:
            seq_c = batch + (1 if eff_prefill > 0 and batch == 0 else 0)
            sw_c = state_writes if batch > 0 else (1 if eff_prefill > 0 else 0)
            hs = self.kda_hb_base + float(seq_c * (1 + sw_c)) * self.kda_state_hs_unit
            ms = rows * self.kda_row_ms
            mod_s += hs if hs >= ms else ms
            if not even_pp:
                chain_s += self._attn_chain_s(
                    batch,
                    rows_per_seq,
                    self.kda_nq,
                    self.kda_vd,
                    False,
                    eff_prefill,
                    self.kda_c0,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                )

        if self.has_dense_ffn:
            hs = self.mlp_hs
            ms = rows * self.mlp_row_ms
            mod_s += hs if hs >= ms else ms
            chain_s += self._moe_chain_s(
                rows,
                0,
                0,
                self.mlp_d_par,
                self.mlp_c0,
                mp,
                tp_ar_t,
                tp_ag_t,
                tp_rs_t,
                row_ar_t,
                self.mlp_sp_ag_ep,
                self.mlp_sp_push,
            )

        self._last_moe_ep_delta = 0.0
        if self.has_moe_ffn:
            global_rows = (
                (decode_rows + eff_prefill * self.pcp_per_ep) * self.dp_per_ep
                if par.ep > 1
                else rows
            )
            if par.moe == "ep":
                ed = _fast_ep_busiest(
                    geom.num_experts, geom.top_k, global_rows, self.ep_ranks
                )
            else:
                ed = (
                    float(geom.num_experts)
                    if geom.top_k >= geom.num_experts
                    else float(geom.num_experts)
                    * (1.0 - (1.0 - geom.top_k / geom.num_experts) ** rows)
                )
            hs = ed * self.routed_hs_unit + self.moe_fixed_hs
            routed_rows = global_rows if par.ep > 1 else rows
            sh_rows = math.ceil(rows / tp) if par.residual == "sp" else rows
            r_ms = routed_rows * self.routed_row_ms
            if self.mxu_m_tile > 1 and routed_rows > 0 and ed > 0.0 and geom.top_k > 0:
                ed_global = (
                    (
                        float(geom.num_experts)
                        if geom.top_k >= geom.num_experts
                        else float(geom.num_experts)
                        * (1.0 - (1.0 - geom.top_k / geom.num_experts) ** routed_rows)
                    )
                    if par.moe == "ep"
                    else ed
                )
                m_per_t = float(routed_rows * geom.top_k) / (
                    ed_global if ed_global >= 1.0 else 1.0
                )
                if m_per_t > 0.0:
                    m_pad = float(
                        math.ceil(m_per_t / self.mxu_m_tile) * self.mxu_m_tile
                    )
                    r_ms *= m_pad / m_per_t
            ms_base = r_ms + sh_rows * self.shared_row_ms + rows * self.router_row_ms
            ms = ms_base * ep_mul
            m_ep = hs if hs >= ms else ms
            if ep_mul != 1.0:
                m_raw = hs if hs >= ms_base else ms_base
                self._last_moe_ep_delta = m_ep - m_raw
            mod_s += m_ep
            if (
                even_pp
                and par.residual == "replicated"
                and par.moe_dispatch == "allreduce"
                and not (par.moe == "ep" and self.ep_ranks > tp)
            ):
                if geom.block_parallel <= 1 and tp > 1:
                    chain_s += self.moe_c0 * tp_ar_t
            else:
                chain_s += self._moe_chain_s(
                    rows,
                    geom.num_experts,
                    geom.top_k,
                    par,
                    self.moe_c0,
                    mp,
                    tp_ar_t,
                    tp_ag_t,
                    tp_rs_t,
                    row_ar_t,
                    self.sp_ag_ep,
                    self.sp_push_disp,
                )

        if self.has_ple:
            hs = self.ple_hb_base + self.ple_lookup_hs * float(rows)
            ms = rows * self.ple_row_ms
            mod_s += hs if hs >= ms else ms

        if par.pp > 1:
            chain_s += _price_coll(self.sp_p2p_pp, float(rows * geom.hidden * 2))
        if tp > 1:
            if par.residual == "sp" and geom.hidden > 0:
                chain_s += tp_ag_t
            if geom.vocab > 0:
                chain_s += _price_coll(self.sp_ag_tp, float(rows * geom.vocab * 2))
        lm_hs = self.lm_hs
        lm_ms = rows * self.lm_row_ms
        mod_s += lm_hs if lm_hs >= lm_ms else lm_ms
        return mod_s + chain_s

    def _get_draft_ev(self) -> _FastPassEvaluator:
        if self._draft_ev is None:
            self._draft_ev = _FastPassEvaluator(
                _make_draft_geom(self.geom),
                self.dt,
                self.chip,
                self.topology,
                self.par,
            )
        return self._draft_ev

    def eval_dec(self, batch: int, context: int, spec_k: int) -> tuple[float, float]:
        ep_tok = batch * self.dp * max(1, spec_k + 1)
        if spec_k <= 0:
            return (
                self.eval_pass(batch, 1, context, 1, 0, 0.0, ep_load_tokens=ep_tok),
                0.0,
            )
        ver_ep = self.eval_pass(
            batch, spec_k + 1, context, 1, 0, 0.0, ep_load_tokens=ep_tok
        )
        ver_raw = ver_ep - self._last_moe_ep_delta
        draft_one = self._get_draft_ev().eval_pass(
            batch, 1, context, 1, 0, 0.0, ep_load_tokens=ep_tok
        )
        fc_hs = self.fc_hs_unit * spec_k
        fc_ms = batch * self.fc_row_ms_unit * spec_k
        fc_mod_s = fc_hs if fc_hs >= fc_ms else fc_ms
        fc_chain_s = (
            spec_k * _price_coll(self.sp_ag_tp, float(batch * self.geom.hidden * 2))
            if self.tp > 1
            else 0.0
        )
        sf_dec_s = ver_ep + spec_k * draft_one + fc_mod_s + fc_chain_s
        dr_s = sf_dec_s - ver_raw
        return (sf_dec_s, dr_s if dr_s > 0.0 else 0.0)


def sweep_prefix_hit_crossovers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    plan: ServingPlan,
    *,
    hit_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
) -> tuple[tuple[float, float] | None, float | None]:
    """Sweep candidate HBM-fit pools across prefix hit rates in under 50 ms."""
    if plan.recommended is None:
        return (None, None)
    raw_cands = (
        plan.sweep_candidates
        if plan.sweep_candidates
        else (plan.candidates if plan.candidates else plan.frontier)
    )
    if not raw_cands:
        return (None, None)
    scen = plan.scenario
    cands = (
        raw_cands
        if plan.sweep_candidates
        else _prune_sweep_candidates(geom, dtypes, chip, scen, raw_cands)
    )
    eff_al = max(1.0, scen.accept_len)
    spec_k = (
        scen.spec_tokens
        if scen.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    dec_steps = max(1.0, scen.osl / eff_al)
    ctx_full = scen.isl + scen.osl
    ctx_mid = scen.isl + scen.osl // 2
    seq_tok = float(ctx_full)
    pcie_core_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
    has_pcie = pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
    has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
    grid = tuple(
        sorted(
            {*(round(float(h), 4) for h in hit_rates), round(scen.prefix_hit_rate, 4)}
        )
    )
    prefill_chunk = max(1, scen.prefill_chunk)
    isl_f = float(scen.isl)
    eff_al_s = eff_al * 1e-3
    inv_al_ms = 1e3 / eff_al
    think_s = scen.think_time_s
    has_think = think_s > 0.0
    ttft_slo = scen.ttft_slo_ms
    tpot_slo = scen.tpot_slo_ms
    need_ttft = has_think or (ttft_slo is not None)
    need_wire = has_dcn and need_ttft
    w_turns = max(0.0, scen.kv_working_set_turns)
    has_w_turns = w_turns > 0.0
    has_offload = scen.kv_offload_gb > 0.0
    offload_bytes = float(scen.kv_offload_gb) * float(1 << 30)
    _ceil = math.ceil

    pool_ids: dict[tuple, int] = {}
    pool_evs: list[_FastPassEvaluator] = []
    pre_cache: dict[tuple[int, float], float] = {}
    pre_step_cache: dict[tuple[int, int, int, int], float] = {}
    dec_cache: dict[tuple[int, int], tuple[float, float]] = {}
    mix_cache: dict[tuple[int, int, int], float] = {}
    kv_b_cache: dict[tuple[int, int, int, int], float] = {}
    sw_w = geom.sliding_window
    if need_wire:
        par_w = Parallel(tp=1, pp=1)
        bd_w0 = hbm_breakdown_per_core_bytes(
            geom, dtypes, par_w, 1, 0, physical_kv_layout=False
        )
        bd_w1 = hbm_breakdown_per_core_bytes(
            geom, dtypes, par_w, 1, 1, physical_kv_layout=False
        )
        wire_state_b = bd_w0.state
        wire_kv_step1 = bd_w1.kv
        if sw_w > 0:
            bd_w_sw = hbm_breakdown_per_core_bytes(
                geom, dtypes, par_w, 1, sw_w, physical_kv_layout=False
            )
            bd_w_sw1 = hbm_breakdown_per_core_bytes(
                geom, dtypes, par_w, 1, sw_w + 1, physical_kv_layout=False
            )
            wire_kv_at_sw = bd_w_sw.kv
            wire_kv_post_sw = bd_w_sw1.kv - bd_w_sw.kv
        else:
            wire_kv_at_sw = 0.0
            wire_kv_post_sw = 0.0
    else:
        wire_state_b = 0.0
        wire_kv_step1 = 0.0
        wire_kv_at_sw = 0.0
        wire_kv_post_sw = 0.0

    def _get_wire_b(uncached_tok: int) -> float:
        tok_k = max(0, uncached_tok)
        if sw_w > 0 and tok_k > sw_w:
            kv_w = wire_kv_at_sw + float(tok_k - sw_w) * wire_kv_post_sw
        else:
            kv_w = float(tok_k) * wire_kv_step1
        return wire_state_b + kv_w

    def _get_b_kv(pool: PoolMeshPlan) -> float:
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                geom, dtypes, par_k, 1, 1
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par_k, 1, 0)
            kv_b_cache[k] = val
        return val

    def _get_pool_id(pool: PoolMeshPlan) -> int:
        pk = (
            pool.topology,
            pool.tp,
            pool.dp,
            pool.ep,
            pool.pp,
            pool.pcp,
            pool.kv_split,
            pool.moe_dispatch,
            pool.mesh_axes,
            pool.pp_layer_partition,
        )
        pid = pool_ids.get(pk)
        if pid is None:
            pid = len(pool_evs)
            pool_ids[pk] = pid
            topo, par = _pool_parallel(pool)
            even_pp = (
                pool.pp <= 1
                or not pool.pp_layer_partition
                or max(pool.pp_layer_partition)
                == max(1, _ceil(geom.num_layers / pool.pp))
            )
            pool_evs.append(
                _FastPassEvaluator(
                    geom,
                    dtypes,
                    chip,
                    topo,
                    par,
                    pp_part=None if even_pp else pool.pp_layer_partition,
                )
            )
        return pid

    def _compute_pre_sf(pid: int, pool: PoolMeshPlan, h_val: float) -> float:
        h_clamped = max(0.0, min(1.0, h_val))
        eff_tok = max(1, _ceil((1.0 - h_clamped) * scen.isl)) if scen.isl > 0 else 0
        eff_pre = (
            max(
                1,
                _ceil(scen.isl * max(0.0, 1.0 - h_clamped) / max(1, pool.pcp)),
            )
            if scen.isl > 0
            else 0
        )
        cached_pre = int(round(scen.isl * h_clamped)) if scen.isl > 0 else 0
        skey = (pid, eff_tok, eff_pre, cached_pre)
        sf_s = pre_step_cache.get(skey)
        if sf_s is None:
            sf_s = pool_evs[pid].eval_pass(
                0, 0, 0, 0, scen.isl, h_clamped, ep_load_tokens=eff_tok
            )
            pre_step_cache[skey] = sf_s
        pre_cache[(pid, h_val)] = sf_s
        return sf_s

    def _get_dec_sf(pid: int, b_res: int) -> tuple[float, float]:
        key = (pid, b_res)
        val = dec_cache.get(key)
        if val is None:
            val = pool_evs[pid].eval_dec(b_res, ctx_mid, spec_k)
            dec_cache[key] = val
        return val

    def _compute_mix_sf(pid: int, b_res: int, chunk_tok: int) -> float:
        val = pool_evs[pid].eval_pass(b_res, spec_k + 1, ctx_mid, 1, chunk_tok, 0.0)
        mix_cache[(pid, b_res, chunk_tok)] = val
        return val

    mix_meta = []
    dis_fast_groups: dict[
        tuple[int, float],
        tuple[
            PoolMeshPlan,
            list[tuple[float, float, float, bool, float, int, ParetoPoint]],
        ],
    ] = {}
    dis_ttft_groups: dict[
        tuple[int, float],
        tuple[
            PoolMeshPlan,
            float,
            list[
                tuple[
                    float,
                    float,
                    float,
                    bool,
                    float,
                    int,
                    float,
                    float,
                    float,
                    float,
                    float,
                    ParetoPoint,
                ]
            ],
        ],
    ] = {}
    for pt in cands:
        d_pool = pt.decode_pool
        d_dp = d_pool.dp if d_pool.dp > 1 else 1
        d_idp = d_pool.instances * d_dp
        if d_idp < 1:
            d_idp = 1
        b_local = 0
        if d_pool.max_num_seqs is not None:
            b_res = d_pool.max_num_seqs
        else:
            b_local = _ceil(pt.concurrency / d_idp)
            if b_local < 1:
                b_local = 1
            b_res = b_local
        cap_tok = pt.kv_cache_capacity_tokens
        if has_w_turns and cap_tok is not None:
            if b_local == 0:
                b_local = _ceil(pt.concurrency / d_idp)
                if b_local < 1:
                    b_local = 1
            w_tok = w_turns * float(b_local * ctx_full)
            cap_t = float(cap_tok)
            r_c = cap_t / (w_tok if w_tok > 1.0 else 1.0)
            cap_ratio = r_c if r_c < 1.0 else 1.0
            if has_offload:
                b_kv = _get_b_kv(d_pool)
                tp_pp = d_pool.tp * d_pool.pp
                off_tok = (
                    (offload_bytes / float(tp_pp if tp_pp > 1 else 1)) / b_kv
                    if b_kv > 0.0
                    else 0.0
                )
                hbm_cap_t = cap_t - off_tok
                if hbm_cap_t < 0.0:
                    hbm_cap_t = 0.0
                r_h = hbm_cap_t / (w_tok if w_tok > 1.0 else 1.0)
                hbm_ratio = r_h if r_h < 1.0 else 1.0
                pcie_slope = (
                    (cap_ratio - hbm_ratio) * isl_f * b_kv / pcie_core_bw
                    if (cap_ratio > hbm_ratio and b_kv > 0.0 and has_pcie)
                    else 0.0
                )
            else:
                pcie_slope = 0.0
        else:
            cap_ratio = 1.0
            pcie_slope = 0.0
        inv_chips = seq_tok / float(pt.total_chips)
        conc_f = float(pt.concurrency)
        if pt.mode == "mixed":
            d_pid = _get_pool_id(d_pool)
            if spec_k == 0:
                sf_dec_s = _get_dec_sf(d_pid, b_res)[0]
                draft_s = 0.0
            else:
                sf_dec_s, draft_s = _get_dec_sf(d_pid, b_res)
            dec_dur_s = dec_steps * sf_dec_s
            res_cap = float(b_res * d_dp)
            c_srv = conc_f if conc_f <= res_cap else res_cap
            if need_ttft:
                if b_local == 0:
                    b_local = _ceil(pt.concurrency / d_idp)
                    if b_local < 1:
                        b_local = 1
                dem_cap = float(b_local * d_dp)
                c_dem = conc_f if conc_f <= dem_cap else dem_cap
                over = max(0.0, (c_dem / max(1e-9, res_cap)) - 1.0)
            else:
                over = 0.0
            mix_meta.append(
                (
                    pt,
                    d_pool,
                    d_pid,
                    b_res,
                    float(b_res),
                    cap_ratio,
                    pcie_slope,
                    sf_dec_s,
                    dec_dur_s,
                    draft_s,
                    res_cap,
                    c_srv,
                    conc_f,
                    pt.concurrency > 1,
                    over,
                    inv_chips,
                    -pt.total_chips,
                )
            )
        else:
            p_pool = pt.prefill_pool
            assert p_pool is not None
            p_pid = _get_pool_id(p_pool)
            sf_dec_s = pt.p90_tpot_ms * eff_al_s
            dec_dur_s = dec_steps * sf_dec_s
            res_cap_d = float(b_res * d_idp)
            c_srv_d = conc_f if conc_f <= res_cap_d else res_cap_d
            rate_d = c_srv_d / (dec_dur_s if dec_dur_s > 1e-9 else 1e-9)
            tpot_d = sf_dec_s * inv_al_ms
            inter_d = 1000.0 / (tpot_d if tpot_d > 1e-6 else 1e-6)
            tpot_ok = tpot_slo is None or tpot_d <= tpot_slo
            p_inst_f = float(p_pool.instances)
            gkey = (p_pid, cap_ratio)
            if not need_ttft:
                g_fast = dis_fast_groups.get(gkey)
                if g_fast is None:
                    g_list_f: list[
                        tuple[float, float, float, bool, float, int, ParetoPoint]
                    ] = []
                    dis_fast_groups[gkey] = (p_pool, g_list_f)
                else:
                    g_list_f = g_fast[1]
                g_list_f.append(
                    (
                        p_inst_f,
                        rate_d,
                        inv_chips,
                        tpot_ok,
                        inter_d,
                        -pt.total_chips,
                        pt,
                    )
                )
            else:
                if b_local == 0:
                    b_local = _ceil(pt.concurrency / d_idp)
                    if b_local < 1:
                        b_local = 1
                dem_cap_d = float(b_local * d_idp)
                c_dem_d = conc_f if conc_f <= dem_cap_d else dem_cap_d
                over_d = max(0.0, (c_dem_d / max(1e-9, res_cap_d)) - 1.0)
                q_wait_d_ms = over_d * dec_dur_s * 1e3
                inv_min_c_dcn_ms = (
                    1e3
                    / (
                        float(
                            max(
                                1,
                                min(
                                    p_pool.chips_per_instance,
                                    d_pool.chips_per_instance,
                                ),
                            )
                        )
                        * chip.dcn_bw
                    )
                    if has_dcn
                    else 0.0
                )
                g_ttft = dis_ttft_groups.get(gkey)
                if g_ttft is None:
                    g_list_t: list[
                        tuple[
                            float,
                            float,
                            float,
                            bool,
                            float,
                            int,
                            float,
                            float,
                            float,
                            float,
                            float,
                            ParetoPoint,
                        ]
                    ] = []
                    dis_ttft_groups[gkey] = (
                        p_pool,
                        float(p_pool.pp - 1),
                        g_list_t,
                    )
                else:
                    g_list_t = g_ttft[2]
                g_list_t.append(
                    (
                        p_inst_f,
                        rate_d,
                        inv_chips,
                        tpot_ok,
                        inter_d,
                        -pt.total_chips,
                        pcie_slope,
                        inv_min_c_dcn_ms,
                        q_wait_d_ms,
                        dec_dur_s,
                        conc_f,
                        pt,
                    )
                )

    winners: list[tuple[float, tuple]] = []
    _pre_get = pre_cache.get
    _mix_get = mix_cache.get
    tol_band = 1.0 + 1e-7
    for h in grid:
        near_slo: list[tuple[float, float, int, ParetoPoint]] = []
        near_all: list[tuple[float, float, int, ParetoPoint]] = []
        max_slo = -1.0
        max_all = -1.0
        eff_isl_1 = max(1, _ceil((1.0 - h) * scen.isl))
        n_chk_1 = max(1, _ceil(eff_isl_1 / prefill_chunk))
        chunk_tok_1 = eff_isl_1 if eff_isl_1 < prefill_chunk else prefill_chunk
        wire_b_1 = _get_wire_b(eff_isl_1) if need_wire else 0.0
        for (
            pt,
            d_pool,
            d_pid,
            b_res,
            b_res_f,
            cap_ratio,
            pcie_slope,
            sf_dec_s,
            dec_dur_s,
            draft_s,
            res_cap,
            c_srv,
            conc_f,
            has_c_gt1,
            over,
            inv_chips,
            neg_chips,
        ) in mix_meta:
            if cap_ratio == 1.0 or h == 0.0:
                h_eff = h
                n_chk = n_chk_1
                chunk_tok_c = chunk_tok_1
            else:
                h_eff = h * cap_ratio
                eff_isl_h = max(1, _ceil((1.0 - h_eff) * scen.isl))
                n_chk = max(1, _ceil(eff_isl_h / prefill_chunk))
                chunk_tok_c = eff_isl_h if eff_isl_h < prefill_chunk else prefill_chunk
            pcie_s = h * pcie_slope if pcie_slope > 0.0 else 0.0
            sf_mix_s = _mix_get((d_pid, b_res, chunk_tok_c))
            if sf_mix_s is None:
                sf_mix_s = _compute_mix_sf(d_pid, b_res, chunk_tok_c)
            mix_serial_s = sf_mix_s + draft_s
            bn = b_res_f * float(n_chk)
            tot_work_s = dec_dur_s + bn * mix_serial_s
            denom_w = tot_work_s if tot_work_s > 1e-9 else 1e-9
            if need_ttft:
                sf_pre_s = _pre_get((d_pid, h_eff))
                if sf_pre_s is None:
                    sf_pre_s = _compute_pre_sf(d_pid, d_pool, h_eff)
                pre_ttft_s = sf_pre_s + pcie_s
                w_srv_final = pre_ttft_s + dec_dur_s
                if has_think:
                    srv_cap_rate = res_cap / denom_w
                    wt = w_srv_final + think_s
                    c_rate = conc_f / (wt if wt > 1e-9 else 1e-9)
                    rate_s = srv_cap_rate if srv_cap_rate <= c_rate else c_rate
                else:
                    rate_s = c_srv / denom_w
                ttft_h = (pre_ttft_s + over * w_srv_final) * 1e3 + (
                    sf_mix_s * 1e3 if has_c_gt1 else 0.0
                )
            else:
                rate_s = c_srv / denom_w
                ttft_h = 0.0
            denom_pm = dec_steps + bn
            p_mix = bn / (denom_pm if denom_pm > 1.0 else 1.0)
            if p_mix > 1.0:
                p_mix = 1.0
            p90_step_s = (1.0 - p_mix) * sf_dec_s + p_mix * mix_serial_s
            tpot_h = p90_step_s * inv_al_ms
            inter_h = 1000.0 / (tpot_h if tpot_h > 1e-6 else 1e-6)
            tput = rate_s * inv_chips
            slo_ok = (ttft_slo is None or ttft_h <= ttft_slo) and (
                tpot_slo is None or tpot_h <= tpot_slo
            )
            if slo_ok:
                if tput * tol_band >= max_slo:
                    if tput > max_slo:
                        max_slo = tput
                    near_slo.append((tput, inter_h, neg_chips, pt))
            elif not near_slo and tput * tol_band >= max_all:
                if tput > max_all:
                    max_all = tput
                near_all.append((tput, inter_h, neg_chips, pt))
        if not need_ttft:
            for (p_pid, cap_ratio), (p_pool, grp_f) in dis_fast_groups.items():
                h_eff = h if (cap_ratio == 1.0 or h == 0.0) else h * cap_ratio
                sf_pre_s = _pre_get((p_pid, h_eff))
                if sf_pre_s is None:
                    sf_pre_s = _compute_pre_sf(p_pid, p_pool, h_eff)
                denom_pre = sf_pre_s if sf_pre_s > 1e-9 else 1e-9
                for (
                    p_inst_f,
                    rate_d,
                    inv_chips,
                    tpot_ok,
                    inter_d,
                    neg_chips,
                    pt,
                ) in grp_f:
                    rate_p = p_inst_f / denom_pre
                    tput = (rate_p if rate_p <= rate_d else rate_d) * inv_chips
                    if tpot_ok:
                        if tput * tol_band >= max_slo:
                            if tput > max_slo:
                                max_slo = tput
                            near_slo.append((tput, inter_d, neg_chips, pt))
                    elif not near_slo and tput * tol_band >= max_all:
                        if tput > max_all:
                            max_all = tput
                        near_all.append((tput, inter_d, neg_chips, pt))
        else:
            for (
                (p_pid, cap_ratio),
                (p_pool, bub_num, grp_t),
            ) in dis_ttft_groups.items():
                if cap_ratio == 1.0 or h == 0.0:
                    h_eff = h
                    n_chk = n_chk_1
                    wire_b_h = wire_b_1
                else:
                    h_eff = h * cap_ratio
                    eff_isl_h = max(1, _ceil((1.0 - h_eff) * scen.isl))
                    n_chk = max(1, _ceil(eff_isl_h / prefill_chunk))
                    wire_b_h = _get_wire_b(eff_isl_h) if need_wire else 0.0
                sf_pre_s = _pre_get((p_pid, h_eff))
                if sf_pre_s is None:
                    sf_pre_s = _compute_pre_sf(p_pid, p_pool, h_eff)
                denom_pre = sf_pre_s if sf_pre_s > 1e-9 else 1e-9
                bub = 1.0 + bub_num / float(n_chk) if bub_num > 0.0 else 1.0
                pre_bub_s = sf_pre_s * bub
                pre_bub_ms = pre_bub_s * 1e3
                for (
                    p_inst_f,
                    rate_d,
                    inv_chips,
                    tpot_ok,
                    inter_d,
                    neg_chips,
                    pcie_slope,
                    inv_min_c_dcn_ms,
                    q_wait_d_ms,
                    dec_dur_s,
                    conc_f,
                    pt,
                ) in grp_t:
                    rate_p = p_inst_f / denom_pre
                    coupled_cap = rate_p if rate_p <= rate_d else rate_d
                    pcie_s = h * pcie_slope if pcie_slope > 0.0 else 0.0
                    xfer_ms = wire_b_h * inv_min_c_dcn_ms
                    ttft_h = pre_bub_ms + pcie_s * 1e3 + xfer_ms + q_wait_d_ms
                    if has_think:
                        wt_d = pre_bub_s + pcie_s + xfer_ms * 1e-3 + dec_dur_s + think_s
                        c_rate_d = conc_f / (wt_d if wt_d > 1e-9 else 1e-9)
                        coupled = coupled_cap if coupled_cap <= c_rate_d else c_rate_d
                    else:
                        coupled = coupled_cap
                    tput = coupled * inv_chips
                    slo_ok = tpot_ok and (ttft_slo is None or ttft_h <= ttft_slo)
                    if slo_ok:
                        if tput * tol_band >= max_slo:
                            if tput > max_slo:
                                max_slo = tput
                            near_slo.append((tput, inter_d, neg_chips, pt))
                    elif not near_slo and tput * tol_band >= max_all:
                        if tput > max_all:
                            max_all = tput
                        near_all.append((tput, inter_d, neg_chips, pt))
        if near_slo:
            pool_h = [x for x in near_slo if x[0] * tol_band >= max_slo]
        else:
            pool_h = [x for x in near_all if x[0] * tol_band >= max_all]
        if pool_h:
            best_pt = _best_frontier_winner(pool_h)
            winners.append((h, _layout_sig(best_pt)))
    if not winners:
        return (None, None)
    h0 = round(scen.prefix_hit_rate, 4)
    idx0 = min(range(len(winners)), key=lambda i: abs(winners[i][0] - h0))
    sig0 = _layout_sig(plan.recommended)
    winners[idx0] = (winners[idx0][0], sig0)
    lo = idx0
    while lo > 0 and winners[lo - 1][1] == sig0:
        lo -= 1
    hi = idx0
    while hi + 1 < len(winners) and winners[hi + 1][1] == sig0:
        hi += 1
    win_range = (winners[lo][0], winners[hi][0])
    cross: float | None = None
    best_dist = math.inf
    for i in range(len(winners) - 1):
        if winners[i][1] != winners[i + 1][1]:
            mid = round(0.5 * (winners[i][0] + winners[i + 1][0]), 4)
            dist = abs(mid - h0)
            if dist < best_dist:
                best_dist = dist
                cross = mid
    return (win_range, cross)
