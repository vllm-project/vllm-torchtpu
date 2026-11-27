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
"""Centralized constants and hardware/model lookup tables for advisor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

from vllm_torchtpu.megakernel.core import errors

from ..core import exec_spec as core_exec_spec
from ..core.dma import vmem as core_vmem
from ..core.profiles import default as core_default_profiles
from ..core.runtime import constants as core_rt_constants


class UnsupportedChipError(errors.MegakernelConfigError, ValueError):
    """Raised when a requested chip or topology is outside supported hardware."""


# Served v7x launch recipes reserve 8% HBM for non-KV runtime allocations.
HBM_UTILIZATION_CAP: float = 0.92
DEFAULT_COLLECTIVE_LATENCY_US_BRACKET: tuple[float, float] = (1.0, 5.0)
CALIBRATION_DIR = Path(__file__).resolve().parent / "data" / "calibration"
PRESETS_DIR = Path(__file__).resolve().parent / "data" / "presets"
MEASUREMENTS_SCHEMA = "advisor.measurements.v1"

FP8_DTYPES: frozenset[str] = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2"})
# ASSUME: FP32 matmul peak = 0.5x BF16 peak; replace with a measured FP32 matmul rate.
FP32_TO_BF16_FLOPS_RATIO: float = 0.5
DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e4m3fn": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "int4_per_token_head": 0.5,
    "turboquant_4bit_nc": 0.5,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}
KV_ONLY_DTYPES: frozenset[str] = frozenset(
    {"int4_per_token_head", "turboquant_4bit_nc"}
)
KV_DTYPE_ALIASES: Mapping[str, str] = {
    "auto": "bf16",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "fp8_e4m3": "fp8",
    "fp8_e4m3fn": "fp8",
}


def canonical_kv_dtype(kv_dtype: str) -> str:
    """Return the canonical normalized KV cache dtype identifier."""
    normalized = kv_dtype.strip().lower()
    return KV_DTYPE_ALIASES.get(normalized, normalized)


EFFICIENCY_BRACKETS: Mapping[str, tuple[float, float]] = {
    "eta_hbm": (0.70, 0.90),
    "eta_dense": (0.60, 0.85),
    "eta_moe_grouped": (0.35, 0.70),
    "eta_ici": (0.60, 0.90),
}

MEGACORE_CORES = 2
USABLE_VMEM_RESERVE_BYTES = 2 << 20

TPU7X_TOPOLOGIES: Mapping[int, str] = {
    8: "2x2x1 (1 host, 4 chips, 8 cores)",
    16: "2x2x2 (2 hosts, 8 chips, 16 cores)",
    32: "2x2x4 (4 hosts, 16 chips, 32 cores)",
    64: "2x4x4 (8 hosts, 32 chips, 64 cores)",
    128: "4x4x4 (16 hosts, 64 chips, 128 cores)",
}
TPU6E_TOPOLOGIES: Mapping[int, str] = {
    1: "1x1 (1 host, 1 chip, 1 core)",
    4: "2x2 (1 host, 4 chips, 4 cores)",
    8: "2x4 (2 hosts, 8 chips, 8 cores)",
    16: "4x4 (4 hosts, 16 chips, 16 cores)",
    32: "4x8 (8 hosts, 32 chips, 32 cores)",
    64: "8x8 (16 hosts, 64 chips, 64 cores)",
}
CHIP_ALIASES: Mapping[str, str] = {
    "7x": "tpu7x",
    "v7x": "tpu7x",
    "tpu7x": "tpu7x",
    "6e": "v6e",
    "v6e": "v6e",
    "tpu6e": "v6e",
}

SHARED_DEFAULT_CHIP_SOURCES: Mapping[str, str] = {
    "collective_latency_s": (
        "ASSUMED: [1.0, 5.0] us/hop floor prior; calibrate via trace"
    ),
    "collective_latency_us_bracket": (
        "ASSUMED: [1.0, 5.0] us/hop bracket; calibrate via trace"
    ),
    "dcn_bw_gbps": "ASSUMED: 100 Gbps/chip nominal host DCN egress prior",
    "dcn_bw_gbps_bracket": "ASSUMED: [50.0, 200.0] Gbps/chip host DCN prior",
    "pcie_h2d_bw_bytes_per_s": (
        "ASSUMED: PCIe Gen5 x16 unidirectional host-to-device bandwidth "
        "64 GB/s (bracket 32-64 GB/s)"
    ),
    "pcie_h2d_bw_bracket_bytes_per_s": (
        "ASSUMED: [32.0, 64.0] GB/s PCIe Gen5 x16 host-to-device bracket"
    ),
}
SCALING_BOOK_VECTOR_URL = (
    "ASSUMED: pre-v7x baseline from How To Scale Your Model, TPU chapter "
    "(jax-ml.github.io/scaling-book/tpus), vector unit section"
)
DEFAULT_SOURCES_7X: Mapping[str, str] = {
    **SHARED_DEFAULT_CHIP_SOURCES,
    "vmem_usable_bytes": "CHOSEN: 64 MiB physical minus 2 MiB reserve (62 MiB/core)",
    "vmem_fit_headroom_bytes": (
        "FACT: megakernel/core/dma/vmem.py FIT_VMEM_HEADROOM_MIB=6 (56 MiB fit budget)"
    ),
    "ici_egress_bw_bytes_per_s": "ASSUMED: 300 GB/s/core unidirectional ICI egress",
    "ici_link_bw_bytes_per_s": "ASSUMED: 100 GB/s/link unidirectional ICI link",
    "min_slice_cores": "FACT: smallest TPU v7x 2x2x1 cube has 4 chips (8 cores)",
    "vector_register_bytes": f"{SCALING_BOOK_VECTOR_URL}: 8 x 128 words x 4 bytes",
    "vector_loads_per_cycle": f"{SCALING_BOOK_VECTOR_URL}: 3 loads per cycle",
    "clock_hz": f"{SCALING_BOOK_VECTOR_URL}: clock of about 2.2 GHz",
}
DEFAULT_SOURCES_PUBLIC: Mapping[str, str] = {
    **SHARED_DEFAULT_CHIP_SOURCES,
    "vmem_usable_bytes": "CHOSEN: physical VMEM capacity minus 2 MiB reserve",
    "vmem_fit_headroom_bytes": "CHOSEN: 0 (no core-planner headroom configured)",
    "ici_egress_bw_bytes_per_s": "ASSUMED: Cloud TPU unidirectional ICI egress prior",
    "ici_link_bw_bytes_per_s": "ASSUMED: Cloud TPU unidirectional per-link ICI prior",
    "min_slice_cores": "FACT: minimum slice core count from Cloud TPU topology docs",
    **dict.fromkeys(
        ("vector_register_bytes", "vector_loads_per_cycle", "clock_hz"),
        "CHOSEN: 0 (unset); no public vector register figure used for this chip",
    ),
}

MIN_INTERVAL = 2
BITS_FP8 = 8
MIN_HEAD_DIM = 8
MULTIMODAL_KEYS: tuple[str, ...] = ("vision_config",)
VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "attention",
        "csa",
        "dense_mlp",
        "dsa",
        "dsa_attention",
        "full_attention",
        "gated_deltanet",
        "gdn",
        "gqa",
        "indexed_attention",
        "kda",
        "kda_attention",
        "linear_attention",
        "mla",
        "mla_attention",
        "mlp",
        "moe",
        "msa",
        "qsa",
    }
)
KDA_MODEL_TYPES: frozenset[str] = frozenset({"kimi_linear"})
# Checkpoint/serving configs storing FP32 recurrent state (see
# gdn_pool_layout.DEFAULT_POOLED_GDN_SSM_STATE_DTYPE = torch.float32).
FP32_RECURRENT_STATE_MODEL_TYPES: frozenset[str] = KDA_MODEL_TYPES | frozenset(
    {"qwen4_exp"}
)
# CSA model types deriving ced_boundary = hf_config.num_hidden_layers // 2 in
# model code (models/vllm/deepseek_v41/v2/model.py) when omitted from config.
CED_HALF_DEPTH_MODEL_TYPES: frozenset[str] = frozenset({"deepseek_v41"})
SUPPORTED_ONCHIP_FAMILIES: frozenset[str] = frozenset(
    {
        "gqa",
        "mla",
        "dsa",
        "qsa",
        "csa",
        "msa",
        "gdn",
        "kda",
        "dense_mlp",
        "moe",
        "hyper_connection",
        "mhc",
    }
)
# Layer families that megakernel/core ships a module package for.
CORE_MODULE_FAMILIES: frozenset[str] = frozenset({"gqa", "gdn", "moe"})
# Weight dtypes the megakernel/core kernels read.
CORE_SUPPORTED_DTYPES: frozenset[str] = frozenset({"fp8", "bf16"})
# Entries of `ModelProfile.layer_families` that describe stack structure.
STRUCTURAL_LAYER_TAGS: frozenset[str] = frozenset({"dense_prefix_moe", "fusion_break"})

# Sparse attention variants:
# - qsa: Query-Selected Sparse Attention
# - csa: Compressed Sparse Attention
# - msa: Multi-Scale Sparse Attention
SparseAttnKind = Literal["qsa", "csa", "msa"]
SparseCacheReplication = Literal["kv_head_group", "per_rank", "tp_sharded"]
# Hyper-connection variants: lowrank or Manifold-Constrained Hyper-Connections (mhc).
HyperConnectionKind = Literal["lowrank", "mhc"]
HostTableShardAxis = Literal["tp", "dp_local", "tp_dp_local"]
DcpSpan = Literal["kv_heads", "tp_group"]
# Execution modes: full decoder pass (`full`) or Chunked Encoder-Decoder (`ced`).
ExecutionMode = Literal["full", "ced"]
FusionScope = Literal["per_op", "single_layer", "layer_group", "whole_step"]
MULTI_LAYER_FUSED_SCOPES: frozenset[FusionScope] = frozenset(
    {"whole_step", "layer_group"}
)


CORE_DEFAULT_TILES: core_exec_spec.Tiles = core_exec_spec.Tiles()


@dataclass(frozen=True)
class CoreLimits:
    """Live execution limits sourced from `megakernel/core`.

    `max_ar_slots_per_layer` mirrors the ring bound in `core/planner.py`
    (verified by `spec_bridge.verify_core_limits_sync()`).
    """

    max_batch_rows: int = core_exec_spec.MAX_BATCH_ROWS
    verified_ranks: tuple[int, ...] = (1, 2, 4, 8, 16)
    chip_kinds: tuple[str, ...] = tuple(core_default_profiles.CHIP_BUDGETS.keys())
    max_ar_slots_per_layer: int = 2
    dense_ring_slots: int = CORE_DEFAULT_TILES.nbd
    expert_ring_slots: int = CORE_DEFAULT_TILES.nbe
    dense_slot_bytes: int = CORE_DEFAULT_TILES.bank
    expert_slot_bytes: int = CORE_DEFAULT_TILES.ebank
    vmem_mib: int = CORE_DEFAULT_TILES.vmem_mib
    smem_mib: int = CORE_DEFAULT_TILES.smem_mib
    wide_rows: int = CORE_DEFAULT_TILES.wide_rows
    default_fit_max_rows: int = core_vmem.DEFAULT_FIT_MAX_ROWS
    fit_vmem_headroom_mib: int = core_vmem.FIT_VMEM_HEADROOM_MIB
    default_decode_bucket_top: int = max(core_rt_constants.DEFAULT_BUCKETS)
    default_verify_bucket_top: int = max(core_rt_constants.DEFAULT_VERIFY_BUCKETS)


CORE_LIMITS = CoreLimits()
MIN_VMEM_MIB = 2
VALID_DMA_BUFFERS = (2, 3)

GainBasis = Literal["calibrated_whole_server_e2e", "fused_scope_device_estimate"]
GAIN_BASIS_CALIBRATED_E2E: GainBasis = "calibrated_whole_server_e2e"
GAIN_BASIS_DEVICE_ONLY: GainBasis = "fused_scope_device_estimate"

MIN_COLLECTIVE_GROUP_SIZE = 2
COLLECTIVE_PATH_XLA = "xla"
VALID_COLLECTIVE_PATHS: frozenset[str] = frozenset({COLLECTIVE_PATH_XLA, "in_kernel"})
PUBLIC_MEASUREMENT_ALIASES: frozenset[str] = frozenset({"bundled", "auto"})

# Sparse attention (DSA) and lightning indexer packing alignment constants.
INDEXER_SCALE_BLOCK_SIZE: int = 128
WORD_ALIGN_BYTES: int = 4

CHIP_KIND_MAP: Mapping[str, str] = {"tpu7x": "TPU7x"}
# core/pool/layout.py:PoolLayout enforces float8_e4m3fn for KV pages.
CORE_KV_DTYPE_ALIASES: Mapping[str, str] = {
    "fp8": "float8_e4m3fn",
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e4m3fn": "float8_e4m3fn",
}
CORE_SUPPORTED_KV_DTYPES: frozenset[str] = frozenset(CORE_KV_DTYPE_ALIASES)
# core/pool/layout.py and core/modules/gdn/decls.py store recurrent state in BF16.
CORE_STATE_DTYPE_ALIASES: Mapping[str, str] = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}
CORE_SUPPORTED_STATE_DTYPES: frozenset[str] = frozenset(CORE_STATE_DTYPE_ALIASES)
# Match measured all-reduce latency within 1 octave [0.5x, 2.0x] of analytical bytes.
AR_BYTE_MATCH_MIN_RATIO: float = 0.5
AR_BYTE_MATCH_MAX_RATIO: float = 2.0
# Recurrent step reads and writes state once each (2x resident bytes).
RECURRENT_STATE_RW_PASSES: float = 2.0
# XLA per-op collectives stage one HBM read and one HBM write around ICI transfer.
XLA_COLLECTIVE_HBM_STAGING_PASSES: float = 2.0
DEFAULT_USABLE_VMEM_MIB = CORE_LIMITS.vmem_mib
# Minimum 256 KiB DMA slot floor and 1/4-layer weight slice for pipeline fill/drain.
MIN_FIRST_DMA_SLOT_BYTES = 1 << 18
FIRST_SLOT_LAYER_WEIGHT_DIVISOR = 4
DOUBLE_BUFFER_SLOTS = 2
GROUPED_MOE_MODES: frozenset[str] = frozenset({"grouped_on_chip"})
ARENA_ALIGN_RESERVE_BYTES = 2 << 20
ATTN_TILE_TOKENS = 128
GDN_STREAMED_HEADS_CAP = 8
DENSE_FFN_TILE_CAP = 1024
BF16_BYTES = 2
FP32_BYTES = 4
LSE_SCALARS_PER_HEAD = 1
SUBCHANNEL_BLOCK_K_THRESHOLD = 512
SUBCHANNEL_VPU_OPS_PER_BLOCK = 2.0
# Elements per 32-bit vector word: 4 at the widest packing of 8-bit elements, and 8
# when unpacking weights stored below one byte (4-bit elements).
VECTOR_MAX_ELEMENTS_PER_WORD = 4
VECTOR_SUB_BYTE_ELEMENTS_PER_WORD = 8
# Per-element vector op counts:
# - RMSNorm + residual: square, sum-reduce/rsqrt, mul, gamma, add (~8 ops).
# - SwiGLU activation: neg, exp, add-1, div/silu, up-proj mul (~4 ops).
# - Sub-byte unpack: shift and mask/sign-extend per weight element (2 ops).
RMSNORM_RESIDUAL_VEC_OPS_PER_ELEM = 8.0
SWIGLU_ACT_VEC_OPS_PER_ELEM = 4.0
SUB_BYTE_UNPACK_VEC_OPS_PER_ELEM = 2.0
SUB_BYTE_EXPERT_DTYPES: frozenset[str] = frozenset({"nvfp4", "mxfp4", "fp4", "int4"})
CORES_PER_HOST = 8
DEFAULT_LAUNCH_GAP_FLOOR_US = 1.0
DEFAULT_LAUNCH_GAP_VAL_US = 2.0
DEFAULT_LAUNCH_GAP_UPPER_US = 8.0
COMPILED_XLA_GAP_P50_US = 1.0
COMPILED_XLA_GAP_P90_US = 1.5
UNFUSED_EAGER_GAP_US = 6.0
PER_OP_COLLECTIVE_CALL_FACTOR = 2
BOUNDARY_SYNC_HOP_FACTOR = 2.0
SUBLAYER_BOUNDARIES_PER_LAYER = 2
DCP_COLLECTIVES_PER_ATTN_LAYER = 2
QUANTILE_DECIMALS = 4
CONSERVED_GAP_DECIMALS = 10
PS_PER_US = 1_000_000.0
NS_PER_US = 1_000.0
# Gaps above 500 us reflect idle pauses rather than kernel launch overhead.
MAX_INTRA_STEP_GAP_US = 500.0
MIN_STEPS_FOR_WARMUP_DROP = 3
MIN_WARMUP_DROP_STEPS = 2
MIN_STEP_OPS_FOR_EXPLICIT_NORM = 2
WARMUP_OUTLIER_RATIO = 2.0
MIN_EVENTS_FOR_WARMUP_DROP = 4
CONSERVED_GAP_UNIT = "us_per_sublayer_boundary"
STEP_COVERAGE_BASIS = "trace"
DEFAULT_PHASE = "decode"
XPLANE_DEVICE_PLANE_PREFIX = "/device:"
XPLANE_SPARSE_FILTER = "sparse"
COLLECTIVE_IGNORE_SUFFIXES: tuple[str, ...] = ("-done", ".done")
XLA_MODULE_MARKERS: tuple[str, ...] = ("xla module", "xla_module")
COLLECTIVE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("all-reduce", "all-reduce"),
    ("allreduce", "all-reduce"),
    ("all-gather", "all-gather"),
    ("allgather", "all-gather"),
    ("reduce-scatter", "reduce-scatter"),
    ("reducescatter", "reduce-scatter"),
    ("all-to-all", "all-to-all"),
    ("alltoall", "all-to-all"),
)
COLLECTIVE_PATH_IN_KERNEL = "in_kernel"
COLLECTIVE_OP_ALL_REDUCE = "all-reduce"
# 128-byte HBM burst/cacheline alignment for scattered sparse-KV row gathers.
HBM_CACHELINE_BYTES: int = 128
# 8 four-bit radix passes sort 32-bit FP32 indexer scores across 2^4 = 16 buckets.
INDEXER_RADIX_PASSES: int = 8
INDEXER_RADIX_BUCKETS: int = 16
# Per-head ReLU + weight multiply-add (2 ops/head) plus block scale/bias (2 ops/key).
INDEXER_HEAD_REDUCE_VEC_OPS: float = 2.0
INDEXER_SCALE_VEC_OPS: float = 2.0
# Per-pass 4-bit radix work: nibble extract, 16-bucket compare, prefix, scatter.
INDEXER_RADIX_PASS_VEC_OPS: float = 36.0
# Scattered KV unpack, 128-lane pad, tail mask, and sub-byte transpose ops per element.
DSA_UNPACK_MASK_VEC_OPS: float = 10.0
# Online softmax per (head, selected_key): max, sub, exp, sum, div, mask, and update.
DSA_SOFTMAX_VEC_OPS: float = 8.0
# Online output accumulator rescale + accumulation per (head, value_dim).
DSA_OUTPUT_RESCALE_VEC_OPS: float = 4.0
# Cross-DCP log-sum-exp partial merge: max, exp-weight, multiply-add, normalize.
DCP_LSE_MERGE_VEC_OPS: float = 6.0
# Unfused per_op out-of-core HBM traffic: FP32 score write+read (2 * 4 B = 8 B)
# and gathered sparse-KV buffer write+read (2x) across standalone kernel boundaries.
PER_OP_INDEXER_SCORE_ROUNDTRIP_BYTES: float = 2.0 * FP32_BYTES
PER_OP_DSA_KV_ROUNDTRIP_FACTOR: float = 2.0
FUSED_ELIGIBLE_PHASES: tuple[str, ...] = ("decode", "verify")
EQUAL_SPLIT_GROUP_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
LAYER_GROUP_PRICING_NOTE: str = (
    "Numbers price one fused kernel per group; a runtime whose groups are "
    "compiled calls of per-op kernels runs at per_op. In this analytical "
    "model, layer_group cost depends on group count rather than individual "
    "group sizes."
)
DEFAULT_CHIP = "tpu7x"
DEFAULT_PREFILL_ISL = 4096
DEFAULT_SPEC_TOKENS = 3
RECOMMEND_DEFAULT_BATCH = 16
RECOMMEND_HIGH_BATCH = 64
RECOMMEND_HIGH_CTX = 32768
SATURATION_BASE_BATCH = 256
SATURATION_BATCH_CAP = 600
INVALID_CONFIG_EXIT_CODE = 2
DEFAULT_PREFILL_OSL = 0
DEFAULT_DECODE_OSL = 1024
TP_CANDIDATE_DEGREES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_DCP_POWER_BITS = 16
DECODE_MED_REF_INDEX = 2
MIN_PROVENANCE_BAND_LEN = 2
MOE_MODE_SHORT_HINTS: Mapping[str, str] = {
    "masked_on_chip": "Routed experts stream into VMEM; token rows are masked "
    "per expert inside the fused call; no reordering",
    "grouped_on_chip": "Tokens are grouped by expert inside VMEM, then each "
    "expert runs on its rows inside the fused call",
    "grouped_in_hbm": "Tokens are sorted by expert in HBM; the MoE is its own "
    "call, so the kernel boundary falls at each MoE",
    "dense_mlp": "Dense feed-forward; no routed experts",
}
SCOPE_HINTS: Mapping[str, str] = {
    "whole_step": "one call for the whole step; [T, H] stays in VMEM. Rows past "
    "max_rows_per_call run as separate calls that stream weights again; "
    "tiling inside each layer past the VMEM ceiling is not implemented",
    "layer_group": "one call per group of N layers, tiled like whole_step",
    "single_layer": "one call per layer",
    "per_op": "one call per operator; activations hand off through HBM",
}
SVG_MIN_FONT_PX = 13
# Width of one glyph as a fraction of font size, a conservative upper bound for
# the system sans-serif stack on digits, Latin letters, and punctuation.
SVG_GLYPH_EM = 0.62
SVG_TICKS: tuple[int, ...] = (1, 8, 64, 512, 4096, 32768)
# Okabe-Ito colorblind-safe palette; each scope also has its own marker shape.
SCOPE_MARKS: Mapping[str, tuple[str, str]] = {
    "whole_step": ("#009E73", "circle"),
    "layer_group": ("#56B4E9", "diamond"),
    "single_layer": ("#E69F00", "triangle"),
    "per_op": ("#CC79A7", "square"),
}
PROVENANCE_TAGS: tuple[str, ...] = ("DERIVED", "ASSUMED", "MEASURED")
MEGAKERNEL_LABELS: Mapping[str, str] = {
    "whole_step": "WHOLE-STEP MEGAKERNEL",
    "layer_group": "LAYER-GROUP MEGAKERNEL",
}
DECISION_TABLE_COLS = 12
BYTES_PER_MIB: float = float(1 << 20)
PHASE_SORT_ORDER: Mapping[str, int] = {
    "decode": 0,
    "verify": 1,
    "mixed": 2,
    "prefill": 3,
}

# 1 folded FP32 scale word per 128-token index page -> 4 B/token overhead.
INT4_INDEX_SCALE_OVERHEAD_BYTES: int = 4
# 8 FP32 scale rows * 128 lanes / 128 tokens per page -> 32 B/stored token.
INT4_SCALE_PAGE_BYTES_PER_TOK: int = 32
# Packed 4-bit TurboQuant page header: 3 BF16 metadata scalars (value scale,
# value zero, key norm) before 128-byte aligned sublane-word pages.
TURBOQUANT_HEADER_BYTES: int = 3 * BF16_BYTES
TURBOQUANT_WORD_BYTES: int = 128
TURBOQUANT_SUBLANE_WORDS: int = 8
TURBOQUANT_DEFAULT_PAGE_SIZE: int = 64
TURBOQUANT_SUPPORTED_PAGE_SIZES: tuple[int, ...] = (16, 64)
# Sinkhorn row/column normalization vector ops per S*S entry per iteration.
SINKHORN_VEC_OPS_PER_ENTRY_ITER: int = 6
HC_SUBLAYERS_PER_LAYER: int = 2
MHC_DYNAMIC_PROJ_TERMS: int = 2
MXFP4_BLOCK_K: int = 32
ENGRAM_SCALE_BLOCK_SIZE: int = 32
