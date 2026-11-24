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
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Mapping

from ..core import errors


class UnsupportedChipError(errors.MegakernelConfigError, ValueError):
    """Raised when a requested chip or topology is outside supported hardware."""


SUPPORTED_CHIPS: frozenset[str] = frozenset({"tpu7x", "v6e"})
DEFAULT_ROW_LIMIT: int = 80
HBM_UTILIZATION_CAP: float = 0.95
DEFAULT_COLLECTIVE_LATENCY_US_BRACKET: tuple[float, float] = (1.0, 5.0)
# Absorbs IEEE-754 float64 1-ULP reassociation noise (~3.6e-12 us at ~2e4 us)
# while staying 1e4x below the 0.01 us display rounding quantum.
FLOOR_TIE_EPSILON_US: float = 1e-6

ADVISOR_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = ADVISOR_DIR / "data" / "calibration"
PRESETS_DIR = ADVISOR_DIR / "data" / "presets"

MEASUREMENTS_SCHEMA = "advisor.measurements.v1"

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

EFFICIENCY_BRACKETS: Mapping[str, tuple[float, float]] = {
    "eta_hbm": (0.70, 0.90),
    "eta_dense": (0.60, 0.85),
    "eta_moe_grouped": (0.35, 0.70),
    "eta_ici": (0.60, 0.90),
}

MEGACORE_CORES = 2
BRACKET_LEN = 2
USABLE_VMEM_RESERVE_BYTES = 2 << 20
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
SYSFS_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")
PCI_DEVICE_TO_CHIP: Mapping[str, str] = {
    "0x006f": "v6e",
    "0x0070": "v6e",
    "0x0076": "tpu7x",
    "0x0077": "tpu7x",
    "0x0078": "tpu7x",
}

REQUIRED_CHIP_SPEC_KEYS: tuple[str, ...] = (
    "vmem_capacity_bytes",
    "hbm_capacity_bytes",
    "hbm_bw_bytes_per_s",
    "bf16_flops_per_s",
)
OPTIONAL_CHIP_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "cores_per_device",
        "vmem_usable_bytes",
        "vmem_fit_headroom_bytes",
        "fp8_flops_per_s",
        "int8_flops_per_s",
        "native_ocp_fp8",
        "vector_f32_ops_per_s",
        "scalar_mem_capacity_bytes",
        "pcie_bw_bytes_per_s",
        "torus_ndim",
        "ici_egress_bw_bytes_per_s",
        "ici_link_bw_bytes_per_s",
        "collective_latency_s",
        "collective_latency_us",
        "collective_latency_us_bracket",
        "dcn_bw_gbps",
        "dcn_bw_gbps_bracket",
        "min_slice_cores",
        "devices_per_chip",
        "supports_megacore",
        "matmul_row_tile_by_dtype",
        "topologies",
        "sources",
        "vector_registers_per_core",
        "vector_register_bytes",
        "vector_loads_per_cycle",
        "vector_stores_per_cycle",
        "vector_port_ops_per_cycle",
        "clock_hz",
    }
)
ALLOWED_CHIP_SPEC_KEYS: frozenset[str] = (
    frozenset(REQUIRED_CHIP_SPEC_KEYS) | OPTIONAL_CHIP_SPEC_KEYS
)

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

DEFAULT_SOURCES_7X: Mapping[str, str] = {
    "vmem_usable_bytes": (
        "CHOSEN: 64 MiB physical minus 2 MiB compiler reserve (62 MiB/core)"
    ),
    "vmem_fit_headroom_bytes": (
        "FACT: megakernel/core/dma/vmem.py FIT_VMEM_HEADROOM_MIB=6 "
        "(6 MiB non-scratch headroom below 62 MiB limit; 56 MiB fit budget)"
    ),
    "ici_egress_bw_bytes_per_s": (
        "ASSUMED: 300 GB/s/core unidirectional ICI egress prior"
    ),
    "ici_link_bw_bytes_per_s": ("ASSUMED: 100 GB/s/link unidirectional ICI link prior"),
    "collective_latency_s": (
        f"ASSUMED: [{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[0]:.1f}, "
        f"{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[1]:.1f}] us/hop floor prior; "
        "calibrate via trace"
    ),
    "collective_latency_us_bracket": (
        f"ASSUMED: [{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[0]:.1f}, "
        f"{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[1]:.1f}] us/hop bracket; "
        "calibrate via trace"
    ),
    "dcn_bw_gbps": "ASSUMED: 100 Gbps/chip nominal host DCN egress prior",
    "dcn_bw_gbps_bracket": (
        "ASSUMED: [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)"
    ),
    "min_slice_cores": (
        "FACT: smallest TPU v7x 2x2x1 cube topology has 4 chips (8 cores)"
    ),
    "pcie_bw_bytes_per_s": ("ASSUMED: 32 GB/s host-to-device PCIe bandwidth prior"),
    "torus_ndim": ("FACT: 3D torus interconnect topology"),
    "vector_registers_per_core": (
        "FACT: How To Scale Your Model, TPU chapter "
        "(jax-ml.github.io/scaling-book/tpus), vector unit section: "
        "64 32-bit vector registers per core"
    ),
    "vector_register_bytes": (
        "DERIVED: 8 x 128 words x 4 bytes per register; How To Scale Your "
        "Model, TPU chapter (jax-ml.github.io/scaling-book/tpus), vector unit "
        "section"
    ),
    "vector_loads_per_cycle": (
        "FACT: How To Scale Your Model, TPU chapter "
        "(jax-ml.github.io/scaling-book/tpus), vector unit section: "
        "3 register loads per cycle"
    ),
    "vector_stores_per_cycle": (
        "FACT: How To Scale Your Model, TPU chapter "
        "(jax-ml.github.io/scaling-book/tpus), vector unit section: "
        "2 register stores per cycle"
    ),
    "vector_port_ops_per_cycle": (
        "FACT: How To Scale Your Model, TPU chapter "
        "(jax-ml.github.io/scaling-book/tpus), vector unit section: "
        "at most 4 loads plus stores per cycle"
    ),
    "clock_hz": (
        "FACT: How To Scale Your Model, TPU chapter "
        "(jax-ml.github.io/scaling-book/tpus), vector unit section: "
        "clock of about 2.2 GHz"
    ),
}

DEFAULT_SOURCES_PUBLIC: Mapping[str, str] = {
    "vmem_usable_bytes": (
        "CHOSEN: physical VMEM capacity minus 2 MiB compiler reserve"
    ),
    "vmem_fit_headroom_bytes": (
        "CHOSEN: 0 (no core-planner VMEM headroom reserve configured on this chip)"
    ),
    "ici_egress_bw_bytes_per_s": (
        "ASSUMED: Cloud TPU interconnect unidirectional ICI egress prior"
    ),
    "ici_link_bw_bytes_per_s": (
        "ASSUMED: Cloud TPU interconnect unidirectional per-link ICI prior"
    ),
    "collective_latency_s": (
        f"ASSUMED: [{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[0]:.1f}, "
        f"{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[1]:.1f}] us/hop floor prior; "
        "calibrate via trace"
    ),
    "collective_latency_us_bracket": (
        f"ASSUMED: [{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[0]:.1f}, "
        f"{DEFAULT_COLLECTIVE_LATENCY_US_BRACKET[1]:.1f}] us/hop bracket; "
        "calibrate via trace"
    ),
    "dcn_bw_gbps": "ASSUMED: 100 Gbps/chip nominal host DCN egress prior",
    "dcn_bw_gbps_bracket": (
        "ASSUMED: [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)"
    ),
    "min_slice_cores": "FACT: minimum slice core count from Cloud TPU topology docs",
    "pcie_bw_bytes_per_s": ("ASSUMED: 32 GB/s nominal host-to-device PCIe bandwidth"),
    "torus_ndim": ("FACT: torus interconnect dimension from topology docs"),
    "vector_registers_per_core": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
    "vector_register_bytes": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
    "vector_loads_per_cycle": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
    "vector_stores_per_cycle": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
    "vector_port_ops_per_cycle": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
    "clock_hz": (
        "CHOSEN: 0 (unset). No public vector register figure is used for "
        "this chip, so the load cap line is omitted"
    ),
}

MIN_INTERVAL = 2
BITS_FP8 = 8
BITS_FP4 = 4
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

SparseAttnKind = Literal["qsa", "csa", "msa"]
SparseCacheReplication = Literal["kv_head_group", "per_rank", "tp_sharded"]
HyperConnectionKind = Literal["lowrank", "mhc"]
HostTableShardAxis = Literal["tp", "dp_local", "tp_dp_local"]
DcpSpan = Literal["kv_heads", "tp_group"]
ExecutionMode = Literal["full", "ced"]


@dataclass(frozen=True)
class CoreLimits:
    """Verified limits mirrored from `megakernel/core` exec, profile, and VMEM specs."""

    max_batch_rows: int = 128
    verified_ranks: tuple[int, ...] = (1, 2, 4, 8, 16)
    chip_kinds: tuple[str, ...] = ("TPU7x",)
    max_ar_slots_per_layer: int = 2
    nbd: int = 14
    nbe: int = 10
    bank: int = 2 << 20
    ebank: int = 1 << 20
    vmem_mib: int = 62
    fit_vmem_headroom_mib: int = 6
    default_decode_bucket_top: int = 32
    default_verify_bucket_top: int = 64


CORE_LIMITS = CoreLimits()
CHIP_KIND_MAP: Mapping[str, str] = {"tpu7x": "TPU7x"}

MIN_VMEM_MIB = 2
DEFAULT_USABLE_VMEM_MIB = CORE_LIMITS.vmem_mib
# Minimum 256 KiB DMA slot floor and 1/4-layer weight slice for pipeline fill/drain.
MIN_FIRST_DMA_SLOT_BYTES = 1 << 18
FIRST_SLOT_LAYER_WEIGHT_DIVISOR = 4
DOUBLE_BUFFER_SLOTS = 2
GROUPED_MOE_MODES: frozenset[str] = frozenset({"grouped_on_chip"})
# Expert layouts a deployment may declare for a megakernel call.
DECLARED_MOE_LAYOUTS: frozenset[str] = frozenset({"grouped", "masked"})
CollectiveStepKind = Literal["routed_transport", "split_kv", "dcp_gather"]
DeclaredExpertMode = Literal["tp", "ep"]
GainBasis = Literal["calibrated_whole_server_e2e", "fused_scope_device_estimate"]
DECLARED_COLLECTIVE_KINDS: frozenset[str] = frozenset(get_args(CollectiveStepKind))
DECLARED_EXPERT_MODES: frozenset[str] = frozenset(get_args(DeclaredExpertMode))
VALID_DMA_BUFFERS = (2, 3)
# Phases a megakernel declaration covers when it lists none.
DEFAULT_DECLARED_PHASES: tuple[str, ...] = ("decode", "verify")
WIDE_DENSE_RING_SLOTS = CORE_LIMITS.nbd
WIDE_EXPERT_RING_SLOTS = CORE_LIMITS.nbe
DENSE_SLOT_DEFAULT_BYTES = CORE_LIMITS.bank
EXPERT_SLOT_DEFAULT_BYTES = CORE_LIMITS.ebank
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
# Floor guarding division by near-zero efficiency ratios.
MIN_EFFICIENCY_FLOOR = 1e-3

DOT_FP8 = "fp8"
DOT_BF16 = "bf16"
AB_KNOB = "ab_step_time"
SETTLE_CMD = "Pass --measurements measurements.json (schema advisor.measurements.v1)"
DEFAULT_DECLARED_CALLS_PER_LAYER = 2
GAIN_BASIS_E2E, GAIN_BASIS_ESTIMATE = get_args(GainBasis)
HOST_HEAD_SETTLE_CMD = (
    "Pass --host-head-us per_op=<us>,fused=<us> or run advisor calibrate on "
    "an XPlane trace with XLA Modules"
)
PER_OP_STEP_SETTLE_CMD = (
    "Pass --per-op-step-us <us> or --measurements measurements.json from the "
    "served baseline image"
)
DECODING_SEQS_SETTLE_CMD = (
    "Pass --decoding-seqs <N> or --output-tok-s <rate> --mean-tpot-ms <ms>"
)
BOUNDARY_SYNC_SETTLE_CMD = (
    "Run an A/B step trace comparing per_op vs fused boundary fill/drain"
)
PER_OP_COLLECTIVE_FACTOR_SETTLE_CMD = (
    "Run an A/B step trace comparing per_op vs fused all-reduce latency"
)
CORNER_AXES: Mapping[str, int] = {
    "launch_gap_us": 0,
    "collective_latency_us": 1,
    "prefill_attn_dot_dtype": 2,
    "boundary_sync_hop_factor": 3,
    "per_op_collective_call_factor": 4,
}

DEFAULT_CHIP = "tpu7x"
DEFAULT_PREFILL_ISL = 4096
DEFAULT_SPEC_TOKENS = 3
RECOMMEND_DEFAULT_BATCH = 16
RECOMMEND_HIGH_BATCH = 64
RECOMMEND_HIGH_CTX = 32768
SATURATION_BASE_BATCH = 256
SATURATION_BATCH_CAP = 600
INVALID_CONFIG_EXIT_CODE = 2

SVG_MIN_FONT_PX = 13
# Width of one glyph as a fraction of font size, a conservative upper bound for
# the system sans-serif stack on digits, Latin letters, and punctuation.
SVG_GLYPH_EM = 0.62
SVG_TICKS: tuple[int, ...] = (1, 8, 64, 512, 4096, 32768)
# Okabe-Ito colorblind-safe palette; each scope also has its own marker shape.
SCOPE_MARKS: Mapping[str, tuple[str, str]] = {
    "whole_step": ("#009E73", "circle"),
    "layer_group": ("#56B4E9", "diamond"),
    "single_layer": ("#E69F00", "diamond"),
    "per_op": ("#CC79A7", "square"),
}
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
    "whole_step": "one call for the whole step; [T, H] stays in VMEM. Rows past a "
    "declared single-call limit run as separate calls that stream weights again; "
    "tiling inside each layer past the VMEM ceiling is not implemented",
    "layer_group": "one call per group of N layers, tiled like whole_step",
    "single_layer": "one call per layer",
    "per_op": "one call per operator; activations hand off through HBM",
}

DEFAULT_PHASE = "decode"
PS_PER_US = 1_000_000.0
NS_PER_US = 1_000.0
# Gaps above 500 us reflect idle pauses rather than kernel launch overhead.
MAX_INTRA_STEP_GAP_US = 500.0
MIN_COLLECTIVE_GROUP_SIZE = 2
MIN_STEPS_FOR_WARMUP_DROP = 3
MIN_WARMUP_DROP_STEPS = 2
DEFAULT_CALIBRATE_LAYERS = 60
SUBLAYER_BOUNDARIES_PER_LAYER = 2
# Unfused per-op operator kernel launches per layer across GQA, MLA, and GDN
# sequence mixers: 3 mixer kernels (input projection, core attention or
# recurrence kernel, output projection), 2 base FFN kernels (gate_up + down or
# router + routed_moe), and 2 shared-expert FFN kernels when present.
PER_OP_MIXER_KERNELS_PER_LAYER = 3
PER_OP_BASE_FFN_KERNELS_PER_LAYER = 2
PER_OP_SHARED_EXPERT_KERNELS_PER_LAYER = 2
MIN_STEP_OPS_FOR_EXPLICIT_NORM = 2
CONSERVED_GAP_DECIMALS = 10
QUANTILE_DECIMALS = 4
WARMUP_OUTLIER_RATIO = 2.0
MIN_EVENTS_FOR_WARMUP_DROP = 4
CONSERVED_GAP_UNIT = "us_per_sublayer_boundary"
STEP_COVERAGE_BASIS = "trace"
FUSED_ELIGIBLE_PHASES: tuple[str, ...] = ("decode", "verify")
XPLANE_DEVICE_PLANE_PREFIX = "/device:"
XPLANE_SPARSE_FILTER = "sparse"
COLLECTIVE_IGNORE_SUFFIXES: tuple[str, ...] = ("-done", ".done")
XLA_MODULE_MARKERS: tuple[str, ...] = ("xla module", "xla_module")
COLLECTIVE_PATH_XLA = "xla"
COLLECTIVE_PATH_IN_KERNEL = "in_kernel"
VALID_COLLECTIVE_PATHS: frozenset[str] = frozenset(
    {COLLECTIVE_PATH_XLA, COLLECTIVE_PATH_IN_KERNEL}
)
COLLECTIVE_OP_ALL_REDUCE = "all-reduce"
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
PUBLIC_MEASUREMENT_ALIASES: frozenset[str] = frozenset({"bundled", "auto"})

PROVENANCE_TAGS: tuple[str, ...] = ("DERIVED", "ASSUMED", "MEASURED")
BAND_PAIR_LEN = 2
MEGAKERNEL_LABELS: Mapping[str, str] = {
    "whole_step": "WHOLE-STEP MEGAKERNEL",
    "layer_group": "LAYER-GROUP MEGAKERNEL",
}
DECISION_TABLE_COLS = 12
PHASE_SORT_ORDER: Mapping[str, int] = {
    "decode": 0,
    "verify": 1,
    "mixed": 2,
    "prefill": 3,
}
VALID_MESH_AXIS_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# Prefill window sizes in tokens.
PHASE_PREFILL_TOKEN_GRID: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
PHASE_T_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PHASE_S_GRID: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    1048576,
)


EQUAL_SPLIT_GROUP_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
LAYER_GROUP_PRICING_NOTE: str = (
    "Numbers price one fused kernel per group; a runtime whose groups are "
    "compiled calls of per-op kernels runs at per_op. In this analytical "
    "model, layer_group cost depends on group count rather than individual "
    "group sizes."
)

# Sparse attention (DSA) and lightning indexer architectural constants.
# 128 FP8 key elements share one 1-byte block scale in the packed indexer cache.
INDEXER_SCALE_BLOCK_SIZE: int = 128
# 32-bit word alignment (4 B) for packed FP8 + scale indexer records and int32 indices.
WORD_ALIGN_BYTES: int = 4
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

# 128-point Sylvester Hadamard butterfly ops (7 stages * 128 elements = 896 ops/head).
HADAMARD_128_VEC_OPS_PER_HEAD: int = 896
# 1 folded FP32 scale word per 128-token index page -> 4 B/token overhead.
INT4_INDEX_SCALE_OVERHEAD_BYTES: int = 4
# 8 FP32 scale rows * 128 lanes / 128 tokens per page -> 32 B/stored token.
INT4_SCALE_PAGE_BYTES_PER_TOK: int = 32
# Packed 4-bit TurboQuant page layout parameters (6-byte header, 128-byte words).
TURBOQUANT_HEADER_BYTES: int = 6
TURBOQUANT_WORD_BYTES: int = 128
TURBOQUANT_SUBLANE_WORDS: int = 8
TURBOQUANT_DEFAULT_PAGE_SIZE: int = 64
# Sinkhorn row/column normalization vector ops per S*S entry per iteration.
SINKHORN_VEC_OPS_PER_ENTRY_ITER: int = 6
HC_SUBLAYERS_PER_LAYER: int = 2
MHC_DYNAMIC_PROJ_TERMS: int = 2
MXFP4_BLOCK_K: int = 32
PLE_PADDED_ROWS_PER_SHARD: int = 2_500_012
PLE_SHARD_DIVISOR: int = 16
ENGRAM_SCALE_BLOCK_SIZE: int = 32
