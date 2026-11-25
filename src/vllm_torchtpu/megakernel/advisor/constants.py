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
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}
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
        "dense_mlp",
        "dsa",
        "dsa_attention",
        "full_attention",
        "gated_deltanet",
        "gdn",
        "gqa",
        "kda",
        "kda_attention",
        "linear_attention",
        "mla",
        "mla_attention",
        "mlp",
        "moe",
    }
)
KDA_MODEL_TYPES: frozenset[str] = frozenset({"kimi_linear"})
SUPPORTED_ONCHIP_FAMILIES: frozenset[str] = frozenset(
    {"gqa", "mla", "dsa", "gdn", "kda", "dense_mlp", "moe"}
)
# Layer families that megakernel/core ships a module package for.
CORE_MODULE_FAMILIES: frozenset[str] = frozenset({"gqa", "gdn", "moe"})
# Weight dtypes the megakernel/core kernels read.
CORE_SUPPORTED_DTYPES: frozenset[str] = frozenset({"fp8", "bf16"})
# Entries of `ModelProfile.layer_families` that describe stack structure.
STRUCTURAL_LAYER_TAGS: frozenset[str] = frozenset({"dense_prefix_moe"})


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
DEFAULT_LAUNCH_GAP_VAL_US = 2.0
DEFAULT_LAUNCH_GAP_UPPER_US = 8.0
COMPILED_XLA_GAP_P50_US = 1.0
COMPILED_XLA_GAP_P90_US = 1.5
UNFUSED_EAGER_GAP_US = 6.0
PER_OP_COLLECTIVE_CALL_FACTOR = 2
BOUNDARY_SYNC_HOP_FACTOR = 2.0
SUBLAYER_BOUNDARIES_PER_LAYER = 2
DCP_COLLECTIVES_PER_ATTN_LAYER = 2
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
