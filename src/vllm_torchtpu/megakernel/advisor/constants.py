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

    `max_ar_slots_per_layer` mirrors the ring bound in `core/planner.py`.
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
