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
"""Sourced TPU hardware specifications and JSON chip spec loader."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NotRequired, TypedDict

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import tpu_info
from vllm_torchtpu.megakernel.core import errors

from .constants import (
    CHIP_ALIASES,
    CORE_LIMITS,
    DEFAULT_COLLECTIVE_LATENCY_US_BRACKET,
    DEFAULT_SOURCES_7X,
    DEFAULT_SOURCES_PUBLIC,
    FP8_DTYPES,
    FP32_TO_BF16_FLOPS_RATIO,
    MEGACORE_CORES,
    TPU6E_TOPOLOGIES,
    TPU7X_TOPOLOGIES,
    USABLE_VMEM_RESERVE_BYTES,
    UnsupportedChipError,
)
from .specs import public_calibration_path

__all__ = [
    "BUILTIN_CHIP_SPECS",
    "TPU6E_TOPOLOGIES",
    "TPU7X_TOPOLOGIES",
    "ChipSpec",
    "UnsupportedChipError",
    "build_builtin_chip_spec",
    "public_calibration_path",
    "resolve_chip_spec",
]

if TYPE_CHECKING:
    from collections.abc import Mapping

BF16_FAMILY_DTYPES: frozenset[str] = frozenset({"bf16", "bfloat16", "fp16", "float16"})
FP32_FAMILY_DTYPES: frozenset[str] = frozenset({"fp32", "float32"})
FP4_FAMILY_DTYPES: frozenset[str] = frozenset({"nvfp4", "mxfp4", "fp4"})
INT_QUANT_DTYPES: frozenset[str] = frozenset({"int8", "int4"})
ALL_PEAK_FLOP_DTYPES: frozenset[str] = (
    FP8_DTYPES
    | FP4_FAMILY_DTYPES
    | INT_QUANT_DTYPES
    | BF16_FAMILY_DTYPES
    | FP32_FAMILY_DTYPES
)


@dataclass(frozen=True)
class ChipSpec:
    """Sourced hardware ceiling parameters for one logical TPU device."""

    name: str
    cores_per_device: int
    vmem_capacity_bytes: int
    vmem_usable_bytes: int
    hbm_capacity_bytes: int
    hbm_bw_bytes_per_s: float
    bf16_flops_per_s: float
    fp8_flops_per_s: float
    int8_flops_per_s: float
    native_ocp_fp8: bool
    ici_egress_bw_bytes_per_s: float
    ici_link_bw_bytes_per_s: float
    collective_latency_s: float
    collective_latency_us_bracket: tuple[float, float] = (1.0, 5.0)
    dcn_bw_gbps: float = 100.0
    dcn_bw_gbps_bracket: tuple[float, float] = (50.0, 200.0)
    min_slice_cores: int = 1
    devices_per_chip: int = 1
    supports_megacore: bool = False
    matmul_row_tile_by_dtype: Mapping[str, int] | None = None
    topologies: Mapping[int, str] = field(default_factory=lambda: TPU7X_TOPOLOGIES)
    sources: Mapping[str, str] | None = None
    vmem_fit_headroom_bytes: int = 0
    vector_f32_ops_per_s: float = 0.0
    vector_register_bytes: int = 0
    vector_loads_per_cycle: int = 0
    clock_hz: float = 0.0

    @property
    def vector_load_cap_bytes_per_s(self) -> float:
        """Derived maximum vector register load bandwidth in bytes per second."""
        return float(
            self.vector_loads_per_cycle * self.vector_register_bytes * self.clock_hz
        )

    def peak_flops(self, dtype: str, act_dtype: str | None = None) -> float:
        """Return peak dense matrix FLOP/s for `dtype` and `act_dtype`."""
        weight_dtype = dtype.strip().lower()
        normalized_act_dtype = (act_dtype or "").strip().lower()
        if weight_dtype not in ALL_PEAK_FLOP_DTYPES or (
            normalized_act_dtype and normalized_act_dtype not in ALL_PEAK_FLOP_DTYPES
        ):
            msg = (
                f"Unsupported dtype={dtype!r} or act_dtype={act_dtype!r} for peak_flops"
            )
            raise errors.MegakernelConfigError(msg)
        if (
            normalized_act_dtype in FP32_FAMILY_DTYPES
            or weight_dtype in FP32_FAMILY_DTYPES
        ):
            return FP32_TO_BF16_FLOPS_RATIO * self.bf16_flops_per_s
        if normalized_act_dtype in BF16_FAMILY_DTYPES:
            return self.bf16_flops_per_s
        if weight_dtype in FP8_DTYPES | FP4_FAMILY_DTYPES:
            return self.fp8_flops_per_s
        if weight_dtype in INT_QUANT_DTYPES:
            return (
                self.fp8_flops_per_s
                if normalized_act_dtype in FP8_DTYPES
                else self.int8_flops_per_s
            )
        return self.bf16_flops_per_s

    def matmul_row_tile(self, dtype: str) -> int:
        """Return the minimum unpadded LHS row tile height for `dtype`."""
        normalized_dtype = dtype.strip().lower()
        table = self.matmul_row_tile_by_dtype or {}
        if normalized_dtype in table:
            return table[normalized_dtype]
        msg = (
            f"Chip {self.name!r} has no matmul_row_tile_by_dtype entry for "
            f"{dtype!r}; known: {sorted(table)}"
        )
        raise UnsupportedChipError(msg)


class _BuiltinExtras(TypedDict):
    """Static non-JAX hardware parameters passed straight into `ChipSpec`."""

    ici_egress_bw_bytes_per_s: float
    ici_link_bw_bytes_per_s: float
    collective_latency_s: float
    collective_latency_us_bracket: tuple[float, float]
    dcn_bw_gbps: float
    dcn_bw_gbps_bracket: tuple[float, float]
    min_slice_cores: int
    topologies: Mapping[int, str]
    vmem_fit_headroom_bytes: NotRequired[int]
    vector_register_bytes: NotRequired[int]
    vector_loads_per_cycle: NotRequired[int]
    clock_hz: NotRequired[float]


JAX_ROW_TILE_DTYPES: Mapping[str, jnp.dtype] = {
    "bf16": jnp.dtype(jnp.bfloat16),
    "fp16": jnp.dtype(jnp.float16),
    "fp8": jnp.dtype(jnp.float8_e4m3fn),
    "fp8_e4m3": jnp.dtype(jnp.float8_e4m3fn),
    "fp8_e4m3fn": jnp.dtype(jnp.float8_e4m3fn),
    "fp8_e5m2": jnp.dtype(jnp.float8_e5m2),
    "int8": jnp.dtype(jnp.int8),
    "fp32": jnp.dtype(jnp.float32),
}
ASSUMED_FP4_DTYPES: tuple[str, ...] = ("int4", "mxfp4", "nvfp4", "fp4")

BUILTIN_CHIP_PARAMS: Mapping[
    str, tuple[pltpu.ChipVersion, Mapping[str, str], _BuiltinExtras]
] = {
    "tpu7x": (
        pltpu.ChipVersion.TPU_7X,
        DEFAULT_SOURCES_7X,
        {
            "ici_egress_bw_bytes_per_s": 300.0e9,
            "ici_link_bw_bytes_per_s": 100.0e9,
            "collective_latency_s": 1.0e-6,
            "collective_latency_us_bracket": DEFAULT_COLLECTIVE_LATENCY_US_BRACKET,
            "dcn_bw_gbps": 100.0,
            "dcn_bw_gbps_bracket": (50.0, 200.0),
            "min_slice_cores": 8,
            "topologies": TPU7X_TOPOLOGIES,
            "vmem_fit_headroom_bytes": CORE_LIMITS.fit_vmem_headroom_mib << 20,
            "vector_register_bytes": 4096,
            "vector_loads_per_cycle": 3,
            "clock_hz": 2.2e9,
        },
    ),
    "v6e": (
        pltpu.ChipVersion.TPU_V6E,
        DEFAULT_SOURCES_PUBLIC,
        {
            "ici_egress_bw_bytes_per_s": 400.0e9,
            "ici_link_bw_bytes_per_s": 100.0e9,
            "collective_latency_s": 1.0e-6,
            "collective_latency_us_bracket": DEFAULT_COLLECTIVE_LATENCY_US_BRACKET,
            "dcn_bw_gbps": 100.0,
            "dcn_bw_gbps_bracket": (50.0, 200.0),
            "min_slice_cores": 1,
            "topologies": TPU6E_TOPOLOGIES,
        },
    ),
}


def _build_row_tiles(info: pltpu.TpuInfo, *, native_ocp_fp8: bool) -> dict[str, int]:
    """Read JAX sublane row tiles and apply the explicit 4-bit ASSUMED rule."""
    tiles = {
        label: int(info.get_sublane_tiling(jax_dtype))
        for label, jax_dtype in JAX_ROW_TILE_DTYPES.items()
    }
    fp4_tile = tiles["fp8"] if native_ocp_fp8 else tiles["bf16"]
    return {**tiles, **dict.fromkeys(ASSUMED_FP4_DTYPES, fp4_tile)}


def _build_jax_sources(
    *, fp8_raw: float, int8_raw: float, num_lanes: int
) -> dict[str, str]:
    """Build provenance source strings for all JAX-sourced and JAX-derived fields."""
    pallas_module = "jax.experimental.pallas.tpu"
    sources = {
        f: f"FACT: {pallas_module} {a}"
        for f, a in (
            ("cores_per_device", "tpu_info.num_cores"),
            ("devices_per_chip", "ChipVersion.num_physical_tensor_cores_per_chip"),
            ("supports_megacore", "ChipVersion.supports_megacore"),
            ("vmem_capacity_bytes", "tpu_info.vmem_capacity_bytes"),
            ("hbm_capacity_bytes", "tpu_info.hbm_capacity_bytes"),
            ("hbm_bw_bytes_per_s", "tpu_info.mem_bw_bytes_per_second"),
            ("bf16_flops_per_s", "tpu_info.bf16_ops_per_second"),
        )
    }
    for key, raw_val, attr in (
        ("fp8_flops_per_s", fp8_raw, "fp8_ops_per_second"),
        ("int8_flops_per_s", int8_raw, "int8_ops_per_second"),
    ):
        sources[key] = (
            f"FACT: {pallas_module} tpu_info.{attr}"
            if raw_val > 0.0
            else f"DERIVED: {pallas_module} tpu_info.{attr}=0; falls back to bf16 rate"
        )
    sources["native_ocp_fp8"] = (
        f"DERIVED: {pallas_module} tpu_info (fp8_ops_per_second > bf16_ops_per_second)"
    )
    sources["matmul_row_tile_by_dtype"] = (
        f"FACT: {pallas_module} tpu_info.get_sublane_tiling (bf16/fp16/fp8/int8/fp32); "
        "ASSUMED: int4/mxfp4/nvfp4/fp4 use fp8 tile when native_ocp_fp8 else bf16 tile"
    )
    sources["vector_f32_ops_per_s"] = (
        f"DERIVED: {pallas_module} tpu_info "
        f"(bf16_ops_per_second / num_lanes={num_lanes})"
    )
    return sources


def build_builtin_chip_spec(name: str) -> ChipSpec:
    """Build a built-in ChipSpec from live JAX Pallas `get_tpu_info_for_chip`."""
    entry = BUILTIN_CHIP_PARAMS.get(name)
    if entry is None:
        msg = (
            f"Unknown built-in chip={name!r}; supported: {sorted(BUILTIN_CHIP_PARAMS)}"
        )
        raise UnsupportedChipError(msg)
    chip_version, static_sources, extras = entry
    info = pltpu.get_tpu_info_for_chip(chip_version, 1)
    bf16_flops = float(info.bf16_ops_per_second)
    fp8_raw = float(info.fp8_ops_per_second or 0)
    int8_raw = float(info.int8_ops_per_second or 0)
    fp8_flops = fp8_raw if fp8_raw > 0.0 else bf16_flops
    int8_flops = int8_raw if int8_raw > 0.0 else bf16_flops
    native_fp8 = fp8_flops > bf16_flops
    vmem_cap, num_lanes = int(info.vmem_capacity_bytes), int(info.num_lanes)
    return ChipSpec(
        name=name,
        cores_per_device=int(info.num_cores),
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_cap - USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=int(info.hbm_capacity_bytes),
        hbm_bw_bytes_per_s=float(info.mem_bw_bytes_per_second),
        bf16_flops_per_s=bf16_flops,
        fp8_flops_per_s=fp8_flops,
        int8_flops_per_s=int8_flops,
        native_ocp_fp8=native_fp8,
        devices_per_chip=int(chip_version.num_physical_tensor_cores_per_chip),
        supports_megacore=bool(chip_version.supports_megacore),
        matmul_row_tile_by_dtype=_build_row_tiles(info, native_ocp_fp8=native_fp8),
        vector_f32_ops_per_s=bf16_flops / float(num_lanes),
        sources={
            **static_sources,
            **_build_jax_sources(
                fp8_raw=fp8_raw, int8_raw=int8_raw, num_lanes=num_lanes
            ),
        },
        **extras,
    )


BUILTIN_CHIP_SPECS: Mapping[str, ChipSpec] = {
    chip_name: build_builtin_chip_spec(chip_name) for chip_name in BUILTIN_CHIP_PARAMS
}


def _scale_megacore(base: ChipSpec, num_cores: int) -> ChipSpec:
    """Scale logical chip ceilings for Megacore."""
    total_cores = base.cores_per_device * num_cores
    chip_name = base.name
    if (
        num_cores > 1
        and total_cores == MEGACORE_CORES
        and not chip_name.endswith("_megacore")
    ):
        chip_name = f"{chip_name}_megacore"
    return replace(
        base,
        name=chip_name,
        cores_per_device=total_cores,
        vmem_capacity_bytes=base.vmem_capacity_bytes * num_cores,
        vmem_usable_bytes=base.vmem_usable_bytes * num_cores,
        vmem_fit_headroom_bytes=base.vmem_fit_headroom_bytes * num_cores,
        hbm_capacity_bytes=base.hbm_capacity_bytes * num_cores,
        hbm_bw_bytes_per_s=base.hbm_bw_bytes_per_s * num_cores,
        bf16_flops_per_s=base.bf16_flops_per_s * num_cores,
        fp8_flops_per_s=base.fp8_flops_per_s * num_cores,
        int8_flops_per_s=base.int8_flops_per_s * num_cores,
        vector_f32_ops_per_s=base.vector_f32_ops_per_s * num_cores,
        vector_loads_per_cycle=base.vector_loads_per_cycle * num_cores,
        ici_egress_bw_bytes_per_s=base.ici_egress_bw_bytes_per_s * num_cores,
    )


def _detect_auto_chip(raw: str) -> str:
    """Detect TPU chip type via `vllm_torchtpu.tpu_info` without claiming PJRT."""
    if raw != "auto":
        return raw
    tpu_type = (tpu_info.get_tpu_type() or "").strip()
    if not tpu_type:
        msg = "chip='auto' requires TPU"
        raise UnsupportedChipError(msg)
    chip_version = tpu_info.get_chip_version(tpu_type)
    prefix = tpu_type.lower().split("-")[0]
    if chip_version == pltpu.ChipVersion.TPU_7X and prefix not in {
        "7x",
        "v7x",
        "tpu7x",
    }:
        msg = f"Unrecognized TPU accelerator type {tpu_type!r}"
        raise UnsupportedChipError(msg)
    return str(chip_version.value).lower()


def resolve_chip_spec(
    chip: ChipSpec | str = "tpu7x",
    *,
    megacore: bool = False,
) -> ChipSpec:
    """Resolve hardware ceiling parameters from sourced built-in tables."""
    if type(chip) is ChipSpec:
        if chip.vector_f32_ops_per_s <= 0.0:
            msg = (
                "Custom ChipSpec requires positive vector_f32_ops_per_s, "
                f"got {chip.vector_f32_ops_per_s}"
            )
            raise errors.MegakernelConfigError(msg)
        base = chip
    else:
        raw = _detect_auto_chip(str(chip).strip().lower().removeprefix("tpu_"))
        if raw.endswith("_megacore"):
            raw, megacore = raw.removesuffix("_megacore"), True
        canonical_chip = CHIP_ALIASES.get(raw, raw)
        if canonical_chip not in BUILTIN_CHIP_SPECS:
            msg = f"Unknown chip={chip!r}; choose from {sorted(BUILTIN_CHIP_SPECS)}"
            raise UnsupportedChipError(msg)
        base = BUILTIN_CHIP_SPECS[canonical_chip]
    if megacore and not base.supports_megacore:
        msg = f"Chip={base.name!r} does not support Megacore (supports_megacore=False)"
        raise UnsupportedChipError(msg)
    scale = MEGACORE_CORES if (megacore and base.cores_per_device == 1) else 1
    return base if scale == 1 else _scale_megacore(base, scale)
