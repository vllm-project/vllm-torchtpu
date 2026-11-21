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
"""Optional JAX Pallas TPU probe and verification bridge."""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Protocol, cast

from .constants import UnsupportedChipError

if TYPE_CHECKING:
    from .chips import ChipSpec


class _ChipVersionEnum(Protocol):
    """Typed protocol for JAX Pallas ChipVersion enum member."""

    value: object


class _ChipVersionType(Protocol):
    """Typed protocol for JAX Pallas ChipVersion enum class."""

    __members__: dict[str, object]


class _TpuInfo(Protocol):
    """Typed protocol for JAX Pallas TPU hardware descriptor."""

    chip_version: _ChipVersionEnum
    num_cores: int
    vmem_capacity_bytes: int
    hbm_capacity_bytes: int
    mem_bw_bytes_per_second: float
    bf16_ops_per_second: float
    fp8_ops_per_second: float | None
    int8_ops_per_second: float | None
    smem_capacity_bytes: int | None = None


class _PallasTpuModule(Protocol):
    """Typed protocol for `jax.experimental.pallas.tpu` module API."""

    ChipVersion: _ChipVersionType

    def get_tpu_info(self) -> _TpuInfo:
        """Return hardware info for attached TPU devices."""

    def get_tpu_info_for_chip(
        self, chip_version: object, cores_per_device: int
    ) -> _TpuInfo:
        """Return hardware info for a named TPU chip generation."""


def _load_pltpu() -> _PallasTpuModule | None:
    """Lazily import jax.experimental.pallas.tpu without loading JAX at import."""
    if importlib.util.find_spec("jax.experimental.pallas.tpu") is None:
        return None
    try:
        mod = importlib.import_module("jax.experimental.pallas.tpu")
    except (ImportError, RuntimeError):
        return None
    return cast("_PallasTpuModule", mod)


def probe_jax_tpu_info() -> tuple[str, int] | None:
    """Probe TPU device version and core count from JAX Pallas if available."""
    pltpu = _load_pltpu()
    if pltpu is None:
        return None
    try:
        info = pltpu.get_tpu_info()
        return str(info.chip_version.value).lower(), int(info.num_cores)
    except (AttributeError, RuntimeError, ValueError):
        return None


CHIP_TO_PALLAS_VERSION: dict[str, str] = {
    "tpu7x": "TPU_7X",
    "v6e": "TPU_V6E",
}


def verify_chip_spec_against_jax(spec: ChipSpec) -> list[str]:
    """Compare a built-in ChipSpec against jax.experimental.pallas.tpu if available."""
    pltpu = _load_pltpu()
    if pltpu is None:
        return []
    base_name = spec.name.removesuffix("_megacore")
    enum_key = CHIP_TO_PALLAS_VERSION.get(base_name)
    if enum_key is None:
        msg = f"Unsupported chip={spec.name!r} for JAX verification"
        raise UnsupportedChipError(msg)
    chip_ver = pltpu.ChipVersion.__members__.get(enum_key)
    if chip_ver is None:
        return []
    info = pltpu.get_tpu_info_for_chip(chip_ver, 1)
    cores = spec.cores_per_device
    mismatches: list[str] = []
    smem_val = getattr(info, "smem_capacity_bytes", None)
    if smem_val is not None and int(smem_val) > 0:
        expected_smem = int(smem_val) * cores
        if spec.scalar_mem_capacity_bytes != expected_smem:
            mismatches.append(
                f"{spec.name}.scalar_mem_capacity_bytes: "
                f"{spec.scalar_mem_capacity_bytes} != {expected_smem}"
            )
    fp8_raw = float(info.fp8_ops_per_second or 0) * cores
    int8_raw = float(info.int8_ops_per_second or 0) * cores
    exp_fp8 = fp8_raw if fp8_raw > 0 else float(spec.bf16_flops_per_s)
    exp_int8 = int8_raw if int8_raw > 0 else float(spec.bf16_flops_per_s)
    checks = (
        (
            "vmem_capacity_bytes",
            spec.vmem_capacity_bytes,
            int(info.vmem_capacity_bytes) * cores,
        ),
        (
            "hbm_capacity_bytes",
            spec.hbm_capacity_bytes,
            int(info.hbm_capacity_bytes) * cores,
        ),
        (
            "hbm_bw_bytes_per_s",
            spec.hbm_bw_bytes_per_s,
            float(info.mem_bw_bytes_per_second) * cores,
        ),
        (
            "bf16_flops_per_s",
            spec.bf16_flops_per_s,
            float(info.bf16_ops_per_second) * cores,
        ),
        (
            "fp8_flops_per_s",
            spec.fp8_flops_per_s,
            exp_fp8,
        ),
        (
            "int8_flops_per_s",
            spec.int8_flops_per_s,
            exp_int8,
        ),
    )
    for field_name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{spec.name}.{field_name}: {actual} != {expected}")
    sources = spec.sources or {}
    fp8_src = str(sources.get("fp8_flops_per_s", ""))
    int8_src = str(sources.get("int8_flops_per_s", ""))
    if fp8_raw == 0 and not fp8_src.startswith("DERIVED:"):
        mismatches.append(
            f"{spec.name}.fp8_flops_per_s: missing DERIVED: source prefix"
        )
    if int8_raw == 0 and not int8_src.startswith("DERIVED:"):
        mismatches.append(
            f"{spec.name}.int8_flops_per_s: missing DERIVED: source prefix"
        )
    return mismatches
