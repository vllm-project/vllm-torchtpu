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
"""Sourced TPU hardware specifications and JSON `--chip-spec` loader."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import errors
from . import jax_bridge
from .constants import (
    ALLOWED_CHIP_SPEC_KEYS,
    BRACKET_LEN,
    CALIBRATION_DIR,
    CHIP_ALIASES,
    DEFAULT_ROW_TILE_FP8,
    DEFAULT_ROW_TILE_WIDE,
    DEFAULT_SOURCES_7X,
    DEFAULT_SOURCES_PUBLIC,
    GOOGLE_PCI_VENDOR_ID,
    MEGACORE_CORES,
    PCI_DEVICE_TO_CHIP,
    REQUIRED_CHIP_SPEC_KEYS,
    ROW_TILES_7X,
    ROW_TILES_V6E,
    SUPPORTED_CHIPS,
    SYSFS_PCI_DEVICES_DIR,
    TPU6E_TOPOLOGIES,
    TPU7X_TOPOLOGIES,
    USABLE_VMEM_RESERVE_BYTES,
    UnsupportedChipError,
)

__all__ = [
    "BUILTIN_CHIP_SPECS",
    "SUPPORTED_CHIPS",
    "ChipSpec",
    "UnsupportedChipError",
    "detect_tpu_from_sysfs",
    "load_chip_spec_file",
    "load_public_calibration",
    "public_calibration_path",
    "resolve_chip_spec",
    "verify_chip_spec_against_jax",
]

if TYPE_CHECKING:
    from collections.abc import Mapping


def _json_number(val: object, field_name: str) -> float:
    """Convert a JSON int/float scalar to float, failing closed on bools/non-numbers."""
    match val:
        case bool():
            msg = f"{field_name} must be numeric, got bool"
            raise errors.MegakernelConfigError(msg)
        case int() | float():
            num = float(val)
            if not math.isfinite(num):
                msg = f"{field_name} must be finite, got {val!r}"
                raise errors.MegakernelConfigError(msg)
            return num
        case _:
            msg = f"{field_name} must be numeric, got {type(val).__name__}"
            raise errors.MegakernelConfigError(msg)


def _json_int(val: object, field_name: str) -> int:
    """Convert a JSON integer scalar to int, rejecting bools and floats."""
    match val:
        case bool():
            msg = f"{field_name} must be an integer, got bool"
            raise errors.MegakernelConfigError(msg)
        case int():
            return val
        case _:
            msg = f"{field_name} must be an integer, got {type(val).__name__}"
            raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class FieldSource:
    """Provenance metadata for one hardware specification field."""

    field: str
    tag: str
    source: str


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
    # JAX devices per physical chip (TPU v7x exposes 2 logical cores per chip).
    devices_per_chip: int = 1
    supports_megacore: bool = False
    matmul_row_tile_by_dtype: Mapping[str, int] | None = None
    topologies: Mapping[int, str] = field(default_factory=lambda: TPU7X_TOPOLOGIES)
    sources: Mapping[str, str] | None = None
    tpu_info: object = None
    vector_f32_ops_per_s: float = 0.0
    scalar_mem_capacity_bytes: int = 1_048_576
    pcie_bw_bytes_per_s: float = 32.0e9
    torus_ndim: int = 2
    vector_registers_per_core: int = 0
    vector_register_bytes: int = 0
    vector_loads_per_cycle: int = 0
    vector_stores_per_cycle: int = 0
    vector_port_ops_per_cycle: int = 0
    clock_hz: float = 0.0

    @property
    def vector_load_cap_bytes_per_s(self) -> float:
        """Derived maximum vector register load bandwidth in bytes per second."""
        return float(
            self.vector_loads_per_cycle * self.vector_register_bytes * self.clock_hz
        )

    def peak_flops(self, dtype: str, act_dtype: str | None = None) -> float:
        """Return peak dense matrix FLOP/s for `dtype` and `act_dtype`.

        Symmetric fp8 matrix rate requires both inputs in fp8 (W8A8); mixed
        W8A16 (bf16 activations x fp8 weights) executes at the bf16 rate.
        """
        dt = dtype.lower()
        act_dt = (act_dtype or "").strip().lower()
        if act_dt in ("bf16", "bfloat16", "fp16", "float16"):
            return self.bf16_flops_per_s
        if act_dt in ("fp32", "float32"):
            return 0.5 * self.bf16_flops_per_s
        if dt in ("fp8", "nvfp4", "mxfp4", "fp4"):
            return self.fp8_flops_per_s
        if dt == "int8":
            return self.int8_flops_per_s
        if dt == "fp32":
            return 0.5 * self.bf16_flops_per_s
        return self.bf16_flops_per_s

    def matmul_row_tile(self, dtype: str) -> int:
        """Return the minimum unpadded LHS row tile height for `dtype`."""
        dt = dtype.lower()
        table = self.matmul_row_tile_by_dtype or {}
        if dt in table:
            return table[dt]
        if dt in ("fp8", "nvfp4", "mxfp4", "fp4") and self.native_ocp_fp8:
            return DEFAULT_ROW_TILE_FP8
        return DEFAULT_ROW_TILE_WIDE

    def lhs_row_floor(self, dtype: str) -> int:
        """Alias for `matmul_row_tile(dtype)` kept for caller compatibility."""
        return self.matmul_row_tile(dtype)


BUILTIN_CHIP_SPECS: Mapping[str, ChipSpec] = {
    "tpu7x": ChipSpec(
        name="tpu7x",
        cores_per_device=1,
        vmem_capacity_bytes=67_108_864,
        vmem_usable_bytes=67_108_864 - USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=103_000_000_000,
        hbm_bw_bytes_per_s=3_700_000_000_000.0,
        bf16_flops_per_s=1_155_000_000_000_000.0,
        fp8_flops_per_s=2_300_000_000_000_000.0,
        int8_flops_per_s=1_155_000_000_000_000.0,
        native_ocp_fp8=True,
        ici_egress_bw_bytes_per_s=300.0e9,
        ici_link_bw_bytes_per_s=100.0e9,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 5.13),
        min_slice_cores=8,
        devices_per_chip=2,
        supports_megacore=False,
        matmul_row_tile_by_dtype=ROW_TILES_7X,
        topologies=TPU7X_TOPOLOGIES,
        sources=DEFAULT_SOURCES_7X,
        vector_f32_ops_per_s=1_155_000_000_000_000.0 / 128.0,
        scalar_mem_capacity_bytes=1_048_576,
        pcie_bw_bytes_per_s=32.0e9,
        torus_ndim=3,
        vector_registers_per_core=64,
        vector_register_bytes=4096,
        vector_loads_per_cycle=3,
        vector_stores_per_cycle=2,
        vector_port_ops_per_cycle=4,
        clock_hz=2.2e9,
    ),
    "v6e": ChipSpec(
        name="v6e",
        cores_per_device=1,
        vmem_capacity_bytes=134_217_728,
        vmem_usable_bytes=134_217_728 - USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=34_400_000_000,
        hbm_bw_bytes_per_s=1_640_000_000_000.0,
        bf16_flops_per_s=920_000_000_000_000.0,
        fp8_flops_per_s=920_000_000_000_000.0,
        int8_flops_per_s=1_840_000_000_000_000.0,
        native_ocp_fp8=False,
        ici_egress_bw_bytes_per_s=400.0e9,
        ici_link_bw_bytes_per_s=100.0e9,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 5.0),
        min_slice_cores=1,
        supports_megacore=False,
        matmul_row_tile_by_dtype=ROW_TILES_V6E,
        topologies=TPU6E_TOPOLOGIES,
        sources=DEFAULT_SOURCES_PUBLIC,
        vector_f32_ops_per_s=920_000_000_000_000.0 / 128.0,
        scalar_mem_capacity_bytes=1_048_576,
        pcie_bw_bytes_per_s=32.0e9,
        torus_ndim=2,
    ),
}


def _parse_bracket(
    raw: Mapping[str, object],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Parse a positive non-decreasing [lo, hi] bracket, failing closed if invalid."""
    if key not in raw:
        return default
    match raw[key]:
        case [lo_raw, hi_raw]:
            lo = _json_number(lo_raw, f"{key}[0]")
            hi = _json_number(hi_raw, f"{key}[1]")
            if lo <= 0.0 or hi < lo:
                msg = f"{key} requires 0 < lo <= hi, got [{lo}, {hi}]"
                raise errors.MegakernelConfigError(msg)
            return (lo, hi)
        case _:
            msg = (
                f"{key} must be a {BRACKET_LEN}-element list [lo, hi], got {raw[key]!r}"
            )
            raise errors.MegakernelConfigError(msg)


def _parse_row_tiles(
    raw: Mapping[str, object], *, native_fp8: bool
) -> Mapping[str, int]:
    """Parse optional `matmul_row_tile_by_dtype` mapping, rejecting non-dicts."""
    if "matmul_row_tile_by_dtype" not in raw:
        return ROW_TILES_7X if native_fp8 else ROW_TILES_V6E
    val = raw["matmul_row_tile_by_dtype"]
    if type(val) is not dict:
        msg = (
            f"matmul_row_tile_by_dtype must be a JSON object, got {type(val).__name__}"
        )
        raise errors.MegakernelConfigError(msg)
    return {
        str(k).lower(): _json_int(v, f"matmul_row_tile_by_dtype[{k!r}]")
        for k, v in val.items()
    }


def _parse_topologies(raw: Mapping[str, object]) -> Mapping[int, str]:
    """Parse optional `{core_count: label}` mapping, rejecting malformed entries."""
    if "topologies" not in raw:
        return {}
    val = raw["topologies"]
    if type(val) is not dict:
        msg = f"topologies must be a JSON object, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    out: dict[int, str] = {}
    for k, v in val.items():
        try:
            cores = int(k)
        except (TypeError, ValueError) as exc:
            msg = f"topologies keys must be integer core counts, got {k!r}"
            raise errors.MegakernelConfigError(msg) from exc
        if cores < 1:
            msg = f"topologies core count must be >= 1, got {cores}"
            raise errors.MegakernelConfigError(msg)
        out[cores] = str(v)
    return out


def _parse_sources(raw: Mapping[str, object], spec_name: str) -> dict[str, str]:
    """Parse optional `sources` provenance mapping, rejecting non-objects."""
    if "sources" not in raw:
        return {"spec": f"user --chip-spec {spec_name}"}
    val = raw["sources"]
    if type(val) is not dict:
        msg = f"sources must be a JSON object, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return {str(k): str(v) for k, v in val.items()}


def load_chip_spec_file(
    path: str | Path, fallback_name: str = "custom_chip"
) -> ChipSpec:
    """Load a user-supplied JSON `--chip-spec` file and validate required keys."""
    spec_path = Path(path)
    if not spec_path.is_file():
        msg = f"Missing --chip-spec file: {spec_path}"
        raise errors.MegakernelConfigError(msg)
    raw = json.loads(spec_path.read_text("utf-8"))
    if type(raw) is not dict:
        msg = f"--chip-spec file {spec_path} must contain a JSON object"
        raise errors.MegakernelConfigError(msg)
    missing = [k for k in REQUIRED_CHIP_SPEC_KEYS if k not in raw]
    if missing:
        msg = (
            f"--chip-spec file {spec_path} missing required keys {missing}; "
            f"required: {list(REQUIRED_CHIP_SPEC_KEYS)}"
        )
        raise errors.MegakernelConfigError(msg)
    unknown = sorted(set(raw) - ALLOWED_CHIP_SPEC_KEYS)
    if unknown:
        msg = (
            f"--chip-spec file {spec_path} has unrecognized keys {unknown}; "
            f"allowed: {sorted(ALLOWED_CHIP_SPEC_KEYS)}"
        )
        raise errors.MegakernelConfigError(msg)
    vmem_cap = _json_int(raw["vmem_capacity_bytes"], "vmem_capacity_bytes")
    vmem_usable = (
        _json_int(raw["vmem_usable_bytes"], "vmem_usable_bytes")
        if "vmem_usable_bytes" in raw
        else max(0, vmem_cap - USABLE_VMEM_RESERVE_BYTES)
    )
    bf16_flops = _json_number(raw["bf16_flops_per_s"], "bf16_flops_per_s")
    fp8_flops = (
        _json_number(raw["fp8_flops_per_s"], "fp8_flops_per_s")
        if "fp8_flops_per_s" in raw
        else bf16_flops
    )
    native_fp8 = bool(raw.get("native_ocp_fp8", fp8_flops > bf16_flops))
    int8_val = (
        _json_number(raw["int8_flops_per_s"], "int8_flops_per_s")
        if raw.get("int8_flops_per_s") is not None
        else 0.0
    )
    int8_flops = int8_val if int8_val > 0.0 else bf16_flops
    bracket = _parse_bracket(raw, "collective_latency_us_bracket", (1.0, 5.0))
    lat_s = (
        _json_number(raw["collective_latency_s"], "collective_latency_s")
        if "collective_latency_s" in raw
        else _json_number(
            raw.get("collective_latency_us", bracket[0]), "collective_latency_us"
        )
        * 1e-6
    )
    dcn_bracket = _parse_bracket(raw, "dcn_bw_gbps_bracket", (50.0, 200.0))
    vec_val = (
        _json_number(raw["vector_f32_ops_per_s"], "vector_f32_ops_per_s")
        if "vector_f32_ops_per_s" in raw
        else 0.0
    )
    vector_f32_ops = vec_val if vec_val > 0.0 else bf16_flops / 128.0
    scalar_mem = (
        _json_int(raw["scalar_mem_capacity_bytes"], "scalar_mem_capacity_bytes")
        if "scalar_mem_capacity_bytes" in raw
        else 1_048_576
    )
    pcie_bw = (
        _json_number(raw["pcie_bw_bytes_per_s"], "pcie_bw_bytes_per_s")
        if "pcie_bw_bytes_per_s" in raw
        else 32.0e9
    )
    torus_dim = _json_int(raw["torus_ndim"], "torus_ndim") if "torus_ndim" in raw else 2
    return ChipSpec(
        name=str(raw.get("name") or fallback_name),
        cores_per_device=_json_int(raw.get("cores_per_device", 1), "cores_per_device"),
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_usable,
        hbm_capacity_bytes=_json_int(raw["hbm_capacity_bytes"], "hbm_capacity_bytes"),
        hbm_bw_bytes_per_s=_json_number(
            raw["hbm_bw_bytes_per_s"], "hbm_bw_bytes_per_s"
        ),
        bf16_flops_per_s=bf16_flops,
        fp8_flops_per_s=fp8_flops,
        int8_flops_per_s=int8_flops,
        native_ocp_fp8=native_fp8,
        ici_egress_bw_bytes_per_s=_json_number(
            raw.get("ici_egress_bw_bytes_per_s", 300.0e9),
            "ici_egress_bw_bytes_per_s",
        ),
        ici_link_bw_bytes_per_s=_json_number(
            raw.get("ici_link_bw_bytes_per_s", 100.0e9), "ici_link_bw_bytes_per_s"
        ),
        collective_latency_s=lat_s,
        collective_latency_us_bracket=bracket,
        dcn_bw_gbps=_json_number(raw.get("dcn_bw_gbps", 100.0), "dcn_bw_gbps"),
        dcn_bw_gbps_bracket=dcn_bracket,
        min_slice_cores=_json_int(raw.get("min_slice_cores", 1), "min_slice_cores"),
        devices_per_chip=_json_int(raw.get("devices_per_chip", 1), "devices_per_chip"),
        supports_megacore=bool(raw.get("supports_megacore", False)),
        matmul_row_tile_by_dtype=_parse_row_tiles(raw, native_fp8=native_fp8),
        topologies=_parse_topologies(raw),
        sources=_parse_sources(raw, spec_path.name),
        vector_f32_ops_per_s=vector_f32_ops,
        scalar_mem_capacity_bytes=scalar_mem,
        pcie_bw_bytes_per_s=pcie_bw,
        torus_ndim=torus_dim,
        vector_registers_per_core=_json_int(
            raw.get("vector_registers_per_core", 0), "vector_registers_per_core"
        ),
        vector_register_bytes=_json_int(
            raw.get("vector_register_bytes", 0), "vector_register_bytes"
        ),
        vector_loads_per_cycle=_json_int(
            raw.get("vector_loads_per_cycle", 0), "vector_loads_per_cycle"
        ),
        vector_stores_per_cycle=_json_int(
            raw.get("vector_stores_per_cycle", 0), "vector_stores_per_cycle"
        ),
        vector_port_ops_per_cycle=_json_int(
            raw.get("vector_port_ops_per_cycle", 0), "vector_port_ops_per_cycle"
        ),
        clock_hz=_json_number(raw.get("clock_hz", 0.0), "clock_hz"),
    )


def _scale_megacore(
    base: ChipSpec,
    num_cores: int,
    ici_bw_gbps: float | None,
    collective_latency_us: float | None,
) -> ChipSpec:
    """Scale a `ChipSpec` by `num_cores` and apply user overrides."""
    vmem_cap = base.vmem_capacity_bytes * num_cores
    base_usable = (
        base.vmem_usable_bytes
        if base.vmem_usable_bytes > 0
        else max(0, base.vmem_capacity_bytes - USABLE_VMEM_RESERVE_BYTES)
    )
    vmem_usable = base_usable * num_cores
    egress = (
        base.ici_egress_bw_bytes_per_s * num_cores
        if ici_bw_gbps is None
        else ici_bw_gbps * 1e9
    )
    lat_s = (
        base.collective_latency_s
        if collective_latency_us is None
        else collective_latency_us * 1e-6
    )
    bracket = (
        base.collective_latency_us_bracket
        if collective_latency_us is None
        else (collective_latency_us, collective_latency_us)
    )
    total_cores = base.cores_per_device * num_cores
    chip_name = (
        f"{base.name}_megacore"
        if total_cores == MEGACORE_CORES and not base.name.endswith("_megacore")
        else base.name
    )
    return replace(
        base,
        name=chip_name,
        cores_per_device=total_cores,
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_usable,
        hbm_capacity_bytes=base.hbm_capacity_bytes * num_cores,
        hbm_bw_bytes_per_s=base.hbm_bw_bytes_per_s * num_cores,
        bf16_flops_per_s=base.bf16_flops_per_s * num_cores,
        fp8_flops_per_s=base.fp8_flops_per_s * num_cores,
        int8_flops_per_s=base.int8_flops_per_s * num_cores,
        vector_f32_ops_per_s=base.vector_f32_ops_per_s * num_cores,
        scalar_mem_capacity_bytes=base.scalar_mem_capacity_bytes * num_cores,
        ici_egress_bw_bytes_per_s=egress,
        ici_link_bw_bytes_per_s=base.ici_link_bw_bytes_per_s * num_cores,
        collective_latency_s=lat_s,
        collective_latency_us_bracket=bracket,
    )


def detect_tpu_from_sysfs(
    pci_dir: Path | None = None,
) -> str | None:
    """Detect the host's TPU generation from PCI sysfs without opening PJRT."""
    root = SYSFS_PCI_DEVICES_DIR if pci_dir is None else pci_dir
    if not root.is_dir():
        return None
    for vendor_file in sorted(root.glob("*/vendor")):
        try:
            vendor = vendor_file.read_text("utf-8").strip().lower()
            if vendor != GOOGLE_PCI_VENDOR_ID:
                continue
            dev_id = (vendor_file.parent / "device").read_text("utf-8").strip().lower()
        except OSError:
            continue
        chip_name = PCI_DEVICE_TO_CHIP.get(dev_id)
        if chip_name is not None:
            return chip_name
    return None


def _detect_auto_chip(raw: str) -> tuple[str, int | None]:
    """Detect TPU chip type and core count from sysfs or JAX probe."""
    if raw != "auto":
        return raw, None
    sysfs_chip = detect_tpu_from_sysfs()
    if sysfs_chip is not None:
        return sysfs_chip, None
    probed = jax_bridge.probe_jax_tpu_info()
    if probed is None:
        msg = "chip='auto' requires TPU"
        raise errors.MegakernelConfigError(msg)
    return probed


def resolve_chip_spec(
    chip: ChipSpec | str = "tpu7x",
    *,
    megacore: bool = False,
    ici_bw_gbps: float | None = None,
    collective_latency_us: float | None = None,
    chip_spec_path: str | Path | None = None,
) -> ChipSpec:
    """Resolve hardware ceiling parameters from sourced table or `--chip-spec`."""
    if type(chip) is ChipSpec:
        return chip
    if chip_spec_path is not None:
        loaded = load_chip_spec_file(chip_spec_path, fallback_name=str(chip))
        if megacore and not loaded.supports_megacore:
            msg = (
                f"Chip={loaded.name!r} does not support Megacore "
                "(supports_megacore=False)"
            )
            raise UnsupportedChipError(msg)
        scale = MEGACORE_CORES if (megacore and loaded.cores_per_device == 1) else 1
        if scale == 1 and ici_bw_gbps is None and collective_latency_us is None:
            return loaded
        return _scale_megacore(loaded, scale, ici_bw_gbps, collective_latency_us)
    raw = str(chip).strip().lower().removeprefix("tpu_")
    auto_cores: int | None = None
    raw, auto_cores = _detect_auto_chip(raw)

    if raw.endswith("_megacore"):
        raw = raw.removesuffix("_megacore")
        megacore = True
    canon = CHIP_ALIASES.get(raw, raw)
    if canon not in BUILTIN_CHIP_SPECS:
        req = list(REQUIRED_CHIP_SPEC_KEYS)
        known = sorted(BUILTIN_CHIP_SPECS)
        msg = (
            f"Unknown chip={chip!r}. Pass --chip-spec <path.json> with required keys "
            f"{req} or choose from {known}"
        )
        raise UnsupportedChipError(msg)
    base = BUILTIN_CHIP_SPECS[canon]
    if megacore and not base.supports_megacore:
        msg = f"Invalid chip={chip!r} or Megacore (supports_megacore=False)"
        raise UnsupportedChipError(msg)
    num_cores = (
        auto_cores
        if (auto_cores is not None and base.supports_megacore)
        else (1 + int(megacore))
    )
    return _scale_megacore(base, num_cores, ici_bw_gbps, collective_latency_us)


def verify_chip_spec_against_jax(spec: ChipSpec) -> list[str]:
    """Compare a built-in ChipSpec against jax.experimental.pallas.tpu if available."""
    return jax_bridge.verify_chip_spec_against_jax(spec)


def public_calibration_path(
    chip_name: str = "tpu7x", *, calibration_dir: Path | None = None
) -> Path | None:
    """Return path to bundled architectural priors JSON for `chip_name` if present."""
    cal_dir = CALIBRATION_DIR if calibration_dir is None else calibration_dir
    raw = chip_name.strip().lower().removeprefix("tpu_").removesuffix("_megacore")
    canon = CHIP_ALIASES.get(raw, raw)
    candidate = cal_dir / f"{canon}_public.json"
    return candidate if candidate.is_file() else None


def load_public_calibration(
    chip_name: str = "tpu7x", *, calibration_dir: Path | None = None
) -> dict[str, object] | None:
    """Load the bundled architectural priors JSON for `chip_name` when available."""
    cal_path = public_calibration_path(chip_name, calibration_dir=calibration_dir)
    if cal_path is None:
        return None
    loaded = json.loads(cal_path.read_text(encoding="utf-8"))
    if type(loaded) is not dict:
        msg = f"Expected JSON object in {cal_path}, got {type(loaded).__name__}"
        raise errors.MegakernelConfigError(msg)
    return {str(k): v for k, v in loaded.items()}
