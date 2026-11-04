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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .. import errors

if TYPE_CHECKING:
    from collections.abc import Mapping

_MEGACORE_CORES = 2
_BRACKET_LEN = 2
_USABLE_VMEM_RESERVE_BYTES = 2 << 20
_DEFAULT_ROW_TILE_FP8 = 32
_DEFAULT_ROW_TILE_WIDE = 16
_DEFAULT_ROW_TILE_NARROW = 8
_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_SYSFS_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")
_PCI_DEVICE_TO_CHIP: Mapping[str, str] = {
    "0x0027": "v4",
    "0x0056": "v5p",
    "0x0063": "v5e",
    "0x006f": "v6e",
    "0x0076": "tpu7x",
}

REQUIRED_CHIP_SPEC_KEYS: tuple[str, ...] = (
    "vmem_capacity_bytes",
    "hbm_capacity_bytes",
    "hbm_bw_bytes_per_s",
    "bf16_flops_per_s",
)
_OPTIONAL_CHIP_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "cores_per_device",
        "vmem_usable_bytes",
        "fp8_flops_per_s",
        "int8_flops_per_s",
        "native_ocp_fp8",
        "ici_egress_bw_bytes_per_s",
        "ici_link_bw_bytes_per_s",
        "collective_latency_us",
        "collective_latency_s",
        "collective_latency_us_bracket",
        "dcn_bw_gbps",
        "dcn_bw_gbps_bracket",
        "min_slice_cores",
        "devices_per_chip",
        "supports_megacore",
        "matmul_row_tile_by_dtype",
        "topologies",
        "sources",
    }
)
_ALLOWED_CHIP_SPEC_KEYS: frozenset[str] = (
    frozenset(REQUIRED_CHIP_SPEC_KEYS) | _OPTIONAL_CHIP_SPEC_KEYS
)

TPU7X_TOPOLOGIES: Mapping[int, str] = {
    8: "2x2x1 (4 chips, 8 cores)",
    16: "2x2x2 (8 chips, 16 cores)",
    32: "2x2x4 (16 chips, 32 cores)",
    64: "2x4x4 (32 chips, 64 cores)",
    128: "4x4x4 (64 chips, 128 cores)",
}

TPU6E_TOPOLOGIES: Mapping[int, str] = {
    1: "1x1 (1 chip, 1 core)",
    4: "2x2 (4 chips, 4 cores)",
    8: "2x4 (8 chips, 8 cores)",
    16: "4x4 (16 chips, 16 cores)",
    32: "4x8 (32 chips, 32 cores)",
    64: "8x8 (64 chips, 64 cores)",
}


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
    # JAX devices per physical chip (TPU v7x exposes each TensorCore).
    devices_per_chip: int = 1
    supports_megacore: bool = False
    matmul_row_tile_by_dtype: Mapping[str, int] | None = None
    topologies: Mapping[int, str] = field(default_factory=lambda: TPU7X_TOPOLOGIES)
    sources: Mapping[str, str] | None = None
    tpu_info: object = None

    def peak_flops(self, dtype: str) -> float:
        """Return peak dense matrix FLOP/s for `dtype` on this logical device."""
        dt = dtype.lower()
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
            return _DEFAULT_ROW_TILE_FP8
        return _DEFAULT_ROW_TILE_WIDE

    def lhs_row_floor(self, dtype: str) -> int:
        """Alias for `matmul_row_tile(dtype)` kept for caller compatibility."""
        return self.matmul_row_tile(dtype)


_ROW_TILES_7X: Mapping[str, int] = {
    "fp8": _DEFAULT_ROW_TILE_FP8,
    "nvfp4": _DEFAULT_ROW_TILE_FP8,
    "mxfp4": _DEFAULT_ROW_TILE_FP8,
    "fp4": _DEFAULT_ROW_TILE_FP8,
    "bf16": _DEFAULT_ROW_TILE_WIDE,
    "fp16": _DEFAULT_ROW_TILE_WIDE,
    "int8": _DEFAULT_ROW_TILE_WIDE,
    "fp32": _DEFAULT_ROW_TILE_WIDE,
}
_ROW_TILES_V6E: Mapping[str, int] = dict.fromkeys(_ROW_TILES_7X, _DEFAULT_ROW_TILE_WIDE)
_ROW_TILES_NARROW: Mapping[str, int] = dict.fromkeys(
    _ROW_TILES_7X, _DEFAULT_ROW_TILE_NARROW
)

_DEFAULT_SOURCES_7X: Mapping[str, str] = {
    "vmem_capacity_bytes": "jax.experimental.pallas.tpu tpu_info (7x: 64 MiB/core)",
    "hbm_capacity_bytes": "jax.experimental.pallas.tpu tpu_info (7x: 103 GB/core)",
    "hbm_bw_bytes_per_s": "jax.experimental.pallas.tpu tpu_info (7x: 3.70 TB/s/core)",
    "bf16_flops_per_s": "jax.experimental.pallas.tpu tpu_info (7x: 1.155 PF/s/core)",
    "fp8_flops_per_s": "jax.experimental.pallas.tpu tpu_info (7x: 2.30 PF/s/core)",
    "ici_egress_bw_bytes_per_s": "Cloud TPU v7x interconnect spec (300 GB/s/core)",
    "collective_latency_s": "ASSUMED [1.0, 5.13] us/hop bracket; calibrate via trace",
    "dcn_bw_gbps": "ASSUMED [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)",
}
_DEFAULT_SOURCES_PUBLIC: Mapping[str, str] = {
    "vmem_capacity_bytes": "jax.experimental.pallas.tpu tpu_info",
    "hbm_capacity_bytes": "jax.experimental.pallas.tpu tpu_info",
    "hbm_bw_bytes_per_s": "jax.experimental.pallas.tpu tpu_info",
    "bf16_flops_per_s": "jax.experimental.pallas.tpu tpu_info",
    "fp8_flops_per_s": "jax.experimental.pallas.tpu tpu_info",
    "ici_egress_bw_bytes_per_s": "Cloud TPU interconnect architecture documentation",
    "collective_latency_s": "ASSUMED [1.0, 5.0] us/hop bracket; calibrate via trace",
    "dcn_bw_gbps": "ASSUMED [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)",
}

BUILTIN_CHIP_SPECS: Mapping[str, ChipSpec] = {
    "tpu7x": ChipSpec(
        name="tpu7x",
        cores_per_device=1,
        vmem_capacity_bytes=67_108_864,
        vmem_usable_bytes=67_108_864 - _USABLE_VMEM_RESERVE_BYTES,
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
        matmul_row_tile_by_dtype=_ROW_TILES_7X,
        topologies=TPU7X_TOPOLOGIES,
        sources=_DEFAULT_SOURCES_7X,
    ),
    "v6e": ChipSpec(
        name="v6e",
        cores_per_device=1,
        vmem_capacity_bytes=134_217_728,
        vmem_usable_bytes=134_217_728 - _USABLE_VMEM_RESERVE_BYTES,
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
        matmul_row_tile_by_dtype=_ROW_TILES_V6E,
        topologies=TPU6E_TOPOLOGIES,
        sources=_DEFAULT_SOURCES_PUBLIC,
    ),
    "v5p": ChipSpec(
        name="v5p",
        cores_per_device=1,
        vmem_capacity_bytes=67_108_864,
        vmem_usable_bytes=67_108_864 - _USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=51_500_000_000,
        hbm_bw_bytes_per_s=1_230_000_000_000.0,
        bf16_flops_per_s=229_500_000_000_000.0,
        fp8_flops_per_s=229_500_000_000_000.0,
        int8_flops_per_s=459_000_000_000_000.0,
        native_ocp_fp8=False,
        ici_egress_bw_bytes_per_s=300.0e9,
        ici_link_bw_bytes_per_s=100.0e9,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 5.0),
        supports_megacore=True,
        matmul_row_tile_by_dtype=_ROW_TILES_NARROW,
        sources=_DEFAULT_SOURCES_PUBLIC,
    ),
    "v5e": ChipSpec(
        name="v5e",
        cores_per_device=1,
        vmem_capacity_bytes=134_217_728,
        vmem_usable_bytes=134_217_728 - _USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=17_200_000_000,
        hbm_bw_bytes_per_s=820_000_000_000.0,
        bf16_flops_per_s=197_000_000_000_000.0,
        fp8_flops_per_s=197_000_000_000_000.0,
        int8_flops_per_s=394_000_000_000_000.0,
        native_ocp_fp8=False,
        ici_egress_bw_bytes_per_s=200.0e9,
        ici_link_bw_bytes_per_s=50.0e9,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 5.0),
        supports_megacore=False,
        matmul_row_tile_by_dtype=_ROW_TILES_NARROW,
        sources=_DEFAULT_SOURCES_PUBLIC,
    ),
    "v4": ChipSpec(
        name="v4",
        cores_per_device=1,
        vmem_capacity_bytes=16_777_216,
        vmem_usable_bytes=16_777_216 - _USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=17_200_000_000,
        hbm_bw_bytes_per_s=615_000_000_000.0,
        bf16_flops_per_s=137_500_000_000_000.0,
        fp8_flops_per_s=137_500_000_000_000.0,
        int8_flops_per_s=137_500_000_000_000.0,
        native_ocp_fp8=False,
        ici_egress_bw_bytes_per_s=135.0e9,
        ici_link_bw_bytes_per_s=45.0e9,
        collective_latency_s=1.0e-6,
        collective_latency_us_bracket=(1.0, 5.0),
        supports_megacore=True,
        matmul_row_tile_by_dtype=_ROW_TILES_NARROW,
        sources=_DEFAULT_SOURCES_PUBLIC,
    ),
}


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
    unknown = sorted(set(raw) - _ALLOWED_CHIP_SPEC_KEYS)
    if unknown:
        msg = (
            f"--chip-spec file {spec_path} has unrecognized keys {unknown}; "
            f"allowed: {sorted(_ALLOWED_CHIP_SPEC_KEYS)}"
        )
        raise errors.MegakernelConfigError(msg)
    vmem_cap = int(raw["vmem_capacity_bytes"])
    vmem_usable = int(
        raw.get("vmem_usable_bytes", max(0, vmem_cap - _USABLE_VMEM_RESERVE_BYTES))
    )
    bf16_flops = float(raw["bf16_flops_per_s"])
    fp8_flops = float(raw.get("fp8_flops_per_s", bf16_flops))
    int8_flops = float(raw.get("int8_flops_per_s", bf16_flops))
    native_fp8 = bool(raw.get("native_ocp_fp8", fp8_flops > bf16_flops))
    bracket_raw = raw.get("collective_latency_us_bracket")
    if type(bracket_raw) is list and len(bracket_raw) == _BRACKET_LEN:
        bracket = (float(bracket_raw[0]), float(bracket_raw[1]))
    else:
        bracket = (1.0, 5.0)
    if "collective_latency_s" in raw:
        lat_s = float(raw["collective_latency_s"])
    else:
        lat_s = float(raw.get("collective_latency_us", bracket[0])) * 1e-6
    dcn_br_raw = raw.get("dcn_bw_gbps_bracket")
    if type(dcn_br_raw) is list and len(dcn_br_raw) == _BRACKET_LEN:
        dcn_bracket = (float(dcn_br_raw[0]), float(dcn_br_raw[1]))
    else:
        dcn_bracket = (50.0, 200.0)
    tiles_raw = raw.get("matmul_row_tile_by_dtype")
    row_tiles: Mapping[str, int] = (
        {str(k).lower(): int(v) for k, v in tiles_raw.items()}
        if type(tiles_raw) is dict
        else (_ROW_TILES_7X if native_fp8 else _ROW_TILES_V6E)
    )
    topo_raw = raw.get("topologies")
    topologies: Mapping[int, str] = (
        {int(k): str(v) for k, v in topo_raw.items()} if type(topo_raw) is dict else {}
    )
    src_raw = raw.get("sources")
    sources: dict[str, str] = (
        {str(k): str(v) for k, v in src_raw.items()}
        if type(src_raw) is dict
        else {"spec": f"user --chip-spec {spec_path.name}"}
    )
    return ChipSpec(
        name=str(raw.get("name") or fallback_name),
        cores_per_device=int(raw.get("cores_per_device", 1)),
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_usable,
        hbm_capacity_bytes=int(raw["hbm_capacity_bytes"]),
        hbm_bw_bytes_per_s=float(raw["hbm_bw_bytes_per_s"]),
        bf16_flops_per_s=bf16_flops,
        fp8_flops_per_s=fp8_flops,
        int8_flops_per_s=int8_flops,
        native_ocp_fp8=native_fp8,
        ici_egress_bw_bytes_per_s=float(raw.get("ici_egress_bw_bytes_per_s", 300.0e9)),
        ici_link_bw_bytes_per_s=float(raw.get("ici_link_bw_bytes_per_s", 100.0e9)),
        collective_latency_s=lat_s,
        collective_latency_us_bracket=bracket,
        dcn_bw_gbps=float(raw.get("dcn_bw_gbps", 100.0)),
        dcn_bw_gbps_bracket=dcn_bracket,
        min_slice_cores=int(raw.get("min_slice_cores", 1)),
        devices_per_chip=int(raw.get("devices_per_chip", 1)),
        supports_megacore=bool(raw.get("supports_megacore", False)),
        matmul_row_tile_by_dtype=row_tiles,
        topologies=topologies,
        sources=sources,
    )


def _scale_megacore(
    base: ChipSpec,
    num_cores: int,
    ici_bw_gbps: float | None,
    collective_latency_us: float | None,
) -> ChipSpec:
    """Scale a single-core `ChipSpec` across `num_cores` and apply user overrides."""
    vmem_cap = base.vmem_capacity_bytes * num_cores
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
    return replace(
        base,
        name=f"{base.name}_megacore" if num_cores == _MEGACORE_CORES else base.name,
        cores_per_device=num_cores,
        vmem_capacity_bytes=vmem_cap,
        vmem_usable_bytes=vmem_cap - _USABLE_VMEM_RESERVE_BYTES,
        hbm_capacity_bytes=base.hbm_capacity_bytes * num_cores,
        hbm_bw_bytes_per_s=base.hbm_bw_bytes_per_s * num_cores,
        bf16_flops_per_s=base.bf16_flops_per_s * num_cores,
        fp8_flops_per_s=base.fp8_flops_per_s * num_cores,
        int8_flops_per_s=base.int8_flops_per_s * num_cores,
        ici_egress_bw_bytes_per_s=egress,
        collective_latency_s=lat_s,
        collective_latency_us_bracket=bracket,
    )


_CHIP_ALIASES: Mapping[str, str] = {
    "7x": "tpu7x",
    "v7x": "tpu7x",
    "tpu7x": "tpu7x",
    "6e": "v6e",
    "v6e": "v6e",
    "tpu6e": "v6e",
    "5p": "v5p",
    "v5p": "v5p",
    "tpu5p": "v5p",
    "5e": "v5e",
    "v5e": "v5e",
    "tpu5e": "v5e",
    "4": "v4",
    "v4": "v4",
    "tpu4": "v4",
}


def _detect_tpu_from_sysfs(
    pci_dir: Path | None = None,
) -> str | None:
    """Detect the host's TPU generation from PCI sysfs without opening PJRT."""
    root = _SYSFS_PCI_DEVICES_DIR if pci_dir is None else pci_dir
    if not root.is_dir():
        return None
    for vendor_file in sorted(root.glob("*/vendor")):
        try:
            vendor = vendor_file.read_text("utf-8").strip().lower()
            if vendor != _GOOGLE_PCI_VENDOR_ID:
                continue
            dev_id = (vendor_file.parent / "device").read_text("utf-8").strip().lower()
        except OSError:
            continue
        chip_name = _PCI_DEVICE_TO_CHIP.get(dev_id)
        if chip_name is not None:
            return chip_name
    return None


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
        return _scale_megacore(
            loaded, loaded.cores_per_device, ici_bw_gbps, collective_latency_us
        )
    raw = str(chip).strip().lower().removeprefix("tpu_")
    auto_cores: int | None = None
    if raw == "auto":
        sysfs_chip = _detect_tpu_from_sysfs()
        try:
            from jax.experimental.pallas import tpu as pltpu

            fn_mod = getattr(pltpu.get_tpu_info, "__module__", "")
            if sysfs_chip is not None and fn_mod.startswith("jax"):
                raw = sysfs_chip
            else:
                info = pltpu.get_tpu_info()
                raw = str(info.chip_version.value).lower()
                auto_cores = int(info.num_cores)
        except Exception as exc:
            if sysfs_chip is not None:
                raw = sysfs_chip
            else:
                msg = "chip='auto' requires TPU"
                raise errors.MegakernelConfigError(msg) from exc

    if raw.endswith("_megacore"):
        raw = raw.removesuffix("_megacore")
        megacore = True
    canon = _CHIP_ALIASES.get(raw, raw)
    if canon not in BUILTIN_CHIP_SPECS:
        req = list(REQUIRED_CHIP_SPEC_KEYS)
        known = sorted(BUILTIN_CHIP_SPECS)
        msg = (
            f"Unknown chip={chip!r}. Pass --chip-spec <path.json> with required keys "
            f"{req} or choose from {known}"
        )
        raise errors.MegakernelConfigError(msg)
    base = BUILTIN_CHIP_SPECS[canon]
    num_cores = auto_cores if auto_cores is not None else (1 + int(megacore))
    if num_cores == _MEGACORE_CORES and not base.supports_megacore:
        msg = f"Invalid chip={chip!r} or Megacore"
        raise errors.MegakernelConfigError(msg)
    return _scale_megacore(base, num_cores, ici_bw_gbps, collective_latency_us)


def verify_chip_spec_against_jax(spec: ChipSpec) -> list[str]:
    """Compare a built-in ChipSpec against jax.experimental.pallas.tpu if available."""
    try:
        from jax.experimental.pallas import tpu as pltpu
    except (ImportError, RuntimeError):
        return []
    base_name = spec.name.removesuffix("_megacore")
    cv_str = "7x" if base_name == "tpu7x" else base_name
    try:
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(cv_str), 1)
    except (ValueError, KeyError, RuntimeError, AttributeError):
        return []
    cores = spec.cores_per_device
    mismatches: list[str] = []
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
    )
    for field_name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{spec.name}.{field_name}: {actual} != {expected}")
    return mismatches
