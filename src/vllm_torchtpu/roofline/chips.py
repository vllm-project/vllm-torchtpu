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
"""Per-core hardware specifications, constants catalog, and 3D topologies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

from .constants import (
    BIN_KIBI,
    BIN_MEBI,
    BITS_PER_BYTE,
    CHIP_COUNT_BRACKET,
    DCN_BRACKET_GBIT_PER_S,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_D2D_BW,
    DEFAULT_DCN_BW,
    DEFAULT_DCN_HOST_GBIT_PER_S,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_MXU_LHS_ROWS_BF16,
    DEFAULT_MXU_LHS_ROWS_FP8,
    DEFAULT_MXU_M_TILE,
    DEFAULT_PCIE_BW,
    DEFAULT_VPU_TO_BF16_RATIO,
    FALLBACK_TPU_SPECS,
    FP8_BYTES,
    GPU_MEM_UTIL_BRACKET,
    ICI_HOP_LATENCY_BRACKET_S,
    MIN_TORUS_DIM,
    MXU_M_TILE_BRACKET,
    PAGE_TABLE_BRACKET_BYTES,
    PAGE_TABLE_BYTE_BUDGET,
    PCIE_BRACKET_BW,
    PERCENT_SCALE,
    RPA_PAGE_SIZE_BRACKET_TOKENS,
    RPA_PAGE_SIZE_TOKENS,
    TOPOLOGY_AXES,
    TPU7X_ICI_LINKS_PER_CHIP,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MILLI,
    UNIT_TERA,
    VPU_TO_BF16_RATIO_BRACKET,
)


class _TpuInfoProto(Protocol):
    """Structural protocol for JAX get_tpu_info_for_chip result."""

    @property
    def hbm_capacity_bytes(self) -> int:
        """Return per-core HBM capacity in bytes."""
        ...

    @property
    def mem_bw_bytes_per_second(self) -> float:
        """Return per-core HBM bandwidth in bytes per second."""
        ...

    @property
    def bf16_ops_per_second(self) -> float:
        """Return per-core BF16 peak FLOP/s."""
        ...

    @property
    def fp8_ops_per_second(self) -> float:
        """Return per-core FP8 peak FLOP/s."""
        ...

    @property
    def vmem_capacity_bytes(self) -> int:
        """Return per-core scratchpad capacity in bytes."""
        ...


class _PltpuProto(Protocol):
    """Structural protocol for jax.experimental.pallas.tpu module."""

    def get_tpu_info_for_chip(
        self, version: object, cores_per_chip: int
    ) -> _TpuInfoProto:
        """Query per-device TPU hardware limits from JAX."""
        ...


try:
    from jax.experimental.pallas import tpu as _pltpu
except ImportError:
    pltpu: _PltpuProto | None = None
    _TPU_7X_VERSION: object = "tpu7x"
else:
    pltpu = _pltpu
    _TPU_7X_VERSION = _pltpu.ChipVersion.TPU_7X

try:
    import jax as _jax
except ImportError:
    JAX_VERSION: str = "unavailable"
else:
    JAX_VERSION = _jax.__version__


@dataclass(frozen=True, slots=True)
class ConstantSpec:
    """Metadata for one hardware or solver constant in the roofline catalog."""

    value: float | int
    unit: str
    scope: str
    kind: str
    source: str
    bracket: tuple[float, float] | None = None
    display: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class ChipOverrides:
    """Optional ICI, DCN, PCIe, and tile overrides for ChipSpec construction."""

    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float | None = None
    ici_bw: float | None = None
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    mxu_m_tile: int = DEFAULT_MXU_M_TILE
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float | None = None
    pcie_bw: float = DEFAULT_PCIE_BW
    vpu_flops: float | None = None


@dataclass(frozen=True, slots=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore and its parent chip/host."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float = DEFAULT_D2D_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    mxu_m_tile: int = DEFAULT_MXU_M_TILE
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float = DEFAULT_DCN_BW
    pcie_bw: float = DEFAULT_PCIE_BW
    vpu_flops: float = 1.155e15 * DEFAULT_VPU_TO_BF16_RATIO
    mxu_lhs_rows_bf16: int = DEFAULT_MXU_LHS_ROWS_BF16
    mxu_lhs_rows_fp8: int = DEFAULT_MXU_LHS_ROWS_FP8

    def lhs_row_floor(self, dtype_bytes: float) -> int:
        """Return the systolic LHS row tile floor for the given datatype width."""
        return (
            self.mxu_lhs_rows_fp8
            if dtype_bytes <= FP8_BYTES
            else self.mxu_lhs_rows_bf16
        )

    def catalog(self) -> dict[str, dict[str, object]]:
        """Return the structured constants catalog for this ChipSpec."""
        return chip_catalog(self)


def _format_byte_count(v: float) -> str:
    """Format byte count using GB, MiB, KiB, or B units."""
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:.1f} GB"
    if v >= BIN_MEBI:
        return f"{v / BIN_MEBI:.1f} MiB"
    if v >= BIN_KIBI:
        return f"{v / BIN_KIBI:.1f} KiB"
    return f"{int(v)} B"


def _format_bandwidth_rate(v: float) -> str:
    """Format bandwidth in TB/s, GB/s, MB/s, or B/s."""
    if v >= UNIT_TERA:
        return f"{v / UNIT_TERA:.2f} TB/s"
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:.1f} GB/s"
    if v >= UNIT_MEGA:
        return f"{v / UNIT_MEGA:.1f} MB/s"
    return f"{v:.1f} B/s"


def _format_flops_rate(v: float) -> str:
    """Format compute throughput in TFLOP/s, GFLOP/s, or FLOP/s."""
    tflops = v / UNIT_TERA
    if tflops >= UNIT_KILO and tflops != int(tflops):
        return f"{tflops:,.1f} TFLOP/s"
    if tflops >= 1.0:
        return f"{tflops:,.0f} TFLOP/s"
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:,.0f} GFLOP/s"
    return f"{v:,.0f} FLOP/s"


def _format_duration_s(v: float) -> str:
    """Format duration in seconds using us, ms, or s units."""
    if v < UNIT_MILLI:
        return f"{v * UNIT_MEGA:.1f} us"
    if v < 1.0:
        return f"{v * UNIT_KILO:.1f} ms"
    return f"{v:.2f} s"


def format_constant_display(value: float, unit: str) -> str:
    """Format numeric constants into clean human-readable strings."""
    numeric_value = float(value)
    dispatch = {
        "B": _format_byte_count,
        "B/s": _format_bandwidth_rate,
        "FLOP/s": _format_flops_rate,
        "s": _format_duration_s,
    }
    if unit in dispatch:
        return dispatch[unit](numeric_value)
    if unit == "Gbit/s":
        return f"{numeric_value:.1f} Gbit/s"
    if unit == "ratio":
        return f"{numeric_value * PERCENT_SCALE:.0f}% ({numeric_value:.2f})"
    if unit in ("cores", "chips", "rows", "tokens"):
        return f"{int(value)} {unit}"
    return f"{value} {unit}".strip()


def _spec_dict(spec: ConstantSpec) -> dict[str, object]:
    """Convert ConstantSpec to an untyped object dictionary without Any."""
    return dict(asdict(spec).items())


def chip_catalog(chip: ChipSpec) -> dict[str, dict[str, object]]:
    """Build structured constants catalog with value, unit, scope, and source."""
    host_chips = float(max(1, chip.chips_per_host))
    dcn_low = (DCN_BRACKET_GBIT_PER_S[0] * UNIT_GIGA / BITS_PER_BYTE) / host_chips
    dcn_high = (DCN_BRACKET_GBIT_PER_S[1] * UNIT_GIGA / BITS_PER_BYTE) / host_chips
    entries: dict[str, ConstantSpec] = {
        "hbm_bytes": ConstantSpec(
            value=chip.hbm_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X hbm_capacity_bytes)",
            display=format_constant_display(chip.hbm_bytes, "B"),
            description="High-Bandwidth Memory (HBM) physical capacity per TensorCore",
        ),
        "hbm_bw": ConstantSpec(
            value=chip.hbm_bw,
            unit="B/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X mem_bw_bytes_per_second)",
            display=format_constant_display(chip.hbm_bw, "B/s"),
            description="Peak HBM memory bandwidth per TensorCore",
        ),
        "bf16_flops": ConstantSpec(
            value=chip.bf16_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X bf16_ops_per_second)",
            display=format_constant_display(chip.bf16_flops, "FLOP/s"),
            description="Peak BF16 matrix-multiply throughput per TensorCore",
        ),
        "fp8_flops": ConstantSpec(
            value=chip.fp8_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X fp8_ops_per_second)",
            display=format_constant_display(chip.fp8_flops, "FLOP/s"),
            description="Peak FP8 matrix-multiply throughput per TensorCore",
        ),
        "vmem_bytes": ConstantSpec(
            value=chip.vmem_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X vmem_capacity_bytes)",
            display=format_constant_display(chip.vmem_bytes, "B"),
            description="On-chip Vector Memory (VMEM / SRAM) capacity per TensorCore",
        ),
        "cores_per_chip": ConstantSpec(
            value=chip.cores_per_chip,
            unit="cores",
            scope="chip",
            kind="fact",
            source="jax._src.tpu_info num_physical_tensor_cores_per_chip",
            display=format_constant_display(chip.cores_per_chip, "cores"),
            description="Physical TensorCores sharing a single TPU chip",
        ),
        "ici_link_bw": ConstantSpec(
            value=chip.ici_link_bw,
            unit="B/s",
            scope="link",
            kind="derived",
            source=(
                "200 GB/s bidirectional link / 2 directions "
                "(cloud.google.com/tpu/docs/tpu7x)"
            ),
            display=format_constant_display(chip.ici_link_bw, "B/s"),
            description=(
                "Unidirectional bandwidth per optical Inter-Chip Interconnect "
                "(ICI) link"
            ),
        ),
        "d2d_bw": ConstantSpec(
            value=chip.d2d_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                "6 links * ici_link_bw across chip (cloud.google.com/tpu/docs/tpu7x)"
            ),
            display=format_constant_display(chip.d2d_bw, "B/s"),
            description=(
                "Aggregate on-chip die-to-die / all-link bandwidth across all 6 "
                "ICI links"
            ),
        ),
        "ici_bw": ConstantSpec(
            value=chip.ici_bw,
            unit="B/s",
            scope="core",
            kind="derived",
            source=(
                "d2d_bw / cores_per_chip per-core ICI share "
                "(cloud.google.com/tpu/docs/tpu7x)"
            ),
            display=format_constant_display(chip.ici_bw, "B/s"),
            description=(
                "Per-TensorCore fair share of aggregate ICI interconnect bandwidth"
            ),
        ),
        "dcn_bw": ConstantSpec(
            value=chip.dcn_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                "dcn_host_gbit_per_s * 1e9 / 8 / chips_per_host "
                "(400 Gbit/s / 8 / 4 chips)"
            ),
            bracket=(dcn_low, dcn_high),
            display=(
                f"{chip.dcn_bw / UNIT_GIGA:.1f} GB/s "
                f"({chip.dcn_bw * BITS_PER_BYTE / UNIT_GIGA:.0f} Gbit/s)"
            ),
            description=(
                "Per-chip slice of datacenter network (DCN) host NIC bandwidth "
                "for disaggregated KV transfer"
            ),
        ),
        "ici_hop_latency_s": ConstantSpec(
            value=chip.ici_hop_latency_s,
            unit="s",
            scope="link",
            kind="assumed",
            source=(
                "Assumed per-hop latency; no TPU7x figure is published. "
                "Used by the serial estimate only, never by the hard floor"
            ),
            bracket=ICI_HOP_LATENCY_BRACKET_S,
            display=format_constant_display(chip.ici_hop_latency_s, "s"),
            description=(
                "Unloaded point-to-point transmission latency per physical ICI "
                "torus hop"
            ),
        ),
        "pcie_bw": ConstantSpec(
            value=chip.pcie_bw,
            unit="B/s",
            scope="chip",
            kind="assumed",
            source=(
                "PCIe 5.0 x16 raw link rate (64 GB/s per chip); "
                "host-to-device effective rate uncalibrated"
            ),
            bracket=PCIE_BRACKET_BW,
            display=format_constant_display(chip.pcie_bw, "B/s"),
            description=(
                "Peak PCIe Gen 5 host-to-device interconnect bandwidth per chip"
            ),
        ),
        "mxu_m_tile": ConstantSpec(
            value=chip.mxu_m_tile,
            unit="rows",
            scope="core",
            kind="assumed",
            source=(
                "Assumed Pallas MoE kernel M-tile multiple on 256x256 MXU "
                "(jax._src.tpu_info mxu_column_size=256)"
            ),
            bracket=MXU_M_TILE_BRACKET,
            display=format_constant_display(chip.mxu_m_tile, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array row granularity "
                "tile floor"
            ),
        ),
        "dcn_host_gbit_per_s": ConstantSpec(
            value=chip.dcn_host_gbit_per_s,
            unit="Gbit/s",
            scope="host",
            kind="assumed",
            source=(
                "Public machine-type search summary (400 Gbit/s per 4-chip host); "
                "not verified offline"
            ),
            bracket=DCN_BRACKET_GBIT_PER_S,
            display=format_constant_display(chip.dcn_host_gbit_per_s, "Gbit/s"),
            description=(
                "Total host-level network interface card (NIC) bandwidth shared "
                "across host chips"
            ),
        ),
        "chips_per_host": ConstantSpec(
            value=chip.chips_per_host,
            unit="chips",
            scope="host",
            kind="assumed",
            source="Public machine-type search summary (4 TPU7x chips per host)",
            bracket=CHIP_COUNT_BRACKET,
            display=format_constant_display(chip.chips_per_host, "chips"),
            description=(
                "Number of TPU chips co-located on a single host tray / CPU subsystem"
            ),
        ),
        "page_table_byte_budget": ConstantSpec(
            value=PAGE_TABLE_BYTE_BUDGET,
            unit="B",
            scope="core",
            kind="assumed",
            source=(
                "Assumed on-chip memory page-table scratch budget "
                "(512 KiB per core; vllm_torchtpu.roofline.memory)"
            ),
            bracket=PAGE_TABLE_BRACKET_BYTES,
            display=format_constant_display(PAGE_TABLE_BYTE_BUDGET, "B"),
            description=(
                "Reserved on-chip fast memory scratchpad budget for Pallas "
                "block-table lookups"
            ),
        ),
        "rpa_page_size_tokens": ConstantSpec(
            value=RPA_PAGE_SIZE_TOKENS,
            unit="tokens",
            scope="core",
            kind="assumed",
            source=(
                "Assumed default RPA KV page granularity constant "
                "(64 tokens; vllm_torchtpu.roofline.constants)"
            ),
            bracket=RPA_PAGE_SIZE_BRACKET_TOKENS,
            display=format_constant_display(RPA_PAGE_SIZE_TOKENS, "tokens"),
            description=(
                "Paged KV-cache token page granularity for Ring-Paged-Attention (RPA)"
            ),
        ),
        "default_gpu_memory_utilization": ConstantSpec(
            value=DEFAULT_GPU_MEMORY_UTILIZATION,
            unit="ratio",
            scope="core",
            kind="config",
            source=(
                "Roofline default --gpu-memory-utilization "
                "(vLLM CacheConfig default is 0.90)"
            ),
            bracket=GPU_MEM_UTIL_BRACKET,
            display=format_constant_display(DEFAULT_GPU_MEMORY_UTILIZATION, "ratio"),
            description=(
                "Default peak HBM memory allocation fraction reserved for weights "
                "and KV cache"
            ),
        ),
        "vpu_flops": ConstantSpec(
            value=chip.vpu_flops,
            unit="FLOP/s",
            scope="core",
            kind="derived",
            source=(
                "8 x 128 lane-sublane pairs x 4 ops per cycle x 2.2e9 Hz = 9e12 "
                "32-bit word ops/s per core (jax-ml.github.io/scaling-book/tpus); "
                "packed elements run up to 4 per word"
            ),
            bracket=(
                chip.bf16_flops * VPU_TO_BF16_RATIO_BRACKET[0],
                chip.bf16_flops * VPU_TO_BF16_RATIO_BRACKET[1],
            ),
            display=format_constant_display(chip.vpu_flops, "FLOP/s"),
            description=(
                "Peak vector processing unit (VPU) arithmetic throughput per TensorCore"
            ),
        ),
        "mxu_lhs_rows_bf16": ConstantSpec(
            value=chip.mxu_lhs_rows_bf16,
            unit="rows",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info get_sublane_tiling",
            display=format_constant_display(chip.mxu_lhs_rows_bf16, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array "
                "LHS row floor for BF16"
            ),
        ),
        "mxu_lhs_rows_fp8": ConstantSpec(
            value=chip.mxu_lhs_rows_fp8,
            unit="rows",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info get_sublane_tiling",
            display=format_constant_display(chip.mxu_lhs_rows_fp8, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array "
                "LHS row floor for FP8"
            ),
        ),
    }
    return {
        entry_name: _spec_dict(constant_spec)
        for entry_name, constant_spec in entries.items()
    }


def from_jax(  # noqa: PLR0913
    name: str,
    version: object,
    overrides: ChipOverrides | None = None,
    *,
    cores_per_chip: int | None = None,
    ici_link_bw: float | None = None,
    ici_bw: float | None = None,
    ici_hop_latency_s: float | None = None,
) -> ChipSpec:
    """Construct ChipSpec from JAX TPU info table or stdlib fallback table."""
    overrides_cfg = overrides or ChipOverrides()
    if (
        cores_per_chip is not None
        or ici_link_bw is not None
        or ici_bw is not None
        or ici_hop_latency_s is not None
    ):
        overrides_cfg = ChipOverrides(
            cores_per_chip=cores_per_chip
            if cores_per_chip is not None
            else overrides_cfg.cores_per_chip,
            ici_link_bw=ici_link_bw
            if ici_link_bw is not None
            else overrides_cfg.ici_link_bw,
            ici_bw=ici_bw if ici_bw is not None else overrides_cfg.ici_bw,
            ici_hop_latency_s=ici_hop_latency_s
            if ici_hop_latency_s is not None
            else overrides_cfg.ici_hop_latency_s,
            d2d_bw=overrides_cfg.d2d_bw,
            mxu_m_tile=overrides_cfg.mxu_m_tile,
            dcn_host_gbit_per_s=overrides_cfg.dcn_host_gbit_per_s,
            chips_per_host=overrides_cfg.chips_per_host,
            dcn_bw=overrides_cfg.dcn_bw,
            pcie_bw=overrides_cfg.pcie_bw,
            vpu_flops=overrides_cfg.vpu_flops,
        )
    eff_d2d_bw = (
        overrides_cfg.d2d_bw
        if overrides_cfg.d2d_bw is not None
        else float(TPU7X_ICI_LINKS_PER_CHIP) * overrides_cfg.ici_link_bw
    )
    eff_ici_bw = (
        overrides_cfg.ici_bw
        if overrides_cfg.ici_bw is not None
        else eff_d2d_bw / float(max(1, overrides_cfg.cores_per_chip))
    )
    eff_dcn_bw = (
        overrides_cfg.dcn_bw
        if overrides_cfg.dcn_bw is not None
        else (overrides_cfg.dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE)
        / float(max(1, overrides_cfg.chips_per_host))
    )
    if pltpu is None or isinstance(version, str):
        if name not in FALLBACK_TPU_SPECS:
            msg = f"Unknown fallback TPU chip {name!r}"
            raise KeyError(msg)
        hbm_b, hbm_bw, bf16, fp8_flops, vmem_b = FALLBACK_TPU_SPECS[name]
    else:
        # Querying with logical cores_per_chip=1 returns per-TensorCore HBM,
        # bandwidth, MXU peaks, and VMEM; ChipOverrides supplies ICI/DCN/PCIe rates.
        info = pltpu.get_tpu_info_for_chip(version, 1)
        hbm_b = info.hbm_capacity_bytes
        hbm_bw = float(info.mem_bw_bytes_per_second)
        bf16 = float(info.bf16_ops_per_second)
        fp8_flops = float(info.fp8_ops_per_second)
        vmem_b = info.vmem_capacity_bytes
    eff_vpu_flops = (
        overrides_cfg.vpu_flops
        if overrides_cfg.vpu_flops is not None
        else bf16 * DEFAULT_VPU_TO_BF16_RATIO
    )
    return ChipSpec(
        name=name,
        hbm_bytes=hbm_b,
        hbm_bw=hbm_bw,
        bf16_flops=bf16,
        fp8_flops=fp8_flops,
        vmem_bytes=vmem_b,
        cores_per_chip=overrides_cfg.cores_per_chip,
        ici_link_bw=overrides_cfg.ici_link_bw,
        d2d_bw=eff_d2d_bw,
        ici_bw=eff_ici_bw,
        ici_hop_latency_s=overrides_cfg.ici_hop_latency_s,
        mxu_m_tile=overrides_cfg.mxu_m_tile,
        dcn_host_gbit_per_s=overrides_cfg.dcn_host_gbit_per_s,
        chips_per_host=overrides_cfg.chips_per_host,
        dcn_bw=eff_dcn_bw,
        pcie_bw=overrides_cfg.pcie_bw,
        vpu_flops=eff_vpu_flops,
        mxu_lhs_rows_bf16=DEFAULT_MXU_LHS_ROWS_BF16,
        mxu_lhs_rows_fp8=DEFAULT_MXU_LHS_ROWS_FP8,
    )


@dataclass(frozen=True, slots=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    wrapped_axes: tuple[bool, bool, bool] | None = None

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the 3D topology."""
        axis_x, axis_y, axis_z = self.chip_axes
        return axis_x * axis_y * axis_z

    @property
    def num_hosts(self) -> int:
        """Return total physical host machines spanned by this slice topology."""
        return max(
            1, (self.num_chips + DEFAULT_CHIPS_PER_HOST - 1) // DEFAULT_CHIPS_PER_HOST
        )

    @property
    def num_cores(self) -> int:
        """Total number of TensorCores across all chips in the topology."""
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        """True when physical chip axes have torus wraparounds."""
        if self.wrapped_axes is not None:
            return any(self.wrapped_axes)
        # Google Cloud TPU7x doc slice table (https://cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4".
        return all(
            dimension_size >= MIN_TORUS_DIM and dimension_size % MIN_TORUS_DIM == 0
            for dimension_size in self.chip_axes
        )

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        """True when a collective spans the full extent of a wrapped physical axis."""
        if self.wrapped_axes is not None:
            return (
                self.wrapped_axes[axis]
                and submesh_extent == self.chip_axes[axis]
                and submesh_extent > 1
            )
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= MIN_TORUS_DIM
        )


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": from_jax(
        "tpu7x",
        _TPU_7X_VERSION,
        ChipOverrides(),
    ),
}

SUPPORTED_CHIPS: frozenset[str] = frozenset(CHIPS)

TOPOLOGIES: dict[str, SliceTopology] = {
    name: SliceTopology(name, axes) for name, axes in TOPOLOGY_AXES.items()
}
