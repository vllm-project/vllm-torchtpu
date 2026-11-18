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
from dataclasses import asdict, dataclass, replace
from typing import TypedDict, Unpack

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .constants import (
    BIN_KIBI,
    BIN_MEBI,
    BITS_PER_BYTE,
    CHIP_COUNT_BRACKET,
    DCN_BRACKET_GBIT_PER_S,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_D2D_BW,
    DEFAULT_DCN_BW,
    DEFAULT_DCN_HOST_GBIT_PER_S,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_PCIE_BW,
    DEFAULT_VECTOR_TO_BF16_RATIO,
    GPU_MEM_UTIL_BRACKET,
    ICI_HOP_LATENCY_BRACKET_S,
    MIN_TORUS_DIM,
    MXU_M_TILE_BRACKET,
    PAGE_TABLE_BRACKET_BYTES,
    PCIE_BRACKET_BW,
    PERCENT_SCALE,
    RPA_PAGE_SIZE_BRACKET_TOKENS,
    RPA_PAGE_SIZE_TOKENS,
    SMEM_PAGE_TABLE_DIVISOR,
    TOPOLOGY_AXES,
    TPU7X_ICI_LINKS_PER_CHIP,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MILLI,
    UNIT_TERA,
    VECTOR_TO_BF16_RATIO_BRACKET,
)

_TORUS_AXES_LEN: int = 3


_TPU_7X_VERSION: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X
JAX_VERSION: str = jax.__version__


@dataclass(frozen=True, slots=True)
class _JaxChipCapabilities:
    """Raw hardware capability fields read from JAX TPU info tables."""

    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    cores_per_chip: int
    mxu_m_tile: int
    mxu_lhs_rows_bf16: int
    mxu_lhs_rows_fp8: int
    num_lanes: int
    num_sublanes: int
    physical_alignment_elems: int
    smem_page_table_bytes: int


def _read_jax_capabilities(name: str, version: object) -> _JaxChipCapabilities:
    """Read per-TensorCore and chip capabilities directly from JAX."""
    if not isinstance(version, pltpu.ChipVersion):
        msg = (
            f"Chip {name!r} requires a JAX pltpu.ChipVersion, "
            f"got {type(version).__name__}"
        )
        raise TypeError(msg)
    # Querying with logical cores_per_chip=1 returns per-TensorCore HBM,
    # bandwidth, MXU peaks, VMEM, SMEM, and sublane tiling directly from JAX.
    info = pltpu.get_tpu_info_for_chip(version, 1)
    num_lanes = int(info.num_lanes)
    return _JaxChipCapabilities(
        hbm_bytes=int(info.hbm_capacity_bytes),
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=int(info.vmem_capacity_bytes),
        cores_per_chip=int(version.num_physical_tensor_cores_per_chip),
        mxu_m_tile=int(info.mxu_column_size),
        mxu_lhs_rows_bf16=int(info.get_sublane_tiling(jnp.bfloat16)),
        mxu_lhs_rows_fp8=int(info.get_sublane_tiling(jnp.float8_e4m3fn)),
        num_lanes=num_lanes,
        num_sublanes=int(info.num_sublanes),
        physical_alignment_elems=num_lanes,
        smem_page_table_bytes=int(info.smem_capacity_bytes) // SMEM_PAGE_TABLE_DIVISOR,
    )


_DEFAULT_TPU_CAPS: _JaxChipCapabilities = _read_jax_capabilities(
    "tpu7x", _TPU_7X_VERSION
)


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
    cores_per_chip: int | None = None
    mxu_m_tile: int | None = None
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float | None = None
    pcie_bw: float = DEFAULT_PCIE_BW
    vector_flops: float | None = None


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
    cores_per_chip: int = _DEFAULT_TPU_CAPS.cores_per_chip
    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float = DEFAULT_D2D_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    mxu_m_tile: int = _DEFAULT_TPU_CAPS.mxu_m_tile
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float = DEFAULT_DCN_BW
    pcie_bw: float = DEFAULT_PCIE_BW
    vector_flops: float = _DEFAULT_TPU_CAPS.bf16_flops * DEFAULT_VECTOR_TO_BF16_RATIO
    mxu_lhs_rows_bf16: int = _DEFAULT_TPU_CAPS.mxu_lhs_rows_bf16
    mxu_lhs_rows_fp8: int = _DEFAULT_TPU_CAPS.mxu_lhs_rows_fp8
    num_lanes: int = _DEFAULT_TPU_CAPS.num_lanes
    num_sublanes: int = _DEFAULT_TPU_CAPS.num_sublanes
    physical_alignment_elems: int = _DEFAULT_TPU_CAPS.physical_alignment_elems
    smem_page_table_bytes: int = _DEFAULT_TPU_CAPS.smem_page_table_bytes

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
                "Dedicated die-to-die bridge approximated numerically as "
                "6 * ici_link_bw (cloud.google.com/tpu/docs/tpu7x)"
            ),
            display=format_constant_display(chip.d2d_bw, "B/s"),
            description=(
                "Dedicated on-chip die-to-die bridge bandwidth (approximated "
                "numerically as 6 * ici_link_bw)"
            ),
        ),
        "ici_bw": ConstantSpec(
            value=chip.ici_bw,
            unit="B/s",
            scope="core",
            kind="derived",
            source=(
                "6 * ici_link_bw / cores_per_chip per-core ICI share "
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
            value=chip.smem_page_table_bytes,
            unit="B",
            scope="core",
            kind="assumed",
            source=(
                "Reserved on-chip SMEM page-table scratch budget "
                "(smem_capacity_bytes // 2 per core; jax._src.tpu_info)"
            ),
            bracket=PAGE_TABLE_BRACKET_BYTES,
            display=format_constant_display(chip.smem_page_table_bytes, "B"),
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
            source=("Roofline default 0.92 (vLLM CacheConfig default is 0.90)"),
            bracket=GPU_MEM_UTIL_BRACKET,
            display=format_constant_display(DEFAULT_GPU_MEMORY_UTILIZATION, "ratio"),
            description=(
                "Default peak HBM memory allocation fraction reserved for weights "
                "and KV cache"
            ),
        ),
        "vector_flops": ConstantSpec(
            value=chip.vector_flops,
            unit="FLOP/s",
            scope="core",
            kind="derived",
            source=(
                "bf16_flops / 128 (DEFAULT_VECTOR_TO_BF16_RATIO = 1/128 of per-core "
                "BF16 peak, jax-ml.github.io/scaling-book/tpus); 32-bit word ops/s "
                "per core, with up to 4 packed elements per word"
            ),
            bracket=(
                chip.bf16_flops * VECTOR_TO_BF16_RATIO_BRACKET[0],
                chip.bf16_flops * VECTOR_TO_BF16_RATIO_BRACKET[1],
            ),
            display=format_constant_display(chip.vector_flops, "FLOP/s"),
            description=("Peak vector unit arithmetic throughput per TensorCore"),
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


class _FromJaxOverrides(TypedDict, total=False):
    """Optional keyword overrides accepted by from_jax."""

    cores_per_chip: int | None
    ici_link_bw: float | None
    ici_bw: float | None
    ici_hop_latency_s: float | None


def from_jax(
    name: str,
    version: object,
    overrides: ChipOverrides | None = None,
    **legacy_kwargs: Unpack[_FromJaxOverrides],
) -> ChipSpec:
    """Construct ChipSpec directly from JAX TPU capability metadata."""
    overrides_cfg = overrides or ChipOverrides()
    cores_per_chip = legacy_kwargs.get("cores_per_chip")
    ici_link_bw = legacy_kwargs.get("ici_link_bw")
    ici_bw = legacy_kwargs.get("ici_bw")
    ici_hop_latency_s = legacy_kwargs.get("ici_hop_latency_s")
    if (
        cores_per_chip is not None
        or ici_link_bw is not None
        or ici_bw is not None
        or ici_hop_latency_s is not None
    ):
        overrides_cfg = replace(
            overrides_cfg,
            cores_per_chip=(
                cores_per_chip
                if cores_per_chip is not None
                else overrides_cfg.cores_per_chip
            ),
            ici_link_bw=(
                ici_link_bw if ici_link_bw is not None else overrides_cfg.ici_link_bw
            ),
            ici_bw=ici_bw if ici_bw is not None else overrides_cfg.ici_bw,
            ici_hop_latency_s=(
                ici_hop_latency_s
                if ici_hop_latency_s is not None
                else overrides_cfg.ici_hop_latency_s
            ),
        )
    caps = _read_jax_capabilities(name, version)
    eff_cores = (
        overrides_cfg.cores_per_chip
        if overrides_cfg.cores_per_chip is not None
        else caps.cores_per_chip
    )
    eff_d2d_bw = (
        overrides_cfg.d2d_bw
        if overrides_cfg.d2d_bw is not None
        else float(TPU7X_ICI_LINKS_PER_CHIP) * overrides_cfg.ici_link_bw
    )
    eff_ici_bw = (
        overrides_cfg.ici_bw
        if overrides_cfg.ici_bw is not None
        else (
            float(TPU7X_ICI_LINKS_PER_CHIP)
            * overrides_cfg.ici_link_bw
            / float(max(1, eff_cores))
        )
    )
    eff_dcn_bw = (
        overrides_cfg.dcn_bw
        if overrides_cfg.dcn_bw is not None
        else (overrides_cfg.dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE)
        / float(max(1, overrides_cfg.chips_per_host))
    )
    eff_vector_flops = (
        overrides_cfg.vector_flops
        if overrides_cfg.vector_flops is not None
        else caps.bf16_flops * DEFAULT_VECTOR_TO_BF16_RATIO
    )
    return ChipSpec(
        name=name,
        hbm_bytes=caps.hbm_bytes,
        hbm_bw=caps.hbm_bw,
        bf16_flops=caps.bf16_flops,
        fp8_flops=caps.fp8_flops,
        vmem_bytes=caps.vmem_bytes,
        cores_per_chip=eff_cores,
        ici_link_bw=overrides_cfg.ici_link_bw,
        d2d_bw=eff_d2d_bw,
        ici_bw=eff_ici_bw,
        ici_hop_latency_s=overrides_cfg.ici_hop_latency_s,
        mxu_m_tile=(
            overrides_cfg.mxu_m_tile
            if overrides_cfg.mxu_m_tile is not None
            else caps.mxu_m_tile
        ),
        dcn_host_gbit_per_s=overrides_cfg.dcn_host_gbit_per_s,
        chips_per_host=overrides_cfg.chips_per_host,
        dcn_bw=eff_dcn_bw,
        pcie_bw=overrides_cfg.pcie_bw,
        vector_flops=eff_vector_flops,
        mxu_lhs_rows_bf16=caps.mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=caps.mxu_lhs_rows_fp8,
        num_lanes=caps.num_lanes,
        num_sublanes=caps.num_sublanes,
        physical_alignment_elems=caps.physical_alignment_elems,
        smem_page_table_bytes=caps.smem_page_table_bytes,
    )


@dataclass(frozen=True, slots=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = _DEFAULT_TPU_CAPS.cores_per_chip
    wrapped_axes: tuple[bool, bool, bool] | None = None

    def __post_init__(self) -> None:
        """Validate 3D axis and wraparound tuple lengths."""
        if len(self.chip_axes) != _TORUS_AXES_LEN:
            msg = (
                f"chip_axes must have length {_TORUS_AXES_LEN}, "
                f"got {len(self.chip_axes)}"
            )
            raise ValueError(msg)
        if self.wrapped_axes is not None and len(self.wrapped_axes) != _TORUS_AXES_LEN:
            msg = (
                f"wrapped_axes must have length {_TORUS_AXES_LEN}, "
                f"got {len(self.wrapped_axes)}"
            )
            raise ValueError(msg)

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the 3D topology."""
        axis_x, axis_y, axis_z = self.chip_axes
        return axis_x * axis_y * axis_z

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
