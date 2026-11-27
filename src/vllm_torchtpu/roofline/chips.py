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

import json
import math
from collections.abc import KeysView, Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .constants import (
    BF16_BYTES,
    BIN_KIBI,
    BIN_MEBI,
    BITS_PER_BYTE,
    CHIP_COUNT_BRACKET,
    CHIP_SPEC_SCHEMA,
    D2D_BW_BRACKET,
    DCN_BRACKET_GBIT_PER_S,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_D2D_BW,
    DEFAULT_D2D_TO_ICI_LINK_MULT,
    DEFAULT_DCN_BW,
    DEFAULT_DCN_HOST_GBIT_PER_S,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_ICI_LINKS_PER_CHIP,
    DEFAULT_PCIE_BW,
    DEFAULT_VECTOR_TO_BF16_RATIO,
    ICI_HOP_LATENCY_BRACKET_S,
    MIN_TORUS_DIM,
    MXU_M_TILE_BRACKET,
    PCIE_BRACKET_BW,
    PERCENT_SCALE,
    SOLVER_ASSUMPTIONS,
    SUBCHANNEL_BLOCK_K_FLOOR,
    TOPOLOGY_AXES,
    TPU7X_FP8_FLOPS_PER_CORE_BRACKET,
    TPU7X_HBM_BYTES_PER_CORE_MAX,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICRO,
    UNIT_MILLI,
    UNIT_TERA,
    VALID_CHIP_PROVENANCE,
    VECTOR_TO_BF16_RATIO_BRACKET,
)

_TORUS_AXES_LEN: int = 3
_TPU_7X_VERSION: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X
JAX_VERSION: str = jax.__version__


@dataclass(frozen=True, slots=True)
class _JaxChipCapabilities:
    """Raw hardware capability fields read directly from JAX TPU info tables."""

    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    cores_per_chip: int
    int8_flops: float
    int4_flops: float
    mxu_m_tile: int
    mxu_lhs_rows_bf16: int
    mxu_lhs_rows_fp8: int
    num_lanes: int
    num_sublanes: int
    physical_alignment_elems: int
    dma_granule_bytes: int
    has_sparse_core: bool
    provenance: str


def _read_jax_capabilities(
    name: str, version: pltpu.ChipVersion
) -> _JaxChipCapabilities:
    """Read per-TensorCore and chip capabilities directly from JAX."""
    if not isinstance(version, pltpu.ChipVersion):
        msg = (
            f"Chip {name!r} requires a JAX pltpu.ChipVersion, "
            f"got {type(version).__name__}"
        )
        raise TypeError(msg)
    # Querying with logical cores_per_chip=1 returns per-TensorCore HBM,
    # bandwidth, matrix-unit peaks, VMEM, sublane tiling, and SparseCore info.
    info = pltpu.get_tpu_info_for_chip(version, 1)
    num_lanes = int(info.num_lanes)
    sparse_core = info.sparse_core
    has_sparse_core = sparse_core is not None
    dma_granule = (
        int(sparse_core.dma_granule_size_bytes) if sparse_core is not None else 0
    )
    return _JaxChipCapabilities(
        hbm_bytes=int(info.hbm_capacity_bytes),
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=int(info.vmem_capacity_bytes),
        cores_per_chip=int(version.num_physical_tensor_cores_per_chip),
        int8_flops=float(info.int8_ops_per_second),
        int4_flops=float(info.int4_ops_per_second),
        mxu_m_tile=int(info.mxu_column_size),
        mxu_lhs_rows_bf16=int(info.get_sublane_tiling(jnp.bfloat16)),
        mxu_lhs_rows_fp8=int(info.get_sublane_tiling(jnp.float8_e4m3fn)),
        num_lanes=num_lanes,
        num_sublanes=int(info.num_sublanes),
        physical_alignment_elems=num_lanes,
        dma_granule_bytes=dma_granule,
        has_sparse_core=has_sparse_core,
        provenance="jax_tpu_info",
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
    """Optional ICI, DCN, PCIe, tile, and capability overrides for ChipSpec."""

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
    hbm_bytes: int | None = None
    hbm_bw: float | None = None
    bf16_flops: float | None = None
    fp8_flops: float | None = None
    int8_flops: float | None = None
    fp4_flops: float | None = None
    int4_flops: float | None = None
    vmem_bytes: int | None = None
    ici_links_per_chip: int | None = None
    min_torus_dim: int | None = None
    num_lanes: int | None = None
    num_sublanes: int | None = None
    vector_to_bf16_ratio: float | None = None
    mxu_lhs_rows_bf16: int | None = None
    mxu_lhs_rows_fp8: int | None = None
    physical_alignment_elems: int | None = None
    dma_granule_bytes: int | None = None
    subchannel_block_k_floor: int | None = None
    has_sparse_core: bool | None = None
    provenance: str | None = None


def override_chip_ici_bw(chip: ChipSpec, ici_bw_per_core: float) -> ChipSpec:
    """Override per-core ICI bandwidth and derive per-link bandwidth cleanly.

    Leaves ``d2d_bw`` unchanged because the intra-chip die-to-die bridge is an
    independent physical link.
    """
    if not math.isfinite(ici_bw_per_core) or ici_bw_per_core <= 0.0:
        msg = f"ici_bw_per_core must be finite and > 0, got {ici_bw_per_core!r}"
        raise ValueError(msg)
    ici_link_bw = (
        ici_bw_per_core
        * float(max(1, chip.cores_per_chip))
        / float(max(1, chip.ici_links_per_chip))
    )
    return replace(chip, ici_bw=ici_bw_per_core, ici_link_bw=ici_link_bw)


@lru_cache(maxsize=256)
def _cached_matmul_peak_kind(
    bf16_flops: float,
    fp8_flops: float,
    int8_flops: float,
    fp4_flops: float,
    int4_flops: float,
    weight_bytes: float,
    act: str,
    weight_dtype: str,
) -> tuple[str, float, bool]:
    """Select matrix-unit precision bucket, peak rate, and unpack flag.

    Both ``a4`` and ``a8`` activations can execute on FP4/FP8 matrix paths
    (FP4 values embed cleanly into FP8 operands when native FP4 throughput is
    absent). Integer matrix peaks (``int4`` and ``int8``) remain gated on
    ``act == "a8"``: emulated ``int8`` falls back directly to BF16, while
    emulated ``int4`` uses the sub-byte fallback path.
    """
    dtype_norm = weight_dtype.strip().lower()
    is_fp4 = (
        dtype_norm in ("nvfp4", "mxfp4", "fp4")
        or "fp4" in dtype_norm
        or (not dtype_norm and weight_bytes < 1.0)
    )
    if "int4" in dtype_norm:
        if int4_flops > 0.0 and act == "a8":
            return ("int4", int4_flops, False)
    elif is_fp4:
        if fp4_flops > 0.0 and act in ("a4", "a8"):
            return ("fp4", fp4_flops, False)
    elif "int8" in dtype_norm:
        if int8_flops > 0.0 and act == "a8":
            return ("int8", int8_flops, False)
        return ("bf16", bf16_flops, False)
    unpack = weight_bytes < 1.0
    if weight_bytes < BF16_BYTES and act in ("a4", "a8") and fp8_flops > 0.0:
        return ("fp8", fp8_flops, unpack)
    return ("bf16", bf16_flops, unpack)


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
    int8_flops: float = 0.0
    fp4_flops: float = 0.0
    int4_flops: float = 0.0
    ici_links_per_chip: int = DEFAULT_ICI_LINKS_PER_CHIP
    min_torus_dim: int = MIN_TORUS_DIM
    num_lanes: int = _DEFAULT_TPU_CAPS.num_lanes
    num_sublanes: int = _DEFAULT_TPU_CAPS.num_sublanes
    physical_alignment_elems: int = _DEFAULT_TPU_CAPS.physical_alignment_elems
    dma_granule_bytes: int = _DEFAULT_TPU_CAPS.dma_granule_bytes
    subchannel_block_k_floor: int = SUBCHANNEL_BLOCK_K_FLOOR
    has_sparse_core: bool = _DEFAULT_TPU_CAPS.has_sparse_core
    provenance: str = "public_spec"
    vector_flops_overridden: bool = False
    dcn_bw_overridden: bool = False

    def matmul_peak_kind(
        self,
        weight_bytes: float,
        act: str = "a8",
        *,
        weight_dtype: str = "",
    ) -> tuple[str, float, bool]:
        """Select matrix-unit precision bucket, peak rate, and unpack requirement."""
        return _cached_matmul_peak_kind(
            self.bf16_flops,
            self.fp8_flops,
            self.int8_flops,
            self.fp4_flops,
            self.int4_flops,
            weight_bytes,
            act,
            weight_dtype,
        )

    def catalog(self) -> dict[str, dict[str, object]]:
        """Return the structured constants catalog for this ChipSpec."""
        return chip_catalog(self)


def _format_byte_count(value: float) -> str:
    """Format byte count using GB, MiB, KiB, or B units."""
    if value >= UNIT_GIGA:
        return f"{value / UNIT_GIGA:.1f} GB"
    if value >= BIN_MEBI:
        return f"{value / BIN_MEBI:.1f} MiB"
    if value >= BIN_KIBI:
        return f"{value / BIN_KIBI:.1f} KiB"
    return f"{int(value)} B"


def _format_bandwidth_rate(value: float) -> str:
    """Format bandwidth in TB/s, GB/s, MB/s, or B/s."""
    if value >= UNIT_TERA:
        return f"{value / UNIT_TERA:.2f} TB/s"
    if value >= UNIT_GIGA:
        return f"{value / UNIT_GIGA:.1f} GB/s"
    if value >= UNIT_MEGA:
        return f"{value / UNIT_MEGA:.1f} MB/s"
    return f"{value:.1f} B/s"


def _format_flops_rate(value: float) -> str:
    """Format compute throughput in TFLOP/s, GFLOP/s, or FLOP/s."""
    tflops = value / UNIT_TERA
    if tflops >= UNIT_KILO and tflops != int(tflops):
        return f"{tflops:,.1f} TFLOP/s"
    if tflops >= 1.0:
        return f"{tflops:,.0f} TFLOP/s"
    if value >= UNIT_GIGA:
        return f"{value / UNIT_GIGA:,.0f} GFLOP/s"
    return f"{value:,.0f} FLOP/s"


def _format_duration_s(value: float) -> str:
    """Format duration in seconds using us, ms, or s units."""
    if value < UNIT_MILLI:
        return f"{value * UNIT_MEGA:.1f} us"
    if value < 1.0:
        return f"{value * UNIT_KILO:.1f} ms"
    return f"{value:.2f} s"


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
    src_prefix = f"{chip.name} ({chip.provenance})"
    is_tpu7x = chip.name == "tpu7x"
    entries: dict[str, ConstantSpec] = {
        "hbm_bytes": ConstantSpec(
            value=chip.hbm_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core HBM capacity",
            bracket=(
                (float(chip.hbm_bytes), float(TPU7X_HBM_BYTES_PER_CORE_MAX))
                if is_tpu7x
                else None
            ),
            display=format_constant_display(chip.hbm_bytes, "B"),
            description="High-Bandwidth Memory (HBM) physical capacity per TensorCore",
        ),
        "hbm_bw": ConstantSpec(
            value=chip.hbm_bw,
            unit="B/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core HBM bandwidth",
            display=format_constant_display(chip.hbm_bw, "B/s"),
            description="Peak HBM memory bandwidth per TensorCore",
        ),
        "bf16_flops": ConstantSpec(
            value=chip.bf16_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core BF16 matrix-unit peak",
            display=format_constant_display(chip.bf16_flops, "FLOP/s"),
            description="Peak BF16 matrix-multiply throughput per TensorCore",
        ),
        "fp8_flops": ConstantSpec(
            value=chip.fp8_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core FP8 matrix-unit peak",
            bracket=TPU7X_FP8_FLOPS_PER_CORE_BRACKET if is_tpu7x else None,
            display=format_constant_display(chip.fp8_flops, "FLOP/s"),
            description="Peak FP8 matrix-multiply throughput per TensorCore",
        ),
        "vmem_bytes": ConstantSpec(
            value=chip.vmem_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core VMEM scratchpad capacity",
            display=format_constant_display(chip.vmem_bytes, "B"),
            description="On-chip Vector Memory (VMEM / SRAM) capacity per TensorCore",
        ),
        "cores_per_chip": ConstantSpec(
            value=chip.cores_per_chip,
            unit="cores",
            scope="chip",
            kind="fact",
            source=f"{src_prefix}: physical TensorCores per chip",
            display=format_constant_display(chip.cores_per_chip, "cores"),
            description="Physical TensorCores sharing a single TPU chip",
        ),
        "ici_link_bw": ConstantSpec(
            value=chip.ici_link_bw,
            unit="B/s",
            scope="link",
            kind="derived",
            source=f"{src_prefix}: unidirectional ICI link bandwidth",
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
                f"{src_prefix}: dedicated bridge modeled as "
                f"{DEFAULT_D2D_TO_ICI_LINK_MULT:g} * ici_link_bw"
                + (
                    " (https://docs.cloud.google.com/tpu/docs/tpu7x)"
                    if is_tpu7x
                    else ""
                )
            ),
            bracket=D2D_BW_BRACKET if is_tpu7x else None,
            display=format_constant_display(chip.d2d_bw, "B/s"),
            description=(
                "Dedicated on-chip die-to-die bridge bandwidth (modeled as "
                f"{DEFAULT_D2D_TO_ICI_LINK_MULT:g} * ici_link_bw)"
            ),
        ),
        "ici_bw": ConstantSpec(
            value=chip.ici_bw,
            unit="B/s",
            scope="core",
            kind="derived",
            source=(
                f"{src_prefix}: {chip.ici_links_per_chip} * ici_link_bw / "
                "cores_per_chip per-core ICI share"
                + (
                    " (https://docs.cloud.google.com/tpu/docs/tpu7x)"
                    if is_tpu7x
                    else ""
                )
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
                f"{src_prefix}: dcn_host_gbit_per_s * 1e9 / 8 / "
                f"{chip.chips_per_host} chips"
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
                f"{src_prefix}: assumed per-hop latency used by serial estimate only, "
                "never by hard floor"
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
                f"{src_prefix}: host-to-device PCIe bandwidth "
                "(effective rate uncalibrated)"
            ),
            bracket=PCIE_BRACKET_BW,
            display=format_constant_display(chip.pcie_bw, "B/s"),
            description="Peak PCIe host-to-device interconnect bandwidth per chip",
        ),
        "mxu_m_tile": ConstantSpec(
            value=chip.mxu_m_tile,
            unit="rows",
            scope="core",
            kind="assumed",
            source=(
                f"{src_prefix}: assumed MoE kernel M-tile multiple on "
                f"{chip.mxu_m_tile}-wide matrix unit"
            ),
            bracket=MXU_M_TILE_BRACKET,
            display=format_constant_display(chip.mxu_m_tile, "rows"),
            description=(
                "Hardware matrix multiply unit systolic array row granularity "
                "tile floor"
            ),
        ),
        "dcn_host_gbit_per_s": ConstantSpec(
            value=chip.dcn_host_gbit_per_s,
            unit="Gbit/s",
            scope="host",
            kind="assumed",
            source=(
                f"{src_prefix}: host NIC bandwidth across "
                f"{chip.chips_per_host} chips per host"
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
            source=f"{src_prefix}: {chip.chips_per_host} chips per host tray",
            bracket=CHIP_COUNT_BRACKET,
            display=format_constant_display(chip.chips_per_host, "chips"),
            description=(
                "Number of TPU chips co-located on a single host tray / CPU subsystem"
            ),
        ),
        "vector_flops": ConstantSpec(
            value=chip.vector_flops,
            unit="FLOP/s",
            scope="core",
            kind="derived",
            source=(
                f"{src_prefix}: vector processing throughput per TensorCore "
                f"({chip.num_lanes}x{chip.num_sublanes} vector layout)"
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
            source=f"{src_prefix}: matrix-unit BF16 LHS sublane tiling floor",
            display=format_constant_display(chip.mxu_lhs_rows_bf16, "rows"),
            description=(
                "Hardware matrix multiply unit systolic array LHS row floor for BF16"
            ),
        ),
        "mxu_lhs_rows_fp8": ConstantSpec(
            value=chip.mxu_lhs_rows_fp8,
            unit="rows",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: matrix-unit FP8 LHS sublane tiling floor",
            display=format_constant_display(chip.mxu_lhs_rows_fp8, "rows"),
            description=(
                "Hardware matrix multiply unit systolic array LHS row floor for FP8"
            ),
        ),
        "dma_granule_bytes": ConstantSpec(
            value=chip.dma_granule_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: SparseCore DMA granule size",
            display=format_constant_display(chip.dma_granule_bytes, "B"),
            description=("Minimum SparseCore DMA transfer alignment granule in bytes"),
        ),
        "has_sparse_core": ConstantSpec(
            value=chip.has_sparse_core,
            unit="",
            scope="chip",
            kind="fact",
            source=f"{src_prefix}: SparseCore hardware capability",
            display="true" if chip.has_sparse_core else "false",
            description="Whether the chip includes dedicated SparseCore offload units",
        ),
    }
    return {
        entry_name: _spec_dict(constant_spec)
        for entry_name, constant_spec in entries.items()
    }


def _build_chip_from_jax(
    name: str,
    version: pltpu.ChipVersion,
    overrides_cfg: ChipOverrides,
) -> ChipSpec:
    """Assemble ChipSpec by merging JAX capabilities with explicit overrides."""
    caps = _read_jax_capabilities(name, version)
    eff_cores = (
        overrides_cfg.cores_per_chip
        if overrides_cfg.cores_per_chip is not None
        else caps.cores_per_chip
    )
    eff_links = (
        overrides_cfg.ici_links_per_chip
        if overrides_cfg.ici_links_per_chip is not None
        else DEFAULT_ICI_LINKS_PER_CHIP
    )
    eff_d2d_bw = (
        overrides_cfg.d2d_bw
        if overrides_cfg.d2d_bw is not None
        else DEFAULT_D2D_TO_ICI_LINK_MULT * overrides_cfg.ici_link_bw
    )
    eff_ici_bw = (
        overrides_cfg.ici_bw
        if overrides_cfg.ici_bw is not None
        else (float(eff_links) * overrides_cfg.ici_link_bw / float(max(1, eff_cores)))
    )
    eff_dcn_bw = (
        overrides_cfg.dcn_bw
        if overrides_cfg.dcn_bw is not None
        else (overrides_cfg.dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE)
        / float(max(1, overrides_cfg.chips_per_host))
    )
    eff_bf16 = (
        overrides_cfg.bf16_flops
        if overrides_cfg.bf16_flops is not None
        else caps.bf16_flops
    )
    eff_ratio = (
        overrides_cfg.vector_to_bf16_ratio
        if overrides_cfg.vector_to_bf16_ratio is not None
        else DEFAULT_VECTOR_TO_BF16_RATIO
    )
    eff_vector = (
        overrides_cfg.vector_flops
        if overrides_cfg.vector_flops is not None
        else eff_bf16 * eff_ratio
    )
    eff_num_lanes = (
        overrides_cfg.num_lanes
        if overrides_cfg.num_lanes is not None
        else caps.num_lanes
    )
    eff_physical_alignment = (
        overrides_cfg.physical_alignment_elems
        if overrides_cfg.physical_alignment_elems is not None
        else (
            eff_num_lanes
            if overrides_cfg.num_lanes is not None
            else caps.physical_alignment_elems
        )
    )
    eff_has_sparse_core = (
        overrides_cfg.has_sparse_core
        if overrides_cfg.has_sparse_core is not None
        else caps.has_sparse_core
    )
    eff_dma_granule = (
        overrides_cfg.dma_granule_bytes
        if overrides_cfg.dma_granule_bytes is not None
        else (
            (
                caps.dma_granule_bytes
                if caps.has_sparse_core
                else _DEFAULT_TPU_CAPS.dma_granule_bytes
            )
            if eff_has_sparse_core
            else 0
        )
    )
    return ChipSpec(
        name=name,
        hbm_bytes=(
            overrides_cfg.hbm_bytes
            if overrides_cfg.hbm_bytes is not None
            else caps.hbm_bytes
        ),
        hbm_bw=(
            overrides_cfg.hbm_bw if overrides_cfg.hbm_bw is not None else caps.hbm_bw
        ),
        bf16_flops=eff_bf16,
        fp8_flops=(
            overrides_cfg.fp8_flops
            if overrides_cfg.fp8_flops is not None
            else caps.fp8_flops
        ),
        vmem_bytes=(
            overrides_cfg.vmem_bytes
            if overrides_cfg.vmem_bytes is not None
            else caps.vmem_bytes
        ),
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
        vector_flops=eff_vector,
        mxu_lhs_rows_bf16=(
            overrides_cfg.mxu_lhs_rows_bf16
            if overrides_cfg.mxu_lhs_rows_bf16 is not None
            else caps.mxu_lhs_rows_bf16
        ),
        mxu_lhs_rows_fp8=(
            overrides_cfg.mxu_lhs_rows_fp8
            if overrides_cfg.mxu_lhs_rows_fp8 is not None
            else caps.mxu_lhs_rows_fp8
        ),
        int8_flops=(
            overrides_cfg.int8_flops
            if overrides_cfg.int8_flops is not None
            else caps.int8_flops
        ),
        fp4_flops=(
            overrides_cfg.fp4_flops if overrides_cfg.fp4_flops is not None else 0.0
        ),
        int4_flops=(
            overrides_cfg.int4_flops
            if overrides_cfg.int4_flops is not None
            else caps.int4_flops
        ),
        ici_links_per_chip=eff_links,
        min_torus_dim=(
            overrides_cfg.min_torus_dim
            if overrides_cfg.min_torus_dim is not None
            else MIN_TORUS_DIM
        ),
        num_lanes=eff_num_lanes,
        num_sublanes=(
            overrides_cfg.num_sublanes
            if overrides_cfg.num_sublanes is not None
            else caps.num_sublanes
        ),
        physical_alignment_elems=eff_physical_alignment,
        dma_granule_bytes=eff_dma_granule,
        subchannel_block_k_floor=(
            overrides_cfg.subchannel_block_k_floor
            if overrides_cfg.subchannel_block_k_floor is not None
            else SUBCHANNEL_BLOCK_K_FLOOR
        ),
        has_sparse_core=eff_has_sparse_core,
        provenance=(
            overrides_cfg.provenance
            if overrides_cfg.provenance is not None
            else caps.provenance
        ),
        vector_flops_overridden=(
            overrides_cfg.vector_flops is not None
            or overrides_cfg.vector_to_bf16_ratio is not None
        ),
    )


def from_jax(
    name: str,
    version: pltpu.ChipVersion,
    overrides: ChipOverrides | None = None,
) -> ChipSpec:
    """Construct ChipSpec directly from JAX TPU capability metadata."""
    overrides_cfg = overrides or ChipOverrides()
    return _build_chip_from_jax(name, version, overrides_cfg)


def count_hosts(
    num_chips: int,
    chip: ChipSpec | None = None,
    *,
    chips_per_host: int | None = None,
) -> int:
    """Return physical host count for ``num_chips`` using ceiling division."""
    per_host = (
        chips_per_host
        if chips_per_host is not None
        else (chip.chips_per_host if chip is not None else DEFAULT_CHIPS_PER_HOST)
    )
    per_host = max(1, per_host)
    return max(1, (num_chips + per_host - 1) // per_host)


@dataclass(frozen=True, slots=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = _DEFAULT_TPU_CAPS.cores_per_chip
    wrapped_axes: tuple[bool, bool, bool] | None = None
    min_torus_dim: int = MIN_TORUS_DIM

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
            w_x, w_y, w_z = self.wrapped_axes
            return w_x or w_y or w_z
        # Google Cloud TPU7x doc slice table
        # (https://docs.cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4" by default (overridable via
        # min_torus_dim).
        axis_x, axis_y, axis_z = self.chip_axes
        min_dim = self.min_torus_dim
        return (
            axis_x >= min_dim
            and axis_x % min_dim == 0
            and axis_y >= min_dim
            and axis_y % min_dim == 0
            and axis_z >= min_dim
            and axis_z % min_dim == 0
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
            and submesh_extent >= self.min_torus_dim
        )


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": from_jax(
        "tpu7x",
        _TPU_7X_VERSION,
        ChipOverrides(),
    ),
}

# Live read-only set view over CHIPS so register_chip stays synchronized globally.
SUPPORTED_CHIPS: KeysView[str] = CHIPS.keys()

TOPOLOGIES: dict[str, SliceTopology] = {
    name: SliceTopology(name, axes) for name, axes in TOPOLOGY_AXES.items()
}

CHIP_TOPOLOGIES: dict[str, dict[str, SliceTopology]] = {
    "tpu7x": TOPOLOGIES,
}


def topologies_for_chip(
    chip: str | ChipSpec,
    *,
    fallback: Mapping[str, SliceTopology] | None = None,
) -> dict[str, SliceTopology]:
    """Return the topology map for a chip name or ChipSpec."""
    chip_name = chip.name if isinstance(chip, ChipSpec) else chip
    if chip_name in CHIP_TOPOLOGIES:
        return dict(CHIP_TOPOLOGIES[chip_name])
    if fallback is not None:
        return dict(fallback)
    if isinstance(chip, ChipSpec):
        return {
            topo_name: replace(
                topology,
                cores_per_chip=chip.cores_per_chip,
                min_torus_dim=chip.min_torus_dim,
            )
            for topo_name, topology in TOPOLOGIES.items()
        }
    return dict(TOPOLOGIES)


def register_chip(
    chip: ChipSpec,
    topologies: Mapping[str, SliceTopology],
) -> None:
    """Register a ChipSpec and topology map for global and SUPPORTED_CHIPS lookup."""
    if not chip.name:
        msg = "ChipSpec.name must be non-empty"
        raise ValueError(msg)
    if not topologies:
        msg = f"Chip {chip.name!r} requires at least one SliceTopology"
        raise ValueError(msg)
    CHIPS[chip.name] = chip
    CHIP_TOPOLOGIES[chip.name] = dict(topologies)


def _read_spec_mapping(
    source: str | Path | Mapping[str, object],
) -> Mapping[str, object]:
    """Load a chip specification JSON path or validate an in-memory mapping."""
    if isinstance(source, (str, Path)):
        loaded = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            msg = "Chip spec JSON root must be an object"
            raise TypeError(msg)
        return loaded
    return source


def _parse_spec_int(
    raw: Mapping[str, object],
    key: str,
    *,
    default: int | None = None,
    allow_zero: bool = False,
) -> int:
    """Extract a positive (or non-negative) integer from a chip spec mapping."""
    val = raw.get(key)
    if val is None:
        if default is not None:
            return default
        msg = f"Chip spec missing required field {key!r}"
        raise ValueError(msg)
    min_val = 0 if allow_zero else 1
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        desc = "non-negative" if allow_zero else "positive"
        msg = f"Chip spec field {key!r} must be a {desc} integer, got {val!r}"
        raise ValueError(msg)
    return int(val)


def _parse_spec_float(
    raw: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
    allow_zero: bool = False,
) -> float:
    """Extract a positive (or non-negative) finite float from a chip spec."""
    val = raw.get(key)
    if val is None:
        if default is not None:
            return default
        msg = f"Chip spec missing required field {key!r}"
        raise ValueError(msg)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        msg = f"Chip spec field {key!r} must be numeric, got {val!r}"
        raise TypeError(msg)
    num = float(val)
    if not math.isfinite(num) or num < 0.0 or (not allow_zero and num <= 0.0):
        bound = ">= 0" if allow_zero else "> 0"
        msg = f"Chip spec field {key!r} must be finite and {bound}, got {val!r}"
        raise ValueError(msg)
    return num


def _parse_spec_bool(
    raw: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    """Extract a boolean capability flag from a chip spec mapping."""
    val = raw.get(key)
    if val is None:
        return default
    if not isinstance(val, bool):
        msg = f"Chip spec field {key!r} must be a boolean, got {val!r}"
        raise TypeError(msg)
    return val


_ALLOWED_TOPOLOGY_KEYS: frozenset[str] = frozenset(
    {"name", "chip_axes", "wrapped_axes"}
)

_ALLOWED_CHIP_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "provenance",
        "name",
        "base",
        "hbm_bytes",
        "hbm_bw",
        "bf16_flops",
        "fp8_flops",
        "int8_flops",
        "fp4_flops",
        "int4_flops",
        "vmem_bytes",
        "cores_per_chip",
        "chips_per_host",
        "ici_links_per_chip",
        "ici_link_bw",
        "d2d_bw",
        "ici_bw",
        "ici_hop_latency_s",
        "mxu_m_tile",
        "mxu_lhs_rows_bf16",
        "mxu_lhs_rows_fp8",
        "dcn_host_gbit_per_s",
        "dcn_bw",
        "pcie_bw",
        "vector_flops",
        "vector_to_bf16_ratio",
        "min_torus_dim",
        "num_lanes",
        "num_sublanes",
        "physical_alignment_elems",
        "dma_granule_bytes",
        "subchannel_block_k_floor",
        "has_sparse_core",
        "topologies",
    }
)


def _parse_one_topology(
    entry: object,
    cores_per_chip: int,
    min_torus_dim: int = MIN_TORUS_DIM,
) -> SliceTopology:
    """Parse one topology entry dict into a SliceTopology instance."""
    if not isinstance(entry, Mapping):
        msg = f"Chip spec topology entry must be an object, got {entry!r}"
        raise TypeError(msg)
    unknown_topo_keys = sorted(set(entry) - _ALLOWED_TOPOLOGY_KEYS)
    if unknown_topo_keys:
        msg = (
            f"Unknown topology key(s) {unknown_topo_keys!r}; "
            f"allowed: {sorted(_ALLOWED_TOPOLOGY_KEYS)}"
        )
        raise ValueError(msg)
    topo_name = entry.get("name")
    if not isinstance(topo_name, str) or not topo_name.strip():
        msg = f"Chip spec topology requires non-empty string 'name', got {topo_name!r}"
        raise ValueError(msg)
    axes_raw = entry.get("chip_axes")
    if (
        not isinstance(axes_raw, (list, tuple))
        or len(axes_raw) != _TORUS_AXES_LEN
        or any(
            isinstance(a, bool) or not isinstance(a, int) or a <= 0 for a in axes_raw
        )
    ):
        msg = (
            f"Topology {topo_name!r} 'chip_axes' must be 3 positive ints, "
            f"got {axes_raw!r}"
        )
        raise ValueError(msg)
    wrapped_raw = entry.get("wrapped_axes")
    wrapped_axes: tuple[bool, bool, bool] | None = None
    if wrapped_raw is not None:
        if (
            not isinstance(wrapped_raw, (list, tuple))
            or len(wrapped_raw) != _TORUS_AXES_LEN
            or any(not isinstance(width, bool) for width in wrapped_raw)
        ):
            msg = (
                f"Topology {topo_name!r} 'wrapped_axes' must be 3 bools, "
                f"got {wrapped_raw!r}"
            )
            raise ValueError(msg)
        wrapped_axes = (
            bool(wrapped_raw[0]),
            bool(wrapped_raw[1]),
            bool(wrapped_raw[2]),
        )
    return SliceTopology(
        name=topo_name,
        chip_axes=(int(axes_raw[0]), int(axes_raw[1]), int(axes_raw[2])),
        cores_per_chip=cores_per_chip,
        wrapped_axes=wrapped_axes,
        min_torus_dim=min_torus_dim,
    )


def _parse_spec_topologies(
    raw: Mapping[str, object],
    cores_per_chip: int,
    base_name: str | None,
    min_torus_dim: int = MIN_TORUS_DIM,
) -> dict[str, SliceTopology]:
    """Parse topology definitions or derive defaults when omitted."""
    topos_raw = raw.get("topologies")
    if topos_raw is None:
        if base_name is not None:
            return {
                t_name: replace(
                    topology,
                    cores_per_chip=cores_per_chip,
                    min_torus_dim=min_torus_dim,
                )
                for t_name, topology in topologies_for_chip(base_name).items()
            }
        return {
            "1x1": SliceTopology(
                name="1x1",
                chip_axes=(1, 1, 1),
                cores_per_chip=cores_per_chip,
                wrapped_axes=(False, False, False),
                min_torus_dim=min_torus_dim,
            )
        }
    if not isinstance(topos_raw, (list, tuple)) or not topos_raw:
        msg = "Chip spec 'topologies' must be a non-empty list when provided"
        raise ValueError(msg)
    parsed = [
        _parse_one_topology(entry, cores_per_chip, min_torus_dim) for entry in topos_raw
    ]
    return {topology.name: topology for topology in parsed}


def _resolve_base_chip(raw: Mapping[str, object]) -> tuple[str | None, ChipSpec | None]:
    """Validate optional 'base' chip reference and enforce schema/provenance."""
    unknown_keys = sorted(set(raw) - _ALLOWED_CHIP_SPEC_KEYS)
    if unknown_keys:
        msg = (
            f"Unknown chip spec key(s) {unknown_keys!r}; "
            f"allowed: {sorted(_ALLOWED_CHIP_SPEC_KEYS)}"
        )
        raise ValueError(msg)
    schema = raw.get("schema_version")
    if schema != CHIP_SPEC_SCHEMA:
        msg = (
            f"Unsupported chip spec schema_version {schema!r}; "
            f"expected {CHIP_SPEC_SCHEMA!r}"
        )
        raise ValueError(msg)
    provenance = raw.get("provenance")
    if not isinstance(provenance, str) or provenance not in VALID_CHIP_PROVENANCE:
        allowed = sorted(VALID_CHIP_PROVENANCE)
        msg = f"Invalid chip spec provenance {provenance!r}; expected one of {allowed}"
        raise ValueError(msg)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = f"Chip spec field 'name' must be a non-empty string, got {name!r}"
        raise ValueError(msg)
    base_raw = raw.get("base")
    if base_raw is None:
        return None, None
    if not isinstance(base_raw, str) or base_raw not in CHIPS:
        msg = f"Unknown base chip {base_raw!r} in chip spec {name!r}"
        raise ValueError(msg)
    return base_raw, CHIPS[base_raw]


def _resolve_spec_vector_flops(
    raw: Mapping[str, object],
    bf16_flops: float,
    base_chip: ChipSpec | None,
) -> float:
    """Resolve and validate per-core vector FLOP/s for a chip spec."""
    if "vector_flops" in raw and raw["vector_flops"] is not None:
        return _parse_spec_float(raw, "vector_flops", allow_zero=False)
    if "vector_to_bf16_ratio" in raw and raw["vector_to_bf16_ratio"] is not None:
        ratio = _parse_spec_float(raw, "vector_to_bf16_ratio", allow_zero=False)
        return bf16_flops * ratio
    if base_chip is not None and base_chip.bf16_flops > 0.0:
        ratio = base_chip.vector_flops / base_chip.bf16_flops
        return bf16_flops * ratio
    return bf16_flops * DEFAULT_VECTOR_TO_BF16_RATIO


def _build_spec_chip(
    raw: Mapping[str, object],
    base_chip: ChipSpec | None,
) -> ChipSpec:
    """Build a validated ChipSpec instance from a parsed JSON mapping."""
    name = str(raw["name"]).strip()
    provenance = str(raw["provenance"])
    hbm_bytes = _parse_spec_int(
        raw, "hbm_bytes", default=base_chip.hbm_bytes if base_chip else None
    )
    hbm_bw = _parse_spec_float(
        raw, "hbm_bw", default=base_chip.hbm_bw if base_chip else None
    )
    bf16_flops = _parse_spec_float(
        raw, "bf16_flops", default=base_chip.bf16_flops if base_chip else None
    )
    vmem_bytes = _parse_spec_int(
        raw, "vmem_bytes", default=base_chip.vmem_bytes if base_chip else None
    )
    cores_per_chip = _parse_spec_int(
        raw, "cores_per_chip", default=base_chip.cores_per_chip if base_chip else None
    )
    chips_per_host = _parse_spec_int(
        raw, "chips_per_host", default=base_chip.chips_per_host if base_chip else None
    )
    fp8_flops = _parse_spec_float(
        raw,
        "fp8_flops",
        default=base_chip.fp8_flops if base_chip else 0.0,
        allow_zero=True,
    )
    int8_flops = _parse_spec_float(
        raw,
        "int8_flops",
        default=base_chip.int8_flops if base_chip else 0.0,
        allow_zero=True,
    )
    fp4_flops = _parse_spec_float(
        raw,
        "fp4_flops",
        default=base_chip.fp4_flops if base_chip else 0.0,
        allow_zero=True,
    )
    int4_flops = _parse_spec_float(
        raw,
        "int4_flops",
        default=base_chip.int4_flops if base_chip else 0.0,
        allow_zero=True,
    )
    ici_links_per_chip = _parse_spec_int(
        raw,
        "ici_links_per_chip",
        default=base_chip.ici_links_per_chip
        if base_chip
        else DEFAULT_ICI_LINKS_PER_CHIP,
    )
    ici_link_bw = _parse_spec_float(
        raw,
        "ici_link_bw",
        default=base_chip.ici_link_bw if base_chip else DEFAULT_ICI_LINK_BW,
    )
    has_d2d_input = raw.get("ici_link_bw") is not None
    d2d_default = (
        base_chip.d2d_bw
        if base_chip is not None and not has_d2d_input
        else DEFAULT_D2D_TO_ICI_LINK_MULT * ici_link_bw
    )
    d2d_bw = _parse_spec_float(
        raw,
        "d2d_bw",
        default=d2d_default,
    )
    has_ici_input = (
        has_d2d_input
        or raw.get("ici_links_per_chip") is not None
        or raw.get("cores_per_chip") is not None
    )
    ici_default = (
        base_chip.ici_bw
        if base_chip is not None and not has_ici_input
        else float(ici_links_per_chip) * ici_link_bw / float(cores_per_chip)
    )
    ici_bw = _parse_spec_float(
        raw,
        "ici_bw",
        default=ici_default,
    )
    dcn_host_gbit_per_s = _parse_spec_float(
        raw,
        "dcn_host_gbit_per_s",
        default=(
            base_chip.dcn_host_gbit_per_s if base_chip else DEFAULT_DCN_HOST_GBIT_PER_S
        ),
    )
    has_dcn_input = (
        raw.get("dcn_host_gbit_per_s") is not None
        or raw.get("chips_per_host") is not None
    )
    dcn_default = (
        base_chip.dcn_bw
        if base_chip is not None and not has_dcn_input
        else (dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE) / float(chips_per_host)
    )
    dcn_bw = _parse_spec_float(
        raw,
        "dcn_bw",
        default=dcn_default,
    )
    vector_flops = _resolve_spec_vector_flops(raw, bf16_flops, base_chip)
    has_sparse_core = _parse_spec_bool(
        raw,
        "has_sparse_core",
        default=(
            base_chip.has_sparse_core
            if base_chip
            else _DEFAULT_TPU_CAPS.has_sparse_core
        ),
    )
    default_dma_granule = (
        (
            base_chip.dma_granule_bytes
            if base_chip is not None and base_chip.has_sparse_core
            else _DEFAULT_TPU_CAPS.dma_granule_bytes
        )
        if has_sparse_core
        else 0
    )
    num_lanes = _parse_spec_int(
        raw,
        "num_lanes",
        default=(base_chip.num_lanes if base_chip else _DEFAULT_TPU_CAPS.num_lanes),
    )
    default_physical_alignment = (
        num_lanes
        if raw.get("num_lanes") is not None
        else (
            base_chip.physical_alignment_elems
            if base_chip
            else _DEFAULT_TPU_CAPS.physical_alignment_elems
        )
    )
    return ChipSpec(
        name=name,
        hbm_bytes=hbm_bytes,
        hbm_bw=hbm_bw,
        bf16_flops=bf16_flops,
        fp8_flops=fp8_flops,
        vmem_bytes=vmem_bytes,
        ici_bw=ici_bw,
        cores_per_chip=cores_per_chip,
        ici_link_bw=ici_link_bw,
        d2d_bw=d2d_bw,
        ici_hop_latency_s=_parse_spec_float(
            raw,
            "ici_hop_latency_s",
            default=(
                base_chip.ici_hop_latency_s if base_chip else DEFAULT_ICI_HOP_LATENCY_S
            ),
            allow_zero=True,
        ),
        mxu_m_tile=_parse_spec_int(
            raw,
            "mxu_m_tile",
            default=(
                base_chip.mxu_m_tile if base_chip else _DEFAULT_TPU_CAPS.mxu_m_tile
            ),
        ),
        dcn_host_gbit_per_s=dcn_host_gbit_per_s,
        chips_per_host=chips_per_host,
        dcn_bw=dcn_bw,
        pcie_bw=_parse_spec_float(
            raw,
            "pcie_bw",
            default=base_chip.pcie_bw if base_chip else DEFAULT_PCIE_BW,
        ),
        vector_flops=vector_flops,
        mxu_lhs_rows_bf16=_parse_spec_int(
            raw,
            "mxu_lhs_rows_bf16",
            default=(
                base_chip.mxu_lhs_rows_bf16
                if base_chip
                else _DEFAULT_TPU_CAPS.mxu_lhs_rows_bf16
            ),
        ),
        mxu_lhs_rows_fp8=_parse_spec_int(
            raw,
            "mxu_lhs_rows_fp8",
            default=(
                base_chip.mxu_lhs_rows_fp8
                if base_chip
                else _DEFAULT_TPU_CAPS.mxu_lhs_rows_fp8
            ),
        ),
        int8_flops=int8_flops,
        fp4_flops=fp4_flops,
        int4_flops=int4_flops,
        ici_links_per_chip=ici_links_per_chip,
        min_torus_dim=_parse_spec_int(
            raw,
            "min_torus_dim",
            default=base_chip.min_torus_dim if base_chip else MIN_TORUS_DIM,
        ),
        num_lanes=num_lanes,
        num_sublanes=_parse_spec_int(
            raw,
            "num_sublanes",
            default=(
                base_chip.num_sublanes if base_chip else _DEFAULT_TPU_CAPS.num_sublanes
            ),
        ),
        physical_alignment_elems=_parse_spec_int(
            raw,
            "physical_alignment_elems",
            default=default_physical_alignment,
        ),
        dma_granule_bytes=_parse_spec_int(
            raw,
            "dma_granule_bytes",
            default=default_dma_granule,
            allow_zero=True,
        ),
        subchannel_block_k_floor=_parse_spec_int(
            raw,
            "subchannel_block_k_floor",
            default=(
                base_chip.subchannel_block_k_floor
                if base_chip
                else SUBCHANNEL_BLOCK_K_FLOOR
            ),
        ),
        has_sparse_core=has_sparse_core,
        provenance=provenance,
        vector_flops_overridden=(
            "vector_flops" in raw
            or "vector_to_bf16_ratio" in raw
            or (base_chip.vector_flops_overridden if base_chip is not None else False)
        ),
    )


def load_chip_spec(
    source: str | Path | Mapping[str, object],
) -> tuple[ChipSpec, dict[str, SliceTopology]]:
    """Validate and load a ChipSpec and topology map from JSON path or mapping."""
    raw = _read_spec_mapping(source)
    base_name, base_chip = _resolve_base_chip(raw)
    chip = _build_spec_chip(raw, base_chip)
    topologies = _parse_spec_topologies(
        raw, chip.cores_per_chip, base_name, chip.min_torus_dim
    )
    return chip, topologies


def get_solver_assumptions(
    chip: ChipSpec | None = None,
) -> list[dict[str, object]]:
    """Return a fresh list of structured physics solver assumption entries."""
    active_chip = chip if chip is not None else CHIPS["tpu7x"]
    rows = [dict(entry) for entry in SOLVER_ASSUMPTIONS]
    rows.append(
        {
            "name": "ici_hop_latency_us",
            "flag": "--chip-spec",
            "kind": "fact",
            "value": active_chip.ici_hop_latency_s / UNIT_MICRO,
            "unit": "us",
            "description": (
                "Per-hop ICI routing latency from the active chip spec "
                "(applied to collective hop estimates)."
            ),
            "source": "chips",
        }
    )
    return rows
