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
"""Per-core hardware specifications and physical 2D/3D slice topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from jax.experimental.pallas import tpu as _pltpu
except ImportError:
    pltpu: Any = None
else:
    pltpu = _pltpu

_MIN_TORUS_DIM: int = 4
DEFAULT_CORES_PER_CHIP: int = 2


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    ici_link_bw: float = 100.0e9
    ici_hop_latency_s: float = 1.0e-6


_FALLBACK_TPU_SPECS: dict[str, tuple[int, float, float, float, int]] = {
    # (hbm_bytes, hbm_bw, bf16_flops, fp8_flops, vmem_bytes) per TensorCore
    # matching jax._src.tpu_info for ChipVersion.TPU_7X and ChipVersion.TPU_V6E.
    "tpu7x": (103_000_000_000, 3.70e12, 1.155e15, 2.30e15, 67_108_864),
    "tpu6e": (34_400_000_000, 1.64e12, 9.20e14, 9.20e14, 134_217_728),
}


def from_jax(
    name: str,
    version: object,
    *,
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP,
    ici_link_bw: float = 100.0e9,
    ici_bw: float = 300.0e9,
    ici_hop_latency_s: float = 1.0e-6,
) -> ChipSpec:
    """Construct ChipSpec from JAX TPU info table or stdlib fallback table."""
    if pltpu is None or type(version) is str:
        hbm_b, hbm_bw, bf16, fp8, vmem_b = _FALLBACK_TPU_SPECS[name]
        return ChipSpec(
            name=name,
            hbm_bytes=hbm_b,
            hbm_bw=hbm_bw,
            bf16_flops=bf16,
            fp8_flops=fp8,
            vmem_bytes=vmem_b,
            cores_per_chip=cores_per_chip,
            ici_link_bw=ici_link_bw,
            ici_bw=ici_bw,
            ici_hop_latency_s=ici_hop_latency_s,
        )
    # One TensorCore per logical device gives per-core HBM, bandwidth, MXU peaks,
    # and VMEM; JAX's table has no ICI rates, so callers pass them.
    info = pltpu.get_tpu_info_for_chip(version, 1)
    return ChipSpec(
        name=name,
        hbm_bytes=info.hbm_capacity_bytes,
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=info.vmem_capacity_bytes,
        cores_per_chip=cores_per_chip,
        ici_link_bw=ici_link_bw,
        ici_bw=ici_bw,
        ici_hop_latency_s=ici_hop_latency_s,
    )


@dataclass(frozen=True)
class SliceTopology:
    """Physical 2D/3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    wrapped_axes: tuple[bool, bool, bool] | None = None

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the slice topology."""
        a, b, c = self.chip_axes
        return a * b * c

    @property
    def num_cores(self) -> int:
        """Total number of TensorCores across all chips in the topology."""
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        """True when at least one physical chip axis has torus wraparound."""
        if self.wrapped_axes is not None:
            return any(self.wrapped_axes)
        # Scaling Book (https://jax-ml.github.io/scaling-book/tpus/):
        # "Any multiple of a full cube ... will have wraparounds."
        # Google Cloud TPU7x doc slice table (https://cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4".
        return all(
            d >= _MIN_TORUS_DIM and d % _MIN_TORUS_DIM == 0 for d in self.chip_axes
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
            and submesh_extent >= _MIN_TORUS_DIM
        )


CHIPS: dict[str, ChipSpec] = {
    # HBM, bandwidth, BF16 and FP8 MXU peaks, and VMEM: JAX's TPU table
    # (ChipVersion.TPU_7X, https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py).
    "tpu7x": from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X if pltpu is not None else "tpu7x",
        # Peak ICI rate: 6 links x 200 GB/s bidirectional = 1,200 GB/s per chip, so
        # 100 GB/s per link per direction and 600 GB/s egress per chip (300 GB/s per
        # core): Google Cloud TPU7x docs (https://cloud.google.com/tpu/docs/tpu7x).
        ici_link_bw=100.0e9,
        ici_bw=300.0e9,
        # Unloaded per-hop ICI latency of about 1 us, the approximation How To Scale
        # Your Model states for TPU v5e (https://jax-ml.github.io/scaling-book/tpus/);
        # no TPU7x value is published.
        ici_hop_latency_s=1.0e-6,
    ),
    # HBM, bandwidth, BF16/FP8 MXU peaks, and VMEM: JAX's TPU table
    # (ChipVersion.TPU_V6E, single TensorCore per chip / is_lite=True).
    "tpu6e": from_jax(
        "tpu6e",
        pltpu.ChipVersion.TPU_V6E if pltpu is not None else "tpu6e",
        cores_per_chip=1,
        # Peak ICI rate: 4 links x 200 GB/s bidirectional = 800 GB/s per chip, so
        # 100 GB/s per link per direction and 400 GB/s egress per chip/core:
        # Google Cloud TPU v6e docs (https://cloud.google.com/tpu/docs/v6e); note
        # How To Scale Your Model uses a 90 GB/s per-link figure.
        ici_link_bw=100.0e9,
        ici_bw=400.0e9,
        ici_hop_latency_s=1.0e-6,
    ),
}

TOPOLOGIES: dict[str, SliceTopology] = {
    # Single-chip (1x1x1, 2 TensorCores) and dual-chip (1x2x1, 4 TensorCores)
    # sub-host topologies for compact models and single-stage preflight checks
    # alongside full-host and pod slices from https://cloud.google.com/tpu/docs/tpu7x.
    "1x1x1": SliceTopology("1x1x1", (1, 1, 1)),
    "1x2x1": SliceTopology("1x2x1", (1, 2, 1)),
    "2x2x1": SliceTopology("2x2x1", (2, 2, 1)),
    "2x2x2": SliceTopology("2x2x2", (2, 2, 2)),
    "2x2x4": SliceTopology("2x2x4", (2, 2, 4)),
    "2x4x4": SliceTopology("2x4x4", (2, 4, 4)),
    "4x4x4": SliceTopology("4x4x4", (4, 4, 4)),
}

TPU6E_TOPOLOGIES: dict[str, SliceTopology] = {
    # Cloud TPU v6e (Trillium) 2D torus slice shapes up to a 16x16 full pod
    # (https://cloud.google.com/tpu/docs/v6e); per How To Scale Your Model,
    # 2D torus wraparound applies along axes of extent 16 (8x16 and 16x16).
    "1x1": SliceTopology("1x1", (1, 1, 1), 1, (False, False, False)),
    "2x2": SliceTopology("2x2", (2, 2, 1), 1, (False, False, False)),
    "2x4": SliceTopology("2x4", (2, 4, 1), 1, (False, False, False)),
    "4x4": SliceTopology("4x4", (4, 4, 1), 1, (False, False, False)),
    "4x8": SliceTopology("4x8", (4, 8, 1), 1, (False, False, False)),
    "8x8": SliceTopology("8x8", (8, 8, 1), 1, (False, False, False)),
    "8x16": SliceTopology("8x16", (8, 16, 1), 1, (False, True, False)),
    "16x16": SliceTopology("16x16", (16, 16, 1), 1, (True, True, False)),
}

CHIP_TOPOLOGIES: dict[str, dict[str, SliceTopology]] = {
    "tpu7x": TOPOLOGIES,
    "tpu6e": TPU6E_TOPOLOGIES,
}


def topologies_for_chip(chip: ChipSpec | str) -> dict[str, SliceTopology]:
    """Return physical slice topologies registered for ``chip``."""
    name = chip.name if isinstance(chip, ChipSpec) else chip
    return CHIP_TOPOLOGIES[name]
