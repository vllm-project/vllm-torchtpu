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
"""Per-core hardware specifications and physical 3D slice topologies."""

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
    cores_per_chip: int = 2
    ici_link_bw: float = 100.0e9
    ici_hop_latency_s: float = 1.0e-6


_FALLBACK_TPU_SPECS: dict[str, tuple[int, float, float, float, int]] = {
    # (hbm_bytes, hbm_bw, bf16_flops, fp8_flops, vmem_bytes) per TensorCore
    # matching jax._src.tpu_info for ChipVersion.TPU_7X.
    "tpu7x": (103_000_000_000, 3.70e12, 1.155e15, 2.30e15, 67_108_864),
}


def _from_jax(
    name: str,
    version: Any,
    *,
    cores_per_chip: int,
    ici_link_bw: float,
    ici_bw: float,
    ici_hop_latency_s: float,
) -> ChipSpec:
    if pltpu is None:
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
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = 2

    @property
    def num_chips(self) -> int:
        a, b, c = self.chip_axes
        return a * b * c

    @property
    def num_cores(self) -> int:
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        # Scaling Book (https://jax-ml.github.io/scaling-book/tpus/):
        # "Any multiple of a full cube ... will have wraparounds."
        # Google Cloud TPU7x doc slice table (https://cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4".
        return all(d >= 4 and d % 4 == 0 for d in self.chip_axes)

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= 4
        )


CHIPS: dict[str, ChipSpec] = {
    # HBM, bandwidth, BF16 and FP8 MXU peaks, and VMEM: JAX's TPU table
    # (ChipVersion.TPU_7X, https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py).
    "tpu7x": _from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X if pltpu is not None else "tpu7x",
        # 2 TensorCores per chip: Google Cloud TPU7x docs
        # (https://cloud.google.com/tpu/docs/tpu7x).
        cores_per_chip=2,
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
}

TOPOLOGIES: dict[str, SliceTopology] = {
    # 1x1x1 and 1x2x1 are sub-host what-if shapes not listed in the public Cloud
    # TPU7x slice table (https://cloud.google.com/tpu/docs/tpu7x); decision pending
    # with the user.
    "1x1x1": SliceTopology("1x1x1", (1, 1, 1)),
    "1x2x1": SliceTopology("1x2x1", (1, 2, 1)),
    "2x2x1": SliceTopology("2x2x1", (2, 2, 1)),
    "2x2x2": SliceTopology("2x2x2", (2, 2, 2)),
    "2x2x4": SliceTopology("2x2x4", (2, 2, 4)),
    "2x4x4": SliceTopology("2x4x4", (2, 4, 4)),
    "4x4x4": SliceTopology("4x4x4", (4, 4, 4)),
}
