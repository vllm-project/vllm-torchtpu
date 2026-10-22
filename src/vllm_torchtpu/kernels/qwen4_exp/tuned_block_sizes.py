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
"""Transfer-sized QSA DMA concurrency, independent of allocator page extent."""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

# Conservative TPU7x performance default, not a hardware concurrency limit.
# Matched1/2/4/8/16 wall measurements do not establish a material gain at16.
# Unmeasured chips stay serial; resource admission remains mandatory.
DEFAULT_DMA_CONCURRENCY = 8
_DMA_BATCHES = tuple(
    1 << power for power in reversed(range(DEFAULT_DMA_CONCURRENCY.bit_length()))
)
_COMPILER_SMEM_RESERVE = 32 * 1024
_MAIN_TILE = 128
_DmaArray = jax.Array | jax.ShapeDtypeStruct


@dataclasses.dataclass(frozen=True)
class DmaHardware:
    """Immutable per-TensorCore capacity and pinned compiler tile facts."""

    chip_version: pltpu.ChipVersion | None
    vmem_bytes: int
    smem_bytes: int
    lanes: int
    bf16_rows: int
    fp8_rows: int
    int32_rows: int

    @classmethod
    def from_tpu_info(cls, info: pltpu.TpuInfo) -> "DmaHardware":
        return cls(
            info.chip_version
            if isinstance(info.chip_version, pltpu.ChipVersion)
            else None,
            info.vmem_capacity_bytes,
            info.smem_capacity_bytes,
            info.num_lanes,
            info.get_sublane_tiling(jnp.bfloat16),
            info.get_sublane_tiling(jnp.float8_e4m3fn),
            info.get_sublane_tiling(jnp.int32),
        )


def _hardware() -> DmaHardware | None:
    # Called at kernel lowering, never at module import or pure CPU planning.
    try:
        info = pltpu.get_tpu_info()
    except ValueError as error:
        if str(error).startswith(
            ("Unsupported TPU device kind:", "Unsupported TPU chip version:")
        ):
            return None
        raise
    if info.generation not in range(2, 9):
        # The pinned get_sublane_tiling API has no layout for later generations.
        return None
    return DmaHardware.from_tpu_info(info)


def _align(value: int, tile: int) -> int:
    return (value + tile - 1) // tile * tile


def _vmem(shape: tuple[int, ...], dtype: DTypeLike, hw: DmaHardware) -> int:
    dtype = jnp.dtype(dtype)
    rows = {
        jnp.dtype(jnp.bfloat16): hw.bf16_rows,
        jnp.dtype(jnp.float8_e4m3fn): hw.fp8_rows,
        jnp.dtype(jnp.int32): hw.int32_rows,
    }[dtype]
    return (
        math.prod(shape[:-2])
        * _align(shape[-2], rows)
        * _align(shape[-1], hw.lanes)
        * dtype.itemsize
    )


def _carrier(pool: _DmaArray, hw: DmaHardware | None) -> bool:
    return (
        hw is not None
        and min(
            hw.vmem_bytes,
            hw.smem_bytes,
            hw.lanes,
            hw.bf16_rows,
            hw.fp8_rows,
            hw.int32_rows,
        )
        > 0
        and len(pool.shape) == 5
        and all(0 < d < 2**31 for d in pool.shape)
        and jnp.dtype(pool.dtype)
        in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float8_e4m3fn))
        and pool.shape[-2] * jnp.dtype(pool.dtype).itemsize == 4
        and pool.shape[-1] % hw.lanes == 0
    )


@dataclasses.dataclass(frozen=True)
class DmaMemory:
    """Conservative padded live blocks plus explicit scheduling headroom."""

    buffers: int
    scratch: int
    headroom: int
    scalar: int

    @property
    def vmem_bytes(self) -> int:
        return self.buffers + self.scratch + self.headroom


def ring_tag_dma_memory(
    pool: _DmaArray,
    batch: int,
    requests: int,
    ring_elements: int,
    tag_elements: int,
    hw: DmaHardware,
) -> DmaMemory:
    rt, rr = region_rows(pool, ring_elements, jnp.bfloat16)
    tt, tr = region_rows(pool, tag_elements, jnp.int32)
    staged = _vmem((batch, rt + tt, *pool.shape[2:]), pool.dtype, hw)
    typed = _vmem((batch, rr, pool.shape[-1]), jnp.bfloat16, hw) + _vmem(
        (batch, tr, pool.shape[-1]), jnp.int32, hw
    )
    vector = hw.int32_rows * hw.lanes * 4
    # Input/output blocks are double buffered by the pinned Pallas pipeline.
    # Reserve another typed+staging block for bitcast/layout scheduling; no MXU
    # operands or accumulators exist in these vector-only kernels.
    return DmaMemory(
        2 * typed,
        staged,
        typed + staged + vector,
        (requests + batch) * 4 + _COMPILER_SMEM_RESERVE,
    )


def main_kv_dma_memory(
    pool: _DmaArray, rows: _DmaArray, batch: int, hw: DmaHardware
) -> DmaMemory:
    source = _vmem((_MAIN_TILE, *rows.shape[1:]), rows.dtype, hw)
    vector = hw.int32_rows * hw.lanes * 4
    # The destination is aliased HBM, not a full-page VMEM allocation.
    # Count explicit prefetched words/flags and DMA counters. As in the shared
    # metadata generator, reserve32KiB for scalar pipeline/compiler bookkeeping;
    # this is headroom, not a hardware allocation-granularity requirement.
    scalar = (
        3 * len(rows) + len(rows) // _MAIN_TILE + len(rows) // batch + batch
    ) * 4 + _COMPILER_SMEM_RESERVE
    return DmaMemory(2 * source, 0, source + vector, scalar)


def ring_tag_dma_batch(
    pool: _DmaArray,
    requests: int,
    *,
    ring_elements: int,
    tag_elements: int,
    hardware: DmaHardware | None,
) -> int:
    """Pure transfer/resource policy. Page length supplies capacity and stride."""
    if (
        not _carrier(pool, hardware)
        or requests <= 0
        or requests >= 2**31
        or min(ring_elements, tag_elements) <= 0
    ):
        return 1
    assert hardware is not None
    rt, _ = region_rows(pool, ring_elements, jnp.bfloat16)
    tt, _ = region_rows(pool, tag_elements, jnp.int32)
    if rt + tt > pool.shape[1]:
        return 1
    batches = (
        _DMA_BATCHES if hardware.chip_version == pltpu.ChipVersion.TPU_7X else (1,)
    )
    for batch in batches:
        if requests % batch:
            continue
        memory = ring_tag_dma_memory(
            pool, batch, requests, ring_elements, tag_elements, hardware
        )
        if (
            memory.vmem_bytes <= hardware.vmem_bytes
            and memory.scalar <= hardware.smem_bytes
        ):
            return batch
    return 1


def main_kv_dma_batch(
    pool: _DmaArray, rows: _DmaArray, *, hardware: DmaHardware | None
) -> int:
    """Pure row-DMA policy; preserve the universal path for partial tiles."""
    if (
        not _carrier(pool, hardware)
        or rows.shape[1:] != pool.shape[2:]
        or rows.dtype != pool.dtype
        or len(rows) == 0
        or len(rows) % _MAIN_TILE
        or pool.shape[1] % _MAIN_TILE
    ):
        return 1
    assert hardware is not None
    batches = (
        _DMA_BATCHES if hardware.chip_version == pltpu.ChipVersion.TPU_7X else (1,)
    )
    for batch in batches:
        memory = main_kv_dma_memory(pool, rows, batch, hardware)
        if (
            memory.vmem_bytes <= hardware.vmem_bytes
            and memory.scalar <= hardware.smem_bytes
        ):
            return batch
    return 1


def get_main_kv_dma_batch(pool: jax.Array, rows: jax.Array) -> int:
    return main_kv_dma_batch(pool, rows, hardware=_hardware())


def get_ring_tag_dma_batch(
    pool: jax.Array, requests: int, *, ring_elements: int, tag_elements: int
) -> int:
    return ring_tag_dma_batch(
        pool,
        requests,
        ring_elements=ring_elements,
        tag_elements=tag_elements,
        hardware=_hardware(),
    )
