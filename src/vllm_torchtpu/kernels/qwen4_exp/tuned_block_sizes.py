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
"""QSA tile defaults and tensor/resource-derived transfer policies."""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

# Maximum concurrent row transfers; padded buffers must fit SMEM and VMEM.
DEFAULT_DMA_CONCURRENCY = 8
_DMA_BATCHES = tuple(
    1 << power for power in reversed(range(DEFAULT_DMA_CONCURRENCY.bit_length()))
)
_COMPILER_SMEM_RESERVE = 32 * 1024
_MAIN_TILE = 128
_DmaArray = jax.Array | jax.ShapeDtypeStruct
DECODE_INDEX_TILE_KEYS = 1024


def get_decode_index_key_packing(
    query: _DmaArray, cache: _DmaArray, *, pages_per_query: int, ratio: int, topk: int
) -> int:
    """Widen a native BF16 reader only when whole physical pages reshape exactly."""
    if (
        query.ndim != 3
        or cache.ndim != 4
        or query.dtype != jnp.bfloat16
        or cache.dtype != jnp.bfloat16
        or cache.shape[2:] != (2, query.shape[-1])
    ):
        return 2
    hw = _hardware()
    page_size = cache.shape[1] * 2
    if hw is None or hw.bf16_rows <= 0:
        return 2
    return hw.bf16_rows if page_size % hw.bf16_rows == 0 else 2


@dataclasses.dataclass(frozen=True)
class GroupedIndexConfig:
    """Measured tiles for sharing a key read across one request's queries."""

    queries: int
    pages: int
    buffers: int


# TPU7x BF16: measured at 1/8/16/32/64 requests, with ragged 1--16-query
# windows and contexts through 128K. Eight queries share each 20-page DMA.
_GROUPED_INDEX_BLOCKS = {
    (batch, 4, 128, 416, 80, 4, 512): GroupedIndexConfig(8, 20, 3)
    for batch in (16, 32, 64, 128, 256)
}


def get_grouped_index_config(
    query: jax.Array,
    cache: jax.Array,
    *,
    pages_per_request: int,
    ratio: int,
    topk: int,
) -> GroupedIndexConfig | None:
    """Admit measured grouped schedules on the native BF16 wide reader."""
    if (
        get_decode_index_key_packing(
            query, cache, pages_per_query=pages_per_request, ratio=ratio, topk=topk
        )
        == 2
    ):
        return None
    hw = _hardware()
    if hw is None or hw.chip_version != pltpu.ChipVersion.TPU_7X:
        return None
    key = (*query.shape, cache.shape[1] * 2, pages_per_request, ratio, topk)
    return _GROUPED_INDEX_BLOCKS.get(key)


def use_decode_state_fusion(
    raw: _DmaArray,
    pool: _DmaArray,
    *,
    requests: int,
    ratio: int,
    position_axes: int = 1,
) -> bool:
    """Fuse resident state and pair writes within the small-query bucket."""
    hw = _hardware()
    if (
        not _carrier(pool, hw)
        or raw.ndim != 2
        or min(raw.shape) <= 0
        or requests <= 0
        # The whole-batch pair writer shares the per-query schedule's bucket.
        # Larger token windows require the bounded prefill/verification writer.
        or raw.shape[0] > requests
        or ratio <= 0
        or position_axes <= 0
        or raw.dtype != jnp.bfloat16
        or pool.dtype != jnp.bfloat16
        or pool.shape[2:] != (1, 2, raw.shape[-1])
    ):
        return False
    assert hw is not None
    # Persistent ring/tag storage pads compression groups to eight rows.
    ring_rows = _align(ratio, 8)
    ring_elements = ring_rows * raw.shape[-1]
    tag_elements = ring_rows * (position_axes + 1)
    rt, _ = region_rows(pool, ring_elements, jnp.bfloat16)
    tt, _ = region_rows(pool, tag_elements, jnp.int32)
    memory = ring_tag_dma_memory(
        pool, min(raw.shape[0], requests), requests, ring_elements, tag_elements, hw
    )
    resident = (
        3 * _vmem(raw.shape, jnp.bfloat16, hw)
        + _vmem(raw.shape, jnp.int32, hw)
        + _vmem((1, raw.shape[1]), jnp.bfloat16, hw)
    )
    return (
        rt + tt <= pool.shape[1]
        and memory.vmem_bytes + resident <= hw.vmem_bytes
        and memory.scalar + (requests * 2 + 1 + raw.shape[0] * position_axes) * 4
        <= hw.smem_bytes
    )


def get_decode_index_page_group(
    query: _DmaArray,
    pool: _DmaArray,
    *,
    logical_pages: int,
    pages_per_query: int,
    ratio: int,
    topk: int,
    keys_per_base_page: int,
) -> int:
    """Coalesce only pages containing logical keys, excluding padding and state."""
    if (
        logical_pages <= 1
        or keys_per_base_page <= 0
        or query.ndim != 3
        or pool.ndim != 5
        or query.dtype != jnp.bfloat16
        or pool.dtype != jnp.bfloat16
        or pool.shape[-1] != query.shape[-1]
    ):
        return 1
    rows = logical_pages * keys_per_base_page
    return logical_pages if math.prod(pool.shape[1:]) == rows * query.shape[-1] else 1


def get_decode_index_pages(query, cache, *, pages_per_query, ratio, topk) -> int | None:
    """Return a lane-aligned tile, or None if its minimum live buffers cannot fit."""
    page_size = cache.shape[1] * 2
    fallback = max(1, 256 // page_size)
    if (
        query.ndim != 3
        or cache.ndim != 4
        or query.dtype != jnp.bfloat16
        or cache.dtype != jnp.bfloat16
        or pages_per_query <= 0
        or min(query.shape) <= 0
    ):
        return fallback
    hw = _hardware()
    if hw is None or min(hw.vmem_bytes, hw.lanes, hw.bf16_rows) <= 0:
        return fallback
    page_alignment = hw.lanes // math.gcd(page_size, hw.lanes)
    pages = max(
        page_alignment,
        _align(math.ceil(DECODE_INDEX_TILE_KEYS / page_size), page_alignment),
    )
    # Score kernels mask page-table tails; keep a complete lane-aligned key tile.
    pages = min(pages, _align(pages_per_query, page_alignment))
    while pages >= page_alignment:
        keys = pages * page_size
        # BF16 key/Q operands plus FP32 per-head score and reduction planes.
        memory = (
            _vmem((keys, cache.shape[-1]), jnp.bfloat16, hw)
            + _vmem(query.shape, jnp.bfloat16, hw)
            + 2 * _vmem((query.shape[1], keys), jnp.float32, hw)
        )
        if memory <= hw.vmem_bytes * 0.8:
            return pages
        pages = (pages // 2 // page_alignment) * page_alignment
    return None


@dataclasses.dataclass(frozen=True)
class DmaHardware:
    """Immutable per-TensorCore capacity and pinned compiler tile facts."""

    vmem_bytes: int
    smem_bytes: int
    lanes: int
    bf16_rows: int
    fp8_rows: int
    int32_rows: int
    chip_version: pltpu.ChipVersion | None = None

    @classmethod
    def from_tpu_info(cls, info: pltpu.TpuInfo) -> "DmaHardware":
        return cls(
            info.vmem_capacity_bytes,
            info.smem_capacity_bytes,
            info.num_lanes,
            info.get_sublane_tiling(jnp.bfloat16),
            info.get_sublane_tiling(jnp.float8_e4m3fn),
            info.get_sublane_tiling(jnp.int32),
            info.chip_version,
        )


def _hardware() -> DmaHardware | None:
    device = jax.sharding.get_abstract_mesh().abstract_device
    if (device is None and jax.default_backend() != "tpu") or (
        device is not None and device.platform != "tpu"
    ):
        return None
    return DmaHardware.from_tpu_info(pltpu.get_tpu_info())


def _align(value: int, tile: int) -> int:
    return (value + tile - 1) // tile * tile


def _vmem(shape: tuple[int, ...], dtype: DTypeLike, hw: DmaHardware) -> int:
    dtype = jnp.dtype(dtype)
    rows = {
        jnp.dtype(jnp.bfloat16): hw.bf16_rows,
        jnp.dtype(jnp.float8_e4m3fn): hw.fp8_rows,
        jnp.dtype(jnp.int32): hw.int32_rows,
        jnp.dtype(jnp.float32): hw.int32_rows,
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
    tile = get_main_kv_tile(rows)
    source = _vmem((tile, *rows.shape[1:]), rows.dtype, hw)
    vector = hw.int32_rows * hw.lanes * 4
    # The destination is aliased HBM, not a full-page VMEM allocation.
    # Count explicit prefetched words/flags and DMA counters. As in the shared
    # metadata generator, reserve32KiB for scalar pipeline/compiler bookkeeping;
    # this is headroom, not a hardware allocation-granularity requirement.
    if tile < _MAIN_TILE:
        # Three prefetched vectors and one DMA counter per decode source row.
        scalar = 4 * len(rows) * 4 + _COMPILER_SMEM_RESERVE
    else:
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
    for batch in _DMA_BATCHES:
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
    """Bound complete resident row transfers by physical layout and capacity."""
    if (
        not _carrier(pool, hardware)
        or rows.shape[1:] != pool.shape[2:]
        or rows.dtype != pool.dtype
        or len(rows) == 0
        or (len(rows) >= _MAIN_TILE and len(rows) % _MAIN_TILE)
        or pool.shape[1] % _MAIN_TILE
    ):
        return 1
    assert hardware is not None
    batches = (len(rows),) if get_main_kv_tile(rows) < _MAIN_TILE else _DMA_BATCHES
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


def get_main_kv_tile(rows: _DmaArray) -> int:
    """Keep the complete small decode bucket in one resident source tile."""
    return min(len(rows), _MAIN_TILE)


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


def use_query_prepare(
    qgate: _DmaArray, norm: _DmaArray, trig: _DmaArray, rotary_dim: int
) -> bool:
    """Fuse norm/RoPE when the complete resident vector planes fit."""
    if (
        qgate.ndim != 3
        or min(qgate.shape) <= 0
        or qgate.shape[-1] % 2
        or qgate.dtype != jnp.bfloat16
        or norm.shape != (qgate.shape[-1] // 2,)
        or norm.dtype != jnp.bfloat16
        or trig.ndim != 2
        or trig.shape[0] <= 0
        or trig.dtype != jnp.bfloat16
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(norm.shape[0], trig.shape[1])
    ):
        return False
    hw = _hardware()
    if hw is None or min(hw.vmem_bytes, hw.lanes, hw.bf16_rows) <= 0:
        return False
    count, heads, _ = qgate.shape
    width = norm.shape[0]
    if width % hw.lanes:
        return False
    memory = (
        _vmem(qgate.shape, qgate.dtype, hw)
        + _vmem((1, width), norm.dtype, hw)
        + _vmem((count, trig.shape[1]), trig.dtype, hw)
        + _vmem((count, heads, width), jnp.bfloat16, hw)
        + 4 * _vmem((count, heads, width), jnp.float32, hw)
    )
    return memory <= hw.vmem_bytes * 0.8


COMPLETED_ROPE_DMA_ROWS = 8

_PREFILL_INDEX_PREPARATION_SHAPES = {((8192, 128), 4, 8)}


def use_prefill_index_preparation(
    values: _DmaArray, ratio: int, *, requests: int
) -> bool:
    """Admit only the measured BF16 prefill bucket on TPU7x."""
    if (
        values.shape,
        ratio,
        requests,
    ) not in _PREFILL_INDEX_PREPARATION_SHAPES or values.dtype != jnp.bfloat16:
        return False
    if jax.default_backend() != "tpu":
        return False
    hw = _hardware()
    return hw is not None and hw.chip_version == pltpu.ChipVersion.TPU_7X


def use_compressed_preparation(
    raw: _DmaArray, trig: _DmaArray, ratio: int, rotary_dim: int
) -> bool:
    """Defer RoPE only when the indexed pair writer's resident planes fit."""
    if (
        raw.ndim != 2
        or min(raw.shape) <= 0
        or raw.dtype != jnp.bfloat16
        or ratio <= 0
        or trig.ndim != 2
        or trig.shape[0] <= 0
        or trig.dtype != jnp.bfloat16
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(raw.shape[1], trig.shape[1])
    ):
        return False
    hw = _hardware()
    if hw is None or raw.shape[1] % hw.lanes:
        return False
    memory = (
        4 * _vmem(raw.shape, jnp.bfloat16, hw)
        + 4 * _vmem(raw.shape, jnp.float32, hw)
        + _vmem(
            (raw.shape[0], COMPLETED_ROPE_DMA_ROWS, trig.shape[1]), jnp.bfloat16, hw
        )
    )
    return memory <= hw.vmem_bytes * 0.8


def use_completed_rope_dma(trig: jax.Array, positions: jax.Array) -> bool:
    hw = _hardware()
    if (
        hw is None
        or trig.ndim != 2
        or trig.shape[0] <= 0
        or trig.shape[0] % COMPLETED_ROPE_DMA_ROWS
        or positions.ndim != 1
        or len(positions) <= 0
        or trig.dtype != jnp.bfloat16
        or positions.dtype != jnp.int32
        or trig.shape[1] % hw.lanes
    ):
        return False
    return (
        _vmem((len(positions), COMPLETED_ROPE_DMA_ROWS, trig.shape[1]), trig.dtype, hw)
        <= hw.vmem_bytes * 0.8
    )


# Output-column DMA tile; the final tile is zero-padded during weight packing.
OUTPUT_PROJECTION_TILE_COLUMNS = 512


def output_projection_tile(weight_shape: tuple[int, ...]) -> int | None:
    if len(weight_shape) != 2 or min(weight_shape) <= 0:
        return None
    hw = _hardware()
    if hw is None:
        return None
    tile = OUTPUT_PROJECTION_TILE_COLUMNS
    tiles = math.ceil(weight_shape[1] / tile)
    # Necessary resident-weight lower bound before making a permanent HBM copy.
    # The launch planner additionally includes attention, HC and scratch buffers.
    weight_bytes = _vmem((tiles, weight_shape[0], tile), jnp.bfloat16, hw)
    return tile if weight_bytes <= hw.vmem_bytes * 0.8 else None
