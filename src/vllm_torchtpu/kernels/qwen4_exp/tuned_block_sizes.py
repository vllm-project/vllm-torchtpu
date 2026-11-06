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
"""Measured QSA tiles and transfer-sized DMA concurrency."""

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
_MAIN_DECODE_TILES = {(16, 1, 4, 256)}
_DmaArray = jax.Array | jax.ShapeDtypeStruct

# (query bucket, heads, width, keys/page, page-table width, ratio, top-k).
# TPU7x BF16 decode: 32 pages amortize pipeline steps without the short-context
# overfetch of 64 pages. Measured at 2K/8K/32K/64K with 8 or 16 live rows;
# keep request groups at 8 because a group of 16 reads padded idle requests.
_DECODE_INDEX_PAGES = {
    (16, 4, 128, 32, 533, 4, 512): 32,
    # Thirteen adjacent subpages already share a physical sidecar allocation.
    # Four whole pages keep the 416-key page extent aligned to 128 lanes.
    (16, 4, 128, 416, 41, 4, 512): 4,
}

# Widen only the measured native BF16 reader; storage and page ownership stay
# unchanged. A full BF16 sublane tile avoids the narrow two-row DMA layout.
_DECODE_INDEX_WIDE_KEYS = {(16, 4, 128, 416, 41, 4, 512)}


def get_decode_index_key_packing(
    query: jax.Array,
    cache: jax.Array,
    *,
    pages_per_query: int,
    ratio: int,
    topk: int,
) -> int:
    """Select a zero-copy page view only for verified geometry and hardware."""
    page_size = cache.shape[1] * 2
    key = (*query.shape, page_size, pages_per_query, ratio, topk)
    if (
        key not in _DECODE_INDEX_WIDE_KEYS
        or query.dtype != jnp.bfloat16
        or cache.dtype != jnp.bfloat16
        or cache.shape[2:] != (2, query.shape[-1])
    ):
        return 2
    hw = _hardware()
    if hw is None or hw.chip_version != pltpu.ChipVersion.TPU_7X:
        return 2
    return hw.bf16_rows if page_size % hw.bf16_rows == 0 else 2


# Small mixed windows share the decode implementation. A single resident tile
# owns every request, so repeated tokens cannot race separate state writers.
_DECODE_STATE_TILES = {(16, 16, 128, 4), (16, 32, 128, 4)}


def use_decode_state_fusion(raw, pool, *, requests: int, ratio: int) -> bool:
    """Admit only measured state geometry and generation, retaining fallback."""
    if (
        (raw.shape[0], requests, raw.shape[-1], ratio) not in _DECODE_STATE_TILES
        or raw.dtype != jnp.bfloat16
        or pool.dtype != jnp.bfloat16
        or pool.ndim != 5
        or pool.shape[2:] != (1, 2, raw.shape[-1])
    ):
        return False
    hw = _hardware()
    if hw is None or hw.chip_version != pltpu.ChipVersion.TPU_7X:
        return False
    ring_elements = 8 * raw.shape[-1]
    rt, _ = region_rows(pool, ring_elements, jnp.bfloat16)
    tt, _ = region_rows(pool, 16, jnp.int32)
    memory = ring_tag_dma_memory(
        pool, min(raw.shape[0], requests), requests, ring_elements, 16, hw
    )
    return (
        rt + tt <= pool.shape[1]
        and memory.vmem_bytes <= hw.vmem_bytes
        and memory.scalar <= hw.smem_bytes
    )


def get_decode_index_page_group(
    query: jax.Array,
    pool: jax.Array,
    *,
    logical_pages: int,
    pages_per_query: int,
    ratio: int,
    topk: int,
) -> int:
    """Coalesce measured sidecars only when every stored row is a logical key.

    Physical padding or ring storage must not become addressable key columns.
    The caller owns the per-query schedule contract and performs the reshape
    inside JAX, never as a Torch view that could materialize the entire pool.
    """
    if (
        logical_pages <= 1
        or query.dtype != jnp.bfloat16
        or pool.dtype != jnp.bfloat16
        or pool.shape[-1] != query.shape[-1]
    ):
        return 1
    rows = logical_pages * (128 // ratio)
    if math.prod(pool.shape[1:]) != rows * query.shape[-1]:
        return 1
    key = (*query.shape, rows, pages_per_query, ratio, topk)
    if key not in _DECODE_INDEX_PAGES:
        return 1
    hardware = _hardware()
    if hardware is None or hardware.chip_version != pltpu.ChipVersion.TPU_7X:
        return 1
    return logical_pages


def get_decode_index_pages(
    query: jax.Array,
    cache: jax.Array,
    *,
    pages_per_query: int,
    ratio: int,
    topk: int,
) -> int:
    """Use measured BF16 decode tiles, preserving other shapes' default."""
    page_size = cache.shape[1] * 2
    fallback = max(1, 256 // page_size)
    if query.dtype != jnp.bfloat16 or cache.dtype != jnp.bfloat16:
        return fallback
    key = (*query.shape, page_size, pages_per_query, ratio, topk)
    if key not in _DECODE_INDEX_PAGES:
        return fallback
    hardware = _hardware()
    if hardware is None or hardware.chip_version != pltpu.ChipVersion.TPU_7X:
        return fallback
    return _DECODE_INDEX_PAGES[key]


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


def _hardware() -> DmaHardware:
    return DmaHardware.from_tpu_info(pltpu.get_tpu_info())


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
    """Pure row-DMA policy for full prefill tiles and measured decode buckets."""
    if (
        not _carrier(pool, hardware)
        or rows.shape[1:] != pool.shape[2:]
        or rows.dtype != pool.dtype
        or len(rows) == 0
        or (
            not (rows.shape in _MAIN_DECODE_TILES and rows.dtype == jnp.float8_e4m3fn)
            and len(rows) % _MAIN_TILE
        )
        or pool.shape[1] % _MAIN_TILE
    ):
        return 1
    assert hardware is not None
    if (
        get_main_kv_tile(rows) < _MAIN_TILE
        and hardware.chip_version != pltpu.ChipVersion.TPU_7X
    ):
        return 1
    # A resident decode bucket owns one semaphore per source row. The older
    # prefill writer retains its measured packet-size candidates.
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


def get_main_kv_tile(rows: jax.Array) -> int:
    """Keep the complete small decode bucket in one resident source tile."""
    return (
        len(rows)
        if rows.shape in _MAIN_DECODE_TILES and rows.dtype == jnp.float8_e4m3fn
        else _MAIN_TILE
    )


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


def use_query_prepare(qgate, norm, trig, rotary_dim):
    """Only the measured TPU7x decode query shape uses the fused preparation."""
    if (
        qgate.shape != (16, 24, 512)
        or qgate.dtype != jnp.bfloat16
        or norm.shape != (256,)
        or norm.dtype != jnp.bfloat16
        or trig.shape[-1] != 128
        or trig.dtype != jnp.bfloat16
        or rotary_dim != 64
    ):
        return False
    hw = _hardware()
    return hw is not None and hw.chip_version == pltpu.ChipVersion.TPU_7X


_COMPRESSED_PREPARATION_SHAPES = {((16, 128), 4, 64)}
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
    return hw.chip_version == pltpu.ChipVersion.TPU_7X


def use_compressed_preparation(raw, trig, ratio, rotary_dim):
    """Further preparation fusion only for measured state-fusion geometry."""
    return (
        (raw.shape, ratio, rotary_dim) in _COMPRESSED_PREPARATION_SHAPES
        and raw.dtype == jnp.bfloat16
        and trig.ndim == 2
        and trig.shape[1] == 128
        and trig.dtype == jnp.bfloat16
    )


def use_completed_rope_dma(trig: jax.Array, positions: jax.Array) -> bool:
    if (
        trig.ndim != 2
        or trig.shape[0] <= 0
        or trig.shape[0] % COMPLETED_ROPE_DMA_ROWS
        or trig.shape[1] != 128
        or positions.shape != (16,)
        or trig.dtype != jnp.bfloat16
        or positions.dtype != jnp.int32
    ):
        return False
    hw = _hardware()
    return hw is not None and hw.chip_version == pltpu.ChipVersion.TPU_7X


# Five contiguous 6 MiB tiles overlap output-weight reads with sparse decode.
# Measured on TPU7x for M=16, K=6144, N=2560; other weights retain the linear path.
def output_projection_tile(weight_shape: tuple[int, ...]) -> int | None:
    if len(weight_shape) != 2:
        return None
    return {(6144, 2560): 512}.get((weight_shape[0], weight_shape[1]))
