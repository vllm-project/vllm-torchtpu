"""V4.1 paged cache allocation, row addressing and mutation."""

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import torch

from vllm_torchtpu.distributed.pallas_shapes import get_kernel_mesh
from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS

ROW_BYTES = ROW_WORDS * 4
INDEX_HEAD_DIM = 128


@dataclass(frozen=True)
class IndexPageGeometry:
    """One compact NoPE key plane and the companion int32 scale plane."""

    nope_bytes: int
    scale_bytes: int
    key_bytes = INDEX_HEAD_DIM
    scale_word_bytes = torch.int32.itemsize
    state_bytes = key_bytes + scale_word_bytes

    def __post_init__(self):
        if self.nope_bytes <= 0 or self.nope_bytes % ROW_BYTES:
            raise ValueError("Index pages require complete native word rows")
        if self.keys_per_page * self.scale_word_bytes > self.scale_bytes:
            raise ValueError("V4.1 index scales exceed the companion array budget")

    @property
    def keys_per_page(self):
        return self.nope_bytes // self.key_bytes


class ByteLayout(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype
    page_stride_bytes: int
    row_offset_bytes: int


def cache_byte_layout(raw, cache, layout):
    """Normalize allocator geometry once at the binding boundary."""
    if layout is not None:
        return layout if isinstance(layout, ByteLayout) else ByteLayout(*layout)
    if raw.device.type != "cpu":
        raise ValueError("TPU native caches require explicit allocator layout")
    return ByteLayout(
        tuple(cache.shape),
        cache.dtype,
        cache.stride(0) * cache.element_size(),
        cache.storage_offset() * cache.element_size(),
    )


@dataclass(frozen=True)
class RecordLayout:
    page_size: int
    page_stride: int
    row_offset: int
    elements: int
    dtype: torch.dtype

    @property
    def geometry(self):
        return self.page_size, self.page_stride, self.row_offset


@dataclass(frozen=True)
class NativeBinding:
    pool: torch.Tensor
    layout: RecordLayout
    scales: torch.Tensor | None = None
    kernel_shape: tuple[int, ...] | None = None


class NativeCacheBinding:
    """Store one validated binding; never wrap tensor reads or writes."""

    tpu_native_kv = True
    _native: NativeBinding | None = None

    def native(self) -> NativeBinding:
        if self._native is None:
            raise RuntimeError(f"{type(self).__name__} requires native cache binding")
        return self._native

    @property
    def has_native_cache(self):
        return self._native is not None

    @property
    def bound_cache(self):
        """Stable allocator object for host-side cache lifetime tracking."""
        return self._bound_cache

    @property
    def kv_cache(self):
        if self._native is not None:
            binding = self._native
            return (
                binding.pool.view(binding.kernel_shape)
                if binding.kernel_shape is not None
                else binding.pool
            )
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_native(
        self, raw, layout: RecordLayout | None, *, scales=None, kernel_shape=None
    ):
        """Store a pool with record geometry already validated by its owner."""
        self._native = None
        if raw is None:
            return
        if layout is None:
            raise ValueError("Native cache binding requires validated record geometry")
        self._native = NativeBinding(raw, layout, scales, kernel_shape)

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None):
        self._native = None
        if not isinstance(cache, torch.Tensor):
            raise TypeError("This cache binding requires a single Tensor")
        cast(Any, super()).bind_kv_cache(cache)
        if raw_cache is not None:
            layout = cache_byte_layout(raw_cache, self.bound_cache, layout)
            self.bind_native(raw_cache, cache_record_layout(raw_cache, layout))

    def get_kv_cache_spec(self, vllm_config):
        from vllm_torchtpu.runner.kv_cache_dsv4 import deepseek_v41_cache_spec

        return deepseek_v41_cache_spec(
            cast(Any, super()).get_kv_cache_spec(vllm_config),
            vllm_config,
        )


@dataclass
class MaterializedNativeKVCache:
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
    raw_tensors: list[torch.Tensor]
    native_raw_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    native_cache_layouts: dict[str, ByteLayout] = field(default_factory=dict)


def native_cache_view(raw, shape, dtype, block_stride_bytes, offset_bytes):
    """Express a native block-compact view with compositional view operations.

    Unlike as_strided followed by a dtype view, these operations can be
    functionalized when a compiled model mutates several views of one pool.
    """
    page_offset = offset_bytes % block_stride_bytes
    pool_offset = offset_bytes - page_offset
    page_bytes = math.prod(shape[1:]) * dtype.itemsize
    pages = raw[pool_offset : pool_offset + shape[0] * block_stride_bytes].reshape(
        shape[0], block_stride_bytes
    )
    return pages[:, page_offset : page_offset + page_bytes].view(dtype).reshape(shape)


_read_values_op = None
_write_op = None
_write_values_op = None


def cache_record_layout(raw, layout):
    """Validate once at binding and retain geometry in record units."""
    shape, dtype, stride_bytes, offset_bytes = layout
    row_bytes = math.prod(shape[2:]) * dtype.itemsize
    if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != ROW_WORDS:
        raise ValueError("Native cache requires DSV4 int32[..., 128] word rows")
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Native cache records must be BF16 or FP32")
    if row_bytes % 4 or (row_bytes % ROW_BYTES and ROW_BYTES % row_bytes):
        raise ValueError("Cache records must divide or be a multiple of 512 bytes")
    if stride_bytes % row_bytes or offset_bytes % row_bytes:
        raise ValueError("Cache pages must be record-aligned")
    if max(raw.shape[0], raw.nbytes // row_bytes) > torch.iinfo(torch.int32).max:
        raise ValueError("Native cache row counts must fit int32")
    return RecordLayout(
        shape[1],
        stride_bytes // row_bytes,
        offset_bytes // row_bytes,
        math.prod(shape[2:]),
        dtype,
    )


def row_indices(slots, page_size, page_stride, row_offset):
    """Map slots using prevalidated record geometry, never pool byte offsets."""
    return row_offset + slots // page_size * page_stride + slots % page_size


def masked_row_indices(slots, page_size, page_stride, row_offset):
    import jax.numpy as jnp

    records = row_indices(jnp.maximum(slots, 0), page_size, page_stride, row_offset)
    return jnp.where(slots >= 0, records, -1)


def read_cache(cache, slots, *, layout=None):
    """Read old state through the existing cache-read operation."""
    if layout is None:
        return cache[slots // cache.shape[1], slots % cache.shape[1]]
    page_size, page_stride, row_offset = layout.geometry
    elements, dtype = layout.elements, layout.dtype
    if cache.device.type == "cpu":
        records = row_indices(slots, page_size, page_stride, row_offset)
        pool = cache.view(torch.int8).reshape(-1, elements * dtype.itemsize)
        return (
            pool[records.long()]
            .contiguous()
            .view(dtype)
            .reshape(*slots.shape, elements)
        )
    assert _read_values_op is not None
    gathered = _read_values_op(
        cache,
        slots.flatten().int(),
        elements,
        dtype == torch.float32,
        page_size,
        page_stride,
        row_offset,
    )
    return gathered.reshape(*slots.shape, elements)


def read_cache_values(
    cache: jax.Array,
    slots: jax.Array,
    elements: int,
    fp32: bool,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> jax.Array:
    records = row_indices(slots, page_size, page_stride, row_offset)
    return gather_cache_values(cache, records, elements, fp32)


def _scatter_rows(cache: jax.Array, slots: jax.Array, values: jax.Array) -> jax.Array:
    import jax.numpy as jnp

    # Invalid/padded tokens must drop their write, not alias the last row.
    rows = cache.reshape((-1, *values.shape[1:]))
    slots = jnp.where(slots >= 0, slots, rows.shape[0])
    return rows.at[slots].set(values, mode="drop").reshape(cache.shape)


def _gather_words(cache, records, record_words):
    import jax.numpy as jnp

    assert cache.dtype == jnp.int32 and cache.shape[1] == ROW_WORDS
    if record_words >= ROW_WORDS:
        chunks = record_words // ROW_WORDS
        indices = (records[:, None] * chunks + jnp.arange(chunks)[None]).reshape(-1)
        return cache[indices].reshape(records.size, record_words)
    # Like DSV4's packed RoPE records, several records can share a word row.
    # Select contiguous segments; a per-element dynamic gather is costly on TPU.
    per_row = ROW_WORDS // record_words
    words = cache[records // per_row]
    selected = words[:, :record_words]
    for segment in range(1, per_row):
        selected = jnp.where(
            (records % per_row == segment)[:, None],
            words[:, segment * record_words : (segment + 1) * record_words],
            selected,
        )
    return selected


def gather_cache_values(
    cache: jax.Array, records: jax.Array, elements: int, fp32: bool = False
) -> jax.Array:
    """Read physical records without flattening or converting the whole pool."""
    import jax.numpy as jnp

    words = _gather_words(cache, records, elements if fp32 else elements // 2)
    if fp32:
        return jax.lax.bitcast_convert_type(words, jnp.float32)
    # Keep the consumer's final shape; rank-increasing bitcasts add TPU padding.
    columns = jnp.arange(elements)[None]
    selected = jnp.take_along_axis(words, columns // 2, axis=1)
    bits = (selected >> ((columns % 2) * 16)) << 16
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def initialize_cache_ops():
    global _read_values_op, _write_op, _write_values_op
    if _write_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _read_values_op = pallas.jax_op(
        "pallas::deepseek_v41_cache_read_values",
        read_cache_values,
        mesh=get_kernel_mesh(get_vllm_model_wrapper_context().mesh),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(), P(), None, None, None, None, None)
        ),
    )
    _read_values_op.register_fake(
        lambda cache, slots, elements, fp32, *geometry: cache.new_empty(
            (slots.numel(), elements), dtype=torch.float32 if fp32 else torch.bfloat16
        )
    )
    _write_op = pallas.jax_op(
        "pallas::deepseek_v41_cache_write",
        _scatter_rows,
        mesh=get_kernel_mesh(get_vllm_model_wrapper_context().mesh),
        input_partition_specs=cast(tuple[tuple[str, ...], ...], (P(), P(), P())),
    )
    _write_op.register_fake(lambda cache, slots, values: torch.empty_like(cache))
    mesh = get_vllm_model_wrapper_context().mesh
    # Pallas DMA runs independently on each device's replicated cache.
    # Mosaic calls require explicit manual partitioning on a multi-device mesh.
    scatter_values = jax.shard_map(
        scatter_cache_values,
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )

    def write_values(
        cache: jax.Array,
        slots: jax.Array,
        values: jax.Array,
        page_size: int,
        page_stride: int,
        row_offset: int,
    ) -> jax.Array:
        records = masked_row_indices(slots, page_size, page_stride, row_offset)
        return scatter_values(cache, records, values)

    _write_values_op = pallas.jax_op(
        "pallas::deepseek_v41_cache_write_values",
        write_values,
        donate_argnums=(0,),
        mesh=get_kernel_mesh(mesh),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(), P(), P(), None, None, None)
        ),
    )
    _write_values_op.register_fake(
        lambda cache, slots, values, *geometry: torch.empty_like(cache)
    )


def write_cache(cache, slots, values, valid, *, layout=None):
    """Valid destinations must be unique; invalid rows have no side effects."""
    count = values.shape[0]
    slots, valid = slots[:count], valid[:count]
    slots = slots.masked_fill(~valid, -1).long()
    if layout is not None:
        page_size, page_stride, row_offset = layout.geometry
        elements, dtype = layout.elements, layout.dtype
        values = values.to(dtype).reshape(slots.shape[0], elements)
        if cache.device.type == "cpu":
            records = row_indices(
                slots.clamp(min=0), page_size, page_stride, row_offset
            )
            pool = cache.view(torch.int8).reshape(-1, elements * dtype.itemsize)
            valid = slots >= 0
            pool.index_copy_(
                0, records[valid].long(), values[valid].contiguous().view(torch.int8)
            )
        else:
            assert _write_values_op is not None
            cache.copy_(_write_values_op(cache, slots.int(), values, *layout.geometry))
        return
    if cache.device.type == "cpu":
        selected = slots[valid]
        if cache.ndim == values.ndim:
            cache.index_copy_(0, selected, values[valid])
        else:
            cache.index_put_(
                (selected // cache.shape[1], selected % cache.shape[1]), values[valid]
            )
    else:
        assert _write_op is not None
        cache.copy_(_write_op(cache, slots.int(), values))


def scatter_cache_values(
    cache: jax.Array, records: jax.Array, values: jax.Array
) -> jax.Array:
    """Write unique physical records, preserving the shared owner and padding.

    Full word rows use DSV4-style aliased DMA. Sub-row writes use independent
    (row, lane) coordinates, so two records sharing a row cannot overwrite
    each other through concurrent read/modify/write of the entire word row.
    """
    import jax.numpy as jnp

    assert cache.dtype == jnp.int32 and cache.ndim == 2 and cache.shape[1] == ROW_WORDS
    assert values.dtype in (jnp.bfloat16, jnp.float32)
    if not values.shape[0]:
        return cache
    words = jax.lax.bitcast_convert_type(
        values.reshape(values.shape[0], -1, 2)
        if values.dtype == jnp.bfloat16
        else values,
        jnp.int32,
    )
    return scatter_cache_words(cache, records, words)


def scatter_cache_words(
    cache: jax.Array, records: jax.Array, words: jax.Array
) -> jax.Array:
    """Publish packed words in full rows or independent sub-row records."""
    import jax.numpy as jnp

    assert cache.dtype == words.dtype == jnp.int32
    assert cache.ndim == words.ndim == 2 and cache.shape[1] == ROW_WORDS
    if not words.shape[0]:
        return cache
    width = words.shape[1]
    assert width % ROW_WORDS == 0 or ROW_WORDS % width == 0
    if width < ROW_WORDS and jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v41.cache_write import write_cache_words

        return write_cache_words(cache, records, words)
    if width < ROW_WORDS or jax.default_backend() != "tpu":
        columns = jnp.arange(width)[None]
        if width < ROW_WORDS:
            per_row = ROW_WORDS // width
            rows = records[:, None] // per_row
            lanes = (records[:, None] % per_row) * width + columns
        else:
            rows = records[:, None] * (width // ROW_WORDS) + columns // ROW_WORDS
            lanes = jnp.broadcast_to(columns % ROW_WORDS, rows.shape)
        rows = jnp.where(records[:, None] >= 0, rows, cache.shape[0])
        return cache.at[rows, lanes].set(words, mode="drop")

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    chunks = width // ROW_WORDS

    def kernel(record_ref, values_ref, _input_ref, output_ref, row_ref, sem):
        token, chunk = pl.program_id(0), pl.program_id(1)
        record = record_ref[token]
        row = record * chunks + chunk

        @pl.when((record >= 0) & (record < cache.shape[0] // chunks))
        def write():
            source = values_ref.at[
                pl.ds((token * chunks + chunk) * ROW_WORDS, ROW_WORDS)
            ]
            destination = output_ref.at[row, :]
            pltpu.make_async_copy(source, row_ref, sem).start()
            pltpu.make_async_copy(source, row_ref, sem).wait()
            pltpu.make_async_copy(row_ref, destination, sem).start()
            pltpu.make_async_copy(row_ref, destination, sem).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.shape[0], chunks),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((ROW_WORDS,), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        # Flatten only the small update batch. A partial DMA from a tiled 2D
        # source can straddle its padded minor dimension; the cache stays 2D.
    )(records, words.reshape(-1), cache)


def block_table(metadata):
    return metadata.block_tables.reshape(metadata.seq_lens.shape[0], -1)


def token_requests(metadata, positions):
    rows = torch.arange(positions.shape[0], device=positions.device)
    return torch.bucketize(rows, metadata.query_start_loc[1:], right=True)


def cache_slots(metadata, positions, *, rows_per_page, compress_ratio=1, ring=False):
    """Derive this cache's addresses from the shared TPU metadata contract."""
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = block_table(metadata)
    requests = token_requests(metadata, positions).clamp(max=table.shape[0] - 1)
    logical = positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    blocks = table[requests, columns.clamp(max=table.shape[1] - 1).long()].long()
    slots = blocks * rows_per_page + logical % rows_per_page
    # Padded request boundaries repeat the last real token offset. Keep the
    # mask on device, including dummy runs and batches with no active tokens.
    rows = torch.arange(positions.shape[0], device=positions.device)
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


def row_slots(block_table, requests, logical, *, rows_per_page):
    """Gather fixed-width token rows without inspecting device values on host."""
    safe = logical.clamp(min=0, max=block_table.shape[1] * rows_per_page - 1)
    blocks = block_table[requests[:, None], safe // rows_per_page].long()
    return blocks * rows_per_page + safe % rows_per_page
