"""V4.1 paged cache allocation, row addressing and mutation."""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import jax
import torch

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS

ROW_BYTES = ROW_WORDS * 4


@dataclass
class MaterializedNativeKVCache:
    kv_caches: dict[str, torch.Tensor]
    raw_tensors: list[torch.Tensor]
    native_raw_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    native_cache_layouts: dict[str, tuple] = field(default_factory=dict)


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


class _CacheBinding(Protocol):
    def bind_kv_cache(self, cache: Any) -> None: ...

    def get_kv_cache_spec(self, vllm_config: Any) -> Any: ...


class PagedCacheRows:
    """Map upstream pages to physical record indices, as in DSV4.

    The shared owner stays in DSV4's native int32[..., 128] layout. Only
    gathered records are reinterpreted. Byte geometry is converted once to
    record units, before any device arithmetic or int32 index conversion.
    """

    def __init__(self, raw, layout):
        self.raw = raw
        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        self.row_bytes = math.prod(self.shape[2:]) * self.dtype.itemsize
        if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != ROW_WORDS:
            raise ValueError("Native cache requires DSV4 int32[..., 128] word rows")
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Native cache records must be BF16 or FP32")
        if self.row_bytes % 4 or (
            self.row_bytes % ROW_BYTES and ROW_BYTES % self.row_bytes
        ):
            raise ValueError("Cache records must divide or be a multiple of 512 bytes")
        if stride_bytes % self.row_bytes or offset_bytes % self.row_bytes:
            raise ValueError("Cache pages must be record-aligned")
        if (
            max(raw.shape[0], raw.nbytes // self.row_bytes)
            > torch.iinfo(torch.int32).max
        ):
            raise ValueError("Native cache row counts must fit int32")
        self.page_stride = stride_bytes // self.row_bytes
        self.row_offset = offset_bytes // self.row_bytes

    def numel(self):
        return math.prod(self.shape)

    def row_indices(self, slots):
        return (
            self.row_offset
            + slots // self.shape[1] * self.page_stride
            + slots % self.shape[1]
        )

    def __getitem__(self, indices):
        blocks, rows = indices
        records = self.row_indices(blocks * self.shape[1] + rows)
        if self.raw.device.type == "cpu":
            pool = self.raw.view(torch.int8).reshape(-1, self.row_bytes)
            return (
                pool[records.long()]
                .contiguous()
                .view(self.dtype)
                .reshape(*records.shape, *self.shape[2:])
            )
        assert _read_values_op is not None
        gathered = _read_values_op(
            self.raw,
            records.flatten().int(),
            self.row_bytes // self.dtype.itemsize,
            self.dtype == torch.float32,
        )
        return gathered.reshape(*records.shape, *self.shape[2:])

    def write(self, slots, values):
        records = self.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        values = values.to(self.dtype).reshape(
            slots.shape[0], self.row_bytes // self.dtype.itemsize
        )
        if self.raw.device.type == "cpu":
            pool = self.raw.view(torch.int8).reshape(-1, self.row_bytes)
            valid = slots >= 0
            pool.index_copy_(
                0, records[valid].long(), values[valid].contiguous().view(torch.int8)
            )
        else:
            assert _write_values_op is not None
            self.raw.copy_(_write_values_op(self.raw, records.int(), values))


class NativeCacheView:
    """Expose each shared physical pool once, deriving layer views in the graph."""

    tpu_native_kv = True

    def get_kv_cache_spec(self, vllm_config):
        from vllm_torchtpu.runner.kv_cache_dsv4 import deepseek_v41_cache_spec

        return deepseek_v41_cache_spec(
            cast(_CacheBinding, super()).get_kv_cache_spec(vllm_config), vllm_config
        )

    @property
    def kv_cache(self):
        raw = getattr(self, "_tpu_raw_cache", None)
        if torch.compiler.is_compiling() and raw is not None:
            return PagedCacheRows(raw, self._tpu_cache_layout)
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None):
        cast(_CacheBinding, super()).bind_kv_cache(cache)
        self._tpu_raw_cache = raw_cache
        if layout is not None:
            self._tpu_cache_layout = layout
            return
        if raw_cache is not None and raw_cache.device.type != "cpu":
            raise ValueError("TPU native caches require explicit allocator layout")
        view = self._bound_cache
        self._tpu_cache_layout = (
            tuple(view.shape),
            view.dtype,
            view.stride(0) * view.element_size(),
            view.storage_offset() * view.element_size(),
        )


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
        gather_cache_values,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), None, None),
    )
    _read_values_op.register_fake(
        lambda cache, records, elements, fp32=False: cache.new_empty(
            (records.numel(), elements), dtype=torch.float32 if fp32 else torch.bfloat16
        )
    )
    _write_op = pallas.jax_op(
        "pallas::deepseek_v41_cache_write",
        _scatter_rows,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P()),
    )
    _write_op.register_fake(lambda cache, slots, values: torch.empty_like(cache))
    mesh = get_vllm_model_wrapper_context().mesh
    # Pallas DMA runs independently on each device's replicated cache.
    # Mosaic calls require explicit manual partitioning on a multi-device mesh.
    write_values = jax.shard_map(
        scatter_cache_values,
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    _write_values_op = pallas.jax_op(
        "pallas::deepseek_v41_cache_write_values",
        write_values,
        donate_argnums=(0,),
        mesh=mesh,
        input_partition_specs=(P(), P(), P()),
    )
    _write_values_op.register_fake(
        lambda cache, records, values: torch.empty_like(cache)
    )


def write_cache(cache, slots, values, valid):
    """Valid destinations must be unique; invalid rows have no side effects."""
    count = values.shape[0]
    slots, valid = slots[:count], valid[:count]
    slots = slots.masked_fill(~valid, -1).long()
    if isinstance(cache, PagedCacheRows):
        cache.write(slots, values)
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


def _block_table(metadata):
    return metadata.block_tables.reshape(metadata.seq_lens.shape[0], -1)


def _token_requests(metadata, positions):
    rows = torch.arange(positions.shape[0], device=positions.device)
    return torch.bucketize(rows, metadata.query_start_loc[1:], right=True)


def _cache_slots(
    cache, metadata, positions, *, compress_ratio=1, ring=False, rows_per_page=None
):
    """Derive this cache's addresses from the shared TPU metadata contract."""
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = _block_table(metadata)
    requests = _token_requests(metadata, positions).clamp(max=table.shape[0] - 1)
    rows_per_page = rows_per_page or cache.shape[1]
    logical = positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    blocks = table[requests, columns.clamp(max=table.shape[1] - 1).long()].long()
    slots = blocks * rows_per_page + logical % rows_per_page
    # Padded request boundaries repeat the last real token offset. Keep the
    # mask on device, including dummy runs and batches with no active tokens.
    rows = torch.arange(positions.shape[0], device=positions.device)
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


def _row_slots(cache, block_table, requests, logical):
    """Gather fixed-width token rows without inspecting device values on host."""
    safe = logical.clamp(min=0, max=block_table.shape[1] * cache.shape[1] - 1)
    blocks = block_table[requests[:, None], safe // cache.shape[1]].long()
    return blocks * cache.shape[1] + safe % cache.shape[1]
