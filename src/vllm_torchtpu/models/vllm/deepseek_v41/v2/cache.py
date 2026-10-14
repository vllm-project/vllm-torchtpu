"""V4.1 paged cache allocation, row addressing and mutation."""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import jax
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import allocate_kv_cache as allocate_upstream_kv_cache

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.fp8_cache import encode_fp8_kv
from vllm_torchtpu.kernels.mla.cache_layout import (
    pack_cache_values,
    packed_record_shape,
    unpack_cache_values,
)

ROW_BYTES = ROW_WORDS * 4
# SparseMLA's gathered RoPE ABI still uses 32 compact words per token.
ROPE_WORDS = 32
FP8_RECORD_WORDS = 2 * ROW_WORDS


# TPU7x SparseCore hardware addressing limit: a single HBM tensor passed to
# SparseCore (`csa_gather` on the shared `int32[rows, 128]` main KV pool) must
# not exceed 16 GiB (1 << 34 = 17,179,869,184 bytes). Leave 64 MiB of page
# alignment headroom below 16 GiB when sizing the shared backing pool.
SPARSECORE_MAX_POOL_BYTES = (16 << 30) - (64 << 20)
SPARSECORE_MAX_POOL_BANKS = 2
SPARSECORE_MIN_SECOND_BANK_BYTES = SPARSECORE_MAX_POOL_BYTES // 4

RawCachePool = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def uses_native_fp8_cache(vllm_config) -> bool:
    return "DeepseekV41ForCausalLM" in (
        vllm_config.model_config.hf_config.architectures or []
    ) and str(vllm_config.cache_config.cache_dtype).startswith("fp8")


def cap_native_cache_budget(vllm_config, available: int, *, auto: bool = False) -> int:
    """Cap the pool; RoPE is now budgeted in each FP8 record's cache spec."""
    if not uses_native_fp8_cache(vllm_config):
        return available
    if auto and available > 0:
        # Keep five fractional budget bits: rounding loses less than 1/32
        # while small HBM measurement changes share the same compiled KV shape.
        # At 32--64 GiB this gives 1-GiB steps.
        quantum = 1 << max(0, available.bit_length() - 6)
        available = available // quantum * quantum
    main_budget = available
    if main_budget < SPARSECORE_MAX_POOL_BYTES + SPARSECORE_MIN_SECOND_BANK_BYTES:
        return min(main_budget, SPARSECORE_MAX_POOL_BYTES)
    return min(
        main_budget,
        SPARSECORE_MAX_POOL_BANKS * SPARSECORE_MAX_POOL_BYTES,
    )


@dataclass
class MaterializedNativeKVCache:
    kv_caches: dict[str, torch.Tensor]
    raw_tensors: list[torch.Tensor]
    native_raw_cache: dict[str, RawCachePool] = field(default_factory=dict)
    native_cache_layouts: dict[str, tuple] = field(default_factory=dict)
    rope_pools: dict[str, RawCachePool] = field(default_factory=dict)


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


def materialize_upstream_kv_cache(
    kv_cache_config, device, layout, *, word_rows=False
) -> MaterializedNativeKVCache:
    """Keep upstream placements, with one TPU input per shared physical region."""
    from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

    views = allocate_upstream_kv_cache(kv_cache_config, torch.device("meta"), layout)
    if not views:
        return MaterializedNativeKVCache({}, [])
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1
    backing_bytes = sizes.pop()
    if layout.is_layer_compact:
        # Reuse the same physical-region split as the hybrid attention/Mamba
        # path. Overlaid groups keep one owner, while unrelated layers no
        # longer functionalize mutations of the entire model cache.
        pool_indices = layer_to_pool_index(kv_cache_config)
        pool_bytes = kv_cache_config.num_blocks * max(
            group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups
        )
    else:
        pool_indices = dict.fromkeys(views, 0)
        pool_bytes = backing_bytes
    # Use DSV4's native 128-word rows at the kernel boundary. Allocate
    # in that layout: reshaping a flat byte pool inside XLA copies the
    # entire pool on every read, even when only a few rows are requested.
    if word_rows:
        assert pool_bytes % ROW_BYTES == 0, pool_bytes
    num_pools = backing_bytes // pool_bytes
    split_banks = word_rows and pool_bytes > SPARSECORE_MAX_POOL_BYTES
    if split_banks:
        num_blocks = kv_cache_config.num_blocks
        block_stride_bytes = pool_bytes // num_blocks
        block_rows = block_stride_bytes // ROW_BYTES
        lo_blocks = SPARSECORE_MAX_POOL_BYTES // block_stride_bytes
        hi_blocks = num_blocks - lo_blocks
        lo_rows = lo_blocks * block_rows
        hi_rows = hi_blocks * block_rows
        assert max(lo_rows, hi_rows) * ROW_BYTES <= SPARSECORE_MAX_POOL_BYTES
        pool_owners: list[RawCachePool] = [
            (
                torch.zeros((lo_rows, ROW_WORDS), dtype=torch.int32, device=device),
                torch.zeros((hi_rows, ROW_WORDS), dtype=torch.int32, device=device),
            )
            for _ in range(num_pools)
        ]
        raw_tensors = [
            bank
            for pair in cast(list[tuple[torch.Tensor, torch.Tensor]], pool_owners)
            for bank in pair
        ]
    else:
        raw_tensors = [
            torch.zeros(
                (pool_bytes // ROW_BYTES, ROW_WORDS) if word_rows else (pool_bytes,),
                dtype=torch.int32 if word_rows else torch.int8,
                device=device,
            )
            for _ in range(num_pools)
        ]
        pool_owners = list(raw_tensors)
    caches = {}
    native_raw_cache: dict[str, RawCachePool] = {}
    native_cache_layouts = {}
    specs = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            specs[name] = (
                spec.kv_cache_specs[name]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
    for name, view in views.items():
        assert view.ndim == 4 and view.shape[1] == 1
        assert view.stride(-1) == 1 and view.stride(-2) == view.shape[-1]
        itemsize = view.element_size()
        pool_index = pool_indices[name]
        stride_bytes = view.stride(0) * itemsize
        offset_bytes = view.storage_offset() * itemsize - pool_index * pool_bytes
        owner = pool_owners[pool_index]
        native_raw_cache[name] = owner
        native_cache_layouts[name] = (
            (view.shape[0], view.shape[2], specs[name].head_size),
            view.dtype,
            stride_bytes,
            offset_bytes,
        )
        if isinstance(owner, tuple):
            single_page = native_cache_view(
                owner[0].view(torch.int8).flatten(),
                (1, *view.shape[1:]),
                view.dtype,
                stride_bytes,
                offset_bytes % stride_bytes,
            )
            caches[name] = single_page.expand(tuple(view.shape))
        else:
            caches[name] = native_cache_view(
                owner.view(torch.int8).flatten(),
                tuple(view.shape),
                view.dtype,
                stride_bytes,
                offset_bytes,
            )
    # NoPE/scales and padded RoPE occupy consecutive rows in the same record.
    # Keep the binding alias, but never allocate a model-wide companion pool.
    fp8_layers = [name for name, view in views.items() if view.dtype == torch.uint8]
    rope_pools = {name: native_raw_cache[name] for name in fp8_layers}
    return MaterializedNativeKVCache(
        kv_caches=caches,
        raw_tensors=raw_tensors,
        native_raw_cache=native_raw_cache,
        native_cache_layouts=native_cache_layouts,
        rope_pools=rope_pools,
    )


_read_values_op = None
_read_values_2bank_op = None
_write_op = None
_write_values_op = None
_write_values_after_op = None
_encode_fp8_op = None


class _CacheBinding(Protocol):
    def bind_kv_cache(self, cache: Any) -> None: ...


class PagedCacheRows:
    """Map upstream pages to physical record indices, as in DSV4.

    The shared owner stays in DSV4's native int32[..., 128] layout. Only
    gathered records are reinterpreted. Byte geometry is converted once to
    record units, before any device arithmetic or int32 index conversion.
    """

    def __init__(self, raw, layout, rope=None):
        if isinstance(raw, tuple):
            if len(raw) != 2:
                raise ValueError("Split native cache requires exactly two banks")
            self.raw, self.raw_hi = raw
        else:
            self.raw, self.raw_hi = raw, None
        if isinstance(rope, tuple):
            if len(rope) != 2:
                raise ValueError("Split RoPE companion requires exactly two banks")
            self.rope, self.rope_hi = rope
        else:
            self.rope, self.rope_hi = rope, None
        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        if self.dtype not in (torch.bfloat16, torch.float32, torch.uint8):
            raise ValueError("Native cache records must be BF16, FP32 or FP8 bytes")
        _, width = packed_record_shape(math.prod(self.shape[2:]), self.dtype.itemsize)
        self.row_bytes = (
            FP8_RECORD_WORDS * 4 if self.dtype == torch.uint8 else width * 4
        )
        banks = (self.raw,) if self.raw_hi is None else (self.raw, self.raw_hi)
        if any(
            bank.dtype != torch.int32 or bank.ndim != 2 or bank.shape[1] != ROW_WORDS
            for bank in banks
        ):
            raise ValueError("Native cache requires DSV4 int32[..., 128] word rows")
        if self.dtype == torch.uint8 and (
            self.shape[-1] != 512
            or self.rope is None
            or (self.rope_hi is not None) != (self.raw_hi is not None)
            or self.rope.shape != self.raw.shape
            or (
                self.raw_hi is not None
                and self.rope_hi is not None
                and self.rope_hi.shape != self.raw_hi.shape
            )
        ):
            raise ValueError("FP8 records contain NoPE and RoPE in the same pool")
        if self.row_bytes % 4 or (
            self.row_bytes % ROW_BYTES and ROW_BYTES % self.row_bytes
        ):
            raise ValueError("Cache records must divide or be a multiple of 512 bytes")
        if (
            stride_bytes % self.row_bytes
            or offset_bytes % self.row_bytes
            or self.raw.nbytes % self.row_bytes
        ):
            raise ValueError("Cache pages must be record-aligned")
        self.total_rows = self.raw.shape[0] + (
            0 if self.raw_hi is None else self.raw_hi.shape[0]
        )
        total_bytes = self.raw.nbytes + (
            0 if self.raw_hi is None else self.raw_hi.nbytes
        )
        if (
            max(self.total_rows, total_bytes // self.row_bytes)
            > torch.iinfo(torch.int32).max
        ):
            raise ValueError("Native cache row counts must fit int32")
        self.bank_records = self.raw.nbytes // self.row_bytes
        self.page_stride = stride_bytes // self.row_bytes
        self.row_offset = offset_bytes // self.row_bytes

    def numel(self):
        return math.prod(self.shape)

    @property
    def device(self):
        return self.raw.device

    def row_indices(self, slots):
        return (
            self.row_offset
            + slots // self.shape[1] * self.page_stride
            + slots % self.shape[1]
        )

    def pages(self, block_table):
        """``block_table`` as pages of ``shape[1]`` raw rows, or None if unaligned.

        Page ``p`` is raw rows ``[p * shape[1], (p + 1) * shape[1])``, and its
        FP8 records occupy two consecutive physical rows each.
        """
        rows = self.shape[1]
        if (
            rows % 4
            or self.row_offset % rows
            or self.page_stride % rows
            or self.bank_records % rows
            or (self.raw_hi is not None and self.raw_hi.nbytes // self.row_bytes % rows)
        ):
            return None
        return block_table * (self.page_stride // rows) + self.row_offset // rows

    def copy_raw(self, updated):
        if self.raw_hi is None:
            self.raw.copy_(updated)
        else:
            updated_lo, updated_hi = updated
            self.raw.copy_(updated_lo)
            self.raw_hi.copy_(updated_hi)

    def __getitem__(self, indices):
        blocks, rows = indices
        records = self.row_indices(blocks * self.shape[1] + rows)
        if self.raw.device.type == "cpu":
            pool = self.raw.view(torch.int8).reshape(-1, self.row_bytes)
            if self.raw_hi is None:
                selected = pool[records.long()]
            else:
                pool_hi = self.raw_hi.view(torch.int8).reshape(-1, self.row_bytes)
                rec = records.long()
                in_hi = rec >= self.bank_records
                lo_idx = rec.masked_fill(in_hi, 0)
                hi_idx = (rec - self.bank_records).masked_fill(~in_hi, 0)
                selected = torch.where(
                    in_hi.unsqueeze(-1), pool_hi[hi_idx], pool[lo_idx]
                )
            words = selected.contiguous().view(torch.int32)
            if self.dtype == torch.bfloat16:
                low = (words & 0xFFFF).to(torch.uint16).view(torch.bfloat16)
                high = ((words >> 16) & 0xFFFF).to(torch.uint16).view(torch.bfloat16)
                values = torch.cat((low, high), -1)
            else:
                values = words.view(self.dtype)
            return values[..., : self.shape[-1]].reshape(
                *records.shape, *self.shape[2:]
            )
        elem_count = math.prod(self.shape[2:])
        is_fp32 = self.dtype == torch.float32
        if self.raw_hi is None:
            assert _read_values_op is not None
            gathered = _read_values_op(
                self.raw,
                records.flatten().int(),
                elem_count,
                is_fp32,
            )
        else:
            assert _read_values_2bank_op is not None
            gathered = _read_values_2bank_op(
                self.raw,
                self.raw_hi,
                records.flatten().int(),
                elem_count,
                is_fp32,
            )
        return gathered.reshape(*records.shape, *self.shape[2:])

    def write(self, slots, values, after=None):
        """``after`` (optional) must be computed before the in-place write."""
        records = self.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        if self.dtype == torch.uint8:
            self._write_fp8(records, values)
            return
        values = values.to(self.dtype).reshape(slots.shape[0], -1)
        if self.raw.device.type == "cpu":
            width = self.row_bytes // 4
            if self.dtype == torch.bfloat16:
                bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
                bits = torch.nn.functional.pad(bits, (0, 2 * width - bits.shape[-1]))
                values = bits[:, :width] | (bits[:, width:] << 16)
            else:
                values = torch.nn.functional.pad(values, (0, width - values.shape[-1]))
                values = values.contiguous().view(torch.int32)
            pool = self.raw.view(torch.int8).reshape(-1, self.row_bytes)
            valid = (slots >= 0) & (
                True if self.raw_hi is None else (records < self.bank_records)
            )
            pool.index_copy_(
                0, records[valid].long(), values[valid].contiguous().view(torch.int8)
            )
            if self.raw_hi is not None:
                pool_hi = self.raw_hi.view(torch.int8).reshape(-1, self.row_bytes)
                valid_hi = (slots >= 0) & (records >= self.bank_records)
                pool_hi.index_copy_(
                    0,
                    (records[valid_hi] - self.bank_records).long(),
                    values[valid_hi].contiguous().view(torch.int8),
                )
            return
        rec_lo = records.int()
        if after is not None:
            assert _write_values_after_op is not None
            self.raw.copy_(_write_values_after_op(self.raw, rec_lo, values, after))
            if self.raw_hi is not None:
                rec_hi = torch.where(
                    rec_lo >= self.bank_records, rec_lo - self.bank_records, -1
                )
                self.raw_hi.copy_(
                    _write_values_after_op(self.raw_hi, rec_hi, values, after)
                )
        else:
            assert _write_values_op is not None
            self.raw.copy_(_write_values_op(self.raw, rec_lo, values))
            if self.raw_hi is not None:
                rec_hi = torch.where(
                    rec_lo >= self.bank_records, rec_lo - self.bank_records, -1
                )
                self.raw_hi.copy_(_write_values_op(self.raw_hi, rec_hi, values))

    def _write_fp8(self, records, values):
        """Write combined NoPE/scales and padded BF16 RoPE records."""
        assert self.rope is not None
        if self.raw.device.type == "cpu":
            import numpy as np

            latent = values.to(torch.bfloat16).float().numpy()
            words, rope = (
                torch.from_numpy(np.array(array))
                for array in encode_fp8_kv(jax.numpy.asarray(latent))
            )
            packed = torch.cat((words, rope), dim=1)
            valid_lo = (records >= 0) & (
                True if self.raw_hi is None else (records < self.bank_records)
            )
            rows_lo = records[valid_lo].long()
            self.raw.view(-1, FP8_RECORD_WORDS).index_copy_(
                0, rows_lo, packed[valid_lo]
            )
            if self.raw_hi is not None:
                assert self.rope_hi is not None
                valid_hi = records >= self.bank_records
                rows_hi = (records[valid_hi] - self.bank_records).long()
                self.raw_hi.view(-1, FP8_RECORD_WORDS).index_copy_(
                    0, rows_hi, packed[valid_hi]
                )
            return
        assert _encode_fp8_op is not None and _write_values_op is not None
        words, rope = _encode_fp8_op(values.to(torch.bfloat16))
        packed = torch.cat((words, rope), dim=1)
        rec_lo = records.int()
        self.raw.copy_(_write_values_op(self.raw, rec_lo, packed))
        if self.raw_hi is not None:
            assert self.rope_hi is not None
            rec_hi = torch.where(
                rec_lo >= self.bank_records, rec_lo - self.bank_records, -1
            )
            self.raw_hi.copy_(_write_values_op(self.raw_hi, rec_hi, packed))


class NativeCacheView:
    """Expose each shared physical pool once, deriving layer views in the graph."""

    tpu_native_kv = True

    @property
    def kv_cache(self):
        raw = getattr(self, "_tpu_raw_cache", None)
        if raw is not None:
            assert self._tpu_paged_cache is not None
            return self._tpu_paged_cache
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None, rope_pool=None):
        cast(_CacheBinding, super()).bind_kv_cache(cache)
        self._tpu_raw_cache = raw_cache
        self._tpu_rope_pool = rope_pool
        if layout is not None:
            self._tpu_cache_layout = layout
        else:
            raw_tensor = raw_cache[0] if isinstance(raw_cache, tuple) else raw_cache
            if raw_tensor is not None and raw_tensor.device.type != "cpu":
                raise ValueError("TPU native caches require explicit allocator layout")
            view = self._bound_cache
            self._tpu_cache_layout = (
                tuple(view.shape),
                view.dtype,
                view.stride(0) * view.element_size(),
                view.storage_offset() * view.element_size(),
            )
        # Engram uses the cache object's identity to retain token history.
        self._tpu_paged_cache = (
            PagedCacheRows(raw_cache, self._tpu_cache_layout, rope_pool)
            if raw_cache is not None
            else None
        )


def _scatter_rows(cache: jax.Array, slots: jax.Array, values: jax.Array) -> jax.Array:
    import jax.numpy as jnp

    # Invalid/padded tokens must drop their write, not alias the last row.
    rows = cache.reshape((-1, *values.shape[1:]))
    slots = jnp.where(slots >= 0, slots, rows.shape[0])
    return rows.at[slots].set(values, mode="drop").reshape(cache.shape)


def _cache_num_rows(cache: jax.Array | tuple[jax.Array, jax.Array]) -> int:
    import jax.numpy as jnp

    if isinstance(cache, tuple):
        cache_lo, cache_hi = cache
        assert cache_lo.dtype == cache_hi.dtype == jnp.int32
        assert cache_lo.shape[1] == cache_hi.shape[1] == ROW_WORDS
        return cache_lo.shape[0] + cache_hi.shape[0]
    assert cache.dtype == jnp.int32 and cache.shape[1] == ROW_WORDS
    return cache.shape[0]


def _gather_raw_rows(
    cache: jax.Array | tuple[jax.Array, jax.Array], indices: jax.Array
) -> jax.Array:
    import jax.numpy as jnp

    if not isinstance(cache, tuple):
        return cache[indices]
    cache_lo, cache_hi = cache
    lo_rows = cache_lo.shape[0]
    hi_rows = cache_hi.shape[0]
    in_hi = indices >= lo_rows
    spread = jnp.arange(indices.size, dtype=indices.dtype).reshape(indices.shape)
    idx_lo = jnp.where(in_hi, spread % lo_rows, indices)
    idx_hi = jnp.where(in_hi, indices - lo_rows, spread % hi_rows)
    return jnp.where(in_hi[..., None], cache_hi[idx_hi], cache_lo[idx_lo])


def _gather_words(cache, records, record_words):
    import jax.numpy as jnp

    _cache_num_rows(cache)
    if record_words >= ROW_WORDS:
        chunks = record_words // ROW_WORDS
        indices = (records[:, None] * chunks + jnp.arange(chunks)[None]).reshape(-1)
        return _gather_raw_rows(cache, indices).reshape(records.size, record_words)
    # Like DSV4's packed RoPE records, several records can share a word row.
    # Select contiguous segments; a per-element dynamic gather is costly on TPU.
    per_row = ROW_WORDS // record_words
    words = _gather_raw_rows(cache, records // per_row)
    selected = words[:, :record_words]
    for segment in range(1, per_row):
        selected = jnp.where(
            (records % per_row == segment)[:, None],
            words[:, segment * record_words : (segment + 1) * record_words],
            selected,
        )
    return selected


def split_halves(x: jax.Array) -> jax.Array:
    """The native feature layout needs no even/odd permutation."""
    return x


def merge_halves(x: jax.Array) -> jax.Array:
    return x


def split_pieces(x: jax.Array, pieces: int) -> jax.Array:
    """Put contiguous 128-feature chunks on a leading axis for attention."""
    split = x.reshape(*x.shape[:-1], pieces, x.shape[-1] // pieces)
    return jax.numpy.moveaxis(split, -2, 0)


def merge_pieces(x: jax.Array) -> jax.Array:
    merged = jax.numpy.moveaxis(x, 0, -2)
    return merged.reshape(*x.shape[1:-1], x.shape[0] * x.shape[-1])


def _valid_records(cache, records, record_words):
    import jax.numpy as jnp

    total_rows = _cache_num_rows(cache)
    spread = jnp.arange(records.size, dtype=records.dtype).reshape(records.shape)
    spread %= total_rows * ROW_WORDS // record_words
    return jnp.where(records >= 0, records, spread)


def _gather_records(cache, records, record_words):
    return _gather_words(
        cache, _valid_records(cache, records, record_words), record_words
    )


def _split_words(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    import jax.numpy as jnp

    def bf16(bits):
        return jax.lax.bitcast_convert_type(bits.astype(jnp.uint16), jnp.bfloat16)

    return bf16(words), bf16(jax.lax.shift_right_logical(words, 16))


def gather_cache_pieces(
    cache: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    elements: int,
) -> jax.Array:
    """Gather contiguous feature chunks as ``[pieces, records, 128]``."""
    import jax.numpy as jnp

    if elements % ROW_WORDS:
        return gather_cache_values(cache, records, elements)[None]
    _, record_words = packed_record_shape(elements, 2)
    chunks = record_words // ROW_WORDS
    records = _valid_records(cache, records, record_words) * chunks
    rows = (records[None] + jnp.arange(chunks, dtype=records.dtype)[:, None]).reshape(
        -1
    )
    low, high = _split_words(
        _gather_raw_rows(cache, rows).reshape(chunks, records.size, ROW_WORDS)
    )
    pieces = jnp.concatenate((low, high), axis=0)
    if elements < ROW_WORDS:
        return pieces[:1, :, :elements]
    return pieces[: elements // ROW_WORDS]


def gather_cache_values(
    cache: jax.Array,
    records: jax.Array,
    elements: int,
    fp32: bool = False,
    split: bool = False,
) -> jax.Array:
    """Gather padded physical records and return their original feature order.

    Negative records read arbitrary distributed rows; consumers mask them.
    `split` is retained for callers whose inputs used the old split ordering.
    """
    import jax.numpy as jnp

    _, width = packed_record_shape(elements, 4 if fp32 else 2)
    words = _gather_records(cache, records, width)
    return unpack_cache_values(words, elements, jnp.float32 if fp32 else jnp.bfloat16)


def gather_cache_page_words(
    cache: jax.Array | tuple[jax.Array, jax.Array],
    first_records: jax.Array,
    page_records: int,
    elements: int,
) -> jax.Array:
    """Raw int32 word rows ``[pages * rows, ROW_WORDS]`` of whole pages.

    Each padded index record occupies one complete 128-word row;
    see `gather_cache_page_values` for the natural feature-order view.
    """
    import jax.numpy as jnp

    _, record_words = packed_record_shape(elements, 2)
    if ROW_WORDS % record_words or (page_records * record_words) % ROW_WORDS:
        raise ValueError("Pages must cover whole pool word rows")
    total_rows = _cache_num_rows(cache)
    per_row = ROW_WORDS // record_words
    rows = page_records * record_words // ROW_WORDS
    first_records = first_records.reshape(-1)
    spread = jnp.arange(first_records.size, dtype=first_records.dtype) * rows
    if isinstance(cache, tuple):
        spread %= max(1, cache[0].shape[0] - rows + 1)
    else:
        spread %= max(1, total_rows - rows + 1)
    starts = jnp.where(first_records >= 0, first_records // per_row, spread)
    # A single-row index gather stays on SparseCore; vmap(dynamic_slice)
    # lowers to a windowed gather that measured ~25x slower on TPU7x.
    indices = (starts[:, None] + jnp.arange(rows, dtype=starts.dtype)[None]).reshape(-1)
    return _gather_raw_rows(cache, indices)


def gather_cache_page_values(
    cache: jax.Array | tuple[jax.Array, jax.Array],
    first_records: jax.Array,
    page_records: int,
    elements: int,
) -> jax.Array:
    """`gather_cache_values` (``split=True``) of whole pages.

    Returns BF16 ``[pages * page_records, elements]`` for records
    ``first + 0 .. first + page_records - 1`` of every ``first`` in
    ``first_records``: one windowed gather of ``page_records`` contiguous
    records per page instead of one gather per record. The caller guarantees
    that each page's records are contiguous and that ``first`` starts a pool
    word row; negative ``first`` reads arbitrary distinct rows (callers mask).
    """
    import jax.numpy as jnp

    words = gather_cache_page_words(cache, first_records, page_records, elements)
    _, width = packed_record_shape(elements, 2)
    words = words.reshape(first_records.size * page_records, width)
    return unpack_cache_values(words, elements, jnp.bfloat16)


def initialize_cache_ops():
    global _read_values_op, _read_values_2bank_op, _write_op, _write_values_op
    global _encode_fp8_op, _write_values_after_op
    if _write_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _read_values_op = pallas.jax_op(
        "pallas::dsv41v2_cache_read_values",
        gather_cache_values,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), None, None),
    )
    _read_values_op.register_fake(
        lambda cache, records, elements, fp32=False: cache.new_empty(
            (records.numel(), elements), dtype=torch.float32 if fp32 else torch.bfloat16
        )
    )

    def read_cache_values_2bank(
        cache_lo: jax.Array,
        cache_hi: jax.Array,
        records: jax.Array,
        elements: int,
        fp32: bool = False,
    ) -> jax.Array:
        return gather_cache_values(
            cast(Any, (cache_lo, cache_hi)), records, elements, fp32
        )

    _read_values_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_cache_read_values_2bank",
        read_cache_values_2bank,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P(), None, None),
    )
    _read_values_2bank_op.register_fake(
        lambda cache_lo, cache_hi, records, elements, fp32=False: cache_lo.new_empty(
            (records.numel(), elements), dtype=torch.float32 if fp32 else torch.bfloat16
        )
    )
    _write_op = pallas.jax_op(
        "pallas::dsv41v2_cache_write",
        _scatter_rows,
        donate_argnums=(0,),
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P()),
    )
    _write_op.register_fake(lambda cache, slots, values: torch.empty_like(cache))
    mesh = get_vllm_model_wrapper_context().mesh

    def write_values(
        cache: jax.Array, records: jax.Array, values: jax.Array
    ) -> jax.Array:
        return scatter_cache_values(cache, records, values)

    # Pallas DMA runs independently on each device's replicated cache.
    # Mosaic calls require explicit manual partitioning on a multi-device mesh.
    write_values = jax.shard_map(
        write_values,
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    _write_values_op = pallas.jax_op(
        "pallas::dsv41v2_cache_write_values",
        write_values,
        donate_argnums=(0,),
        mesh=mesh,
        input_partition_specs=(P(), P(), P()),
    )
    _write_values_op.register_fake(
        lambda cache, records, values: torch.empty_like(cache)
    )

    def write_values_after(
        cache: jax.Array, records: jax.Array, values: jax.Array, after: jax.Array
    ) -> jax.Array:
        return scatter_cache_values(cache, records, values, after=after)

    write_values_after = jax.shard_map(
        write_values_after,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )

    _write_values_after_op = pallas.jax_op(
        "pallas::dsv41v2_cache_write_values_after",
        write_values_after,
        donate_argnums=(0,),
        mesh=mesh,
        input_partition_specs=(P(), P(), P(), P()),
    )
    _write_values_after_op.register_fake(
        lambda cache, records, values, after: torch.empty_like(cache)
    )
    _encode_fp8_op = pallas.jax_op(
        "pallas::dsv41v2_encode_fp8_kv",
        encode_fp8_kv,
        mesh=mesh,
        input_partition_specs=(P(),),
    )
    _encode_fp8_op.register_fake(
        lambda latent: (
            latent.new_empty((latent.shape[0], ROW_WORDS), dtype=torch.int32),
            latent.new_empty((latent.shape[0], ROW_WORDS), dtype=torch.int32),
        )
    )


def write_cache(cache, slots, values, valid, after=None):
    """Valid destinations must be unique; invalid rows have no side effects.

    ``after``: a value (e.g. rows read from ``cache``) to finish first.
    """
    count = values.shape[0]
    slots, valid = slots[:count], valid[:count]
    slots = slots.masked_fill(~valid, -1).long()
    if isinstance(cache, PagedCacheRows):
        cache.write(slots, values, after)
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


def merge_sub_rows(
    cache: jax.Array, records: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Turn unique sub-row records into full word-row writes.

    Each valid record yields its row: the current row with every segment
    written in this batch replaced. Records sharing a row yield identical
    rows, so writing them in any order is equivalent to the lane scatter.
    Returns ``(row records, int32 [n, ROW_WORDS])``; invalid records stay -1.
    """
    import jax.numpy as jnp

    width = values.shape[1]
    per_row = ROW_WORDS // width
    valid = (records >= 0) & (records < cache.shape[0] * per_row)
    rows = jnp.where(valid, records // per_row, -1)
    segments = records % per_row
    merged = cache[jnp.maximum(rows, 0)]
    # [n, n]: which batch records land in each record's row. Batches are a
    # step's tokens (<= max_num_batched_tokens), so this stays small.
    same_row = (rows[:, None] == rows[None, :]) & valid[:, None] & valid[None, :]
    pieces = []
    for segment in range(per_row):
        writers = same_row & (segments[None, :] == segment)
        source = jnp.take(values, jnp.argmax(writers, axis=1), axis=0)
        pieces.append(
            jnp.where(
                writers.any(axis=1)[:, None],
                source,
                merged[:, segment * width : (segment + 1) * width],
            )
        )
    return rows, jnp.concatenate(pieces, axis=1)


def _pack_bf16_words(values: jax.Array) -> jax.Array:
    return pack_cache_values(values)


def scatter_cache_values(
    cache: jax.Array,
    records: jax.Array,
    values: jax.Array,
    *,
    after: jax.Array | None = None,
) -> jax.Array:
    """Write unique physical records, preserving the shared owner and padding.

    Full word rows use DSV4-style aliased DMA. Sub-row writes use independent
    (row, lane) coordinates, so two records sharing a row cannot overwrite
    each other through concurrent read/modify/write of the entire word row.
    """
    import jax.numpy as jnp

    assert values.dtype in (jnp.bfloat16, jnp.float32, jnp.int32)
    if isinstance(cache, tuple):
        cache_lo, cache_hi = cache
        record_words = (
            values.shape[1]
            if values.dtype == jnp.int32
            else packed_record_shape(values.shape[1], values.dtype.itemsize)[1]
        )
        bank_records = cache_lo.shape[0] * ROW_WORDS // record_words
        records_lo = jnp.where(records < bank_records, records, -1)
        records_hi = jnp.where(records >= bank_records, records - bank_records, -1)
        return cast(
            Any,
            (
                scatter_cache_values(cache_lo, records_lo, values, after=after),
                scatter_cache_values(cache_hi, records_hi, values, after=after),
            ),
        )
    assert cache.dtype == jnp.int32 and cache.ndim == 2 and cache.shape[1] == ROW_WORDS
    if not values.shape[0]:
        return cache
    words = values if values.dtype == jnp.int32 else pack_cache_values(values)
    width = words.shape[1]
    if jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.mla.word_cache import scatter_word_records

        return scatter_word_records(cache, records, words, after=after)
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
