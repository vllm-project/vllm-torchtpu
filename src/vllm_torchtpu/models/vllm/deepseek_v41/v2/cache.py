"""V4.1 paged cache allocation, row addressing and mutation."""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import jax
import torch
from vllm.v1.worker.utils import allocate_kv_cache as allocate_upstream_kv_cache

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS

ROW_BYTES = ROW_WORDS * 4
# DSV4 FP8 NoPE record: 448 e4m3 values, then one e8m0 scale per lane of a
# 64-lane period (groups {j, j + 64, ...}); 448 + 64 = 512 bytes = one pool row.
NOPE_DIM = 448
NOPE_SCALE_PERIOD = 64
# Each 512-byte pool row owns 128 RoPE bytes (32 words) in a companion array,
# addressed by the same row index, as `csa_gather` expects.
ROPE_WORDS = 32


# TPU7x SparseCore hardware addressing limit: a single HBM tensor passed to
# SparseCore (`csa_gather` on the shared `int32[rows, 128]` main KV pool) must
# not exceed 16 GiB (1 << 34 = 17,179,869,184 bytes). Leave 64 MiB of page
# alignment headroom below 16 GiB when sizing the shared backing pool.
SPARSECORE_MAX_POOL_BYTES = (16 << 30) - (64 << 20)


def uses_fp8_rope_companion(vllm_config) -> bool:
    return "DeepseekV41ForCausalLM" in (
        vllm_config.model_config.hf_config.architectures or []
    ) and str(vllm_config.cache_config.cache_dtype).startswith("fp8")


def reserve_rope_companion(vllm_config, available: int) -> int:
    """Leave room for the RoPE companion and cap at SparseCore's 16 GiB tensor limit."""
    if not uses_fp8_rope_companion(vllm_config):
        return available
    return min(available * 4 // 5, SPARSECORE_MAX_POOL_BYTES)


@dataclass
class MaterializedNativeKVCache:
    kv_caches: dict[str, torch.Tensor]
    raw_tensors: list[torch.Tensor]
    native_raw_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    native_cache_layouts: dict[str, tuple] = field(default_factory=dict)
    rope_pools: dict[str, torch.Tensor] = field(default_factory=dict)


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
    raw_tensors = [
        torch.zeros(
            (pool_bytes // ROW_BYTES, ROW_WORDS) if word_rows else (pool_bytes,),
            dtype=torch.int32 if word_rows else torch.int8,
            device=device,
        )
        for _ in range(backing_bytes // pool_bytes)
    ]
    caches = {}
    native_raw_cache = {}
    native_cache_layouts = {}
    for name, view in views.items():
        assert view.ndim == 4 and view.shape[1] == 1
        assert view.stride(-1) == 1 and view.stride(-2) == view.shape[-1]
        itemsize = view.element_size()
        pool_index = pool_indices[name]
        native_raw_cache[name] = raw_tensors[pool_index]
        native_cache_layouts[name] = (
            (view.shape[0], *view.shape[2:]),
            view.dtype,
            view.stride(0) * itemsize,
            view.storage_offset() * itemsize - pool_index * pool_bytes,
        )
        caches[name] = native_cache_view(
            raw_tensors[pool_index].view(torch.int8).flatten(),
            tuple(view.shape),
            view.dtype,
            view.stride(0) * itemsize,
            view.storage_offset() * itemsize - pool_index * pool_bytes,
        )
    # FP8 NoPE layers (uint8 records) keep RoPE in a companion with one
    # 32-word slot per pool row, budgeted by `reserve_rope_companion`.
    fp8_layers = [name for name, view in views.items() if view.dtype == torch.uint8]
    rope_pools = {}
    if fp8_layers:
        assert word_rows
        companions = [
            torch.zeros(
                (raw.shape[0] * ROPE_WORDS // ROW_WORDS, ROW_WORDS),
                dtype=torch.int32,
                device=device,
            )
            for raw in raw_tensors
        ]
        rope_pools = {name: companions[pool_indices[name]] for name in fp8_layers}
        raw_tensors = raw_tensors + companions
    return MaterializedNativeKVCache(
        kv_caches=caches,
        raw_tensors=raw_tensors,
        native_raw_cache=native_raw_cache,
        native_cache_layouts=native_cache_layouts,
        rope_pools=rope_pools,
    )


_read_values_op = None
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
        self.raw = raw
        self.rope = rope
        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        self.row_bytes = math.prod(self.shape[2:]) * self.dtype.itemsize
        if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != ROW_WORDS:
            raise ValueError("Native cache requires DSV4 int32[..., 128] word rows")
        if self.dtype not in (torch.bfloat16, torch.float32, torch.uint8):
            raise ValueError("Native cache records must be BF16, FP32 or FP8 bytes")
        if self.dtype == torch.uint8 and (
            self.row_bytes != ROW_BYTES
            or rope is None
            or rope.shape != (raw.shape[0] * ROPE_WORDS // ROW_WORDS, ROW_WORDS)
        ):
            raise ValueError("FP8 records are one pool row plus a RoPE companion")
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

    def pages(self, block_table):
        """``block_table`` as pages of ``shape[1]`` raw rows, or None if unaligned.

        Page ``p`` is raw rows ``[p * shape[1], (p + 1) * shape[1])``, and its
        RoPE companion rows ``[p * shape[1] / 4, ...)``.
        """
        rows = self.shape[1]
        if (
            rows % 4
            or self.row_offset % rows
            or self.page_stride % rows
            or self.raw.shape[0] % rows
        ):
            return None
        return block_table * (self.page_stride // rows) + self.row_offset // rows

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

    def write(self, slots, values, after=None):
        """``after`` (optional) must be computed before the in-place write."""
        records = self.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        if self.dtype == torch.uint8:
            self._write_fp8(records, values)
            return
        values = values.to(self.dtype).reshape(
            slots.shape[0], self.row_bytes // self.dtype.itemsize
        )
        if self.raw.device.type == "cpu":
            pool = self.raw.view(torch.int8).reshape(-1, self.row_bytes)
            valid = slots >= 0
            pool.index_copy_(
                0, records[valid].long(), values[valid].contiguous().view(torch.int8)
            )
        elif after is not None:
            assert _write_values_after_op is not None
            self.raw.copy_(
                _write_values_after_op(self.raw, records.int(), values, after)
            )
        else:
            assert _write_values_op is not None
            self.raw.copy_(_write_values_op(self.raw, records.int(), values))

    def _write_fp8(self, records, values):
        """Quantize BF16 latents into NoPE rows and their RoPE companion slots."""
        assert self.rope is not None
        if self.raw.device.type == "cpu":
            import numpy as np

            latent = values.to(torch.bfloat16).float().numpy()
            words, rope = (
                torch.from_numpy(np.array(array))
                for array in encode_fp8_kv(jax.numpy.asarray(latent))
            )
            valid = records >= 0
            rows = records[valid].long()
            self.raw.index_copy_(0, rows, words[valid])
            companion = self.rope.view(-1, ROPE_WORDS)
            companion.index_copy_(0, rows, rope[valid])
            return
        assert _encode_fp8_op is not None and _write_values_op is not None
        words, rope = _encode_fp8_op(values.to(torch.bfloat16))
        records = records.int()
        self.raw.copy_(_write_values_op(self.raw, records, words))
        self.rope.copy_(_write_values_op(self.rope, records, rope))


class NativeCacheView:
    """Expose each shared physical pool once, deriving layer views in the graph."""

    tpu_native_kv = True

    @property
    def kv_cache(self):
        raw = getattr(self, "_tpu_raw_cache", None)
        if torch.compiler.is_compiling() and raw is not None:
            return PagedCacheRows(
                raw, self._tpu_cache_layout, getattr(self, "_tpu_rope_pool", None)
            )
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


def _permute(x: jax.Array, source: jax.Array) -> jax.Array:
    """``x[..., source]`` as an exact 0/1 matmul (finite ``x``; subnormals flush).

    Lane-stride slices and lane gathers are slow on TPU; the MXU is not.
    """
    import jax.numpy as jnp

    permutation = (jnp.arange(x.shape[-1])[:, None] == source[None]).astype(x.dtype)
    return jnp.einsum(
        "...d,de->...e",
        x,
        permutation,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def piece_order(elements: int) -> jax.Array:
    """Element order of `gather_cache_pieces`: per word row, even then odd."""
    import jax.numpy as jnp

    width = min(elements // 2, ROW_WORDS)
    return jnp.concatenate(
        [
            start + 2 * jnp.arange(width) + parity
            for start in range(0, elements, 2 * width)
            for parity in (0, 1)
        ]
    )


def split_halves(x: jax.Array) -> jax.Array:
    """Reorder the last axis as `piece_order` (``split`` reads)."""
    return _permute(x, piece_order(x.shape[-1]))


def merge_halves(x: jax.Array) -> jax.Array:
    """Inverse of `split_halves`."""
    import jax.numpy as jnp

    return _permute(x, jnp.argsort(piece_order(x.shape[-1])))


def split_pieces(x: jax.Array, pieces: int) -> jax.Array:
    """Reorder the last axis into ``[pieces, *x.shape[:-1], d // pieces]``.

    Placing ``pieces`` on the leading axis instead of the second-to-last axis
    preserves ``(heads, 128)`` as the minor dimensions on TPU, avoiding 5D
    sublane-padded ``(..., pieces=4, 128)`` reshapes (~5.95 ms/chunk).
    """
    import jax.numpy as jnp

    dim = x.shape[-1]
    order = piece_order(dim)
    perm = (jnp.arange(dim)[:, None] == order[None]).reshape(dim, pieces, dim // pieces)
    return jnp.einsum(
        "...d,dpe->p...e",
        x,
        perm.astype(x.dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def merge_pieces(x: jax.Array) -> jax.Array:
    """Inverse of `split_pieces`, contracting leading ``pieces`` into the last axis."""
    import jax.numpy as jnp

    pieces = x.shape[0]
    piece_dim = x.shape[-1]
    dim = pieces * piece_dim
    order = piece_order(dim)
    perm = (jnp.arange(dim)[:, None] == order[None]).reshape(dim, pieces, piece_dim)
    return jnp.einsum(
        "p...e,dpe->...d",
        x,
        perm.astype(x.dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def _valid_records(cache, records, record_words):
    import jax.numpy as jnp

    spread = jnp.arange(records.size, dtype=records.dtype).reshape(records.shape)
    spread %= cache.shape[0] * ROW_WORDS // record_words
    return jnp.where(records >= 0, records, spread)


def _gather_records(cache, records, record_words):
    return _gather_words(
        cache, _valid_records(cache, records, record_words), record_words
    )


def _split_words(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    import jax.numpy as jnp

    def bf16(bits):
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)

    return bf16(words << 16), bf16(words & jnp.int32(-65536))


def gather_cache_pieces(
    cache: jax.Array, records: jax.Array, elements: int
) -> jax.Array:
    """BF16 records as stacked ``[pieces, n, width]`` in `piece_order`.

    See `gather_cache_values`. Records spanning several word rows are read
    word-row-major in one gather (``[rows, n, 128]`` needs no relayout, unlike
    an interleaved ``[2n, 128] -> [n, 256]``; one gather stays on SparseCore).
    """
    import jax.numpy as jnp

    record_words = elements // 2
    if record_words <= ROW_WORDS:
        return jnp.stack(_split_words(_gather_records(cache, records, record_words)))
    chunks = record_words // ROW_WORDS
    records = _valid_records(cache, records, record_words) * chunks
    rows = (records[None] + jnp.arange(chunks, dtype=records.dtype)[:, None]).reshape(
        -1
    )
    even, odd = _split_words(cache[rows].reshape(chunks, records.size, ROW_WORDS))
    return jnp.stack([even, odd], axis=1).reshape(2 * chunks, records.size, ROW_WORDS)


def gather_cache_values(
    cache: jax.Array,
    records: jax.Array,
    elements: int,
    fp32: bool = False,
    split: bool = False,
) -> jax.Array:
    """Read physical records without flattening or converting the whole pool.

    Negative records are invalid and return arbitrary rows; callers mask them.
    They read distinct rows: clamping them all to row 0 serializes the
    SparseCore gather on the repeated index (padded decode, short top-k).
    ``split`` returns BF16 elements reordered as `piece_order` (see
    `split_halves`), which needs no lane gather.
    """
    import jax.numpy as jnp

    if split:
        if elements // 2 <= ROW_WORDS:
            words = _gather_records(cache, records, elements // 2)
            return jnp.concatenate(_split_words(words), axis=1)
        return jnp.concatenate(
            list(gather_cache_pieces(cache, records, elements)), axis=1
        )
    record_words = elements if fp32 else elements // 2
    words = _gather_records(cache, records, record_words)
    if fp32:
        return jax.lax.bitcast_convert_type(words, jnp.float32)
    # Keep the consumer's final shape; rank-increasing bitcasts add TPU padding.
    columns = jnp.arange(elements)[None]
    selected = jnp.take_along_axis(words, columns // 2, axis=1)
    bits = (selected >> ((columns % 2) * 16)) << 16
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def gather_cache_page_values(
    cache: jax.Array, first_records: jax.Array, page_records: int, elements: int
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

    record_words = elements // 2
    if ROW_WORDS % record_words or (page_records * record_words) % ROW_WORDS:
        raise ValueError("Pages must cover whole pool word rows")
    per_row = ROW_WORDS // record_words
    rows = page_records * record_words // ROW_WORDS
    first_records = first_records.reshape(-1)
    spread = jnp.arange(first_records.size, dtype=first_records.dtype) * rows
    spread %= max(1, cache.shape[0] - rows + 1)
    starts = jnp.where(first_records >= 0, first_records // per_row, spread)
    # A single-row index gather stays on SparseCore; vmap(dynamic_slice)
    # lowers to a windowed gather that measured ~25x slower on TPU7x.
    indices = (starts[:, None] + jnp.arange(rows, dtype=starts.dtype)[None]).reshape(-1)
    words = cache[indices].reshape(first_records.size * page_records, record_words)
    return jnp.concatenate(_split_words(words), axis=1)


def encode_fp8_kv(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """BF16 ``[n, 512]`` latents -> DSV4 FP8 NoPE rows and RoPE words.

    Returns int32 ``[n, 128]`` rows (the ``(4, 128)`` uint8 slab of
    `csa_cache_layout`: 448 e4m3 values, then 64 lane-periodic e8m0 scales)
    and int32 ``[n, 32]`` RoPE words for the companion. Matches the V4
    compressor's quantizer, so `sparse_ragged_paged_attention` reads it.
    """
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

    nope = latent[:, :NOPE_DIM].astype(jnp.float32)
    groups = NOPE_DIM // NOPE_SCALE_PERIOD
    amax = jnp.abs(nope).reshape(-1, groups, NOPE_SCALE_PERIOD).max(axis=1)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    exponent = jnp.ceil(jnp.log2(jnp.maximum(amax, 1e-4) / fp8_max)).astype(jnp.int32)
    # Build the power of two from its bits: exp2 is approximate on some
    # backends, and a scale off by one ulp flips e4m3 rounding at ties.
    biased = exponent + 127
    scales = jax.lax.bitcast_convert_type(biased << 23, jnp.float32)
    values = (nope / jnp.tile(scales, groups)).astype(jnp.float8_e4m3fn)
    record = jnp.concatenate(
        (jax.lax.bitcast_convert_type(values, jnp.uint8), biased.astype(jnp.uint8)),
        axis=-1,
    )
    words = csa_cache_layout.slabs_to_words(
        record.reshape(-1, csa_cache_layout.SLAB_ROWS, ROW_WORDS)
    )
    rope = csa_cache_layout.rope_words(latent[:, NOPE_DIM:].astype(jnp.bfloat16))
    return words, rope


def initialize_cache_ops():
    global _read_values_op, _write_op, _write_values_op, _encode_fp8_op
    global _write_values_after_op
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
    _write_op = pallas.jax_op(
        "pallas::dsv41v2_cache_write",
        _scatter_rows,
        donate_argnums=(0,),
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
        # An in-place write that may overlap an async gather of the old pool
        # makes XLA copy the whole pool first; finish the gather instead.
        cache, _ = jax.lax.optimization_barrier((cache, after))
        return write_values(cache, records, values)

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
            latent.new_empty((latent.shape[0], ROPE_WORDS), dtype=torch.int32),
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
    cache: jax.Array, records: jax.Array, words: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Turn unique sub-row records into full word-row writes.

    Each valid record yields its row: the current row with every segment
    written in this batch replaced. Records sharing a row yield identical
    rows, so writing them in any order is equivalent to the lane scatter.
    Returns ``(row records, int32 [n, ROW_WORDS])``; invalid records stay -1.
    """
    import jax.numpy as jnp

    width = words.shape[1]
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
        source = jnp.take(words, jnp.argmax(writers, axis=1), axis=0)
        pieces.append(
            jnp.where(
                writers.any(axis=1)[:, None],
                source,
                merged[:, segment * width : (segment + 1) * width],
            )
        )
    return rows, jnp.concatenate(pieces, axis=1)


def _pack_bf16_words(values: jax.Array) -> jax.Array:
    """Pack BF16 ``[n, 2 * width]`` into int32 ``[n, width]`` via `split_halves`.

    A direct ``bitcast_convert_type(values.reshape(n, -1, 2), int32)`` lowers on
    TPU to stride-2 lane slices ``[0:511:2]`` and ``[1:512:2]`` (~6.0 ms/chunk
    across 21 SWA/compressor writes). Reordering via `split_halves` (MXU 0/1
    matmul) places even and odd elements into contiguous 128-lane halves first.
    """
    import jax.numpy as jnp

    half_width = min(values.shape[1] // 2, ROW_WORDS)
    split = split_halves(values).reshape(values.shape[0], -1, 2, half_width)
    even = jax.lax.bitcast_convert_type(
        split[:, :, 0, :].astype(jnp.float32), jnp.int32
    )
    odd = jax.lax.bitcast_convert_type(split[:, :, 1, :].astype(jnp.float32), jnp.int32)
    return (jax.lax.shift_right_logical(even, 16) | (odd & jnp.int32(-65536))).reshape(
        values.shape[0], values.shape[1] // 2
    )


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
    assert values.dtype in (jnp.bfloat16, jnp.float32, jnp.int32)
    if not values.shape[0]:
        return cache
    words = (
        values
        if values.dtype == jnp.int32
        else (
            _pack_bf16_words(values)
            if (
                values.dtype == jnp.bfloat16
                and jax.default_backend() == "tpu"
                and values.shape[1] >= 2 * ROW_WORDS
                and values.shape[1] % (2 * ROW_WORDS) == 0
            )
            else jax.lax.bitcast_convert_type(
                values.reshape(values.shape[0], -1, 2)
                if values.dtype == jnp.bfloat16
                else values,
                jnp.int32,
            )
        )
    )
    width = words.shape[1]
    if width < ROW_WORDS and jax.default_backend() == "tpu":
        # An elementwise (row, lane) scatter is slow on TPU. Rebuild whole word
        # rows instead and reuse the full-row DMA below.
        records, words = merge_sub_rows(cache, records, words)
        width = ROW_WORDS
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
    # Records per grid step whose row DMAs are in flight together. A serial
    # start/wait per 512 B row is DMA-latency bound.
    block = 64
    count = records.shape[0]
    records = jnp.pad(records, (0, -count % block), constant_values=-1)

    def kernel(record_ref, values_ref, _input_ref, output_ref, sem):
        def copies(index):
            token = pl.program_id(0) * block + index
            record = record_ref[token]
            return (record >= 0) & (record < cache.shape[0] // chunks), [
                pltpu.make_async_copy(
                    values_ref.at[
                        pl.ds((token * chunks + chunk) * ROW_WORDS, ROW_WORDS)
                    ],
                    output_ref.at[record * chunks + chunk, :],
                    sem,
                )
                for chunk in range(chunks)
            ]

        def run(method):
            def body(index, carry):
                valid, row_copies = copies(index)

                @pl.when(valid)
                def _():
                    for copy in row_copies:
                        getattr(copy, method)()

                return carry

            jax.lax.fori_loop(0, block, body, 0)

        run("start")
        run("wait")

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.shape[0] // block,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        # Flatten only the small update batch. A partial DMA from a tiled 2D
        # source can straddle its padded minor dimension; the cache stays 2D.
    )(records, words.reshape(-1), cache)
