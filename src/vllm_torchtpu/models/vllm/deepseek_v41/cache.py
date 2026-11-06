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
"""DeepSeek-V4.1 paged KV cache allocation, row addressing, and mutation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from vllm.v1.worker.utils import allocate_kv_cache as allocate_upstream_kv_cache

from vllm_torchtpu import utils
from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import cache_op

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

ROW_WORDS: int = 128
ROPE_WORDS: int = 32
WORD_BYTES: int = torch.int32.itemsize
ROW_BYTES: int = ROW_WORDS * WORD_BYTES
ROPE_COMPANION_DIVISOR: int = ROW_WORDS // ROPE_WORDS
PAGE_ROW_ALIGNMENT: int = 4
BF16_SHIFT_BITS: int = 16
BF16_EVEN_MASK: int = 0xFFFF
BF16_ODD_MASK: int = -65536
DEFAULT_PAGE_TOKENS: int = 16
ZERO_BLOCKS_SMALL_BUCKET: int = 256
ZERO_BLOCKS_LARGE_BUCKET: int = 2048
AUTO_MEMORY_MANTISSA_BITS: int = 6
COMPANION_BUDGET_NUMERATOR: int = 4
COMPANION_BUDGET_DENOMINATOR: int = 5
SPARSECORE_MAX_POOL_BANKS: int = 2
# TPU7x SparseCore 16 GiB (1 << 34) single-tensor HBM addressing limit minus
# 64 MiB alignment headroom.
SPARSECORE_MAX_POOL_BYTES: int = (16 << 30) - (64 << 20)


def _alloc_word_pools(
    low_rows: int,
    high_rows: int,
    num_pools: int,
    device: torch.device,
) -> tuple[list[utils.RawCachePool], list[Tensor]]:
    """Allocate 1-bank or 2-bank ``int32[rows, 128]`` word pools on ``device``.

    Args:
        low_rows: Row count per pool in the primary bank (``<= 16 GiB`` window).
        high_rows: Row count per pool in the overflow bank (``0`` for 1-bank).
        num_pools: Number of independent raw pools to materialize.
        device: Target PyTorch device for the backing ``int32`` tensors.

    Returns:
        Tuple ``(owners, flat_tensors)`` where ``owners`` contains either single
        ``Tensor`` banks or ``(low_bank, high_bank)`` pairs (Created).
    """
    low_banks = [
        torch.zeros((low_rows, ROW_WORDS), dtype=torch.int32, device=device)
        for _ in range(num_pools)
    ]
    if high_rows <= 0:
        return list(low_banks), low_banks
    high_banks = [
        torch.zeros((high_rows, ROW_WORDS), dtype=torch.int32, device=device)
        for _ in range(num_pools)
    ]
    owners: list[utils.RawCachePool] = list(zip(low_banks, high_banks))
    flat_tensors = [bank for pair in zip(low_banks, high_banks) for bank in pair]
    return owners, flat_tensors


def _expand_chunks(records: Tensor, chunks_per_record: int) -> Tensor:
    """Expand record indices into consecutive 128-word row indices.

    Args:
        records: ``int32[num_records]`` logical record indices (``< 0`` invalid).
        chunks_per_record: Number of 128-word rows per logical record.

    Returns:
        ``int32[num_records * chunks_per_record]`` physical row indices (Created).
    """
    offsets = torch.arange(chunks_per_record, dtype=torch.int32, device=records.device)
    expanded = torch.where(
        records[:, None] >= 0,
        records[:, None] * chunks_per_record + offsets,
        -1,
    )
    return expanded.reshape(-1)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _zero_pools_tpu(
    low_banks: list[Tensor],
    high_banks: list[Tensor | None],
    records: Tensor,
    zero_rows: Tensor,
    high_row_offset: int,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> None:
    """Zero selected 128-word rows in-place across 1-bank or 2-bank TPU pools.

    Args:
        low_banks: Primary ``int32[rows, 128]`` HBM bank tensors (Changed).
        high_banks: Optional overflow ``int32[rows, 128]`` HBM bank tensors (Changed).
        records: ``int32[bucket_rows]`` target row indices (Used; ``< 0`` ignored).
        zero_rows: ``int32[bucket_rows, 128]`` zero payload tensor (Used).
        high_row_offset: Row count of each primary bank where ``high_banks`` start.
        page_tokens: Contiguous rows per physical page for burst DMA writes.
    """
    for low_bank, high_bank in zip(low_banks, high_banks):
        low_bank.copy_(
            cache_op.dsv41_write_sub_rows(
                low_bank, zero_rows, records, 0, page_tokens=page_tokens
            )
        )
        if high_bank is not None:
            high_bank.copy_(
                cache_op.dsv41_write_sub_rows(
                    high_bank,
                    zero_rows,
                    records,
                    high_row_offset,
                    page_tokens=page_tokens,
                )
            )


class PagedCacheRows:
    """Map upstream pages to physical record indices in ``int32[rows, 128]`` banks.

    Data lifecycle:
        - Created: ``__getitem__`` unpacks gathered ``int32`` words into a fresh
          ``float32`` or ``bfloat16`` tensor; ``pages``/``row_indices`` return new
          ``int32`` index tensors.
        - Used: ``raw``, ``raw_hi``, ``rope``, and ``rope_hi`` bank metadata.
        - Changed: ``write`` mutates ``raw``/``raw_hi`` (and ``rope``/``rope_hi``
          for ``uint8`` FP8 caches) in-place via donated ``copy_`` updates.

    Order-coupling lifecycle:
        While ``torch.compiler.is_compiling()`` is true, ``__getitem__`` records a
        1-word slice of the raw ``dsv41_gather_cache_rows`` output in
        ``_last_read``. Every ``write`` call clears ``_last_read`` back to ``None``,
        forwarding it into ``dsv41_write_sub_rows`` only when ``after_read`` is
        ``True`` so XLA orders prior gathers before donated writes without leaking
        stale tensors across calls.
    """

    def __init__(
        self,
        raw: utils.RawCachePool,
        layout: tuple[tuple[int, ...], torch.dtype, int, int],
        rope: utils.RawCachePool | None = None,
        *,
        null_block: bool = False,
    ) -> None:
        self._last_read: Tensor | None = None
        self.null_block: bool = null_block
        if isinstance(raw, tuple):
            self.raw: Tensor = raw[0]
            self.raw_hi: Tensor | None = raw[1]
        else:
            self.raw = raw
            self.raw_hi = None
        if isinstance(rope, tuple):
            self.rope: Tensor | None = rope[0]
            self.rope_hi: Tensor | None = rope[1]
        else:
            self.rope = rope
            self.rope_hi = None

        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        row_bytes = math.prod(self.shape[2:]) * self.dtype.itemsize
        self.row_bytes: int = row_bytes

        banks = (self.raw,) if self.raw_hi is None else (self.raw, self.raw_hi)
        invalid_bank = any(
            bank.dtype != torch.int32 or bank.shape[1:] != (ROW_WORDS,)
            for bank in banks
        )
        if invalid_bank or self.dtype not in (
            torch.bfloat16,
            torch.float32,
            torch.uint8,
        ):
            raise ValueError("Invalid native cache bank or record dtype")

        expected_rope_shape = (
            self.raw.shape[0] // ROPE_COMPANION_DIVISOR,
            ROW_WORDS,
        )
        rope_ok = (
            self.raw.shape[0] % ROPE_COMPANION_DIVISOR == 0
            and self.rope is not None
            and self.rope.shape == expected_rope_shape
        )
        high_ok = (self.rope_hi is not None) == (self.raw_hi is not None)
        if self.dtype == torch.uint8 and (
            row_bytes != ROW_BYTES or not (rope_ok and high_ok)
        ):
            raise ValueError("FP8 records are one pool row plus a RoPE companion")

        unaligned_record = row_bytes % WORD_BYTES != 0 or (
            row_bytes % ROW_BYTES != 0 and ROW_BYTES % row_bytes != 0
        )
        if (
            unaligned_record
            or stride_bytes % row_bytes != 0
            or offset_bytes % row_bytes != 0
            or self.raw.nbytes % row_bytes != 0
        ):
            raise ValueError("Cache pages and records must be word/row aligned")

        self.bank_records: int = self.raw.nbytes // row_bytes
        self.page_stride: int = stride_bytes // row_bytes
        self.row_offset: int = offset_bytes // row_bytes

    def numel(self) -> int:
        """Return logical element count across all pages in ``shape``."""
        return math.prod(self.shape)

    @property
    def device(self) -> torch.device:
        """Return the PyTorch device of the primary raw bank."""
        return self.raw.device

    def row_indices(self, slots: Tensor) -> Tensor:
        """Translate logical flat ``slots`` into physical bank record indices.

        Args:
            slots: ``int32`` or ``int64`` tensor of logical flat slot indices.

        Returns:
            ``int32`` tensor of physical bank record indices (Created).
        """
        slots_i32 = slots.int()
        rows_per_page = self.shape[1]
        if self.page_stride == rows_per_page:
            base_records = slots_i32
        else:
            page_gap = self.page_stride - rows_per_page
            base_records = slots_i32 + (slots_i32 // rows_per_page) * page_gap
        if self.row_offset == 0:
            return base_records
        return self.row_offset + base_records

    def pages(self, block_table: Tensor) -> Tensor | None:
        """Map logical ``block_table`` IDs to physical page indices when aligned.

        Args:
            block_table: ``int32`` or ``int64`` block table ``[num_reqs, max_blocks]``.

        Returns:
            ``int32[num_reqs, max_blocks]`` physical page indices (Created), or
            ``None`` if the layer's page layout is not block-aligned.
        """
        rows_per_page = self.shape[1]
        high_bank = self.raw_hi
        misaligned_layout = (
            rows_per_page % PAGE_ROW_ALIGNMENT != 0
            or self.row_offset % rows_per_page != 0
            or self.page_stride % rows_per_page != 0
        )
        misaligned_banks = self.raw.shape[0] % rows_per_page != 0 or (
            high_bank is not None and high_bank.shape[0] % rows_per_page != 0
        )
        if misaligned_layout or misaligned_banks:
            return None
        min_block = 1 if self.null_block else 0
        page_scale = self.page_stride // rows_per_page
        page_offset = self.row_offset // rows_per_page
        mapped_pages = block_table.int() * page_scale + page_offset
        return torch.where(block_table < min_block, -1, mapped_pages)

    def __getitem__(self, indices: tuple[Tensor, Tensor]) -> Tensor:
        """Gather ``float32`` or ``bfloat16`` records at ``(block_ids, row_ids)``.

        Args:
            indices: Tuple ``(block_ids, row_ids)`` of ``int32``/``int64`` index
                tensors (Used).

        Returns:
            Unpacked ``float32`` or ``bfloat16`` tensor of shape
            ``(*block_ids.shape, *self.shape[2:])`` (Created).
        """
        block_ids = indices[0].int()
        row_ids = indices[1].int()
        min_block = 1 if self.null_block else 0
        invalid = (block_ids < min_block) | (row_ids < 0)
        flat_slots = block_ids.clamp(min=0) * self.shape[1] + row_ids.clamp(min=0)
        records = self.row_indices(flat_slots).masked_fill(invalid, -1)

        flat_records = records.reshape(-1)
        record_words = self.row_bytes // WORD_BYTES
        if record_words >= ROW_WORDS:
            query_rows = _expand_chunks(flat_records, record_words // ROW_WORDS)
            raw_words = cache_op.dsv41_gather_cache_rows(
                self.raw, query_rows, self.raw_hi
            )
            words = raw_words.reshape(-1, record_words)
        else:
            records_per_row = ROW_WORDS // record_words
            query_rows = torch.where(
                flat_records >= 0, flat_records // records_per_row, -1
            )
            raw_words = cache_op.dsv41_gather_cache_rows(
                self.raw, query_rows, self.raw_hi
            )
            batch_iota = torch.arange(flat_records.numel(), device=flat_records.device)
            sub_col = flat_records.clamp(min=0) % records_per_row
            words = raw_words.reshape(-1, records_per_row, record_words)[
                batch_iota, sub_col
            ]

        if torch.compiler.is_compiling():
            self._last_read = raw_words.reshape(-1)[:1]
        if self.dtype == torch.float32:
            values = words.view(torch.float32)
        else:
            even_lanes = (
                (words << BF16_SHIFT_BITS).view(torch.float32).to(torch.bfloat16)
            )
            odd_lanes = (words & BF16_ODD_MASK).view(torch.float32).to(torch.bfloat16)
            values = torch.stack((even_lanes, odd_lanes), dim=-1).flatten(-2)
        return values.reshape(*records.shape, *self.shape[2:])

    def _write_words(
        self,
        low_bank: Tensor,
        high_bank: Tensor | None,
        records: Tensor,
        words: Tensor,
        dependency: Tensor | None = None,
    ) -> None:
        """Write packed ``int32`` ``words`` into ``low_bank`` and ``high_bank``.

        Args:
            low_bank: Primary ``int32[rows, 128]`` HBM bank tensor (Changed).
            high_bank: Optional overflow ``int32[rows, 128]`` HBM bank tensor (Changed).
            records: ``int32[num_tokens]`` physical record indices (Used).
            words: ``int32[num_tokens, record_words]`` packed word payload (Used).
            dependency: Optional prior gather dependency token for XLA ordering (Used).
        """
        high_offset = self.bank_records
        page_tokens = self.shape[1]
        if words.shape[1] > ROW_WORDS:
            chunks_per_record = words.shape[1] // ROW_WORDS
            records = _expand_chunks(records, chunks_per_record)
            words = words.reshape(-1, ROW_WORDS)
            high_offset = low_bank.shape[0]
            page_tokens = self.shape[1] * chunks_per_record
        low_bank.copy_(
            cache_op.dsv41_write_sub_rows(
                low_bank,
                words,
                records,
                0,
                after=dependency,
                page_tokens=page_tokens,
            )
        )
        if high_bank is not None:
            high_bank.copy_(
                cache_op.dsv41_write_sub_rows(
                    high_bank,
                    words,
                    records,
                    high_offset,
                    after=dependency,
                    page_tokens=page_tokens,
                )
            )

    def write(
        self,
        slots: Tensor,
        values: Tensor,
        *,
        records: Tensor | None = None,
        valid: Tensor | None = None,
        after_read: bool = False,
    ) -> None:
        """Pack ``values`` and write active ``slots`` into the backing banks.

        Args:
            slots: ``int32``/``int64`` flat logical slot indices ``[num_tokens]``.
            values: ``[num_tokens, ...]`` payload tensor to pack and write.
            records: Optional pre-computed physical record indices ``[num_tokens]``.
            valid: Optional boolean mask ``[num_tokens]`` selecting active writes.
            after_read: When ``True``, forwards ``_last_read`` to order prior reads
                before donated writes under ``torch.compile``.
        """
        last_read = self._last_read
        self._last_read = None
        dependency = (
            last_read if (after_read and torch.compiler.is_compiling()) else None
        )
        if records is None:
            min_slot = self.shape[1] if self.null_block else 0
            invalid = slots < min_slot if valid is None else (slots < min_slot) | ~valid
            records = self.row_indices(slots.clamp(min=0)).masked_fill(invalid, -1)
        record_ids = records.int()
        if self.dtype == torch.uint8:
            assert self.rope is not None
            kv_words, rope_words = cache_op.dsv41_pack_fp8_kv_rows(
                values.to(torch.bfloat16)
            )
            self._write_words(self.raw, self.raw_hi, record_ids, kv_words, dependency)
            self._write_words(
                self.rope, self.rope_hi, record_ids, rope_words, dependency
            )
            return
        flat_values = values.to(self.dtype).reshape(values.shape[0], -1)
        if self.dtype == torch.float32:
            words = flat_values.view(torch.int32)
        else:
            even_bits = (
                flat_values[:, 0::2].float().view(torch.int32) >> BF16_SHIFT_BITS
            ) & BF16_EVEN_MASK
            odd_bits = flat_values[:, 1::2].float().view(torch.int32) & BF16_ODD_MASK
            words = even_bits | odd_bits
        self._write_words(self.raw, self.raw_hi, record_ids, words, dependency)


class NativeCacheView(nn.Module, utils.NativeKVCacheLayer):
    """Mutable cache-binding mixin exposing ``PagedCacheRows`` on TPU.

    Each layer instance updates its mutable binding state (``_bound_cache``,
    ``_tpu_raw_cache``, ``_tpu_rope_pool``, ``_tpu_cache_layout``, and
    ``_tpu_paged_cache_rows``) in :meth:`bind_kv_cache`, while the first
    registered native layer (``native_layers[0]``) acts as the shared-pool
    owner that materializes pools and populates ``_tpu_zero_groups``.

    Attributes:
        null_block: True when raw pool block ``0`` is reserved as vLLM's padding
            sink and must never be read or written by active requests.
    """

    null_block: bool = False
    _bound_cache: Any = None
    _tpu_raw_cache: Any = None
    _tpu_rope_pool: Any = None
    _tpu_cache_layout: Any = None
    _tpu_paged_cache_rows: PagedCacheRows | None = None
    _tpu_zero_groups: tuple[Any, ...] = ()

    @property
    def paged_cache_rows(self) -> PagedCacheRows | None:
        """Return the bound ``PagedCacheRows`` wrapper when present."""
        return self._tpu_paged_cache_rows

    @property
    def kv_cache(self) -> Any:
        """Return ``PagedCacheRows`` during Dynamo trace, else the dense view."""
        if torch.compiler.is_compiling() and self._tpu_paged_cache_rows is not None:
            return self._tpu_paged_cache_rows
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache: Any) -> None:
        self._bound_cache = cache

    def bind_kv_cache(
        self,
        cache: Any,
        *,
        raw_cache: Any = None,
        layout: Any = None,
        rope_pool: Any = None,
    ) -> None:
        """Bind dense ``cache`` view and optional raw TPU ``int32`` word banks."""
        super().bind_kv_cache(cache)
        self._tpu_raw_cache = raw_cache
        self._tpu_rope_pool = rope_pool
        bound_view = self._bound_cache
        elem_size = bound_view.element_size()
        # NOTE: Fallback for standalone unit tests without full runner materialization.
        default_layout = (
            tuple(bound_view.shape),
            bound_view.dtype,
            bound_view.stride(0) * elem_size,
            bound_view.storage_offset() * elem_size,
        )
        resolved_layout = layout if layout is not None else default_layout
        self._tpu_cache_layout = resolved_layout
        self.null_block = raw_cache is not None
        if raw_cache is None:
            self._tpu_paged_cache_rows = None
        else:
            self._tpu_paged_cache_rows = PagedCacheRows(
                raw_cache,
                resolved_layout,
                rope_pool,
                null_block=True,
            )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Delegate KV cache specification lookup to the concrete cache layer."""
        return super().get_kv_cache_spec(vllm_config)

    def materialize_kv_caches(
        self,
        kv_cache_config: KVCacheConfig,
        device: torch.device,
        layout: Any,
    ) -> utils.NativeKVCacheMaterialization:
        """Allocate SparseCore-safe ``int32[rows, 128]`` KV and RoPE companion pools."""
        views = allocate_upstream_kv_cache(
            kv_cache_config, torch.device("meta"), layout
        )
        if not views:
            self._tpu_zero_groups = ()
            return utils.NativeKVCacheMaterialization({}, [], {}, {}, {})

        num_blocks = kv_cache_config.num_blocks
        backing_bytes = kv_cache_config.kv_cache_tensors[0].size
        if layout.is_layer_compact:
            pool_index_by_layer = layer_to_pool_index(kv_cache_config)
            max_page_bytes = max(
                group.kv_cache_spec.page_size_bytes
                for group in kv_cache_config.kv_cache_groups
            )
            pool_bytes = num_blocks * max_page_bytes
        else:
            pool_index_by_layer = dict.fromkeys(views, 0)
            pool_bytes = backing_bytes

        stride_bytes = pool_bytes // num_blocks
        num_pools = backing_bytes // pool_bytes
        low_rows = pool_bytes // ROW_BYTES
        high_rows = 0
        if pool_bytes > SPARSECORE_MAX_POOL_BYTES:
            low_blocks = SPARSECORE_MAX_POOL_BYTES // stride_bytes
            rows_per_block = stride_bytes // ROW_BYTES
            low_rows = low_blocks * rows_per_block
            high_rows = (num_blocks - low_blocks) * rows_per_block

        owners, raw_tensors = _alloc_word_pools(low_rows, high_rows, num_pools, device)
        caches: dict[str, Tensor] = {}
        native_raw_cache: dict[str, utils.RawCachePool] = {}
        layouts: dict[str, tuple[tuple[int, ...], torch.dtype, int, int]] = {}
        for name, view in views.items():
            pool_idx = pool_index_by_layer[name]
            item_bytes = view.element_size()
            view_stride = view.stride(0) * item_bytes
            view_offset = (
                int(view.storage_offset()) * item_bytes - pool_idx * pool_bytes
            )
            owner = owners[pool_idx]
            is_two_bank = isinstance(owner, tuple)
            native_raw_cache[name] = owner
            layouts[name] = (
                (view.shape[0], *view.shape[2:]),
                view.dtype,
                view_stride,
                view_offset,
            )
            primary_bank = owner[0] if is_two_bank else owner
            raw_bytes = primary_bank.view(torch.int8).flatten()
            view_shape = (1, *view.shape[1:]) if is_two_bank else tuple(view.shape)
            page_offset = view_offset % view_stride
            payload_bytes = math.prod(view_shape[1:]) * item_bytes
            base_offset = 0 if is_two_bank else view_offset - page_offset
            pages = raw_bytes[
                base_offset : base_offset + view_shape[0] * view_stride
            ].reshape(view_shape[0], view_stride)
            compact_view = (
                pages[:, page_offset : page_offset + payload_bytes]
                .view(view.dtype)
                .reshape(view_shape)
            )
            caches[name] = (
                compact_view.expand(tuple(view.shape)) if is_two_bank else compact_view
            )

        fp8_layers = [name for name, view in views.items() if view.dtype == torch.uint8]
        rope_pools: dict[str, utils.RawCachePool] = {}
        if fp8_layers:
            rope_owners, rope_tensors = _alloc_word_pools(
                low_rows // ROPE_COMPANION_DIVISOR,
                high_rows // ROPE_COMPANION_DIVISOR,
                num_pools,
                device,
            )
            rope_pools = {
                name: rope_owners[pool_index_by_layer[name]] for name in fp8_layers
            }
            raw_tensors = raw_tensors + rope_tensors
            self._tpu_zero_groups = ((owners, num_blocks), (rope_owners, num_blocks))
        else:
            self._tpu_zero_groups = ((owners, num_blocks),)
        return utils.NativeKVCacheMaterialization(
            caches, raw_tensors, native_raw_cache, layouts, rope_pools
        )

    def zero_blocks(self, block_ids: Any) -> None:
        """Zero newly allocated ``block_ids`` (> 0) across all raw and RoPE banks.

        Args:
            block_ids: Iterable of logical block indices to clear (Used).
        """
        groups: Any = self._tpu_zero_groups
        # NOTE: Fallback for standalone unit tests without full runner materialization.
        if not groups and self._tpu_raw_cache is not None and self._tpu_cache_layout:
            num_blocks_bound = int(self._tpu_cache_layout[0][0])
            rope_pool = self._tpu_rope_pool
            raw_group = ([self._tpu_raw_cache], num_blocks_bound)
            if rope_pool is not None:
                groups = (raw_group, ([rope_pool], num_blocks_bound))
            else:
                groups = (raw_group,)
        if not groups:
            bound = self._bound_cache
            if isinstance(bound, Tensor) and bound.numel():
                valid_blocks = {
                    int(block_id)
                    for block_id in block_ids
                    if 0 < int(block_id) < bound.shape[0]
                }
                for block_id in valid_blocks:
                    bound[block_id].zero_()
            return

        for pools, num_blocks in groups:
            sorted_blocks = sorted(
                {
                    int(block_id)
                    for block_id in block_ids
                    if 0 < int(block_id) < num_blocks
                }
            )
            if not sorted_blocks or not pools:
                continue
            low_banks = [pool[0] if isinstance(pool, tuple) else pool for pool in pools]
            high_banks = [
                pool[1] if isinstance(pool, tuple) else None for pool in pools
            ]
            high_offset = low_banks[0].shape[0]
            high_rows = 0 if high_banks[0] is None else high_banks[0].shape[0]
            rows_per_block = (high_offset + high_rows) // num_blocks
            if low_banks[0].device.type != "tpu":
                for block_id in sorted_blocks:
                    start_row = block_id * rows_per_block
                    local_offset = (
                        start_row
                        if start_row < high_offset
                        else start_row - high_offset
                    )
                    for low_bank, high_bank in zip(low_banks, high_banks):
                        target_bank = low_bank if start_row < high_offset else high_bank
                        assert target_bank is not None
                        target_bank[
                            local_offset : local_offset + rows_per_block
                        ].zero_()
                continue

            rows = [
                block_id * rows_per_block + row_in_block
                for block_id in sorted_blocks
                for row_in_block in range(rows_per_block)
            ]
            bucket_rows = (
                ZERO_BLOCKS_SMALL_BUCKET
                if len(rows) <= ZERO_BLOCKS_SMALL_BUCKET
                else ZERO_BLOCKS_LARGE_BUCKET
            )
            device = low_banks[0].device
            zero_rows = torch.zeros(
                (bucket_rows, ROW_WORDS), dtype=torch.int32, device=device
            )
            for start in range(0, len(rows), bucket_rows):
                chunk = rows[start : start + bucket_rows]
                padded = chunk + [-1] * (bucket_rows - len(chunk))
                records = torch.tensor(padded, dtype=torch.int32, device=device)
                _zero_pools_tpu(
                    low_banks,
                    high_banks,
                    records,
                    zero_rows,
                    high_offset,
                    rows_per_block,
                )
            active_banks = [
                bank for bank in (*low_banks, *high_banks) if bank is not None
            ]
            utils.synchronize_tensors(active_banks)

    def reserve_companion_memory(
        self, vllm_config: VllmConfig, available: int, *, auto: bool = False
    ) -> int:
        """Reserve 20% companion RoPE pool budget and cap at two SparseCore banks.

        Args:
            vllm_config: Active vLLM runtime configuration (Used).
            available: Available KV cache HBM byte budget before reservation.
            auto: When ``True``, quantizes ``available`` down to a 6-bit mantissa.

        Returns:
            Adjusted byte budget for primary KV cache allocation.
        """
        architectures = vllm_config.model_config.hf_config.architectures or ()
        cache_dtype = str(vllm_config.cache_config.cache_dtype)
        if "DeepseekV41ForCausalLM" not in architectures or not cache_dtype.startswith(
            "fp8"
        ):
            return available
        # NOTE: Round auto memory down to a 6-bit mantissa quantum and reserve 20%
        # for the 1:4 RoPE companion pool, capped at 2 SparseCore banks.
        if auto and available > 0:
            available &= -(
                1 << max(0, available.bit_length() - AUTO_MEMORY_MANTISSA_BITS)
            )
        budget = available * COMPANION_BUDGET_NUMERATOR // COMPANION_BUDGET_DENOMINATOR
        single_bank_threshold = (
            SPARSECORE_MAX_POOL_BYTES
            * COMPANION_BUDGET_DENOMINATOR
            // COMPANION_BUDGET_NUMERATOR
        )
        bank_limit = 1 if budget < single_bank_threshold else SPARSECORE_MAX_POOL_BANKS
        return min(budget, bank_limit * SPARSECORE_MAX_POOL_BYTES)


def write_cache(
    cache: PagedCacheRows | Tensor,
    slots: Tensor,
    values: Tensor,
    valid: Tensor,
    after_read: bool = False,
    *,
    records: Tensor | None = None,
) -> None:
    """Write ``values`` into ``cache`` at ``slots`` where ``valid`` is True.

    Args:
        cache: Target ``PagedCacheRows`` or dense PyTorch cache tensor (Changed).
        slots: ``int32``/``int64`` flat logical slot indices ``[num_tokens]`` (Used).
        values: ``[num_tokens, ...]`` payload tensor to write (Used).
        valid: Boolean mask ``[num_tokens]`` selecting active token writes (Used).
        after_read: When ``True``, orders prior gathers before donated TPU writes.
        records: Optional pre-computed physical record indices ``[num_tokens]``.
    """
    num_rows = values.shape[0]
    valid_mask = valid[:num_rows]
    if isinstance(cache, PagedCacheRows):
        masked_records = (
            None if records is None else records[:num_rows].masked_fill(~valid_mask, -1)
        )
        cache.write(
            slots[:num_rows],
            values,
            records=masked_records,
            valid=valid_mask,
            after_read=after_read,
        )
        return
    valid_slots = slots[:num_rows].masked_fill(~valid_mask, -1)[valid_mask].long()
    if cache.ndim == values.ndim:
        cache[(valid_slots,)] = values[valid_mask]
    else:
        rows_per_page = cache.shape[1]
        cache[valid_slots // rows_per_page, valid_slots % rows_per_page] = values[
            valid_mask
        ]
