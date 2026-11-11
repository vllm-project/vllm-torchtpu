# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer cache specifications, native owner binding and FP8 row access."""

import dataclasses

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.models.deepseek_v41 import attention as upstream

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
)

from . import indexer_fp8
from .cache import (
    NativeCacheView,
    PagedCacheRows,
    RawCachePool,
    block_stride_alignment,
)

logger = init_logger(__name__)

# Eight 512-byte owner rows: one int32 [8, 128] tile.
INDEX_PAGE_ALIGNMENT = 4096


def indexer_uses_fp4(config: VllmConfig) -> bool:
    """Upstream ``dsa_indexer_uses_fp4`` without its CUDA capability checks.

    TPU implements the FP8 indexer cache only.
    """
    dtype = config.attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype != "fp8":
        raise ValueError(f"TPU V4.1 indexer requires indexer_kv_dtype=fp8, got {dtype}")
    return False


class PackedFP8IndexCacheRows(PagedCacheRows):
    """Logical D128 keys backed by transposed FP8 pages in shared native owners."""

    def __init__(
        self,
        raw: RawCachePool,
        layout: tuple[tuple[int, ...], torch.dtype, int, int],
    ) -> None:
        physical_shape, dtype, stride_bytes, offset_bytes = layout
        if (
            dtype != torch.uint8
            or len(physical_shape) != 3
            or physical_shape[2] != 132
            or stride_bytes % 512
            or offset_bytes % 512
        ):
            raise ValueError(
                "Packed FP8 index cache requires 132 bytes per record "
                "and 512-byte aligned owners"
            )
        banks = raw if isinstance(raw, tuple) else (raw,)
        if len(banks) not in (1, 2) or any(
            bank.dtype != torch.int32 or bank.ndim != 2 or bank.shape[1] != 128
            for bank in banks
        ):
            raise ValueError("Packed FP8 index cache requires int32 [rows,128] owners")
        self.layout = PackedFP8Layout(
            physical_shape[0],
            stride_bytes // 512,
            offset_bytes // 512,
            physical_shape[1],
        )
        self.layout.validate(tuple(bank.shape[0] for bank in banks))
        self.raw, self.raw_hi = banks[0], banks[1] if len(banks) == 2 else None
        self.shape = (*physical_shape[:2], 128)
        self.dtype = torch.bfloat16
        # Addressing above the format boundary is in logical record slots.
        self.row_offset = 0
        self.page_stride = self.layout.page_size

    def row_indices(self, slots: torch.Tensor) -> torch.Tensor:
        return slots

    def __getitem__(self, indices: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        blocks, rows = indices
        records = (blocks * self.layout.page_size + rows).int()
        assert indexer_fp8._read_op is not None
        values = indexer_fp8._read_op(
            self.banks, records.flatten(), self._layout_args()
        )
        return values.reshape(*records.shape, 128)

    @property
    def banks(self) -> tuple[torch.Tensor, ...]:
        return (self.raw,) if self.raw_hi is None else (self.raw, self.raw_hi)

    def _layout_args(self) -> tuple[int, int, int, int]:
        return (
            self.layout.pages,
            self.layout.page_stride,
            self.layout.page_offset,
            self.layout.page_size,
        )

    def write(
        self,
        slots: torch.Tensor,
        values: torch.Tensor,
        after: torch.Tensor | None = None,
    ) -> None:
        after = values.new_empty((0,)) if after is None else after
        assert indexer_fp8._write_op is not None
        updated = indexer_fp8._write_op(
            self.banks, slots.int(), values, after, self._layout_args()
        )
        for bank, value in zip(self.banks, updated):
            bank.copy_(value)


class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    """Upstream's 132-byte FP8 index record in transposed native pages."""

    def __init__(self, *args, **kwargs):
        config = get_current_vllm_config()
        if getattr(config, "kv_transfer_config", None) is not None:
            raise ValueError("Packed FP8 Index-K does not support external KV transfer")
        if getattr(config.cache_config, "kv_offloading_size", 0):
            raise ValueError("Packed FP8 Index-K does not support external KV offload")
        indexer_uses_fp4(config)
        if config.model_config.hf_config.index_head_dim != 128:
            raise ValueError("Packed FP8 indexer requires index_head_dim=128")
        super().__init__(*args, **kwargs)
        indexer_fp8.initialize_fp8_indexer()

    def get_kv_cache_spec(self, vllm_config):
        # Upstream's spec with whole 8-row (4 KiB) pages: `PackedFP8Layout`
        # addresses pages in 512-byte owner rows, and every page then starts
        # on an int32 tile boundary, as the indexer kernels were tuned.
        return dataclasses.replace(
            super().get_kv_cache_spec(vllm_config),
            alignment=INDEX_PAGE_ALIGNMENT,
            block_stride_alignment=block_stride_alignment(
                vllm_config.cache_config.block_size
            ),
        )

    @property
    def kv_cache(self):
        return PackedFP8IndexCacheRows(self._tpu_raw_cache, self._tpu_cache_layout)

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None, scale_pool=None):
        if raw_cache is None or layout is None:
            raise ValueError("Index-K requires an explicit native owner/layout")
        if scale_pool is not None:
            raise ValueError("Index-K owners cannot have scale companions")
        bound = PackedFP8IndexCacheRows(raw_cache, layout)
        logger.info_once(
            "Packed FP8 Index-K: %d records/page, %d physical pages, "
            "%d-byte page stride",
            bound.layout.page_size,
            bound.layout.pages,
            bound.layout.page_stride * 512,
        )
        super().bind_kv_cache(cache, raw_cache=raw_cache, layout=layout)

    def get_attn_backend(self):
        from .attention import AttentionBackend

        return AttentionBackend
