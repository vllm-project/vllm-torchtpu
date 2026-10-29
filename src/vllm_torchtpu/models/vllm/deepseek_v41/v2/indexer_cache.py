# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer cache specifications, native owner binding and FP8 row access."""

import dataclasses

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.models.deepseek_v41 import attention as upstream
from vllm.v1.kv_cache_interface import MLAAttentionSpec, replace_as

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
)

from . import indexer_impl as indexer_ops
from .cache import IndexCacheRows, NativeCacheView, PagedCacheRows, RawCachePool

logger = init_logger(__name__)


def indexer_uses_fp8(config: VllmConfig) -> bool:
    dtype = config.attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype not in ("bf16", "fp8"):
        raise ValueError(f"TPU V4.1 indexer requires bf16 or fp8, got {dtype}")
    return dtype == "fp8"


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
        assert indexer_ops._read_op is not None
        values = indexer_ops._read_op(
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
        assert indexer_ops._write_op is not None
        updated = indexer_ops._write_op(
            self.banks, slots.int(), values, after, self._layout_args()
        )
        for bank, value in zip(self.banks, updated):
            bank.copy_(value)


class PackedFP8IndexerSpec(MLAAttentionSpec):
    """Keep 132-byte records compatible with hybrid power-of-two pages."""

    def __post_init__(self):
        record_bytes = 1 << (self.state_content_size_bytes - 1).bit_length()
        min_states = max(1, (self.alignment or 1) // record_bytes)
        object.__setattr__(
            self, "block_size", max(self.block_size, min_states * self.tokens_per_state)
        )
        super().__post_init__()
        size = max(self.alignment or 1, self.real_page_size_bytes)
        object.__setattr__(self, "page_size_padded", 1 << (size - 1).bit_length())

    @classmethod
    def merge(cls, specs):
        return dataclasses.replace(super().merge(specs), alignment=specs[0].alignment)


class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    packed_fp8 = False

    def __init__(self, *args, **kwargs):
        config = get_current_vllm_config()
        if getattr(config, "kv_transfer_config", None) is not None:
            raise ValueError("Halfplane Index-K does not support external KV transfer")
        if getattr(config.cache_config, "kv_offloading_size", 0):
            raise ValueError("Halfplane Index-K does not support external KV offload")
        self.packed_fp8 = indexer_uses_fp8(config)
        head_dim = config.model_config.hf_config.index_head_dim
        if self.packed_fp8 and head_dim != 128:
            raise ValueError("Packed FP8 indexer requires index_head_dim=128")
        kwargs["head_dim"] = head_dim + 4 if self.packed_fp8 else head_dim
        kwargs["dtype"] = torch.uint8 if self.packed_fp8 else torch.bfloat16
        super().__init__(*args, **kwargs)
        if self.packed_fp8:
            indexer_ops.initialize_fp8_indexer()
            logger.info_once(
                "DSV4.1 indexer backend=packed_fp8: transposed FP8 cache, "
                "BF16 weighted products and rounded head sums; model quality "
                "and end-to-end performance remain to be validated."
            )

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return replace_as(spec, PackedFP8IndexerSpec) if self.packed_fp8 else spec

    @property
    def kv_cache(self):
        if self.packed_fp8:
            return PackedFP8IndexCacheRows(self._tpu_raw_cache, self._tpu_cache_layout)
        return IndexCacheRows(self._tpu_raw_cache, self._tpu_cache_layout)

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_kv_cache(
        self, cache, *, raw_cache=None, layout=None, rope_pool=None, scale_pool=None
    ):
        if raw_cache is None or layout is None:
            raise ValueError("Index-K requires an explicit native owner/layout")
        if rope_pool is not None or scale_pool is not None:
            raise ValueError("Index-K owners cannot have RoPE or scale companions")
        cache_type = PackedFP8IndexCacheRows if self.packed_fp8 else IndexCacheRows
        bound = cache_type(raw_cache, layout)
        if isinstance(bound, PackedFP8IndexCacheRows):
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
