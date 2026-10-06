"""V1 binding of the shared DeepSeek V4.1 Engram."""

import weakref
from typing import override

import torch

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_cache import (
    write_cache,
)

from ..common import engram as common
from ..common.engram import Engram

__all__ = ["Engram", "NgramHashState"]


class NgramHashState(common.NgramHashState):
    @override
    def write_cache(self, cache, slots, values, valid):
        write_cache(cache, slots, values, valid)

    @override
    def ensure_cache(self) -> bool:
        if torch.compiler.is_compiling():
            return super().ensure_cache()
        # The kernel property derives a view of the canonical pool. Lifetime
        # tracking must use the stable allocator object, not that temporary view.
        kv_cache = self.swa_cache_module.bound_cache
        if not kv_cache.numel():
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        # Hybrid allocation can grow the physical SWA page beyond the
        # block size copied at model construction. Match the bound view
        # used by slot_mapping and block_table before upstream sizes its
        # generated-token history cache.
        block_size = self.swa_cache_module.block_size
        if self.block_size != block_size:
            self.block_size = block_size
            self._kv_cache_ref = None
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True
