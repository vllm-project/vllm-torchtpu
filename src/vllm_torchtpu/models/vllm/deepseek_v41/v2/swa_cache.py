# SPDX-License-Identifier: Apache-2.0
"""Native SWA storage with vLLM-owned page lifetimes."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Self

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class NativeSWASpec(SlidingWindowMLASpec):
    """BF16 page owners; ordinary prefix retention keeps scheduler addresses."""


@dataclass(frozen=True, kw_only=True)
class CompactSWASpec(NativeSWASpec):
    """An independent pool whose IDs are issued by SlidingWindowManager."""

    max_num_seqs: int
    max_in_flight_tokens: int

    @property
    def pool_capacity(self) -> int:
        retained = self.sliding_window - 1 + self.extra_retained_tokens
        # One page for the null block. Two partial pages per request cover
        # window alignment and splitting the shared in-flight token budget.
        return (
            1
            + self.max_num_seqs * (cdiv(retained, self.block_size) + 2)
            + cdiv(self.max_in_flight_tokens, self.block_size)
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, CompactSWASpec) and spec == self
            for spec in kv_cache_specs.values()
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = SlidingWindowMLASpec.merge(specs)
        first = specs[0]
        assert all(
            spec.max_num_seqs == first.max_num_seqs
            and spec.max_in_flight_tokens == first.max_in_flight_tokens
            for spec in specs
        )
        return replace(
            first,
            **{field.name: getattr(merged, field.name) for field in fields(merged)},
        )


class CompactSWAManager(SlidingWindowManager):
    """Reuse vLLM allocation/reclamation; check a private SWA pool's capacity.

    Like HiSparseSourceManager, private-pool demand contributes zero to the
    shared pool, or an admission failure when the private pool is exhausted.
    Each cache group owns its pool, so checks cannot overcommit another group.
    """

    def __init__(self, kv_cache_spec: CompactSWASpec, **kwargs: Any) -> None:
        if kwargs["enable_caching"] and kv_cache_spec.prefix_cacheable:
            raise ValueError("Compact SWA requires bounded prefix replay")
        self.shared_pool = kwargs["block_pool"]
        kwargs["block_pool"] = BlockPool(
            num_gpu_blocks=kv_cache_spec.pool_capacity,
            enable_caching=False,
            hash_block_size=self.shared_pool.hash_block_size,
        )
        super().__init__(kv_cache_spec, **kwargs)
        # SWA writes each visible token before reading it. Its local IDs must
        # never enter the shared pool's unqualified block-zeroing messages.
        self._record_new_block_ids = False

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        required = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        if required > self.block_pool.get_num_free_blocks():
            return self.shared_pool.num_gpu_blocks + 1
        return 0


def register_native_swa_specs() -> None:
    """Called by the platform's upstream KV-cache registration hook."""
    KVCacheSpecRegistry.register(
        NativeSWASpec, SlidingWindowManager, uniform_type_base_spec=NativeSWASpec
    )
    KVCacheSpecRegistry.register(
        CompactSWASpec, CompactSWAManager, uniform_type_base_spec=CompactSWASpec
    )


def native_swa_reservation(specs: dict[str, KVCacheSpec]) -> int:
    """Physical BF16 bytes deducted before sizing the shared word pool."""
    return sum(
        spec.pool_capacity * spec.unpadded_page_size_bytes
        for spec in specs.values()
        if isinstance(spec, CompactSWASpec)
    )
