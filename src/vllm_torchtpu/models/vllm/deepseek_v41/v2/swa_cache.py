# SPDX-License-Identifier: Apache-2.0
"""CSA2 SWA pages: one spec, one manager, one window-sized physical pool.

DeepSeek-V4.1 rebuilds sliding-window KV after a prefix hit by bounded
replay, so SWA pages never outlive the request window that owns them, with
or without prefix caching. The manager therefore issues block IDs from a
private pool sized for live windows, and those IDs are the physical page
indices of the BF16 owner `cache._materialize_native_swa` allocates.
"""

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
class DeepseekV41SWASpec(SlidingWindowMLASpec):
    """Native BF16 SWA pages owned by `DeepseekV41SWAManager`'s pool."""

    max_num_seqs: int
    max_in_flight_tokens: int
    num_lookahead_tokens: int = 0

    @property
    def prefix_cacheable(self) -> bool:
        # Bounded replay rebuilds the window after a hit; with prefix caching
        # off there are no hits. Either way no page is shared or retained.
        return False

    @property
    def pool_capacity(self) -> int:
        """Pages for the worst case; the pool never limits admission.

        `allocate_slots` frees below the processed basis and allocates through
        the lookahead slots, so a request holds at most
        cdiv(W - 1 + extra + lookahead + x, bs) + 1 pages, where x is its
        in-flight plus newly scheduled tokens. x sums to at most
        `max_in_flight_tokens` over all requests; cdiv(a + x) <= cdiv(a) +
        cdiv(x) gives one more page per request. Block lists never shrink, so
        after a rejection a request still holds pages allocated for its
        rejected drafts and their lookahead; the second lookahead term covers
        that stale tail. Page 0 is the null block.
        """
        retained = (
            self.sliding_window
            - 1
            + self.extra_retained_tokens
            + 2 * self.num_lookahead_tokens
        )
        return (
            1
            + self.max_num_seqs * (cdiv(retained, self.block_size) + 2)
            + cdiv(self.max_in_flight_tokens, self.block_size)
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, DeepseekV41SWASpec) and spec == self
            for spec in kv_cache_specs.values()
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = SlidingWindowMLASpec.merge(specs)
        first = specs[0]
        assert all(
            (spec.max_num_seqs, spec.max_in_flight_tokens, spec.num_lookahead_tokens)
            == (
                first.max_num_seqs,
                first.max_in_flight_tokens,
                first.num_lookahead_tokens,
            )
            for spec in specs
        )
        return replace(
            first,
            **{field.name: getattr(merged, field.name) for field in fields(merged)},
        )


class DeepseekV41SWAManager(SlidingWindowManager):
    """Upstream sliding-window allocation and reclamation over a private pool.

    Demand is charged to the private pool only. `pool_capacity` bounds the
    peak, so a shortfall is a capacity bug: fail loudly rather than returning
    an admission failure that the scheduler would turn into preemption.
    """

    def __init__(self, kv_cache_spec: DeepseekV41SWASpec, **kwargs: Any) -> None:
        shared_pool = kwargs["block_pool"]
        kwargs["block_pool"] = BlockPool(
            num_gpu_blocks=kv_cache_spec.pool_capacity,
            enable_caching=False,
            hash_block_size=shared_pool.hash_block_size,
        )
        super().__init__(kv_cache_spec, **kwargs)
        # SWA writes each visible token before reading it. Private IDs must
        # never enter the shared pool's block-zeroing or copy messages.
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
        if apply_admission_cap:
            # Whole-sequence admission estimates; the pool is sized for peak.
            return 0
        required = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            prefill_end=prefill_end,
        )
        if required > self.block_pool.get_num_free_blocks():
            raise RuntimeError(
                f"DeepSeek-V4.1 SWA pool exhausted: {request_id} needs "
                f"{required} pages, {self.block_pool.get_num_free_blocks()} free "
                f"of {self.block_pool.num_gpu_blocks}"
            )
        return 0


def register_swa_specs() -> None:
    """Called by the platform's upstream KV-cache registration hook."""
    KVCacheSpecRegistry.register(
        DeepseekV41SWASpec,
        DeepseekV41SWAManager,
        uniform_type_base_spec=DeepseekV41SWASpec,
    )


def num_physical_blocks(spec: KVCacheSpec, num_blocks: int) -> int:
    """Pages a group's owner holds: the private pool, or the shared block count."""
    if isinstance(spec, DeepseekV41SWASpec):
        return spec.pool_capacity
    return num_blocks


def swa_pool_bytes(specs: dict[str, KVCacheSpec]) -> int:
    """Physical BF16 bytes deducted before sizing the shared word pool."""
    return sum(
        spec.pool_capacity * spec.unpadded_page_size_bytes
        for spec in specs.values()
        if isinstance(spec, DeepseekV41SWASpec)
    )
