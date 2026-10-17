# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-DP-group KV block regions for the K3 decoder under attention DP.

Under ``attention_dp`` (the decode role's loader config) the decode ranks form
DP groups of ``32 / dp`` ranks, and a request's MLA pages live only on its
group's ranks. Every rank holds a block id's row of its pools, so with one id
space the other groups' copies of a page would sit idle; instead the
scheduler's id space is larger than the pools (``N`` physical blocks):

* ids ``[0, S)``: shared blocks, the same row on every rank (the null block
  and the KDA state blocks, which every rank's importer reads);
* ids ``g N + [S, N)``: MLA pages of DP group ``g`` at physical row
  ``id % N`` on group ``g``'s ranks only.

New requests choose the group with the fewest assigned requests. A pending
request can leave a full group after 16 scheduling steps with capacity in
another group, before a prefix lookup. This lets short admission stalls drain
without changing the placement. Requests holding blocks keep their group.
Prefix hits only reuse pages of the chosen group. The KV transfer writes every
destination block on every rank, so while a request's pages load the
scheduler also holds the other groups' ids of the same rows ("twins").
"""

from __future__ import annotations

import functools
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SHARED = 0
_REASSIGN_AFTER_STEPS = 16

# The region pool's settings while a decode scheduler builds its KV cache
# manager (see ``_scheduler_init``).
_PENDING: dict[str, int] | None = None
_INSTALLED = False


def physical_block_ids(block_ids, num_physical: int) -> list[int]:
    """The pools' rows of scheduler block ids (``id % num_physical``)."""
    return [int(b) % num_physical for b in block_ids]


class RegionBlockPool(BlockPool):
    """A :class:`BlockPool` over ``groups * N`` ids: a shared region and one
    MLA region per DP group (see the module docstring)."""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: Any = None,
        *,
        groups: int,
        shared: int,
        cap: int,
    ) -> None:
        physical = int(num_gpu_blocks)
        if not 1 < shared < physical:
            raise ValueError(
                f"K3 DP blocks: {shared} shared blocks do not fit {physical} blocks"
            )
        super().__init__(
            groups * physical,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
        self.physical, self.shared, self.groups, self.cap = (
            physical,
            int(shared),
            int(groups),
            int(cap),
        )
        blocks = self.blocks
        self.queues = [FreeKVCacheBlockQueue(blocks[1:shared])] + [
            FreeKVCacheBlockQueue(blocks[g * physical + shared : (g + 1) * physical])
            for g in range(groups)
        ]
        self.free_block_queue = self.queues[SHARED]
        self.num_gpu_blocks = shared + groups * (physical - shared)
        # Requests' DP groups (placed at their first lookup) and those that
        # hold blocks; twin ids held while a request's pages load.
        self.req_group: dict[str, int] = {}
        self.active: dict[str, int] = {}
        self.twins: dict[str, list[KVCacheBlock]] = {}
        self.blocked_since: dict[str, int] = {}
        # The group of the request being looked up or allocated, and the KV
        # cache groups holding MLA pages.
        self.group = 0
        self.mla_group_ids: set[int] = set()

    # ------------------------------------------------------------ regions
    def region_of(self, block_id: int) -> int:
        if block_id < self.shared:
            return SHARED
        group, row = divmod(block_id, self.physical)
        if row < self.shared:
            raise ValueError(f"K3 DP blocks: id {block_id} is in no region")
        return 1 + group

    def free_in(self, region: int) -> int:
        return self.queues[region].num_free_blocks

    def _take(self, region: int, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > self.free_in(region):
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from region {region}"
            )
        taken = self.queues[region].popleft_n(num_blocks)
        self._claim(taken)
        return taken

    def _claim(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)

    def _twin(self, block: KVCacheBlock, group: int) -> KVCacheBlock:
        return self.blocks[group * self.physical + block.block_id % self.physical]

    def _paired(self, group: int, limit: int):
        """Free blocks of ``group``'s region whose rows are free in every
        other group too (at most ``limit``)."""
        queue = self.queues[1 + group]
        others = [g for g in range(self.groups) if g != group]
        block = queue.fake_free_list_head.next_free_block
        found = 0
        while found < limit and block is not queue.fake_free_list_tail:
            following = block.next_free_block
            if all(self._twin(block, g).ref_cnt == 0 for g in others):
                found += 1
                yield block
            block = following

    def count_pairs(self, group: int, limit: int) -> int:
        return sum(1 for _ in self._paired(group, limit))

    def take_pairs(
        self, group: int, num_blocks: int
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        """``num_blocks`` blocks of ``group``'s region and the other groups'
        blocks of the same rows (the twins), all allocated."""
        chosen = list(self._paired(group, num_blocks))
        if len(chosen) < num_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} paired blocks for DP group {group}"
            )
        twins = []
        for block in chosen:
            self.queues[1 + group].remove(block)
            for other in range(self.groups):
                if other != group:
                    twin = self._twin(block, other)
                    self.queues[1 + other].remove(twin)
                    twins.append(twin)
        self._claim(chosen)
        self._claim(twins)
        return chosen, twins

    # ------------------------------------------------------------ BlockPool
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._take(SHARED, num_blocks)

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        unwritten = self.__dict__.get("_tpu_unwritten_block_ids") or ()
        cache = self.cached_block_hash_to_block._cache
        found = []
        for group_id in kv_cache_group_ids:
            entry = cache.get(make_block_hash_with_group_id(block_hash, group_id))
            if entry is None:
                return None
            candidates = entry.values() if isinstance(entry, dict) else (entry,)
            region = 1 + self.group if group_id in self.mla_group_ids else SHARED
            block = next(
                (
                    b
                    for b in candidates
                    if self.region_of(b.block_id) == region
                    and b.block_id not in unwritten
                ),
                None,
            )
            if block is None:
                return None
            found.append(block)
        return found

    def touch(self, blocks) -> None:
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.queues[self.region_of(block.block_id)].remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks) -> None:
        first: dict[int, list[KVCacheBlock]] = {}
        last: dict[int, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                target = (
                    first
                    if block.block_hash is None or not self.enable_caching
                    else last
                )
                target.setdefault(self.region_of(block.block_id), []).append(block)
        for region, blocks in first.items():
            self.queues[region].prepend_n(blocks)
        for region, blocks in last.items():
            self.queues[region].append_n(blocks)

    def get_num_free_blocks(self) -> int:
        return sum(queue.num_free_blocks for queue in self.queues)

    def get_usage(self) -> float:
        usable = self.num_gpu_blocks - 1
        return 1.0 - self.get_num_free_blocks() / usable if usable else 0.0

    # ------------------------------------------------------------ requests
    def place(self, request_id: str, *, step: int | None = None) -> int:
        """Balance new placements; reconsider stalled ones before prefix lookup."""
        group = self.req_group.get(request_id)
        if group is None:
            counts = [0] * self.groups
            for g in self.req_group.values():
                counts[g] += 1
            group = min(range(self.groups), key=counts.__getitem__)
            self.req_group[request_id] = group
        if step is not None and request_id not in self.active:
            counts = [0] * self.groups
            for g in self.active.values():
                counts[g] += 1
            best = min(range(self.groups), key=counts.__getitem__)
            if counts[group] >= self.cap and counts[best] < self.cap:
                since = self.blocked_since.setdefault(request_id, step)
                if step - since >= _REASSIGN_AFTER_STEPS:
                    group = best
                    self.req_group[request_id] = group
                    self.blocked_since.pop(request_id)
            else:
                self.blocked_since.pop(request_id, None)
        self.group = group
        return group

    def release(self, request_id: str) -> list[KVCacheBlock]:
        """Forget ``request_id``'s group; returns its twins (not yet freed)."""
        self.req_group.pop(request_id, None)
        self.active.pop(request_id, None)
        self.blocked_since.pop(request_id, None)
        return self.twins.pop(request_id, [])


class _GroupView:
    """A manager's view of the pool that allocates from the current
    request's DP group region."""

    def __init__(self, pool: RegionBlockPool) -> None:
        self._pool = pool

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._pool._take(1 + self._pool.group, num_blocks)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _state_blocks_per_request(kv_cache_config: Any) -> int:
    """The most shared blocks a request holds at once: vLLM's ``align`` mode
    keeps at most ``2 + num_speculative_blocks + num_prefill_checkpoint_blocks``
    per KDA (Mamba) group (``MambaSpec.max_memory_usage_bytes``)."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    total = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, MambaSpec):
            total += (
                2 + spec.num_speculative_blocks + spec.num_prefill_checkpoint_blocks
            )
    return total


def _install(scheduler: Any) -> None:
    """Point the scheduler's MLA managers at their group regions, pair the
    pages a KV load writes, and check each region's capacity."""
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    coordinator = scheduler.kv_cache_manager.coordinator
    pool = coordinator.block_pool
    managers = coordinator.single_type_managers
    mla = [i for i, m in enumerate(managers) if isinstance(m, FullAttentionManager)]
    if len(mla) != 1:
        raise RuntimeError(f"K3 DP blocks: expected one MLA KV cache group, got {mla}")
    (mla_index,) = mla
    manager = managers[mla_index]
    pool.mla_group_ids = {manager.kv_cache_group_id}
    manager.block_pool = _GroupView(pool)

    def allocate_external_computed_blocks(
        request_id, num_local_computed_tokens, num_external_computed_tokens
    ):
        if num_external_computed_tokens <= 0:
            return
        req_blocks = manager.req_to_blocks[request_id]
        total = num_local_computed_tokens + num_external_computed_tokens
        num_new = max(0, _cdiv(total, manager.block_size) - len(req_blocks))
        blocks, twins = pool.take_pairs(pool.group, num_new)
        req_blocks.extend(blocks)
        pool.twins.setdefault(request_id, []).extend(twins)

    manager.allocate_external_computed_blocks = allocate_external_computed_blocks

    def get_num_blocks_to_allocate(
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        group = pool.place(request_id)
        need = [0] * len(pool.queues)
        for i, m in enumerate(managers):
            need[1 + group if i == mla_index else SHARED] += (
                m.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            )
        first = request_id not in pool.active
        pairs = 0
        if first and total_computed_tokens > num_local_computed_tokens:
            pairs = _cdiv(total_computed_tokens, manager.block_size) - (
                num_local_computed_tokens // manager.block_size
            )
        full = (
            (first and sum(g == group for g in pool.active.values()) >= pool.cap)
            or need[1 + group] > pool.free_in(1 + group)
            or need[SHARED] > pool.free_in(SHARED)
            or (pairs and pool.count_pairs(group, pairs) < pairs)
        )
        if full:
            return pool.get_num_free_blocks() + 1
        return sum(need) + pairs * (pool.groups - 1)

    coordinator.get_num_blocks_to_allocate = get_num_blocks_to_allocate
    logger.info(
        "K3 DP blocks: %d DP groups over %d physical blocks: %d shared, %d MLA "
        "per group (%d scheduler ids), at most %d requests per group",
        pool.groups,
        pool.physical,
        pool.shared,
        pool.physical - pool.shared,
        pool.num_gpu_blocks,
        pool.cap,
    )


def install() -> None:
    """Engine patches: a decode scheduler under attention DP gets a
    :class:`RegionBlockPool`."""
    import vllm.v1.core.kv_cache_coordinator as coordinator_module
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    base_pool = coordinator_module.BlockPool

    def block_pool(*args, **kwargs):
        if _PENDING is None:
            return base_pool(*args, **kwargs)
        return RegionBlockPool(*args, **kwargs, **_PENDING)

    coordinator_module.BlockPool = block_pool

    original_init = Scheduler.__init__

    @functools.wraps(original_init)
    def _scheduler_init(self, vllm_config, *args, **kwargs):
        global _PENDING
        from vllm_torchtpu.megakernel import backend

        dp = backend.attention_dp(vllm_config)
        if dp > 1:
            max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
            shared = backend.shared_blocks(vllm_config)
            if shared is None:
                kv_cache_config = kwargs.get("kv_cache_config") or args[0]
                shared = _state_blocks_per_request(kv_cache_config) * max_seqs + 1
            _PENDING = {
                "groups": dp,
                "shared": shared,
                "cap": max_seqs // dp,
            }
        try:
            original_init(self, vllm_config, *args, **kwargs)
        finally:
            pending, _PENDING = _PENDING, None
        if pending is not None:
            _install(self)

    Scheduler.__init__ = _scheduler_init

    original_lookup = Scheduler._get_local_prefix_cache_hit

    @functools.wraps(original_lookup)
    def _get_local_prefix_cache_hit(self, request):
        pool = self.kv_cache_manager.block_pool
        if isinstance(pool, RegionBlockPool):
            pool.place(request.request_id, step=self.current_step)
        return original_lookup(self, request)

    Scheduler._get_local_prefix_cache_hit = _get_local_prefix_cache_hit

    original_remote_kv = Scheduler._update_waiting_for_remote_kv

    @functools.wraps(original_remote_kv)
    def _update_waiting_for_remote_kv(self, request):
        original_remote_kv(self, request)
        pool = self.kv_cache_manager.block_pool
        if isinstance(pool, RegionBlockPool):
            twins = pool.twins.pop(request.request_id, [])
            if twins:
                pool.free_blocks(twins)

    Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv

    original_allocate = KVCacheManager.allocate_slots

    @functools.wraps(original_allocate)
    def allocate_slots(self, request, *args, **kwargs):
        pool = self.block_pool
        if not isinstance(pool, RegionBlockPool):
            return original_allocate(self, request, *args, **kwargs)
        group = pool.place(request.request_id)
        blocks = original_allocate(self, request, *args, **kwargs)
        if blocks is not None:
            pool.active.setdefault(request.request_id, group)
        return blocks

    KVCacheManager.allocate_slots = allocate_slots

    original_free = KVCacheManager.free

    @functools.wraps(original_free)
    def free(self, request):
        original_free(self, request)
        pool = self.block_pool
        if isinstance(pool, RegionBlockPool):
            twins = pool.release(request.request_id)
            if twins:
                pool.free_blocks(twins)

    KVCacheManager.free = free

    original_pop = KVCacheManager.pop_blocks_for_free

    @functools.wraps(original_pop)
    def pop_blocks_for_free(self, request):
        blocks = original_pop(self, request)
        pool = self.block_pool
        if isinstance(pool, RegionBlockPool):
            twins = pool.release(request.request_id)
            # Freed in reverse order: the twins go after the request's blocks.
            blocks = twins[::-1] + list(blocks)
        return blocks

    KVCacheManager.pop_blocks_for_free = pop_blocks_for_free
    logger.info("Applied K3 DP block regions patch to the decode scheduler")
