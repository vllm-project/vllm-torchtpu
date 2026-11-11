# SPDX-License-Identifier: Apache-2.0
"""A separate LRU budget for cached Mamba state blocks (engine core).

``VLLM_TPU_MAMBA_LRU_BUDGET=<blocks>`` (0 = off, the default).

The hybrid KV cache manager keeps attention KV blocks and Mamba "align"
state blocks in one free queue. A cached state block (a prefix checkpoint)
and a cached KV block then evict each other in plain LRU order, and losing
either half of a hybrid hit leaves the other half cached but useless: KV
with no state at its end cannot be resumed, and a state whose prefix KV is
gone cannot be reached.

With a budget, in the style of SGLang's ``--max-mamba-cache-size``:

- A cached Mamba state block whose last reference is dropped goes to a
  separate LRU (the "state LRU") instead of the shared free queue. The state
  LRU holds at most ``budget`` blocks; when it is over budget, the oldest
  checkpoint (all Mamba groups' blocks at that position together) is evicted
  from it. Attention allocations take blocks from the shared queue only, so
  they never evict a checkpoint, and state LRU evictions never evict KV.
  Only when the shared queue cannot satisfy an allocation does the oldest
  checkpoint spill (counted; the pool would otherwise have to preempt).
- Blocks in the state LRU count as free, so admission and KV usage see the
  same capacity as without the budget.
- Nothing is stranded. Every checkpoint is a node of a prefix tree keyed by
  the request's block hashes; it references the hash positions between its
  nearest ancestor checkpoint and itself (extended to the end of the
  enclosing attention block). When a checkpoint is evicted and no
  descendant checkpoint needs its prefix, the cached KV blocks that only it
  needed move to the reuse-first end of the shared queue, hashless. When a
  cached KV block is evicted, the checkpoints whose prefix included it (and
  their descendants) can no longer be reached: they leave the state LRU, and
  the KV that only they needed is released the same way. A request that
  loses its newest checkpoint therefore still hits the previous one, whose
  KV and state are both kept.

Everything is installed on the block pool instance of each KV cache manager
that has an align-mode Mamba group, so with the budget at 0 no code path
changes.
"""

import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import wraps

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_mamba_state_lru_patch"
_LOG_INTERVAL_S = 60.0


class _Node:
    __slots__ = ("parent", "children", "segment")

    def __init__(self, parent, segment: tuple) -> None:
        self.parent = parent
        self.children: set = set()
        self.segment = segment


class MambaStateLRU:
    """State LRU and prefix tree attached to one ``BlockPool``."""

    def __init__(
        self,
        pool,
        mamba_group_ids: Iterable[int],
        attention_group_ids: Iterable[int],
        attention_block_sizes: Iterable[int],
        budget: int,
    ) -> None:
        from vllm.v1.core.kv_cache_utils import (
            get_block_hash,
            get_group_id,
            make_block_hash_with_group_id,
        )

        self._get_block_hash = get_block_hash
        self._get_group_id = get_group_id
        self._make_key = make_block_hash_with_group_id
        self.pool = pool
        self.queue = pool.free_block_queue
        self.mamba_gids = frozenset(mamba_group_ids)
        self.attention_gids = tuple(sorted(attention_group_ids))
        self.attention_block_sizes = tuple(sorted(set(attention_block_sizes)))
        self.budget = budget
        # Checkpoint position hash -> its free cached state blocks (all
        # Mamba groups). Order: first = evicted first.
        self.lru: OrderedDict = OrderedDict()
        self.lru_key: dict[int, object] = {}  # block_id -> position hash
        self.lru_blocks = 0
        self.nodes: dict = {}
        self.hash_nodes: dict = {}
        self._depth = 0
        self._evicting = 0
        self._request = None
        self._lost_states: set = set()
        self._lost_kv: set = set()
        self._orphans: list = []
        self.stats = dict.fromkeys(
            (
                "parked",
                "hits",
                "budget_evictions",
                "spills",
                "kv_kills",
                "stranded_kv",
                "nodes_removed",
            ),
            0,
        )
        self._last_log = 0.0
        self._install()

    # ------------------------------------------------------------------ #
    # Installation
    # ------------------------------------------------------------------ #
    def _install(self) -> None:
        pool = self.pool
        self._orig = {
            name: getattr(pool, name)
            for name in (
                "free_blocks",
                "get_new_blocks",
                "touch",
                "get_num_free_blocks",
                "reset_prefix_cache",
                "evict_blocks",
                "_remove_cached_block_hashes",
                "_maybe_evict_cached_block",
                "_insert_block_hash",
                "cache_full_blocks",
                "cache_partial_block",
                "move_block_hashes",
            )
        }
        bound = {
            "free_blocks": self.free_blocks,
            "get_new_blocks": self.get_new_blocks,
            "touch": self.touch,
            "get_num_free_blocks": self.get_num_free_blocks,
            "reset_prefix_cache": self.reset_prefix_cache,
            "evict_blocks": self._boundary(self._orig["evict_blocks"]),
            "_remove_cached_block_hashes": self._remove_cached_block_hashes,
            "_maybe_evict_cached_block": self._maybe_evict_cached_block,
            "_insert_block_hash": self._insert_block_hash,
            "cache_full_blocks": self._with_request(self._orig["cache_full_blocks"]),
            "cache_partial_block": self._with_request(
                self._orig["cache_partial_block"]
            ),
            "move_block_hashes": self._boundary(self._orig["move_block_hashes"]),
        }
        for name, fn in bound.items():
            setattr(pool, name, fn)
        pool._tpu_mamba_state_lru = self

    def _boundary(self, fn):
        """Run ``fn``; settle deferred evictions when the outermost call ends."""

        @wraps(fn)
        def wrapped(*args, **kwargs):
            self._depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._settle()

        return wrapped

    def _with_request(self, fn):
        boundary = self._boundary(fn)

        @wraps(fn)
        def wrapped(request, *args, **kwargs):
            saved, self._request = self._request, request
            try:
                return boundary(request, *args, **kwargs)
            finally:
                self._request = saved

        return wrapped

    # ------------------------------------------------------------------ #
    # BlockPool overrides
    # ------------------------------------------------------------------ #
    def get_num_free_blocks(self) -> int:
        return self.queue.num_free_blocks + self.lru_blocks

    def free_blocks(self, ordered_blocks: Iterable) -> None:
        self._depth += 1
        try:
            states, rest = [], []
            for block in ordered_blocks:
                key = block.block_hash
                if (
                    key is not None
                    and block.ref_cnt == 1
                    and not block.is_null
                    and (block.pool is None or block.pool is self.pool)
                    and self._get_group_id(key) in self.mamba_gids
                ):
                    states.append(block)
                else:
                    rest.append(block)
            if rest:
                self._orig["free_blocks"](rest)
            for block in states:
                block.ref_cnt -= 1
                self._park(block)
            if states:
                self._enforce_budget()
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._settle()

    def get_new_blocks(self, num_blocks: int):
        self._depth += 1
        try:
            while num_blocks > self.queue.num_free_blocks and self.lru:
                key, blocks = self.lru.popitem(last=False)
                self.stats["spills"] += 1
                self._drop_entry(key, blocks)
            return self._orig["get_new_blocks"](num_blocks)
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._settle()

    def touch(self, blocks: Sequence) -> None:
        rest = []
        for block in blocks:
            key = self.lru_key.pop(block.block_id, None)
            if key is None:
                rest.append(block)
                continue
            entry = self.lru[key]
            entry.remove(block)
            if not entry:
                del self.lru[key]
            self.lru_blocks -= 1
            self.stats["hits"] += 1
            block.ref_cnt += 1
            if self.pool.metrics_collector:
                self.pool.metrics_collector.on_block_accessed(block)
        if rest:
            self._orig["touch"](rest)

    def reset_prefix_cache(self) -> bool:
        parked = [b for blocks in self.lru.values() for b in blocks]
        self.lru.clear()
        self.lru_key.clear()
        self.lru_blocks = 0
        self.queue.append_n(parked)
        self.nodes.clear()
        self.hash_nodes.clear()
        self._lost_states.clear()
        self._lost_kv.clear()
        self._orphans.clear()
        return self._orig["reset_prefix_cache"]()

    def _remove_cached_block_hashes(self, block):
        removed = self._orig["_remove_cached_block_hashes"](block)
        for key in removed:
            if self._get_group_id(key) in self.mamba_gids:
                self._lost_states.add(self._get_block_hash(key))
            elif self._evicting:
                self._lost_kv.add(key)
        if removed and block.block_id in self.lru_key:
            self._orphans.append(block)
        return removed

    def _maybe_evict_cached_block(self, block) -> bool:
        self._depth += 1
        self._evicting += 1
        try:
            return self._orig["_maybe_evict_cached_block"](block)
        finally:
            self._evicting -= 1
            self._depth -= 1
            if self._depth == 0:
                self._settle()

    def _insert_block_hash(self, key, block, num_tokens) -> None:
        self._orig["_insert_block_hash"](key, block, num_tokens)
        if (
            self._request is not None
            and num_tokens is not None
            and self._get_group_id(key) in self.mamba_gids
        ):
            self._register(self._get_block_hash(key), num_tokens, self._request)

    # ------------------------------------------------------------------ #
    # State LRU
    # ------------------------------------------------------------------ #
    def _park(self, block) -> None:
        key = self._get_block_hash(block.block_hash)
        entry = self.lru.get(key)
        if entry is None:
            self.lru[key] = [block]
        else:
            entry.append(block)
            self.lru.move_to_end(key)
        self.lru_key[block.block_id] = key
        self.lru_blocks += 1
        self.stats["parked"] += 1

    def _enforce_budget(self) -> None:
        while self.lru_blocks > self.budget and self.lru:
            key, blocks = self.lru.popitem(last=False)
            self.stats["budget_evictions"] += 1
            self._drop_entry(key, blocks)

    def _drop_entry(self, key, blocks: list) -> None:
        """Evict a state LRU entry: hashless, to the reuse-first queue end."""
        for block in blocks:
            del self.lru_key[block.block_id]
        self.lru_blocks -= len(blocks)
        self._depth += 1
        try:
            for block in blocks:
                self._orig["_maybe_evict_cached_block"](block)
        finally:
            self._depth -= 1
        self.queue.prepend_n(blocks)
        self._lost_states.add(key)

    def _state_alive(self, key) -> bool:
        cached = self.pool.cached_block_hash_to_block
        return all(
            cached.get_one_block(self._make_key(key, gid)) is not None
            for gid in self.mamba_gids
        )

    # ------------------------------------------------------------------ #
    # Prefix tree
    # ------------------------------------------------------------------ #
    def _register(self, key, num_tokens: int, request) -> None:
        if key in self.nodes:
            return
        hashes = request.block_hashes
        hash_block_size = self.pool.hash_block_size
        idx = num_tokens // hash_block_size - 1
        if num_tokens % hash_block_size or not 0 <= idx < len(hashes):
            return
        if hashes[idx] != key:
            return
        j = idx - 1
        while j >= 0 and hashes[j] not in self.nodes:
            j -= 1
        parent = hashes[j] if j >= 0 else None
        # A hit at ``num_tokens`` also needs the attention block holding it.
        end = idx
        for block_size in self.attention_block_sizes:
            block_end = -(-num_tokens // block_size) * block_size
            end = max(end, min(block_end // hash_block_size, len(hashes)) - 1)
        segment = tuple(hashes[i] for i in range(j + 1, end + 1))
        self.nodes[key] = _Node(parent, segment)
        if parent is not None:
            self.nodes[parent].children.add(key)
        for h in segment:
            refs = self.hash_nodes.get(h)
            if refs is None:
                self.hash_nodes[h] = {key}
            else:
                refs.add(key)

    def _remove_node(self, key) -> None:
        while key is not None:
            node = self.nodes.pop(key)
            self.stats["nodes_removed"] += 1
            for h in node.segment:
                refs = self.hash_nodes.get(h)
                if refs is None:
                    continue
                refs.discard(key)
                if not refs:
                    del self.hash_nodes[h]
                    self._strand(h)
            parent = self.nodes.get(node.parent) if node.parent is not None else None
            if parent is None:
                return
            parent.children.discard(key)
            key = node.parent
            if parent.children or self._state_alive(key):
                return

    def _strand(self, h) -> None:
        """Release the free cached KV at hash position ``h``: no checkpoint
        can resume past it any more."""
        cached = self.pool.cached_block_hash_to_block._cache
        demoted = []
        for gid in self.attention_gids:
            blocks = cached.get(self._make_key(h, gid))
            if blocks is None:
                continue
            candidates = (
                list(blocks.values()) if isinstance(blocks, dict) else [blocks]
            )
            for block in candidates:
                if block.ref_cnt != 0 or block.is_null:
                    continue
                self.queue.remove(block)
                self._orig["_maybe_evict_cached_block"](block)
                demoted.append(block)
        if demoted:
            self.stats["stranded_kv"] += len(demoted)
            self.queue.prepend_n(demoted)

    def _kill(self, key) -> None:
        """The prefix of ``key`` lost a KV block: drop it and its subtree."""
        stack = [key]
        while stack:
            k = stack.pop()
            node = self.nodes.get(k)
            if node is None:
                continue
            stack.extend(node.children)
            blocks = self.lru.pop(k, None)
            if blocks is not None:
                self._drop_entry(k, blocks)
            self._lost_states.add(k)

    def _settle(self) -> None:
        """Resolve evictions recorded during the outermost pool call."""
        self._depth += 1
        try:
            while self._orphans or self._lost_kv or self._lost_states:
                while self._orphans:
                    block = self._orphans.pop()
                    key = self.lru_key.pop(block.block_id, None)
                    if key is None or block.block_hash is not None:
                        if key is not None:
                            self.lru_key[block.block_id] = key
                        continue
                    entry = self.lru[key]
                    entry.remove(block)
                    if not entry:
                        del self.lru[key]
                    self.lru_blocks -= 1
                    self.queue.prepend_n([block])
                cached = self.pool.cached_block_hash_to_block
                while self._lost_kv:
                    key = self._lost_kv.pop()
                    if cached.get_one_block(key) is not None:
                        continue
                    for node_key in tuple(
                        self.hash_nodes.get(self._get_block_hash(key), ())
                    ):
                        self.stats["kv_kills"] += 1
                        self._kill(node_key)
                while self._lost_states:
                    key = self._lost_states.pop()
                    if self._state_alive(key):
                        continue
                    blocks = self.lru.pop(key, None)
                    if blocks is not None:
                        # Another group's block of this checkpoint is gone.
                        self._drop_entry(key, blocks)
                    node = self.nodes.get(key)
                    if node is not None and not node.children:
                        self._remove_node(key)
        finally:
            self._depth -= 1
        self._maybe_log()

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S or not self.stats["parked"]:
            return
        self._last_log = now
        logger.info(
            "VLLM_TPU_MAMBA_LRU_BUDGET: %d/%d state blocks parked (%d "
            "checkpoints), %d tree nodes; parked %d, hits %d, budget evictions "
            "%d, spills %d, KV-loss kills %d, stranded KV released %d.",
            self.lru_blocks,
            self.budget,
            len(self.lru),
            len(self.nodes),
            self.stats["parked"],
            self.stats["hits"],
            self.stats["budget_evictions"],
            self.stats["spills"],
            self.stats["kv_kills"],
            self.stats["stranded_kv"],
        )


def install_mamba_state_lru(coordinator, budget: int) -> MambaStateLRU | None:
    """Attach the state LRU to ``coordinator``'s block pool, if it applies."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    pool = coordinator.block_pool
    if not pool.enable_caching:
        return None
    mamba, attention, sizes = [], [], []
    for gid, manager in enumerate(coordinator.single_type_managers):
        if manager.block_pool is not pool:
            logger.warning(
                "VLLM_TPU_MAMBA_LRU_BUDGET: group %d has its own block pool; "
                "the budget is off.",
                gid,
            )
            return None
        if isinstance(manager, MambaManager):
            if manager.mamba_cache_mode != "align":
                return None
            mamba.append(gid)
        else:
            attention.append(gid)
            sizes.append(manager.block_size)
    if not mamba:
        return None
    return MambaStateLRU(pool, mamba, attention, sizes, budget)


def patch_mamba_state_lru() -> None:
    budget = envs.VLLM_TPU_MAMBA_LRU_BUDGET
    if budget <= 0:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_init = KVCacheManager.__init__
    if getattr(original_init, _PATCH_FLAG, False):
        return

    @wraps(original_init)
    def __init__(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        lru = install_mamba_state_lru(self.coordinator, budget)
        if lru is None:
            logger.warning(
                "VLLM_TPU_MAMBA_LRU_BUDGET=%d: no align-mode Mamba group with "
                "prefix caching on one shared block pool; the budget is off.",
                budget,
            )
            return
        logger.info(
            "VLLM_TPU_MAMBA_LRU_BUDGET: on; cached Mamba state blocks get their "
            "own LRU of %d blocks (Mamba groups %s, attention groups %s).",
            budget,
            sorted(lru.mamba_gids),
            list(lru.attention_gids),
        )

    setattr(__init__, _PATCH_FLAG, True)
    KVCacheManager.__init__ = __init__
