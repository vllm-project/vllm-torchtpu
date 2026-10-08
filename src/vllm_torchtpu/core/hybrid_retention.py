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
"""Prefix-cache retention for hybrid attention + Mamba/GDN ("align") models.

With ``--mamba-cache-mode align`` and ``--prefix-cache-retention-interval 0``,
the next turn of a conversation can only hit at the GDN state snapshot taken
at the previous prompt's boundary. If that snapshot is gone, the hit is 0 even
though every attention block is still cached.

vLLM frees the snapshot early. The floor-block state is freed by
``remove_skipped_blocks`` at the first decode block crossing, and the
partial-tail CoW copy is freed one step after prefill. The request's attention
blocks are only freed at finish. So in the LRU free queue the snapshot is
older than the attention blocks by the whole decode time. Under pressure it is
evicted first: the conversation becomes a full re-prefill while its 100k+
attention blocks remain in cache without a matching Mamba state. Under
multi-turn long-context serving, this causes avoidable re-prefills when follow-up
turns arrive after multi-second client think times.

This module keeps both the full-block Mamba boundary snapshot (e.g. 640-token
aligned) and the fine-grained CoW partial-tail snapshot (e.g. 128-token
aligned) alive during decode and across multi-turn think-time:

1. Pin running (full boundary + partial tail). Once a decoding request has
   released its full boundary snapshot or its CoW partial-tail snapshot
   (ref_cnt 0), take a reference on it. Hooks ``MambaManager._cache_partial_tail_block``
   and ``MambaManager.allocate_new_blocks`` (CoW hash move) in addition to
   dual-boundary probing so both blocks are tracked and pinned.
2. Refresh & Demote at turn end. When a request finishes, its snapshot blocks
   are placed at the MRU end of the free queue after its attention blocks, and
   unreachable decode-output attention blocks past the last retained snapshot
   (+1 EAGLE block) are demoted to the head of the free queue.
3. Cross-turn Conversation LRU. At the end of a turn, ownership of at most the
   2 latest Mamba blocks per Mamba group (full boundary + partial tail) is
   transferred into a bounded cross-turn conversation LRU
   (``TPU_HYBRID_RETENTION_MAX_CONVS``, default 64). When a newer turn of the
   same conversation finishes (or a 65th conversation arrives), the older
   turn's Mamba blocks are unpinned. A guard on ``block_pool.get_new_blocks``
   (and ``allocate_slots`` / ``reset_prefix_cache``) automatically evicts the
   oldest conversation entries whenever free blocks run low, guaranteeing zero
   allocation failures or extra preemptions.

Models without a Mamba group, or with prefix caching off, are untouched.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, NamedTuple

from vllm.logger import init_logger

import vllm_torchtpu.envs as envs

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000
# Give up pinning when decode has gone this many Mamba blocks past the prompt
# without the snapshot being released (e.g. another running request shares it).
_PIN_MAX_DECODE_BLOCKS = 4
# Probes that found no boundary snapshot scan the whole prompt; cap them.
_PIN_MAX_MISSED_PROBES = 3


def _is_spec(spec: Any, name: str) -> bool:
    return any(c.__name__ == name for c in type(spec).__mro__)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _ConvLRUEntry(NamedTuple):
    conv_key: Any
    req_id: str
    req_prefix: str | None
    first_hash: Any
    boundary_idx: int
    boundary_hash: Any
    blocks: list[Any]
    hit: int


class HybridStateRetention:
    """Keeps GDN boundary and partial-tail snapshots alive during decode and
    across multi-turn conversation think-time."""

    def __init__(
        self,
        kv_cache_manager: Any,
        max_conversations: int | None = None,
    ) -> None:
        self.kvm = kv_cache_manager
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        groups = list(getattr(coordinator, "attention_groups", None) or ())
        self.coordinator = coordinator
        self.mamba_groups = [g for g in groups if _is_spec(g.spec, "MambaSpec")]
        self.attn_group_ids = [
            gid
            for g in groups
            if _is_spec(g.spec, "FullAttentionSpec")
            for gid in g.group_ids
        ]
        self.enabled = bool(
            getattr(kv_cache_manager, "enable_caching", False)
            and self.mamba_groups
            and self.attn_group_ids
        )
        self.block_pool = getattr(coordinator, "block_pool", None)
        if max_conversations is None:
            max_conversations = int(envs.TPU_HYBRID_RETENTION_MAX_CONVS)
        self.max_conversations = max(0, int(max_conversations))

        # request_id -> (pinned snapshot blocks, snapshot hit length)
        self._pinned: dict[str, tuple[list[Any], int]] = {}
        # request_ids that pinned their full boundary block and are still
        # waiting for their CoW partial-tail block to be created/released.
        self._pending_partial: set[str] = set()
        # request_id -> {(gid, boundary_tokens): (block, boundary_tokens)}
        # populated by MambaManager._cache_partial_tail_block / CoW hooks.
        self._tracked_blocks: dict[str, dict[tuple[int, int], tuple[Any, int]]] = {}
        # Cross-turn conversation LRU: conv_key -> _ConvLRUEntry
        self._conv_lru: OrderedDict[Any, _ConvLRUEntry] = OrderedDict()

        # request_id -> number of probes that found no snapshot
        self._missed: dict[str, int] = {}
        self._gave_up: set[str] = set()
        self._n_freed = 0
        self._n_pinned = 0
        self._n_refreshed = 0
        self._n_demoted_blocks = 0
        self._n_no_state = 0
        self._n_conv_retained = 0
        self._n_conv_evicted = 0

        if self.enabled:
            self._install_mamba_manager_hooks()
            self._install_block_pool_guards()

    # ------------------------------------------------------------------- hooks
    def _install_mamba_manager_hooks(self) -> None:
        """Hook MambaManager partial-tail caching and CoW hash movement so both
        the full 640-token boundary block and 128-token CoW partial-tail block
        are tracked for running requests."""
        managers = getattr(self.coordinator, "single_type_managers", None) or ()
        for group in self.mamba_groups:
            for gid in group.group_ids:
                if gid >= len(managers):
                    continue
                mgr = managers[gid]
                if getattr(mgr, "_hybrid_retention_hooked", False):
                    continue
                mgr._hybrid_retention_hooked = True

                orig_cache_partial = getattr(
                    mgr, "_cache_partial_tail_block", None
                )
                if callable(orig_cache_partial):

                    def _wrapped_cache_partial(
                        request: Any,
                        num_tokens: int,
                        _orig: Any = orig_cache_partial,
                        _mgr: Any = mgr,
                        _gid: int = gid,
                    ) -> Any:
                        res = _orig(request, num_tokens)
                        if res is not None:
                            rid = request.request_id
                            partial_reqs = getattr(_mgr, "_partial_hit_reqs", None)
                            if isinstance(partial_reqs, dict) and rid in partial_reqs:
                                _, src_blk = partial_reqs[rid]
                                self._record_tracked_block(
                                    rid, _gid, src_blk, int(num_tokens)
                                )
                            else:
                                req_blocks = getattr(
                                    _mgr, "req_to_blocks", {}
                                ).get(rid)
                                bidx = int(num_tokens) // int(_mgr.block_size)
                                if req_blocks and bidx < len(req_blocks):
                                    self._record_tracked_block(
                                        rid, _gid, req_blocks[bidx], int(num_tokens)
                                    )
                        return res

                    mgr._cache_partial_tail_block = _wrapped_cache_partial

                orig_alloc_new = getattr(mgr, "allocate_new_blocks", None)
                if callable(orig_alloc_new):

                    def _wrapped_allocate_new_blocks(
                        request_id: str,
                        num_tokens: int,
                        num_tokens_main_model: int,
                        _orig: Any = orig_alloc_new,
                        _mgr: Any = mgr,
                        _gid: int = gid,
                    ) -> Any:
                        partial_reqs = getattr(_mgr, "_partial_hit_reqs", None)
                        had_partial = (
                            partial_reqs.get(request_id)
                            if isinstance(partial_reqs, dict)
                            else None
                        )
                        pending_cows = getattr(_mgr, "_pending_cow_copies", None)
                        cow_len_before = (
                            len(pending_cows)
                            if isinstance(pending_cows, list)
                            else 0
                        )
                        ret = _orig(
                            request_id, num_tokens, num_tokens_main_model
                        )
                        if had_partial is not None and isinstance(
                            pending_cows, list
                        ):
                            _, src_blk = had_partial
                            for src_c, cow_blk in pending_cows[cow_len_before:]:
                                if src_c is src_blk:
                                    self._replace_tracked_block(
                                        request_id, _gid, src_blk, cow_blk
                                    )
                        return ret

                    mgr.allocate_new_blocks = _wrapped_allocate_new_blocks

    def _record_tracked_block(
        self, request_id: str, gid: int, block: Any, boundary_tokens: int
    ) -> None:
        if block is None or getattr(block, "is_null", False):
            return
        per_req = self._tracked_blocks.setdefault(request_id, {})
        per_req[(gid, boundary_tokens)] = (block, boundary_tokens)

    def _replace_tracked_block(
        self, request_id: str, gid: int, old_block: Any, new_block: Any
    ) -> None:
        per_req = self._tracked_blocks.get(request_id)
        if not per_req:
            return
        for key, (blk, b_tokens) in list(per_req.items()):
            if key[0] == gid and blk is old_block:
                per_req[key] = (new_block, b_tokens)

    def _install_block_pool_guards(self) -> None:
        """Guard BlockPool.get_new_blocks (and KVCacheManager.allocate_slots /
        reset_prefix_cache) so cross-turn Mamba pins are automatically evicted
        whenever free blocks run low."""
        pool = self.block_pool
        if pool is not None and not getattr(
            pool, "_hybrid_retention_guarded", False
        ):
            pool._hybrid_retention_guarded = True
            orig_get_new_blocks = getattr(pool, "get_new_blocks", None)
            if callable(orig_get_new_blocks):

                def _guarded_get_new_blocks(
                    num_blocks: int, _orig: Any = orig_get_new_blocks
                ) -> Any:
                    while (
                        self._conv_lru
                        and self._get_num_free_blocks() < num_blocks
                    ):
                        if not self.evict_oldest_conversation():
                            break
                    return _orig(num_blocks)

                pool.get_new_blocks = _guarded_get_new_blocks

            orig_reset = getattr(pool, "reset_prefix_cache", None)
            if callable(orig_reset):

                def _guarded_reset_prefix_cache(
                    _orig: Any = orig_reset,
                ) -> Any:
                    self.evict_all_conversations()
                    return _orig()

                pool.reset_prefix_cache = _guarded_reset_prefix_cache

        kvm = self.kvm
        if kvm is not None and not getattr(
            kvm, "_hybrid_retention_guarded", False
        ):
            kvm._hybrid_retention_guarded = True
            orig_allocate_slots = getattr(kvm, "allocate_slots", None)
            if callable(orig_allocate_slots):

                def _guarded_allocate_slots(
                    *args: Any,
                    _orig: Any = orig_allocate_slots,
                    **kwargs: Any,
                ) -> Any:
                    res = _orig(*args, **kwargs)
                    while res is None and self._conv_lru:
                        if not self.evict_oldest_conversation():
                            break
                        res = _orig(*args, **kwargs)
                    return res

                kvm.allocate_slots = _guarded_allocate_slots

    def _get_num_free_blocks(self) -> int:
        pool = self.block_pool
        if pool is None:
            return 0
        get_free = getattr(pool, "get_num_free_blocks", None)
        if callable(get_free):
            return int(get_free())
        q = getattr(pool, "free_block_queue", None)
        if hasattr(q, "num_free_blocks"):
            return int(q.num_free_blocks)
        if hasattr(q, "q"):
            return len(q.q)
        if isinstance(q, (list, tuple, set)):
            return len(q)
        return 0

    # ------------------------------------------------------------------ probes
    def _mamba_block_size(self) -> int:
        gid = self.mamba_groups[0].group_ids[0]
        return int(self.coordinator.single_type_managers[gid].block_size)

    def _alignment_tokens(self) -> int:
        alignment = int(
            getattr(self.coordinator, "_cache_hit_alignment_tokens", 0) or 0
        )
        return alignment if alignment > 0 else self._mamba_block_size()

    def _max_retained_blocks_per_turn(self) -> int:
        num_mamba_gids = max(
            1, sum(len(g.group_ids) for g in self.mamba_groups)
        )
        return 2 * num_mamba_gids

    def _probe(self, request: Any, max_length: int) -> tuple[list[Any], int]:
        """Return (snapshot blocks of every Mamba group, hit length) for the
        longest retained snapshot at or below ``max_length`` (0 if none)."""
        if max_length <= 0:
            return [], 0
        coordinator = self.coordinator
        alignment = coordinator._cache_hit_alignment_tokens
        states: list[Any] = []
        hit = max_length
        for group in self.mamba_groups:
            mgr = coordinator.single_type_managers[group.group_ids[0]]
            hit_blocks, group_hit = group.manager_cls.find_longest_cache_hit(
                block_hashes=request.block_hashes,
                max_length=max_length,
                kv_cache_group_ids=group.group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=group.spec,
                drop_eagle_block=False,
                alignment_tokens=alignment,
                dcp_world_size=getattr(mgr, "dcp_world_size", 1),
                pcp_world_size=getattr(mgr, "pcp_world_size", 1),
            )
            if group_hit <= 0:
                return [], 0
            hit = min(hit, group_hit)
            for blocks in hit_blocks:
                if blocks and not blocks[-1].is_null:
                    states.append(blocks[-1])
        return states, hit

    def _probe_own_states(self, request: Any) -> tuple[list[Any], int]:
        """Probe both the full Mamba block boundary (floor640(prompt)) and the
        fine partial-tail boundary (floor128(prompt)), plus any tracked CoW
        partial-tail blocks, returning at most 2 blocks per Mamba group."""
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if prompt <= 0:
            return [], 0
        bs = self._mamba_block_size()
        hb = self._alignment_tokens()
        fine_boundary = (prompt // hb) * hb
        full_boundary = (fine_boundary // bs) * bs

        states: list[Any] = []
        seen_ids: set[int] = set()
        hit = 0

        # 1. Full block boundary (e.g. 640-token aligned) when distinct from
        #    the fine partial-tail boundary.
        if full_boundary >= bs and full_boundary < fine_boundary:
            full_states, full_hit = self._probe(request, full_boundary)
            if full_states and self._is_own_boundary(request, full_hit):
                for b in full_states:
                    if id(b) not in seen_ids:
                        seen_ids.add(id(b))
                        states.append(b)
                hit = max(hit, full_hit)

        # 2. Fine-grained partial-tail boundary (e.g. 128-token aligned) or
        #    full boundary when prompt is block-aligned.
        fine_states, fine_hit = self._probe(request, prompt)
        if fine_states and self._is_own_boundary(request, fine_hit):
            for b in fine_states:
                if id(b) not in seen_ids:
                    seen_ids.add(id(b))
                    states.append(b)
            hit = max(hit, fine_hit)

        # 3. Any CoW partial-tail blocks recorded via MambaManager hooks.
        tracked = self._tracked_blocks.get(request.request_id)
        if tracked:
            for (_, _), (b, b_tokens) in sorted(tracked.items()):
                if (
                    b is not None
                    and not getattr(b, "is_null", False)
                    and getattr(b, "block_hash", None) is not None
                    and self._is_own_boundary(request, b_tokens)
                ):
                    if id(b) not in seen_ids:
                        seen_ids.add(id(b))
                        states.append(b)
                    hit = max(hit, b_tokens)

        max_blocks = self._max_retained_blocks_per_turn()
        if len(states) > max_blocks:
            states = states[-max_blocks:]
        return states, hit

    def _is_own_boundary(self, request: Any, hit: int) -> bool:
        # The replay boundary is floor(prompt) at hash or block granularity;
        # anything older is a shared-prefix junction, not this turn's snapshot.
        return hit > 0 and hit > request.num_prompt_tokens - 2 * self._mamba_block_size()

    # ------------------------------------------------------- conversation LRU
    def _conv_key(self, request: Any) -> Any:
        block_hashes = getattr(request, "block_hashes", None)
        if block_hashes:
            return ("hash", len(block_hashes), block_hashes[-1])
        return ("id", str(getattr(request, "request_id", "") or ""))

    def _claim_from_conv_lru_if_sole_owner(self, block: Any) -> bool:
        """If ``block`` is currently held solely by ``_conv_lru`` (ref_cnt == 1),
        transfer its +1 ref out of ``_conv_lru`` to the caller."""
        if getattr(block, "ref_cnt", 0) != 1:
            return False
        for key, entry in list(self._conv_lru.items()):
            for idx, b in enumerate(entry.blocks):
                if b is block:
                    remaining = entry.blocks[:idx] + entry.blocks[idx + 1 :]
                    if remaining:
                        self._conv_lru[key] = entry._replace(blocks=remaining)
                    else:
                        del self._conv_lru[key]
                    return True
        return False

    def _evict_matching_conv_entries(self, request: Any) -> None:
        """Evict any prior turn's entry in ``_conv_lru`` for ``request``'s
        conversation before storing ``request``'s new turn blocks."""
        if not self._conv_lru:
            return
        conv_key = self._conv_key(request)
        rid = str(getattr(request, "request_id", "") or "")
        req_prefix = rid.split("_", 1)[0] if "_" in rid else None
        block_hashes = list(getattr(request, "block_hashes", None) or ())

        keys_to_evict: list[Any] = []
        for key, entry in self._conv_lru.items():
            if key == conv_key:
                keys_to_evict.append(key)
                continue
            if req_prefix is not None and entry.req_prefix is not None:
                if req_prefix == entry.req_prefix:
                    keys_to_evict.append(key)
                continue
            if (
                0 <= entry.boundary_idx < len(block_hashes)
                and entry.boundary_hash is not None
                and block_hashes[entry.boundary_idx] == entry.boundary_hash
            ):
                keys_to_evict.append(key)

        for key in keys_to_evict:
            self.evict_conversation(key)

    def evict_conversation(self, conv_key: Any) -> bool:
        """Unpin and remove ``conv_key`` from the cross-turn conversation LRU."""
        entry = self._conv_lru.pop(conv_key, None)
        if entry is None:
            return False
        to_free = [
            b
            for b in entry.blocks
            if not getattr(b, "is_null", False) and getattr(b, "ref_cnt", 0) > 0
        ]
        if to_free:
            self.block_pool.free_blocks(to_free)
        self._n_conv_evicted += 1
        return True

    def evict_oldest_conversation(self) -> bool:
        """Evict the least-recently-used conversation from ``_conv_lru``."""
        if not self._conv_lru:
            return False
        oldest_key = next(iter(self._conv_lru))
        return self.evict_conversation(oldest_key)

    def evict_all_conversations(self) -> int:
        """Evict all conversations from ``_conv_lru`` and return count."""
        count = 0
        while self._conv_lru:
            if self.evict_oldest_conversation():
                count += 1
            else:
                break
        return count

    def _store_in_conv_lru(
        self, request: Any, blocks: list[Any], hit: int
    ) -> None:
        if self.max_conversations <= 0 or not blocks or hit <= 0:
            return
        max_blocks = self._max_retained_blocks_per_turn()
        candidates = [
            b
            for b in blocks[-max_blocks:]
            if not getattr(b, "is_null", False)
            and getattr(b, "block_hash", None) is not None
            and getattr(b, "ref_cnt", 0) == 0
        ]
        if not candidates:
            return

        self._evict_matching_conv_entries(request)

        #Re-filter in case _evict_matching_conv_entries changed ref_cnt.
        retainable = [
            b
            for b in candidates
            if not getattr(b, "is_null", False)
            and getattr(b, "block_hash", None) is not None
            and getattr(b, "ref_cnt", 0) == 0
        ]
        if not retainable:
            return

        self.block_pool.touch(retainable)
        rid = str(getattr(request, "request_id", "") or "")
        req_prefix = rid.split("_", 1)[0] if "_" in rid else None
        block_hashes = list(getattr(request, "block_hashes", None) or ())
        first_hash = block_hashes[0] if block_hashes else None
        hb = self._alignment_tokens()
        boundary_idx = hit // hb - 1
        boundary_hash = (
            block_hashes[boundary_idx]
            if 0 <= boundary_idx < len(block_hashes)
            else None
        )
        conv_key = self._conv_key(request)
        self._conv_lru[conv_key] = _ConvLRUEntry(
            conv_key=conv_key,
            req_id=rid,
            req_prefix=req_prefix,
            first_hash=first_hash,
            boundary_idx=boundary_idx,
            boundary_hash=boundary_hash,
            blocks=retainable,
            hit=hit,
        )
        self._conv_lru.move_to_end(conv_key)
        self._n_conv_retained += 1

        while len(self._conv_lru) > self.max_conversations:
            if not self.evict_oldest_conversation():
                break

    # -------------------------------------------------------------------- pin
    def pin_running(self, running: Any) -> None:
        """Pin released boundary and partial-tail snapshots of decoding
        requests. Call once per schedule() before allocation."""
        if not self.enabled:
            return
        bs = self._mamba_block_size()
        hb = self._alignment_tokens()
        for request in running:
            rid = request.request_id
            if rid in self._gave_up:
                continue
            if rid in self._pinned and rid not in self._pending_partial:
                continue
            prompt = request.num_prompt_tokens
            if prompt <= 0 or request.num_computed_tokens < prompt:
                continue
            states, hit = self._probe_own_states(request)
            if not states or not self._is_own_boundary(request, hit):
                missed = self._missed.get(rid, 0) + 1
                self._missed[rid] = missed
                if missed >= _PIN_MAX_MISSED_PROBES:
                    self._pending_partial.discard(rid)
                    if rid not in self._pinned:
                        self._gave_up.add(rid)
                continue

            cur_pinned, cur_hit = self._pinned.get(rid, ([], 0))
            cur_pinned = list(cur_pinned)
            cur_ids = {id(b) for b in cur_pinned}
            to_touch: list[Any] = []
            still_unreleased: list[Any] = []

            for b in states:
                if id(b) in cur_ids:
                    continue
                if b.ref_cnt == 0:
                    to_touch.append(b)
                elif self._claim_from_conv_lru_if_sole_owner(b):
                    cur_pinned.append(b)
                    cur_ids.add(id(b))
                else:
                    still_unreleased.append(b)

            if to_touch:
                self.block_pool.touch(to_touch)
                cur_pinned.extend(to_touch)

            if cur_pinned:
                was_pinned = rid in self._pinned
                new_hit = max(cur_hit, hit)
                self._pinned[rid] = (cur_pinned, new_hit)
                self._missed.pop(rid, None)
                if not was_pinned:
                    self._n_pinned += 1

            fine_boundary = (prompt // hb) * hb
            full_boundary = (fine_boundary // bs) * bs
            expects_partial = (
                full_boundary >= bs
                and fine_boundary > full_boundary
                and max(cur_hit, hit) < fine_boundary
            )
            if still_unreleased or expects_partial:
                decoded = request.num_computed_tokens - prompt
                if decoded > _PIN_MAX_DECODE_BLOCKS * bs:
                    self._pending_partial.discard(rid)
                    if not cur_pinned:
                        self._gave_up.add(rid)
                elif cur_pinned:
                    self._pending_partial.add(rid)
            else:
                self._pending_partial.discard(rid)

    # ------------------------------------------------------------------- free
    def snapshot_attention_blocks(self, request: Any) -> list[tuple[int, list[Any]]]:
        """Copy the request's full-attention block lists before they are freed."""
        if not self.enabled:
            return []
        out = []
        managers = self.coordinator.single_type_managers
        for gid in self.attn_group_ids:
            mgr = managers[gid]
            blocks = mgr.req_to_blocks.get(request.request_id)
            if blocks:
                out.append((int(mgr.block_size), list(blocks)))
        return out

    def on_request_freed(
        self, request: Any, attn_blocks: list[tuple[int, list[Any]]]
    ) -> None:
        """Call right after the scheduler freed a request's blocks (finish,
        abort or preemption; all go through ``_free_request_blocks``)."""
        if not self.enabled:
            return
        rid = request.request_id
        self._missed.pop(rid, None)
        self._gave_up.discard(rid)
        self._pending_partial.discard(rid)
        queue = self.block_pool.free_block_queue
        pinned = self._pinned.pop(rid, None)
        probed_states, probed_hit = self._probe_own_states(request)
        self._tracked_blocks.pop(rid, None)

        if pinned is not None:
            states, hit = list(pinned[0]), pinned[1]
            # Release pinned blocks first so hashed blocks are appended to MRU.
            self.block_pool.free_blocks(states)
            pinned_ids = {id(b) for b in states}
            if probed_states and self._is_own_boundary(request, probed_hit):
                for block in probed_states:
                    if id(block) in pinned_ids:
                        continue
                    if self._claim_from_conv_lru_if_sole_owner(block):
                        self.block_pool.free_blocks([block])
                        states.append(block)
                        pinned_ids.add(id(block))
                    elif block.ref_cnt == 0 and not block.is_null:
                        queue.remove(block)
                        queue.append(block)
                        states.append(block)
                        pinned_ids.add(id(block))
                hit = max(hit, probed_hit)
        else:
            states, hit = probed_states, probed_hit
            if states and self._is_own_boundary(request, hit):
                moved = False
                for block in states:
                    if self._claim_from_conv_lru_if_sole_owner(block):
                        self.block_pool.free_blocks([block])
                        moved = True
                    elif block.ref_cnt == 0 and not block.is_null:
                        queue.remove(block)
                        queue.append(block)
                        moved = True
                self._n_refreshed += int(moved)
            else:
                states = []
                hit = 0
                self._n_no_state += 1

        if hit > 0:
            # Longest snapshot anywhere in the sequence: with dense retention
            # decode-time states are hashed too and nothing is demoted.
            _, reach = self._probe(request, request.num_tokens - 1)
            reach_hit = max(hit, reach)
            dead = []
            for block_size, blocks in attn_blocks:
                # +1 block: EAGLE groups match one block past the boundary.
                for block in blocks[_cdiv(reach_hit, block_size) + 1 :]:
                    if (
                        block.ref_cnt == 0
                        and not block.is_null
                        and block.block_hash is not None
                    ):
                        dead.append(block)
            if dead:
                for block in dead:
                    queue.remove(block)
                queue.prepend_n(dead)
                self._n_demoted_blocks += len(dead)

            # Transfer ownership of at most the 2 latest Mamba blocks per group
            # (full boundary + partial tail) into the cross-turn conversation LRU.
            self._store_in_conv_lru(request, states, hit)

        self._n_freed += 1
        if self._n_freed % _LOG_EVERY == 0:
            logger.info(
                "hybrid-retention: freed=%d pinned=%d refreshed=%d no_state=%d "
                "demoted_blocks=%d live_pins=%d conv_lru=%d conv_evicted=%d",
                self._n_freed,
                self._n_pinned,
                self._n_refreshed,
                self._n_no_state,
                self._n_demoted_blocks,
                len(self._pinned),
                len(self._conv_lru),
                self._n_conv_evicted,
            )
