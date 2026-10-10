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
"""Prefix-cache change log for PrefixAwareScheduler's cross-step memo.

Layered on top of the same-step prefix-hit patch in ``vllm_torchtpu/__init__``
(which owns ``_tpu_prefix_cache_epoch`` and ``_tpu_unwritten_block_ids``) and
kept out of that file on purpose: ``vllm_torchtpu/__init__.py`` also defines a
patched VocabParallelEmbedding.forward that Dynamo traces, so editing it would
change vLLM's traced-file code hash and force a full recompile.

Once enabled on a BlockPool, every event that can change the answer of
``BlockPool.get_cached_block(h, groups)`` appends ``(epoch, h)`` to
``pool._tpu_prefix_change_log`` (``h`` = group-less block hash):

* a key inserted (all keys of the block are logged: the block joining the
  unwritten set flips the answer of every key whose first block it is);
* a key removed (eviction / re-pointing);
* the unwritten set cleared at ``Scheduler.schedule`` start (the plugin wrapper
  bumps the epoch by one and clears; we log the affected keys with that epoch
  just before);
* ``reset_prefix_cache`` drops the log and raises the log base epoch.
"""

import functools

LOG = "_tpu_prefix_change_log"
BASE = "_tpu_prefix_change_log_base"
EPOCH = "_tpu_prefix_cache_epoch"
UNWRITTEN = "_tpu_unwritten_block_ids"
_MARK = "_tpu_prefix_change_log_patch"


def _log_block_keys(d, log, epoch, block, extra_key=None) -> None:
    if extra_key is not None:
        log.append((epoch, extra_key[:-4]))
    key = getattr(block, "block_hash", None)
    if key is not None and key is not extra_key:
        log.append((epoch, key[:-4]))
    by_block = d.get("cached_block_hashes_by_block")
    if by_block:
        for key in by_block.get(block.block_id, ()):
            if key is not extra_key:
                log.append((epoch, key[:-4]))


def log_unwritten_clear(pool) -> None:
    """Log keys whose answer flips when the unwritten set is cleared by the
    plugin's schedule wrapper (which bumps the epoch by exactly one)."""
    d = pool.__dict__
    log = d.get(LOG)
    unwritten = d.get(UNWRITTEN)
    if log is None or not unwritten:
        return
    epoch = d.get(EPOCH, 0) + 1
    blocks = pool.blocks
    for block_id in unwritten:
        _log_block_keys(d, log, epoch, blocks[block_id])


def install() -> bool:
    """Wrap BlockPool/Scheduler (idempotent). Requires the plugin's same-step
    prefix-hit patch; returns False (memo stays per-step) without it."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(BlockPool._insert_block_hash, _MARK, False):
        return True
    for fn in (
        BlockPool._insert_block_hash,
        BlockPool._remove_cached_block_hashes,
        BlockPool.reset_prefix_cache,
        BlockPool.get_cached_block,
        Scheduler.schedule,
    ):
        if not getattr(fn, "_tpu_same_step_prefix_hit_patch", False):
            return False

    inner_insert = BlockPool._insert_block_hash
    inner_remove = BlockPool._remove_cached_block_hashes
    inner_reset = BlockPool.reset_prefix_cache
    inner_schedule = Scheduler.schedule

    @functools.wraps(inner_insert)
    def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens):
        inner_insert(self, block_hash_with_group_id, block, num_tokens)
        d = self.__dict__
        log = d.get(LOG)
        if log is not None:
            _log_block_keys(d, log, d.get(EPOCH, 0), block, block_hash_with_group_id)

    @functools.wraps(inner_remove)
    def _remove_cached_block_hashes(self, *args, **kwargs):
        evicted = inner_remove(self, *args, **kwargs)
        if evicted:
            d = self.__dict__
            log = d.get(LOG)
            if log is not None:
                epoch = d.get(EPOCH, 0)
                for key in evicted:
                    log.append((epoch, key[:-4]))
        return evicted

    @functools.wraps(inner_reset)
    def reset_prefix_cache(self):
        res = inner_reset(self)
        d = self.__dict__
        log = d.get(LOG)
        if log is not None:
            log.clear()
            d[BASE] = d.get(EPOCH, 0)
        return res

    @functools.wraps(inner_schedule)
    def schedule(self, *args, **kwargs):
        log_unwritten_clear(self.kv_cache_manager.block_pool)
        return inner_schedule(self, *args, **kwargs)

    for wrapper in (_insert_block_hash, _remove_cached_block_hashes, reset_prefix_cache, schedule):
        setattr(wrapper, _MARK, True)
        wrapper._tpu_same_step_prefix_hit_patch = True
    BlockPool._insert_block_hash = _insert_block_hash
    BlockPool._remove_cached_block_hashes = _remove_cached_block_hashes
    BlockPool.reset_prefix_cache = reset_prefix_cache
    Scheduler.schedule = schedule
    return True


def enable(pool) -> None:
    d = pool.__dict__
    d.setdefault(LOG, [])
    d.setdefault(BASE, d.get(EPOCH, 0))


def record_probes(pool, fn, *args):
    """Run ``fn(*args)`` while recording every hash passed to
    ``pool.get_cached_block``; returns ``(result, frozenset(hashes))``.

    Uses a temporary instance attribute, so the class-level lookup path pays
    nothing when no memo computation is in flight."""
    inner = pool.get_cached_block
    rec = []

    def get_cached_block(block_hash, kv_cache_group_ids):
        rec.append(block_hash)
        return inner(block_hash, kv_cache_group_ids)

    d = pool.__dict__
    d["get_cached_block"] = get_cached_block
    try:
        result = fn(*args)
    finally:
        d.pop("get_cached_block", None)
    return result, frozenset(rec)
