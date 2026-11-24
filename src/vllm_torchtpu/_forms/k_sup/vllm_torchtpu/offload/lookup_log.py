# SPDX-License-Identifier: Apache-2.0
"""Per-request prefix lookup line for the store-backed offload connector (VLLM_TPU_PREFIX_LOOKUP_LOG=1).

A hybrid prefix hit needs every attention block up to the hit and a recurrent-state checkpoint exactly at it, and
each level answers on its own: the HBM hit (``HybridKVCacheCoordinator.find_longest_cache_hit``) is the rightmost
cached checkpoint inside the contiguous HBM attention run, and the tier hit
(``OffloadingConnectorScheduler._lookup_complete_chunks``) is the rightmost tier checkpoint inside the attention run
the tier holds from the HBM hit on. The line records, at a request's first connector lookup, the two hits and what
each level held of the prompt's blocks, so a miss can be told apart as age (nothing left), a split chain (a hit
existed in HBM and the tier together) or a missing checkpoint.

Fields, in scheduler blocks of ``block`` tokens unless named ``*_tokens``:

- ``cap``: the longest hit the lookup allows, ``(num_tokens - 1) // block``.
- ``local_tokens`` / ``tier_tokens``: the hits the scheduler took (``tier_tokens`` is None for a deferred lookup).
- ``hbm_attn``: attention blocks cached in HBM, contiguous from block 0.
- ``hbm_ckpts``: boundaries up to ``cap`` with every state group cached in HBM (the last 8, ascending).
- ``tier_attn``: attention blocks in the tier contiguous from the HBM hit; ``tier_attn_any``: attention blocks the
  tier holds, counted inside the resident prefix of one probe over every group's keys in order. A store lookup answers
  only up to its first miss, so a chain whose block 0 is missing from the tier counts 0 here (and adds nothing to
  ``union_*``) even when the tier holds the rest of it; ``tier_runs`` shows such blocks.
- ``tier_ckpts``: boundaries with every state group in the tier (the last 8).
- ``union_attn``: blocks present in either level, contiguous from 0; ``union_ckpt``: the rightmost boundary inside
  it with a checkpoint in either level.
- ``junction_tokens``: the Marconi shared-prefix boundary the HBM lookup set.
- ``tier_class`` (tier classes only, ``VLLM_TPU_OFFLOAD_TIER_CLASSES``): per class, what that class alone holds. An
  attention class gives the end of its run of blocks from the HBM hit (``tier_attn`` is the shortest of them); a state
  class gives the boundaries where every one of its groups has a checkpoint (the last 8). A hit cut short because one
  class lacked its partner shows as classes that disagree.
- ``tier_runs`` (tier classes only): per attention class, the runs ``[start, end)`` of the prompt's blocks up to
  ``cap`` that every group of the class holds, each block probed on its own without pins (the last 8 runs). A split
  chain (head only in HBM, the rest and its checkpoint in the tier) shows a run that starts past block 0 while
  ``tier_class`` reads 0.

The HBM side is read in ``KVCacheManager.get_computed_blocks`` or, when the connector takes divergent hybrid hits
(VLLM_TPU_OFFLOAD_DIVERGENT_HITS), ``get_computed_blocks_for_connector`` (the scheduler's local lookup, which runs
just before the connector's) and kept on the request; the tier side is probed without pinning, so the line moves nothing in either
LRU. Host only.

The same flag writes one ``tier-restore`` line per host-tier load job, when the job ends (``RestoreLog``). A request's
load is one job; a request loaded again after preemption gets another job and another line. Times are ``time.time()``
seconds:

- ``submit``: the scheduler asks for the load (``prepare_load``: the request has its blocks and its host entries are
  pinned).
- ``start``: the store launches the transfer (``submit_load_job``, which issues the asynchronous ``load`` /
  ``read_remote``).
- ``end``: the first poll that reports the transfer landed; ``end_lo``: the poll before it, or ``start`` if later. The
  transfer landed in (``end_lo``, ``end``]; polls run once a schedule pass.
- ``keys``: offload keys loaded, every group; ``bytes``: ``keys`` times the all-ranks bytes of one offloaded block, the
  figure the host tier is sized with (with the tier classes, each key at its class's entry size).
- ``ok``: the job loaded every key (false: launch failure, lost pin, failed transfer).

It also writes one ``request-arrival`` line per request when the engine core's scheduler first sees it
(``Scheduler.add_request``, which puts it in the waiting queue):

- ``wall`` / ``mono``: ``time.time()`` and ``time.monotonic()`` seconds at that call.
- ``api_arrival``: the request's ``arrival_time``, the API server's ``time.time()`` before it renders and tokenizes the
  prompt (the same host clock as ``wall``). ``wall - api_arrival`` is the API server's part of the intake; the wait
  in the queue runs from ``wall``.
- ``prompt_tokens``: prompt length. The line has no cached count: nothing is looked up yet, since the HBM and tier
  lookups run when the scheduler first takes the request from the waiting queue, and the prefix-lookup line records
  them.
- ``waiting`` / ``running``: lengths of the scheduler's waiting and running queues before the request joins.

The engine core adds requests between steps, so ``wall`` trails the request's receipt on the engine's input socket by
up to one step. A streaming-input chunk for a request the scheduler already holds writes no line.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_prefix_lookup_log_patch"
_HBM_ATTR = "_tpu_hbm_lookup"
_LOGGED_ATTR = "_tpu_lookup_logged"
_LAST = 8
_clock = time.time
_monotonic = time.monotonic


def enabled() -> bool:
    return bool(envs.VLLM_TPU_PREFIX_LOOKUP_LOG)


def contiguous(present: Sequence[bool], start: int = 0) -> int:
    """End of the run of present blocks that starts at ``start`` (``start`` when block ``start`` is absent)."""
    end = start
    while end < len(present) and present[end]:
        end += 1
    return end


def rightmost(ckpt: Sequence[bool], lo: int, hi: int) -> int:
    """Rightmost boundary b in (lo, hi] whose checkpoint is present (``ckpt[b - 1]``), else 0."""
    for b in range(min(hi, len(ckpt)), lo, -1):
        if ckpt[b - 1]:
            return b
    return 0


def summarize(hbm_attn: Sequence[bool] | None, hbm_ckpt: Sequence[bool] | None, tier_attn: Sequence[bool],
              tier_ckpt: Sequence[bool], local_blocks: int, cap: int) -> dict:
    """The line's block fields from per-block presence (index k: block k, or the boundary after it)."""
    tier_attn = list(tier_attn[:cap]); tier_ckpt = list(tier_ckpt[:cap])
    out = {
        "tier_attn": contiguous(tier_attn, local_blocks),
        "tier_attn_any": sum(tier_attn),
        "tier_ckpts": [b + 1 for b, p in enumerate(tier_ckpt) if p][-_LAST:],
    }
    if hbm_attn is None or hbm_ckpt is None:
        out.update(hbm_attn=None, hbm_ckpts=None, union_attn=None, union_ckpt=None)
        return out
    hbm_attn = list(hbm_attn[:cap]); hbm_ckpt = list(hbm_ckpt[:cap])
    union = [a or t for a, t in zip(hbm_attn, tier_attn)]
    union_attn = contiguous(union)
    out.update(
        hbm_attn=contiguous(hbm_attn),
        hbm_ckpts=[b + 1 for b, p in enumerate(hbm_ckpt) if p][-_LAST:],
        union_attn=union_attn,
        union_ckpt=rightmost([a or t for a, t in zip(hbm_ckpt, tier_ckpt)], 0, union_attn),
    )
    return out


def hbm_presence(block_pool, block_hashes, attn_gids: list[int], state_gids: list[int], cap: int):
    """Per block up to ``cap``: every attention group cached, and every state group cached (the checkpoint at the
    boundary after the block). Uses the pool's own cached-block lookup, so it sees what the hit sees."""
    attn, ckpt = [], []
    for k in range(min(cap, len(block_hashes))):
        attn.append(block_pool.get_cached_block(block_hashes[k], attn_gids) is not None)
        ckpt.append(bool(state_gids) and block_pool.get_cached_block(block_hashes[k], state_gids) is not None)
    return attn, ckpt


def _group_kinds(kv_cache_config, block_pool, managers):
    """Prefix-cacheable attention and state group ids, or None when a group's block is not one hash block (the line
    reads presence per hash block)."""
    from vllm.v1.kv_cache_interface import MambaSpec

    attn, state = [], []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not spec.prefix_cacheable:
            continue
        if managers[gid].block_size != block_pool.hash_block_size:
            return None
        (state if isinstance(spec, MambaSpec) else attn).append(gid)
    return attn, state


def record_hbm(manager, request) -> None:
    """Keeps the HBM presence of the request's blocks on the request (first lookup only)."""
    if getattr(request, _HBM_ATTR, None) is not None or request.num_computed_tokens:
        return
    kinds = _group_kinds(manager.kv_cache_config, manager.block_pool, manager.coordinator.single_type_managers)
    if kinds is None:
        setattr(request, _HBM_ATTR, (None, None))
        return
    cap = max(request.num_tokens - 1, 0) // manager.block_pool.hash_block_size
    setattr(request, _HBM_ATTR, hbm_presence(manager.block_pool, request.block_hashes, kinds[0], kinds[1], cap))


def log_lookup(sched, request, local_tokens: int, tier_tokens: int | None) -> None:
    """Writes the line once per request, after the connector's lookup."""
    if getattr(request, _LOGGED_ATTR, False):
        return
    req_status = sched._req_status.get(request.request_id)
    if req_status is None:
        return
    from vllm.v1.kv_cache_interface import MambaSpec

    configs = sched.config.kv_group_configs
    block = configs[0].tokens_per_chunk
    if any(g.tokens_per_chunk != block for g in configs):
        return
    cap = max(request.num_tokens - 1, 0) // block
    attn_keys, state_keys = [], []
    for g, gs in zip(configs, req_status.group_states):
        keys = list(gs.offload_keys[:cap])
        (state_keys if isinstance(g.kv_cache_spec, MambaSpec) else attn_keys).append(keys)
    manager = sched._raiden_manager
    flat = [k for keys in attn_keys + state_keys for k in keys]
    found = {h for h, blk in manager._probe([s for k in flat for s in manager._sub_hashes(k)])
             if blk.status in (manager._block_status.HOST, manager._block_status.HOST_AND_HBM)}

    def present(group_keys):
        n = min((len(k) for k in group_keys), default=0)
        return [bool(group_keys) and all(all(s in found for s in manager._sub_hashes(keys[i])) for keys in group_keys)
                for i in range(n)]

    from vllm_torchtpu.offload.tier_classes import TieredOffloadingManager

    hbm = getattr(request, _HBM_ATTR, None) or (None, None)
    line = {"req_id": request.request_id, "num_tokens": request.num_tokens, "block": block, "cap": cap,
            "local_tokens": local_tokens, "tier_tokens": tier_tokens,
            "junction_tokens": getattr(request, "shared_prefix_boundary", 0)}
    line.update(summarize(hbm[0], hbm[1], present(attn_keys), present(state_keys), local_tokens // block, cap))
    if isinstance(manager, TieredOffloadingManager):
        keys_of = {g.group_idx: list(gs.offload_keys[:cap]) for g, gs in zip(configs, req_status.group_states)}
        line["tier_class"] = class_presence(manager, keys_of, local_tokens // block)
        line["tier_runs"] = class_runs(manager, keys_of)
    setattr(request, _LOGGED_ATTR, True)
    logger.info("prefix-lookup %s", json.dumps(line, separators=(",", ":")))


def class_presence(manager, keys_of: dict[int, list], local_blocks: int) -> dict:
    """Per tier class, what it holds of a request's keys (``keys_of``: group -> keys by block), probed without pins."""
    out: dict = {}
    for tier_class, class_manager in zip(manager.classes, manager.managers):
        groups = [keys_of[g] for g in tier_class.groups if g in keys_of]
        if not groups:
            continue
        if tier_class.per_checkpoint:
            n = min(len(keys) for keys in groups)
            boundaries = [
                b + 1 for b in range(n)
                if all(class_manager.host_resident_prefix([keys[b]]) == 1 for keys in groups)
            ]
            out[tier_class.name] = boundaries[-_LAST:]
        else:
            out[tier_class.name] = local_blocks + min(
                class_manager.host_resident_prefix(keys[local_blocks:]) for keys in groups
            )
    return out


def class_runs(manager, keys_of: dict[int, list]) -> dict:
    """Per attention class, the runs ``[start, end)`` of a request's blocks that every group of the class holds, each
    block probed on its own (``host_resident_prefix`` of one key: no pins, no LRU move), the last ``_LAST``."""
    out: dict = {}
    for tier_class, class_manager in zip(manager.classes, manager.managers):
        if tier_class.per_checkpoint:
            continue
        groups = [keys_of[g] for g in tier_class.groups if g in keys_of]
        if not groups:
            continue
        runs: list[list[int]] = []
        start = None
        n = min(len(keys) for keys in groups)
        for b in range(n):
            held = all(class_manager.host_resident_prefix([keys[b]]) == 1 for keys in groups)
            if held and start is None:
                start = b
            elif not held and start is not None:
                runs.append([start, b])
                start = None
        if start is not None:
            runs.append([start, n])
        out[tier_class.name] = runs[-_LAST:]
    return out


def patch_prefix_lookup_log() -> None:
    """Wraps KVCacheManager.get_computed_blocks to record the HBM side and Scheduler.add_request to write the
    arrival line. Inert unless the flag is set."""
    if not enabled():
        return
    _patch_request_arrival()
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    # Both local lookups: get_computed_blocks_for_connector is the one the scheduler calls when the connector takes
    # divergent hybrid hits; record_hbm keeps the first record, so a fallback to get_computed_blocks adds nothing.
    for name in ("get_computed_blocks", "get_computed_blocks_for_connector"):
        original = getattr(KVCacheManager, name, None)
        if original is None or getattr(original, _PATCH_FLAG, False):
            continue

        def wrapped(self, request, _original=original):
            out = _original(self, request)
            try:
                record_hbm(self, request)
            except Exception:  # a diagnostic must never fail a schedule pass
                logger.exception("prefix-lookup: HBM presence failed for %s", request.request_id)
            return out

        setattr(wrapped, _PATCH_FLAG, True)
        setattr(KVCacheManager, name, wrapped)
    logger.info("Applied TPU patch: per-request prefix lookup line (VLLM_TPU_PREFIX_LOOKUP_LOG).")


def arrival_fields(scheduler, request) -> dict:
    """The request-arrival line's fields, read before the scheduler adds the request."""
    return {"req_id": request.request_id, "wall": round(_clock(), 4), "mono": round(_monotonic(), 4),
            "api_arrival": round(request.arrival_time, 4), "prompt_tokens": request.num_prompt_tokens,
            "waiting": len(scheduler.waiting), "running": len(scheduler.running)}


def _patch_request_arrival() -> None:
    """Wraps Scheduler.add_request, which every scheduler class on this tree inherits, to write the arrival line."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.add_request
    if getattr(original, _PATCH_FLAG, False):
        return

    def add_request(self, request):
        line = None
        if request.request_id not in self.requests:
            try:
                line = arrival_fields(self, request)
            except Exception:  # a diagnostic must never fail a request
                logger.exception("request-arrival: line failed for %s", request.request_id)
        original(self, request)
        if line is not None:
            logger.info("request-arrival %s", json.dumps(line, separators=(",", ":")))

    setattr(add_request, _PATCH_FLAG, True)
    Scheduler.add_request = add_request
    logger.info("Applied TPU patch: per-request arrival line (VLLM_TPU_PREFIX_LOOKUP_LOG).")


class RestoreLog:
    """Writes one ``tier-restore`` line per host-tier load job of a ``RaidenOffloadingManager``.

    Installed on the manager instance only when the flag is set, so the class methods run unchanged with it off. The
    wrappers read the clock and keep one tuple per open job; the calls they wrap get the same arguments and return the
    same values."""

    def __init__(self, manager, bytes_per_key):
        # An int (every key the same size) or a function of a job's keys.
        self.bytes_per_key = bytes_per_key
        self.asked: dict[str, float] = {}
        self.jobs: dict[int, tuple[str, int, int, float, float]] = {}
        self.last_poll = 0.0
        from vllm_torchtpu.offload.raiden_store import AdmissionOp

        self._load_op = AdmissionOp.LOAD
        prepare, submit = manager.prepare_load, manager.submit_load_job
        poll, drain = manager.poll_finished_jobs, manager.drain_jobs

        def prepare_load(keys, req_context):
            self.asked[req_context.req_id] = _clock()
            return prepare(keys, req_context)

        def submit_load_job(job_id, keys, device_block_ids, req_id, pinned=True):
            start = _clock()
            per_key = self.bytes_per_key
            nbytes = per_key(keys) if callable(per_key) else len(keys) * per_key
            self.jobs[job_id] = (req_id, len(keys), nbytes, self.asked.pop(req_id, start), start)
            return submit(job_id, keys, device_block_ids, req_id, pinned=pinned)

        def poll_finished_jobs():
            finished = poll()
            self.ended(finished, _clock(), poll=True)
            return finished

        def drain_jobs(job_ids):
            # A drain's own polls also finalize other jobs, which the next poll_finished_jobs returns; only that call
            # moves last_poll, so their (end_lo, end] still holds the landing.
            finished = drain(job_ids)
            self.ended(finished, _clock(), poll=False)
            return finished

        manager.prepare_load = prepare_load
        manager.submit_load_job = submit_load_job
        manager.poll_finished_jobs = poll_finished_jobs
        manager.drain_jobs = drain_jobs

    def ended(self, finished, now: float, poll: bool) -> None:
        for job in finished:
            rec = self.jobs.pop(job.job_id, None) if job.op is self._load_op else None
            if rec is None:
                continue
            req_id, keys, nbytes, submit, start = rec
            end_lo = max(self.last_poll, start)
            line = {"req_id": req_id, "job": job.job_id, "keys": keys, "bytes": nbytes,
                    "submit": round(submit, 4), "start": round(start, 4), "end_lo": round(end_lo, 4),
                    "end": round(now, 4), "ok": job.success}
            try:
                logger.info("tier-restore %s", json.dumps(line, separators=(",", ":")))
            except Exception:  # a diagnostic must never fail a schedule pass
                logger.exception("tier-restore: line failed for job %s", job.job_id)
        if poll:
            self.last_poll = now


def install_restore_log(manager, bytes_per_key) -> RestoreLog | None:
    """Times the manager's load jobs when the flag is set; returns None and touches nothing otherwise.

    ``bytes_per_key``: an int, or a function giving the bytes of a job's keys."""
    if not enabled():
        return None
    logger.info(
        "prefix-lookup: tier-restore line on (%s)",
        "bytes by tier class" if callable(bytes_per_key) else f"{bytes_per_key} bytes a key",
    )
    return RestoreLog(manager, bytes_per_key)
