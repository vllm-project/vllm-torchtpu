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
"""Single-step warm prefill for hybrid attention + GDN ("align") models.

``TPU_SCHED_SINGLE_STEP_PREFILL=1`` (default on). With ``--mamba-cache-mode
align`` a warm turn (~94k cached + ~2k new tokens) is split into ~3 prefill
steps because every Mamba state that may be cached -- the last full block
``f640`` and the 128-token partial tail ``f128`` -- is only written at a
chunk end. Here one step writes up to two of them inside the chunk instead
(GDN v3 "virtual segments", ``fused_conv1d_gdn(seg_bounds=...)``):

* Scheduler (``plan_single_step_split``): drop every cacheable stop that one
  chunk can checkpoint internally -- the full block ending where the chunk's
  last (running) state block starts, and one partial tail inside that block
  -- and record them as cuts. Any other cacheable stop still ends the chunk.
* KV manager (``MambaManager`` patches): allocate a real block for the full
  cut's column (upstream's internal-checkpoint block, generalized to chunks
  that start mid-block), register the partial tail on the request's first
  speculative block (checkpoint 1, which a prefill never writes) and move it
  to a private copy before that block is next written. A cut is
  *confirmed* only once its block exists; only confirmed cuts reach workers.
* Worker: ``TPUModelRunner`` turns confirmed cuts into ``seg_bounds`` /
  per-group ``seg_state_indices`` (``build_seg_arrays``).

Invariant: a Mamba block is hashed (by upstream ``cache_blocks``) only if its
state is written in the same step: full cuts are always reachable boundaries
whose column holds the confirmed checkpoint block, and the planner never lets
a chunk cross any other cacheable boundary.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger("vllm." + __name__)

_PATCH_FLAG = "_tpu_single_step_prefill_patch"


def enabled() -> bool:
    return bool(envs.TPU_SCHED_SINGLE_STEP_PREFILL)


# --------------------------------------------------------------- scheduler
def mamba_managers(kv_cache_manager: Any) -> list[Any]:
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    managers = getattr(coordinator, "single_type_managers", None) or ()
    return [m for m in managers if getattr(m, "mamba_cache_mode", None) == "align"]


def supported(scheduler: Any) -> tuple[bool, str]:
    """Whether the single-step split may be used by ``scheduler``."""
    if not getattr(scheduler, "need_mamba_block_aligned_split", False):
        return False, "mamba_cache_mode != align"
    if getattr(scheduler, "mamba_has_prefill_checkpoint_blocks", False):
        return False, "upstream prefill checkpoint blocks are active"
    kvm = getattr(scheduler, "kv_cache_manager", None)
    if not getattr(kvm, "enable_caching", False):
        return False, "prefix caching is off"
    if getattr(getattr(kvm, "coordinator", None), "retention_interval", None) != 0:
        return False, "prefix_cache_retention_interval != 0"
    if getattr(scheduler, "use_eagle", False) and getattr(
        scheduler, "use_eagle_block_drop", True
    ):
        return False, "EAGLE block drop is on (VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=0)"
    cache_bs = getattr(getattr(scheduler, "cache_config", None), "block_size", None)
    if cache_bs != getattr(scheduler, "block_size", None):
        return False, "scheduler and cache block sizes differ (PCP)"
    managers = mamba_managers(kvm)
    if not managers or any(m.block_size != scheduler.block_size for m in managers):
        return False, "no align Mamba group at the scheduler block size"
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is not None:
        # Same VllmConfig-only predicates as TPUModelRunner._mamba_seg_enabled:
        # a worker that cannot write in-chunk checkpoints raises on the first
        # cut, so refuse here instead.
        from vllm_torchtpu.block_major_pool import unified_block_major_enabled
        from vllm_torchtpu.gdn_pool_layout import QWEN_GDN_ARCHITECTURES
        from vllm_torchtpu.platforms.tpu_block_size_utils import (
            unified_kv_layout_enabled,
        )

        if not unified_kv_layout_enabled(vllm_config):
            return False, "the unified KV block pool is off"
        if unified_block_major_enabled(vllm_config):
            return False, "the block-major unified pool (VLLM_TPU_BLOCK_MAJOR_KV)"
        # Only the GDN pooled op takes seg_bounds; e.g. Kimi's pooled KDA op
        # would silently skip the checkpoint writes behind hashed blocks.
        arch = vllm_config.model_config.architecture
        if arch not in QWEN_GDN_ARCHITECTURES:
            return False, f"{arch} does not run the GDN pooled op"
    return True, ""


def needed_checkpoints(
    scheduler: Any, request: Any, start: int, retention: Any | None
) -> set[int]:
    """Cacheable state positions in ``(start, prefill_end)`` that the stock
    split would stop at: upstream's last cacheable block and partial tail,
    the shared-prefix junction, and the retention stops (smart-fork pre-final
    block, safe-tail full block and fine tail). The mid-block boundary after
    a partial hit is not one: its state is never hashed under sparse
    retention."""
    bs = scheduler.block_size
    prompt = request.num_prompt_tokens
    num_tokens = request.num_tokens
    prefill_end = max(prompt, num_tokens - 1)
    out: set[int] = {num_tokens - num_tokens % bs}
    if getattr(scheduler, "mamba_partial_cache_hit", False):
        hb = scheduler.hash_block_size
        tail = max(prompt - 1, 0) // hb * hb
        last = num_tokens - num_tokens % bs
        if last < tail < prompt:
            # Under the retention safe-tail rule the raw tail is never cached
            # (HybridStateRetention._install_mamba_manager_hooks).
            uncached = False
            if retention is not None and retention.enabled:
                eff = retention._effective_prompt_tokens(prompt)
                fine = (
                    eff // retention._alignment_tokens() * retention._alignment_tokens()
                )
                uncached = eff < prompt and tail != fine
            if not uncached:
                out.add(tail)
    spb = int(getattr(request, "shared_prefix_boundary", 0) or 0)
    if spb:
        out.add(spb // bs * bs)
    if retention is not None:
        out.update(retention.prefill_checkpoint_stops(request, start))
    return {p for p in out if start < p < prefill_end}


def plan_single_step_split(
    scheduler: Any,
    request: Any,
    start: int,
    num_new_tokens: int,
    retention: Any | None,
) -> tuple[int, tuple[int, int] | None]:
    """Returns (tokens to schedule, (full_cut, partial_cut) or None).

    Cuts are absolute token positions (0 = none). ``num_new_tokens`` is the
    budget- and cap-limited chunk length before Mamba alignment.
    """
    prompt = request.num_prompt_tokens
    prefill_end = max(prompt, request.num_tokens - 1)
    if start >= prefill_end or num_new_tokens <= 0:
        return num_new_tokens, None
    bs = scheduler.block_size
    end = start + num_new_tokens
    if end < prefill_end:
        # Same rule as upstream: non-final chunks end block aligned.
        max_prefill = scheduler.max_num_scheduled_tokens
        threshold = scheduler.scheduler_config.long_prefill_token_threshold
        if threshold > 0:
            max_prefill = min(max_prefill, threshold)
        aligned = end // bs * bs
        if aligned > start or bs <= max_prefill:
            end = aligned
        if end <= start:
            return max(end - start, 0), None
    needed = needed_checkpoints(scheduler, request, start, retention)
    num_spec_blocks = min(
        (m.num_speculative_blocks for m in mamba_managers(scheduler.kv_cache_manager)),
        default=0,
    )
    full_cut = partial_cut = 0
    while True:
        run_start = (end - 1) // bs * bs
        full_cut = run_start if (run_start in needed and start < run_start) else 0
        partials = sorted(
            p for p in needed if p % bs and max(run_start, start) < p < end
        )
        partial_cut = partials[0] if partials and num_spec_blocks >= 1 else 0
        uncovered = [
            p for p in needed if start < p < end and p not in (full_cut, partial_cut)
        ]
        if not uncovered:
            break
        end = min(uncovered)
    if full_cut or partial_cut:
        return end - start, (full_cut, partial_cut)
    return end - start, None


def set_plan(kv_cache_manager: Any, request_id: str, plan: tuple | None) -> None:
    for mgr in mamba_managers(kv_cache_manager):
        plans = mgr.__dict__.setdefault("_tpu_seg_plan", {})
        if plan is None:
            plans.pop(request_id, None)
        else:
            plans[request_id] = plan


def begin_step(kv_cache_manager: Any) -> None:
    for mgr in mamba_managers(kv_cache_manager):
        mgr.__dict__["_tpu_seg_plan"] = {}
        mgr.__dict__["_tpu_seg_confirmed"] = {}


def confirmed_cuts(kv_cache_manager: Any, num_scheduled_tokens: dict) -> dict:
    """Cuts every Mamba group confirmed for a request scheduled this step."""
    managers = mamba_managers(kv_cache_manager)
    if not managers:
        return {}
    per_group = [mgr.__dict__.get("_tpu_seg_confirmed", {}) for mgr in managers]
    out = {}
    for rid, cuts in per_group[0].items():
        if rid not in num_scheduled_tokens:
            continue
        if all(g.get(rid) == cuts for g in per_group[1:]):
            out[rid] = cuts
        else:
            logger.warning(
                "single-step prefill: Mamba groups disagree on cuts for %s", rid
            )
    return out


# -------------------------------------------------------------- KV manager
def install_mamba_manager_patches() -> None:
    from vllm.v1.core import single_type_kv_cache_manager as _stm
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    _ckpt_pos_fn = getattr(_stm, "get_mamba_prefill_checkpoint_position", None)

    if getattr(MambaManager.cache_blocks, _PATCH_FLAG, False):
        return
    orig_needs_ckpt = MambaManager._needs_internal_checkpoint
    orig_cache_partial = MambaManager._cache_partial_tail_block
    orig_cache_blocks = MambaManager.cache_blocks
    orig_allocate = MambaManager.allocate_new_blocks
    orig_pop = MambaManager.pop_blocks_for_free

    orig_relocate = MambaManager._relocate_speculative_block

    def _plan(self, request_id):
        return self.__dict__.get("_tpu_seg_plan", {}).get(request_id)

    def _ckpt_slot_kind(self, request_id, ckpt_idx):
        """How the full cut's column gets a real block, or None if it holds
        one already (the previous state block, or the CoW'd hit block):
        "null" -> upstream's internal-checkpoint block lands there;
        "spec" -> it holds a speculative scratch block that this allocation
        would relocate (the chunk starts at a block boundary), keep it in
        place as the checkpoint and let upstream append one more block."""
        blocks = self.req_to_blocks[request_id]
        if ckpt_idx >= len(blocks) or blocks[ckpt_idx].is_null:
            return "null"
        if (
            request_id in self._allocated_block_reqs
            and ckpt_idx >= len(blocks) - self.num_speculative_blocks
            and self.block_pool.is_block_writable(blocks[ckpt_idx])
        ):
            return "spec"
        return None

    @wraps(orig_relocate)
    def _relocate_speculative_block(self, req_blocks, block_idx):
        keep = self.__dict__.get("_tpu_keep_spec")
        if keep is not None and keep[0] is req_blocks and keep[1] == block_idx:
            return
        return orig_relocate(self, req_blocks, block_idx)

    @wraps(orig_needs_ckpt)
    def _needs_internal_checkpoint(self, request_id, *args):
        # v0.32+: (request_id, query_start, query_end, checkpoint_position, prefill_end)
        # v0.29:  (request_id, num_tokens, num_computed_tokens)
        num_tokens = args[1] if len(args) >= 4 else args[0]
        plan = _plan(self, request_id)
        if plan is not None and plan[1]:
            full_cut = plan[1]
            ckpt_idx = full_cut // self.block_size - 1
            run_start = (num_tokens - 1) // self.block_size * self.block_size
            if plan[0] < full_cut == run_start < num_tokens:
                return _ckpt_slot_kind(self, request_id, ckpt_idx) is not None
            return False
        return orig_needs_ckpt(self, request_id, *args)

    @wraps(orig_cache_partial)
    def _cache_partial_tail_block(
        self, request, num_tokens, retention_interval=None, *args, **kwargs
    ):
        override = self.__dict__.get("_tpu_seg_partial_override")
        if override is None or override[0] != request.request_id:
            return orig_cache_partial(
                self, request, num_tokens, retention_interval, *args, **kwargs
            )
        # Same checks as upstream, but the tail state lives in the request's
        # first speculative block instead of block num_tokens // block_size.
        block_idx = override[1]
        hash_block_size = self.block_pool.hash_block_size
        if callable(_ckpt_pos_fn):
            latest_prompt_hash_boundary = _ckpt_pos_fn(
                request.num_prompt_tokens,
                hash_block_size,
                getattr(self, "drop_eagle_checkpoint_block", False),
            )
        else:
            latest_prompt_hash_boundary = (
                max(request.num_prompt_tokens - 1, 0)
                // hash_block_size
                * hash_block_size
            )
        if (
            self.block_size == hash_block_size
            or num_tokens <= 0
            or num_tokens % self.block_size == 0
            or num_tokens % hash_block_size != 0
            or num_tokens != latest_prompt_hash_boundary
        ):
            return None
        blocks = self.req_to_blocks[request.request_id]
        if block_idx >= len(blocks) or blocks[block_idx].is_null:
            return None
        source_block = blocks[block_idx]
        partial_hash = self.block_pool.cache_partial_block(
            request=request,
            block=source_block,
            num_tokens=num_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )
        if partial_hash is not None:
            self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
            self.num_cached_block[request.request_id] = num_tokens // self.block_size
            self._producer_partial_tail_reqs[request.request_id] = (
                source_block,
                num_tokens,
            )
            self.__dict__.setdefault("_tpu_spec_partial", set()).add(request.request_id)
        return partial_hash

    @wraps(orig_cache_blocks)
    def cache_blocks(self, request, num_tokens, retention_interval=None, **kwargs):
        plan = _plan(self, request.request_id)
        if plan is not None and plan[1]:
            # Prevent v0.32's _cache_partial_tail_block from overwriting the
            # full-cut checkpoint block's hash with checkpoint_position.
            getattr(self, "_checkpoints", {}).pop(request.request_id, None)
        orig_cache_blocks(
            self,
            request,
            num_tokens,
            retention_interval=retention_interval,
            **kwargs,
        )
        if plan is None:
            return
        rid = request.request_id
        start, full_cut, partial_cut = plan
        blocks = self.req_to_blocks[rid]
        bs = self.block_size
        run_col = (num_tokens - 1) // bs
        ok_full = 0
        if full_cut and start < full_cut == run_col * bs < num_tokens:
            idx = full_cut // bs - 1
            if idx < len(blocks) and not blocks[idx].is_null:
                ok_full = full_cut
        ok_partial = 0
        spec_idx = run_col + 1
        if (
            partial_cut
            and self.num_speculative_blocks >= 1
            and max(start, run_col * bs) < partial_cut < num_tokens
            and partial_cut // bs == run_col
            and spec_idx < len(blocks)
            and not blocks[spec_idx].is_null
            and blocks[spec_idx].block_hash is None
            and rid not in self._partial_hit_reqs
        ):
            self.__dict__["_tpu_seg_partial_override"] = (rid, spec_idx)
            try:
                # Through the instance attribute so HybridStateRetention's
                # hook sees the real boundary (safe-tail prompt, tracking).
                partial_hash = self._cache_partial_tail_block(
                    request, partial_cut, retention_interval=retention_interval
                )
            finally:
                self.__dict__["_tpu_seg_partial_override"] = None
            if partial_hash is not None:
                self.cached_blocks_this_step.add(partial_hash)
                ok_partial = partial_cut
        if ok_full or ok_partial:
            self.__dict__.setdefault("_tpu_seg_confirmed", {})[rid] = (
                ok_full,
                ok_partial,
            )

    @wraps(orig_allocate)
    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        spec_partial = self.__dict__.get("_tpu_spec_partial")
        if (
            spec_partial
            and request_id in spec_partial
            and request_id in self._partial_hit_reqs
            and request_id in self._allocated_block_reqs
        ):
            # The partial tail lives in a speculative block, which the
            # request writes on its next step (and upstream may relocate it
            # first). Move the cache entry to a private copy now, exactly as
            # upstream does for a running request's partial tail.
            spec_partial.discard(request_id)
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            assert self.req_to_blocks[request_id][block_idx] is source_block
            (cow_block,) = self.block_pool.get_new_blocks(1)
            self.block_pool.move_block_hashes(source_block, cow_block)
            self._pending_cow_copies.append((source_block, cow_block))
            source_block.ref_cnt += 1
            producer_tail = self._producer_partial_tail_reqs.pop(request_id, None)
            if producer_tail is not None:
                boundary_tokens = (
                    producer_tail[1]
                    if isinstance(producer_tail, tuple)
                    else producer_tail
                )
                self._pending_boundary_state_offloads.append(
                    (request_id, self.kv_cache_group_id, cow_block, boundary_tokens)
                )
            if cow_block.block_hash is not None:
                self.cached_blocks_this_step.add(cow_block.block_hash)
        keep = None
        plan = _plan(self, request_id)
        has_ckpt = request_id in getattr(self, "_checkpoints", ()) or bool(
            getattr(self, "_num_checkpoint_blocks", {}).get(request_id, 0)
        )
        if (
            plan is not None
            and plan[1]
            and has_ckpt
            and _ckpt_slot_kind(self, request_id, plan[1] // self.block_size - 1)
            == "spec"
        ):
            keep = (self.req_to_blocks[request_id], plan[1] // self.block_size - 1)
        self.__dict__["_tpu_keep_spec"] = keep
        try:
            return orig_allocate(self, request_id, num_tokens, num_tokens_main_model)
        finally:
            self.__dict__["_tpu_keep_spec"] = None

    @wraps(orig_pop)
    def pop_blocks_for_free(self, request_id):
        spec_partial = self.__dict__.get("_tpu_spec_partial")
        if spec_partial:
            spec_partial.discard(request_id)
        return orig_pop(self, request_id)

    for fn in (
        _relocate_speculative_block,
        _needs_internal_checkpoint,
        _cache_partial_tail_block,
        cache_blocks,
        allocate_new_blocks,
        pop_blocks_for_free,
    ):
        setattr(fn, _PATCH_FLAG, True)
    MambaManager._relocate_speculative_block = _relocate_speculative_block
    MambaManager._needs_internal_checkpoint = _needs_internal_checkpoint
    MambaManager._cache_partial_tail_block = _cache_partial_tail_block
    MambaManager.cache_blocks = cache_blocks
    MambaManager.allocate_new_blocks = allocate_new_blocks
    MambaManager.pop_blocks_for_free = pop_blocks_for_free
    logger.info("TPU_SCHED_SINGLE_STEP_PREFILL: MambaManager patches installed.")


# ------------------------------------------------------------------ worker
def build_seg_arrays(
    cuts: dict,
    req_ids: list,
    start_index: int,
    num_reqs: int,
    target_num_reqs: int,
    num_computed: Any,
    num_scheduled_tokens: dict,
    block_tables: dict,
    block_size: int,
):
    """Host-side ``(seg_bounds, {gid: seg_state_indices})`` for one chunk.

    ``block_tables`` maps mamba kv_cache_group_id -> CPU block table
    ``[num_rows, num_cols]``. Returns (None, None) if no request in the
    chunk has a cut. A cut whose destination block is the null block is
    dropped (the KV manager hashed nothing there).
    """
    import numpy as np

    rows = []
    for i in range(num_reqs):
        rid = req_ids[start_index + i]
        cut = cuts.get(rid) if rid is not None else None
        if cut:
            rows.append((i, rid, cut))
    if not rows:
        return None, None
    bounds = np.zeros((target_num_reqs, 2), np.int32)
    per_gid = {gid: np.zeros((target_num_reqs, 2), np.int32) for gid in block_tables}
    for i, rid, (full_cut, partial_cut) in rows:
        computed = int(num_computed[i])
        end = computed + int(num_scheduled_tokens[rid])
        run_col = (end - 1) // block_size
        row = start_index + i
        for k, (cut, col) in enumerate(
            ((full_cut, full_cut // block_size - 1), (partial_cut, run_col + 1))
        ):
            if not (cut and computed < cut < end):
                continue
            ids = [
                int(bt[row, col]) if col < bt.shape[1] else 0
                for bt in block_tables.values()
            ]
            if any(b == 0 for b in ids):
                logger.warning(
                    "single-step prefill: %s cut %d has no block; dropped", rid, cut
                )
                continue
            bounds[i, k] = cut - computed
            for gid, b in zip(block_tables, ids):
                per_gid[gid][i, k] = b
    return bounds, per_gid
