# SPDX-License-Identifier: Apache-2.0
"""Fold a prefill's tail into its last Mamba-aligned chunk (env-gated).

In Mamba "align" mode a reusable GDN / short-conv state exists only at block
boundaries, and the state is written at the end of a step. So the pinned
``Scheduler._mamba_block_aligned_split`` ends a request's last prefill chunk
at its last block boundary (``(num_tokens - 1) // B * B``) and runs the rest,
up to ``B`` tokens, as one more prefill step of its own. That step carries
the full fixed cost of a long-context prefill step and stalls every decode
in the batch.

The pinned vLLM release (0.32) can instead export that boundary state from
inside the step: a ``MambaSpec`` with ``num_prefill_checkpoint_blocks=1``
makes the scheduler keep the tail in the chunk (``use_internal_checkpoint``)
and makes ``MambaManager`` reserve one block at column ``cdiv(end, B) - 2``,
which it hashes at ``(end - 1) // B * B`` after the step. The backend must
then write the state at that position into that block. This module is the
TPU side of that contract for Qwen4Exp's unified pool:

- ``with_prefill_checkpoint`` sets the spec fields on the pooled Mamba specs
  (GDN and PLE groups, ``models/vllm/qwen4_exp/pool_layout.py``).
- ``prefill_checkpoints`` is the worker-side mirror of the manager's
  decision: per request, the checkpoint's offset in the step's query and its
  block-table column. The runner uploads it with the step metadata; the GDN
  kernel writes the state after the tile that ends at the offset (the
  offset is a whole number of blocks, so a tile boundary), and the PLE layer
  writes the short-conv history before it.
- ``patch_mamba_tail_fold`` (engine core) checks that the scheduler and the
  managers use the geometry the worker assumes, and turns the internal
  checkpoint off if not. It also counts the tails it folds:
  ``VLLM_TPU_MAMBA_TAIL_FOLD: folded N tails / M splits``.

The worker mirror ignores the manager's "the chunk ends the prefill" test,
which needs the request's prompt length. It may then also write a state
into a column the manager did not reserve; that column is null or holds one
of the request's own speculative scratch blocks, which the next step does
not read (a step after a prefill chunk resumes from the state block). What
it must never do is miss a reserved one, so it errs on that side.

``VLLM_TPU_MAMBA_TAIL_FOLD=1`` enables it; off, nothing here is installed and
the specs are unchanged.
"""

import time
from dataclasses import replace
from functools import wraps

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_mamba_tail_fold_patch"
_LOG_INTERVAL_S = 60.0


def gdn_max_prefill_tile() -> int:
    """Largest GDN PER_SEQ prefill tile (rows) the kernel can pick.

    Dynamic tiling picks at most 128 rows (``tiling.MIXED_TILE_SIZES``), the
    static default is 64, and ``TPU_GDN_CHUNK_PAIR`` doubles the tile.
    """
    return 256 if envs.TPU_GDN_CHUNK_PAIR else 128


def fold_unit(block_size: int) -> int:
    """Checkpoint / prefix-hit granularity of the fold for a Mamba block.

    ``VLLM_TPU_MAMBA_TAIL_FOLD_UNIT`` (0: the block itself, the original
    geometry). A finer unit must divide the block, so a hit still maps to one
    state block, and be a multiple of the largest GDN prefill tile, so the
    checkpoint offset always ends a tile and the kernel needs no change.
    """
    unit = envs.VLLM_TPU_MAMBA_TAIL_FOLD_UNIT
    if unit <= 0 or unit == block_size:
        return block_size
    tile = gdn_max_prefill_tile()
    if block_size % unit or unit % tile:
        raise ValueError(
            f"VLLM_TPU_MAMBA_TAIL_FOLD_UNIT={unit} must divide the Mamba block "
            f"({block_size}) and be a multiple of the largest GDN prefill tile "
            f"({tile}{', TPU_GDN_CHUNK_PAIR=1' if tile == 256 else ''})."
        )
    return unit


_DCP_LOGGED: set[str] = set()


def fold_blocked_by_context_parallel(cfg, main_kv_dcp: int) -> str | None:
    """Why the fold must stay off for this deployment (None: allowed).

    Any context parallelism turns it off: vLLM DCP / PCP, or the Qwen4Exp
    main-KV DCP (``VLLM_TPU_QSA_MAIN_KV_DCP``). Under main-KV DCP the
    scheduler block is ``dcp x`` the Mamba block, so the scheduler guard
    rejects the geometry, but the worker still saw checkpoint-carrying specs
    and ran the fold's metadata path (za5 c48 j-5378b21d). Leaving the
    checkpoint off the specs keeps both sides off consistently. Logged once
    per reason and process.
    """
    p = getattr(cfg, "parallel_config", None)
    reason = None
    if main_kv_dcp > 1:
        reason = f"VLLM_TPU_QSA_MAIN_KV_DCP={main_kv_dcp}"
    elif int(getattr(p, "decode_context_parallel_size", 1) or 1) > 1:
        reason = f"decode_context_parallel_size={p.decode_context_parallel_size}"
    elif int(getattr(p, "prefill_context_parallel_size", 1) or 1) > 1:
        reason = f"prefill_context_parallel_size={p.prefill_context_parallel_size}"
    if reason is not None and reason not in _DCP_LOGGED:
        _DCP_LOGGED.add(reason)
        logger.warning(
            "VLLM_TPU_MAMBA_TAIL_FOLD: disabled, %s (context parallelism); the "
            "Mamba specs carry no prefill checkpoint block and prefills keep "
            "the block-aligned tail split.",
            reason,
        )
    return reason


def with_prefill_checkpoint(spec):
    """``spec`` with one internal prefill checkpoint block per request.

    The alignment is the fold unit (``fold_unit``; the spec's block size by
    default). The scheduler only keeps a tail when the chunk starts on a hash
    block, and the guard requires the hash block to equal the unit, so the
    checkpoint offset is always a whole number of units. That makes it a
    multiple of every GDN prefill tile (at most 256 rows).
    """
    return replace(
        spec,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=fold_unit(spec.block_size),
    )


def prefill_checkpoints(
    num_computed: np.ndarray,
    num_scheduled: np.ndarray,
    block_size: int,
    hash_block_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per request: (offset in the step's query, block-table column).

    Mirrors ``MambaManager._needs_internal_checkpoint`` and
    ``get_mamba_prefill_checkpoint_position`` with the hash block, the cache
    hit alignment and the checkpoint alignment all equal to
    ``hash_block_size`` (the fold unit; default ``block_size``), Mamba block
    ``block_size`` and no EAGLE block drop (``patch_mamba_tail_fold`` turns
    the fold off for any other geometry). Offset 0 means no checkpoint.
    """
    unit = block_size if hash_block_size is None else hash_block_size
    start = num_computed.astype(np.int64)
    end = start + num_scheduled.astype(np.int64)
    position = np.maximum((end - 1) // unit * unit, 0)
    column = -(-end // block_size) - 2
    initial_column = (start - 1) // block_size
    valid = (
        (num_scheduled > 0)
        & (start % unit == 0)
        & (column > initial_column)
        & (start + unit <= position)
        & (start < position)
        & (position < end)
        & (column >= 0)
    )
    offsets = np.where(valid, position - start, 0).astype(np.int32)
    columns = np.where(valid, column, 0).astype(np.int32)
    return offsets, columns


def prefill_checkpoint_targets(
    prefill_ckpt: torch.Tensor, block_tables: torch.Tensor, num_reqs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """In the compiled forward: per request (offset, destination block), int32.

    ``prefill_ckpt`` is the step's ``(2, num_reqs)`` offsets and columns
    (``AttentionMetadata.mamba_prefill_ckpt``); ``block_tables`` the group's
    flat ``(num_reqs * width,)`` table. Each Mamba group owns its checkpoint
    block, so each layer reads it from its own group's table. Rows without
    a checkpoint get block 0, which the writers skip.
    """
    offsets = prefill_ckpt[0].to(torch.int32)
    table = block_tables.reshape(num_reqs, -1)
    columns = prefill_ckpt[1].to(torch.int64).clamp(0, table.shape[1] - 1)
    blocks = torch.gather(table, 1, columns[:, None]).squeeze(1).to(torch.int32)
    blocks = torch.where(offsets > 0, blocks, torch.zeros_like(blocks))
    return offsets, blocks


class _Counter:
    def __init__(self, label: str, what: str) -> None:
        self.label = label
        self.what = what
        self.hits = 0
        self.total = 0
        self.logged = 0
        self.last_log = 0.0

    def add(self, hits: int, total: int) -> None:
        if not total:
            return
        self.hits += hits
        self.total += total
        now = time.monotonic()
        if self.hits and (not self.logged or now - self.last_log >= _LOG_INTERVAL_S):
            self.logged += 1
            self.last_log = now
            logger.info(
                "VLLM_TPU_MAMBA_TAIL_FOLD: %s %d %s / %d %s",
                self.label,
                self.hits,
                self.what[0],
                self.total,
                self.what[1],
            )


SCHEDULER_COUNTER = _Counter("folded", ("tails", "splits"))
WORKER_COUNTER = _Counter("exported", ("checkpoints", "prefill rows"))


def _geometry_mismatch(scheduler) -> str | None:
    """Why the scheduler/manager geometry differs from the worker mirror."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    if getattr(scheduler, "use_eagle_block_drop", False):
        return "EAGLE block drop is on (set VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1)"
    managers = [
        m
        for m in scheduler.kv_cache_manager.coordinator.single_type_managers
        if isinstance(m, MambaManager)
    ]
    if not managers:
        return "no Mamba groups"
    sizes = {m.block_size for m in managers}
    if len(sizes) != 1:
        return f"Mamba groups with different block sizes {sorted(sizes)}"
    (block,) = sizes
    # The worker mirrors the manager with hash == cache-hit alignment ==
    # checkpoint alignment == the fold unit (the Mamba block by default).
    unit = fold_unit(block)
    what = "Mamba block" if unit == block else "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT"
    hint = "" if unit == block else f" (set --prefix-match-unit {unit})"
    for m in managers:
        if not m.has_prefill_checkpoint_blocks:
            return "a Mamba group without a prefill checkpoint block"
        if m.drop_eagle_checkpoint_block:
            return "a Mamba group drops the EAGLE block"
        alignment = getattr(
            getattr(m, "kv_cache_spec", None), "prefill_checkpoint_alignment", unit
        )
        if alignment != unit:
            return f"checkpoint alignment {alignment} != {what} {unit}"
        if m.block_pool.hash_block_size != unit:
            return f"hash block {m.block_pool.hash_block_size} != {what} {unit}{hint}"
        if m.cache_hit_alignment_tokens != unit:
            return (
                f"cache hit alignment {m.cache_hit_alignment_tokens} != "
                f"{what} {unit}{hint}"
            )
    if scheduler.hash_block_size != unit:
        return f"scheduler hash block {scheduler.hash_block_size} != {unit}{hint}"
    checkpoint_block = (
        scheduler.hash_block_size
        if scheduler.mamba_partial_cache_hit
        else scheduler.block_size
    )
    if checkpoint_block != unit:
        return f"scheduler checkpoint block {checkpoint_block} != {unit}"
    return None


def _disable_internal_checkpoint(scheduler) -> None:
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    scheduler.mamba_has_prefill_checkpoint_blocks = False
    for m in scheduler.kv_cache_manager.coordinator.single_type_managers:
        if isinstance(m, MambaManager):
            m.has_prefill_checkpoint_blocks = False


def _verify(scheduler) -> bool:
    """Once per scheduler, at its first split (after every __init__ patch)."""
    state = scheduler.__dict__.get("_tpu_tail_fold_ok")
    if state is not None:
        return state
    ok = bool(getattr(scheduler, "mamba_has_prefill_checkpoint_blocks", False))
    if ok:
        reason = _geometry_mismatch(scheduler)
        if reason is not None:
            logger.warning(
                "VLLM_TPU_MAMBA_TAIL_FOLD: disabled, %s; prefills keep the "
                "block-aligned tail split.",
                reason,
            )
            _disable_internal_checkpoint(scheduler)
            ok = False
        else:
            logger.info(
                "VLLM_TPU_MAMBA_TAIL_FOLD: on; a prefill that fits one chunk "
                "keeps its tail and checkpoints its Mamba state at a %d-token "
                "boundary inside the step (Mamba block %d).",
                scheduler.hash_block_size,
                scheduler.cache_config.block_size,
            )
    else:
        logger.warning(
            "VLLM_TPU_MAMBA_TAIL_FOLD: set, but the Mamba specs carry no "
            "prefill checkpoint block (only the Qwen4Exp unified pool in "
            "align mode does); prefills keep the tail split."
        )
    scheduler._tpu_tail_fold_ok = ok
    return ok


def patch_mamba_tail_fold() -> None:
    if not envs.VLLM_TPU_MAMBA_TAIL_FOLD:
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    original_split = Scheduler._mamba_block_aligned_split
    if getattr(original_split, _PATCH_FLAG, False):
        return

    @wraps(original_split)
    def split(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        args = (num_new_local_computed_tokens, num_external_computed_tokens)
        if not _verify(self):
            return original_split(self, request, num_new_tokens, *args)
        folded = original_split(self, request, num_new_tokens, *args)
        start = request.num_computed_tokens + sum(args)
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end or start + num_new_tokens < prefill_end:
            return folded
        # The chunk reaches the prefill end: count whether the split without
        # an internal checkpoint would have stopped it short. Pure function;
        # schedule() calls it synchronously on the EngineCore thread.
        self.mamba_has_prefill_checkpoint_blocks = False
        try:
            unfolded = original_split(self, request, num_new_tokens, *args)
        finally:
            self.mamba_has_prefill_checkpoint_blocks = True
        if unfolded < num_new_tokens:
            SCHEDULER_COUNTER.add(int(folded > unfolded), 1)
        return folded

    setattr(split, _PATCH_FLAG, True)
    Scheduler._mamba_block_aligned_split = split
    logger.info("VLLM_TPU_MAMBA_TAIL_FOLD: scheduler patch installed.")
