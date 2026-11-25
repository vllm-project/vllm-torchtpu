# SPDX-License-Identifier: Apache-2.0
"""q38 HIX700 GDN-SIDECAR runtime policy (VLLM_TPU_MK_GDN_SIDECAR=1): when the 16-row whole step may read its GDN states from the sidecar,
and when the sidecar must be flushed into the pool first.

The sidecar (the whole-step module's `side_rec` / `side_conv`, filled by the last sidecar step, with that step's write ids `side_wb`) holds
every live request's verify checkpoints; the pool's checkpoint blocks are stale while it is valid. Anything that reads the pool's
checkpoint blocks must therefore see a flush first. A FLUSH writes the sidecar's checkpoints into the pool with exactly the flag-off scatter
bytes (`scatter_gdn_ckpts_fast` on `side_wb`, G1's `full` op), with the write ids of rows whose request has left the batch set to 0 (their
blocks may already belong to someone else), and marks the sidecar invalid.

Decision (`before_execute`, at the start of execute_model, before any seed copy; `before_forward`, just before the megakernel's forward):
  * valid stays only if the scheduler output has no new, finished or preempted request (scheduler_output.scheduled_new_reqs,
    finished_req_ids, and `preempted_req_ids` if present), every scheduled request is a pure decode row (num_scheduled_tokens ==
    num_spec + 1), no request can cross a mamba state block this step (conservatively: the window of positions it may end at spans a
    block edge), and the batch's row -> request map equals the one the sidecar was filled with;
  * otherwise FLUSH (if valid) and run with side_on = 0 (the kernel gathers from the pool: admission / after a flush);
  * the module's `side_on` buffer is set to 1 / 0 accordingly before the forward;
  * `after_sample` (sample_tokens, after the read offsets are applied) marks the sidecar valid iff the whole step ran at the sidecar bucket
    this step (the module's hit mark), and snapshots the row map; after a step in which a crossing was possible it flushes at once, so a
    block the step completes (and the prefix cache may publish) holds exactly today's bytes. Warmup / dummy / multi-chunk forwards are followed by no sample, so they
    never validate the sidecar.
Counters (steps on the sidecar, flushes by reason) are logged at the first 4 flushes, every 4096 flushes, and every LOG_EVERY execute_model
steps (`q38 GDN-SIDECAR runtime: step N: ...`: the steady-state engagement line readers key on).
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("vllm." + __name__)
ENGAGED = "q38 GDN-SIDECAR runtime"
LOG_EVERY = 1024


class State:
    valid = False
    armed = False  # before_execute ran for the coming forward (a forward without it: warmup / dummy / profile run)
    crossing = False  # a state-block crossing was possible this step: the pool gets this step's checkpoints right after the sample
    rows: tuple = ()
    side_steps = 0
    flushes: dict = {}
    steps = 0
    last_hit = False
    logged_at = 0


def module(runner: Any) -> Any:
    """The whole-step module holding a sidecar, or None (flag off / no whole step / no 16-row bucket)."""
    drv = getattr(runner, "_megakernel", None)
    whole = getattr(drv, "whole_module", None) if drv is not None else None
    if whole is None or getattr(whole, "side_rec", None) is None:
        return None
    return whole


def _rows(runner: Any) -> tuple:
    ids = getattr(runner.input_batch, "req_ids", [])
    return tuple(x for x in ids if x is not None)


def _count(reason: str) -> None:
    State.flushes[reason] = State.flushes.get(reason, 0) + 1
    n = sum(State.flushes.values())
    if n <= 4 or n % 4096 == 0:
        _log.info("%s: flush %d (%s); sidecar steps %d of %d; flushes by reason %s", ENGAGED, n, reason, State.side_steps, State.steps,
                  dict(sorted(State.flushes.items())))


def flush(runner: Any, reason: str, keep_rows: tuple | None = None) -> bool:
    """Write the valid sidecar into the pool (flag-off bytes), rows whose request is not in `keep_rows` (None = keep all) nulled; invalidate.
    Returns whether it wrote."""
    mod = module(runner)
    if mod is None or not State.valid:
        State.valid = False
        return False
    wb = mod.side_wb
    if keep_rows is not None:
        gone = [i for i, r in enumerate(State.rows) if r not in keep_rows and i < wb.shape[1]]
        if gone:
            wb = wb.clone()
            wb[:, _index(gone, wb), :] = 0
    _full_op(mod.kind.layout).full(mod.pool, mod.side_conv, mod.side_rec, wb)
    State.valid = False
    _count(reason)
    return True


def _full_op(layout: Any) -> Any:
    """G1's compiled flag-off scatter op (`full`: pool, conv, rec, wb -> the pool written in place)."""
    from .qwen38_gdn_commit import _ops

    return _ops(layout)


def _index(rows: list, like: Any) -> Any:
    try:
        import torch

        return torch.tensor(rows, dtype=torch.long, device=like.device)
    except ImportError:  # the CPU chain simulation's fake tensors
        return rows


def _crossing_possible(runner: Any, scheduler_output: Any, k: int) -> bool:
    """Conservative: some request's possible end positions this step (num_computed + scheduled - 1, down to that minus k rejected drafts)
    span a mamba state block edge (the allocator's stride: spec block size x CP world, as `_collect_mamba_state_seed_copies` derives it)."""
    stride = getattr(runner, "_mamba_state_block_size", None)
    if not stride:
        return True
    try:
        from vllm.distributed import get_pcp_group  # the runner's import (tpu_runner.py:41)

        cp = get_pcp_group().world_size
    except Exception:  # noqa: BLE001
        cp = 1
    stride = int(stride) * int(cp)
    ids = runner.input_batch.req_ids
    comp = runner.input_batch.num_computed_tokens_cpu
    for i, rid in enumerate(ids):
        if rid is None:
            continue
        n = scheduler_output.num_scheduled_tokens.get(rid, 0)
        end = int(comp[i]) + int(n) - 1
        lo = end - k - 1
        if end // stride != max(lo, 0) // stride:
            return True
    return False


def _pure_decode(runner: Any, scheduler_output: Any, k: int) -> bool:
    ns = scheduler_output.num_scheduled_tokens
    if not ns:
        return False
    return all(int(v) == k + 1 for v in ns.values()) and sum(int(v) for v in ns.values()) <= 16


def _num_spec(runner: Any) -> int:
    sc = getattr(runner, "speculative_config", None) or getattr(getattr(runner, "vllm_config", None), "speculative_config", None)
    return int(getattr(sc, "num_speculative_tokens", 0) or 0)


def before_execute(runner: Any, scheduler_output: Any) -> None:
    """execute_model start (before input prep and seed copies): flush when anything this step could read the pool's checkpoints or change
    the row map."""
    if module(runner) is None:
        return
    State.steps += 1
    State.armed = True
    k = _num_spec(runner)
    State.crossing = _crossing_possible(runner, scheduler_output, k)
    if not State.valid:
        return
    if getattr(scheduler_output, "scheduled_new_reqs", None):
        flush(runner, "new request", keep_rows=_keep(runner, scheduler_output))
    elif getattr(scheduler_output, "finished_req_ids", None):
        flush(runner, "finished request", keep_rows=_keep(runner, scheduler_output))
    elif getattr(scheduler_output, "preempted_req_ids", None):
        flush(runner, "preempted request", keep_rows=_keep(runner, scheduler_output))
    elif not _pure_decode(runner, scheduler_output, k):
        flush(runner, "not a pure 16-row decode step")
    elif State.crossing:
        flush(runner, "state-block crossing possible")


def _keep(runner: Any, scheduler_output: Any) -> tuple:
    gone = set(getattr(scheduler_output, "finished_req_ids", ()) or ()) | set(getattr(scheduler_output, "preempted_req_ids", ()) or ())
    return tuple(r for r in State.rows if r not in gone)


def before_forward(runner: Any) -> None:
    """Just before the megakernel forward: the row map must still be the sidecar's; set the module's side_on."""
    mod = module(runner)
    if mod is None:
        return
    if State.valid and not State.armed:  # a forward outside execute_model writes the sidecar in place and samples nothing: flush first
        flush(runner, "forward without execute_model")
    State.armed = False
    if State.valid and _rows(runner) != State.rows:
        flush(runner, "row map changed", keep_rows=_rows(runner))
    mod.side_on.fill_(1 if State.valid else 0)
    runner.__dict__["_q38_side_hit"] = 0  # the driver sets the bucket when the whole step runs this forward
    if State.valid:
        State.side_steps += 1


def after_sample(runner: Any, whole_hit: bool) -> None:
    """sample_tokens, after the read offsets: the sidecar is valid iff this step's whole step ran at the sidecar bucket."""
    if module(runner) is None:
        return
    State.valid = bool(whole_hit)
    State.rows = _rows(runner) if whole_hit else ()
    if State.steps and State.steps % LOG_EVERY == 0 and State.logged_at != State.steps:
        State.logged_at = State.steps
        _log.info("%s: step %d: sidecar steps %d (%.1f %%), flushes %d by reason %s", ENGAGED, State.steps, State.side_steps,
                  100.0 * State.side_steps / State.steps, sum(State.flushes.values()), dict(sorted(State.flushes.items())))
    if State.valid and State.crossing:
        # A block this step completes (and may publish to the prefix cache) must hold exactly today's bytes: write this step's
        # checkpoints into the pool now, as today's scatter would have (the next step then gathers from the pool).
        flush(runner, "after a possible crossing")
    State.crossing = False


def reset() -> None:
    State.valid, State.rows, State.side_steps, State.flushes, State.steps, State.armed, State.crossing = False, (), 0, {}, 0, False, False
    State.logged_at = 0


__all__ = ["ENGAGED", "State", "after_sample", "before_execute", "before_forward", "flush", "module", "reset"]
