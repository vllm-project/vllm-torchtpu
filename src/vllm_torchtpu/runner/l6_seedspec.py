# SPDX-License-Identifier: Apache-2.0
"""q38 HCDEC-HOSTADM SEED-SPEC: the GDN seed copy at a mamba block edge without the blocking copy wait.

Lever ``seedspec`` (``VLLM_TPU_L6_SEED_SPEC``: 0 / 1 / ab:<classes>), read once per worker step through
``core/l6_ab.py``; host-only, never part of the compile key, no new compiled program.

The wait it removes. ``TPUModelRunner._collect_mamba_state_seed_copies`` positions every request as if all of the
previous step's drafts were accepted. When a request's last token of the step lies within ``drafts`` tokens past a
state-column edge, the true column depends on how many of those drafts were rejected, so the stock path waits for the
previous step's sampled-token copy (``previous.wait_for_copy()``, sub-mark ``~pi_seed_sync``) to learn it. On the
served T93x decode (j-cdcd4675 pod h2j6t, c112) that wait runs on 17 % of steps (43.5 per 250, HOSTSUBS) at a median
of 8.6 ms, and the steps that take it run 7.6-8.9 ms over their local baseline (device-0 dispatch stamps).

What the lever does instead, only in the canonical case (the request's recorded column ``prev`` is the low side
``c = (end - drafts) // stride``, the all-accepted column is ``c + 1``, and the seed takes the speculative checkpoint
path: read offsets on, checkpoint window > 1). The decision uses host data every TP rank shares (the scheduler output,
the previous results' draft counts, the recorded columns), never whether this rank's copy has landed, so every rank
stages the same seed programs:

* it keeps the all-accepted position, so the collector stages exactly the crossing ``(prev, c + 1)`` that the stock
  path stages when the request did cross: the same packed seed row ``[block(c), new_owner, 1, block(c + 1),
  checkpoint blocks of column c]``, the same read-offset seed at ``block(c + 1)``, the same programs and shapes;
* it records the request as pending and, at the first collector call of a LATER step (whose ``previous`` is a newer
  results object), reads the rejected count from the results object the stock path would have waited on (landed by
  then: the device ran it before the step whose verify the host has since waited on) and sets ``_mamba_state_pos`` to
  the true column, which is the value the stock path would have stored.

Why the result is bitwise the stock path's. If the request crossed, the device work is identical. If it did not, the
only extra device effects are the copy ``block(c + 1) := checkpoint[read_offsets[block(c)]]`` and ``read_offsets[
block(c + 1)] := 0``. ``block(c + 1)`` is checkpoint 1 of the live group. Its bytes are read only as the checkpoint at
``read_offsets[block(c)] == 1``: at the copy itself that makes the copy a self-copy (the seed row picks the source on
device), and afterwards a read offset of 1 is set only by a verify whose forward wrote (or committed) checkpoint 1
after the copy. Otherwise they are next touched by the seed of the real crossing into ``c + 1``, which overwrites
them and resets the same read offset to 0 before any forward reads it. The read offset of a checkpoint block other
than the state block is not read by the forward. The block is the request's own, freshly allocated decode-time block
(the canonical case puts ``c + 1`` past the request's previous state column), so no prefix-cache hit names it.

Port onto Benliang Wang's tree (q38 BLSTACK-SEED, from q38-hcdec-seedspec 3552fb4b2, credit HCDEC-HOSTADM): ``core/l6_ab.py``
is a minimal shim there (plain "1" only, "ab:" refused at import); ``edge`` also declines a multi-chunk previous step
(his collector asserts ``previous.num_chunks == 1`` before it reads the sampled copy by ``previous_row``); his GDN
window-log groups (VLLM_TPU_GDN_WINDOW_LOG: the seed copies checkpoint 0 with its log and migrates the read offset)
take the same canonical decision: the not-crossed extra copy and offset land on ``block(c + 1)``, which a window-log
group never reads while its state column is ``c`` (only the state block is read), and the real crossing rewrites both.

Engagement (rule 14) on the worker L6AB line: ``seedspec.spec`` (speculative seeds staged), ``seedspec.cross`` /
``seedspec.stay`` (resolved outcomes), ``seedspec.rwait`` / ``seedspec.rwait_us`` (resolutions whose copy had not
landed, and their wait), ``seedspec.rsync`` (a pending request met again with the same results object: resolved by
waiting), ``seedspec.drop`` (pending requests that finished or were invalidated before resolution), ``seedspec.late`` (an
entry ``resolve`` should have taken, met by ``settle``: 0 in correct operation). The D1N line's ``syncs`` counts the
stock waits that remain.
"""

from __future__ import annotations

import time
from typing import Any

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "seedspec"
# q38 BLSTACK-SEED: read the lever (and refuse "ab:") when the runner module loads, not at the first block edge.
_l6_ab.configured(LEVER)
_INVALID = -1  # tpu_runner_async_output.INVALID_TOKEN_ID
_LOGGED = [False]


class _Pending:
    __slots__ = ("results", "row", "drafts", "end", "low", "high", "stride")

    def __init__(self, results, row, drafts, end, low, high, stride):
        self.results = results
        self.row = row
        self.drafts = drafts
        self.end = end
        self.low = low
        self.high = high
        self.stride = stride


def _pending(runner: Any) -> dict:
    p = runner.__dict__.get("_seedspec_pending")
    if p is None:
        p = runner._seedspec_pending = {}
    return p


def on() -> bool:
    """The lever's state for this worker step (False before the first begin_step)."""
    return _l6_ab.on(LEVER)


def edge(
    runner: Any,
    req_id: str,
    previous: Any,
    previous_row: int | None,
    drafts: int,
    end_position: int,
    col_stride: int,
) -> bool:
    """At a block edge whose stock handling would wait for ``previous``'s copy: True when this request's seed is
    staged speculatively instead (the caller keeps the all-accepted ``end_position``), False for the stock wait."""
    if not on():
        return False
    if getattr(previous, "copy_state", None) is None:
        return False  # nothing to wait for: the stock path costs nothing
    if previous_row is None or drafts <= 0:
        return False
    if getattr(previous, "num_chunks", 1) != 1:
        return False  # q38 BLSTACK-SEED: the stock path's single-chunk assert keeps this case
    if runner.mamba_slot_read_offsets is None or runner._mamba_ckpt_window <= 1:
        return False
    high = end_position // col_stride
    low = (end_position - drafts) // col_stride
    if high != low + 1 or low < 0:
        return False
    if runner._mamba_state_pos.get(req_id) != low:
        return False
    pend = _pending(runner)
    if req_id in pend:
        return False
    pend[req_id] = _Pending(previous, int(previous_row), int(drafts), int(end_position), low, high, int(col_stride))
    _l6_ab.note(LEVER, "spec")
    if not _LOGGED[0]:
        _LOGGED[0] = True
        logger.info(
            "q38 L6 SEEDSPEC (VLLM_TPU_L6_SEED_SPEC): block-edge GDN seed staged without the copy wait, crossing "
            "resolved at the next step (first: column %d -> %d, stride %d)",
            low,
            high,
            col_stride,
        )
    return True


def _resolve_one(runner: Any, req_id: str, p: _Pending) -> None:
    copy_state = getattr(p.results, "copy_state", None)
    waited = copy_state is not None and not getattr(copy_state, "completed", False)
    t0 = time.perf_counter_ns() if waited else 0
    sampled = p.results.wait_for_copy()
    if waited:
        _l6_ab.note(LEVER, "rwait")
        _l6_ab.note(LEVER, "rwait_us", (time.perf_counter_ns() - t0) // 1000)
    accepted = int((sampled[p.row] != _INVALID).sum().item())
    rejected = max(0, p.drafts + 1 - accepted)
    col = (p.end - rejected) // p.stride
    runner._mamba_state_pos[req_id] = col
    _l6_ab.note(LEVER, "cross" if col == p.high else "stay")


def resolve(runner: Any, previous: Any) -> None:
    """Resolve every pending request whose results object is not ``previous`` (it belongs to an earlier step). Runs
    at the top of every collector call, before any request's column is read, whatever the lever's state."""
    pend = runner.__dict__.get("_seedspec_pending")
    if not pend:
        return
    live = runner.requests
    for req_id in list(pend):
        p = pend[req_id]
        if p.results is previous:
            continue
        del pend[req_id]
        # _update_states drops the column of a finished / preempted / resumed request: then the stock path would
        # start it afresh, and so does this one.
        if req_id not in live or runner._mamba_state_pos.get(req_id) != p.high:
            _l6_ab.note(LEVER, "drop")
            continue
        _resolve_one(runner, req_id, p)


def settle(runner: Any, req_id: str, previous: Any) -> None:
    """Before reading ``req_id``'s column, for an entry ``resolve`` left in place: one staged against ``previous``
    itself (the collector met the request again before a newer results object existed, e.g. a step that produced no
    sampled-token copy). The column is then only known from ``previous``'s copy: resolve it by waiting, as the stock
    path would. An entry from an OLDER results object here means ``resolve`` did not run; it is resolved the same
    way and counted ``seedspec.late`` (0 in correct operation)."""
    pend = runner.__dict__.get("_seedspec_pending")
    if not pend or req_id not in pend:
        return
    p = pend.pop(req_id)
    _l6_ab.note(LEVER, "rsync" if p.results is previous else "late")
    if req_id in runner.requests and runner._mamba_state_pos.get(req_id) == p.high:
        _resolve_one(runner, req_id, p)  # settle: by waiting on the entry's own results
