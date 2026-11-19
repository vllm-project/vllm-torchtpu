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
"""q38 L6 lever ``clean`` (H2 CARRY-LEAN): the CARRY host check and record, lean.

Lever ``clean`` (``VLLM_TPU_L6_CARRY_LEAN``), read once per worker step through
``core/l6_ab.py`` (0 / 1 / ab:<classes>; never part of the compile key; no
compiled program changes). It acts only next to CARRY (runner/decode_carry.py)
and STEADY (runner/l6_steady.py); off, CARRY runs its stock check and record.

**Check.** ``decode_carry._check`` rebuilds the request-id tuple and the
mapped-position tuple, gathers the rejected counts with np.fromiter over a
generator, and runs four array compares and an np.repeat every step. When this
step's STEADY reorder verdict was ``ro_fast`` (``l6_steady.reorder_try`` stamps
``fast_seq``) against the very reorder cache that was in effect when the carry
state was recorded (``CarryState.ro``, the epoch: held by reference, so ``is``
is exact), the batch is the recorded one: the same request ids in the same
order, the same scheduled-token dict (so the same ``nsched`` and total), and no
request joined, left, resumed or was preempted (C l6_steady.reorder_try). Then:

* request order and scheduled counts need no compare (the verdict);
* ``seq_lens_np[:n] = num_computed + nsched`` and ``seq_lens_np[n:] = 0`` every
  step (``tpu_runner.fill_step_seq_lens``), and ``positions_np[:total]`` is
  ``num_computed[req] + arange`` from the same ``nsched`` (the general path,
  ptrims' cached form, or the 1-token path); so the stock position compare,
  the sequence-padding compare and ``np.repeat`` follow from the sequence-length
  test, and the position padding is written by nothing between two steps
  (tests/q38_l6/test_l6_pshort.py scans every writer of these mirrors);
* the mapped-position tuple is replaced by one dict compare of the previous
  results' ``req_id_to_index_copy`` against the recorded one (equal dicts give
  equal positions for every recorded request, so ``map_same`` holds);
* what is left is one vector test, ``seq_lens_np[:n] - S + r_prev == step_delta``
  on the recorded ``S`` and ``step_delta``, with ``r_prev`` gathered by one
  C-level ``map``.

Any condition that does not hold hands the step to the stock ``_check``, so
every decision (carried or not, and the reason) is the stock one; the lean form
only ever accepts a step the stock check accepts, with the same ``map_same``.

**Record.** On a chunk the lean check accepted this step (same batch, same map),
``_record`` updates the state in place: the step, the source tag, ``S`` and
``pos`` by ``np.copyto``, the device tensors and the map
reference. Everything else in the state is unchanged by the batch identity
(request ids, ``nsched``, target, total, padded total, the mapped tuple, the
padding snapshots: the sequence padding is zero, the position padding
unwritten). Otherwise the stock record runs and the lean fields are stamped.

**K-step bursts** (reports/kstep_design_v1.md §10 item 1). The test reads the
recorded ``S`` and ``step_delta`` at check time and keeps the stock step pairing
(``st.step == step - 1``), so a shifted re-record that rewrites the stock fields
(``S``, ``pos``, ``step_delta``) is honoured as the stock check honours it, and a
head after a burst fails the vector test exactly where the stock check fails
``why_seq``.

``VLLM_TPU_L6_CARRY_LEAN_CHECK=N``: every N lean accepts the stock check runs as
well and must accept the same state with the same ``map_same``; a mismatch logs
one ERROR, counts ``clean.mismatch``, uses the stock result and turns the lean
form off for the process. ``VLLM_TPU_L6_CHECK_STEPS`` bounds it, as every L6
CHECK mode.

Engagement on the L6AB window line: ``clean.fast`` (lean accepts),
``clean.full`` (handed to the stock check), ``clean.inplace`` / ``clean.stamp``
(records), ``clean.check``, ``clean.mismatch``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import decode_carry as _dc

logger = init_logger(__name__)

LEVER = "clean"

# VLLM_TPU_L6_CARRY_LEAN_CHECK, resolved on first use (tests reset it).
_CHECK: list[int | None] = [None]


def check_every() -> int:
    v = _CHECK[0]
    if v == 0:
        return 0
    if v is None:
        raw = os.environ.get("VLLM_TPU_L6_CARRY_LEAN_CHECK", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"VLLM_TPU_L6_CARRY_LEAN_CHECK must be >= 0, got {v}")
        _CHECK[0] = v
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return v if _l6_ab.check_allowed() else 0


def on(runner: Any) -> bool:
    """The lever's state for this step (False once a CHECK found a mismatch)."""
    return _l6_ab.on(LEVER) and not runner.__dict__.get("_l6_clean_off", False)


def stamping() -> bool:
    """Whether a CARRY record keeps the lean fields (lever clean or pshort on)."""
    return _l6_ab.on(LEVER) or _l6_ab.on("pshort")


def fast(runner: Any, start_index: int, num_reqs: int, total: int, padded_total: int,
         target: int, layout_plan: Any):
    """The recorded carry state (``map_same`` set) when the lean test holds, else
    None. Reads only; never notes. The gates are ``decode_carry._check``'s (chunk,
    config, ``_layout_ok``, ``_pre_ok``) inlined, plus the epoch / verdict."""
    d = runner.__dict__
    st = d.get("_l6_carry_state")
    if st is None or start_index != 0:
        return None
    ro = getattr(st, "ro", None)
    step = _l6_ab.step_seq()
    c = d.get("_l6_steady_ro")
    # STEADY's same-step ro_fast verdict against the recorded epoch.
    if ro is None or c is not ro or c.fast_seq != step or st.step != step - 1:
        return None
    if (
        num_reqs != runner.input_batch.num_reqs
        or num_reqs != len(st.req_ids)
        or total != st.total
        or padded_total != st.padded_total
        or target != st.target
    ):
        return None
    sl = layout_plan.token_slice
    if (
        not envs.VLLM_TPU_POSTFWD_FUSED
        or getattr(runner, "_pcp_enabled", False)
        or getattr(runner, "uses_mrope", False)
        or layout_plan.kind is not SequenceLayoutKind.ALL
        or (sl.start or 0) != 0
        or sl.stop != padded_total
        or sl.step not in (None, 1)
        or not runner.scheduler_config.async_scheduling
        or runner.speculative_config is None
    ):
        return None
    return fast_tail(runner, st, num_reqs, step)


def fast_tail(runner: Any, st: Any, num_reqs: int, step: int):
    """The per-step half of ``fast`` (the previous results, the rejected counts and
    the vector test), for a caller that established the rest (the short path)."""
    pre = runner._pre_async_results
    if pre is None or pre.spec_decode_num_rejected_tokens is None or pre.next_tokens_tpu is None:
        return None
    # Equal to the recorded (non-empty: every recorded request was mapped) dict, so non-empty.
    if pre.req_id_to_index_copy != st.map_d:
        return None
    t = pre.__dict__.get("_l6_carry_tag")  # decode_carry.tag(pre) without assigning one
    if t is not None and t == st.src_tag:
        return None
    if pre.next_tokens_tpu.dim() != 1:
        return None
    rec = runner.__dict__.get("_l6_carry_rej")
    if rec is None or rec[0] != st.step or rec[1] != st.src_tag:
        return None
    try:
        r = np.fromiter(map(rec[2].__getitem__, st.req_ids), np.int64, num_reqs)
    except KeyError:
        return None
    r += runner.seq_lens_np[:num_reqs]
    r -= st.S
    if r.tobytes() != _delta_bytes(num_reqs, st.step_delta):  # every entry == step_delta
        return None
    st.map_same = True
    st.lean_step = step
    return st


_DELTA: dict[tuple[int, int], bytes] = {}


def _delta_bytes(n: int, step_delta: int) -> bytes:
    """The bytes of an int64 vector of `n` entries equal to `step_delta`."""
    b = _DELTA.get((n, step_delta))
    if b is None:
        b = _DELTA[(n, step_delta)] = np.full(n, step_delta, dtype=np.int64).tobytes()
    return b


def cadence(runner: Any, st: Any, args: tuple):
    """The CHECK cadence after a lean accept (the caller noted ``fast`` and saw a
    non-zero ``check_every()``): None, or the stock result when this step's CHECK
    found a mismatch (the lean form is then off for the process)."""
    k = runner.__dict__.get("_l6_clean_since", 0) + 1
    if k < _CHECK[0]:
        runner._l6_clean_since = k
        return None
    runner._l6_clean_since = 0
    _l6_ab.note(LEVER, "check")
    ref = _dc._check(*args)
    if ref[0] is not st or ref[1] != "" or st.map_same is not True:
        _disable(runner, f"stock check gave {ref[1] or 'carry'} with map_same="
                         f"{getattr(ref[0], 'map_same', None)}")
        st.lean_step = -1
        return ref
    return None


def check(runner: Any, start_index: int, num_reqs: int, nsched: np.ndarray, total: int,
          padded_total: int, target: int, layout_plan: Any):
    """``decode_carry._check``'s (state, reason): the lean accept when it holds,
    else the stock check."""
    st = fast(runner, start_index, num_reqs, total, padded_total, target, layout_plan)
    if st is None:
        _l6_ab.note(LEVER, "full")
        return _dc._check(runner, start_index, num_reqs, nsched, total, padded_total, target,
                          layout_plan)
    _l6_ab.note(LEVER, "fast")
    if _CHECK[0] != 0 and check_every() > 0:
        ref = cadence(runner, st, (runner, start_index, num_reqs, nsched, total, padded_total,
                                   target, layout_plan))
        if ref is not None:
            return ref
    return st, ""


def _disable(runner: Any, detail: str) -> None:
    runner._l6_clean_off = True
    _l6_ab.note(LEVER, "mismatch")
    logger.error(
        "q38 L6 CARRY-LEAN: CHECK mismatch (%s); the lean carry check is off for the rest "
        "of this process", detail,
    )


def record(runner: Any, ctx: Any, num_reqs: int, nsched: np.ndarray, seq_lens: Any,
           positions: Any, input_ids: Any, pending: Any, subst: Any) -> bool:
    """In-place record on a chunk the lean check accepted this step (True), else
    False (the stock record runs)."""
    st = ctx.carry
    step = _l6_ab.step_seq()
    if st is None or getattr(st, "lean_step", -1) != step or runner.__dict__.get("_l6_carry_state") is not st:
        return False
    pre = runner._pre_async_results
    st.step = step
    st.src_tag = _dc.tag(pre)
    np.copyto(st.S, runner.seq_lens_np[:num_reqs])
    np.copyto(st.pos, runner.positions_np[: st.total])
    st.seq_lens = seq_lens
    st.positions = positions
    st.input_ids = input_ids
    st.map_d = pre.req_id_to_index_copy
    st.seq_idx_dev = pending[1]
    st.pos_idx_dev = pending[2]
    st.qsl_dev = ctx.qsl_dev
    st.subst = subst
    st.map_same = False
    st.lean_step = -1
    ctx.recorded = True
    _l6_ab.note(LEVER, "inplace")
    return True


def stamp(runner: Any, st: Any) -> None:
    """After a stock record: the epoch (the STEADY reorder cache in effect now)
    and the map reference."""
    st.ro = runner.__dict__.get("_l6_steady_ro")
    st.map_d = runner._pre_async_results.req_id_to_index_copy
    st.lean_step = -1
    _l6_ab.note(LEVER, "stamp")
