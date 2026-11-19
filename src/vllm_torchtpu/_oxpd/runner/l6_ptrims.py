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
"""q38 L6 lever ``ptrims`` (H0-2 PREP-TRIMS): host trims of the worker's input prep.

Lever ``ptrims`` (``VLLM_TPU_L6_PREP_TRIMS``), read once per worker step
through ``core/l6_ab.py`` (0 / 1 / ab:<classes>; never part of the compile key;
no compiled program changes). Off, ``_prepare_inputs`` takes the stock code
unchanged. On, the steps below produce the same host arrays with less Python /
numpy work; they are what is left after CARRY (runner/decode_carry.py) and
STEADY (runner/l6_steady.py), sized by tests/q38_l6/bench_l6_ptrims.py.

* **Token indices** (``~pi_tokens``, the general path, i.e. every MTP step):
  the request indices, the batched arange and the row offsets
  ``req_index * token_ids stride`` depend only on (start index, request count,
  stride, the scheduled-token counts), so they are cached under exactly that
  key (at most 32 entries) together with a preallocated int64 token-index
  buffer; a step then does two in-place adds and the gather. Port of the
  ``_prepare_inputs`` index cache of IX a46d91d20 and the in-place token
  indices of IX 84c8f2177 (Jordan Dotzel, Qwen3.5 lane PR09 branch). The
  positions and input ids written are the stock values (same dtypes and
  casts).
* **Combined request distribution** (``~pi_post``): the six int32 fields of
  the combined RPA / GDN request distribution staging tensor are written with
  one numpy assignment instead of six torch scalar ``__setitem__`` calls.
* **Mamba slot indices** (``~pi_post``): ``_build_mamba_state_indices`` writes
  the per-request recurrent slots with one numpy assignment into the staging
  tensor instead of one torch scalar ``__setitem__`` per request; slots are
  looked up, and new ones popped from the free list, in the same order (the
  Mamba slot fast path of IX a46d91d20, Jordan Dotzel). Also used by the draft
  path's call, which runs the same function.

``VLLM_TPU_L6_PREP_TRIMS_CHECK=N``: every N trimmed steps the stock code runs
as well and the host results are compared (positions, input ids, the combined
request distribution, mamba slot indices). A mismatch logs one ERROR,
counts ``ptrims.mismatch``, applies the stock values for that step and turns
the trims off for the process. ``VLLM_TPU_L6_CHECK_STEPS`` bounds it, as every
L6 CHECK mode.

Engagement: counts on the L6AB window line (``core/l6_ab.py``) as
``ptrims.<key>=n``: ``steps`` (trimmed ``_prepare_inputs`` calls),
``idx_hit`` / ``idx_build`` (token-index cache), ``comb`` and ``mamba`` (the
two writes), ``check`` (compared items) and ``mismatch``.

Measured and left out (tests/q38_l6/bench_l6_ptrims.py, CPU pod, STEADY +
CARRY + envcache on): the scheduled-count gather (2 us saved at 25 requests),
the query_start_loc cumsum (3 us in all) and the attention schedule check (0.2
us: no attention layer runs the batched RPA schedule, so it returns at once),
the draft-count gather (under 1 us saved) and the cached spec metadata (its
draft gather feeds the deferred host drafts), the carry program's shape key and
wrapper (6 us in all), and the second substitution-index build (CARRY already
reuses its own indices on every step it records).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "ptrims"
_IDX_CACHE_MAX = 32

# VLLM_TPU_L6_PREP_TRIMS_CHECK, resolved on first use (tests reset it).
_CHECK: list[int | None] = [None]


def check_every() -> int:
    v = _CHECK[0]
    if v is None:
        raw = os.environ.get("VLLM_TPU_L6_PREP_TRIMS_CHECK", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"VLLM_TPU_L6_PREP_TRIMS_CHECK must be >= 0, got {v}")
        _CHECK[0] = v
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return v if _l6_ab.check_allowed() else 0


def on(runner: Any) -> bool:
    """The lever's state for this step (False once a CHECK found a mismatch)."""
    return _l6_ab.on(LEVER) and not runner.__dict__.get("_l6_ptrims_off", False)


def drop(runner: Any) -> None:
    """Forget every cached array (the lever is off this step)."""
    runner.__dict__.pop("_l6_ptrims_idx", None)


class _State:
    __slots__ = ("seq", "check", "since_check")

    def __init__(self) -> None:
        self.seq = -1
        self.check = False
        self.since_check = 0


def begin(runner: Any) -> _State | None:
    """Called once per ``_prepare_inputs``: None when the lever is off (the
    stock code runs and the caches are dropped), else the step's state with
    its CHECK decision (fixed for every chunk of the step)."""
    if not on(runner):
        drop(runner)
        return None
    st = runner.__dict__.get("_l6_ptrims_st")
    if st is None:
        st = runner._l6_ptrims_st = _State()
    seq = _l6_ab.step_seq()
    if st.seq != seq:
        st.seq = seq
        n = check_every()
        st.check = False
        if n > 0:
            st.since_check += 1
            if st.since_check >= n:
                st.since_check = 0
                st.check = True
    _l6_ab.note(LEVER, "steps")
    return st


def _disable(runner: Any, what: str, detail: str) -> None:
    runner._l6_ptrims_off = True
    drop(runner)
    _l6_ab.note(LEVER, "mismatch")
    logger.error(
        "q38 L6 PTRIMS: CHECK mismatch in %s (%s); the input-prep trims are off "
        "for the rest of this process",
        what,
        detail,
    )


# --------------------------------------------------------------------------
# ~pi_tokens: token indices (the general path)


def tokens(
    runner: Any,
    st: _State,
    start_index: int,
    num_reqs: int,
    nsched: np.ndarray,
    total: int,
) -> None:
    """Write ``positions_np[:total]`` and ``input_ids_cpu[:total]`` as the stock
    general path does, from cached index arrays."""
    ib = runner.input_batch
    stride = int(ib.token_ids_cpu.shape[1])
    key = (start_index, num_reqs, stride, nsched.tobytes())
    cache = runner.__dict__.get("_l6_ptrims_idx")
    if cache is None:
        cache = runner._l6_ptrims_idx = {}
    ent = cache.get(key)
    if ent is None:
        req_indices = np.repeat(runner.arange_np[start_index : start_index + num_reqs], nsched)
        arange = np.concatenate([runner.arange_np[:n] for n in nsched])
        row_off = req_indices * stride
        tok = np.empty(total, dtype=np.result_type(runner.positions_np.dtype, row_off.dtype))
        ent = (req_indices, arange, row_off, tok, torch.from_numpy(tok))
        if len(cache) >= _IDX_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = ent
        _l6_ab.note(LEVER, "idx_build")
    else:
        _l6_ab.note(LEVER, "idx_hit")
    req_indices, arange, row_off, tok, tok_t = ent
    positions_np = runner.positions_np[:total]
    np.add(ib.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
    np.add(positions_np, row_off, out=tok)
    torch.index_select(
        ib.token_ids_cpu_tensor.flatten(), 0, tok_t, out=runner.input_ids_cpu[:total]
    )
    if st.check:
        _l6_ab.note(LEVER, "check")
        ref_pos, ref_ids = _stock_tokens(runner, start_index, num_reqs, nsched, total)
        why = None
        if not np.array_equal(ref_pos, positions_np):
            why = "positions"
        elif not torch.equal(ref_ids, runner.input_ids_cpu[:total]):
            why = "input ids"
        if why is not None:
            positions_np[:] = ref_pos
            runner.input_ids_cpu[:total] = ref_ids
            _disable(runner, "token indices", why)


def _stock_tokens(runner: Any, start_index: int, num_reqs: int, nsched: np.ndarray, total: int):
    """The stock general path into fresh arrays, for CHECK."""
    ib = runner.input_batch
    req_indices = np.repeat(runner.arange_np[start_index : start_index + num_reqs], nsched)
    arange = np.concatenate([runner.arange_np[:n] for n in nsched])
    pos = np.empty(total, dtype=runner.positions_np.dtype)
    np.add(ib.num_computed_tokens_cpu[req_indices], arange, out=pos)
    token_indices = pos + req_indices * ib.token_ids_cpu.shape[1]
    ids = torch.index_select(ib.token_ids_cpu_tensor.flatten(), 0, torch.from_numpy(token_indices))
    return pos, ids


# --------------------------------------------------------------------------
# ~pi_post: mamba slot indices


def mamba_slots(runner: Any, indices: torch.Tensor, start_index: int, num_live: int) -> None:
    """The stock per-request loop of ``_build_mamba_state_indices`` (``indices``
    already zero-filled): look up each live request's slot, pop a fresh one
    for a request without, in request order; then one numpy write."""
    ids = runner.input_batch.req_ids[start_index : start_index + num_live]
    slots = _gather_slots(runner, ids)
    if num_live > 0:
        indices.numpy()[:num_live] = slots
    _l6_ab.note(LEVER, "mamba")
    st = runner.__dict__.get("_l6_ptrims_st")
    if st is not None and st.check and st.seq == _l6_ab.step_seq():
        _l6_ab.note(LEVER, "check")
        # The stock writes, from the slot map the gather left.
        ref = torch.zeros_like(indices)
        for i, req_id in enumerate(ids):
            ref[i] = runner._mamba_slot_by_req_id[req_id]
        if not torch.equal(ref, indices):
            indices.copy_(ref)
            _disable(runner, "mamba slot indices", "values")


def _gather_slots(runner: Any, ids: list) -> list[int]:
    slot_by_req_id = runner._mamba_slot_by_req_id
    try:
        return list(map(slot_by_req_id.__getitem__, ids))
    except KeyError:
        pass
    slots = []
    free = runner._free_mamba_slots
    for req_id in ids:
        assert req_id is not None
        slot = slot_by_req_id.get(req_id)
        if slot is None:
            # The scheduler bounds running requests by max_num_reqs;
            # preempted owners release their slots in _update_states.
            slot = free.pop()
            slot_by_req_id[req_id] = slot
        slots.append(slot)
    return slots


# --------------------------------------------------------------------------
# ~pi_post: combined RPA / GDN request distribution


def comb(
    runner: Any,
    st: _State,
    combined: torch.Tensor,
    num_decode: int,
    num_windowed: int,
    num_reqs: int,
) -> None:
    """The six stock ``combined[i] = ...`` writes as one numpy assignment."""
    combined.numpy()[:6] = _comb_values(num_decode, num_windowed, num_reqs)
    _l6_ab.note(LEVER, "comb")
    if st.check:
        _l6_ab.note(LEVER, "check")
        # The stock writes, on a copy.
        ref = combined.clone()
        ref[0] = num_decode
        ref[1] = num_decode
        ref[2] = num_reqs
        ref[3] = num_windowed
        ref[4] = num_windowed
        ref[5] = num_reqs
        if not torch.equal(ref, combined):
            combined.copy_(ref)
            _disable(runner, "combined request distribution", "values")


def _comb_values(num_decode: int, num_windowed: int, num_reqs: int) -> tuple[int, ...]:
    return (num_decode, num_decode, num_reqs, num_windowed, num_windowed, num_reqs)
