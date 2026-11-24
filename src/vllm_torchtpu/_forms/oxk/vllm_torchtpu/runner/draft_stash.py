# SPDX-License-Identifier: Apache-2.0
"""q38 L1-DSTASH: carry a held-back request's drafts across a DECODE_TRIM skip (VLLM_TPU_DECODE_TRIM_DRAFT_STASH).

Why: DECODE_TRIM (core/decode_trim.py) holds some running decode requests back for one step. Under async scheduling
with MTP, vLLM's AsyncScheduler ships ``[-1] * k`` placeholder drafts, and the TPU runner fills a request's input
slots from the previous step's device results only for requests present in them
(``TPUModelRunner._prepare_async_token_substitution_indices`` skips any request id absent from
``_pre_async_results.req_id_to_index_copy``). A request held back at step t+1 is absent from step t+1's results, so
at its re-entry at t+2 it verifies -1 drafts and yields only its bonus token.

What the stash does (runner side only; the scheduler is untouched):
  capture  (``_update_states`` of step t+1, before the base update): every request that has a source row in step
           t's results but is not scheduled now and is still live (not finished, preempted or resumed) gets an entry
           ``(step t's next_tokens_tpu, its row position)``. The device tensor is only referenced, never copied: it
           already holds that request's ``[bonus, draft_1..K]`` at ``position * (1 + K)``, and nothing writes it
           after the step that produced it.
  apply    (``_prepare_inputs`` at re-entry, right after the input ids are uploaded and before any other
           substitution): the request's input slots are filled from its entry with the same compiled
           ``_substitute_placeholder_token`` program, called exactly as ``_apply_async_token_substitution`` calls it
           (input bucket, int32 indices padded to it, the source through ``_warmed_next_tokens`` when
           VLLM_TPU_WARMED_SUBSTITUTE_SOURCE is set, as in the served stack, i.e. a precompiled signature), so no new
           device program exists. The entry is consumed.
           The request stays absent from the current results' map, so the regular substitution leaves its slots
           alone, and the rejection correction (keyed by the same map) and the GDN read offset (kept per state slot
           across the skip) are exactly what they are without the stash: the scheduler shipped exact positions
           (update(t) runs before schedule(t+2)), so the carried rejected count is zero.
  count    (``_modify_prev_results`` of the step after re-entry): the tokens committed by each stash-hit re-entry.

It is a no-op unless VLLM_TPU_DECODE_TRIM is also set and a request was actually held back: with the flag off no
runner attribute is created and every hook is a dictionary lookup. (The runner sees "held back" as "had a row in the
previous results, not scheduled now, still live"; vLLM's own skip of a request whose previous step already reached
max_tokens matches too, and its entry is dropped when the request finishes.) Refused (logged once, never engaged) under prefill
context parallelism or a multi-stage pipeline, and skipped for a chunk of a step with structured-output requests
(the scheduler's grammar bitmask was built for the placeholders).

Env (host-only, compile-ignored):
  VLLM_TPU_DECODE_TRIM_DRAFT_STASH  0 / 1 (default 0).
  VLLM_TPU_DECODE_TRIM_LOG          engine steps per 'q38 DSTASH' window line (shared with DECODE_TRIM; 0 = install
                                    line only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_COUNTERS = (
    "steps", "captured", "hits", "reentry_nostash", "accept_n", "accept_tokens", "dropped_invalidated",
    "dropped_dead", "conflict", "skip_shape", "skip_rows", "skip_structured", "held_nonspec",
)


def trim_on() -> bool:
    raw = str(envs.VLLM_TPU_DECODE_TRIM or "").strip().lower()
    return raw not in ("", "0", "off", "false")


def enabled() -> bool:
    """Both flags: the stash only exists to undo DECODE_TRIM's re-entry draft loss."""
    return bool(envs.VLLM_TPU_DECODE_TRIM_DRAFT_STASH) and trim_on()


class DraftStash:
    """Per-runner stash state and counters."""

    def __init__(self, k: int, log_every: int):
        self.k = int(k)
        self.stride = 1 + self.k
        self.log_every = int(log_every)
        # req_id -> (source tensor of the step the request last ran in, its row position in that source)
        self.entries: dict[str, tuple[torch.Tensor, int]] = {}
        # stash-hit re-entries whose committed token count is read when their step's results are committed
        self.pending_accept: set[str] = set()
        self.last_applied: tuple[str, ...] = ()
        self.totals = dict.fromkeys(_COUNTERS, 0)
        self.totals["accept_hist"] = {}
        self._reset_window()

    def _reset_window(self) -> None:
        self.w = dict.fromkeys(_COUNTERS, 0)
        self.w_hist: dict[int, int] = {}

    def _add(self, key: str, n: int = 1) -> None:
        self.w[key] += n
        self.totals[key] += n

    # ------------------------------------------------------------------ capture (step t+1, _update_states)
    def capture(self, pre: Any, scheduler_output: Any, invalidated: Any, live: Any, batch: Any) -> None:
        """`pre` = step t's AsyncPreResults (invalidated requests already popped from its map); `live` = the runner's
        request states before the base update; `batch` = persistent-batch membership before the base update."""
        self._add("steps")
        self.last_applied = ()
        for rid in invalidated:
            self.pending_accept.discard(rid)
            if self.entries.pop(rid, None) is not None:
                self._add("dropped_invalidated")
        for rid in [r for r in self.entries if r not in live]:
            del self.entries[rid]
            self._add("dropped_dead")
        if self.pending_accept:
            self.pending_accept.intersection_update(live)
        scheduled = scheduler_output.num_scheduled_tokens
        cached = scheduler_output.scheduled_cached_reqs
        resumed = cached.resumed_req_ids
        mapping = pre.req_id_to_index_copy if pre is not None else None
        held = [r for r in mapping if r not in scheduled and r in live] if mapping else []
        if held:
            src = pre.next_tokens_tpu
            if pre.spec_decode_num_rejected_tokens is None:
                self._add("held_nonspec", len(held))  # stride-1 source: no drafts to carry
            elif src is None or src.dim() != 1:
                self._add("skip_shape", len(held))  # not the flat [rows * (1 + K)] source
            else:
                numel = int(src.numel())
                for rid in held:
                    pos = int(mapping[rid])
                    if (pos + 1) * self.stride > numel:
                        self._add("skip_shape")
                        continue
                    self.entries[rid] = (src, pos)
                    self._add("captured")
        for rid in cached.req_ids:
            if rid not in batch and rid not in resumed and rid not in self.entries:
                self._add("reentry_nostash")  # back after a gap, nothing to carry
        if self.log_every and self.w["steps"] >= self.log_every:
            self.log_window()

    # ------------------------------------------------------------------ apply (re-entry, _prepare_inputs)
    def plan(
        self,
        req_ids: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray,
        pre_map: Any,
        layout_plan: Any,
        structured: bool,
    ) -> list[tuple[torch.Tensor, list[int], list[int]]]:
        """Consume this chunk's entries; return one (source, input slots, source slots) group per source tensor."""
        groups: dict[int, tuple[torch.Tensor, list[int], list[int]]] = {}
        applied: list[str] = []
        acc = 0
        for i in range(num_reqs):
            rid = req_ids[start_index + i]
            n = int(num_scheduled_tokens_per_req[i])
            acc += n
            ent = self.entries.pop(rid, None)
            if ent is None:
                continue
            if structured:
                self._add("skip_structured")
                continue
            if pre_map and rid in pre_map:
                self._add("conflict")  # it ran last step after all: the regular substitution owns it
                continue
            if n > self.stride:
                self._add("skip_rows")  # not a decode verify span
                continue
            src, pos = ent
            base = pos * self.stride
            g = groups.get(id(src))
            if g is None:
                g = groups[id(src)] = (src, [], [])
            for j in range(n):
                rm = acc - n + j
                cur = rm if layout_plan is None else layout_plan.local_index_for_request_major_token(rm)
                if cur is None:
                    continue
                g[1].append(int(cur))
                g[2].append(base + j)
            applied.append(rid)
        if applied:
            self._add("hits", len(applied))
            self.pending_accept.update(applied)
            self.last_applied = self.last_applied + tuple(applied)
        return [g for g in groups.values() if g[1]]

    # ------------------------------------------------------------------ count (commit of the re-entry step)
    def note_committed(self, records: Any, valid_sampled_token_ids: Any) -> None:
        """`records` = the committed step's request_seq_lens. A stash-hit request is in exactly the results of the
        step it re-entered in (it was held back in the step before), so membership identifies the step."""
        for pre_req_idx, _state, _seq_len, rid in records:
            if rid in self.pending_accept:
                self.pending_accept.discard(rid)
                n = len(valid_sampled_token_ids[pre_req_idx])
                self._add("accept_n")
                self._add("accept_tokens", n)
                self.w_hist[n] = self.w_hist.get(n, 0) + 1
                th = self.totals["accept_hist"]
                th[n] = th.get(n, 0) + 1

    def log_window(self) -> None:
        w = self.w
        hist = "/".join(f"{k}:{v}" for k, v in sorted(self.w_hist.items())) or "-"
        logger.info(
            "q38 DSTASH steps=%d captured=%d hits=%d reentry_nostash=%d accepted_tokens=%d accepted_per_hit=%.3f "
            "accept_hist=%s dropped_invalidated=%d dropped_dead=%d conflict=%d skip_shape=%d skip_rows=%d "
            "skip_structured=%d held_nonspec=%d live_entries=%d totals=%s",
            w["steps"], w["captured"], w["hits"], w["reentry_nostash"], w["accept_tokens"],
            (w["accept_tokens"] / w["accept_n"]) if w["accept_n"] else float("nan"), hist,
            w["dropped_invalidated"], w["dropped_dead"], w["conflict"], w["skip_shape"], w["skip_rows"],
            w["skip_structured"], w["held_nonspec"], len(self.entries), self.totals,
        )
        self._reset_window()


def padded_indices(cur: list[int], src: list[int], length: int) -> tuple[np.ndarray, np.ndarray]:
    """The padding `_apply_async_token_substitution` uses: real slots first, then every other slot of the input with
    a -1 source (left unchanged by the program)."""
    cur_np = np.asarray(cur, dtype=np.int32)
    missing = np.ones(length, dtype=bool)
    missing[cur_np] = False
    cur_full = np.concatenate((cur_np, np.flatnonzero(missing).astype(np.int32)))
    src_full = np.pad(np.asarray(src, dtype=np.int32), (0, length - len(src)), mode="constant", constant_values=-1)
    return cur_full, src_full
