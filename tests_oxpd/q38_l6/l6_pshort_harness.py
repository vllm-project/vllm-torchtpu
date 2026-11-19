# SPDX-License-Identifier: Apache-2.0
"""q38 L6 levers clean (CARRY-LEAN) / pshort (PREP-SHORT): the CARRY harness
(l6_carry_harness.py) with what the two levers read added.

* ``SSim`` (after the ptrims test's PSim) adds preemption with resumption and a
  widened token table to H.Sim, and fills the SchedulerOutput fields STEADY's
  reorder verdict reads (new / finished / resumed / preempted requests).
* ``run_stream`` wraps ``_prepare_inputs`` as ``execute_model`` calls it:
  STEADY's ``drop`` / ``reorder_and_count`` first (the full reorder and chunk
  count are the harness's pure-decode values: every request in one windowed
  segment, one chunk), then ``_prepare_inputs`` with the counts the reorder
  returned, then execute_model's ``spec_decode_metadata.draft_token_ids =
  <fused drafts>``. Per step it records the H.Sim digests, the spec metadata's
  ``draft_token_ids`` at return, every host mirror and the carry state
  (``snapshot``), the compiled-program shapes called and the upload count, and
  the CARRY decisions noted (``carried`` / ``uploaded`` / ``why_*``).
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import decode_carry, l6_steady, tpu_runner

SERVED_PADS = (16, 32, 64, 128, 256, 512)
SHAPES = {
    # served decode bucket (max_num_seqs 64) at 4 / 11 / 25 running requests
    "s04_served": dict(nreq=4, max_num_reqs=64, paddings=SERVED_PADS, mamba=True, comb=True),
    "s11_served": dict(nreq=11, max_num_reqs=64, paddings=SERVED_PADS, mamba=True, comb=True),
    "s25_served": dict(nreq=25, max_num_reqs=64, paddings=SERVED_PADS, mamba=True, comb=True),
    # pooled cache: no compact mamba slots
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    # no GDN read offsets: no combined distribution (the short path refuses: config)
    "b6_nocomb": dict(nreq=6, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
}
# The one host mirror the short path leaves stale on a short step, with its readers pinned by
# tests/q38_l6/test_l6_pshort.py::test_skipped_mirrors_have_no_new_reader: input_ids_cpu (uploads,
# which StepCtx.materialize precedes; the spec manager's deferred host drafts, which the short path
# does not build; the PLE host ids, never under async scheduling).
SKIPPED_MIRRORS = ("h.input_ids_cpu",)


class SSim(H.Sim):
    def __init__(self, runner, **kw):
        self.parked: list = []
        self._resume = None
        self.seen: dict[str, int] = {}
        self.cur_events: tuple = ()
        super().__init__(runner, **kw)

    def _invalidate(self, rid):
        """tpu_runner._update_states for a preempted / resumed request."""
        r = self.r
        slot = r._mamba_slot_by_req_id.pop(rid, None)
        if slot is not None:
            r._free_mamba_slots.append(slot)
        r._mamba_offset_seeded.discard(rid)
        pre = r._pre_async_results
        if pre is not None:
            pre.req_id_to_index_copy.pop(rid, None)
            pre.request_seq_lens = [x for x in pre.request_seq_lens if x[3] != rid]

    def _add(self, computed):
        if self._resume is None:
            return super()._add(computed)
        rid, comp, row, prompt, ntns, req = self._resume
        self._resume = None
        r, ib = self.r, self.r.input_batch
        self._invalidate(rid)
        i = ib.num_reqs
        ib.req_ids.append(rid)
        ib.req_id_to_index[rid] = i
        ib.num_reqs += 1
        ib.num_prompt_tokens[i] = prompt
        ib.num_tokens_no_spec[i] = ntns
        ib.token_ids_cpu[i] = row
        r.requests[rid] = req
        self.computed[rid] = max(1, comp - 5)
        return rid

    def preempt(self, j):
        r, ib = self.r, self.r.input_batch
        rid = ib.req_ids[j]
        self._invalidate(rid)
        self.parked.append((rid, self.computed[rid], ib.token_ids_cpu[j].copy(),
                            int(ib.num_prompt_tokens[j]), int(ib.num_tokens_no_spec[j]), r.requests[rid]))
        last = ib.num_reqs - 1
        if j != last:
            self._move(last, j)
        ib.req_ids.pop()
        ib.num_reqs -= 1
        del ib.req_id_to_index[rid]
        del self.computed[rid]

    def widen(self):
        ib = self.r.input_batch
        old = ib.token_ids_cpu
        new = np.zeros((old.shape[0], old.shape[1] + 64), dtype=old.dtype)
        new[:, : old.shape[1]] = old
        ib.token_ids_cpu = new
        ib.token_ids_cpu_tensor = torch.from_numpy(new)

    def step(self, events=()):
        ev = list(events)
        for e in ev:
            self.seen[e] = self.seen.get(e, 0) + 1
        if "preempt" in ev:
            self.preempt(int(self.rng.integers(0, self.r.input_batch.num_reqs)))
        if "resume" in ev and self.parked:
            self._resume = self.parked.pop(0)
            ev.append("join")
        if "widen" in ev:
            self.widen()
        self.cur_events = tuple(ev)
        rest = tuple(e for e in ev if e not in ("preempt", "resume", "widen"))
        return super().step(rest)


def make_runner(shape, seed):
    r = H.make_runner(seed=seed, **{k: v for k, v in shape.items() if k != "pooled"})
    # execute_model's reorder + chunk count on the harness's pure decode / verify batch:
    # every request in the windowed segment, one chunk (H.Sim passes (0, n) itself).
    r._reorder_batch_for_rpa = lambda so: (0, r.input_batch.num_reqs)
    r._count_input_chunks = lambda so: (1, r.input_batch.num_reqs)
    return r


def snapshot(r) -> dict[str, bytes]:
    d = {
        "h.positions_np": H._b(r.positions_np),
        "h.input_ids_cpu": H._b(r.input_ids_cpu),
        "h.seq_lens_np": H._b(r.seq_lens_np),
        "h.query_start_loc_np": H._b(r.query_start_loc_np),
        "h.mamba_state_indices_cpu": H._b(r.mamba_state_indices_cpu),
        "h.combined_distribution": H._b(r._combined_request_distribution_cpu),
        "h.slot_map": repr(sorted(r._mamba_slot_by_req_id.items())).encode(),
        "h.free_slots": repr(list(r._free_mamba_slots)).encode(),
        "h.seeded": repr(sorted(r._mamba_offset_seeded)).encode(),
        "h.layout_plan": repr(r._last_sequence_layout_plan).encode(),
    }
    st = r.__dict__.get("_l6_carry_state")
    if st is not None:
        # (src_tag is a process-unique counter, different in every run.)
        for f in ("step", "req_ids", "nsched_b", "target", "total", "padded_total", "step_delta", "map",
                  "map_same"):
            d[f"c.{f}"] = repr(getattr(st, f)).encode()
        for f in ("nsched", "S", "seq_pad", "pos", "pos_pad", "seq_lens", "positions", "input_ids",
                  "seq_idx_dev", "pos_idx_dev", "qsl_dev"):
            d[f"c.{f}"] = H._b(getattr(st, f))
        d["c.subst"] = b"".join(H._b(x) for x in st.subst)
    else:
        d["c.none"] = b"1"
    return d


def random_events(rng, sim) -> tuple:
    n = sim.r.input_batch.num_reqs
    cap = sim.r.max_num_reqs - 1
    ev = []
    u = rng.random()
    if u < 0.07 and n < cap:
        ev.append("join")
    elif u < 0.12 and sim.parked and n < cap:
        ev.append("resume")
    elif u < 0.19 and n > 2:
        ev.append("leave")
    elif u < 0.24 and n > 2:
        ev.append("preempt")
    for name, p in (("swap", 0.06), ("partial", 0.04), ("nonspec", 0.03), ("discard", 0.04),
                    ("remap", 0.05), ("skew", 0.03), ("flush", 0.02), ("pad", 0.04)):
        if rng.random() < p:
            ev.append(name)
    if "nonspec" in ev and ("join" in ev or "resume" in ev):
        ev.remove("nonspec")
    return tuple(ev)


def _so_fields(so, sim):
    ev = sim.cur_events
    rids = list(sim.r.input_batch.req_ids)
    so.scheduled_new_reqs = [NS(req_id=rids[-1])] if "join" in ev and "resume" not in ev else []
    so.finished_req_ids = {"gone"} if "leave" in ev else set()
    so.preempted_req_ids = {"pre"} if "preempt" in ev else set()
    so.scheduled_cached_reqs = NS(resumed_req_ids={rids[-1]} if "resume" in ev else set())


def run_stream(shape, steps, *, seed, rand=True, extra_events=None, on_step=None, around=None,
               notes=None):
    """One engine for `steps` steps; returns (per-step dicts, sim). `around(sim, call, so,
    start, nd, nw)`: wraps one _prepare_inputs call (call(so, start, nd, nw) runs it); `notes`:
    a list the CARRY notes of each step are appended to as (sim step, key)."""
    r = make_runner(shape, seed)
    sim = SSim(r, nreq=shape.get("nreq", 3), seed=seed, pooled=shape.get("pooled", False))
    rng = np.random.default_rng(5000 + seed)
    progs: list = []
    saved = [(tpu_runner, "_postfwd_subst_fused"), (decode_carry, "carry_fused"), (decode_carry, "_note")]
    saved = [(m, a, getattr(m, a)) for m, a in saved]
    for name, (mod, attr, fn) in zip(("fused", "carry"), saved[:2]):

        def rec(*a, _fn=fn, _name=name):
            progs.append((_name, tuple((tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else t
                                       for t in a)))
            return _fn(*a)

        setattr(mod, attr, rec)
    real_note = saved[2][2]
    step_notes: list = []

    def note(key):
        step_notes.append(key)
        if notes is not None:
            notes.append((sim.n, key))
        real_note(key)

    decode_carry._note = note
    orig = r._prepare_inputs
    extra: dict = {}

    def call(so, start, nd, nw):
        return orig(so, start, nd, nw)

    def prep(so, start, nd, nw):
        _so_fields(so, sim)
        if l6_ab.ACTIVE and not l6_steady.on(r):
            l6_steady.drop(r)
        if l6_ab.ACTIVE and l6_steady.on(r):
            nd, nw, _, _ = l6_steady.reorder_and_count(r, so)
        out = around(sim, call, so, start, nd, nw) if around is not None else call(so, start, nd, nw)
        md = out[7]
        extra["spec_draft_token_ids_at_return"] = H._b(None if md is None else md.draft_token_ids)
        extra["spec_md_fields"] = b"" if md is None else b"".join(
            H._b(getattr(md, f)) for f in ("draft_lengths", "draft_lengths_cpu", "target_logits_indices",
                                            "bonus_logits_indices", "final_logits_indices", "segment_ids",
                                            "group_indices"))
        extra.update(snapshot(r))
        fused = r.__dict__.get("_postfwd_fused_out")
        if fused is not None and md is not None and fused[0] == start:
            md.draft_token_ids = fused[2]  # execute_model
        return out

    r._prepare_inputs = prep
    rows = []
    try:
        for _ in range(steps):
            if on_step is not None:
                on_step(sim)
            ev = random_events(rng, sim) if rand else ()
            ev += tuple((extra_events or {}).get(sim.n, ()))
            progs.clear()
            step_notes.clear()
            extra.clear()
            d = sim.step(ev)
            d.update(extra)
            d["progs"] = repr(progs).encode()
            d["uploads"] = repr(sim.uploads[-1]).encode()
            d["carry_notes"] = repr(sorted(step_notes)).encode()
            rows.append(d)
    finally:
        for mod, attr, fn in saved:
            setattr(mod, attr, fn)
    return rows, sim


def first_difference(a, b, skip=()):
    if len(a) != len(b):
        return ("len", len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        kx = {k for k in x if k not in skip}
        ky = {k for k in y if k not in skip}
        if kx != ky:
            return (i, "keys", sorted(kx ^ ky))
        for k in sorted(kx):
            if x[k] != y[k]:
                return (i, k)
    return None
