# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): DECODE-TRIM's held-back requests (core/decode_trim.py, q38 L1-C48) x the L6
lever engalloc (core/l6_engine_alloc.py, VLLM_TPU_L6_ENGINE_ALLOC_FAST). Host-only; run in a CDK CPU pod.

DECODE-TRIM makes the scheduler skip some running decode requests for one engine step (never two in a row): a held-back
request makes no allocate_slots call that step, keeps its KV blocks and RUNNING status, and re-enters the next step
with its previous step's rejections rolled back and the -1 placeholder drafts of vLLM's AsyncScheduler. engalloc reuses
a per-request certificate from the request's previous stock call to skip later quiet calls. The case to test is the
re-entry: the certificate was made two steps earlier and the pool changed in between.

Harness: L6's own engalloc stream (tests/q38_l6/test_l6_engine_alloc.py ``Stream``, PD-decode-shaped, async, MTP k=3,
preemptions, prefix hits, landings, finishes, evictions), with a trim rule layered on its scheduling loop: each step,
from an RNG independent of the stream's, a running decode request is held back with probability HOLD unless it was
held in the previous step (the trim's one-step starvation bound) and at least two requests run. The stock and the fast
twin draw the same trim decisions, so after every step the full KV state digests must be equal (the stream's own
equality check), check mode (every predicted-quiet call also run stock) must count no mismatch, and the composition
must be exercised (holds, re-entries, and re-entry calls that the fast path served). Positive control: L6's broken
certificate (one block of slack) under the same trim rule diverges without the check and is caught with it.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "q38_l6"))
import test_l6_engine_alloc as E  # noqa: E402  (L6's harness module; none of its tests is re-exported here)

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.core import l6_engine_alloc as ea_mod  # noqa: E402

HOLD = 0.3


@pytest.fixture(autouse=True)
def lever(monkeypatch):
    """engalloc on (uniform), as L6's own fixture; module state restored after."""
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_DECODE_TRIM"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(E.ENV, "1")
    l6_ab._resolve()
    ea_mod._INSTANCES.clear()
    yield monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    ea_mod._INSTANCES.clear()


class TrimStream(E.Stream):
    """L6's engalloc stream with DECODE-TRIM's hold-back on its scheduling loop."""

    def __init__(self, *args, seed=0, hold=HOLD, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.trim_rnd = random.Random(9000 + seed)
        self.hold = hold
        self.held_prev: set = set()
        self.held_now: set = set()
        self.n_hold = 0
        self.n_reentry = 0
        self.n_reentry_fast = 0
        self.max_consec = 0
        self._consec: dict = {}

    def step(self):
        self.held_prev, self.held_now = self.held_now, set()
        super().step()
        for rid in list(self._consec):
            if rid not in self.held_now:
                del self._consec[rid]

    def _schedule_one(self, r):
        rid = r.request_id
        decode = r.num_computed_tokens >= r.num_tokens - 1 and r.num_output_tokens > 0
        if decode and rid not in self.held_prev and len(self.running) >= 2 and self.trim_rnd.random() < self.hold:
            # held back: no allocate_slots this step (DECODE-TRIM's hook returns 0 for it)
            self.held_now.add(rid)
            self.n_hold += 1
            self._consec[rid] = self._consec.get(rid, 0) + 1
            self.max_consec = max(self.max_consec, self._consec[rid])
            self.trace.append(("held", rid))
            return None
        if rid not in self.held_prev:
            return super()._schedule_one(r)
        self.n_reentry += 1
        before = self.ea.n["fast"] if self.ea is not None else 0
        out = super()._schedule_one(r)
        if self.ea is not None and self.ea.n["fast"] > before:
            self.n_reentry_fast += 1
        return out


def _pair(steps, **kw):
    base = TrimStream(fast=False, **kw)
    db = base.run(steps)
    fast = TrimStream(fast=True, **kw)
    df = fast.run(steps)
    return base, db, fast, df


@pytest.mark.parametrize("kind,twins,seed", [("fine", True, 1), ("fine", False, 2), ("skew", True, 3),
                                             ("coarse", True, 4), ("fine", True, 5)])
def test_fast_path_matches_stock_with_trim_holds(kind, twins, seed):
    base, db, fast, df = _pair(500, kind=kind, twins=twins, seed=seed)
    assert E._first_diff(db, df) is None, E._first_diff(db, df)
    assert base.calls == fast.calls
    assert (base.n_hold, base.n_reentry) == (fast.n_hold, fast.n_reentry)
    # not vacuous: the trim engaged, requests re-entered, the fast path served calls, re-entries included
    assert fast.n_hold >= 40 and fast.n_reentry >= 30, (fast.n_hold, fast.n_reentry)
    assert fast.max_consec == 1, fast.max_consec
    assert fast.ea.n["fast"] > 100 and fast.ea.n["mismatch"] == 0, fast.ea.n
    assert fast.seen["preempt"] > 0 and fast.seen["finish"] > 0, fast.seen
    print(f"TRIMxENGALLOC {kind} twins={twins} seed={seed} holds={fast.n_hold} reentries={fast.n_reentry} "
          f"reentry_fast={fast.n_reentry_fast} ea={fast.ea.n} seen={fast.seen}")


def test_reentries_take_the_fast_path_somewhere():
    """Across seeds, some re-entry allocate_slots call is served by a certificate made before the held step."""
    total = 0
    for seed in (11, 12, 13):
        _, db, fast, df = _pair(400, kind="fine", twins=True, seed=seed)
        assert db == df
        total += fast.n_reentry_fast
    print(f"TRIMxENGALLOC reentry calls served by the fast path: {total}")
    assert total > 0


def test_check_mode_under_trim_finds_no_mismatch():
    base = TrimStream(fast=False, twins=True, seed=7)
    db = base.run(400)
    st = TrimStream(fast=True, twins=True, seed=7, check=1)
    df = st.run(400)
    assert db == df
    assert st.n_reentry >= 20 and st.ea.n["check"] > 20 and st.ea.n["mismatch"] == 0, (st.n_reentry, st.ea.n)


def test_broken_certificate_under_trim_diverges_without_check():
    def hook(ea):
        E._loose_need(ea)
        ea._mismatch = lambda *a, **k: None

    base = TrimStream(fast=False, twins=True, seed=8)
    db = base.run(400)
    st = TrimStream(fast=True, twins=True, seed=8, cert_hook=hook)
    df = st.run(400)
    assert E._first_diff(db, df) is not None


def test_broken_certificate_under_trim_is_caught_by_check(lever):
    errors = []
    lever.setattr(ea_mod.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    base = TrimStream(fast=False, twins=True, seed=8)
    db = base.run(400)
    st = TrimStream(fast=True, twins=True, seed=8, check=1, cert_hook=E._loose_need)
    df = st.run(400)
    assert db == df
    assert st.ea.n["mismatch"] == 1 and st.ea.off
    assert any("MISMATCH" in e for e in errors)
