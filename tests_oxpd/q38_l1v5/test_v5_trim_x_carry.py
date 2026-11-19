# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): DECODE-TRIM's held-back requests (core/decode_trim.py, q38 L1-C48) x the L6
lever carry (runner/decode_carry.py, VLLM_TPU_L6_DECODE_CARRY). Host-only; run in a CDK CPU pod.

DECODE-TRIM makes the scheduler skip a running decode request for one engine step (never two in a row). On the TPU
runner such a request leaves the persistent batch for that step (its request state and its GDN slot stay) and re-enters
the next step, absent from the previous step's async results, with the scheduler's position already corrected for
its last rejections. CARRY reuses the previous step's corrected device tensors when the chunk's batch is unchanged.
The cases to test are the held step and the re-entry: the batch looks steady except for the one request, and on a
re-entry at the same row the request ids can equal those of the step before the hold.

Harness: L6's carry harness (tests/q38_l6/l6_carry_harness.py: the real TPUModelRunner._prepare_inputs,
_modify_prev_results and _update_placeholder, device side simulated), extended with two batch edits: hold (the request
leaves the batch, keeping its runner state, computed position and last rejected count) and rejoin (the same request id
re-enters at the end of the batch, position corrected by its last rejections, no rejection in flight). Every scenario
runs once with the lever unset (the upload path) and once with it on, from the same seed, and every output that leaves
_prepare_inputs for the device must be equal byte for byte; CARRY must upload (not carry) on each held and re-entry
step and carry again afterwards. Positive control: a carry forced on the re-entry step changes the outputs (or fails).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "q38_l6"))
import l6_carry_harness as H  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.runner import decode_carry  # noqa: E402

ENV_KEYS = (
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
    "VLLM_TPU_DECODE_TRIM",
    "VLLM_TPU_DECODE_TRIM_DRAFT_STASH",
)
SHAPE = dict(nreq=4, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True)


class TrimSim(H.Sim):
    """L6's Sim with DECODE-TRIM's hold-back and re-entry as batch edits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held: list = []

    def hold(self, j: int) -> str:
        r, ib = self.r, self.r.input_batch
        rid = ib.req_ids[j]
        row = (ib.token_ids_cpu[j].copy(), int(ib.num_tokens_no_spec[j]), int(ib.num_prompt_tokens[j]))
        self.held.append((rid, row, self.computed[rid], self.last_rej.get(rid, 0)))
        last = ib.num_reqs - 1
        if j != last:
            self._move(last, j)
        ib.req_ids.pop()
        ib.num_reqs -= 1
        del ib.req_id_to_index[rid]  # r.requests[rid] and its GDN slot stay (the runner keeps unscheduled requests)
        return rid

    def rejoin(self) -> str:
        rid, (tok, ntok, nprompt), comp, lrej = self.held.pop(0)
        ib = self.r.input_batch
        i = ib.num_reqs
        ib.req_ids.append(rid)
        ib.req_id_to_index[rid] = i
        ib.num_reqs += 1
        ib.token_ids_cpu[i] = tok
        ib.num_tokens_no_spec[i] = ntok
        ib.num_prompt_tokens[i] = nprompt
        self.computed[rid] = comp - lrej  # the scheduler learned its last step's rejections while it was held
        self.last_rej[rid] = 0  # nothing of it was in flight during the held step
        return rid

    def step(self, events=()):
        rest = []
        for e in events:
            if e == "rejoin":
                self.rejoin()
            elif e == "hold0":
                self.hold(0)
            elif e == "holdL":
                self.hold(self.r.input_batch.num_reqs - 1)
            else:
                rest.append(e)
        return super().step(tuple(rest))


def run(steps, *, seed, events, on_step=None):
    r = H.make_runner(seed=seed, **SHAPE)
    sim = TrimSim(r, nreq=SHAPE["nreq"], seed=seed)
    digests = []
    for _ in range(steps):
        if on_step is not None:
            on_step(sim)
        digests.append(sim.step(events.get(sim.n, ())))
    return digests, sim


@pytest.fixture
def env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = H.install_eager_programs()
    notes: list = []
    cur = {"step": -1}
    real_note = decode_carry._note

    def note(key):
        notes.append((cur["step"], key))
        real_note(key)

    monkeypatch.setattr(decode_carry, "_note", note)

    def lever(value=None):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        if value is not None:
            monkeypatch.setenv("VLLM_TPU_L6_DECODE_CARRY", value)
        l6_ab._resolve()

    yield NS(lever=lever, notes=notes, cur=cur, monkeypatch=monkeypatch)
    undo()
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def both(env, steps, *, seed, events, hook=None):
    env.lever(None)
    base, _ = run(steps, seed=seed, events=events)
    env.lever("1")
    env.notes.clear()

    def on_step(sim):
        env.cur["step"] = sim.n
        if hook is not None:
            hook(sim)

    carry, csim = run(steps, seed=seed, events=events, on_step=on_step)
    d: dict = {}
    for s, k in env.notes:
        d.setdefault(s, set()).add(k)
    return base, carry, csim, d


# name -> (events, steps that must upload, first step from which every step must carry again, steps)
SCENARIOS = {
    "hold_row0_rejoin": ({6: ("hold0",), 7: ("rejoin",)}, (6, 7), 9, 14),
    "hold_last_rejoin_same_row": ({6: ("holdL",), 7: ("rejoin",)}, (6, 7), 9, 14),
    "rotation_three": ({6: ("hold0",), 7: ("rejoin", "hold0"), 8: ("rejoin", "hold0"), 9: ("rejoin",)},
                       (6, 7, 8, 9), 11, 16),
    "periodic_trim": ({s: ev for s0 in (5, 10, 15, 20) for s, ev in ((s0, ("holdL",)), (s0 + 1, ("rejoin",)))},
                      (5, 6, 10, 11, 15, 16, 20, 21), 23, 26),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_carry_matches_upload_path_across_holds_and_reentries(env, name):
    events, upload_steps, resume, steps = SCENARIOS[name]
    base, carry, csim, d = both(env, steps, seed=21, events=events)
    assert H.first_difference(base, carry) is None, H.first_difference(base, carry)
    assert not csim.held
    assert all("carried" in d.get(s, set()) for s in range(1, min(upload_steps))), {s: d.get(s) for s in range(8)}
    for s in upload_steps:
        assert "carried" not in d.get(s, set()) and "uploaded" in d.get(s, set()), (s, d.get(s))
    assert all("carried" in d.get(s, set()) for s in range(resume, steps)), {s: d.get(s) for s in range(resume, steps)}
    whys = sorted({k for s in upload_steps for k in d.get(s, ()) if k.startswith("why_")})
    print(f"TRIMxCARRY {name} upload reasons {whys} carried {sum('carried' in v for v in d.values())}/{steps}")


def test_reentry_at_the_same_row_is_refused_although_request_ids_match(env):
    """hold the last row at 6, re-enter it at the same row at 7: the step-7 batch has the step-5 request ids; CARRY
    must still upload at 7 (its state is from step 6 or stale), and the outputs match the upload path."""
    events = {6: ("holdL",), 7: ("rejoin",)}
    ids = {}

    def hook(sim):
        ids[sim.n] = tuple(sim.r.input_batch.req_ids[: sim.r.input_batch.num_reqs])

    base, carry, csim, d = both(env, 10, seed=22, events=events, hook=hook)
    assert H.first_difference(base, carry) is None
    # ids are read before the step's batch edits: ids[8] is the batch step 7 ran
    assert ids[8] == ids[6] and ids[7] != ids[6], ids
    assert "carried" not in d[7] and ({"why_batch", "why_stale", "why_map"} & d[7]), d[7]


def test_forced_carry_on_reentry_is_caught(env):
    """Positive control: a carry forced on the re-entry step (the host check overridden for that step only) does not
    reproduce the upload path (outputs differ, or the step fails)."""
    events = {6: ("hold0",), 7: ("rejoin",)}
    real = decode_carry._check

    def forced(runner, *a, **k):
        st, why = real(runner, *a, **k)
        if st is None and env.cur["step"] == 7:
            prev = runner.__dict__.get("_l6_carry_state")
            if prev is not None:
                prev.map_same = False
                return prev, ""
        return st, why

    env.lever(None)
    base, _ = run(10, seed=23, events=events)
    env.lever("1")
    env.monkeypatch.setattr(decode_carry, "_check", forced)
    try:
        carry, _ = run(10, seed=23, events=events, on_step=lambda sim: env.cur.__setitem__("step", sim.n))
    except Exception as e:  # the forced carry cannot even run: also a refusal that matters
        print(f"TRIMxCARRY forced carry failed: {type(e).__name__}: {e}")
        return
    assert H.first_difference(base, carry) is not None
