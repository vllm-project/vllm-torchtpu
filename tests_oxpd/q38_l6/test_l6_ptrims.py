# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever ptrims (runner/l6_ptrims.py, H0-2 PREP-TRIMS). Host-only: runs in a CDK CPU pod.

Every stream test drives the real ``TPUModelRunner._prepare_inputs`` (and the
real ``_modify_prev_results`` / ``_update_placeholder``) through the CARRY
helper's harness (l6_carry_harness.py) twice from the same seed, once with the
lever unset and once with ``VLLM_TPU_L6_PREP_TRIMS=1``, and compares at every
step, byte for byte:

* every output that leaves ``_prepare_inputs`` for the device (the harness
  digests: metadata context, positions, substituted input ids and drafts,
  logits / substitution indices, spec metadata, the adapter's fused input prep);
* every host array the trims touch or feed: ``positions_np``,
  ``input_ids_cpu``, ``seq_lens_np``, ``query_start_loc_np``,
  ``mamba_state_indices_cpu``, the combined request distribution, the mamba
  slot map and free list;
* with STEADY + CARRY on in both runs, the carry state the step leaves;
* the compiled-program shape keys used (fused and carry programs) and the
  upload count per step (the trims change no program and no upload).

Streams mix admissions, finishes, preemptions with resumption, batch-order
changes, schedule changes (a partial draft, a non-spec step, a prompt tail),
remapped and discarded previous results, flushes and DP padding.
"""

from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import decode_carry, l6_ptrims, tpu_runner

ENV_KEYS = (
    "VLLM_TPU_L6_PREP_TRIMS",
    "VLLM_TPU_L6_PREP_TRIMS_CHECK",
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_STEADY_STEP",
    "VLLM_TPU_L6_STEADY_STEP_CHECK",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_STEPS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_AB_CONFIRM",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)
SERVED_PADS = (16, 32, 64, 128, 256, 512)
SHAPES = {
    # served decode bucket: max_num_seqs 64, 11 running requests
    "s11_served": dict(nreq=11, max_num_reqs=64, paddings=SERVED_PADS, mamba=True, comb=True),
    # 25 running requests, request bucket 64
    "s25_served": dict(nreq=25, max_num_reqs=64, paddings=SERVED_PADS, mamba=True, comb=True),
    # three requests, small buckets
    "b3_padded": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    # no compact mamba slots (pooled cache): the slot trim is not reached
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    # no GDN read offsets: the combined distribution is not built
    "b6_nocomb": dict(nreq=6, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
}
STEPS = 120
CARRY_ON = {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"}


@pytest.fixture
def env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = H.install_eager_programs()
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)  # one wall-clock window
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    errors: list[str] = []
    monkeypatch.setattr(l6_ptrims.logger, "error", lambda fmt, *a: errors.append(fmt % a))

    def lever(value=None, check=None, **extra):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        if value is not None:
            monkeypatch.setenv("VLLM_TPU_L6_PREP_TRIMS", value)
        if check is not None:
            monkeypatch.setenv("VLLM_TPU_L6_PREP_TRIMS_CHECK", str(check))
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_ptrims._CHECK[0] = None
        try:
            from vllm_torchtpu.runner import l6_steady

            l6_steady._CHECK[0] = None
        except ImportError:
            pass

    yield NS(lever=lever, errors=errors, monkeypatch=monkeypatch)
    undo()
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_ptrims._CHECK[0] = None


# ---------------------------------------------------------------- the stream


class PSim(H.Sim):
    """H.Sim plus preemption / resumption (as _update_states invalidates them),
    a widened token table, and host-array snapshots of every step."""

    def __init__(self, runner, **kw):
        self.parked: list = []
        self._resume = None
        self.seen: dict[str, int] = {}
        super().__init__(runner, **kw)

    def _invalidate(self, rid):
        """tpu_runner._update_states for a preempted / resumed request."""
        r = self.r
        r._mamba_state_pos.pop(rid, None) if hasattr(r, "_mamba_state_pos") else None
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
        self.computed[rid] = max(1, comp - 5)  # recompute a 5-token tail
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
        del self.computed[rid]  # r.requests keeps it (preempted, not finished)

    def widen(self):
        """The token table reallocated wider (same contents): the row stride changes."""
        ib = self.r.input_batch
        old = ib.token_ids_cpu
        new = np.zeros((old.shape[0], old.shape[1] + 64), dtype=old.dtype)
        new[:, : old.shape[1]] = old
        t = torch.from_numpy(new)
        ib.token_ids_cpu = new
        ib.token_ids_cpu_tensor = t

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
        rest = tuple(e for e in ev if e not in ("preempt", "resume", "widen"))
        return super().step(rest)


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
    }
    st = r.__dict__.get("_l6_carry_state")
    if st is not None:
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
    for name, p in (("swap", 0.08), ("partial", 0.05), ("nonspec", 0.03), ("discard", 0.04),
                    ("remap", 0.05), ("skew", 0.03), ("flush", 0.02), ("pad", 0.03)):
        if rng.random() < p:
            ev.append(name)
    if "nonspec" in ev and ("join" in ev or "resume" in ev):
        ev.remove("nonspec")
    return tuple(ev)


def run_stream(shape, steps, *, seed, extra_events=None, on_step=None, chunk=None, rand=True):
    """One engine for `steps` steps; returns (per-step dicts, sim). `chunk`:
    cap the requests per _prepare_inputs call (num_reqs_max_model_len) and run
    the remaining requests as a second chunk, as execute_model's chunk loop."""
    r = H.make_runner(seed=seed, **{k: v for k, v in shape.items() if k != "pooled"})
    if chunk is not None:
        r.num_reqs_max_model_len = chunk
    sim = PSim(r, nreq=shape.get("nreq", 3), seed=seed, pooled=shape.get("pooled", False))
    if chunk is not None:
        # The adapter digest models one chunk per step; the host snapshots of both chunks are compared.
        sim._adapter = lambda ctx, ids, n: {}
    rng = np.random.default_rng(5000 + seed)
    progs: list = []
    saved = [(tpu_runner, "_postfwd_subst_fused"), (decode_carry, "carry_fused")]
    saved = [(m, a, getattr(m, a)) for m, a in saved]
    for name, (mod, attr, fn) in zip(("fused", "carry"), saved):

        def rec(*a, _fn=fn, _name=name):
            progs.append((_name, tuple((tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else t
                                       for t in a)))
            return _fn(*a)

        setattr(mod, attr, rec)
    orig = r._prepare_inputs
    snaps: list = []

    def prep(so, start, nd, nw):
        out = orig(so, start, nd, nw)
        snaps.append(snapshot(r))
        if chunk is not None and r.input_batch.num_reqs > chunk:
            orig(so, chunk, nd, nw)
            snaps.append(snapshot(r))
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
            snaps.clear()
            d = sim.step(ev)
            for i, snap in enumerate(snaps):
                d.update({f"{k}#{i}": v for k, v in snap.items()})
            d["progs"] = repr(progs).encode()
            d["uploads"] = repr(sim.uploads[-1]).encode()
            rows.append(d)
    finally:
        for mod, attr, fn in saved:
            setattr(mod, attr, fn)
    return rows, sim


def both(env, shape, steps, *, seed, base_extra=None, value="1", check=None, **kw):
    """(stock rows, stock sim, trimmed rows, trimmed sim, trimmed counts, stock counts)."""
    env.lever(None, **(base_extra or {}))
    base, bsim = run_stream(shape, steps, seed=seed, **kw)
    base_counts = dict(l6_ab._state.get("counts", {}))
    env.lever(value, check=check, **(base_extra or {}))
    trim, tsim = run_stream(shape, steps, seed=seed, **kw)
    counts = dict(l6_ab._state.get("counts", {}))
    return base, bsim, trim, tsim, counts, base_counts


def first_difference(a, b):
    return H.first_difference(a, b)


def corrupt_token_cache(mp):
    """After every trimmed token step, shift the cached batched arange by one in
    place (a stale / corrupted cache entry the next hit reads)."""
    real = l6_ptrims.tokens

    def bad(r, *a):
        out = real(r, *a)
        for ent in r.__dict__.get("_l6_ptrims_idx", {}).values():
            ent[1][:] += 1
        return out

    mp.setattr(l6_ptrims, "tokens", bad)


def break_trim(mp, what):
    if what == "tokens":
        corrupt_token_cache(mp)
    elif what == "mamba":
        real_g = l6_ptrims._gather_slots
        mp.setattr(l6_ptrims, "_gather_slots", lambda r, ids: list(reversed(real_g(r, ids))))
    else:
        mp.setattr(l6_ptrims, "_comb_values", lambda d, w, n: (w, w, n, d, d, n))


# ---------------------------------------------------------------- trimmed == stock


@pytest.mark.parametrize("carry", ["alone", "steady_carry"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_trims_match_stock_over_random_streams(env, name, carry):
    shape = SHAPES[name]
    extra = CARRY_ON if carry == "steady_carry" else None
    base, bsim, trim, tsim, c, bc = both(env, shape, STEPS, seed=sorted(SHAPES).index(name) + 11,
                                         base_extra=extra)
    assert first_difference(base, trim) is None
    assert not any(k.startswith("ptrims.") for k in bc)
    # Engaged on every step: one trimmed _prepare_inputs per step.
    assert c.get("ptrims.steps") == STEPS
    assert c.get("ptrims.idx_hit", 0) > STEPS // 3
    assert c.get("ptrims.idx_build", 0) >= 2  # the schedule changed in the stream
    assert ("ptrims.mamba" in c) == shape["mamba"]
    assert ("ptrims.comb" in c) == shape["comb"]
    assert c.get("ptrims.mismatch", 0) == 0 and c.get("ptrims.check", 0) == 0
    if extra:
        # The carry state is compared on the steps it carried.
        assert c.get("carry.carried", 0) > STEPS // 5
        assert sum(1 for d in trim if "c.step#0" in d) > STEPS // 5


def test_streams_are_not_vacuous(env):
    """The comparison covers what it claims: every event kind occurs over the
    stream, and the mamba slots stay conserved (each admission / resumption took
    a slot from the free list through the trimmed gather's fallback)."""
    shape = SHAPES["s11_served"]
    base, bsim, trim, tsim, c, _ = both(env, shape, 240, seed=21, base_extra=CARRY_ON)
    assert first_difference(base, trim) is None
    for e in ("join", "resume", "leave", "preempt", "swap", "partial", "nonspec", "discard", "remap"):
        assert tsim.seen.get(e, 0) > 0, (e, tsim.seen)
    r = tsim.r
    assert len(r._mamba_slot_by_req_id) + len(r._free_mamba_slots) == 4 * r.max_num_reqs
    assert c.get("ptrims.mamba", 0) == 240


def test_two_chunk_steps_match_stock(env):
    """start_index > 0 (execute_model's chunk loop): two equal chunks of 4
    requests share every key field but the start index."""
    shape = dict(nreq=8, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True)
    env.lever(None)
    base, _ = run_stream(shape, 30, seed=3, chunk=4, rand=False)
    env.lever("1")
    trim, _ = run_stream(shape, 30, seed=3, chunk=4, rand=False)
    assert first_difference(base, trim) is None
    assert all("h.positions_np#1" in d for d in trim)


def test_widened_token_table_matches_stock(env):
    """The token table reallocated with a wider row (the stride in the key) on a
    steady schedule."""
    ev = {k: ("widen",) for k in (7, 19)}
    base, bsim, trim, tsim, c, _ = both(env, SHAPES["b3_padded"], 30, seed=4, extra_events=ev, rand=False)
    assert tsim.seen.get("widen") == 2
    assert first_difference(base, trim) is None


def test_lever_off_takes_the_stock_path(env):
    """Unset, '0', or no L6 lever at all: no trim runs and nothing is cached."""

    def boom(*a, **k):
        raise AssertionError("a trim ran with the lever off")

    for name in ("tokens", "mamba_slots", "comb"):
        env.monkeypatch.setattr(l6_ptrims, name, boom)
    for value, extra in ((None, None), ("0", None), (None, CARRY_ON), ("0", CARRY_ON)):
        env.lever(value, **(extra or {}))
        _, sim = run_stream(SHAPES["s11_served"], 20, seed=1)
        assert "_l6_ptrims_idx" not in sim.r.__dict__
        assert not any(k.startswith("ptrims.") for k in l6_ab._state.get("counts", {}))


def test_ab_windows_switch_the_lever(env):
    """ab:1 with step windows of 5: the trims run only in class-1 windows and
    every step equals stock."""
    calls: list[int] = []
    real = l6_ptrims.tokens

    def spy(*a, **k):
        calls.append(l6_ab._state["j"])
        return real(*a, **k)

    shape = SHAPES["b3_padded"]
    env.lever(None)
    base, _ = run_stream(shape, 40, seed=6)
    # l6_ab requires the lockstep confirm distance below the window (1 <= D < S).
    env.lever("ab:1", VLLM_TPU_L6_AB_STEPS="5", VLLM_TPU_L6_AB_CLASSES="2", VLLM_TPU_L6_AB_CONFIRM="2")
    env.monkeypatch.setattr(l6_ptrims, "tokens", spy)
    trim, sim = run_stream(shape, 40, seed=6)
    assert first_difference(base, trim) is None
    assert calls and all((j // 5) % 2 == 1 for j in calls), calls
    assert len(calls) >= 15


def test_check_mode_compares_and_finds_no_mismatch(env):
    shape = SHAPES["s11_served"]
    base, bsim, trim, tsim, c, _ = both(env, shape, 60, seed=8, check=1, base_extra=CARRY_ON)
    assert first_difference(base, trim) is None
    # Most steps compare the token indices, and every step the combined distribution and the slots.
    assert c.get("ptrims.check", 0) >= 2 * 60 + 60 // 2
    assert c.get("ptrims.mismatch", 0) == 0 and not env.errors
    assert not tsim.r.__dict__.get("_l6_ptrims_off", False)


@pytest.mark.parametrize("what", ["tokens", "mamba", "comb"])
def test_check_catches_a_broken_trim(env, what):
    """A broken trim under CHECK=1: the first wrong step is found, the stock
    values are applied, one ERROR is logged and the trims turn off, so the whole
    stream still equals stock."""
    break_trim(env.monkeypatch, what)
    base, bsim, trim, tsim, c, _ = both(env, SHAPES["s11_served"], 30, seed=9, check=1)
    assert first_difference(base, trim) is None
    assert c.get("ptrims.mismatch") == 1
    assert len(env.errors) == 1 and "CHECK mismatch" in env.errors[0]
    assert tsim.r._l6_ptrims_off is True


@pytest.mark.parametrize("what", ["tokens", "mamba", "comb"])
def test_broken_trim_diverges_without_check(env, what):
    """The same broken trims without CHECK make the stream differ from stock
    (the comparison has teeth for every trim)."""
    break_trim(env.monkeypatch, what)
    base, bsim, trim, tsim, c, _ = both(env, SHAPES["s11_served"], 30, seed=9)
    assert first_difference(base, trim) is not None


def test_check_steps_bound_the_check(env):
    """VLLM_TPU_L6_CHECK_STEPS=4 with CHECK=1: compares only in the first 4 worker steps."""
    env.lever("1", check=1, VLLM_TPU_L6_CHECK_STEPS="4")
    seen: list[int] = []
    real = l6_ptrims._stock_tokens

    def spy(*a, **k):
        seen.append(l6_ab._state["j"])
        return real(*a, **k)

    env.monkeypatch.setattr(l6_ptrims, "_stock_tokens", spy)
    run_stream(SHAPES["b3_padded"], 20, seed=12, rand=False)
    assert seen == [1, 2, 3, 4], seen


# ---------------------------------------------------------------- registration / staging


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["ptrims"] == "VLLM_TPU_L6_PREP_TRIMS"
    for name in ("VLLM_TPU_L6_PREP_TRIMS", "VLLM_TPU_L6_PREP_TRIMS_CHECK"):
        assert name in envs.environment_variables
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    # The compile-key registration test carries both rows.
    reg = pathlib.Path(__file__).resolve().parents[1] / "compilation" / "test_tpu_compile_key_registration.py"
    text = reg.read_text()
    assert '("VLLM_TPU_L6_PREP_TRIMS", "1", "")' in text
    assert '("VLLM_TPU_L6_PREP_TRIMS_CHECK", "200", 0)' in text


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
