"""Decode-only steps between prefill chunks (core/decode_interleave.py).

The patched vLLM ``Scheduler`` / ``AsyncScheduler`` is stepped on the host
with a fake model runner (one step in flight when async), the same harness
as test_prefill_srpt.py.
"""

import os

import pytest

from tests.test_prefill_srpt import BUDGET, _request, _run, _scheduler
from vllm_torchtpu import envs
from vllm_torchtpu.core import decode_interleave, prefill_srpt
from vllm_torchtpu.core.decode_interleave import DecodeInterleavePolicy


def _install(monkeypatch, steps=0, adaptive=False, finish=0, finish_max=32, srpt=False):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler

    for name in ("get_cached_block", "_insert_block_hash"):
        monkeypatch.setattr(BlockPool, name, getattr(BlockPool, name))
    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    monkeypatch.setattr(Scheduler, "_tpu_prefill_srpt", False, raising=False)
    monkeypatch.setattr(Scheduler, "_tpu_decode_interleave", False, raising=False)
    monkeypatch.setenv("TPU_SCHED_PREFILL_SRPT", "1" if srpt else "0")
    monkeypatch.setenv("TPU_SCHED_PREFILL_SRPT_MAX_HOLD_STEPS", "0")
    monkeypatch.setenv("TPU_SCHED_DECODE_INTERLEAVE_STEPS", str(steps))
    monkeypatch.setenv("TPU_SCHED_DECODE_INTERLEAVE_ADAPTIVE", "1" if adaptive else "0")
    monkeypatch.setenv("TPU_SCHED_DECODE_FINISH_TOKENS", str(finish))
    monkeypatch.setenv("TPU_SCHED_DECODE_FINISH_MAX_STEPS", str(finish_max))
    prefill_srpt.patch_scheduler_prefill_srpt()
    decode_interleave.patch_scheduler_decode_interleave()


def _invariants(sched):
    mnbt = sched.max_num_scheduled_tokens
    mns = sched.max_num_running_reqs

    def check(s, out):
        assert s.max_num_scheduled_tokens == mnbt
        assert s.max_num_running_reqs == mns
        assert sum(out.num_scheduled_tokens.values()) <= BUDGET
        # a deferred step is never empty: an empty step has no prefill left
        if not out.num_scheduled_tokens:
            assert not s.waiting
            assert not any(
                r.num_computed_tokens < r.num_prompt_tokens for r in s.running
            )

    return check


def _trace(monkeypatch, async_scheduling, arrivals, n=40, **kw):
    _install(monkeypatch, **kw)
    sched = _scheduler(async_scheduling)
    return sched, _run(sched, arrivals, n, async_scheduling, _invariants(sched))


def _scenario():
    """A decode ``d`` runs; a 2000-token prompt ``L`` starts chunking at
    step 1 (budget 256 -> ~8 chunks)."""
    return {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 2000)],
    }


def _l_steps(trace, prompt=2000):
    """Steps that carry a chunk of L's prompt (not its later decodes)."""
    out, total = [], 0
    for i, s in enumerate(trace):
        if "L" in s and total < prompt:
            out.append(i)
            total += s["L"]
    return out


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_off_is_the_unpatched_scheduler(monkeypatch, async_scheduling):
    from vllm.v1.core.sched.scheduler import Scheduler

    _install(monkeypatch)
    assert not getattr(Scheduler, "_tpu_decode_interleave", False)
    _, a = _trace(monkeypatch, async_scheduling, _scenario())
    # baseline: L chunks every step from its admission until done
    ls = _l_steps(a)
    assert ls == list(range(ls[0], ls[0] + len(ls)))


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_k_decode_only_steps_follow_each_mixed_step(monkeypatch, async_scheduling, k):
    _, off = _trace(monkeypatch, async_scheduling, _scenario())
    _, on = _trace(monkeypatch, async_scheduling, _scenario(), n=80, steps=k)
    ls = _l_steps(on)
    # L still gets the same total prompt tokens, in the same number of chunks
    assert sum(on[i]["L"] for i in ls) == sum(off[i]["L"] for i in _l_steps(off))
    assert len(ls) == len(_l_steps(off))
    # between consecutive L chunks: exactly k steps, each decode-only with d
    for a, b in zip(ls, ls[1:]):
        assert b - a == k + 1
        for i in range(a + 1, b):
            assert set(on[i]) == {"d"}
    # d is scheduled every step (it never loses a step)
    first_d = next(i for i, s in enumerate(on) if "d" in s)
    assert all("d" in s for s in on[first_d : ls[-1] + 1])


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_no_decode_no_deferral(monkeypatch, async_scheduling):
    arrivals = {0: [_request("L", 2000)]}
    _, on = _trace(monkeypatch, async_scheduling, arrivals, n=20, steps=3)
    ls = _l_steps(on)
    assert ls == list(range(ls[0], ls[0] + len(ls)))


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_new_requests_wait_during_a_decode_only_step(monkeypatch, async_scheduling):
    arrivals = _scenario()
    arrivals[4] = [_request("S", 100)]
    _, on = _trace(monkeypatch, async_scheduling, arrivals, n=80, steps=2)
    s_first = next(i for i, s in enumerate(on) if "S" in s)
    assert set(on[s_first]) - {"d"}  # admitted in a prefill-capable step
    ls = _l_steps(on)
    # S never starts in a step right after a mixed step (that is decode-only)
    assert all(s_first != a + 1 for a in ls if a + 1 < len(on))


def test_adaptive_scales_by_prefill_tokens():
    pol = DecodeInterleavePolicy(steps=4, adaptive=True)

    class _S:
        max_num_scheduled_tokens = 4096

    class _Out:
        def __init__(self, d):
            self.num_scheduled_tokens = d

    pol.observe(_S(), _Out({"L": 4092, "d": 4}), False, {"d"})
    assert pol._left == 4
    pol.observe(_S(), _Out({"L": 700, "d": 4}), False, {"d"})
    assert pol._left == 1  # ceil(4 * 700 / 4096)
    pol.observe(_S(), _Out({"L": 4096}), False, set())
    assert pol._left == 0  # no decode in the step: nothing owed


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_finish_holds_prefill_until_the_near_done_decode_finishes(
    monkeypatch, async_scheduling
):
    def arrivals():  # fresh requests per run
        return {
            0: [_request("d", 32, max_tokens=12)],
            1: [_request("L", 2000)],
        }

    _, off = _trace(monkeypatch, async_scheduling, arrivals(), n=40)
    _, on = _trace(monkeypatch, async_scheduling, arrivals(), n=40, finish=16)
    d_last_on = max(i for i, s in enumerate(on) if "d" in s)
    d_last_off = max(i for i, s in enumerate(off) if "d" in s)
    # d is within 16 tokens of max_tokens from its start: L waits for it
    assert all("L" not in s for s in on[: d_last_on + 1])
    assert d_last_on <= d_last_off
    # L is not starved: its whole prompt still goes in
    assert sum(on[i]["L"] for i in _l_steps(on)) >= 2000


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_finish_cap_bounds_consecutive_decode_only_steps(monkeypatch, async_scheduling):
    arrivals = {
        0: [_request("d", 32, max_tokens=60)],
        1: [_request("L", 2000)],
    }
    _, on = _trace(
        monkeypatch, async_scheduling, arrivals, n=60, finish=1000, finish_max=3
    )
    ls = _l_steps(on)
    assert len(ls) > 1
    for a, b in zip(ls, ls[1:]):
        if any("d" in on[i] for i in range(a + 1, b)):
            assert b - a <= 3 + 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_composes_with_srpt(monkeypatch, async_scheduling):
    arrivals = _scenario()
    arrivals[4] = [_request("S", 100)]
    sched, on = _trace(
        monkeypatch, async_scheduling, arrivals, n=80, steps=2, srpt=True
    )
    # SRPT still lets S in ahead of L's remaining chunks
    s_done = next(i for i, s in enumerate(on) if "S" in s)
    assert s_done < max(_l_steps(on))
    pol = sched._tpu_srpt_policy
    # deferred steps bypass SRPT: it saw only prefill-capable steps
    assert pol.steps < len([s for s in on if s])
    assert sched._tpu_interleave_policy.n_deferred > 0


def test_patch_is_a_no_op_when_off(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_tpu_decode_interleave", False, raising=False)
    for k in ("STEPS", "FINISH_TOKENS"):
        monkeypatch.setenv(
            f"TPU_SCHED_DECODE_{'INTERLEAVE_' if k == 'STEPS' else ''}{k}", "0"
        )
    before = Scheduler.schedule
    decode_interleave.patch_scheduler_decode_interleave()
    assert Scheduler.schedule is before


def test_registry_installs_after_srpt_on_the_engine_core():
    from vllm_torchtpu import patch_registry

    targets = [p.target for p in patch_registry.PATCHES]
    t_il = "vllm_torchtpu.core.decode_interleave:patch_scheduler_decode_interleave"
    t_srpt = "vllm_torchtpu.core.prefill_srpt:patch_scheduler_prefill_srpt"
    assert targets.index(t_il) > targets.index(t_srpt)
    (patch,) = [p for p in patch_registry.PATCHES if p.target == t_il]
    assert "engine_core" in patch.stages


def test_env_defaults_are_off_and_validated():
    ev = envs.environment_variables
    for name, off in (
        ("TPU_SCHED_DECODE_INTERLEAVE_STEPS", 0),
        ("TPU_SCHED_DECODE_INTERLEAVE_ADAPTIVE", False),
        ("TPU_SCHED_DECODE_FINISH_TOKENS", 0),
    ):
        if name not in os.environ:
            assert ev[name]() == off
    assert not DecodeInterleavePolicy().enabled
    with pytest.raises(ValueError):
        DecodeInterleavePolicy(steps=-1)
    with pytest.raises(ValueError):
        DecodeInterleavePolicy(finish_tokens=8, finish_max_steps=0)
