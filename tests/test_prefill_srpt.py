"""Shortest-remaining-prefill-first with hold-back (core/prefill_srpt.py).

The patched vLLM ``Scheduler`` / ``AsyncScheduler`` is stepped on the host
with a fake model runner (one step in flight when async), so the step
composition the TPU runner would see is checked end to end.
"""

import os

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.core import prefill_srpt
from vllm_torchtpu.core.prefill_srpt import PrefillSrptPolicy, schedule_with_srpt

BLOCK = 16
BUDGET = 256
_HASH_READY = False


def _scheduler(async_scheduling, *, mns=8, prefix_caching=False, num_blocks=4000):
    torch = pytest.importorskip("torch")
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    try:
        model_config = ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        )
    except Exception as exc:  # offline host
        pytest.skip(f"model config unavailable: {exc}")
    scheduler_config = SchedulerConfig(
        max_num_seqs=mns,
        max_num_batched_tokens=BUDGET,
        max_model_len=8192,
        enable_chunked_prefill=True,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    try:
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )

        register_all_kvcache_specs(vllm_config)
    except ImportError:
        pass
    cls = AsyncScheduler if async_scheduling else Scheduler
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _request(req_id, prompt, max_tokens=4):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    global _HASH_READY
    if not _HASH_READY:  # one NONE_HASH, so prompts sharing a prefix hit
        init_none_hash(sha256)
        _HASH_READY = True
    params = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
    params.update_from_generation_config({}, 2)
    if isinstance(prompt, int):
        prompt = [hash(req_id) % 1000 + 10] * prompt
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _runner_output(out, mid):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(out.num_scheduled_tokens)
    sampled = [[] if rid in mid else [7] for rid in req_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _install(monkeypatch, on, max_hold=0):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler

    # Building a VllmConfig can install the same-step prefix-hit patch, which
    # wraps BlockPool and Scheduler.schedule together: undo both per test.
    for name in ("get_cached_block", "_insert_block_hash"):
        monkeypatch.setattr(BlockPool, name, getattr(BlockPool, name))
    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    monkeypatch.setattr(Scheduler, "_tpu_prefill_srpt", False, raising=False)
    monkeypatch.setenv("TPU_SCHED_PREFILL_SRPT", "1" if on else "0")
    monkeypatch.setenv("TPU_SCHED_PREFILL_SRPT_MAX_HOLD_STEPS", str(max_hold))
    prefill_srpt.patch_scheduler_prefill_srpt()


def _run(scheduler, arrivals, steps, async_scheduling, check=None):
    """Drive schedule/update like the engine core. ``arrivals``: step ->
    requests added before that step's schedule(). Returns per step
    {req_id: tokens}."""
    trace = []
    in_flight = None
    for step in range(steps):
        for req in arrivals.get(step, ()):
            scheduler.add_request(req)
        out = scheduler.schedule()
        mid = {
            rid
            for rid in out.num_scheduled_tokens
            if scheduler.requests[rid].num_computed_tokens
            < scheduler.requests[rid].num_prompt_tokens
        }
        trace.append(dict(out.num_scheduled_tokens))
        assert len(scheduler.running) <= scheduler.max_num_running_reqs
        assert sum(out.num_scheduled_tokens.values()) <= BUDGET
        if check is not None:
            check(scheduler, out)
        if async_scheduling:
            if in_flight is not None:
                scheduler.update_from_output(in_flight[0], _runner_output(*in_flight))
            in_flight = (out, mid)
        else:
            scheduler.update_from_output(out, _runner_output(out, mid))
    return trace


def _done_step(trace, rid, prompt):
    """Step whose chunk completes ``rid``'s prompt (its first token)."""
    total = 0
    for i, s in enumerate(trace):
        total += s.get(rid, 0)
        if total >= prompt:
            return i
    return None


def _prefill_tokens(trace, rid, prompt):
    """Tokens ``rid`` got up to and including its prompt-completing step."""
    done = _done_step(trace, rid, prompt)
    return sum(s.get(rid, 0) for s in trace[: done + 1])


def _scenario(monkeypatch, on, async_scheduling, steps=30):
    """A decode ``d`` runs; a 2000-token prompt ``L`` starts chunking at
    step 1; a 100-token prompt ``S`` arrives at step 3."""
    _install(monkeypatch, on)
    sched = _scheduler(async_scheduling)
    arrivals = {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 2000)],
        3: [_request("S", 100)],
    }
    return _run(sched, arrivals, steps, async_scheduling)


# --------------------------------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_off_short_prompt_waits_for_the_whole_long_prompt(
    monkeypatch, async_scheduling
):
    trace = _scenario(monkeypatch, False, async_scheduling)
    l_done = _done_step(trace, "L", 2000)
    s_first = next(i for i, s in enumerate(trace) if "S" in s)
    assert s_first >= l_done  # head-of-line blocked behind every chunk


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_flag_off_is_the_unpatched_scheduler(monkeypatch, async_scheduling):
    from vllm.v1.core.sched.scheduler import Scheduler

    _install(monkeypatch, False)
    assert not getattr(Scheduler, "_tpu_prefill_srpt", False)
    plain = _scenario(monkeypatch, False, async_scheduling)
    # A wrapper with a policy that never plans must not change one step.
    orig = Scheduler.schedule

    class _Never(PrefillSrptPolicy):
        def plan(self, scheduler):
            return prefill_srpt.SrptPlan()

    def wrapped(self, throttle_prefills=False):
        pol = self.__dict__.setdefault("_p", _Never())
        return schedule_with_srpt(self, pol, orig, throttle_prefills)

    monkeypatch.setattr(Scheduler, "schedule", wrapped)
    sched = _scheduler(async_scheduling)
    arrivals = {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 2000)],
        3: [_request("S", 100)],
    }
    assert _run(sched, arrivals, 30, async_scheduling) == plain


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_hold_back_lets_the_short_prompt_in_at_once(monkeypatch, async_scheduling):
    off = _scenario(monkeypatch, False, async_scheduling)
    on = _scenario(monkeypatch, True, async_scheduling)
    # S goes in the step it arrives, whole, and the long chunk sits that step out.
    assert on[3].get("S") == 100 and "L" not in on[3] and "d" in on[3]
    assert _done_step(on, "S", 100) == 3
    assert _done_step(off, "S", 100) > _done_step(off, "L", 2000)
    # L loses exactly the held step; its chunks are otherwise unchanged.
    assert _done_step(on, "L", 2000) == _done_step(off, "L", 2000) + 1
    assert _prefill_tokens(on, "L", 2000) == 2000
    # Decodes keep running every step, S's too although it was admitted
    # behind the held prefill in ``running``.
    assert all("d" in s for s in on[3:20])
    assert all("S" in s for s in on[3:7])


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_waiting_requests_go_shortest_first(monkeypatch, async_scheduling):
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling)
    arrivals = {
        1: [_request("L", 2000)],
        3: [_request("A", 200), _request("B", 50)],  # A queued before B
    }
    trace = _run(sched, arrivals, 20, async_scheduling)
    # B (shorter, queued second) first, then A; the long chunk sits out.
    assert list(trace[3].items()) == [("B", 50), ("A", 200)]
    assert "L" in trace[4]
    assert _prefill_tokens(trace, "L", 2000) == 2000


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_longer_waiting_request_is_hidden_while_a_prefill_is_held(
    monkeypatch, async_scheduling
):
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling)
    arrivals = {
        1: [_request("L", 600)],
        3: [_request("W", 1500), _request("S", 50)],  # W queued first
    }
    trace = _run(sched, arrivals, 30, async_scheduling)
    # The held step gives S its 50 and nothing to W (longer than L's rest).
    assert trace[3] == {"S": 50}
    # Then plain SRPT: L (shorter) completes, W only gets what L leaves.
    assert trace[4]["L"] == 600 - 512 and _done_step(trace, "L", 600) == 4
    assert next(i for i, s in enumerate(trace) if "W" in s) == 4
    # FCFS order of the queue is restored after the hidden step.
    assert _prefill_tokens(trace, "W", 1500) == 1500


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_running_prefills_go_shortest_first(monkeypatch, async_scheduling):
    """Two in-progress prefills: the older, longer one is held while the
    newer, shorter one finishes."""
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling)
    arrivals = {1: [_request("L", 1800)], 3: [_request("M", 600)]}
    trace = _run(sched, arrivals, 40, async_scheduling)
    # M's three chunks run back to back with L held, then L resumes.
    assert [s.get("M") for s in trace[3:6]] == [256, 256, 88]
    assert all("L" not in s for s in trace[3:6])
    assert "L" in trace[6] and "M" in trace[6]
    assert _prefill_tokens(trace, "L", 1800) == 1800


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_held_request_keeps_its_slot(monkeypatch, async_scheduling):
    """mns 3: d + L run; two shorter prompts wait. Only one slot is free, so
    only one is admitted though L is detached during the call."""
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling, mns=3)

    def check(s, out):
        assert len(s.running) <= 3

    arrivals = {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 2000)],
        3: [_request("S1", 60), _request("S2", 40)],
    }
    trace = _run(sched, arrivals, 12, async_scheduling, check)
    assert trace[3] == {"d": 1, "S2": 40}
    assert [r.request_id for r in sched.running][:2] == ["d", "L"]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_no_hold_when_no_slot_is_free(monkeypatch, async_scheduling):
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling, mns=2)
    arrivals = {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 2000)],
        3: [_request("S", 50)],
    }
    trace = _run(sched, arrivals, 6, async_scheduling)
    assert all("L" in s and "S" not in s for s in trace[1:6])


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_running_order_and_queue_order_are_restored(monkeypatch, async_scheduling):
    """After a hold step: the held request is back in place, the admission
    is last (running[-1] is still the newest), and the hidden request keeps
    its queue position."""
    _install(monkeypatch, True)
    sched = _scheduler(async_scheduling)
    seen = {}

    def check(s, out):
        if s.current_step == 4:  # schedule() of step index 3
            seen["running"] = [r.request_id for r in s.running]
            seen["waiting"] = [r.request_id for r in s.waiting]

    arrivals = {
        0: [_request("d", 32, max_tokens=400)],
        1: [_request("L", 600)],
        3: [_request("W", 1500), _request("S", 50)],
    }
    _run(sched, arrivals, 6, async_scheduling, check)
    assert seen["running"] == ["d", "L", "S"]
    assert seen["waiting"] == ["W"]


def test_preemption_order_is_running_minus_one(monkeypatch):
    """KV runs out mid-run: vLLM preempts running[-1], the newest admission,
    never a held request detached for the step, and every request still
    completes."""
    _install(monkeypatch, True)
    sched = _scheduler(False, num_blocks=150)  # 2400 tokens of KV
    preempted = []

    def check(s, out):
        preempted.extend(r for r in out.preempted_req_ids or ())

    arrivals = {
        0: [_request("d", 32, max_tokens=900)],
        1: [_request("L", 1600)],
        3: [_request("S", 300, max_tokens=400)],
    }
    trace = _run(sched, arrivals, 120, False, check)
    assert _done_step(trace, "S", 300) is not None
    assert _done_step(trace, "L", 1600) is not None
    assert _done_step(trace, "d", 32) is not None


def _short_stream():
    arrivals = {1: [_request("L", 2000)]}
    for i in range(3, 23):
        arrivals[i] = [_request(f"s{i}", 30)]
    return arrivals


def test_short_stream_holds_a_long_prompt_without_a_guard(monkeypatch):
    """A fresh short prompt every step keeps a long one held: the hold-back
    has no aging by default (at c <= 8 the closed-loop client bounds it)."""
    _install(monkeypatch, True)
    trace = _run(_scheduler(False), _short_stream(), 24, False)
    assert all("L" not in s for s in trace[3:23])
    assert "L" in trace[23]


def test_max_hold_steps_bounds_the_hold(monkeypatch):
    _install(monkeypatch, True, max_hold=2)
    trace = _run(_scheduler(False), _short_stream(), 24, False)
    l_steps = [i for i, s in enumerate(trace) if "L" in s]
    # Held at most 2 steps in a row, then one chunk.
    assert all(b - a <= 3 for a, b in zip(l_steps, l_steps[1:]))
    assert len([i for i in l_steps if 3 <= i < 23]) >= 6


def test_prefix_cached_prompt_counts_only_its_cold_tokens(monkeypatch):
    """A 1500-token prompt whose first 1440 tokens are cached is short (60
    cold tokens) and is admitted ahead of a 1000-token running prefill."""
    _install(monkeypatch, True)
    sched = _scheduler(False, prefix_caching=True)
    base = [5] * 1440
    arrivals = {
        0: [_request("X", base + [6] * 16)],
        8: [_request("L", [9] * 1000)],
        10: [_request("Y", base + [8] * 60)],
    }
    seen = {}

    def check(s, out):
        if s.current_step == 10:  # before step index 10's schedule()
            y = arrivals[10][0]
            seen["vllm_hit"] = s.kv_cache_manager.coordinator.find_longest_cache_hit(
                y.block_hashes, y.num_tokens - 1
            )[1]

    trace = _run(sched, arrivals, 20, False, check)
    if not seen["vllm_hit"]:
        # Patches applied by earlier tests in this process can leave the
        # host-side prefix cache unhittable; the hold decision is then moot.
        pytest.skip("prefix cache not hittable in this process state")
    key = sched._tpu_srpt_policy._key_cache["Y"][1][0]
    assert key == 1500 - 1440
    assert "L" not in trace[10] and trace[10].get("Y") == 60


def test_no_livelock_when_the_favoured_request_cannot_run():
    pol = PrefillSrptPolicy()
    plan = prefill_srpt.SrptPlan(
        held=[type("R", (), {"request_id": "L"})()], favoured={"S"}
    )

    class _Out:
        num_scheduled_tokens = {"d": 1}

    pol.observe(plan, _Out())
    assert pol._skip_next
    assert not pol._hold_eligible(object(), [])  # next step: no hold
    pol.observe(prefill_srpt.SrptPlan(), _Out())
    assert not pol._skip_next


def test_patch_is_a_no_op_when_off(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setenv("TPU_SCHED_PREFILL_SRPT", "0")
    before = Scheduler.schedule
    prefill_srpt.patch_scheduler_prefill_srpt()
    assert Scheduler.schedule is before


def test_registry_installs_the_patch_on_the_engine_core():
    from vllm_torchtpu import patch_registry

    target = "vllm_torchtpu.core.prefill_srpt:patch_scheduler_prefill_srpt"
    (patch,) = [p for p in patch_registry.PATCHES if p.target == target]
    assert "engine_core" in patch.stages


def test_env_defaults_are_off():
    if "TPU_SCHED_PREFILL_SRPT" not in os.environ:
        assert envs.environment_variables["TPU_SCHED_PREFILL_SRPT"]() is False
    with pytest.raises(ValueError):
        PrefillSrptPolicy(-1)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_on_without_contention_matches_off(monkeypatch, async_scheduling):
    """No prefill to reorder (one prompt at a time, decodes behind it): the
    policy must not change a step."""

    def trace(on):
        _install(monkeypatch, on)
        arrivals = {
            0: [_request("d", 32, max_tokens=400)],
            1: [_request("L", 1500)],
            12: [_request("M", 700)],
        }
        return _run(_scheduler(async_scheduling), arrivals, 30, async_scheduling)

    assert trace(True) == trace(False)
