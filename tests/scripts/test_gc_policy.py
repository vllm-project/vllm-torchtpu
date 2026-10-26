# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the serving garbage-collection policy."""

import gc

import pytest

pytestmark = pytest.mark.cpu_test

from vllm_torchtpu import gc_policy  # noqa: E402


@pytest.fixture(autouse=True)
def restore_gc():
    enabled = gc.isenabled()
    yield
    gc.unfreeze()
    if enabled:
        gc.enable()


def test_arm_freezes_the_heap_and_stops_automatic_collection():
    policy = gc_policy.StepGCPolicy(collect_every=4)
    policy.arm()
    assert not gc.isenabled()
    assert gc.get_freeze_count() > 0
    with pytest.raises(RuntimeError):
        policy.arm()


def test_collection_runs_on_the_step_cadence_only():
    policy = gc_policy.StepGCPolicy(collect_every=4)
    policy.arm()
    for _ in range(3):
        policy.step()
    assert policy.collections == 0
    policy.step()
    assert policy.collections == 1
    assert policy.collect_ns > 0
    for _ in range(4):
        policy.step()
    assert policy.collections == 2


def test_cadence_zero_never_collects():
    policy = gc_policy.StepGCPolicy(collect_every=0)
    policy.arm()
    for _ in range(10):
        policy.step()
    assert policy.collections == 0


def test_unarmed_policy_counts_nothing():
    policy = gc_policy.StepGCPolicy(collect_every=1)
    policy.step()
    assert (policy.steps, policy.collections) == (0, 0)


def test_default_mode_installs_no_policy(monkeypatch):
    monkeypatch.setattr(gc_policy, "_POLICY", None)
    assert gc_policy.install("default", 1000) is None
    gc_policy.on_coordinated_step()  # a no-op without a policy
    with pytest.raises(ValueError):
        gc_policy.install("collect_always", 1000)


def test_negative_cadence_is_rejected():
    with pytest.raises(ValueError):
        gc_policy.StepGCPolicy(collect_every=-1)


@pytest.fixture
def engine_classes(monkeypatch):
    """vLLM's engine-core classes with stand-ins for the two methods the
    policy wraps; monkeypatch restores the real ones afterwards."""
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

    monkeypatch.setenv("TPU_GC_POLICY", "freeze")
    monkeypatch.setenv("TPU_GC_COLLECT_EVERY", "3")
    monkeypatch.setattr(gc_policy, "_ENGINE_POLICY", None)
    iterations = []
    monkeypatch.setattr(
        EngineCoreProc,
        "_process_engine_step",
        lambda self: iterations.append(1) or True,
    )

    def has_global_unfinished_reqs(self, local_unfinished):
        self.step_counter += 1
        return self.running

    monkeypatch.setattr(
        DPEngineCoreProc, "_has_global_unfinished_reqs", has_global_unfinished_reqs
    )
    return EngineCoreProc, DPEngineCoreProc, iterations


def test_engine_core_arms_on_the_first_iteration(engine_classes):
    """The engine core's heap is the scheduler's and is built before the busy
    loop runs, so the policy arms on the loop's first iteration; without DP
    the cadence is the engine core's own iteration count."""
    EngineCoreProc, _, iterations = engine_classes
    gc_policy.install_engine_core()
    wrapped = EngineCoreProc._process_engine_step
    gc_policy.install_engine_core()  # idempotent
    assert EngineCoreProc._process_engine_step is wrapped
    policy = gc_policy._ENGINE_POLICY
    assert policy is not gc_policy._POLICY
    assert not policy.armed
    core = object.__new__(EngineCoreProc)
    for _ in range(3):
        EngineCoreProc._process_engine_step(core)
    assert policy.armed
    assert (policy.steps, policy.collections, len(iterations)) == (3, 1, 3)


def test_dp_engine_core_collects_on_vllms_step_counter(engine_classes):
    """An idle DP rank runs loop iterations its peers do not, so its own
    iteration count drifts; only vLLM's synchronized step counter places the
    collection on the same step on every rank."""
    EngineCoreProc, DPEngineCoreProc, iterations = engine_classes
    gc_policy.install_engine_core()
    policy = gc_policy._ENGINE_POLICY
    core = object.__new__(DPEngineCoreProc)
    core.step_counter, core.running = 0, True
    for _ in range(5):  # arms, then idle iterations that must not count
        EngineCoreProc._process_engine_step(core)
    assert policy.armed and policy.collections == 0
    for _ in range(2):
        assert DPEngineCoreProc._has_global_unfinished_reqs(core, True)
    assert policy.collections == 0
    DPEngineCoreProc._has_global_unfinished_reqs(core, True)  # step_counter 3
    assert policy.collections == 1
    assert len(iterations) == 5


def test_dp_engine_core_collects_when_the_wave_ends(engine_classes):
    EngineCoreProc, DPEngineCoreProc, _ = engine_classes
    gc_policy.install_engine_core()
    policy = gc_policy._ENGINE_POLICY
    core = object.__new__(DPEngineCoreProc)
    core.step_counter, core.running = 0, True
    EngineCoreProc._process_engine_step(core)
    DPEngineCoreProc._has_global_unfinished_reqs(core, True)
    core.running = False  # every rank reaches this iteration together
    assert not DPEngineCoreProc._has_global_unfinished_reqs(core, False)
    assert policy.collections == 1


def test_engine_core_install_is_inert_under_the_default_policy(monkeypatch):
    monkeypatch.setenv("TPU_GC_POLICY", "default")
    monkeypatch.setattr(gc_policy, "_ENGINE_POLICY", None)
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

    step = EngineCoreProc._process_engine_step
    sync = DPEngineCoreProc._has_global_unfinished_reqs
    gc_policy.install_engine_core()
    assert EngineCoreProc._process_engine_step is step
    assert DPEngineCoreProc._has_global_unfinished_reqs is sync
    assert gc_policy._ENGINE_POLICY is None
