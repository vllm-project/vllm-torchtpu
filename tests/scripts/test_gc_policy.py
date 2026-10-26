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
