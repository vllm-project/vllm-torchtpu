# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.megakernel.dp_blocks import RegionBlockPool


def test_waiting_request_can_leave_a_full_group():
    pool = RegionBlockPool(128, False, 16, groups=2, shared=16, cap=2)
    for index in range(4):
        request_id = f"active_{index}"
        pool.active[request_id] = pool.place(request_id)
    assert pool.place("waiting", step=0) == 0
    pool.release("active_1")
    assert pool.place("waiting") == 0
    assert pool.place("waiting", step=1) == 0
    assert pool.place("waiting", step=16) == 0
    assert pool.place("waiting", step=17) == 1
    assert pool.place("waiting") == 1
    assert pool.group == 1


def test_allocated_request_keeps_its_group_when_other_group_has_space():
    pool = RegionBlockPool(128, False, 16, groups=2, shared=16, cap=2)
    pool.req_group = {"transferring": 0, "running": 0}
    pool.active = pool.req_group.copy()
    for step in (0, 16, 32):
        assert pool.place("transferring", step=step) == 0
        assert pool.place("running", step=step) == 0


def test_deferred_requests_balance_when_they_are_allocated():
    pool = RegionBlockPool(128, False, 16, groups=2, shared=16, cap=2)
    request_ids = [f"deferred_{index}" for index in range(4)]
    assert [pool.place(req) for req in request_ids] == [0, 1, 0, 1]
    for request_id in request_ids:
        pool.active[request_id] = pool.place(request_id)
    assert list(pool.active.values()) == [0, 1, 0, 1]


def test_preferred_group_recovery_restarts_the_retry_interval():
    pool = RegionBlockPool(128, False, 16, groups=2, shared=16, cap=2)
    pool.active = {"active_a": 0, "active_b": 0}
    pool.req_group = dict(pool.active, waiting=0)
    assert pool.place("waiting", step=0) == 0
    pool.release("active_a")
    assert pool.place("waiting", step=1) == 0
    pool.active["replacement"] = pool.req_group["replacement"] = 0
    assert pool.place("waiting", step=16) == 0
    assert pool.place("waiting", step=32) == 1


def test_releasing_a_waiting_request_clears_its_retry_state():
    pool = RegionBlockPool(128, False, 16, groups=2, shared=16, cap=2)
    pool.active = {"active_a": 0, "active_b": 0}
    pool.req_group = dict(pool.active, waiting=0)
    pool.place("waiting", step=0)
    assert "waiting" in pool.blocked_since
    pool.release("waiting")
    assert "waiting" not in pool.blocked_since
    assert "waiting" not in pool.req_group
