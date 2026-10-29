# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler


def _make_scheduler(waiting=(), drain_prefill=False):
    scheduler = TpuDpScheduler.__new__(TpuDpScheduler)
    scheduler.waiting = list(waiting)
    scheduler.kv_holding_waiting = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler._drain_prefill = drain_prefill
    scheduler.prefill_capacity_bound = False
    return scheduler


def test_unthrottled_step_opens_drain_window():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()

    with patch.object(AsyncScheduler, "schedule", return_value=expected) as schedule:
        result = scheduler.schedule(throttle_prefills=False)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert scheduler._drain_prefill is True
    # Upstream throttling is neutralized so the cadence is the only clock.
    assert scheduler.prefill_capacity_bound is True


def test_throttled_step_pauses_new_requests():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return expected

    with patch.object(AsyncScheduler, "schedule", side_effect=capture) as schedule:
        result = scheduler.schedule(throttle_prefills=True)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._pause_state is PauseState.UNPAUSED


def test_drain_spans_throttled_steps_until_queue_empties():
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return object()

    # A throttled step keeps draining while requests are still waiting.
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.UNPAUSED
    assert scheduler._drain_prefill is True

    # Once the queue drains, the window closes and buffering resumes.
    scheduler.waiting = []
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._drain_prefill is False


def test_pause_state_restored_when_schedule_raises():
    scheduler = _make_scheduler(waiting=[object()])

    with (
        patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError),
        pytest.raises(RuntimeError),
    ):
        scheduler.schedule(throttle_prefills=True)

    assert scheduler._pause_state is PauseState.UNPAUSED


class _FakeReq:
    def __init__(self, req_id: str, isl: int = 8192, osl: int = 1024) -> None:
        self.request_id = req_id
        self.status = "WAITING"
        self.num_prompt_tokens = isl
        self.num_tokens = isl
        self.max_tokens = osl
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.next_decode_eligible_step = 0

    @property
    def is_prefill_chunk(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens


def _make_wave_harness(
    total_blocks: int,
    waiting: list[_FakeReq],
    max_num_seqs: int = 64,
    block_size: int = 1024,
    chunk_size: int = 1024,
    watermark_blocks: int = 5,
):
    usable_blocks = total_blocks - 1
    allocated: dict[str, int] = {}

    def get_num_free_blocks() -> int:
        return usable_blocks - sum(allocated.values())

    def get_num_blocks_to_allocate(request_id: str, num_tokens: int, **kwargs) -> int:
        needed_total = math.ceil(num_tokens / block_size)
        return max(needed_total - allocated.get(request_id, 0), 0)

    scheduler = TpuDpScheduler.__new__(TpuDpScheduler)
    scheduler.waiting = list(waiting)
    scheduler.kv_holding_waiting = []
    scheduler.running = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler._drain_prefill = False
    scheduler._prefill_wave_enabled = True
    scheduler._decode_wave_locked = False
    scheduler.prefill_capacity_bound = False
    scheduler.current_step = 0
    scheduler.max_num_active_reqs = max_num_seqs
    scheduler.max_num_running_reqs = max_num_seqs
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_model_len = 9216
    scheduler.num_lookahead_tokens = 0
    scheduler._spec_decode_step_blocks = 0
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._is_blocked_waiting_status = lambda _status: False
    empty_blocks = SimpleNamespace(blocks=([],))
    scheduler._get_local_prefix_cache_hit = lambda _req: (empty_blocks, 0, 0, False)
    scheduler.kv_cache_manager = SimpleNamespace(
        empty_kv_cache_blocks=empty_blocks,
        watermark_blocks=watermark_blocks,
        coordinator=SimpleNamespace(
            get_num_blocks_to_allocate=get_num_blocks_to_allocate
        ),
        block_pool=SimpleNamespace(get_num_free_blocks=get_num_free_blocks),
    )

    def fake_super_schedule(_throttle: bool = False):
        # Simulate EngineCore popping batch_queue of size 2 before scheduling:
        # any request scheduled 2 steps ago (not in prev_step_scheduled_req_ids)
        # has its placeholder resolved into a real token.
        for r in scheduler.running:
            if (
                r.num_output_placeholders > 0
                and r.request_id not in scheduler.prev_step_scheduled_req_ids
            ):
                r.num_output_placeholders -= 1
                r.num_tokens += 1

        scheduler.current_step += 1
        budget = chunk_size
        scheduled: dict[str, int] = {}

        for r in list(scheduler.running):
            if budget <= 0:
                break
            if scheduler.current_step < r.next_decode_eligible_step:
                continue
            rem = r.num_tokens + r.num_output_placeholders - r.num_computed_tokens
            take = min(rem, budget)
            if take <= 0:
                continue
            r.num_computed_tokens += take
            allocated[r.request_id] = math.ceil(r.num_computed_tokens / block_size)
            scheduled[r.request_id] = take
            budget -= take
            if not r.is_prefill_chunk:
                r.num_output_placeholders += 1

        if scheduler._pause_state == PauseState.UNPAUSED:
            while (
                budget > 0
                and scheduler.waiting
                and len(scheduler.running) < scheduler.max_num_active_reqs
            ):
                r = scheduler.waiting[0]
                need_now = math.ceil(min(r.num_tokens, budget) / block_size)
                if get_num_free_blocks() - need_now < watermark_blocks:
                    break
                scheduler.waiting.pop(0)
                r.status = "RUNNING"
                scheduler.running.append(r)
                take = min(r.num_tokens, budget)
                r.num_computed_tokens += take
                allocated[r.request_id] = math.ceil(r.num_computed_tokens / block_size)
                scheduled[r.request_id] = take
                budget -= take
                if not r.is_prefill_chunk:
                    r.num_output_placeholders += 1

        scheduler.prev_step_scheduled_req_ids = set(scheduled.keys())
        return SimpleNamespace(
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
        )

    return scheduler, allocated, fake_super_schedule


def test_prefill_wave_single_wave_64_reqs_at_580_blocks():
    reqs = [_FakeReq(f"r{i}") for i in range(64)]
    scheduler, allocated, fake_super = _make_wave_harness(580, reqs)

    with patch.object(AsyncScheduler, "schedule", side_effect=fake_super):
        # 64 prompts * 8 chunks = 512 pure prefill steps, even when throttled!
        for step in range(512):
            out = scheduler.schedule(throttle_prefills=(step % 8 != 0))
            assert out.total_num_scheduled_tokens == 1024
            assert len(out.num_scheduled_tokens) == 1

        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 64

        # Step 512: all 64 completed prefills enter C=64 decode together!
        decode_out = scheduler.schedule(throttle_prefills=True)
        assert decode_out.total_num_scheduled_tokens == 64
        assert len(decode_out.num_scheduled_tokens) == 64
        assert scheduler._decode_wave_locked is True
        # 64 * 9 = 576 <= 579 usable blocks (3 free blocks left, 0 preemptions).
        assert sum(allocated.values()) == 576


def test_prefill_wave_two_waves_at_292_blocks():
    reqs = [_FakeReq(f"r{i}") for i in range(64)]
    scheduler, allocated, fake_super = _make_wave_harness(292, reqs)

    with patch.object(AsyncScheduler, "schedule", side_effect=fake_super):
        # Wave 1 admits exactly 32 requests (32 * 9 = 288 <= 291 usable blocks).
        for step in range(32 * 8):
            out = scheduler.schedule(throttle_prefills=(step % 8 != 0))
            assert out.total_num_scheduled_tokens == 1024

        assert len(scheduler.running) == 32
        assert len(scheduler.waiting) == 32

        # Wave 1 releases into C=32 decode and locks out R32 even on unthrottled step.
        decode_out = scheduler.schedule(throttle_prefills=False)
        assert decode_out.total_num_scheduled_tokens == 32
        assert len(decode_out.num_scheduled_tokens) == 32
        assert scheduler._decode_wave_locked is True
        assert sum(allocated.values()) == 288

        # Even if one request finishes early, Wave 1 stays locked until drained.
        finished = scheduler.running.pop(0)
        allocated.pop(finished.request_id)
        decode_out_2 = scheduler.schedule(throttle_prefills=False)
        assert decode_out_2.total_num_scheduled_tokens == 31
        assert len(scheduler.waiting) == 32

        # Once Wave 1 finishes completely, Wave 2 unlocks and admits remaining 32.
        scheduler.running.clear()
        allocated.clear()
        wave2_first = scheduler.schedule(throttle_prefills=True)
        assert scheduler._decode_wave_locked is False
        assert wave2_first.total_num_scheduled_tokens == 1024
        assert len(scheduler.waiting) == 31


def test_prefill_wave_drains_disjoint_async_placeholder_before_release():
    reqs = [_FakeReq("r0", isl=1024, osl=16), _FakeReq("r1", isl=1024, osl=16)]
    scheduler, _allocated, fake_super = _make_wave_harness(580, reqs)

    with patch.object(AsyncScheduler, "schedule", side_effect=fake_super):
        scheduler.schedule(throttle_prefills=False)  # r0 prefill completes
        scheduler.schedule(throttle_prefills=True)  # r1 prefill completes

        # Simulate batch_queue_size > 2: r0 still has an unpopped placeholder
        # even though prev_step_scheduled_req_ids only contains r1.
        r0 = scheduler.running[0]
        r0.num_output_placeholders = 1
        r0.num_tokens = 1024
        scheduler.prev_step_scheduled_req_ids = {"r1"}

        # Drain 1 pops r0; drain 2 pops r1 (whose step is no longer in
        # prev_step_scheduled_req_ids after drain 1); then both decode cleanly!
        drain_out_1 = scheduler.schedule(throttle_prefills=True)
        assert drain_out_1.total_num_scheduled_tokens == 0
        drain_out_2 = scheduler.schedule(throttle_prefills=True)
        assert drain_out_2.total_num_scheduled_tokens == 0

        release_out = scheduler.schedule(throttle_prefills=True)
        assert release_out.total_num_scheduled_tokens == 2
        assert set(release_out.num_scheduled_tokens.keys()) == {"r0", "r1"}


def test_prefill_wave_zero_livelock_when_target_rejected_upstream():
    reqs = [_FakeReq("r0", isl=1024, osl=16), _FakeReq("r1", isl=1024, osl=16)]
    scheduler, _allocated, fake_super = _make_wave_harness(580, reqs)

    with patch.object(AsyncScheduler, "schedule", side_effect=fake_super):
        scheduler.schedule(throttle_prefills=False)  # r0 finishes prefill

    # Now r1 is waiting and fits KV, so scheduler tries to admit r1 while
    # holding r0, but upstream rejects r1 and returns 0 tokens.
    def reject_once(_throttle: bool = False):
        scheduler.current_step += 1
        scheduler.prev_step_scheduled_req_ids = set()
        return SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0)

    with patch.object(AsyncScheduler, "schedule", side_effect=reject_once):
        empty_out = scheduler.schedule(throttle_prefills=False)
        assert empty_out.total_num_scheduled_tokens == 0
        assert scheduler._decode_wave_locked is True

    # Next step: wave is locked; after 1 placeholder drain step, r0 decodes!
    with patch.object(AsyncScheduler, "schedule", side_effect=fake_super):
        scheduler.schedule(throttle_prefills=False)
        decode_out = scheduler.schedule(throttle_prefills=False)
        assert decode_out.total_num_scheduled_tokens == 1
        assert "r0" in decode_out.num_scheduled_tokens
