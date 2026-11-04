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
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.pp_chunk_scheduler import (
    SCHEDULER_CLS,
    TpuPipelineChunkScheduler,
    install_chunk_limits_from_workers,
    patch_engine_core_for_pp_chunks,
    uses_dynamic_chunks,
)
from vllm_torchtpu.core.pp_chunks import chunk_pairs

_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


def _scheduler(
    mamba_split=False,
    granularity=None,
    schedule=None,
    running=(),
    waiting=(),
    step_limit=None,
    kv_holding_waiting=(),
) -> TpuPipelineChunkScheduler:
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    scheduler._granularity = granularity
    scheduler._schedule = schedule
    scheduler._step_limit = step_limit
    scheduler._pp_size = 8
    scheduler.waiting = list(waiting)
    scheduler.kv_holding_waiting = list(kv_holding_waiting)
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=16384)
    scheduler.vllm_config = SimpleNamespace(
        scheduler_config=scheduler.scheduler_config,
        parallel_config=SimpleNamespace(pipeline_parallel_size=8),
    )
    scheduler._mamba_split = mamba_split
    scheduler.need_mamba_block_aligned_split = True
    scheduler.running = list(running)
    scheduler._step_tokens = 0
    scheduler._step_prefill_tokens = 0
    scheduler._step_schedule = 0
    scheduler._seeded = set()
    scheduler._clips = 0
    scheduler._waits = 0
    return scheduler


def _request(computed: int, prompt: int, req_id="r", num_tokens=None):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt if num_tokens is None else num_tokens,
        request_id=req_id,
    )


def _decode(req_id: str, computed: int = 8192):
    return _request(computed, computed, req_id, num_tokens=computed + 1)


def test_without_limits_chunks_pass_through():
    scheduler = _scheduler()
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 16384
    assert scheduler._step_tokens == 16384
    assert scheduler._step_prefill_tokens == 16384


def test_prefix_free_first_chunk_is_untouched():
    scheduler = _scheduler(granularity=4096, schedule=(10000, 256, 256))
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 16384
    assert scheduler._clips == 0


def test_the_first_prefill_of_a_step_advances_even_behind_decodes():
    # 512 decodes fill the 512-token step a lone 4K prompt is spread into.
    decodes = [_decode(f"d{i}") for i in range(512)]
    lone = _request(0, 4096, "p")
    scheduler = _scheduler(granularity=512, running=decodes, waiting=[lone])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 512
    assert scheduler._step_tokens == 512
    assert scheduler._step_prefill_tokens == 0
    # One granule for the first prompt anyway.
    assert scheduler._mamba_block_aligned_split(lone, 4096) == 512
    # A second prompt waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(0, 4096, "s"), 4096) == 0
    assert scheduler._waits == 1


def test_decodes_are_counted_once_before_prefill_chunks_are_sized():
    decodes = [_decode(f"d{i}", 8192) for i in range(37)]
    lone = _request(0, 32768, "p")
    scheduler = _scheduler(granularity=2048, running=decodes, waiting=[lone])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 4096
    assert scheduler._step_tokens == 37
    # The running loop visits the decodes: nothing is added twice.
    assert scheduler._mamba_block_aligned_split(decodes[0], 1) == 1
    assert scheduler._step_tokens == 37
    # The prefill chunk lands the step total on the limit.
    assert scheduler._mamba_block_aligned_split(lone, 16384) == 4096 - 37
    assert scheduler._step_tokens == 4096


def test_a_prefill_still_in_flight_is_not_seeded_as_a_decode():
    in_flight = [_request(8192, 8192, f"p{i}") for i in range(6)]
    scheduler = _scheduler(granularity=4096, running=in_flight)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 0 and not scheduler._seeded
    # Two whole 8K prompts still share one step.
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "a"), 8192) == 8192
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "b"), 8192) == 8192


def test_a_decode_admitted_after_the_seed_is_counted_when_visited():
    scheduler = _scheduler(granularity=4096, step_limit=4096)
    scheduler._step_tokens = 4096
    scheduler._step_prefill_tokens = 4096
    assert scheduler._mamba_block_aligned_split(_decode("late"), 1) == 1
    assert scheduler._clips == 0
    assert scheduler._step_tokens == 4097


def test_a_resumed_request_replaying_output_is_sized_like_prefill():
    schedule = (100000, 256, 256)
    scheduler = _scheduler(granularity=4096, schedule=schedule)
    scheduler._step_tokens = 4096
    scheduler._step_prefill_tokens = 4096
    request = _request(65536, 65536, num_tokens=65540)
    assert scheduler._mamba_block_aligned_split(request, 3) == 3
    assert scheduler._clips == 0
    assert scheduler._step_prefill_tokens == 4099
    # One query block attending to the 65,539 tokens up to its end.
    assert scheduler._step_schedule == chunk_pairs(schedule, 3, 65536) == 257


def test_external_and_local_computed_tokens_count_as_prefix():
    # 16K tokens need 2,080 pairs of 256x256 tiles from the start of a
    # prompt and 6,176 behind 16K of context.
    scheduler = _scheduler(granularity=4096, schedule=(4000, 256, 256))
    got = scheduler._mamba_block_aligned_split(
        _request(0, 65536),
        16384,
        num_new_local_computed_tokens=8192,
        num_external_computed_tokens=8192,
    )
    assert got == 8192
    scheduler = _scheduler(granularity=4096, schedule=(4000, 256, 256))
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 16384


def test_mamba_alignment_runs_after_the_clip_when_the_model_needs_it():
    scheduler = _scheduler(mamba_split=True, granularity=4096, step_limit=12288)
    seen = {}

    def align(self, request, num_new_tokens, local=0, external=0):
        seen["tokens"] = num_new_tokens
        return num_new_tokens - 1

    with patch.object(Scheduler, "_mamba_block_aligned_split", align):
        got = scheduler._mamba_block_aligned_split(_request(16384, 65536), 16384)
    assert seen["tokens"] == 12288
    assert got == 12287
    assert scheduler._step_tokens == 12287


def test_a_limited_step_takes_one_small_chunk_and_fills_with_it():
    scheduler = _scheduler(granularity=2048, step_limit=4096)
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 4096
    assert scheduler._clips == 1
    # The step is full: the next request waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"), 16384) == 0
    assert scheduler._waits == 1


def test_the_pair_capacity_backs_a_deep_chunk_off_by_granules():
    schedule = (10000, 256, 256)
    scheduler = _scheduler(granularity=2048, schedule=schedule)
    # 16K behind 48K needs 64 * 192 + 2080 pairs: over 10000.
    got = scheduler._mamba_block_aligned_split(_request(49152, 131072), 16384)
    assert got == 10240 and scheduler._clips == 1
    assert chunk_pairs(schedule, got, 49152) <= 10000
    assert chunk_pairs(schedule, got + 2048, 49152) > 10000
    assert scheduler._step_schedule == chunk_pairs(schedule, got, 49152)
    # Pairs already in the step leave less room, and the step total stays
    # on a granule multiple.
    got = scheduler._mamba_block_aligned_split(_request(0, 65536, "s"), 16384)
    assert got == 12288
    assert scheduler._step_tokens % 2048 == 0
    assert scheduler._step_schedule <= 10000


def test_the_step_limit_spreads_the_pending_prefill_over_the_stages():
    lone = [_request(0, 32768)]
    scheduler = _scheduler(granularity=2048, running=lone)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    # 32K over 8 stages: 4K steps.
    assert scheduler._step_limit == 4096
    # 8K left of a prompt: a granule per stage.
    scheduler = _scheduler(granularity=2048, running=[_request(24576, 32768)])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 2048
    # Waiting requests count; enough work keeps full steps.
    waiting = [_request(0, 65536, f"w{i}") for i in range(4)]
    scheduler = _scheduler(granularity=2048, running=lone, waiting=waiting)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None
    # Requests whose prefill is fully scheduled add nothing: no limit.
    scheduler = _scheduler(granularity=2048, running=[_decode("d")])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None
    # Without the chunk limits there is no limit.
    scheduler = _scheduler(running=lone)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None


def test_a_busy_server_leaves_chunks_at_the_token_budget():
    # Five prompts fill a full step on every stage.
    waiting = [_request(0, 32768, f"w{i}") for i in range(5)]
    scheduler = _scheduler(granularity=2048, waiting=waiting)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None
    assert scheduler._mamba_block_aligned_split(waiting[0], 16384) == 16384
    assert scheduler._clips == 0 and scheduler._waits == 0


def test_kv_holding_queue_does_not_shrink_the_pending_work_budget():
    pending = [_request(0, 8192, f"w{i}") for i in range(4)]
    scheduler = _scheduler(granularity=2048, waiting=pending)
    assert scheduler._pending_step_limit() == 4096

    # Queue placement alone does not consume prompt tokens. Requests holding
    # KV blocks must contribute the same pending work as other waiting requests.
    scheduler.waiting = []
    scheduler.kv_holding_waiting = pending
    assert scheduler._pending_step_limit() == 4096

    # Splitting requests across both queues must preserve the same budget.
    scheduler.waiting = pending[:2]
    scheduler.kv_holding_waiting = pending[2:]
    assert scheduler._pending_step_limit() == 4096
    scheduler.waiting = []

    # Count only the remaining prompt work, even in the KV-holding queue.
    scheduler.kv_holding_waiting = [_request(24576, 32768)]
    assert scheduler._pending_step_limit() == 2048
    scheduler.kv_holding_waiting = [_decode("done")]
    assert scheduler._pending_step_limit() is None


def test_schedule_resets_the_step_accounting():
    scheduler = _scheduler(granularity=4096)
    scheduler._step_tokens = 5
    scheduler._step_prefill_tokens = 5
    scheduler._step_schedule = 9
    scheduler._seeded = {"x"}
    with patch.object(Scheduler, "schedule", return_value="out") as schedule:
        assert scheduler.schedule(throttle_prefills=True) == "out"
    schedule.assert_called_once_with(True)
    assert scheduler._step_tokens == 0 and scheduler._step_schedule == 0
    assert scheduler._step_prefill_tokens == 0 and not scheduler._seeded


def _config(pp=8, cls=SCHEDULER_CLS):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(scheduler_cls=cls),
    )


def test_uses_dynamic_chunks_follows_the_installed_scheduler_class():
    assert uses_dynamic_chunks(_config())
    assert not uses_dynamic_chunks(_config(pp=1))
    assert not uses_dynamic_chunks(_config(cls=None))
    assert not uses_dynamic_chunks(_config(cls="other.Scheduler"))
    assert not uses_dynamic_chunks(None)


class _Executor:
    def __init__(self, reports):
        self.reports = reports
        self.calls = []

    def collective_rpc(self, method):
        self.calls.append(method)
        return self.reports


def _report(granularity=4096, schedule=(14000, 256, 256)):
    return {"granularity": granularity, "schedule": schedule}


def test_install_chunk_limits_merges_every_stage():
    executor = _Executor(
        [
            _report(4096, (14000, 256, 256)),
            _report(4352, (13980, 256, 256)),
            _report(4096, None),
        ]
    )
    scheduler = _scheduler()
    scheduler._pp_size = 1
    install_chunk_limits_from_workers(scheduler, executor)
    assert executor.calls == ["pipeline_chunk_limits"]
    # The largest granule and the smallest stage capacity are kept.
    assert scheduler._granularity == 4352
    assert scheduler._schedule == (13980, 256, 256)
    assert scheduler._pp_size == 8


def test_install_chunk_limits_without_a_schedule_capacity():
    scheduler = _scheduler()
    install_chunk_limits_from_workers(scheduler, _Executor([_report(2048, None)]))
    assert scheduler._granularity == 2048
    assert scheduler._schedule is None


def test_install_chunk_limits_skips_other_schedulers():
    executor = _Executor([])
    install_chunk_limits_from_workers(object(), executor)
    assert executor.calls == []


def test_engine_core_patch_installs_after_init_of_subclasses(monkeypatch):
    from vllm.v1.engine import core as engine_core

    class FakeCore:
        def __init__(self, *args, **kwargs):
            self.scheduler = _scheduler()
            self.model_executor = _Executor([_report()])

    class FakeProc(FakeCore):
        def __init__(self):
            super().__init__()
            self.after_init = self.scheduler._granularity

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    patch_engine_core_for_pp_chunks(_config())
    assert FakeCore._tpu_pp_chunks_patch
    assert FakeCore().scheduler._granularity == 4096
    assert FakeProc().after_init == 4096
    # A second install is a no-op.
    init = FakeCore.__init__
    patch_engine_core_for_pp_chunks(_config())
    assert FakeCore.__init__ is init


def test_engine_core_patch_is_skipped_without_dynamic_chunks(monkeypatch):
    from vllm.v1.engine import core as engine_core

    class FakeCore:
        pass

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    patch_engine_core_for_pp_chunks(_config(cls=None))
    assert not hasattr(FakeCore, "_tpu_pp_chunks_patch")
