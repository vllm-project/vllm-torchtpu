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

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import pp_chunk_scheduler
from vllm_torchtpu.core.pp_chunk_scheduler import (
    SCHEDULER_CLS,
    TpuPipelineChunkScheduler,
    patch_engine_core_for_pp_chunks,
    profile_and_install_cost_model,
    uses_dynamic_chunks,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel

BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


def _model(attn=1.5e-7, slack=0.1, granularity=4096) -> StepCostModel:
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack), granularity)


def _scheduler(
    mamba_split=False, cost=None, running=(), waiting=(), step_limit=None
) -> TpuPipelineChunkScheduler:
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    scheduler._cost = cost
    scheduler._step_limit = step_limit
    scheduler._pp_size = 8
    scheduler.waiting = list(waiting)
    scheduler.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        parallel_config=SimpleNamespace(pipeline_parallel_size=8),
    )
    scheduler._mamba_split = mamba_split
    scheduler.need_mamba_block_aligned_split = True
    scheduler.running = list(running)
    scheduler._step_tokens = 0
    scheduler._step_pairs = 0.0
    scheduler._step_squares = 0.0
    scheduler._step_prefill_tokens = 0
    scheduler._step_schedule = 0
    scheduler._seeded = set()
    scheduler._inflight_ms = []
    scheduler._clips = 0
    scheduler._waits = 0
    scheduler._retention = None
    scheduler._single_step = False
    scheduler._n_single_step_reqs = 0
    scheduler.kv_cache_manager = SimpleNamespace()
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


def test_without_a_model_chunks_pass_through():
    scheduler = _scheduler()
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 16384
    assert scheduler._step_tokens == 16384
    assert scheduler._step_prefill_tokens == 16384


def test_prefix_free_first_chunk_is_untouched():
    scheduler = _scheduler(cost=_model())
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 16384
    assert scheduler._clips == 0


def test_chunk_behind_a_prefix_is_clipped_and_accounted():
    scheduler = _scheduler(cost=_model())
    got = scheduler._mamba_block_aligned_split(_request(16384, 65536), 16384)
    assert got == 12288
    assert scheduler._step_tokens == 12288
    assert scheduler._step_pairs == pytest.approx(12288.0 * 16384)
    assert scheduler._step_squares == pytest.approx(12288.0**2)
    assert scheduler._clips == 1
    # The step is at its target: the next request waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"), 16384) == 0
    assert scheduler._waits == 1
    assert scheduler._step_tokens == 12288


def test_a_light_chunk_leaves_room_for_the_next_request():
    scheduler = _scheduler(cost=_model())
    assert scheduler._mamba_block_aligned_split(_request(16384, 20480), 4096) == 4096
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"), 16384) == 12288
    assert scheduler._step_tokens == 16384


def test_the_first_prefill_of_a_step_advances_even_behind_decodes():
    # Decodes so deep in their contexts that they alone exceed the target.
    decodes = [_decode(f"d{i}", 40_000_000) for i in range(32)]
    scheduler = _scheduler(cost=_model(), running=decodes)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 32
    assert scheduler._step_prefill_tokens == 0
    assert scheduler._cost.chunk(65536, 16384, 32, scheduler._step_pairs, 32.0) == 0
    # One granule for the first prompt anyway.
    assert scheduler._mamba_block_aligned_split(_request(65536, 131072), 16384) == 4096
    # A second prompt waits for the next step.
    assert (
        scheduler._mamba_block_aligned_split(_request(65536, 131072, "s"), 16384) == 0
    )
    assert scheduler._waits == 1


def test_decodes_are_counted_once_before_prefill_chunks_are_sized():
    decodes = [_decode(f"d{i}", 8192) for i in range(37)]
    scheduler = _scheduler(cost=_model(), running=decodes)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 37
    assert scheduler._step_pairs == pytest.approx(37.0 * 8192)
    # The running loop visits the decodes: nothing is added twice.
    assert scheduler._mamba_block_aligned_split(decodes[0], 1) == 1
    assert scheduler._step_tokens == 37
    # A prefill chunk lands the step total on a granule multiple.
    assert (
        scheduler._mamba_block_aligned_split(_request(0, 65536), 16384 - 37)
        == 16384 - 37
    )
    assert scheduler._step_tokens == 16384


def test_a_prefill_still_in_flight_is_not_seeded_as_a_decode():
    in_flight = [_request(8192, 8192, f"p{i}") for i in range(6)]
    scheduler = _scheduler(cost=_model(), running=in_flight)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 0 and not scheduler._seeded
    # Two whole 8K prompts still share one step.
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "a"), 8192) == 8192
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "b"), 8192) == 8192


def test_a_decode_admitted_after_the_seed_is_counted_when_visited():
    scheduler = _scheduler(cost=_model(attn=1e-2))
    scheduler._step_tokens = 4096
    scheduler._step_prefill_tokens = 4096
    assert scheduler._mamba_block_aligned_split(_decode("late"), 1) == 1
    assert scheduler._clips == 0
    assert scheduler._step_tokens == 4097
    assert scheduler._step_pairs == pytest.approx(8192.0)


def test_a_resumed_request_replaying_output_is_sized_like_prefill():
    scheduler = _scheduler(cost=_model())
    scheduler._step_tokens = 4096
    scheduler._step_prefill_tokens = 4096
    request = _request(65536, 65536, num_tokens=65540)
    assert scheduler._mamba_block_aligned_split(request, 3) == 3
    assert scheduler._clips == 0
    assert scheduler._step_prefill_tokens == 4099
    assert scheduler._step_pairs == pytest.approx(3.0 * 65536)


def test_external_and_local_computed_tokens_count_as_prefix():
    scheduler = _scheduler(cost=_model())
    got = scheduler._mamba_block_aligned_split(
        _request(0, 65536),
        16384,
        num_new_local_computed_tokens=8192,
        num_external_computed_tokens=8192,
    )
    assert got == 12288


def test_mamba_alignment_runs_after_the_clip_when_the_model_needs_it():
    scheduler = _scheduler(mamba_split=True, cost=_model())
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
    scheduler = _scheduler(cost=_model(granularity=2048), step_limit=4096)
    assert scheduler._mamba_block_aligned_split(_request(0, 65536), 16384) == 4096
    assert scheduler._clips == 1
    # The step is full: the next request waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"), 16384) == 0
    assert scheduler._waits == 1


def test_the_step_limit_spreads_the_pending_prefill_over_the_stages():
    model = _model(granularity=2048)
    lone = [_request(0, 32768)]
    scheduler = _scheduler(cost=model, running=lone)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    # 32K over 8 stages: 4K steps.
    assert scheduler._step_limit == 4096
    # 8K left of a prompt: 2K steps pad to the 4K bucket and take as long
    # as 4K ones, so two 4K steps finish sooner than four.
    scheduler = _scheduler(cost=model, running=[_request(24576, 32768)])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 4096
    # Waiting requests count; enough work keeps full steps.
    waiting = [_request(0, 65536, f"w{i}") for i in range(4)]
    scheduler = _scheduler(cost=model, running=lone, waiting=waiting)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 16384
    # Requests whose prefill is fully scheduled add nothing: no limit.
    scheduler = _scheduler(cost=model, running=[_decode("d")])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None
    # Without a model there is no limit.
    scheduler = _scheduler(running=lone)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit is None


def test_the_step_limit_keeps_a_prompt_whole_when_halves_are_no_faster():
    # A 4K step costs more than an 8K one: a lone 8K prompt on 4 stages
    # leaves sooner as one step (4 steps of 109 ms) than as two (5 of 118).
    buckets = [4096, 8192, 12288, 16384]
    linear = {4096: 118.0, 8192: 109.0, 12288: 133.0, 16384: 164.0}
    model = StepCostModel(buckets, linear, 3.3e-8, 330.0, 4096)
    scheduler = _scheduler(cost=model, running=[_request(0, 8192)])
    scheduler._pp_size = 4
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
        assert scheduler._step_limit == 8192
        # Two such prompts: one each per step, not both in one.
        scheduler.waiting = [_request(0, 8192, "w")]
        scheduler.schedule()
        assert scheduler._step_limit == 8192
        # Four: still one prompt per step, all four stages busy.
        scheduler.waiting = [_request(0, 8192, f"w{i}") for i in range(3)]
        scheduler.schedule()
        assert scheduler._step_limit == 8192


def test_schedule_resets_the_step_accounting():
    scheduler = _scheduler(cost=_model())
    scheduler._step_tokens = 5
    scheduler._step_pairs = 7.0
    scheduler._step_squares = 9.0
    scheduler._step_prefill_tokens = 5
    scheduler._seeded = {"x"}
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP) as schedule:
        assert scheduler.schedule(throttle_prefills=True) is _EMPTY_STEP
    schedule.assert_called_once_with(True)
    assert scheduler._step_tokens == 0 and scheduler._step_pairs == 0.0
    assert scheduler._step_squares == 0.0
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


def _report(offset: float, granularity=4096):
    return {
        "granularity": granularity,
        "samples": [(b, 0, offset + b * 0.01) for b in BUCKETS],
        "schedule": (14000 - int(offset), 256, 256),
    }


def test_profile_and_install_merges_every_stage(monkeypatch):
    monkeypatch.setattr(pp_chunk_scheduler.envs, "TPU_PP_CHUNK_SLACK", 0.1)
    executor = _Executor([_report(10.0), _report(20.0)])
    scheduler = _scheduler()
    profile_and_install_cost_model(scheduler, executor)
    assert executor.calls == ["profile_pipeline_chunks"]
    assert scheduler._cost.linear_ms[16384] == pytest.approx(20.0 + 163.84)
    assert scheduler._cost.target_ms == pytest.approx(1.1 * (20.0 + 163.84))
    assert scheduler._cost.granularity == 4096
    # The smallest stage capacity is kept.
    assert scheduler._cost.schedule == (13980, 256, 256)


def test_profile_and_install_skips_other_schedulers(monkeypatch):
    executor = _Executor([])
    profile_and_install_cost_model(object(), executor)
    assert executor.calls == []


def test_engine_core_patch_installs_after_init_of_subclasses(monkeypatch):
    from vllm.v1.engine import core as engine_core

    monkeypatch.setattr(pp_chunk_scheduler.envs, "TPU_PP_CHUNK_SLACK", 0.1)

    class FakeCore:
        def __init__(self, *args, **kwargs):
            self.scheduler = _scheduler()
            self.model_executor = _Executor([_report(10.0)])

    class FakeProc(FakeCore):
        def __init__(self):
            super().__init__()
            self.after_init = self.scheduler._cost

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    patch_engine_core_for_pp_chunks(_config())
    assert FakeCore._tpu_pp_chunks_patch
    assert FakeCore().scheduler._cost is not None
    assert FakeProc().after_init is not None
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


def test_the_step_estimate_prices_the_step_the_cost_model_sees():
    cost = _model()
    scheduler = _scheduler(cost=cost)
    # schedule() has already counted the step's tokens as computed
    scheduler.requests = {
        "a": _request(12288, 32768, "a"),
        "d": _request(8193, 8192, "d", num_tokens=8193),
    }
    step = SimpleNamespace(
        num_scheduled_tokens={"a": 4096, "d": 1, "gone": 7},
        total_num_scheduled_tokens=4104,
    )
    pairs = 4096 * 8192 + 1 * 8192
    squares = 4096**2 + 1 + 49
    assert scheduler.step_estimate_ms(step) == cost.step_ms(8192, pairs, squares)
    # past the largest bucket the step is priced at that bucket
    step.total_num_scheduled_tokens = 20000
    assert scheduler.step_estimate_ms(step) == cost.step_ms(16384, pairs, squares)
    assert _scheduler().step_estimate_ms(step) is None


def test_the_step_limit_costs_the_spreads_against_the_steps_in_flight():
    model = _model()
    full = model.step_ms(16384, 0.0, float(16384**2))
    # A lone 8K prompt on 4 idle stages leaves soonest as two 4K steps.
    scheduler = _scheduler(cost=model, running=[_request(0, 8192)])
    scheduler._pp_size = 4
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
        assert scheduler._step_limit == 4096
        # Behind a full step the next three steps last a full step
        # whatever enters, so one 8K step gets the prompt out a step earlier.
        scheduler._inflight_ms = [full]
        scheduler.schedule()
        assert scheduler._step_limit == 8192
        # A full step that leaves the pipeline this step no longer counts.
        scheduler._inflight_ms = [0.0, 0.0, full]
        scheduler.schedule()
        assert scheduler._step_limit == 4096


def test_the_live_history_window_never_wraps_around():
    # PP4, pending 8K, history [4K, 16K, 4K] newest first. Candidate 4K
    # steps run five steps; the fifth overlaps nothing in flight. A
    # negative slice bound there would read the full step back in and
    # make the 8K step look faster.
    model = _model()
    small = model.step_ms(4096, 0.0, float(4096**2))
    full = model.step_ms(16384, 0.0, float(16384**2))
    scheduler = _scheduler(cost=model, running=[_request(0, 8192)])
    scheduler._pp_size = 4
    scheduler._inflight_ms = [small, full, small]
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 4096
    # The same history costed by hand: the 8K step waits out both full
    # steps and runs two of its own; the 4K steps wait them out and run
    # three of their own, which is the cheaper of the two.
    eight = model.step_ms(8192, 0.0, float(8192**2))
    assert 2 * full + 3 * small < 2 * full + 2 * eight


def test_schedule_keeps_the_last_stage_count_minus_one_step_estimates():
    model = _model()
    scheduler = _scheduler(cost=model)
    scheduler._pp_size = 4
    scheduler.requests = {"a": _request(4096, 32768, "a")}
    step = SimpleNamespace(
        num_scheduled_tokens={"a": 4096}, total_num_scheduled_tokens=4096
    )
    estimate = model.step_ms(4096, 0.0, float(4096**2))
    with patch.object(Scheduler, "schedule", return_value=step):
        for _ in range(5):
            scheduler.schedule()
    assert scheduler._inflight_ms == [estimate] * 3
    # An empty step launches nothing and leaves the pipeline as it was.
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._inflight_ms == [estimate] * 3


def test_pushes_and_settles_age_the_steps_in_flight():
    model = _model()
    scheduler = _scheduler(cost=model)
    scheduler._pp_size = 4
    estimate = model.step_ms(4096, 0.0, float(4096**2))
    scheduler._inflight_ms = [estimate] * 3
    # Each push is a step that carries nothing: it enters at no cost and
    # the oldest step leaves.
    scheduler.pipeline_advanced(pushes=2)
    assert scheduler._inflight_ms == [0.0, 0.0, estimate]
    scheduler.pipeline_advanced(pushes=1)
    assert scheduler._inflight_ms == [0.0, 0.0, 0.0]
    # A settle is sent once nothing is in flight.
    scheduler._inflight_ms = [estimate] * 3
    scheduler.pipeline_advanced(settled=True)
    assert scheduler._inflight_ms == []
    # After a settle a new burst is costed against an empty pipeline.
    scheduler.running = [_request(0, 8192)]
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 4096
    scheduler.pipeline_advanced(pushes=0)
    assert scheduler._inflight_ms == []


def test_the_engine_core_patch_wires_the_executor_pushes_to_the_scheduler(
    monkeypatch,
):
    from vllm.v1.engine import core as engine_core

    from vllm_torchtpu.distributed import pp_push

    monkeypatch.setattr(pp_chunk_scheduler.envs, "TPU_PP_CHUNK_SLACK", 0.1)

    class Executor(_Executor):
        def __init__(self, reports):
            super().__init__(reports)
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(pipeline_parallel_size=4)
            )
            self.output_rank = 3

        def collective_rpc(self, method, **kwargs):
            self.calls.append(method)
            return self.reports

    class FakeCore:
        def __init__(self):
            self.scheduler = _scheduler()
            self.model_executor = Executor([_report(10.0)])

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    patch_engine_core_for_pp_chunks(_config(pp=4))
    core = FakeCore()
    assert core.scheduler._cost is not None
    # install_cost_model read the fake config's eight stages; test four.
    core.scheduler._pp_size = 4
    state = pp_push.wave_state(core.model_executor)
    assert state.listener == core.scheduler.pipeline_advanced
    core.scheduler._inflight_ms = [5.0, 5.0, 5.0]
    state.push(core.model_executor)
    assert core.scheduler._inflight_ms == [0.0, 5.0, 5.0]
    state.settle(core.model_executor)
    assert core.scheduler._inflight_ms == []
    # Executors without a pipeline config are left alone.
    pp_chunk_scheduler.attach_pipeline_listener(_scheduler(), _Executor([]))


class _Retention:
    """Records the retention calls and clips every chunk at ``stop``."""

    enabled = True

    def __init__(self, stop=None, fail=False):
        self.stop = stop
        self.fail = fail
        self.calls: list[tuple] = []

    def pin_running(self, running):
        self.calls.append(("pin", tuple(r.request_id for r in running)))
        if self.fail:
            raise RuntimeError("pin")

    def begin_step(self):
        self.calls.append(("begin",))

    def prepare_prefill_split(self, request, start, num_new_tokens):
        self.calls.append(("prepare", request.request_id, start, num_new_tokens))

    def adjust_prefill_split(self, request, start, aligned):
        self.calls.append(("adjust", request.request_id, start, aligned))
        if self.stop is not None and start < self.stop < start + aligned:
            return self.stop - start
        return aligned

    def snapshot_attention_blocks(self, request):
        self.calls.append(("snapshot", request.request_id))
        return ["attn"]

    def on_request_freed(self, request, attn_blocks):
        self.calls.append(("freed", request.request_id, attn_blocks))


def test_retention_stops_apply_after_the_time_budget_clip():
    # The clip leaves 4096 of 16384; retention stops the chunk 1024 in.
    scheduler = _scheduler(cost=_model(), step_limit=4096)
    scheduler._retention = _Retention(stop=1024)
    request = _request(0, 65536)
    got = scheduler._mamba_block_aligned_split(request, 16384)
    assert got == 1024
    assert scheduler._retention.calls == [
        ("prepare", "r", 0, 4096),
        ("adjust", "r", 0, 4096),
    ]
    assert scheduler._step_tokens == 1024
    assert scheduler._step_prefill_tokens == 1024


def test_retention_runs_after_the_mamba_alignment():
    scheduler = _scheduler(mamba_split=True, cost=_model(), step_limit=4096)
    scheduler._retention = _Retention()
    request = _request(0, 65536)
    with patch.object(Scheduler, "_mamba_block_aligned_split", return_value=3840):
        assert scheduler._mamba_block_aligned_split(request, 16384) == 3840
    assert scheduler._retention.calls[-1] == ("adjust", "r", 0, 3840)


def test_single_step_plan_replaces_the_mamba_alignment(monkeypatch):
    scheduler = _scheduler(mamba_split=True, cost=_model(), step_limit=4096)
    scheduler._single_step = True
    scheduler._retention = _Retention()
    plans = []
    monkeypatch.setattr(
        pp_chunk_scheduler.single_step_prefill,
        "plan_single_step_split",
        lambda sched, request, start, tokens, retention: (tokens - 128, (640, 1152)),
    )
    monkeypatch.setattr(
        pp_chunk_scheduler.single_step_prefill,
        "set_plan",
        lambda kvm, rid, plan: plans.append((rid, plan)),
    )
    request = _request(0, 65536)
    with patch.object(Scheduler, "_mamba_block_aligned_split") as base:
        assert scheduler._mamba_block_aligned_split(request, 16384) == 4096 - 128
    base.assert_not_called()
    assert plans == [("r", (0, 640, 1152))]
    assert scheduler._retention.calls == [("prepare", "r", 0, 4096)]
    assert scheduler._step_tokens == 4096 - 128


def test_a_failing_retention_hook_disables_retention():
    scheduler = _scheduler(running=[_decode("d")])
    scheduler._retention = _Retention(fail=True)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._retention is None


def test_schedule_pins_the_running_requests_and_attaches_the_cuts(monkeypatch):
    scheduler = _scheduler(running=[_decode("d")])
    retention = _Retention()
    scheduler._retention = retention
    scheduler._single_step = True
    output = SimpleNamespace(
        total_num_scheduled_tokens=1, num_scheduled_tokens={"d": 1}
    )
    begun = []
    monkeypatch.setattr(
        pp_chunk_scheduler.single_step_prefill,
        "begin_step",
        lambda kvm: begun.append(kvm),
    )
    monkeypatch.setattr(
        pp_chunk_scheduler.single_step_prefill,
        "confirmed_cuts",
        lambda kvm, scheduled: {"d": (0, 640, 0)},
    )
    with patch.object(Scheduler, "schedule", return_value=output):
        assert scheduler.schedule() is output
    assert retention.calls[:2] == [("pin", ("d",)), ("begin",)]
    assert begun == [scheduler.kv_cache_manager]
    assert output.tpu_mamba_seg_cuts == {"d": (0, 640, 0)}
    assert scheduler._n_single_step_reqs == 1


def test_freeing_a_request_snapshots_its_attention_blocks_first():
    scheduler = _scheduler()
    retention = _Retention()
    scheduler._retention = retention
    request = _request(100, 100, "f")
    with patch.object(Scheduler, "_free_request_blocks", return_value=None) as base:
        scheduler._free_request_blocks(request)
    base.assert_called_once_with(request)
    assert retention.calls == [("snapshot", "f"), ("freed", "f", ["attn"])]


def test_a_finished_connector_request_is_pinned():
    scheduler = _scheduler()
    retention = _Retention()
    scheduler._retention = retention
    request = _request(100, 100, "c")
    with patch.object(Scheduler, "_connector_finished", return_value=(True, None)):
        assert scheduler._connector_finished(request) == (True, None)
    with patch.object(Scheduler, "_connector_finished", return_value=(False, None)):
        scheduler._connector_finished(request)
    assert retention.calls == [("pin", ("c",))]


def test_a_padded_last_prompt_token_is_cut_back_to_the_prompt_end():
    scheduler = _scheduler(cost=_model())
    request = _request(0, 2 * 2176 + 1)
    got = scheduler._mamba_block_aligned_split(
        request, 2, num_new_local_computed_tokens=2 * 2176
    )
    assert got == 1


def test_chunk_sizing_uses_live_lockstep_wave_ceiling():
    scheduler = _scheduler(cost=_model())
    # Without in-flight steps above target_ms, 16K behind 16K clips to 12288.
    assert scheduler._mamba_block_aligned_split(_request(16384, 65536), 16384) == 12288
    # When an in-flight step in the wave already takes 260 ms, the next step
    # can fill to 16384 without slowing the lockstep wave.
    scheduler = _scheduler(cost=_model())
    scheduler._inflight_ms = [260.0]
    assert scheduler._mamba_block_aligned_split(_request(16384, 65536), 16384) == 16384


def test_running_srpf_orders_by_quadratic_effective_cost():
    model = _model()
    # r_deep has fewer raw remaining tokens (3800) than r_shallow (4096),
    # but r_deep is behind a 160K prefix so its quadratic prefill cost is
    # much higher than r_shallow at prefix 0.
    r_deep = _request(163840, 163840 + 3800, "deep")
    r_shallow = _request(0, 4096, "shallow")
    scheduler = _scheduler(cost=model, running=[r_deep, r_shallow])
    scheduler._srpf_running = True
    scheduler._paw_aging_tok_per_s = 0.0
    scheduler._paw_max_wait_s = 0.0
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ["shallow", "deep"]


def test_complementary_step_packing_pairs_warm_and_cold_chunks():
    model = _model()
    prefix = 112 * 1024
    # Two warm requests behind 112K prefix (each clips to 4096) and one cold
    # request (0 prefix, 65536 tokens). After warm_1 takes 4096 tokens, warm_2
    # cannot fit in the same step without blowing target_ms, so cold_1 is
    # promoted into the 2nd slot to fill 8192 complementary tokens while
    # warm_2 stays next for the following step.
    w1 = _request(prefix, prefix + 16384, "warm1")
    w2 = _request(prefix, prefix + 16384, "warm2")
    cold = _request(0, 65536, "cold")
    scheduler = _scheduler(cost=model, waiting=[w1, w2, cold])
    scheduler.WINDOW = 16
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.waiting] == ["warm1", "cold", "warm2"]
    assert scheduler._mamba_block_aligned_split(scheduler.waiting[0], 16384) == 4096
    assert scheduler._mamba_block_aligned_split(scheduler.waiting[1], 12288) == 8192
    assert scheduler._mamba_block_aligned_split(scheduler.waiting[2], 4096) == 0
    assert scheduler._step_tokens == 12288


def test_complementary_step_packing_across_running_and_waiting():
    model = _model()
    prefix = 112 * 1024
    # Running warm_1 clips at 4096; waiting has [warm_2, cold]. Because warm_1
    # already clipped on attention in running, cold is promoted to the head of
    # waiting so super().schedule()'s waiting loop does not break on warm_2.
    w1 = _request(prefix, prefix + 16384, "warm1")
    w2 = _request(prefix, prefix + 16384, "warm2")
    cold = _request(0, 65536, "cold")
    scheduler = _scheduler(cost=model, running=[w1], waiting=[w2, cold])
    scheduler.WINDOW = 16
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.waiting] == ["cold", "warm2"]
