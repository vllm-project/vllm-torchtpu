from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import pp_chunk_scheduler
from vllm_torchtpu.core.pp_chunk_scheduler import (
    SCHEDULER_CLS, TpuPipelineChunkScheduler, patch_engine_core_for_pp_chunks,
    profile_and_install_cost_model, uses_dynamic_chunks)
from vllm_torchtpu.core.pp_chunks import StepCostModel

BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


def _model(attn=1.5e-7, slack=0.1, granularity=4096) -> StepCostModel:
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack),
                         granularity)


def _scheduler(mamba_split=False,
               cost=None,
               running=(),
               waiting=(),
               step_limit=None) -> TpuPipelineChunkScheduler:
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    scheduler._cost = cost
    scheduler._step_limit = step_limit
    scheduler._pp_size = 8
    scheduler.waiting = list(waiting)
    scheduler.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        parallel_config=SimpleNamespace(pipeline_parallel_size=8))
    scheduler._mamba_split = mamba_split
    scheduler.need_mamba_block_aligned_split = True
    scheduler.running = list(running)
    scheduler._step_tokens = 0
    scheduler._step_pairs = 0.0
    scheduler._step_squares = 0.0
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
        request_id=req_id)


def _decode(req_id: str, computed: int = 8192):
    return _request(computed, computed, req_id, num_tokens=computed + 1)


def test_without_a_model_chunks_pass_through():
    scheduler = _scheduler()
    assert scheduler._mamba_block_aligned_split(_request(0, 65536),
                                                16384) == 16384
    assert scheduler._step_tokens == 16384
    assert scheduler._step_prefill_tokens == 16384


def test_prefix_free_first_chunk_is_untouched():
    scheduler = _scheduler(cost=_model())
    assert scheduler._mamba_block_aligned_split(_request(0, 65536),
                                                16384) == 16384
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
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"),
                                                16384) == 0
    assert scheduler._waits == 1
    assert scheduler._step_tokens == 12288


def test_a_light_chunk_leaves_room_for_the_next_request():
    scheduler = _scheduler(cost=_model())
    assert scheduler._mamba_block_aligned_split(_request(16384, 20480),
                                                4096) == 4096
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"),
                                                16384) == 12288
    assert scheduler._step_tokens == 16384


def test_the_first_prefill_of_a_step_advances_even_behind_decodes():
    # Decodes so deep in their contexts that they alone exceed the target.
    decodes = [_decode(f"d{i}", 40_000_000) for i in range(32)]
    scheduler = _scheduler(cost=_model(), running=decodes)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 32
    assert scheduler._step_prefill_tokens == 0
    assert scheduler._cost.chunk(65536, 16384, 32, scheduler._step_pairs,
                                 32.0) == 0
    # One granule for the first prompt anyway.
    assert scheduler._mamba_block_aligned_split(_request(65536, 131072),
                                                16384) == 4096
    # A second prompt waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(65536, 131072, "s"),
                                                16384) == 0
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
    assert scheduler._mamba_block_aligned_split(_request(0, 65536),
                                                16384 - 37) == 16384 - 37
    assert scheduler._step_tokens == 16384


def test_a_prefill_still_in_flight_is_not_seeded_as_a_decode():
    in_flight = [_request(8192, 8192, f"p{i}") for i in range(6)]
    scheduler = _scheduler(cost=_model(), running=in_flight)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_tokens == 0 and not scheduler._seeded
    # Two whole 8K prompts still share one step.
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "a"),
                                                8192) == 8192
    assert scheduler._mamba_block_aligned_split(_request(0, 8192, "b"),
                                                8192) == 8192


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
        num_external_computed_tokens=8192)
    assert got == 12288


def test_mamba_alignment_runs_after_the_clip_when_the_model_needs_it():
    scheduler = _scheduler(mamba_split=True, cost=_model())
    seen = {}

    def align(self, request, num_new_tokens, local=0, external=0):
        seen["tokens"] = num_new_tokens
        return num_new_tokens - 1

    with patch.object(Scheduler, "_mamba_block_aligned_split", align):
        got = scheduler._mamba_block_aligned_split(_request(16384, 65536),
                                                   16384)
    assert seen["tokens"] == 12288
    assert got == 12287
    assert scheduler._step_tokens == 12287


def test_a_limited_step_takes_one_small_chunk_and_fills_with_it():
    scheduler = _scheduler(cost=_model(granularity=2048), step_limit=4096)
    assert scheduler._mamba_block_aligned_split(_request(0, 65536),
                                                16384) == 4096
    assert scheduler._clips == 1
    # The step is full: the next request waits for the next step.
    assert scheduler._mamba_block_aligned_split(_request(0, 65536, "s"),
                                                16384) == 0
    assert scheduler._waits == 1


def test_the_step_limit_spreads_the_pending_prefill_over_the_stages():
    model = _model(granularity=2048)
    lone = [_request(0, 32768)]
    scheduler = _scheduler(cost=model, running=lone)
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    # 32K over 8 stages: 4K steps.
    assert scheduler._step_limit == 4096
    # 8K left of a prompt: one granule per stage is the floor.
    scheduler = _scheduler(cost=model, running=[_request(24576, 32768)])
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        scheduler.schedule()
    assert scheduler._step_limit == 2048
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


def test_schedule_resets_the_step_accounting():
    scheduler = _scheduler(cost=_model())
    scheduler._step_tokens = 5
    scheduler._step_pairs = 7.0
    scheduler._step_squares = 9.0
    scheduler._step_prefill_tokens = 5
    scheduler._seeded = {"x"}
    with patch.object(Scheduler, "schedule", return_value="out") as schedule:
        assert scheduler.schedule(throttle_prefills=True) == "out"
    schedule.assert_called_once_with(True)
    assert scheduler._step_tokens == 0 and scheduler._step_pairs == 0.0
    assert scheduler._step_squares == 0.0
    assert scheduler._step_prefill_tokens == 0 and not scheduler._seeded


def _config(pp=8, cls=SCHEDULER_CLS):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(scheduler_cls=cls))


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
