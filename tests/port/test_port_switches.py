# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port switches that restore what the old trees ran.

Each defaults to this branch's (gated) behaviour:
- TPU_PP_QUAD_COST=0: the PP chunk scheduler without PR #2243's quadratic cost (linear
  keys, waiting work from num_computed_tokens, no pending pairs, no live target ceiling).
- TPU_PP_INFLIGHT_COST=0: no in-flight step costing (4ec32d152 / 06f7b9bf9).
- TPU_SCHED_PAW_PROMOTE_WARMEST=0: FCFS admission, no warmest-first promotion.
- TPU_SCHED_DECODE_INFLIGHT_THROTTLE=1 (default 0): the final-lineage decode admission
  throttle (_decode_inflight_reserved_blocks, fp2q-r5 0447ff58b).
"""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import pp_chunk_scheduler as pcs
from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


def _model(attn=1.5e-7, slack=0.1, granularity=4096) -> StepCostModel:
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack), granularity)


def _sched(cost=None, running=(), waiting=(), **attrs) -> TpuPipelineChunkScheduler:
    s = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    s._cost = cost
    s._step_limit = None
    s._pp_size = 8
    s.waiting = list(waiting)
    s.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        parallel_config=SimpleNamespace(pipeline_parallel_size=8),
    )
    s._mamba_split = False
    s.need_mamba_block_aligned_split = True
    s.running = list(running)
    s._step_tokens = 0
    s._step_pairs = 0.0
    s._step_squares = 0.0
    s._step_prefill_tokens = 0
    s._step_schedule = 0
    s._seeded = set()
    s._inflight_ms = []
    s._clips = 0
    s._waits = 0
    s._retention = None
    s._single_step = False
    s._n_single_step_reqs = 0
    s.kv_cache_manager = SimpleNamespace()
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _req(computed, prompt, rid="r", num_tokens=None):
    return SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt,
                           num_tokens=prompt if num_tokens is None else num_tokens, request_id=rid)


# ---------------- switch parsing and wiring ----------------


@pytest.mark.parametrize("val,expect", [(None, True), ("1", True), ("0", False), ("false", False),
                                        ("off", False), ("OFF", False)])
def test_env_switch(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("TPU_PP_QUAD_COST", raising=False)
    else:
        monkeypatch.setenv("TPU_PP_QUAD_COST", val)
    assert pcs._env_switch("TPU_PP_QUAD_COST") is expect


def test_init_reads_the_three_switches():
    src = inspect.getsource(TpuPipelineChunkScheduler.__init__)
    for name, attr in (("TPU_SCHED_PAW_PROMOTE_WARMEST", "_promote_waiting"),
                       ("TPU_PP_QUAD_COST", "_quad"), ("TPU_PP_INFLIGHT_COST", "_inflight_cost")):
        assert f'self.{attr} = _env_switch("{name}")' in src


# ---------------- TPU_PP_QUAD_COST ----------------


def test_quad_off_drops_the_live_target_ceiling():
    # With a 260 ms step in flight the PR lets 16K behind 16K fill to 16384; without it the
    # chunk clips to 12288, as before PR #2243.
    on = _sched(cost=_model(), _inflight_ms=[260.0])
    assert on._mamba_block_aligned_split(_req(16384, 65536), 16384) == 16384
    off = _sched(cost=_model(), _inflight_ms=[260.0], _quad=False)
    assert off._mamba_block_aligned_split(_req(16384, 65536), 16384) == 12288


def test_quad_off_cost_is_linear():
    s = _sched(cost=_model())
    assert s._quad_cost(3800, 163840) > 3800
    s._quad = False
    assert s._quad_cost(3800, 163840) == 3800.0
    assert s._quad_cost(-5, 10) == 0.0


def test_quad_off_running_srpf_is_linear():
    deep = _req(163840, 163840 + 3800, "deep")
    shallow = _req(0, 4096, "shallow")
    for quad, order in ((True, ["shallow", "deep"]), (False, ["deep", "shallow"])):
        s = _sched(cost=_model(), running=[deep, shallow], _quad=quad, _srpf_running=True,
                   _paw_aging_tok_per_s=0.0, _paw_max_wait_s=0.0)
        with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
            s.schedule()
        assert [r.request_id for r in s.running] == order


def test_quad_off_pending_limit_has_no_pairs_and_no_waiting_warmth():
    model = _model()
    seen = []
    real = model.step_ms

    def spy(bucket, pairs, squares):
        seen.append(pairs)
        return real(bucket, pairs, squares)

    def no_warmth(_r):
        raise AssertionError("_waiting_warmth consulted with TPU_PP_QUAD_COST=0")

    running = [_req(32768, 65536, "a")]
    waiting = [_req(0, 40000, "w")]
    s = _sched(cost=model, running=running, waiting=waiting, _quad=False)
    s._waiting_warmth = no_warmth
    with patch.object(model, "step_ms", side_effect=spy):
        s._pending_step_limit()
    assert seen and all(p == 0.0 for p in seen)
    seen.clear()
    s = _sched(cost=model, running=running, waiting=waiting)
    s._waiting_warmth = lambda r: 0
    with patch.object(model, "step_ms", side_effect=spy):
        s._pending_step_limit()
    assert any(p > 0.0 for p in seen)


# ---------------- TPU_PP_INFLIGHT_COST ----------------


def test_inflight_off_keeps_no_history():
    model = _model()
    est = model.step_ms(4096, 0.0, float(4096**2))
    on = _sched(cost=model)
    on._pp_size = 4
    on._enter(est)
    on.pipeline_advanced(pushes=2)
    assert on._inflight_ms == [0.0, 0.0, est]
    off = _sched(cost=model, _inflight_cost=False)
    off._pp_size = 4
    off._enter(est)
    off.pipeline_advanced(pushes=2)
    off.pipeline_advanced(settled=True)
    assert off._inflight_ms == []
    off.running = [_req(0, 8192)]
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        off.schedule()
    assert off._inflight_ms == []


# ---------------- TPU_SCHED_PAW_PROMOTE_WARMEST ----------------


@pytest.mark.parametrize("val,expect", [(None, True), ("1", True), ("0", False), ("off", False)])
def test_promote_switch(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("TPU_SCHED_PAW_PROMOTE_WARMEST", raising=False)
    else:
        monkeypatch.setenv("TPU_SCHED_PAW_PROMOTE_WARMEST", val)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    assert s._promote_warmest_on() is expect


def test_chunk_scheduler_promotion_follows_the_switch():
    calls = []
    for on in (True, False):
        s = _sched(cost=None, _promote_waiting=on)
        s._select_waiting_queue_for_scheduling = lambda: calls.append(on)
        with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
            s.schedule()
    assert calls == [True]


def test_pas_schedule_guards_the_promotion():
    src = inspect.getsource(PrefixAwareScheduler.schedule)
    assert "if self._promote_warmest_on():\n                self._select_waiting_queue_for_scheduling()" in src
