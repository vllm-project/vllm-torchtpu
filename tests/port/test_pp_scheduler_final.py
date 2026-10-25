# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port: TPU_PP_SCHEDULER=final installs the final lineage's PP8 chunk scheduler
(a PrefixAwareScheduler subclass, final-p2 code); unset keeps unified's flavour."""
import collections
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import pp_chunk_scheduler as pcs
from vllm_torchtpu.core.pp_chunk_scheduler import (
    SCHEDULER_CLS,
    SCHEDULER_CLS_FINAL,
    TpuPipelineChunkScheduler,
    uses_dynamic_chunks,
)
from vllm_torchtpu.core.pp_chunk_scheduler_final import TpuPipelineChunkSchedulerFinal
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.platforms import tpu_platform

BUCKETS = [2176, 4352, 6528, 8192]
G = 2176
_EMPTY = SimpleNamespace(total_num_scheduled_tokens=0)


def _model(attn=9.2e-8, slack=0.5):
    linear = {2176: 33.3, 4352: 49.1, 6528: 64.3, 8192: 76.6}  # pilot_D prefill log (M)
    full = linear[8192] + attn * 8192**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack), G)


# ---------------- platform selection ----------------


def _config(cls=None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=8, data_parallel_size=1),
        scheduler_config=SimpleNamespace(scheduler_cls=cls, enable_chunked_prefill=True),
    )


def _configure(monkeypatch, config):
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_DYNAMIC_CHUNKS", True)
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", False)
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_CHUNK_SLACK", 0.5)
    with patch("vllm_torchtpu.core.pp_chunk_scheduler.patch_engine_core_for_pp_chunks") as hook:
        tpu_platform._configure_pipeline_chunks(config)
    return hook


@pytest.mark.parametrize("val,expect", [(None, SCHEDULER_CLS), ("", SCHEDULER_CLS),
                                        ("unified", SCHEDULER_CLS), ("final", SCHEDULER_CLS_FINAL),
                                        ("FINAL", SCHEDULER_CLS_FINAL)])
def test_platform_installs_the_selected_flavour(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("TPU_PP_SCHEDULER", raising=False)
    else:
        monkeypatch.setenv("TPU_PP_SCHEDULER", val)
    for cls in (None, "vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler"):
        config = _config(cls)
        hook = _configure(monkeypatch, config)
        assert config.scheduler_config.scheduler_cls == expect
        hook.assert_called_once_with(config)
        assert uses_dynamic_chunks(config)


def test_final_flavour_is_a_prefix_aware_scheduler_and_resolves():
    from vllm.utils.import_utils import resolve_obj_by_qualname

    assert resolve_obj_by_qualname(SCHEDULER_CLS_FINAL) is TpuPipelineChunkSchedulerFinal
    assert issubclass(TpuPipelineChunkSchedulerFinal, PrefixAwareScheduler)
    assert not issubclass(TpuPipelineChunkScheduler, PrefixAwareScheduler)


def test_cost_model_installs_on_the_final_flavour():
    s = TpuPipelineChunkSchedulerFinal.__new__(TpuPipelineChunkSchedulerFinal)
    s.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=8))
    report = {"granularity": G, "samples": [(b, 0.0, float(b * b), 30.0 + b / 100) for b in BUCKETS]}
    ex = SimpleNamespace(collective_rpc=lambda name: [report])
    with patch.object(StepCostModel, "fit", return_value=_model()):
        pcs.profile_and_install_cost_model(s, ex)
    assert s._cost is not None and s._pp_size == 8


# ---------------- behaviour (harness as in tests/test_pp_chunk_scheduler.py) ----------------


def _base(s, cost, running=(), waiting=()):
    s._cost = cost
    s._step_limit = None
    s._pp_size = 8
    s.waiting = list(waiting)
    s.running = list(running)
    s.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=8))
    s._mamba_split = True
    s.need_mamba_block_aligned_split = True
    s.block_size = G
    s._step_tokens = 0
    s._step_pairs = 0.0
    s._step_squares = 0.0
    s._step_prefill_tokens = 0
    s._step_schedule = 0
    s._seeded = set()
    s._clips = 0
    s._waits = 0
    s.kv_cache_manager = SimpleNamespace()
    return s


def _unified(cost, quad=True, inflight=True, **kw):
    s = _base(TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler), cost, **kw)
    s._inflight_ms = []
    s._retention = None
    s._single_step = False
    s._n_single_step_reqs = 0
    s._quad = quad
    s._inflight_cost = inflight
    s._promote_waiting = False
    return s


def _final(cost, **kw):
    return _base(TpuPipelineChunkSchedulerFinal.__new__(TpuPipelineChunkSchedulerFinal), cost, **kw)


def _req(computed, prompt, rid="r"):
    return SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=prompt,
                           request_id=rid)


def _align_to_blocks(self, request, n, local=0, ext=0):
    """Stand-in for the base Mamba split: end on a block boundary or the prompt end, else 0."""
    start = request.num_computed_tokens + local + ext
    end = start + n
    if end >= request.num_prompt_tokens:
        return n
    return max((end // G) * G - start, 0)


def test_final_step_limit_ignores_pairs_and_in_flight_steps():
    model = _model()
    seen = []
    real = model.step_ms
    def spy(bucket, pairs, squares):
        seen.append(pairs)
        return real(bucket, pairs, squares)
    with patch.object(model, "step_ms", side_effect=spy):
        _final(model, running=[_req(32768, 65536, "a")], waiting=[_req(0, 40000, "w")])._pending_step_limit()
    assert seen and all(p == 0.0 for p in seen)
    running = [_req(32768, 65536, "a")]
    waiting = [_req(0, 40000, "w")]
    f = _final(model, running=running, waiting=waiting)
    u = _unified(model, quad=False, inflight=False, running=running, waiting=waiting)
    u._waiting_warmth = lambda r: 0
    assert f._pending_step_limit() == u._pending_step_limit()
    f._inflight_ms = [500.0]  # not an attribute the final flavour reads
    assert f._pending_step_limit() == u._pending_step_limit()


def test_final_retries_a_declined_first_chunk_at_the_minimum_step():
    # The base split declines the step's first chunk (ends mid-block: 1000 + 2176): unified's
    # flavour waits (0 tokens this step); the final flavour retries at max(block, granularity).
    model = _model()
    r = _req(1000, 200000)
    with patch.object(PrefixAwareScheduler, "_mamba_block_aligned_split",
                      lambda self, req, n, *a: 0 if n > G else n), \
         patch.object(Scheduler, "_mamba_block_aligned_split", lambda self, req, n, *a: 0):
        f = _final(model)
        f._step_limit = 8192
        u = _unified(model, quad=False, inflight=False)
        u._step_limit = 8192
        assert f._mamba_block_aligned_split(r, 8192) == G
        assert u._mamba_block_aligned_split(r, 8192) == 0


def _simulate(make, steps=40, n_reqs=24, seed=7):
    """A synthetic PP8 prefill queue: long prompts behind 128-token-granular prefix hits.
    Each step runs schedule() (base patched out) and then sizes chunks for the queue in order
    up to the 8192-token budget, as the base scheduler loop would. Returns the step-bucket mix."""
    import random

    rnd = random.Random(seed)
    reqs = []
    for i in range(n_reqs):
        hit = rnd.randrange(300, 1200) * 128  # agentic turns: 38K-154K cached prefix
        reqs.append(_req(hit, hit + rnd.randrange(2000, 30000), f"q{i}"))
    s = make(_model(), waiting=reqs)
    mix = collections.Counter()
    with patch.object(Scheduler, "schedule", return_value=_EMPTY), \
         patch.object(PrefixAwareScheduler, "schedule", lambda self, *a, **k: _EMPTY), \
         patch.object(Scheduler, "_mamba_block_aligned_split", _align_to_blocks), \
         patch.object(PrefixAwareScheduler, "_mamba_block_aligned_split", _align_to_blocks):
        for _ in range(steps):
            s.schedule()
            total = 0
            for r in list(s.waiting):
                left = r.num_prompt_tokens - r.num_computed_tokens
                if left <= 0 or total >= 8192:
                    continue
                n = s._mamba_block_aligned_split(r, min(left, 8192 - total))
                r.num_computed_tokens += n
                total += n
            if total:
                mix[next(b for b in BUCKETS if b >= total)] += 1
                if hasattr(s, "_enter") and s._cost is not None:
                    s._enter(s._cost.step_ms(next(b for b in BUCKETS if b >= total), 0.0, float(total) ** 2))
    return mix


def test_synthetic_queue_chunk_sizing_matches_in_a_fixed_order_loop():
    # Measured finding (recorded in reports/unified_port.md): in a fixed-order loop the two
    # flavours size chunks identically, also with unified's quad / in-flight costing on, so the
    # band-D step-mix shift (more 4352-token steps, fewer 8192) is not chunk sizing. It must come
    # from admission order (PrefixAwareScheduler / AsyncScheduler base vs vLLM's Scheduler),
    # which this loop does not model; the final flavour carries that base class.
    final = _simulate(lambda m, **kw: _final(m, **kw), steps=80)
    for quad, inflight in ((True, True), (False, False)):
        assert _simulate(lambda m, **kw: _unified(m, quad=quad, inflight=inflight, **kw), steps=80) == final
    assert sum(final.values()) > 0 and len(final) >= 3
