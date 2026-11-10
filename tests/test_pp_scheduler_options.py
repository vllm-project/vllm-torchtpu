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
"""The ``tpu_pp_scheduler`` options of the pipeline chunk scheduler.

The platform installs the scheduler on every engine with more than one
pipeline stage that names no other scheduler class, so with no option set it
must schedule exactly as main's scheduler does, under each setting of main's
lone-prompt and equal-cost chunking; each option turns on one lever.
"""

import itertools
import math
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import additional_config_sections
from vllm_torchtpu.core import pp_chunk_scheduler, pp_scheduler_options
from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler
from vllm_torchtpu.core.pp_chunks import StepCostModel, step_limit
from vllm_torchtpu.runner.utils import (
    HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
    PP_SCHEDULER_OPTIONS_SECTION,
)

BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)
_IX_PREFILL = {
    "quadratic_cost": True,
    "prefix_aware": True,
    "paw_aging_tok_per_s": 3000,
}


def _model(attn=1.5e-7, slack=0.1, granularity=4096) -> StepCostModel:
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack), granularity)


def _req(computed, prompt, rid="r", num_tokens=None):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt if num_tokens is None else num_tokens,
        request_id=rid,
    )


def _sched(cost=None, running=(), waiting=(), **attrs) -> TpuPipelineChunkScheduler:
    """A scheduler built without __init__, after the engine hook installed
    ``cost`` (or, with ``_granularity`` and no cost, the chunk limits alone);
    attributes default as on an engine that sets no option, with main's
    lone-prompt and equal-cost chunking on."""
    s = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    s._lone_prompt = True
    s._equal_cost = True
    s._cost = cost
    s._granularity = cost.granularity if cost is not None else None
    s._schedule = cost.schedule if cost is not None else None
    s._step_limit = None
    s._pp_size = 8
    s.waiting = list(waiting)
    s.kv_holding_waiting = []
    s.scheduler_config = SimpleNamespace(max_num_batched_tokens=16384)
    s.vllm_config = SimpleNamespace(
        scheduler_config=s.scheduler_config,
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
    s._clips = 0
    s._waits = 0
    s._retention = None
    s.kv_cache_manager = SimpleNamespace()
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


# -- parsing -----------------------------------------------------------------


def test_defaults_are_the_schedulers_behaviour_without_options():
    opts = pp_scheduler_options.parse(None)
    assert opts == pp_scheduler_options.PPSchedulerOptions()
    assert pp_scheduler_options.parse({}) == opts
    assert (
        opts.quadratic_cost,
        opts.prefix_aware,
        opts.paw_aging_tok_per_s,
    ) == (False, False, 0.0)
    assert pp_scheduler_options.get(SimpleNamespace(additional_config=None)) == opts
    assert pp_scheduler_options.get(SimpleNamespace()) == opts


def test_parse_reads_every_option():
    opts = pp_scheduler_options.get(
        SimpleNamespace(additional_config={"tpu_pp_scheduler": _IX_PREFILL})
    )
    assert opts.quadratic_cost and opts.prefix_aware
    # A JSON integer is taken for the float option.
    assert opts.paw_aging_tok_per_s == 3000.0
    assert type(opts.paw_aging_tok_per_s) is float


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"inflight": True}, "unknown keys"),
        # Main's pipeline hand-off sets the steps in flight, and the in-flight
        # costing modelled the lockstep hand-off main removed: neither is an
        # option.
        ({"inflight_cost": True}, "unknown keys"),
        ({"extra_steps_in_flight": 1}, "unknown keys"),
        ({"quadratic_cost": 1}, "must be a bool"),
        ({"prefix_aware": "true"}, "must be a bool"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": True}, "must be a float"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": "3000"}, "must be a float"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": -1.0}, ">= 0"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": math.inf}, "finite"),
        ({"paw_aging_tok_per_s": 3000}, "set prefix_aware"),
        ([], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, match):
    with pytest.raises(ValueError, match=match):
        pp_scheduler_options.parse(raw)


def test_section_is_registered_and_kept_out_of_the_compile_cache_key():
    assert pp_scheduler_options.SECTION == PP_SCHEDULER_OPTIONS_SECTION
    assert pp_scheduler_options.SECTION in additional_config_sections.KNOWN_SECTIONS
    assert pp_scheduler_options.SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


# -- __init__ wiring -----------------------------------------------------------


def _built(
    additional=None, kv_role="kv_producer", retention_enabled=True, equal_cost=True
):
    """Runs TpuPipelineChunkScheduler.__init__ over a stub base __init__, with
    main's equal-cost chunking ``equal_cost``; returns the scheduler and the
    order of the prefix-aware setup calls."""
    calls: list[str] = []

    def base_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            additional_config=additional,
            kv_transfer_config=(
                None if kv_role is None else SimpleNamespace(kv_role=kv_role)
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=8),
        )
        self.need_mamba_block_aligned_split = False
        self.kv_cache_manager = SimpleNamespace()

    class FakeRetention:
        def __init__(self, kvm):
            calls.append("retention")
            self.enabled = retention_enabled

    def memo(self):
        calls.append("memo")
        self._memo_xstep = True

    with (
        patch.object(Scheduler, "__init__", base_init),
        patch.object(pp_chunk_scheduler.envs, "TPU_PP_EQUAL_COST_CHUNKS", equal_cost),
        patch.object(pp_chunk_scheduler, "HybridStateRetention", FakeRetention),
        patch.object(
            pp_chunk_scheduler.prefix_change_log,
            "install_epoch",
            lambda: calls.append("epoch"),
        ),
        patch.object(TpuPipelineChunkScheduler, "_enable_memo_xstep", memo),
    ):
        s = TpuPipelineChunkScheduler()
    return s, calls


def test_no_option_turns_on_no_lever():
    s, calls = _built()
    assert not s._quadratic_cost
    assert not s._prefix_aware
    assert not s._srpf_running
    assert not s._paw_sort
    assert not s._promote_waiting
    assert not s._memo_xstep
    assert s._paw_aging_tok_per_s == 0.0
    assert "epoch" not in calls and "memo" not in calls


def test_prefix_aware_installs_the_epoch_before_the_memo():
    s, calls = _built({"tpu_pp_scheduler": _IX_PREFILL})
    assert calls.index("epoch") < calls.index("memo")
    assert s._memo_xstep
    assert s._prefix_aware and s._srpf_running
    assert s._quadratic_cost
    assert s._paw_aging_tok_per_s == 3000.0
    assert s._get_paw_aging_rate() == 3000.0
    # The waiting-queue sort runs on a KV producer.
    assert s._paw_sort


def test_quadratic_cost_needs_the_equal_cost_model():
    # Without equal-cost chunking the stages are not timed and no cost model
    # exists to price the attention pairs, so the option is refused.
    with pytest.raises(ValueError, match="TPU_PP_EQUAL_COST_CHUNKS"):
        _built({"tpu_pp_scheduler": {"quadratic_cost": True}}, equal_cost=False)
    s, _ = _built({"tpu_pp_scheduler": {"quadratic_cost": True}})
    assert s._quadratic_cost
    # The prefix-aware admission needs no cost model.
    s, _ = _built({"tpu_pp_scheduler": {"prefix_aware": True}}, equal_cost=False)
    assert s._prefix_aware and not s._quadratic_cost


def test_promotion_needs_prefix_aware_and_follows_promote_warmest():
    s, _ = _built({"tpu_scheduler": {"promote_warmest": True}})
    assert not s._promote_waiting
    s, _ = _built({"tpu_pp_scheduler": {"prefix_aware": True}})
    assert s._promote_waiting
    s, _ = _built(
        {
            "tpu_pp_scheduler": {"prefix_aware": True},
            "tpu_scheduler": {"promote_warmest": False},
        }
    )
    assert not s._promote_waiting


def test_retention_follows_the_kv_producer_rule():
    s, calls = _built(kv_role="kv_producer")
    assert s._retention is not None and "retention" in calls
    s, calls = _built(kv_role="kv_consumer")
    assert s._retention is None and "retention" not in calls
    s, calls = _built(kv_role=None)
    assert s._retention is None
    s, _ = _built(kv_role="kv_producer", retention_enabled=False)
    assert s._retention is None


# -- default behaviour: the scheduler as it was without options ----------------


def _reference_step_limit(cost, pp_size, running, waiting, granularity=None):
    """Main's lone-prompt spread, as the scheduler had it before the options
    existed: waiting requests by their computed tokens and no (chunk, prefix)
    pairs; without a cost model, main's ``step_limit``."""

    def end(r):
        return max(r.num_prompt_tokens, r.num_tokens - 1)

    pending = sum(max(end(r) - r.num_computed_tokens, 0) for r in running)
    pending += sum(max(end(r) - r.num_computed_tokens, 0) for r in waiting)
    if cost is None:
        return step_limit(pending, pp_size, granularity, 16384)
    if pending == 0:
        return None
    limit, soonest = cost.buckets[-1], math.inf
    for spread in range(1, pp_size + 1):
        share = -(-pending // spread)
        share = -(-share // cost.granularity) * cost.granularity
        share = min(cost.buckets[-1], max(cost.granularity, share))
        bucket = next(b for b in cost.buckets if b >= share)
        steps = -(-pending // share) + pp_size - 1
        done_ms = steps * cost.step_ms(bucket, 0.0, float(share) * share)
        if done_ms < soonest:
            limit, soonest = share, done_ms
    return limit


def _scenarios(seed=0, n=200):
    rng = random.Random(seed)
    for i in range(n):
        running = []
        for j in range(rng.randint(0, 3)):
            prompt = rng.choice([4096, 8192, 65536, 163840 + 3800])
            computed = rng.choice([0, prompt // 2, prompt])
            running.append(_req(computed, prompt, f"run{i}.{j}"))
        waiting = [
            _req(0, rng.choice([2048, 8192, 32768]), f"w{i}.{j}")
            for j in range(rng.randint(0, 3))
        ]
        yield running, waiting, rng.choice([4, 8])


def test_without_options_the_step_limit_is_mains():
    model = _model(granularity=2048)
    kvm = SimpleNamespace(enable_caching=True)
    differs = 0
    for running, waiting, pp in _scenarios():
        s = _sched(cost=model, running=running, waiting=waiting)
        s._pp_size = pp
        want = _reference_step_limit(model, pp, running, waiting)
        assert s._pending_step_limit() == want
        q = _sched(
            cost=model,
            running=running,
            waiting=waiting,
            kv_cache_manager=kvm,
            _quadratic_cost=True,
        )
        q._prefix_warmth = lambda request: 0
        q._external_warmth = lambda request, local: 0
        q._pp_size = pp
        differs += q._pending_step_limit() != want
        # With lone-prompt chunking off no step limit is set, option or not.
        for sched in (s, q):
            sched._lone_prompt = False
            assert sched._pending_step_limit() is None
        # Lone-prompt chunking without the cost model (equal-cost off) is
        # main's step_limit, which no option changes.
        for attrs in ({}, {"_quadratic_cost": True}):
            n = _sched(running=running, waiting=waiting, _granularity=2048, **attrs)
            n._pp_size = pp
            assert n._pending_step_limit() == _reference_step_limit(
                None, pp, running, waiting, 2048
            )
    # quadratic_cost does change the spread somewhere in these scenarios.
    assert differs > 0


def test_without_options_schedule_runs_none_of_the_prefix_aware_steps():
    polls, packs = [], []
    connector = SimpleNamespace(
        poll_and_complete_transfers=lambda sched: polls.append(sched),
        supports_divergent_local_hybrid_hits=False,
    )
    deep = _req(163840, 163840 + 3800, "deep")
    shallow = _req(0, 4096, "shallow")
    step = SimpleNamespace(total_num_scheduled_tokens=0, scheduled_new_reqs=[1])
    for prefix_aware in (False, True):
        s = _sched(
            cost=_model(),
            running=[deep, shallow],
            connector=connector,
            _prefix_aware=prefix_aware,
            _srpf_running=prefix_aware,
            _quadratic_cost=True,
            _paw_aging_tok_per_s=0.0,
        )
        s._maybe_pack_prompt_ids = lambda out, on=prefix_aware: packs.append(on)
        s._select_waiting_queue_for_scheduling = lambda: pytest.fail("promoted")
        s._sort_waiting_queue = lambda: pytest.fail("sorted")
        with patch.object(Scheduler, "schedule", return_value=step):
            s.schedule()
        order = [r.request_id for r in s.running]
        if prefix_aware:
            assert order == ["shallow", "deep"]
        else:
            assert order == ["deep", "shallow"]
    assert len(polls) == 1 and packs == [True]


def test_promotion_and_sort_run_when_their_flags_are_set():
    calls = []
    for on in (True, False):
        s = _sched(_promote_waiting=on, _paw_sort=on, _prefix_aware=on)
        s._select_waiting_queue_for_scheduling = lambda on=on: calls.append(
            ("promote", on)
        )
        s._sort_waiting_queue = lambda on=on: calls.append(("sort", on))
        with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
            s.schedule()
    assert calls == [("sort", True), ("promote", True)]


def test_without_prefix_aware_the_local_hit_is_the_base_schedulers():
    seen = []

    def base(self, request, max_cache_hit_length=None):
        seen.append(max_cache_hit_length)
        return "blocks", 4096, 0, True

    resolve = []
    connector = SimpleNamespace(
        supports_divergent_local_hybrid_hits=True,
        resolve_divergent_local_hit_tokens=lambda *a: resolve.append(a) or 0,
    )
    request = _req(0, 8192)
    with patch.object(Scheduler, "_get_local_prefix_cache_hit", base):
        s = _sched(connector=connector)
        # vLLM passes the cap positionally.
        assert s._get_local_prefix_cache_hit(request, 2048) == (
            "blocks",
            4096,
            0,
            True,
        )
        assert seen == [2048] and resolve == []
        assert not hasattr(request, "_tpu_local_hit_tokens")


def test_prefix_aware_resolves_a_divergent_hit_through_the_connector():
    seen = []

    def base(self, request, max_cache_hit_length=None):
        seen.append(max_cache_hit_length)
        return "blocks", 8192, 0, True

    kvm = SimpleNamespace(
        get_computed_blocks=lambda request, cap: ("reconciled", 2048, 0),
        truncate_computed_blocks=lambda blocks, n: f"{blocks}[:{n}]",
        empty_kv_cache_blocks="empty",
    )
    connector = SimpleNamespace(
        supports_divergent_local_hybrid_hits=True,
        resolve_divergent_local_hit_tokens=lambda req, n, rec, bs: 4096,
    )
    request = _req(0, 16384)
    with patch.object(Scheduler, "_get_local_prefix_cache_hit", base):
        s = _sched(connector=connector, _prefix_aware=True, kv_cache_manager=kvm)
        s._get_block_size = lambda: 2048
        got = s._get_local_prefix_cache_hit(request, 12288)
    assert got == ("blocks[:4096]", 4096, 0, True)
    assert request._tpu_local_hit_tokens == 4096
    assert seen == [12288]


def test_without_prefix_aware_the_resend_is_not_trimmed():
    trimmed = []
    cached = SimpleNamespace(all_token_ids={"r": [1, 2, 3]})
    with (
        patch.object(Scheduler, "_make_cached_request_data", return_value=cached),
        patch.object(
            pp_chunk_scheduler, "trim_resend_token_ids", lambda c: trimmed.append(c)
        ),
    ):
        assert _sched()._make_cached_request_data() is cached
        assert trimmed == []
        _sched(_prefix_aware=True)._make_cached_request_data()
        assert trimmed == [cached]


def test_without_quadratic_cost_the_cost_is_the_token_count():
    model = _model()
    plain = _sched(cost=model)
    quad = _sched(cost=model, _quadratic_cost=True)
    for tokens, prefix in itertools.product((0, 4096), (0, 131072)):
        assert plain._quad_cost(tokens, prefix) == float(tokens)
        assert quad._quad_cost(tokens, prefix) == model.effective_tokens(tokens, prefix)
    assert quad._quad_cost(4096, 131072) > 4096


def test_a_waiting_prefix_hit_counts_as_done_only_under_quadratic_cost():
    kvm = SimpleNamespace(enable_caching=True)
    plain = _sched(cost=_model(), kv_cache_manager=kvm)
    quad = _sched(cost=_model(), kv_cache_manager=kvm, _quadratic_cost=True)
    # Without the cost model (main's lone-prompt chunking alone) the pending
    # work is main's: computed tokens only, whatever the option says.
    bare = _sched(kv_cache_manager=kvm, _granularity=2048, _quadratic_cost=True)
    for s in (plain, quad, bare):
        s._prefix_warmth = lambda request: 4096
        s._external_warmth = lambda request, local: 0
    waiting = _req(0, 8192)
    assert plain._waiting_warmth(waiting) == 0
    assert quad._waiting_warmth(waiting) == 4096
    assert bare._waiting_warmth(waiting) == 0
    bare.waiting = [waiting]
    assert bare._pending_step_limit() == step_limit(8192, 8, 2048, 16384)
    # A started request counts its computed tokens either way.
    assert plain._waiting_warmth(_req(2048, 8192)) == 2048
    assert quad._waiting_warmth(_req(2048, 8192)) == 2048
