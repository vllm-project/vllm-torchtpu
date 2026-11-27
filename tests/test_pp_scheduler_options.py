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
import json
import math
import pathlib
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import additional_config_sections, config_section
from vllm_torchtpu.core import (
    pp_chunk_scheduler,
    pp_scheduler_options,
    scheduler_options,
)
from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler
from vllm_torchtpu.core.pp_chunks import StepCostModel, step_limit
from vllm_torchtpu.models import qwen3_5
from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)
_ENTRY_PREFILL = {
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
    s._hit_estimator = None
    s._paw_aging_tok_per_s = 0.0
    s._step_wall_time = None
    s._transfer_poller = None
    s._hit_resolver = None
    s.kv_cache_manager = SimpleNamespace(enable_caching=False)
    s.connector = None
    s._is_kv_consumer = False
    s._quadratic_cost = False
    s._prefix_aware = False
    s._srpf_running = False
    s._paw_sort = False
    s._promote_waiting = False
    s._memo_xstep = False
    s._trim_resend = True
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
    ) == (False, False, 3000)
    assert pp_scheduler_options.get(SimpleNamespace(additional_config=None)) == opts
    # VllmConfig always has additional_config: a config without it is an error.
    with pytest.raises(AttributeError, match="additional_config"):
        pp_scheduler_options.get(SimpleNamespace())


def test_parse_reads_every_option():
    opts = pp_scheduler_options.get(
        SimpleNamespace(additional_config={"tpu_pp_scheduler": _ENTRY_PREFILL})
    )
    assert opts.quadratic_cost and opts.prefix_aware
    assert opts.paw_aging_tok_per_s == 3000
    assert type(opts.paw_aging_tok_per_s) is int


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
        ({"prefix_aware": True, "paw_aging_tok_per_s": True}, "must be a int"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": "3000"}, "must be a int"),
        # A whole number of tokens per second: no fraction, infinity or NaN.
        ({"prefix_aware": True, "paw_aging_tok_per_s": 3000.0}, "must be a int"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": math.inf}, "must be a int"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": math.nan}, "must be a int"),
        ({"prefix_aware": True, "paw_aging_tok_per_s": -1}, ">= 0"),
        ({"paw_aging_tok_per_s": 3000}, "set prefix_aware"),
        ([], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, match):
    with pytest.raises(ValueError, match=match):
        pp_scheduler_options.parse(raw)


def test_get_applies_the_section_checks():
    """``get`` parses through ``parse``, so the aging checks hold at start-up
    too, not only the shared parser's type checks."""
    cfg = SimpleNamespace(
        additional_config={"tpu_pp_scheduler": {"paw_aging_tok_per_s": 5}}
    )
    with pytest.raises(ValueError, match="set prefix_aware"):
        pp_scheduler_options.get(cfg)


@pytest.mark.parametrize("module", [pp_scheduler_options, scheduler_options])
def test_the_sections_parse_with_the_shared_parser(module):
    """Both scheduler sections go through ``config_section.parse``, the one
    strict section parser, instead of a copy of it."""
    calls = []

    def shared(options_cls, section, raw):
        calls.append((options_cls, section, raw))
        return options_cls()

    with patch.object(config_section, "parse", shared):
        module.parse({"x": 1})
    assert calls == [(calls[0][0], module.SECTION, {"x": 1})]


def test_section_is_registered_and_kept_out_of_the_compile_cache_key():
    assert pp_scheduler_options.SECTION in additional_config_sections.KNOWN_SECTIONS
    assert pp_scheduler_options.SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


# -- __init__ wiring -----------------------------------------------------------


def _kv(role):
    """A KV transfer config stand-in with vLLM's role properties (kv_both is
    both a producer and a consumer)."""
    return SimpleNamespace(
        kv_role=role,
        is_kv_producer=role in ("kv_producer", "kv_both"),
        is_kv_consumer=role in ("kv_consumer", "kv_both"),
    )


def _built(
    additional=None,
    kv_role="kv_producer",
    retention_enabled=True,
    architecture="Qwen3_5MoeForConditionalGeneration",
    equal_cost=True,
    connector=None,
):
    """Runs TpuPipelineChunkScheduler.__init__ over a stub base __init__, with
    main's equal-cost chunking ``equal_cost``; returns the scheduler and the
    order of the prefix-aware setup calls."""
    calls: list[str] = []

    def base_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            additional_config=additional,
            kv_transfer_config=None if kv_role is None else _kv(kv_role),
            parallel_config=SimpleNamespace(pipeline_parallel_size=8),
            model_config=SimpleNamespace(architectures=[architecture]),
        )
        self.need_mamba_block_aligned_split = False
        self.kv_cache_manager = SimpleNamespace()
        self.connector = connector

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
    assert "memo" not in calls


def test_prefix_aware_opts_the_pool_into_the_memo():
    # The epoch and change-log wrappers are registry patches; the scheduler
    # only opts its pool in, through _enable_memo_xstep.
    s, calls = _built({"tpu_pp_scheduler": _ENTRY_PREFILL})
    assert "memo" in calls
    assert s._memo_xstep
    assert s._prefix_aware and s._srpf_running
    assert s._quadratic_cost
    assert s._paw_aging_tok_per_s == 3000.0
    assert s._get_paw_aging_rate() == 3000.0
    # The waiting-queue sort runs on a KV producer.
    assert s._paw_sort


def test_prefix_aware_ages_its_orders_by_default():
    """With prefix_aware on and no aging rate set, the orders age at 3000
    tokens per second, so a large prompt is not overtaken without bound; 0
    turns the aging off, and without prefix_aware nothing ages."""
    s, _ = _built({"tpu_pp_scheduler": {"prefix_aware": True}})
    assert s._get_paw_aging_rate() == 3000.0
    s, _ = _built(
        {"tpu_pp_scheduler": {"prefix_aware": True, "paw_aging_tok_per_s": 0}}
    )
    assert s._get_paw_aging_rate() == 0.0
    s, _ = _built({"tpu_pp_scheduler": {"quadratic_cost": True}})
    assert s._get_paw_aging_rate() == 0.0
    # An explicit 0 without prefix_aware is accepted, as before, and so is a
    # missing section or a section without the rate.
    assert pp_scheduler_options.parse({"paw_aging_tok_per_s": 0}).prefix_aware is False
    for raw in (None, {}, {"quadratic_cost": True}):
        assert pp_scheduler_options.parse(raw).paw_aging_tok_per_s == 3000


def test_quadratic_cost_opts_the_pool_into_the_memo_without_the_admission():
    # Under quadratic_cost the pending work counts every waiting request's
    # prefix hit on every step, so the memo is on; the prefix-aware admission
    # stays off.
    s, calls = _built({"tpu_pp_scheduler": {"quadratic_cost": True}})
    assert "memo" in calls and s._memo_xstep
    assert not s._prefix_aware and not s._paw_sort and not s._promote_waiting


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


@pytest.mark.parametrize(
    "kv_role,consumer",
    [
        ("kv_producer", False),
        ("kv_consumer", True),
        ("kv_both", False),
        (None, False),
    ],
)
def test_kv_both_is_not_a_kv_consumer(kv_role, consumer):
    """kv_both (an offloading connector on an aggregated engine) has neither
    role here: vLLM's role properties count it as a producer too, so it is
    not a consumer (as in PrefixAwareScheduler), and only kv_producer is a
    producer, so it gets neither the producer's sort nor retention."""
    s, _ = _built(kv_role=kv_role)
    assert s._is_kv_consumer is consumer
    assert s._is_kv_producer is (kv_role == "kv_producer")


class _HookedConnector:
    """A connector with the two hooks the prefix-aware admission calls, as
    the multi-connector and the Raiden offloading connector define them."""

    def poll_and_complete_transfers(self, scheduler):
        return 0, 0, 0, False

    def resolve_divergent_local_hit_tokens(
        self, request, num_local, num_reconciled, block_size
    ):
        return None


def test_the_connector_hooks_are_read_through_named_interfaces(caplog):
    connector = _HookedConnector()
    with caplog.at_level("WARNING"):
        s, _ = _built({"tpu_pp_scheduler": {"prefix_aware": True}}, connector=connector)
    assert s._transfer_poller is connector and s._hit_resolver is connector
    assert "has no" not in caplog.text


def test_a_connector_without_the_hooks_is_named_in_a_warning(caplog):
    """The prefix-aware admission runs without the poll and the divergent-hit
    decision when the connector has neither, and says so instead of skipping
    them silently."""
    with caplog.at_level("WARNING"):
        s, _ = _built({"tpu_pp_scheduler": {"prefix_aware": True}}, connector=object())
    assert s._transfer_poller is None and s._hit_resolver is None
    assert "has no poll_and_complete_transfers or" in caplog.text
    caplog.clear()
    # Without the prefix-aware admission nothing calls them, so nothing is said.
    with caplog.at_level("WARNING"):
        _built(connector=object())
    assert "has no" not in caplog.text


def test_retention_follows_the_kv_producer_rule():
    s, calls = _built(kv_role="kv_producer")
    assert s._retention is not None and "retention" in calls
    s, calls = _built(kv_role="kv_consumer")
    assert s._retention is None and "retention" not in calls
    s, calls = _built(kv_role=None)
    assert s._retention is None
    s, _ = _built(kv_role="kv_producer", retention_enabled=False)
    assert s._retention is None


@pytest.mark.parametrize(
    "architecture, retained",
    [
        ("Qwen3_5MoeForConditionalGeneration", True),
        ("Qwen3_5MoeForCausalLM", True),
        # Other hybrid (GDN or Mamba + attention) models keep main's scheduler.
        ("Qwen3NextForCausalLM", False),
        ("Qwen3_5ForConditionalGeneration", False),
        ("JambaForCausalLM", False),
    ],
)
def test_retention_is_scoped_to_the_qwen3_5_moe_architectures(architecture, retained):
    s, calls = _built(kv_role="kv_producer", architecture=architecture)
    assert (s._retention is not None) is retained
    assert ("retention" in calls) is retained


# -- default behaviour: every option off -----------------------------------------


def _reference_step_limit(cost, pp_size, running, waiting, granularity=None):
    """The lone-prompt step limit with every option off: the pending tokens of
    the running and waiting requests, each counted from its computed tokens and
    with no (chunk, prefix) pairs, cut into the chunk share whose spread
    finishes them soonest on the cost model; without a cost model, main's
    ``step_limit``."""

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
            _transfer_poller=connector,
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


def test_a_failing_sort_turns_the_sort_off_and_the_step_still_schedules():
    s = _sched(_paw_sort=True, _prefix_aware=True)

    def failing_sort():
        raise RuntimeError("simulated sort failure")

    s._sort_waiting_queue = failing_sort
    with (
        patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP) as base,
        patch.object(pp_chunk_scheduler.logger, "exception") as log,
    ):
        s.schedule()
        s.schedule()
    assert s._paw_sort is False
    assert base.call_count == 2
    log.assert_called_once()
    assert "waiting-queue sort failed" in log.call_args.args[0]


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
        s = _sched(connector=connector, _hit_resolver=connector)
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
        s = _sched(
            connector=connector,
            _hit_resolver=connector,
            _prefix_aware=True,
            kv_cache_manager=kvm,
        )
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


# -- the validation entries ----------------------------------------------------


def _point_entries():
    folder = pathlib.Path(qwen3_5.__file__).parent / "validation"
    for path in sorted(folder.glob("p[0-9][0-9]-*.yaml")):
        with path.open() as f:
            yield path.stem, yaml.safe_load(f)


def test_the_prefill_legs_run_what_the_record_jobs_ran():
    seen = []
    for name, entry in _point_entries():
        cfg = SimpleNamespace(
            additional_config=entry["legs"]["prefill"]["additional_config"]
        )
        opts = pp_scheduler_options.get(cfg)
        assert opts.quadratic_cost and opts.prefix_aware, name
        assert opts.paw_aging_tok_per_s == 3000.0, name
        # Warmest-first promotion ran at #1 only.
        point_one = entry["point"] == 1
        assert scheduler_options.get(cfg).promote_warmest is point_one, name
        # The leg gets this scheduler (no other scheduler class) with main's
        # equal-cost chunking, which quadratic_cost needs.
        prefill = entry["legs"]["prefill"]
        assert "--scheduler-cls" not in prefill["serve_args"], name
        for flag in ("TPU_PP_LONE_PROMPT_CHUNKS", "TPU_PP_EQUAL_COST_CHUNKS"):
            assert (prefill.get("env") or {}).get(flag, "1") not in ("0", "false"), name
        # The decode legs run PrefixAwareScheduler, not this scheduler.
        assert "tpu_pp_scheduler" not in entry["legs"]["decode"]["additional_config"]
        seen.append(entry["point"])
    assert seen == [1, 5, 8, 10, 12]


def test_the_prefill_legs_are_kv_producers_not_consumers():
    """Every prefill leg is a kv_producer: a producer, and by vLLM's role
    properties not a consumer, so its sort keeps the producer tiers."""
    for name, entry in _point_entries():
        args = entry["legs"]["prefill"]["serve_args"]
        kv = json.loads(args[args.index("--kv-transfer-config") + 1])
        assert kv["kv_role"] == "kv_producer", name
        s, _ = _built(kv_role=kv["kv_role"])
        assert not s._is_kv_consumer and s._is_kv_producer, name


# -- the prefix-aware admission's promotion, free-block count and estimate -----


def _promote(head, other, warmth, free_blocks, step_time, aging=3_000.0):
    """Runs the warmest-first promotion over ``[head, other]`` at
    ``step_time`` and returns the request id it leaves first."""
    queue = FCFSRequestQueue()
    queue.add_request(head)
    queue.add_request(other)
    s = _sched(_paw_aging_tok_per_s=aging, _step_wall_time=step_time)
    s.policy = SchedulingPolicy.FCFS
    s._paw_n = s._paw_cold_tokens = s._paw_prompt_tokens = 0
    s._prefix_warmth = lambda request: warmth[request.request_id]
    s._get_block_size = lambda: 2_176
    s._get_num_free_blocks = lambda: free_blocks
    s._promote_warmest(queue)
    return next(iter(queue)).request_id


def _waiting(request_id: str, arrival_time: float) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=10_000,
        num_preemptions=0,
        arrival_time=arrival_time,
    )


def test_promotion_ranks_with_the_free_block_fit():
    # The head (old, so aged ahead) does not fit the free blocks; a warm
    # request that fits takes the free-block-fit tier and is promoted. Without
    # the fit input the head's aging keeps it first.
    head, fits = _waiting("head", 990.0), _waiting("fits", 1_000.0)
    warmth = {"head": 2_000, "fits": 5_000}
    assert _promote(head, fits, warmth, free_blocks=3, step_time=1_000.0) == "fits"
    # With room for the head there is no fit input: aging keeps the head first.
    assert _promote(head, fits, warmth, free_blocks=10_000, step_time=1_000.0) == "head"


def test_promotion_ages_the_head_and_the_window_to_the_step_time():
    # Aging is linear in the clock, so what matters is that the head and the
    # rest of the window are aged to the same one, the step time. The step
    # time here is far in the past of the real clock: a request aged to
    # time.time() instead would win on aging alone. At the step time, one
    # second of aging (3,000 tokens) is less than the 4,000-token cold-prefix
    # difference, so the warmer request ranks first.
    old_cold, new_warm = _waiting("old-cold", 999.0), _waiting("new-warm", 1_000.0)
    warmth = {"old-cold": 2_000, "new-warm": 6_000}
    # The head is the older, colder request: the warmer one is promoted
    # (the head must be aged to the step time).
    assert _promote(old_cold, new_warm, warmth, 10_000, step_time=1_000.0) == "new-warm"
    # The head is the warmer one: it stays first (the window's other
    # requests must be aged to the step time too).
    assert _promote(new_warm, old_cold, warmth, 10_000, step_time=1_000.0) == "new-warm"


def test_a_ranking_without_a_time_ages_to_the_step_time():
    """A ranking called during schedule() without a time ages the request to
    the step's wall time, as the promotion does, not to the real clock."""
    s = _sched(_paw_aging_tok_per_s=3_000.0, _step_wall_time=1_000.0)
    s._prefix_warmth = lambda request: 2_000
    s._get_block_size = lambda: 2_176
    request = _waiting("r", 990.0)
    assert s._rank_and_warmth(request) == s._rank_and_warmth(request, now=1_000.0)
    s._step_wall_time = None
    with patch.object(pp_chunk_scheduler.time, "time", return_value=1_005.0):
        assert s._rank_and_warmth(request) == s._rank_and_warmth(request, now=1_005.0)


def test_free_blocks_count_the_blocks_retention_pins():
    pool = SimpleNamespace(get_num_free_blocks=lambda: 10)
    s = _sched(kv_cache_manager=SimpleNamespace(block_pool=pool))
    assert s._get_num_free_blocks() == 10
    s._retention = SimpleNamespace(num_pinned_blocks=lambda: 5)
    assert s._get_num_free_blocks() == 15


def test_the_connector_estimate_is_read_through_the_named_interface():
    class Estimator:
        def estimate_external_hit_tokens(self, request, num_local_tokens):
            return 0

    connector = Estimator()
    s, _ = _built(connector=connector)
    assert s._hit_estimator is connector
    # A connector without the method (main's connectors) gives no estimate.
    s, _ = _built(connector=SimpleNamespace())
    assert s._hit_estimator is None
