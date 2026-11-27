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
"""``tpu_scheduler.prefix_aware_levers``: PrefixAwareScheduler's decode-leg
levers on an engine that is not a KV consumer."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import KVTransferConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core import prefix_aware_scheduler, scheduler_options
from vllm_torchtpu.core.packed_ids import PackedPromptIds
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

# The engines that are not KV consumers: no KV transfer, a KV producer, and
# kv_both (an offloading connector on an aggregated engine).
OTHER_ROLES = [None, "kv_producer", "kv_both"]

# ---------------- parsing ----------------


def test_the_option_defaults_off():
    assert scheduler_options.parse({}).prefix_aware_levers is False
    assert scheduler_options.parse(None).prefix_aware_levers is False


def test_parse_reads_the_option():
    opts = scheduler_options.get(
        SimpleNamespace(
            additional_config={"tpu_scheduler": {"prefix_aware_levers": True}}
        )
    )
    assert opts.prefix_aware_levers is True
    # The other options keep their defaults.
    assert opts.promote_warmest is True
    assert opts.kv_prefetch is False


@pytest.mark.parametrize("value", [1, "true", None])
def test_the_option_takes_a_bool_only(value):
    with pytest.raises(ValueError, match="prefix_aware_levers must be a bool"):
        scheduler_options.parse({"prefix_aware_levers": value})


# ---------------- the scheduler ----------------


class _Resolver:
    """A KV connector that claims divergent hybrid hits and resolves them to
    ``target`` local tokens."""

    supports_divergent_local_hybrid_hits = True

    def __init__(self, target=None):
        self.target = target
        self.calls: list[tuple] = []

    def resolve_divergent_local_hit_tokens(
        self, request, num_local, num_reconciled, block_size
    ):
        self.calls.append((num_local, num_reconciled, block_size))
        return self.target


def _built(monkeypatch, role, section=None, connector="default", calls=None):
    """Runs PrefixAwareScheduler.__init__ over a stub base __init__ for an
    engine with KV role ``role`` and ``section`` as its tpu_scheduler
    options; ``calls`` records the memo set-up."""
    kv_transfer_config = (
        None
        if role is None
        else KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=role)
    )
    if connector == "default":
        connector = None if role is None else object()

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            additional_config=None if section is None else {"tpu_scheduler": section},
            scheduler_config=SimpleNamespace(max_num_seqs=8),
        )
        self.connector = connector
        self.kv_cache_manager = MagicMock()
        # Enough for decode_inflight_throttle, so only the role refuses it.
        self.dcp_world_size = 4

    calls = [] if calls is None else calls
    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    monkeypatch.setattr(
        PrefixAwareScheduler, "_enable_memo_xstep", lambda self: calls.append("memo")
    )
    return PrefixAwareScheduler()


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("role", OTHER_ROLES)
def test_the_option_turns_the_levers_on_off_a_kv_consumer(monkeypatch, role, on):
    calls = []
    s = _built(monkeypatch, role, {"prefix_aware_levers": on}, calls=calls)
    assert s._is_kv_consumer is False
    assert s._levers_on() is on
    assert calls == (["memo"] if on else [])


@pytest.mark.parametrize("on", [False, True])
def test_on_a_kv_consumer_the_option_changes_nothing(monkeypatch, caplog, on):
    calls = []
    connector = _Resolver()
    with caplog.at_level("INFO"):
        s = _built(
            monkeypatch,
            "kv_consumer",
            {"prefix_aware_levers": on},
            connector=connector,
            calls=calls,
        )
    # The levers run either way; divergent hits keep vLLM's lookup.
    assert s._levers_on() and calls == ["memo"]
    assert s._levers_switch is False and s._hit_resolver is None
    assert ("adds nothing on a KV consumer" in caplog.text) is on


@pytest.mark.parametrize("option", ["kv_prefetch", "decode_inflight_throttle"])
@pytest.mark.parametrize("role", OTHER_ROLES)
def test_the_levers_leave_the_kv_consumer_refusals(monkeypatch, role, option):
    """The levers do not make an engine a KV consumer: the options that
    start or throttle remote loads stay refused."""
    with pytest.raises(ValueError, match=f"{option} .*needs a KV consumer"):
        _built(monkeypatch, role, {"prefix_aware_levers": True, option: True})


def _make_request(request_id, arrival_time):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=1000,
        arrival_time=arrival_time,
        num_preemptions=0,
        num_computed_tokens=0,
        skip_reading_prefix_cache=False,
        block_hashes=[object()],
    )


def _running_request(request_id, prompt, computed):
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        status=RequestStatus.RUNNING,
    )


def _step_ready(s, warmth):
    """Gives a built scheduler two queued and two KV-holding requests (one
    of each cached ``warmth`` tokens deep), a partial prefill ahead of a
    decoder in ``running``, and what the levers' telemetry reads."""
    now = time.time()
    queued = [_make_request("cold", now - 10), _make_request("warm", now - 5)]
    holding = [
        _make_request("hold_cold", now - 10),
        _make_request("hold_warm", now - 5),
    ]
    hits = {"warm": warmth, "hold_warm": warmth}

    def find_hit(block_hashes, max_len):
        for r in queued + holding:
            if r.block_hashes is block_hashes:
                return ({}, hits.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    s.kv_cache_manager.enable_caching = True
    s.kv_cache_manager.coordinator.find_longest_cache_hit.side_effect = find_hit
    s.waiting = FCFSRequestQueue()
    for r in queued:
        s.waiting.add_request(r)
    s.kv_holding_waiting = FCFSRequestQueue()
    for r in holding:
        s.kv_holding_waiting.add_request(r)
    s.policy = SchedulingPolicy.FCFS
    s.running = [
        _running_request("p1", prompt=10_000, computed=640),
        _running_request("d1", prompt=500, computed=501),
    ]
    s._inflight_prefills = set()
    s.max_num_active_reqs = 8
    s.num_waiting_for_streaming_input = 0
    # A list: the stand-in requests are not hashable.
    s.deferred_waiting = []
    s.max_num_scheduled_tokens = 8192
    s.block_size = 16


def _new_request(prompt):
    return SimpleNamespace(
        prompt_token_ids=prompt,
        mm_features=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        sampling_params=SimpleNamespace(
            thinking_token_budget=None, prompt_logprobs=None, extra_args=None
        ),
    )


def _fake_cached(self, *args, **kwargs):
    return CachedRequestData(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={"a": [1, 2, 3, 4, 5]},
        new_block_ids=[None],
        num_computed_tokens=[3],
        num_output_tokens=[2],
    )


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("role", OTHER_ROLES)
def test_schedule_runs_the_decode_leg_levers_with_the_option(monkeypatch, role, on):
    """With the option: decoders ahead of partial prefills, the warmest
    request of both waiting queues promoted, packed prompt ids and the
    trimmed resend. Without it: the warmest-first admission alone."""
    s = _built(monkeypatch, role, {"prefix_aware_levers": on})
    _step_ready(s, warmth=800)
    prompt = list(range(4096))
    new_req = _new_request(prompt)
    out = SimpleNamespace(scheduled_new_reqs=[new_req], total_num_scheduled_tokens=1)
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["running"] = [r.request_id for r in self.running]
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    monkeypatch.setattr(AsyncScheduler, "_make_cached_request_data", _fake_cached)
    monkeypatch.setattr(Scheduler, "_inflight_prefill_reserved_blocks", lambda s: 0)
    assert s.schedule() is out
    cached = s._make_cached_request_data([], [], {}, {}, {})
    assert [r.request_id for r in s.kv_holding_waiting] == ["hold_warm", "hold_cold"]
    if on:
        assert seen["running"] == ["d1", "p1"]
        assert [r.request_id for r in s.waiting] == ["warm", "cold"]
        assert isinstance(new_req.prompt_token_ids, PackedPromptIds)
        assert cached.all_token_ids == {"a": [4, 5]}
    else:
        assert seen["running"] == ["p1", "d1"]
        assert [r.request_id for r in s.waiting] == ["cold", "warm"]
        assert new_req.prompt_token_ids is prompt
        assert cached.all_token_ids == {"a": [1, 2, 3, 4, 5]}


@pytest.mark.parametrize("on", [False, True])
def test_the_levers_telemetry_records_why_the_waiting_loop_stopped(monkeypatch, on):
    s = _built(monkeypatch, "kv_both", {"prefix_aware_levers": on})
    s._step_wait_event = None
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.WAITING), "align0")
    assert s._step_wait_event == ("align0" if on else None)


# ---------------- divergent hybrid hits ----------------


def _diverged_lookup(monkeypatch, s, block_size=2048):
    """vLLM's lookup answers a diverged 8192-token local hit whose
    reconciled hit is 2048 tokens."""
    seen = []

    def base(self, request, max_cache_hit_length=None):
        seen.append(max_cache_hit_length)
        return "blocks", 8192, 0, True

    monkeypatch.setattr(Scheduler, "_get_local_prefix_cache_hit", base)
    s.kv_cache_manager = SimpleNamespace(
        get_computed_blocks=lambda request, cap: ("reconciled", 2048, 64),
        truncate_computed_blocks=lambda blocks, n: f"{blocks}[:{n}]",
        empty_kv_cache_blocks="empty",
    )
    s.block_size = block_size
    return seen


@pytest.mark.parametrize(
    "target, expected",
    [
        (4096, ("blocks[:4096]", 4096, 0, True)),
        (8192, ("blocks", 8192, 0, True)),
        (None, ("reconciled", 2048, 64, False)),
        (0, ("empty", 0, 64, False)),
    ],
)
@pytest.mark.parametrize("role", ["kv_both", "kv_producer"])
def test_the_option_resolves_a_divergent_hit_through_the_connector(
    monkeypatch, role, target, expected
):
    connector = _Resolver(target)
    s = _built(monkeypatch, role, {"prefix_aware_levers": True}, connector=connector)
    assert s._hit_resolver is connector
    seen = _diverged_lookup(monkeypatch, s)
    request = SimpleNamespace(request_id="r", num_tokens=16384)
    assert s._get_local_prefix_cache_hit(request, 12288) == expected
    assert connector.calls == [(8192, 2048, 2048)]
    assert seen == [12288]


@pytest.mark.parametrize("memo", [False, True])
def test_the_option_records_the_local_hit_for_the_connector(monkeypatch, memo):
    """From the memo or from the lookup, as the pipeline chunk scheduler's
    prefix-aware admission records it; not without the option."""
    hit = ("blocks", 640, 0, False)
    monkeypatch.setattr(
        prefix_aware_scheduler,
        "memo_local_prefix_cache_hit",
        lambda s, request, cap: hit if memo else None,
    )
    monkeypatch.setattr(
        Scheduler, "_get_local_prefix_cache_hit", lambda self, request, cap=None: hit
    )
    for role, section, recorded in [
        ("kv_both", {"prefix_aware_levers": True}, True),
        (None, {"prefix_aware_levers": True}, True),
        ("kv_both", None, False),
        ("kv_consumer", {"prefix_aware_levers": True}, False),
    ]:
        s = _built(monkeypatch, role, section)
        request = SimpleNamespace(request_id="r")
        assert s._get_local_prefix_cache_hit(request) == hit
        assert getattr(request, "_tpu_local_hit_tokens", None) == (
            640 if recorded else None
        ), (role, section)


def test_a_hit_that_did_not_diverge_is_not_resolved(monkeypatch):
    connector = _Resolver(0)
    s = _built(
        monkeypatch, "kv_both", {"prefix_aware_levers": True}, connector=connector
    )
    monkeypatch.setattr(
        Scheduler,
        "_get_local_prefix_cache_hit",
        lambda self, request, cap=None: ("blocks", 640, 0, False),
    )
    assert s._get_local_prefix_cache_hit(SimpleNamespace()) == ("blocks", 640, 0, False)
    assert connector.calls == []


def test_without_the_option_a_divergent_hit_keeps_vllms_lookup(monkeypatch):
    connector = _Resolver(0)
    s = _built(monkeypatch, "kv_both", None, connector=connector)
    assert s._hit_resolver is None
    _diverged_lookup(monkeypatch, s)
    got = s._get_local_prefix_cache_hit(SimpleNamespace(request_id="r"), None)
    assert got == ("blocks", 8192, 0, True)
    assert connector.calls == []


@pytest.mark.parametrize("claims", [True, False])
def test_init_says_whether_the_resolution_is_live(monkeypatch, caplog, claims):
    """vLLM's lookup reports a diverged hit only for a connector that claims
    divergent hybrid hits."""
    connector = _Resolver()
    connector.supports_divergent_local_hybrid_hits = claims
    with caplog.at_level("INFO"):
        s = _built(
            monkeypatch, "kv_both", {"prefix_aware_levers": True}, connector=connector
        )
    assert s._hit_resolver is connector
    assert ("_Resolver resolves divergent hybrid hits" in caplog.text) is claims
    assert ("the resolution stays idle" in caplog.text) is not claims


def test_a_connector_without_the_hook_is_named_once(monkeypatch, caplog):
    class Offload:
        supports_divergent_local_hybrid_hits = False

    with caplog.at_level("INFO"):
        s = _built(
            monkeypatch, "kv_both", {"prefix_aware_levers": True}, connector=Offload()
        )
    assert s._levers_on() and s._hit_resolver is None
    assert "Offload has no resolve_divergent_local_hit_tokens" in caplog.text
    caplog.clear()
    # Without a connector (no KV transfer) there is nothing to say.
    with caplog.at_level("INFO"):
        s = _built(monkeypatch, None, {"prefix_aware_levers": True})
    assert s._levers_on() and s._hit_resolver is None
    assert "resolve_divergent_local_hit_tokens" not in caplog.text
