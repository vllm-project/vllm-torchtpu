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

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


def _make_request(
    request_id: str,
    num_tokens: int = 1000,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
    num_computed_tokens: int = 0,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        block_hashes=[object()],
    )


def _make_scheduler(
    requests: list[SimpleNamespace],
    warmth_map: dict[str, int],
    window: int = 64,
    enable_caching: bool = True,
    raise_on_lookup: bool = False,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.kv_holding_waiting = FCFSRequestQueue()
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = window
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False

    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        if raise_on_lookup:
            raise RuntimeError("simulated coordinator failure")
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth_map.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit

    kvm = MagicMock()
    kvm.enable_caching = enable_caching
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    return s


def test_default_knobs():
    assert PrefixAwareScheduler.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})

    expected = object()

    def schedule(throttle_prefills):
        assert throttle_prefills is True
        assert [r.request_id for r in s.waiting] == ["warm", "cold"]
        return expected

    with patch.object(AsyncScheduler, "schedule", side_effect=schedule):
        assert s.schedule(throttle_prefills=True) is expected


def test_ranks_by_uncached_tokens_not_hit_ratio():
    now = time.time()
    # 100k tokens, 90% cached -> 10k uncached tokens
    r_long = _make_request("long", num_tokens=100_000, arrival_time=now - 10)
    # 20k tokens, 80% cached -> 4k uncached tokens (fewer uncached tokens wins)
    r_short = _make_request("short", num_tokens=20_000, arrival_time=now - 5)
    s = _make_scheduler([r_long, r_short], {"long": 90_000, "short": 16_000})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["short", "long"]


def test_fcfs_order_preserved_when_all_cold():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["r1", "r2"]


def test_preempted_tier_outranks_normal():
    now = time.time()
    r_preempted = _make_request(
        "preempted", num_tokens=1000, arrival_time=now - 5, num_preemptions=1
    )
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 10)
    s = _make_scheduler([r_preempted, r_warm], {"preempted": 0, "warm": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["preempted", "warm"]


def test_window_bounds_evaluation():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 15)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 10)
    r_outside = _make_request("outside", num_tokens=1000, arrival_time=now - 5)
    # window = 2, so r_outside is beyond the inspection window
    s = _make_scheduler(
        [r1, r2, r_outside], {"r1": 100, "r2": 200, "outside": 999}, window=2
    )

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order[0] == "r2"  # r2 was warmest in window [r1, r2]


def test_single_or_empty_queue_is_noop():
    s = _make_scheduler([_make_request("single")], {"single": 500})
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["single"]


def test_uses_num_computed_tokens_when_present():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, num_computed_tokens=900
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]


def test_skip_reading_prefix_cache_or_caching_disabled():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, skip_reading_prefix_cache=True
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]

    s_disabled = _make_scheduler(
        [r1, _make_request("r3", num_tokens=1000, arrival_time=now - 5)],
        {"r1": 0, "r3": 900},
        enable_caching=False,
    )
    q_disabled = s_disabled._select_waiting_queue_for_scheduling()
    assert q_disabled is not None
    assert [r.request_id for r in q_disabled] == ["r1", "r3"]


def test_coordinator_failure_falls_back_to_fcfs():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True)

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]
    assert s._paw_scoring_broken is True


def test_prefix_aware_srpt_scheduler_chains_both_layers():
    from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import (
        PrefixAwareSrptScheduler,
        SrptPrefillScheduler,
    )

    mro = PrefixAwareSrptScheduler.__mro__
    assert mro.index(PrefixAwareScheduler) < mro.index(SrptPrefillScheduler)
    assert mro.index(SrptPrefillScheduler) < mro.index(AsyncScheduler)
    calls = []

    def srpt_schedule(self, throttle_prefills=False):
        calls.append(("srpt", throttle_prefills))

    s = _make_scheduler([], {})
    s.__class__ = PrefixAwareSrptScheduler
    with patch.object(SrptPrefillScheduler, "schedule", srpt_schedule):
        s.schedule()
    # PAW reorders the waiting queue, then forwards to the SRPT layer.
    assert calls == [("srpt", False)]


def test_prompt_ids_shipped_as_ndarray_when_enabled():
    import numpy as np

    s = _make_scheduler([], {})
    s._prompt_np = True
    ids = [5, 7, 2**31 - 1, 0]
    request = SimpleNamespace(request_id="r", prompt_token_ids=ids)
    with patch.object(AsyncScheduler, "add_request", return_value=None):
        s.add_request(request)
    np.testing.assert_array_equal(request._prompt_ids_np, np.asarray(ids, np.int32))
    assert request.prompt_token_ids is ids  # the engine keeps its list
    grown = SimpleNamespace(request_id="g", prompt_token_ids=[1, 2])
    grown._prompt_ids_np = np.asarray([1], np.int32)  # list grew after add
    s.requests = {"r": request, "g": grown}
    new_r = SimpleNamespace(req_id="r", prompt_token_ids=ids)
    new_g = SimpleNamespace(req_id="g", prompt_token_ids=grown.prompt_token_ids)
    out = SimpleNamespace(scheduled_new_reqs=[new_r, new_g])
    with patch.object(AsyncScheduler, "schedule", return_value=out):
        assert s.schedule() is out
    assert isinstance(new_r.prompt_token_ids, np.ndarray)
    assert new_r.prompt_token_ids.tolist() == ids
    assert new_g.prompt_token_ids == [1, 2]


def test_prompt_ids_left_as_list_by_default():
    s = _make_scheduler([], {})
    ids = [1, 2, 3]
    request = SimpleNamespace(request_id="r", prompt_token_ids=ids)
    with patch.object(AsyncScheduler, "add_request", return_value=None):
        s.add_request(request)
    assert not hasattr(request, "_prompt_ids_np")
    s.requests = {"r": request}
    new_r = SimpleNamespace(req_id="r", prompt_token_ids=ids)
    out = SimpleNamespace(scheduled_new_reqs=[new_r])
    with patch.object(AsyncScheduler, "schedule", return_value=out):
        s.schedule()
    assert new_r.prompt_token_ids is ids


def test_skip_full_skips_scoring_only_when_no_seat_is_free():
    reqs = [_make_request("cold", 1000, 0.0), _make_request("warm", 1000, 1.0)]
    s = _make_scheduler(reqs, {"warm": 900})
    s._paw_skip_full = True
    s.max_num_active_reqs = 2
    s.num_waiting_for_streaming_input = 0
    s.running = [object(), object()]
    s._promote_warmest(s.waiting)
    assert [r.request_id for r in s.waiting] == ["cold", "warm"]
    s.kv_cache_manager.coordinator.find_longest_cache_hit.assert_not_called()
    s.running = [object()]
    s._promote_warmest(s.waiting)
    assert [r.request_id for r in s.waiting] == ["warm", "cold"]


def _shipped_payload(s, ids, early):
    """Pickled scheduler-output payload of one new request, with the prompt
    converted in the input thread (``early``) or in ``add_request``."""
    import pickle

    request = SimpleNamespace(request_id="r", prompt_token_ids=ids)
    if early:
        from vllm_torchtpu.core.prefix_aware_scheduler import prompt_ids_to_ndarray

        prompt_ids_to_ndarray(request)
        converted = request._prompt_ids_np
    with patch.object(AsyncScheduler, "add_request", return_value=None):
        s.add_request(request)
    if early:
        assert request._prompt_ids_np is converted  # add_request reused it
    s.requests = {"r": request}
    new_r = SimpleNamespace(req_id="r", prompt_token_ids=ids)
    out = SimpleNamespace(scheduled_new_reqs=[new_r])
    with patch.object(AsyncScheduler, "schedule", return_value=out):
        s.schedule()
    return pickle.dumps(new_r.prompt_token_ids)


def test_prompt_ids_ndarray_early_ships_identical_payload():
    s = _make_scheduler([], {})
    s._prompt_np = True
    ids = [5, 7, 2**31 - 1, 0] * 1000
    assert _shipped_payload(s, list(ids), early=True) == _shipped_payload(
        s, list(ids), early=False
    )


def test_prompt_ids_ndarray_early_patch_wraps_preprocess(monkeypatch):
    import numpy as np
    from vllm.v1.engine.core import EngineCore

    from vllm_torchtpu import envs
    from vllm_torchtpu.core import prefix_aware_scheduler as paw

    ids = [3, 1, 4, 1, 5]
    request = SimpleNamespace(request_id="r", prompt_token_ids=ids)

    def fake_preprocess(self, req):
        return request, 0

    monkeypatch.setattr(EngineCore, "preprocess_add_request", fake_preprocess)
    # Off (either flag unset): no patch.
    monkeypatch.setattr(envs, "VLLM_TPU_PROMPT_IDS_NDARRAY", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_PROMPT_IDS_NDARRAY_EARLY", False, raising=False)
    paw.patch_prompt_ids_ndarray_early()
    assert EngineCore.preprocess_add_request is fake_preprocess
    # On: the input-thread hook converts once; a second patch call is a no-op.
    monkeypatch.setattr(envs, "VLLM_TPU_PROMPT_IDS_NDARRAY_EARLY", True, raising=False)
    paw.patch_prompt_ids_ndarray_early()
    patched = EngineCore.preprocess_add_request
    assert patched is not fake_preprocess
    paw.patch_prompt_ids_ndarray_early()
    assert EngineCore.preprocess_add_request is patched
    out = EngineCore.preprocess_add_request(None, object())
    assert out == (request, 0)
    np.testing.assert_array_equal(request._prompt_ids_np, np.asarray(ids, np.int32))
    assert request.prompt_token_ids is ids


def test_prompt_ids_ndarray_early_real_preprocess(monkeypatch):
    """The wrapped real EngineCore.preprocess_add_request builds a real
    Request whose cached ndarray add_request then reuses."""
    import numpy as np
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.core import EngineCore

    from vllm_torchtpu import envs
    from vllm_torchtpu.core import prefix_aware_scheduler as paw

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request
    )
    monkeypatch.setattr(envs, "VLLM_TPU_PROMPT_IDS_NDARRAY", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_PROMPT_IDS_NDARRAY_EARLY", True, raising=False)
    paw.patch_prompt_ids_ndarray_early()
    ids = list(range(1000, 1000 + 4096))
    core_req = EngineCoreRequest(
        request_id="r0",
        prompt_token_ids=ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    fake_core = SimpleNamespace(
        mm_receiver_cache=None, request_block_hasher=None, structured_output_manager=None
    )
    request, wave = EngineCore.preprocess_add_request(fake_core, core_req)
    assert wave == 0
    converted = request._prompt_ids_np
    np.testing.assert_array_equal(converted, np.asarray(ids, np.int32))
    s = _make_scheduler([], {})
    s._prompt_np = True
    with patch.object(AsyncScheduler, "add_request", return_value=None):
        s.add_request(request)
    assert request._prompt_ids_np is converted
