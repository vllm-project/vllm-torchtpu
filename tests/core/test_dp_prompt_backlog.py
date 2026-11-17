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
"""The engine choice by pending prompt tokens, on vLLM's own engine-core
publisher, msgpack wire format and data-parallel client class."""

import queue
import random
from collections import Counter
from types import SimpleNamespace

import pytest
from vllm.pooling_params import PoolingParams
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.pool.late_interaction import (
    build_late_interaction_query_params,
    get_late_interaction_engine_index,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.core import dp_prompt_backlog as backlog

ENGINES = 8


@pytest.fixture
def installed(monkeypatch):
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(
        DPEngineCoreProc, "_maybe_publish_request_counts", backlog.make_publisher()
    )
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "lb_engines",
        property(backlog._get_lb_engines, backlog._set_lb_engines),
        raising=False,
    )
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        backlog.make_router(stock, margin=1.0),
    )
    return stock


class Client(DPLBAsyncMPClient):
    """The attributes get_core_engine_for_request reads, without the engine
    processes and sockets __init__ starts."""

    def __init__(self, client_count: int = 1):
        self.client_count = client_count
        self.core_engines = [i.to_bytes(2, "little") for i in range(ENGINES)]
        self.reqs_in_flight = {}
        self.engine_inflight = Counter()
        self.eng_start_index = 0
        self.lb_engines = [[0, 0, 0.0] for _ in range(ENGINES)]


def request(request_id: str, prompt_tokens: int, **fields) -> EngineCoreRequest:
    values = dict(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    values.update(fields)
    return EngineCoreRequest(**values)


class Request(SimpleNamespace):
    @property
    def num_tokens(self):
        return self.num_prompt_tokens


class Coordinator:
    """vLLM's coordinator path: decode the engine's message, copy
    [waiting, running, kv_cache_usage] into the engine's row, relay the rows."""

    def __init__(self):
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)
        self.rows = [[0, 0, 0.0] for _ in range(ENGINES)]

    def receive(self, engine_index: int, outputs: EngineCoreOutputs) -> None:
        wire = self.encoder.encode(outputs)
        stats = self.decoder.decode(wire).scheduler_stats
        row = self.rows[engine_index]
        row[0] = stats.num_waiting_reqs
        row[1] = stats.num_running_reqs
        row[2] = stats.kv_cache_usage

    def relay(self):
        relay_encoder, relay_decoder = MsgpackEncoder(), MsgpackDecoder()
        return relay_decoder.decode(relay_encoder.encode(self.rows))


def engine(waiting, running, cached_by_id, kv_usage=0.25, kv_holding_waiting=()):
    lookups = []

    def find_longest_cache_hit(block_hashes, max_length):
        lookups.append(block_hashes)
        return None, min(cached_by_id[block_hashes[0]], max_length), 0

    manager = SimpleNamespace(
        prefix_cache_lookup_enabled=lambda request: True,
        coordinator=SimpleNamespace(find_longest_cache_hit=find_longest_cache_hit),
    )
    kv_holding_waiting = list(kv_holding_waiting)
    scheduler = SimpleNamespace(
        waiting=waiting,
        kv_holding_waiting=kv_holding_waiting,
        running=running,
        kv_cache_manager=manager,
        get_request_counts=lambda: (
            len(running),
            len(waiting) + len(kv_holding_waiting),
        ),
        get_kv_cache_usage=lambda: kv_usage,
    )
    core = SimpleNamespace(
        publish_dp_lb_stats=True,
        scheduler=scheduler,
        last_counts=(0, 0),
        step_counter=7,
        current_wave=1,
        output_queue=queue.Queue(),
    )
    return core, lookups


def test_pack_round_trip():
    for pending, kv in [(0, 0.0), (1, 0.5), (123_456, 0.999), (10**7, 1.0)]:
        got_pending, got_kv = backlog.unpack(backlog.pack(pending, kv))
        assert got_pending == pending
        assert got_kv == pytest.approx(min(kv, backlog.KV_USAGE_CAP), abs=1e-6)


def test_engine_publishes_pending_prompt_tokens(installed):
    waiting = [
        Request(
            request_id="w0",
            num_prompt_tokens=50_000,
            num_computed_tokens=0,
            block_hashes=["w0"],
        ),
        Request(
            request_id="w1",
            num_prompt_tokens=8_000,
            num_computed_tokens=0,
            block_hashes=["w1"],
        ),
    ]
    running = [
        Request(
            request_id="r0",
            num_prompt_tokens=100_000,
            num_computed_tokens=30_000,
            block_hashes=["r0"],
        ),
        Request(
            request_id="r1",
            num_prompt_tokens=4_000,
            num_computed_tokens=4_100,
            block_hashes=["r1"],
        ),
    ]
    core, lookups = engine(waiting, running, {"w0": 45_000, "w1": 0})
    DPEngineCoreProc._maybe_publish_request_counts(core)

    _, outputs = core.output_queue.get_nowait()
    coordinator = Coordinator()
    coordinator.receive(3, outputs)
    client = Client()
    client.lb_engines = coordinator.relay()

    assert client.lb_engines[3] == [2, 2, pytest.approx(0.25)]
    assert client._lb_pending[3] == 5_000 + 8_000 + 70_000
    assert outputs.scheduler_stats.step_counter == 7

    # Unchanged counts and backlog publish nothing; the cached prefix of a
    # waiting request is looked up once.
    DPEngineCoreProc._maybe_publish_request_counts(core)
    assert core.output_queue.empty()
    assert len(lookups) == 2

    # A prefill step changes only the backlog, and that is published.
    running[0].num_computed_tokens += 1024
    DPEngineCoreProc._maybe_publish_request_counts(core)
    _, outputs = core.output_queue.get_nowait()
    pending, kv = backlog.unpack(outputs.scheduler_stats.kv_cache_usage)
    assert pending == 83_000 - 1024
    assert kv == pytest.approx(0.25)


def test_requests_holding_kv_while_waiting_count_as_pending(installed):
    """vLLM admits from kv_holding_waiting before waiting: a request there
    still has its uncomputed prompt ahead of it."""
    holding = Request(
        request_id="h0",
        num_prompt_tokens=6_000,
        num_computed_tokens=1_000,
        block_hashes=["h0"],
    )
    waiting = [
        Request(
            request_id="w0",
            num_prompt_tokens=2_000,
            num_computed_tokens=0,
            block_hashes=["w0"],
        )
    ]
    core, _ = engine(waiting, [], {"h0": 0, "w0": 0}, kv_holding_waiting=[holding])
    DPEngineCoreProc._maybe_publish_request_counts(core)
    _, outputs = core.output_queue.get_nowait()
    pending, _ = backlog.unpack(outputs.scheduler_stats.kv_cache_usage)
    assert pending == 5_000 + 2_000
    assert outputs.scheduler_stats.num_waiting_reqs == 2


def test_equal_backlogs_keep_vllms_choice(installed):
    stock = installed
    rng = random.Random(0)
    patched, reference = Client(client_count=4), Client(client_count=4)
    for i in range(400):
        rows = [
            [rng.randrange(4), rng.randrange(8), rng.random()] for _ in range(ENGINES)
        ]
        patched.lb_engines = [[w, r, backlog.pack(12_000, kv)] for w, r, kv in rows]
        reference.__dict__["_lb_counts"] = [list(row) for row in rows]
        reference.__dict__["_lb_pending"] = [0] * ENGINES
        req = request(f"q{i}", rng.randrange(100, 60_000))
        got = DPLBAsyncMPClient.get_core_engine_for_request(patched, req)
        want = stock(reference, req)
        assert got == want
        assert patched.engine_inflight == reference.engine_inflight
        for got_row, want_row in zip(patched.lb_engines, reference.lb_engines):
            assert got_row == pytest.approx(want_row, abs=1e-9)


def test_request_leaves_engine_behind_long_prefill(installed):
    client = Client()
    # Equal request counts everywhere; engine 0 holds a 100,000-token prefill.
    rows = [[0, 3, backlog.pack(0, 0.2)] for _ in range(ENGINES)]
    rows[0][2] = backlog.pack(100_000, 0.2)
    for idx in range(1, ENGINES):
        rows[idx][1] = 4
    client.lb_engines = rows
    assert DPLBAsyncMPClient.get_core_engine_for_request(client, request("a", 1)) != 0

    client = Client()
    client.lb_engines = rows
    client.lb_engines[0][1] = 1
    stats = DPLBAsyncMPClient.get_core_engine_for_request.backlog_stats
    moved = stats.moved
    chosen = DPLBAsyncMPClient.get_core_engine_for_request(client, request("b", 20_000))
    assert chosen != client.core_engines[0]
    assert stats.moved == moved + 1
    assert client.engine_inflight[client.core_engines[0]] == 0
    assert client.reqs_in_flight["b"] == chosen
    assert client.lb_engines[0][0] == 0
    # The moved prompt counts on its new engine until the next relay.
    assert client._lb_pending[client.core_engines.index(chosen)] == 20_000


def test_long_prompt_stays_where_its_cache_may_be(installed):
    client = Client()
    rows = [[0, 4, backlog.pack(0, 0.2)] for _ in range(ENGINES)]
    rows[0] = [0, 1, backlog.pack(100_000, 0.2)]
    client.lb_engines = rows
    chosen = DPLBAsyncMPClient.get_core_engine_for_request(
        client, request("c", 150_000)
    )
    assert chosen == client.core_engines[0]


def test_follow_up_and_pinned_requests_bypass(installed):
    client = Client()
    rows = [[0, 0, backlog.pack(0, 0.0)] for _ in range(ENGINES)]
    rows[2] = [0, 0, backlog.pack(500_000, 0.0)]
    client.lb_engines = rows
    pinned = DPLBAsyncMPClient.get_core_engine_for_request(
        client, request("d", 10, data_parallel_rank=2)
    )
    assert pinned == client.core_engines[2]


def test_late_interaction_request_stays_on_its_engine(installed):
    client = Client()
    params = PoolingParams(
        late_interaction_params=build_late_interaction_query_params("q", 1)
    )
    home = get_late_interaction_engine_index(params, ENGINES)
    rows = [[0, 0, backlog.pack(0, 0.0)] for _ in range(ENGINES)]
    rows[home] = [0, 0, backlog.pack(500_000, 0.0)]
    client.lb_engines = rows
    chosen = DPLBAsyncMPClient.get_core_engine_for_request(
        client, request("e", 10, pooling_params=params)
    )
    assert chosen == client.core_engines[home]
    assert client.lb_engines[home][0] == 0


def test_burst_spreads_over_idle_engines(installed):
    """Between relays, each routed prompt counts on its engine, so a burst
    does not pile onto the engine that looked emptiest."""
    client = Client()
    rows = [[0, 2, backlog.pack(300_000, 0.1)] for _ in range(ENGINES)]
    rows[5] = [0, 2, backlog.pack(0, 0.1)]
    rows[6] = [0, 2, backlog.pack(0, 0.1)]
    client.lb_engines = rows
    chosen = [
        client.core_engines.index(
            DPLBAsyncMPClient.get_core_engine_for_request(
                client, request(f"e{i}", 30_000)
            )
        )
        for i in range(4)
    ]
    assert sorted(chosen) == [5, 5, 6, 6]


def test_install_patches_both_sides(monkeypatch):
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        DPLBAsyncMPClient.get_core_engine_for_request,
    )
    monkeypatch.setattr(
        DPEngineCoreProc,
        "_maybe_publish_request_counts",
        DPEngineCoreProc._maybe_publish_request_counts,
    )
    monkeypatch.setattr(DPLBAsyncMPClient, "lb_engines", None, raising=False)
    monkeypatch.delattr(DPLBAsyncMPClient, "lb_engines")
    monkeypatch.setenv("VLLM_TPU_DP_PROMPT_BACKLOG", "1")
    backlog.install_if_enabled()
    assert DPLBAsyncMPClient.get_core_engine_for_request.backlog_stats is not None
    assert isinstance(DPLBAsyncMPClient.__dict__["lb_engines"], property)
    assert DPEngineCoreProc._maybe_publish_request_counts.__module__ == backlog.__name__


def test_client_built_before_install(installed):
    client = object.__new__(Client)
    client.__dict__.update(
        client_count=1,
        core_engines=[i.to_bytes(2, "little") for i in range(ENGINES)],
        reqs_in_flight={},
        engine_inflight=Counter(),
        eng_start_index=0,
        lb_engines=[[0, 0, 0.0] for _ in range(ENGINES)],
    )
    chosen = DPLBAsyncMPClient.get_core_engine_for_request(client, request("f", 100))
    assert chosen == client.core_engines[0]
    assert client._lb_pending[0] == 100
