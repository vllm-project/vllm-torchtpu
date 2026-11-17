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
"""The prefix-aware engine choice on vLLM's own data-parallel client class,
driven with the client state the stock balancer reads."""

import random
from collections import Counter

import pytest
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_torchtpu.core import dp_prefix_affinity as affinity
from vllm_torchtpu.core import dp_prompt_backlog as backlog

CHUNK = 64
ENGINES = 8


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

    def finish(self, request_id: str) -> None:
        engine = self.reqs_in_flight.pop(request_id)
        self.engine_inflight[engine] -= 1

    def snapshot(self) -> None:
        """The coordinator's periodic stats: every in-flight request running."""
        for idx, engine in enumerate(self.core_engines):
            self.lb_engines[idx] = [0, self.engine_inflight[engine], 0.0]


def request(request_id: str, token_ids: list[int], **fields) -> EngineCoreRequest:
    values = dict(
        request_id=request_id,
        prompt_token_ids=token_ids,
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


@pytest.fixture
def routed(monkeypatch):
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        affinity.make_router(stock, CHUNK, load_margin=2, capacity=4096),
    )
    return stock


def conversations(count: int, turns: int, seed: int = 0):
    """Agentic conversations: each turn repeats the previous prompt and its
    answer and adds new text; a shared system prompt opens every one."""
    rng = random.Random(seed)
    system = [rng.randrange(1, 50000) for _ in range(3 * CHUNK)]
    prompts = {
        c: system + [rng.randrange(1, 50000) for _ in range(CHUNK)]
        for c in range(count)
    }
    for turn in range(turns):
        for c in rng.sample(range(count), count):
            yield c, turn, list(prompts[c])
            prompts[c] += [
                rng.randrange(1, 50000)
                for _ in range(rng.randrange(CHUNK // 2, 3 * CHUNK))
            ]


def drive(client: Client, count: int = 32, turns: int = 12) -> tuple[int, int, Counter]:
    """Routes every turn while `count` conversations each hold one request
    in flight, and returns (follow-up turns, those on their previous engine,
    requests per engine)."""
    previous: dict[int, bytes] = {}
    in_flight: dict[int, str] = {}
    same = follow_ups = 0
    per_engine: Counter = Counter()
    for c, turn, prompt in conversations(count, turns):
        if c in in_flight:
            client.finish(in_flight.pop(c))
        rid = f"c{c}-t{turn}"
        engine = client.get_core_engine_for_request(request(rid, prompt))
        in_flight[c] = rid
        client.snapshot()
        per_engine[engine] += 1
        if c in previous:
            follow_ups += 1
            same += engine == previous[c]
        previous[c] = engine
    return follow_ups, same, per_engine


def test_hashes_name_prefixes():
    a = list(range(5 * CHUNK + 7))
    b = a[: 3 * CHUNK] + [-1] * (2 * CHUNK)
    ha, hb = affinity.boundary_hashes(a, CHUNK), affinity.boundary_hashes(b, CHUNK)
    assert len(ha) == 5 and len(hb) == 5
    assert ha[:3] == hb[:3] and ha[3] != hb[3]
    assert affinity.boundary_hashes(a[: CHUNK - 1], CHUNK) == []


def test_stock_choice_scatters_turns():
    """The check can fail: without the change most follow-up turns change
    engine on this workload."""
    follow_ups, same, _ = drive(Client())
    assert same / follow_ups < 0.5


def test_turns_follow_their_prefix(routed):
    follow_ups, same, per_engine = drive(Client())
    assert same / follow_ups > 0.95
    assert max(per_engine.values()) - min(per_engine.values()) <= 0.25 * (
        sum(per_engine.values()) / ENGINES
    )


def test_busy_engine_is_left_for_the_stock_choice(routed):
    client = Client()
    prompt = list(range(4 * CHUNK))
    first = client.get_core_engine_for_request(request("a", prompt))
    owner = client.core_engines.index(first)
    client.lb_engines[owner] = [0, 2, 0.0]
    assert client.get_core_engine_for_request(request("b", prompt + [7])) == first
    client.lb_engines[owner] = [0, 4, 0.0]
    other = client.get_core_engine_for_request(request("c", prompt + [8]))
    assert other != first
    assert client.get_core_engine_for_request(request("d", prompt + [9])) == other


def test_bookkeeping_matches_the_stock_choice(routed):
    client = Client()
    prompt = list(range(2 * CHUNK))
    engine = client.get_core_engine_for_request(request("a", prompt))
    client.finish("a")
    before = [list(x) for x in client.lb_engines]
    again = client.get_core_engine_for_request(request("b", prompt))
    idx = client.core_engines.index(again)
    assert again == engine
    assert client.reqs_in_flight["b"] == again and client.engine_inflight[again] == 1
    assert client.lb_engines[idx][0] == before[idx][0] + client.client_count


def test_requests_the_stock_choice_owns(routed):
    client = Client()
    prompt = list(range(2 * CHUNK))
    client.get_core_engine_for_request(request("a", prompt))
    pinned = client.get_core_engine_for_request(
        request("b", prompt, data_parallel_rank=5)
    )
    assert pinned == client.core_engines[5]
    short = client.get_core_engine_for_request(request("c", prompt[: CHUNK - 1]))
    assert short in client.core_engines
    assert client.get_core_engine_for_request(request("c", prompt)) == short


def test_prefix_choice_falls_back_to_the_backlog_choice(monkeypatch):
    """With both switches on, the prefix holder is kept whatever its prompt
    backlog; a prompt without one gets the backlog choice, and the table
    remembers the engine the request finally went to."""
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "lb_engines",
        property(backlog._get_lb_engines, backlog._set_lb_engines),
        raising=False,
    )
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        affinity.make_router(
            backlog.make_router(stock, margin=1.0), CHUNK, load_margin=2, capacity=4096
        ),
    )
    client = Client()
    prompt = list(range(4 * CHUNK))
    first = client.get_core_engine_for_request(request("a", prompt))
    owner = client.core_engines.index(first)
    client.finish("a")
    backlogged_rows = [
        [0, 0, float(100 * len(prompt))] if idx == owner else [0, 1, 0.0]
        for idx in range(ENGINES)
    ]
    client.lb_engines = backlogged_rows
    assert client.get_core_engine_for_request(request("b", prompt + [7])) == first
    client.finish("b")

    client.lb_engines = backlogged_rows
    other_prompt = list(range(10**6, 10**6 + 4 * CHUNK))
    moved = client.get_core_engine_for_request(request("c", other_prompt))
    assert moved != first
    client.finish("c")
    assert client.get_core_engine_for_request(request("d", other_prompt + [7])) == moved


def test_prefix_every_engine_holds_leaves_the_backlog_choice(monkeypatch):
    """A prefix all engines hold favors none, so the backlog choice decides."""
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "lb_engines",
        property(backlog._get_lb_engines, backlog._set_lb_engines),
        raising=False,
    )
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        affinity.make_router(
            backlog.make_router(stock, margin=1.0), CHUNK, load_margin=2, capacity=4096
        ),
    )
    client = Client()
    system = list(range(2 * CHUNK))
    client._prefix_table = affinity.PrefixTable(4096)
    for idx in range(ENGINES):
        client._prefix_table.remember(affinity.boundary_hashes(system, CHUNK), idx)
    client.lb_engines = [
        [0, 0, float(100 * len(system))] if idx == 0 else [0, 1, 0.0]
        for idx in range(ENGINES)
    ]
    assert (
        client.get_core_engine_for_request(request("a", system + [7]))
        != (client.core_engines[0])
    )


def test_table_is_bounded():
    table = affinity.PrefixTable(capacity=10)
    for i in range(30):
        table.remember([i], i % ENGINES)
    assert len(table.engines) == 10 and table.deepest([29]) == {29 % ENGINES}
    assert table.deepest([0]) is None


def test_switch_installs_on_the_client_class(monkeypatch):
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(DPLBAsyncMPClient, "get_core_engine_for_request", stock)
    monkeypatch.delenv("VLLM_TPU_DP_PREFIX_AFFINITY", raising=False)
    affinity.install_if_enabled()
    assert DPLBAsyncMPClient.get_core_engine_for_request is stock
    monkeypatch.setenv("VLLM_TPU_DP_PREFIX_AFFINITY", "1")
    affinity.install_if_enabled()
    affinity.install_if_enabled()
    patched = DPLBAsyncMPClient.get_core_engine_for_request
    assert patched is not stock and patched.stock is stock


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("VLLM_TPU_DP_PREFIX_AFFINITY_CHUNK_TOKENS", "0"),
        ("VLLM_TPU_DP_PREFIX_AFFINITY_CHUNK_TOKENS", "-64"),
        ("VLLM_TPU_DP_PREFIX_AFFINITY_TABLE_SIZE", "0"),
        ("VLLM_TPU_DP_PREFIX_AFFINITY_TABLE_SIZE", "-1"),
        ("VLLM_TPU_DP_PREFIX_AFFINITY_LOAD_MARGIN", "-1"),
    ],
)
def test_knob_that_cannot_work_raises_at_install(monkeypatch, knob, value):
    """A value that would fail every routed request, or silently turn the
    choice off, raises before the client class is patched."""
    stock = DPLBAsyncMPClient.get_core_engine_for_request
    monkeypatch.setattr(DPLBAsyncMPClient, "get_core_engine_for_request", stock)
    monkeypatch.setenv("VLLM_TPU_DP_PREFIX_AFFINITY", "1")
    monkeypatch.setenv(knob, value)
    with pytest.raises(ValueError, match=knob):
        affinity.install_if_enabled()
    assert DPLBAsyncMPClient.get_core_engine_for_request is stock
