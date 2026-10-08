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
import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from vllm.v1.engine import core_client
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_torchtpu.core import dp_placement
from vllm_torchtpu.core.dp_release_client import ReleaseDPLBAsyncMPClient

pytestmark = pytest.mark.cpu_test


class _Socket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


def _client(n=4, low=2, counts=None, fail_engine=None):
    """The router over vLLM's own send path, with the engines' sockets
    replaced by a recorder. Every engine has 8 in flight except `low`."""
    c = ReleaseDPLBAsyncMPClient.__new__(ReleaseDPLBAsyncMPClient)
    c.core_engines = [f"e{i}".encode() for i in range(n)]
    c.reqs_in_flight = {}
    c.engine_inflight = Counter({e: 8 for e in c.core_engines})
    c.engine_inflight[c.core_engines[low]] = 7
    c.lb_engines = counts if counts is not None else [[1, 8, 0.5] for _ in range(n)]
    c.client_count = 1
    c.client_index = 0
    c.current_wave = 0
    c.engines_running = True
    c.first_req_send_socket = _Socket()
    c.resources = SimpleNamespace(engine_dead=False, stats_update_task=object())
    c._held = []
    c._assigned = Counter()
    c._next_engine = 0
    c._poller = None
    c._release = None
    c.sent = []
    c._ensure_output_queue_task = lambda: None

    def send_input(request_type, request, engine):
        if engine == fail_engine:
            raise RuntimeError(f"{engine.decode()} is down")
        if request_type == core_client.EngineCoreRequestType.ADD:
            c.sent.append((request.request_id, engine.decode()))

        async def done():
            pass

        return done()

    c._send_input = send_input
    return c


def _request(i, rank=None):
    return SimpleNamespace(
        request_id=str(i),
        current_wave=None,
        client_index=None,
        data_parallel_rank=rank,
        pooling_params=None,
    )


def _add(c, r):
    """An add returns once its request is sent, so held adds run as tasks."""
    return asyncio.get_running_loop().create_task(c.add_request_async(r))


def _held(c):
    return [h.request.request_id for h in c._held]


@pytest.fixture(autouse=True)
def _slow_clocks(monkeypatch):
    """Only the thresholds and the idle trigger release, unless a test
    shortens a clock."""
    monkeypatch.setattr(ReleaseDPLBAsyncMPClient, "DEADLINE_S", 30.0)
    monkeypatch.setattr(ReleaseDPLBAsyncMPClient, "QUIET_S", 30.0)


def test_holds_until_threshold_then_one_per_engine():
    c = _client()

    async def go():
        held = [_add(c, _request(i)) for i in range(3)]
        await asyncio.sleep(0.01)
        assert c.sent == [] and _held(c) == ["0", "1", "2"]
        assert not any(t.done() for t in held)
        await c.add_request_async(_request(3))
        await asyncio.gather(*held)

    asyncio.run(go())
    # The fewest in flight first, then cyclic, each engine once.
    assert [e for _, e in c.sent] == ["e2", "e3", "e0", "e1"]
    assert c.engine_inflight[b"e2"] == 8 and c.reqs_in_flight["0"] == b"e2"
    assert [w for w, _, _ in c.lb_engines] == [2, 2, 2, 2]
    assert c._held == []


def test_idle_engine_and_deadline_and_abort(monkeypatch):
    monkeypatch.setattr(ReleaseDPLBAsyncMPClient, "DEADLINE_S", 0.05)
    c = _client(counts=[[0, 0, 0.0], [1, 8, 0.5], [1, 8, 0.5], [1, 8, 0.5]])

    async def go():
        # Engine 0 is idle: released at once, to the least in flight.
        await c.add_request_async(_request(0))
        first = list(c.sent)
        c.lb_engines[0] = [1, 8, 0.5]
        one = _add(c, _request(1))
        await asyncio.sleep(0.01)
        held = _held(c)
        await c.abort_requests_async(["1"])
        with pytest.raises(asyncio.CancelledError):
            await one
        # The poller releases this one at the deadline.
        await c.add_request_async(_request(2))
        return first, held

    first, held = asyncio.run(go())
    assert first == [("0", "e2")] and held == ["1"]
    assert c.sent[1:] == [("2", "e3")]


def test_quiet_release(monkeypatch):
    monkeypatch.setattr(ReleaseDPLBAsyncMPClient, "QUIET_S", 0.05)
    c = _client()

    async def go():
        adds = [_add(c, _request(0)), _add(c, _request(1))]
        await asyncio.sleep(0.01)
        held = _held(c)
        await asyncio.wait_for(asyncio.gather(*adds), 2)
        return held

    assert asyncio.run(go()) == ["0", "1"]
    assert [r for r, _ in c.sent] == ["0", "1"]


def test_pinned_request_keeps_its_engine():
    c = _client(n=2, low=1)

    async def go():
        pinned = _add(c, _request(0, rank=0))
        await asyncio.sleep(0.01)
        await c.add_request_async(_request(1))
        await pinned

    asyncio.run(go())
    assert c.sent == [("0", "e0"), ("1", "e1")]


def test_failed_send_reaches_its_own_request():
    """The failed request's add raises, the rest go out, and the counts
    claim nothing for the failed one."""
    c = _client(n=2, low=1, fail_engine=b"e1")

    async def go():
        failing = _add(c, _request(0))
        await asyncio.sleep(0.01)
        await c.add_request_async(_request(1))
        with pytest.raises(RuntimeError, match="e1 is down"):
            await failing

    asyncio.run(go())
    assert c.sent == [("1", "e0")] and c._held == []
    assert c.engine_inflight[b"e1"] == 7 and "0" not in c.reqs_in_flight
    assert c.reqs_in_flight["1"] == b"e0"
    assert [w for w, _, _ in c.lb_engines] == [2, 1]


def test_dead_engine_fails_every_held_request():
    c = _client()

    async def go():
        held = [_add(c, _request(i)) for i in range(2)]
        await asyncio.sleep(0.01)
        c.resources.engine_dead = True
        with pytest.raises(EngineDeadError):
            await c.add_request_async(_request(2))
        for t in held:
            with pytest.raises(EngineDeadError):
                await t

    asyncio.run(go())
    assert c.sent == [] and c._held == []


def test_shutdown_fails_every_held_request(monkeypatch):
    monkeypatch.setattr(core_client.DPLBAsyncMPClient, "shutdown", lambda *a: None)
    c = _client()

    async def go():
        held = [_add(c, _request(i)) for i in range(2)]
        await asyncio.sleep(0.01)
        c.shutdown()
        for t in held:
            with pytest.raises(EngineDeadError):
                await t
        return c._poller

    assert asyncio.run(go()).cancelled()
    assert c.sent == [] and c._held == []


def test_install_if_enabled_selects_the_client(monkeypatch):
    monkeypatch.setattr(core_client, "DPLBAsyncMPClient", core_client.DPLBAsyncMPClient)
    monkeypatch.delenv("DP_PLACEMENT", raising=False)
    dp_placement.install_if_enabled()
    assert core_client.DPLBAsyncMPClient is not ReleaseDPLBAsyncMPClient

    monkeypatch.setenv("DP_PLACEMENT", "relase")
    with pytest.raises(ValueError, match="release"):
        dp_placement.install_if_enabled()

    monkeypatch.setenv("DP_PLACEMENT", "release")
    dp_placement.install_if_enabled()
    assert core_client.DPLBAsyncMPClient is ReleaseDPLBAsyncMPClient


def test_refuses_more_than_one_api_server():
    with pytest.raises(NotImplementedError, match="--api-server-count=1"):
        ReleaseDPLBAsyncMPClient(None, None, False, None, 2, 0)


def _block_first_send(c):
    """Make the client's first send block until `finish` is set."""
    started, finish = asyncio.Event(), asyncio.Event()

    def send_input(request_type, request, engine):
        if request_type != core_client.EngineCoreRequestType.ADD:
            return asyncio.sleep(0)
        c.sent.append((request.request_id, engine.decode()))
        first = len(c.sent) == 1

        async def done():
            if first:
                started.set()
                await finish.wait()

        return done()

    c._send_input = send_input
    return started, finish


def test_abort_during_a_release_is_honoured_once():
    """A request aborted while another is being sent is never sent, and the
    rest are sent once."""
    c = _client(n=2, low=1)
    started, finish = _block_first_send(c)

    async def go():
        first = _add(c, _request(0))
        await asyncio.sleep(0.01)
        releasing = _add(c, _request(1))
        await asyncio.wait_for(started.wait(), 5)
        await c.abort_requests_async(["1"])
        with pytest.raises(asyncio.CancelledError):
            await releasing
        finish.set()
        await first
        held = _held(c)
        await asyncio.gather(_add(c, _request(2)), _add(c, _request(3)))
        return held

    assert asyncio.run(go()) == []
    assert [rid for rid, _ in c.sent] == ["0", "2", "3"]


def test_cancelled_add_leaves_the_other_sends_alone():
    """A client that leaves while another request's send is in flight: that
    send completes, and the leaver is never sent."""
    c = _client(n=2, low=1)
    started, finish = _block_first_send(c)

    async def go():
        first = _add(c, _request(0))
        await asyncio.sleep(0.01)
        leaver = _add(c, _request(1))
        await asyncio.wait_for(started.wait(), 5)
        leaver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaver
        finish.set()
        await asyncio.wait_for(first, 5)
        await asyncio.sleep(0.01)

    asyncio.run(go())
    assert [rid for rid, _ in c.sent] == ["0"] and c._held == []
    assert c.reqs_in_flight["0"] == b"e1" and "1" not in c.reqs_in_flight
