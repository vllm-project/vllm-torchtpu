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
"""The DP router's release placement (core/dp_placement.py) on vLLM's real
DPLBAsyncMPClient with its engine sockets replaced by a recorder: hold until
the threshold then one per engine, the idle-engine trigger, the deadline,
abort of a held request, the quiet release, a pinned request, streaming-input
and late-interaction requests that keep vLLM's routing, a failed send
delivered to its own request, a cancelled add that leaves the other sends
alone, a dead engine, and the mode check. An add returns once its request has
been sent, so held requests are added as tasks."""

import asyncio
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_torchtpu.core import dp_placement


@dataclass
class Request:
    """The EngineCoreRequest fields the router reads and writes."""

    request_id: str
    data_parallel_rank: int | None = None
    resumable: bool = False
    pooling_params: PoolingParams | None = None
    current_wave: int | None = None
    client_index: int | None = None


def _release_client(
    num_engines=4,
    lb_engines=None,
    failing_engine=None,
    threshold=4,
    deadline_s=30.0,
    idle_trigger=False,
    quiet_s=0.0,
):
    """A DPLBAsyncMPClient (a fresh subclass, so the patch stays local) with
    the state vLLM's __init__ sets and recorded sends instead of sockets."""

    class RecordingClient(DPLBAsyncMPClient):
        def _send_input(self, request_type, request, engine=None):
            if engine == failing_engine:
                raise RuntimeError(f"{engine!r} is down")
            if request_type == EngineCoreRequestType.ADD:
                self.sent.append((request.request_id, engine))
            else:
                self.aborted.append((list(request), engine))

            async def sent():
                pass

            return sent()

        def _ensure_stats_update_task(self):
            pass

        def _ensure_output_queue_task(self):
            pass

    dp_placement.install_release(
        RecordingClient,
        threshold=threshold,
        deadline_s=deadline_s,
        idle_trigger=idle_trigger,
        quiet_s=quiet_s,
    )
    client = object.__new__(RecordingClient)
    client.core_engines = [f"e{i}".encode() for i in range(num_engines)]
    client.client_count = 1
    client.client_index = 0
    client.current_wave = 0
    client.engines_running = True
    client.eng_start_index = 0
    client.reqs_in_flight = {}
    client.streaming_reqs = set()
    client.engine_inflight = Counter({e: 8 for e in client.core_engines})
    client.engine_inflight[b"e2"] = 7
    client.lb_engines = (
        lb_engines
        if lb_engines is not None
        else [[1, 8, 0.5] for _ in range(num_engines)]
    )
    client.resources = SimpleNamespace(engine_dead=False)
    client.sent = []
    client.aborted = []
    return client


def _add(client, request):
    """A held request's add waits for its send: adds run as tasks."""
    return asyncio.get_running_loop().create_task(client.add_request_async(request))


def test_release_holds_until_threshold_then_one_per_engine():
    client = _release_client(idle_trigger=True)

    async def run():
        held = [_add(client, Request(str(i))) for i in range(3)]
        await asyncio.sleep(0.01)
        # Held: no engine idle, below the threshold.
        assert client.sent == [] and len(client._dp_release_held) == 3
        assert not any(task.done() for task in held)
        await client.add_request_async(Request("3"))
        await asyncio.gather(*held)  # Every add returns once its request is sent.

    asyncio.run(run())
    # The fewest in flight first, then cyclic, each engine once.
    assert [engine for _, engine in client.sent] == [b"e2", b"e3", b"e0", b"e1"]
    assert client.engine_inflight[b"e2"] == 8
    assert client.reqs_in_flight["0"] == b"e2"
    assert [counts[0] for counts in client.lb_engines] == [2, 2, 2, 2]
    assert client._dp_release_held == []


def test_release_idle_engine_and_deadline_and_abort():
    client = _release_client(
        lb_engines=[[0, 0, 0.0], [1, 8, 0.5], [1, 8, 0.5], [1, 8, 0.5]],
        deadline_s=0.05,
        idle_trigger=True,
    )

    async def run():
        # Engine 0 idle: released at once, to the least in flight.
        await client.add_request_async(Request("0"))
        first = list(client.sent)
        client.lb_engines[0] = [1, 8, 0.5]
        aborted_add = _add(client, Request("1"))
        await asyncio.sleep(0.01)
        held = len(client._dp_release_held)
        await client.abort_requests_async(["1"])
        with pytest.raises(asyncio.CancelledError):
            await aborted_add  # The aborted request's add returns with the cancel.
        # The poller releases the next one by the deadline.
        await client.add_request_async(Request("2"))
        return first, held

    first, held = asyncio.run(run())
    assert first == [("0", b"e2")] and held == 1
    assert ("2", b"e3") in client.sent  # All at 8 now: the cyclic next after e2.
    assert "1" not in client.reqs_in_flight and client.aborted == []


def test_release_quiet_trigger():
    """A partial group releases once arrivals have stopped for the quiet
    time, without an idle engine or the deadline."""
    client = _release_client(idle_trigger=True, quiet_s=0.05)

    async def run():
        adds = [_add(client, Request("0")), _add(client, Request("1"))]
        await asyncio.sleep(0.01)
        held = len(client._dp_release_held)
        await asyncio.wait_for(asyncio.gather(*adds), 2)
        return held

    assert asyncio.run(run()) == 2
    assert [request_id for request_id, _ in client.sent] == ["0", "1"]


def test_pinned_request_is_held_and_keeps_its_engine():
    client = _release_client(threshold=2)

    async def run():
        pinned = _add(client, Request("0", data_parallel_rank=1))
        await asyncio.sleep(0.01)
        assert client.sent == []  # Pinned requests are held with the rest.
        await client.add_request_async(Request("1"))
        await pinned

    asyncio.run(run())
    # Pinned to rank 1, not the least loaded; the other goes to e2.
    assert client.sent == [("0", b"e1"), ("1", b"e2")]
    assert client.reqs_in_flight == {"0": b"e1", "1": b"e2"}
    # As in vLLM's router, only the load-balanced pick raises the local counts.
    assert [counts[0] for counts in client.lb_engines] == [1, 1, 2, 1]


def test_streaming_session_keeps_vllm_routing_and_affinity():
    """A resumable (streaming-input) request is sent at once through vLLM's
    router, and its later chunks, including the final non-resumable one, go to
    the same engine without being held."""
    client = _release_client()

    async def run():
        await client.add_request_async(Request("s", resumable=True))
        engine = client.reqs_in_flight["s"]
        assert client.sent == [("s", engine)] and "s" in client.streaming_reqs
        await client.add_request_async(Request("s", resumable=True))
        await client.add_request_async(Request("s"))  # End of stream.
        return engine

    engine = asyncio.run(run())
    assert client.sent == [("s", engine)] * 3
    assert "s" not in client.streaming_reqs
    assert "_dp_release_held" not in vars(client)  # Never held.


def test_open_streaming_session_is_not_held():
    """A session whose earlier chunk already finished (no longer in flight)
    still bypasses the hold while vLLM tracks it as an open session."""
    client = _release_client()
    client.streaming_reqs.add("s")

    async def run():
        await client.add_request_async(Request("s"))

    asyncio.run(run())
    assert len(client.sent) == 1 and "_dp_release_held" not in vars(client)


def test_late_interaction_request_keeps_vllm_engine():
    client = _release_client()
    pooling_params = PoolingParams(
        late_interaction_params=LateInteractionParams(mode="cache_query", query_key="q")
    )

    async def run():
        await client.add_request_async(Request("0", pooling_params=pooling_params))

    asyncio.run(run())
    from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

    index = get_late_interaction_engine_index(pooling_params, 4)
    assert client.sent == [("0", client.core_engines[index])]


def test_failed_send_reaches_its_own_request_and_the_rest_go_out():
    """The first request's send fails: its own add raises, the request that
    triggered the release is sent, and the counts claim nothing for the
    failed one."""
    client = _release_client(failing_engine=b"e2", threshold=2)

    async def run():
        failing = _add(client, Request("0"))
        await asyncio.sleep(0.01)
        await client.add_request_async(Request("1"))  # The first goes to e2.
        with pytest.raises(RuntimeError, match="is down"):
            await failing

    asyncio.run(run())
    assert client.sent == [("1", b"e3")] and client._dp_release_held == []
    assert client.engine_inflight[b"e2"] == 7 and "0" not in client.reqs_in_flight
    assert client.reqs_in_flight["1"] == b"e3"
    assert client.lb_engines[2][0] == 1


def test_a_dead_engine_fails_every_held_request():
    client = _release_client()

    async def run():
        held = [_add(client, Request(str(i))) for i in range(2)]
        await asyncio.sleep(0.01)
        client.resources.engine_dead = True
        with pytest.raises(EngineDeadError):
            await client.add_request_async(Request("2"))
        for task in held:
            with pytest.raises(EngineDeadError):
                await task

    asyncio.run(run())
    assert client.sent == [] and client._dp_release_held == []


def test_install_if_enabled_checks_the_mode(monkeypatch):
    monkeypatch.setenv("DP_PLACEMENT", "relase")
    with pytest.raises(ValueError):
        dp_placement.install_if_enabled()
    monkeypatch.setenv("DP_PLACEMENT", "")
    dp_placement.install_if_enabled()  # Off: nothing to install.
    assert not getattr(DPLBAsyncMPClient, "_tpu_dp_release_patch", False)


def _block_first_send(client):
    """Makes the client's first send block until `finish` is set."""
    started, finish = asyncio.Event(), asyncio.Event()

    def send_input(request_type, request, engine=None):
        client.sent.append((request.request_id, engine))
        is_first = len(client.sent) == 1

        async def sent():
            if is_first:
                started.set()
                await finish.wait()

        return sent()

    client._send_input = send_input
    return started, finish


def test_a_cancel_during_a_release_is_honoured_once():
    """A request aborted while the release loop is sending another one is
    never sent, and the requests still held are sent once, not twice."""
    client = _release_client(threshold=2, deadline_s=100.0)

    async def run():
        started, finish = _block_first_send(client)
        first = _add(client, Request("0"))  # Held: below the threshold.
        await asyncio.sleep(0.01)
        # The threshold: the release starts and its first send blocks.
        releasing = _add(client, Request("1"))
        await asyncio.wait_for(started.wait(), 5)
        await client.abort_requests_async(["1"])  # Aborted while still held.
        with pytest.raises(asyncio.CancelledError):
            await releasing
        finish.set()
        await first  # The blocked send completes and its add returns.
        held = list(client._dp_release_held)
        await asyncio.gather(_add(client, Request("2")), _add(client, Request("3")))
        return held

    held = asyncio.run(run())
    assert held == []
    assert [request_id for request_id, _ in client.sent] == ["0", "2", "3"]


def test_a_cancelled_add_leaves_the_other_sends_alone():
    """The client of the request that triggered the release leaves (its add
    is cancelled) while another request's send is in flight: that send
    completes and its add returns; the leaver is never sent."""
    client = _release_client(threshold=2, deadline_s=100.0)

    async def run():
        started, finish = _block_first_send(client)
        first = _add(client, Request("0"))
        await asyncio.sleep(0.01)
        leaver = _add(client, Request("1"))
        await asyncio.wait_for(started.wait(), 5)
        leaver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaver
        finish.set()
        await asyncio.wait_for(first, 5)
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert [request_id for request_id, _ in client.sent] == ["0"]
    assert client._dp_release_held == []
    assert client.reqs_in_flight["0"] == b"e2" and "1" not in client.reqs_in_flight
