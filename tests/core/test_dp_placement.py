"""The DP router's release rules (core/dp_placement.py) on a stand-in
load-balancing client: hold until the threshold then one per engine, the
idle-engine trigger, the deadline, abort of a held request, the quiet
release, a pinned request, a failed send, and the mode check."""
import asyncio
from collections import Counter

import pytest

from vllm_torchtpu.core import dp_placement


def _release_client(n=4, counts=None, fail_engine=None):
    """A fake DP client with what the release mode touches."""

    class Sock:

        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

    class C:

        def __init__(self):
            self.core_engines = [f"e{i}" for i in range(n)]
            self.reqs_in_flight = {}
            self.engine_inflight = Counter({f"e{i}": 8 for i in range(n)})
            self.engine_inflight["e2"] = 7
            self.lb_engines = counts if counts is not None else [
                [1, 8, 0.5] for _ in range(n)
            ]
            self.client_count = 1
            self.engines_running = True
            self.current_wave = 0
            self.client_index = 0
            self.first_req_send_socket = Sock()
            self.sent = []

        def _ensure_stats_update_task(self):
            pass

        def _ensure_output_queue_task(self):
            pass

        def _send_input(self, kind, request, engine):
            if engine == fail_engine:
                raise RuntimeError(f"{engine} is down")
            self.sent.append((request.request_id, engine))

            async def done():
                pass

            return done()

        async def abort_requests_async(self, ids):
            self.aborted = list(ids)

    return C


class R:

    def __init__(self, i, rank=None):
        self.request_id = str(i)
        self.current_wave = None
        self.client_index = None
        self.data_parallel_rank = rank


def test_release_holds_until_threshold_then_one_per_engine():
    C = _release_client()
    dp_placement.install_release(C,
                                 threshold=4,
                                 deadline_s=30.0,
                                 idle_trigger=True,
                                 quiet_s=0.0)
    c = C()

    async def go():
        for i in range(3):
            await c.add_request_async(R(i))
        assert c.sent == [] and len(
            c._release_buf) == 3  # held: no engine idle, below the threshold
        await c.add_request_async(R(3))
        return c.sent

    sent = asyncio.run(go())
    assert [e for _, e in sent
            ] == ["e2", "e3", "e0",
                  "e1"]  # the fewest in flight first, then cyclic, each once
    assert c.engine_inflight["e2"] == 8 and c.reqs_in_flight["0"] == "e2"
    assert c._release_buf == []


def test_release_idle_engine_and_deadline_and_abort():
    C = _release_client(
        counts=[[0, 0, 0.0], [1, 8, 0.5], [1, 8, 0.5], [1, 8, 0.5]])
    dp_placement.install_release(C,
                                 threshold=4,
                                 deadline_s=0.05,
                                 idle_trigger=True,
                                 quiet_s=0.0)
    c = C()

    async def go():
        await c.add_request_async(
            R(0))  # engine 0 idle: released at once, to the least in flight
        first = list(c.sent)
        c.lb_engines[0] = [1, 8, 0.5]
        await c.add_request_async(R(1))
        held = len(c._release_buf)
        await c.abort_requests_async(["1"])
        dropped = len(c._release_buf)
        await c.add_request_async(R(2))
        await asyncio.sleep(0.12)  # the poller releases it by the deadline
        return first, held, dropped, c.sent

    first, held, dropped, sent = asyncio.run(go())
    assert first == [("0", "e2")] and held == 1 and dropped == 0
    assert ("2", "e3") in sent  # all at 8 now: the cyclic next after e2


def test_release_quiet_trigger():
    """A partial buffer releases once arrivals have stopped for the quiet
    time, without an idle engine or the deadline."""
    C = _release_client()
    dp_placement.install_release(C,
                                 threshold=4,
                                 deadline_s=30.0,
                                 idle_trigger=True,
                                 quiet_s=0.05)
    c = C()

    async def go():
        await c.add_request_async(R(0))
        await c.add_request_async(R(1))
        held = len(c._release_buf)
        await asyncio.sleep(0.12)
        return held, c.sent

    held, sent = asyncio.run(go())
    assert held == 2 and [r for r, _ in sent] == ["0", "1"]


def test_pinned_request_keeps_its_engine():
    C = _release_client()
    dp_placement.install_release(C,
                                 threshold=2,
                                 deadline_s=30.0,
                                 idle_trigger=False,
                                 quiet_s=0.0)
    c = C()

    async def go():
        await c.add_request_async(R(0, rank=1))
        await c.add_request_async(R(1))
        return c.sent

    sent = asyncio.run(go())
    assert sent[0] == ("0", "e1")  # pinned to rank 1, not the least loaded


def test_failed_send_keeps_the_counts_and_the_rest():
    C = _release_client(fail_engine="e2")
    dp_placement.install_release(C,
                                 threshold=2,
                                 deadline_s=30.0,
                                 idle_trigger=False,
                                 quiet_s=0.0)
    c = C()

    async def go():
        await c.add_request_async(R(0))
        with pytest.raises(RuntimeError):
            await c.add_request_async(R(1))  # the first goes to e2 and fails
        return c.sent, len(c._release_buf)

    sent, held = asyncio.run(go())
    assert sent == [] and held == 1  # the failed one dropped, the other held
    assert c.engine_inflight["e2"] == 7 and "0" not in c.reqs_in_flight


def test_install_if_enabled_checks_the_mode(monkeypatch):
    monkeypatch.setenv("DP_PLACEMENT", "relase")
    with pytest.raises(ValueError):
        dp_placement.install_if_enabled()
    monkeypatch.setenv("DP_PLACEMENT", "")
    dp_placement.install_if_enabled()  # off: nothing to install


def test_a_cancel_during_a_release_is_honoured_once():
    """A request cancelled while the release loop is sending another one is
    never sent, and the requests still held are sent once, not twice."""
    C = _release_client()
    started, finish = asyncio.Event(), asyncio.Event()

    def send_input(self, kind, request, engine):
        self.sent.append((request.request_id, engine))
        first = len(self.sent) == 1

        async def done():
            if first:
                started.set()
                await finish.wait()

        return done()

    C._send_input = send_input
    dp_placement.install_release(C,
                                 threshold=2,
                                 deadline_s=100.0,
                                 idle_trigger=False,
                                 quiet_s=0.0)
    c = C()

    async def go():
        await c.add_request_async(R(0))  # held: below the threshold
        # The threshold: the release starts and its first send blocks.
        releasing = asyncio.create_task(c.add_request_async(R(1)))
        await asyncio.wait_for(started.wait(), 5)
        await c.abort_requests_async(["1"])  # cancelled while still held
        finish.set()
        await releasing
        held = list(c._release_buf)
        await c.add_request_async(R(2))
        await c.add_request_async(R(3))
        return held, [rid for rid, _ in c.sent]

    held, sent = asyncio.run(go())
    assert held == [] and sent == ["0", "2", "3"]
