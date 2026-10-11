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
"""The driver status endpoint and the follower rules, in one process."""

import json
import socket
import threading
import time

import pytest
from plan_fixtures import Data, build
from runtime_helpers import free_port
from scenario.runtime import control, exits, net
from scenario.runtime.control import (
    Follower,
    FollowerRules,
    Status,
    StatusServer,
    Verdict,
)
from scenario.runtime.exits import RunError
from scenario.runtime.placement import Placement

SHA = "0123456789ab"


@pytest.fixture
def port(monkeypatch: pytest.MonkeyPatch) -> int:
    found = free_port()
    monkeypatch.setattr(control, "STATUS_PORT", found)
    return found


@pytest.fixture
def server(port: int) -> StatusServer:
    started = StatusServer(SHA, ["server/1", "server/2"]).start()
    yield started
    started.stop()


def get(port: int, path: str) -> tuple[int, dict]:
    found = net.http_get(f"http://127.0.0.1:{port}{path}")
    assert found is not None
    return found[0], json.loads(found[1])


def test_the_status_names_the_plan_state_and_exit(
    server: StatusServer, port: int
) -> None:
    assert get(port, "/status?part=server/1") == (
        200,
        {"plan": SHA, "state": "starting", "exit": None},
    )
    server.set("done", 1)
    assert get(port, "/status?part=server/2")[1] == {
        "plan": SHA,
        "state": "done",
        "exit": 1,
    }
    assert get(port, "/other")[0] == 404


def test_the_driver_notes_who_has_asked(server: StatusServer, port: int) -> None:
    assert server.silent() == ["server/1", "server/2"]
    get(port, "/status?part=server/2")
    assert server.silent() == ["server/1"]


def test_a_taken_port_is_exit_3(port: int) -> None:
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", port))
        taken.listen()
        with pytest.raises(RunError) as error:
            StatusServer(SHA, []).start()
    assert error.value.code == exits.BRINGUP


def test_the_driver_lingers_until_every_follower_saw_the_end(
    server: StatusServer, port: int
) -> None:
    get(port, "/status?part=server/1")
    get(port, "/status?part=server/2")
    server.set("done", 0)
    # Asking before the end does not count.
    assert server.linger(0.05) == ["server/1", "server/2"]

    def ask_later() -> None:
        time.sleep(0.1)
        get(port, "/status?part=server/1")
        get(port, "/status?part=server/2")

    threading.Thread(target=ask_later).start()
    started = time.monotonic()
    assert server.linger(5) == []
    assert time.monotonic() - started < 2


def test_the_driver_does_not_linger_for_followers_never_heard_from(
    server: StatusServer, port: int
) -> None:
    get(port, "/status?part=server/2")
    server.set("done", 2)
    started = time.monotonic()
    assert server.linger(0.3) == ["server/2"]
    get(port, "/status?part=server/2")
    assert server.linger(5) == []
    assert time.monotonic() - started < 2


def test_a_status_is_parsed_or_refused() -> None:
    assert Status.parse(b'{"plan": "x", "state": "serving", "exit": null}') == Status(
        "x", "serving", None
    )
    for body in (b"<html>", b"[]", b'{"plan": "x", "state": "idle", "exit": 0}'):
        assert Status.parse(body) is None


STARTING = Status(SHA, "starting")
SERVING = Status(SHA, "serving")
DONE = Status(SHA, "done", 0)
ABORTED = Status(SHA, "aborted", 143)


@pytest.mark.parametrize(
    ("answers", "verdict"),
    [
        # (seconds since start, answer); the verdict after the last one.
        ([(0, STARTING), (2, SERVING), (4, DONE)], exits.OK),
        ([(0, SERVING), (2, ABORTED)], exits.DRIVER_LOST),
        ([(0, Status("other", "starting"))], exits.USAGE),
        # No answer for 20 s straight after a first one: lost.
        ([(0, SERVING), (2, None), (21, None)], None),
        ([(0, SERVING), (2, None), (22, None)], exits.DRIVER_LOST),
        # A gap that heals starts the count again.
        ([(0, SERVING), (2, None), (15, SERVING), (16, None), (35, None)], None),
        # Never an answer: give up after 1800 s, not 20.
        ([(0, None), (1799, None)], None),
        ([(0, None), (1800, None)], exits.BRINGUP),
    ],
)
def test_the_follower_rules(
    answers: list[tuple[float, Status | None]], verdict: int | None
) -> None:
    rules = FollowerRules(SHA, 0.0)
    found: Verdict | None = None
    for now, status in answers:
        found = rules.observe(status, now)
    assert (found.code if found else None) == verdict


@pytest.mark.parametrize(
    ("seen", "answer", "code"),
    [
        (STARTING, STARTING, exits.NOT_READY),
        (STARTING, None, exits.NOT_READY),
        (SERVING, SERVING, exits.SERVER_DIED),
        (SERVING, None, exits.SERVER_DIED),
        (STARTING, SERVING, exits.SERVER_DIED),
        # The driver's end took the server with it.
        (SERVING, DONE, exits.OK),
        (SERVING, ABORTED, exits.DRIVER_LOST),
    ],
)
def test_a_followers_own_server_exiting(
    seen: Status, answer: Status | None, code: int
) -> None:
    rules = FollowerRules(SHA, 0.0)
    rules.observe(seen, 0.0)
    assert rules.server_exited(answer, 1.0).code == code


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "POLL_S", 0.02)
    monkeypatch.setattr(control, "GONE_S", 0.3)


@pytest.mark.usefixtures("quick")
def test_a_follower_leaves_when_the_driver_is_done(
    server: StatusServer, port: int
) -> None:
    lines: list[str] = []
    url = control.status_url("127.0.0.1", "server/1")
    follower = Follower(url, SHA, say=lines.append).start()
    try:
        assert follower.wait(0.2) is None
        server.set("serving")
        time.sleep(0.1)
        server.set("done", 0)
        verdict = follower.wait(5)
    finally:
        follower.stop()
    assert verdict is not None and verdict.code == exits.OK
    assert follower.seen_done
    assert lines[1:] == [
        "The driver answered: starting",
        "The driver is serving",
        "Leaving with exit 0: the driver is done (exit 0)",
    ]
    assert server.silent() == ["server/2"]


@pytest.mark.usefixtures("quick")
def test_a_follower_leaves_when_the_driver_is_lost(
    server: StatusServer, port: int
) -> None:
    url = control.status_url("127.0.0.1", "server/1")
    follower = Follower(url, SHA, say=lambda line: None).start()
    try:
        deadline = time.monotonic() + 5
        while not follower.rules.contacted and time.monotonic() < deadline:
            time.sleep(0.01)
        server.stop()
        verdict = follower.wait(5)
    finally:
        follower.stop()
    assert verdict is not None and verdict.code == exits.DRIVER_LOST


@pytest.mark.usefixtures("quick")
def test_a_follower_asks_once_more_when_its_server_exits(
    server: StatusServer, port: int
) -> None:
    url = control.status_url("127.0.0.1", "server/1")
    follower = Follower(url, SHA, say=lambda line: None)
    server.set("serving")
    assert follower.server_exited().code == exits.SERVER_DIED
    # The first verdict stays.
    server.set("done", 0)
    assert follower.server_exited().code == exits.SERVER_DIED


def test_the_parts_and_the_driver(variants: dict[str, Data]) -> None:
    ray = build(variants["single"])
    assert control.parts(ray) == ["server/0", "server/1"]
    assert control.followers(ray) == ["server/1"]
    pd = build(variants["pd"])
    assert control.driver_part(pd) == "client/0"
    assert control.followers(pd) == ["prefill/0", "decode/0"]
    k3 = build(variants["k3_pd"])
    assert control.driver_part(k3) == "prefill/0"
    assert control.followers(k3)[:2] == ["prefill/1", "prefill/2"]


def test_where_followers_find_the_driver(variants: dict[str, Data]) -> None:
    ray = build(variants["single"])
    worker = Placement("server", 1, head_address="h0")
    assert control.driver_address(ray, worker) == "h0"
    pd = build(variants["pd"])
    hosts = {"prefill": ("p0",), "decode": ("d0",), "client": ("c0",)}
    decode = Placement("decode", 0, head_address="d0", role_hosts=hosts)
    assert control.driver_address(pd, decode) == "c0"
    for plan, alone in ((ray, Placement("server", 1)), (pd, Placement("decode", 0))):
        with pytest.raises(RunError) as error:
            control.driver_address(plan, alone)
        assert error.value.code == exits.USAGE
    assert "SCENARIO_ROLE_HOSTS" in error.value.message
