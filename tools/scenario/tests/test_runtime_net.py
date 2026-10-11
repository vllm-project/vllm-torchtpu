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
"""Names, ports and the readiness wait, against local sockets only."""

import http.server
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from scenario.runtime import exits, net
from scenario.runtime.exits import RunError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Health(http.server.BaseHTTPRequestHandler):
    status = 503

    def do_GET(self) -> None:
        code = type(self).status if self.path == "/health" else 404
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, type[Health]]]:
    handler = type("Handler", (Health,), {"status": 503})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_an_address_needs_no_lookup() -> None:
    def never(*args: Any) -> list[Any]:
        raise AssertionError("looked up")

    assert net.resolve_ipv4("10.1.2.3", getaddrinfo=never) == "10.1.2.3"


def test_resolution_retries_until_the_name_appears() -> None:
    clock = FakeClock()
    calls = []

    def lookup(name: str, *args: Any) -> list[Any]:
        calls.append(name)
        if len(calls) < 3:
            raise socket.gaierror("not yet")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 0))]

    found = net.resolve_ipv4(
        "pod-0.wl", getaddrinfo=lookup, clock=clock, sleep=clock.sleep
    )
    assert found == "10.0.0.7"
    assert calls == ["pod-0.wl"] * 3
    assert clock.now == 2 * net.RESOLVE_INTERVAL_S


def test_resolution_gives_up_with_a_bring_up_failure() -> None:
    clock = FakeClock()

    def lookup(*args: Any) -> list[Any]:
        raise socket.gaierror("Name or service not known")

    with pytest.raises(RunError) as raised:
        net.resolve_ipv4(
            "gone.wl", timeout_s=30, getaddrinfo=lookup, clock=clock, sleep=clock.sleep
        )
    assert raised.value.code == exits.BRINGUP
    assert "gone.wl did not resolve to IPv4 within 30s" in raised.value.message
    assert clock.now >= 30


def test_port_probe() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert net.port_open("127.0.0.1", port)
    assert not net.port_open("127.0.0.1", free_port())


def test_http_status(server: tuple[str, type[Health]]) -> None:
    url, handler = server
    assert net.http_status(f"{url}/health") == 503
    handler.status = 200
    assert net.http_status(f"{url}/health") == 200
    assert net.http_status(f"{url}/other") == 404
    assert net.http_status(f"http://127.0.0.1:{free_port()}/health") is None
    with pytest.raises(ValueError):
        net.http_status("https://example.invalid/health")


def test_http_ignores_proxy_settings(
    server: tuple[str, type[Health]], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, handler = server
    handler.status = 200
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{free_port()}")
    monkeypatch.setenv("no_proxy", "")
    assert net.http_status(f"{url}/health") == 200


class Dies:
    def __init__(self, after: int) -> None:
        self.polls = after

    def alive(self) -> bool:
        self.polls -= 1
        return self.polls > 0


def test_wait_returns_once_ready(server: tuple[str, type[Health]]) -> None:
    url, handler = server
    clock = FakeClock()

    def flip(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 3:
            handler.status = 200

    outcome = net.wait_http(
        f"{url}/health", deadline=100, clock=clock, sleep=flip, proc=Dies(after=100)
    )
    assert outcome == "ready"
    assert clock.now == 3


def test_wait_notices_a_dead_process(server: tuple[str, type[Health]]) -> None:
    url, _ = server
    clock = FakeClock()
    outcome = net.wait_http(
        f"{url}/health",
        deadline=100,
        proc=Dies(after=3),
        clock=clock,
        sleep=clock.sleep,
    )
    assert outcome == "died"
    assert clock.now == 2


def test_wait_runs_out_and_reports_progress(server: tuple[str, type[Health]]) -> None:
    url, _ = server
    clock = FakeClock()
    seen: list[float] = []
    outcome = net.wait_http(
        f"{url}/health",
        deadline=150,
        interval_s=10,
        progress=seen.append,
        clock=clock,
        sleep=clock.sleep,
    )
    assert outcome == "deadline"
    assert clock.now == 150
    assert seen == [60, 120]


def test_exit_code_precedence() -> None:
    assert exits.worst() == exits.OK
    assert exits.worst(exits.AFTER_TASK, exits.TASK_FAILED) == exits.TASK_FAILED
    assert exits.worst(exits.TASK_FAILED, exits.DRIVER_LOST) == exits.DRIVER_LOST
    assert exits.worst(exits.DRIVER_LOST, exits.BRINGUP) == exits.BRINGUP
    assert exits.worst(exits.BRINGUP, exits.SERVER_DIED) == exits.SERVER_DIED
    assert exits.worst(exits.NOT_READY, exits.SERVER_DIED) == exits.NOT_READY
    assert exits.worst(exits.SERVER_DIED, exits.SIGTERM) == exits.SIGTERM
    assert exits.for_signal(15) == exits.SIGTERM
    assert set(exits.MEANINGS) == {0, 1, 2, 3, 4, 5, 6, 7, 130, 143}
