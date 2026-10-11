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
"""Names, ports and HTTP: bounded resolution, probes and the readiness wait.

HTTP goes through http.client, not urllib, so a proxy set in the
environment never sits between a host and its own server.
"""

import http.client
import ipaddress
import socket
import time
import urllib.parse
from collections.abc import Callable
from typing import Any, Literal, Protocol

from scenario.runtime.exits import BRINGUP, RunError

# A headless service's names are published some time after the pod starts.
RESOLVE_TIMEOUT_S = 1800
RESOLVE_INTERVAL_S = 5
CONNECT_TIMEOUT_S = 2
READ_TIMEOUT_S = 5
# The readiness wait reports progress this often.
PROGRESS_EVERY_S = 60

Resolver = Callable[..., list[tuple[Any, ...]]]


def is_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text)
    except ValueError:
        return False
    return True


def resolve_ipv4(
    name: str,
    *,
    timeout_s: float = RESOLVE_TIMEOUT_S,
    interval_s: float = RESOLVE_INTERVAL_S,
    getaddrinfo: Resolver = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """`name` as an IPv4 address, retrying until `timeout_s` has passed.

    Raises:
        RunError: exit 3, the name did not resolve in time.
    """
    if is_ipv4(name):
        return name
    deadline = clock() + timeout_s
    while True:
        try:
            infos = getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM)
            if infos:
                return str(infos[0][4][0])
            problem = "no IPv4 address"
        except OSError as error:
            problem = str(error)
        if clock() >= deadline:
            raise RunError(
                BRINGUP,
                f"{name} did not resolve to IPv4 within {timeout_s:.0f}s: {problem}",
            )
        sleep(interval_s)


def port_open(host: str, port: int, timeout_s: float = CONNECT_TIMEOUT_S) -> bool:
    """Whether something accepts a TCP connection on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def http_status(url: str, timeout_s: float = READ_TIMEOUT_S) -> int | None:
    """The status of GET `url`, or None when no answer came."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or parts.hostname is None:
        raise ValueError(f"not an http URL: {url}")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn = http.client.HTTPConnection(
        parts.hostname, parts.port or 80, timeout=min(CONNECT_TIMEOUT_S, timeout_s)
    )
    try:
        conn.connect()
        assert conn.sock is not None
        conn.sock.settimeout(timeout_s)
        conn.request("GET", path)
        response = conn.getresponse()
        response.read()
        return response.status
    except (OSError, http.client.HTTPException):
        return None
    finally:
        conn.close()


class Watched(Protocol):
    """What the readiness wait watches: a process that may die."""

    def alive(self) -> bool: ...


Readiness = Literal["ready", "died", "deadline"]


def wait_http(
    url: str,
    *,
    deadline: float,
    proc: Watched | None = None,
    interval_s: float = 1.0,
    progress: Callable[[float], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Readiness:
    """Poll GET `url` until it answers 200, `proc` dies, or `deadline` passes.

    `deadline` is on `clock`. `progress` gets the seconds waited, once per
    PROGRESS_EVERY_S.
    """
    started = clock()
    reported = started
    while True:
        if http_status(url) == 200:
            return "ready"
        if proc is not None and not proc.alive():
            return "died"
        now = clock()
        if now >= deadline:
            return "deadline"
        if progress is not None and now - reported >= PROGRESS_EVERY_S:
            reported = now
            progress(now - started)
        sleep(min(interval_s, max(deadline - now, 0.0)))
