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
"""The driver's status endpoint, and how the hosts that wait on it leave.

Every host other than the driver's is a follower: the other hosts of a
multi-host role (mp or Ray workers), and the hosts of the other roles. A
follower starts its own part at once and polls the driver's
GET /status?part=<role>/<i> every POLL_S. The answer is the driver's Plan
sha, its state (starting, serving, done, aborted) and its exit code once it
has one. The driver records when each part last asked.

A follower leaves when:

    the driver is done                          exit 0
    the driver aborted                          exit 7
    no answer for GONE_S, after a first one     exit 7: the driver was lost
    no first answer within FIRST_CONTACT_S      exit 3
    the driver runs another Plan                exit 2

GONE_S straight rather than one miss: a name-service blip looks the same as
a driver that has gone, and leaving cannot be undone. Polling every POLL_S
still notices the end within about GONE_S.

The state tells completion from abort, which a closed port cannot. A
follower exits non-zero when the driver aborted or was lost because a target
may recreate a disrupted driver alone: followers that exited 0 would never
come back, and the new driver would hold its chips through its whole
startup timeout. A non-zero exit ends the run at once instead.
"""

import http.server
import json
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from scenario.plan.types import CLIENT_ROLE, STATUS_PORT, Plan
from scenario.runtime import exits, net
from scenario.runtime.exits import RunError
from scenario.runtime.placement import ENV_HEAD_ADDRESS, ENV_ROLE_HOSTS, Placement

# Followers ask this often.
POLL_S = 2.0
# A follower that has had an answer leaves once the driver has not answered
# for this long.
GONE_S = 20.0
# A follower that never had an answer gives up after this long: the
# driver's host may start later than its own, and its name may take as long
# to resolve as any other.
FIRST_CONTACT_S = float(net.RESOLVE_TIMEOUT_S)
# A status request gives up after this long, so a driver whose address drops
# packets counts as a miss rather than a hang.
REQUEST_TIMEOUT_S = 5.0
# After its end, the driver waits at most this long for every follower to
# have asked once more.
LINGER_S = 10.0
# After an abort it waits less: its stop grace (20 s) and this fit in the
# usual 30 s a signalled host gets. Long enough for a follower whose engine
# died with the driver's to ask once more and learn of the abort.
ABORT_LINGER_S = 3.0

State = Literal["starting", "serving", "done", "aborted"]
STATES: tuple[State, ...] = ("starting", "serving", "done", "aborted")


@dataclass(frozen=True)
class Status:
    """The driver's answer: its Plan, its state, and its exit code once known."""

    plan: str
    state: State
    exit: int | None = None

    def to_data(self) -> dict[str, Any]:
        return {"plan": self.plan, "state": self.state, "exit": self.exit}

    @staticmethod
    def parse(body: bytes) -> "Status | None":
        """The Status in a response body, or None when it is something else."""
        try:
            data = json.loads(body)
        except ValueError:
            return None
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("plan"), str)
            or data.get("state") not in STATES
            or not isinstance(data.get("exit"), int | None)
        ):
            return None
        return Status(data["plan"], data["state"], data["exit"])


def part_name(role: str, index: int) -> str:
    return f"{role}/{index}"


def driver_part(plan: Plan) -> str:
    """The part that drives: the driver role's first host, or the client."""
    return part_name(plan.driver.role or CLIENT_ROLE, 0)


def parts(plan: Plan) -> list[str]:
    """Every part of the Plan: each host of each role, then the client."""
    found = [
        part_name(role.name, index)
        for role in plan.roles
        for index in range(role.hosts)
    ]
    if plan.driver.chipless:
        found.append(part_name(CLIENT_ROLE, 0))
    return found


def followers(plan: Plan) -> list[str]:
    """The parts that wait on the driver."""
    driver = driver_part(plan)
    return [part for part in parts(plan) if part != driver]


def driver_address(plan: Plan, placement: Placement) -> str:
    """Where a follower finds the driver.

    The first host of the driver's role in the role hosts; for a plan of one
    role, the head address will do.

    Raises:
        RunError: exit 2, the placement names neither.
    """
    role = plan.driver.role or CLIENT_ROLE
    hosts = placement.role_hosts.get(role)
    if hosts:
        return hosts[0]
    if len(plan.roles) == 1 and not plan.driver.chipless and placement.head_address:
        return placement.head_address
    if len(plan.roles) == 1 and not plan.driver.chipless:
        need = f"set {ENV_HEAD_ADDRESS} or pass --head-address"
    else:
        need = f"set {ENV_ROLE_HOSTS} or pass --role-hosts"
    raise RunError(
        exits.USAGE,
        f"{placement.part} waits on the driver {role}/0 at an address: {need}",
    )


def status_url(address: str, part: str) -> str:
    query = urllib.parse.urlencode({"part": part})
    return f"http://{address}:{STATUS_PORT}/status?{query}"


class StatusServer:
    """The driver's GET /status, for the hosts that follow it."""

    def __init__(
        self,
        plan_sha: str,
        followers: Iterable[str],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.plan_sha = plan_sha
        self.followers = tuple(followers)
        self.port = STATUS_PORT
        self._clock = clock
        self._cond = threading.Condition()
        self._state: State = "starting"
        self._exit: int | None = None
        self._changed = clock()
        self._polls: dict[str, float] = {}
        self._httpd: http.server.ThreadingHTTPServer | None = None

    def start(self) -> "StatusServer":
        """Serve on every address.

        Raises:
            RunError: exit 3, the port is taken.
        """
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parts = urllib.parse.urlsplit(self.path)
                if parts.path != "/status":
                    self._reply(404, {"error": f"no {parts.path}"})
                    return
                part = urllib.parse.parse_qs(parts.query).get("part", [""])[0]
                self._reply(200, server.answer(part).to_data())

            def _reply(self, code: int, data: dict[str, Any]) -> None:
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                # Every follower asks every few seconds; the log stays quiet.
                pass

        try:
            httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as error:
            raise RunError(
                exits.BRINGUP,
                f"cannot serve the driver status on port {self.port}: {error}",
            ) from None
        httpd.daemon_threads = True
        self._httpd = httpd
        threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        return self

    def answer(self, part: str) -> Status:
        with self._cond:
            if part:
                self._polls[part] = self._clock()
                self._cond.notify_all()
            return Status(self.plan_sha, self._state, self._exit)

    @property
    def state(self) -> State:
        with self._cond:
            return self._state

    def set(self, state: State, exit: int | None = None) -> None:
        with self._cond:
            self._state, self._exit = state, exit
            self._changed = self._clock()

    def silent(self) -> list[str]:
        """The followers that have never asked."""
        with self._cond:
            return [part for part in self.followers if part not in self._polls]

    def linger(self, timeout_s: float | None = None) -> list[str]:
        """Wait until every follower that has asked has asked again since the
        last state change.

        A follower never heard from is not waited for: nothing says it is
        coming. Returns the followers still behind once `timeout_s`
        (LINGER_S) has passed.
        """
        timeout = LINGER_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                behind = [
                    part
                    for part in self.followers
                    if part in self._polls and self._polls[part] < self._changed
                ]
                remaining = deadline - time.monotonic()
                if not behind or remaining <= 0:
                    return behind
                self._cond.wait(remaining)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


@dataclass(frozen=True)
class Verdict:
    """Why a follower leaves, and its exit code."""

    code: int
    reason: str


class FollowerRules:
    """What a follower makes of each answer, or of no answer, at time `now`."""

    def __init__(self, plan_sha: str, started: float) -> None:
        self.plan_sha = plan_sha
        self.started = started
        self.contacted = False
        self.gone_since: float | None = None
        self.last: Status | None = None
        self.seen_serving = False

    def observe(self, status: Status | None, now: float) -> Verdict | None:
        """The verdict once one applies, else None."""
        if status is None:
            if not self.contacted:
                if now - self.started >= FIRST_CONTACT_S:
                    return Verdict(
                        exits.BRINGUP,
                        f"the driver did not answer within {FIRST_CONTACT_S:.0f}s",
                    )
                return None
            if self.gone_since is None:
                self.gone_since = now
            if now - self.gone_since >= GONE_S:
                return Verdict(
                    exits.DRIVER_LOST,
                    f"the driver has not answered for {GONE_S:.0f}s: it was lost "
                    "before it finished",
                )
            return None
        self.contacted = True
        self.gone_since = None
        self.last = status
        if status.plan != self.plan_sha:
            return Verdict(
                exits.USAGE,
                f"the driver runs Plan {status.plan} and this host Plan "
                f"{self.plan_sha}: they belong to different runs",
            )
        if status.state == "serving":
            self.seen_serving = True
        if status.state == "done":
            return Verdict(exits.OK, f"the driver is done (exit {status.exit})")
        if status.state == "aborted":
            return Verdict(exits.DRIVER_LOST, "the driver aborted")
        return None

    def server_exited(self, status: Status | None, now: float) -> Verdict:
        """The follower's own server exited; `status` is the driver's answer to one
        more request.

        A done driver is a normal end, as is an aborted one ending its
        followers: the server stopped because its peers did. Otherwise the
        server failed: before the driver was serving, while starting (4);
        after, while serving (5).
        """
        if status is not None:
            verdict = self.observe(status, now)
            if verdict is not None:
                return verdict
        if self.seen_serving:
            return Verdict(exits.SERVER_DIED, "this host's server died")
        return Verdict(exits.NOT_READY, "this host's server died while starting")

    @property
    def seen_done(self) -> bool:
        return self.last is not None and self.last.state == "done"


def fetch_status(url: str) -> Status | None:
    """The driver's Status, or None when no Status came back."""
    found = net.http_get(url, REQUEST_TIMEOUT_S)
    if found is None or found[0] != 200:
        return None
    return Status.parse(found[1])


class Follower:
    """Polls the driver from a thread until a verdict applies.

    `say` gets a line when the driver first answers, when it starts serving,
    and with the verdict.
    """

    def __init__(
        self,
        url: str,
        plan_sha: str,
        *,
        say: Callable[[str], None],
        fetch: Callable[[str], Status | None] = fetch_status,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self._say = say
        self._fetch = fetch
        self._clock = clock
        self._lock = threading.Lock()
        self.rules = FollowerRules(plan_sha, clock())
        self.verdict: Verdict | None = None
        self._decided = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> "Follower":
        self._say(f"Following the driver at {self.url}")
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stopping.is_set():
            status = self._fetch(self.url)
            with self._lock:
                contacted, serving = self.rules.contacted, self.rules.seen_serving
                verdict = self.rules.observe(status, self._clock())
                if not contacted and self.rules.contacted:
                    self._say(f"The driver answered: {status and status.state}")
                if not serving and self.rules.seen_serving:
                    self._say("The driver is serving")
            if verdict is not None:
                self._decide(verdict)
                return
            self._stopping.wait(POLL_S)

    def _decide(self, verdict: Verdict) -> None:
        with self._lock:
            if self.verdict is not None:
                return
            self.verdict = verdict
        self._say(f"Leaving with exit {verdict.code}: {verdict.reason}")
        self._decided.set()

    def wait(self, timeout_s: float | None = None) -> Verdict | None:
        """The verdict, once there is one or `timeout_s` has passed."""
        self._decided.wait(timeout_s)
        return self.verdict

    def server_exited(self) -> Verdict:
        """The verdict when this host's server exited: one more request decides."""
        status = self._fetch(self.url)
        with self._lock:
            verdict = self.rules.server_exited(status, self._clock())
        self._decide(verdict)
        assert self.verdict is not None
        return self.verdict

    @property
    def seen_done(self) -> bool:
        with self._lock:
            return self.rules.seen_done

    def stop(self) -> None:
        """Stop polling. A request in flight is not waited for: the thread
        is a daemon, and the verdict is already known."""
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(min(POLL_S, 1.0))
