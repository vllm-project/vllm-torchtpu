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
"""Bringing up this host's part: its pre commands, then its server.

Pre commands run in order before the server (ray start, the remote-code
warm), each within its own timeout. A best-effort one that fails is a
warning; any other is a bring-up failure (exit 3).

Ray: the first host of a ray role starts the head, then waits until every
host of the role is a Ray node with TPUs before its server starts. The
others wait until the head's port answers, then join. `ray start` gets the
host's whole env, the scenario's included, so the raylet and every Ray
worker it starts inherit it. `ray start` returns once its daemons run, and
they stay in its process group until teardown runs `ray stop --force` and
stops what is left of the group.

The readiness wait is Readiness.timeout_s from the spawn, the number `show`
prints: the engine's own startup timeout plus a margin, so a slow start
fails with vLLM's message rather than the wait's. A driver waits for the
heads of the other roles as long, plus the names' budget, counted from its
own start: their hosts may start later than its own. The router starts
once every head is ready, so it never forwards to a head that cannot yet
answer, and has ROUTER_READY_S to answer its own health check.
"""

import contextlib
import pathlib
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence

from scenario.plan.types import Command, HostPlan, Readiness
from scenario.runtime import net
from scenario.runtime.exits import BRINGUP, NOT_READY, RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import LOOPBACK, Refs
from scenario.runtime.proc import Proc
from scenario.runtime.proc import run as run_command

POLL_INTERVAL_S = 1.0
# A Ray worker probes the head's port this often, and gives up after the
# names' budget: the head's host may start as late as any other.
RAY_JOIN_PROBE_S = 10.0
RAY_JOIN_TIMEOUT_S = float(net.RESOLVE_TIMEOUT_S)
# The Ray head counts nodes with TPUs this often, until every host is one.
RAY_CLUSTER_POLL_S = 15.0
RAY_CLUSTER_TIMEOUT_S = 1800.0
# A Ray query, `ray status` and `ray stop` each give up after this long.
RAY_COMMAND_TIMEOUT_S = 120.0
# The heads of other roles are asked this often.
REMOTE_POLL_S = 5.0
# The router has this long from its spawn to answer its health check.
ROUTER_READY_S = 120.0
# Lines of the server's output repeated in a failure report; all of it was
# streamed above.
FAILURE_TAIL = 50


def host_env(
    base: Mapping[str, str], host: HostPlan, refs: Refs
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """The host's env: inherited, then the scenario's, then the tool's.

    Returns that env and the two parts the runtime adds, for the spawn log.
    """
    scenario = refs.env(host.scenario_env)
    tool = refs.env(host.tool_env)
    return {**base, **scenario, **tool}, scenario, tool


def run_pre_command(
    what: str,
    command: Command,
    *,
    refs: Refs,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log: Log,
    check: Callable[[], None] | None = None,
) -> None:
    """Run one pre command within its timeout.

    `check` is called while the command runs and may raise to stop it.

    Raises:
        RunError: exit 3, the command is not best effort and failed.
    """
    argv = refs.argv(command.argv)
    extra = refs.env(command.env)
    log.spawn(what, argv, {**added, **({"env": extra} if extra else {})})
    try:
        done = run_command(
            argv,
            env={**env, **extra},
            cwd=cwd,
            timeout_s=command.timeout_s,
            check=check,
        )
    except OSError as error:
        raise RunError(BRINGUP, f"{what}: cannot run {argv[0]}: {error}") from None
    if done.ok:
        return
    if done.timed_out:
        problem = f"{what} ({argv[0]}) did not finish in {command.timeout_s}s"
    else:
        problem = f"{what} ({argv[0]}) exited {done.code}"
    if not command.best_effort:
        raise RunError(BRINGUP, problem)
    warning = f"{problem}; it is best effort, so bring-up goes on"
    log.warning(warning)


def run_pre(
    host: HostPlan,
    *,
    refs: Refs,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log: Log,
    check: Callable[[], None] | None = None,
) -> None:
    """Run the host's pre commands in order; see run_pre_command."""
    for index, command in enumerate(host.pre):
        run_pre_command(
            f"pre[{index}]",
            command,
            refs=refs,
            env=env,
            added=added,
            cwd=cwd,
            log=log,
            check=check,
        )


def is_ray_start(argv: Sequence[str]) -> bool:
    return list(argv[:2]) == ["ray", "start"]


def ray_head_address(argv: Sequence[str]) -> tuple[str, int] | None:
    """The head a `ray start --address=HOST:PORT` joins; None for the head."""
    for word in argv:
        if word.startswith("--address="):
            host, _, port = word.removeprefix("--address=").rpartition(":")
            return host, int(port)
    return None


def start_ray(
    what: str,
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log: Log,
    pause: Callable[[float], None] = time.sleep,
) -> Proc:
    """Start this host's Ray node; the Proc whose group holds its daemons.

    A worker first waits until the head's port answers: `ray start
    --address` fails outright against a head that is not up. Not --block,
    which the Plan never asks for: a blocking worker would hold its chips
    until something killed it.

    Raises:
        RunError: exit 3, the head never answered or `ray start` failed.
    """
    head = ray_head_address(argv)
    if head is not None:
        host, port = head
        log.say(f"Waiting for the Ray head at {host}:{port}")
        deadline = time.monotonic() + RAY_JOIN_TIMEOUT_S
        while not net.port_open(host, port, timeout_s=5):
            if time.monotonic() >= deadline:
                raise RunError(
                    BRINGUP,
                    f"no Ray head at {host}:{port} after {RAY_JOIN_TIMEOUT_S:.0f}s",
                )
            pause(RAY_JOIN_PROBE_S)
    log.spawn(what, argv, added)
    try:
        proc = Proc(argv, env=env, cwd=cwd).start()
    except OSError as error:
        raise RunError(BRINGUP, f"{what}: cannot run {argv[0]}: {error}") from None
    try:
        code = proc.wait()
    except BaseException:
        proc.stop()
        raise
    if code != 0:
        proc.stop()
        raise RunError(BRINGUP, f"{what} (ray start) exited {code}")
    return proc


# How many Ray nodes are alive with TPUs, written to the file named by
# argv[1]: Ray's own logging floods stdout once a driver attaches, and a
# count printed there would be lost in it. A node counts only once it
# advertises chips: a raylet registers before libtpu has handed it its
# chips, and a server started against such a cluster fails an hour later in
# a placement timeout that looks nothing like its cause.
RAY_NODES = """\
import sys
import ray
ray.init(address="auto", logging_level="error")
count = sum(
    1
    for node in ray.nodes()
    if node["Alive"] and node.get("Resources", {}).get("TPU", 0) > 0
)
with open(sys.argv[1], "w") as report:
    report.write(str(count))
"""


def ray_tpu_nodes(
    env: Mapping[str, str],
    cwd: pathlib.Path,
    report: pathlib.Path,
    errors: pathlib.Path,
) -> int:
    """The Ray nodes alive with TPUs, or 0 when the query failed."""
    report.unlink(missing_ok=True)
    with errors.open("w") as err:
        try:
            subprocess.run(
                ["python3", "-c", RAY_NODES, str(report)],
                env=dict(env),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                timeout=RAY_COMMAND_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            err.write(f"{error}\n")
    try:
        return int(report.read_text())
    except (OSError, ValueError):
        return 0


def ray_status(env: Mapping[str, str], cwd: pathlib.Path) -> None:
    """`ray status` into the log, best effort."""
    with contextlib.suppress(OSError):
        run_command(
            ["ray", "status"], env=env, cwd=cwd, timeout_s=RAY_COMMAND_TIMEOUT_S
        )


def wait_for_ray_nodes(
    count: int,
    *,
    env: Mapping[str, str],
    cwd: pathlib.Path,
    log: Log,
    pause: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until `count` Ray nodes are alive with TPUs.

    Raises:
        RunError: exit 3, fewer after RAY_CLUSTER_TIMEOUT_S.
    """
    log.say(f"Waiting for {count} Ray nodes with TPUs")
    deadline = time.monotonic() + RAY_CLUSTER_TIMEOUT_S
    reported = -1
    with tempfile.TemporaryDirectory(prefix="scenario-ray-") as scratch:
        report = pathlib.Path(scratch, "nodes")
        errors = pathlib.Path(scratch, "errors")
        while True:
            alive = ray_tpu_nodes(env, cwd, report, errors)
            if alive >= count:
                log.say(f"Ray cluster complete: {alive} nodes")
                break
            if alive != reported:
                log.say(f"  {alive}/{count} Ray nodes with TPUs")
                reported = alive
            if time.monotonic() >= deadline:
                # The head is a node with TPUs itself, so 0 means the query
                # failed, not that nobody joined.
                lines = errors.read_text().splitlines()[-20:]
                if lines:
                    log.say("Last Ray query error:\n" + "\n".join(lines))
                ray_status(env, cwd)
                raise RunError(
                    BRINGUP,
                    f"the Ray cluster had {alive} of {count} nodes with TPUs "
                    f"after {RAY_CLUSTER_TIMEOUT_S:.0f}s",
                )
            pause(RAY_CLUSTER_POLL_S)
    ray_status(env, cwd)


def stop_ray(
    daemons: Proc, *, env: Mapping[str, str], cwd: pathlib.Path, log: Log
) -> None:
    """`ray stop --force`, then whatever is left of `ray start`'s group."""
    try:
        done = run_command(
            ["ray", "stop", "--force"],
            env=env,
            cwd=cwd,
            timeout_s=RAY_COMMAND_TIMEOUT_S,
        )
        if not done.ok:
            warning = f"ray stop --force exited {done.code}"
            log.warning(warning)
    except OSError as error:
        warning = f"cannot run ray stop: {error}"
        log.warning(warning)
    daemons.stop()


def start_server(
    host: HostPlan,
    readiness: Readiness,
    *,
    refs: Refs,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    log: Log,
) -> Proc:
    """Spawn the host's server; its output goes to stdout and `log_path`.

    Raises:
        RunError: exit 3, something already answers on the API port of a
            host that serves the API.
    """
    assert host.serve is not None
    if host.head and net.port_open(LOOPBACK, readiness.port):
        raise RunError(
            BRINGUP,
            f"port {readiness.port} is already taken: another server runs on this host",
        )
    argv = refs.argv(host.serve.argv)
    serve_env = refs.env(host.serve.env)
    log.spawn("serve", argv, {**added, **({"env": serve_env} if serve_env else {})})
    try:
        return Proc(argv, env={**env, **serve_env}, cwd=cwd, log=log_path).start()
    except OSError as error:
        raise RunError(BRINGUP, f"cannot run {argv[0]}: {error}") from None


def wait_ready(
    server: Proc,
    readiness: Readiness,
    *,
    spawned_at: float,
    log: Log,
    note: Callable[[], str] = lambda: "",
) -> float:
    """Wait for the server's readiness; the seconds it took.

    `note` adds to each progress line, such as the hosts not heard from.

    Raises:
        RunError: exit 4, the server died while starting or the wait ran out.
    """
    url = f"http://{LOOPBACK}:{readiness.port}{readiness.path}"
    outcome = net.wait_http(
        url,
        deadline=spawned_at + readiness.timeout_s,
        proc=server,
        interval_s=POLL_INTERVAL_S,
        progress=lambda waited: log.say(
            f"Waiting for server... ({waited:.0f}s elapsed){note()}"
        ),
    )
    waited = time.monotonic() - spawned_at
    if outcome == "ready":
        log.say(f"Server ready on port {readiness.port} ({waited:.0f}s)")
        return waited
    if outcome == "died":
        code = server.poll()
        report_tail(server, log)
        raise RunError(
            NOT_READY,
            f"the server died while starting (exit {code}) after {waited:.0f}s",
        )
    report_tail(server, log)
    raise RunError(
        NOT_READY,
        f"the server was not ready within {readiness.timeout_s}s "
        f"(GET {readiness.path} on port {readiness.port})",
    )


def wait_remote(
    role: str,
    url: str,
    *,
    deadline: float,
    own: Proc | None,
    log: Log,
    note: Callable[[], str] = lambda: "",
) -> None:
    """Wait for another role's head to answer `url`.

    Raises:
        RunError: exit 4, this host's own server died meanwhile, or the
            deadline passed.
    """
    log.say(f"Waiting for role {role} at {url}")
    started = time.monotonic()
    outcome = net.wait_http(
        url,
        deadline=deadline,
        proc=own,
        interval_s=REMOTE_POLL_S,
        progress=lambda waited: log.say(
            f"Waiting for role {role}... ({waited:.0f}s elapsed){note()}"
        ),
    )
    waited = time.monotonic() - started
    if outcome == "ready":
        log.say(f"Role {role} is ready ({waited:.0f}s)")
        return
    if outcome == "died":
        assert own is not None
        report_tail(own, log)
        raise RunError(
            NOT_READY,
            f"this host's server died (exit {own.poll()}) while waiting for "
            f"role {role}",
        )
    raise RunError(NOT_READY, f"role {role} was not ready at {url} in time")


def start_router(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    log: Log,
) -> Proc:
    """Spawn the router; its output goes to the log, prefixed, and `log_path`.

    Raises:
        RunError: exit 3, it cannot be run.
    """
    log.spawn("router", argv, added)
    try:
        return Proc(argv, env=env, cwd=cwd, log=log_path, prefix="[router] ").start()
    except OSError as error:
        raise RunError(BRINGUP, f"cannot run the router: {error}") from None


def wait_router(router: Proc, port: int, path: str, *, log: Log) -> None:
    """Wait ROUTER_READY_S for the router's health check.

    Raises:
        RunError: exit 4, it died or did not answer in time.
    """
    started = time.monotonic()
    outcome = net.wait_http(
        f"http://{LOOPBACK}:{port}{path}",
        deadline=started + ROUTER_READY_S,
        proc=router,
        interval_s=POLL_INTERVAL_S,
    )
    if outcome == "ready":
        log.say(f"Router ready on port {port} ({time.monotonic() - started:.0f}s)")
        return
    report_tail(router, log, "router.log")
    if outcome == "died":
        raise RunError(
            NOT_READY, f"the router died while starting (exit {router.poll()})"
        )
    raise RunError(
        NOT_READY,
        f"the router was not ready within {ROUTER_READY_S:.0f}s (GET {path} on "
        f"port {port})",
    )


def report_tail(
    proc: Proc, log: Log, name: str = "server.log", count: int = FAILURE_TAIL
) -> None:
    lines = proc.tail(count)
    log.say(f"========== Last {len(lines)} lines of {name} ==========")
    log.say("\n".join(lines))
    log.say("=" * 51)
