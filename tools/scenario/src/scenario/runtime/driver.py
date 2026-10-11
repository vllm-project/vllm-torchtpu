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
"""One host's part of a run.

Every host of a run runs the same Plan and finds its part from its
placement: a role and an index, or the chipless client. One part drives:
the first host of the only role, or the part tasks.driver names (a role's
first host, or the client). The others follow it (runtime/control.py).
The task verbs run against a server at a URL instead.

A part runs its pre commands (ray start, the remote-code warm), then
starts its server if it has one; a Ray worker and the client have none.
The driver serves its status to the followers and waits for its own server
(on an mp head, every host has joined; a Ray head starts its server only
once every host is a node), then for the heads of the other roles, then
starts the router and waits for it. It runs the tasks in Plan order and
tears down: the router, its server, `ray stop --force`. A follower starts
at once, without waiting on the driver, and leaves as the driver's status
says.

Tasks run in the Plan's order. A failed task does not stop the next one; a
dead server or router does (exit 5), and so does a custom task's
skip_rest_if_exit_ge.
At its end the driver says done, with its exit code, before it stops its
server, so a follower whose engine dies with the head's still leaves with
0. SIGTERM or SIGINT says aborted, stops every child (20 s, then SIGKILL)
and exits 143 or 130 without after-tasks. Otherwise, after teardown, the
target's end after-tasks run, and a fatal one that fails turns exit 0 into
6. Either way the driver then answers a little longer, until every follower
it has heard from has seen its end (at most 10 s, or 3 s after an abort).
Core dumps are off for the run and everything it starts.
"""

import datetime
import functools
import pathlib
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scenario import __version__
from scenario.plan import codec
from scenario.plan.types import CLIENT_ROLE, Command, HostPlan, Plan, Ref, RolePlan
from scenario.runtime import (
    bringup,
    control,
    exec_custom,
    exec_eval,
    exec_perf,
    exits,
    net,
    results,
)
from scenario.runtime.context import (
    Every,
    Liveness,
    LocalRouter,
    LocalServer,
    RemoteServer,
    ServerDied,
    TaskContext,
    TaskOutcome,
    task_name,
)
from scenario.runtime.control import StatusServer, Verdict
from scenario.runtime.exits import RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import (
    ENV_IMAGE,
    ENV_RESOLVED,
    ENV_TARGET_DATA,
    LOOPBACK,
    Placement,
    PlacementFlags,
    Refs,
    decode_resolved,
    decode_target_data,
    place,
)
from scenario.runtime.proc import Proc, disable_core_dumps
from scenario.runtime.publish import Publisher
from scenario.target import TargetData, TargetInfo

Mode = Literal["run", "up", "url"]

# A follower checks on its own server this often.
OWN_SERVER_CHECK_S = 1.0
# A follower's own readiness check (logged, not waited on) gives up after this.
OWN_READY_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class RunOptions:
    """How to run a Plan on this host.

    `env` is the inherited environment: children start from it, and
    SCENARIO_* placement is read from it. `mode` run serves and drives (with
    no tasks it serves until stopped), up only serves, url drives tasks
    against the server at `url`.
    """

    env: Mapping[str, str]
    flags: PlacementFlags = PlacementFlags()
    results_root: pathlib.Path | None = None
    repo_root: pathlib.Path = pathlib.Path(".")
    resolved: Mapping[str, Any] | None = None
    target: TargetData | None = None
    mode: Mode = "run"
    url: str | None = None
    lm_eval_cache: str | None = None


class Interrupted(BaseException):
    """SIGTERM or SIGINT arrived; raised wherever the main thread was."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class Decided(Exception):
    """The driver's status decided how a follower ends, while it was busy."""

    def __init__(self, verdict: Verdict) -> None:
        super().__init__(verdict.reason)
        self.verdict = verdict


class Signals:
    """SIGTERM and SIGINT raise Interrupted while armed, and are noted while held.

    Teardown runs held, so a second signal cannot cut it short.
    """

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self) -> None:
        self.armed = True
        self.received: int | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "Signals":
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum: int, _frame: object) -> None:
        if self.received is None:
            self.received = signum
        if self.armed:
            self.armed = False
            raise Interrupted(signum)

    def hold(self) -> None:
        self.armed = False

    def consume(self) -> None:
        """The signal was handled as a request to stop, not an abort."""
        self.received = None

    def arm(self) -> None:
        if self.received is not None:
            raise Interrupted(self.received)
        self.armed = True


Executor = Callable[[TaskContext, Any, dict[str, Any]], TaskOutcome]
EXECUTORS: dict[str, Executor] = {
    "perf": exec_perf.run_perf,
    "agentx": exec_custom.run_agentx,
    "custom": exec_custom.run_custom,
    "eval": exec_eval.run_eval,
}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Host:
    """What every part does: its files, Refs and env, pre commands and server."""

    def __init__(
        self,
        plan: Plan,
        options: RunOptions,
        placement: Placement,
        now: Callable[[], datetime.datetime] = _now,
    ) -> None:
        self.plan = plan
        self.options = options
        self.placement = placement
        self.started = now()
        self.begun = time.monotonic()
        self.log = Log(placement.part, codec.digest(plan), __version__)
        self.summary: results.Summary | None = None
        self.server: Proc | None = None
        # The group of `ray start`, which holds this host's Ray daemons.
        self.ray: Proc | None = None
        self.spawned_at = 0.0
        self.role: RolePlan | None = None
        self.host: HostPlan | None = None
        if options.mode != "url" and placement.role != CLIENT_ROLE:
            self.role = plan.role(placement.role)
            self.host = self.role.host_plans[placement.host_index]
        self.host_env: dict[str, str] = dict(options.env)
        self.host_added: dict[str, dict[str, str]] = {}

    # Waits that a follower cuts short once the driver has decided.

    def _pause(self, seconds: float) -> None:
        time.sleep(seconds)

    def _check(self) -> None:
        """Raises Decided when this part must stop; never on the driver."""

    # Setup -------------------------------------------------------------

    def _target(self) -> TargetData | None:
        env = self.options.env
        if self.options.target is not None:
            return self.options.target
        if env.get(ENV_TARGET_DATA):
            return decode_target_data(env[ENV_TARGET_DATA])
        return None

    def _run_dir(self, target: TargetData | None) -> pathlib.Path:
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        root = results.results_root(
            self.options.results_root,
            target.info.results_root if target is not None else None,
            self.plan.scenario_name,
            stamp,
        )
        self.root = root
        return results.run_dir(root, self.plan)

    def _summary(self, directory: pathlib.Path) -> results.Summary:
        summary = results.Summary(
            directory / results.SUMMARY,
            self.plan,
            self.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        summary.save()
        return summary

    def _prepare(self, run_dir: pathlib.Path, directory: pathlib.Path) -> None:
        """Refs, and this host's env: inherited, the scenario's, the tool's."""
        self.directory = directory
        self.refs = Refs(
            self.plan,
            self.placement,
            run_dir=run_dir,
            server_log=directory / results.SERVER_LOG,
            env=self.options.env,
            resolve=functools.partial(net.resolve_ipv4, sleep=self._pause),
            warn=self.log.say,
        )
        if self.host is not None:
            self.host_env, scenario_env, tool_env = bringup.host_env(
                self.options.env, self.host, self.refs
            )
            self.host_added = {"scenario env": scenario_env, "tool env": tool_env}

    def _phase(self, name: str) -> None:
        self.log.phase(name)
        if self.summary is not None:
            self.summary.phase(name)

    # Bring-up ----------------------------------------------------------

    def _bring_up_part(self) -> None:
        """This host's pre commands, then its server, if it has them."""
        host, role = self.host, self.role
        if host is None or role is None:
            return
        for index, command in enumerate(host.pre):
            what = f"pre[{index}]"
            if role.backend == "ray" and bringup.is_ray_start(command.argv):
                self._start_ray(what, command, role)
                continue
            bringup.run_pre_command(
                what,
                command,
                refs=self.refs,
                env=self.host_env,
                added=self.host_added,
                cwd=self.options.repo_root,
                log=self.log,
                check=self._check,
            )
        if host.pre:
            self._phase("pre-done")
        if host.serve is None:
            return
        self._phase("server-spawn")
        self.spawned_at = time.monotonic()
        self.server = bringup.start_server(
            host,
            role.readiness,
            refs=self.refs,
            env=self.host_env,
            added=self.host_added,
            cwd=self.options.repo_root,
            log_path=self.directory / results.SERVER_LOG,
            log=self.log,
        )

    def _start_ray(self, what: str, command: Command, role: RolePlan) -> None:
        """This host's Ray node; on the head, the wait for every host's."""
        argv = self.refs.argv(command.argv)
        self.ray = bringup.start_ray(
            what,
            argv,
            env=self.host_env,
            added=self.host_added,
            cwd=self.options.repo_root,
            log=self.log,
            pause=self._pause,
        )
        if bringup.ray_head_address(argv) is not None:
            self._phase("ray-joined")
            return
        bringup.wait_for_ray_nodes(
            role.hosts,
            env=self.host_env,
            cwd=self.options.repo_root,
            log=self.log,
            pause=self._pause,
        )
        self._phase("ray-cluster-ready")

    # Teardown ----------------------------------------------------------

    def _command_failed(self) -> None:
        if self.summary is not None:
            self.summary.command_failed()

    def _stop_part(self, first: Sequence[Proc] = ()) -> None:
        """Stop `first`, then this host's server, then its Ray node."""
        procs = [*first, *([self.server] if self.server is not None else [])]
        if not procs and self.ray is None:
            return
        self.log.phase("stopping")
        for proc in procs:
            proc.stop()
        if self.ray is not None:
            bringup.stop_ray(
                self.ray, env=self.host_env, cwd=self.options.repo_root, log=self.log
            )

    def _finish(self, code: int, aborted: bool) -> None:
        if self.summary is not None:
            self.summary.data["phases"]["exit"] = self.summary.elapsed()
            self.summary.set(state="aborted" if aborted else "done", exit=code)
        self.log.phase(f"exit {code}")


class Driver(Host):
    """The part that runs the tasks: the first host of the driver's role."""

    def __init__(
        self,
        plan: Plan,
        options: RunOptions,
        placement: Placement,
        executors: Mapping[str, Executor] | None = None,
        now: Callable[[], datetime.datetime] = _now,
    ) -> None:
        super().__init__(plan, options, placement, now)
        self.executors = dict(EXECUTORS if executors is None else executors)
        self.liveness: Liveness = Liveness()
        self.publisher: Publisher | None = None
        self.status: StatusServer | None = None
        self.router: Proc | None = None

    def run(self) -> int:
        code = exits.OK
        aborted = False
        disable_core_dumps()
        with Signals() as signals:
            self.signals = signals
            try:
                self._setup()
                if self.options.mode == "url":
                    self._check_url()
                else:
                    self._bring_up()
                if self.options.mode == "up" or (
                    self.options.mode == "run" and not self.plan.driver.tasks
                ):
                    self._serve()
                else:
                    code = self._tasks()
            except Interrupted as stop:
                aborted = True
                code = exits.for_signal(stop.signum)
                self.log.say(f"Stopping on signal {signal.Signals(stop.signum).name}")
            except RunError as error:
                self.log.error(error.message)
                code = error.code
                if code in (exits.BRINGUP, exits.NOT_READY, exits.SERVER_DIED):
                    self._command_failed()
            finally:
                signals.hold()
                self._announce(code, aborted)
                self._teardown()
            if not aborted and self.summary is not None:
                try:
                    signals.arm()
                    code = self._end(code)
                except Interrupted as stop:
                    aborted = True
                    code = exits.for_signal(stop.signum)
                finally:
                    signals.hold()
            self._close_status(aborted)
        self._finish(code, aborted)
        return code

    # Setup -------------------------------------------------------------

    def _setup(self) -> None:
        plan, options = self.plan, self.options
        env = options.env
        self.log.phase("start")
        self.target = self._target() or TargetData(TargetInfo("none", None, None))
        resolved = options.resolved
        if resolved is None and env.get(ENV_RESOLVED):
            resolved = decode_resolved(env[ENV_RESOLVED])
        self.resolved = resolved
        self.image = env.get(ENV_IMAGE) or plan.image or ""
        self.run_dir = self._run_dir(self.target)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary = self._summary(self.run_dir)
        scenario = resolved.get("scenario") if resolved is not None else None
        timestamp = self.started.strftime("%Y%m%d-%H%M%S")
        results.write_json(
            self.run_dir / results.CONFIG,
            results.config_json(plan, scenario, timestamp),
        )
        results.write_json(
            self.run_dir / results.SCENARIO_RUN,
            results.scenario_run(
                plan,
                resolved=resolved,
                target=self.target.info,
                placement=self.placement.to_data(),
                image=self.image,
            ),
        )
        if options.mode != "url" and plan.has_followers():
            self.status = StatusServer(
                codec.digest(plan), control.followers(plan)
            ).start()
            self.log.say(
                f"Serving the driver status on port {self.status.port} to "
                f"{', '.join(self.status.followers)}"
            )
        self._prepare(self.run_dir, self.run_dir)
        self.task_added = self.refs.env(plan.driver.task_env)
        self.publisher = Publisher(
            self.target.after_tasks,
            run_dir=self.run_dir,
            image=self.image,
            run_stamp=timestamp,
            experiment=bool(plan.experiment),
            env={**self.host_env, **self.task_added},
            cwd=options.repo_root,
            log=self.log,
            summary=self.summary,
        )
        self._phase("placement")

    def _silent(self) -> str:
        """A progress line's note of the followers not heard from yet."""
        silent = self.status.silent() if self.status is not None else []
        return f"; not heard from: {', '.join(silent)}" if silent else ""

    # Bring-up ----------------------------------------------------------

    def _bring_up(self) -> None:
        self._bring_up_part()
        parts: list[Liveness] = []
        if self.server is not None and self.role is not None:
            parts.append(LocalServer(self.server, self.log, self.run_dir))
            bringup.wait_ready(
                self.server,
                self.role.readiness,
                spawned_at=self.spawned_at,
                log=self.log,
                note=self._silent,
            )
            self._phase("ready")
        others = [r for r in self.plan.roles if r.name != self.placement.role]
        for role in others:
            head = self.refs.ref(Ref("role_head", role=role.name))
            readiness = role.readiness
            bringup.wait_remote(
                role.name,
                f"http://{head}:{readiness.port}{readiness.path}",
                deadline=self.begun + net.RESOLVE_TIMEOUT_S + readiness.timeout_s,
                own=self.server,
                log=self.log,
                note=self._silent,
            )
        if others:
            self._phase("roles-ready")
        router = self.plan.driver.router
        if router is not None:
            self.router = bringup.start_router(
                self.refs.argv(router.argv),
                env={**self.host_env, **self.task_added},
                added={"env": self.task_added},
                cwd=self.options.repo_root,
                log_path=self.run_dir / results.ROUTER_LOG,
                log=self.log,
            )
            bringup.wait_router(
                self.router, router.port, router.health_path, log=self.log
            )
            parts.append(LocalRouter(self.router, self.log))
            self._phase("router-ready")
        if parts:
            self.liveness = parts[0] if len(parts) == 1 else Every(*parts)
        self._serving()

    def _serving(self) -> None:
        assert self.summary is not None
        self.summary.set(state="serving")
        if self.status is not None:
            self.status.set("serving")

    def _check_url(self) -> None:
        assert self.summary is not None and self.options.url is not None
        self.liveness = RemoteServer(self.options.url)
        try:
            self.liveness.check()
        except ServerDied as error:
            raise RunError(
                exits.NOT_READY, f"no server is ready: {error.message}"
            ) from None
        self.summary.set(state="serving")

    def _serve(self) -> None:
        """Serve until the server or router exits (exit 5) or a signal stops it."""
        driver = self.plan.driver
        host = self.refs.value(driver.endpoint_host)
        self.log.say(
            f"Serving {self.plan.model_name} at http://{host}:"
            f"{driver.endpoint_port}; SIGINT or SIGTERM stops it"
        )
        try:
            while True:
                if self.server is None:
                    time.sleep(1.0)
                elif (code := self.server.wait(1.0)) is not None:
                    raise ServerDied(f"the server exited (exit {code})")
                code = self.router.poll() if self.router is not None else None
                if code is not None:
                    raise ServerDied(
                        f"the router exited (exit {code})", "the router died"
                    )
        except Interrupted as stop:
            self.signals.consume()
            self.log.say(f"Stopping on signal {signal.Signals(stop.signum).name}")

    # Tasks -------------------------------------------------------------

    def _context(self) -> TaskContext:
        assert self.summary is not None
        return TaskContext(
            plan=self.plan,
            run_dir=self.run_dir,
            refs=self.refs,
            env={**self.host_env, **self.task_added},
            added=self.task_added,
            cwd=self.options.repo_root,
            log=self.log,
            summary=self.summary,
            liveness=self.liveness,
            lm_eval_cache=self.options.lm_eval_cache or self.target.lm_eval_cache_root,
            publisher=self.publisher,
        )

    def _tasks(self) -> int:
        summary = self.summary
        assert summary is not None
        ctx = self._context()
        entries = [
            (task, summary.task(task.kind, task_name(task)))
            for task in self.plan.driver.tasks
        ]
        summary.save()
        code = exits.OK
        skipped: str | None = None
        for position, (task, entry) in enumerate(entries):
            if skipped is not None:
                entry.update(status="skipped", reason=skipped)
                continue
            name = entry["name"]
            executor = self.executors.get(task.kind)
            if executor is None:
                entry.update(status="failed", reason=f"no {task.kind} executor")
                code = exits.worst(code, exits.TASK_FAILED)
                continue
            self.log.phase(f"task-start {name}")
            entry.update(status="running", started=summary.elapsed())
            summary.save()
            try:
                ctx.alive()
                outcome = executor(ctx, task, entry)
            except (ServerDied, Interrupted) as stop:
                why = stop.reason if isinstance(stop, ServerDied) else "interrupted"
                entry.update(status="failed", ended=summary.elapsed(), reason=why)
                for _, rest in entries[position + 1 :]:
                    rest.update(status="skipped", reason=why)
                summary.save()
                raise
            entry.update(status=outcome.status, ended=summary.elapsed())
            summary.save()
            self.log.phase(f"task-end {name} {outcome.status}")
            if outcome.status == "failed":
                code = exits.worst(code, exits.TASK_FAILED)
            if outcome.skip_rest is not None:
                skipped = outcome.skip_rest
                self.log.say(f"Skipping the remaining tasks: {skipped}")
        return code

    # Teardown ----------------------------------------------------------

    def _announce(self, code: int, aborted: bool) -> None:
        """Tell the followers how the run ended, before anything stops."""
        if self.status is not None:
            self.status.set("aborted" if aborted else "done", code)

    def _teardown(self) -> None:
        self._stop_part([self.router] if self.router is not None else [])

    def _end(self, code: int) -> int:
        """The target's end after-tasks, on normal and failure exits alike.

        The summary says done first, so the files they hand on are final.
        """
        assert self.summary is not None
        self.summary.set(state="done", exit=code)
        if self.publisher is None:
            return code
        self.publisher.end()
        if self.publisher.fatal_failed:
            code = exits.worst(code, exits.AFTER_TASK)
        return code

    def _close_status(self, aborted: bool) -> None:
        """Wait briefly until every follower has seen the end, then stop serving."""
        if self.status is None:
            return
        behind = self.status.linger(control.ABORT_LINGER_S if aborted else None)
        if behind:
            self.log.say(f"Not heard from since the end: {', '.join(behind)}")
        self.status.stop()


class Follower(Host):
    """A part that waits on the driver: a host of a role other than its first."""

    def __init__(
        self,
        plan: Plan,
        options: RunOptions,
        placement: Placement,
        now: Callable[[], datetime.datetime] = _now,
    ) -> None:
        super().__init__(plan, options, placement, now)
        self.follower: control.Follower | None = None
        self.ready = False

    def _pause(self, seconds: float) -> None:
        if self.follower is None:
            time.sleep(seconds)
            return
        verdict = self.follower.wait(seconds)
        if verdict is not None:
            raise Decided(verdict)

    def _check(self) -> None:
        if self.follower is not None and self.follower.verdict is not None:
            raise Decided(self.follower.verdict)

    def run(self) -> int:
        code = exits.OK
        aborted = False
        disable_core_dumps()
        with Signals() as signals:
            try:
                self._setup()
                self._bring_up_part()
                code = self._follow()
            except Decided as decided:
                code = decided.verdict.code
            except Interrupted as stop:
                self.log.say(f"Stopping on signal {signal.Signals(stop.signum).name}")
                if self.follower is not None and self.follower.seen_done:
                    code = exits.OK
                else:
                    aborted = True
                    code = exits.for_signal(stop.signum)
            except RunError as error:
                self.log.error(error.message)
                code = error.code
            finally:
                signals.hold()
                if self.follower is not None:
                    self.follower.stop()
                self._stop_part()
        self._finish(code, aborted)
        return code

    def _setup(self) -> None:
        plan, placement = self.plan, self.placement
        self.log.phase("start")
        address = control.driver_address(plan, placement)
        run_dir = self._run_dir(self._target())
        directory = results.part_dir(run_dir, placement.role, placement.host_index)
        directory.mkdir(parents=True, exist_ok=True)
        self.summary = self._summary(directory)
        self.summary.set(part=placement.part)
        self.follower = control.Follower(
            control.status_url(address, placement.part),
            codec.digest(plan),
            say=self.log.say,
        ).start()
        self._prepare(run_dir, directory)
        self._phase("placement")

    def _follow(self) -> int:
        """Wait for the driver's verdict, or for this host's server to exit."""
        assert self.follower is not None
        while True:
            verdict = self.follower.wait(OWN_SERVER_CHECK_S)
            if verdict is not None:
                return verdict.code
            if self.server is None:
                continue
            exited = self.server.poll()
            if exited is None:
                self._note_ready()
                continue
            self.log.say(f"This host's server exited (exit {exited})")
            verdict = self.follower.server_exited()
            if verdict.code in (exits.NOT_READY, exits.SERVER_DIED):
                bringup.report_tail(self.server, self.log)
                self._command_failed()
            return verdict.code

    def _note_ready(self) -> None:
        """Log once that this host's API answers; the driver is what waits on it."""
        host, role = self.host, self.role
        if self.ready or host is None or role is None or not host.head:
            return
        readiness = role.readiness
        url = f"http://{LOOPBACK}:{readiness.port}{readiness.path}"
        if net.http_status(url, OWN_READY_TIMEOUT_S) == 200:
            self.ready = True
            self._phase("ready")
            assert self.summary is not None
            self.summary.set(state="serving")


def run(
    plan: Plan,
    options: RunOptions,
    executors: Mapping[str, Executor] | None = None,
) -> int:
    """Run this host's part of `plan`; the exit code (runtime/exits.py)."""
    if options.mode == "url":
        at = Placement(role=plan.driver.role or CLIENT_ROLE, host_index=0)
        return Driver(plan, options, at, executors).run()
    try:
        placement = place(plan, options.flags, options.env)
    except RunError as error:
        log = Log("?", codec.digest(plan), __version__)
        log.error(error.message)
        log.phase(f"exit {error.code}")
        return error.code
    part = control.part_name(placement.role, placement.host_index)
    if part == control.driver_part(plan):
        return Driver(plan, options, placement, executors).run()
    return Follower(plan, options, placement).run()
