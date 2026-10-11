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
"""One host's run: the server, then the driver's tasks, then teardown.

This version runs single-host, single-role Plans, where the one host both
serves and drives, and the task verbs against a server at a URL. Hosts that
wait on a driver elsewhere come with the driver status endpoint.

Tasks run in the Plan's order. A failed task does not stop the next one; a
dead server does (exit 5), and so does a custom task's skip_rest_if_exit_ge.
SIGTERM or SIGINT stops every child (20 s, then SIGKILL) and exits 143 or
130 without after-tasks.
"""

import datetime
import pathlib
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scenario import __version__
from scenario.plan import codec
from scenario.plan.types import CLIENT_ROLE, Plan
from scenario.runtime import bringup, exec_perf, exits, results
from scenario.runtime.context import (
    Liveness,
    LocalServer,
    RemoteServer,
    ServerDied,
    TaskContext,
    TaskOutcome,
    task_name,
)
from scenario.runtime.exits import RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import (
    ENV_IMAGE,
    ENV_RESOLVED,
    ENV_TARGET_DATA,
    Placement,
    PlacementFlags,
    Refs,
    decode_resolved,
    decode_target_data,
    place,
)
from scenario.runtime.proc import Proc
from scenario.target import TargetData, TargetInfo

Mode = Literal["run", "up", "url"]


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
EXECUTORS: dict[str, Executor] = {"perf": exec_perf.run_perf}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Driver:
    def __init__(
        self,
        plan: Plan,
        options: RunOptions,
        executors: Mapping[str, Executor] | None = None,
        now: Callable[[], datetime.datetime] = _now,
    ) -> None:
        self.plan = plan
        self.options = options
        self.executors = dict(EXECUTORS if executors is None else executors)
        self.started = now()
        self.log = Log("?", codec.digest(plan), __version__)
        self.summary: results.Summary | None = None
        self.server: Proc | None = None
        self.liveness: Liveness = Liveness()

    def run(self) -> int:
        code = exits.OK
        aborted = False
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
        self._finish(code, aborted)
        return code

    # Setup -------------------------------------------------------------

    def _setup(self) -> None:
        plan, options = self.plan, self.options
        env = options.env
        self.placement = self._place()
        self.log.part = self.placement.part
        self.log.phase("start")
        target = options.target
        if target is None and env.get(ENV_TARGET_DATA):
            target = decode_target_data(env[ENV_TARGET_DATA])
        self.target = target or TargetData(TargetInfo("none", None, None))
        resolved = options.resolved
        if resolved is None and env.get(ENV_RESOLVED):
            resolved = decode_resolved(env[ENV_RESOLVED])
        self.resolved = resolved
        self.image = env.get(ENV_IMAGE) or plan.image or ""
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        root = results.results_root(
            options.results_root,
            self.target.info.results_root,
            plan.scenario_name,
            stamp,
        )
        self.root = root
        self.run_dir = results.run_dir(root, plan)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary = results.Summary(
            self.run_dir / results.SUMMARY,
            plan,
            self.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self.summary.save()
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
        self.refs = Refs(
            plan,
            self.placement,
            run_dir=self.run_dir,
            server_log=self.run_dir / results.SERVER_LOG,
            env=env,
            warn=self.log.say,
        )
        self.host_env: dict[str, str] = dict(env)
        self.host_added: dict[str, dict[str, str]] = {}
        if options.mode != "url":
            host = self.plan.role(self.placement.role).host_plans[
                self.placement.host_index
            ]
            self.host = host
            self.host_env, scenario_env, tool_env = bringup.host_env(
                env, host, self.refs
            )
            self.host_added = {"scenario env": scenario_env, "tool env": tool_env}
        self.log.phase("placement")
        self.summary.phase("placement")

    def _place(self) -> Placement:
        plan = self.plan
        if self.options.mode == "url":
            role = plan.driver.role or CLIENT_ROLE
            return Placement(role=role, host_index=0)
        placement = place(plan, self.options.flags, self.options.env)
        role = placement.role
        if (
            len(plan.roles) != 1
            or plan.driver.chipless
            or role == CLIENT_ROLE
            or plan.role(role).hosts != 1
        ):
            raise RunError(
                exits.USAGE,
                "this version runs single-host, single-role plans; bring-up over "
                "several hosts or roles comes with the driver status endpoint",
            )
        return placement

    # Bring-up ----------------------------------------------------------

    def _bring_up(self) -> None:
        assert self.summary is not None
        role = self.plan.role(self.placement.role)
        self.log.phase("server-spawn")
        self.summary.phase("server-spawn")
        spawned_at = time.monotonic()
        self.server = bringup.start_server(
            self.host,
            role.readiness,
            refs=self.refs,
            env=self.host_env,
            added=self.host_added,
            cwd=self.options.repo_root,
            log_path=self.run_dir / results.SERVER_LOG,
            log=self.log,
        )
        self.liveness = LocalServer(self.server, self.log, self.run_dir)
        bringup.wait_ready(
            self.server, role.readiness, spawned_at=spawned_at, log=self.log
        )
        self.log.phase("ready")
        self.summary.phase("ready")
        self.summary.set(state="serving")

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
        """Serve until the server exits (exit 5) or a signal asks to stop."""
        assert self.server is not None
        port = self.plan.role(self.placement.role).readiness.port
        self.log.say(
            f"Serving {self.plan.model_name} at http://127.0.0.1:{port}; "
            "SIGINT or SIGTERM stops it"
        )
        try:
            while True:
                code = self.server.wait(1.0)
                if code is not None:
                    raise ServerDied(f"the server exited (exit {code})")
        except Interrupted as stop:
            self.signals.consume()
            self.log.say(f"Stopping on signal {signal.Signals(stop.signum).name}")

    # Tasks -------------------------------------------------------------

    def _context(self) -> TaskContext:
        assert self.summary is not None
        added = self.refs.env(self.plan.driver.task_env)
        return TaskContext(
            plan=self.plan,
            run_dir=self.run_dir,
            refs=self.refs,
            env={**self.host_env, **added},
            added=added,
            cwd=self.options.repo_root,
            log=self.log,
            summary=self.summary,
            liveness=self.liveness,
            lm_eval_cache=self.options.lm_eval_cache or self.target.lm_eval_cache_root,
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
                why = (
                    "the server died" if isinstance(stop, ServerDied) else "interrupted"
                )
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

    def _command_failed(self) -> None:
        if self.summary is not None:
            self.summary.command_failed()

    def _teardown(self) -> None:
        if self.server is None:
            return
        self.log.phase("stopping")
        self.server.stop()

    def _end(self, code: int) -> int:
        """Work after teardown, on normal and failure exits alike."""
        return code

    def _finish(self, code: int, aborted: bool) -> None:
        if self.summary is not None:
            self.summary.data["phases"]["exit"] = self.summary.elapsed()
            self.summary.set(state="aborted" if aborted else "done", exit=code)
        self.log.phase(f"exit {code}")


def run(
    plan: Plan,
    options: RunOptions,
    executors: Mapping[str, Executor] | None = None,
) -> int:
    """Run this host's part of `plan`; the exit code (runtime/exits.py)."""
    return Driver(plan, options, executors).run()
