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
"""What a task executor gets from the driver, and what it gives back.

An executor runs one task of the Plan through TaskContext: it fills Refs,
runs commands with the tasks' env, records progress in the summary, and
checks that the server is still alive between runs.
"""

import pathlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from scenario.plan.types import Command, CustomTaskPlan, Plan, TaskPlan
from scenario.runtime import bringup, exits, net, results
from scenario.runtime.exits import RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import Refs
from scenario.runtime.proc import Finished, Proc
from scenario.runtime.proc import run as run_command
from scenario.runtime.publish import Publisher

# A server at a URL counts as alive while GET /health answers within this.
REMOTE_HEALTH_TIMEOUT_S = 10


class ServerDied(RunError):
    """A server or the router died; `reason` is what the summary records."""

    def __init__(self, message: str, reason: str = "the server died") -> None:
        super().__init__(exits.SERVER_DIED, message)
        self.reason = reason


class Liveness:
    """Whether the server the tasks talk to is still there."""

    def check(self) -> None:
        """Raises ServerDied when it is not."""


class LocalServer(Liveness):
    def __init__(self, server: Proc, log: Log, run_dir: pathlib.Path) -> None:
        self.server = server
        self.log = log
        self.run_dir = run_dir

    def check(self) -> None:
        code = self.server.poll()
        if code is None:
            return
        self.log.say("ERROR: Server process died (detected after a completed run).")
        bringup.report_tail(self.server, self.log)
        for path in (self.run_dir, pathlib.Path("/dev/shm")):
            if path.is_dir():
                usage = shutil.disk_usage(path)
                self.log.say(
                    f"disk {path}: {usage.used >> 30} GiB used of "
                    f"{usage.total >> 30} GiB, {usage.free >> 30} GiB free"
                )
        raise ServerDied(f"the server died (exit {code})")


class LocalRouter(Liveness):
    """The router on this host: a dead one stops the tasks (exit 5)."""

    def __init__(self, router: Proc, log: Log) -> None:
        self.router = router
        self.log = log

    def check(self) -> None:
        code = self.router.poll()
        if code is None:
            return
        self.log.say("ERROR: the router died; nothing can reach the servers.")
        bringup.report_tail(self.router, self.log, "router.log")
        raise ServerDied(f"the router died (exit {code})", "the router died")


class Every(Liveness):
    """Alive while every part is."""

    def __init__(self, *parts: Liveness) -> None:
        self.parts = parts

    def check(self) -> None:
        for part in self.parts:
            part.check()


class RemoteServer(Liveness):
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def check(self) -> None:
        status = net.http_status(f"{self.url}/health", REMOTE_HEALTH_TIMEOUT_S)
        if status != 200:
            raise ServerDied(f"{self.url}/health answers {status or 'nothing'}")


@dataclass(frozen=True)
class TaskOutcome:
    status: Literal["passed", "failed"]
    # Why the remaining tasks are skipped, if they are.
    skip_rest: str | None = None


@dataclass
class TaskContext:
    """What an executor gets: the Plan, the run directory, and how to run."""

    plan: Plan
    run_dir: pathlib.Path
    refs: Refs
    # The tasks' env: the host's, then the driver's task env.
    env: dict[str, str]
    # What the driver's task env adds, for spawn logs.
    added: dict[str, str]
    cwd: pathlib.Path
    log: Log
    summary: results.Summary
    liveness: Liveness
    lm_eval_cache: str | None = None
    publisher: Publisher | None = None

    @property
    def experiment(self) -> bool:
        return bool(self.plan.experiment)

    def alive(self) -> None:
        self.liveness.check()

    def command_failed(self) -> None:
        self.summary.command_failed()
        self.summary.save()

    def run(
        self,
        what: str,
        command: Command,
        *,
        log_path: pathlib.Path | None = None,
        timeout_s: float | None = None,
        extra_argv: Sequence[str] = (),
        extra_env: Mapping[str, str] | None = None,
    ) -> Finished:
        """Run one Plan command with its Refs filled; its own env over the tasks'."""
        argv = [*self.refs.argv(command.argv), *extra_argv]
        env = {**self.refs.env(command.env), **(extra_env or {})}
        self.log.spawn(what, argv, {"env": {**self.added, **env}})
        timeout = timeout_s if timeout_s is not None else command.timeout_s
        return run_command(
            argv, env={**self.env, **env}, cwd=self.cwd, log=log_path, timeout_s=timeout
        )


def task_name(task: TaskPlan) -> str:
    return task.name if isinstance(task, CustomTaskPlan) else task.kind
