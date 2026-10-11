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
"""The runtime's own lines: phases, spawns and messages.

A phase line carries the UTC time, the seconds since the runtime started
and the host's role/index, so the hosts of one run line up in a merged log;
its `--- ` start makes CI log viewers fold the output into sections. A spawn
names the command, the env the runtime adds and the Plan it came from;
inherited env is never printed.
"""

import datetime
import shlex
import time
from collections.abc import Callable, Mapping, Sequence

from scenario.runtime.proc import echo


def _utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _env_text(env: Mapping[str, str]) -> str:
    if not env:
        return "none"
    return " ".join(
        f"{name}={shlex.quote(value) if value else ''}"
        for name, value in sorted(env.items())
    )


class Log:
    def __init__(
        self,
        part: str,
        plan_sha: str,
        tool_version: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime.datetime] = _utc,
    ) -> None:
        self.part = part
        self.plan_sha = plan_sha
        self.tool_version = tool_version
        self._clock = clock
        self._wall = wall
        self._start = clock()

    def phase(self, name: str) -> None:
        stamp = self._wall().strftime("%Y-%m-%dT%H:%M:%SZ")
        elapsed = self._clock() - self._start
        echo(f"--- [scenario] {stamp} +{elapsed:.1f}s {self.part} {name}")

    def say(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            echo(line)

    def warning(self, text: str) -> None:
        echo(f"::warning::{text}")

    def error(self, text: str) -> None:
        echo(f"::error::{text}")

    def spawn(
        self,
        what: str,
        argv: Sequence[str],
        env: Mapping[str, Mapping[str, str]],
    ) -> None:
        """`env` maps a label (scenario env, tool env, env) to what it adds."""
        origin = f"plan {self.plan_sha}, tool {self.tool_version}"
        lines = [f"[spawn] {self.part} {what} ({origin})", f"  {shlex.join(argv)}"]
        lines += [f"  {label}: {_env_text(values)}" for label, values in env.items()]
        for line in lines:
            echo(line)
