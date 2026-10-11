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
"""Exit codes of a run, each naming the stage that failed.

A launcher that collects several hosts' codes can tell a server that never
came up (4) from one that died under load (5) and from a task that failed
(1). When several apply, the one closest to the cause wins: a signal, then
a server that was not ready or died, then bring-up, then a lost driver,
then a failed task, then a failed after-task.
"""

OK = 0
TASK_FAILED = 1
USAGE = 2
BRINGUP = 3
NOT_READY = 4
SERVER_DIED = 5
AFTER_TASK = 6
DRIVER_LOST = 7
SIGINT = 130
SIGTERM = 143

MEANINGS = {
    OK: "every task passed (soft gate failures allowed)",
    TASK_FAILED: "a task failed: a command exited non-zero, a hard gate failed, "
    "a result is missing, or a cell had too many failed requests",
    USAGE: "usage, Plan or placement error",
    BRINGUP: "bring-up failed before the server started: a name did not "
    "resolve, a port was taken",
    NOT_READY: "a server was not ready: it died while starting, or the "
    "readiness wait ran out",
    SERVER_DIED: "a server or router died after it was ready",
    AFTER_TASK: "a fatal after-task failed while every task passed",
    DRIVER_LOST: "the driver aborted, or was lost before it finished",
    SIGINT: "interrupted (SIGINT)",
    SIGTERM: "terminated (SIGTERM)",
}

_RANK = {
    OK: 0,
    AFTER_TASK: 1,
    TASK_FAILED: 2,
    DRIVER_LOST: 3,
    BRINGUP: 4,
    USAGE: 4,
    NOT_READY: 5,
    SERVER_DIED: 5,
    SIGINT: 6,
    SIGTERM: 6,
}


def worst(*codes: int) -> int:
    """The code that wins among `codes`; the first of equals."""
    found = OK
    for code in codes:
        if _RANK.get(code, 2) > _RANK.get(found, 2):
            found = code
    return found


def for_signal(signum: int) -> int:
    return 128 + signum


class RunError(Exception):
    """A run cannot go on; `code` is its exit code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
