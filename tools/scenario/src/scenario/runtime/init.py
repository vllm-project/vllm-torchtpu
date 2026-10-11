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
"""The PID 1 shim: when the runtime is a container's first process.

A container's PID 1 inherits every orphaned process in it and must reap
them, or they stay as zombies. The kernel also gives PID 1 no default signal
actions, so a SIGTERM it does not handle is ignored. The runtime therefore
forks when it is PID 1: the parent forwards signals to the child, reaps
every process that ends up as its child, and exits with the child's status.
"""

import contextlib
import os
import signal

FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def supervise() -> int | None:
    """Fork; the parent reaps and forwards, the child carries on.

    Returns None in the child. In the parent, returns the child's exit
    status (128 + n for signal n) once the child has exited.
    """
    child = os.fork()
    if child == 0:
        return None

    def forward(signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signum)

    for sig in FORWARDED:
        signal.signal(sig, forward)
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            return 1
        if pid == child:
            code = os.waitstatus_to_exitcode(status)
            return 128 - code if code < 0 else code


def maybe_supervise() -> int | None:
    """supervise() when this process is PID 1, else None."""
    if os.getpid() != 1:
        return None
    return supervise()
