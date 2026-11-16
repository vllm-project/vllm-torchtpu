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

"""Helpers for the "Yanshu debug here" request-lifecycle log lines.

A learning aid, not for merging. Stages S16 and S17 are logged in
worker/tpu_worker.py; the other stages (S01-S15, S18-S32) are log calls in
vLLM itself. `grep -rn "Yanshu debug here"` lists the sites.
"""

import threading
import time

_seen: set[str] = set()


def first_time(key: str) -> bool:
    """Return True the first time this process asks about `key`."""
    if key in _seen:
        return False
    _seen.add(key)
    return True


def stamp() -> str:
    """Return the time with milliseconds, and the thread name.

    vLLM's own log timestamp has whole seconds, which is too coarse to put
    lines from the API server, EngineCore and worker processes in one order.
    """
    now = time.time()
    clock = time.strftime("%H:%M:%S", time.localtime(now))
    millis = int(now * 1000) % 1000
    return f"t={clock}.{millis:03d} thr={threading.current_thread().name}"
