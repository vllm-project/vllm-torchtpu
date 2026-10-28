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
"""Timed export spans, logged as `trace-start-event` / `trace-end-event` lines.

The line format is the one CDK's job trace generator parses into a Perfetto
trace, so export runs on CDK get a per-rank timeline without extra setup; in
any other log the lines still timestamp each stage.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
_HOST = socket.gethostname()


def _pid() -> str:
    rank = os.environ.get("RANK")
    process = "export" if rank is None else f"export-rank{rank}"
    return f"{_HOST}/{process}/{os.getpid()}"


def _emit(kind: str, name: str) -> None:
    sys.stderr.write(
        f"trace-{kind}-event: name={name} ts={time.time()} pid={_pid()} "
        f"tid={threading.get_ident()}\n"
    )
    sys.stderr.flush()


class trace(contextlib.ContextDecorator):
    """Logs a span around a block (`with trace("x"):`) or a function (`@trace()`).

    A decorated function without an explicit name is logged as
    `export.<qualified function name>`. The end event is logged even when the
    block raises.
    """

    def __init__(self, name: str | None = None) -> None:
        if name is not None and any(c.isspace() for c in name):
            raise ValueError(f"trace name must not contain whitespace: {name!r}")
        self.name = name

    def __call__(self, func: _F) -> _F:
        if self.name is None:
            return trace(f"export.{func.__qualname__}")(func)
        return super().__call__(func)  # type: ignore[return-value]

    def __enter__(self) -> trace:
        if self.name is None:
            raise ValueError("trace used as a context manager needs a name")
        _emit("start", self.name)
        return self

    def __exit__(self, *exc: object) -> bool:
        assert self.name is not None
        _emit("end", self.name)
        return False
