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
"""Child processes: one session each, output pumped by line, stopped as a group.

Every server, router and task runs in a session of its own, so stopping it
reaches everything it started: vLLM's engine cores stay in their parent's
process group. Nothing outside those sessions is ever signalled, so a run
never kills another server on a shared host.

Output is split on \\n and on \\r: progress bars (lm_eval, tqdm) redraw with
\\r and end their line only when they finish, and a log read by line would
show nothing until then.
"""

import collections
import contextlib
import io
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import IO

# SIGTERM to the group, then this long before SIGKILL.
STOP_GRACE_S = 20
# Lines of a child's output kept for error reports.
TAIL_LINES = 200
# A line longer than this is emitted in pieces, so one huge line cannot hold
# the whole output in memory.
MAX_LINE_BYTES = 1 << 20
# How long to wait for the output pump to drain after a child is gone.
PUMP_JOIN_S = 5

_BREAK = re.compile(rb"\r\n|\r|\n")
_ECHO = threading.RLock()


def echo(line: str) -> None:
    """One line to stdout, never interleaved with another thread's."""
    with _ECHO:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


@contextlib.contextmanager
def capture_output() -> Iterator[io.StringIO]:
    """What this thread prints, apart from the children's output.

    sys.stdout is process-wide, so the output pumps wait on the echo lock
    while it is redirected; a child's lines come out after, not inside.
    """
    buffer = io.StringIO()
    with _ECHO, contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def exit_status(returncode: int) -> int:
    """A child's exit status as a shell reports it: 128 + n for signal n."""
    return 128 - returncode if returncode < 0 else returncode


class LineSplitter:
    """Bytes to lines, each ended by \\n, \\r or \\r\\n."""

    def __init__(self) -> None:
        self._rest = b""
        self._after_cr = False

    def feed(self, data: bytes) -> list[str]:
        # A \r\n split across two reads is one break, not two.
        if self._after_cr and data.startswith(b"\n"):
            data = data[1:]
        text = self._rest + data
        self._after_cr = text.endswith(b"\r")
        parts = _BREAK.split(text)
        self._rest = parts.pop()
        lines = [part.decode("utf-8", "replace") for part in parts]
        while len(self._rest) > MAX_LINE_BYTES:
            piece, self._rest = (
                self._rest[:MAX_LINE_BYTES],
                self._rest[MAX_LINE_BYTES:],
            )
            lines.append(piece.decode("utf-8", "replace"))
        return lines

    def close(self) -> list[str]:
        rest, self._rest = self._rest, b""
        return [rest.decode("utf-8", "replace")] if rest else []


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def session_members(sid: int) -> list[int]:
    """Processes of session `sid` that /proc lists; none where there is no /proc.

    A process that moved to a group of its own is still in its session.
    """
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return []
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may hold spaces and parentheses; fields after it
        # are: state, ppid, pgrp, session.
        fields = stat.rpartition(")")[2].split()
        if len(fields) > 3 and fields[3] == str(sid):
            found.append(int(entry.name))
    return found


class Proc:
    """One child in its own session; its output goes to stdout, a log and a tail.

    `prefix` starts each line on stdout; the log gets the lines as written.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str | os.PathLike[str] | None = None,
        log: pathlib.Path | None = None,
        prefix: str = "",
        echo_output: bool = True,
    ) -> None:
        self.argv = list(argv)
        self.env = dict(env)
        self.cwd = cwd
        self.log = log
        self.prefix = prefix
        self.echo_output = echo_output
        self.popen: subprocess.Popen[bytes] | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._pump: threading.Thread | None = None
        self._log_file: IO[str] | None = None
        self._lock = threading.Lock()

    def start(self) -> "Proc":
        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log.open("a", encoding="utf-8")
        try:
            self.popen = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self._close_log()
            raise
        self._pump = threading.Thread(target=self._read, daemon=True)
        self._pump.start()
        return self

    @property
    def pid(self) -> int:
        assert self.popen is not None, "not started"
        return self.popen.pid

    def poll(self) -> int | None:
        """The exit status once the child has exited, else None."""
        assert self.popen is not None, "not started"
        code = self.popen.poll()
        return None if code is None else exit_status(code)

    def alive(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        """The exit status, or None when `timeout` passes first."""
        assert self.popen is not None, "not started"
        try:
            code = self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        # What the leader left in its group still holds the output pipe open;
        # stop() drains once those are gone.
        if not _group_exists(self.popen.pid):
            self._drain()
        return exit_status(code)

    def tail(self, count: int = TAIL_LINES) -> list[str]:
        with self._lock:
            lines = list(self._tail)
        return lines[-count:]

    def stop(self, grace: float = STOP_GRACE_S) -> int | None:
        """SIGTERM the group, SIGKILL it after `grace`, then any session leftover.

        The leader may exit before the rest of its group (engine cores
        shutting down), so the grace covers the whole group, and what is left
        once the leader has gone gets a second SIGTERM. Returns the leader's
        exit status.
        """
        if self.popen is None:
            return None
        pgid = self.popen.pid
        _signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        resent = False
        while time.monotonic() < deadline:
            leader_gone = self.popen.poll() is not None
            if not _group_exists(pgid):
                break
            # A child forked while the first signal was delivered never got
            # it; once the leader is gone, its stragglers get one more.
            if leader_gone and not resent:
                _signal_group(pgid, signal.SIGTERM)
                resent = True
            time.sleep(0.02)
        else:
            _signal_group(pgid, signal.SIGKILL)
        code = self.popen.wait()
        for pid in session_members(pgid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        self._drain()
        return exit_status(code)

    def _drain(self) -> None:
        if self._pump is not None:
            self._pump.join(PUMP_JOIN_S)
        if self._pump is None or not self._pump.is_alive():
            self._close_log()

    def _close_log(self) -> None:
        with self._lock:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def _emit(self, line: str) -> None:
        with self._lock:
            self._tail.append(line)
            if self._log_file is not None:
                self._log_file.write(line + "\n")
                self._log_file.flush()
        if self.echo_output:
            echo(self.prefix + line)

    def _read(self) -> None:
        assert self.popen is not None and self.popen.stdout is not None
        fd = self.popen.stdout.fileno()
        splitter = LineSplitter()
        try:
            while True:
                data = os.read(fd, 65536)
                if not data:
                    break
                for line in splitter.feed(data):
                    self._emit(line)
            for line in splitter.close():
                self._emit(line)
        finally:
            self.popen.stdout.close()


@dataclass(frozen=True)
class Finished:
    """How a command ended: its exit status, or a timeout."""

    code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out


# What a timed-out command reports, as timeout(1) does.
TIMEOUT_STATUS = 124


def run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str | os.PathLike[str] | None = None,
    log: pathlib.Path | None = None,
    prefix: str = "",
    timeout_s: float | None = None,
) -> Finished:
    """Run a command to its end; on a timeout or any exception, stop its group.

    A timeout reports status 124, as timeout(1) does.
    """
    proc = Proc(argv, env=env, cwd=cwd, log=log, prefix=prefix).start()
    try:
        code = proc.wait(timeout_s)
    except BaseException:
        proc.stop()
        raise
    if code is None:
        proc.stop()
        return Finished(TIMEOUT_STATUS, timed_out=True)
    # The leader is gone; nothing it started may outlive it.
    if _group_exists(proc.pid):
        proc.stop()
    return Finished(code)
