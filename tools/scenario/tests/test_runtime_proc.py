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
"""Child processes: output by line, group stop with a grace, the PID 1 shim."""

import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

import pytest
from scenario.runtime import proc
from scenario.runtime.proc import LineSplitter, Proc

ENV = dict(os.environ)


def gone(pid: int, within: float = 3.0) -> bool:
    """Whether `pid` no longer exists (reaped) within `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def read_pid(path: pathlib.Path, within: float = 5.0) -> int:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError(f"no pid in {path}")


def test_lines_split_on_every_break() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"a\nb\rc\r") == ["a", "b", "c"]
    # The \n of a \r\n split across reads ends nothing more.
    assert splitter.feed(b"\nd\r\n\ne") == ["d", ""]
    assert splitter.feed("é".encode()[:1]) == []
    assert splitter.feed("é".encode()[1:] + b"\n") == ["eé"]
    assert splitter.close() == []
    splitter.feed(b"tail without a break")
    assert splitter.close() == ["tail without a break"]


def test_a_huge_line_comes_out_in_pieces() -> None:
    splitter = LineSplitter()
    lines = splitter.feed(b"x" * (proc.MAX_LINE_BYTES * 2 + 3))
    assert [len(line) for line in lines] == [proc.MAX_LINE_BYTES] * 2
    assert splitter.close() == ["xxx"]


def test_output_goes_to_stdout_log_and_tail(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "logs" / "child.log"
    child = Proc(
        ["sh", "-c", "printf 'one\\r10%%\\r20%%\\r\\ntwo\\n'; echo $FOO; pwd"],
        env={**ENV, "FOO": "bar"},
        cwd=tmp_path,
        log=log,
        prefix="[c] ",
    ).start()
    assert child.wait(10) == 0
    expected = ["one", "10%", "20%", "two", "bar", str(tmp_path.resolve())]
    assert log.read_text().splitlines()[: len(expected) - 1] == expected[:-1]
    assert child.tail(2)[0] == "bar"
    out = capsys.readouterr().out.splitlines()
    assert out[:5] == [f"[c] {line}" for line in expected[:5]]


def test_the_tail_keeps_the_last_lines(tmp_path: pathlib.Path) -> None:
    child = Proc(
        ["sh", "-c", "i=0; while [ $i -lt 300 ]; do echo $i; i=$((i+1)); done"],
        env=ENV,
        echo_output=False,
    ).start()
    assert child.wait(10) == 0
    tail = child.tail()
    assert len(tail) == proc.TAIL_LINES and tail[-1] == "299"
    assert child.tail(3) == ["297", "298", "299"]


def test_stop_reaches_the_whole_group(tmp_path: pathlib.Path) -> None:
    pidfile = tmp_path / "grandchild"
    child = Proc(
        ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"],
        env=ENV,
        echo_output=False,
    ).start()
    grandchild = read_pid(pidfile)
    assert child.stop(grace=5) == 128 + signal.SIGTERM
    assert gone(grandchild)
    assert not child.alive()


def test_stop_kills_after_the_grace(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    script = textwrap.dedent(
        f"""
        import pathlib, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(ready)!r}).write_text("1")
        time.sleep(60)
        """
    )
    child = Proc([sys.executable, "-c", script], env=ENV, echo_output=False).start()
    read_pid(ready)
    started = time.monotonic()
    assert child.stop(grace=0.5) == 128 + signal.SIGKILL
    assert time.monotonic() - started < 5


def test_stop_reaches_a_group_whose_leader_already_left(tmp_path: pathlib.Path) -> None:
    pidfile = tmp_path / "orphan"
    child = Proc(
        ["sh", "-c", f"sleep 60 & echo $! > {pidfile}"],
        env=ENV,
        echo_output=False,
    ).start()
    orphan = read_pid(pidfile)
    assert child.wait(10) == 0
    child.stop(grace=5)
    assert gone(orphan)


def test_stop_reaches_a_child_forked_after_the_signal(tmp_path: pathlib.Path) -> None:
    ready, late = tmp_path / "ready", tmp_path / "late"
    script = (
        f'trap "sleep 60 & echo \\$! > {late}; exit 0" TERM; '
        f"echo 1 > {ready}; while :; do sleep 0.05; done"
    )
    child = Proc(["sh", "-c", script], env=ENV, echo_output=False).start()
    read_pid(ready)
    started = time.monotonic()
    child.stop(grace=10)
    assert time.monotonic() - started < 5
    assert gone(read_pid(late))


def test_run_reports_the_status_and_leaves_nothing(tmp_path: pathlib.Path) -> None:
    pidfile = tmp_path / "left"
    finished = proc.run(
        ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; exit 3"], env=ENV
    )
    assert finished == proc.Finished(3)
    assert not finished.ok
    assert gone(read_pid(pidfile))


def test_run_times_out(tmp_path: pathlib.Path) -> None:
    started = time.monotonic()
    finished = proc.run(["sleep", "60"], env=ENV, timeout_s=0.3)
    assert finished == proc.Finished(proc.TIMEOUT_STATUS, timed_out=True)
    assert time.monotonic() - started < 10


def test_an_interrupted_run_stops_its_child(tmp_path: pathlib.Path) -> None:
    pidfile = tmp_path / "child"

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        with pytest.raises(KeyboardInterrupt):
            proc.run(["sh", "-c", f"echo $$ > {pidfile}; exec sleep 60"], env=ENV)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert gone(read_pid(pidfile))


def test_exit_status_names_signals() -> None:
    assert proc.exit_status(0) == 0
    assert proc.exit_status(3) == 3
    assert proc.exit_status(-signal.SIGTERM) == 143


SUPERVISED = textwrap.dedent(
    """
    import os, pathlib, signal, sys, time
    from scenario.runtime import init

    state = pathlib.Path(sys.argv[1])
    if sys.argv[2] == "orphan":
        # Stands in for PID 1: orphans of the child are reparented here.
        import ctypes

        PR_SET_CHILD_SUBREAPER = 36
        ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    code = init.supervise()
    if code is not None:
        state.joinpath("parent").write_text(str(code))
        sys.exit(code)

    def on_term(signum, frame):
        state.joinpath("child-term").write_text("1")
        sys.exit(7)

    signal.signal(signal.SIGTERM, on_term)
    state.joinpath("child").write_text(str(os.getpid()))
    if sys.argv[2] == "exit":
        sys.exit(3)
    if sys.argv[2] == "orphan":
        middle = os.fork()
        if middle == 0:
            # The orphan's parent leaves at once; the orphan ends soon after.
            orphan = os.fork()
            if orphan == 0:
                time.sleep(0.2)
                os._exit(0)
            state.joinpath("orphan").write_text(str(orphan))
            os._exit(0)
        os.waitpid(middle, 0)
        time.sleep(1)
        orphan = state.joinpath("orphan").read_text()
        # Reaped: no /proc entry. Not reaped: a zombie entry stays.
        reaped = not pathlib.Path("/proc", orphan).exists()
        sys.exit(0 if reaped else 9)
    time.sleep(60)
    """
)


def supervised(tmp_path: pathlib.Path, mode: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", SUPERVISED, str(tmp_path), mode], env=ENV
    )


def test_the_shim_exits_with_the_childs_status(tmp_path: pathlib.Path) -> None:
    parent = supervised(tmp_path, "exit")
    assert parent.wait(20) == 3
    assert (tmp_path / "parent").read_text() == "3"


def test_the_shim_forwards_sigterm(tmp_path: pathlib.Path) -> None:
    parent = supervised(tmp_path, "wait")
    child = read_pid(tmp_path / "child")
    parent.send_signal(signal.SIGTERM)
    assert parent.wait(20) == 7
    assert (tmp_path / "child-term").is_file()
    assert gone(child)


@pytest.mark.skipif(sys.platform != "linux", reason="needs a Linux subreaper")
def test_the_shim_reaps_orphans(tmp_path: pathlib.Path) -> None:
    assert supervised(tmp_path, "orphan").wait(20) == 0


def test_the_shim_only_runs_as_pid_1() -> None:
    from scenario.runtime import init

    assert os.getpid() != 1
    assert init.maybe_supervise() is None
