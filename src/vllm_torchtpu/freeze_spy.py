# SPDX-License-Identifier: Apache-2.0
"""Out-of-process native stack dumps on a serving freeze (VLLM_TPU_FREEZE_PYSPY_S=N).

The in-process watchdogs (stall_watchdog.py) run Python code in the stalled
process: a thread that needs the GIL, or faulthandler, which prints Python
frames only. When a C call holds the GIL, neither shows where the process
is. With VLLM_TPU_STALL_WATCHDOG_S > 0 and N > 0, the API server starts this
module as a separate process (its own session, no GIL shared with vLLM):

- heartbeats: the API server's event-loop watch and every worker's step
  watch write ``<wall time> <live requests> <last progress wall time>`` to
  ``$VLLM_TPU_FREEZE_DIR/<role>.hb`` about once a second;
- trigger: a heartbeat file not rewritten for N s (its process cannot run
  Python), or a worker reporting live requests with no step for N s, or the
  API server's loop lagging N s;
- dump, once per freeze plus a second dump N s later (to see whether the
  stacks move): per-thread state and CPU time from /proc for the API server
  and all its descendants, then ``py-spy dump --native`` of each, written
  to the server log (stderr) and to ``$CDK_OUTPUT_DIR/freeze_spy/``.

Every vLLM process allows any process to ptrace it (PR_SET_PTRACER_ANY), so
py-spy works under Yama ptrace_scope 1 without CAP_SYS_PTRACE.
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time

PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value


def seconds() -> float:
    try:
        return float(os.environ.get("VLLM_TPU_FREEZE_PYSPY_S", "120") or 0)
    except ValueError:
        return 0.0


def allow_ptrace() -> None:
    """Let the monitor (not an ancestor of this process) attach py-spy."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def heartbeat(role: str, live: int, progress_wall: float) -> None:
    """Write this process's heartbeat (cheap; called ~once a second)."""
    directory = os.environ.get("VLLM_TPU_FREEZE_DIR")
    if not directory:
        return
    path = os.path.join(directory, f"{role}.hb")
    try:
        with open(path + ".tmp", "w") as f:
            f.write(f"{time.time():.3f} {live} {progress_wall:.3f} {os.getpid()}\n")
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def launch(limit: float) -> None:
    """API server: create the heartbeat directory (children inherit it via the
    environment) and start the monitor in its own session."""
    if os.environ.get("VLLM_TPU_FREEZE_DIR"):
        return
    pid = os.getpid()
    directory = f"/dev/shm/vllm-tpu-freeze-{pid}"
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return
    os.environ["VLLM_TPU_FREEZE_DIR"] = directory
    out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "freeze_spy")
    cmd = [
        # Run the file, not the package: importing vllm_torchtpu would load
        # torch and the plugin into the monitor.
        sys.executable, "-I", os.path.abspath(__file__),
        "--pid", str(pid), "--dir", directory, "--limit", str(limit), "--out", out,
    ]
    try:
        subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
            start_new_session=True, close_fds=True,
        )
    except OSError as e:
        print(f"Freeze spy: cannot start the monitor: {e}", file=sys.stderr, flush=True)
        return
    print(
        f"Freeze spy: monitor watches pid {pid} heartbeats in {directory} "
        f"(py-spy --native dump after {limit:.0f} s without progress)",
        file=sys.stderr, flush=True,
    )


# ---------------------------------------------------------------- monitor


def _children(pid: int) -> list[int]:
    out = []
    try:
        for task in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{task}/children") as f:
                out += [int(c) for c in f.read().split()]
    except OSError:
        pass
    return out


def _tree(root: int) -> list[int]:
    pids, todo = [], [root]
    while todo:
        pid = todo.pop()
        pids.append(pid)
        todo += _children(pid)
    return pids


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _name(pid: int) -> str:
    return _read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()[:120] or "?"


def _threads(pid: int) -> list[tuple[int, str, str, int]]:
    """``(tid, comm, state, utime+stime ticks)`` of every thread."""
    rows = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return rows
    for tid in tids:
        stat = _read(f"/proc/{pid}/task/{tid}/stat")
        if not stat:
            continue
        comm = stat[stat.find("(") + 1 : stat.rfind(")")]
        fields = stat[stat.rfind(")") + 2 :].split()
        try:
            rows.append((int(tid), comm, fields[0], int(fields[11]) + int(fields[12])))
        except (IndexError, ValueError):
            continue
    return rows


def _proc_report(pids: list[int], interval: float = 2.0) -> str:
    """Per-thread state and CPU use over ``interval`` s: shows a pegged
    thread (R, ~100%) or threads stuck in the kernel (D)."""
    before = {pid: {t[0]: t for t in _threads(pid)} for pid in pids}
    time.sleep(interval)
    hz = os.sysconf("SC_CLK_TCK")
    lines = []
    for pid in pids:
        rows = _threads(pid)
        busy = []
        states: dict[str, int] = {}
        for tid, comm, state, ticks in rows:
            states[state] = states.get(state, 0) + 1
            old = before.get(pid, {}).get(tid)
            pct = 100.0 * (ticks - old[3]) / hz / interval if old else 0.0
            wchan = _read(f"/proc/{pid}/task/{tid}/wchan")
            if pct >= 5 or state == "D":
                busy.append(f"    tid {tid} {comm!r} {state} {pct:.0f}% wchan={wchan}")
        lines.append(
            f"  pid {pid} [{_name(pid)}] threads={len(rows)} states={states}"
        )
        lines += busy
    return "\n".join(lines)


def _heartbeats(directory: str) -> dict[str, tuple[float, int, float, float]]:
    """``role -> (written wall, live, progress wall, file age s)``."""
    beats = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return beats
    now = time.time()
    for name in names:
        if not name.endswith(".hb"):
            continue
        parts = _read(os.path.join(directory, name)).split()
        try:
            wall, live, progress = float(parts[0]), int(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            continue
        beats[name[:-3]] = (wall, live, progress, now - wall)
    return beats


def _stalled(beats, limit: float) -> list[str]:
    now = time.time()
    reasons = []
    for role, (wall, live, progress, age) in sorted(beats.items()):
        if age >= limit:
            reasons.append(f"{role}: heartbeat not written for {age:.0f} s")
        elif progress and live > 0 and now - progress >= limit:
            reasons.append(f"{role}: no progress for {now - progress:.0f} s with {live} live")
    return reasons


def _dump(root: int, reasons: list[str], out: str, tag: str) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    pids = [p for p in _tree(root) if p != os.getpid()]
    header = [
        f"===== FREEZE SPY {tag} {stamp} UTC root={root} =====",
        *[f"  trigger: {r}" for r in reasons],
        _proc_report(pids),
    ]
    text = ["\n".join(header)]
    spy = shutil.which("py-spy")
    for pid in pids:
        if spy is None:
            text.append("  py-spy not found")
            break
        try:
            res = subprocess.run(
                [spy, "dump", "--native", "--pid", str(pid)],
                capture_output=True, text=True, timeout=180,
            )
            body = res.stdout + (res.stderr if res.returncode else "")
        except subprocess.TimeoutExpired:
            body = "py-spy dump timed out after 180 s\n"
        text.append(f"----- py-spy --native pid {pid} [{_name(pid)}] -----\n{body}")
    text.append(f"===== FREEZE SPY {tag} end =====")
    report = "\n".join(text)
    print(report, file=sys.stderr, flush=True)
    try:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"freeze_{stamp}_{tag}.txt"), "w") as f:
            f.write(report)
    except OSError:
        pass


def _alive(pid: int) -> bool:
    stat = _read(f"/proc/{pid}/stat")
    return bool(stat) and stat[stat.rfind(")") + 2 :].split()[0] not in ("Z", "X")


def monitor(root: int, directory: str, limit: float, out: str) -> None:
    period = min(5.0, limit / 4)
    frozen_since = 0.0
    dumps = 0
    while _alive(root):
        time.sleep(period)
        beats = _heartbeats(directory)
        if "api" not in beats:
            continue  # the loop watch starts with the first request
        reasons = _stalled(beats, limit)
        if not reasons:
            if frozen_since:
                print(
                    f"FREEZE SPY: progress resumed after {time.time() - frozen_since:.0f} s",
                    file=sys.stderr, flush=True,
                )
            frozen_since, dumps = 0.0, 0
            continue
        if not frozen_since:
            frozen_since = time.time()
        # First dump at the trigger, a second one `limit` s later.
        if dumps < 2 and time.time() - frozen_since >= dumps * limit:
            dumps += 1
            _dump(root, reasons, out, f"dump{dumps}")
    try:
        shutil.rmtree(directory)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    monitor(args.pid, args.dir, args.limit, args.out)


if __name__ == "__main__":
    main()
