# SPDX-License-Identifier: Apache-2.0
"""q38 L6 P-METER (host_levers_h2.md section 4.1 row 1): per-step, per-rank host
launch stamps, a per-window thread census and the DELAY-INJECT classes.

Everything here is host-only and log-only (no compiled program and no call ABI
changes; every flag is in ``_TPU_COMPILE_ENV_IGNORED``). Three l6_ab levers,
read once per worker step through ``core/l6_ab.py`` (values 0 / 1 / ab:<c>):

* ``pmeter`` (VLLM_TPU_L6_PMETER): the HOSTLAUNCH ring and THREADSTAT.
* ``inject`` (VLLM_TPU_L6_INJECT): rank VLLM_TPU_L6_INJECT_RANK (default 0)
  runs a GIL-held busy spin of VLLM_TPU_L6_INJECT_US (default 250) us just
  before the step's first forward dispatch (H2 hunt_3 section 1.4 (d1)).
* ``injectall`` (VLLM_TPU_L6_INJECT_ALL): every rank spins
  VLLM_TPU_L6_INJECT_ALL_US (default 1500) us at the same point ((d2)). In a
  served car the all-rank spin also moves the race between the main thread and
  the async output thread at the resync, so a prereg must say so.
  With both on in one step, the all-rank spin wins (one spin per step).

HOSTLAUNCH. With ``pmeter`` on for a step, ``time.time_ns()`` (wall clock, so
the stamps can be joined to the PJRT VLOG Start lines) is taken at:

* ``deq``: the first statement of ``TPUModelRunner.execute_model``, i.e. right
  after the worker took the step's SchedulerOutput off the RPC queue (the
  worker's ``execute_model`` is a one-line pass-through);
* ``fwd``: the forward enqueue, immediately before the step's first
  ``forward_model`` call (after any inject spin);
* ``w0`` / ``w1``: the start and the return of the blocking previous-step copy
  wait inside ``_modify_prev_results`` (the step's last one);
* ``loop``: the dp_loop enqueue, right after the draft's ``dp_loop_setup``.

The step is keyed by ``l6_ab.step_seq()``, which every rank of a decode
advances in lockstep (they all receive the same SchedulerOutputs). Every
VLLM_TPU_L6_PMETER_WINDOW steps (default 200), when the step whose seq is a
multiple of the window has completed (so at the same seq on every rank), each
rank logs one line:

``PMETER rank=<r> host=<h> pid=<p> window=<W> seq0=<s> n=<k> t0_ns=<ns> e=<e1>;<e2>;...``

with one entry per seq ``s .. s + W - 1``: ``d,f,a,b,l`` in integer
microseconds, ``d`` = deq - t0 (t0 = the window's first stamped deq) and
``f`` / ``a`` / ``b`` / ``l`` = fwd / w0 / w1 / loop - deq; ``-`` = not stamped
in that step, ``x`` = ``pmeter`` was off for that step. Ranks on two hosts do
not share a clock, so the reader aligns ranks by seq and compares step-relative
durations (and the absolute stamps only within one host).

THREADSTAT, right after each PMETER line (Linux ``/proc``; ``-`` elsewhere):

``THREADSTAT rank=<r> seq=<s> n=<k> <name>:<tid>=<run_ns>/<wait_ns>/<slices>/<vcsw>/<nvcsw> ...``

for every live Python thread (main, the async output thread, the KV
connector's threads, ...): ``/proc/self/task/<tid>/schedstat`` (on-CPU ns,
run-queue wait ns, time slices) and the voluntary / nonvoluntary context
switches from ``status``; cumulative, so a reader takes window deltas.
Threads that Python did not start (torch_tpu's workers) are not named until
torch_tpu names them, so they are not listed.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_NF = 6  # ring fields per step: seq, deq, fwd, w0, w1, loop


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


WINDOW = 200
INJECT_US = 250
INJECT_RANK = 0
INJECT_ALL_US = 1500
_ring: list[int] = []
_cur: list[int] = [0, 0, -1, -1, -1, -1]
_st: dict = {}


def _resolve() -> None:
    """(Re)read the parameters; tests set os.environ and call it."""
    global WINDOW, INJECT_US, INJECT_RANK, INJECT_ALL_US, _ring, _cur
    WINDOW = _env_int("VLLM_TPU_L6_PMETER_WINDOW", 200)
    if WINDOW < 1:
        raise ValueError("VLLM_TPU_L6_PMETER_WINDOW must be >= 1")
    INJECT_US = _env_int("VLLM_TPU_L6_INJECT_US", 250)
    INJECT_RANK = _env_int("VLLM_TPU_L6_INJECT_RANK", 0)
    INJECT_ALL_US = _env_int("VLLM_TPU_L6_INJECT_ALL_US", 1500)
    _ring = [0] * (WINDOW * _NF)
    _cur = [0, 0, -1, -1, -1, -1]
    _st.clear()
    _st.update(on=False, deq=0, spin=0, rank=None, host=None)


_resolve()


def _rank() -> int:
    if _st["rank"] is None:
        _st["rank"] = l6_ab._rank_world()[0]
    return _st["rank"]


def deq() -> None:
    """First statement of execute_model: the SchedulerOutput dequeue stamp."""
    if l6_ab.ACTIVE:
        _st["deq"] = time.time_ns()


def on_step() -> None:
    """After l6_ab.begin_step(): close the previous step's entry (and log the
    window it completes), open this step's entry, and fix the inject spin."""
    if _cur[0]:
        i = ((_cur[0] - 1) % WINDOW) * _NF
        _ring[i:i + _NF] = _cur
        _cur[0] = 0
    prev = l6_ab.step_seq() - 1
    if prev > 0 and prev % WINDOW == 0:
        _emit(prev)
    on = l6_ab.on("pmeter")
    _st["on"] = on
    if on:
        _cur[:] = [l6_ab.step_seq(), _st["deq"], -1, -1, -1, -1]
    spin = 0
    if l6_ab.on("injectall"):
        spin = INJECT_ALL_US
    elif l6_ab.on("inject") and _rank() == INJECT_RANK:
        spin = INJECT_US
    _st["spin"] = spin


def _spin(us: int) -> None:
    """A GIL-held busy spin: Python bytecode in a loop (the interpreter only
    hands the GIL over at its switch interval, as for any host work)."""
    end = time.perf_counter_ns() + us * 1000
    while time.perf_counter_ns() < end:
        pass


def before_forward() -> None:
    """Immediately before the step's forward_model call: the inject spin (once
    per step) and the forward enqueue stamp (the step's first forward)."""
    spin = _st["spin"]
    if spin:
        _st["spin"] = 0
        _spin(spin)
        l6_ab.note("inject", "spins")
        l6_ab.note("inject", "us", spin)
    if _st["on"] and _cur[2] < 0:
        _cur[2] = time.time_ns()


def wait_begin() -> None:
    if _st["on"]:
        _cur[3] = time.time_ns()


def wait_end() -> None:
    if _st["on"]:
        _cur[4] = time.time_ns()


def loop_enqueue() -> None:
    if _st["on"] and _cur[5] < 0:
        _cur[5] = time.time_ns()


def _us(v: int, base: int) -> str:
    return "-" if v < 0 else str((v - base) // 1000)


def window_line(seq_end: int) -> str:
    """The PMETER line of the window ending at seq_end (also used by tests)."""
    first = seq_end - WINDOW + 1
    t0 = None
    parts = []
    for s in range(first, seq_end + 1):
        i = ((s - 1) % WINDOW) * _NF
        if _ring[i] != s:
            parts.append("x")
            continue
        d = _ring[i + 1]
        if t0 is None:
            t0 = d
        parts.append(",".join((str((d - t0) // 1000), _us(_ring[i + 2], d), _us(_ring[i + 3], d),
                               _us(_ring[i + 4], d), _us(_ring[i + 5], d))))
    if _st["host"] is None:
        _st["host"] = socket.gethostname()
    return (f"PMETER rank={_rank()} host={_st['host']} pid={os.getpid()} window={WINDOW} seq0={first} "
            f"n={sum(p != 'x' for p in parts)} t0_ns={t0 if t0 is not None else '-'} e={';'.join(parts)}")


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def thread_line(seq: int) -> str:
    parts = []
    for t in threading.enumerate():
        tid = getattr(t, "native_id", None)
        name = _SAFE.sub("_", t.name)
        stat = _read(f"/proc/self/task/{tid}/schedstat") if tid is not None else None
        status = _read(f"/proc/self/task/{tid}/status") if tid is not None else None
        if stat is None or status is None:
            parts.append(f"{name}:{tid}=-")
            continue
        run_ns, wait_ns, slices = (stat.split() + ["-", "-", "-"])[:3]
        vol = nvol = "-"
        for ln in status.splitlines():
            if ln.startswith("voluntary_ctxt_switches:"):
                vol = ln.split()[1]
            elif ln.startswith("nonvoluntary_ctxt_switches:"):
                nvol = ln.split()[1]
        parts.append(f"{name}:{tid}={run_ns}/{wait_ns}/{slices}/{vol}/{nvol}")
    return f"THREADSTAT rank={_rank()} seq={seq} n={len(parts)} " + " ".join(parts)


def _emit(seq_end: int) -> None:
    first = seq_end - WINDOW + 1
    if not any(_ring[((s - 1) % WINDOW) * _NF] == s for s in range(first, seq_end + 1)):
        return
    logger.info("%s", window_line(seq_end))
    logger.info("%s", thread_line(seq_end))
    for k in range(0, len(_ring), _NF):
        _ring[k] = 0
