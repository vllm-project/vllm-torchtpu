# SPDX-License-Identifier: Apache-2.0
"""q38 L4 host sub-mark instrument: the HOSTSUBS sub-phase census
(VLLM_TPU_HOST_PHASES_SUBS=1 with VLLM_TPU_HOST_PHASES=N > 0).

Every N steps each rank logs, on its own HOSTSUBS line, every ``sub`` /
``charge`` sub-phase as p50 / p90 (over the steps that had it) / mean per step
(over all N steps) / step count. Sub-phases split a phase, so the HOSTPHASES
line must be byte-identical with the census on or off, and with the flag unset
nothing is recorded.
"""

import os
import re
import subprocess
import sys

from vllm_torchtpu.runner import host_phases


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


class _Clock:
    """Virtual time: reads do not advance it; the test advances it."""

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


def _reset(monkeypatch, every, subs, slow_ms=0.0):
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_SLOW_NS", int(slow_ms * 1e6))
    monkeypatch.setattr(host_phases, "_SUBS", subs)
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_cur", {})
    monkeypatch.setattr(host_phases, "_ctx", [])
    monkeypatch.setattr(host_phases, "_sub_last", None)
    monkeypatch.setattr(host_phases, "_cur_wall", 0.0)
    monkeypatch.setattr(host_phases, "_sub_acc", {})
    monkeypatch.setattr(host_phases, "_sub_samples", {})
    monkeypatch.setattr(host_phases, "_rank", 3)
    log = _Log()
    monkeypatch.setattr(host_phases, "logger", log)
    clock = _Clock()
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", clock)
    return log, clock


def _steps7(clock):
    """7 steps of phases a (with sub-phases) and b, then one more begin_step,
    so with N=4 the second window holds exactly steps 3..6 (the first
    begin_step only starts the clock, as in the runner). Step i: idle 100 us;
    in phase a, sub s1 takes 10*((i % 4) + 1) us, sub s2 20 us, then 5 us more
    of a; phase b 50 us. Sub s3 is a 7 us charge in steps 4 and 6 only."""
    for i in range(7):
        host_phases.begin_step()
        clock.tick(10 * ((i % 4) + 1))
        host_phases.sub("s1")
        clock.tick(20)
        host_phases.sub("s2")
        if i in (4, 6):
            host_phases.charge("s3", 7_000)
        clock.tick(5)
        host_phases.mark("a")
        clock.tick(50)
        host_phases.mark("b")
        clock.tick(100)
    host_phases.begin_step()


def test_census_line_values(monkeypatch):
    log, clock = _reset(monkeypatch, every=4, subs=True)
    _steps7(clock)
    subs = [ln for ln in log.lines if ln.startswith("HOSTSUBS")]
    assert len(subs) == 2, log.lines
    # Window 1 (steps 0..2): s1 = 10, 20, 30 us; mean per step over N=4.
    assert subs[0] == (
        "HOSTSUBS rank=3 steps=4 per-step sub-phase host us "
        "(p50/p90/mean-per-step/n-steps) ~s1=20/28/15/3 ~s2=20/20/15/3"
    ), subs[0]
    # Window 2 (steps 3..6): s1 = 40, 10, 20, 30 us: p50 25, p90 37, mean 25, 4
    # steps; s2 = 20 x4; s3 = 7 us in 2 of 4 steps: mean 14 / 4 = 3.5 -> "4".
    assert subs[1] == (
        "HOSTSUBS rank=3 steps=8 per-step sub-phase host us "
        "(p50/p90/mean-per-step/n-steps) ~s1=25/37/25/4 ~s2=20/20/20/4 ~s3=7/7/4/2"
    ), subs[1]
    phases = [ln for ln in log.lines if ln.startswith("HOSTPHASES")]
    assert len(phases) == 2 and not any("~" in ln for ln in phases), phases


def test_hostphases_line_is_identical_with_the_census_on_or_off(monkeypatch):
    log_off, clock = _reset(monkeypatch, every=4, subs=False)
    _steps7(clock)
    off = [ln for ln in log_off.lines if ln.startswith("HOST")]
    log_on, clock = _reset(monkeypatch, every=4, subs=True)
    _steps7(clock)
    on_phases = [ln for ln in log_on.lines if ln.startswith("HOSTPHASES")]
    assert off == on_phases, (off, on_phases)
    assert len(off) == 2 and not any(ln.startswith("HOSTSUBS") for ln in off)
    # Window 2: a = s1 + s2 + 5 us = 35 / 45 / 55 / 65; the sub-phases are
    # inside a and not added to the total (idle 100 + a 50 + b 50).
    assert " total_mean=200 " in off[1], off[1]
    assert "a=50/62/50" in off[1] and "b=50/50/50" in off[1], off[1]


def test_off_records_nothing(monkeypatch):
    # Census off, with the slow dump off and on: nothing accumulates (no leak).
    for slow_ms in (0.0, 0.1):
        log, clock = _reset(monkeypatch, every=4, subs=False, slow_ms=slow_ms)
        _steps7(clock)
        assert host_phases._sub_acc == {} and host_phases._sub_samples == {}, slow_ms
        assert not any(ln.startswith("HOSTSUBS") for ln in log.lines)


def test_census_and_slow_dump_together(monkeypatch):
    # 0.1 ms threshold: every step is slow; HOSTSLOW keeps its ~ entries and the
    # census sees the same sub-phases.
    log, clock = _reset(monkeypatch, every=4, subs=True, slow_ms=0.1)
    _steps7(clock)
    slow = [ln for ln in log.lines if ln.startswith("HOSTSLOW")]
    # Step 0's record has no idle (85 us < 100 us); steps 1..6 are logged.
    assert len(slow) == 6 and all("~s1=" in ln and "~s2=" in ln for ln in slow), slow
    assert any(ln.startswith("HOSTSUBS") and "~s1=25/37/25/4" in ln for ln in log.lines)


_PROBE = (
    "from vllm_torchtpu.runner import host_phases as h; "
    "print('SUBS', h._SUBS, 'EVERY', h._EVERY)"
)


def _probe(extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VLLM_TPU_HOST_PHASES", "VLLM_TPU_HOST_PHASES_SUBS")
    }
    env.update(extra)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("SUBS ")]
    assert len(line) == 1, out.stdout[-2000:]
    return line[0]


def test_fresh_process_flag_gating():
    assert _probe({}) == "SUBS False EVERY 0"
    assert _probe({"VLLM_TPU_HOST_PHASES": "64"}) == "SUBS False EVERY 64"
    # Without host phases the census stays off (nothing would ever flush it).
    assert _probe({"VLLM_TPU_HOST_PHASES_SUBS": "1"}) == "SUBS False EVERY 0"
    assert (
        _probe({"VLLM_TPU_HOST_PHASES": "64", "VLLM_TPU_HOST_PHASES_SUBS": "1"})
        == "SUBS True EVERY 64"
    )


def _src(rel):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    with open(os.path.join(root, "src", "vllm_torchtpu", rel)) as f:
        return f.read()


def _order(text, needles):
    pos = []
    for n in needles:
        assert text.count(n) == 1, (n, text.count(n))
        pos.append(text.index(n))
    assert pos == sorted(pos), list(zip(needles, pos))


def test_sub_marks_sit_inside_their_phases():
    """Each new sub-mark lies between the mark that opens its phase and the
    mark that closes it (a sub charges the time since the last mark or sub)."""
    runner = _src("runner/tpu_runner.py")
    _order(runner, [
        'host_phases.mark("forward_dispatch")',
        'host_phases.sub("pf_counters")',
        'host_phases.sub("pf_logits")',
        'host_phases.mark("post_forward")',
    ])
    _order(runner, [
        'host_phases.mark("modify_prev_results")',
        'host_phases.sub("st_kvconn_out")',
        'host_phases.sub("st_async")',
        'host_phases.sub("st_assembly")',
        'host_phases.mark("sample_tail_deferred")',
        'host_phases.mark("sample_tail_sync")',
        'host_phases.mark("sample_tail")\n',
    ])
    # No other mark between modify_prev_results and the first st_ sub.
    seg = runner[runner.index('host_phases.mark("modify_prev_results")') + 1:
                 runner.index('host_phases.sub("st_kvconn_out")')]
    assert "host_phases.mark(" not in seg
    seg = runner[runner.index('host_phases.mark("forward_dispatch")') + 1:
                 runner.index('host_phases.mark("post_forward")')]
    assert "host_phases.mark(" not in seg
    driver = _src("megakernel_runtime/qwen38_driver.py")
    _order(driver, [
        'host_phases.sub("fd_pre_whole")',
        'out = self.whole_program(',
        'host_phases.sub("fd_whole_call")',
    ])
    assert re.search(r"^from vllm_torchtpu\.runner import host_phases$", driver, re.M)
