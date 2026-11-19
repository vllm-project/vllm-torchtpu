# SPDX-License-Identifier: Apache-2.0
"""q38 L6 P-METER (host_levers_h2.md section 4.1 row 1): runner/l6_pmeter.py.

* The PMETER window line is byte-exact on a virtual wall clock that differs
  from the perf clock (a stamp taken on the wrong clock changes the line), with
  windows rolled by l6_ab.step_seq(), '-' for an unstamped point and 'x' for a
  step with the meter off (ab: windows).
* THREADSTAT follows every PMETER line (fixture /proc reads; one real read).
* Off: no lever set -> no clock read, no ring write, no line; another lever set
  but pmeter off -> no ring write, no line.
* DELAY-INJECT: one rank (inject) or every rank (injectall) spins once per
  step, before the forward enqueue stamp; the spin really holds the host.
* ~mpr_wait is exactly the blocking copy wait of the real
  TPUModelRunner._modify_prev_results (virtual clock), and the P-METER wait
  stamps bracket the same wait.
* Wiring in source (execute_model / forward / drafter) and the registry rows.
"""

from __future__ import annotations

import ast
import os
import re
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import host_phases, l6_pmeter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src", "vllm_torchtpu")
RUNNER = os.path.join(SRC, "runner", "tpu_runner.py")
EAGLE = os.path.join(SRC, "spec_decode", "eagle3.py")
BASE = 1_000_000_000_000


class _Clock:
    def __init__(self, ns=0):
        self.ns = ns

    def __call__(self):
        return self.ns


@pytest.fixture
def pm(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    lines = []
    monkeypatch.setattr(l6_pmeter.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)
    wall = _Clock(BASE)
    perf = _Clock(BASE + 5_555_555_555)
    monkeypatch.setattr(l6_pmeter.time, "time_ns", wall)
    monkeypatch.setattr(l6_pmeter.time, "perf_counter_ns", perf)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (3, 8))
    monkeypatch.setattr(l6_pmeter.os, "getpid", lambda: 77)

    def setup(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_pmeter._resolve()
        l6_pmeter._st["host"] = "hostA"
        return l6_pmeter

    yield setup, lines, wall
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_pmeter._resolve()


def _fake_threads(monkeypatch):
    monkeypatch.setattr(l6_pmeter.threading, "enumerate",
                        lambda: [NS(name="MainThread", native_id=11), NS(name="out put", native_id=12),
                                 NS(name="nat", native_id=None)])
    files = {"/proc/self/task/11/schedstat": "1000 200 3\n",
             "/proc/self/task/11/status": "Name:\tx\nvoluntary_ctxt_switches:\t5\nnonvoluntary_ctxt_switches:\t2\n"}
    monkeypatch.setattr(l6_pmeter, "_read", lambda p: files.get(p))


def _step(m, wall, k, fwd=True, wait=True, loop=True):
    """One worker step k on the virtual wall clock (7 ms apart)."""
    b = BASE + k * 7_000_000
    wall.ns = b
    m.deq()
    l6_ab.begin_step()
    if l6_ab.ACTIVE:
        m.on_step()
    if fwd:
        wall.ns = b + 1_200_500
        m.before_forward()
    if wait:
        wall.ns = b + 3_000_000
        m.wait_begin()
        wall.ns = b + 3_450_999
        m.wait_end()
    if loop:
        wall.ns = b + 4_100_000
        m.loop_enqueue()


E = "1200,3000,3450,4100"
TS = "THREADSTAT rank=3 seq={} n=3 MainThread:11=1000/200/3/5/2 out_put:12=- nat:None=-"


def test_window_line_is_byte_exact(pm, monkeypatch):
    setup, lines, wall = pm
    m = setup(VLLM_TPU_L6_PMETER="1", VLLM_TPU_L6_PMETER_WINDOW="4")
    _fake_threads(monkeypatch)
    for k in range(1, 10):
        _step(m, wall, k, loop=k != 2, wait=k != 3)
        if k == 4:
            assert lines == []  # the window closes when the next step begins
    b1, b5 = BASE + 7_000_000, BASE + 5 * 7_000_000
    assert lines == [
        f"PMETER rank=3 host=hostA pid=77 window=4 seq0=1 n=4 t0_ns={b1} "
        f"e=0,{E};7000,1200,3000,3450,-;14000,1200,-,-,4100;21000,{E}",
        TS.format(4),
        f"PMETER rank=3 host=hostA pid=77 window=4 seq0=5 n=4 t0_ns={b5} e=0,{E};7000,{E};14000,{E};21000,{E}",
        TS.format(8),
    ], lines


def test_ab_window_steps_show_x(pm, monkeypatch):
    setup, lines, wall = pm
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (3, 1))
    m = setup(VLLM_TPU_L6_PMETER="ab:1", VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1",
              VLLM_TPU_L6_PMETER_WINDOW="4")
    _fake_threads(monkeypatch)
    for k in range(1, 13):
        _step(m, wall, k)
    b4, b5 = BASE + 4 * 7_000_000, BASE + 5 * 7_000_000
    # l6_ab step windows: j 1-3 class 0 (off), 4-7 class 1 (on), 8-11 class 0, 12- class 1.
    assert lines == [
        f"PMETER rank=3 host=hostA pid=77 window=4 seq0=1 n=1 t0_ns={b4} e=x;x;x;0,{E}",
        TS.format(4),
        f"PMETER rank=3 host=hostA pid=77 window=4 seq0=5 n=3 t0_ns={b5} e=0,{E};7000,{E};14000,{E};x",
        TS.format(8),
    ], lines


def test_off_reads_no_clock_and_logs_nothing(pm, monkeypatch):
    setup, lines, wall = pm
    m = setup()
    assert not l6_ab.ACTIVE

    def boom():
        raise AssertionError("a clock read with every L6 lever off")
    monkeypatch.setattr(l6_pmeter.time, "time_ns", boom)
    monkeypatch.setattr(l6_pmeter.time, "perf_counter_ns", boom)
    for _ in range(12):
        m.deq()
        l6_ab.begin_step()
        m.before_forward()
        m.wait_begin()
        m.wait_end()
        m.loop_enqueue()
    assert lines == [] and not any(m._ring) and m._cur == [0, 0, -1, -1, -1, -1]


def test_other_lever_on_pmeter_off_writes_nothing(pm):
    setup, lines, wall = pm
    m = setup(VLLM_TPU_L6_DEFER_PREV="1", VLLM_TPU_L6_PMETER_WINDOW="4")
    assert l6_ab.ACTIVE
    for k in range(1, 12):
        _step(m, wall, k)
    assert lines == [] and not any(m._ring[0::6]) and m._cur[0] == 0


@pytest.mark.parametrize("rank,env,want", [
    (3, {"VLLM_TPU_L6_INJECT": "1", "VLLM_TPU_L6_INJECT_RANK": "3", "VLLM_TPU_L6_INJECT_US": "100"}, 100),
    (2, {"VLLM_TPU_L6_INJECT": "1", "VLLM_TPU_L6_INJECT_RANK": "3", "VLLM_TPU_L6_INJECT_US": "100"}, 0),
    (2, {"VLLM_TPU_L6_INJECT_ALL": "1", "VLLM_TPU_L6_INJECT_ALL_US": "700"}, 700),
    (3, {"VLLM_TPU_L6_INJECT": "1", "VLLM_TPU_L6_INJECT_RANK": "3", "VLLM_TPU_L6_INJECT_ALL": "1"}, 1500),
])
def test_inject_one_rank_or_all_ranks_once_per_step_before_the_stamp(pm, monkeypatch, rank, env, want):
    setup, lines, wall = pm
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (rank, 8))
    m = setup(VLLM_TPU_L6_PMETER="1", **env)
    ev = []
    monkeypatch.setattr(l6_pmeter, "_spin", lambda us: ev.append(("spin", us)))

    def stamp():
        ev.append(("stamp",))
        return BASE
    monkeypatch.setattr(l6_pmeter.time, "time_ns", stamp)
    for _ in range(3):
        m.deq()
        l6_ab.begin_step()
        m.on_step()
        m.before_forward()
        m.before_forward()  # a second chunk's forward: no second spin, no second stamp
    one = ([("stamp",), ("spin", want), ("stamp",)] if want else [("stamp",), ("stamp",)])
    assert ev == one * 3, ev
    assert l6_ab._state["counts"].get("inject.spins", 0) == (3 if want else 0)
    assert l6_ab._state["counts"].get("inject.us", 0) == 3 * want


def test_spin_holds_the_host_for_its_length():
    t = time.perf_counter_ns()
    l6_pmeter._spin(3000)
    assert time.perf_counter_ns() - t >= 3_000_000


def test_thread_line_real_proc_read():
    line = l6_pmeter.thread_line(7)
    assert line.startswith("THREADSTAT rank=") and " seq=7 " in line
    main = [p for p in line.split()[4:] if p.startswith("MainThread:")]
    assert len(main) == 1, line
    tid = main[0].split(":")[1].split("=")[0]
    if os.path.exists(f"/proc/self/task/{tid}/schedstat"):
        assert re.fullmatch(r"MainThread:\d+=\d+/\d+/\d+/\d+/\d+", main[0]), main[0]


# ------------------------------------------------------------------ the copy wait (real path)

class _Clk:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


def _hp_reset(monkeypatch, subs):
    for name, v in (("_EVERY", 1), ("_SLOW_NS", 0), ("_SUBS", subs), ("_GC", False), ("_last", None),
                    ("_steps", 0), ("_samples", {}), ("_cur", {}), ("_ctx", []), ("_sub_last", None),
                    ("_sub_acc", {}), ("_sub_samples", {}), ("_rank", 3)):
        monkeypatch.setattr(host_phases, name, v)
    log = []
    monkeypatch.setattr(host_phases, "logger", NS(info=lambda fmt, *a: log.append(fmt % a)))
    return log


@pytest.mark.parametrize("subs", [True, False])
def test_mpr_wait_is_exactly_the_blocking_copy_wait(pm, monkeypatch, subs):
    """~mpr_pre (40 us before the wait), ~mpr_wait (exactly the 700 us wait),
    ~mpr_post (the rollback, 50 us); the phase holding them is their sum plus
    the 30 us after. The P-METER stamps bracket the same 700 us."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    setup, lines, wall = pm
    m = setup(VLLM_TPU_L6_PMETER="1")
    log = _hp_reset(monkeypatch, subs)
    clk = _Clk()
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", clk)
    monkeypatch.setattr(l6_pmeter.time, "time_ns", clk)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)

    class _Pre:
        req_ids = ["a"]
        discard_sampled_tokens_req_indices = []

        def __init__(self, state):
            self.request_seq_lens = [(0, state, 9, "a")]

        @property
        def num_draft_per_req(self):
            clk.tick(40)
            return {0: 3}

        def wait_for_copy(self):
            clk.tick(700)
            return torch.tensor([[77, 88, -1, -1]])

    class _Batch:
        num_tokens_no_spec = np.array([10])
        token_ids_cpu = np.zeros((1, 32), dtype=np.int64)

        @property
        def req_id_to_index(self):
            clk.tick(25)
            return {"a": 0}

    state = NS(output_token_ids=[1, 2, 3, 0, 0, 0, 0])
    fake = NS(_pre_async_results=_Pre(state), input_batch=_Batch())
    monkeypatch.setattr("vllm_torchtpu.runner.tpu_runner.INVALID_TOKEN_ID", -1)
    l6_ab.begin_step()
    m.on_step()
    host_phases.begin_step()
    host_phases.mark("token_d2h_bookkeeping")
    TPUModelRunner._modify_prev_results(fake)
    clk.tick(30)
    host_phases.mark("dp_entry")
    host_phases.begin_step()
    assert state.output_token_ids == [1, 2, 3, 77, 88]  # the real rollback ran
    assert m._cur[4] - m._cur[3] == 700_000
    ph = [ln for ln in log if ln.startswith("HOSTPHASES") and "dp_entry=" in ln]
    assert len(ph) == 1 and "dp_entry=820/820/820" in ph[0], log
    subl = [ln for ln in log if ln.startswith("HOSTSUBS") and "~mpr" in ln]
    if not subs:
        assert not any(ln.startswith("HOSTSUBS") for ln in log) and host_phases._sub_acc == {}
        return
    assert len(subl) == 1, log
    assert subl[0].split("(p50/p90/mean-per-step/n-steps) ", 1)[1] == (
        "~mpr_pre=40/40/40/1 ~mpr_wait=700/700/700/1 ~mpr_post=50/50/50/1")


# ------------------------------------------------------------------ wiring and registry

def _func(path, name, cls=None):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (cls is None or node.name == cls):
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise AssertionError(name)


def _src(path, f):
    return "\n".join(open(path).read().splitlines()[f.lineno - 1:f.end_lineno])


def test_wiring_in_source():
    f = _func(RUNNER, "execute_model", "TPUModelRunner")
    first = f.body[0]
    assert ast.unparse(first) == "_l6_pmeter.deq()", ast.unparse(first)
    body = _src(RUNNER, f)
    i_hp = body.index("host_phases.begin_step()")
    assert body.index("_l6_pmeter.deq()") < i_hp
    # after l6_ab.begin_step and behind the envcache / jitter hooks and the steady drop, which
    # test_l6_ab_envcache (240 chars) and test_l6_steady (160 chars) pin right behind the guard
    assert re.search(r"_l6_ab\.begin_step\(fp_fn=[^\n]*\)\n\s+_l6_envcache\.on_step\(\)\n\s+_l6_jitter\.on_step\(\)\n"
                     r"\s+if _l6_ab\.ACTIVE and not _l6_steady\.on\(self\):\n\s+_l6_steady\.drop\(self\)\n"
                     r"\s+if _l6_ab\.ACTIVE:\n(\s+#[^\n]*\n)*\s+_l6_pmeter\.on_step\(\)\n", body)
    stmts = [ln.strip() for ln in open(RUNNER).read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    k = stmts.index("_l6_pmeter.before_forward()")
    assert stmts[k + 1] == "hidden_states, aux_hidden_states = self.forward_model(", stmts[k + 1]
    assert stmts.count("_l6_pmeter.before_forward()") == 1
    mpr = [ln.strip() for ln in _src(RUNNER, _func(RUNNER, "_modify_prev_results", "TPUModelRunner")).splitlines()
           if ln.strip() and not ln.strip().startswith("#")]
    k = mpr.index('host_phases.sub("mpr_pre")')
    assert mpr[k:k + 5] == ['host_phases.sub("mpr_pre")', "_l6_pmeter.wait_begin()",
                            "pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()",
                            "_l6_pmeter.wait_end()", 'host_phases.sub("mpr_wait")'], mpr[k:k + 5]
    assert mpr[-1] == 'host_phases.sub("mpr_post")'
    es = [ln.strip() for ln in open(EAGLE).read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    k = es.index('host_phases.mark("dp_loop_setup")')
    assert es[k + 1] == "_l6_pmeter.loop_enqueue()" and es.count("_l6_pmeter.loop_enqueue()") == 1


def test_registry_rows():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["pmeter"] == "VLLM_TPU_L6_PMETER"
    assert l6_ab.LEVERS["inject"] == "VLLM_TPU_L6_INJECT"
    assert l6_ab.LEVERS["injectall"] == "VLLM_TPU_L6_INJECT_ALL"
    for name, default in (("VLLM_TPU_L6_PMETER", ""), ("VLLM_TPU_L6_PMETER_WINDOW", 200), ("VLLM_TPU_L6_INJECT", ""),
                          ("VLLM_TPU_L6_INJECT_US", 250), ("VLLM_TPU_L6_INJECT_RANK", 0),
                          ("VLLM_TPU_L6_INJECT_ALL", ""), ("VLLM_TPU_L6_INJECT_ALL_US", 1500)):
        assert name in _TPU_COMPILE_ENV_IGNORED, name
        if name not in os.environ:
            assert getattr(envs, name) == default, name


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6PM_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6PM_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
