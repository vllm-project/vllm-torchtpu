# SPDX-License-Identifier: Apache-2.0
"""q38 L4 AGREE units (runner/l4_agree.py), torch CPU only (CDK CPU pod). AGREE-BUILD (helper of L4-E).

The Agreement alone, with stand-in levers: the boundary schedule on engine steps (step 1, the union of the levers'
periods, l6_ab step-window rolls only; wall-clock rolls never), warm-up steps not counted, the periods frozen at the
first serving step, one gather per boundary whatever the number of levers, the vector layout (header, name order,
zero padding to CAP), each lever's apply seeing exactly its own slice of every rank's row, the header rule (unequal
lever set / step / magic raises), the registration rules, and fail-stop on a failed all_gather. Plus the real
transport: two processes in a gloo group agree through l4_agree.gather with two levers in one vector, and two
processes with unequal lever sets both raise instead of hanging.
"""

import json
import os
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from vllm_torchtpu.runner import l4_agree as A

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTS: dict = {}


class Lever:
    def __init__(self, name, width, period=0, roll=False, vec=None):
        self.AGREE_NAME, self.AGREE_WIDTH = name, width
        self.period, self._roll = period, roll
        self.vec = list(vec) if vec is not None else list(range(1, width + 1))
        self.rows = []
        self.agreement = None

    def local_vector(self):
        return list(self.vec)

    def apply(self, rows):
        self.rows.append([list(r) for r in rows])

    def agree_period(self):
        return self.period

    def agree_roll(self):
        return self._roll


class Ranks:
    """A stand-in transport: this rank's vector plus `others` (functions of this rank's vector)."""

    def __init__(self, others=()):
        self.others, self.calls, self.sent = list(others), 0, []

    def __call__(self, vec):
        self.calls += 1
        self.sent.append(list(vec))
        return [list(vec)] + [f(list(vec)) for f in self.others]


def _agreement(*levers, transport=None, l6=None):
    a = A.Agreement()
    a._l6[:] = [l6]
    a.transport = transport or Ranks()
    for lv in levers:
        a.add(lv)
    return a


def _ticks(a, n, warm=True):
    runner = types.SimpleNamespace(warmup_done=warm)
    return [s for s in range(1, n + 1) if a.tick(runner)]


def test_boundaries_are_engine_steps_in_the_union_of_the_periods():
    t = Ranks()
    a = _agreement(Lever("draftasm", 5, period=4), Lever("vfyone", 8, period=6), transport=t)
    got = _ticks(a, 24)
    assert got == [1, 4, 6, 8, 12, 16, 18, 20, 24], got
    assert t.calls == a.gathers == a.boundaries == len(got)  # ONE gather per boundary, two levers
    assert all(len(lv.rows) == len(got) for lv in (a.slots[0].lever, a.slots[1].lever))
    COUNTS["union_boundaries"] = len(got)


def test_warmup_steps_are_not_counted():
    t = Ranks()
    a = _agreement(Lever("x", 2, period=3), transport=t)
    assert _ticks(a, 10, warm=False) == [] and a.step == 0 and t.calls == 0
    assert _ticks(a, 3) == [1, 3] and a.step == 3


def test_periods_are_frozen_at_the_first_serving_step():
    lv = Lever("x", 2, period=5)
    a = _agreement(lv)
    assert _ticks(a, 1) == [1]
    lv.period = 2
    runner = types.SimpleNamespace(warmup_done=True)
    assert [s for s in range(2, 11) if a.tick(runner)] == [5, 10]


def test_vector_layout_and_each_lever_sees_its_own_slice():
    vf = Lever("vfyone", 8, period=2, vec=[11, 12, 13, 14, 15, 16, 17, 18])
    da = Lever("draftasm", 5, period=2, vec=[1, 2, 3, 4, 5])
    other = lambda v: v[:A.HEADER] + [x + 100 for x in v[A.HEADER:]]  # noqa: E731 - rank 1: same header
    t = Ranks([other])
    a = _agreement(vf, da, transport=t)  # registered vfyone first: the layout is in name order
    assert [s.name for s in a.slots] == ["draftasm", "vfyone"]
    _ticks(a, 1)
    vec = t.sent[0]
    assert len(vec) == A.CAP and vec[:A.HEADER] == [A.MAGIC, a.schema, 1, 0]
    assert vec[A.HEADER:A.HEADER + 13] == [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18]
    assert vec[A.HEADER + 13:] == [0] * (A.CAP - A.HEADER - 13)
    assert da.rows == [[[1, 2, 3, 4, 5], [101, 102, 103, 104, 105]]]
    assert vf.rows == [[[11, 12, 13, 14, 15, 16, 17, 18], [111, 112, 113, 114, 115, 116, 117, 118]]]
    assert da.agreement is a and vf.agreement is a
    COUNTS["layout_entries"] = 13


class StepL6:
    def __init__(self, steps):
        self.STEPS, self.j = steps, 0

    def current_class(self):
        return str((self.j // 5) % 2)


def test_rolls_make_boundaries_only_with_step_windows():
    l6 = StepL6(5)
    a = _agreement(Lever("vfyone", 8, period=0, roll=True), l6=l6)
    runner = types.SimpleNamespace(warmup_done=True)
    got = []
    for s in range(1, 21):
        l6.j = s  # l6_ab's step counter (begin_step runs before the tick)
        if a.tick(runner):
            got.append(s)
    assert got == [1, 5, 10, 15, 20], got
    # wall-clock windows (STEPS = 0): a class that rolls every step makes no boundary
    l6w = StepL6(0)
    l6w.current_class = lambda: str(l6w.j % 2)
    b = _agreement(Lever("vfyone", 8, period=0, roll=True), l6=l6w)
    got = []
    for s in range(1, 21):
        l6w.j = s
        if b.tick(runner):
            got.append(s)
    assert got == [1], got
    # step windows but no lever asks for rolls: none
    c = _agreement(Lever("draftasm", 5, period=7), l6=StepL6(5))
    assert _ticks(c, 20) == [1, 7, 14]
    COUNTS["roll_cases"] = 3


@pytest.mark.parametrize("what", ["step", "schema", "magic", "boundary"])
def test_unequal_headers_raise(what):
    idx = {"magic": 0, "schema": 1, "step": 2, "boundary": 3}[what]

    def bad(v):
        v = list(v)
        v[idx] += 1
        return v
    a = _agreement(Lever("draftasm", 5, period=1), Lever("vfyone", 8, period=1), transport=Ranks([lambda v: v, bad]))
    with pytest.raises(RuntimeError, match="headers differ"):
        _ticks(a, 1)
    assert all(not s.lever.rows for s in a.slots)  # no lever decided on mismatched rows


def test_a_rank_without_a_lever_raises_instead_of_reading_wrong_slices():
    # rank 1 registered only draftasm: its schema differs (and its vfyone entries are padding)
    lone = A.Agreement()
    lone.add(Lever("draftasm", 5))
    lone_vec = lone.local_vector()
    a = _agreement(Lever("draftasm", 5, period=1), Lever("vfyone", 8, period=1),
                   transport=Ranks([lambda v: list(lone_vec[:2]) + v[2:4] + lone_vec[4:]]))
    with pytest.raises(RuntimeError, match="headers differ"):
        _ticks(a, 1)


def test_rows_of_the_wrong_length_raise():
    a = _agreement(Lever("x", 2, period=1), transport=lambda v: [v, v[:-1]])
    with pytest.raises(RuntimeError, match="expected %d" % A.CAP):
        _ticks(a, 1)


def test_registration_rules():
    a = A.Agreement()
    a.add(Lever("draftasm", 5))
    with pytest.raises(ValueError, match="already registered"):
        a.add(Lever("draftasm", 5))
    with pytest.raises(ValueError, match="needs 0 entries"):
        a.add(Lever("zero", 0))
    with pytest.raises(ValueError, match="free"):
        a.add(Lever("big", A.CAP - A.HEADER - 5 + 1))
    a.add(Lever("fits", A.CAP - A.HEADER - 5))
    runner = types.SimpleNamespace(warmup_done=True)
    a.transport = Ranks()
    a.tick(runner)
    with pytest.raises(RuntimeError, match="before the first serving step"):
        a.add(Lever("late", 1))
    # the runner hook: for_runner creates once and stores it as runner._l4_agree
    r = types.SimpleNamespace()
    x = A.register(r, Lever("vfyone", 8))
    assert r._l4_agree is x and A.for_runner(r) is x and A.register(r, Lever("draftasm", 5)) is x
    assert [s.name for s in x.slots] == ["draftasm", "vfyone"]


def test_a_lever_vector_of_the_wrong_width_raises():
    lv = Lever("x", 3)
    a = _agreement(lv)
    lv.vec = [1, 2]
    with pytest.raises(RuntimeError, match="gave 2 entries, registered 3"):
        _ticks(a, 1)


def test_a_failed_gather_raises_and_stops_the_tick(monkeypatch):
    group = types.SimpleNamespace(world_size=8, cpu_group=object())
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)

    def boom(*a, **k):
        raise RuntimeError("gloo timeout")
    monkeypatch.setattr(torch.distributed, "all_gather", boom)
    with pytest.raises(RuntimeError, match="could not agree on the lever paths"):
        A.gather([1, 2, 3])
    a = A.Agreement()
    a._l6[:] = [None]
    a.add(Lever("draftasm", 5, period=4))
    with pytest.raises(RuntimeError, match="could not agree on the lever paths"):
        a.tick(types.SimpleNamespace(warmup_done=True))


def test_no_tp_group_is_one_rank(monkeypatch):
    def none():
        raise AssertionError("no TP group")
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", none)
    assert A.gather([4, 5]) == [[4, 5]]
    group = types.SimpleNamespace(world_size=1, cpu_group=object())
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)
    assert A.gather([4, 5]) == [[4, 5]]


# ---------------------------------------------------------------- the real gloo transport

CHILD = textwrap.dedent(r"""
    import datetime, json, os, sys, types
    import torch, torch.distributed as dist
    sys.path.insert(0, os.environ["Q38_VFY_HARNESS_DIR"])
    import vfy_harness as H
    rank, world = int(os.environ["Q38_RANK"]), 2
    dist.init_process_group("gloo", init_method="file://" + os.environ["Q38_INIT"], rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: types.SimpleNamespace(world_size=world, cpu_group=dist.group.WORLD)
    vd = types.ModuleType("vllm.distributed")
    vd.get_tensor_model_parallel_rank = lambda: dist.get_rank()
    sys.modules["vllm.distributed.parallel_state"] = ps
    sys.modules["vllm.distributed"] = vd
    AG = H.load_agree("gloo%d" % rank)
    real = dist.all_gather
    n_calls = [0]
    def counted(*a, **k):
        n_calls[0] += 1
        return real(*a, **k)
    dist.all_gather = counted

    class Lever:
        def __init__(self, name, width, period):
            self.AGREE_NAME, self.AGREE_WIDTH, self.period, self.rows = name, width, period, []
            self.disabled = 0
        def local_vector(self):
            return [1, self.disabled] + [rank * 0 + i for i in range(self.AGREE_WIDTH - 2)]
        def apply(self, rows):
            self.rows.append(rows)
        def agree_period(self):
            return self.period

    da, vf = Lever("draftasm", 5, 4), Lever("vfyone", 8, 6)
    a = AG.Agreement()
    a._l6[:] = [None]
    a.add(vf); a.add(da)
    runner = types.SimpleNamespace(warmup_done=True)
    bounds = []
    for s in range(1, 13):
        if s == 5 and rank == 1:
            da.disabled = 1
        if a.tick(runner):
            bounds.append(s)
    out = {"bounds": bounds, "gathers": a.gathers, "all_gather_calls": n_calls[0], "da": da.rows, "vf": vf.rows}
    # unequal lever sets on the two ranks: both must raise (equal-size tensors, no hang)
    b = AG.Agreement()
    b._l6[:] = [None]
    b.add(Lever("draftasm", 5, 1))
    if rank == 1:
        b.add(Lever("vfyone", 8, 1))
    try:
        b.tick(runner)
        out["mismatch"] = "no error"
    except RuntimeError as exc:
        out["mismatch"] = "headers differ" if "headers differ" in str(exc) else repr(exc)
    print("Q38_GLOO " + json.dumps(out), flush=True)
    dist.destroy_process_group()
""")


def test_real_gloo_one_gather_two_levers_two_processes(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    env = dict(os.environ, Q38_VFY_HARNESS_DIR=os.path.join(HERE, "..", "l4vfy"), Q38_INIT=str(tmp_path / "init"),
               PYTHONPATH=os.environ.get("PYTHONPATH", ""), GLOO_SOCKET_IFNAME="lo")
    procs = [subprocess.Popen([sys.executable, "-u", str(script)], env=dict(env, Q38_RANK=str(r)),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(2)]
    outs = [p.communicate(timeout=240)[0] for p in procs]
    res = []
    for p, o in zip(procs, outs):
        assert p.returncode == 0, o[-3000:]
        line = [ln for ln in o.splitlines() if ln.startswith("Q38_GLOO ")]
        assert len(line) == 1, o[-3000:]
        res.append(json.loads(line[0][len("Q38_GLOO "):]))
    assert res[0] == res[1], res  # the same rows on both ranks
    r = res[0]
    assert r["bounds"] == [1, 4, 6, 8, 12] and r["gathers"] == 5 and r["all_gather_calls"] == 5, r
    # rank 1's disable (set at step 5) reaches both ranks in the boundary-6 gather, not before
    assert r["da"][1] == [[1, 0, 0, 1, 2], [1, 0, 0, 1, 2]] and r["da"][2] == [[1, 0, 0, 1, 2], [1, 1, 0, 1, 2]], r["da"]
    assert r["vf"][2] == [[1, 0, 0, 1, 2, 3, 4, 5]] * 2, r["vf"]
    assert r["mismatch"] == "headers differ", r
    COUNTS["gloo_ranks"] = 2


def test_zz_counts():
    out = os.environ.get("Q38_L4AGREE_UNIT_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(COUNTS, f, sort_keys=True)
    print("Q38_L4AGREE_UNIT_COUNTS " + json.dumps(COUNTS, sort_keys=True))
