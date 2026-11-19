# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE: the all-rank agreement (lead safety rule) in runner/vfy_one.py. CDK CPU pod only.

The VFY program and the D83 chain carry their TP all-gather in different programs, so every TP rank must
take the same path on every step, and a path change (window switch, local disable, CHECK verdict) may take
effect only at a window boundary, through one all-rank CPU-group agreement (MIN of the enable bit).

1. Threaded W-rank simulation: each rank is its own copy of vfy_one.py (and of core/l6_ab.py in the windowed
   cases), stepping in lockstep (a per-step barrier stands for the device collectives), gathering through a
   barrier-based all_gather. Per-step decisions must be equal on every rank at every step, and may change only
   at boundary steps: uniform switch; l6_ab step windows (ab:1); a rank-local disable mid-window; one rank's
   l6_ab lockstep disable (its window wish drops); a CHECK mismatch on one rank; a clean CHECK; unequal warm
   keys; unequal settings.
2. The real gather: two processes in a gloo group (file init) agree through vfy_one.gather (MIN, sticky); a
   gather that raises stops the rank (fail-stop RuntimeError); no TP group = single-rank rows.
3. Window-switch parsing: ab: without the l6_ab lever or without step windows keeps the D83 chain, one ERROR.

q38 L4 AGREE (q38-l4-agree): the boundaries and the gather are the runner's engine-step clock's (runner/l4_agree.py,
loaded here as a private copy). Each simulated rank registers its VfyOne on its own Agreement (as the runner does),
runs begin_step then the Agreement's tick per step (execute_model's two hooks) and gathers through the Agreement's
transport; the gloo and fail-stop cases call l4_agree.gather. Assertions unchanged.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vfy_harness as H  # noqa: E402

SAB = H.load_sab()
AG = H.load_agree()
AB_ENVS = ("VLLM_TPU_VPS_VFY_ONE_AB", "VLLM_TPU_L6_AB_STEPS", "VLLM_TPU_L6_AB_PERIOD_MS", "VLLM_TPU_L6_AB_CLASSES",
           "VLLM_TPU_L6_AB_CONFIRM", "VLLM_TPU_L6_AB_DIR", "VLLM_TPU_SERVED_AB", "VLLM_TPU_HOST_AB_PERIOD_MS",
           "VLLM_TPU_L6_ENVS_CACHE", "VLLM_TPU_L6_FAST_FWD_CTX", "VLLM_TPU_L6_GC_FREEZE", "VLLM_TPU_L6_SWITCH",
           "VLLM_TPU_L6_MAIN_PIN", "VLLM_TPU_L6_DEFER_PREV")
COUNTS: dict = {}


def _runner():
    return types.SimpleNamespace(warmup_done=True, _vps_head=object(), _vps_fb_path=True, _vps_ro_fold=True,
                                 mamba_slot_read_offsets=object(), _vps_checks_left=0)


class AllGather:
    """A barrier all_gather among W threads (the TP group's CPU group)."""

    def __init__(self, world):
        self.world, self.rows = world, [None] * world
        self.b = threading.Barrier(world, timeout=60)
        self.calls = [0] * world

    def __call__(self, rank, vec):
        self.rows[rank] = list(vec)
        self.calls[rank] += 1
        self.b.wait()
        out = [list(r) for r in self.rows]
        self.b.wait()
        return out


def _ranks(world, ab="", check=0, l6env=None, tmp=None):
    """W (vfy, l6) pairs, each its own module copy; l6 None when the switch is uniform."""
    old = {k: os.environ.get(k) for k in AB_ENVS}
    for k in AB_ENVS:
        os.environ.pop(k, None)
    os.environ.update(l6env or {})
    if ab:
        os.environ["VLLM_TPU_VPS_VFY_ONE_AB"] = ab
    if tmp:
        os.environ["VLLM_TPU_L6_AB_DIR"] = tmp
    try:
        out = []
        for r in range(world):
            vf = H.load_vfy(f"r{r}")
            l6 = None
            if l6env is not None:
                l6 = H.load_l6(f"r{r}", SAB)
                l6._state["rank_world"] = (r, world)
                l6._state["store"] = tmp  # the lockstep files of this test only
            vf._L6_AB[:] = [l6]
            vf._tp_rank = lambda r=r: r
            out.append((vf, l6, vf.VfyOne(1, check, ab)))
        return out
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _simulate(ranks, steps, events=None, warm=None, single=True):
    """Run `steps` serving steps on every rank in lockstep. events: {(rank, step): fn(vfy_obj, l6, vf)} run
    before that step's begin_step. Returns per rank the list of (engaged, step_mode) and the boundary steps."""
    world = len(ranks)
    ag = AllGather(world)
    step_barrier = threading.Barrier(world, timeout=60)
    recs = [[] for _ in range(world)]
    bounds = [[] for _ in range(world)]
    errors = []
    agrees = []
    for r, (vf, l6, v) in enumerate(ranks):
        a = AG.Agreement()  # the runner's engine-step clock (registered as tpu_runner registers VFY-ONE)
        a._l6[:] = [l6]
        a.transport = lambda vec, r=r: ag(r, vec)
        a.add(v)
        agrees.append(a)
        if warm is not None:
            v.warm = set(warm[r])

    def run(r):
        vf, l6, v = ranks[r]
        runner = _runner()
        try:
            for j in range(1, steps + 1):
                ev = (events or {}).get((r, j))
                if ev is not None:
                    ev(v, l6, vf)
                if l6 is not None:
                    H.l6_begin(l6, SAB)
                n0 = v.boundaries
                v.begin_step(runner)
                agrees[r].tick(runner)
                if v.boundaries != n0:
                    bounds[r].append(j)
                d = v.decide(runner, single)
                recs[r].append((d, v.step_mode, v.engaged, v.failed))
                step_barrier.wait()
        except Exception as exc:  # noqa: BLE001
            errors.append((r, repr(exc)))
            step_barrier.abort()
            ag.b.abort()

    ts = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    assert not errors, errors
    return recs, bounds, ag


def _assert_lockstep(recs, bounds):
    """Equal decisions on every rank at every step; changes only at boundary steps."""
    world = len(recs)
    for j in range(len(recs[0])):
        row = {tuple(recs[r][j]) for r in range(world)}
        assert len(row) == 1, ("ranks diverge at step", j + 1, [recs[r][j] for r in range(world)])
    assert all(b == bounds[0] for b in bounds), bounds
    for j in range(1, len(recs[0])):
        if recs[0][j][2] != recs[0][j - 1][2]:  # the agreed state changed
            assert (j + 1) in bounds[0], ("agreed state changed off a boundary at step", j + 1, bounds[0])
    COUNTS["lockstep_checks"] = COUNTS.get("lockstep_checks", 0) + 1


W = 4


def test_uniform_all_ranks_engage_together():
    ranks = _ranks(W)
    recs, bounds, ag = _simulate(ranks, 450)
    _assert_lockstep(recs, bounds)
    assert bounds[0] == [1, 200, 400]  # first serving step, then every AGREE_EVERY
    assert all(r[0] and r[1] == "vfy" for r in recs[0])
    assert ag.calls == [3] * W  # one gather per boundary, never per step


def test_windowed_switches_only_at_window_rolls(tmp_path):
    S = 10
    env = {"VLLM_TPU_L6_AB_STEPS": str(S), "VLLM_TPU_L6_AB_CLASSES": "2", "VLLM_TPU_L6_AB_CONFIRM": "3"}
    ranks = _ranks(W, ab="ab:1", l6env=env, tmp=str(tmp_path))
    assert all(l6.ACTIVE and l6.WINDOWED and not l6.REFUSED for _, l6, _ in ranks)
    recs, bounds, ag = _simulate(ranks, 6 * S)
    _assert_lockstep(recs, bounds)
    # l6_ab's step j = serving step here (no warm-up steps); window k = j // S, class k % 2; window 0 is
    # never confirmed on a multi-rank lockstep, so it runs off.
    on = [j for j in range(1, 6 * S + 1) if recs[0][j - 1][0]]
    want = [j for j in range(1, 6 * S + 1) if (j // S) % 2 == 1]
    assert on == want, (on, want)
    assert bounds[0] == [1] + [k * S for k in range(1, 7)]  # every window roll in steps 1..6S, the last at 6S
    COUNTS["windowed_on_steps"] = len(on)


def test_local_disable_takes_effect_at_the_next_boundary_on_every_rank():
    ranks = _ranks(W)
    ev = {(2, 250): lambda v, l6, vf: v.disable_locally("test")}
    recs, bounds, _ = _simulate(ranks, 450, ev)
    _assert_lockstep(recs, bounds)
    assert all(recs[r][249][0] for r in range(W))  # step 250: still on everywhere (mid-window)
    assert all(recs[r][398][0] for r in range(W))  # step 399: still on (no boundary since 200)
    assert all(not recs[r][399][0] and recs[r][399][3] for r in range(W))  # step 400 boundary: off + failed
    assert all(not x[0] for r in range(W) for x in recs[r][399:])


def test_one_ranks_l6_lockstep_disable_turns_every_rank_off_at_the_next_window(tmp_path):
    S = 10
    env = {"VLLM_TPU_L6_AB_STEPS": str(S), "VLLM_TPU_L6_AB_CLASSES": "2", "VLLM_TPU_L6_AB_CONFIRM": "3"}
    ranks = _ranks(W, ab="ab:1", l6env=env, tmp=str(tmp_path))
    # rank 1's l6_ab disables its ab: levers in the middle of an ON window (class 1: steps 30..39)
    ev = {(1, 33): lambda v, l6, vf: l6._sync_disable("test")}
    recs, bounds, _ = _simulate(ranks, 6 * S, ev)
    _assert_lockstep(recs, bounds)
    assert all(recs[r][34][0] for r in range(W))  # step 35: the ON window keeps going on every rank
    assert not any(x[0] for r in range(W) for x in recs[r][39:])  # from step 40 off everywhere (MIN)
    assert not any(x[3] for r in range(W) for x in recs[r])  # a wish drop is not a failure


def test_min_of_the_wishes_without_a_failure():
    """One rank's wish drops for a window (no disable): every rank runs the D83 chain from the next boundary
    to the boundary after the wish comes back, and nothing fails."""
    ranks = _ranks(W)

    def drop(v, l6, vf):
        v.wish = lambda: False

    def back(v, l6, vf):
        del v.wish
    recs, bounds, _ = _simulate(ranks, 650, {(1, 150): drop, (1, 350): back})
    _assert_lockstep(recs, bounds)
    on = [x[0] for x in recs[0]]
    assert all(on[:199]) and not any(on[199:399]) and all(on[399:])
    assert not any(x[3] for r in range(W) for x in recs[r])


def test_check_mismatch_on_one_rank_keeps_every_rank_on_base():
    ranks = _ranks(W, check=5)

    def bad(v, l6, vf):
        v.check_mismatches += 3
    recs, bounds, _ = _simulate(ranks, 450, {(3, 4): bad})
    _assert_lockstep(recs, bounds)
    assert [x[1] for x in recs[0][:6]] == ["check"] * 5 + [None]  # 5 check steps, then wait for the boundary
    assert all(not x[0] for x in recs[0][5:]) and recs[0][-1][3]  # never armed; failed at 200


def test_clean_check_arms_at_the_next_boundary():
    ranks = _ranks(W, check=5)
    recs, bounds, _ = _simulate(ranks, 450)
    _assert_lockstep(recs, bounds)
    modes = [x[1] for x in recs[0]]
    assert modes[:5] == ["check"] * 5 and set(modes[5:199]) == {None} and set(modes[199:]) == {"vfy"}


def test_unequal_warm_keys_or_settings_fail_at_the_first_boundary():
    ranks = _ranks(W)
    warm = [{("k", 1)}] * W
    warm[1] = {("k", 2)}
    recs, bounds, _ = _simulate(ranks, 5, warm=warm)
    _assert_lockstep(recs, bounds)
    assert all(not x[0] and x[3] for x in recs[0])
    ranks = _ranks(W)
    ranks[2][2].config_hash += 1
    recs, bounds, _ = _simulate(ranks, 5)
    _assert_lockstep(recs, bounds)
    assert all(not x[0] and x[3] for x in recs[0])


def test_multi_chunk_and_static_steps_never_engage():
    ranks = _ranks(W)
    recs, bounds, _ = _simulate(ranks, 5, single=False)
    _assert_lockstep(recs, bounds)
    assert not any(x[0] for x in recs[0]) and all(x[2] for x in recs[0])


# ---------------------------------------------------------------- 2. the real gather

CHILD = textwrap.dedent(r"""
    import json, os, sys, types
    import torch, torch.distributed as dist
    sys.path.insert(0, os.environ["Q38_HARNESS_DIR"])
    import vfy_harness as H
    rank, world = int(os.environ["Q38_RANK"]), 2
    import datetime
    dist.init_process_group("gloo", init_method="file://" + os.environ["Q38_INIT"], rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: types.SimpleNamespace(world_size=world, cpu_group=dist.group.WORLD)
    vd = types.ModuleType("vllm.distributed")
    vd.get_tensor_model_parallel_rank = lambda: dist.get_rank()
    sys.modules["vllm.distributed.parallel_state"] = ps
    sys.modules["vllm.distributed"] = vd
    vf = H.load_vfy("gloo%d" % rank)
    ag = H.load_agree("gloo%d" % rank)
    vf._L6_AB[:] = [None]
    out = []
    v = vf.VfyOne(1, 0, "")
    v.warm = {("k", 1)}
    runner = types.SimpleNamespace(warmup_done=True)
    for step, disable in ((1, False), (200, rank == 1), (400, False)):
        if disable:
            v.disable_locally("gloo test")
        v.serving = step - 1
        rows = ag.gather(v.local_vector())
        v.apply(rows)
        out.append({"step": step, "rows": rows, "engaged": v.engaged, "failed": v.failed})
    print("Q38_GLOO " + json.dumps(out), flush=True)
    dist.destroy_process_group()
""")


def test_real_gloo_gather_two_processes_agree(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    init = tmp_path / "init"
    env = dict(os.environ, Q38_HARNESS_DIR=os.path.dirname(os.path.abspath(__file__)), Q38_INIT=str(init),
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
    assert res[0] == res[1], res  # the same rows and the same decisions on both ranks
    a, b, c = res[0]
    assert len(a["rows"]) == 2 and a["engaged"] and not a["failed"]
    assert b["rows"][1][0] == 0 and b["rows"][1][1] == 1 and b["rows"][0][0] == 1  # rank 1 disabled, rank 0 not
    assert not b["engaged"] and b["failed"] and not c["engaged"] and c["failed"]  # MIN, sticky
    COUNTS["gloo_ranks"] = 2


def test_failed_gather_raises(monkeypatch):
    vf = H.load_vfy("fail")
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: types.SimpleNamespace(world_size=2, cpu_group=object())
    import torch

    def boom(*a, **k):
        raise RuntimeError("gloo down")
    monkeypatch.setattr(torch.distributed, "all_gather", boom)
    with H.modules({"vllm.distributed.parallel_state": ps}):
        with pytest.raises(RuntimeError, match="could not agree"):
            AG.gather([1, 0, 0, 0, 5, 6, 7, -1])
    # and a VfyOne boundary propagates it (fail-stop, not a silent D83 chain)
    vf._L6_AB[:] = [None]
    v = vf.VfyOne(1, 0, "")
    a = AG.Agreement()
    a._l6[:] = [None]
    a.add(v)
    runner = types.SimpleNamespace(warmup_done=True)
    with H.modules({"vllm.distributed.parallel_state": ps}):
        with pytest.raises(RuntimeError, match="could not agree"):
            v.begin_step(runner)
            a.tick(runner)


def test_no_tp_group_is_one_rank():
    vf = H.load_vfy("solo")
    with H.modules({"vllm.distributed.parallel_state": types.ModuleType("vllm.distributed.parallel_state")}):
        assert AG.gather([1, 2]) == [[1, 2]]


# ---------------------------------------------------------------- 3. window-switch parsing


def test_ab_needs_the_lever_and_step_windows(caplog, tmp_path):
    vf = H.load_vfy("parse")
    vf._L6_AB[:] = [None]
    v = vf.VfyOne(1, 0, "ab:1")
    with caplog.at_level(logging.ERROR):
        assert v.wish() is False and v.wish() is False
    assert len([m for m in caplog.messages if "needs core/l6_ab.py" in m]) == 1
    l6 = types.SimpleNamespace(LEVERS={"vfyone": "X"}, STEPS=0, on=lambda n: True)
    vf._L6_AB[:] = [l6]
    v = vf.VfyOne(1, 0, "ab:1")
    assert v.wish() is False
    l6.STEPS = 10
    assert v.wish() is True
    for raw, want in (("", True), ("1", True), ("0", False), ("bogus", False)):
        assert vf.VfyOne(1, 0, raw).wish() is want
    # the l6_ab copy of this tree registers the lever to the env var
    os.environ.pop("VLLM_TPU_VPS_VFY_ONE_AB", None)
    real = H.load_l6("parse", SAB)
    assert real.LEVERS.get("vfyone") == "VLLM_TPU_VPS_VFY_ONE_AB"


def test_check_gate_follows_l6_check_allowed():
    vf = H.load_vfy("gate")
    vf._L6_AB[:] = [types.SimpleNamespace(check_allowed=lambda: False)]
    assert vf.VfyOne(1, 7, "").check_steps == 0
    vf._L6_AB[:] = [types.SimpleNamespace()]  # no check_allowed on this tree: allowed
    assert vf.VfyOne(1, 7, "").check_steps == 7


def test_zz_counts():
    out = os.environ.get("Q38_L4VFY_AGREE_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(COUNTS, f, sort_keys=True)
    print("Q38_L4VFY_AGREE_COUNTS " + json.dumps(COUNTS, sort_keys=True))
