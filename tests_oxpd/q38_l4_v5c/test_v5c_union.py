# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c): the all-rank agreement on the union of l6_ab levers. CDK CPU pod only.
V5COMP-BUILD (helper of L4-E).

The branch's core/l6_ab.py LEVERS is the union of L6's six (v4), VPSHOST's sampfb / sampsnap / statsasync, VFY-ONE's
vfyone and DRAFT-ASM / DRAFT1-B's draftasm / draftasmcl / draftasmone. Torch-free W-rank simulation on VFY-ONE's
harness (tests/l4vfy: each rank its own copy of core/l6_ab.py and runner/vfy_one.py, a barrier all_gather):

U1  every union lever windowed at once (step windows, 4 classes, each lever its own class set): on every step all
    ranks hold the same on / off state for every lever, each lever follows its own classes only (no cross-talk),
    and VFY-ONE's agreed path (MIN over the ranks, at window rolls) is the same on every rank.
U2  one rank's lockstep fingerprint differs for window 5 (VFY-ONE's second on window): every rank reads it at the
    confirm step and turns every ab: lever off from there; VFY-ONE stays off on every rank at the roll where it would
    have engaged (a wish drop, not a failure).
U3  LEVERS is exactly the union (14 entries with v6's k1fuse, 14 distinct env vars); no two levers share an env var.
Run: python -m pytest -q -p no:cacheprovider --noconftest -s tests/q38_l4_v5c/test_v5c_union.py
"""

import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "l4vfy"))
import test_vfy_agree as ta  # noqa: E402  (VFY-ONE's agreement harness: _ranks, _assert_lockstep)
import vfy_harness as H  # noqa: E402

UNION = {
    "envcache": "VLLM_TPU_L6_ENVS_CACHE",
    "fctx": "VLLM_TPU_L6_FAST_FWD_CTX",
    "gcfreeze": "VLLM_TPU_L6_GC_FREEZE",
    "switch": "VLLM_TPU_L6_SWITCH",
    "mainpin": "VLLM_TPU_L6_MAIN_PIN",
    "defer": "VLLM_TPU_L6_DEFER_PREV",
    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER",
    "sampsnap": "VLLM_TPU_L4_SAMP_SNAP",
    "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC",
    "vfyone": "VLLM_TPU_VPS_VFY_ONE_AB",
    "draftasm": "VLLM_TPU_DRAFT_ASM_AB",
    "draftasmcl": "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "draftasmone": "VLLM_TPU_DRAFT_ASM_ONE_AB",
    "k1fuse": "VLLM_TPU_K1FUSE_AB",  # v6 (q38-l4-v6c2, V6COMP-BUILD): T1-K1FUSE's window switch
}
CLASSES = {  # 4 classes; overlapping and disjoint sets, so cross-talk between any two levers would show
    "envcache": {1}, "fctx": {2}, "gcfreeze": {3}, "switch": {1, 2}, "mainpin": {2, 3}, "defer": {0, 1},
    "sampfb": {1, 3}, "sampsnap": {2}, "statsasync": {3}, "vfyone": {1}, "draftasm": {2, 3},
    "draftasmcl": {1}, "draftasmone": {3}, "k1fuse": {0, 2},
}
S, NCLS, D, W = 8, 4, 2, ta.W
COUNTS: dict = {}


def _env(tmp):
    env = {"VLLM_TPU_L6_AB_STEPS": str(S), "VLLM_TPU_L6_AB_CLASSES": str(NCLS), "VLLM_TPU_L6_AB_CONFIRM": str(D)}
    for lever, var in UNION.items():
        if lever != "vfyone":  # _ranks sets VFY's own window variable from its `ab` argument
            env[var] = "ab:" + "+".join(str(c) for c in sorted(CLASSES[lever]))
    return env


def _simulate(ranks, steps, fps):
    """ta._simulate (q38 L4 AGREE's port: per-rank Agreement clock + transport), also recording every union lever's
    l6_ab state per step; fps(rank, step) -> fingerprint."""
    ag = ta.AllGather(W)
    bar = threading.Barrier(W, timeout=60)
    recs, bounds, states, errors = [[] for _ in range(W)], [[] for _ in range(W)], [[] for _ in range(W)], []
    agrees = []
    for r, (vf, l6, v) in enumerate(ranks):
        # q38 L4 AGREE: the runner's engine-step clock carries VFY-ONE's agreement (as test_vfy_agree._simulate)
        a = ta.AG.Agreement()
        a._l6[:] = [l6]
        a.transport = lambda vec, r=r: ag(r, vec)
        a.add(v)
        agrees.append(a)

    def run(r):
        vf, l6, v = ranks[r]
        runner = ta._runner()
        try:
            for j in range(1, steps + 1):
                H.l6_begin(l6, ta.SAB, fps(r, j))
                states[r].append(tuple(sorted((k, bool(l6.on(k))) for k in UNION)))
                n0 = v.boundaries
                v.begin_step(runner)
                agrees[r].tick(runner)
                if v.boundaries != n0:
                    bounds[r].append(j)
                recs[r].append((v.decide(runner, True), v.step_mode, v.engaged, v.failed))
                bar.wait()
        except Exception as exc:  # noqa: BLE001
            errors.append((r, repr(exc)))
            bar.abort()
            ag.b.abort()

    ts = [threading.Thread(target=run, args=(r,)) for r in range(W)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    assert not errors, errors
    return recs, bounds, states


def _ranks(monkeypatch, tmp):
    env = _env(tmp)
    for var in UNION.values():  # the variables ta._ranks does not save / restore
        if var not in ta.AB_ENVS and var != "VLLM_TPU_VPS_VFY_ONE_AB":
            monkeypatch.setenv(var, env[var])
    ranks = ta._ranks(W, ab="ab:1", l6env=env, tmp=tmp)
    for _, l6, _ in ranks:
        assert l6.ACTIVE and l6.WINDOWED and not l6.REFUSED and l6.STEPS == S and l6.CLASSES == NCLS
        assert dict(l6.LEVERS) == UNION
        assert all(l6._cfg[k] == (False, frozenset(CLASSES[k])) for k in UNION), l6._cfg
    return ranks


def _want(j, lever, off_step=None):
    """l6_ab's state at step j: window k = j // S, class k % NCLS; window 0 is never confirmed; after a desync is
    read (off_step = the confirm step k*S - 1 of the failed window) every ab: lever is off."""
    k = j // S
    if k == 0 or (off_step is not None and j >= off_step):
        return False
    return (k % NCLS) in CLASSES[lever]


def test_u1_every_union_lever_windowed_agrees_on_every_rank(monkeypatch, tmp_path):
    ranks = _ranks(monkeypatch, str(tmp_path))
    steps = 7 * S
    recs, bounds, states = _simulate(ranks, steps, lambda r, j: "fp")
    ta._assert_lockstep(recs, bounds)
    for j in range(steps):
        assert len({states[r][j] for r in range(W)}) == 1, ("ranks diverge", j + 1, [states[r][j] for r in range(W)])
        got = dict(states[0][j])
        want = {k: _want(j + 1, k) for k in UNION}
        assert got == want, (j + 1, {k for k in UNION if got[k] != want[k]})
    vfy_on = [j for j in range(1, steps + 1) if recs[0][j - 1][0]]
    assert vfy_on == [j for j in range(1, steps + 1) if _want(j, "vfyone")], vfy_on
    assert any(recs[0][j][0] for j in range(steps)) and not any(recs[0][j][3] for j in range(steps))
    n_on = sum(v for row in states[0] for _, v in row)
    COUNTS["u1_lever_on_steps"] = n_on
    print(f"\nV5C_UNION U1 ranks={W} steps={steps} levers={len(UNION)} lever-on states={n_on} identical on every "
          f"rank and per lever class only; VFY on {len(vfy_on)} steps in lockstep (boundaries {bounds[0]})")


def test_u2_one_ranks_desync_turns_every_lever_off_on_every_rank(monkeypatch, tmp_path):
    ranks = _ranks(monkeypatch, str(tmp_path))
    steps = 7 * S
    bad, off_step = 5 * S - D, 5 * S - 1  # rank 2 writes a different fingerprint for window 5; read at 5S - 1
    recs, bounds, states = _simulate(ranks, steps, lambda r, j: "other" if (r == 2 and j == bad) else "fp")
    ta._assert_lockstep(recs, bounds)
    for j in range(steps):
        assert len({states[r][j] for r in range(W)}) == 1, ("ranks diverge", j + 1)
        assert dict(states[0][j]) == {k: _want(j + 1, k, off_step=off_step) for k in UNION}, j + 1
    assert all(l6._state["sync_off"] for _, l6, _ in ranks)
    vfy_on = [j for j in range(1, steps + 1) if recs[0][j - 1][0]]
    assert vfy_on == list(range(S, 2 * S)), vfy_on  # window 1 only: window 5 never engages
    assert not any(recs[r][j][3] for r in range(W) for j in range(steps))
    COUNTS["u2"] = 1
    print(f"\nV5C_UNION U2 rank 2 fingerprint differs for window 5: every rank sync_off at step {off_step}; all "
          f"{len(UNION)} levers off from there on every rank; VFY on in window 1 only ({len(vfy_on)} steps), not failed")


def test_u3_levers_is_exactly_the_union_with_distinct_env_vars():
    l6 = H.load_l6("v5c_u3", ta.SAB)
    assert dict(l6.LEVERS) == UNION, set(l6.LEVERS.items()) ^ set(UNION.items())
    assert len(set(l6.LEVERS.values())) == len(l6.LEVERS) == 14
    COUNTS["u3_levers"] = len(l6.LEVERS)
    print("\nV5C_UNION U3 LEVERS " + ",".join(f"{k}={v}" for k, v in l6.LEVERS.items()))


def test_zz_counts():
    print("\nV5C_UNION_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
