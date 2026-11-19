# SPDX-License-Identifier: Apache-2.0
"""q38 L1-S3DGATE: the S3D load gate (VLLM_TPU_S3_DEFER_GATE_MIN_REQS, core/l6_ab.py S3D_GATE_*; HUNT-11 §5.1
form b). Host-only; CDK CPU pod.

Lever s3defer (ab:<classes>, step windows) runs its classes in window k only if window k-1's mean scheduled-request
count over its first S - D steps is at least T. The decision rides the lockstep fingerprint of window k.

A. One process (rank / world monkeypatched): unset / 0 is v8 (no line, no state, no fingerprint suffix, GATE as
   before); the validators; the gate follows the previous window's counted steps (the last D steps of a window never
   count), crosses T both ways, holds exactly at T, keeps the class pattern inside eligible windows and leaves every
   other lever alone; the per-step GATES (carry / steady) behave exactly as without it and gate_step never raises
   (the _GATE_KEY trap); the engine core prints no line; without gate_step calls every window stays closed; the
   line's exact format (the reader's regex); the fingerprint carries the decision and a peer that decided otherwise
   disables the ab: levers; the runner hook records the count with only the S3D gate set.
B. Four gloo ranks (spawned), production windows (S = 250, D = 16, 8 classes, ab:1+2+4+7, T = 4), a synthetic load
   trace that crosses T both ways (exact-T and just-below windows, decoys in the uncounted steps), one barrier per
   step as the device collectives: every rank reaches the same lever states at every step and the same decision and
   line at every window, equal to an independent oracle; no rank is disabled. With rank 1 alone at T = 5 the
   lockstep must disable every rank at the first window the two thresholds decide differently, and the states must
   stay identical on every rank at every step (the fail-safe; without the fingerprint fold the ranks would silently
   run S3D in different windows).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from types import SimpleNamespace as NS

import pytest

GATE_ENV = "VLLM_TPU_S3_DEFER_GATE_MIN_REQS"
S3D = "VLLM_TPU_S3_DEFER_AR"
WORLD = 4
# The reader's regex for the engagement line (q38_s3a_read.py RX_GATE, kept identical).
RX_GATE = re.compile(
    r"S3DEFER gate rank(\d+): window=(\d+) class=(\S+) win_eligible=([01]) win_on=([01]) mean_reqs_prev=(\S+) "
    r"n_prev=(\d+) T=(\d+) windows=(\d+) eligible=(\d+) on=(\d+) share_on=([0-9.]+) share_on_eligible=(\S+)$")
ONCLS = {1, 2, 4, 7}


def _clear_env(setenv_del):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k.startswith("VLLM_TPU_S3_") or k in (
                "VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB"):
            setenv_del(k)


@pytest.fixture
def gt(monkeypatch, tmp_path):
    from vllm_torchtpu.core import l6_ab

    _clear_env(lambda k: monkeypatch.delenv(k, raising=False))
    monkeypatch.setenv("VLLM_TPU_L6_AB_DIR", str(tmp_path))
    lines: list[str] = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    world = {"rank": 0, "n": 1}
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (world["rank"], world["n"]))

    def setup(n_world=1, **env):
        _clear_env(lambda k: monkeypatch.delenv(k, raising=False))
        monkeypatch.setenv("VLLM_TPU_L6_AB_DIR", str(tmp_path))
        world["n"] = n_world
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab

    yield NS(setup=setup, lines=lines, tmp=tmp_path, mp=monkeypatch)
    _clear_env(lambda k: monkeypatch.delenv(k, raising=False))
    l6_ab._resolve()


def _env(S=20, D=4, T="4", **extra):
    e = {"VLLM_TPU_L6_AB_STEPS": str(S), "VLLM_TPU_L6_AB_CONFIRM": str(D), "VLLM_TPU_L6_AB_CLASSES": "8",
         S3D: "ab:1+2+4+7"}
    if T is not None:
        e[GATE_ENV] = T
    e.update(extra)
    return e


def _so(n):
    return NS(num_scheduled_tokens={f"r{i}": 4 for i in range(n)}, total_num_scheduled_tokens=4 * n)


def _trace(means, S, D, decoy=True):
    """Per-step counts (j = 1, 2, ...) for windows 0..len(means)-1. A window's first S - D steps carry the window's
    target mean exactly ('4-' = 4 with one step one short, so the mean is just below 4; '4.5' alternates 4 and 5);
    its last D steps carry a decoy that would flip the decision if they counted."""
    out = []
    for k, m in enumerate(means):
        lo = 1 if k == 0 else k * S
        steps = list(range(lo, (k + 1) * S))
        counted = [j for j in steps if j % S < S - D]
        if m == "4-":
            vals = [4] * len(counted)
            vals[0] = 3
        elif m == "4.5":
            vals = [4 + (i % 2) for i in range(len(counted))]  # 4, 5, 4, 5, ...: a mean in [4, 5)
        elif m == 4:
            vals = [3 + 2 * (i % 2) for i in range(len(counted))]  # 3, 5, 3, 5, ... (+ a 4 if odd): mean exactly 4
            if len(vals) % 2:
                vals[-1] = 4
        else:
            vals = [int(m) + (1 if i % 2 else -1) * (1 if m >= 1 else 0) for i in range(len(counted))]
            if len(vals) % 2:
                vals[-1] = int(m)
        assert len(vals) == len(counted)
        mean = sum(vals) / len(vals)
        dec = 0 if mean >= 4 else 30
        out.extend(vals + [dec if decoy else int(mean)] * (len(steps) - len(counted)))
    return out


def _oracle(trace, S, D, T, world_confirms=True):
    """Window k >= 1 open iff window k-1's first S - D counted steps average >= T (integer test)."""
    sums, ns = {}, {}
    for j, n in enumerate(trace, start=1):
        if j % S < S - D:
            sums[j // S] = sums.get(j // S, 0) + n
            ns[j // S] = ns.get(j // S, 0) + 1
    nwin = (len(trace)) // S + 1
    openk = {0: False}
    for k in range(1, nwin):
        s, n = sums.get(k - 1, 0), ns.get(k - 1, 0)
        openk[k] = n > 0 and s >= T * n
    return openk, sums, ns


def _run1(m, trace, gate_calls=True, role="worker"):
    st = []
    for n in trace:
        so = _so(n)
        m.begin_step(role=role, fp_fn=lambda so=so: m.sched_fingerprint(so))
        if gate_calls and m.GATE:
            m.gate_step(len(so.num_scheduled_tokens))
        st.append((m.on("s3defer"), m.on("envcache")))
    return st


MEANS = [6, 2, 2, 6, 6, 3, 8, 4, "4-", 5, 1, 4, 12, 0, 7, 2, 2, 9, 3, "4.5", "4.5", 2, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 10,
         2, 5, 5]


# ---------------------------------------------------------------- A. one process


def test_unset_or_zero_is_v8(gt):
    for val in (None, "0", ""):
        env = _env(T=val)
        m = gt.setup(**env)
        assert m.S3D_GATE_T == 0 and m.GATE == 0 and "g3" not in m._state, val
        gt.lines.clear()
        st = _run1(m, [3] * (5 * 20 - 1))
        # the class pattern exactly as v8: window k on iff k % 8 in {1, 2, 4, 7} (single rank: no lockstep)
        for j, (s3d, _) in enumerate(st, start=1):
            assert s3d is ((j // 20) % 8 in ONCLS), (val, j)
        assert not any("S3DEFER gate" in ln for ln in gt.lines), val
    # with the carry gate the module GATE is the carry N, as before
    m = gt.setup(**_env(T=None, VLLM_TPU_L6_CARRY_MIN_REQS="6", VLLM_TPU_L6_DECODE_CARRY="1"))
    assert m.GATE == 6 and m.S3D_GATE_T == 0


def test_unset_fingerprint_files_are_the_scheduler_fingerprint(gt):
    m = gt.setup(n_world=2, **_env(T=None))
    fp = m.sched_fingerprint(_so(2))
    for k in range(1, 4):
        (gt.tmp / f"w{k}_r1").write_text(fp)  # the peer's fingerprint of the same steps
    st = _run1(m, [2] * (3 * 20))
    for k in range(1, 4):
        assert (gt.tmp / f"w{k}_r0").read_text() == fp  # byte-identical to v8's file: no gate suffix
    assert [s for s, _ in st[19:39]] == [True] * 20 and not any(ln.startswith("ERROR") for ln in gt.lines)


def test_validators(gt):
    with pytest.raises(ValueError, match="step windows"):
        gt.setup(**{S3D: "ab:1", "VLLM_TPU_L6_AB_PERIOD_MS": "20000", GATE_ENV: "4"})
    with pytest.raises(ValueError, match="ab:<classes>"):
        gt.setup(**_env(**{S3D: "1"}))
    with pytest.raises(ValueError, match="ab:<classes>"):
        gt.setup(**_env(**{S3D: ""}))
    with pytest.raises(ValueError):
        gt.setup(**_env(T="-1"))
    m = gt.setup(**_env(T="4"))
    assert m.S3D_GATE_T == 4 and m.GATE == 4 and m._state["g3"]["dec"] is None


def test_gate_follows_the_previous_windows_counted_steps(gt):
    S, D, T = 20, 4, 4
    m = gt.setup(**_env(S=S, D=D, T=str(T), VLLM_TPU_L6_ENVS_CACHE="ab:1+3+5+7"))
    trace = _trace(MEANS, S, D)
    st = _run1(m, trace)
    openk, sums, ns = _oracle(trace, S, D, T)
    seen = {}
    for j, (s3d, ec) in enumerate(st, start=1):
        k = j // S
        seen.setdefault(k, set()).add(s3d)
        assert s3d is (openk[k] and k % 8 in ONCLS), (j, k, openk[k])
        assert ec is (k % 8 in {1, 3, 5, 7}), (j, k)  # an ungated ab: lever is untouched by the gate
    assert all(len(v) == 1 for v in seen.values())  # one state per window
    trans = [(openk[k - 1], openk[k]) for k in range(2, len(openk))]
    assert trans.count((False, True)) >= 3 and trans.count((True, False)) >= 3
    # exactly-at-T opens, just-below closes (windows 7 = 4 and 8 = '4-' decide windows 8 and 9)
    assert openk[8] is True and openk[9] is False and openk[12] is True
    assert sum(openk[k] and k % 8 in ONCLS for k in openk) >= 4 and sum(openk[k] and k % 8 not in ONCLS for k in openk) >= 4
    lines = [ln for ln in gt.lines if ln.startswith("S3DEFER gate")]
    assert len(lines) == len(openk)
    on_n = el_n = 0
    for i, ln in enumerate(lines):
        mm = RX_GATE.match(ln)
        assert mm, ln
        rank, k, cls, el, on, mean, n, tt, w, e, o, sh, she = mm.groups()
        k = int(k)
        assert k == i and rank == "0" and cls == str(k % 8) and int(tt) == T
        assert el == str(int(openk[k])) and on == str(int(openk[k] and k % 8 in ONCLS)), ln
        n_exp = ns.get(k - 1, 0) if k else 0
        assert int(n) == n_exp, ln
        assert mean == ("%.3f" % (sums[k - 1] / ns[k - 1]) if k else "nan"), ln
        on_n += on == "1"
        el_n += el == "1"
        assert (int(w), int(e), int(o)) == (i + 1, el_n, on_n)
        assert sh == "%.3f" % (on_n / (i + 1)) and she == ("%.3f" % (on_n / el_n) if el_n else "nan")


def test_last_d_steps_never_count(gt):
    S, D = 20, 4
    m = gt.setup(**_env(S=S, D=D))
    # window 0: 2 requests on its counted steps, 40 on its last D (decoy): window 1 must stay closed
    trace = [2] * (S - D - 1) + [40] * D + [2] * S
    _run1(m, trace)
    assert m._state["g3"]["dec"][:2] == (2, False)
    lines = [ln for ln in gt.lines if ln.startswith("S3DEFER gate")]
    assert RX_GATE.match(lines[1]).group(4) == "0" and RX_GATE.match(lines[1]).group(7) == str(S - D - 1)


def test_per_step_gates_unchanged_and_gate_step_never_raises(gt):
    """The _GATE_KEY trap: gate_step with the per-step gates AND the S3D gate set returns exactly what it returns
    without the S3D gate, step by step, and never raises (s3defer is not a GATES entry)."""
    S, D = 20, 4
    trace = _trace(MEANS[:12], S, D)
    runs = {}
    for T in (None, "4"):
        m = gt.setup(**_env(S=S, D=D, T=T, VLLM_TPU_L6_DECODE_CARRY="1", VLLM_TPU_L6_STEADY_STEP="ab:0+1+2+3",
                            VLLM_TPU_L6_CARRY_MIN_REQS="5", VLLM_TPU_L6_STEADY_MIN_REQS="3"))
        out = []
        for n in trace:
            m.begin_step()
            off = m.gate_step(n)
            out.append((off, m.on("carry"), m.on("steady")))
            assert "s3defer" not in off
        runs[T] = (out, [ln for ln in gt.lines if ln.startswith("q38 L6AB")])
        gt.lines.clear()
    assert runs[None][0] == runs["4"][0]
    strip = lambda ls: [re.sub(r"on=\S+", "", ln) for ln in ls]  # s3defer's own on= differs by design
    assert strip(runs[None][1]) == strip(runs["4"][1])


def test_engine_core_prints_no_line_and_no_count_keeps_windows_closed(gt):
    m = gt.setup(**_env())
    st = _run1(m, [9] * 99, role="engine_core")
    assert not any("S3DEFER gate" in ln for ln in gt.lines)
    m = gt.setup(**_env())
    st = _run1(m, [9] * 99, gate_calls=False)  # a runner that never calls gate_step: S3D stays off (fail-safe)
    assert not any(s for s, _ in st)
    lines = [ln for ln in gt.lines if ln.startswith("S3DEFER gate")]
    assert lines and all(RX_GATE.match(ln).group(4) == "0" and RX_GATE.match(ln).group(7) == "0" for ln in lines)


def test_fingerprint_carries_the_decision_and_a_disagreeing_peer_disables(gt):
    S, D = 20, 4
    m = gt.setup(n_world=2, **_env(S=S, D=D, VLLM_TPU_L6_ENVS_CACHE="ab:1+3+5+7"))
    trace = [6] * (6 * S)
    so = {j: _so(n) for j, n in enumerate(trace, start=1)}
    sums = {0: 6 * (S - D - 1)}
    for k in range(1, 6):
        sums[k] = 6 * (S - D)
    for j in range(1, len(trace) + 1):
        k = (j + D) // S
        if (j + D) % S == 0:
            fp = m.sched_fingerprint(so[j])
            n = S - D - 1 if k == 1 else S - D
            dec = f"{k}:1:{sums[k - 1]}:{n}"
            if k == 4:
                dec = f"{k}:0:{sums[k - 1]}:{n}"  # the peer decided otherwise at window 4
            (gt.tmp / f"w{k}_r1").write_text(f"{fp}|s3dgate={dec}")
        m.begin_step(fp_fn=lambda j=j: m.sched_fingerprint(so[j]))
        m.gate_step(len(so[j].num_scheduled_tokens))
        w = j // S
        if 1 <= w and j < 4 * S - 1:  # windows 1-3 confirmed and open
            assert m.on("s3defer") is (w % 8 in ONCLS) and m.on("envcache") is (w % 2 == 1), j
        if j >= 4 * S - 1:  # window 4's confirm (its step 4S - 1) disables every ab: lever from that step on
            assert not m.on("s3defer") and not m.on("envcache"), j
    assert (gt.tmp / "w2_r0").read_text().endswith(f"|s3dgate=2:1:{sums[1]}:{S - D}")
    errs = [ln for ln in gt.lines if ln.startswith("ERROR q38 L6AB rank 0 DISABLED")]
    assert len(errs) == 1 and "window 4" in errs[0], errs
    # the windows after the disable are logged as not eligible (not armed)
    g = {int(RX_GATE.match(ln).group(2)): RX_GATE.match(ln).group(4) for ln in gt.lines if ln.startswith("S3DEFER gate")}
    assert g[1] == g[2] == g[3] == "1" and g[4] == g[5] == "0"


def test_runner_hook_records_the_count_with_only_the_s3d_gate(gt):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    m = gt.setup(**_env())
    assert m.GATE == 4 and m.MIN_REQS == {}  # so execute_model's 'if _l6_ab.GATE:' calls the hook
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.input_batch = NS(num_reqs=0, req_ids=[])
    m.begin_step()
    r._l6_gate_step(_so(7))
    m.begin_step()
    r._l6_gate_step(_so(3))
    assert (m._state["g3"]["acc_k"], m._state["g3"]["acc_sum"], m._state["g3"]["acc_n"]) == (0, 10, 2)


# ---------------------------------------------------------------- B. four gloo ranks


def _rank_main(rank, init_file, store, env, trace, q):
    try:
        _clear_env(lambda k: os.environ.pop(k, None))
        os.environ.update(env)
        os.environ["VLLM_TPU_L6_AB_DIR"] = store
        import torch.distributed as dist

        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        from vllm_torchtpu.core import l6_ab

        lines = []
        l6_ab.logger.info = lambda fmt, *a: lines.append(fmt % a)
        l6_ab.logger.error = lambda fmt, *a: lines.append("ERROR " + fmt % a)
        l6_ab._resolve()
        states = []
        for n in trace:
            so = _so(n)
            l6_ab.begin_step(fp_fn=lambda so=so: l6_ab.sched_fingerprint(so))
            if l6_ab.GATE:
                l6_ab.gate_step(len(so.num_scheduled_tokens))
            states.append((l6_ab.on("s3defer"), l6_ab.on("envcache")))
            dist.barrier()  # the step's device collectives keep the ranks in step
        q.put((rank, states, lines, l6_ab._state["rank_world"]))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001
        import traceback

        q.put((rank, None, [traceback.format_exc()[-2000:] + repr(exc)], None))


def _run4(envs, trace):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    timeout = int(os.environ.get("Q38_S3DGATE_TIMEOUT", "600"))
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "store")
        os.makedirs(store)
        procs = [ctx.Process(target=_rank_main, args=(r, os.path.join(d, "init"), store, envs[r], trace, q))
                 for r in range(WORLD)]
        for p in procs:
            p.start()
        res, deadline = [], time.time() + timeout
        try:
            for _ in procs:
                res.append(q.get(timeout=max(1, deadline - time.time())))
        except Exception as exc:  # noqa: BLE001 - a hang is a failure
            res.append((-1, None, [f"timeout: {len(res)} of {WORLD} ranks reported: {exc!r}"], None))
        for p in procs:
            p.join(timeout=20)
            if p.is_alive():
                p.kill()
    res = sorted(res, key=lambda x: x[0])
    for rank, states, lines, rw in res:
        assert states is not None, (rank, lines)
    return res


S4, D4, T4 = 250, 16, 4


def _envs4(t_rank1="4"):
    base = _env(S=S4, D=D4, T="4", VLLM_TPU_L6_ENVS_CACHE="ab:1+3+5+7")
    out = []
    for r in range(WORLD):
        e = dict(base)
        if r == 1:
            e[GATE_ENV] = t_rank1
        out.append(e)
    return out


def _gate_rows(lines):
    rows = []
    for ln in lines:
        if ln.startswith("S3DEFER gate"):
            mm = RX_GATE.match(ln)
            assert mm, ln
            rows.append(mm.groups()[1:])  # without the rank
    return rows


def test_four_ranks_identical_gate_state_every_step_and_window():
    trace = _trace(MEANS, S4, D4)
    res = _run4(_envs4(), trace)
    openk, sums, ns = _oracle(trace, S4, D4, T4)
    ref_states, ref_rows = res[0][1], _gate_rows(res[0][2])
    for rank, states, lines, rw in res:
        assert rw == (rank, WORLD), rw
        assert not [ln for ln in lines if ln.startswith("ERROR")], (rank, [ln for ln in lines if "ERROR" in ln][:2])
        assert states == ref_states, rank  # every step
        assert _gate_rows(lines) == ref_rows, rank  # every window
    for j, (s3d, ec) in enumerate(ref_states, start=1):
        k = j // S4
        assert s3d is (k >= 1 and openk[k] and k % 8 in ONCLS), (j, k)
        assert ec is (k >= 1 and k % 8 in {1, 3, 5, 7}), (j, k)
    assert len(ref_rows) == len(openk)
    for k, row in enumerate(ref_rows):
        assert int(row[0]) == k and row[2] == str(int(k >= 1 and openk[k])), (k, row)
    trans = [(openk[k - 1], openk[k]) for k in range(2, len(openk))]
    assert trans.count((False, True)) >= 3 and trans.count((True, False)) >= 3
    el = [k for k in openk if openk[k]]
    print("Q38S3DGATE_PROTOCOL windows=%d eligible=%d eligible_on=%d steps=%d ranks=%d" % (
        len(openk), len(el), sum(k % 8 in ONCLS for k in el), len(trace), len(res)))


def test_one_rank_with_a_different_threshold_disables_every_rank_together():
    trace = _trace(MEANS, S4, D4)
    res = _run4(_envs4(t_rank1="5"), trace)
    o4, _, _ = _oracle(trace, S4, D4, 4)
    o5, _, _ = _oracle(trace, S4, D4, 5)
    first = min(k for k in o4 if o4[k] != o5[k])
    ref = res[0][1]
    for rank, states, lines, rw in res:
        assert states == ref, rank  # identical on every rank at every step: the fail-safe
        errs = [ln for ln in lines if ln.startswith(f"ERROR q38 L6AB rank {rank} DISABLED")]
        assert len(errs) == 1 and f"window {first}:" in errs[0], (rank, errs)
    for j, (s3d, ec) in enumerate(ref, start=1):
        k = j // S4
        if j >= first * S4 - 1:  # from the confirm step of window `first` (its predecessor's last step) on
            assert not s3d and not ec, (j, k)
        else:
            assert s3d is (k >= 1 and o4[k] and k % 8 in ONCLS), (j, k)
