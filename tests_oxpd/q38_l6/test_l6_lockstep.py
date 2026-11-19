# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST: the l6_ab lockstep check (SAB's fingerprint protocol, q38 L5, credited) and
the composition rules with D1-norm windows and L5's SAB. Host-only; CDK CPU pod."""

import os

import pytest

S, D = 8, 2  # step window and confirmation delay used below


@pytest.fixture
def lk(monkeypatch, tmp_path):
    from vllm_torchtpu.core import l6_ab

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_L6_AB_DIR", str(tmp_path))
    lines = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    world = {"n": 2}
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, world["n"]))

    def setup(n_world=2, **env):
        world["n"] = n_world
        env.setdefault("VLLM_TPU_L6_AB_STEPS", str(S))
        env.setdefault("VLLM_TPU_L6_AB_CONFIRM", str(D))
        env.setdefault("VLLM_TPU_L6_ENVS_CACHE", "ab:1")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab

    yield setup, lines, tmp_path
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k == "VLLM_TPU_SERVED_AB":
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _fp(j):
    return f"fp{j}"


def _peer(tmp_path, k, fp, rank=1):
    (tmp_path / f"w{k}_r{rank}").write_text(fp)


def _run(m, steps):
    state = {}
    for j in range(1, steps + 1):
        m.begin_step(fp_fn=lambda j=j: _fp(j))
        state[j] = m.on("envcache")
    return state


def _window_states(state):
    """Lever state per window k (steps k*S .. k*S+S-1, j from 1); asserts one state per window."""
    out = {}
    for j, v in state.items():
        out.setdefault(j // S, set()).add(v)
    for k, vs in out.items():
        assert len(vs) == 1, (k, vs)
    return {k: vs.pop() for k, vs in out.items()}


def test_two_ranks_confirm_each_window(lk):
    setup, lines, tmp = lk
    m = setup()
    for k in range(1, 6):
        _peer(tmp, k, _fp(k * S - D))  # the other rank's fingerprint of the same step
    st = _window_states(_run(m, 5 * S - 1))
    # window 0 is never confirmed; then class 1 on / class 0 off, confirmed each time
    assert st == {0: False, 1: True, 2: False, 3: True, 4: False}
    assert (tmp / "w1_r0").read_text() == _fp(S - D)
    assert not any(ln.startswith("ERROR") for ln in lines)


def test_desynchronised_rank_disables_the_rest_of_the_run(lk):
    setup, lines, tmp = lk
    m = setup()
    for k in range(1, 6):
        _peer(tmp, k, _fp(k * S - D) if k != 3 else _fp(k * S - D + 1))  # rank 1 a step off at window 3
    st = _window_states(_run(m, 5 * S - 1))
    assert st == {0: False, 1: True, 2: False, 3: False, 4: False}
    errs = [ln for ln in lines if ln.startswith("ERROR q38 L6AB rank 0 DISABLED")]
    assert len(errs) == 1 and "window 3" in errs[0], lines


def test_missing_rank_keeps_ab_levers_off(lk):
    setup, lines, tmp = lk
    m = setup()
    st = _window_states(_run(m, 3 * S - 1))
    assert st == {0: False, 1: False, 2: False}
    assert sum(ln.startswith("ERROR q38 L6AB rank 0 DISABLED") for ln in lines) == 1


def test_uniform_levers_ignore_the_lockstep(lk):
    setup, lines, tmp = lk
    m = setup(VLLM_TPU_L6_ENVS_CACHE="1", VLLM_TPU_L6_FAST_FWD_CTX="ab:1")
    for j in range(1, 2 * S):
        m.begin_step(fp_fn=lambda j=j: _fp(j))
        assert m.on("envcache") is True


def test_single_rank_needs_no_files(lk):
    setup, lines, tmp = lk
    m = setup(n_world=1)
    st = _window_states(_run(m, 3 * S - 1))
    assert st == {0: False, 1: True, 2: False}
    assert not list(tmp.iterdir())


def test_d1n_windows_compose(lk):
    setup, lines, tmp = lk
    m = setup(n_world=1, VLLM_TPU_D1N_AB_PERIOD_MS="20000")
    assert m.REFUSED is False
    assert _window_states(_run(m, 2 * S - 1)) == {0: False, 1: True}


def test_balanced_inside_sab_windows(lk):
    """Every SAB window (P calls) holds equal shares of each L6 class when S * CLASSES divides P,
    whatever the offset between the two counters; otherwise the ab: levers are refused."""
    setup, lines, tmp = lk
    m = setup(n_world=1, VLLM_TPU_SERVED_AB="whole:4000:64", VLLM_TPU_L6_AB_STEPS="200",
              VLLM_TPU_L6_AB_CONFIRM="16", VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_ENVS_CACHE="ab:1,3")
    assert m.REFUSED is False
    for offset in (0, 37, 199, 401, 799):
        counts = [0] * 4
        for j in range(offset + 1, offset + 4001):
            counts[(j // 200) % 4] += 1
        assert counts == [1000] * 4, (offset, counts)
    m = setup(n_world=1, VLLM_TPU_SERVED_AB="whole:4000:64", VLLM_TPU_L6_AB_STEPS="300",
              VLLM_TPU_L6_AB_CONFIRM="16", VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_ENVS_CACHE="ab:1,3")
    assert m.REFUSED is True
    m.begin_step()
    assert any("not balanced inside SAB" in ln for ln in lines if ln.startswith("ERROR"))


def test_fingerprint_is_the_scheduler_output():
    from types import SimpleNamespace as NS

    from vllm_torchtpu.core import l6_ab

    a = NS(num_scheduled_tokens={"r1": 4, "r2": 4}, total_num_scheduled_tokens=8)
    b = NS(num_scheduled_tokens={"r1": 4, "r2": 4}, total_num_scheduled_tokens=8)
    c = NS(num_scheduled_tokens={"r2": 4, "r1": 4}, total_num_scheduled_tokens=8)
    assert l6_ab.sched_fingerprint(a) == l6_ab.sched_fingerprint(b)
    assert l6_ab.sched_fingerprint(a) != l6_ab.sched_fingerprint(c)  # batch order matters


@pytest.mark.parametrize("spec", ["", "  ", "whole:4000", "whole:4000:32", "whole", "whole:100:64", "whole:10:0",
                                  "seg:4000", "whole:x", "whole:4000:64:1"])
def test_sab_period_follows_served_ab_parse(monkeypatch, spec):
    """l6_ab reads SAB's period through served_ab.parse: same value on every valid spec (L5's
    test_parse cases), and SAB off (None) wherever parse rejects the spec."""
    from vllm_torchtpu.core import l6_ab
    from vllm_torchtpu.megakernel_runtime import served_ab

    monkeypatch.setenv("VLLM_TPU_SERVED_AB", spec)
    try:
        want = served_ab.parse(spec)
        want = None if want is None else want[0]
    except ValueError:
        want = None
    assert l6_ab._sab_period() == want


def test_check_window_closes_after_n_steps(lk):
    """VLLM_TPU_L6_CHECK_STEPS=N: check_allowed() is True for worker steps 1..N, False after,
    with one INFO line; envcache's and steady's CHECK modes follow it."""
    setup, lines, tmp = lk
    m = setup(n_world=1, VLLM_TPU_L6_ENVS_CACHE="1", VLLM_TPU_L6_CHECK_STEPS="5")
    from vllm_torchtpu.runner import decode_carry, l6_steady

    import os as _os

    l6_steady._CHECK[0] = 7
    _os.environ["VLLM_TPU_L6_DECODE_CARRY_CHECK"] = "3"
    seen = []
    try:
        for _ in range(8):
            m.begin_step()
            seen.append((m.check_allowed(), l6_steady.check_every(), decode_carry._check_every()))
    finally:
        l6_steady._CHECK[0] = None
        del _os.environ["VLLM_TPU_L6_DECODE_CARRY_CHECK"]
    assert seen == [(True, 7, 3)] * 5 + [(False, 0, 0)] * 3, seen
    assert sum("q38 L6 CHECK window closed after 5 worker steps" in ln for ln in lines) == 1


def test_no_bound_keeps_checks_unbounded(lk):
    setup, lines, tmp = lk
    m = setup(n_world=1, VLLM_TPU_L6_ENVS_CACHE="1")
    for _ in range(50):
        m.begin_step()
        assert m.check_allowed()


def test_sixteen_class_factorial_with_lockstep(lk, tmp_path):
    """STACK-D v4's 2^4 factorial: CLASSES=16 with four levers on the bit patterns of the class (A bit 0, B bit 1,
    C bit 2, D bit 3). Two ranks in lockstep: every confirmed window k >= 1 turns on exactly the levers whose bit is set
    in k mod 16 (window 0 is never confirmed). S * 16 must divide SAB's period: 250 * 16 = 4000 is accepted, 200 * 16 is
    refused."""
    setup, lines, tmp = lk
    bits = {"envcache": 0, "gcfreeze": 1, "carry": 2, "steady": 3}
    env = {"VLLM_TPU_L6_ENVS_CACHE": "", "VLLM_TPU_L6_GC_FREEZE": "", "VLLM_TPU_L6_DECODE_CARRY": "",
           "VLLM_TPU_L6_STEADY_STEP": ""}
    keys = {"envcache": "VLLM_TPU_L6_ENVS_CACHE", "gcfreeze": "VLLM_TPU_L6_GC_FREEZE",
            "carry": "VLLM_TPU_L6_DECODE_CARRY", "steady": "VLLM_TPU_L6_STEADY_STEP"}
    for lev, b in bits.items():
        env[keys[lev]] = "ab:" + "+".join(str(c) for c in range(16) if c >> b & 1)
    m = setup(n_world=2, VLLM_TPU_L6_AB_CLASSES="16", **env)
    assert m.CLASSES == 16 and not m.REFUSED
    n_win = 2 * 16 + 1
    for k in range(1, n_win + 1):
        _peer(tmp_path, k, _fp(k * S - D))
    seen = {}
    for j in range(1, n_win * S + 1):
        m.begin_step(fp_fn=lambda j=j: _fp(j))
        k = j // S
        seen.setdefault(k, set())
        seen[k].add(tuple(sorted(lev for lev in bits if m.on(lev))))
    assert not any("DISABLED" in ln for ln in lines), [ln for ln in lines if "ERROR" in ln]
    assert seen[0] == {()}
    for k in range(1, n_win):
        want = tuple(sorted(lev for lev, b in bits.items() if (k % 16) >> b & 1))
        assert seen[k] == {want}, (k, seen[k], want)
    # SAB balance at 16 classes
    m = setup(n_world=1, VLLM_TPU_SERVED_AB="whole:4000:64", VLLM_TPU_L6_AB_STEPS="250",
              VLLM_TPU_L6_AB_CONFIRM="16", VLLM_TPU_L6_AB_CLASSES="16", **env)
    assert m.REFUSED is False
    m = setup(n_world=1, VLLM_TPU_SERVED_AB="whole:4000:64", VLLM_TPU_L6_AB_STEPS="200",
              VLLM_TPU_L6_AB_CONFIRM="16", VLLM_TPU_L6_AB_CLASSES="16", **env)
    assert m.REFUSED is True
