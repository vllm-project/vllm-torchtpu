# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST: step-time lever switches / A/B windows (core/l6_ab.py) and the
envcache lever (core/l6_envcache.py). Host-only; runs in a CDK CPU pod."""

import ast
import os
import pathlib

import pytest

P = 20000  # window period, ms
NS = 1_000_000


def _t(window: int, frac: float = 0.5) -> int:
    """A wall-clock time (ns) inside window `window`."""
    return int((window + frac) * P * NS)


@pytest.fixture
def ab(monkeypatch):
    from vllm_torchtpu.core import l6_ab

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    lines = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    clock = {"t": _t(100)}
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])

    def setup(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab

    yield setup, lines, clock
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def test_no_flag_is_inactive(ab):
    setup, lines, _ = ab
    m = setup()
    assert not m.ACTIVE
    m.begin_step()
    assert m.on("envcache") is False
    assert lines == []


def test_uniform_on_every_step_and_logs_each_20s(ab):
    setup, lines, clock = ab
    m = setup(VLLM_TPU_L6_ENVS_CACHE="1")
    assert m.ACTIVE
    for _ in range(3):
        m.begin_step()
        assert m.on("envcache")
    clock["t"] += 20000 * NS  # next 20 s uniform log window
    m.begin_step()
    assert m.on("envcache")
    assert len(lines) == 1 and "class=uniform steps=3 on=envcache" in lines[0], lines


def test_ab_classes_select_windows(ab):
    setup, lines, clock = ab
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_ENVS_CACHE="ab:1,3")
    seen = {}
    for w in range(100, 108):
        clock["t"] = _t(w, 0.1)
        m.begin_step()
        first = m.on("envcache")
        clock["t"] = _t(w, 0.9)  # later in the same window: state is fixed per window
        m.begin_step()
        assert m.on("envcache") == first
        seen[w % 4] = first
    assert seen == {0: False, 1: True, 2: False, 3: True}
    # one line per finished window, with its class, step count and lever state
    assert len(lines) == 7
    assert lines[0] == "q38 L6AB role=worker window=100 class=0 steps=2 on=- counts=-"
    assert lines[1] == "q38 L6AB role=worker window=101 class=1 steps=2 on=envcache counts=-"


def test_step_windows_align_with_hostphases_windows(ab):
    """With VLLM_TPU_L6_AB_STEPS=S the j-th step (j from 1) is in window j // S, so the
    HOSTPHASES window logged with steps=T (steps T-S .. T-1, runner/host_phases.py)
    lies in exactly one L6 window, (T - 1) // S."""
    setup, lines, clock = ab
    S = 4
    m = setup(VLLM_TPU_L6_AB_STEPS=str(S), VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_ENVS_CACHE="ab:1")
    state = {}
    for j in range(1, 17):
        m.begin_step()
        state[j] = m.on("envcache")
    for T in (8, 12, 16):  # HOSTPHASES lines with steps=T cover steps T-S .. T-1
        cover = {state[j] for j in range(T - S, T)}
        assert len(cover) == 1, (T, cover)
        assert cover.pop() == (((T - 1) // S) % 2 == 1)
    assert [state[j] for j in range(1, 13)] == [False] * 3 + [True] * 4 + [False] * 4 + [True]
    assert lines[0] == "q38 L6AB role=worker window=0 class=0 steps=3 on=- counts=-"
    assert lines[1] == "q38 L6AB role=worker window=1 class=1 steps=4 on=envcache counts=-"


def test_period_and_steps_are_exclusive(ab):
    setup, _, _ = ab
    with pytest.raises(ValueError):
        setup(VLLM_TPU_L6_AB_STEPS="200", VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_ENVS_CACHE="ab:1")


def test_plus_separates_classes_like_comma(ab):
    """ab:1+3 (served envs are ','-split by the recipe) parses exactly as ab:1,3."""
    setup, _, _ = ab
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_ENVS_CACHE="ab:1+3")
    plus = m._cfg["envcache"]
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_ENVS_CACHE="ab:1,3")
    assert plus == m._cfg["envcache"] == (False, frozenset({1, 3}))
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_GC_FREEZE="ab:2+3")
    plus = m._cfg["gcfreeze"]
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_AB_CLASSES="4", VLLM_TPU_L6_GC_FREEZE="ab:2,3")
    assert plus == m._cfg["gcfreeze"] == (False, frozenset({2, 3}))


def test_ab_needs_a_period_and_valid_classes(ab):
    setup, _, _ = ab
    with pytest.raises(ValueError):
        setup(VLLM_TPU_L6_ENVS_CACHE="ab:1")
    with pytest.raises(ValueError):
        setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_ENVS_CACHE="ab:2")  # 2 classes: 0, 1
    with pytest.raises(ValueError):
        setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_ENVS_CACHE="maybe")


def test_ab_refused_next_to_another_windowed_ab(ab):
    setup, lines, clock = ab
    m = setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_ENVS_CACHE="ab:1", VLLM_TPU_HOST_AB_PERIOD_MS="20000")
    assert m.REFUSED
    for w in (101, 103):
        clock["t"] = _t(w)
        m.begin_step()
        assert m.on("envcache") is False
    assert sum(ln.startswith("ERROR q38 L6AB") for ln in lines) == 1


def test_note_counts_land_in_the_window_line(ab):
    setup, lines, clock = ab
    m = setup(VLLM_TPU_L6_ENVS_CACHE="1")
    m.begin_step()
    m.note("envcache", "steps", 2)
    m.note("envcache", "steps")
    clock["t"] += 20000 * NS
    m.begin_step()
    assert lines[-1].endswith("counts=envcache.steps=3"), lines


# ---------------------------------------------------------------- envcache


@pytest.fixture
def ec(ab, monkeypatch):
    setup, lines, clock = ab
    from vllm_torchtpu import envs
    from vllm_torchtpu.core import l6_envcache

    calls = {"n": 0}
    name = "VLLM_TPU_HOST_PHASES"  # a real registered flag
    real = envs.environment_variables[name]

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setitem(envs.environment_variables, name, counting)
    eclines = []
    monkeypatch.setattr(l6_envcache.logger, "info", lambda fmt, *a: eclines.append(fmt % a))
    monkeypatch.setattr(l6_envcache.logger, "warning", lambda fmt, *a: eclines.append("WARN " + fmt % a))
    l6_envcache.restore()
    yield setup, clock, envs, l6_envcache, calls, name, eclines
    l6_envcache.restore()
    monkeypatch.setattr(l6_envcache, "CENSUS", 0)
    monkeypatch.setattr(l6_envcache, "CHECK", 0)


def test_cached_reads_skip_the_parser_and_keep_the_value(ec, monkeypatch):
    setup, _, envs, m, calls, name, _ = ec
    monkeypatch.setenv(name, "7")
    m.apply(True)
    vals = [getattr(envs, name) for _ in range(5)]
    assert vals == [7] * 5
    assert calls["n"] == 1
    m.apply(False)
    calls["n"] = 0
    vals = [getattr(envs, name) for _ in range(3)]
    assert vals == [7] * 3 and calls["n"] == 3


def test_off_clears_the_cache(ec, monkeypatch):
    setup, _, envs, m, calls, name, _ = ec
    monkeypatch.setenv(name, "7")
    m.apply(True)
    assert getattr(envs, name) == 7
    m.apply(False)
    monkeypatch.setenv(name, "9")
    m.apply(True)
    assert getattr(envs, name) == 9  # a stale 7 would mean the cache survived the off window


def test_lever_follows_the_ab_windows(ec, monkeypatch):
    setup, clock, envs, m, calls, name, _ = ec
    monkeypatch.setenv(name, "5")
    setup(VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_L6_ENVS_CACHE="ab:1")
    from vllm_torchtpu.core import l6_ab

    states = []
    for w in (100, 101, 102, 103):
        clock["t"] = _t(w)
        l6_ab.begin_step()
        m.on_step()
        calls["n"] = 0
        for _ in range(4):
            assert getattr(envs, name) == 5
        states.append(calls["n"])
    # class 0: raw (4 parses); class 1: cached (at most one parse)
    assert states[0] == 4 and states[2] == 4
    assert states[1] <= 1 and states[3] <= 1


def test_check_mode_drops_a_stale_cache(ec, monkeypatch):
    setup, clock, envs, m, calls, name, eclines = ec
    monkeypatch.setenv(name, "3")
    m.apply(True)
    assert getattr(envs, name) == 3
    assert m.check_now() == 0
    monkeypatch.setenv(name, "4")  # a runtime write the cache cannot see
    assert getattr(envs, name) == 3
    assert m.check_now() == 1
    assert getattr(envs, name) == 4
    assert any(ln.startswith("WARN q38 L6 envcache CHECK mismatch on " + name) for ln in eclines)


def test_check_mode_runs_from_on_step(ec, monkeypatch):
    setup, clock, envs, m, calls, name, _ = ec
    monkeypatch.setattr(m, "CHECK", 2)
    monkeypatch.setenv(name, "3")
    setup(VLLM_TPU_L6_ENVS_CACHE="1")
    from vllm_torchtpu.core import l6_ab

    l6_ab.begin_step(); m.on_step()
    assert getattr(envs, name) == 3
    monkeypatch.setenv(name, "8")
    l6_ab.begin_step(); m.on_step()  # 2nd cached step: check runs and finds the stale value
    assert getattr(envs, name) == 8
    assert l6_ab._state["counts"].get("envcache.mismatch") == 1


def test_census_counts_reads_per_step(ec, monkeypatch):
    setup, clock, envs, m, calls, name, eclines = ec
    monkeypatch.setattr(m, "CENSUS", 2)
    monkeypatch.setenv(name, "1")
    for _ in range(2):
        m.on_step()
        for _ in range(3):
            getattr(envs, name)
    m.on_step()  # the third step's call closes the window of 2 complete steps (6 reads)
    lines = [ln for ln in eclines if ln.startswith("q38 L6ENVCENSUS")]
    assert lines == [f"q38 L6ENVCENSUS steps=2 reads=6 reads_per_step=3.0 top={name}:6"], lines


def test_runner_calls_the_hooks_in_execute_model():
    """execute_model runs l6_ab.begin_step() and l6_envcache.on_step() under one guard."""
    import vllm_torchtpu.runner.tpu_runner as tr

    src = pathlib.Path(tr.__file__).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    body = ast.unparse(fn)
    assert "if _l6_ab.ACTIVE or _l6_envcache.CENSUS:" in body
    i = body.index("if _l6_ab.ACTIVE or _l6_envcache.CENSUS:")
    seg = body[i:i + 240]
    assert "_l6_ab.begin_step(fp_fn=lambda: _l6_ab.sched_fingerprint(scheduler_output))" in seg
    assert "_l6_envcache.on_step()" in seg and "_l6_jitter.on_step()" in seg
    assert body.index("host_phases.begin_step()") < i < body.index("self._update_states(scheduler_output)")


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
