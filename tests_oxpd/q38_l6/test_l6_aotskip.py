# SPDX-License-Identifier: Apache-2.0
"""q38 L6-H2 lever aotskip: CF-2 AOTSKIP-AB (compilation/l6_aotskip.py). Host-only; runs in a
CDK CPU pod on the served image (torch's AOTCompiledFunction and vLLM's
TorchCompileWithNoGuardsWrapper.aot_compile are real here).

The core tests compile real AOT artifacts through vLLM's own aot_compile with the lever's
patch installed, drive l6_ab step windows (lever on in class 1), call the worker hook
(``on_step``) once per step as execute_model does, and count guard evaluations with a spy on
the artifact: exactly one per call in lever-off steps, none in lever-on steps, and every
output equal to the eager function's, bitwise. A wrong-shape call fails the shape guard in
lever-off steps only. A cache-loaded artifact (guard check already off, as vLLM leaves it) is
never switched on. An artifact whose guards vLLM evaluates (evaluate_guards True or no config on
a fresh compile; a cache-loaded artifact vLLM left with its check on) is never kept, so its guard
is evaluated, and a wrong shape raises, in every window.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.compilation import l6_aotskip as la
from vllm_torchtpu.core import l6_ab

ENV = "VLLM_TPU_L6_AOT_SKIP"
CHECK = "VLLM_TPU_L6_AOT_SKIP_CHECK"
S = 10  # l6_ab step-window length


def serializable_eager(gm, example_inputs):
    """torch's AOT compile needs a SerializableCallable (as tests/compilation/test_aot_guard_skip.py)."""
    import torch._dynamo.aot_compile as aotc

    G = getattr(aotc, "GraphModuleSerializableCallable", None)
    if G is None:
        try:
            from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable as G
        except Exception:  # noqa: BLE001
            G = None
    if G is not None:
        return G(gm)

    class _Eager(aotc.SerializableCallable):
        def __init__(self, g):
            self.g = g

        @classmethod
        def serialize_compile_artifacts(cls, fn):
            return b""

        @classmethod
        def deserialize_compile_artifacts(cls, data):
            raise NotImplementedError

        def __call__(self, *a, **k):
            return self.g(*a, **k)

    return _Eager(gm)


SCALE = 2


def f1(x):
    return x * SCALE + 1


@pytest.fixture
def wrapped(monkeypatch):
    """The lever's patch installed over vLLM's own aot_compile / loader (restored after)."""
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "VLLM_TPU_HOST_AB_PERIOD_MS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(W, "aot_compile", W.aot_compile)
    monkeypatch.setattr(decorators, "_try_load_aot_compiled_fn", decorators._try_load_aot_compiled_fn)
    la.reset()
    yield NS(mp=monkeypatch, W=W, decorators=decorators)
    la.reset()
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _arm(mp, value="ab:1", **extra):
    mp.setenv(ENV, value)
    mp.setenv("VLLM_TPU_L6_AB_STEPS", str(S))
    mp.setenv("VLLM_TPU_L6_AB_CONFIRM", "4")  # l6_ab requires 1 <= CONFIRM < STEPS (default 16)
    for k, v in extra.items():
        mp.setenv(k, v)
    l6_ab._resolve()
    la.install()


def _compile(W, fn=f1, x=None, evaluate=False, config=True):
    """A real AOT artifact through vLLM's aot_compile; the owner's evaluate_guards is `evaluate`
    (`config=False`: the owner has no compilation_config at all)."""
    x = torch.arange(4, dtype=torch.float32) if x is None else x
    torch._dynamo.reset()
    with torch._dynamo.config.patch(enable_aot_compile=True):
        c = torch.compile(fn, fullgraph=True, dynamic=False, backend=serializable_eager)
    owner = NS(_compiled_callable=c)
    if config:
        owner.compilation_config = NS(dynamic_shapes_config=NS(evaluate_guards=evaluate))
    return W.aot_compile(owner, x)


class Spy:
    """Counts guard evaluations of one artifact: its per-call guard_check (the attribute
    AOTCompiledFunction.__call__ calls when _guard_check_enabled is True)."""

    def __init__(self, fn):
        self.n = 0
        orig = fn.guard_check

        def spy(*a, **k):
            self.n += 1
            return orig(*a, **k)

        fn.guard_check = spy


def _drive(fn, steps=200, seed=0, bad_every=0):
    """One step per iteration as the worker runs it: begin_step, on_step, then the calls.
    `evals` counts the guard evaluations of the step's valid call; with `bad_every` a
    wrong-shape call follows and `guard_error` says whether it failed the shape guard."""
    g = torch.Generator().manual_seed(seed)
    spy = Spy(fn)
    rows = []
    for _ in range(steps):
        l6_ab.begin_step()
        if la._handles:
            la.on_step()
        on = l6_ab.on("aotskip")
        n0 = spy.n
        x = torch.randn(4, generator=g)
        out = fn(x)
        evals = spy.n - n0
        guard_error = None
        if bad_every:
            try:
                fn(torch.arange(5, dtype=torch.float32))
                guard_error = False
            except Exception as e:  # noqa: BLE001
                assert "uard" in str(e), repr(e)
                guard_error = True
        rows.append(dict(on=on, evals=evals, equal=torch.equal(out, f1(x)), guard_error=guard_error))
    return rows


def test_windows_flip_guard_evaluation_on_a_real_artifact(wrapped):
    _arm(wrapped.mp)
    fn = _compile(wrapped.W)
    assert type(fn).__name__ == "AOTCompiledFunction"
    assert len(la._handles) == 1 and la._handles[0][1] is True
    rows = _drive(fn, steps=200)
    on = [r for r in rows if r["on"]]
    off = [r for r in rows if not r["on"]]
    assert len(on) >= 80 and len(off) >= 80, (len(on), len(off))
    assert all(r["evals"] == 0 for r in on), [i for i, r in enumerate(rows) if r["on"] and r["evals"]]
    assert all(r["evals"] == 1 for r in off), [i for i, r in enumerate(rows) if not r["on"] and r["evals"] != 1]
    assert all(r["equal"] for r in rows)
    c = l6_ab._state["counts"]
    print(f"AOTSKIP_WINDOWS on={len(on)} off={len(off)} counts={dict(sorted(c.items()))}")


def test_wrong_shape_fails_the_guard_only_with_the_lever_off(wrapped):
    _arm(wrapped.mp)
    fn = _compile(wrapped.W)
    rows = _drive(fn, steps=60, bad_every=1)
    assert all(r["guard_error"] is (not r["on"]) for r in rows), [(r["on"], r["guard_error"]) for r in rows]
    assert any(r["on"] for r in rows) and any(not r["on"] for r in rows)


def test_uniform_on_and_off(wrapped):
    _arm(wrapped.mp, "1")
    fn = _compile(wrapped.W)
    rows = _drive(fn, steps=30)
    assert all(r["on"] and r["evals"] == 0 and r["equal"] for r in rows)
    wrapped.mp.setenv(ENV, "0")
    l6_ab._resolve()
    rows = _drive(fn, steps=30)
    # Lever off: the original state is written back once, guards evaluated on every call.
    assert all((not r["on"]) and r["evals"] == 1 and r["equal"] for r in rows)
    assert fn._guard_check_enabled is True


def test_cache_loaded_artifact_is_never_switched_on(wrapped):
    """vLLM disables the guard check on an artifact it loads from its cache (evaluate_guards off);
    the lever keeps that state in every window, so a loaded artifact never evaluates guards and
    never raises. Uses a real saved + loaded artifact when torch can load one, else vLLM's state."""
    import torch._dynamo.aot_compile as aotc

    fresh = _compile(wrapped.W)
    la.reset()
    loaded = None
    loader = getattr(torch.compiler, "load_compiled_function", None)
    if loader is not None:
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "f1.aot")
        try:
            fresh.save_compiled_function(path)
            with open(path, "rb") as fh:
                loaded = loader(fh)
        except Exception as e:  # noqa: BLE001
            print(f"AOTSKIP_LOAD real load unavailable: {e!r}"[:300])
            loaded = None
    kind = "real" if loaded is not None else "fresh+disabled"
    art = loaded if loaded is not None else _compile(wrapped.W)
    art.disable_guard_check()  # what vLLM's _try_load_aot_compiled_fn leaves
    wrapped.mp.setattr(wrapped.decorators, "_try_load_aot_compiled_fn", lambda model, *a, **k: art)
    _arm(wrapped.mp)
    got = wrapped.decorators._try_load_aot_compiled_fn(NS(), "/cache")
    assert got is art and len(la._handles) == 1 and la._handles[0][1] is False
    rows = _drive(art, steps=120)
    assert all(r["evals"] == 0 and r["equal"] for r in rows), rows[:3]
    assert any(r["on"] for r in rows) and any(not r["on"] for r in rows)
    assert art._guard_check_enabled is False
    print(f"AOTSKIP_LOADED kind={kind} type={type(art).__name__} steps={len(rows)}")
    assert isinstance(art, aotc.AOTCompiledFunction)


@pytest.mark.parametrize("config", [True, False], ids=["evaluate_guards", "no_config"])
def test_fresh_artifact_with_evaluated_guards_is_never_kept(wrapped, config):
    """vLLM evaluates this fresh artifact's guards (evaluate_guards True, or no config, which
    X-H14-AOTSKIP also reads as True): the lever never keeps it, so in every window each call
    evaluates its guard and a wrong-shape call fails it."""
    _arm(wrapped.mp)
    fn = _compile(wrapped.W, evaluate=True, config=config)
    assert type(fn).__name__ == "AOTCompiledFunction"
    assert la._handles == [] and la._state["not_kept"] == 1 and fn._guard_check_enabled is True
    la.register(NS(_guard_check_enabled=True))  # one kept handle, so on_step runs every step
    l6_ab.begin_step()
    la.on_step()
    assert l6_ab._state["counts"].get("aotskip.not_kept") == 1, l6_ab._state["counts"]
    rows = _drive(fn, steps=60, bad_every=1)
    assert any(r["on"] for r in rows) and any(not r["on"] for r in rows)
    assert all(r["evals"] == 1 and r["equal"] and r["guard_error"] is True for r in rows), rows[:3]


def test_cache_loaded_artifact_with_the_check_on_is_never_kept(wrapped):
    """A cache-loaded artifact that vLLM left with its guard check on (it evaluates guards) is
    never kept: its guard is evaluated in every window."""
    art = _compile(wrapped.W)  # not armed yet: vLLM's own aot_compile, check on
    assert art._guard_check_enabled is True
    wrapped.mp.setattr(wrapped.decorators, "_try_load_aot_compiled_fn", lambda model, *a, **k: art)
    _arm(wrapped.mp)
    got = wrapped.decorators._try_load_aot_compiled_fn(NS(), "/cache")
    assert got is art and la._handles == [] and la._state["not_kept"] == 1
    la.register(NS(_guard_check_enabled=True))  # one kept handle, so on_step runs every step
    rows = _drive(art, steps=60)
    assert any(r["on"] for r in rows) and any(not r["on"] for r in rows)
    assert all(r["evals"] == 1 and r["equal"] for r in rows), rows[:3]
    assert art._guard_check_enabled is True


def test_check_mode_evaluates_guards_every_nth_lever_on_step(wrapped):
    _arm(wrapped.mp, "1", **{CHECK: "4"})
    fn = _compile(wrapped.W)
    # l6_ab clears its counts at every window boundary (10-step windows here), so count the
    # CHECK notes themselves.
    notes = []
    real_note = l6_ab.note
    wrapped.mp.setattr(l6_ab, "note", lambda lever, key, n=1: (notes.append((lever, key)), real_note(lever, key, n)))
    rows = _drive(fn, steps=40)
    evals = [r["evals"] for r in rows]
    assert evals.count(1) == 10 and evals.count(0) == 30, evals
    assert all(evals[i] == (1 if (i + 1) % 4 == 0 else 0) for i in range(40)), evals
    assert notes.count(("aotskip", "check")) == 10, notes


def test_check_steps_bound_the_check(wrapped):
    _arm(wrapped.mp, "1", **{CHECK: "1", "VLLM_TPU_L6_CHECK_STEPS": "5"})
    fn = _compile(wrapped.W)
    rows = _drive(fn, steps=20)
    assert [r["evals"] for r in rows] == [1] * 5 + [0] * 15, [r["evals"] for r in rows]


def test_writes_only_when_the_state_changes(wrapped):
    """40 steps of 10-step windows (off, on, off, on, then step 40 opens an off window): the
    artifacts are written 5 times (the first step and each window change), never in between."""
    _arm(wrapped.mp)
    fns = [_compile(wrapped.W) for _ in range(3)]
    calls = []
    real = la.apply
    wrapped.mp.setattr(la, "apply", lambda skip: calls.append(skip) or real(skip))
    seen = []
    for _ in range(4 * S):
        l6_ab.begin_step()
        la.on_step()
        seen.append(l6_ab.on("aotskip"))
        assert all(f._guard_check_enabled is (not seen[-1]) for f in fns)
    assert calls == [False, True, False, True, False], calls
    assert seen == [False] * 9 + [True] * 10 + [False] * 10 + [True] * 10 + [False], seen


def test_registered_mid_run_takes_the_current_state(wrapped):
    _arm(wrapped.mp, "1")
    first = _compile(wrapped.W)
    l6_ab.begin_step()
    la.on_step()
    assert first._guard_check_enabled is False
    late = _compile(wrapped.W)
    assert late._guard_check_enabled is False  # register() applied the current state


def test_not_installed_next_to_the_hashed_aotskip(wrapped):
    wrapped.mp.setenv("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "1")
    before = wrapped.W.aot_compile
    _arm(wrapped.mp, "1")
    assert wrapped.W.aot_compile is before


def test_flag_unset_installs_nothing(wrapped):
    before = wrapped.W.aot_compile
    load = wrapped.decorators._try_load_aot_compiled_fn
    la.install()
    assert wrapped.W.aot_compile is before and wrapped.decorators._try_load_aot_compiled_fn is load


_PROBE = r"""
import sys
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W
orig = W.aot_compile
from vllm_torchtpu import patch_registry as r
rows = [p for p in r.PATCHES if p.target == "vllm_torchtpu:_patch_vllm_l6_aotskip"]
print("L6AOT_ROWS", len(rows))
print("L6AOT_STAGES", ",".join(sorted(rows[0].stages)) if rows else "-")
for p in rows:
    r._invoke(p.target)
print("L6AOT_WRAPPED", W.aot_compile is not orig)
print("L6AOT_MOD", "vllm_torchtpu.compilation.l6_aotskip" in sys.modules)
"""


def _probe(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_TPU_L6_") and k != "VLLM_TPU_AOT_SKIP_GUARD_CHECK"}
    env.update(extra)
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return dict(ln.split(" ", 1) for ln in out.stdout.splitlines() if ln.startswith("L6AOT_"))


def test_registry_row_fresh_process_off_identity_and_on_install():
    """Fresh processes: with the flag unset the registry row imports nothing and leaves vLLM's
    aot_compile alone; with it set the wrapper is installed."""
    off = _probe({})
    assert off == {"L6AOT_ROWS": "1", "L6AOT_STAGES": "platform_activation,worker_init",
                   "L6AOT_WRAPPED": "False", "L6AOT_MOD": "False"}, off
    on = _probe({ENV: "ab:1", "VLLM_TPU_L6_AB_STEPS": "10", "VLLM_TPU_L6_AB_CONFIRM": "4"})
    assert on == {"L6AOT_ROWS": "1", "L6AOT_STAGES": "platform_activation,worker_init",
                  "L6AOT_WRAPPED": "True", "L6AOT_MOD": "True"}, on


def test_worker_call_site_is_guarded():
    import inspect

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    src = inspect.getsource(TPUModelRunner.execute_model)
    i = src.index("_l6_ab.begin_step(")
    j = src.index("self._update_states(scheduler_output)")
    seg = src[i:j]
    assert "if _l6_aot._handles:" in seg and "_l6_aot.on_step()" in seg, seg
    assert seg.index("if _l6_aot._handles:") < seg.index("_l6_aot.on_step()")


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["aotskip"] == ENV == la.ENV
    ignored = tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert ENV in ignored and CHECK in ignored
    assert hasattr(envs, ENV) and hasattr(envs, CHECK)


def test_import_resolves_to_the_tree_under_test():
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
