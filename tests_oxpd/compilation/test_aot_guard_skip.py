# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_AOT_SKIP_GUARD_CHECK (q38 L4 D98 port of MM's test,
mm/code/vllm-torchtpu-host-rb/tests/compilation/test_aot_guard_skip.py), plus
the API this tree's vLLM / torch must provide for it to engage."""

import os
from types import SimpleNamespace as NS


def test_fresh_aot_compile_disables_guard_check(monkeypatch):
    """A freshly AOT-compiled function skips the per-call guard check when
    guards are not evaluated, and keeps it when they are."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_guard_skip

    class Compiled:
        def __init__(self):
            self._guard_check_enabled = True

        @property
        def checking(self):
            return self._guard_check_enabled

        def disable_guard_check(self):
            self._guard_check_enabled = False

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "aot_compile", lambda self, *a, **k: Compiled()
    )
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", True, raising=False)
    monkeypatch.delenv("RANK", raising=False)
    lines = []
    monkeypatch.setattr(aot_guard_skip.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    aot_guard_skip.patch_aot_guard_skip()
    for evaluate, checking in ((False, False), (True, True)):
        owner = NS(
            compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=evaluate))
        )
        compiled = TorchCompileWithNoGuardsWrapper.aot_compile(owner, 1)
        assert compiled.checking is checking
    # One engagement line per compile, saying whether the skip applied, tagged
    # with the rank (-1 without torch.distributed and RANK) and the artifact state.
    eng = [ln for ln in lines if ln.startswith("AOT_GUARD_SKIP")]
    assert eng == [
        "AOT_GUARD_SKIP engaged: rank=-1 cls=SimpleNamespace guard_check_enabled=False",
        "AOT_GUARD_SKIP not engaged: rank=-1 cls=SimpleNamespace (evaluate_guards=True, "
        "disable_guard_check=True)",
    ], eng


def test_skip_that_does_not_take_reads_not_engaged(monkeypatch):
    """disable_guard_check that leaves the check on is reported, not counted."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_guard_skip

    class Stubborn:
        _guard_check_enabled = True

        def disable_guard_check(self):
            pass

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "aot_compile", lambda self, *a, **k: Stubborn()
    )
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", True, raising=False)
    monkeypatch.delenv("RANK", raising=False)
    lines = []
    monkeypatch.setattr(aot_guard_skip.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    aot_guard_skip.patch_aot_guard_skip()
    owner = NS(compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=False)))
    TorchCompileWithNoGuardsWrapper.aot_compile(owner, 1)
    eng = [ln for ln in lines if ln.startswith("AOT_GUARD_SKIP")]
    assert eng == [
        "AOT_GUARD_SKIP not engaged: rank=-1 cls=SimpleNamespace (disable_guard_check "
        "left guard_check_enabled=True)"
    ], eng


def test_cache_loaded_artifact_is_logged_with_its_guard_state(monkeypatch):
    """vLLM's own load path (decorators._try_load_aot_compiled_fn) is wrapped to
    log each loaded artifact's guard state; a miss (None) logs nothing."""
    from vllm.compilation import decorators

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_guard_skip

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    results = [NS(_guard_check_enabled=False), NS(_guard_check_enabled=True), None]
    monkeypatch.setattr(decorators, "_try_load_aot_compiled_fn", lambda m, p: results.pop(0))
    # patch_aot_guard_skip also wraps aot_compile: register vLLM's own for restore.
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "aot_compile", TorchCompileWithNoGuardsWrapper.aot_compile
    )
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", True, raising=False)
    monkeypatch.setenv("RANK", "5")
    lines = []
    monkeypatch.setattr(aot_guard_skip.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    aot_guard_skip.patch_aot_guard_skip()

    class Model:
        pass

    got = [decorators._try_load_aot_compiled_fn(Model(), "/x") for _ in range(3)]
    assert got[2] is None
    eng = [ln for ln in lines if ln.startswith("AOT_GUARD_SKIP")]
    assert eng == [
        "AOT_GUARD_SKIP cache-loaded: rank=5 cls=Model guard_check_enabled=False",
        "AOT_GUARD_SKIP cache-loaded: rank=5 cls=Model guard_check_enabled=True",
    ], eng


def test_flag_off_leaves_aot_compile_alone(monkeypatch):
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    import vllm_torchtpu
    from vllm_torchtpu import envs

    before = TorchCompileWithNoGuardsWrapper.aot_compile
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", False, raising=False)
    vllm_torchtpu._patch_vllm_aot_guard_skip()
    assert TorchCompileWithNoGuardsWrapper.aot_compile is before


def test_this_trees_vllm_and_torch_provide_the_hooks():
    """The patch only engages if vLLM's wrapper has ``aot_compile`` and torch's
    AOT-compiled function has a guard check that ``disable_guard_check`` turns
    off (vLLM 0.31.0 / torch 2.13 on the image)."""
    import torch._dynamo.aot_compile as aot
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    assert callable(getattr(TorchCompileWithNoGuardsWrapper, "aot_compile", None))
    fn = aot.AOTCompiledFunction
    assert fn._guard_check_enabled is True
    probe = object.__new__(fn)
    fn.disable_guard_check(probe)
    assert probe._guard_check_enabled is False


# ---- q38 L4 X-H14-AOTSKIP (D14 serve tree): registry wiring, fresh-process
# identity, and a real torch AOT artifact through vLLM's own aot_compile. ----

_PROBE = r"""
import sys
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W
orig = W.aot_compile
from vllm_torchtpu import patch_registry as r
rows = [p for p in r.PATCHES if p.target == "vllm_torchtpu:_patch_vllm_aot_guard_skip"]
print("AOTSKIP_ROWS", len(rows))
print("AOTSKIP_STAGES", ",".join(sorted(rows[0].stages)) if rows else "-")
for p in rows:
    r._invoke(p.target)
print("AOTSKIP_WRAPPED", W.aot_compile is not orig)
print("AOTSKIP_MARK", bool(getattr(W.aot_compile, "_tpu_guard_skip", False)))
print("AOTSKIP_MOD", "vllm_torchtpu.compilation.aot_guard_skip" in sys.modules)
"""


def _probe(extra_env):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_AOT_SKIP_GUARD_CHECK"}
    env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    got = {}
    for line in out.stdout.splitlines():
        if line.startswith("AOTSKIP_"):
            k, _, v = line.partition(" ")
            got[k] = v.strip()
    return got


def test_registry_row_and_fresh_process_flag_off_identity_and_on_engagement():
    """The registry installs the skip in worker processes (platform_activation +
    worker_init), and only when the flag is set: with the flag unset a fresh
    process keeps vLLM's own aot_compile and never imports the module."""
    off = _probe({})
    assert off == {
        "AOTSKIP_ROWS": "1",
        "AOTSKIP_STAGES": "platform_activation,worker_init",
        "AOTSKIP_WRAPPED": "False",
        "AOTSKIP_MARK": "False",
        "AOTSKIP_MOD": "False",
    }, off
    on = _probe({"VLLM_TPU_AOT_SKIP_GUARD_CHECK": "1"})
    assert on == {
        "AOTSKIP_ROWS": "1",
        "AOTSKIP_STAGES": "platform_activation,worker_init",
        "AOTSKIP_WRAPPED": "True",
        "AOTSKIP_MARK": "True",
        "AOTSKIP_MOD": "True",
    }, on
    print("AOTSKIP_FRESH off", off, "on", on)


SCALE = 2


def _scaled(x):
    return x * SCALE + 1


_BACKEND_SRC = r"""
def serializable_eager(gm, example_inputs):
    # torch's AOT compile needs a SerializableCallable (torch 2.13 rejects aot_eager's
    # plain function, M: l4host-smoke-v1 j-9afd37c9). Prefer torch's own graph-module
    # wrapper; else a minimal subclass that runs the graph eagerly.
    import torch._dynamo.aot_compile as aotc
    G = getattr(aotc, "GraphModuleSerializableCallable", None)
    if G is None:
        try:
            from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable as G
        except Exception:
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
"""
exec(_BACKEND_SRC)


def test_real_aot_artifact_through_vllms_aot_compile_skips_guards(monkeypatch):
    """vLLM's own TorchCompileWithNoGuardsWrapper.aot_compile on a real torch AOT
    artifact (a serializable eager backend): with evaluate_guards off the patched call
    returns an artifact whose per-call guard check is off (an input of another shape
    no longer fails the static-shape tensor guard); with evaluate_guards on the guard
    still fires. (Smoke j-0553a580, M: on torch 2.13 a rebound module global did not
    fail the guard, so the probe is the tensor shape guard.)"""
    import torch
    from types import SimpleNamespace as NS
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_guard_skip

    real = TorchCompileWithNoGuardsWrapper.aot_compile
    assert not getattr(real, "_tpu_guard_skip", False), "start from vLLM's own"
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", True, raising=False)
    lines = []
    monkeypatch.setattr(aot_guard_skip.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    # monkeypatch restores vLLM's attribute after the test.
    monkeypatch.setattr(TorchCompileWithNoGuardsWrapper, "aot_compile", real)
    aot_guard_skip.patch_aot_guard_skip()
    assert TorchCompileWithNoGuardsWrapper.aot_compile is not real

    x = torch.arange(4, dtype=torch.float32)
    results = {}
    for evaluate in (False, True):
        torch._dynamo.reset()
        monkeypatch.setitem(globals(), "SCALE", 2)
        with torch._dynamo.config.patch(enable_aot_compile=True):
            compiled = torch.compile(
                _scaled, fullgraph=True, dynamic=False, backend=serializable_eager  # noqa: F821
            )
        owner = NS(
            _compiled_callable=compiled,
            compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=evaluate)),
        )
        fn = TorchCompileWithNoGuardsWrapper.aot_compile(owner, x)
        assert type(fn).__name__ == "AOTCompiledFunction", type(fn)
        assert torch.equal(fn(x), x * 2 + 1)
        x5 = torch.arange(5, dtype=torch.float32)  # dynamic=False: a shape guard fails
        try:
            out = fn(x5)
            assert torch.equal(out, x5 * 2 + 1), out
            results[evaluate] = "no-guard-error"
        except Exception as e:  # torch raises RuntimeError("GuardManager check failed ...")
            assert "uard" in str(e), repr(e)
            results[evaluate] = "guard-error"
    print("AOTSKIP_REAL", results)
    assert results == {False: "no-guard-error", True: "guard-error"}, results
    eng = [ln.split(":")[0] for ln in lines if ln.startswith("AOT_GUARD_SKIP")]
    assert eng == ["AOT_GUARD_SKIP engaged", "AOT_GUARD_SKIP not engaged"], lines


# ---- Lead ruling 18:3xZ: hashed key + an engagement check that bites. ----

_SCENARIO = _BACKEND_SRC + r"""
import logging, os, sys
import torch
# Smoke j-0553a580 (M): vLLM's own handler printed none of these INFO lines in the CPU
# pod; attach an explicit stdout handler so the reader sees exactly what the skip logs.
_h = logging.StreamHandler(sys.stdout)
_h.setLevel(logging.INFO)
for _name in ("vllm", "vllm.vllm_torchtpu.compilation.aot_guard_skip"):
    logging.getLogger(_name).setLevel(logging.INFO)
_lg = logging.getLogger("vllm.vllm_torchtpu.compilation.aot_guard_skip")
_lg.addHandler(_h)
_lg.propagate = False  # exactly one copy of each line, whatever vLLM's own config does
from types import SimpleNamespace as NS
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W
from vllm_torchtpu import patch_registry as r
for p in r.PATCHES:
    if p.target == "vllm_torchtpu:_patch_vllm_aot_guard_skip":
        r._invoke(p.target)
SCALE = 2
def f(x):
    return x * SCALE + 1
x = torch.arange(4, dtype=torch.float32)
for name in ("WholeStep", "MtpForward0", "MtpLoop"):
    torch._dynamo.reset()
    with torch._dynamo.config.patch(enable_aot_compile=True):
        c = torch.compile(f, fullgraph=True, dynamic=False, backend=serializable_eager)
    owner = type(name, (), {})()
    owner._compiled_callable = c
    owner.compilation_config = NS(dynamic_shapes_config=NS(evaluate_guards=False))
    fn = W.aot_compile(owner, x)
    assert torch.equal(fn(x), x * 2 + 1)
print("SCENARIO_DONE", flush=True)
"""


def _scenario(extra_env, tmp_path, tag):
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "RANK")}
    env.update(extra_env)
    env["RANK"] = "0"
    out = subprocess.run(
        [sys.executable, "-c", _SCENARIO], env=env, capture_output=True, text=True,
        timeout=900,
    )
    log = tmp_path / f"scenario_{tag}.log"
    log.write_text(out.stdout + out.stderr)
    assert out.returncode == 0 and "SCENARIO_DONE" in out.stdout, (out.stdout + out.stderr)[-4000:]
    return log


def _engage(args):
    import subprocess
    import sys

    reader = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_aotskip",
                          "q38_aotskip_engage.py")
    out = subprocess.run([sys.executable, "-I", reader, *map(str, args)],
                         capture_output=True, text=True, timeout=120)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("AOTSKIP_ENG")]
    assert len(lines) == 1, out.stdout + out.stderr
    return out.returncode, lines[0]


def test_engagement_check_bites_on_real_aot_artifacts(tmp_path):
    """Three real AOT artifacts through vLLM's aot_compile in a fresh process:
    with the flag the reader says ENGAGED (3 per rank); without it the same
    reader in engaged mode says NOT_ENGAGED, and in off mode OFF."""
    on = _scenario({"VLLM_TPU_AOT_SKIP_GUARD_CHECK": "1"}, tmp_path, "on")
    off = _scenario({}, tmp_path, "off")
    rc, line = _engage([on, "--ranks", 1, "--min-per-rank", 3])
    print("AOTSKIP_ENG_ON", line)
    assert rc == 0 and " verdict=ENGAGED " in line and "classes=MtpForward0:1,MtpLoop:1,WholeStep:1" in line, line
    rc, line = _engage([off, "--ranks", 1, "--min-per-rank", 3])
    print("AOTSKIP_ENG_OFF_AS_ON", line)
    assert rc == 1 and " verdict=NOT_ENGAGED " in line, line
    rc, line = _engage([off, "--expect-off"])
    print("AOTSKIP_ENG_OFF", line)
    assert rc == 0 and " verdict=OFF " in line, line


def test_engagement_reader_selftest():
    rc, out = None, None
    import subprocess
    import sys

    reader = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_aotskip",
                          "q38_aotskip_engage.py")
    p = subprocess.run([sys.executable, "-I", reader, "--selftest"], capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0 and "AOTSKIP_ENG_SELFTEST PASS (11/11 cases)" in p.stdout, p.stdout + p.stderr
