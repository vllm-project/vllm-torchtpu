# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D60 (lever 5c): the decode adapter's compact_pipe dispatch, op key and arm line.

``FUSED_MOE_DECODE_COMPACT_PIPE=N`` (default 0 = off) makes ``layers/adapter/fused_moe_decode._layer`` pass
``compact_rows=FUSED_MOE_DECODE_COMPACT_PIPE_ROWS, compact_pipe=True`` to the kernel for folded whole-expert buckets of
at least N tokens, ahead of lever 5b's ``FUSED_MOE_DECODE_COMPACT_ROWS`` there; smaller buckets keep 5b / dense. With
the flag off, ``_layer`` passes exactly today's kwargs and ``_build_op`` builds today's key, partial and op name.
The kernel's bitwise identity is test_compact_pipe_kernel.py. Same harness as DEC-MOEKV's
tests/q38_moekv/test_compact_rows_wiring.py (credit): the adapter functions' exact source, read with ``ast``,
compiled against recording stubs (CPU venvs lack vLLM's compilation stack).
"""

import ast
import functools
import pathlib
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "src/vllm_torchtpu/layers/adapter/fused_moe_decode.py"
ENVS = ROOT / "src/vllm_torchtpu/envs.py"
FUNCS = ("_layer", "_build_op", "_compact_policy", "_compact_pipe_policy")


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


class _X:
    def __init__(self, tokens=1, hidden=2560):
        self.shape = (tokens, hidden)
        self.dtype = "float8_e4m3fn"

    def reshape(self, *shape):
        return self


def _closure_op_name():
    import importlib.util

    path = ADAPTER.parents[2] / "kernels" / "closure_op_name.py"
    spec = importlib.util.spec_from_file_location("q38_closure_op_name_d60", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.closure_op_name


def _namespace(*, ib=None, pipe=0, pipe_rows=32, compact_rows=0, with_pipe_attrs=True):
    calls = {"kernel": [], "gmm": []}
    env = dict(FUSED_MOE_DECODE_KERNEL_FOLD="xla", FUSED_MOE_DECODE_COMPACT_ROWS=compact_rows,
               FUSED_MOE_DECODE_COMPACT_MIN_TOKENS=256, FUSED_MOE_DECODE_KERNEL_SCOPES=0)
    if with_pipe_attrs:
        env.update(FUSED_MOE_DECODE_COMPACT_PIPE=pipe, FUSED_MOE_DECODE_COMPACT_PIPE_ROWS=pipe_rows)
    ns = {
        "logger": _Log(),
        "plan_weight_ring": lambda *a: (ib, 6),
        "_kernel_body": lambda *a, **k: calls["kernel"].append(k) or "kernel",
        "_gmm_layer": lambda *a, **k: calls["gmm"].append(k) or "gmm",
        "functools": functools,
        "PartitionSpec": lambda *a: ("spec", a),
        "EP_AXIS_NAME": "ep",
        "_OPS": {},
        "closure_op_name": _closure_op_name(),
        "jax": types.SimpleNamespace(Array=object),
        "jnp": types.SimpleNamespace(int32="int32"),
        "torch": types.SimpleNamespace(empty_like=lambda x: x),
        "envs": types.SimpleNamespace(**env),
    }

    class _Op:
        def __init__(self, name, fn, **kwargs):
            self.name, self.fn = name, fn

        def register_fake(self, fake):
            self.fake = fake

    ns["jax_op"] = _Op
    tree = ast.parse(ADAPTER.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert sorted(f.name for f in funcs) == sorted(FUNCS)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(ADAPTER), "exec"), ns)
    return ns, calls


def _call_layer(ns, tokens, *, fold="xla", compact_rows=0, compact_min_tokens=256, max_tokens=256, pipe=None):
    args = [np.zeros((1, 2), np.int32), _X(tokens)] + [_X() for _ in range(11)]
    kw = dict(n_ranks=8, topk=10, expert_parallel=True, max_tokens=max_tokens, fold=fold, compact_rows=compact_rows,
              compact_min_tokens=compact_min_tokens, scopes=0)
    if pipe is not None:
        kw.update(compact_pipe_min=pipe[0], compact_pipe_rows=pipe[1])
    return ns["_layer"](*args, **kw)


@pytest.mark.parametrize("tokens", [16, 64, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("compact_rows", [0, 32])
def test_flag_off_passes_todays_kwargs(tokens, compact_rows):
    ns, calls = _namespace()
    _call_layer(ns, tokens, compact_rows=compact_rows, pipe=(0, 32))
    kw = calls["kernel"][-1]
    assert "compact_pipe" not in kw
    assert kw["compact_rows"] == (compact_rows if compact_rows and tokens >= 256 else None)
    assert "pipelined" not in ns["logger"].lines[-1]


@pytest.mark.parametrize("tokens,want_pipe", [(256, True), (224, True), (208, True), (192, True), (191, False),
                                              (160, False), (64, False)])
def test_pipe_applies_from_threshold(tokens, want_pipe):
    ns, calls = _namespace()
    _call_layer(ns, tokens, compact_rows=32, pipe=(192, 32))
    kw = calls["kernel"][-1]
    assert kw.get("compact_pipe", False) is want_pipe
    want_rows = 32 if want_pipe else (32 if tokens >= 256 else None)
    assert kw["compact_rows"] == want_rows
    line = ns["logger"].lines[-1]
    assert ("compact_rows 32 pipelined (q38 D60 5c)" in line) == want_pipe
    print(f"Q38_D60_WIRING tokens {tokens} pipe {kw.get('compact_pipe', False)} rows {kw['compact_rows']}: {line}")


def test_pipe_rows_and_precedence_over_5b():
    ns, calls = _namespace()
    _call_layer(ns, 256, compact_rows=16, pipe=(224, 48))
    assert calls["kernel"][-1]["compact_rows"] == 48 and calls["kernel"][-1]["compact_pipe"] is True
    _call_layer(ns, 160, compact_rows=16, compact_min_tokens=128, pipe=(224, 48))
    assert calls["kernel"][-1]["compact_rows"] == 16 and "compact_pipe" not in calls["kernel"][-1]


def test_unfolded_or_blocked_not_applied():
    ns, calls = _namespace()
    _call_layer(ns, 256, fold="off", pipe=(192, 32))
    assert "compact_pipe" not in calls["kernel"][-1]
    assert any("compact_pipe rows=32 not applied" in s for s in ns["logger"].lines)
    ns, calls = _namespace(ib=128)
    _call_layer(ns, 256, pipe=(192, 32))
    assert "compact_pipe" not in calls["kernel"][-1] and calls["kernel"][-1]["compact_rows"] is None
    assert any("not applied" in s for s in ns["logger"].lines)


def test_above_max_tokens_takes_gmm():
    ns, calls = _namespace()
    _call_layer(ns, 512, pipe=(192, 32), max_tokens=256)
    assert calls["gmm"] and not calls["kernel"]


def _partial(op):
    cells = op.fn.__closure__ or ()
    parts = [c.cell_contents for c in cells if isinstance(c.cell_contents, functools.partial)]
    assert parts
    return parts[0]


def test_build_op_flag_off_is_todays_key_partial_and_name():
    """Off: the key, the partial's keywords and the op name equal a build from stubs WITHOUT the new attributes
    (today's adapter reads only the older flags), so flags-off programs and compile keys are unchanged."""
    args = (object(), 8, 10, True, 256)
    ns0, _ = _namespace(with_pipe_attrs=False, compact_rows=32)
    ns1, _ = _namespace(pipe=0, compact_rows=32)
    op0, op1 = ns0["_build_op"](*args), ns1["_build_op"](*args)
    assert list(ns0["_OPS"]) == list(ns1["_OPS"])
    assert op0.name == op1.name
    assert _partial(op0).keywords == _partial(op1).keywords
    assert "compact_pipe_min" not in _partial(op1).keywords
    k = list(ns1["_OPS"])[0]
    assert len(k) == 11 and k[-3:] == (32, 256, 0)


def test_build_op_flag_on_extends_key_partial_and_name():
    args = (object(), 8, 10, True, 256)
    ns, _ = _namespace(compact_rows=32)
    off = ns["_build_op"](*args)
    ns["envs"].FUSED_MOE_DECODE_COMPACT_PIPE = 192
    on = ns["_build_op"](*args)
    assert on is not off and on.name != off.name, "compact_pipe must be part of the op key and name"
    ns["envs"].FUSED_MOE_DECODE_COMPACT_PIPE_ROWS = 48
    on48 = ns["_build_op"](*args)
    assert on48 is not on and on48.name != on.name, "pipe rows must be part of the op key and name"
    keys = list(ns["_OPS"])
    assert keys[1][-1] == ("compact_pipe", 192, 32) and keys[2][-1] == ("compact_pipe", 192, 48)
    p = _partial(on).keywords
    assert p["compact_pipe_min"] == 192 and p["compact_pipe_rows"] == 32 and p["compact_rows"] == 32


@pytest.mark.parametrize("bad", [0, 8, 24])
def test_build_op_rejects_bad_pipe_rows(bad):
    ns, _ = _namespace(pipe=192, pipe_rows=bad)
    with pytest.raises(ValueError, match="FUSED_MOE_DECODE_COMPACT_PIPE_ROWS"):
        ns["_build_op"](object(), 8, 10, True, 256)


def test_compact_pipe_policy_arm_line():
    ns, _ = _namespace()
    f = ns["_compact_pipe_policy"]
    assert f("xla", 0, 32, 256) is None
    assert "NOT applied" in f("off", 192, 32, 256) and "FOLD=off" in f("off", 192, 32, 256)
    assert "NOT applied" in f("xla", 512, 32, 256)
    on = f("xla", 192, 32, 256)
    assert on.startswith("Fused MoE decode compact_pipe=192 ARMED (q38 L3 D60 5c)") and "192..256" in on
    assert "-cr_32p" in on


def test_envs_register_both_flags_with_defaults():
    """envs.py declares both flags (typed default and parser); the compile-key side is
    tests/compilation/test_tpu_compile_key_registration.py (both in HASHED_FLAGS)."""
    text = ENVS.read_text()
    assert "FUSED_MOE_DECODE_COMPACT_PIPE: int = 0" in text
    assert "FUSED_MOE_DECODE_COMPACT_PIPE_ROWS: int = 32" in text
    assert '"FUSED_MOE_DECODE_COMPACT_PIPE": env_nonnegative_int(\n        "FUSED_MOE_DECODE_COMPACT_PIPE", 0\n    )' in text
    assert ('"FUSED_MOE_DECODE_COMPACT_PIPE_ROWS": env_nonnegative_int(\n        '
            '"FUSED_MOE_DECODE_COMPACT_PIPE_ROWS", 32\n    )') in text
