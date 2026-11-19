# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-MOEKV lever 5b: the decode adapter's compact_rows dispatch.

``FUSED_MOE_DECODE_COMPACT_ROWS`` (default 0) makes ``layers/adapter/fused_moe_decode._layer`` pass
``compact_rows=R`` to the fused MoE decode kernel for folded, whole-expert token buckets of at least
``FUSED_MOE_DECODE_COMPACT_MIN_TOKENS`` (default 256). The kernel's bitwise identity to the dense
loop is DEC-FB-2's ``tests/decfb/test_fused_moe_decode_compact.py``. This file checks only the
dispatch, the op cache key and the argument check.

The adapter package imports vLLM's compilation stack, which CPU test venvs do not have. So this test
compiles the two functions' exact source (``_layer``, ``_build_op``, read with ``ast`` from the
adapter file) against recording stubs of their collaborators.
"""

import ast
import functools
import pathlib
import types

import numpy as np
import pytest

ADAPTER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/layers/adapter/fused_moe_decode.py"
)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


class _X:
    """Shape-carrying operand stand-in (the stubbed kernel never reads values)."""

    def __init__(self, tokens=1, hidden=2560):
        self.shape = (tokens, hidden)
        self.dtype = "float8_e4m3fn"

    def reshape(self, *shape):
        return self


def _closure_op_name():
    import importlib.util

    path = ADAPTER.parents[2] / "kernels" / "closure_op_name.py"
    spec = importlib.util.spec_from_file_location("q38_closure_op_name", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.closure_op_name


def _namespace(*, ib=None):
    calls = {"kernel": [], "gmm": []}
    ns = {
        "logger": _Log(),
        "plan_weight_ring": lambda *a: (ib, 6),
        "_kernel_body": lambda *a, **k: calls["kernel"].append(k) or "kernel",
        "_gmm_layer": lambda *a, **k: calls["gmm"].append(k) or "gmm",
        "functools": functools,
        "PartitionSpec": lambda *a: ("spec", a),
        "EP_AXIS_NAME": "ep",
        "_OPS": {},
        # q38 DEC-O4-4: _build_op names its op with closure_op_name (a pure stdlib module, loaded by path).
        "closure_op_name": _closure_op_name(),
        "jax": types.SimpleNamespace(Array=object),
        "jnp": types.SimpleNamespace(int32="int32"),
        "torch": types.SimpleNamespace(empty_like=lambda x: x),
        "envs": types.SimpleNamespace(
            FUSED_MOE_DECODE_KERNEL_FOLD="xla",
            FUSED_MOE_DECODE_COMPACT_ROWS=0,
            FUSED_MOE_DECODE_COMPACT_MIN_TOKENS=256,
            FUSED_MOE_DECODE_KERNEL_SCOPES=0,
        ),
    }

    class _Op:
        def __init__(self, name, fn, **kwargs):
            self.name, self.fn = name, fn

        def register_fake(self, fake):
            self.fake = fake

    ns["jax_op"] = _Op
    tree = ast.parse(ADAPTER.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_layer", "_build_op", "_compact_policy")]
    assert sorted(f.name for f in funcs) == ["_build_op", "_compact_policy", "_layer"]
    module = ast.Module(body=funcs, type_ignores=[])
    exec(compile(module, str(ADAPTER), "exec"), ns)
    return ns, calls


def _call_layer(ns, tokens, *, fold="xla", compact_rows=0, compact_min_tokens=256, max_tokens=256, scopes=0):
    args = [np.zeros((1, 2), np.int32), _X(tokens)] + [_X() for _ in range(11)]
    return ns["_layer"](
        *args, n_ranks=8, topk=10, expert_parallel=True, max_tokens=max_tokens,
        fold=fold, compact_rows=compact_rows, compact_min_tokens=compact_min_tokens, scopes=scopes,
    )


def test_default_off_is_dense():
    ns, calls = _namespace()
    _call_layer(ns, 256)
    assert calls["kernel"][-1]["compact_rows"] is None
    assert "dense experts" in ns["logger"].lines[-1]


@pytest.mark.parametrize("tokens,want", [(256, 16), (255, None), (128, None), (64, None)])
def test_on_applies_from_min_tokens(tokens, want):
    ns, calls = _namespace()
    _call_layer(ns, tokens, compact_rows=16, max_tokens=512)
    assert calls["kernel"][-1]["compact_rows"] == want
    assert ("compact_rows 16" in ns["logger"].lines[-1]) == (want is not None)


def test_unfolded_call_stays_dense():
    ns, calls = _namespace()
    _call_layer(ns, 256, fold="off", compact_rows=16)
    assert "compact_rows" not in calls["kernel"][-1]
    assert "not applied" in ns["logger"].lines[0]
    assert "dense experts" in ns["logger"].lines[-1]


def test_blocked_weight_ring_stays_dense():
    ns, calls = _namespace(ib=4)
    _call_layer(ns, 256, compact_rows=32)
    assert calls["kernel"][-1]["compact_rows"] is None
    assert "not applied" in ns["logger"].lines[0]


def test_above_max_tokens_takes_gmm():
    ns, calls = _namespace()
    _call_layer(ns, 512, compact_rows=16, max_tokens=256)
    assert calls["gmm"] and not calls["kernel"]


def test_build_op_key_and_partial_carry_compact():
    ns, _ = _namespace()
    envs = ns["envs"]
    args = (object(), 8, 10, True, 256)
    dense = ns["_build_op"](*args)
    envs.FUSED_MOE_DECODE_COMPACT_ROWS = 16
    compact = ns["_build_op"](*args)
    assert compact is not dense, "compact_rows must be part of the op cache key"
    envs.FUSED_MOE_DECODE_COMPACT_MIN_TOKENS = 192
    assert ns["_build_op"](*args) is not compact, "min tokens must be part of the op cache key"
    keys = list(ns["_OPS"])
    assert [k[-3:] for k in keys] == [(0, 256, 0), (16, 256, 0), (16, 192, 0)]
    # The op's per-rank body is _layer with the env values bound (a partial in fn's closure).
    op = ns["_OPS"][keys[1]]
    cells = op.fn.__closure__ or ()
    partials = [c.cell_contents for c in cells if isinstance(c.cell_contents, functools.partial)]
    assert partials and partials[0].func is ns["_layer"]
    assert partials and partials[0].keywords["compact_rows"] == 16
    assert partials[0].keywords["compact_min_tokens"] == 256


@pytest.mark.parametrize("bad", [8, 24, 1])
def test_build_op_rejects_non_multiple_of_16(bad):
    ns, _ = _namespace()
    ns["envs"].FUSED_MOE_DECODE_COMPACT_ROWS = bad
    with pytest.raises(ValueError, match="multiple of 16"):
        ns["_build_op"](object(), 8, 10, True, 256)


def test_compact_policy_arm_line():
    """The arm-time engagement line: None when off, NOT applied when unfolded or min > max, ARMED otherwise."""
    ns, _ = _namespace()
    f = ns["_compact_policy"]
    assert f("xla", 0, 256, 256) is None
    assert "NOT applied" in f("off", 32, 256, 256) and "FOLD=off" in f("off", 32, 256, 256)
    assert "NOT applied" in f("xla", 32, 512, 256)
    on = f("xla", 32, 256, 256)
    assert on.startswith("Fused MoE decode compact_rows=32 ARMED") and "256..256" in on
    assert "160..256" in f("kernel", 16, 160, 256)


@pytest.mark.parametrize("fold", ["xla", "off"])
def test_scopes_reach_the_kernel(fold):
    """IQD #10: FUSED_MOE_DECODE_KERNEL_SCOPES reaches both kernel calls (folded and unfolded)."""
    ns, calls = _namespace()
    _call_layer(ns, 128, fold=fold, scopes=1)
    assert calls["kernel"][-1]["scopes"] == 1
    ns, calls = _namespace()
    _call_layer(ns, 128, fold=fold)
    assert calls["kernel"][-1]["scopes"] == 0


def test_build_op_key_and_partial_carry_scopes():
    ns, _ = _namespace()
    args = (object(), 8, 10, True, 256)
    base = ns["_build_op"](*args)
    ns["envs"].FUSED_MOE_DECODE_KERNEL_SCOPES = 1
    traced = ns["_build_op"](*args)
    assert traced is not base, "scopes must be part of the op cache key"
    cells = traced.fn.__closure__ or ()
    partials = [c.cell_contents for c in cells if isinstance(c.cell_contents, functools.partial)]
    assert partials and partials[0].keywords["scopes"] == 1
