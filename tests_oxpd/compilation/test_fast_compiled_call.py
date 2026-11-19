# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D98: the fast compiled call (compilation/fast_compiled_call.py) on CPU.

The "tpu" backend cannot run on CPU, so these tests compile with counting
stand-ins built on ``aot_eager``: the same Dynamo cache entries, guard
managers, transformed code objects and AOTAutograd runtime wrapper as the TPU
helpers. Each stand-in counts its compiles and its compiled-graph runs, so a
test can tell a direct dispatch (graph run, no compile, no Dynamo frame) from
a normal call and from eager Python.

Run with ``--noconftest`` (tests/conftest.py imports vLLM). The tests marked
``needs_pkg`` import the real runner / vLLM; tests/q38_d98/run_cpu.sh requires
0 skipped on the CDK CPU pod.
"""

import importlib.util
import os
import subprocess
import sys
import types

import pytest
import torch
from torch._dynamo.backends.debugging import aot_eager

from vllm_torchtpu.compilation import fast_compiled_call as fcc

needs_pkg = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="needs vLLM / the real package"
)

COMPILES: dict[str, int] = {}
RUNS: dict[str, int] = {}
FORBID_COMPILE = {"on": False}


def _counting_backend(name):
    def backend(gm, example_inputs):
        if FORBID_COMPILE["on"]:
            raise AssertionError(f"backend {name} compiled while compiles are forbidden")
        COMPILES[name] = COMPILES.get(name, 0) + 1
        inner = aot_eager(gm, example_inputs)

        def run(*args):
            RUNS[name] = RUNS.get(name, 0) + 1
            return inner(*args)

        return run

    backend.__name__ = f"d98_{name}"
    return backend


BK_FAST = _counting_backend("fast")
BK_PLAIN = _counting_backend("plain")
BK_OTHER = _counting_backend("other")


def _n(d, k):
    return d.get(k, 0)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(fcc, "ELIGIBLE_BACKENDS", ("tpu", BK_FAST, BK_OTHER))
    monkeypatch.setattr(fcc, "MODULE_PREFIXES", ("vllm_torchtpu", __name__))
    monkeypatch.setattr(fcc, "_MEMO_MODE", "")
    monkeypatch.setattr(fcc, "_WRAPPED", [])
    fcc._reset_stats_for_tests()
    FORBID_COMPILE["on"] = False
    torch._dynamo.reset()
    with torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=4096):
        yield
    FORBID_COMPILE["on"] = False
    torch._dynamo.reset()


def clone(fn):
    """Same body, globals and closure; its own code object (own Dynamo cache)."""
    g = types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    g.__kwdefaults__ = fn.__kwdefaults__
    g.__module__ = fn.__module__
    g.__qualname__ = fn.__qualname__
    return g


def hook(drop_global_state=False):
    return fcc.make_hook(torch.compile, fast=True, drop_global_state=drop_global_state)


def fast_of(fn, backend=BK_FAST, drop_global_state=False, same_code=False):
    out = hook(drop_global_state)(
        fn if same_code else clone(fn), backend=backend, fullgraph=True, dynamic=False
    )
    assert hasattr(out, "_d98_fast_state"), "eligible helper was not wrapped"
    return out


def plain_of(fn, backend=BK_PLAIN):
    return torch.compile(clone(fn), backend=backend, fullgraph=True, dynamic=False)


def st(w):
    return w._d98_fast_state


def bits(x):
    """Bitwise identity: dtype, shape, stride and raw bytes."""
    if isinstance(x, torch.Tensor):
        c = x.detach().contiguous()
        raw = c.view(torch.uint8) if c.element_size() == 1 else c.view(-1).view(torch.uint8)
        return ("T", x.dtype, tuple(x.shape), tuple(x.stride()), raw.tolist())
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(bits(v) for v in x))
    return ("V", x)


def clone_args(args):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(a.clone())
        elif isinstance(a, list):
            out.append([t.clone() if isinstance(t, torch.Tensor) else t for t in a])
        else:
            out.append(a)
    return out


# ---- helpers under test (module level, so __module__ is this test module) ----
GFLAG = 1


def h_tensors(x, y, k: int, scale: float, flag: bool, maybe=None):
    z = x * scale + y[:, :k].sum(-1, keepdim=True)
    if flag:
        z = z.relu()
    if maybe is not None:
        z = z + maybe
    return z, (z > 0).sum(0)


def h_mutate(buf, idx, val):
    buf.index_put_((idx,), val)
    return buf.sum(0)


def h_list(xs, n):
    return torch.stack([a * n for a in xs]).sum(0)


def h_default(x, n: int = 3):
    return x + n


def h_varargs(*args):
    """Shaped like _postfwd_subst_compiled(*args)."""
    a, b, c = args
    return a + b * c, c.sum()


def h_kwtuple(x, pair, n: int = 2, *, scale: float = 1.0):
    """Shaped like fused_step_inputs: called with keywords and tuple args."""
    return x * pair[0] + pair[1] * n * scale


def h_global(x):
    return x + GFLAG


def h_alias(x, y):
    x.add_(1)
    return x * y


def h_small(x):
    return x * 2


def make_closure(c):
    def h_closure(x):
        return x + c

    return h_closure


class Owner:
    def __init__(self, k):
        self.k = k

    def h_method(self, x):
        return x * self.k


INNER = {}


def outer_fn(x, y):
    return INNER["fn"](x, y, 2, 1.0, True)[0] * 2


# --------------------------------------------------------------------------------


def _run_case(fn, calls, memo, monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", memo)
    fast, plain = fast_of(fn), plain_of(fn)
    c0f, c0p, r0 = _n(COMPILES, "fast"), _n(COMPILES, "plain"), _n(RUNS, "fast")
    for args, kwargs in calls:
        a_fast, a_plain, a_eager = clone_args(args), clone_args(args), clone_args(args)
        out_f = fast(*a_fast, **kwargs)
        out_p = plain(*a_plain, **kwargs)
        out_e = fn(*a_eager, **kwargs)
        assert bits(out_f) == bits(out_p), fn.__name__
        assert bits(a_fast) == bits(a_plain), f"{fn.__name__}: mutated inputs differ"
        torch.testing.assert_close(out_f, out_e, rtol=0, atol=0)
    s = st(fast)
    compiles = _n(COMPILES, "fast") - c0f
    # The fast path compiles exactly what the plain compiled call compiles.
    assert compiles == _n(COMPILES, "plain") - c0p
    # Every call ran its compiled graph exactly once (direct calls included).
    assert _n(RUNS, "fast") - r0 == len(calls)
    # Every call that did not compile went through the direct dispatch.
    assert s["slow"] == compiles and s["fast"] == len(calls) - compiles, s
    return fast, s


def _calls_tensors():
    base = torch.randn(40, 6)
    y = torch.randn(8, 6)
    out = []
    for i in range(10):
        x = base[(i % 3) : (i % 3) + 8, :4] if i % 2 else torch.randn(8, 4)
        k = 3 if i < 6 else 2
        maybe = None if i % 4 else torch.randn(8, 4)
        out.append(((x, y, k, 0.5, bool(i % 2), maybe), {}))
    out.append(((torch.randn(8, 4), y, 3, 0.5, True), {"maybe": None}))  # kwargs
    out.append(((torch.randn(8, 4), y, 3, 0.5, True, None), {}))
    return out


def _calls_mutate():
    out = []
    for i in range(8):
        buf = torch.zeros(16, dtype=torch.int32)
        idx = torch.tensor([1, 5, 9][: 2 + (i % 2)])
        val = torch.tensor([7, 8, 9][: 2 + (i % 2)], dtype=torch.int32)
        out.append(((buf, idx, val), {}))
    return out


@pytest.mark.parametrize("memo", ["", "1", "check"])
def test_bitwise_vs_plain_compiled_and_engaged(memo, monkeypatch):
    _run_case(h_tensors, _calls_tensors(), memo, monkeypatch)
    _run_case(h_mutate, _calls_mutate(), memo, monkeypatch)
    lst = [(([torch.randn(5), torch.randn(5)], 2), {}) for _ in range(4)]
    lst += [(([torch.randn(5), torch.randn(5), torch.randn(5)], 3), {}) for _ in range(3)]
    _run_case(h_list, lst, memo, monkeypatch)
    _run_case(h_default, [((torch.randn(4),), {}) for _ in range(3)]
              + [((torch.randn(4), 5), {}) for _ in range(3)], memo, monkeypatch)
    _run_case(make_closure(torch.ones(4)), [((torch.randn(4),), {}) for _ in range(4)],
              memo, monkeypatch)
    _run_case(Owner.h_method, [((Owner(2), torch.randn(4)), {}) for _ in range(1)]
              + [((o, torch.randn(4)), {}) for o in [Owner(2)] * 3], memo, monkeypatch)
    _run_case(h_varargs, [((torch.randn(4), torch.randn(4), torch.randn(4)), {})
                          for _ in range(4)], memo, monkeypatch)
    kw = []
    for i in range(6):
        pair = (torch.randn(4), torch.randn(4))
        kw.append(((torch.randn(4),), {"pair": pair, "n": 2 + (i % 2), "scale": 0.5}))
    kw.append(((torch.randn(4), (torch.randn(4), torch.randn(4))), {}))
    kw.append(((torch.randn(4), (torch.randn(4), torch.randn(4)), 3), {"scale": 0.5}))
    h0 = fcc.STATS["memo_hit"]
    _run_case(h_kwtuple, kw, memo, monkeypatch)
    if memo:
        assert fcc.STATS["memo_hit"] > h0  # keyword calls are memoized too
        assert fcc.STATS["memo_store"] > 0
        assert fcc.STATS["memo_hit"] > 0
    if memo == "check":
        assert fcc.STATS["memo_check_ok"] == fcc.STATS["memo_hit"]
        assert fcc.STATS["memo_mismatch"] == 0


def test_method_bound_through_the_class():
    w = fast_of(Owner.h_method)

    class OwnerFast(Owner):
        h_method = w

    o = OwnerFast(3)
    x = torch.randn(4)
    for _ in range(3):
        assert torch.equal(o.h_method(x), x * 3)
    assert st(w)["fast"] == 2 and st(w)["slow"] == 1


def test_fast_path_never_compiles(monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    w = fast_of(h_tensors)
    y = torch.randn(8, 6)
    shapes = [torch.randn(8, 4), torch.randn(16, 4)]
    for x in shapes:
        w(x, torch.randn(x.shape[0], 6), 3, 0.5, True)
    c0 = _n(COMPILES, "fast")
    FORBID_COMPILE["on"] = True
    for i in range(200):
        x = torch.randn(8 if i % 2 else 16, 4)
        w(x, torch.randn(x.shape[0], 6), 3, 0.5, True)
    FORBID_COMPILE["on"] = False
    assert _n(COMPILES, "fast") == c0
    assert st(w)["fast"] == 200
    del y


def test_recompile_drops_the_memo(monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    w = fast_of(h_small)
    a, b = torch.randn(4), torch.randn(6)
    w(a)  # compile
    w(a)  # guard path, memo store
    assert len(st(w)["memo"]) == 1
    h0 = fcc.STATS["memo_hit"]
    w(a)
    assert fcc.STATS["memo_hit"] == h0 + 1
    c0 = _n(COMPILES, "fast")
    w(b)  # new shape: normal call compiles; the refresh drops the memo
    assert _n(COMPILES, "fast") == c0 + 1
    assert st(w)["memo"] == {}
    h1 = fcc.STATS["memo_hit"]
    w(a)  # guard path again (re-store), no memo hit
    assert fcc.STATS["memo_hit"] == h1
    w(a)
    assert fcc.STATS["memo_hit"] == h1 + 1


def test_eviction_and_reset_invalidate(monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    f = clone(h_small)
    w = fast_of(f, same_code=True)
    a = torch.randn(4)
    for _ in range(3):
        w(a)
    assert fcc.STATS["memo_hit"] == 1
    # Eviction / reset of the code's entries: the next call must be a normal
    # call that compiles again, never a stale direct dispatch.
    torch._C._dynamo.eval_frame.reset_code(f.__code__)
    c0, s0, h0 = _n(COMPILES, "fast"), st(w)["slow"], fcc.STATS["memo_hit"]
    assert torch.equal(w(a), a * 2)
    assert _n(COMPILES, "fast") == c0 + 1 and st(w)["slow"] == s0 + 1
    assert fcc.STATS["memo_hit"] == h0
    w(a)
    w(a)
    assert fcc.STATS["memo_hit"] == h0 + 1
    # Same entry count, different entries: reset, then another compile of the
    # same code object lands one entry (another backend). The count is back to
    # 1, but the wrapper must not run its stale entry.
    torch._C._dynamo.eval_frame.reset_code(f.__code__)
    other = torch.compile(f, backend=BK_OTHER, fullgraph=True, dynamic=False)
    other(a)
    assert torch._C._dynamo.eval_frame._get_total_cache_entry_count(f.__code__) == 1
    c1, s1, h1, f1 = _n(COMPILES, "fast"), st(w)["slow"], fcc.STATS["memo_hit"], st(w)["fast"]
    assert torch.equal(w(a), a * 2)
    assert st(w)["slow"] == s1 + 1 and st(w)["fast"] == f1
    assert fcc.STATS["memo_hit"] == h1
    assert _n(COMPILES, "fast") == c1 + 1


@pytest.mark.parametrize("memo", ["1", "check"])
def test_storage_offset_change_misses_the_memo(memo, monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", memo)
    w, p = fast_of(h_small), plain_of(h_small)
    base = torch.randn(20, 4)
    a, b = base[0:8], base[1:9]
    assert a.shape == b.shape and a.stride() == b.stride()
    assert a.storage_offset() != b.storage_offset()
    assert fcc.memo_key((a,)) != fcc.memo_key((b,))
    w(a)
    w(a)
    w(a)
    h0, s0, f0 = fcc.STATS["memo_hit"], fcc.STATS["memo_store"], st(w)["fast"]
    out = w(b)  # new storage_offset: no memo hit; the full guard check decides
    assert fcc.STATS["memo_hit"] == h0
    assert fcc.STATS["memo_store"] == s0 + 1
    assert st(w)["fast"] == f0 + 1
    assert bits(out) == bits(p(b))
    w(b)
    assert fcc.STATS["memo_hit"] == h0 + 1


def test_global_state_change_is_not_served_from_the_memo(monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    w, p = fast_of(h_small), plain_of(h_small)
    x = torch.randn(4)
    with torch.no_grad():
        for _ in range(3):
            w(x)
            p(x)
    assert fcc.STATS["memo_hit"] == 1
    h0, cf, cp = fcc.STATS["memo_hit"], _n(COMPILES, "fast"), _n(COMPILES, "plain")
    with torch.enable_grad():
        out = w(x)
        ref = p(x)
    assert fcc.STATS["memo_hit"] == h0  # GLOBAL_STATE differs: no memo hit
    assert bits(out) == bits(ref)
    # Same as the plain compiled call: the GLOBAL_STATE guard forces a compile.
    assert _n(COMPILES, "fast") - cf == _n(COMPILES, "plain") - cp == 1
    n = torch.get_num_threads()
    try:
        with torch.no_grad():
            w(x)
            h1 = fcc.STATS["memo_hit"]
            torch.set_num_threads(max(1, n - 1) if n > 1 else 2)
            w(x)
        assert fcc.STATS["memo_hit"] == h1
    finally:
        torch.set_num_threads(n)


def test_global_state_guard_without_memo_takes_the_normal_call():
    w = fast_of(h_small)
    x = torch.randn(4)
    with torch.no_grad():
        w(x)
        w(x)
    s0 = st(w)["slow"]
    with torch.enable_grad():
        w(x)
    assert st(w)["slow"] == s0 + 1  # the guard check failed: normal call (compile)


def test_drop_global_state_guard_reuses_the_entry(monkeypatch):
    w = fast_of(h_small, drop_global_state=True)
    p = plain_of(h_small)
    x = torch.randn(4)
    with torch.no_grad():
        w(x)
        p(x)
    c0 = _n(COMPILES, "fast")
    with torch.enable_grad():
        out = w(x)
        ref = p(x)
    assert _n(COMPILES, "fast") == c0  # no recompile
    assert st(w)["fast"] == 1
    assert bits(out) == bits(ref)
    parts = [p for gm, _d, _s in st(w)["entries"] for p in fcc._guard_parts(gm)]
    assert parts and not any("___check_global_state" in p for p in parts)
    assert any("check_tensor" in p for p in parts)
    assert fcc._drop_global_state(
        [types.SimpleNamespace(guard_type="GLOBAL_STATE"),
         types.SimpleNamespace(guard_type="TENSOR_MATCH")]
    ) == [False, True]


def test_check_mode_counts_a_mismatch_on_a_global_rebind(monkeypatch):
    global GFLAG
    monkeypatch.setattr(fcc, "_MEMO_MODE", "check")
    w = fast_of(h_global)
    x = torch.randn(4)
    try:
        for _ in range(3):
            assert torch.equal(w(x), x + 1)
        assert fcc.STATS["memo_check_ok"] == 1 and fcc.STATS["memo_mismatch"] == 0
        GFLAG = 2
        assert torch.equal(w(x), x + 2)  # guard check catches it; normal call
        assert fcc.STATS["memo_mismatch"] == 1
    finally:
        GFLAG = 1


def test_memo_mode_without_check_skips_global_guards(monkeypatch):
    """Documents the hazard check mode measures: memo=1 never runs the guards,
    so a rebound global is served the stale entry."""
    global GFLAG
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    w = fast_of(h_global)
    x = torch.randn(4)
    try:
        for _ in range(3):
            w(x)
        GFLAG = 2
        assert torch.equal(w(x), x + 1)
    finally:
        GFLAG = 1


def test_attribute_guards_are_never_memoized(monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", "1")
    w = fast_of(Owner.h_method)
    o = Owner(2)
    x = torch.randn(4)
    for _ in range(3):
        assert torch.equal(w(o, x), x * 2)
    assert fcc.STATS["memo_unsafe"] >= 1 and fcc.STATS["memo_hit"] == 0
    o.k = 3
    assert torch.equal(w(o, x), x * 3)
    assert fcc.STATS["memo_hit"] == 0


@pytest.mark.parametrize("memo", ["1", "check"])
def test_aliased_arguments_miss_the_memo(memo, monkeypatch):
    monkeypatch.setattr(fcc, "_MEMO_MODE", memo)
    w, p = fast_of(h_alias), plain_of(h_alias)
    for _ in range(3):
        x, y = torch.randn(4), torch.randn(4)
        w(x.clone(), y.clone())
        p(x.clone(), y.clone())
    assert fcc.STATS["memo_hit"] == 1
    k_dist = fcc.memo_key((torch.randn(4), torch.randn(4)))
    z = torch.randn(4)
    assert fcc.memo_key((z, z)) != k_dist
    h0 = fcc.STATS["memo_hit"]
    z1, z2, z3 = z.clone(), z.clone(), z.clone()
    out = w(z1, z1)
    ref = p(z2, z2)
    eag = h_alias(z3, z3)
    assert fcc.STATS["memo_hit"] == h0
    assert bits(out) == bits(ref) and bits(z1) == bits(z2)
    assert torch.equal(out, eag) and torch.equal(z1, z3)


def test_only_its_own_backend_entries_are_dispatched():
    f = clone(h_small)
    other = torch.compile(f, backend=BK_OTHER, fullgraph=True, dynamic=False)
    w = fast_of(f, same_code=True)
    a, b = torch.randn(4), torch.randn(6)
    other(a)  # another backend's entry for shape a
    w(b)  # own entry for shape b; the refresh sees both entries
    ro, cf = _n(RUNS, "other"), _n(COMPILES, "fast")
    out = w(a)  # must not run the other backend's entry
    assert torch.equal(out, a * 2)
    assert _n(RUNS, "other") == ro
    assert _n(COMPILES, "fast") == cf + 1
    rf = _n(RUNS, "fast")
    w(a)
    assert _n(RUNS, "fast") == rf + 1 and _n(RUNS, "other") == ro


def test_traced_region_inlines_the_original():
    w = fast_of(h_tensors)
    INNER["fn"] = w
    outer = torch.compile(clone(outer_fn), backend=BK_PLAIN, fullgraph=True, dynamic=False)
    x, y = torch.randn(8, 4), torch.randn(8, 6)
    out = outer(x, y)
    assert st(w)["fast"] == 0 and st(w)["slow"] == 0  # inlined, not called
    torch.testing.assert_close(out, h_tensors(x, y, 2, 1.0, True)[0] * 2, rtol=0, atol=0)


def test_fallback_when_a_dynamo_callback_is_active():
    w = fast_of(h_small)
    x = torch.randn(4)
    w(x)
    w(x)
    f0, s0 = st(w)["fast"], st(w)["slow"]
    with torch.compiler.set_stance("force_eager"):
        out = w(x)
    assert torch.equal(out, x * 2)
    assert st(w)["fast"] == f0 and st(w)["slow"] == s0 + 1


def test_hook_eligibility_decorator_and_idempotence():
    h = hook()
    ok = dict(backend=BK_FAST, fullgraph=True, dynamic=False)
    assert hasattr(h(clone(h_small), **ok), "_d98_fast_state")
    assert hasattr(h(**ok)(clone(h_small)), "_d98_fast_state")
    for bad in (
        dict(ok, fullgraph=False),
        dict(ok, dynamic=None),
        dict(ok, dynamic=True),
        dict(ok, backend=BK_PLAIN),
        dict(ok, mode="default"),
        dict(ok, options={"x": 1}),
    ):
        assert not hasattr(h(clone(h_small), **bad), "_d98_fast_state"), bad
    foreign = clone(h_small)
    foreign.__module__ = "somewhere_else"
    assert not hasattr(h(foreign, **ok), "_d98_fast_state")
    assert not hasattr(h(torch.nn.Linear(2, 2), **ok), "_d98_fast_state")
    # A hook around a hook wraps once.
    hh = fcc.make_hook(h, fast=True, drop_global_state=False)
    out = hh(clone(h_small), **ok)
    assert hasattr(out, "_d98_fast_state")
    assert not hasattr(out._d98_compiled, "_d98_fast_state")


def test_install_hooks_torch_tpu_loader_copy_in_either_order(monkeypatch):
    orig = torch.compile
    monkeypatch.setattr(torch, "compile", orig)
    monkeypatch.setattr(fcc, "_INSTALLED", False)
    monkeypatch.setattr(fcc, "_LOG_EVERY", 0)
    loader = types.ModuleType("torch_tpu._loader")
    loader._torch_compile = orig  # captured before the install

    def _default_tpu_compile(*a, **k):
        return loader._torch_compile(*a, **k)

    monkeypatch.setitem(sys.modules, "torch_tpu._loader", loader)
    assert fcc.install(fast=True, memo="", log_every=0)
    assert getattr(torch.compile, "_d98_hook", False)
    assert getattr(loader._torch_compile, "_d98_hook", False)
    first = torch.compile
    fcc.install(fast=True, memo="", log_every=0)  # idempotent
    assert torch.compile is first
    # torch_tpu's device init re-installs its wrapper around its captured copy.
    torch.compile = _default_tpu_compile
    w = torch.compile(clone(h_small), backend=BK_FAST, fullgraph=True, dynamic=False)
    assert hasattr(w, "_d98_fast_state")
    x = torch.randn(4)
    w(x)
    w(x)
    assert st(w)["fast"] == 1


def test_engagement_line_and_step_tick(monkeypatch):
    lines = []
    monkeypatch.setattr(fcc.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(fcc, "_LOG_EVERY", 3)
    monkeypatch.setattr(fcc, "_MEMO_MODE", "check")
    fcc.set_rank(5)
    w = fast_of(h_small)
    x = torch.randn(4)
    for _ in range(4):
        w(x)
    lines.clear()
    for _ in range(6):
        fcc.step_tick()
    d98 = [ln for ln in lines if ln.startswith("D98FAST")]
    assert len(d98) == 2
    first = dict(kv.split("=", 1) for kv in d98[0].split()[1:] if "=" in kv)
    assert first["rank"] == "5" and first["steps"] == "3"
    assert first["fast"] == "3" and first["slow"] == "1" and first["memo"] == "check"
    # call 1 compiles, call 2 stores, calls 3-4 are checked memo hits
    assert first["memo_mismatch"] == "0" and first["memo_check_ok"] == "2"
    assert first["memo_hit"] == "2" and first["fast_share"] == "0.7500"
    assert first["win_fast"] == "3" and first["wrapped"] == "1" and first["engaged"] == "1"
    second = dict(kv.split("=", 1) for kv in d98[1].split()[1:] if "=" in kv)
    assert second["win_fast"] == "0" and second["steps"] == "6"


# ---- real decode helpers (runner/tpu_runner.py) ----------------------------------


def _runner_impls():
    from vllm_torchtpu.runner import tpu_runner as r

    return r


@needs_pkg
@pytest.mark.parametrize("memo", ["", "1", "check"])
def test_real_decode_helpers_bitwise(memo, monkeypatch):
    r = _runner_impls()
    g = torch.Generator().manual_seed(0)

    def ro():
        return torch.randint(0, 4, (64,), dtype=torch.int32, generator=g)

    def groups(w):
        return [torch.randperm(64, generator=g)[:w].to(torch.int32) for _ in range(2)]

    upd = []
    for i in range(8):
        w = 16 if i % 2 else 32
        nt = torch.randint(0, 100, (12, 4), generator=g)
        nt[:, 2 + (i % 2):] = r.INVALID_TOKEN_ID
        upd.append(((ro(), groups(w), nt, torch.tensor(9 + (i % 3))), {}))
    _run_case(r._update_read_offsets_impl, upd, memo, monkeypatch)
    rst = []
    for i in range(6):
        rst.append(((ro(), torch.rand(16, generator=g) > 0.5,
                     torch.randint(1, 64, (2, 16), generator=g), torch.tensor(0)), {}))
    _run_case(r._reset_read_offsets_impl, rst, memo, monkeypatch)
    seed = [((ro(), torch.randint(0, 64, (5 + (i % 2),), generator=g)), {}) for i in range(6)]
    _run_case(r._rollback_offsets_seed_impl, seed, memo, monkeypatch)
    mig = [((ro(), torch.randint(0, 64, (5,), generator=g),
             torch.randint(0, 64, (5,), generator=g)), {}) for _ in range(5)]
    _run_case(r._rollback_offsets_migrate_impl, mig, memo, monkeypatch)
    ssr = []
    for i in range(6):
        packed = torch.randint(0, 8, (4, 4 + 3), generator=g)
        ssr.append(((torch.randint(0, 3, (16,), dtype=torch.int32, generator=g), packed,
                     2 if i < 3 else 1), {}))
    _run_case(r._spec_seed_rows_impl, ssr, memo, monkeypatch)


# ---- flags off / on in a fresh process ---------------------------------------------

_PROBE = r"""
import sys, torch
import vllm_torchtpu
from vllm_torchtpu.runner import tpu_runner as r
from vllm_torchtpu.spec_decode import eagle3 as e
print("HOOK", bool(getattr(torch.compile, "_d98_hook", False)))
print("MOD", "vllm_torchtpu.compilation.fast_compiled_call" in sys.modules)
print("TICKNONE", r._d98_fastcall is None)
print("WRAPPED_UPDATE", hasattr(r._update_read_offsets_compiled, "_d98_fast_state"))
print("WRAPPED_VPSFB", hasattr(r._vps_verify_fb_compiled, "_d98_fast_state"))
m = sys.modules.get("vllm_torchtpu.compilation.fast_compiled_call")
names = sorted(s["name"] for s in m.wrapped_states()) if m else []
print("NWRAPPED", len(names))
for n in names:
    print("WRAPPED", n)
"""


def _probe(env_on):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_TPU_FAST_C")}
    env.pop("VLLM_TPU_AOT_SKIP_GUARD_CHECK", None)
    env.update(env_on)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=900
    )
    assert out.returncode == 0, out.stderr[-4000:]
    keys = {"HOOK", "MOD", "TICKNONE", "WRAPPED_UPDATE", "WRAPPED_VPSFB", "NWRAPPED"}
    pairs = [ln.split(" ", 1) for ln in out.stdout.splitlines() if " " in ln]
    return {k: v for k, v in pairs if k in keys}, [v for k, v in pairs if k == "WRAPPED"]


@needs_pkg
def test_flags_off_identity_and_on_engagement_in_fresh_processes():
    off, off_names = _probe({})
    assert off == {
        "HOOK": "False",
        "MOD": "False",
        "TICKNONE": "True",
        "WRAPPED_UPDATE": "False",
        "WRAPPED_VPSFB": "False",
        "NWRAPPED": "0",
    }, off
    on, on_names = _probe({"VLLM_TPU_FAST_COMPILED_CALL": "1"})
    assert on["HOOK"] == "True" and on["MOD"] == "True" and on["TICKNONE"] == "False"
    assert on["WRAPPED_UPDATE"] == "True" and on["WRAPPED_VPSFB"] == "True"
    assert int(on["NWRAPPED"]) >= 10
    print("D98_WRAPPED_AT_IMPORT", len(on_names))
    for n in on_names:
        print("D98_WRAPPED", n)
