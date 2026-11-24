# SPDX-License-Identifier: Apache-2.0
"""q38 L6-MK: host trims of the D14 MK-type compiled calls (levers ``mkctx`` and ``mkthin``).

Both levers are L6 step-time switches (``core/l6_ab.py``: 0 / 1 / ab:<classes>), read per call
on the host, never at trace time: no compiled program, no compile key and no call ABI changes,
so an arm that sets them reuses the other arm's programs and nothing extra is precompiled.

``mkctx`` (VLLM_TPU_L6_MK_CTX; host_levers_h1.md §4.2 MK-CTX, driver part; hunt_4 v1 §5):
in ``Qwen38SegmentDriver.try_forward`` on a whole-step call

* the contexts the whole-step program runs under (``set_model_tag("model_mk_whole")`` and
  ``set_current_vllm_config(whole_cfg)``) are entered from factories resolved once
  (``_StackedContexts``, as the segment path's ``_serve_contexts``), instead of building an
  ExitStack with three imports per call; the same two context managers, entered and exited
  around every call, with the same arguments;
* ``check_real`` (a numpy max over the computed-token counts) is computed only when the
  shadow-check branch can use it;
* ``_whole_eligible`` (a numpy max) is evaluated once per call instead of at every site;
* the whole-step call module is imported once instead of per call.

The draft call sites (``set_model_tag("eagle_head")`` in propose) belong to L4-D and are not
touched. The ``_note`` running counter in the hunt's list is dropped: the sum it replaces runs
over the step-mix dict only, below the cost of the bookkeeping that would keep a counter exact
across lever windows.

CHECK (VLLM_TPU_L6_MK_CTX_CHECK=N > 0, bounded by VLLM_TPU_L6_CHECK_STEPS through
``l6_ab.check_allowed()``): on every Nth whole-step call with the lever on, the memoized
eligibility is re-evaluated at each site and the cached contexts' tag / config / noaot are
compared with what ``_contexts`` would build; a mismatch turns the lever off for the process
(one ERROR, ``mkctx.mismatch`` on the L6AB line).

``mkthin`` (VLLM_TPU_L6_MK_THIN_TT; host_levers_h1.md §4.2 MK-THIN-TT; hunt_4 v1 §5): torch_tpu
calls every compiled program through ``_SplitCompiledExecutable.__call__`` → its split fx
GraphModule (``nn.Module.__call__``) → ``_WrapperModule.forward`` (``nn.Module.__call__``) →
``TorchTpuCompiledExecutable.__call__`` → ``_fast_runner(args)``. When the split graph holds
exactly one partition and nothing else (placeholders, one ``call_module`` of a
``_WrapperModule`` without CPU arguments whose arguments are placeholders, ``getitem``s of its
result and an output tuple of those), the patched ``__call__`` calls the partition's
``_fast_runner`` directly with the same argument tuple and rebuilds the same output tuple from
the same objects. Anything else (a handshake / recompile function, CPU partitions or CPU
arguments, symint or constant nodes, kwargs, module hooks, a released executable, an
``output_shapes`` call) takes torch_tpu's own path. The plan is derived once per executable
from its graph.

CHECK (VLLM_TPU_L6_MK_THIN_TT_CHECK=N > 0, bounded as above): every Nth thin-eligible call runs
torch_tpu's own path with the partition's ``_fast_runner`` observed, and requires that it
received exactly (by identity) the argument tuple the thin path builds, and that the stock
result equals (by identity, element by element) the tuple the thin path builds from the same
runner result. The program runs once. A mismatch disables the thin path for that executable
(one ERROR, ``mkthin.mismatch``).

Engagement: ``mkctx.whole`` / ``mkthin.thin`` / ``mkthin.stock`` / ``mkthin.plan_refused`` /
``*.check_ok`` / ``*.mismatch`` on the L6AB window line; one INFO line per refused plan
(reason) and per installed patch.
"""

from __future__ import annotations

import operator
import os
from typing import Any

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MKCTX = "mkctx"
MKTHIN = "mkthin"

_CHECK_ENVS = {MKCTX: "VLLM_TPU_L6_MK_CTX_CHECK", MKTHIN: "VLLM_TPU_L6_MK_THIN_TT_CHECK"}
# Resolved on first use (tests reset them).
_CHECK: dict[str, int | None] = {MKCTX: None, MKTHIN: None}
_TURN: dict[str, int] = {MKCTX: 0, MKTHIN: 0}


def _check_every(lever: str) -> int:
    v = _CHECK[lever]
    if v is None:
        raw = os.environ.get(_CHECK_ENVS[lever], "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"{_CHECK_ENVS[lever]} must be >= 0, got {v}")
        _CHECK[lever] = v
    return v


def check_turn(lever: str) -> bool:
    """Whether this on-call of `lever` runs its CHECK (every Nth call, inside the CHECK window)."""
    every = _check_every(lever)
    if every <= 0:
        return False
    _TURN[lever] += 1
    if _TURN[lever] < every:
        return False
    _TURN[lever] = 0
    return _l6_ab.check_allowed()


def _reset_for_tests() -> None:
    for k in _CHECK:
        _CHECK[k] = None
        _TURN[k] = 0
    _WSC[0] = None


# ----------------------------------------------------------------------------------------- mkctx
_WSC: list[Any] = [None]


def wsc_module() -> Any:
    """megakernel_runtime.qwen38_wholestep_call, imported once."""
    m = _WSC[0]
    if m is None:
        from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as m

        _WSC[0] = m
    return m


def mkctx_on(driver: Any) -> bool:
    """The lever's state for this call (False once a CHECK found a mismatch)."""
    return _l6_ab.on(MKCTX) and not driver.__dict__.get("_l6_mkctx_off", False)


def _whole_cfg(driver: Any, runner: Any) -> Any:
    """The config ``Qwen38SegmentDriver._contexts(runner, "model_mk_whole")`` enters."""
    cfg = getattr(driver, "whole_cfg", None)
    return cfg or (getattr(driver, "cfg", None) or runner.vllm_config)


def whole_contexts(driver: Any, runner: Any) -> Any:
    """``_contexts(runner, "model_mk_whole")`` from factories resolved once (falls back to it
    with ``noaot``, whose env override is per call)."""
    if getattr(driver, "noaot", False):
        return driver._contexts(runner, "model_mk_whole")
    cfg = _whole_cfg(driver, runner)
    cached = driver.__dict__.get("_l6_mkctx_ctx")
    if cached is None or cached[0] is not cfg:
        from vllm.compilation.backends import set_model_tag
        from vllm.config import set_current_vllm_config

        cached = (
            cfg,
            (lambda: set_model_tag("model_mk_whole"), lambda: set_current_vllm_config(cfg)),
        )
        driver._l6_mkctx_ctx = cached
    from vllm_torchtpu.megakernel_runtime.qwen38_driver import _StackedContexts

    return _StackedContexts(cached[1])


def mkctx_check(driver: Any, runner: Any, n_req: int, rows: int, elig: Any) -> None:
    """CHECK turn: the memoized eligibility and the cached contexts against the stock ones."""
    bad = []
    want = driver._whole_eligible(runner, n_req, rows)
    if elig is not None and bool(elig) != bool(want):
        bad.append(f"whole_eligible memo {elig} != {want}")
    if not getattr(driver, "noaot", False):
        cached = driver.__dict__.get("_l6_mkctx_ctx")
        cfg = _whole_cfg(driver, runner)
        if cached is not None and cached[0] is not cfg:
            bad.append("cached whole-step config is not the config _contexts enters")
    if bad:
        driver._l6_mkctx_off = True
        _l6_ab.note(MKCTX, "mismatch")
        logger.error("q38 L6 MK-CTX: CHECK mismatch (%s); the lever is off for the rest of this process",
                     "; ".join(bad))
    else:
        _l6_ab.note(MKCTX, "check_ok")


# ---------------------------------------------------------------------------------------- mkthin
_THIN_INSTALLED: list[bool] = [False]
# Plan attribute on a _SplitCompiledExecutable: a _ThinPlan, or a str (the refusal reason).
_PLAN_ATTR = "_l6_thin_plan"
# sha256[:16] of inspect.getsource of the torch_tpu functions the thin path replicates, as on the
# served image (torch_tpu 0.2.0; read by the L6-MK probe job j-3be93ddd-938e-47d8-853b). Any other
# source leaves the lever inert (one ERROR), since the thin path mirrors exactly these bodies.
THIN_SOURCES = {
    "_SplitCompiledExecutable.__call__": "2d850e8963a2789e",
    "_SplitCompiledExecutable._maybe_handshake_and_recompile": "47b74dfd6cc57f98",
    "_WrapperModule.forward": "f25c9f603ec86f23",
    "TorchTpuCompiledExecutable.__call__": "060cb304d3b45824",
}
_M: list[Any] = [None]  # torch.nn.modules.module, bound at install


def _count(key: str) -> None:
    """l6_ab.note(MKTHIN, key) without the per-call name formatting (12 calls per step)."""
    if _l6_ab.ACTIVE:
        c = _l6_ab._state["counts"]
        c[key] = c.get(key, 0) + 1


def note_whole() -> None:
    _count("mkctx.whole")


class _ThinPlan:
    __slots__ = ("exe", "gm", "wrapper", "nph", "arg_idx", "identity", "unwrap", "outs", "single", "off")

    def __init__(self, exe, gm, wrapper, nph, arg_idx, unwrap, outs, single):
        self.exe = exe  # the partition's TorchTpuCompiledExecutable
        self.gm = gm  # the split GraphModule
        self.wrapper = wrapper  # the partition's _WrapperModule
        self.nph = nph  # number of placeholders (the call's argument count)
        self.arg_idx = arg_idx  # placeholder index per runner argument
        self.identity = arg_idx == tuple(range(nph))
        self.unwrap = unwrap  # _WrapperModule.unwrap_singleton_tuple
        self.outs = outs  # per output entry: ("r", i) = x[i], ("x", None) = x, ("a", j) = argument j
        self.single = single  # the graph returns one value, not a tuple
        self.off = False


def _hooked(gm: Any, wrapper: Any) -> bool:
    """Whether nn.Module.__call__ would not go straight to forward for either module (hooks or a
    compiled call impl): then torch_tpu's own path."""
    M = _M[0]
    if (
        M._global_forward_hooks
        or M._global_forward_pre_hooks
        or M._global_backward_hooks
        or M._global_backward_pre_hooks
    ):
        return True
    for m in (gm, wrapper):
        if (
            m._forward_hooks
            or m._forward_pre_hooks
            or m._backward_hooks
            or m._backward_pre_hooks
            or m.__dict__.get("_compiled_call_impl") is not None
        ):
            return True
    return False


def thin_plan(split_exe: Any) -> Any:
    """A _ThinPlan for a single-partition split executable, or a str saying why not."""
    import torch
    from torch.fx.graph_module import _WrappedCall
    from torch_tpu._internal.compile import split_compiler as sc
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable

    d = split_exe.__dict__
    if d.get("recompile_fn") is not None:
        return "recompile_fn (dispatch-stage handshake)"
    gm = d.get("_split_gm")
    if not isinstance(gm, torch.fx.GraphModule):
        return "no split GraphModule"
    # fx gives every GraphModule its own class whose __call__ goes through a class-level
    # _WrappedCall; with cls_call None that is nn.Module.__call__ -> forward (the generated code).
    wc = type(gm).__dict__.get("_wrapped_call")
    if (
        type(wc) is not _WrappedCall
        or wc.cls_call is not None
        or not getattr(type(gm).__call__, "__qualname__", "").endswith("call_wrapped")
        or "forward" in gm.__dict__
    ):
        return "split module call or forward overridden"
    if type(gm.graph._codegen) is not torch.fx.graph.CodeGen:
        return "non-default fx codegen"
    if gm.graph.python_code(root_module="self").src != gm.code:
        return "generated forward differs from the graph"
    placeholders, calls, getitems, output = [], [], set(), None
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            if n.args or n.kwargs or str(n.target).startswith("*"):
                return "placeholder with a default or varargs"
            placeholders.append(n)
        elif n.op == "call_module":
            calls.append(n)
        elif n.op == "call_function" and n.target is operator.getitem:
            getitems.add(n)
        elif n.op == "output":
            output = n
        else:
            return f"node {n.op}:{n.target}"
    if len(calls) != 1 or output is None:
        return f"{len(calls)} partitions"
    call = calls[0]
    wrapper = gm.get_submodule(str(call.target))
    if type(wrapper) is not sc._WrapperModule or "forward" in wrapper.__dict__:
        return f"partition is {type(wrapper).__name__}"
    if getattr(wrapper, "cpu_arg_indices", ()):
        return "CPU arguments"
    exe = wrapper.__dict__.get("submod", getattr(wrapper, "submod", None))
    if type(exe) is not TorchTpuCompiledExecutable:
        return f"partition runs {type(exe).__name__}"
    if call.kwargs:
        return "call kwargs"
    pos = {p: i for i, p in enumerate(placeholders)}
    arg_idx = []
    for a in call.args:
        if not isinstance(a, torch.fx.Node) or a not in pos:
            return "partition argument is not a placeholder"
        arg_idx.append(pos[a])
    for g in getitems:
        if len(g.args) != 2 or g.args[0] is not call or type(g.args[1]) is not int or g.kwargs:
            return "getitem not on the partition result"
    out = output.args[0] if output.args else None
    single = not isinstance(out, (tuple, list))
    if not single and type(out) is not tuple:
        return "output is a list"
    if output.kwargs:
        return "output kwargs"
    outs = []
    for e in ((out,) if single else out):
        if not isinstance(e, torch.fx.Node):
            return "constant output"
        if e in getitems:
            outs.append(("r", e.args[1]))
        elif e is call:
            outs.append(("x", None))
        elif e in pos:
            outs.append(("a", pos[e]))
        else:
            return f"output from {e.op}"
    return _ThinPlan(exe, gm, wrapper, len(placeholders), tuple(arg_idx),
                     bool(wrapper.unwrap_singleton_tuple), tuple(outs), single)


def _thin_build(plan: _ThinPlan, args: Any, x: Any) -> Any:
    """The split graph's output, built from the partition's (unwrapped) result `x`."""
    if plan.single:
        kind, i = plan.outs[0]
        return x[i] if kind == "r" else (x if kind == "x" else args[i])
    return tuple(x[i] if kind == "r" else (x if kind == "x" else args[i]) for kind, i in plan.outs)


def _thin_disable(plan: _ThinPlan, why: str) -> None:
    plan.off = True
    _l6_ab.note(MKTHIN, "mismatch")
    logger.error("q38 L6 MK-THIN-TT: CHECK mismatch (%s); thin dispatch off for this executable", why)


def _thin_check(orig_call: Any, split_exe: Any, plan: _ThinPlan, a: Any, call_args: tuple) -> Any:
    """Run torch_tpu's own path once with the partition's runner observed, and compare what it
    passed and returned with what the thin path builds from the same objects."""
    exe = plan.exe
    fr = exe._fast_runner
    seen: list = []

    def observed(t):
        r = fr(t)
        seen.append((t, r))
        return r

    exe._fast_runner = observed
    try:
        out = orig_call(split_exe, *call_args)
    finally:
        exe._fast_runner = fr
    if len(seen) != 1:
        _thin_disable(plan, f"runner called {len(seen)} times")
        return out
    got, r = seen[0]
    want = tuple(a) if plan.identity else tuple(a[i] for i in plan.arg_idx)
    if type(got) is not tuple or len(got) != len(want) or any(g is not w for g, w in zip(got, want)):
        _thin_disable(plan, "runner arguments differ")
        return out
    thin = _thin_build(plan, a, r[0] if plan.unwrap and isinstance(r, (tuple, list)) else r)
    if plan.single:
        same = thin is out
    else:
        same = type(out) is tuple and len(out) == len(thin) and all(p is q for p, q in zip(out, thin))
    if same:
        _l6_ab.note(MKTHIN, "check_ok")
    else:
        _thin_disable(plan, "outputs differ")
    return out


def _source_hashes() -> dict[str, str]:
    import hashlib
    import inspect

    from torch_tpu._internal.compile import split_compiler as sc
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable

    objs = {
        "_SplitCompiledExecutable.__call__": sc._SplitCompiledExecutable.__dict__["__call__"],
        "_SplitCompiledExecutable._maybe_handshake_and_recompile":
            sc._SplitCompiledExecutable._maybe_handshake_and_recompile,
        "_WrapperModule.forward": sc._WrapperModule.forward,
        "TorchTpuCompiledExecutable.__call__": TorchTpuCompiledExecutable.__call__,
    }
    out = {}
    for k, f in objs.items():
        f = getattr(f, "__wrapped__", f) if getattr(f, "_l6_thin", False) else f
        out[k] = hashlib.sha256(inspect.getsource(f).encode()).hexdigest()[:16]
    return out


def install_mkthin() -> bool:
    """Patch torch_tpu's _SplitCompiledExecutable.__call__ (idempotent); False if refused."""
    if _THIN_INSTALLED[0]:
        return True
    try:
        import torch.nn.modules.module as M
        from torch_tpu._internal.compile import split_compiler as sc

        got = _source_hashes()
    except Exception as exc:  # noqa: BLE001 - no torch_tpu / no sources: the lever stays inert
        logger.error("q38 L6 MK-THIN-TT: NOT installed (%s: %s); the lever stays inert", type(exc).__name__, exc)
        return False
    bad = {k: v for k, v in got.items() if THIN_SOURCES.get(k) != v}
    if bad:
        logger.error("q38 L6 MK-THIN-TT: NOT installed: torch_tpu sources differ from the verified ones %s; "
                     "the lever stays inert", bad)
        return False
    _M[0] = M
    cls = sc._SplitCompiledExecutable
    orig = cls.__dict__["__call__"]
    if getattr(orig, "_l6_thin", False):
        _THIN_INSTALLED[0] = True
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs or not _l6_ab.on(MKTHIN):
            return orig(self, *args, **kwargs)
        d = self.__dict__
        plan = d.get(_PLAN_ATTR)
        if plan is None:
            try:
                plan = thin_plan(self)
            except Exception as exc:  # noqa: BLE001 - never fail a call the stock path runs
                plan = f"plan error {type(exc).__name__}: {exc}"
            d[_PLAN_ATTR] = plan
            if isinstance(plan, str):
                _count("mkthin.plan_refused")
                logger.info("q38 L6 MK-THIN-TT: an executable keeps torch_tpu's path (%s)", plan)
        if type(plan) is str or plan.off or d.get("recompile_fn") is not None:
            _count("mkthin.stock")
            return orig(self, *args)
        a = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        fr = plan.exe._fast_runner
        if (
            fr is None
            or len(a) != plan.nph
            or d.get("_split_gm") is not plan.gm
            or plan.wrapper.__dict__.get("submod", plan.exe) is not plan.exe
            or _hooked(plan.gm, plan.wrapper)
        ):
            _count("mkthin.stock")
            return orig(self, *args)
        if check_turn(MKTHIN):
            return _thin_check(orig, self, plan, a, args)
        r = fr(tuple(a) if plan.identity else tuple(a[i] for i in plan.arg_idx))
        _count("mkthin.thin")
        return _thin_build(plan, a, r[0] if plan.unwrap and isinstance(r, (tuple, list)) else r)

    __call__._l6_thin = True  # type: ignore[attr-defined]
    __call__.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.__call__ = __call__
    _THIN_INSTALLED[0] = True
    logger.info("q38 L6 MK-THIN-TT: installed on %s.%s (%s=%s, check every %d)", cls.__module__, cls.__name__,
                _l6_ab.LEVERS[MKTHIN], os.environ.get(_l6_ab.LEVERS[MKTHIN], ""), _check_every(MKTHIN))
    return True


def uninstall_mkthin_for_tests() -> None:
    from torch_tpu._internal.compile import split_compiler as sc

    f = sc._SplitCompiledExecutable.__dict__["__call__"]
    if getattr(f, "_l6_thin", False):
        sc._SplitCompiledExecutable.__call__ = f.__wrapped__
    _THIN_INSTALLED[0] = False


def configured(lever: str) -> bool:
    """Whether the lever's flag is set to anything but off."""
    return _l6_ab._cfg.get(lever, (False, None)) != (False, None)


def install_worker_patches() -> None:
    """Worker init: install the class patches of the configured MK levers (no-op otherwise)."""
    if configured(MKTHIN):
        install_mkthin()
    from vllm_torchtpu.core import l6_fuse_rpc

    if l6_fuse_rpc.configured():
        l6_fuse_rpc.install_worker()
